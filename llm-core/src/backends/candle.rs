use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, Context};
use candle_core::{Device, Tensor, DType, Module};
use candle_core::quantized::QMatMul;
use parking_lot::Mutex;

use crate::types::*;
use crate::backend::LlmBackend;
use crate::model::config::parse_config;
use crate::sampler::sample_logits;
use crate::graph::{ComputeGraph, Operator, scan_tensors, build_graph, map_gguf_name};
use crate::backends::vision::VisionEncoder;
use crate::backends::attention::{RmsNorm, RawKvCache, apply_rope, apply_rope_q, repeat_kv, rms_norm_no_scale};
use crate::backends::multimodal::splice_visual_embeddings;
use crate::backends::weights::{ExecContext, WeightStore, meta_u32, meta_u32_agnostic, meta_f32_agnostic, find_meta_key};
use crate::backends::audio::AudioEncoder;





#[derive(Clone)]
pub(crate) struct BlockData {
    pub k: Tensor,
    pub k_scale: Option<Tensor>,
    pub v: Tensor,
    pub v_scale: Option<Tensor>,
}

fn generate_hadamard_orthogonal(d: usize) -> Vec<f32> {
    let mut p2 = 1;
    while p2 * 2 <= d {
        p2 *= 2;
    }
    
    let mut h = vec![0.0f32; d * d];
    let mut h_sub = vec![1.0f32; p2 * p2];
    let mut step = 1;
    while step < p2 {
        for i in 0..step {
            for j in 0..step {
                let val = h_sub[i * p2 + j];
                h_sub[i * p2 + (j + step)] = val;
                h_sub[(i + step) * p2 + j] = val;
                h_sub[(i + step) * p2 + (j + step)] = -val;
            }
        }
        step *= 2;
    }
    
    let scale = 1.0 / (p2 as f32).sqrt();
    for i in 0..p2 {
        for j in 0..p2 {
            h[i * d + j] = h_sub[i * p2 + j] * scale;
        }
    }
    for i in p2..d {
        h[i * d + i] = 1.0;
    }
    h
}

fn quantize(x: &Tensor, dtype: KvDtype, comp_dtype: DType) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let abs_x = x.abs()?;
    let max_val = abs_x.max_keepdim(3)?;
    
    let max_quant = match dtype {
        KvDtype::Q8 => 127.0f32,
        KvDtype::Q4 => 7.0f32,
        _ => 1.0f32,
    };
    
    let scale = max_val.affine(1.0 / max_quant as f64, 0.0)?.broadcast_add(&Tensor::new(1e-5f32, dev)?.to_dtype(comp_dtype)?)?;
    let scaled = x.broadcast_div(&scale)?;
    let rounded = scaled.round()?;
    
    let offset = match dtype {
        KvDtype::Q8 => 128.0f32,
        KvDtype::Q4 => 8.0f32,
        _ => 0.0f32,
    };
    
    let u8_tensor = rounded.affine(1.0, offset as f64)?.to_dtype(DType::U8)?;
    Ok((u8_tensor, scale))
}

fn dequantize(u8_tensor: &Tensor, scale: &Tensor, dtype: KvDtype, comp_dtype: DType) -> Result<Tensor> {
    let f_tensor = u8_tensor.to_dtype(comp_dtype)?;
    let offset = match dtype {
        KvDtype::Q8 => 128.0f32,
        KvDtype::Q4 => 8.0f32,
        _ => 0.0f32,
    };
    let centered = f_tensor.affine(1.0, -offset as f64)?;
    let dequantized = centered.broadcast_mul(scale)?;
    Ok(dequantized)
}

fn update_block_tensor(
    block_tensor: &Tensor,
    slice_tensor: &Tensor,
    start_offset: usize,
    chunk_len: usize,
) -> Result<Tensor> {
    let block_size = block_tensor.dim(1)?;
    let end_offset = start_offset + chunk_len;
    
    let left = if start_offset > 0 {
        Some(block_tensor.narrow(1, 0, start_offset)?)
    } else {
        None
    };
    
    let right = if end_offset < block_size {
        Some(block_tensor.narrow(1, end_offset, block_size - end_offset)?)
    } else {
        None
    };
    
    let res = match (left, right) {
        (Some(l), Some(r)) => {
            Tensor::cat(&[&l, slice_tensor, &r], 1)?
        }
        (Some(l), None) => {
            Tensor::cat(&[&l, slice_tensor], 1)?
        }
        (None, Some(r)) => {
            Tensor::cat(&[slice_tensor, &r], 1)?
        }
        (None, None) => {
            slice_tensor.clone()
        }
    };
    Ok(res)
}

pub struct CandleBackend {
    /// Full-precision weights (safetensors path; loaded directly onto device)
    weights: HashMap<String, Tensor>,
    /// Quantized GGUF weights kept on CPU; dequantized lazily
    quantized_weights: HashMap<String, candle_core::quantized::QTensor>,
    /// Dequantization cache: embed/norm weights stored as f32 to avoid f16 overflow
    deq_cache: Mutex<HashMap<String, Tensor>>,
    /// Running GPU bytes used by deq_cache
    gpu_cache_bytes: std::sync::atomic::AtomicU64,
    /// GPU VRAM budget for weight cache (bytes); set at load time
    gpu_cache_budget: u64,
    graph: Option<ComputeGraph>,
    meta: Option<ModelMeta>,
    kv_cache: Option<Mutex<RawKvCache>>,
    kv_config: KvCacheConfig,
    eos_token_id: u32,
    /// Primary compute device (CUDA or CPU)
    device: Device,
    /// Compute dtype: F32 on CPU (prevents f16 overflow), F16 on CUDA.
    compute_dtype: DType,
    vision_encoder: Option<VisionEncoder>,
    visual_embeddings: Mutex<Option<Tensor>>,
    last_image_path: Mutex<Option<String>>,
    last_pixel_values: Mutex<Option<Tensor>>,
    audio_encoder: Option<AudioEncoder>,
    audio_embeddings: Mutex<Option<Tensor>>,
    last_audio_path: Mutex<Option<String>>,
    last_audio_values: Mutex<Option<Tensor>>,
    gpu_kv_cache: Mutex<HashMap<(usize, BlockId), BlockData>>,
    seq_blocks: Mutex<HashMap<SeqId, Vec<BlockId>>>,
    explicit_dequantize: bool,
    use_vram_embeddings: bool,
    qmatmul_cache: Mutex<HashMap<String, QMatMul>>,
}


impl CandleBackend {
    pub fn new() -> Self {
        let explicit_dequantize = std::env::var("LLM_EXPLICIT_DEQUANTIZE").is_ok();
        let use_vram_embeddings = std::env::var("LLM_USE_VRAM_EMBEDDINGS").is_ok();
        Self {
            weights: HashMap::new(),
            quantized_weights: HashMap::new(),
            deq_cache: Mutex::new(HashMap::new()),
            gpu_cache_bytes: std::sync::atomic::AtomicU64::new(0),
            gpu_cache_budget: 0,
            graph: None,
            meta: None,
            kv_cache: None,
            kv_config: KvCacheConfig {
                n_layers: 32,
                n_kv_heads: 32,
                head_dim: 128,
                block_size: 16,
                dtype: match std::env::var("LLM_KV_DTYPE").as_deref() {
                    Ok("q8") | Ok("Q8") => KvDtype::Q8,
                    Ok("q4") | Ok("Q4") => KvDtype::Q4,
                    _ => KvDtype::F16,
                },
            },
            eos_token_id: 2,
            device: Device::Cpu,
            // F32 on CPU prevents f16 accumulation overflow in matmul;
            // updated to F16 when CUDA device is selected at load time.
            compute_dtype: DType::F32,
            vision_encoder: None,
            visual_embeddings: Mutex::new(None),
            last_image_path: Mutex::new(None),
            last_pixel_values: Mutex::new(None),
            audio_encoder: None,
            audio_embeddings: Mutex::new(None),
            last_audio_path: Mutex::new(None),
            last_audio_values: Mutex::new(None),
            gpu_kv_cache: Mutex::new(HashMap::new()),
            seq_blocks: Mutex::new(HashMap::new()),
            explicit_dequantize,
            use_vram_embeddings,
            qmatmul_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Thin wrapper: resolve a weight tensor by delegating to `WeightStore`.
    ///
    /// This keeps the ~20 call sites in `forward_pass` unchanged while the
    /// actual resolution logic lives in `weights.rs`.
    fn get_weight(&self, name: &str, local_cache: &mut HashMap<String, Tensor>) -> Result<Tensor> {
        let store = WeightStore {
            weights: &self.weights,
            deq_cache: &self.deq_cache,
            quantized_weights: &self.quantized_weights,
            gpu_cache_bytes: &self.gpu_cache_bytes,
            gpu_cache_budget: self.gpu_cache_budget,
            explicit_dequantize: self.explicit_dequantize,
            device: &self.device,
        };
        store.get(name, local_cache)
    }

    /// Set pre-computed visual embeddings (called by the CLI before forward_pass).
    pub fn set_visual_embeddings(&self, embeds: Tensor) {
        *self.visual_embeddings.lock() = Some(embeds);
    }
}

impl LlmBackend for CandleBackend {

    fn name(&self) -> &str {
        match self.device {
            Device::Cpu => "candle-cpu",
            Device::Cuda(_) => "candle-cuda",
            _ => "candle-unknown",
        }
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn clear_sequence(&self, seq_id: SeqId) {
        let blocks_to_clear = {
            let mut seq_blocks_map = self.seq_blocks.lock();
            seq_blocks_map.remove(&seq_id)
        };
        if let Some(blocks) = blocks_to_clear {
            let mut cache = self.gpu_kv_cache.lock();
            let n_layers = self.kv_config.n_layers;
            for layer_idx in 0..n_layers {
                for &block_id in &blocks {
                    cache.remove(&(layer_idx, block_id));
                }
            }
        }
    }

    fn set_explicit_dequantize(&mut self, val: bool) {
        self.explicit_dequantize = val;
    }

    fn set_use_vram_embeddings(&mut self, val: bool) {
        self.use_vram_embeddings = val;
    }

    fn load_weights(&mut self, path: &Path) -> Result<ModelMeta> {
        let is_gguf = path.is_file() && path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase() == "gguf").unwrap_or(false);

        // Auto-detect hardware accelerator: CUDA first, then Metal, falling back to CPU
        let dev = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or_else(|e| {
                tracing::warn!("CUDA init failed ({e}), falling back to CPU");
                Device::Cpu
            })
        } else if let Ok(metal_dev) = Device::new_metal(0) {
            metal_dev
        } else {
            Device::Cpu
        };
        self.device = dev.clone();
        self.compute_dtype = if dev.is_cpu() { DType::F32 } else { DType::F16 };

        // Query free VRAM for cache budget (leave 2.0 GB headroom for activations/KV cache)
        let vram_headroom_bytes: u64 = 2_000 * 1024 * 1024;
        let mut free_vram = if candle_core::utils::cuda_is_available() {
            std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.lines().next().and_then(|l| l.trim().parse::<u64>().ok()))
                .map(|mib| mib * 1024 * 1024)
                .unwrap_or(0)
        } else {
            0
        };
        if free_vram == 0 && dev.is_cuda() {
            tracing::info!("nvidia-smi query returned 0, but CUDA device is active. Falling back to 6.0 GB estimated free VRAM.");
            free_vram = 6_000 * 1024 * 1024;
        }
        self.gpu_cache_budget = free_vram.saturating_sub(vram_headroom_bytes);
        tracing::info!(
            "GPU cache budget: {:.1} GB (free VRAM: {:.1} GB)",
            self.gpu_cache_budget as f64 / 1e9,
            free_vram as f64 / 1e9
        );

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        let mut quantized_weights: HashMap<String, candle_core::quantized::QTensor> = HashMap::new();

        let meta = if is_gguf {
            let mut file = std::fs::File::open(path)
                .context(format!("Failed to open GGUF file: {:?}", path))?;
            let model = candle_core::quantized::gguf_file::Content::read(&mut file)
                .context("Failed to read GGUF content")?;

            // Discover and load mmproj metadata if a matching mmproj file exists
            let mut mmproj_metadata = HashMap::new();
            let base = path.to_string_lossy();
            let stem = if let Some(pos) = base.rfind('.') {
                let without_ext = &base[..pos];
                if let Some(q_pos) = without_ext.rfind('.') {
                    without_ext[..q_pos].to_string()
                } else {
                    without_ext.to_string()
                }
            } else {
                base.to_string()
            };
            let suffixes = [
                "-mmproj-f16.gguf",
                ".mmproj.gguf",
                "-mmproj.gguf",
                ".BF16-mmproj.gguf",
                "-mmproj-f32.gguf",
            ];
            let mut mmproj_path = None;
            for suffix in &suffixes {
                let candidate = format!("{}{}", stem, suffix);
                if std::path::Path::new(&candidate).exists() {
                    mmproj_path = Some(std::path::PathBuf::from(candidate));
                    break;
                }
            }
            if mmproj_path.is_none() {
                if let Some(parent) = path.parent() {
                    if let Ok(entries) = std::fs::read_dir(parent) {
                        for entry in entries.flatten() {
                            let p = entry.path();
                            if p.is_file() {
                                let name = p.file_name().unwrap_or_default().to_string_lossy();
                                if name.contains("mmproj") && name.ends_with(".gguf") {
                                    mmproj_path = Some(p);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if let Some(ref mp_path) = mmproj_path {
                tracing::info!("Discovered mmproj file at {:?}; loading its metadata...", mp_path);
                if let Ok(mut mp_file) = std::fs::File::open(mp_path) {
                    if let Ok(mp_model) = candle_core::quantized::gguf_file::Content::read(&mut mp_file) {
                        mmproj_metadata = mp_model.metadata;
                    }
                }
            }

            let arch = match model.metadata.get("general.architecture") {
                Some(candle_core::quantized::gguf_file::Value::String(s)) => s.as_str(),
                _ => "llama",
            };

            let vocab_size = match model.metadata.get("tokenizer.ggml.tokens") {
                Some(candle_core::quantized::gguf_file::Value::Array(arr)) => arr.len(),
                _ => 151936,
            };

            let n_layers = meta_u32_agnostic(&model.metadata, "block_count")
                .ok_or_else(|| anyhow!("Missing block_count in GGUF metadata"))? as usize;

            let hidden_dim = meta_u32_agnostic(&model.metadata, "embedding_length")
                .ok_or_else(|| anyhow!("Missing embedding_length in GGUF metadata"))? as usize;

            let intermediate_dim = meta_u32_agnostic(&model.metadata, "feed_forward_length")
                .ok_or_else(|| anyhow!("Missing feed_forward_length in GGUF metadata"))? as usize;

            let n_heads = meta_u32_agnostic(&model.metadata, "attention.head_count")
                .ok_or_else(|| anyhow!("Missing head_count in GGUF metadata"))? as usize;

            let n_kv_heads = meta_u32_agnostic(&model.metadata, "attention.head_count_kv")
                .unwrap_or(n_heads as u32) as usize;

            let max_seq_len = meta_u32_agnostic(&model.metadata, "context_length")
                .unwrap_or(4096) as usize;

            let rope_theta = meta_f32_agnostic(&model.metadata, "rope.freq_base")
                .unwrap_or(10000.0);

            let head_dim = if let Some(info) = model.tensor_infos.get("blk.0.attn_q.weight") {
                info.shape.dims()[0] / n_heads
            } else if let Some(info) = model.tensor_infos.get("blk.0.attn_q") {
                info.shape.dims()[0] / n_heads
            } else {
                meta_u32_agnostic(&model.metadata, "attention.key_length")
                    .unwrap_or((hidden_dim / n_heads) as u32) as usize
            };

            let rms_norm_eps = meta_f32_agnostic(&model.metadata, "attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6);

            let eos_token_id = meta_u32(&model.metadata, "tokenizer.ggml.eos_token_id")
                .unwrap_or(2);
            self.eos_token_id = eos_token_id;

            tracing::info!("Loading {} GGUF tensors as QTensor", model.tensor_infos.len());
            let tie_word_embeddings = arch == "gemma" || arch == "gemma2" || arch == "gemma4"; // pre-check, refined below
            let mut remaining_vram_budget = self.gpu_cache_budget;
            for name in model.tensor_infos.keys() {
                let name_lower = name.to_lowercase();
                let is_token_embd = name == "token_embd.weight";
                let is_non_matmul = name_lower.contains("embed") 
                    || name_lower.contains("embd")
                    || name_lower.contains("norm") 
                    || name_lower.contains("bias") 
                    || name_lower.contains("scale")
                    || name_lower.contains("freq")
                    || name_lower.contains("rotors")
                    || name_lower.contains("rot");
                let mut fit_in_vram = false;
                if is_non_matmul && !name_lower.contains("freq") && !name_lower.contains("rot") && self.device.is_cuda() {
                    if let Some(info) = model.tensor_infos.get(name) {
                        let size_bytes = (info.shape.dims().iter().product::<usize>() as u64) * 2;
                        if remaining_vram_budget >= size_bytes {
                            remaining_vram_budget -= size_bytes;
                            fit_in_vram = true;
                        }
                    }
                }
                let load_device = if self.explicit_dequantize {
                    Device::Cpu
                } else if is_non_matmul {
                    if fit_in_vram || self.use_vram_embeddings {
                        self.device.clone()
                    } else {
                        Device::Cpu
                    }
                } else {
                    self.device.clone()
                };
                let qt = model.tensor(&mut file, name, &load_device)
                    .with_context(|| format!("Failed to read tensor {}", name))?;
                let hf_name = map_gguf_name(name);
                quantized_weights.insert(hf_name, qt);

                // For tied-embedding architectures (Gemma), also load the token embedding on CUDA
                // so the lm_head MatMul can use it as a QMatMul without CPU→GPU copy every decode step.
                if is_token_embd && tie_word_embeddings && !self.explicit_dequantize {
                    if let Ok(qt_cuda) = model.tensor(&mut file, name, &self.device) {
                        quantized_weights.insert("lm_head.weight".to_string(), qt_cuda);
                    }
                }
            }

            let has_vision_encoder = match model.metadata.get("clip.has_vision_encoder")
                .or_else(|| mmproj_metadata.get("clip.has_vision_encoder")) {
                Some(candle_core::quantized::gguf_file::Value::Bool(b)) => *b,
                _ => false,
            };
            let has_audio_encoder = match model.metadata.get("clip.has_audio_encoder")
                .or_else(|| mmproj_metadata.get("clip.has_audio_encoder")) {
                Some(candle_core::quantized::gguf_file::Value::Bool(b)) => *b,
                _ => false,
            };
            let audio_hidden_dim = meta_u32(&model.metadata, "clip.audio.embedding_length")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.audio.embedding_length"))
                .map(|v| v as usize);
            let audio_block_count = meta_u32(&model.metadata, "clip.audio.block_count")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.audio.block_count"))
                .map(|v| v as usize);
            let audio_embedding_length = meta_u32(&model.metadata, "clip.audio.embedding_length")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.audio.embedding_length"))
                .map(|v| v as usize);
            let audio_num_mel_bins = meta_u32(&model.metadata, "clip.audio.num_mel_bins")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.audio.num_mel_bins"))
                .map(|v| v as usize);
            let vision_hidden_dim = meta_u32(&model.metadata, "clip.vision.embedding_length")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.embedding_length"))
                .map(|v| v as usize);
            let vision_patch_size = meta_u32(&model.metadata, "clip.vision.patch_size")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.patch_size"))
                .map(|v| v as usize);
            let vision_image_size = meta_u32(&model.metadata, "clip.vision.image_size")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.image_size"))
                .map(|v| v as usize);
            let vision_num_layers = meta_u32(&model.metadata, "clip.vision.block_count")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.block_count"))
                .map(|v| v as usize);
            let vision_num_heads = meta_u32(&model.metadata, "clip.vision.attention.head_count")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.attention.head_count"))
                .map(|v| v as usize);
            let vision_projection_dim = meta_u32(&model.metadata, "clip.vision.projection_dim")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.projection_dim"))
                .map(|v| v as usize);
            let spatial_merge_size = meta_u32(&model.metadata, "clip.vision.spatial_merge_size")
                .or_else(|| meta_u32(&mmproj_metadata, "clip.vision.spatial_merge_size"))
                .map(|v| v as usize);

            let is_deepstack_layers = match model.metadata.get("clip.vision.is_deepstack_layers")
                .or_else(|| mmproj_metadata.get("clip.vision.is_deepstack_layers")) {
                Some(candle_core::quantized::gguf_file::Value::Array(arr)) => {
                    Some(arr.iter().map(|v| match v {
                        candle_core::quantized::gguf_file::Value::Bool(b) => *b,
                        _ => false,
                    }).collect())
                }
                _ => None,
            };

            let projector_type = match model.metadata.get("clip.vision.projector_type")
                .or_else(|| mmproj_metadata.get("clip.vision.projector_type"))
                .or_else(|| model.metadata.get("clip.projector_type"))
                .or_else(|| mmproj_metadata.get("clip.projector_type")) {
                Some(candle_core::quantized::gguf_file::Value::String(s)) => Some(s.clone()),
                _ => None,
            };

            let shared_kv_layers = meta_u32_agnostic(&model.metadata, "attention.shared_kv_layers")
                .map(|v| v as usize);

            let sliding_window_pattern = match find_meta_key(&model.metadata, "attention.sliding_window_pattern") {
                Some(key) => match model.metadata.get(&key) {
                    Some(candle_core::quantized::gguf_file::Value::Array(arr)) => {
                        Some(arr.iter().map(|v| match v {
                            candle_core::quantized::gguf_file::Value::Bool(b) => *b,
                            _ => false,
                        }).collect())
                    }
                    _ => None,
                },
                None => None,
            };

            let sliding_window = meta_u32_agnostic(&model.metadata, "attention.sliding_window")
                .map(|v| v as usize);

            let key_length = meta_u32_agnostic(&model.metadata, "attention.key_length")
                .map(|v| v as usize);

            let key_length_swa = meta_u32_agnostic(&model.metadata, "attention.key_length_swa")
                .map(|v| v as usize);

            let rope_theta_swa = meta_f32_agnostic(&model.metadata, "rope.freq_base_swa");

            let final_logit_softcapping = meta_f32_agnostic(&model.metadata, "final_logit_softcapping");

            let ple_dim = meta_u32_agnostic(&model.metadata, "embedding_length_per_layer_input")
                .map(|v| v as usize);

            let is_gemma = arch == "gemma" || arch == "gemma2" || arch == "gemma4";

            // Detect activation function from GGUF metadata first, then fall back to arch heuristic.
            // GGUF key: "<arch>.feed_forward_type" (e.g. "llama.feed_forward_type" = "SiLU")
            let feed_forward_type = find_meta_key(&model.metadata, "feed_forward_type")
                .and_then(|k| model.metadata.get(&k))
                .and_then(|v| if let candle_core::quantized::gguf_file::Value::String(s) = v { Some(s.to_lowercase()) } else { None });
            let hidden_act = match feed_forward_type.as_deref() {
                Some(s) if s.contains("gelu") => HiddenAct::GeLU,
                Some(s) if s.contains("silu") || s.contains("swiglu") => HiddenAct::SiLU,
                // Fallback: Gemma family uses GeLU, everything else (LLaMA/Qwen/Mistral) uses SiLU.
                _ => if is_gemma { HiddenAct::GeLU } else { HiddenAct::SiLU },
            };

            // Detect tied embeddings from GGUF tensor presence rather than arch name.
            // If neither "output.weight" nor "lm_head.weight" exists as a tensor, the model
            // uses tied embeddings (shares token_embd.weight for both input and output projection).
            let has_dedicated_lm_head = model.tensor_infos.contains_key("output.weight")
                || model.tensor_infos.contains_key("lm_head.weight")
                || model.tensor_infos.contains_key("output_norm.weight") && model.tensor_infos.contains_key("output.weight");
            // For Gemma, always tie (embed_scale requires the shared weight).
            let tie_word_embeddings = is_gemma || !has_dedicated_lm_head;

            let embed_scale = if is_gemma {
                Some((hidden_dim as f32).sqrt())
            } else {
                None
            };

            // Model-agnostic chat template - loaded directly from GGUF metadata
            let chat_template = match model.metadata.get("tokenizer.chat_template") {
                Some(candle_core::quantized::gguf_file::Value::String(s)) => Some(s.clone()),
                _ => None,
            };

            // EOS token string for chat formatting (e.g. "<|im_end|>" or "<end_of_turn>")
            let eos_token_str = if is_gemma {
                Some("<end_of_turn>".to_string())
            } else {
                // Try to infer from chat_template
                chat_template.as_deref().and_then(|t| {
                    if t.contains("<|im_end|>") {
                        Some("<|im_end|>".to_string())
                    } else if t.contains("</s>") {
                        Some("</s>".to_string())
                    } else {
                        None
                    }
                })
            };

            ModelMeta {
                vocab_size,
                hidden_dim,
                n_layers,
                n_heads,
                n_kv_heads,
                head_dim,
                intermediate_dim,
                max_seq_len,
                rope_theta,
                weight_dtype: WeightDtype::F32,
                rms_norm_eps,
                tie_word_embeddings,
                hidden_act,
                no_rope_layers: vec![false; n_layers],
                has_vision_encoder,
                vision_hidden_dim,
                vision_patch_size,
                vision_image_size,
                vision_num_layers,
                vision_num_heads,
                vision_projection_dim,
                spatial_merge_size,
                is_deepstack_layers,
                projector_type,
                has_audio_encoder,
                audio_hidden_dim,
                audio_block_count,
                audio_embedding_length,
                audio_num_mel_bins,
                shared_kv_layers,
                sliding_window_pattern,
                sliding_window,
                key_length,
                key_length_swa,
                rope_theta_swa,
                final_logit_softcapping,
                is_gemma,
                ple_dim,
                embed_scale,
                arch: arch.to_string(),
                chat_template,
                eos_token_str,
            }
        } else {
            let config_path = path.join("config.json");
            let mut meta = parse_config(&config_path)
                .context("Failed to parse config.json in CandleBackend")?;

            let eos_token_id = {
                let mut file = std::fs::File::open(&config_path)?;
                let mut contents = String::new();
                use std::io::Read;
                file.read_to_string(&mut contents)?;
                let v: serde_json::Value = serde_json::from_str(&contents)?;
                v.get("eos_token_id")
                    .and_then(|id| id.as_u64())
                    .map(|id| id as u32)
                    .unwrap_or(2)
            };
            self.eos_token_id = eos_token_id;

            let mut sf_files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                        sf_files.push(p);
                    }
                }
            }
            if sf_files.is_empty() {
                return Err(anyhow!("No safetensors files found in {:?}", path));
            }

            // Load all SafeTensors tensors to CPU memory in compute_dtype first
            let mut cpu_tensors = HashMap::new();
            for sf_file in sf_files {
                let loaded = candle_core::safetensors::load(&sf_file, &Device::Cpu)?;
                for (name, tensor) in loaded {
                    let casted = tensor.to_dtype(self.compute_dtype)?;
                    cpu_tensors.insert(name, casted);
                }
            }

            // Sort weight names layer-wise to prioritize caching the earlier layers
            let mut names: Vec<String> = cpu_tensors.keys().cloned().collect();
            names.sort_by(|a, b| {
                let get_layer = |name: &str| -> usize {
                    if name.starts_with("model.layers.") {
                        let remain = &name["model.layers.".len()..];
                        if let Some(pos) = remain.find('.') {
                            if let Ok(idx) = remain[..pos].parse::<usize>() {
                                return idx;
                            }
                        }
                    }
                    if name == "model.embed_tokens.weight" {
                        0
                    } else {
                        999 // put other weights last
                    }
                };
                get_layer(a).cmp(&get_layer(b))
            });

            let mut simulated_used = 0u64;
            for name in names {
                if let Some(t) = cpu_tensors.remove(&name) {
                    let tensor_bytes = t.shape().elem_count() as u64 * 2;
                    if simulated_used + tensor_bytes <= self.gpu_cache_budget {
                        // Move to GPU
                        match t.to_device(&self.device) {
                            Ok(gpu_t) => {
                                weights.insert(name.clone(), gpu_t);
                                simulated_used += tensor_bytes;
                                self.gpu_cache_bytes.fetch_add(tensor_bytes, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(_) => {
                                // Fallback to CPU if GPU transfer fails
                                weights.insert(name.clone(), t);
                            }
                        }
                    } else {
                        // Keep on CPU
                        weights.insert(name.clone(), t);
                    }
                }
            }

            meta.weight_dtype = WeightDtype::F16;
            // Defaults for safetensors path - try to read tokenizer_config.json for chat_template
            if meta.arch.is_empty() {
                meta.arch = "unknown".to_string();
            }
            if meta.chat_template.is_none() {
                // Try to load chat_template from tokenizer_config.json alongside model
                let tc_path = path.join("tokenizer_config.json");
                if let Ok(tc_content) = std::fs::read_to_string(&tc_path) {
                    if let Ok(tc_json) = serde_json::from_str::<serde_json::Value>(&tc_content) {
                        if let Some(tmpl) = tc_json.get("chat_template").and_then(|v| v.as_str()) {
                            meta.chat_template = Some(tmpl.to_string());
                        }
                    }
                }
            }
            meta
        };

        // Build graph from ALL known weight names (GGUF: from quantized_weights)
        let names: Vec<String> = if weights.is_empty() {
            quantized_weights.keys().cloned().collect()
        } else {
            weights.keys().cloned().collect()
        };
        let group = scan_tensors(&names);
        let graph = build_graph(&meta, &group);
        tracing::info!("Compute graph built: {} operators", graph.ops.len());

        let kv_cache = RawKvCache::new();

        self.kv_config = KvCacheConfig {
            n_layers: meta.n_layers,
            n_kv_heads: meta.n_kv_heads,
            head_dim: meta.head_dim,
            block_size: 16,
            dtype: match std::env::var("LLM_KV_DTYPE").as_deref() {
                Ok("q8") | Ok("Q8") => KvDtype::Q8,
                Ok("q4") | Ok("Q4") => KvDtype::Q4,
                _ => KvDtype::F16,
            },
        };

        self.weights = weights;
        self.quantized_weights = quantized_weights;
        self.graph = Some(graph);
        self.meta = Some(meta.clone());
        self.kv_cache = Some(Mutex::new(kv_cache));

        // Eagerly pre-dequantize all weights in parallel/sequence into deq_cache to eliminate TTFT latency
        if !self.quantized_weights.is_empty() {
            if self.explicit_dequantize {
                println!("Eagerly dequantizing and caching {} GGUF weights...", self.quantized_weights.len());
                let mut local_cache = HashMap::new();
                let mut simulated_used = 0u64;
                
                // Sort weight names layer-wise to prioritize caching the earlier layers
                let mut names: Vec<String> = self.quantized_weights.keys().cloned().collect();
                names.sort_by(|a, b| {
                    let get_layer = |name: &str| -> usize {
                        if name.starts_with("model.layers.") {
                            let remain = &name["model.layers.".len()..];
                            if let Some(pos) = remain.find('.') {
                                if let Ok(idx) = remain[..pos].parse::<usize>() {
                                    return idx;
                                }
                            }
                        }
                        if name == "model.embed_tokens.weight" {
                            0
                        } else {
                            999 // put other weights (like lm_head or output norm) last
                        }
                    };
                    get_layer(a).cmp(&get_layer(b))
                });

                for name in &names {
                    if let Some(qt) = self.quantized_weights.get(name) {
                        let tensor_bytes = qt.shape().elem_count() as u64 * 2;
                        if simulated_used + tensor_bytes <= self.gpu_cache_budget {
                            simulated_used += tensor_bytes;
                            if let Err(e) = self.get_weight(name, &mut local_cache) {
                                println!("Warning: Eager dequantize failed for {}: {:?}", name, e);
                                // Fallback to CPU dequantize if GPU fails
                                if let Ok(t) = qt.dequantize(&Device::Cpu) {
                                    if let Ok(t) = t.to_dtype(self.compute_dtype) {
                                        local_cache.insert(name.clone(), t);
                                    }
                                }
                            }
                        } else {
                            // Dequantize to CPU directly for offloaded weights (never touch GPU)
                            if let Ok(t) = qt.dequantize(&Device::Cpu) {
                                if let Ok(t) = t.to_dtype(self.compute_dtype) {
                                    local_cache.insert(name.clone(), t);
                                }
                            } else {
                                println!("Warning: Eager CPU dequantize failed for {}", name);
                            }
                        }
                    }
                }
                println!("Eager dequantization loop done. local_cache len: {}. model.embed_tokens.weight cached: {}", local_cache.len(), local_cache.contains_key("model.embed_tokens.weight"));
                // Now transfer all local_cache entries to the Mutex deq_cache
                { let mut cache = self.deq_cache.lock();
                    *cache = local_cache.clone();
                }
                
                let mut gpu_count = 0;
                let mut gpu_bytes = 0u64;
                let mut cpu_count = 0;
                let mut cpu_bytes = 0u64;

                for tensor in local_cache.values() {
                    let bytes = tensor.shape().elem_count() as u64 * 2; // F16 = 2 bytes
                    if tensor.device().is_cuda() {
                        gpu_count += 1;
                        gpu_bytes += bytes;
                    } else {
                        cpu_count += 1;
                        cpu_bytes += bytes;
                    }
                }

                println!("Eager dequantization finished.");
                println!("  - GPU (CUDA): {} weights ({:.2} GB / {:.2} MB)", gpu_count, gpu_bytes as f64 / 1e9, gpu_bytes as f64 / (1024.0 * 1024.0));
                println!("  - CPU (Offloaded): {} weights ({:.2} GB / {:.2} MB)", cpu_count, cpu_bytes as f64 / 1e9, cpu_bytes as f64 / (1024.0 * 1024.0));
                if cpu_count > 0 {
                    println!("Note: Model is not fully loadable on CUDA. Offloaded {:.1}% of weights to CPU.", (cpu_count as f64 / (gpu_count + cpu_count) as f64) * 100.0);
                } else {
                    println!("Success: Model is fully loaded on CUDA.");
                }
            } else {
                println!("Eagerly caching quantized weights and dequantizing embedding/norm weights...");
                let mut local_deq = HashMap::new();
                let mut local_qmatmul = HashMap::new();
                
                let keys: Vec<String> = self.quantized_weights.keys().cloned().collect();
                for name in keys {
                    if let Some(qt) = self.quantized_weights.remove(&name) {
                        let name_lower = name.to_lowercase();
                        let is_embedding = name_lower.contains("embed") || name_lower.contains("embd");
                        let is_non_matmul = is_embedding
                            || name_lower.contains("norm") 
                            || name_lower.contains("bias") 
                            || name_lower.contains("scale")
                            || name_lower.contains("freq")
                            || name_lower.contains("rotors")
                            || name_lower.contains("rot");

                        if is_non_matmul {
                            let target_dev = qt.device().clone();
                            match qt.dequantize(&target_dev) {
                                Ok(t) => {
                                    // Keep as f32 — f16 can overflow (~65504) for large
                                    // residual activations (e.g. Qwen 2.5 reaches ~43k by layer 0).
                                    local_deq.insert(name.clone(), t);
                                }
                                Err(e) => {
                                    println!("Error: Failed to dequantize tensor {} on CPU: {:?}", name, e);
                                }
                            }
                        } else {
                            // qt is already loaded on self.device, directly initialize QMatMul
                            match QMatMul::from_qtensor(qt) {
                                Ok(qmatmul) => {
                                    local_qmatmul.insert(name.clone(), qmatmul);
                                }
                                Err(e) => {
                                    println!("Warning: QMatMul::from_qtensor failed for {}: {:?}", name, e);
                                }
                            }
                        }
                    }
                }
                { let mut cache = self.deq_cache.lock();
                    *cache = local_deq;
                }
                { let mut cache = self.qmatmul_cache.lock();
                    *cache = local_qmatmul;
                }
                let gpu_deq_bytes = self.deq_cache.lock().values().map(|t| t.shape().elem_count() as u64 * 2).sum::<u64>();
                println!("Quantized weight loading finished.");
                println!("  - Cached standard GPU/device tensors (embed/norm): {} weights ({:.2} MB)", 
                         self.deq_cache.lock().len(), 
                         gpu_deq_bytes as f64 / (1024.0 * 1024.0));
                println!("  - Cached QMatMul layers: {} weights", self.qmatmul_cache.lock().len());
            }
        }

        if meta.has_vision_encoder {
            // Model-agnostic mmproj discovery: strip the last extension component
            // and try common multimodal projection suffixes
            let vision_path = if is_gguf {
                let base = path.to_string_lossy();
                // Strip any quantisation suffix (e.g. .Q4_K_M, .Q8_0, .BF16, .F16)
                let stem = if let Some(pos) = base.rfind('.') {
                    // Find the base name up to the quantisation marker
                    let without_ext = &base[..pos]; // remove .gguf
                    // Remove quantisation tag if present (e.g. .Q4_K_M)
                    if let Some(q_pos) = without_ext.rfind('.') {
                        without_ext[..q_pos].to_string()
                    } else {
                        without_ext.to_string()
                    }
                } else {
                    base.to_string()
                };
                let suffixes = [
                    "-mmproj-f16.gguf",
                    ".mmproj.gguf",
                    "-mmproj.gguf",
                    ".BF16-mmproj.gguf",
                    "-mmproj-f32.gguf",
                ];
                let mut found = None;
                for suffix in &suffixes {
                    let candidate = format!("{}{}", stem, suffix);
                    if std::path::Path::new(&candidate).exists() {
                        found = Some(std::path::PathBuf::from(candidate));
                        break;
                    }
                }
                // Also try same directory pattern matching
                if found.is_none() {
                    if let Some(parent) = path.parent() {
                        if let Ok(entries) = std::fs::read_dir(parent) {
                            for entry in entries.flatten() {
                                let n = entry.file_name().to_string_lossy().to_lowercase();
                                if n.contains("mmproj") && n.ends_with(".gguf") {
                                    found = Some(entry.path());
                                    break;
                                }
                            }
                        }
                    }
                }
                found.unwrap_or_else(|| path.to_path_buf())
            } else {
                path.to_path_buf()
            };
            tracing::info!("Attempting to load VisionEncoder from {:?}", vision_path);
            match VisionEncoder::load(&vision_path, &dev) {
                Ok(enc) => { self.vision_encoder = Some(enc); }
                Err(e) => { tracing::warn!("VisionEncoder load skipped: {e}"); }
            }
        }

        if meta.has_audio_encoder {
            let audio_path = if is_gguf {
                let base = path.to_string_lossy();
                let stem = if let Some(pos) = base.rfind('.') {
                    let without_ext = &base[..pos]; // remove .gguf
                    if let Some(q_pos) = without_ext.rfind('.') {
                        without_ext[..q_pos].to_string()
                    } else {
                        without_ext.to_string()
                    }
                } else {
                    base.to_string()
                };
                let suffixes = [
                    "-mmproj-f16.gguf",
                    ".mmproj.gguf",
                    "-mmproj.gguf",
                    ".BF16-mmproj.gguf",
                    "-mmproj-f32.gguf",
                ];
                let mut found = None;
                for suffix in &suffixes {
                    let candidate = format!("{}{}", stem, suffix);
                    if std::path::Path::new(&candidate).exists() {
                        found = Some(std::path::PathBuf::from(candidate));
                        break;
                    }
                }
                if found.is_none() {
                    if let Some(parent) = path.parent() {
                        if let Ok(entries) = std::fs::read_dir(parent) {
                            for entry in entries.flatten() {
                                let n = entry.file_name().to_string_lossy().to_lowercase();
                                if n.contains("mmproj") && n.ends_with(".gguf") {
                                    found = Some(entry.path());
                                    break;
                                }
                            }
                        }
                    }
                }
                found.unwrap_or_else(|| path.to_path_buf())
            } else {
                path.to_path_buf()
            };
            tracing::info!("Attempting to load AudioEncoder from {:?}", audio_path);
            match AudioEncoder::load(&audio_path, &dev) {
                Ok(enc) => { self.audio_encoder = Some(enc); }
                Err(e) => { tracing::warn!("AudioEncoder load skipped: {e}"); }
            }
        }

        Ok(meta)
    }

    fn forward_pass(&self, batch: &BatchInput) -> Result<BatchOutput> {
        let graph = self.graph.as_ref().ok_or_else(|| anyhow!("Compute graph not built"))?;
        let meta = self.meta.as_ref().ok_or_else(|| anyhow!("Model metadata not loaded"))?;
        let kv_cache_mutex = self.kv_cache.as_ref().ok_or_else(|| anyhow!("KV Cache not initialized"))?;
        let mut kv_cache = kv_cache_mutex.lock();

        if batch.seq_ids[0] == 1 && kv_cache.get_seq_len(batch.seq_ids[0]) == 0 {
            for (idx, op) in graph.ops.iter().enumerate().take(60) {
                println!("Graph OP idx={}: {:?}", idx, op);
            }
        }

        // Initialize execution context
        let mut ctx = ExecContext::new();

        // Create a local clone of the deq_cache to avoid lock contention on every weight resolution
        let mut local_cache = {
            let cache = self.deq_cache.lock();
            cache.clone()
        };

        // Feed input_ids to context
        let dev = self.device.clone();
        let tokens_t = Tensor::new(batch.token_ids.as_slice(), &dev)?;
        let b_sz = batch.seq_ids.len();
        let seq_len = batch.token_ids.len() / b_sz;
        let tokens_t = tokens_t.reshape((b_sz, seq_len))?;
        ctx.insert("input_ids".to_string(), tokens_t);

        // Feed pixel_values if vision encoder is present
        if meta.has_vision_encoder {
            let img_size = self.vision_encoder.as_ref().map(|e| e.image_size).unwrap_or(768);
            let active_path = crate::backends::ACTIVE_IMAGE_PATH.lock().clone();
            let mut cache = self.last_pixel_values.lock();
            let mut last_path = self.last_image_path.lock();
            let needs_load = match (&active_path, &*last_path, &*cache) {
                (Some(curr), Some(prev), Some(_)) if curr == prev => false,
                _ => true,
            };
            let pixel_val = if needs_load {
                let vision_dtype = self.compute_dtype;
                let loaded = if let Some(ref path_str) = active_path {
                    match crate::backends::vision::load_image(Path::new(path_str), img_size, &dev) {
                        Ok(t) => t.to_dtype(vision_dtype)?,
                        Err(e) => {
                            tracing::warn!("Failed to load image from {path_str}: {e}, using zeros");
                            Tensor::zeros((1, 3, img_size, img_size), vision_dtype, &dev)?
                        }
                    }
                } else {
                    Tensor::zeros((1, 3, img_size, img_size), vision_dtype, &dev)?
                };
                *cache = Some(loaded.clone());
                *last_path = active_path.clone();
                loaded
            } else {
                cache.as_ref().unwrap().clone()
            };
            ctx.insert("pixel_values".to_string(), pixel_val);
        }

        // Feed audio_values if audio encoder is present
        if meta.has_audio_encoder {
            let active_path = crate::backends::ACTIVE_AUDIO_PATH.lock().clone();
            let mut cache = self.last_audio_values.lock();
            let mut last_path = self.last_audio_path.lock();
            let needs_load = match (&active_path, &*last_path, &*cache) {
                (Some(curr), Some(prev), Some(_)) if curr == prev => false,
                _ => true,
            };
            let audio_val = if needs_load {
                let audio_dtype = self.compute_dtype;
                let loaded = if let Some(ref path_str) = active_path {
                    match crate::backends::audio::load_audio(Path::new(path_str), &dev) {
                        Ok(t) => t.to_dtype(audio_dtype)?,
                        Err(e) => {
                            tracing::warn!("Failed to load audio from {path_str}: {e}, using zeros");
                            Tensor::zeros((1, 128, 3000), audio_dtype, &dev)?
                        }
                    }
                } else {
                    Tensor::zeros((1, 128, 3000), audio_dtype, &dev)?
                };
                *cache = Some(loaded.clone());
                *last_path = active_path.clone();
                loaded
            } else {
                cache.as_ref().unwrap().clone()
            };
            ctx.insert("audio_values".to_string(), audio_val);
        }

        // Compute last use index for each activation tensor to free VRAM immediately
        let mut last_use: HashMap<String, usize> = HashMap::new();
        for (idx, op) in graph.ops.iter().enumerate() {
            let inputs = match op {
                Operator::Embed { input_ids, .. } => vec![input_ids.clone()],
                Operator::RMSNorm { input, .. } => vec![input.clone()],
                Operator::MatMul { input, .. } => vec![input.clone()],
                Operator::Rope { q, k, .. } => vec![q.clone(), k.clone()],
                Operator::RopeQ { q, .. } => vec![q.clone()],
                Operator::RopeSkip { q, k, .. } => vec![q.clone(), k.clone()],
                Operator::PagedAttention { q, k, v, .. } => vec![q.clone(), k.clone(), v.clone()],
                Operator::Activation { input, .. } => vec![input.clone()],
                Operator::Mul { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
                Operator::Add { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
                Operator::VisualEmbed { pixel_values, .. } => vec![pixel_values.clone()],
                Operator::SpliceTensors { text_embeds, visual_embeds, .. } => vec![text_embeds.clone(), visual_embeds.clone()],
                Operator::AudioEmbed { audio_values, .. } => vec![audio_values.clone()],
                Operator::SpliceAudioTensors { text_embeds, audio_embeds, .. } => vec![text_embeds.clone(), audio_embeds.clone()],
                Operator::DeepStackFuse { input, .. } => vec![input.clone(), "visual_embeddings".to_string()],
                Operator::Softcap { input, .. } => vec![input.clone()],
                Operator::Scale { input, .. } => vec![input.clone()],
                Operator::TensorScale { input, .. } => vec![input.clone()],
                Operator::PleInput { input_ids, text_embeddings, .. } => vec![input_ids.clone(), text_embeddings.clone()],
                Operator::PleLayer { input, per_layer_input, .. } => vec![input.clone(), per_layer_input.clone()],
            };
            for input in inputs {
                last_use.insert(input, idx);
            }
        }

        // Execute operators sequentially
        for (idx, op) in graph.ops.iter().enumerate() {
            /*
            if batch.seq_ids[0] == 1 && kv_cache.get_seq_len(batch.seq_ids[0]) == 0 {
                use std::io::Write;
                println!("Executing OP idx={}/{}: {:?}", idx, graph.ops.len(), op);
                std::io::stdout().flush().ok();
            }
            */
            match op {
                Operator::Embed { input_ids, weight, output } => {
                    let ids = ctx.get(input_ids)?;
                    let table = self.get_weight(weight, &mut local_cache)?;
                    let (b_sz, seq_len) = ids.dims2()?;
                    let out = if table.device().is_cpu() {
                        let ids_cpu = ids.to_device(&Device::Cpu)?;
                        let ids_flat = ids_cpu.flatten_all()?;
                        let out_flat = table.index_select(&ids_flat, 0)?;
                        let out_cpu = out_flat.reshape((b_sz, seq_len, table.dim(1)?))?;
                        out_cpu.to_device(&self.device)?
                    } else {
                        let ids_flat = ids.flatten_all()?;
                        let out_flat = table.index_select(&ids_flat, 0)?;
                        out_flat.reshape((b_sz, seq_len, table.dim(1)?))?
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::RMSNorm { input, weight, output, eps } => {
                    let in_t = ctx.get(input)?;
                    let w_t = self.get_weight(weight, &mut local_cache)?.to_device(in_t.device())?;
                    let is_gemma = self.meta.as_ref().map(|m| m.is_gemma).unwrap_or(false);
                    let is_gemma_hf = is_gemma && !self.weights.is_empty();
                    let out = RmsNorm::new(w_t, *eps as f64, is_gemma_hf).forward(&in_t)?;
                    ctx.insert(output.clone(), out);
                }
                Operator::MatMul { input, weight, bias, output } => {
                    let in_t = ctx.get(input)?;
                    let qmatmul_opt = if self.explicit_dequantize {
                        None
                    } else {
                        let cache = self.qmatmul_cache.lock();
                        cache.get(weight).cloned()
                    };

                    let out = if let Some(qmatmul) = qmatmul_opt {
                        let in_t_f32 = in_t.to_dtype(DType::F32)?;
                        // Keep output as f32 — do NOT downcast back to f16.
                        // f16 overflows at ~65504; Qwen 2.5 residuals reach ~43k by layer 0
                        // and easily overflow by layer 4, producing NaN that poisons all
                        // subsequent layers. Staying in f32 costs 2x memory but is correct.
                        qmatmul.forward(&in_t_f32)?
                    } else {
                        let w_t = self.get_weight(weight, &mut local_cache)?.to_device(in_t.device())?;

                        // Auto-transpose: weight stored row-major [out_features, in_features]
                        let last_dim = in_t.dim(in_t.rank() - 1)?;
                        let w_t_final = if w_t.rank() == 2 && last_dim == w_t.dim(1)? {
                            w_t.transpose(0, 1)?
                        } else {
                            w_t
                        };

                        let rank_in = in_t.rank();
                        if rank_in == 3 {
                            let (b, m, k) = in_t.dims3()?;
                            let in_t_2d = in_t.reshape((b * m, k))?;
                            let res_2d = in_t_2d.matmul(&w_t_final)?;
                            let n = res_2d.dim(1)?;
                            res_2d.reshape((b, m, n))?
                        } else {
                            in_t.matmul(&w_t_final)?
                        }
                    };

                    let mut out = out;
                    if let Some(bias_name) = bias {
                        let b_t = self.get_weight(bias_name, &mut local_cache)?.to_device(in_t.device())?;
                        out = out.broadcast_add(&b_t)?;
                    }

                    ctx.insert(output.clone(), out);
                }
                Operator::Rope { q, k, output_q, output_k, layer_idx, rope_theta } => {
                    let q_t = ctx.get(q)?;
                    let k_t = ctx.get(k)?;
                    let (b_sz, seq_len, q_dim) = q_t.dims3()?;
                    let (_, _, k_dim) = k_t.dims3()?;

                    let head_dim = meta.get_head_dim(*layer_idx);
                    let layer_n_heads = q_dim / head_dim;
                    let layer_n_kv_heads = k_dim / head_dim;
                    let layer_head_dim = head_dim;
                    let layer_k_head_dim = head_dim;

                    let q_4d = q_t.reshape((b_sz, seq_len, layer_n_heads, layer_head_dim))?;
                    let k_4d = k_t.reshape((b_sz, seq_len, layer_n_kv_heads, layer_k_head_dim))?;
                    let (q_out_4d, k_out_4d) = apply_rope(&q_4d, &k_4d, batch, &kv_cache, *rope_theta)?;
                    let q_out = q_out_4d.reshape((b_sz, seq_len, layer_n_heads * layer_head_dim))?;
                    let k_out = k_out_4d.reshape((b_sz, seq_len, layer_n_kv_heads * layer_k_head_dim))?;
                    ctx.insert(output_q.clone(), q_out);
                    ctx.insert(output_k.clone(), k_out);
                }
                Operator::RopeQ { q, output_q, layer_idx, rope_theta } => {
                    let q_t = ctx.get(q)?;
                    let (b_sz, seq_len, q_dim) = q_t.dims3()?;

                    let head_dim = meta.get_head_dim(*layer_idx);
                    let layer_n_heads = q_dim / head_dim;
                    let layer_head_dim = head_dim;

                    let q_4d = q_t.reshape((b_sz, seq_len, layer_n_heads, layer_head_dim))?;
                    let q_out_4d = apply_rope_q(&q_4d, batch, &kv_cache, *rope_theta)?;
                    let q_out = q_out_4d.reshape((b_sz, seq_len, layer_n_heads * layer_head_dim))?;
                    ctx.insert(output_q.clone(), q_out);
                }
                Operator::RopeSkip { q, k, output_q, output_k } => {
                    let q_t = ctx.get(q)?;
                    let k_t = ctx.get(k)?;
                    ctx.insert(output_q.clone(), q_t.clone());
                    ctx.insert(output_k.clone(), k_t.clone());
                }
                Operator::PagedAttention { q, k, v, output, layer_idx, n_heads: _, n_kv_heads: _, head_dim: _ } => {
                    let q_t = ctx.get(q)?;

                    let (b_sz, seq_len, q_dim) = q_t.dims3()?;
                    
                    let head_dim = meta.get_head_dim(*layer_idx);
                    let layer_n_heads = q_dim / head_dim;
                    let layer_head_dim = head_dim;
                    
                    let q_4d = q_t.reshape((b_sz, seq_len, layer_n_heads, layer_head_dim))?;

                    let is_shared = meta.is_kv_shared(*layer_idx);
                    let q_dev = q_4d.device();

                    {
                        let mut seq_blocks_map = self.seq_blocks.lock();
                        for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
                            seq_blocks_map.insert(seq_id, batch.block_tables[i].clone());
                        }
                    }

                    let is_quantized = self.kv_config.dtype == KvDtype::Q8 || self.kv_config.dtype == KvDtype::Q4;
                    let r_tensor = if is_quantized {
                        let r_vec = generate_hadamard_orthogonal(head_dim);
                        Some(Tensor::from_vec(r_vec, (head_dim, head_dim), q_dev)?.to_dtype(q_4d.dtype())?)
                    } else {
                        None
                    };

                    let q_4d_rotated = if let Some(ref r) = r_tensor {
                        q_4d.reshape(((), head_dim))?.matmul(r)?.reshape(q_4d.dims())?
                    } else {
                        q_4d.clone()
                    };

                    let (k_4d, v_4d, n_kv_heads) = if !is_shared {
                        let k_t = ctx.get(k)?;
                        let v_t = ctx.get(v)?;
                        let (_, _, k_dim) = k_t.dims3()?;
                        let n_kv_heads = k_dim / head_dim;
                        let k_4d = k_t.reshape((b_sz, seq_len, n_kv_heads, layer_head_dim))?;
                        let mut v_4d = v_t.reshape((b_sz, seq_len, n_kv_heads, layer_head_dim))?;
                        if meta.ple_dim.is_some() {
                            v_4d = rms_norm_no_scale(&v_4d, meta.rms_norm_eps as f64)?;
                        }
                        
                        let k_4d_rotated = if let Some(ref r) = r_tensor {
                            k_4d.reshape(((), head_dim))?.matmul(r)?.reshape(k_4d.dims())?
                        } else {
                            k_4d
                        };
                        (Some(k_4d_rotated), Some(v_4d), n_kv_heads)
                    } else {
                        (None, None, meta.n_kv_heads)
                    };

                    let mut gpu_cache = self.gpu_kv_cache.lock();
                    let block_size = self.kv_config.block_size;
                    let mut att_outputs = Vec::with_capacity(b_sz);

                    for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
                        let block_table = &batch.block_tables[i];
                        let offset = kv_cache.get_seq_len(seq_id);
                        
                        if !is_shared {
                            let k_4d_val = k_4d.as_ref().unwrap();
                            let v_4d_val = v_4d.as_ref().unwrap();
                            let k_i = k_4d_val.narrow(0, i, 1)?;
                            let v_i = v_4d_val.narrow(0, i, 1)?;
                            
                            let mut t_start = 0;
                            while t_start < seq_len {
                                let abs_idx = offset + t_start;
                                let block_idx = abs_idx / block_size;
                                let start_block_offset = abs_idx % block_size;
                                let block_id = block_table[block_idx];
                                
                                let chunk_len = std::cmp::min(seq_len - t_start, block_size - start_block_offset);
                                let t_end = t_start + chunk_len;
                                
                                let k_chunk = k_i.narrow(1, t_start, chunk_len)?;
                                let v_chunk = v_i.narrow(1, t_start, chunk_len)?;
                                
                                let block_data = gpu_cache.entry((*layer_idx, block_id)).or_insert_with(|| {
                                    let dev = q_dev;
                                    let dtype = self.kv_config.dtype;
                                    let comp_dtype = self.compute_dtype;
                                    
                                    let (k_block, k_scale) = if dtype == KvDtype::Q8 || dtype == KvDtype::Q4 {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), DType::U8, dev).unwrap();
                                        let s = Tensor::zeros((1, block_size, n_kv_heads, 1), comp_dtype, dev).unwrap();
                                        (d, Some(s))
                                    } else {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), comp_dtype, dev).unwrap();
                                        (d, None)
                                    };
                                    
                                    let (v_block, v_scale) = if dtype == KvDtype::Q8 || dtype == KvDtype::Q4 {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), DType::U8, dev).unwrap();
                                        let s = Tensor::zeros((1, block_size, n_kv_heads, 1), comp_dtype, dev).unwrap();
                                        (d, Some(s))
                                    } else {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), comp_dtype, dev).unwrap();
                                        (d, None)
                                    };
                                    
                                    BlockData { k: k_block, k_scale, v: v_block, v_scale }
                                });
                                
                                if self.kv_config.dtype == KvDtype::Q8 || self.kv_config.dtype == KvDtype::Q4 {
                                    let (q_k, q_k_scale) = quantize(&k_chunk, self.kv_config.dtype, self.compute_dtype)?;
                                    let (q_v, q_v_scale) = quantize(&v_chunk, self.kv_config.dtype, self.compute_dtype)?;
                                    
                                    block_data.k = update_block_tensor(&block_data.k, &q_k, start_block_offset, chunk_len)?;
                                    block_data.k_scale = Some(update_block_tensor(block_data.k_scale.as_ref().unwrap(), &q_k_scale, start_block_offset, chunk_len)?);
                                    
                                    block_data.v = update_block_tensor(&block_data.v, &q_v, start_block_offset, chunk_len)?;
                                    block_data.v_scale = Some(update_block_tensor(block_data.v_scale.as_ref().unwrap(), &q_v_scale, start_block_offset, chunk_len)?);
                                } else {
                                    block_data.k = update_block_tensor(&block_data.k, &k_chunk, start_block_offset, chunk_len)?;
                                    block_data.v = update_block_tensor(&block_data.v, &v_chunk, start_block_offset, chunk_len)?;
                                }
                                
                                t_start = t_end;
                            }
                        }

                        let src_layer = if is_shared {
                            meta.get_kv_source_layer(*layer_idx)
                        } else {
                            *layer_idx
                        };

                        let num_active_blocks = std::cmp::min(
                            (offset + seq_len + block_size - 1) / block_size,
                            block_table.len()
                        );
                        let mut k_blocks = Vec::with_capacity(num_active_blocks);
                        let mut v_blocks = Vec::with_capacity(num_active_blocks);

                        for &block_id in &block_table[0..num_active_blocks] {
                            let block_data = gpu_cache.get(&(src_layer, block_id))
                                .ok_or_else(|| anyhow!("Block ID {} not found for layer {}", block_id, src_layer))?;
                            
                            let k_deq = if self.kv_config.dtype == KvDtype::Q8 || self.kv_config.dtype == KvDtype::Q4 {
                                dequantize(&block_data.k, block_data.k_scale.as_ref().unwrap(), self.kv_config.dtype, self.compute_dtype)?
                            } else {
                                block_data.k.clone()
                            };
                            
                            let v_deq = if self.kv_config.dtype == KvDtype::Q8 || self.kv_config.dtype == KvDtype::Q4 {
                                dequantize(&block_data.v, block_data.v_scale.as_ref().unwrap(), self.kv_config.dtype, self.compute_dtype)?
                            } else {
                                block_data.v.clone()
                            };

                            k_blocks.push(k_deq.squeeze(0)?);
                            v_blocks.push(v_deq.squeeze(0)?);
                        }

                        let mut k_hist_squeezed = Tensor::cat(&k_blocks, 0)?;
                        let mut v_hist_squeezed = Tensor::cat(&v_blocks, 0)?;

                        let total_len = offset + seq_len;
                        k_hist_squeezed = k_hist_squeezed.narrow(0, 0, total_len)?;
                        v_hist_squeezed = v_hist_squeezed.narrow(0, 0, total_len)?;

                        if let Some(window_len) = meta.get_sliding_window_len(*layer_idx) {
                            if total_len > window_len {
                                let start_idx = total_len - window_len;
                                k_hist_squeezed = k_hist_squeezed.narrow(0, start_idx, window_len)?;
                                v_hist_squeezed = v_hist_squeezed.narrow(0, start_idx, window_len)?;
                            }
                        }

                        let total_seq_len = k_hist_squeezed.dim(0)?;
                        let layer_n_kv_heads = k_hist_squeezed.dim(1)?;

                        let q_i = q_4d_rotated.narrow(0, i, 1)?.squeeze(0)?;
                        let q_i = q_i.transpose(0, 1)?.contiguous()?; // (n_heads, seq_len, head_dim)
                        
                        let n_rep = layer_n_heads / layer_n_kv_heads;
                        let k_hist_rep = repeat_kv(k_hist_squeezed.transpose(0, 1)?, n_rep)?.contiguous()?;
                        let v_hist_rep = repeat_kv(v_hist_squeezed.transpose(0, 1)?, n_rep)?.contiguous()?;

                        let scores = q_i.matmul(&k_hist_rep.transpose(1, 2)?.contiguous()?)?;
                        let scores_scaled = if meta.ple_dim.is_some() {
                            scores
                        } else {
                            (scores / (layer_head_dim as f64).sqrt())?
                        };

                        let num_tokens = seq_len;
                        let seq_len_before = total_seq_len - num_tokens;

                        let scores = if num_tokens > 1 {
                            let mut mask_vec = vec![0.0f32; num_tokens * total_seq_len];
                            for q_idx in 0..num_tokens {
                                for k_idx in 0..total_seq_len {
                                    if k_idx > seq_len_before + q_idx {
                                        mask_vec[q_idx * total_seq_len + k_idx] = f32::NEG_INFINITY;
                                    }
                                }
                            }
                            let mask = Tensor::from_vec(mask_vec, (1, num_tokens, total_seq_len), q_dev)?.to_dtype(scores_scaled.dtype())?;
                            scores_scaled.broadcast_add(&mask)?
                        } else {
                            scores_scaled
                        };

                        let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
                        let out_i = probs.matmul(&v_hist_rep)?;
                        let out_i = out_i.transpose(0, 1)?.contiguous()?; // (seq_len, n_heads, head_dim)
                        let out_i = out_i.reshape((seq_len, layer_n_heads * layer_head_dim))?;
                        att_outputs.push(out_i);
                    }

                    let att_out = Tensor::stack(&att_outputs, 0)?;
                    ctx.insert(output.clone(), att_out);
                }
                Operator::Activation { input, output, act } => {
                    let in_t = ctx.get(input)?;
                    let out = match act {
                        HiddenAct::SiLU => (&in_t * &candle_nn::ops::sigmoid(&in_t)?)?,
                        HiddenAct::GeLU => in_t.gelu()?,
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::Mul { lhs, rhs, output } => {
                    let lhs_t = ctx.get(lhs)?;
                    let rhs_t = ctx.get(rhs)?;
                    ctx.insert(output.clone(), lhs_t.broadcast_mul(&rhs_t)?);
                }
                Operator::Add { lhs, rhs, output } => {
                    let lhs_t = ctx.get(lhs)?;
                    let rhs_t = ctx.get(rhs)?;
                    ctx.insert(output.clone(), lhs_t.broadcast_add(&rhs_t)?);
                }
                Operator::VisualEmbed { pixel_values, output } => {
                    let active_path = crate::backends::ACTIVE_IMAGE_PATH.lock().clone();
                    let out = if let Some(ref enc) = self.vision_encoder {
                        let mut cache = self.visual_embeddings.lock();
                        let mut last_path = self.last_image_path.lock();
                        let needs_encode = match (&active_path, &*last_path, &*cache) {
                            (Some(curr), Some(prev), Some(_)) if curr == prev => false,
                            _ => true,
                        };
                        if needs_encode {
                            let p_val = ctx.get(pixel_values)?;
                            let encoded = enc.encode(&p_val)?;
                            *cache = Some(encoded.clone());
                            *last_path = active_path.clone();
                            encoded
                        } else {
                            cache.as_ref().unwrap().clone()
                        }
                    } else if let Some(ref preloaded) = *self.visual_embeddings.lock() {
                        preloaded.clone()
                    } else {
                        return Err(anyhow!("VisionEncoder / visual_embeddings not loaded for VisualEmbed"));
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::SpliceTensors { text_embeds, visual_embeds, output } => {
                    let t_emb = ctx.get(text_embeds)?;
                    let v_emb = ctx.get(visual_embeds)?;
                    // Unify dtypes: visual embeddings (may be F32 on CPU) must match text embedding dtype
                    let v_emb = v_emb.to_dtype(t_emb.dtype())?;
                    let v_emb_narrowed = if v_emb.dim(2)? > t_emb.dim(2)? {
                        v_emb.narrow(2, 0, t_emb.dim(2)?)?
                    } else {
                        v_emb.clone()
                    };
                    let token_ids = &batch.token_ids;
                    let out = splice_visual_embeddings(&t_emb, &v_emb_narrowed, token_ids, 0, 0)?;
                    ctx.insert(output.clone(), out);
                }
                Operator::AudioEmbed { audio_values, output } => {
                    let active_path = crate::backends::ACTIVE_AUDIO_PATH.lock().clone();
                    // If no audio is provided, emit a dummy embedding and skip encoding.
                    // This avoids running the conformer on zero tensors when the prompt is text-only.
                    let out = if active_path.is_none() {
                        // Dummy: (1, 1, embed_dim) so downstream SpliceAudioTensors is a no-op
                        let t_emb = ctx.get("text_embeddings").or_else(|_| ctx.get("spliced_visual_embeddings"))?;
                        let embed_dim = t_emb.dim(2)?;
                        Tensor::zeros((1, 1, embed_dim), t_emb.dtype(), t_emb.device())?
                    } else if let Some(ref enc) = self.audio_encoder {
                        let mut cache = self.audio_embeddings.lock();
                        let mut last_path = self.last_audio_path.lock();
                        let needs_encode = match (&active_path, &*last_path, &*cache) {
                            (Some(curr), Some(prev), Some(_)) if curr == prev => false,
                            _ => true,
                        };
                        if needs_encode {
                            let a_val = ctx.get(audio_values)?;
                            let encoded = enc.encode(&a_val)?;
                            *cache = Some(encoded.clone());
                            *last_path = active_path.clone();
                            encoded
                        } else {
                            cache.as_ref().unwrap().clone()
                        }
                    } else if let Some(ref preloaded) = *self.audio_embeddings.lock() {
                        preloaded.clone()
                    } else {
                        // No encoder and no preloaded — dummy passthrough
                        let t_emb = ctx.get("text_embeddings").or_else(|_| ctx.get("spliced_visual_embeddings"))?;
                        let embed_dim = t_emb.dim(2)?;
                        Tensor::zeros((1, 1, embed_dim), t_emb.dtype(), t_emb.device())?
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::SpliceAudioTensors { text_embeds, audio_embeds, output } => {
                    let t_emb = ctx.get(text_embeds)?;
                    // Only splice if audio is actually present in the token sequence.
                    // When no audio pad tokens exist, pass text embeddings through unchanged.
                    let active_audio = crate::backends::ACTIVE_AUDIO_PATH.lock().clone();
                    let out = if active_audio.is_none() {
                        t_emb.clone()
                    } else {
                        let a_emb = ctx.get(audio_embeds)?;
                        let a_emb = a_emb.to_dtype(t_emb.dtype())?;
                        let a_emb_narrowed = if a_emb.dim(2)? > t_emb.dim(2)? {
                            a_emb.narrow(2, 0, t_emb.dim(2)?)?
                        } else {
                            a_emb.clone()
                        };
                        let token_ids = &batch.token_ids;
                        crate::backends::multimodal::splice_audio_embeddings(&t_emb, &a_emb_narrowed, token_ids)?
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::DeepStackFuse { input, layer_idx, output } => {
                    let in_t = ctx.get(input)?;
                    let vis_embeds = ctx.get("visual_embeddings")?;
                    let hidden_dim = self.meta.as_ref().map(|m| m.hidden_dim).unwrap_or(2560);
                    
                    let mut ds_idx = None;
                    if let Some(ref meta) = self.meta {
                        if let Some(ref flags) = meta.is_deepstack_layers {
                            let mut count = 0;
                            for (idx, &flag) in flags.iter().enumerate() {
                                if flag {
                                    if idx == *layer_idx {
                                        ds_idx = Some(count);
                                        break;
                                    }
                                    count += 1;
                                }
                            }
                        }
                    }
                    
                    if let Some(idx) = ds_idx {
                        let ds_feat = vis_embeds.narrow(2, (1 + idx) * hidden_dim, hidden_dim)?;
                        let token_ids = &batch.token_ids;
                        let fused = splice_visual_embeddings(&in_t, &ds_feat, token_ids, 0, 0)?;
                        ctx.insert(output.clone(), fused);
                    } else {
                        ctx.insert(output.clone(), in_t.clone());
                    }
                }
                Operator::Softcap { input, output, cap } => {
                    let in_t = ctx.get(input)?;
                    let scaled = (in_t / *cap as f64)?;
                    let tanhed = scaled.tanh()?;
                    let out = (tanhed * *cap as f64)?;
                    ctx.insert(output.clone(), out);
                }
                Operator::Scale { input, scale, output } => {
                    let in_t = ctx.get(input)?;
                    let out = (in_t * (*scale as f64))?;
                    ctx.insert(output.clone(), out);
                }
                Operator::TensorScale { input, scale_tensor, output } => {
                    let in_t = ctx.get(input)?;
                    let scale = self.get_weight(scale_tensor, &mut local_cache)?.to_device(in_t.device())?;
                    let out = in_t.broadcast_mul(&scale)?;
                    ctx.insert(output.clone(), out);
                }
                Operator::PleInput {
                    input_ids,
                    text_embeddings,
                    per_layer_token_embd,
                    per_layer_model_proj,
                    per_layer_proj_norm,
                    output,
                } => {
                    let ids = ctx.get(input_ids)?;
                    let table = self.get_weight(per_layer_token_embd, &mut local_cache)?;
                    let (b_sz, seq_len) = ids.dims2()?;
                    let num_tokens = b_sz * seq_len;
                    
                    let n_layers = self.meta.as_ref().map(|m| m.n_layers).unwrap_or(35);
                    let ple_dim = self.meta.as_ref().and_then(|m| m.ple_dim).unwrap_or(256);
                    let hidden_dim = self.meta.as_ref().map(|m| m.hidden_dim).unwrap_or(1536);

                    // Host-side slicing / index selection:
                    let lookup_out = if table.device().is_cpu() {
                        let ids_cpu = ids.to_device(&Device::Cpu)?;
                        let ids_flat = ids_cpu.flatten_all()?;
                        let out_flat = table.index_select(&ids_flat, 0)?;
                        let out_cpu = out_flat.reshape((num_tokens, n_layers * ple_dim))?;
                        out_cpu.to_device(&self.device)?
                    } else {
                        let ids_flat = ids.flatten_all()?;
                        let out_flat = table.index_select(&ids_flat, 0)?;
                        out_flat.reshape((num_tokens, n_layers * ple_dim))?
                    };

                    let token_identity = lookup_out.reshape((num_tokens, n_layers, ple_dim))?.contiguous()?;
                    let token_identity = (token_identity * ((ple_dim as f64).sqrt()))?;

                    // Context-aware projection:
                    let text_emb = ctx.get(text_embeddings)?;
                    let text_emb_flat = text_emb.reshape((num_tokens, hidden_dim))?.contiguous()?;
                    let qmatmul_opt = if self.explicit_dequantize {
                        None
                    } else {
                        let cache = self.qmatmul_cache.lock();
                        cache.get(per_layer_model_proj).cloned()
                    };

                    let context_proj = if let Some(qmatmul) = qmatmul_opt {
                        let text_emb_flat_f32 = text_emb_flat.to_dtype(DType::F32)?;
                        let out_f32 = qmatmul.forward(&text_emb_flat_f32)?;
                        out_f32.to_dtype(text_emb_flat.dtype())?
                    } else {
                        let model_proj_w = self.get_weight(per_layer_model_proj, &mut local_cache)?.to_device(&self.device)?;
                        text_emb_flat.matmul(&model_proj_w.t()?)?
                    };
                    let context_proj = (context_proj / ((hidden_dim as f64).sqrt()))?;
                    let context_proj_reshaped = context_proj.reshape((num_tokens, n_layers, ple_dim))?.contiguous()?;

                    // Apply RMSNorm specifically to the context projection
                    let norm_w = self.get_weight(per_layer_proj_norm, &mut local_cache)?.to_device(&self.device)?;
                    let is_gemma = self.meta.as_ref().map(|m| m.is_gemma).unwrap_or(false);
                    let is_gemma_hf = is_gemma && !self.weights.is_empty();
                    let context_aware = RmsNorm::new(norm_w, 1e-6, is_gemma_hf).forward(&context_proj_reshaped)?;

                    // Combined: (token_identity + context_aware) * (1 / sqrt(2))
                    let combined = ((token_identity + context_aware)? * (1.0 / 2.0f64.sqrt()))?;

                    ctx.insert(output.clone(), combined);
                }
                Operator::PleLayer {
                    input,
                    per_layer_input,
                    layer_idx,
                    per_layer_input_gate,
                    per_layer_projection,
                    post_per_layer_input_norm,
                    output,
                } => {
                    let in_t = ctx.get(input)?;
                    let (b_sz, seq_len, hidden_dim) = in_t.dims3()?;
                    let num_tokens = b_sz * seq_len;
                    let in_flat = in_t.reshape((num_tokens, hidden_dim))?.contiguous()?;

                    let ple_in = ctx.get(per_layer_input)?;
                    let ple_slice = ple_in.narrow(1, *layer_idx, 1)?.reshape((num_tokens, ()))?.contiguous()?;

                    let qmatmul_gate_opt = if self.explicit_dequantize {
                        None
                    } else {
                        let cache = self.qmatmul_cache.lock();
                        cache.get(per_layer_input_gate).cloned()
                    };

                    let gate_input = if let Some(qmatmul) = qmatmul_gate_opt {
                        let in_flat_f32 = in_flat.to_dtype(DType::F32)?;
                        let out_f32 = qmatmul.forward(&in_flat_f32)?;
                        out_f32.to_dtype(in_flat.dtype())?
                    } else {
                        let gate_w = self.get_weight(per_layer_input_gate, &mut local_cache)?.to_device(in_t.device())?;
                        in_flat.matmul(&gate_w.t()?)?
                    };

                    let act_fn = self.meta.as_ref().map(|m| m.hidden_act).unwrap_or(HiddenAct::SiLU);
                    let gate_active = match act_fn {
                        HiddenAct::SiLU => (&gate_input * &candle_nn::ops::sigmoid(&gate_input)?)?,
                        HiddenAct::GeLU => gate_input.gelu()?,
                    };

                    let gated = gate_active.broadcast_mul(&ple_slice)?;

                    let qmatmul_proj_opt = if self.explicit_dequantize {
                        None
                    } else {
                        let cache = self.qmatmul_cache.lock();
                        cache.get(per_layer_projection).cloned()
                    };

                    let proj_out = if let Some(qmatmul) = qmatmul_proj_opt {
                        let gated_f32 = gated.to_dtype(DType::F32)?;
                        let out_f32 = qmatmul.forward(&gated_f32)?;
                        out_f32.to_dtype(gated.dtype())?
                    } else {
                        let proj_w = self.get_weight(per_layer_projection, &mut local_cache)?.to_device(in_t.device())?;
                        gated.matmul(&proj_w.t()?)?
                    };

                    let norm_w = self.get_weight(post_per_layer_input_norm, &mut local_cache)?.to_device(in_t.device())?;
                    let is_gemma = self.meta.as_ref().map(|m| m.is_gemma).unwrap_or(false);
                    let is_gemma_hf = is_gemma && !self.weights.is_empty();
                    let proj_normed = RmsNorm::new(norm_w, 1e-6, is_gemma_hf).forward(&proj_out)?;

                    let proj_normed_3d = proj_normed.reshape((b_sz, seq_len, hidden_dim))?.contiguous()?;
                    let out = (in_t + proj_normed_3d)?;
                    ctx.insert(output.clone(), out);
                }
            }



            // Evict tensors that are no longer needed
            let mut to_remove = Vec::new();
            for (key, &last_idx) in &last_use {
                if last_idx == idx && key != "logits" {
                    to_remove.push(key.clone());
                }
            }
            for key in to_remove {
                ctx.remove(&key);
            }
        }

        // Retrieve logits
        let logits = ctx.get("logits")?;

        // Update sequence lengths in KV cache after all layers have processed the batch
        for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
            let start_offset = batch.cu_seqlens[i] as usize;
            let end_offset = batch.cu_seqlens[i+1] as usize;
            let num_tokens = end_offset - start_offset;
            let seq_len_before = kv_cache.get_seq_len(seq_id);
            kv_cache.set_seq_len(seq_id, seq_len_before + num_tokens);
        }

        // Extract logits for next token prediction
        let mut next_tokens = Vec::with_capacity(b_sz);
        let mut batch_logits = Vec::with_capacity(b_sz);

        for i in 0..b_sz {
            let seq_logits_t = logits.narrow(0, i, 1)?.narrow(1, seq_len - 1, 1)?.squeeze(0)?.squeeze(0)?;
            let seq_logits = seq_logits_t.to_dtype(DType::F32)?.to_vec1::<f32>()?;



            let mut max_idx = 0;
            let mut max_val_ref = seq_logits[0];
            for (idx, &val) in seq_logits.iter().enumerate() {
                if val > max_val_ref {
                    max_val_ref = val;
                    max_idx = idx;
                }
            }

            next_tokens.push(max_idx as TokenId);
            batch_logits.push(seq_logits);
        }

        Ok(BatchOutput {
            seq_ids: batch.seq_ids.clone(),
            next_tokens,
            logits: Some(batch_logits),
        })
    }

    fn sample(&self, logits: &[f32], params: &SampleParams, token_history: &[TokenId]) -> Result<TokenId> {
        sample_logits(logits, params, token_history)
    }

    fn kv_cache_config(&self) -> KvCacheConfig {
        self.kv_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, Device};

    #[test]
    fn test_hadamard_orthogonal() {
        let d = 128;
        let h_vec = generate_hadamard_orthogonal(d);
        assert_eq!(h_vec.len(), d * d);
        
        let dev = &Device::Cpu;
        let h_tensor = Tensor::from_vec(h_vec, (d, d), dev).unwrap();
        let h_t = h_tensor.t().unwrap();
        let identity = h_tensor.matmul(&h_t).unwrap();
        
        let identity_expected = Tensor::eye(d, DType::F32, dev).unwrap();
        let diff = (identity - identity_expected).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff < 1e-4, "Hadamard matrix is not orthogonal: max diff = {}", max_diff);
    }

    #[test]
    fn test_kv_quantization_q8() {
        let dev = &Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, -2.0, 3.5, -4.0, 0.5, 0.0], (1, 1, 1, 6), dev).unwrap();
        
        let (u8_tensor, scale) = quantize(&x, KvDtype::Q8, DType::F32).unwrap();
        let deq = dequantize(&u8_tensor, &scale, KvDtype::Q8, DType::F32).unwrap();
        
        let diff = (&x - &deq).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff < 0.1, "Q8 quantization loss too high: max diff = {}", max_diff);
    }

    #[test]
    fn test_kv_quantization_q4() {
        let dev = &Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, -2.0, 3.5, -4.0, 0.5, 0.0], (1, 1, 1, 6), dev).unwrap();
        
        let (u8_tensor, scale) = quantize(&x, KvDtype::Q4, DType::F32).unwrap();
        let deq = dequantize(&u8_tensor, &scale, KvDtype::Q4, DType::F32).unwrap();
        
        let diff = (&x - &deq).unwrap().abs().unwrap();
        let max_diff = diff.max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(max_diff < 0.7, "Q4 quantization loss too high: max diff = {}", max_diff);
    }
}
