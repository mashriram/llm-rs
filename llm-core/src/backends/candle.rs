use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, bail, Context};
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

/// Best-effort EOS token id resolution from a GGUF file's `tokenizer.ggml.tokens`
/// vocabulary array, tried only when the explicit `tokenizer.ggml.eos_token_id`
/// metadata key is absent. GGUF's tokens array is index-addressed by token id,
/// so a candidate string's array position IS its token id.
fn find_eos_in_gguf_tokens(
    metadata: &std::collections::HashMap<String, candle_core::quantized::gguf_file::Value>,
    candidates: &[&str],
) -> Option<u32> {
    if let Some(candle_core::quantized::gguf_file::Value::Array(arr)) = metadata.get("tokenizer.ggml.tokens") {
        for cand in candidates {
            if let Some(idx) = arr.iter().position(
                |v| matches!(v, candle_core::quantized::gguf_file::Value::String(s) if s == cand)
            ) {
                return Some(idx as u32);
            }
        }
    }
    None
}

/// Best-effort EOS token id resolution from a HF model directory's
/// `tokenizer_config.json` (`added_tokens_decoder`) or `tokenizer.json`
/// (`added_tokens`), tried only when `config.json`'s `eos_token_id` is absent.
fn find_eos_in_hf_tokenizer_files(model_dir: &Path, candidates: &[&str]) -> Option<u32> {
    for fname in ["tokenizer_config.json", "tokenizer.json"] {
        let Ok(contents) = std::fs::read_to_string(model_dir.join(fname)) else { continue };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&contents) else { continue };

        if let Some(decoder) = v.get("added_tokens_decoder").and_then(|d| d.as_object()) {
            for (id_str, tok) in decoder {
                if let Some(content) = tok.get("content").and_then(|c| c.as_str()) {
                    if candidates.contains(&content) {
                        if let Ok(id) = id_str.parse::<u32>() {
                            return Some(id);
                        }
                    }
                }
            }
        }
        if let Some(arr) = v.get("added_tokens").and_then(|a| a.as_array()) {
            for tok in arr {
                if let Some(content) = tok.get("content").and_then(|c| c.as_str()) {
                    if candidates.contains(&content) {
                        if let Some(id) = tok.get("id").and_then(|i| i.as_u64()) {
                            return Some(id as u32);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Model-agnostic mmproj (multimodal projector) file discovery for a GGUF base
/// model path: strips the base model's extension/quantization-tag suffix and
/// tries a handful of common mmproj filename patterns, falling back to a
/// same-directory scan for any `*mmproj*.gguf` file. Returns `None` if nothing
/// is found (callers decide their own fallback — e.g. some fall back to the
/// base model path itself, on the assumption the vision/audio tower may be
/// embedded in the same GGUF file; others just skip the encoder).
///
/// Previously this ~50-line block was duplicated 3 times (once each for
/// mmproj-metadata merging, VisionEncoder loading, and AudioEncoder loading) —
/// factored into this single helper so the discovery logic can't drift.
fn find_mmproj_path(base_path: &Path) -> Option<std::path::PathBuf> {
    let base = base_path.to_string_lossy();
    // Strip any quantisation suffix (e.g. .Q4_K_M, .Q8_0, .BF16, .F16) and the
    // .gguf extension to get the base model name stem.
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
    for suffix in &suffixes {
        let candidate = format!("{}{}", stem, suffix);
        if std::path::Path::new(&candidate).exists() {
            return Some(std::path::PathBuf::from(candidate));
        }
    }
    // Fall back to a same-directory scan for any file matching `*<stem_prefix>*mmproj*.gguf`.
    let model_file_stem = base_path.file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let model_prefix = model_file_stem.split('-').next().unwrap_or(&model_file_stem);

    if let Some(parent) = base_path.parent() {
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() {
                    let name = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                    if name.contains("mmproj") && name.ends_with(".gguf") && (name.contains(model_prefix) || model_file_stem.contains(name.split('-').next().unwrap_or(""))) {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
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

/// Full block-gather-and-reconstruct fallback: dequantizes and concatenates
/// every stored block for this sequence/layer, from scratch. This is what
/// every decode step did unconditionally before the `kv_history_cache` fast
/// path was added below - O(total context length) of tensor work per layer,
/// per single new token, which is the dominant real cost of long-context
/// decode (confirmed by direct benchmarking: decode throughput dropped from
/// ~31 t/s at a 28-token context to ~16 t/s at a 278-token context on the
/// same 3B model/hardware, far more than a model this size should degrade
/// over such a small range). Kept as the correctness-guaranteed fallback for
/// cache misses (new sequence, evicted/reused seq_id, quantized KV dtype).
#[allow(clippy::too_many_arguments)]
fn rebuild_kv_history_from_blocks(
    gpu_cache: &HashMap<(usize, BlockId), BlockData>,
    src_layer: usize,
    block_table: &[BlockId],
    offset: usize,
    this_seq_len: usize,
    block_size: usize,
    dtype: KvDtype,
    compute_dtype: DType,
    n_rep: usize,
) -> Result<(Tensor, Tensor)> {
    let total_len = offset + this_seq_len;
    let num_active_blocks = std::cmp::min(
        (total_len + block_size - 1) / block_size,
        block_table.len(),
    );
    let mut k_blocks = Vec::with_capacity(num_active_blocks);
    let mut v_blocks = Vec::with_capacity(num_active_blocks);

    for &block_id in &block_table[0..num_active_blocks] {
        let block_data = gpu_cache.get(&(src_layer, block_id))
            .ok_or_else(|| anyhow!("Block ID {} not found for layer {}", block_id, src_layer))?;

        let k_deq = if dtype == KvDtype::Q8 || dtype == KvDtype::Q4 {
            dequantize(&block_data.k, block_data.k_scale.as_ref().unwrap(), dtype, compute_dtype)?
        } else {
            block_data.k.clone()
        };
        let v_deq = if dtype == KvDtype::Q8 || dtype == KvDtype::Q4 {
            dequantize(&block_data.v, block_data.v_scale.as_ref().unwrap(), dtype, compute_dtype)?
        } else {
            block_data.v.clone()
        };

        k_blocks.push(k_deq.squeeze(0)?);
        v_blocks.push(v_deq.squeeze(0)?);
    }

    let k_hist_squeezed = Tensor::cat(&k_blocks, 0)?.narrow(0, 0, total_len)?;
    let v_hist_squeezed = Tensor::cat(&v_blocks, 0)?.narrow(0, 0, total_len)?;

    let k_hist_rep = repeat_kv(k_hist_squeezed.transpose(0, 1)?, n_rep)?.contiguous()?;
    let v_hist_rep = repeat_kv(v_hist_squeezed.transpose(0, 1)?, n_rep)?.contiguous()?;

    Ok((k_hist_rep, v_hist_rep))
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
    /// `None` when no EOS token id could be determined from model metadata/config
    /// or a best-effort tokenizer lookup — see `LlmBackend::eos_token_id`'s doc
    /// comment for why this must not silently default to Llama's `2`.
    eos_token_id: Option<u32>,
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
    /// Per-(sequence, source-layer) cache of the already-repeated/transposed/
    /// contiguous full K/V history, so a decode step only needs to append its
    /// one new token instead of rebuilding the entire past from block storage
    /// every time (see `rebuild_kv_history_from_blocks`'s doc comment for
    /// why this matters). Keyed by `src_layer` (not `layer_idx`) so KV-shared
    /// layers (e.g. Gemma's local/global sharing) reuse the same entry the
    /// owning layer already populated in the same forward_pass call, instead
    /// of redundantly caching identical data per sharing layer. Always holds
    /// the un-windowed length - sliding-window trimming happens on a local
    /// view at use time, never persisted here. Only populated for
    /// non-quantized (F16/F32) KV; the Q8/Q4 path always uses the full
    /// rebuild fallback (see `is_quantized` checks at the call site).
    kv_history_cache: Mutex<HashMap<(SeqId, usize), (usize, Tensor, Tensor)>>,
    explicit_dequantize: bool,
    use_vram_embeddings: bool,
    custom_mmproj_path: Option<std::path::PathBuf>,
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
            eos_token_id: None,
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
            kv_history_cache: Mutex::new(HashMap::new()),
            explicit_dequantize,
            use_vram_embeddings,
            custom_mmproj_path: None,
            qmatmul_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Set an explicit custom mmproj path for vision/audio encoders.
    pub fn set_mmproj_path(&mut self, path: std::path::PathBuf) {
        self.custom_mmproj_path = Some(path);
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
            Device::Metal(_) => "candle-metal",
        }
    }

    fn eos_token_id(&self) -> Option<u32> {
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
        // Must be cleared alongside gpu_kv_cache: if `seq_id` gets reused for
        // a different sequence later, a stale entry here would make the next
        // forward_pass call wrongly believe it can append onto someone else's
        // history instead of rebuilding from scratch.
        self.kv_history_cache.lock().retain(|(sid, _), _| *sid != seq_id);
    }

    fn set_explicit_dequantize(&mut self, val: bool) {
        self.explicit_dequantize = val;
    }

    fn set_use_vram_embeddings(&mut self, val: bool) {
        self.use_vram_embeddings = val;
    }

    fn load_weights(&mut self, path: &Path) -> Result<ModelMeta> {
        let is_gguf = path.is_file() && path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase() == "gguf").unwrap_or(false);

        // Estimate model size in bytes from files on disk
        let mut estimated_bytes = 0;
        if path.is_file() {
            estimated_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        } else if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                            let ext_lower = ext.to_lowercase();
                            if ext_lower == "safetensors" || ext_lower == "bin" || ext_lower == "gguf" {
                                estimated_bytes += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
            }
        }

        // Hardware-aware dynamic device selection via HardwareProfile.
        // `choose_device` deliberately returns `Err` when the model won't fit in
        // either VRAM or system RAM ("clean error > OS kill" — see its doc comment);
        // swallowing that with `unwrap_or(Cpu)` would defeat that safety mechanism
        // entirely, so we propagate it instead of guessing.
        let profile = crate::profile::HardwareProfile::get();
        let target_backend = profile.choose_device(estimated_bytes)
            .map_err(|e| anyhow!("Refusing to load model: {e}"))?;

        // Explicit, narrow opt-in escape hatch: if a GPU device fails to initialize
        // (driver/runtime issue, not a capacity issue — HardwareProfile already
        // decided the model fits), the default is a hard `Err` per CLAUDE.md's
        // "no silent fallback to a different backend" rule. Setting
        // LLM_ALLOW_CPU_FALLBACK=1 opts into the old silent-downgrade behavior,
        // loudly logged, for environments that would rather run slow than not run.
        let allow_cpu_fallback = std::env::var("LLM_ALLOW_CPU_FALLBACK").is_ok();
        let dev = match target_backend {
            crate::profile::BackendChoice::Cuda => {
                match Device::new_cuda(0) {
                    Ok(d) => d,
                    Err(e) if allow_cpu_fallback => {
                        tracing::warn!(
                            "CUDA init failed ({e}); LLM_ALLOW_CPU_FALLBACK=1 set, falling back to CPU (SLOW)"
                        );
                        Device::Cpu
                    }
                    Err(e) => {
                        return Err(anyhow!(
                            "CUDA device init failed ({e}) and HardwareProfile selected CUDA \
                             for this model. Not silently falling back to CPU (set \
                             LLM_ALLOW_CPU_FALLBACK=1 to opt into that instead)."
                        ));
                    }
                }
            }
            crate::profile::BackendChoice::Metal => {
                match Device::new_metal(0) {
                    Ok(d) => d,
                    Err(e) if allow_cpu_fallback => {
                        tracing::warn!(
                            "Metal init failed ({e}); LLM_ALLOW_CPU_FALLBACK=1 set, falling back to CPU (SLOW)"
                        );
                        Device::Cpu
                    }
                    Err(e) => {
                        return Err(anyhow!(
                            "Metal device init failed ({e}) and HardwareProfile selected Metal \
                             for this model. Not silently falling back to CPU (set \
                             LLM_ALLOW_CPU_FALLBACK=1 to opt into that instead)."
                        ));
                    }
                }
            }
            crate::profile::BackendChoice::Cpu => Device::Cpu,
        };
        self.device = dev.clone();
        self.compute_dtype = if dev.is_cpu() { DType::F32 } else { DType::F16 };

        // Determine cache budget dynamically (leave 2.0 GB headroom for activations/KV cache)
        let vram_headroom_bytes: u64 = 2_000 * 1024 * 1024;
        let mut free_vram = profile.gpu_vram_free_bytes.unwrap_or(0);
        if free_vram == 0 && !dev.is_cpu() {
            // HardwareProfile could not measure free VRAM/unified memory. Rather than
            // guessing a generous 6-8 GB is available (which contradicts this project's
            // own HardwareProfile design of never guessing VRAM — see choose_device's
            // "clean error > OS kill" doc comment), fall back to a small, conservative
            // fixed budget and say so loudly. Worst case this under-utilizes VRAM;
            // it will never overcommit it.
            const CONSERVATIVE_UNMEASURED_VRAM_BYTES: u64 = 1_500 * 1024 * 1024; // 1.5 GB
            tracing::warn!(
                "Free VRAM/Unified Memory could not be measured for {:?}; using a conservative \
                 {:.1} GB budget instead of assuming several GB are free.",
                dev, CONSERVATIVE_UNMEASURED_VRAM_BYTES as f64 / 1e9
            );
            free_vram = CONSERVATIVE_UNMEASURED_VRAM_BYTES;
        }
        self.gpu_cache_budget = free_vram.saturating_sub(vram_headroom_bytes);
        tracing::info!(
            "GPU cache budget: {:.1} GB (free VRAM/Unified Memory: {:.1} GB)",
            self.gpu_cache_budget as f64 / 1e9,
            free_vram as f64 / 1e9
        );

        let mut weights: HashMap<String, Tensor> = HashMap::new();
        let mut quantized_weights: HashMap<String, candle_core::quantized::QTensor> = HashMap::new();

        let meta = if is_gguf {
            let mut file = std::fs::File::open(path)
                .context(format!("Failed to open GGUF file: {:?}", path))?;
            let model = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
                // candle-core's GGUF header parser aborts the ENTIRE file scan
                // (not just the offending tensor) the moment it hits a GGML
                // tensor dtype id it doesn't recognize - confirmed via a real
                // GGUF file (a Qwen2.5-0.5B "Q2_K" export that mixes in
                // llama.cpp's newer IQ4_NL "importance quantization" format
                // for most weight tensors, dtype id 20). candle-core 0.9.2
                // has no dequantization support for ANY IQ-series type
                // (IQ1_S/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ4_NL/IQ3_S/IQ2_S/IQ4_XS/
                // IQ1_M - confirmed by grepping its source), so this is a
                // real, currently-unsupported quantization family, not a
                // corrupt file. Give an actionable message instead of the
                // raw parser error, which just says "unknown dtype for
                // tensor N" with no indication of what that means or what
                // to do about it.
                let msg = e.to_string();
                if msg.contains("unknown dtype for tensor") {
                    anyhow!(
                        "GGUF file {:?} uses a quantization format this build doesn't support \
                         yet ({msg}). This usually means an \"IQ\"-series type (IQ4_NL, IQ2_XXS, \
                         etc. - llama.cpp's newer \"importance quantization\" formats), which the \
                         underlying candle-core tensor library has no dequantization support for \
                         at all. Classic quant types (F16/F32/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, \
                         and the K-quant family Q2_K..Q8_K) all work fine - try a different quant \
                         of the same model (Q4_K_M and Q8_0 are usually safe choices; `llm pull` \
                         warns about this before downloading when it can detect it from the \
                         filename).",
                        path
                    )
                } else {
                    anyhow::Error::new(e).context("Failed to read GGUF content")
                }
            })?;

            // Discover and load mmproj metadata if a matching mmproj file exists
            let mut mmproj_metadata = HashMap::new();
            let mmproj_path = find_mmproj_path(path);
            if let Some(ref mp_path) = mmproj_path {
                tracing::info!("Discovered mmproj file at {:?}; loading its metadata...", mp_path);
                if let Ok(mut mp_file) = std::fs::File::open(mp_path) {
                    if let Ok(mp_model) = candle_core::quantized::gguf_file::Content::read(&mut mp_file) {
                        mmproj_metadata = mp_model.metadata;
                    }
                }
            }

            // `general.architecture` and `tokenizer.ggml.tokens` are mandatory GGUF
            // fields for any real model file. Guessing "llama"/a fixed vocab size
            // when they're absent would silently misclassify the architecture (and
            // thus mis-route arch-specific behavior like Gemma's tied-embedding /
            // activation heuristics below) or size the lm_head against the wrong
            // vocabulary — a corrupt/non-model GGUF file must fail loudly here
            // instead of quietly loading with wrong assumptions.
            let arch = match model.metadata.get("general.architecture") {
                Some(candle_core::quantized::gguf_file::Value::String(s)) => s.as_str(),
                _ => bail!(
                    "GGUF file {:?} is missing required 'general.architecture' metadata; \
                     cannot determine model architecture",
                    path
                ),
            };

            let vocab_size = match model.metadata.get("tokenizer.ggml.tokens") {
                Some(candle_core::quantized::gguf_file::Value::Array(arr)) => arr.len(),
                _ => bail!(
                    "GGUF file {:?} is missing required 'tokenizer.ggml.tokens' metadata; \
                     cannot determine vocabulary size",
                    path
                ),
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

            // Do not silently assume Llama's EOS id (2) when metadata lacks an
            // explicit one — try the tokenizer's own vocabulary for common EOS
            // strings as a best-effort fallback, and if that also fails, leave
            // it as `None` (loudly logged) rather than guessing wrong.
            let eos_token_id = meta_u32(&model.metadata, "tokenizer.ggml.eos_token_id")
                .or_else(|| find_eos_in_gguf_tokens(
                    &model.metadata,
                    &["<end_of_turn>", "<|im_end|>", "<|endoftext|>", "</s>"],
                ));
            if eos_token_id.is_none() {
                tracing::warn!(
                    "Could not determine an EOS token id for {:?} from GGUF metadata or \
                     common tokenizer vocab entries; generation will not stop on EOS \
                     (only max_new_tokens bounds it) unless the caller resolves \
                     additional EOS candidates itself.",
                    path
                );
            }
            self.eos_token_id = eos_token_id;

            tracing::info!("Loading {} GGUF tensors as QTensor", model.tensor_infos.len());

            // Single source of truth for Gemma-family detection, shared with model/config.rs
            // (see `is_gemma_arch`'s doc comment). GGUF only exposes a single
            // `general.architecture` string, so `architectures` is empty here.
            let is_gemma = crate::types::is_gemma_arch(arch, &[]);

            // Detect tied embeddings from GGUF tensor presence rather than arch name.
            // If neither "output.weight" nor "lm_head.weight" exists as a tensor, the model
            // uses tied embeddings (shares token_embd.weight for both input and output projection).
            // Computed up-front (tensor_infos is fully populated before any tensor is read) so the
            // VRAM-preload fast path below can use this generic, metadata-driven flag directly
            // instead of re-deriving a Gemma-only arch-name check.
            let has_dedicated_lm_head = model.tensor_infos.contains_key("output.weight")
                || model.tensor_infos.contains_key("lm_head.weight")
                || model.tensor_infos.contains_key("output_norm.weight") && model.tensor_infos.contains_key("output.weight");
            // For Gemma, always tie (embed_scale requires the shared weight).
            let tie_word_embeddings = is_gemma || !has_dedicated_lm_head;

            let mut remaining_vram_budget = self.gpu_cache_budget;
            for name in model.tensor_infos.keys() {
                let name_lower = name.to_lowercase();
                let is_token_embd = name == "token_embd.weight";
                let is_per_layer_embd = name_lower.contains("per_layer_token_embd");
                let is_non_matmul = name_lower.contains("embed") 
                    || name_lower.contains("embd")
                    || name_lower.contains("norm") 
                    || name_lower.contains("bias") 
                    || name_lower.contains("scale")
                    || name_lower.contains("freq")
                    || name_lower.contains("rotors")
                    || name_lower.contains("rot");
                let mut fit_in_vram = false;
                if is_non_matmul && !name_lower.contains("freq") && !name_lower.contains("rot") && !is_per_layer_embd && self.device.is_cuda() {
                    if let Some(info) = model.tensor_infos.get(name) {
                        // dequantize() produces F32 tensors (4 bytes per element)
                        let size_bytes = (info.shape.dims().iter().product::<usize>() as u64) * 4;
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
                let qt = match model.tensor(&mut file, name, &load_device) {
                    Ok(qt) => qt,
                    Err(e) if !load_device.is_cpu() => {
                        tracing::warn!("Failed to load tensor {} on {:?} ({e}); falling back to CPU", name, load_device);
                        model.tensor(&mut file, name, &Device::Cpu)
                            .with_context(|| format!("Failed to read tensor {}", name))?
                    }
                    Err(e) => return Err(anyhow::Error::new(e).context(format!("Failed to read tensor {}", name))),
                };
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
            let model_dir = if path.is_file() {
                path.parent().unwrap_or(Path::new("."))
            } else {
                path
            };
            let config_path = model_dir.join("config.json");
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
                    // Do not silently assume Llama's EOS id (2) when config.json
                    // lacks an explicit one — try the tokenizer files' own special
                    // tokens for common EOS strings as a best-effort fallback.
                    .or_else(|| find_eos_in_hf_tokenizer_files(
                        model_dir,
                        &["<|im_end|>", "<end_of_turn>", "<|endoftext|>", "</s>"],
                    ))
            };
            if eos_token_id.is_none() {
                tracing::warn!(
                    "Could not determine an EOS token id for {:?} from config.json or \
                     tokenizer files; generation will not stop on EOS (only \
                     max_new_tokens bounds it) unless the caller resolves additional \
                     EOS candidates itself.",
                    path
                );
            }
            self.eos_token_id = eos_token_id;

            let mut sf_files = Vec::new();
            if path.is_file() {
                sf_files.push(path.to_path_buf());
            } else if let Ok(entries) = std::fs::read_dir(path) {
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

            // Detect AWQ/GPTQ pre-quantization from config.json (bitsandbytes
            // is rejected earlier in `parse_config`; awq/gptq are allowed
            // through to be dequantized here). Read directly rather than
            // threading a new field through `ModelMeta` (which has ~9
            // construction sites across the codebase) to keep this change's
            // blast radius small.
            let packed_quant: Option<(String, usize)> = std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| {
                    let qc = v.get("quantization_config")?;
                    let method = qc.get("quant_method")?.as_str()?.to_lowercase();
                    if method == "awq" || method == "gptq" {
                        let group_size = qc.get("group_size").and_then(|g| g.as_u64()).unwrap_or(128) as usize;
                        Some((method, group_size))
                    } else {
                        None
                    }
                });

            // Load all SafeTensors tensors to CPU memory first, at their
            // native dtype (NOT cast to compute_dtype yet: AWQ/GPTQ pack
            // weights as I32, and a raw numeric cast of packed integer bit
            // patterns to F16/F32 would silently produce meaningless values
            // instead of a dequantization - exactly the kind of silent
            // wrong-output failure this project's rules forbid).
            let mut raw_tensors: HashMap<String, Tensor> = HashMap::new();
            for sf_file in sf_files {
                let loaded = candle_core::safetensors::load(&sf_file, &Device::Cpu)?;
                for (name, tensor) in loaded {
                    raw_tensors.insert(name, tensor);
                }
            }

            let mut cpu_tensors = HashMap::new();
            if let Some((method, group_size)) = packed_quant {
                // Group the three (AWQ) or four (GPTQ) packed components per
                // logical linear layer, dequantize each into one dense
                // `{base}.weight` tensor; copy anything else (embeddings,
                // norms, lm_head, GPTQ's per-layer biases) through unchanged.
                let mut bases: std::collections::HashSet<String> = std::collections::HashSet::new();
                for name in raw_tensors.keys() {
                    let comp = if method == "awq" {
                        crate::loader::awq::awq_component(name)
                    } else {
                        crate::loader::gptq::gptq_component(name)
                    };
                    if let Some((base, _)) = comp {
                        bases.insert(base.to_string());
                    }
                }
                for base in &bases {
                    let qweight = raw_tensors.get(&format!("{base}.qweight"))
                        .ok_or_else(|| anyhow!("{method}: missing {base}.qweight"))?;
                    let qzeros = raw_tensors.get(&format!("{base}.qzeros"))
                        .ok_or_else(|| anyhow!("{method}: missing {base}.qzeros"))?;
                    let scales = raw_tensors.get(&format!("{base}.scales"))
                        .ok_or_else(|| anyhow!("{method}: missing {base}.scales"))?;
                    let dequantized = if method == "awq" {
                        crate::loader::awq::dequantize_awq_linear(qweight, qzeros, scales, group_size, &Device::Cpu)?
                    } else {
                        let g_idx = raw_tensors.get(&format!("{base}.g_idx"));
                        crate::loader::gptq::dequantize_gptq_linear(qweight, qzeros, scales, g_idx, group_size, &Device::Cpu)?
                    };
                    cpu_tensors.insert(format!("{base}.weight"), dequantized.to_dtype(self.compute_dtype)?);
                }
                for (name, tensor) in raw_tensors {
                    let comp = if method == "awq" {
                        crate::loader::awq::awq_component(&name)
                    } else {
                        crate::loader::gptq::gptq_component(&name)
                    };
                    if comp.is_none() {
                        cpu_tensors.insert(name, tensor.to_dtype(self.compute_dtype)?);
                    }
                }
                tracing::warn!(
                    "Dequantized {} {method} linear layer(s) to dense {:?} weights at load time \
                     (correctness-first path, not yet a fast tensor-core kernel - see \
                     quant-performance-plan.md phase 4.3). This dequant path has been checked \
                     against real safetensors headers but not yet numerically verified against \
                     Python (transformers/autoawq/auto-gptq) output on real hardware.",
                    bases.len(), self.compute_dtype
                );
            } else {
                for (name, tensor) in raw_tensors {
                    cpu_tensors.insert(name, tensor.to_dtype(self.compute_dtype)?);
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
                let tc_path = model_dir.join("tokenizer_config.json");
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
                            let deq_res = qt.dequantize(&target_dev).or_else(|_| {
                                if !target_dev.is_cpu() {
                                    qt.dequantize(&Device::Cpu)
                                } else {
                                    Err(candle_core::Error::Msg("CPU dequantize failed".to_string()))
                                }
                            });
                            match deq_res {
                                Ok(t) => {
                                    local_deq.insert(name.clone(), t);
                                }
                                Err(e) => {
                                    println!("Error: Failed to dequantize tensor {}: {:?}", name, e);
                                    self.quantized_weights.insert(name, qt);
                                }
                            }
                        } else {
                            // QMatMul requires a 2D matrix weight. If 1D or 3D+, keep in quantized_weights.
                            if qt.shape().dims().len() == 2 {
                                match QMatMul::from_qtensor(qt) {
                                    Ok(qmatmul) => {
                                        local_qmatmul.insert(name.clone(), qmatmul);
                                    }
                                    Err(e) => {
                                        tracing::warn!("QMatMul::from_qtensor failed for {}: {:?}", name, e);
                                    }
                                }
                            } else {
                                self.quantized_weights.insert(name, qt);
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

        let custom_mmproj = self.custom_mmproj_path.clone();

        if meta.has_vision_encoder {
            // Explicit mmproj path from CLI/caller, or model-agnostic mmproj discovery,
            // falling back to the base model path itself.
            let vision_path = custom_mmproj.clone().unwrap_or_else(|| {
                if is_gguf {
                    find_mmproj_path(path).unwrap_or_else(|| path.to_path_buf())
                } else {
                    path.to_path_buf()
                }
            });
            println!("  [mmproj] Loading VisionEncoder weights from: {}", vision_path.display());
            match VisionEncoder::load(&vision_path, &dev) {
                Ok(enc) => {
                    println!("  [mmproj] VisionEncoder loaded successfully ✓");
                    self.vision_encoder = Some(enc);
                }
                Err(e) => {
                    println!("  [mmproj] VisionEncoder load skipped: {e}");
                }
            }
        }

        if meta.has_audio_encoder {
            let audio_path = custom_mmproj.clone().unwrap_or_else(|| {
                if is_gguf {
                    find_mmproj_path(path).unwrap_or_else(|| path.to_path_buf())
                } else {
                    path.to_path_buf()
                }
            });
            println!("  [mmproj] Loading AudioEncoder weights from: {}", audio_path.display());
            match AudioEncoder::load(&audio_path, &dev) {
                Ok(enc) => {
                    println!("  [mmproj] AudioEncoder loaded successfully ✓");
                    self.audio_encoder = Some(enc);
                }
                Err(e) => {
                    println!("  [mmproj] AudioEncoder load skipped: {e}");
                }
            }
        }

        Ok(meta)
    }

    fn forward_pass(&self, batch: &BatchInput) -> Result<BatchOutput> {
        if batch.seq_ids.is_empty() {
            return Err(anyhow!("empty batch: seq_ids is empty"));
        }
        // Validate the packed (varlen) batch shape up-front, before touching any
        // model state: `cu_seqlens` must have exactly one more entry than
        // `seq_ids` (N sequence boundaries for N sequences), and its last entry
        // must equal the total token count — otherwise the per-sequence slicing
        // done throughout this function (RoPE position ids, PagedAttention's
        // token-axis narrowing, next-token logit extraction) would silently read
        // out-of-range or mis-attributed data instead of erroring.
        let num_seqs = batch.seq_ids.len();
        let total_tokens = batch.token_ids.len();
        if batch.cu_seqlens.len() != num_seqs + 1 {
            return Err(anyhow!(
                "forward_pass: cu_seqlens length {} does not match seq_ids length {} + 1",
                batch.cu_seqlens.len(), num_seqs
            ));
        }
        if batch.cu_seqlens.last().copied() != Some(total_tokens as u32) {
            return Err(anyhow!(
                "forward_pass: cu_seqlens last entry ({:?}) does not match total token count ({})",
                batch.cu_seqlens.last(), total_tokens
            ));
        }
        let graph = self.graph.as_ref().ok_or_else(|| anyhow!("Compute graph not built"))?;
        let meta = self.meta.as_ref().ok_or_else(|| anyhow!("Model metadata not loaded"))?;
        let kv_cache_mutex = self.kv_cache.as_ref().ok_or_else(|| anyhow!("KV Cache not initialized"))?;
        let mut kv_cache = kv_cache_mutex.lock();

        if tracing::enabled!(tracing::Level::TRACE)
            && batch.seq_ids[0] == 1
            && kv_cache.get_seq_len(batch.seq_ids[0]) == 0
        {
            for (idx, op) in graph.ops.iter().enumerate().take(60) {
                tracing::trace!("Graph OP idx={}: {:?}", idx, op);
            }
        }

        // Initialize execution context
        let mut ctx = ExecContext::new();

        // Create a local clone of the deq_cache to avoid lock contention on every weight resolution
        let mut local_cache = {
            let cache = self.deq_cache.lock();
            cache.clone()
        };

        // Feed input_ids to context.
        //
        // IMPORTANT: `batch.token_ids` is a PACKED (varlen) buffer — the scheduler
        // concatenates each sequence's tokens back-to-back (a prefill sequence
        // contributes its full prompt length, a decode sequence contributes exactly
        // 1 token), with `batch.cu_seqlens` giving each sequence's boundary in that
        // flat buffer. Sequences in the same batch are NOT required to have equal
        // length (mixed prefill+decode, or multiple prefills of different prompt
        // lengths, is the normal/expected case — see `Scheduler::step`). Reshaping
        // to `(num_seqs, uniform_len)` here would therefore be WRONG whenever
        // lengths differ (previously caused `shape mismatch in reshape` panics on
        // exactly this scenario).
        //
        // Instead we thread the whole batch through the graph as a single
        // "batch of 1" packed sequence of shape `(1, total_tokens)`. This is
        // shape-compatible with every other operator in this function (Embed,
        // RMSNorm, MatMul, Activation, Mul, Add, Scale, TensorScale, Softcap,
        // PleInput/PleLayer all only care about `num_tokens = b_sz * seq_len`,
        // not how it is factored into `(b_sz, seq_len)`). The two operators that
        // DO need real per-sequence boundaries — RoPE position ids (`apply_rope`/
        // `apply_rope_q`, via `cu_seqlens`) and `PagedAttention` (which narrows
        // along the token axis using `cu_seqlens` per sequence, see below) — are
        // updated accordingly.
        let dev = self.device.clone();
        // `num_seqs`/`total_tokens` are validated against `batch.cu_seqlens` above.
        let tokens_t = Tensor::new(batch.token_ids.as_slice(), &dev)?;
        let tokens_t = tokens_t.reshape((1, total_tokens))?;
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
                // `needs_load` is only `false` when the match above matched the
                // `Some(_)` cache arm, so `cache` is guaranteed to be populated here.
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
                let num_mel_bins = self.audio_encoder.as_ref()
                    .map(|e| e.architecture.num_mel_bins())
                    .unwrap_or(128);
                let loaded = if let Some(ref path_str) = active_path {
                    match crate::backends::audio::load_audio(Path::new(path_str), &dev, num_mel_bins) {
                        Ok(t) => t.to_dtype(audio_dtype)?,
                        Err(e) => {
                            tracing::warn!("Failed to load audio from {path_str}: {e}, using zeros");
                            Tensor::zeros((1, num_mel_bins, 3000), audio_dtype, &dev)?
                        }
                    }
                } else {
                    Tensor::zeros((1, num_mel_bins, 3000), audio_dtype, &dev)?
                };
                *cache = Some(loaded.clone());
                *last_path = active_path.clone();
                loaded
            } else {
                // `needs_load` is only `false` when the match above matched the
                // `Some(_)` cache arm, so `cache` is guaranteed to be populated here.
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
                    // Normalize to compute_dtype at the embedding stage so all downstream
                    // activations flow in a consistent dtype (avoids F32/F16 mix from GGUF dequant)
                    let out = if out.dtype() != self.compute_dtype {
                        out.to_dtype(self.compute_dtype)?
                    } else {
                        out
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::RMSNorm { input, weight, output, eps } => {
                    let in_t = ctx.get(input)?;
                    let w_t_raw = self.get_weight(weight, &mut local_cache)?.to_device(in_t.device())?;
                    // Align weight dtype to match the input activation dtype
                    let w_t = if w_t_raw.dtype() != in_t.dtype() {
                        w_t_raw.to_dtype(in_t.dtype())?
                    } else {
                        w_t_raw
                    };
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

                        // Align dtypes: dequantize() always returns F32, but safetensors weights
                        // may be F16. Align both to compute_dtype to avoid matmul dtype mismatch.
                        let in_t_aligned = if in_t.dtype() != self.compute_dtype {
                            in_t.to_dtype(self.compute_dtype)?
                        } else {
                            in_t.clone()
                        };
                        let w_t_aligned = if w_t.dtype() != self.compute_dtype {
                            w_t.to_dtype(self.compute_dtype)?
                        } else {
                            w_t
                        };

                        // Auto-transpose: weight stored row-major [out_features, in_features]
                        let last_dim = in_t_aligned.dim(in_t_aligned.rank() - 1)?;
                        let w_t_final = if w_t_aligned.rank() == 2 && last_dim == w_t_aligned.dim(1)? {
                            w_t_aligned.transpose(0, 1)?
                        } else {
                            w_t_aligned
                        };

                        let rank_in = in_t_aligned.rank();
                        if rank_in == 3 {
                            let (b, m, k) = in_t_aligned.dims3()?;
                            let in_t_2d = in_t_aligned.reshape((b * m, k))?;
                            let res_2d = in_t_2d.matmul(&w_t_final)?;
                            let n = res_2d.dim(1)?;
                            res_2d.reshape((b, m, n))?
                        } else {
                            in_t_aligned.matmul(&w_t_final)?
                        }
                    };

                    let mut out = out;
                    if let Some(bias_name) = bias {
                        let b_t = self.get_weight(bias_name, &mut local_cache)?.to_device(out.device())?;
                        let b_t_aligned = if b_t.dtype() != out.dtype() {
                            b_t.to_dtype(out.dtype())?
                        } else {
                            b_t
                        };
                        out = out.broadcast_add(&b_t_aligned)?;
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
                    let mut att_outputs = Vec::with_capacity(batch.seq_ids.len());

                    for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
                        let block_table = &batch.block_tables[i];
                        let offset = kv_cache.get_seq_len(seq_id);
                        // `q_4d`/`k_4d`/`v_4d` are packed as `(1, total_tokens, ...)` (see the
                        // `input_ids` construction earlier in `forward_pass`), so this
                        // sequence's slice of the token axis is `cu_seqlens[i]..cu_seqlens[i+1]`
                        // — NOT `narrow(0, i, 1)`, which would assume (incorrectly, whenever
                        // sequences in this batch have different lengths) that dim 0 is a
                        // per-sequence batch axis of matching uniform length.
                        let tok_start = batch.cu_seqlens[i] as usize;
                        let this_seq_len = (batch.cu_seqlens[i + 1] - batch.cu_seqlens[i]) as usize;

                        if !is_shared {
                            // `k_4d`/`v_4d` are `Some(..)` exactly when `!is_shared` (see the
                            // construction above), so both are guaranteed populated here.
                            let k_4d_val = k_4d.as_ref().unwrap();
                            let v_4d_val = v_4d.as_ref().unwrap();
                            let k_i = k_4d_val.narrow(1, tok_start, this_seq_len)?;
                            let v_i = v_4d_val.narrow(1, tok_start, this_seq_len)?;

                            let mut t_start = 0;
                            while t_start < this_seq_len {
                                let abs_idx = offset + t_start;
                                let block_idx = abs_idx / block_size;
                                let start_block_offset = abs_idx % block_size;
                                let block_id = *block_table.get(block_idx).ok_or_else(|| anyhow!(
                                    "block_idx {} out of range for this sequence's block_table \
                                     (len={}) — scheduler/backend KV-allocation mismatch",
                                    block_idx, block_table.len()
                                ))?;

                                let chunk_len = std::cmp::min(this_seq_len - t_start, block_size - start_block_offset);
                                let t_end = t_start + chunk_len;
                                
                                let k_chunk = k_i.narrow(1, t_start, chunk_len)?;
                                let v_chunk = v_i.narrow(1, t_start, chunk_len)?;
                                
                                // `HashMap::entry(..).or_insert_with(..)` requires an infallible
                                // closure, but `Tensor::zeros` can fail (e.g. allocation
                                // failure), so we build the entry fallibly outside of
                                // `or_insert_with` and propagate any error via `?` instead of
                                // unwrapping inside the closure.
                                if !gpu_cache.contains_key(&(*layer_idx, block_id)) {
                                    let dev = q_dev;
                                    let dtype = self.kv_config.dtype;
                                    let comp_dtype = self.compute_dtype;

                                    let (k_block, k_scale) = if dtype == KvDtype::Q8 || dtype == KvDtype::Q4 {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), DType::U8, dev)?;
                                        let s = Tensor::zeros((1, block_size, n_kv_heads, 1), comp_dtype, dev)?;
                                        (d, Some(s))
                                    } else {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), comp_dtype, dev)?;
                                        (d, None)
                                    };

                                    let (v_block, v_scale) = if dtype == KvDtype::Q8 || dtype == KvDtype::Q4 {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), DType::U8, dev)?;
                                        let s = Tensor::zeros((1, block_size, n_kv_heads, 1), comp_dtype, dev)?;
                                        (d, Some(s))
                                    } else {
                                        let d = Tensor::zeros((1, block_size, n_kv_heads, head_dim), comp_dtype, dev)?;
                                        (d, None)
                                    };

                                    gpu_cache.insert(
                                        (*layer_idx, block_id),
                                        BlockData { k: k_block, k_scale, v: v_block, v_scale },
                                    );
                                }
                                let block_data = gpu_cache
                                    .get_mut(&(*layer_idx, block_id))
                                    .ok_or_else(|| anyhow!("Block ID {} for layer {} was just inserted but is missing", block_id, layer_idx))?;
                                
                                if self.kv_config.dtype == KvDtype::Q8 || self.kv_config.dtype == KvDtype::Q4 {
                                    // Cast to compute_dtype before quantizing (dequantize() returns F32)
                                    let k_chunk_cast = k_chunk.to_dtype(self.compute_dtype)?;
                                    let v_chunk_cast = v_chunk.to_dtype(self.compute_dtype)?;
                                    let (q_k, q_k_scale) = quantize(&k_chunk_cast, self.kv_config.dtype, self.compute_dtype)?;
                                    let (q_v, q_v_scale) = quantize(&v_chunk_cast, self.kv_config.dtype, self.compute_dtype)?;
                                    
                                    // `k_scale`/`v_scale` are always `Some` for Q8/Q4 blocks:
                                    // they are created `Some` above whenever `dtype` is Q8/Q4
                                    // (this same branch condition), and never reset to `None`.
                                    block_data.k = update_block_tensor(&block_data.k, &q_k, start_block_offset, chunk_len)?;
                                    block_data.k_scale = Some(update_block_tensor(block_data.k_scale.as_ref().unwrap(), &q_k_scale, start_block_offset, chunk_len)?);

                                    block_data.v = update_block_tensor(&block_data.v, &q_v, start_block_offset, chunk_len)?;
                                    block_data.v_scale = Some(update_block_tensor(block_data.v_scale.as_ref().unwrap(), &q_v_scale, start_block_offset, chunk_len)?);
                                } else {
                                    // Ensure dtype matches the initialized block (F16 on Metal/CUDA, F32 on CPU)
                                    let k_chunk_cast = if k_chunk.dtype() != self.compute_dtype {
                                        k_chunk.to_dtype(self.compute_dtype)?
                                    } else {
                                        k_chunk
                                    };
                                    let v_chunk_cast = if v_chunk.dtype() != self.compute_dtype {
                                        v_chunk.to_dtype(self.compute_dtype)?
                                    } else {
                                        v_chunk
                                    };
                                    block_data.k = update_block_tensor(&block_data.k, &k_chunk_cast, start_block_offset, chunk_len)?;
                                    block_data.v = update_block_tensor(&block_data.v, &v_chunk_cast, start_block_offset, chunk_len)?;
                                }
                                
                                t_start = t_end;
                            }
                        }

                        let src_layer = if is_shared {
                            meta.get_kv_source_layer(*layer_idx)
                        } else {
                            *layer_idx
                        };

                        let total_len = offset + this_seq_len;
                        let n_rep = layer_n_heads / n_kv_heads;

                        // Fast path: extend the cached, already-repeated/transposed/contiguous
                        // K/V history instead of rebuilding it from block storage every step
                        // (see `rebuild_kv_history_from_blocks`'s doc comment for why this is
                        // the dominant real cost of long-context decode). Restricted to
                        // non-quantized KV: the Q8/Q4 path applies a Hadamard rotation +
                        // lossy quantization round-trip this shortcut does not replicate.
                        let cached_entry = if is_quantized {
                            None
                        } else {
                            self.kv_history_cache.lock().get(&(seq_id, src_layer)).cloned()
                        };

                        let (k_hist_rep, v_hist_rep, total_seq_len) = match cached_entry {
                            Some((cached_len, cached_k, cached_v)) if cached_len == total_len => {
                                // A KV-shared layer whose owning layer already extended the
                                // cache to this exact length earlier in this same
                                // forward_pass call - nothing left to do.
                                (cached_k, cached_v, cached_len)
                            }
                            Some((cached_len, cached_k, cached_v)) if cached_len == offset && !is_shared => {
                                // Common decode-step case: append just the newly computed
                                // tokens onto the cached history instead of re-reading,
                                // re-dequantizing, and re-repeating the entire past.
                                let k_4d_val = k_4d.as_ref().unwrap();
                                let v_4d_val = v_4d.as_ref().unwrap();
                                let k_new = k_4d_val.narrow(1, tok_start, this_seq_len)?.squeeze(0)?.transpose(0, 1)?;
                                let v_new = v_4d_val.narrow(1, tok_start, this_seq_len)?.squeeze(0)?.transpose(0, 1)?;
                                let k_new_rep = repeat_kv(k_new, n_rep)?.contiguous()?;
                                let v_new_rep = repeat_kv(v_new, n_rep)?.contiguous()?;
                                let k_new_rep = if k_new_rep.dtype() != cached_k.dtype() {
                                    k_new_rep.to_dtype(cached_k.dtype())?
                                } else {
                                    k_new_rep
                                };
                                let v_new_rep = if v_new_rep.dtype() != cached_v.dtype() {
                                    v_new_rep.to_dtype(cached_v.dtype())?
                                } else {
                                    v_new_rep
                                };
                                let k_full = Tensor::cat(&[&cached_k, &k_new_rep], 1)?.contiguous()?;
                                let v_full = Tensor::cat(&[&cached_v, &v_new_rep], 1)?.contiguous()?;
                                (k_full, v_full, total_len)
                            }
                            _ => {
                                // Cache miss, or stale/mismatched length (new sequence,
                                // evicted/reused seq_id, prefix-cache reuse, etc.) - safe
                                // fallback to full reconstruction rather than risk building
                                // on top of the wrong history.
                                let (k, v) = rebuild_kv_history_from_blocks(
                                    &gpu_cache, src_layer, block_table, offset, this_seq_len,
                                    block_size, self.kv_config.dtype, self.compute_dtype, n_rep,
                                )?;
                                (k, v, total_len)
                            }
                        };

                        if !is_quantized {
                            self.kv_history_cache.lock().insert(
                                (seq_id, src_layer),
                                (total_seq_len, k_hist_rep.clone(), v_hist_rep.clone()),
                            );
                        }

                        // Sliding window is applied to a local view only - the cache above
                        // always holds the full un-windowed history so later steps can keep
                        // extending it correctly.
                        let (k_hist_rep, v_hist_rep, total_seq_len) = if let Some(window_len) = meta.get_sliding_window_len(*layer_idx) {
                            if total_seq_len > window_len {
                                let start_idx = total_seq_len - window_len;
                                (
                                    k_hist_rep.narrow(1, start_idx, window_len)?,
                                    v_hist_rep.narrow(1, start_idx, window_len)?,
                                    window_len,
                                )
                            } else {
                                (k_hist_rep, v_hist_rep, total_seq_len)
                            }
                        } else {
                            (k_hist_rep, v_hist_rep, total_seq_len)
                        };

                        let q_i = q_4d_rotated.narrow(1, tok_start, this_seq_len)?.squeeze(0)?;
                        let q_i = q_i.transpose(0, 1)?.contiguous()?; // (n_heads, this_seq_len, head_dim)

                        // Align K dtype to Q for scores matmul
                        let k_hist_rep = if k_hist_rep.dtype() != q_i.dtype() {
                            k_hist_rep.to_dtype(q_i.dtype())?
                        } else {
                            k_hist_rep
                        };

                        let scores = q_i.matmul(&k_hist_rep.transpose(1, 2)?.contiguous()?)?;
                        let scores_scaled = if meta.ple_dim.is_some() {
                            scores
                        } else {
                            (scores / (layer_head_dim as f64).sqrt())?
                        };

                        let num_tokens = this_seq_len;
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
                        // Align V dtype to probs (softmax may upcast to F32 on Metal even with F16 input)
                        let v_hist_rep = if v_hist_rep.dtype() != probs.dtype() {
                            v_hist_rep.to_dtype(probs.dtype())?
                        } else {
                            v_hist_rep
                        };
                        let out_i = probs.matmul(&v_hist_rep)?;
                        let out_i = out_i.transpose(0, 1)?.contiguous()?; // (this_seq_len, n_heads, head_dim)
                        let out_i = out_i.reshape((this_seq_len, layer_n_heads * layer_head_dim))?;
                        att_outputs.push(out_i);
                    }

                    // Concatenate along the TOKEN axis (not stack along a batch axis):
                    // each `out_i` has its own `this_seq_len`, which need not match across
                    // sequences in this batch, so `Tensor::stack` (which requires identical
                    // shapes) would panic on a mixed-length batch. Concatenating back-to-back
                    // in `cu_seqlens` order reconstructs the same packed `(1, total_tokens, ..)`
                    // layout as every other tensor flowing through this graph.
                    let att_out_flat = Tensor::cat(&att_outputs, 0)?;
                    let total_tokens_out = att_out_flat.dim(0)?;
                    let att_out = att_out_flat.reshape((1, total_tokens_out, layer_n_heads * layer_head_dim))?;
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
                    // Align dtypes for element-wise mul
                    let rhs_t = if rhs_t.dtype() != lhs_t.dtype() {
                        rhs_t.to_dtype(lhs_t.dtype())?
                    } else {
                        rhs_t
                    };
                    ctx.insert(output.clone(), lhs_t.broadcast_mul(&rhs_t)?);
                }
                Operator::Add { lhs, rhs, output } => {
                    let lhs_t = ctx.get(lhs)?;
                    let rhs_t = ctx.get(rhs)?;
                    // Align dtypes (residual connections can mix F32 and F16)
                    let rhs_t = if rhs_t.dtype() != lhs_t.dtype() {
                        rhs_t.to_dtype(lhs_t.dtype())?
                    } else {
                        rhs_t
                    };
                    ctx.insert(output.clone(), lhs_t.broadcast_add(&rhs_t)?);
                }
                Operator::VisualEmbed { pixel_values, output } => {
                    let active_path = crate::backends::ACTIVE_IMAGE_PATH.lock().clone();
                    // If no image is provided, emit a dummy embedding and skip
                    // encoding entirely - mirrors AudioEmbed's identical pattern
                    // just below. Previously this ran the vision encoder on a
                    // zero-valued dummy image on EVERY request to any vision-
                    // capable model (even pure-text or audio-only ones) and
                    // unconditionally cached the result into
                    // `self.visual_embeddings`. That cache write is exactly
                    // what SpliceTensors' "is a real image actually active"
                    // guard checks (`has_preloaded`) - so the dummy encode on
                    // the first op of a forward pass silently made every
                    // later op in that SAME pass believe a real image was
                    // preloaded, defeating the guard it runs into moments
                    // later. Confirmed via a real /audio-only request to a
                    // vision+audio-capable model: the dummy vision embedding
                    // got spliced into the audio placeholder token run,
                    // producing a length-mismatch crash unrelated to audio at
                    // all. Never touching the cache when no image is active
                    // fixes this at the source rather than patching every
                    // downstream consumer of `has_preloaded`.
                    let out = if active_path.is_none() {
                        // Dummy: (1, 1, embed_dim) so downstream SpliceTensors/
                        // DeepStackFuse are guaranteed no-ops for this request.
                        let t_emb = ctx.get("text_embeddings")?;
                        let embed_dim = t_emb.dim(2)?;
                        Tensor::zeros((1, 1, embed_dim), t_emb.dtype(), t_emb.device())?
                    } else if let Some(ref enc) = self.vision_encoder {
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
                            // `needs_encode` is only `false` when the match above matched
                            // the `Some(_)` cache arm, so `cache` is guaranteed populated.
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
                    // Only splice if an image is actually attached to this request.
                    // Mirrors the audio path's SpliceAudioTensors guard below: without this
                    // check, ordinary text containing a run of >=16 identical token IDs
                    // (e.g. padding or repeated punctuation) would be silently overwritten
                    // with a zero-valued dummy visual embedding on every vision-capable model.
                    let active_image = crate::backends::ACTIVE_IMAGE_PATH.lock().clone();
                    let has_preloaded = self.visual_embeddings.lock().is_some();
                    let out = if active_image.is_none() && !has_preloaded {
                        t_emb.clone()
                    } else {
                        let v_emb = ctx.get(visual_embeds)?;
                        // Unify dtypes: visual embeddings (may be F32 on CPU) must match text embedding dtype
                        let v_emb = v_emb.to_dtype(t_emb.dtype())?;
                        let v_emb_narrowed = if v_emb.dim(2)? > t_emb.dim(2)? {
                            v_emb.narrow(2, 0, t_emb.dim(2)?)?
                        } else {
                            v_emb.clone()
                        };
                        let token_ids = &batch.token_ids;
                        splice_visual_embeddings(&t_emb, &v_emb_narrowed, token_ids, 0, 0)?
                    };
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
                            // `needs_encode` is only `false` when the match above matched
                            // the `Some(_)` cache arm, so `cache` is guaranteed populated.
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
                    
                    let active_image = crate::backends::ACTIVE_IMAGE_PATH.lock().clone();
                    let has_preloaded = self.visual_embeddings.lock().is_some();
                    if active_image.is_none() && !has_preloaded {
                        // No image attached to this request: DeepStack fusion must not
                        // splice the (meaningless, zero-input-derived) vision features
                        // into ordinary text — same guard as SpliceTensors above.
                        ctx.insert(output.clone(), in_t.clone());
                    } else if let Some(idx) = ds_idx {
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
                    let scale_raw = self.get_weight(scale_tensor, &mut local_cache)?.to_device(in_t.device())?;
                    // Align dtype: weight dequant returns F32 but activations may be F16
                    let scale = if scale_raw.dtype() != in_t.dtype() {
                        scale_raw.to_dtype(in_t.dtype())?
                    } else {
                        scale_raw
                    };
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

                    let lookup_out = if lookup_out.dtype() != self.compute_dtype {
                        lookup_out.to_dtype(self.compute_dtype)?
                    } else {
                        lookup_out
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

                    let context_aware_aligned = if context_aware.dtype() != token_identity.dtype() {
                        context_aware.to_dtype(token_identity.dtype())?
                    } else {
                        context_aware
                    };

                    // Combined: (token_identity + context_aware) * (1 / sqrt(2))
                    let combined = ((token_identity + context_aware_aligned)? * (1.0 / 2.0f64.sqrt()))?;

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
                    let proj_normed_3d_aligned = if proj_normed_3d.dtype() != in_t.dtype() {
                        proj_normed_3d.to_dtype(in_t.dtype())?
                    } else {
                        proj_normed_3d
                    };
                    let out = (in_t + proj_normed_3d_aligned)?;
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

        // Extract logits for next token prediction.
        //
        // `logits` is `(1, total_tokens, vocab)` — the whole batch packed as a
        // single row (see the `input_ids` construction above). Each sequence's
        // next-token logits live at the ABSOLUTE token-axis position of that
        // sequence's LAST token, i.e. `cu_seqlens[i + 1] - 1`, not at a shared
        // `seq_len - 1` offset (which was only ever correct when every sequence
        // in the batch had the same length).
        let mut next_tokens = Vec::with_capacity(num_seqs);
        let mut batch_logits = Vec::with_capacity(num_seqs);

        for i in 0..num_seqs {
            // Per-sequence length (from cu_seqlens, the authoritative source of how many
            // tokens sequence `i` actually contributed to this batch) must be > 0 before
            // we can subtract 1 to index the last token's logits — a batch entry with
            // zero tokens would otherwise underflow (usize) and panic/wrap.
            let this_seq_len = (batch.cu_seqlens[i + 1] - batch.cu_seqlens[i]) as usize;
            if this_seq_len == 0 {
                return Err(anyhow!(
                    "forward_pass: sequence at batch index {} has zero tokens (cu_seqlens[{}]={}, cu_seqlens[{}]={}) — cannot extract next-token logits",
                    i, i, batch.cu_seqlens[i], i + 1, batch.cu_seqlens[i + 1]
                ));
            }
            let last_abs_idx = batch.cu_seqlens[i + 1] as usize - 1;
            let seq_logits_t = logits.narrow(0, 0, 1)?.narrow(1, last_abs_idx, 1)?.squeeze(0)?.squeeze(0)?;
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

    /// Regression test: `forward_pass` must reject a batch whose `cu_seqlens`
    /// doesn't match `token_ids`'s actual total length, rather than silently
    /// reshaping to a wrong uniform `(num_seqs, len)` grid (the original bug —
    /// see `let seq_len = batch.token_ids.len() / b_sz` in the pre-fix code,
    /// which either panicked with a candle "shape mismatch in reshape" error
    /// on a genuinely mixed-length batch, or — worse — silently produced a
    /// wrong-but-valid reshape whenever `token_ids.len()` happened to be evenly
    /// divisible by `seq_ids.len()` despite the per-sequence lengths differing).
    /// This validation runs before the compute graph is touched, so it is
    /// exercised here even on a `CandleBackend` with no model loaded.
    #[test]
    fn forward_pass_rejects_cu_seqlens_mismatched_with_token_ids() {
        let backend = CandleBackend::new();

        // Two sequences of DIFFERENT lengths (3 and 5 tokens) packed back to
        // back, exactly as `Scheduler::step` would build them — but with a
        // deliberately wrong `cu_seqlens` last entry to prove the check fires.
        let batch = BatchInput {
            seq_ids: vec![1, 2],
            token_ids: vec![0u32; 8],
            cu_seqlens: vec![0, 3, 7], // wrong: should end at 8
            block_tables: vec![vec![0], vec![1]],
            is_prefill: vec![true, true],
        };

        let err = backend.forward_pass(&batch).expect_err(
            "cu_seqlens whose last entry doesn't match total token count must be a hard error"
        );
        assert!(
            err.to_string().contains("cu_seqlens"),
            "expected a cu_seqlens-related error, got: {err}"
        );
    }

    #[test]
    fn forward_pass_rejects_cu_seqlens_wrong_length() {
        let backend = CandleBackend::new();
        let batch = BatchInput {
            seq_ids: vec![1, 2],
            token_ids: vec![0u32; 8],
            cu_seqlens: vec![0, 8], // wrong: needs 3 entries for 2 sequences
            block_tables: vec![vec![0], vec![1]],
            is_prefill: vec![true, true],
        };
        let err = backend.forward_pass(&batch).expect_err(
            "cu_seqlens with the wrong number of entries must be a hard error"
        );
        assert!(
            err.to_string().contains("cu_seqlens"),
            "expected a cu_seqlens-related error, got: {err}"
        );
    }
}
