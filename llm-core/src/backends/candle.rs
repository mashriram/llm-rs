use std::path::Path;
use std::sync::Mutex;
use std::collections::HashMap;
use anyhow::{Result, anyhow, Context};
use candle_core::{Device, Tensor, DType};

use crate::types::*;
use crate::backend::LlmBackend;
use crate::model::config::parse_config;
use crate::sampler::sample_logits;
use crate::graph::{ComputeGraph, Operator, scan_tensors, build_graph, map_gguf_name};
// HardwareProfile not needed: CUDA device always selected when available, GGUF weights stay on CPU
use crate::backends::vision::VisionEncoder;

// RMSNorm implementation in Candle
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        
        let w_len = self.weight.dim(0)?;
        let last_dim = x_f32.dim(x_f32.rank() - 1)?;
        
        if last_dim != w_len && last_dim % w_len == 0 {
            let rank = x_f32.rank();
            let reshaped = if rank == 3 {
                let (b, s, _) = x_f32.dims3()?;
                let h = last_dim / w_len;
                x_f32.reshape((b, s, h, w_len))?
            } else if rank == 2 {
                let (s, _) = x_f32.dims2()?;
                let h = last_dim / w_len;
                x_f32.reshape((s, h, w_len))?
            } else {
                return Err(anyhow!("Unsupported rank {} in RmsNorm with QK reshaping", rank));
            };
            
            let variance = reshaped.sqr()?.mean_keepdim(reshaped.rank() - 1)?;
            let x_norm_f32 = reshaped.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            let x_norm = x_norm_f32.to_dtype(orig_dtype)?;
            
            let out_reshaped = x_norm.broadcast_mul(&self.weight)?;
            let out = if rank == 3 {
                let (b, s, _, _) = out_reshaped.dims4()?;
                out_reshaped.reshape((b, s, last_dim))?
            } else {
                let (s, _, _) = out_reshaped.dims3()?;
                out_reshaped.reshape((s, last_dim))?
            };
            Ok(out)
        } else {
            let variance = x_f32.sqr()?.mean_keepdim(x_f32.rank() - 1)?;
            let x_norm_f32 = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            let x_norm = x_norm_f32.to_dtype(orig_dtype)?;
            let out = x_norm.broadcast_mul(&self.weight)?;
            Ok(out)
        }
    }
}

fn apply_rope(
    q: &Tensor, // (b_sz, seq_len, n_heads, head_dim)
    k: &Tensor, // (b_sz, seq_len, n_kv_heads, head_dim)
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    rope_theta: f32,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device();
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    let (_, _, n_kv_heads, _) = k.dims4()?;
    
    let mut q_vec = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut k_vec = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    
    let q_stride_seq = n_heads * head_dim;
    let q_stride_batch = seq_len * q_stride_seq;
    
    let k_stride_seq = n_kv_heads * head_dim;
    let k_stride_batch = seq_len * k_stride_seq;
    
    for b in 0..b_sz {
        let seq_id = batch.seq_ids[b];
        let seq_len_before = kv_cache.get_seq_len(seq_id);
        
        for t in 0..seq_len {
            let pos = (seq_len_before + t) as f32;
            
            // Apply to Q
            for h in 0..n_heads {
                let base_idx = b * q_stride_batch + t * q_stride_seq + h * head_dim;
                let half_dim = head_dim / 2;
                for i in 0..half_dim {
                    let idx_1 = base_idx + i;
                    let idx_2 = base_idx + i + half_dim;
                    
                    let theta = pos / (rope_theta).powf((2 * i) as f32 / head_dim as f32);
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();
                    
                    let q1 = q_vec[idx_1];
                    let q2 = q_vec[idx_2];
                    q_vec[idx_1] = q1 * cos_theta - q2 * sin_theta;
                    q_vec[idx_2] = q1 * sin_theta + q2 * cos_theta;
                }
            }
            
            // Apply to K
            for h in 0..n_kv_heads {
                let base_idx = b * k_stride_batch + t * k_stride_seq + h * head_dim;
                let half_dim = head_dim / 2;
                for i in 0..half_dim {
                    let idx_1 = base_idx + i;
                    let idx_2 = base_idx + i + half_dim;
                    
                    let theta = pos / (rope_theta).powf((2 * i) as f32 / head_dim as f32);
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();
                    
                    let k1 = k_vec[idx_1];
                    let k2 = k_vec[idx_2];
                    k_vec[idx_1] = k1 * cos_theta - k2 * sin_theta;
                    k_vec[idx_2] = k1 * sin_theta + k2 * cos_theta;
                }
            }
        }
    }
    
    let q_out = Tensor::from_vec(q_vec, (b_sz, seq_len, n_heads, head_dim), dev)?.to_dtype(q.dtype())?;
    let k_out = Tensor::from_vec(k_vec, (b_sz, seq_len, n_kv_heads, head_dim), dev)?.to_dtype(k.dtype())?;
    Ok((q_out, k_out))
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (n_kv_heads, seq_len, head_dim) = xs.dims3()?;
        let xs = xs.unsqueeze(1)?;
        let xs = xs.expand((n_kv_heads, n_rep, seq_len, head_dim))?;
        let xs = xs.reshape((n_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(xs)
    }
}

// Host-side raw KV cache storage
struct RawKvCache {
    block_size: usize,
    _n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    data_k: Vec<f32>,
    data_v: Vec<f32>,
    num_blocks: usize,
    seq_lengths: HashMap<SeqId, usize>,
}

impl RawKvCache {
    fn new(num_blocks: usize, block_size: usize, n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let size = n_layers * num_blocks * block_size * n_kv_heads * head_dim;
        Self {
            block_size,
            _n_layers: n_layers,
            n_kv_heads,
            head_dim,
            data_k: vec![0.0; size],
            data_v: vec![0.0; size],
            num_blocks,
            seq_lengths: HashMap::new(),
        }
    }

    fn get_index(&self, layer: usize, block: usize, slot: usize, head: usize, dim: usize) -> usize {
        layer * (self.num_blocks * self.block_size * self.n_kv_heads * self.head_dim)
            + block * (self.block_size * self.n_kv_heads * self.head_dim)
            + slot * (self.n_kv_heads * self.head_dim)
            + head * self.head_dim
            + dim
    }



    fn get_slice_k(&self, layer: usize, block: usize, slot: usize) -> &[f32] {
        let start = self.get_index(layer, block, slot, 0, 0);
        let len = self.n_kv_heads * self.head_dim;
        &self.data_k[start..start + len]
    }

    fn get_slice_v(&self, layer: usize, block: usize, slot: usize) -> &[f32] {
        let start = self.get_index(layer, block, slot, 0, 0);
        let len = self.n_kv_heads * self.head_dim;
        &self.data_v[start..start + len]
    }

    fn write_slice_k(&mut self, layer: usize, block: usize, slot: usize, src: &[f32]) {
        let start = self.get_index(layer, block, slot, 0, 0);
        let len = self.n_kv_heads * self.head_dim;
        self.data_k[start..start + len].copy_from_slice(src);
    }

    fn write_slice_v(&mut self, layer: usize, block: usize, slot: usize, src: &[f32]) {
        let start = self.get_index(layer, block, slot, 0, 0);
        let len = self.n_kv_heads * self.head_dim;
        self.data_v[start..start + len].copy_from_slice(src);
    }

    fn get_seq_len(&self, seq_id: SeqId) -> usize {
        *self.seq_lengths.get(&seq_id).unwrap_or(&0)
    }

    fn set_seq_len(&mut self, seq_id: SeqId, len: usize) {
        self.seq_lengths.insert(seq_id, len);
    }
}

// Helper function to splice visual tokens into text embeddings.
fn splice_visual_embeddings(
    text_embeds: &Tensor,
    visual_embeds: &Tensor,
    token_ids: &[u32],
    vision_start_id: u32,
    vision_end_id: u32,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_dim) = text_embeds.dims3()?;
    if b_sz != 1 {
        // Fallback: only support batch size 1 for multimodal prefill
        return Ok(text_embeds.clone());
    }
    
    let mut start_idx = None;
    let mut end_idx = None;
    for (idx, &tok) in token_ids.iter().enumerate() {
        if tok == vision_start_id {
            start_idx = Some(idx);
        } else if tok == vision_end_id {
            end_idx = Some(idx);
        }
    }

    // Fallback: if start/end markers not found, look for common image placeholder token IDs
    if start_idx.is_none() || end_idx.is_none() {
        let mut first_img = None;
        let mut last_img = None;
        for (idx, &tok) in token_ids.iter().enumerate() {
            if tok == 151655 || tok == 32000 || tok == 88253 || tok == 151652 || tok == 151653 {
                if first_img.is_none() {
                    first_img = Some(idx);
                }
                last_img = Some(idx);
            }
        }
        if let (Some(first), Some(last)) = (first_img, last_img) {
            start_idx = Some(first.saturating_sub(1));
            end_idx = Some((last + 1).min(token_ids.len() - 1));
        }
    }

    if let (Some(start), Some(end)) = (start_idx, end_idx) {
        if end > start + 1 {
            let num_pads = end - start - 1;
            let visual_len = visual_embeds.dim(1)?;
            
            let before = text_embeds.narrow(1, 0, start + 1)?;
            let after = text_embeds.narrow(1, end, seq_len - end)?;
            
            let middle = if visual_len == num_pads {
                visual_embeds.clone()
            } else if visual_len > num_pads {
                visual_embeds.narrow(1, 0, num_pads)?
            } else {
                let pad_len = num_pads - visual_len;
                let pad = Tensor::zeros((1, pad_len, hidden_dim), visual_embeds.dtype(), visual_embeds.device())?;
                Tensor::cat(&[visual_embeds, &pad], 1)?
            };
            
            return Ok(Tensor::cat(&[&before, &middle, &after], 1)?);
        }
    }
    Ok(text_embeds.clone())
}

struct ExecContext {
    activations: HashMap<String, Tensor>,
}

impl ExecContext {
    fn new() -> Self {
        Self {
            activations: HashMap::new(),
        }
    }

    fn get(&self, name: &str) -> Result<Tensor> {
        self.activations.get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Activation tensor not found in context: {}", name))
    }

    fn insert(&mut self, name: String, tensor: Tensor) {
        self.activations.insert(name, tensor);
    }

    fn remove(&mut self, name: &str) {
        self.activations.remove(name);
    }
}

pub struct CandleBackend {
    /// Full-precision weights (safetensors path; loaded directly onto device)
    weights: HashMap<String, Tensor>,
    /// Quantized GGUF weights kept on CPU; dequantized lazily
    quantized_weights: HashMap<String, candle_core::quantized::QTensor>,
    /// Dequantization cache: GPU-tier if weight fits in budget, CPU-tier otherwise
    deq_cache: std::sync::Mutex<HashMap<String, Tensor>>,
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
    vision_encoder: Option<VisionEncoder>,
    visual_embeddings: Mutex<Option<Tensor>>,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            quantized_weights: HashMap::new(),
            deq_cache: std::sync::Mutex::new(HashMap::new()),
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
                dtype: KvDtype::F16,
            },
            eos_token_id: 2,
            device: Device::Cpu,
            vision_encoder: None,
            visual_embeddings: Mutex::new(None),
        }
    }

    /// Resolve a weight tensor.
    /// - safetensors weights: already on the primary device, returned directly.
    /// - GGUF QTensors: dequantized once directly to target device:
    ///   * Cache tier: if the dequantized bytes fit in `gpu_cache_budget`, cached on GPU.
    ///   * Offload tier: dequantized once, stored in CPU memory as F16, and copied to GPU on demand.
    fn get_weight(&self, name: &str, local_cache: &mut HashMap<String, Tensor>) -> Result<Tensor> {
        // 1. Full-precision map (safetensors)
        if let Some(t) = self.weights.get(name) {
            return Ok(t.clone());
        }
        // 2. Already in local_cache (GPU or CPU tiered)
        if let Some(t) = local_cache.get(name) {
            // If it is on the target device, return directly.
            // Otherwise (offloaded to CPU), copy it to the target device.
            if std::mem::discriminant(t.device()).eq(&std::mem::discriminant(&self.device)) {
                return Ok(t.clone());
            } else {
                return Ok(t.to_device(&self.device)?);
            }
        }
        // 3. Dequantize from QTensor (done only ONCE per weight name)
        if let Some(qt) = self.quantized_weights.get(name) {
            let t = qt.dequantize(&self.device)
                .with_context(|| format!("dequantize {} to {:?}", name, self.device))?;
            let t = t.to_dtype(DType::F16)?;
            let tensor_bytes = t.elem_count() as u64 * 2; // F16 = 2 bytes

            let mut cache = self.deq_cache.lock().map_err(|_| anyhow!("deq_cache poisoned"))?;
            let used = self.gpu_cache_bytes.load(std::sync::atomic::Ordering::Relaxed);
            if used + tensor_bytes <= self.gpu_cache_budget {
                // GPU cached
                self.gpu_cache_bytes.fetch_add(tensor_bytes, std::sync::atomic::Ordering::Relaxed);
                cache.insert(name.to_string(), t.clone());
                local_cache.insert(name.to_string(), t.clone());
                return Ok(t);
            } else {
                // CPU offloaded - move the F16 tensor to CPU, cache it there,
                // and return the target device (GPU) tensor for immediate use.
                let cpu_t = t.to_device(&Device::Cpu)?;
                cache.insert(name.to_string(), cpu_t.clone());
                local_cache.insert(name.to_string(), cpu_t);
                return Ok(t);
            }
        }
        Err(anyhow!("Weight not found: {}", name))
    }



    pub fn set_visual_embeddings(&self, embeds: Tensor) {
        if let Ok(mut lock) = self.visual_embeddings.lock() {
            *lock = Some(embeds);
        }
    }

    /// Query free VRAM via nvidia-smi; returns 0 on CPU-only systems.
    fn query_free_vram_bytes() -> u64 {
        if !candle_core::utils::cuda_is_available() {
            return 0;
        }
        std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.lines().next().and_then(|l| l.trim().parse::<u64>().ok()))
            .map(|mib| mib * 1024 * 1024)
            .unwrap_or(4 * 1024 * 1024 * 1024) // 4 GB conservative default
    }

    fn get_metadata_u32(metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>, key: &str) -> Option<u32> {
        match metadata.get(key) {
            Some(candle_core::quantized::gguf_file::Value::U8(v)) => Some(*v as u32),
            Some(candle_core::quantized::gguf_file::Value::I8(v)) => Some(*v as u32),
            Some(candle_core::quantized::gguf_file::Value::U16(v)) => Some(*v as u32),
            Some(candle_core::quantized::gguf_file::Value::I16(v)) => Some(*v as u32),
            Some(candle_core::quantized::gguf_file::Value::U32(v)) => Some(*v),
            Some(candle_core::quantized::gguf_file::Value::I32(v)) => Some(*v as u32),
            Some(candle_core::quantized::gguf_file::Value::U64(v)) => Some(*v as u32),
            Some(candle_core::quantized::gguf_file::Value::I64(v)) => Some(*v as u32),
            _ => None,
        }
    }

    fn get_metadata_f32(metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>, key: &str) -> Option<f32> {
        match metadata.get(key) {
            Some(candle_core::quantized::gguf_file::Value::F32(v)) => Some(*v),
            Some(candle_core::quantized::gguf_file::Value::F64(v)) => Some(*v as f32),
            _ => None,
        }
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

    fn load_weights(&mut self, path: &Path) -> Result<ModelMeta> {
        let is_gguf = path.is_file() && path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase() == "gguf").unwrap_or(false);

        // Always try CUDA first; GGUF weights stay quantized on CPU.
        // For safetensors, we load to CPU first and move to GPU layer by layer.
        let dev = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap_or_else(|e| {
                tracing::warn!("CUDA init failed ({e}), falling back to CPU");
                Device::Cpu
            })
        } else {
            Device::Cpu
        };
        self.device = dev.clone();

        // Query free VRAM for cache budget (leave 2.5 GB headroom for activations/KV cache)
        let vram_headroom_bytes: u64 = 2_500 * 1024 * 1024;
        let free_vram = Self::query_free_vram_bytes();
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

            let arch = match model.metadata.get("general.architecture") {
                Some(candle_core::quantized::gguf_file::Value::String(s)) => s.as_str(),
                _ => "llama",
            };

            let vocab_size = match model.metadata.get("tokenizer.ggml.tokens") {
                Some(candle_core::quantized::gguf_file::Value::Array(arr)) => arr.len(),
                _ => 151936,
            };

            let n_layers = Self::get_metadata_u32(&model.metadata, &format!("{}.block_count", arch))
                .ok_or_else(|| anyhow!("Missing block_count in GGUF metadata"))? as usize;

            let hidden_dim = Self::get_metadata_u32(&model.metadata, &format!("{}.embedding_length", arch))
                .ok_or_else(|| anyhow!("Missing embedding_length in GGUF metadata"))? as usize;

            let intermediate_dim = Self::get_metadata_u32(&model.metadata, &format!("{}.feed_forward_length", arch))
                .ok_or_else(|| anyhow!("Missing feed_forward_length in GGUF metadata"))? as usize;

            let n_heads = Self::get_metadata_u32(&model.metadata, &format!("{}.attention.head_count", arch))
                .ok_or_else(|| anyhow!("Missing head_count in GGUF metadata"))? as usize;

            let n_kv_heads = Self::get_metadata_u32(&model.metadata, &format!("{}.attention.head_count_kv", arch))
                .unwrap_or(n_heads as u32) as usize;

            let max_seq_len = Self::get_metadata_u32(&model.metadata, &format!("{}.context_length", arch))
                .unwrap_or(4096) as usize;

            let rope_theta = Self::get_metadata_f32(&model.metadata, &format!("{}.rope.freq_base", arch))
                .unwrap_or(10000.0);

            let head_dim = Self::get_metadata_u32(&model.metadata, &format!("{}.attention.key_length", arch))
                .unwrap_or((hidden_dim / n_heads) as u32) as usize;

            let rms_norm_eps = Self::get_metadata_f32(&model.metadata, &format!("{}.attention.layer_norm_rms_epsilon", arch))
                .unwrap_or(1e-6);

            let eos_token_id = Self::get_metadata_u32(&model.metadata, "tokenizer.ggml.eos_token_id")
                .unwrap_or(2);
            self.eos_token_id = eos_token_id;

            tracing::info!("Loading {} GGUF tensors as QTensor (lazy GPU dequantize)", model.tensor_infos.len());
            let cpu = Device::Cpu;
            for name in model.tensor_infos.keys() {
                let qt = model.tensor(&mut file, name, &cpu)
                    .with_context(|| format!("Failed to read tensor {}", name))?;
                let hf_name = map_gguf_name(name);
                quantized_weights.insert(hf_name, qt);
            }

            let has_vision_encoder = match model.metadata.get("clip.has_vision_encoder") {
                Some(candle_core::quantized::gguf_file::Value::Bool(b)) => *b,
                _ => false,
            };
            let vision_hidden_dim = Self::get_metadata_u32(&model.metadata, "clip.vision.embedding_length").map(|v| v as usize);
            let vision_patch_size = Self::get_metadata_u32(&model.metadata, "clip.vision.patch_size").map(|v| v as usize);
            let vision_image_size = Self::get_metadata_u32(&model.metadata, "clip.vision.image_size").map(|v| v as usize);
            let vision_num_layers = Self::get_metadata_u32(&model.metadata, "clip.vision.block_count").map(|v| v as usize);
            let vision_num_heads = Self::get_metadata_u32(&model.metadata, "clip.vision.attention.head_count").map(|v| v as usize);
            let vision_projection_dim = Self::get_metadata_u32(&model.metadata, "clip.vision.projection_dim").map(|v| v as usize);
            let spatial_merge_size = Self::get_metadata_u32(&model.metadata, "clip.vision.spatial_merge_size").map(|v| v as usize);

            let is_deepstack_layers = match model.metadata.get("clip.vision.is_deepstack_layers") {
                Some(candle_core::quantized::gguf_file::Value::Array(arr)) => {
                    Some(arr.iter().map(|v| match v {
                        candle_core::quantized::gguf_file::Value::Bool(b) => *b,
                        _ => false,
                    }).collect())
                }
                _ => None,
            };

            let projector_type = match model.metadata.get("clip.projector_type") {
                Some(candle_core::quantized::gguf_file::Value::String(s)) => Some(s.clone()),
                _ => None,
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
                tie_word_embeddings: false,
                hidden_act: HiddenAct::SiLU,
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

            // Load SafeTensors files directly into weights map in F16 on selected device
            for sf_file in sf_files {
                let loaded = candle_core::safetensors::load(&sf_file, &Device::Cpu)?;
                for (name, tensor) in loaded {
                    let casted = tensor.to_dtype(DType::F16)?.to_device(&dev)?;
                    weights.insert(name, casted);
                }
            }

            meta.weight_dtype = WeightDtype::F16;
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

        let num_blocks = 1024;
        let kv_cache = RawKvCache::new(
            num_blocks,
            16,
            meta.n_layers,
            meta.n_kv_heads,
            meta.head_dim,
        );

        self.kv_config = KvCacheConfig {
            n_layers: meta.n_layers,
            n_kv_heads: meta.n_kv_heads,
            head_dim: meta.head_dim,
            block_size: 16,
            dtype: KvDtype::F16,
        };

        self.weights = weights;
        self.quantized_weights = quantized_weights;
        self.graph = Some(graph);
        self.meta = Some(meta.clone());
        self.kv_cache = Some(Mutex::new(kv_cache));

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

        Ok(meta)
    }

    fn forward_pass(&self, batch: &BatchInput) -> Result<BatchOutput> {
        let graph = self.graph.as_ref().ok_or_else(|| anyhow!("Compute graph not built"))?;
        let meta = self.meta.as_ref().ok_or_else(|| anyhow!("Model metadata not loaded"))?;
        let kv_cache_mutex = self.kv_cache.as_ref().ok_or_else(|| anyhow!("KV Cache not initialized"))?;
        let mut kv_cache = kv_cache_mutex.lock().map_err(|_| anyhow!("KV Cache mutex poisoned"))?;

        // Initialize execution context
        let mut ctx = ExecContext::new();

        // Create a local clone of the deq_cache to avoid lock contention on every weight resolution
        let mut local_cache = {
            let cache = self.deq_cache.lock().map_err(|_| anyhow!("deq_cache poisoned"))?;
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
            // Retrieve pixel values. We will supply them through a dummy tensor if not preloaded.
            let pixel_val = Tensor::zeros((1, 3, 224, 224), DType::F16, &dev)?;
            ctx.insert("pixel_values".to_string(), pixel_val);
        }

        // Compute last use index for each activation tensor to free VRAM immediately
        let mut last_use: HashMap<String, usize> = HashMap::new();
        for (idx, op) in graph.ops.iter().enumerate() {
            let inputs = match op {
                Operator::Embed { input_ids, .. } => vec![input_ids.clone()],
                Operator::RMSNorm { input, .. } => vec![input.clone()],
                Operator::MatMul { input, .. } => vec![input.clone()],
                Operator::Rope { q, k, .. } => vec![q.clone(), k.clone()],
                Operator::RopeSkip { q, k, .. } => vec![q.clone(), k.clone()],
                Operator::PagedAttention { q, k, v, .. } => vec![q.clone(), k.clone(), v.clone()],
                Operator::Activation { input, .. } => vec![input.clone()],
                Operator::Mul { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
                Operator::Add { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
                Operator::VisualEmbed { pixel_values, .. } => vec![pixel_values.clone()],
                Operator::SpliceTensors { text_embeds, visual_embeds, .. } => vec![text_embeds.clone(), visual_embeds.clone()],
                Operator::DeepStackFuse { input, .. } => vec![input.clone(), "visual_embeddings".to_string()],
            };
            for input in inputs {
                last_use.insert(input, idx);
            }
        }

        // Execute operators sequentially
        for (idx, op) in graph.ops.iter().enumerate() {
            match op {
                Operator::Embed { input_ids, weight, output } => {
                    let ids = ctx.get(input_ids)?;
                    let table = self.get_weight(weight, &mut local_cache)?;
                    let (b_sz, seq_len) = ids.dims2()?;
                    let ids_flat = ids.flatten_all()?;
                    let out_flat = table.index_select(&ids_flat, 0)?;
                    let out = out_flat.reshape((b_sz, seq_len, table.dim(1)?))?;
                    ctx.insert(output.clone(), out);
                }
                Operator::RMSNorm { input, weight, output, eps } => {
                    let in_t = ctx.get(input)?;
                    let w_t = self.get_weight(weight, &mut local_cache)?;
                    let out = RmsNorm::new(w_t, *eps as f64).forward(&in_t)?;
                    ctx.insert(output.clone(), out);
                }
                Operator::MatMul { input, weight, bias, output } => {
                    let in_t = ctx.get(input)?;
                    let w_t = self.get_weight(weight, &mut local_cache)?;

                    // Auto-transpose: weight stored row-major [out_features, in_features]
                    let last_dim = in_t.dim(in_t.rank() - 1)?;
                    let w_t_final = if w_t.rank() == 2 && last_dim == w_t.dim(1)? {
                        w_t.transpose(0, 1)?
                    } else {
                        w_t
                    };

                    let rank_in = in_t.rank();
                    let out = if rank_in == 3 {
                        let (b, m, k) = in_t.dims3()?;
                        let in_t_2d = in_t.reshape((b * m, k))?;
                        let res_2d = in_t_2d.matmul(&w_t_final)?;
                        let n = res_2d.dim(1)?;
                        res_2d.reshape((b, m, n))?
                    } else {
                        in_t.matmul(&w_t_final)?
                    };

                    let mut out = out;
                    if let Some(bias_name) = bias {
                        let b_t = self.get_weight(bias_name, &mut local_cache)?;
                        out = out.broadcast_add(&b_t)?;
                    }
                    ctx.insert(output.clone(), out);
                }
                Operator::Rope { q, k, output_q, output_k, layer_idx: _, rope_theta } => {
                    let q_t = ctx.get(q)?;
                    let k_t = ctx.get(k)?;
                    let (b_sz, seq_len, _) = q_t.dims3()?;
                    let q_4d = q_t.reshape((b_sz, seq_len, meta.n_heads, meta.head_dim))?;
                    let k_4d = k_t.reshape((b_sz, seq_len, meta.n_kv_heads, meta.head_dim))?;
                    let (q_out_4d, k_out_4d) = apply_rope(&q_4d, &k_4d, batch, &kv_cache, *rope_theta)?;
                    let q_out = q_out_4d.reshape((b_sz, seq_len, meta.n_heads * meta.head_dim))?;
                    let k_out = k_out_4d.reshape((b_sz, seq_len, meta.n_kv_heads * meta.head_dim))?;
                    ctx.insert(output_q.clone(), q_out);
                    ctx.insert(output_k.clone(), k_out);
                }
                Operator::RopeSkip { q, k, output_q, output_k } => {
                    let q_t = ctx.get(q)?;
                    let k_t = ctx.get(k)?;
                    ctx.insert(output_q.clone(), q_t.clone());
                    ctx.insert(output_k.clone(), k_t.clone());
                }
                Operator::PagedAttention { q, k, v, output, layer_idx, n_heads, n_kv_heads, head_dim } => {
                    let q_t = ctx.get(q)?;
                    let k_t = ctx.get(k)?;
                    let v_t = ctx.get(v)?;

                    let (b_sz, seq_len, _) = q_t.dims3()?;
                    
                    let q_4d = q_t.reshape((b_sz, seq_len, *n_heads, *head_dim))?;
                    let k_4d = k_t.reshape((b_sz, seq_len, *n_kv_heads, *head_dim))?;
                    let v_4d = v_t.reshape((b_sz, seq_len, *n_kv_heads, *head_dim))?;

                    let q_dev = q_4d.device();
                    let k_flat = k_4d.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let v_flat = v_4d.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

                    let block_size = kv_cache.block_size;
                    let stride_t = n_kv_heads * head_dim;
                    let stride_i = seq_len * stride_t;

                    for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
                        let blocks = &batch.block_tables[i];
                        let start_offset = batch.cu_seqlens[i] as usize;
                        let end_offset = batch.cu_seqlens[i+1] as usize;
                        let num_tokens = end_offset - start_offset;

                        let seq_len_before = kv_cache.get_seq_len(seq_id);

                        for t in 0..num_tokens {
                            let curr_token_idx = seq_len_before + t;
                            let block_idx = curr_token_idx / block_size;
                            let slot_idx = curr_token_idx % block_size;

                            if block_idx >= blocks.len() {
                                return Err(anyhow!("Block table overflow for seq {}: block_idx {}, table len {}", seq_id, block_idx, blocks.len()));
                            }
                            let physical_block = blocks[block_idx] as usize;

                            let start_idx = i * stride_i + t * stride_t;
                            let len = n_kv_heads * head_dim;
                            let k_slice = &k_flat[start_idx..start_idx + len];
                            let v_slice = &v_flat[start_idx..start_idx + len];

                            kv_cache.write_slice_k(*layer_idx, physical_block, slot_idx, k_slice);
                            kv_cache.write_slice_v(*layer_idx, physical_block, slot_idx, v_slice);
                        }
                    }

                    let mut att_outputs = Vec::with_capacity(b_sz);

                    for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
                        let start_offset = batch.cu_seqlens[i] as usize;
                        let end_offset = batch.cu_seqlens[i+1] as usize;
                        let num_tokens = end_offset - start_offset;

                        let seq_len_before = kv_cache.get_seq_len(seq_id);
                        let total_seq_len = seq_len_before + num_tokens;
                        let blocks = &batch.block_tables[i];

                        let mut k_hist = Vec::with_capacity(total_seq_len * n_kv_heads * head_dim);
                        let mut v_hist = Vec::with_capacity(total_seq_len * n_kv_heads * head_dim);

                        for t in 0..total_seq_len {
                            let block_idx = t / block_size;
                            let slot_idx = t % block_size;
                            let physical_block = blocks[block_idx] as usize;

                            k_hist.extend_from_slice(kv_cache.get_slice_k(*layer_idx, physical_block, slot_idx));
                            v_hist.extend_from_slice(kv_cache.get_slice_v(*layer_idx, physical_block, slot_idx));
                        }

                        let k_hist_t = Tensor::from_vec(k_hist, (total_seq_len, *n_kv_heads, *head_dim), q_dev)?.to_dtype(q_t.dtype())?;
                        let v_hist_t = Tensor::from_vec(v_hist, (total_seq_len, *n_kv_heads, *head_dim), q_dev)?.to_dtype(q_t.dtype())?;

                        let q_i = q_4d.narrow(0, i, 1)?.squeeze(0)?;
                        let q_i = q_i.transpose(0, 1)?.contiguous()?; // (n_heads, num_tokens, head_dim)
                        
                        let n_rep = n_heads / n_kv_heads;
                        let k_hist_t = repeat_kv(k_hist_t.transpose(0, 1)?, n_rep)?.contiguous()?;
                        let v_hist_t = repeat_kv(v_hist_t.transpose(0, 1)?, n_rep)?.contiguous()?;

                        let scores = q_i.matmul(&k_hist_t.transpose(1, 2)?.contiguous()?)?;
                        let scores_scaled = (scores / (*head_dim as f64).sqrt())?;

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
                        let out_i = probs.matmul(&v_hist_t)?;
                        let out_i = out_i.transpose(0, 1)?.contiguous()?; // (num_tokens, n_heads, head_dim)
                        let out_i = out_i.reshape((num_tokens, n_heads * head_dim))?;
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
                    let p_val = ctx.get(pixel_values)?;
                    let out = if let Some(ref enc) = self.vision_encoder {
                        enc.encode(&p_val)?
                    } else if let Some(ref preloaded) = *self.visual_embeddings.lock().unwrap() {
                        preloaded.clone()
                    } else {
                        return Err(anyhow!("VisionEncoder / visual_embeddings not loaded for VisualEmbed"));
                    };
                    ctx.insert(output.clone(), out);
                }
                Operator::SpliceTensors { text_embeds, visual_embeds, output } => {
                    let t_emb = ctx.get(text_embeds)?;
                    let v_emb = ctx.get(visual_embeds)?;
                    let token_ids = &batch.token_ids;
                    let out = splice_visual_embeddings(&t_emb, &v_emb, token_ids, 151652, 151653)?;
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
                        let fused = splice_visual_embeddings(&in_t, &ds_feat, token_ids, 151652, 151653)?;
                        ctx.insert(output.clone(), fused);
                    } else {
                        ctx.insert(output.clone(), in_t.clone());
                    }
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
