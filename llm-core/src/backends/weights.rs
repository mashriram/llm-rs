//! Weight resolution, ExecContext, and GGUF metadata helpers.
//!
//! This module owns:
//! - `ExecContext` — the per-forward-pass activation tensor map
//! - `get_weight()` — unified weight lookup across safetensors / GGUF / dequant cache
//! - GGUF metadata accessor helpers (arch-agnostic key lookup)
//!
//! The actual loading / caching of weights stays in `CandleBackend::load_weights()`.
//! This module only provides the *resolution* path called during forward passes.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};

use crate::types::GgufMeta;

// ---------------------------------------------------------------------------
// ExecContext — forward-pass activation store
// ---------------------------------------------------------------------------

/// Holds the named intermediate tensors produced during a single forward pass.
///
/// Operators read their inputs from here and write their outputs back. Cleared
/// between forward passes; never shared across threads.
pub(crate) struct ExecContext {
    activations: HashMap<String, Tensor>,
}

impl ExecContext {
    pub fn new() -> Self {
        Self { activations: HashMap::new() }
    }

    /// Retrieve a tensor by name, returning a clear error if it is missing.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.activations
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Activation '{}' not found in forward-pass context", name))
    }

    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.activations.insert(name, tensor);
    }

    pub fn remove(&mut self, name: &str) {
        self.activations.remove(name);
    }
}

// ---------------------------------------------------------------------------
// Weight resolver
// ---------------------------------------------------------------------------

/// Bundled reference to all weight stores owned by `CandleBackend`.
///
/// Passed by reference into `get_weight()` so it can be called without
/// requiring a full `&CandleBackend` borrow (which would conflict with the
/// mutable `local_cache` borrow taken in `forward_pass`).
pub(crate) struct WeightStore<'a> {
    /// Full-precision SafeTensors weights (already on device).
    pub weights: &'a HashMap<String, Tensor>,
    /// Dequantized embedding / norm weights (f32 on CPU for GGUF path).
    pub deq_cache: &'a parking_lot::Mutex<HashMap<String, Tensor>>,
    /// Still-quantized GGUF projection weights; lazily dequantized via qmatmul.
    pub quantized_weights: &'a HashMap<String, candle_core::quantized::QTensor>,
    /// Running byte count of GPU-cached tensors.
    pub gpu_cache_bytes: &'a std::sync::atomic::AtomicU64,
    /// Maximum bytes to keep in GPU dequant cache.
    pub gpu_cache_budget: u64,
    /// If `true`, always dequantize fully to F32 (skips the budget logic).
    pub explicit_dequantize: bool,
    /// Primary compute device.
    pub device: &'a Device,
}

impl<'a> WeightStore<'a> {
    /// Resolve a named weight tensor.
    ///
    /// Priority order:
    /// 1. Full-precision SafeTensors map (already on device).
    /// 2. `local_cache` (forward-pass cache, avoids repeated lock acquisitions).
    /// 3. Dequantization cache (populated eagerly at load time for embed/norm weights).
    /// 4. On-demand dequantization from QTensor, with GPU vs CPU tiering.
    /// 5. `lm_head.weight` alias → `model.embed_tokens.weight` (tied embeddings).
    pub fn get(
        &self,
        name: &str,
        local_cache: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        // 1. SafeTensors
        if let Some(t) = self.weights.get(name) {
            return Ok(t.clone());
        }
        // 2. Forward-pass local cache (avoids locking on every op for the same weight)
        if let Some(t) = local_cache.get(name) {
            return Ok(t.clone());
        }
        // 3. Eager dequantization cache (embed / norm weights pre-dequantized at load time)
        {
            let cache = self.deq_cache.lock();
            if let Some(t) = cache.get(name) {
                local_cache.insert(name.to_string(), t.clone());
                return Ok(t.clone());
            }
        }
        // 4. On-demand dequantization of remaining QTensors
        if let Some(qt) = self.quantized_weights.get(name) {
            let tensor_bytes = qt.shape().elem_count() as u64 * 4; // f32 = 4 bytes

            if !self.explicit_dequantize {
                // Non-budget path: dequantize to f32 and cache in deq_cache.
                let t = qt
                    .dequantize(self.device)
                    .with_context(|| format!("dequantize '{}' to {:?}", name, self.device))?;
                self.deq_cache.lock().insert(name.to_string(), t.clone());
                local_cache.insert(name.to_string(), t.clone());
                return Ok(t);
            }

            let used = self.gpu_cache_bytes.load(Ordering::Relaxed);
            if used + tensor_bytes <= self.gpu_cache_budget {
                // GPU-cached tier: stays on the compute device.
                let t = qt
                    .dequantize(self.device)
                    .with_context(|| format!("dequantize '{}' to {:?}", name, self.device))?;
                self.gpu_cache_bytes.fetch_add(tensor_bytes, Ordering::Relaxed);
                self.deq_cache.lock().insert(name.to_string(), t.clone());
                local_cache.insert(name.to_string(), t.clone());
                Ok(t)
            } else {
                // CPU-offload tier: kept in system RAM, copied to device on each use.
                let t = qt
                    .dequantize(&Device::Cpu)
                    .with_context(|| format!("dequantize '{}' to CPU", name))?;
                self.deq_cache.lock().insert(name.to_string(), t.clone());
                local_cache.insert(name.to_string(), t.clone());
                Ok(t)
            }
        }
        // 5. Tied-embedding fallback: lm_head.weight → model.embed_tokens.weight
        else if name == "lm_head.weight" {
            let embed = "model.embed_tokens.weight";
            if let Some(t) = self.weights.get(embed) {
                let t = t.to_device(self.device)?;
                local_cache.insert(name.to_string(), t.clone());
                return Ok(t);
            }
            let cache = self.deq_cache.lock();
            if let Some(t) = cache.get(embed) {
                let t = t.to_device(self.device)?;
                local_cache.insert(name.to_string(), t.clone());
                return Ok(t);
            }
            Err(anyhow!("Weight '{}' not found (tried tied-embedding alias '{}')", name, embed))
        } else {
            Err(anyhow!("Weight '{}' not found", name))
        }
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata helpers
// ---------------------------------------------------------------------------

type GgufValue = candle_core::quantized::gguf_file::Value;

/// Extract a `u32` from any integer-typed GGUF metadata value.
pub(crate) fn meta_u32(metadata: &GgufMeta, key: &str) -> Option<u32> {
    match metadata.get(key) {
        Some(GgufValue::U8(v))  => Some(*v as u32),
        Some(GgufValue::I8(v))  => Some(*v as u32),
        Some(GgufValue::U16(v)) => Some(*v as u32),
        Some(GgufValue::I16(v)) => Some(*v as u32),
        Some(GgufValue::U32(v)) => Some(*v),
        Some(GgufValue::I32(v)) => Some(*v as u32),
        Some(GgufValue::U64(v)) => Some(*v as u32),
        Some(GgufValue::I64(v)) => Some(*v as u32),
        Some(GgufValue::Array(arr)) => arr.first().and_then(|v| match v {
            GgufValue::U8(v)  => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }),
        _ => None,
    }
}

/// Extract a `f32` from any float-typed GGUF metadata value.
pub(crate) fn meta_f32(metadata: &GgufMeta, key: &str) -> Option<f32> {
    match metadata.get(key) {
        Some(GgufValue::F32(v)) => Some(*v),
        Some(GgufValue::F64(v)) => Some(*v as f32),
        Some(GgufValue::Array(arr)) => arr.first().and_then(|v| match v {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }),
        _ => None,
    }
}

/// Find the full key in `metadata` that ends with `.suffix` or equals `suffix`.
///
/// GGUF keys are architecture-prefixed (e.g. `llama.context_length`). This
/// helper lets callers pass just the suffix (`context_length`) and get back
/// the actual key for whichever architecture the model uses.
pub(crate) fn find_meta_key(metadata: &GgufMeta, suffix: &str) -> Option<String> {
    if metadata.contains_key(suffix) {
        return Some(suffix.to_string());
    }
    for key in metadata.keys() {
        if key.ends_with(&format!(".{}", suffix)) {
            return Some(key.clone());
        }
    }
    None
}

/// Architecture-agnostic `u32` metadata lookup using suffix matching.
pub(crate) fn meta_u32_agnostic(metadata: &GgufMeta, suffix: &str) -> Option<u32> {
    find_meta_key(metadata, suffix).and_then(|k| meta_u32(metadata, &k))
}

/// Architecture-agnostic `f32` metadata lookup using suffix matching.
pub(crate) fn meta_f32_agnostic(metadata: &GgufMeta, suffix: &str) -> Option<f32> {
    find_meta_key(metadata, suffix).and_then(|k| meta_f32(metadata, &k))
}
