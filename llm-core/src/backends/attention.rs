//! Attention-related math primitives: RMSNorm, RoPE, and KV repeat.
//!
//! These are pure functions/types with no state. They take tensors and
//! return tensors. The `CandleBackend` forward pass imports them directly.

use anyhow::{anyhow, Result};
use candle_core::{DType, Tensor};

use crate::types::BatchInput;

// ---------------------------------------------------------------------------
// RawKvCache — host-side sequence-length tracker
// ---------------------------------------------------------------------------

/// Tracks how many tokens have already been processed per sequence.
/// The actual KV tensors live in `CandleBackend::gpu_kv_cache`.
pub(crate) struct RawKvCache {
    seq_lengths: std::collections::HashMap<crate::types::SeqId, usize>,
}

impl RawKvCache {
    pub fn new() -> Self {
        Self { seq_lengths: std::collections::HashMap::new() }
    }

    pub fn get_seq_len(&self, seq_id: crate::types::SeqId) -> usize {
        *self.seq_lengths.get(&seq_id).unwrap_or(&0)
    }

    pub fn set_seq_len(&mut self, seq_id: crate::types::SeqId, len: usize) {
        self.seq_lengths.insert(seq_id, len);
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// When `is_gemma` is `true`, applies Gemma-style `(1 + weight)` scaling
/// (for HuggingFace SafeTensors format only — GGUF Gemma weights are already
/// stored without the +1 offset).
pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f64,
    is_gemma: bool,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64, is_gemma: bool) -> Self {
        Self { weight, eps, is_gemma }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;

        let w_len = self.weight.dim(0)?;
        let last_dim = x_f32.dim(x_f32.rank() - 1)?;

        // Gemma HF format stores weights as the delta from 1.0.
        let w_scaled = if self.is_gemma {
            self.weight.affine(1.0, 1.0)?
        } else {
            self.weight.clone()
        };

        // QK-Norm case: weight covers a head_dim slice but x has full concat dim.
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

            let out_reshaped = x_norm.broadcast_mul(&w_scaled)?;
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
            let denom = (variance + self.eps)?.sqrt()?;
            let x_norm_f32 = x_f32.broadcast_div(&denom)?;
            let x_norm = x_norm_f32.to_dtype(orig_dtype)?;
            Ok(x_norm.broadcast_mul(&w_scaled)?)
        }
    }
}

/// RMSNorm without a learned scale weight (used in vision encoder blocks).
pub(crate) fn rms_norm_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(x_f32.rank() - 1)?;
    let denom = (variance + eps)?.sqrt()?;
    let x_norm_f32 = x_f32.broadcast_div(&denom)?;
    Ok(x_norm_f32.to_dtype(orig_dtype)?)
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (interleaved, GGUF-compatible)
// ---------------------------------------------------------------------------

/// Apply interleaved RoPE to both Q and K.
///
/// Weights stored in GGUF use an interleaved rotation layout where positions
/// `(2i, 2i+1)` are rotated as a pair. This matches the GPU CubeCL kernel.
///
/// Shapes:
/// - `q`: `[b, seq_len, n_heads, head_dim]`
/// - `k`: `[b, seq_len, n_kv_heads, head_dim]`
pub(crate) fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    rope_theta: f32,
) -> Result<(Tensor, Tensor)> {
    if rope_theta == 0.0 {
        return Ok((q.clone(), k.clone()));
    }
    let dev = q.device();
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    let (_, _, n_kv_heads, _) = k.dims4()?;
    let half_dim = head_dim / 2;

    let inv_freq = build_inv_freq(half_dim, head_dim, rope_theta, dev)?;
    let (cos, sin) = build_cos_sin(b_sz, seq_len, batch, kv_cache, &inv_freq, q.dtype(), dev)?;
    let cos = cos.reshape((b_sz, seq_len, 1, half_dim, 1))?;
    let sin = sin.reshape((b_sz, seq_len, 1, half_dim, 1))?;

    let q_out = rotate_interleaved(q, &cos, &sin, b_sz, seq_len, n_heads, head_dim)?;
    let k_out = rotate_interleaved(k, &cos, &sin, b_sz, seq_len, n_kv_heads, head_dim)?;
    Ok((q_out, k_out))
}

/// Apply interleaved RoPE to Q only (for architectures with separate Q/K rope).
pub(crate) fn apply_rope_q(
    q: &Tensor,
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    rope_theta: f32,
) -> Result<Tensor> {
    if rope_theta == 0.0 {
        return Ok(q.clone());
    }
    let dev = q.device();
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    let half_dim = head_dim / 2;

    let inv_freq = build_inv_freq(half_dim, head_dim, rope_theta, dev)?;
    let (cos, sin) = build_cos_sin(b_sz, seq_len, batch, kv_cache, &inv_freq, q.dtype(), dev)?;
    let cos = cos.reshape((b_sz, seq_len, 1, half_dim, 1))?;
    let sin = sin.reshape((b_sz, seq_len, 1, half_dim, 1))?;

    rotate_interleaved(q, &cos, &sin, b_sz, seq_len, n_heads, head_dim)
}

/// Build the inverse frequency vector: `1 / theta^(2i / head_dim)`.
fn build_inv_freq(
    half_dim: usize,
    head_dim: usize,
    rope_theta: f32,
    dev: &candle_core::Device,
) -> Result<Tensor> {
    let inv_freq_vec: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    Ok(Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?)
}

/// Build per-position cos/sin matrices incorporating KV-cache offset.
fn build_cos_sin(
    b_sz: usize,
    seq_len: usize,
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    inv_freq: &Tensor,
    dtype: DType,
    dev: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let mut pos_vec = Vec::with_capacity(b_sz * seq_len);
    for b in 0..b_sz {
        let seq_id = batch.seq_ids[b];
        let offset = kv_cache.get_seq_len(seq_id);
        for t in 0..seq_len {
            pos_vec.push((offset + t) as f32);
        }
    }
    let pos = Tensor::from_vec(pos_vec, (b_sz * seq_len, 1), dev)?;
    let freqs = pos.matmul(inv_freq)?;
    let cos = freqs.cos()?.to_dtype(dtype)?.reshape((b_sz, seq_len, freqs.dim(1)?))?;
    let sin = freqs.sin()?.to_dtype(dtype)?.reshape((b_sz, seq_len, freqs.dim(1)?))?;
    Ok((cos, sin))
}

/// Apply interleaved rotation to a Q or K tensor.
fn rotate_interleaved(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    b_sz: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let half_dim = head_dim / 2;
    let x_reshaped = x.reshape((b_sz, seq_len, n_heads, half_dim, 2))?;
    let x1 = x_reshaped.narrow(candle_core::D::Minus1, 0, 1)?;
    let x2 = x_reshaped.narrow(candle_core::D::Minus1, 1, 1)?;

    let out1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let out2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
    Ok(Tensor::cat(&[out1, out2], candle_core::D::Minus1)?.reshape((b_sz, seq_len, n_heads, head_dim))?)
}

// ---------------------------------------------------------------------------
// GQA helper
// ---------------------------------------------------------------------------

/// Repeat KV heads to match the number of Q heads (Grouped Query Attention).
pub(crate) fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (n_kv_heads, seq_len, head_dim) = xs.dims3()?;
    let xs = xs.unsqueeze(1)?;
    let xs = xs.expand((n_kv_heads, n_rep, seq_len, head_dim))?;
    Ok(xs.reshape((n_kv_heads * n_rep, seq_len, head_dim))?)
}
