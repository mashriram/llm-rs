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
        let w_scaled = if is_gemma {
            weight.affine(1.0, 1.0).unwrap_or_else(|_| weight.clone())
        } else {
            weight
        };
        Self { weight: w_scaled, eps, is_gemma }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let w_len = self.weight.dim(0)?;
        let last_dim = x.dim(x.rank() - 1)?;

        let w_scaled = if self.weight.dtype() != orig_dtype {
            self.weight.to_dtype(orig_dtype)?
        } else {
            self.weight.clone()
        };

        // QK-Norm case: weight covers a head_dim slice but x has full concat dim.
        if last_dim != w_len && last_dim % w_len == 0 {
            let rank = x.rank();
            let reshaped = if rank == 3 {
                let (b, s, _) = x.dims3()?;
                let h = last_dim / w_len;
                x.reshape((b, s, h, w_len))?
            } else if rank == 2 {
                let (s, _) = x.dims2()?;
                let h = last_dim / w_len;
                x.reshape((s, h, w_len))?
            } else {
                return Err(anyhow!("Unsupported rank {} in RmsNorm with QK reshaping", rank));
            };

            let variance = reshaped.sqr()?.mean_keepdim(reshaped.rank() - 1)?;
            let x_norm = reshaped.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

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
            let variance = x.sqr()?.mean_keepdim(x.rank() - 1)?;
            let denom = (variance + self.eps)?.sqrt()?;
            let x_norm = x.broadcast_div(&denom)?;
            Ok(x_norm.broadcast_mul(&w_scaled)?)
        }
    }
}

/// RMSNorm without a learned scale weight (used in vision encoder blocks).
pub(crate) fn rms_norm_no_scale(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x_f32 = if orig_dtype == DType::F32 { x.clone() } else { x.to_dtype(DType::F32)? };
    let variance = x_f32.sqr()?.mean_keepdim(x_f32.rank() - 1)?;
    let denom = (variance + eps)?.sqrt()?;
    let x_norm_f32 = x_f32.broadcast_div(&denom)?;
    if orig_dtype == DType::F32 { Ok(x_norm_f32) } else { Ok(x_norm_f32.to_dtype(orig_dtype)?) }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (interleaved, GGUF-compatible)
// ---------------------------------------------------------------------------

/// Apply interleaved RoPE to both Q and K using precomputed cos/sin.
pub(crate) fn apply_rope_with_cos_sin(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_theta: f32,
) -> Result<(Tensor, Tensor)> {
    if rope_theta == 0.0 {
        return Ok((q.clone(), k.clone()));
    }
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    let (_, _, n_kv_heads, _) = k.dims4()?;

    let q_out = rotate_interleaved(q, cos, sin, b_sz, seq_len, n_heads, head_dim)?;
    let k_out = rotate_interleaved(k, cos, sin, b_sz, seq_len, n_kv_heads, head_dim)?;
    Ok((q_out, k_out))
}

/// Apply interleaved RoPE to Q only using precomputed cos/sin.
pub(crate) fn apply_rope_q_with_cos_sin(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_theta: f32,
) -> Result<Tensor> {
    if rope_theta == 0.0 {
        return Ok(q.clone());
    }
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    rotate_interleaved(q, cos, sin, b_sz, seq_len, n_heads, head_dim)
}

/// Apply interleaved RoPE to both Q and K.
pub(crate) fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    rope_theta: f32,
    inv_freq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if rope_theta == 0.0 {
        return Ok((q.clone(), k.clone()));
    }
    let dev = q.device();
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    let (_, _, n_kv_heads, _) = k.dims4()?;
    let half_dim = head_dim / 2;

    let (cos, sin) = build_cos_sin(b_sz, seq_len, batch, kv_cache, inv_freq, q.dtype(), dev)?;
    let cos = cos.reshape((b_sz, seq_len, 1, half_dim, 1))?;
    let sin = sin.reshape((b_sz, seq_len, 1, half_dim, 1))?;

    let q_out = rotate_interleaved(q, &cos, &sin, b_sz, seq_len, n_heads, head_dim)?;
    let k_out = rotate_interleaved(k, &cos, &sin, b_sz, seq_len, n_kv_heads, head_dim)?;
    Ok((q_out, k_out))
}

/// Apply interleaved RoPE to Q only.
pub(crate) fn apply_rope_q(
    q: &Tensor,
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    rope_theta: f32,
    inv_freq: &Tensor,
) -> Result<Tensor> {
    if rope_theta == 0.0 {
        return Ok(q.clone());
    }
    let dev = q.device();
    let (b_sz, seq_len, n_heads, head_dim) = q.dims4()?;
    let half_dim = head_dim / 2;

    let (cos, sin) = build_cos_sin(b_sz, seq_len, batch, kv_cache, inv_freq, q.dtype(), dev)?;
    let cos = cos.reshape((b_sz, seq_len, 1, half_dim))?;
    let sin = sin.reshape((b_sz, seq_len, 1, half_dim))?;

    rotate_interleaved(q, &cos, &sin, b_sz, seq_len, n_heads, head_dim)
}

/// Build the inverse frequency vector: `1 / theta^(2i / head_dim)`. Pure
/// function of `(half_dim, head_dim, rope_theta)` - callers should cache
/// the result across calls when these don't change (see
/// `CandleBackend::get_inv_freq`), since this used to be rebuilt (host
/// `Vec<f32>` alloc + fresh device upload) on every single RoPE
/// application before that caching existed.
pub(crate) fn build_inv_freq(
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
///
/// `q`/`k` are packed as `(1, total_tokens, n_heads, head_dim)` — a single
/// "batch of 1" row holding every sequence's tokens back-to-back (see
/// `CandleBackend::forward_pass`'s `input_ids` construction). Each token's
/// RoPE position must therefore be derived from ITS OWN sequence's KV-cache
/// offset plus its offset WITHIN that sequence — not from a single shared
/// `kv_cache.get_seq_len` call for a `b`-th "batch row", which silently
/// assumed every sequence in the batch had the same length and were arranged
/// one-per-batch-row. `batch.cu_seqlens` gives the exact token-axis boundary
/// of each sequence in the packed buffer, so we walk those boundaries instead.
pub(crate) fn build_cos_sin(
    b_sz: usize,
    seq_len: usize,
    batch: &BatchInput,
    kv_cache: &RawKvCache,
    inv_freq: &Tensor,
    dtype: DType,
    dev: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let total_tokens = b_sz * seq_len;
    if batch.cu_seqlens.last().copied() != Some(total_tokens as u32) {
        return Err(anyhow!(
            "build_cos_sin: cu_seqlens last entry ({:?}) does not match q/k total token count ({})",
            batch.cu_seqlens.last(), total_tokens
        ));
    }
    let mut pos_vec = Vec::with_capacity(total_tokens);
    for (i, &seq_id) in batch.seq_ids.iter().enumerate() {
        let offset = kv_cache.get_seq_len(seq_id);
        let start = batch.cu_seqlens[i] as usize;
        let end = batch.cu_seqlens[i + 1] as usize;
        for t in 0..(end - start) {
            pos_vec.push((offset + t) as f32);
        }
    }
    if pos_vec.len() != total_tokens {
        return Err(anyhow!(
            "build_cos_sin: cu_seqlens-derived token count ({}) does not match q/k total token count ({})",
            pos_vec.len(), total_tokens
        ));
    }
    let pos = Tensor::from_vec(pos_vec, (total_tokens, 1), dev)?;
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
    _b_sz: usize,
    _seq_len: usize,
    _n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let half_dim = head_dim / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;
    let neg_x2 = x2.neg()?;
    let rotate_x = Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)?;
    let cos_cat = Tensor::cat(&[cos, cos], candle_core::D::Minus1)?;
    let sin_cat = Tensor::cat(&[sin, sin], candle_core::D::Minus1)?;
    Ok((x.broadcast_mul(&cos_cat)? + rotate_x.broadcast_mul(&sin_cat)?)?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BatchInput;
    use candle_core::Device;

    /// Regression test for the mixed-length-batch bug: a batch containing two
    /// PREFILL sequences of DIFFERENT prompt lengths, packed back-to-back per
    /// `cu_seqlens` (exactly what `Scheduler::step` produces — see
    /// `llm-scheduler/src/scheduler.rs`'s `BatchInput` construction). Before the
    /// fix, `CandleBackend::forward_pass` reshaped `token_ids` to a uniform
    /// `(num_seqs, token_ids.len() / num_seqs)` grid, which panics outright when
    /// lengths differ, and `build_cos_sin` assigned RoPE positions assuming one
    /// batch "row" per sequence of equal length. This test drives `build_cos_sin`
    /// directly (the position-id half of the fix) with a 3-token and a 5-token
    /// sequence packed into one `total_tokens = 8` batch and asserts:
    /// 1. it does not error / panic on the length mismatch, and
    /// 2. each token gets the RoPE position belonging to ITS OWN sequence
    ///    (continuing from that sequence's individual KV-cache offset), with no
    ///    cross-contamination between the two sequences' position ranges.
    #[test]
    fn build_cos_sin_handles_mixed_length_batch_without_cross_contamination() {
        let dev = Device::Cpu;

        // Sequence A: seq_id=1, already has 10 cached tokens, contributes 3 new
        // tokens this step (e.g. a partial-prefill continuation).
        // Sequence B: seq_id=2, fresh (0 cached tokens), contributes 5 new
        // tokens this step (an initial prompt prefill).
        // Packed token_ids buffer would be [A0,A1,A2, B0,B1,B2,B3,B4] (8 total);
        // cu_seqlens = [0, 3, 8] marks that boundary.
        let batch = BatchInput {
            seq_ids: vec![1, 2],
            token_ids: vec![0u32; 8], // token values are irrelevant to build_cos_sin
            cu_seqlens: vec![0, 3, 8],
            block_tables: vec![vec![0], vec![1]],
            is_prefill: vec![true, true],
        };

        let mut kv_cache = RawKvCache::new();
        kv_cache.set_seq_len(1, 10);
        kv_cache.set_seq_len(2, 0);

        let head_dim = 4;
        let half_dim = head_dim / 2;
        let inv_freq = build_inv_freq(half_dim, head_dim, 10000.0, &dev).unwrap();

        let total_tokens = batch.token_ids.len();
        let (cos, _sin) = build_cos_sin(1, total_tokens, &batch, &kv_cache, &inv_freq, DType::F32, &dev)
            .expect("build_cos_sin must not error on a mixed-length packed batch");

        assert_eq!(cos.dims(), &[1, total_tokens, half_dim]);

        // Recover the per-token cos(pos * inv_freq[0]) implied value and back out
        // the effective position for the first frequency component, then check
        // sequence A's 3 tokens continue from offset 10 (positions 10,11,12) and
        // sequence B's 5 tokens start fresh from offset 0 (positions 0,1,2,3,4) —
        // i.e. no cross-contamination of KV-cache offsets between sequences, and
        // no assumption that both sequences share the same length.
        let cos_vals = cos.reshape((total_tokens, half_dim)).unwrap()
            .narrow(1, 0, 1).unwrap()
            .squeeze(1).unwrap()
            .to_vec1::<f32>().unwrap();

        let expected_positions: [f32; 8] = [10.0, 11.0, 12.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        for (i, &expected_pos) in expected_positions.iter().enumerate() {
            let expected_cos = expected_pos.cos(); // inv_freq[0] == 1.0
            assert!(
                (cos_vals[i] - expected_cos).abs() < 1e-4,
                "token {i}: expected cos(pos={expected_pos}) = {expected_cos}, got {}",
                cos_vals[i]
            );
        }
    }

    #[test]
    fn build_cos_sin_rejects_cu_seqlens_token_count_mismatch() {
        let dev = Device::Cpu;
        let batch = BatchInput {
            seq_ids: vec![1],
            token_ids: vec![0u32; 4],
            cu_seqlens: vec![0, 3], // wrong: should be 4 to match token_ids.len()
            block_tables: vec![vec![0]],
            is_prefill: vec![true],
        };
        let kv_cache = RawKvCache::new();
        let head_dim = 4;
        let inv_freq = build_inv_freq(head_dim / 2, head_dim, 10000.0, &dev).unwrap();
        let result = build_cos_sin(1, 4, &batch, &kv_cache, &inv_freq, DType::F32, &dev);
        assert!(result.is_err(), "cu_seqlens/token-count mismatch must be a hard error, not silently truncated/padded");
    }
}
