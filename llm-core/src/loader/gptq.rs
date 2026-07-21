//! GPTQ dequantization (non-act-order / `desc_act: false` layout).
//!
//! Format confirmed by direct inspection of a real repo's safetensors header
//! (`TheBloke/Llama-2-7B-Chat-GPTQ`, `quant_method: "gptq"`, `desc_act:
//! false`, `sym: true`, `group_size: 128`, `bits: 4`), not assumed from
//! memory:
//!   - `qweight`: I32, shape `[in_features/8, out_features]` - 8 packed
//!     int4 values per int32, packed along the INPUT axis (opposite of
//!     AWQ, which packs along the output axis).
//!   - `qzeros`:  I32, shape `[in_features/group_size, out_features/8]`.
//!   - `scales`:  F16, shape `[in_features/group_size, out_features]`.
//!   - `g_idx`:   I32, shape `[in_features]` - per-input-row group index.
//!     With `desc_act: false` this is just `[0]*group_size + [1]*group_size
//!     + ...` (sequential), but is read and honored rather than assumed, so
//!     an act-order (`desc_act: true`) file with a permuted `g_idx` degrades
//!     to a correctness-preserving slow path (see below) instead of
//!     silently producing wrong output.
//!
//! Dequantized weight: `w[k, n] = (unpack(qweight)[k, n] - (unpack(qzeros)[g_idx[k], n] + 1)) * scales[g_idx[k], n]`.
//! The `+ 1` on the zero-point is a well-documented AutoGPTQ export quirk,
//! not a typo - GPTQ's reference dequant kernel applies it.
//!
//! IMPORTANT — unverified numerically, same caveat as `awq.rs`: this has not
//! been checked against a real GPTQ model's dequantized output from Python
//! (`transformers`/`auto-gptq`) on real hardware. Do that before trusting it
//! beyond a first correctness pass.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};

/// Returns `Some((base_name, kind))` if `name` is one of GPTQ's per-linear
/// tensor suffixes, so the safetensors-loading pass can group them.
pub fn gptq_component(name: &str) -> Option<(&str, &'static str)> {
    for (suffix, kind) in [
        (".qweight", "qweight"),
        (".qzeros", "qzeros"),
        (".scales", "scales"),
        (".g_idx", "g_idx"),
    ] {
        if let Some(base) = name.strip_suffix(suffix) {
            return Some((base, kind));
        }
    }
    None
}

/// Dequantizes one GPTQ linear layer's packed weight into a dense tensor.
///
/// `qweight`: I32 `[in_features/8, out_features]`.
/// `qzeros`:  I32 `[in_features/group_size, out_features/8]`.
/// `scales`:  F16/F32 `[in_features/group_size, out_features]`.
/// `g_idx`:   optional I32 `[in_features]`; when absent, `k / group_size` is
///            used (equivalent to the `desc_act: false` sequential case).
///
/// Returns a dense `[out_features, in_features]` tensor in `scales`' dtype,
/// on `device`, matching the same `[out, in]` HF `nn.Linear` convention
/// `dequantize_awq_linear` returns.
pub fn dequantize_gptq_linear(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: Option<&Tensor>,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let (packed_in, out_features) = qweight.dims2()?;
    let in_features = packed_in * 8;
    let (n_groups, packed_out_zeros) = qzeros.dims2()?;
    if packed_out_zeros * 8 != out_features {
        return Err(anyhow!(
            "GPTQ qweight/qzeros out-axis mismatch: qweight has {} out_features, qzeros implies {}",
            out_features, packed_out_zeros * 8
        ));
    }
    if n_groups != in_features.div_ceil(group_size) {
        return Err(anyhow!(
            "GPTQ qzeros row count {} does not match in_features {} / group_size {}",
            n_groups, in_features, group_size
        ));
    }

    let qweight_u32 = qweight.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let qzeros_u32 = qzeros.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let scales_f32 = scales.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let g_idx_vec: Vec<usize> = match g_idx {
        Some(g) => g.to_dtype(DType::U32)?.to_vec1::<u32>()?.into_iter().map(|v| v as usize).collect(),
        None => (0..in_features).map(|k| k / group_size).collect(),
    };
    if g_idx_vec.len() != in_features {
        return Err(anyhow!("GPTQ g_idx length {} does not match in_features {}", g_idx_vec.len(), in_features));
    }

    // GPTQ packs 8 int4 rows per int32 sequentially (row 0 = bits 0..4, row 1
    // = bits 4..8, ...) - no AWQ-style interleave.
    let mut dequant = vec![0f32; in_features * out_features];
    for k in 0..in_features {
        let packed_row = k / 8;
        let shift_in_row = ((k % 8) * 4) as u32;
        let group = g_idx_vec[k];
        for n in 0..out_features {
            let w_word = qweight_u32[packed_row][n];
            let w_nibble = (w_word >> shift_in_row) & 0xF;

            let packed_col = n / 8;
            let shift_in_col = ((n % 8) * 4) as u32;
            let z_word = qzeros_u32[group][packed_col];
            let z_nibble = (z_word >> shift_in_col) & 0xF;

            let scale = scales_f32[group][n];
            dequant[k * out_features + n] = (w_nibble as f32 - (z_nibble as f32 + 1.0)) * scale;
        }
    }

    let t = Tensor::from_vec(dequant, (in_features, out_features), device)?;
    let t = t.transpose(0, 1)?.contiguous()?; // -> [out_features, in_features]
    Ok(t.to_dtype(scales.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip check: pack 8 known nibble values in GPTQ's sequential
    /// (non-interleaved) row order, dequantize with zero=-1 (so the `+1`
    /// zero-point offset cancels to 0)/scale=1, and confirm the unpacked
    /// values land back in their original row positions. Verifies the
    /// unpacking logic is internally self-consistent - does NOT substitute
    /// for the real numerical check against a Python (auto-gptq/
    /// transformers) reference this module's doc comment calls for.
    #[test]
    fn gptq_sequential_order_round_trips() {
        let device = Device::Cpu;
        // logical_values[k]: output column 0's raw int4 weight for input row k
        // (GPTQ packs 8 input rows per int32, per output column).
        let logical_values: [u32; 8] = [10, 14, 11, 15, 12, 0, 13, 1];
        let mut packed_col0: u32 = 0;
        for (k, &v) in logical_values.iter().enumerate() {
            packed_col0 |= v << (k as u32 * 4);
        }
        // qweight: [in_features/8=1, out_features=8] - only column 0 is
        // meaningful; columns 1..8 are zero-filled and unchecked.
        let mut qweight_data = vec![0u32; 8];
        qweight_data[0] = packed_col0;
        let qweight = Tensor::from_vec(qweight_data, (1, 8), &device).unwrap();
        // qzeros: [n_groups=1, out_features/8=1], all-zero nibbles.
        let qzeros = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();
        let scales = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &device).unwrap();

        let dequant = dequantize_gptq_linear(&qweight, &qzeros, &scales, None, 8, &device).unwrap();
        // dequantize_gptq_linear returns [out_features=8, in_features=8]; row 0
        // is output column 0 across all 8 input positions.
        let got: Vec<f32> = dequant.narrow(0, 0, 1).unwrap().squeeze(0).unwrap().to_vec1().unwrap();
        // formula: (w_nibble - (z_nibble + 1)) * scale = w_nibble - 1
        let expected: Vec<f32> = logical_values.iter().map(|&v| v as f32 - 1.0).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn gptq_component_recognizes_all_four_suffixes() {
        assert_eq!(gptq_component("model.layers.0.mlp.down_proj.qweight"), Some(("model.layers.0.mlp.down_proj", "qweight")));
        assert_eq!(gptq_component("model.layers.0.mlp.down_proj.qzeros"), Some(("model.layers.0.mlp.down_proj", "qzeros")));
        assert_eq!(gptq_component("model.layers.0.mlp.down_proj.scales"), Some(("model.layers.0.mlp.down_proj", "scales")));
        assert_eq!(gptq_component("model.layers.0.mlp.down_proj.g_idx"), Some(("model.layers.0.mlp.down_proj", "g_idx")));
        assert_eq!(gptq_component("model.layers.0.input_layernorm.weight"), None);
    }
}
