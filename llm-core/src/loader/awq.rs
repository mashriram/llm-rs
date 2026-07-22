//! AWQ ("Activation-aware Weight Quantization") dequantization.
//!
//! Format confirmed by direct inspection of a real repo's safetensors header
//! (`TheBloke/Llama-2-7B-AWQ`, `quant_method: "awq"`, `version: "gemm"`,
//! `group_size: 128`, `bits: 4`), not assumed from memory:
//!   - `qweight`: I32, shape `[in_features, out_features/8]` - 8 packed
//!     int4 values per int32, packed along the OUTPUT axis.
//!   - `qzeros`:  I32, shape `[in_features/group_size, out_features/8]` -
//!     same packing, one zero-point per group per output channel.
//!   - `scales`:  F16, shape `[in_features/group_size, out_features]`.
//!
//! Dequantized weight: `w[k, n] = (unpack(qweight)[k, n] - unpack(qzeros)[k/group_size, n]) * scales[k/group_size, n]`.
//!
//! Verified against AutoAWQ's actual reference implementation
//! (`awq/utils/packing_utils.py`'s `unpack_awq`/`reverse_awq_order`/
//! `dequantize_gemm`, casper-hansen/AutoAWQ) and against a real end-to-end
//! generation test on real CUDA hardware (`Qwen/Qwen2.5-1.5B-Instruct-AWQ`).
//!
//! AWQ's packing has two logically separate steps: (1) plain sequential
//! nibble unpacking (nibble `i`, 0..8, at bit-shift `i*4` - no interleave at
//! this stage), then (2) a column permutation applied to *groups of 8*
//! already-unpacked columns, using `AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3,
//! 7]` - i.e. logical position `i` within a group of 8 reads its value from
//! the nibble unpacked at raw position `AWQ_REVERSE_ORDER[i]`.
//!
//! An earlier version of this file used `AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5,
//! 7]` directly as the unpack order. That constant is real (it appears in
//! AutoAWQ too), but it's the *pack* direction's permutation, not the
//! *unpack* direction's - the two are inverse permutations of each other
//! (`AWQ_REVERSE_ORDER` is literally `AWQ_ORDER` inverted), so using the
//! wrong one here silently produced a real 4-bit weight matrix, just the
//! wrong one, everywhere. This ran end-to-end without erroring and produced
//! plausible-looking (but garbage) generated text - only caught by actually
//! reading real generated tokens on real hardware, not by the unit tests
//! (which only proved this file was internally self-consistent, and by
//! comparing against a from-scratch Python reference implementing the *same*
//! assumed order, which is the same mistake twice, not independent
//! verification).
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};

/// AWQ's GEMM-kernel unpack order: logical position `i` (0..8) within one
/// packed `qweight`/`qzeros` int32 group of 8 columns is read from the nibble
/// at bit-shift `AWQ_REVERSE_ORDER[i] * 4`. See module doc comment.
const AWQ_REVERSE_ORDER: [u32; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Returns `Some(base_name)` if `name` is one of AWQ's three per-linear
/// tensor suffixes, so the safetensors-loading pass can group them.
pub fn awq_component(name: &str) -> Option<(&str, &'static str)> {
    for (suffix, kind) in [(".qweight", "qweight"), (".qzeros", "qzeros"), (".scales", "scales")] {
        if let Some(base) = name.strip_suffix(suffix) {
            return Some((base, kind));
        }
    }
    None
}

/// Dequantizes one AWQ linear layer's packed weight into a dense tensor.
///
/// `qweight`: I32 `[in_features, out_features/8]` (CPU tensor).
/// `qzeros`:  I32 `[in_features/group_size, out_features/8]`.
/// `scales`:  F16/F32 `[in_features/group_size, out_features]`.
///
/// Returns a dense `[out_features, in_features]` tensor in `scales`' dtype,
/// on `device` - matching the `[out, in]` convention the rest of this
/// codebase's safetensors loading path expects for a standard HF `nn.Linear`
/// weight (see `vision.rs`'s `linear()` helper for the same convention
/// question on the GGUF/mmproj side).
pub fn dequantize_awq_linear(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let (in_features, packed_out) = qweight.dims2()?;
    let out_features = packed_out * 8;
    let (n_groups, packed_out_zeros) = qzeros.dims2()?;
    if packed_out_zeros != packed_out {
        return Err(anyhow!(
            "AWQ qweight/qzeros out-axis mismatch: qweight implies {} packed columns, qzeros has {}",
            packed_out, packed_out_zeros
        ));
    }
    if n_groups != in_features.div_ceil(group_size) {
        return Err(anyhow!(
            "AWQ qzeros row count {} does not match in_features {} / group_size {}",
            n_groups, in_features, group_size
        ));
    }

    let qweight_u32 = qweight.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let qzeros_u32 = qzeros.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let scales_f32 = scales.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    // Unpack into [in_features, out_features] row-major, f32, then transpose
    // to [out_features, in_features] once at the end (one contiguous copy,
    // rather than scattering into transposed layout element-by-element).
    let mut dequant = vec![0f32; in_features * out_features];
    for k in 0..in_features {
        let group = k / group_size;
        for packed_col in 0..packed_out {
            let w_word = qweight_u32[k][packed_col];
            let z_word = qzeros_u32[group][packed_col];
            for i in 0..8u32 {
                let shift = AWQ_REVERSE_ORDER[i as usize] * 4;
                let w_nibble = (w_word >> shift) & 0xF;
                let z_nibble = (z_word >> shift) & 0xF;
                let n = packed_col * 8 + i as usize;
                let scale = scales_f32[group][n];
                dequant[k * out_features + n] = (w_nibble as f32 - z_nibble as f32) * scale;
            }
        }
    }

    let t = Tensor::from_vec(dequant, (in_features, out_features), device)?;
    let t = t.transpose(0, 1)?.contiguous()?; // -> [out_features, in_features]
    Ok(t.to_dtype(scales.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip check: pack 8 known nibble values using AWQ's real unpack
    /// order (`AWQ_REVERSE_ORDER`, verified against AutoAWQ's actual source -
    /// see module doc comment), dequantize with zero=0/scale=1, and confirm
    /// the unpacked values land back in their original logical positions.
    /// This is a self-consistency check on the arithmetic, not a substitute
    /// for the real end-to-end verification (numeric dequant check against
    /// AutoAWQ's source + real CUDA generation test) documented above.
    #[test]
    fn awq_nibble_order_round_trips() {
        let device = Device::Cpu;
        let logical_values: [u32; 8] = [10, 14, 11, 15, 12, 0, 13, 1];
        let mut packed: u32 = 0;
        for (i, &v) in logical_values.iter().enumerate() {
            let shift = AWQ_REVERSE_ORDER[i] * 4;
            packed |= v << shift;
        }
        let qweight = Tensor::from_vec(vec![packed], (1, 1), &device).unwrap();
        let qzeros = Tensor::from_vec(vec![0u32], (1, 1), &device).unwrap();
        let scales = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &device).unwrap();

        let dequant = dequantize_awq_linear(&qweight, &qzeros, &scales, 1, &device).unwrap();
        // dequantize_awq_linear returns [out_features, in_features] = [8, 1]
        let got: Vec<f32> = dequant.squeeze(1).unwrap().to_vec1().unwrap();
        let expected: Vec<f32> = logical_values.iter().map(|&v| v as f32).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn awq_component_recognizes_all_three_suffixes() {
        assert_eq!(awq_component("model.layers.0.mlp.down_proj.qweight"), Some(("model.layers.0.mlp.down_proj", "qweight")));
        assert_eq!(awq_component("model.layers.0.mlp.down_proj.qzeros"), Some(("model.layers.0.mlp.down_proj", "qzeros")));
        assert_eq!(awq_component("model.layers.0.mlp.down_proj.scales"), Some(("model.layers.0.mlp.down_proj", "scales")));
        assert_eq!(awq_component("model.layers.0.input_layernorm.weight"), None);
    }
}
