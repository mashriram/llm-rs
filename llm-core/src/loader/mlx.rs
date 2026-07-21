//! MLX (`mlx.core.quantize`, Apple's MLX framework) affine-quantized weight
//! dequantization - lets llm-rs load a real `mlx-community/*` HF repo's
//! safetensors checkpoint directly, dequantizing at load time and routing
//! through this codebase's existing dense-weight/`QMatMul` execution path
//! (candle's own kernels execute it - no separate MLX runtime dependency),
//! per `quant-performance-plan.md` Phase 3's "one engine, one `LlmBackend`
//! trait" constraint.
//!
//! Format confirmed by direct inspection of a real repo AND byte-level
//! dequantization ground truth - not assumed from documentation alone.
//! `mlx-community/gemma-4-e2b-it-4bit`'s `config.json` carries:
//! ```json
//! "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
//! ```
//! and its `model.safetensors` header shows, for a quantized linear layer
//! (e.g. `language_model.model.layers.0.self_attn.q_proj`):
//!   - `{base}.weight`:  U32, shape `[out_features, in_features / (32/bits)]`
//!     - `32/bits` packed `bits`-wide unsigned values per u32 word.
//!   - `{base}.scales`:  same dtype as the rest of the model (BF16 in the
//!     reference checkpoint), shape `[out_features, in_features/group_size]`.
//!   - `{base}.biases`:  identical shape/dtype to `{base}.scales`.
//!
//! The packing order and dequant formula were verified numerically, not
//! assumed: loaded `language_model.model.layers.0.self_attn.q_proj`'s real
//! `weight`/`scales`/`biases` tensors via `mlx.core.load`, ran the real
//! `mlx.core.dequantize(weight, scales, biases, group_size=64, bits=4)`,
//! and reverse-derived the packed integer values from the dequantized
//! output (`value = round((dequantized - bias) / scale)`) for the first 16
//! input positions of output row 0. The recovered values
//! (`[12,2,9,7,9,14,5,9, 12,10,12,14,8,6,13,9]`) match EXACTLY extracting
//! 4-bit nibbles from the two raw packed words
//! (`2515106092`, `2640899244`) via `value_i = (word >> (4*i)) & 0xF` for
//! `i` in `0..8`, i.e. **low nibble = the first (lowest-input-index)
//! element of the group of 8**, consecutive words covering consecutive
//! input positions. Dequant formula confirmed the same way:
//! `w = value * scale + bias` (e.g. `value=12`, `scale=-0.00299072265625`,
//! `bias=0.0269775390625` → `-0.0089111328125`, matching the real
//! `mlx.core.dequantize` output exactly). This matches MLX's documented
//! "affine" quantization scheme, now confirmed against real bytes rather
//! than taken on faith. The regression test below (`matches_real_mlx_checkpoint_bytes`)
//! encodes this exact real data point.
//!
//! Already in `[out_features, in_features]` row convention - matching this
//! codebase's standard `nn.Linear` weight layout (no transpose needed,
//! unlike AWQ, which packs along the opposite axis - see `awq.rs`'s doc
//! comment for that contrast).
//!
//! Not every tensor in an MLX-quantized checkpoint is actually quantized:
//! smaller submodules (in the reference checkpoint, the audio/vision towers'
//! linear layers) are shipped as plain dense `{base}.linear.weight` BF16
//! tensors with no `.scales`/`.biases` siblings at all. `.weight` alone is
//! therefore an AMBIGUOUS suffix - callers must confirm a `.scales`+
//! `.biases` sibling pair exists (and that `.weight`'s dtype is U32) before
//! treating a tensor group as MLX-quantized; see `mlx_component`'s doc
//! comment.

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};

/// This codebase's only implemented MLX quantization mode/bit-widths.
/// `mode` values other than `"affine"`, or `bits` values other than 4/8,
/// are refused with a clear error rather than silently misinterpreted -
/// see `detect_mlx_quantization`.
pub struct MlxQuantConfig {
    pub group_size: usize,
    pub bits: usize,
}

/// Parse the `"quantization"` block MLX-published `config.json` files carry
/// (confirmed real key name via `mlx-community/gemma-4-e2b-it-4bit`'s own
/// `config.json` - NOT `quantization_config`, a separate/legacy field some
/// MLX repos also include for HF-ecosystem-tool compatibility but which
/// `mlx_lm` itself does not read for this purpose). Returns `Ok(None)` if
/// the model isn't MLX-quantized at all (the common case for a plain HF
/// safetensors repo), so callers can fall through to the existing
/// unquantized-safetensors path unchanged.
pub fn detect_mlx_quantization(config_json: &serde_json::Value) -> Result<Option<MlxQuantConfig>> {
    let Some(q) = config_json.get("quantization") else { return Ok(None) };
    let mode = q.get("mode").and_then(|v| v.as_str()).unwrap_or("affine");
    if mode != "affine" {
        return Err(anyhow!(
            "this model's MLX quantization mode is '{mode}', but llm-rs only implements \
             MLX's 'affine' quantization scheme - refusing to load it as if it were affine, \
             which would silently produce wrong weights"
        ));
    }
    let group_size = q.get("group_size").and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("MLX quantization config is missing group_size"))? as usize;
    let bits = q.get("bits").and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("MLX quantization config is missing bits"))? as usize;
    if bits != 4 && bits != 8 {
        return Err(anyhow!(
            "MLX quantization bits={bits} is not supported (only 4-bit and 8-bit are \
             implemented) - refusing to load it as if it were, which would silently \
             produce wrong weights"
        ));
    }
    if group_size == 0 {
        return Err(anyhow!("MLX quantization group_size must be nonzero"));
    }
    Ok(Some(MlxQuantConfig { group_size, bits }))
}

/// Returns `Some((base_name, kind))` if `name` is the `.weight`/`.scales`/
/// `.biases` suffix of a POTENTIALLY MLX-quantized linear layer. `.weight`
/// alone is ambiguous - ordinary dense linear/norm/embedding tensors use
/// the same suffix - so callers MUST additionally confirm a `.scales` AND
/// `.biases` sibling both exist (and that `.weight`'s own dtype is U32,
/// candle's own unsigned-32-bit type) before treating a tensor group as
/// MLX-quantized; anything without that full sibling set should load
/// through the existing plain-safetensors path unchanged.
pub fn mlx_component(name: &str) -> Option<(&str, &'static str)> {
    for (suffix, kind) in [(".weight", "weight"), (".scales", "scales"), (".biases", "biases")] {
        if let Some(base) = name.strip_suffix(suffix) {
            return Some((base, kind));
        }
    }
    None
}

/// Dequantizes one MLX affine-quantized linear layer's packed weight into a
/// dense `[out_features, in_features]` tensor, in `scales`' own dtype, on
/// `device` - see this module's doc comment for the verified packing
/// format and dequant formula.
pub fn dequantize_mlx_linear(
    weight: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    bits: usize,
    device: &Device,
) -> Result<Tensor> {
    let per_word = 32 / bits;
    let mask: u32 = (1u32 << bits) - 1;

    let (out_features, packed_in) = weight.dims2()?;
    let in_features = packed_in * per_word;

    let (scales_out, n_groups) = scales.dims2()?;
    if scales_out != out_features {
        return Err(anyhow!(
            "MLX weight/scales out-axis mismatch: weight implies {out_features} output rows, \
             scales has {scales_out}"
        ));
    }
    let expected_groups = in_features.div_ceil(group_size);
    if n_groups != expected_groups {
        return Err(anyhow!(
            "MLX scales column count {n_groups} does not match in_features {in_features} \
             / group_size {group_size} (expected {expected_groups})"
        ));
    }
    let (biases_out, biases_groups) = biases.dims2()?;
    if biases_out != out_features || biases_groups != n_groups {
        return Err(anyhow!(
            "MLX biases shape [{biases_out}, {biases_groups}] does not match scales shape \
             [{out_features}, {n_groups}]"
        ));
    }

    let weight_u32 = weight.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let scales_f32 = scales.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let biases_f32 = biases.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let mut dequant = vec![0f32; out_features * in_features];
    for o in 0..out_features {
        for packed_col in 0..packed_in {
            let word = weight_u32[o][packed_col];
            for i in 0..per_word {
                let p = packed_col * per_word + i;
                let group = p / group_size;
                let value = (word >> (bits * i)) & mask;
                dequant[o * in_features + p] = value as f32 * scales_f32[o][group] + biases_f32[o][group];
            }
        }
    }

    let t = Tensor::from_vec(dequant, (out_features, in_features), device)?;
    Ok(t.to_dtype(scales.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_component_recognizes_all_three_suffixes() {
        assert_eq!(
            mlx_component("model.layers.0.self_attn.q_proj.weight"),
            Some(("model.layers.0.self_attn.q_proj", "weight"))
        );
        assert_eq!(
            mlx_component("model.layers.0.self_attn.q_proj.scales"),
            Some(("model.layers.0.self_attn.q_proj", "scales"))
        );
        assert_eq!(
            mlx_component("model.layers.0.self_attn.q_proj.biases"),
            Some(("model.layers.0.self_attn.q_proj", "biases"))
        );
        // `.weight` is intentionally ambiguous - this function alone can't
        // tell a quantized linear's packed weight from an ordinary dense
        // tensor (embeddings, norms) that happens to share the suffix.
        // Disambiguation (checking for `.scales`/`.biases` siblings AND a
        // U32 dtype) is the CALLER's job - see this module's doc comment.
        assert_eq!(
            mlx_component("model.layers.0.input_layernorm.weight"),
            Some(("model.layers.0.input_layernorm", "weight"))
        );
        assert_eq!(mlx_component("model.layers.0.input_layernorm.bias"), None);
    }

    #[test]
    fn detect_mlx_quantization_parses_real_config_shape() {
        let cfg: serde_json::Value = serde_json::json!({
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        });
        let q = detect_mlx_quantization(&cfg).unwrap().unwrap();
        assert_eq!(q.group_size, 64);
        assert_eq!(q.bits, 4);
    }

    #[test]
    fn detect_mlx_quantization_none_for_plain_model() {
        let cfg: serde_json::Value = serde_json::json!({"hidden_size": 4096});
        assert!(detect_mlx_quantization(&cfg).unwrap().is_none());
    }

    #[test]
    fn detect_mlx_quantization_rejects_unsupported_mode() {
        let cfg: serde_json::Value = serde_json::json!({
            "quantization": {"group_size": 64, "bits": 4, "mode": "logarithmic"}
        });
        assert!(detect_mlx_quantization(&cfg).is_err());
    }

    #[test]
    fn detect_mlx_quantization_rejects_unsupported_bits() {
        let cfg: serde_json::Value = serde_json::json!({
            "quantization": {"group_size": 64, "bits": 3, "mode": "affine"}
        });
        assert!(detect_mlx_quantization(&cfg).is_err());
    }

    /// Round-trip check: pack 8 known 4-bit values in the verified low-
    /// nibble-first order, dequantize with a trivial scale/bias, and confirm
    /// they land back in their original logical positions.
    #[test]
    fn mlx_4bit_nibble_order_round_trips() {
        let device = Device::Cpu;
        let logical_values: [u32; 8] = [12, 2, 9, 7, 9, 14, 5, 9];
        let mut packed: u32 = 0;
        for (i, &v) in logical_values.iter().enumerate() {
            packed |= v << (4 * i);
        }
        let weight = Tensor::from_vec(vec![packed], (1, 1), &device).unwrap();
        let scales = Tensor::from_vec(vec![1.0f32], (1, 1), &device).unwrap();
        let biases = Tensor::from_vec(vec![0.0f32], (1, 1), &device).unwrap();

        let dequant = dequantize_mlx_linear(&weight, &scales, &biases, 8, 4, &device).unwrap();
        assert_eq!(dequant.dims(), &[1, 8]);
        let got: Vec<f32> = dequant.squeeze(0).unwrap().to_vec1().unwrap();
        let expected: Vec<f32> = logical_values.iter().map(|&v| v as f32).collect();
        assert_eq!(got, expected);
    }

    /// Round-trip check for 8-bit packing (4 values per u32 word).
    #[test]
    fn mlx_8bit_nibble_order_round_trips() {
        let device = Device::Cpu;
        let logical_values: [u32; 4] = [255, 128, 0, 64];
        let mut packed: u32 = 0;
        for (i, &v) in logical_values.iter().enumerate() {
            packed |= v << (8 * i);
        }
        let weight = Tensor::from_vec(vec![packed], (1, 1), &device).unwrap();
        let scales = Tensor::from_vec(vec![2.0f32], (1, 1), &device).unwrap();
        let biases = Tensor::from_vec(vec![1.0f32], (1, 1), &device).unwrap();

        let dequant = dequantize_mlx_linear(&weight, &scales, &biases, 4, 8, &device).unwrap();
        let got: Vec<f32> = dequant.squeeze(0).unwrap().to_vec1().unwrap();
        let expected: Vec<f32> = logical_values.iter().map(|&v| v as f32 * 2.0 + 1.0).collect();
        assert_eq!(got, expected);
    }

    /// Regression fixture encoding the REAL bytes read from
    /// `mlx-community/gemma-4-e2b-it-4bit`'s `model.safetensors`
    /// (`language_model.model.layers.0.self_attn.q_proj`, output row 0,
    /// input positions 0..16 - two packed u32 words, one scale/bias group)
    /// and the REAL dequantized values `mlx.core.dequantize` produced for
    /// them on this exact data - not synthetic/self-consistent-only data.
    /// See this module's doc comment for how these were obtained.
    #[test]
    fn matches_real_mlx_checkpoint_bytes() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(
            vec![2515106092u32, 2640899244u32],
            (1, 2),
            &device,
        ).unwrap();
        let scales = Tensor::from_vec(vec![-0.00299072265625f32], (1, 1), &device).unwrap();
        let biases = Tensor::from_vec(vec![0.0269775390625f32], (1, 1), &device).unwrap();

        let dequant = dequantize_mlx_linear(&weight, &scales, &biases, 64, 4, &device).unwrap();
        let got: Vec<f32> = dequant.squeeze(0).unwrap().to_vec1().unwrap();
        let expected = vec![
            -0.0089111328125, 0.02099609375, 6.103515625e-05, 0.00604248046875,
            6.103515625e-05, -0.014892578125, 0.01202392578125, 6.103515625e-05,
            -0.0089111328125, -0.0029296875, -0.0089111328125, -0.014892578125,
            0.0030517578125, 0.009033203125, -0.01190185546875, 6.103515625e-05,
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }
}
