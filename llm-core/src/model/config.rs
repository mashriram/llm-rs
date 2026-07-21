use std::path::Path;
use std::fs::File;
use std::io::Read;
use anyhow::{Result, anyhow};
use crate::types::{ModelMeta, WeightDtype, HiddenAct};

pub fn parse_config(path: &Path) -> Result<ModelMeta> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let val: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

    // AWQ and GPTQ are dequantized at load time in CandleBackend::load_weights
    // (see `llm-core/src/loader/awq.rs`/`gptq.rs`) - detected here only to
    // reject formats with genuinely no dequant path. bitsandbytes (NF4/FP4)
    // packs weights in a layout this codebase does not implement at all;
    // rather than silently loading its packed bytes as if they were plain
    // F16/BF16 weights (which would load "successfully" and produce
    // meaningless output with no indication anything is wrong), fail with a
    // clear error. This is a defense-in-depth check for models loaded
    // directly from disk (not via `llm pull`, which already warns about this
    // before download).
    if let Some(qc) = val.get("quantization_config") {
        let method = qc.get("quant_method").and_then(|v| v.as_str()).unwrap_or("unknown");
        if matches!(method.to_lowercase().as_str(), "bitsandbytes" | "bnb" | "bnb_4bit") {
            return Err(anyhow!(
                "this model is pre-quantized with '{method}', which llm-rs does not yet support \
                 (no dequantization kernel for this packed-weight format) - use a GGUF-quantized \
                 version of this model instead, or its unquantized/F16 safetensors version",
            ));
        }
    }

    // Real multimodal HF configs (LLaVA-style, Qwen2-VL, Gemma3/Gemma4, ...)
    // commonly nest the actual text-decoder dimensions under a `text_config`
    // object, with the top level carrying only multimodal glue fields
    // (image/audio token ids, `vision_config`/`audio_config`, etc.) and NO
    // top-level `vocab_size`/`hidden_size`/... at all. Confirmed against a
    // real checkpoint, not assumed: `mlx-community/gemma-4-e2b-it-4bit`'s
    // own `config.json` has no top-level `vocab_size` - only
    // `text_config.vocab_size`. Without this fallback, `parse_config` would
    // reject every such model with "vocab_size missing from config" even
    // though the real value is right there, one level down - a direct
    // violation of "any model from HF should run." `lookup` checks the top
    // level first (so a model that DOES set these at the top level, or
    // deliberately overrides a text_config value, still wins), falling back
    // to `text_config` only when the top-level key is absent.
    let text_config = val.get("text_config");
    let lookup = |key: &str| -> Option<&serde_json::Value> {
        val.get(key).or_else(|| text_config.and_then(|tc| tc.get(key)))
    };

    let vocab_size = lookup("vocab_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("vocab_size missing from config"))? as usize;
    let hidden_size = lookup("hidden_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("hidden_size missing from config"))? as usize;
    let num_hidden_layers = lookup("num_hidden_layers").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("num_hidden_layers missing from config"))? as usize;
    let num_attention_heads = lookup("num_attention_heads").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("num_attention_heads missing from config"))? as usize;
    if num_attention_heads == 0 {
        return Err(anyhow!("num_attention_heads must be nonzero (malformed config.json)"));
    }

    let num_key_value_heads = lookup("num_key_value_heads").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(num_attention_heads);
    let head_dim = lookup("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(hidden_size / num_attention_heads);

    let intermediate_size = lookup("intermediate_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("intermediate_size missing from config"))? as usize;
    let max_position_embeddings = lookup("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(2048) as usize;
    let mut rope_theta = lookup("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;

    // Gemma3/Gemma4-style hybrid local/global attention: real HF configs
    // (confirmed against `mlx-community/gemma-4-e2b-it-4bit`'s own
    // `text_config`) describe this via `layer_types` (one
    // "sliding_attention"/"full_attention" string per layer),
    // `sliding_window`, `num_kv_shared_layers`, and a nested
    // `rope_parameters.{full_attention,sliding_attention}.rope_theta` dict
    // (NOT the flat `rope_theta` field read above, which this family of
    // configs may omit or use only as an unrelated default). The GGUF
    // loading path already reads GGUF's equivalent metadata keys into these
    // exact `ModelMeta` fields (`shared_kv_layers`/`sliding_window_pattern`/
    // `sliding_window`/`key_length`/`key_length_swa`/`rope_theta_swa` -
    // see `backends/candle.rs`), and the graph/forward-pass consumption of
    // these fields is already fully generic - only THIS safetensors config
    // parser was missing the equivalent read, silently leaving every layer
    // as plain full-attention/non-shared-KV for any hybrid-attention model
    // loaded from safetensors (correct-looking but numerically wrong
    // output, with no error). Ported from the GGUF path's field semantics,
    // NOT independently verified against a real forward pass here - same
    // "written, not numerically verified" posture as this codebase's
    // AWQ/GPTQ CUDA dequant path (see quant-performance-plan.md phase 4).
    let sliding_window_pattern: Option<Vec<bool>> = lookup("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().map(|v| v.as_str() == Some("sliding_attention")).collect());
    let sliding_window = lookup("sliding_window").and_then(|v| v.as_u64()).map(|v| v as usize);
    let shared_kv_layers = lookup("num_kv_shared_layers").and_then(|v| v.as_u64()).map(|v| v as usize);
    let rope_parameters = lookup("rope_parameters");
    let rope_theta_full = rope_parameters
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|t| t.get("rope_theta"))
        .and_then(|v| v.as_f64());
    let rope_theta_swa = rope_parameters
        .and_then(|rp| rp.get("sliding_attention"))
        .and_then(|t| t.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);
    // `rope_parameters.full_attention.rope_theta`, when present, is this
    // family's real primary/non-sliding rope_theta - takes priority over
    // the flat `rope_theta` field read above (which this family's configs
    // may set to a default that's actually only correct for sliding
    // layers, per `ModelMeta::get_rope_theta`'s non-swa branch using the
    // top-level `rope_theta`).
    if let Some(v) = rope_theta_full {
        rope_theta = v as f32;
    }
    // `global_head_dim`, when present, is the K/V head dim for full-
    // attention layers specifically (Gemma-4's `head_dim`/`global_head_dim`
    // split - confirmed present side-by-side in the real config); the base
    // `head_dim` computed above serves sliding-window layers via
    // `ModelMeta::get_head_dim`'s swa branch.
    let key_length_swa = Some(head_dim);
    let key_length = lookup("global_head_dim").and_then(|v| v.as_u64()).map(|v| v as usize);

    let torch_dtype = lookup("torch_dtype").or_else(|| lookup("dtype")).and_then(|v| v.as_str()).unwrap_or("float16");
    let weight_dtype = match torch_dtype {
        "float32" => WeightDtype::F32,
        "float16" => WeightDtype::F16,
        "bfloat16" => WeightDtype::BF16,
        _ => WeightDtype::F16,
    };

    let rms_norm_eps = lookup("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
    let tie_word_embeddings = lookup("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    // Matches candle.rs's GGUF-metadata activation detection: substring match
    // so variants like "gelu_new"/"gelu_pytorch_tanh" still resolve to GeLU,
    // and "silu"/"swish" (the same function under two names, both common in
    // real HF configs) resolve to SiLU explicitly.
    //
    // Real Gemma2/Gemma3/Gemma4 HF configs use a DIFFERENT field name,
    // `hidden_activation`, instead of the more common `hidden_act` -
    // confirmed against `mlx-community/gemma-4-e2b-it-4bit`'s own
    // `text_config.hidden_activation = "gelu_pytorch_tanh"` (it has no
    // `hidden_act` field at all). Missing this alias would make every real
    // Gemma-family HF checkpoint hit the "missing field" bail below - a
    // regression from even the old silent-SiLU-default behavior for the
    // one family that behavior was ALSO already wrong for (Gemma uses GeLU,
    // not SiLU). Check both names; anything else - a genuinely missing
    // field under either name, or a real but unrecognized value like
    // "relu"/"geglu"/"swiglu" - is a clear error instead of a silent guess:
    // SiLU-gated and e.g. ReLU MLPs compute different math, not just a
    // different nonlinearity curve, so guessing wrong here produces
    // numerically wrong output with no warning - exactly the class of bug
    // this codebase's "no silent fallback" rule (see the unknown-GGML-
    // quant-type and missing-architecture-metadata bails elsewhere in this
    // file/candle.rs) exists to prevent.
    let hidden_act_raw = lookup("hidden_act").or_else(|| lookup("hidden_activation"));
    let hidden_act = match hidden_act_raw.and_then(|v| v.as_str()) {
        Some(s) if s.contains("gelu") => HiddenAct::GeLU,
        Some(s) if s.contains("silu") || s.contains("swish") => HiddenAct::SiLU,
        Some(other) => return Err(anyhow!(
            "config.json's hidden_act = \"{other}\" is not a recognized activation \
             function (this codebase currently supports SiLU and GeLU variants only) - \
             defaulting silently here would run the wrong MLP math with no warning"
        )),
        None => return Err(anyhow!(
            "config.json is missing the hidden_act field - cannot determine this model's \
             MLP activation function without guessing, which risks running the wrong MLP \
             math silently"
        )),
    };

    let no_rope_layers = if let Some(arr) = val.get("no_rope_layers").and_then(|v| v.as_array()) {
        arr.iter().map(|v| {
            if let Some(b) = v.as_bool() {
                b
            } else if let Some(i) = v.as_i64() {
                i != 0
            } else if let Some(u) = v.as_u64() {
                u != 0
            } else {
                false
            }
        }).collect()
    } else {
        vec![false; num_hidden_layers]
    };

    let mut has_vision_encoder = false;
    let mut vision_hidden_dim = None;
    let mut vision_patch_size = None;
    let mut vision_image_size = None;
    let mut vision_num_layers = None;
    let mut vision_num_heads = None;
    let mut vision_projection_dim = None;
    let mut spatial_merge_size = None;
    let mut is_deepstack_layers = None;
    let mut projector_type = None;

    let mut has_audio_encoder = false;
    let mut audio_hidden_dim = None;
    let mut audio_block_count = None;
    let mut audio_embedding_length = None;
    let mut audio_num_mel_bins = None;

    // Vision and audio modalities are detected (and their config blocks resolved)
    // SEPARATELY: audio_config/audio_*-prefixed keys are evidence of an AUDIO
    // encoder, not a vision one — treating them as vision (as a previous version
    // of this function did) mislabeled audio-only multimodal models.
    let vision_keys = ["vision_config", "vision_config_dict", "image_config", "multimodal_config"];
    let mut vision_config_block = None;
    for m in vision_keys {
        if let Some(block) = val.get(m) {
            vision_config_block = Some(block);
            has_vision_encoder = true;
            break;
        }
    }

    let audio_keys = ["audio_config", "audio_config_dict"];
    let mut audio_config_block = None;
    for m in audio_keys {
        if let Some(block) = val.get(m) {
            audio_config_block = Some(block);
            has_audio_encoder = true;
            break;
        }
    }

    if !has_vision_encoder {
        if let Some(obj) = val.as_object() {
            for key in obj.keys() {
                if key.starts_with("vision_") || key.starts_with("image_") || key.starts_with("video_") || key.starts_with("multimodal_") {
                    has_vision_encoder = true;
                    break;
                }
            }
        }
    }
    if !has_audio_encoder {
        if let Some(obj) = val.as_object() {
            for key in obj.keys() {
                if key.starts_with("audio_") {
                    has_audio_encoder = true;
                    break;
                }
            }
        }
    }

    if let Some(vc) = vision_config_block {
        vision_hidden_dim = vc.get("hidden_size").or_else(|| vc.get("hidden_dim")).and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_patch_size = vc.get("patch_size").and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_image_size = vc.get("image_size").and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_num_layers = vc.get("num_hidden_layers").or_else(|| vc.get("num_layers")).and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_num_heads = vc.get("num_attention_heads").or_else(|| vc.get("num_heads")).and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_projection_dim = vc.get("projection_dim").or_else(|| vc.get("projection_size")).and_then(|v| v.as_u64()).map(|v| v as usize);
        spatial_merge_size = vc.get("spatial_merge_size").and_then(|v| v.as_u64()).map(|v| v as usize);

        is_deepstack_layers = vc.get("is_deepstack_layers").and_then(|v| v.as_array()).map(|arr| {
            arr.iter().map(|item| item.as_bool().unwrap_or(false)).collect::<Vec<bool>>()
        });
        projector_type = vc.get("projector_type").and_then(|v| v.as_str()).map(|s| s.to_string());
    } else if has_vision_encoder {
        vision_hidden_dim = val.get("vision_hidden_size").or_else(|| val.get("vision_hidden_dim")).and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_patch_size = val.get("vision_patch_size").and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_image_size = val.get("vision_image_size").and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_num_layers = val.get("vision_num_hidden_layers").or_else(|| val.get("vision_num_layers")).and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_num_heads = val.get("vision_num_attention_heads").or_else(|| val.get("vision_num_heads")).and_then(|v| v.as_u64()).map(|v| v as usize);
        vision_projection_dim = val.get("vision_projection_dim").and_then(|v| v.as_u64()).map(|v| v as usize);
        spatial_merge_size = val.get("spatial_merge_size").and_then(|v| v.as_u64()).map(|v| v as usize);
        projector_type = val.get("projector_type").and_then(|v| v.as_str()).map(|s| s.to_string());
    }

    if let Some(ac) = audio_config_block {
        audio_hidden_dim = ac.get("hidden_size").or_else(|| ac.get("hidden_dim")).or_else(|| ac.get("d_model")).and_then(|v| v.as_u64()).map(|v| v as usize);
        audio_block_count = ac.get("num_hidden_layers").or_else(|| ac.get("num_layers")).or_else(|| ac.get("encoder_layers")).and_then(|v| v.as_u64()).map(|v| v as usize);
        audio_embedding_length = ac.get("embedding_length").or_else(|| ac.get("hidden_size")).and_then(|v| v.as_u64()).map(|v| v as usize);
        audio_num_mel_bins = ac.get("num_mel_bins").or_else(|| ac.get("n_mels")).and_then(|v| v.as_u64()).map(|v| v as usize);
    } else if has_audio_encoder {
        audio_hidden_dim = val.get("audio_hidden_size").or_else(|| val.get("audio_hidden_dim")).and_then(|v| v.as_u64()).map(|v| v as usize);
        audio_block_count = val.get("audio_num_hidden_layers").or_else(|| val.get("audio_num_layers")).and_then(|v| v.as_u64()).map(|v| v as usize);
        audio_embedding_length = val.get("audio_embedding_length").and_then(|v| v.as_u64()).map(|v| v as usize);
        audio_num_mel_bins = val.get("audio_num_mel_bins").and_then(|v| v.as_u64()).map(|v| v as usize);
    }

    let final_logit_softcapping = lookup("final_logit_softcapping").and_then(|v| v.as_f64()).map(|v| v as f32);

    let model_type = val.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let architectures: Vec<String> = val.get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|a| a.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    // Single source of truth shared with backends/candle.rs's GGUF loading path —
    // see `is_gemma_arch`'s doc comment for why this must not be duplicated independently.
    let is_gemma = crate::types::is_gemma_arch(model_type, &architectures);

    let ple_dim = lookup("hidden_size_per_layer_input").and_then(|v| v.as_u64()).map(|v| v as usize);
    let embed_scale = if is_gemma {
        Some((hidden_size as f32).sqrt())
    } else {
        None
    };

    Ok(ModelMeta {
        vocab_size,
        hidden_dim: hidden_size,
        n_layers: num_hidden_layers,
        n_heads: num_attention_heads,
        n_kv_heads: num_key_value_heads,
        head_dim,
        intermediate_dim: intermediate_size,
        max_seq_len: max_position_embeddings,
        rope_theta,
        weight_dtype,
        rms_norm_eps,
        tie_word_embeddings,
        hidden_act,
        no_rope_layers,
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
        arch: model_type.to_string(),
        chat_template: None,  // Will be populated from tokenizer_config.json in candle.rs
        eos_token_str: if is_gemma { Some("<end_of_turn>".to_string()) } else { None },
    })
}

#[cfg(test)]
mod hidden_act_tests {
    use super::parse_config;
    use crate::types::HiddenAct;
    use std::io::Write;

    fn write_config(dir: &std::path::Path, extra_hidden_act_line: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!("llm_rs_hidden_act_test_{}_{}.json", std::process::id(), id));
        let json = format!(
            r#"{{
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008
                {extra_hidden_act_line}
            }}"#
        );
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        path
    }

    /// A missing `hidden_act` must be a clear load-time error, not a silent
    /// SiLU guess - see the doc comment on this field in `parse_config`.
    #[test]
    fn missing_hidden_act_is_an_error_not_a_silent_default() {
        let path = write_config(&std::env::temp_dir(), "");
        let err = parse_config(&path).unwrap_err();
        assert!(err.to_string().contains("hidden_act"));
    }

    /// A real but unrecognized `hidden_act` value (this codebase only
    /// supports SiLU/GeLU today) must also be a clear error, not silently
    /// treated as SiLU.
    #[test]
    fn unrecognized_hidden_act_is_an_error_not_a_silent_default() {
        let path = write_config(&std::env::temp_dir(), r#", "hidden_act": "relu""#);
        let err = parse_config(&path).unwrap_err();
        assert!(err.to_string().contains("relu"));
    }

    #[test]
    fn silu_and_swish_both_resolve_to_silu() {
        let path = write_config(&std::env::temp_dir(), r#", "hidden_act": "silu""#);
        assert!(matches!(parse_config(&path).unwrap().hidden_act, HiddenAct::SiLU));
        let path = write_config(&std::env::temp_dir(), r#", "hidden_act": "swish""#);
        assert!(matches!(parse_config(&path).unwrap().hidden_act, HiddenAct::SiLU));
    }

    #[test]
    fn gelu_variants_resolve_to_gelu() {
        let path = write_config(&std::env::temp_dir(), r#", "hidden_act": "gelu_pytorch_tanh""#);
        assert!(matches!(parse_config(&path).unwrap().hidden_act, HiddenAct::GeLU));
    }
}
