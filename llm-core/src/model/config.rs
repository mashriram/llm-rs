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

    let vocab_size = val.get("vocab_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("vocab_size missing from config"))? as usize;
    let hidden_size = val.get("hidden_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("hidden_size missing from config"))? as usize;
    let num_hidden_layers = val.get("num_hidden_layers").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("num_hidden_layers missing from config"))? as usize;
    let num_attention_heads = val.get("num_attention_heads").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("num_attention_heads missing from config"))? as usize;
    
    let num_key_value_heads = val.get("num_key_value_heads").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(num_attention_heads);
    let head_dim = val.get("head_dim").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(hidden_size / num_attention_heads);
    
    let intermediate_size = val.get("intermediate_size").and_then(|v| v.as_u64()).ok_or_else(|| anyhow!("intermediate_size missing from config"))? as usize;
    let max_position_embeddings = val.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(2048) as usize;
    let rope_theta = val.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0) as f32;
    
    let torch_dtype = val.get("torch_dtype").and_then(|v| v.as_str()).unwrap_or("float16");
    let weight_dtype = match torch_dtype {
        "float32" => WeightDtype::F32,
        "float16" => WeightDtype::F16,
        "bfloat16" => WeightDtype::BF16,
        _ => WeightDtype::F16,
    };

    let rms_norm_eps = val.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
    let tie_word_embeddings = val.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false);
    let hidden_act = match val.get("hidden_act").and_then(|v| v.as_str()) {
        Some("gelu") => HiddenAct::GeLU,
        _ => HiddenAct::SiLU,
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

    let modalities = ["vision_config", "vision_config_dict", "audio_config", "audio_config_dict", "image_config", "multimodal_config"];
    let mut config_block = None;
    for m in modalities {
        if let Some(block) = val.get(m) {
            config_block = Some(block);
            has_vision_encoder = true;
            break;
        }
    }

    if !has_vision_encoder {
        if let Some(obj) = val.as_object() {
            for key in obj.keys() {
                if key.starts_with("vision_") || key.starts_with("image_") || key.starts_with("audio_") || key.starts_with("video_") || key.starts_with("multimodal_") {
                    has_vision_encoder = true;
                    break;
                }
            }
        }
    }

    if let Some(vc) = config_block {
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

    let final_logit_softcapping = val.get("final_logit_softcapping").and_then(|v| v.as_f64()).map(|v| v as f32);

    let model_type = val.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    let is_gemma = model_type == "gemma" || model_type == "gemma2" || model_type == "gemma4" ||
        val.get("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().any(|arch| {
                let arch_str = arch.as_str().unwrap_or("");
                arch_str.contains("Gemma")
            })).unwrap_or(false);

    let ple_dim = val.get("hidden_size_per_layer_input").and_then(|v| v.as_u64()).map(|v| v as usize);
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
        shared_kv_layers: None,
        sliding_window_pattern: None,
        sliding_window: None,
        key_length: None,
        key_length_swa: None,
        rope_theta_swa: None,
        final_logit_softcapping,
        is_gemma,
        ple_dim,
        embed_scale,
        arch: model_type.to_string(),
        chat_template: None,  // Will be populated from tokenizer_config.json in candle.rs
        eos_token_str: if is_gemma { Some("<end_of_turn>".to_string()) } else { None },
    })
}
