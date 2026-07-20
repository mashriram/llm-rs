use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone)]
pub struct LayerTensors {
    pub index: usize,
    pub input_layernorm: Option<String>,
    pub q_proj: Option<String>,
    pub q_bias: Option<String>,
    pub k_proj: Option<String>,
    pub k_bias: Option<String>,
    pub v_proj: Option<String>,
    pub v_bias: Option<String>,
    pub q_norm: Option<String>,
    pub k_norm: Option<String>,
    pub o_proj: Option<String>,
    pub o_bias: Option<String>,
    pub post_attention_layernorm: Option<String>,
    pub post_attention_norm: Option<String>,
    pub post_ffw_norm: Option<String>,
    pub gate_proj: Option<String>,
    pub gate_bias: Option<String>,
    pub up_proj: Option<String>,
    pub up_bias: Option<String>,
    pub down_proj: Option<String>,
    pub down_bias: Option<String>,
    pub per_layer_input_gate: Option<String>,
    pub per_layer_projection: Option<String>,
    pub post_per_layer_input_norm: Option<String>,
    pub layer_output_scale: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TensorGroupMap {
    pub embed_tokens: Option<String>,
    pub final_norm: Option<String>,
    pub lm_head: Option<String>,
    pub per_layer_token_embd: Option<String>,
    pub per_layer_model_proj: Option<String>,
    pub per_layer_proj_norm: Option<String>,
    pub layers: Vec<LayerTensors>,
}

pub fn map_gguf_name(name: &str) -> String {
    if name == "token_embd.weight" {
        return "model.embed_tokens.weight".to_string();
    }
    if name == "output_norm.weight" {
        return "model.norm.weight".to_string();
    }
    if name == "output.weight" {
        return "lm_head.weight".to_string();
    }
    if name == "per_layer_inp.weight" || name == "per_layer_token_embd.weight" || name == "per_layer_embed.weight" {
        return "per_layer_token_embd.weight".to_string();
    }
    if name == "per_layer_proj.weight" || name == "per_layer_model_proj.weight" {
        return "per_layer_model_proj.weight".to_string();
    }
    if name == "per_layer_norm.weight" || name == "per_layer_proj_norm.weight" {
        return "per_layer_proj_norm.weight".to_string();
    }
    
    if name.starts_with("blk.") {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 3 {
            if let Ok(layer_idx) = parts[1].parse::<usize>() {
                let suffix = parts[2..].join(".");
                let mapped_suffix = match suffix.as_str() {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    "attn_norm.weight" => "input_layernorm.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_output.bias" => "self_attn.o_proj.bias",
                    "ffn_gate.bias" => "mlp.gate_proj.bias",
                    "ffn_up.bias" => "mlp.up_proj.bias",
                    "ffn_down.bias" => "mlp.down_proj.bias",
                    "attn_q_norm.weight" => "attn_q_norm.weight",
                    "attn_k_norm.weight" => "attn_k_norm.weight",
                    "inp_gate.weight" => "per_layer_input_gate.weight",
                    "proj.weight" => "per_layer_projection.weight",
                    "post_norm.weight" => "post_per_layer_input_norm.weight",
                    "layer_output_scale.weight" => "layer_output_scale.weight",
                    _ => &suffix,
                };
                return format!("model.layers.{}.{}", layer_idx, mapped_suffix);
            }
        }
    }
    name.to_string()
}

pub fn scan_tensors(names: &[String]) -> TensorGroupMap {
    let mut embed_tokens = None;
    let mut final_norm = None;
    let mut lm_head = None;
    let mut per_layer_token_embd = None;
    let mut per_layer_model_proj = None;
    let mut per_layer_proj_norm = None;
    
    let mut layer_map: HashMap<usize, LayerTensors> = HashMap::new();
    
    for raw_name in names {
        let name = map_gguf_name(raw_name);
        
        if name == "model.embed_tokens.weight" {
            embed_tokens = Some(name.clone());
        } else if name == "model.norm.weight" {
            final_norm = Some(name.clone());
        } else if name == "lm_head.weight" {
            lm_head = Some(name.clone());
        } else if name == "per_layer_token_embd.weight" {
            per_layer_token_embd = Some(name.clone());
        } else if name == "per_layer_model_proj.weight" {
            per_layer_model_proj = Some(name.clone());
        } else if name == "per_layer_proj_norm.weight" {
            per_layer_proj_norm = Some(name.clone());
        } else if name.starts_with("model.layers.") {
            let remain = &name["model.layers.".len()..];
            let parts: Vec<&str> = remain.splitn(2, '.').collect();
            if parts.len() == 2 {
                if let Ok(layer_idx) = parts[0].parse::<usize>() {
                    let suffix = parts[1];
                    let layer = layer_map.entry(layer_idx).or_insert_with(|| LayerTensors {
                        index: layer_idx,
                        input_layernorm: None,
                        q_proj: None,
                        q_bias: None,
                        k_proj: None,
                        k_bias: None,
                        v_proj: None,
                        v_bias: None,
                        q_norm: None,
                        k_norm: None,
                        o_proj: None,
                        o_bias: None,
                        post_attention_layernorm: None,
                        post_attention_norm: None,
                        post_ffw_norm: None,
                        gate_proj: None,
                        gate_bias: None,
                        up_proj: None,
                        up_bias: None,
                        down_proj: None,
                        down_bias: None,
                        per_layer_input_gate: None,
                        per_layer_projection: None,
                        post_per_layer_input_norm: None,
                        layer_output_scale: None,
                    });
                    
                    match suffix {
                        "input_layernorm.weight" => layer.input_layernorm = Some(name.clone()),
                        "self_attn.q_proj.weight" => layer.q_proj = Some(name.clone()),
                        "self_attn.q_proj.bias" => layer.q_bias = Some(name.clone()),
                        "self_attn.k_proj.weight" => layer.k_proj = Some(name.clone()),
                        "self_attn.k_proj.bias" => layer.k_bias = Some(name.clone()),
                        "self_attn.v_proj.weight" => layer.v_proj = Some(name.clone()),
                        "self_attn.v_proj.bias" => layer.v_bias = Some(name.clone()),
                        "attn_q_norm.weight" => layer.q_norm = Some(name.clone()),
                        "attn_k_norm.weight" => layer.k_norm = Some(name.clone()),
                        "self_attn.o_proj.weight" => layer.o_proj = Some(name.clone()),
                        "self_attn.o_proj.bias" => layer.o_bias = Some(name.clone()),
                        "post_attention_layernorm.weight" => layer.post_attention_layernorm = Some(name.clone()),
                        "post_attention_norm.weight" => layer.post_attention_norm = Some(name.clone()),
                        "post_ffw_norm.weight" => layer.post_ffw_norm = Some(name.clone()),
                        "mlp.gate_proj.weight" => layer.gate_proj = Some(name.clone()),
                        "mlp.gate_proj.bias" => layer.gate_bias = Some(name.clone()),
                        "mlp.up_proj.weight" => layer.up_proj = Some(name.clone()),
                        "mlp.up_proj.bias" => layer.up_bias = Some(name.clone()),
                        "mlp.down_proj.weight" => layer.down_proj = Some(name.clone()),
                        "mlp.down_proj.bias" => layer.down_bias = Some(name.clone()),
                        "per_layer_input_gate.weight" => layer.per_layer_input_gate = Some(name.clone()),
                        "per_layer_projection.weight" => layer.per_layer_projection = Some(name.clone()),
                        "post_per_layer_input_norm.weight" => layer.post_per_layer_input_norm = Some(name.clone()),
                        "layer_output_scale.weight" => layer.layer_output_scale = Some(name.clone()),
                        _ => {}
                    }
                }
            }
        }
    }

    if per_layer_token_embd.is_none() && per_layer_model_proj.is_some() {
        per_layer_token_embd = embed_tokens.clone();
    }
    
    let mut layers: Vec<LayerTensors> = layer_map.into_values().collect();
    layers.sort_by_key(|l| l.index);
    
    info!("Scanned tensors: found {} layers", layers.len());
    TensorGroupMap {
        embed_tokens,
        final_norm,
        lm_head,
        per_layer_token_embd,
        per_layer_model_proj,
        per_layer_proj_norm,
        layers,
    }
}
