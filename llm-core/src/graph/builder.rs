use crate::types::ModelMeta;
use crate::graph::scan::TensorGroupMap;
use crate::graph::ops::{ComputeGraph, Operator};

pub fn build_graph(meta: &ModelMeta, group: &TensorGroupMap) -> ComputeGraph {
    let mut graph = ComputeGraph::new();

    // 1. Text embedding
    let embed_weight = group.embed_tokens.clone().unwrap_or_else(|| "model.embed_tokens.weight".to_string());
    graph.add_op(Operator::Embed {
        input_ids: "input_ids".to_string(),
        weight: embed_weight.clone(),
        output: "text_embeddings".to_string(),
    });

    let mut current_hidden = if meta.has_vision_encoder {
        // Run vision encoder if present
        graph.add_op(Operator::VisualEmbed {
            pixel_values: "pixel_values".to_string(),
            output: "visual_embeddings".to_string(),
        });
        // Splice visual embeddings into text token embeddings at placeholder positions
        graph.add_op(Operator::SpliceTensors {
            text_embeds: "text_embeddings".to_string(),
            visual_embeds: "visual_embeddings".to_string(),
            output: "spliced_embeddings".to_string(),
        });
        "spliced_embeddings".to_string()
    } else {
        "text_embeddings".to_string()
    };

    // 2. Decode layers
    for layer in &group.layers {
        let l_idx = layer.index;
        let layer_input = current_hidden.clone();
        let pre_norm_out = format!("layer_{}_pre_norm", l_idx);
        
        // Input Norm
        let norm_weight = layer.input_layernorm.clone().unwrap_or_else(|| {
            format!("model.layers.{}.input_layernorm.weight", l_idx)
        });
        graph.add_op(Operator::RMSNorm {
            input: layer_input.clone(),
            weight: norm_weight,
            output: pre_norm_out.clone(),
            eps: meta.rms_norm_eps,
        });

        // QKV projections
        let q_out = format!("layer_{}_q_proj", l_idx);
        let k_out = format!("layer_{}_k_proj", l_idx);
        let v_out = format!("layer_{}_v_proj", l_idx);

        let q_proj_w = layer.q_proj.clone().unwrap_or_else(|| format!("model.layers.{}.self_attn.q_proj.weight", l_idx));
        let k_proj_w = layer.k_proj.clone().unwrap_or_else(|| format!("model.layers.{}.self_attn.k_proj.weight", l_idx));
        let v_proj_w = layer.v_proj.clone().unwrap_or_else(|| format!("model.layers.{}.self_attn.v_proj.weight", l_idx));

        graph.add_op(Operator::MatMul {
            input: pre_norm_out.clone(),
            weight: q_proj_w,
            bias: layer.q_bias.clone(),
            output: q_out.clone(),
        });
        graph.add_op(Operator::MatMul {
            input: pre_norm_out.clone(),
            weight: k_proj_w,
            bias: layer.k_bias.clone(),
            output: k_out.clone(),
        });
        graph.add_op(Operator::MatMul {
            input: pre_norm_out.clone(),
            weight: v_proj_w,
            bias: layer.v_bias.clone(),
            output: v_out.clone(),
        });

        // Apply QK Norms if present (e.g. Qwen3-VL)
        let q_post_norm = if let Some(q_norm_w) = &layer.q_norm {
            let out = format!("layer_{}_q_normed", l_idx);
            graph.add_op(Operator::RMSNorm {
                input: q_out,
                weight: q_norm_w.clone(),
                output: out.clone(),
                eps: 1e-6,
            });
            out
        } else {
            q_out
        };

        let k_post_norm = if let Some(k_norm_w) = &layer.k_norm {
            let out = format!("layer_{}_k_normed", l_idx);
            graph.add_op(Operator::RMSNorm {
                input: k_out,
                weight: k_norm_w.clone(),
                output: out.clone(),
                eps: 1e-6,
            });
            out
        } else {
            k_out
        };

        // Rotary Position Embedding
        let q_rope = format!("layer_{}_q_rope", l_idx);
        let k_rope = format!("layer_{}_k_rope", l_idx);
        
        let skip_rope = if l_idx < meta.no_rope_layers.len() {
            meta.no_rope_layers[l_idx]
        } else {
            false
        };

        if skip_rope {
            graph.add_op(Operator::RopeSkip {
                q: q_post_norm,
                k: k_post_norm,
                output_q: q_rope.clone(),
                output_k: k_rope.clone(),
            });
        } else {
            graph.add_op(Operator::Rope {
                q: q_post_norm,
                k: k_post_norm,
                output_q: q_rope.clone(),
                output_k: k_rope.clone(),
                layer_idx: l_idx,
                rope_theta: meta.rope_theta,
            });
        }

        // Paged Attention
        let attn_out = format!("layer_{}_attn_out", l_idx);
        graph.add_op(Operator::PagedAttention {
            q: q_rope,
            k: k_rope,
            v: v_out,
            output: attn_out.clone(),
            layer_idx: l_idx,
            n_heads: meta.n_heads,
            n_kv_heads: meta.n_kv_heads,
            head_dim: meta.head_dim,
        });

        // Attention Output projection
        let o_out = format!("layer_{}_o_proj", l_idx);
        let o_proj_w = layer.o_proj.clone().unwrap_or_else(|| format!("model.layers.{}.self_attn.o_proj.weight", l_idx));
        graph.add_op(Operator::MatMul {
            input: attn_out,
            weight: o_proj_w,
            bias: layer.o_bias.clone(),
            output: o_out.clone(),
        });

        // Residual Add
        let post_attn = format!("layer_{}_post_attn", l_idx);
        graph.add_op(Operator::Add {
            lhs: layer_input,
            rhs: o_out,
            output: post_attn.clone(),
        });

        // Post-Attention Norm
        let post_norm_out = format!("layer_{}_post_norm", l_idx);
        let post_norm_w = layer.post_attention_layernorm.clone().unwrap_or_else(|| {
            format!("model.layers.{}.post_attention_layernorm.weight", l_idx)
        });
        graph.add_op(Operator::RMSNorm {
            input: post_attn.clone(),
            weight: post_norm_w,
            output: post_norm_out.clone(),
            eps: meta.rms_norm_eps,
        });

        // MLP Block (Architecture Gated vs Non-Gated)
        let mlp_out = format!("layer_{}_mlp_out", l_idx);
        let down_proj_w = layer.down_proj.clone().unwrap_or_else(|| format!("model.layers.{}.mlp.down_proj.weight", l_idx));

        if let Some(gate_proj_w) = &layer.gate_proj {
            // Gated MLP (SwiGLU)
            let gate_out = format!("layer_{}_gate_proj", l_idx);
            let up_out = format!("layer_{}_up_proj", l_idx);
            let act_out = format!("layer_{}_act", l_idx);
            let mlp_inter = format!("layer_{}_mlp_inter", l_idx);

            let up_proj_w = layer.up_proj.clone().unwrap_or_else(|| format!("model.layers.{}.mlp.up_proj.weight", l_idx));

            graph.add_op(Operator::MatMul {
                input: post_norm_out.clone(),
                weight: gate_proj_w.clone(),
                bias: layer.gate_bias.clone(),
                output: gate_out.clone(),
            });
            graph.add_op(Operator::MatMul {
                input: post_norm_out,
                weight: up_proj_w,
                bias: layer.up_bias.clone(),
                output: up_out.clone(),
            });
            graph.add_op(Operator::Activation {
                input: gate_out,
                output: act_out.clone(),
                act: meta.hidden_act,
            });
            graph.add_op(Operator::Mul {
                lhs: act_out,
                rhs: up_out,
                output: mlp_inter.clone(),
            });
            graph.add_op(Operator::MatMul {
                input: mlp_inter,
                weight: down_proj_w,
                bias: layer.down_bias.clone(),
                output: mlp_out.clone(),
            });
        } else {
            // Non-Gated MLP
            let up_out = format!("layer_{}_up_proj", l_idx);
            let act_out = format!("layer_{}_act", l_idx);
            let up_proj_w = layer.up_proj.clone().unwrap_or_else(|| format!("model.layers.{}.mlp.up_proj.weight", l_idx));

            graph.add_op(Operator::MatMul {
                input: post_norm_out,
                weight: up_proj_w,
                bias: layer.up_bias.clone(),
                output: up_out.clone(),
            });
            graph.add_op(Operator::Activation {
                input: up_out,
                output: act_out.clone(),
                act: meta.hidden_act,
            });
            graph.add_op(Operator::MatMul {
                input: act_out,
                weight: down_proj_w,
                bias: layer.down_bias.clone(),
                output: mlp_out.clone(),
            });
        }

        // Residual Add
        let mut layer_output = format!("layer_{}_out", l_idx);
        graph.add_op(Operator::Add {
            lhs: post_attn,
            rhs: mlp_out,
            output: layer_output.clone(),
        });

        if meta.has_vision_encoder {
            if let Some(ds_flags) = &meta.is_deepstack_layers {
                if ds_flags.get(l_idx).copied().unwrap_or(false) {
                    let fused_out = format!("layer_{}_fused", l_idx);
                    graph.add_op(Operator::DeepStackFuse {
                        input: layer_output.clone(),
                        layer_idx: l_idx,
                        output: fused_out.clone(),
                    });
                    layer_output = fused_out;
                }
            }
        }

        current_hidden = layer_output;
    }

    // 3. Final norm
    let final_norm_w = group.final_norm.clone().unwrap_or_else(|| "model.norm.weight".to_string());
    graph.add_op(Operator::RMSNorm {
        input: current_hidden,
        weight: final_norm_w,
        output: "final_norm_out".to_string(),
        eps: meta.rms_norm_eps,
    });

    // 4. LM Head
    let lm_head_w = if meta.tie_word_embeddings {
        embed_weight
    } else {
        group.lm_head.clone().unwrap_or(embed_weight)
    };
    graph.add_op(Operator::MatMul {
        input: "final_norm_out".to_string(),
        weight: lm_head_w,
        bias: None,
        output: "logits".to_string(),
    });

    graph
}
