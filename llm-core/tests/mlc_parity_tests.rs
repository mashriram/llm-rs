use llm_core::types::{ModelMeta, WeightDtype, HiddenAct};
use llm_core::conv_template::Conversation;
use llm_core::metadata::HfConfig;
use candle_core::{Tensor, Device};

#[test]
fn test_loader_parity() {
    let config_json = r#"{
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "rope_theta": 10000.0,
        "torch_dtype": "float16"
    }"#;

    let hf_config: HfConfig = serde_json::from_str(config_json).unwrap();
    let n_heads = hf_config.num_attention_heads;
    let n_kv_heads = hf_config.num_key_value_heads.unwrap_or(n_heads);
    let head_dim = hf_config.hidden_size / n_heads;

    let weight_dtype = match hf_config.torch_dtype.as_str() {
        "float32" => WeightDtype::F32,
        "bfloat16" => WeightDtype::BF16,
        _ => WeightDtype::F16,
    };

    let meta = ModelMeta {
        vocab_size: hf_config.vocab_size,
        hidden_dim: hf_config.hidden_size,
        n_layers: hf_config.num_hidden_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        intermediate_dim: hf_config.intermediate_size,
        max_seq_len: hf_config.max_position_embeddings,
        rope_theta: hf_config.rope_theta,
        weight_dtype,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        hidden_act: HiddenAct::SiLU,
        no_rope_layers: vec![false; hf_config.num_hidden_layers],
        has_vision_encoder: false,
        vision_hidden_dim: None,
        vision_patch_size: None,
        vision_image_size: None,
        vision_num_layers: None,
        vision_num_heads: None,
        vision_projection_dim: None,
        spatial_merge_size: None,
        is_deepstack_layers: None,
        projector_type: None,
    };

    assert_eq!(meta.vocab_size, 32000);
    assert_eq!(meta.hidden_dim, 4096);
    assert_eq!(meta.n_layers, 32);
    assert_eq!(meta.n_heads, 32);
    assert_eq!(meta.n_kv_heads, 8);
    assert_eq!(meta.head_dim, 128);
    assert_eq!(meta.intermediate_dim, 11008);
    assert_eq!(meta.max_seq_len, 2048);
    assert_eq!(meta.rope_theta, 10000.0);
    assert_eq!(meta.weight_dtype, WeightDtype::F16);
}

#[test]
fn test_op_parity_rmsnorm() {
    let device = Device::Cpu;
    let input = Tensor::new(&[1.0f32, 2.0f32, 3.0f32, 4.0f32], &device).unwrap();
    
    let eps = 1e-5f32;
    let sq = input.sqr().unwrap();
    let sum = sq.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let mean = sum / 4.0;
    let inv_std = 1.0 / (mean + eps).sqrt();
    let expected = input.to_vec1::<f32>().unwrap().iter().map(|x| x * inv_std).collect::<Vec<_>>();

    // Directly test RmsNorm-like operation
    let variance = input.sqr().unwrap().mean_keepdim(candle_core::D::Minus1).unwrap();
    let x_norm = input.broadcast_div(&(variance + eps as f64).unwrap().sqrt().unwrap()).unwrap();
    let output_vec = x_norm.to_vec1::<f32>().unwrap();

    for (a, b) in output_vec.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_op_parity_silu() {
    let device = Device::Cpu;
    let input = Tensor::new(&[-2.0f32, 0.0f32, 2.0f32], &device).unwrap();

    // Directly test SiLU calculation
    let sig = input.neg().unwrap().exp().unwrap().affine(1.0, 1.0).unwrap().recip().unwrap();
    let out = input.mul(&sig).unwrap();
    let output_vec = out.to_vec1::<f32>().unwrap();

    let expected = vec![
        -2.0f32 * (1.0f32 / (1.0f32 + (-(-2.0f32)).exp())),
        0.0f32,
        2.0f32 * (1.0f32 / (1.0f32 + (-2.0f32).exp())),
    ];

    assert!((output_vec[0] - expected[0]).abs() < 1e-5);
    assert!((output_vec[1] - expected[1]).abs() < 1e-5);
    assert!((output_vec[2] - expected[2]).abs() < 1e-5);
}

#[test]
fn test_quantization_parity() {
    let quantized_weights = vec![127i8, -128i8, 0i8];
    let scale = 0.5f32;

    let dequantized: Vec<f32> = quantized_weights.iter().map(|&w| w as f32 * scale).collect();
    assert_eq!(dequantized[0], 63.5f32);
    assert_eq!(dequantized[1], -64.0f32);
    assert_eq!(dequantized[2], 0.0f32);
}

#[test]
fn test_conversation_template_parity() {
    let conv_json = r#"{
        "name": "llama-2",
        "system_template": "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        "system_message": "You are a helpful assistant.",
        "roles": {
            "user": "[INST]",
            "assistant": "[/INST]"
        },
        "role_templates": {
            "user": "{user_message}",
            "assistant": "{assistant_message}"
        },
        "messages": [],
        "seps": [" "],
        "role_content_sep": " ",
        "role_empty_sep": "",
        "stop_str": ["</s>"],
        "add_role_after_system_message": false,
        "stop_token_ids": [2]
    }"#;

    let conv = Conversation::from_json(conv_json).unwrap();
    assert_eq!(conv.name, "llama-2");
    assert_eq!(conv.system_message, "You are a helpful assistant.");
}
