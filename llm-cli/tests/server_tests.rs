use std::sync::Arc;
use axum::{
    body::Body,
    http::{self, Request, StatusCode},
};
use tower::ServiceExt; // for `oneshot` and `ready`
use serde_json::Value;

use llm_core::types::{BatchInput, BatchOutput, KvCacheConfig, KvDtype, ModelMeta, WeightDtype, TokenId, SampleParams};
use llm_core::backend::LlmBackend;
use llm_scheduler::engine::ServingEngine;
use llm_cli::{create_router, AppState, ChatCompletionRequest, ChatMessage};
use anyhow::Result;

// Mock Backend for server testing
struct MockBackend;

impl LlmBackend for MockBackend {
    fn load_weights(&mut self, _path: &std::path::Path) -> Result<ModelMeta> {
        Ok(ModelMeta {
            vocab_size: 32000,
            hidden_dim: 4096,
            n_layers: 2,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 11008,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            weight_dtype: WeightDtype::F16,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            hidden_act: llm_core::types::HiddenAct::SiLU,
            no_rope_layers: vec![false; 2],
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
            shared_kv_layers: None,
            sliding_window_pattern: None,
            sliding_window: None,
            key_length: None,
            key_length_swa: None,
            rope_theta_swa: None,
            final_logit_softcapping: None,
            is_gemma: false,
            ple_dim: None,
            embed_scale: None,
        })
    }

    fn forward_pass(&self, input: &BatchInput) -> Result<BatchOutput> {
        // Return mock token IDs (e.g. ASCII values of "Hello" or 42)
        let next_tokens = vec![2; input.seq_ids.len()]; // Return 2 (EOS) to finish immediately in one step
        Ok(BatchOutput {
            seq_ids: input.seq_ids.clone(),
            next_tokens,
            logits: None,
        })
    }

    fn sample(&self, _logits: &[f32], _params: &SampleParams, _token_history: &[TokenId]) -> Result<TokenId> {
        Ok(2)
    }

    fn kv_cache_config(&self) -> KvCacheConfig {
        KvCacheConfig {
            n_layers: 2,
            n_kv_heads: 8,
            head_dim: 128,
            block_size: 4,
            dtype: KvDtype::F16,
        }
    }

    fn name(&self) -> &str {
        "mock"
    }
}

#[tokio::test]
async fn test_chat_completions_non_streaming() {
    let backend = Box::new(MockBackend);
    let engine = Arc::new(ServingEngine::new(backend, 16));
    let tokenizer_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("qwen2.5-0.5b/tokenizer.json");
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(tokenizer_path).unwrap());
    let state = Arc::new(AppState {
        engine,
        model_name: "mock-model".to_string(),
        tokenizer,
    });

    let app = create_router(state);

    let req_body = ChatCompletionRequest {
        model: "mock-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }],
        stream: false,
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(10),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri("/v1/chat/completions")
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["model"], "mock-model");
    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"].is_array());
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn test_chat_completions_streaming() {
    let backend = Box::new(MockBackend);
    let engine = Arc::new(ServingEngine::new(backend, 16));
    let tokenizer_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("qwen2.5-0.5b/tokenizer.json");
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(tokenizer_path).unwrap());
    let state = Arc::new(AppState {
        engine,
        model_name: "mock-model".to_string(),
        tokenizer,
    });

    let app = create_router(state);

    let req_body = ChatCompletionRequest {
        model: "mock-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }],
        stream: true,
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(10),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri("/v1/chat/completions")
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
}
