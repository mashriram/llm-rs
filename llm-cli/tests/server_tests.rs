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
use llm_cli::{create_router, create_router_with_body_limit, AppState, ChatCompletionRequest, ChatMessage};
use anyhow::Result;

fn mock_meta(arch: &str) -> ModelMeta {
    ModelMeta {
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
        has_audio_encoder: false,
        audio_hidden_dim: None,
        audio_block_count: None,
        audio_embedding_length: None,
        audio_num_mel_bins: None,
        shared_kv_layers: None,
        sliding_window_pattern: None,
        sliding_window: None,
        key_length: None,
        key_length_swa: None,
        rope_theta_swa: None,
        final_logit_softcapping: None,
        is_gemma: arch == "gemma4",
        ple_dim: None,
        embed_scale: None,
        arch: arch.to_string(),
        chat_template: None,
        eos_token_str: None,
    }
}

// Mock Backend for server testing
struct MockBackend;

impl LlmBackend for MockBackend {
    fn load_weights(&mut self, _path: &std::path::Path) -> Result<ModelMeta> {
        Ok(mock_meta("mock"))
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

fn tokenizer_path() -> std::path::PathBuf {
    let parent = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let candidates = [
        parent.join("qwen2.5-0.5b/tokenizer.json"),
        parent.join("qwen2.5-1.5b/tokenizer.json"),
        parent.join("models/qwen2.5-0.5b/tokenizer.json"),
    ];
    for p in &candidates {
        if p.exists() {
            return p.clone();
        }
    }
    parent.join("qwen2.5-0.5b/tokenizer.json")
}

fn build_state(arch: &str, max_tokens_limit: usize) -> Arc<AppState> {
    let backend = Box::new(MockBackend);
    let engine = Arc::new(ServingEngine::new(backend, 16));
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(tokenizer_path()).unwrap());
    Arc::new(AppState {
        engine,
        model_name: "mock-model".to_string(),
        tokenizer,
        meta: Arc::new(mock_meta(arch)),
        max_tokens_limit,
    })
}

#[tokio::test]
async fn test_chat_completions_non_streaming() {
    let state = build_state("mock", llm_cli::DEFAULT_MAX_TOKENS_LIMIT);
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
    let state = build_state("mock", llm_cli::DEFAULT_MAX_TOKENS_LIMIT);
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

/// Fix #4: a client asking for more than the configured max_tokens ceiling
/// must get a clean 400, not have the request silently clamped or accepted.
#[tokio::test]
async fn test_max_tokens_over_limit_is_rejected() {
    let state = build_state("mock", 100);
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
        max_tokens: Some(999_999_999),
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"]["message"].as_str().unwrap().contains("max_tokens"));
}

/// Fix #4: request bodies larger than the configured cap must be rejected
/// rather than fully buffered into memory.
#[tokio::test]
async fn test_oversized_request_body_is_rejected() {
    let state = build_state("mock", llm_cli::DEFAULT_MAX_TOKENS_LIMIT);
    // Tiny cap so the test doesn't need to construct megabytes of payload.
    let app = create_router_with_body_limit(state, 64);

    let big_content = "x".repeat(10_000);
    let req_body = ChatCompletionRequest {
        model: "mock-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: big_content,
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

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

/// Fix #7: a non-ChatML architecture (Gemma) must be rendered with its own
/// per-arch format, not the previously-hardcoded ChatML `<|im_start|>` tags.
#[test]
fn test_gemma_arch_uses_gemma_format_not_chatml() {
    let meta = mock_meta("gemma4");
    let messages = vec![ChatMessage { role: "user".to_string(), content: "hi".to_string() }];
    let rendered = llm_cli::render_prompt(&messages, &meta);

    assert!(rendered.contains("<|turn>user"), "expected gemma4 turn format, got: {}", rendered);
    assert!(!rendered.contains("<|im_start|>"), "gemma4 must not use hardcoded ChatML format, got: {}", rendered);
}

/// Fix #7 (regression guard): a plain qwen/ChatML-style arch still renders
/// with the ChatML fallback format.
#[test]
fn test_qwen_arch_uses_chatml_format() {
    let meta = mock_meta("qwen2");
    let messages = vec![ChatMessage { role: "user".to_string(), content: "hi".to_string() }];
    let rendered = llm_cli::render_prompt(&messages, &meta);
    assert!(rendered.contains("<|im_start|>user"), "expected ChatML format, got: {}", rendered);
}
