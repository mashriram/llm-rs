//! Concurrency integration tests for llm-rs ServingEngine and Scheduler.
//! Exercises 100 to 500 concurrent requests simulating high-load server usage.

use llm_core::backend::LlmBackend;
use llm_core::types::*;
use llm_scheduler::engine::ServingEngine;
use std::sync::Arc;

struct DummyConcurrencyBackend {
    kv_config: KvCacheConfig,
}

impl DummyConcurrencyBackend {
    fn new() -> Self {
        Self {
            kv_config: KvCacheConfig {
                n_layers: 4,
                n_kv_heads: 4,
                head_dim: 32,
                block_size: 16,
                dtype: KvDtype::F16,
            },
        }
    }
}

impl LlmBackend for DummyConcurrencyBackend {
    fn name(&self) -> &str {
        "dummy-concurrency"
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(2)
    }

    fn clear_sequence(&self, _seq_id: SeqId) {}
    fn set_explicit_dequantize(&mut self, _val: bool) {}
    fn set_use_vram_embeddings(&mut self, _val: bool) {}

    fn load_weights(&mut self, _path: &std::path::Path) -> anyhow::Result<ModelMeta> {
        Ok(ModelMeta {
            vocab_size: 32000,
            hidden_dim: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 32,
            intermediate_dim: 512,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            weight_dtype: WeightDtype::F16,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            hidden_act: HiddenAct::SiLU,
            no_rope_layers: vec![false; 4],
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
            is_gemma: false,
            ple_dim: None,
            embed_scale: None,
            arch: "test".to_string(),
            chat_template: None,
            eos_token_str: None,
        })
    }

    fn forward_pass(&self, batch: &BatchInput) -> anyhow::Result<BatchOutput> {
        let num_seqs = batch.seq_ids.len();
        let next_tokens = vec![42u32; num_seqs];
        Ok(BatchOutput {
            seq_ids: batch.seq_ids.clone(),
            next_tokens,
            logits: None,
        })
    }

    fn sample(&self, logits: &[f32], _params: &SampleParams, _token_history: &[TokenId]) -> anyhow::Result<TokenId> {
        Ok(logits.first().map(|&f| f as TokenId).unwrap_or(42))
    }

    fn kv_cache_config(&self) -> KvCacheConfig {
        self.kv_config
    }
}

async fn run_concurrency_test(num_concurrent_users: usize) {
    let backend = Box::new(DummyConcurrencyBackend::new());
    let engine = Arc::new(ServingEngine::new(backend, num_concurrent_users * 64));

    let mut handles = Vec::with_capacity(num_concurrent_users);

    for user_idx in 0..num_concurrent_users {
        let engine_clone = engine.clone();
        let seq_id = (user_idx + 1) as u64;

        let handle = tokio::spawn(async move {
            let mut rx = engine_clone.subscribe();
            let req = InferRequest {
                seq_id,
                prompt_tokens: vec![1, 10, 20, 30],
                max_new_tokens: 10,
                sample_params: SampleParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    repetition_penalty: 1.0,
                    max_new_tokens: 10,
                },
            };

            engine_clone.add_request(req).expect("Failed to add request");

            let mut generated_count = 0;
            let mut is_finished = false;

            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if event.seq_id == seq_id {
                            if event.is_eos {
                                is_finished = true;
                                break;
                            } else {
                                generated_count += 1;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }

            (seq_id, generated_count, is_finished)
        });

        handles.push(handle);
    }

    let mut finished_count = 0;
    for handle in handles {
        let (seq_id, count, finished) = handle.await.expect("Task failed");
        assert!(finished, "Request {} failed to finish cleanly", seq_id);
        assert!(count > 0, "Request {} generated 0 tokens", seq_id);
        finished_count += 1;
    }

    assert_eq!(
        finished_count, num_concurrent_users,
        "All {} concurrent requests must complete cleanly",
        num_concurrent_users
    );
}

#[tokio::test]
async fn test_concurrency_100_users() {
    run_concurrency_test(100).await;
}

#[tokio::test]
async fn test_concurrency_250_users() {
    run_concurrency_test(250).await;
}

#[tokio::test]
async fn test_concurrency_500_users() {
    run_concurrency_test(500).await;
}
