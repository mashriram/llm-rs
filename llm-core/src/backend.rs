use std::path::Path;
use crate::types::*;
use anyhow::Result;

/// The central abstraction for model execution.
/// 
/// Invariant: after `load_weights` returns Ok, the backend is ready to serve
/// `forward_pass` calls indefinitely without re-loading.
/// 
/// Implementors: CandleBackend (CPU/reference), CubeclBackend (GPU/production).
pub trait LlmBackend: Send + Sync {
    /// Load weights from `path` (GGUF or SafeTensors directory).
    /// Returns metadata extracted from the model config.
    fn load_weights(&mut self, path: &Path) -> Result<ModelMeta>;

    /// Execute one forward pass over the batch.
    /// The block tables in `batch` are managed by the scheduler; 
    /// the backend must not allocate or free blocks.
    fn forward_pass(&self, batch: &BatchInput) -> Result<BatchOutput>;

    /// Apply sampling to logits. Separated from forward_pass so the 
    /// scheduler can apply speculative decoding or beam search on raw logits.
    fn sample(&self, logits: &[f32], params: &SampleParams, 
              token_history: &[TokenId]) -> Result<TokenId>;

    /// Return the KV cache configuration this backend requires.
    /// Called once by the scheduler at startup to size the block pool.
    fn kv_cache_config(&self) -> KvCacheConfig;
    
    /// Backend name for logging/metrics.
    fn name(&self) -> &str;

    /// Clean up any resources associated with the sequence.
    fn clear_sequence(&self, _seq_id: SeqId) {}

    /// Get the EOS token ID for this backend.
    fn eos_token_id(&self) -> u32 {
        2
    }
}

/// A reusable DummyBackend for testing.
pub struct DummyBackend {
    pub meta: ModelMeta,
}

impl Default for DummyBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyBackend {
    pub fn new() -> Self {
        Self {
            meta: ModelMeta {
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
                hidden_act: crate::types::HiddenAct::SiLU,
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
            }
        }
    }
}

impl LlmBackend for DummyBackend {
    fn load_weights(&mut self, _path: &Path) -> Result<ModelMeta> {
        Ok(self.meta.clone())
    }

    fn forward_pass(&self, input: &BatchInput) -> Result<BatchOutput> {
        let next_tokens = input.seq_ids.iter().map(|&id| (id % 100) as u32 + 2).collect();
        Ok(BatchOutput {
            seq_ids: input.seq_ids.clone(),
            next_tokens,
            logits: None,
        })
    }

    fn sample(&self, _logits: &[f32], _params: &SampleParams, _token_history: &[TokenId]) -> Result<TokenId> {
        Ok(42)
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
        "dummy"
    }
}
