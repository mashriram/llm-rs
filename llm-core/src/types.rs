use serde::{Serialize, Deserialize};

/// A unique identifier for a sequence within the scheduler.
/// Monotonically increasing. Never reused within a session.
pub type SeqId = u64;

/// A token identifier (index into vocabulary).
pub type TokenId = u32;

/// A physical block identifier in the PagedAttention block pool.
/// Valid range: 0..BlockPool::capacity(). u32 to keep block tables cache-friendly.
pub type BlockId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiddenAct {
    SiLU,
    GeLU,
}

/// Static metadata extracted from a model's config file.
/// Populated once at load time; never mutated during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,       // may differ from n_heads (GQA)
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub weight_dtype: WeightDtype,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub hidden_act: HiddenAct,
    pub no_rope_layers: Vec<bool>,
    pub has_vision_encoder: bool,
    pub vision_hidden_dim: Option<usize>,
    pub vision_patch_size: Option<usize>,
    pub vision_image_size: Option<usize>,
    pub vision_num_layers: Option<usize>,
    pub vision_num_heads: Option<usize>,
    pub vision_projection_dim: Option<usize>,
    pub spatial_merge_size: Option<usize>,
    pub is_deepstack_layers: Option<Vec<bool>>,
    pub projector_type: Option<String>,
}

/// Weight quantization format. Maps to MLC-LLM's dtype strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum WeightDtype {
    F32,
    F16,
    BF16,
    Q8_0,    // 8-bit symmetric, scale per 32-weight block
    Q4_0,    // 4-bit symmetric, scale per 32-weight block
    Q4_K,    // 4-bit with k-means quantization (GGUF)
}

/// Configuration for the KV cache, derived from ModelMeta at load time.
#[derive(Debug, Clone, Copy)]
pub struct KvCacheConfig {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,     // tokens per physical block (default: 16)
    pub dtype: KvDtype,
}

/// KV cache element type. Independent of weight dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype { 
    F16, 
    BF16, 
    Q8 
}

/// A single request from the HTTP layer to the scheduler.
#[derive(Debug, Clone)]
pub struct InferRequest {
    pub seq_id: SeqId,
    pub prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub sample_params: SampleParams,
}

/// Sampling hyperparameters. Mirrors MLC-LLM's SamplerConfig.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub temperature: f32,    // 0.0 = greedy
    pub top_p: f32,          // nucleus sampling threshold
    pub top_k: usize,        // 0 = disabled
    pub repetition_penalty: f32,  // 1.0 = disabled
    pub max_new_tokens: usize,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self { 
            temperature: 1.0, 
            top_p: 1.0, 
            top_k: 0, 
            repetition_penalty: 1.0, 
            max_new_tokens: 512 
        }
    }
}

/// Input to a single forward pass: a batch of sequences.
/// Prefill and decode sequences may be mixed (Varlen path).
#[derive(Debug, Clone)]
pub struct BatchInput {
    pub seq_ids: Vec<SeqId>,
    pub token_ids: Vec<TokenId>,       // flattened; use cu_seqlens to index
    pub cu_seqlens: Vec<u32>,          // cumulative sequence lengths (len = batch+1)
    pub block_tables: Vec<Vec<BlockId>>, // per-sequence logical->physical mapping
    pub is_prefill: Vec<bool>,         // per-sequence flag
}

/// Output from a single forward pass.
#[derive(Debug, Clone)]
pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,     // one per sequence
    pub logits: Option<Vec<Vec<f32>>>, // only populated if caller needs raw logits
}
