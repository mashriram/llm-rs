use llm_core::types::{
    InferRequest, SampleParams, BatchInput, BatchOutput, KvCacheConfig, KvDtype, ModelMeta, WeightDtype, TokenId
};
use llm_core::backend::LlmBackend;
use llm_scheduler::block_allocator::BlockAllocator;
use llm_scheduler::prefix_cache::PrefixCache;
use llm_scheduler::scheduler::Scheduler;
use anyhow::Result;

// A simple mock backend to test Scheduler orchestration without loading actual model weights.
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
        // Return mock token 42 for each sequence in the batch
        let next_tokens = vec![42; input.seq_ids.len()];
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
        "mock"
    }
}

#[test]
fn test_block_allocator_lifecycle() {
    // 1. Initialize allocator with 4 blocks
    let mut allocator = BlockAllocator::new(4);
    assert_eq!(allocator.free_blocks(), 4);

    // 2. Allocate blocks
    let block1 = allocator.allocate().expect("Allocation failed");
    let block2 = allocator.allocate().expect("Allocation failed");
    assert_eq!(allocator.free_blocks(), 2);

    // 3. Verify reference counting
    allocator.increment_ref(block1).unwrap();
    
    // 4. Free blocks
    let freed1 = allocator.free(block1).unwrap(); // Ref count goes to 1, not freed yet
    assert!(!freed1);
    assert_eq!(allocator.free_blocks(), 2);

    let freed2 = allocator.free(block1).unwrap(); // Ref count goes to 0, block is reclaimed
    assert!(freed2);
    assert_eq!(allocator.free_blocks(), 3);

    let freed3 = allocator.free(block2).unwrap();
    assert!(freed3);
    assert_eq!(allocator.free_blocks(), 4);
}

#[test]
fn test_prefix_cache_radix_tree() {
    // 1. Initialize prefix cache with max 2 recycling sequences
    let mut cache = PrefixCache::new(2);

    // 2. Insert sequence 1: [1, 2, 3, 4]
    let seq1 = vec![1, 2, 3, 4];
    let match1 = cache.insert_sequence(101, &seq1, -1);
    assert_eq!(match1.matched_offset, 0); // No prefix matched

    // 3. Insert sequence 2: [1, 2, 3, 5]
    let seq2 = vec![1, 2, 3, 5];
    let match2 = cache.insert_sequence(102, &seq2, -1);
    
    // The common prefix is [1, 2, 3], length 3.
    assert_eq!(match2.matched_offset, 3);
    assert_eq!(match2.fork_seq_id, Some(101));
}

#[test]
fn test_scheduler_orchestration() {
    let backend = Box::new(MockBackend);
    let mut scheduler = Scheduler::new(backend, 16);

    // 1. Add a prefill request
    let req1 = InferRequest {
        seq_id: 1001,
        prompt_tokens: vec![1, 2, 3, 5],
        sample_params: SampleParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            max_new_tokens: 3,
        },
        max_new_tokens: 3,
    };

    scheduler.add_request(req1);

    // 2. Step 1: Run prefill
    let step1_res = scheduler.step().expect("Step 1 failed");
    assert_eq!(step1_res.len(), 1);
    assert_eq!(step1_res[0].0, 1001); // seq_id
    assert_eq!(step1_res[0].1, 42);   // generated token from MockBackend
    assert!(!step1_res[0].2);          // is_eos should be false (max_new_tokens is 3)

    // 3. Step 2: Run decode (first token)
    let step2_res = scheduler.step().expect("Step 2 failed");
    assert_eq!(step2_res.len(), 1);
    assert_eq!(step2_res[0].1, 42);
    assert!(!step2_res[0].2);

    // 4. Step 3: Run decode (second token, reaches max_new_tokens)
    let step3_res = scheduler.step().expect("Step 3 failed");
    assert_eq!(step3_res.len(), 1);
    assert_eq!(step3_res[0].1, 42);
    assert!(step3_res[0].2); // Should be finished (reached_max = true)
}
