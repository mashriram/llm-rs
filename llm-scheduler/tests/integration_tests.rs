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
            arch: "mock".to_string(),
            chat_template: None,
            eos_token_str: None,
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
    let mut scheduler = Scheduler::new(backend, 16).expect("Scheduler::new failed");

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

// Regression test for audit finding #1: when a sequence is preempted due to
// out-of-memory (no free blocks left to grow its KV cache), the client must
// still receive a terminal event (is_eos=true) rather than being left hanging
// forever with no final signal.
#[test]
fn test_oom_preemption_sends_terminal_event() {
    let backend = Box::new(MockBackend);
    // Only 1 physical block available, block_size=4 tokens/block -> capacity
    // for a single sequence is exactly 4 tokens with no room to grow.
    let mut scheduler = Scheduler::new(backend, 1).expect("Scheduler::new failed");

    let req = InferRequest {
        seq_id: 42,
        prompt_tokens: vec![1, 2, 3, 4], // exactly fills the only block (4 tokens)
        sample_params: SampleParams::default(),
        max_new_tokens: 50, // far more than the pool can ever support
    };
    scheduler.add_request(req);

    // Prefill consumes the only block (4/4 tokens used). The generated token
    // pushes total_tokens to 5, exceeding the 4-token capacity, and there are
    // no free blocks left to grow into -> OOM preemption on this very step.
    let step_res = scheduler.step().expect("step failed");
    assert_eq!(step_res.len(), 1);
    assert_eq!(step_res[0].0, 42);
    assert!(
        step_res[0].2,
        "OOM-preempted sequence must be signaled with is_eos=true, not left hanging"
    );

    // The sequence must actually be gone from the running queue (freed, not
    // silently stuck around waiting for blocks that will never come).
    assert_eq!(scheduler.running_tasks(), 0);
}

// Regression test for audit finding #6: a request whose prompt needs more
// blocks than the pool will EVER have (its total capacity) must be rejected
// immediately, and must not permanently head-of-line-block every other
// request queued behind it.
#[test]
fn test_oversized_request_rejected_without_blocking_queue() {
    let backend = Box::new(MockBackend);
    // Pool capacity: 2 blocks * block_size 4 = 8 tokens, ever.
    let mut scheduler = Scheduler::new(backend, 2).expect("Scheduler::new failed");

    let oversized_req = InferRequest {
        seq_id: 1,
        prompt_tokens: vec![0; 100], // needs 25 blocks; pool can never provide that
        sample_params: SampleParams::default(),
        max_new_tokens: 5,
    };
    let normal_req = InferRequest {
        seq_id: 2,
        prompt_tokens: vec![1, 2],
        sample_params: SampleParams::default(),
        max_new_tokens: 3,
    };

    scheduler.add_request(oversized_req);
    scheduler.add_request(normal_req);

    let step_res = scheduler.step().expect("step failed");

    // The oversized request must have been rejected with a terminal event...
    let rejected = step_res.iter().find(|(id, _, _)| *id == 1);
    assert!(rejected.is_some(), "oversized request must produce a terminal event");
    assert!(rejected.unwrap().2, "rejected request's event must be terminal (is_eos=true)");

    // ...and must not have blocked the normal request behind it in the queue:
    // it should have been admitted and processed in this very same step.
    let normal = step_res.iter().find(|(id, _, _)| *id == 2);
    assert!(normal.is_some(), "normal request behind an oversized one must still be admitted");

    assert_eq!(scheduler.waiting_tasks(), 0, "queue must not still be stuck behind the oversized request");
}

// Regression test for audit finding #14: a request for zero new tokens must
// be completed immediately (not silently produce exactly one generated
// token as the pre-fix behavior did).
#[test]
fn test_zero_max_new_tokens_rejected_upfront() {
    let backend = Box::new(MockBackend);
    let mut scheduler = Scheduler::new(backend, 16).expect("Scheduler::new failed");

    let req = InferRequest {
        seq_id: 7,
        prompt_tokens: vec![1, 2, 3],
        sample_params: SampleParams::default(),
        max_new_tokens: 0,
    };
    scheduler.add_request(req);

    let step_res = scheduler.step().expect("step failed");
    assert_eq!(step_res.len(), 1);
    assert_eq!(step_res[0].0, 7);
    assert!(step_res[0].2, "zero-token request must be completed immediately with a terminal event");
    assert_eq!(scheduler.running_tasks(), 0, "zero-token request must never enter the running queue");
}

/// A backend whose `sample()` deliberately fails for sequences using a
/// sentinel `top_k == 999` (standing in for e.g. a NaN/malformed sampling
/// configuration), while behaving normally for every other sequence. Used to
/// prove that one bad sequence's sampling failure doesn't take down every
/// other concurrent sequence's generation (audit finding #2).
struct FaultyBackend;

impl LlmBackend for FaultyBackend {
    fn load_weights(&mut self, path: &std::path::Path) -> Result<ModelMeta> {
        let mut mock = MockBackend;
        mock.load_weights(path)
    }

    fn forward_pass(&self, input: &BatchInput) -> Result<BatchOutput> {
        // Populate logits so that `sample()` is actually invoked per-sequence.
        let logits = vec![vec![0.0f32; 4]; input.seq_ids.len()];
        Ok(BatchOutput {
            seq_ids: input.seq_ids.clone(),
            next_tokens: vec![42; input.seq_ids.len()],
            logits: Some(logits),
        })
    }

    fn sample(&self, _logits: &[f32], params: &SampleParams, _token_history: &[TokenId]) -> Result<TokenId> {
        if params.top_k == 999 {
            Err(anyhow::anyhow!("simulated sampling failure (e.g. NaN logits)"))
        } else {
            Ok(42)
        }
    }

    fn kv_cache_config(&self) -> KvCacheConfig {
        MockBackend.kv_cache_config()
    }

    fn name(&self) -> &str {
        "faulty"
    }
}

#[test]
fn test_one_bad_sequence_does_not_abort_other_concurrent_sequences() {
    let backend = Box::new(FaultyBackend);
    let mut scheduler = Scheduler::new(backend, 16).expect("Scheduler::new failed");

    let good_req = InferRequest {
        seq_id: 100,
        prompt_tokens: vec![1, 2, 3],
        sample_params: SampleParams::default(), // top_k = 0, samples fine
        max_new_tokens: 5,
    };
    let bad_req = InferRequest {
        seq_id: 200,
        prompt_tokens: vec![4, 5, 6],
        sample_params: SampleParams { top_k: 999, ..SampleParams::default() }, // triggers simulated failure
        max_new_tokens: 5,
    };

    scheduler.add_request(good_req);
    scheduler.add_request(bad_req);

    // Prefill step: both get admitted and run through the (faulty) forward pass.
    let step_res = scheduler.step().expect(
        "step() must not return Err just because one sequence's sampling failed"
    );

    let good = step_res.iter().find(|(id, _, _)| *id == 100).expect("good sequence must have a result");
    let bad = step_res.iter().find(|(id, _, _)| *id == 200).expect("bad sequence must have a terminal event");

    assert!(!good.2, "good sequence must not be terminated by the other sequence's failure");
    assert!(bad.2, "bad sequence must be terminated with a terminal event after its sampling failure");

    // The good sequence must still be alive and progressing in the running queue.
    assert_eq!(scheduler.running_sequence_ids(), vec![100]);

    // A further step must keep making progress for the surviving sequence.
    let step2_res = scheduler.step().expect("subsequent step for surviving sequence must succeed");
    assert!(step2_res.iter().any(|(id, _, _)| *id == 100));
}
