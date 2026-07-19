use std::collections::{HashMap, VecDeque};
use anyhow::{Result, anyhow, Context};
use tracing::error;
use llm_core::backend::LlmBackend;
use llm_core::types::*;
use crate::block_allocator::BlockAllocator;
use crate::prefix_cache::PrefixCache;

/// Represents an active, running sequence in the scheduler.
#[derive(Debug, Clone)]
pub struct ActiveSequence {
    pub seq_id: SeqId,
    pub token_history: Vec<TokenId>,
    pub block_table: Vec<BlockId>,
    pub sample_params: SampleParams,
    pub max_new_tokens: usize,
    pub tokens_generated: usize,
    pub is_prefill: bool,
}

pub struct Scheduler {
    backend: Box<dyn LlmBackend>,
    block_allocator: BlockAllocator,
    prefix_cache: PrefixCache,
    waiting_queue: VecDeque<InferRequest>,
    running_queue: Vec<ActiveSequence>,
    block_size: usize,
}

impl Scheduler {
    pub fn new(backend: Box<dyn LlmBackend>, block_pool_size: usize) -> Result<Self> {
        let kv_config = backend.kv_cache_config();
        let block_size = kv_config.block_size;

        // `block_size` comes straight from the backend's reported KV cache
        // config and is used as a division denominator below (needed_blocks
        // computation, capacity math). A backend that reports 0 would cause
        // a panic on every step(); fail loudly at construction time instead.
        if block_size == 0 {
            return Err(anyhow!(
                "Backend reported a KV cache block_size of 0, which is invalid \
                 (would cause a division by zero in the scheduler)"
            ));
        }

        Ok(Self {
            backend,
            block_allocator: BlockAllocator::new(block_pool_size),
            prefix_cache: PrefixCache::new(64), // Max 64 recycling sequences
            waiting_queue: VecDeque::new(),
            running_queue: Vec::new(),
            block_size,
        })
    }

    /// Add a new inference request to the scheduler's waiting queue.
    pub fn add_request(&mut self, request: InferRequest) {
        self.waiting_queue.push_back(request);
    }

    /// Execute a single serving engine step.
    /// Prefills waiting requests if blocks are available, and runs decode for running requests.
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId, bool)>> {

        // Results for requests that are rejected/completed during admission
        // (oversized prompts, zero-token requests) before ever entering the
        // running queue. Collected separately so a single bad request can be
        // resolved (with a terminal event) without blocking admission of
        // everything behind it in the waiting queue.
        let mut admission_results: Vec<(SeqId, TokenId, bool)> = Vec::new();

        while let Some(req) = self.waiting_queue.front() {
            let prompt_len = req.prompt_tokens.len();
            let needed_blocks = (prompt_len + self.block_size - 1) / self.block_size;

            // A request that needs more blocks than the pool will EVER have
            // (its total capacity, not just currently-free blocks) can never
            // be admitted. Reject it immediately instead of `break`-ing the
            // admission loop forever and head-of-line-blocking every other
            // waiting request behind it.
            if needed_blocks > self.block_allocator.capacity() {
                let req = self
                    .waiting_queue
                    .pop_front()
                    .expect("waiting_queue was just observed non-empty via front()");
                error!(
                    "Rejecting request {}: prompt requires {} blocks but pool capacity is only {}",
                    req.seq_id, needed_blocks, self.block_allocator.capacity()
                );
                admission_results.push((req.seq_id, 0, true));
                continue;
            }

            // A request asking for zero new tokens has nothing to generate;
            // complete it immediately with a terminal event rather than
            // silently running one decode step (the old behavior) or
            // admitting it into the running queue at all.
            if req.max_new_tokens == 0 {
                let req = self
                    .waiting_queue
                    .pop_front()
                    .expect("waiting_queue was just observed non-empty via front()");
                admission_results.push((req.seq_id, 0, true));
                continue;
            }

            if self.block_allocator.free_blocks() >= needed_blocks {
                // `waiting_queue.front()` returned `Some` above and `&mut self` gives
                // us exclusive access with no intervening mutation, so the queue is
                // still non-empty here.
                let req = self
                    .waiting_queue
                    .pop_front()
                    .expect("waiting_queue was just observed non-empty via front()");

                // Allocate blocks
                let mut block_table = Vec::with_capacity(needed_blocks);
                for _ in 0..needed_blocks {
                    block_table.push(self.block_allocator.allocate()?);
                }

                // Check prefix cache for matching blocks (Radix reuse).
                //
                // NOTE(#7 audit finding): `match_res` (fork_seq_id / reuse_seq_id)
                // is intentionally NOT used to share physical blocks below. Doing
                // so correctly requires the block allocator/cleanup path to keep a
                // recycled sequence's block_table alive (today `step()`'s cleanup
                // unconditionally frees every block of a finished sequence, even
                // ones recycled into the prefix cache, so there is nothing left to
                // share by the time a future request could reuse it), plus mapping
                // the radix tree's token-granularity match onto this allocator's
                // block-granularity ref-counting. That is a real feature-completion
                // task, not a small bug fix, and was left out of this pass to avoid
                // rushing changes to KV-cache sharing/ref-counting correctness.
                // Today this subsystem computes correct match results but provides
                // ZERO actual KV-block-sharing benefit — every sequence still gets
                // freshly allocated blocks. Wiring this in is tracked as follow-up
                // work; do not assume prefix caching provides any speedup.
                let _match_res = self.prefix_cache.insert_sequence(req.seq_id, &req.prompt_tokens, -1);

                let active_seq = ActiveSequence {
                    seq_id: req.seq_id,
                    token_history: req.prompt_tokens.clone(),
                    block_table,
                    sample_params: req.sample_params,
                    max_new_tokens: req.max_new_tokens,
                    tokens_generated: 0,
                    is_prefill: true,
                };
                self.running_queue.push(active_seq);
            } else {
                // Not enough blocks *right now*; stop prefilling new requests to
                // avoid thrashing. Requests that can never fit were already
                // rejected above, so this can no longer permanently stall the
                // queue -- it will be retried on a subsequent step() once blocks
                // free up.
                break;
            }
        }

        if self.running_queue.is_empty() {
            return Ok(admission_results);
        }

        // 2. Construct BatchInput
        let mut seq_ids = Vec::new();
        let mut token_ids = Vec::new();
        let mut cu_seqlens = vec![0];
        let mut block_tables = Vec::new();
        let mut is_prefill = Vec::new();

        let mut curr_offset = 0;
        for seq in &self.running_queue {
            seq_ids.push(seq.seq_id);
            block_tables.push(seq.block_table.clone());
            is_prefill.push(seq.is_prefill);

            if seq.is_prefill {
                // Prefill: process all prompt tokens
                token_ids.extend_from_slice(&seq.token_history);
                curr_offset += seq.token_history.len() as u32;
            } else {
                // Decode: process only the last generated token
                let last_token = *seq.token_history.last().ok_or_else(|| anyhow!("Empty token history"))?;
                token_ids.push(last_token);
                curr_offset += 1;
            }
            cu_seqlens.push(curr_offset);
        }

        let batch_input = BatchInput {
            seq_ids,
            token_ids,
            cu_seqlens,
            block_tables,
            is_prefill,
        };

        // 3. Execute Forward Pass
        let batch_output = self.backend.forward_pass(&batch_input)
            .context("Backend forward pass failed in Scheduler")?;

        // 4. Post-process and update sequence states
        let mut step_results = admission_results;
        let mut finished_seqs = Vec::new();

        // Index running_queue by seq_id once up front instead of doing an
        // O(n) linear `iter_mut().find()` scan per sequence per step (which
        // made this loop O(n^2) in the batch size).
        let mut index: HashMap<SeqId, usize> = HashMap::with_capacity(self.running_queue.len());
        for (idx, seq) in self.running_queue.iter().enumerate() {
            index.insert(seq.seq_id, idx);
        }

        for (i, seq_id) in batch_output.seq_ids.iter().enumerate() {
            // Find the active sequence in our running queue
            let Some(&pos) = index.get(seq_id) else { continue };
            let seq = &mut self.running_queue[pos];

            // Sampling (or any other per-sequence post-processing) failing
            // for ONE sequence must never take down every other concurrent
            // request being served in this batch. Isolate the failure to
            // just this sequence: log it, send it a terminal event, abort
            // only it, and keep processing the rest of the batch normally.
            let next_token = if let Some(ref logits_vec) = batch_output.logits {
                match self.backend.sample(&logits_vec[i], &seq.sample_params, &seq.token_history) {
                    Ok(token) => token,
                    Err(e) => {
                        error!(
                            "Sampling failed for seq {} (isolating failure to this sequence only): {:?}",
                            seq_id, e
                        );
                        step_results.push((*seq_id, 0, true));
                        finished_seqs.push(*seq_id);
                        continue;
                    }
                }
            } else {
                batch_output.next_tokens[i]
            };
            // Append next token to history
            seq.token_history.push(next_token);
            seq.tokens_generated += 1;

            let is_eos = next_token == self.backend.eos_token_id();
            let reached_max = seq.tokens_generated >= seq.max_new_tokens;

            if is_eos || reached_max {
                step_results.push((*seq_id, next_token, true));
                finished_seqs.push(*seq_id);
            } else {
                // Transition from prefill to decode
                seq.is_prefill = false;

                // Allocate a new KV block if we exceeded the capacity of our current block table
                let total_tokens = seq.token_history.len();
                let capacity = seq.block_table.len() * self.block_size;
                if total_tokens >= capacity {
                    if self.block_allocator.free_blocks() > 0 {
                        let new_block = self.block_allocator.allocate()?;
                        seq.block_table.push(new_block);
                        step_results.push((*seq_id, next_token, false));
                    } else {
                        // Out of memory: preempt this sequence (abort it for reference
                        // simplicity). The client must still be told generation has
                        // ended -- send this token's event with is_eos=true (rather
                        // than silently dropping the final signal) so no caller is
                        // left waiting forever for a token that will never arrive.
                        step_results.push((*seq_id, next_token, true));
                        finished_seqs.push(*seq_id);
                    }
                } else {
                    step_results.push((*seq_id, next_token, false));
                }
            }
        }

        // 5. Clean up finished sequences and free their blocks
        for seq_id in finished_seqs {
            if let Some(pos) = self.running_queue.iter().position(|s| s.seq_id == seq_id) {
                let seq = self.running_queue.remove(pos);
                for block in seq.block_table {
                    self.block_allocator.free(block)?;
                }
                self.prefix_cache.recycle_sequence(seq_id);
                self.backend.clear_sequence(seq_id);
            }
        }

        Ok(step_results)
    }

    /// Abort all currently running sequences, freeing their allocated blocks.
    pub fn abort_all_running(&mut self) {
        for seq in self.running_queue.drain(..) {
            for block in seq.block_table {
                let _ = self.block_allocator.free(block);
            }
            self.prefix_cache.recycle_sequence(seq.seq_id);
        }
    }

    pub fn running_sequence_ids(&self) -> Vec<SeqId> {
        self.running_queue.iter().map(|s| s.seq_id).collect()
    }

    pub fn running_tasks(&self) -> usize {
        self.running_queue.len()
    }

    pub fn waiting_tasks(&self) -> usize {
        self.waiting_queue.len()
    }
}
