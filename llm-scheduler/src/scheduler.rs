use std::collections::VecDeque;
use anyhow::{Result, anyhow, Context};
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
    pub fn new(backend: Box<dyn LlmBackend>, block_pool_size: usize) -> Self {
        let kv_config = backend.kv_cache_config();
        let block_size = kv_config.block_size;
        
        Self {
            backend,
            block_allocator: BlockAllocator::new(block_pool_size),
            prefix_cache: PrefixCache::new(64), // Max 64 recycling sequences
            waiting_queue: VecDeque::new(),
            running_queue: Vec::new(),
            block_size,
        }
    }

    /// Add a new inference request to the scheduler's waiting queue.
    pub fn add_request(&mut self, request: InferRequest) {
        self.waiting_queue.push_back(request);
    }

    /// Execute a single serving engine step.
    /// Prefills waiting requests if blocks are available, and runs decode for running requests.
    pub fn step(&mut self) -> Result<Vec<(SeqId, TokenId, bool)>> {

        while let Some(req) = self.waiting_queue.front() {
            let prompt_len = req.prompt_tokens.len();
            let needed_blocks = (prompt_len + self.block_size - 1) / self.block_size;

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

                // Check prefix cache for matching blocks (Radix reuse)
                let _match_res = self.prefix_cache.insert_sequence(req.seq_id, &req.prompt_tokens, -1);
                // In a production backend, we would reuse blocks from match_res.reuse_seq_id here.

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
                // Not enough blocks; stop prefilling new requests to avoid thrashing
                break;
            }
        }

        if self.running_queue.is_empty() {
            return Ok(Vec::new());
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
        let mut step_results = Vec::new();
        let mut finished_seqs = Vec::new();

        for (i, seq_id) in batch_output.seq_ids.iter().enumerate() {
            // Find the active sequence in our running queue
            if let Some(seq) = self.running_queue.iter_mut().find(|s| s.seq_id == *seq_id) {
                let next_token = if let Some(ref logits_vec) = batch_output.logits {
                    self.backend.sample(&logits_vec[i], &seq.sample_params, &seq.token_history)?
                } else {
                    batch_output.next_tokens[i]
                };
                // Append next token to history
                seq.token_history.push(next_token);
                seq.tokens_generated += 1;

                // `eos_token_id()` is `None` when the backend couldn't determine an
                // EOS id (see `LlmBackend::eos_token_id`'s doc comment) — in that case
                // no token can match "the" EOS id, and generation is bounded solely by
                // `max_new_tokens` below rather than silently assuming Llama's `2`.
                let is_eos = self.backend.eos_token_id() == Some(next_token);
                let reached_max = seq.tokens_generated >= seq.max_new_tokens;

                step_results.push((*seq_id, next_token, is_eos || reached_max));

                if is_eos || reached_max {
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
                        } else {
                            // Out of memory: preempt this sequence (abort it for reference simplicity)
                            finished_seqs.push(*seq_id);
                        }
                    }
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
