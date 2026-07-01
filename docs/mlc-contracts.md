# MLC-LLM Contracts & Invariants

This document captures the logical contracts, invariants, and state transitions extracted function-by-function from the C++ `mlc-llm` codebase.

## 1. `cpp/serve/engine.cc`

### `StreamBackErrorImpl(Request, Callback, finish_reason)`
- **Invariant**: When streaming back an error or aborting a request, the engine *must always* append a dummy usage JSON block `{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}` at the end. Without this, the HTTP frontend may fail to terminate the connection properly.

### `AbortRequestImpl(EngineState, ...)`
- **Invariant**: Aborting a request involves exactly these steps in order:
  1. Find and erase it from `request_states`.
  2. If it is in `running_queue`, erase it.
  3. For every state entry in the request, check if it's cached in the prefix cache. If yes, recycle the sequence lazily. If not, forcefully remove the request from the backend model and recycle the ID.
  4. If it was in the `waiting_queue`, simply erase it.
  5. Fire a stream back error.

### `EngineImpl::Create(...)`
- **Invariant**: The `Engine` creates components in a strict order:
  1. Load Model Configs.
  2. Create a Disco Session (for tensor/pipeline parallelism).
  3. Infer and finalize the `EngineConfig` (e.g., fallback hybrid prefill to chunked prefill if speculative mode is enabled).
  4. Create `PrefixCache` (Radix or None).
  5. For each model: Load params, Set Max Sequence, Create `KVCache` pool (allocates VRAM blocks).
  6. Create `DraftTokenWorkspaceManager` if speculative decoding is used.
  7. Initialize `EngineAction` array (this defines the Step pipeline).

### `EngineImpl::HandleDisaggRequest(Request)`
- **Invariant**: For disaggregated serving (splitting prefill and decode across machines), a request's input sequence is forcibly split at `kv_window_begin`. The left hand side stays in the current request for prefill, while the right hand side is queued or streamed depending on the mode.

### `EngineImpl::AddRequest(Request)`
- **Invariant**: 
  - Tokenization happens *before* a request is added to the waiting queue (`Request::FromUntokenized`).
  - If `request.generation_cfg.n > 1` (parallel generation), the engine creates `n` parallel `RequestStateEntry` objects. They share the same prompt but each gets a unique `rng_seed = base_seed + i + 1`.
  - New requests are always pushed to the back of `estate_->waiting_queue`.

### `EngineImpl::Step()`
- **Invariant**: The main scheduler loop iteratively tries every registered `EngineAction`. If any action successfully processes requests, it breaks the loop and runs `ActionStepPostProcess`. If no action processes any request, but the `running_queue` is not empty, the engine panics (violates progression invariant).

## 2. `cpp/serve/engine_state.cc`

### `EngineStateObj::GetRunningRequestStateEntries()`
- **Invariant**: A request entry is only considered "running for decode" if it is a leaf (`child_indices.empty()`), alive, and has finished all input prefill (`inputs.empty()`). This is cached to avoid O(N) recalculations on every tick.

## 3. `cpp/serve/request_state.cc`

### `RequestModelStateNode::CommitToken(SampleResult)`
- **Invariant**: When a token is committed:
  1. It is appended to `committed_tokens`.
  2. Its occurrence is logged in `appeared_token_ids` (used for repetition penalty).
  3. If a `grammar_matcher` exists, it MUST accept the token (otherwise it triggers a fatal error).

### `RequestStateEntryNode::GetDeltaRequestReturn(...)`
- **Invariant**: Generates the delta stream for the frontend, checking stop conditions in exactly this order:
  1. Any of the stop strings is matched (uses `StopStrHandler`).
  2. Any of the stop tokens appears in the generated output.
  3. The `grammar_matcher` (if present) reaches a terminated state.
  4. Generation reaches `max_tokens` (model output length limit).
  5. Sequence reaches `max_single_sequence_length` (context window limit).

## 4. `cpp/serve/prefix_cache.cc`

### `PrefixCacheImpl::InsertSequence(...)`
- **Invariant**: When matching a sequence:
  - If sliding window is enabled, re-usage is strictly limited to EXACT matches. No rolling back is permitted.
  - If sliding window is disabled, the cache greedily re-uses the *shortest* recycling sequence (to minimize the rollback penalty), and rolls back the trailing tokens if necessary.
  - If no recycling sequence can be reused, it falls back to forking from the longest matching sequence.

### `PrefixCacheImpl::RecycleSequence(seq_id, lazy)`
- **Invariant**: 
  - Sequences are lazily removed (`lazy=true`) to act as a cache. 
  - They are added to an LRU tracker. 
  - If the cache exceeds `max_num_recycling_seqs_`, the oldest sequence is forcibly freed (which calls a removal callback to free KV cache blocks).

## 5. `cpp/serve/sampler/cpu_sampler.cc`

### `SampleTopPFromProb(...)`
- **Invariant**: If `top_p == 0`, sampling falls back to `argmax` (greedy search). If `top_p == 1`, it samples uniformly without truncation. When doing Top-P sampling, it pre-filters with a cutoff threshold (`top_p / 1024`) to minimize the array size before sorting, avoiding a full vocabulary sort.

### `RenormalizeProbByTopP(...)`
- **Invariant**: Modifies probability distributions in-place. If `top_p == 1.0`, it skips processing. Otherwise, it sorts the upper partition of probabilities, masks everything below the boundary to `0.0`, and renormalizes the remaining probabilities so they sum to `1.0`.

### `BatchVerifyDraftTokensWithProbAfterTopP(...)`
- **Invariant**: Speculative decoding verification happens in parallel. If a draft token is rejected (i.e. random number `r >= p/q`), it recalculates the residual probability distribution `max(p - q, 0)` and samples a new token directly from this residual distribution.

## 6. `python/mlc_llm/interface/compile.py`

### `_apply_preproc_to_params_and_check_pipeline(...)`
- **Invariant**: Pipeline and tensor parallel settings are enforced at compilation time by injecting shard strategies and stage mapping directly into `param.attrs["preprocs"]` and `param.attrs["pipeline_stages"]`. 

### `_infer_kv_state_kind(model_type)`
- **Invariant**: The `kv_state_kind` is structurally tied to the model architecture name. `rwkv` requires `rnn_state`, `medusa` requires `none`, `qwen3_5` requires `hybrid`, and everything else defaults to `kv_cache`.

### `_get_variable_bounds(model_config)`
- **Invariant**: Dynamic shape boundaries (`seq_len`, `batch_size`, `rolling_cache_len`) are explicitly extracted from the model config and provided to TVM Relax as compile-time variable bounds.

## 7. `cpp/serve/grammar/grammar.cc` (via `xgrammar`)
*(Note: Grammar functionality is offloaded to the `xgrammar` 3rdparty library)*

### Grammar Initialization & Matching
- **Invariant**: When a request specifies a JSON schema response format, `xgrammar` is initialized in `engine.cc` via `cached_grammar_compiler_.GetCompiledGrammarForJSONSchema()`.
- **Invariant**: The grammar state is tied to `RequestModelStateNode`. On every `CommitToken()`, the grammar matcher `AcceptToken(token)` MUST return true, otherwise it is considered an illegal state violation. When the grammar state reaches termination, `GetDeltaRequestReturn` forces a `finish_reason="stop"`.
