# Progress

## Current task
Tasks 7, 8, 9 DONE. Ready for Task 10 (tag v1) pending user confirmation
on pushing/releasing (see below — local tagging/merge is done, remote
push is a separate explicit ask per session safety rules).

## Task 8 — CLI + end-to-end verification, 2026-07-19
- **Chat TUI** (`llm-cli/src/bin/chat.rs`, release build, `--features
  metal`): multi-turn conversation against SmolLM3-3B-Q4_K_M — Jinja chat
  template loaded from GGUF and rendered correctly, streaming token
  output to stdout works, produced a correct answer to "what is 2+2?"
  with proper stats (TTFT/prefill/decode t/s) printed after each turn.
  Found and fixed a real bug while testing: the compute graph's first-
  token debug dump was a bare `println!` firing on every session's first
  token, spamming stdout with ~60 lines of internal graph ops — not
  appropriate for a production chat UI. Changed to `tracing::trace!`
  gated behind `tracing::enabled!(Level::TRACE)` so it's available for
  debugging but silent by default.
- **HTTP server** (`llm-cli` main bin, OpenAI-compatible API): verified
  `/health` (200 OK), `/v1/models` (lists the loaded model), and
  `/v1/chat/completions` both non-streaming (correct JSON response,
  correct answer) and streaming (SSE).
  **Found and fixed a real bug**: the streaming SSE response never
  terminated the HTTP connection after the final token. Root cause: the
  handler built the SSE stream directly on top of a `broadcast::Receiver`
  whose sender lives for the whole server process; a `take_while`-based
  cutoff (first attempted fix) still needs to observe one more broadcast
  item before it can stop, which may never arrive once this request's
  generation is done — so the stream (and the HTTP response) hung until
  the client's own timeout, which real OpenAI-compatible clients (e.g.
  the openai-python SDK) don't have, meaning it would have hung forever.
  Fixed properly: the handler now spawns a small forwarding task that
  reads from the broadcast receiver into a dedicated `mpsc` channel and
  explicitly `break`s (dropping the sender, ending the stream) the moment
  it sees `is_eos`, also emitting an OpenAI-convention `data: [DONE]`
  sentinel first. Verified: `curl -N .../v1/chat/completions` with
  `stream:true` now completes and exits 0 (was hanging to the `-m`
  timeout, exit 28, before the fix). Existing `server_tests.rs` streaming
  test still passes (it wasn't asserting on termination, only content, so
  it hadn't caught this).
- **`llm devices`**: goal.md's CLI spec names this command but no binary
  implemented it. Added `llm-cli/src/bin/devices.rs` (prints the
  auto-detected `HardwareProfile`: backend, CPU cores, SIMD caps, RAM,
  GPU VRAM/Unified Memory). Verified output on this machine matches the
  actual hardware (Metal, 12 cores, NEON, ~19GB Unified Memory).
- All of the above re-verified against real GGUF checkpoints in
  `./models/`, not synthetic/mock data.
- Full test suite re-run clean after every fix: `cargo test --workspace
  --exclude llm-kernel --lib` (9/9), `mlc_test` (94/94),
  `integration_tests` (3/3), `server_tests` (2/2).
- Hardware verification matrix: unchanged from Task 7 (CPU+Metal ✅ on
  this machine; CUDA/x86 not verified here, user will check in on that
  hardware separately).

## Task 9 — Benchmarks, 2026-07-19
Ran `benchmark_speed` (release, `--features metal`) against both real
checkpoints on this machine (Apple Silicon, Metal backend):
- SmolLM3-3B-Q4_K_M: TTFT 242ms, prefill 82.5 tok/s, decode 6.6 tok/s
  (100 new tokens).
- Gemma-4-E2B-Q4_K_M: TTFT 262ms, prefill 68.7 tok/s, decode 39.8 tok/s
  (short completion, model stopped itself early via EOS).
- **Against goal.md's Verification Matrix — reported honestly, not
  rounded up**:
  - Quantized GEMV speed vs llama.cpp: **not measured** — no llama.cpp
    build available in this session to compare against.
  - Concurrent throughput (96 concurrent, target >150 req/s) and KV
    waste (<4%): **not measured** — would require a multi-request load
    harness against the HTTP server; out of scope for what could be
    exercised in this session. Logging as an explicit open gap, not
    silently assumed passing.
  - Numerical parity vs HuggingFace reference (tolerance 1e-3): **not
    measured** — no HF reference inference environment (transformers +
    matching checkpoint) available in this sandbox.
  - Day-zero model support: partially demonstrated — SmolLM3 (llama-style
    dense) and Gemma-4 (its own arch, GQA + QK-norm + tied embeddings)
    both auto-classified and ran correctly with zero model-specific code
    changes, which is real evidence for the auto-classification claim,
    but this is 2 architectures, not a genuinely new/unseen release.
  - Memory safety (`cargo miri`): **not run** — flagging as open; the
    unsafe-code audit in Task 7 was manual review + fixes, not a miri
    pass. `cargo miri test` on the CPU path is worth doing in a future
    session (candle's own CUDA/Metal FFI code is generally not
    miri-compatible, so this would cover `llm-core`'s own unsafe only —
    the mmap calls — which is now down to just 2 sites after Task 7's
    fixes).
  - Fault recovery (cluster pause-replicate-retry): **not exercised** —
    would need a multi-node USB/network setup; llm-cluster's code exists
    (recovery.rs) but wasn't run end-to-end here.
  These gaps are real and should not be presented as "benchmarked" —
  they're the honest state after what's feasible on one Apple Silicon
  laptop in one session.

## Real-model + cross-backend verification, 2026-07-19
Ran `llm-cli`'s `run_model` bin (release build) against the two real GGUF
checkpoints in `./models/` on this machine (Apple Silicon / macOS):
- `models/huggingfacetb-smollm/SmolLM3-3B-Q4_K_M.gguf` (dense, text-only,
  3B) — prompt "The capital of France is" → correct output
  ("The capital of France is Paris.") on BOTH the default (CPU) build and
  a `--features metal` build. **Token IDs were byte-identical across CPU
  and Metal** (`[128002, 271, 128003, 198, 791, 6864, 315, 9822, 374,
  12366, 13, 128012]`), confirming the two backends agree numerically on
  a real model, not just on unit tests.
- `models/gemma/gemma-4-E2B-it-Q4_K_M.gguf` (Gemma-4 E2B, multimodal-
  capable checkpoint, text-only prompt exercised here) — correct output
  ("The capital of France is **Paris**.") on Metal.
- `models/qwen2.5-0.5b/` only contains a `tokenizer.json`, no actual
  weight file (gguf/safetensors) — **could not be tested**, flagged as an
  incomplete model directory, not a code issue.
- Confirmed via logs that `HardwareProfile` auto-detected this machine's
  hardware correctly and with no user flag: `CPU SIMD caps: AVX2=false,
  AVX-512F=false, NEON=true` (correct for Apple Silicon) and, on the
  metal build, `GPU: 18186.25 MiB Metal Unified Memory limit ... →
  Metal.` — the dispatch decision matches CLAUDE.md's rule (one
  HardwareProfile-driven choice, not hardcoded).
- CUDA and x86_64 could not be exercised on this machine (no NVIDIA GPU,
  Apple Silicon only). Attempted a `cargo check --target
  x86_64-unknown-linux-gnu` cross-compile sanity check as a substitute for
  "does the non-macOS code path even build" — this failed for
  environmental reasons (no cross C-toolchain in this sandbox for
  `onig`/`esaxx-rs`'s build scripts, unrelated to our code) rather than a
  code defect, so it's inconclusive, not a pass or fail signal. CUDA/x86
  code paths were instead reviewed statically (see Backend dispatch audit
  above + Task 7 section below) rather than executed.

## Task 7 — Production hardening pass, 2026-07-19
Full-tree audit + fix (not just report) across llm-core, llm-scheduler,
llm-cli, llm-cluster, llm-ffi `src/` directories, done via a dedicated
hardening pass plus follow-up manual fixes:

- **unwrap() in library code**: of 58 call sites found, ~44 were inside
  `#[cfg(test)] mod tests` (left as-is, acceptable). Of the rest, fixed
  the genuine failure-path ones: KV-cache block allocation in candle.rs
  (was panicking on allocation failure inside `entry().or_insert_with()`,
  now propagates via `?`), an unknown-GGML-quant-type silent fallback in
  gguf.rs (was reinterpreting unknown types as Q4_K — silently corrupting
  weights; now `bail!`s with a clear error), a `partial_cmp().unwrap()` on
  MoE router logits in llm-cluster (NaN logits would panic; now uses
  `total_cmp`), a `SystemTime` unwrap and an SSE-serialization unwrap in
  llm-cli (now degrade gracefully / skip-and-log instead of panicking a
  request handler), a `path.parent().unwrap()` in main.rs (now a proper
  `Result` error), and 2 redundant `contains_key`+`get_mut().unwrap()`
  pairs in llm-scheduler's prefix_cache.rs (rewritten as single
  `if let Some(...)` lookups). Remaining unwraps left in place are either
  guarded by an immediately-preceding check in the same function (verified
  each one) or operate on literals that cannot fail (e.g. `CString::new("
  ").unwrap()`).
- **unsafe without SAFETY comment**: added accurate `// SAFETY:` comments
  to the two mmap creations (gguf.rs, safetensors.rs) explaining the
  standard "don't mutate the file while mapped" caveat. Found and fixed a
  **real unsoundness** in `llm-cluster/src/collective.rs`: 4 raw-pointer
  reinterpret-casts between `Vec<u8>` and `&[f32]` for network
  send/receive during all-reduce, which assumed alignment `Vec<u8>` does
  not guarantee — replaced with safe `to_le_bytes`/`from_le_bytes`
  round-trips (semantically identical, verified via full test suite +
  real-model runs afterward).
  Found and fixed a **second real unsoundness**: `gguf.rs`'s and
  `safetensors.rs`'s loaders stored tensor data as `&'static [u8]` behind
  a `pub` field (`GgufFile::tensors` / `SafeTensorsFile::tensors`),
  meaning any caller could copy a tensor's `&'static` byte slice out,
  drop the owning file, and hold a dangling reference (use-after-free).
  Fixed properly (not just flagged) by storing byte offsets instead of
  borrowed slices, and replacing the public `tensors` field with a
  `tensor(&self, name) -> Option<TensorView<'_>>` accessor that borrows
  directly from `&self`'s mmap — the borrow checker now makes the bug
  impossible to reintroduce, and the fix eliminates the `unsafe` lifetime
  cast entirely (down to just the one unavoidable `unsafe { mmap(...) }`
  per loader). Updated the one real caller (`llm-tests/tests/mlc_test.rs`)
  to the new API. All 94 tests in `mlc_test` still pass.
- **Hardcoded-dispatch grep** (bypassing HardwareProfile): none found.
  `Device::new_cuda`/`new_metal` calls and `.is_cpu()`/`.is_cuda()` checks
  in candle.rs/vision.rs/audio.rs all operate on `self.device` (already
  chosen once via `HardwareProfile::choose_device`) for dtype/algorithm
  bookkeeping, not re-routing — consistent with CLAUDE.md's rule.
- **Silent-fallback grep**: beyond the GGML-quant-type fix above, found
  and fixed two in `llm-core/src/backends/candle.rs`'s GGUF loading path:
  missing `general.architecture` metadata was silently assumed to be
  `"llama"`, and missing `tokenizer.ggml.tokens` was silently assumed to
  be vocab size 151936 (an arbitrary Qwen2-specific guess) — both are
  mandatory GGUF fields; a file missing them is malformed and should fail
  loudly rather than mis-route architecture-specific behavior (e.g.
  Gemma's tied-embedding logic) or size the lm_head against the wrong
  vocabulary. Both now `bail!` with a clear message. Verified both real
  GGUF checkpoints in `./models/` still load fine after this change (they
  have proper metadata, as any real model export would). Also tightened
  `model/config.rs`'s `hidden_act` parsing to substring-match "gelu" (was
  exact-match), matching candle.rs's own more-robust GGUF-metadata
  heuristic, so variants like "gelu_new" don't silently fall through to
  SiLU. Reviewed but left as-is: `config.rs`'s `torch_dtype` fallback to
  F16 for unrecognized values — confirmed via grep that
  `ModelMeta.weight_dtype` from this HF-config path is unconditionally
  overwritten later in candle.rs's actual GGUF loading path anyway, so
  this default is inert for the real inference path, not a live bug.
- Verified throughout: `cargo check --workspace --exclude llm-kernel` and
  `cargo test --workspace --exclude llm-kernel --lib` (9/9) and
  `cargo test --test mlc_test` (94/94) all pass clean after every round of
  fixes, plus real-model generation re-verified byte-identical afterward
  (see "Real-model + cross-backend verification" above).

## Hardware verification matrix
| Hardware | Status |
|---|---|
| CPU (ARM/Apple Silicon, NEON) | ✅ Verified — real models, correct output |
| Metal (Apple Silicon GPU) | ✅ Verified — real models, correct output, numerically matches CPU |
| CPU (x86_64, AVX2/AVX-512) | ⬜ Not verified — no x86 hardware in this session; code reviewed statically (SIMD detection in profile/mod.rs is properly `#[cfg(target_arch = "x86_64")]`-gated, defaults to `false` off-arch, no panics) |
| CUDA | ⬜ Not verified — no NVIDIA GPU in this session; code reviewed statically (candle_core::utils::cuda_is_available() gates the whole branch, nvidia-smi/CUDA-driver-API query failures fall back to CPU with a clear warning rather than crashing or guessing, LLM_FORCE_CUDA opt-in documented) |
| Vulkan | ⬜ Not implemented in this codebase at all — goal.md's Phase 5 (CubeCL/Vulkan) is the unwired `llm-kernel` crate; real inference has no Vulkan backend today (known gap, logged previously) |

## Completed
- Task 1 — Branch audit (read-only), 2026-07-19.
- Manual regression pass vs vision-stability, 2026-07-19 (found + will fix
  the Whisper audio-encoder gap; everything else confirmed clean).
- Created `v1-unified` branch from `mlx` (fast-forward, cherry-picked the
  CLAUDE.md/PROGRESS.md commits from `cpu` on top), 2026-07-19.
- **Fixed the confirmed regression**: `llm-core/src/backends/audio.rs` now
  supports BOTH the Gemma-4 Conformer encoder and vision-stability's
  Whisper-style encoder, auto-selected via `detect_architecture()` based
  on tensor names present in the loaded checkpoint (no user flag, per the
  model-agnostic principle) — `audio_encoder.*`/`audio_projector.*` naming
  -> Whisper, else Gemma-4 native `a.conv1d.*`/`a.blk.*` naming ->
  GemmaConformer. `AudioArchitecture::num_mel_bins()` (80 for Whisper, 128
  for Gemma-4) now threads through to `load_audio()`'s mel-bin count and
  `candle.rs`'s audio-tensor fallback-zero shape, replacing the old
  hardcoded-128 assumption. Also fixed a latent bug while porting: vision-
  stability's `normalize_audio_tensors` hardcoded `DType::F16` for missing
  q/k/v bias zero-fill, which would panic/mismatch on CPU (F32 compute
  dtype) — now infers dtype from whichever bias tensor is actually
  present. Added 4 unit tests (`arch_detection_tests`) covering
  architecture detection and mel-bin selection for both variants.
  `cargo check --workspace --exclude llm-kernel` and
  `cargo test --workspace --exclude llm-kernel --lib` both pass clean (9
  tests, 0 failures) on this machine (macOS/CPU backend).
  Not yet verified: an actual Whisper-derived checkpoint end-to-end
  (no such checkpoint available on this machine) — architecture
  detection and tensor-name mapping are unit-tested, but a real load+
  encode pass against a genuine Whisper mmproj file is still open.

## Known blockers / open questions
- **MAJOR REVISION TO v1-plan.md's assumption**: `cpu`, `mlx`,
  `origin/vision-stability`, and `master` are NOT divergent branches with
  conflicting struct definitions. They are all points on the SAME linear
  commit history (`git log --oneline --all --graph` shows a single line,
  no merge commits, no forks). Verified: `vision-stability` and `master`
  are both ancestors of `cpu`, and `cpu` is an ancestor of `mlx`
  (`git merge-base --is-ancestor` confirms all four). `mlx` is the tip —
  it is a strict superset of cpu/vision-stability/master. There is no
  merge to perform and no struct-conflict resolution needed; Tasks 2-6 of
  v1-plan.md (create canonical types, port cpu/mlx/vision-stability
  backends one at a time) are largely moot as originally scoped, since
  `mlx` already contains everything cpu and vision-stability have, plus a
  Metal-aware HardwareProfile on top.
  - Proposed replacement for Tasks 2-6: branch `v1-unified` directly from
    `mlx` (fast-forward, no merge conflicts expected), then go straight to
    hardening/build/test verification (original Tasks 7-9) since there is
    no type-unification or backend-porting work to do.
- `llm-kernel` (the CubeCL JIT-kernel crate matching goal.md Phase 5) is
  NOT depended on by any other crate in the workspace (`llm-core`,
  `llm-cli`, `llm-scheduler` never reference `llm_kernel`). The real
  inference path dispatches entirely through `candle-core`'s own
  CUDA/Metal/CPU device backends (see `llm-core/src/backends/candle.rs`),
  not through custom CubeCL kernels. This is a deviation from goal.md's
  architecture (Phase 5 exists as scaffolding, unwired) — flagging as a
  known gap, not silently treating llm-kernel as "done".
- Have not yet run `cargo build`/`cargo test` on any branch — Task 1 was
  audit-only per plan discipline. First build/test check is the natural
  next step once the user confirms the revised Task 2.

## Struct conflict log
- `ModelMeta`/types (`llm-core/src/types.rs`): cpu == mlx byte-for-byte.
  vision-stability and master both lack: `GgufMeta` type alias,
  `has_audio_encoder`/`audio_hidden_dim`/`audio_block_count`/
  `audio_embedding_length`/`audio_num_mel_bins` fields, and the `Q4`
  variant of the quant enum (only have `Q8`). Master additionally lacks
  `arch`/`chat_template`/`eos_token_str` fields. cpu/mlx are strictly
  ahead — nothing to reconcile, just take cpu/mlx's version.
- `LlmBackend` trait (`llm-core/src/backend.rs`): cpu == mlx. master is
  missing `set_explicit_dequantize`/`set_use_vram_embeddings` default
  methods and the audio fields in its dummy/test impls. Same conclusion:
  cpu/mlx's version is the superset, nothing to merge.
- `llm-core/src/backends/{vision,audio}.rs`: cpu/mlx are materially more
  correct/robust than vision-stability's older versions (dynamic dtype
  resolution — F32 on CPU for accuracy vs hardcoded F16, optional-bias
  handling, more general positional-embedding logic, GGUF value-type
  coverage). vision-stability's code here predates and was superseded by
  cpu's own vision/audio work — not something to port forward.

## Manual regression pass (cpu/mlx vs vision-stability), 2026-07-19
Per user request, did a deep line-by-line diff review (via subagent) beyond
the Task 1 skim, specifically hunting for functionality present in
vision-stability that cpu/mlx silently dropped.

- **CONFIRMED REGRESSION**: `llm-core/src/backends/audio.rs` — vision-
  stability's Whisper-style audio encoder (Conv1d x2 → abs. pos embed →
  standard MHA transformer blocks, tensor names like
  `audio_encoder.layers.N.self_attn.{q,k,v}_proj`, defaults matching
  Whisper-large-v3: hidden_dim=1280/layers=32/heads=20) was fully replaced,
  not relocated, by a Gemma-4-style Conformer audio encoder (SubSample
  Conv2d, chunked local attention, `a.conv1d.N.*`/`a.blk.N.*` tensor names,
  defaults 1024/12/8). Confirmed via full-tree grep: no remnant of the old
  Whisper naming/architecture exists in cpu or mlx (mlx == cpu here,
  byte-identical). **Any checkpoint with a Whisper-derived audio encoder
  that vision-stability could load will fail or produce garbage on
  cpu/mlx.** This is a real loss of model-family support that needs an
  explicit decision (see below), not something to silently accept.
- Minor/unconfirmed: in `candle.rs`, `splice_visual_embeddings(...)` calls
  now pass hardcoded `vision_start_id=0, vision_end_id=0` instead of
  vision-stability's Qwen2-VL-specific `151652/151653`. Likely mitigated
  by a new "longest-run" placeholder-detection heuristic in
  `multimodal.rs` used as the primary strategy, with marker IDs as
  fallback only — but for small images where the placeholder run is under
  the heuristic's threshold (~16 tokens), behavior could differ from
  vision-stability. Flagged, not yet verified either way.
- Everything else checked (candle.rs core forward pass, vision.rs,
  gguf.rs, sampler.rs, profile/mod.rs, graph/builder.rs, graph/ops.rs,
  tokenizer.rs, model/config.rs, backends/mod.rs, cli bins,
  llm-cluster/profiler.rs) came back as genuine supersets — relocated
  intact into attention.rs/weights.rs/multimodal.rs, or strictly more
  defensive (removed unwraps, added Result-returning error paths, added
  RAM-OOM guards). No further deletions found.

## Backend dispatch audit
- `cpu` branch: `candle.rs`'s `Backend::new()` (approx) picks device via a
  hardcoded chain: `candle_core::utils::cuda_is_available()` → else try
  `Device::new_metal(0)` → else `Device::Cpu`. This does NOT go through
  `HardwareProfile` — it's a local ad hoc fallback chain. Violates the
  "single HardwareProfile dispatch point" rule in CLAUDE.md.
- `mlx` branch: same call site rewritten to estimate model size on disk,
  then call `HardwareProfile::get().choose_device(estimated_bytes)`,
  matching on `BackendChoice::{Cuda,Metal,Cpu}` — this IS the correct
  single-dispatch-point pattern the plan calls for. `HardwareProfile` in
  `llm-core/src/profile/mod.rs` was extended on mlx with `BackendChoice::
  Metal`, `query_metal_vram()` (via the `metal` crate, macOS-only,
  cfg-gated), and VRAM/unified-memory budget checks that apply to both
  Cuda and Metal.
- `vision-stability` / `master`: same profile/mod.rs (108 lines, pre-Metal
  support) and same hardcoded cuda→metal→cpu fallback as cpu — behind
  mlx on hardware dispatch.
- Conclusion: `mlx` is the one branch with a real HardwareProfile-driven
  dispatch point. Adopting it as the base for v1-unified satisfies
  CLAUDE.md's dispatch rule immediately rather than requiring new work.
