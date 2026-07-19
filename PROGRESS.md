# Progress

## Current task
Task 1 DONE (audit). Awaiting user decision on revised Task 2 plan (see
"Known blockers / open questions" — the original plan assumed real branch
divergence; that assumption is false, see below).

## Completed
- Task 1 — Branch audit (read-only), 2026-07-19.

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
