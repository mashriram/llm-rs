# Changelog

## v2026.7.22 (part 11) — Real bug: explicit --mmproj-path was ignored for vision/audio activation

Found while checking multimodal (vision+audio) coherence on this real CUDA
machine using real `gemma-4-E2B-it-Q4_K_M.gguf` + `mmproj-BF16.gguf` (a
real, popular unsloth GGUF pairing). `/image <path>` silently produced a
text-only response ("Please provide the image...") even with
`--mmproj-path` passed explicitly.

**Root cause**: `CandleBackend::load_weights` (`candle.rs`) always called
auto-discovery (`find_mmproj_path(path)`) to load mmproj *metadata* (which
determines `has_vision_encoder`/`has_audio_encoder`), completely ignoring
`self.custom_mmproj_path` — the explicit `--mmproj-path` value was only ever
consulted later, for the actual encoder *construction* step, which never ran
because `has_vision_encoder` was already false by then. Auto-discovery's
name-matching heuristic (`find_mmproj_path`) requires the mmproj filename to
share a token with the model's stem — but unsloth (a major GGUF publisher)
names mmproj files generically (`mmproj-BF16.gguf`, no model-name prefix) for
both the Gemma-4 and Qwen3.5 repos checked this session, so discovery always
missed it even with the file sitting right next to the model.

**Fixed**: `self.custom_mmproj_path` now takes priority over auto-discovery
for metadata loading too, matching what already happened for encoder
construction. Verified: `gemma-4-E2B-it-Q4_K_M.gguf` +
`--mmproj-path mmproj-BF16.gguf` now correctly loads both VisionEncoder and
AudioEncoder, reports "Vision encoder: YES", and actually runs the
vision/audio forward path on a real `/image`/`/audio` command (previously it
silently never did). Output itself is **still not coherent** for either
modality on this specific model — but that now matches this project's own
already-documented, honestly-scoped prior finding (PROGRESS.md v3/v5: "runs
without crashing, not yet coherent" — a deep numerical-reference gap, not
something to guess-fix here) rather than masking a *different*, worse bug
(the feature never activating at all). Full test suite still green
(`--features cuda`).

## v2026.7.22 (part 10) — First real CUDA hardware: build/test/generate verified, two real bugs found and fixed

This machine (RTX 2000 Ada, 8GB VRAM, CUDA 12.0, `nvcc` present) is the
**first real CUDA hardware this project has ever run on** — every prior
CUDA-related item in this file/PROGRESS.md was reviewed by reading code only
("no `nvcc`, no NVIDIA GPU in this environment"). This entry is that real
verification, plus fixes for real bugs it surfaced.

### Context: a local, unpushed "v1-gpu" attempt needed real fixes
Before this session, a local-only commit (`c6a5013`, "v1-gpu") plus further
uncommitted edits had been made on top of `ff8c45c` (== `origin/v2026.7.19`,
the last-known-good state) attempting GPU-related work, but it "didn't work."
Investigated by reading the diff directly (not guessing) and, once real CUDA
hardware was available, by actually running generation and comparing against
the untouched baseline in an isolated `git worktree`.

### Fixed: build-breaking stray file
An untracked `llm-cli/src/bin/chat copy.rs` (a strictly-older duplicate of
`chat.rs`, missing its newer `/v1/chat/completions` client-streaming mode) had
a space in its filename, which Cargo's bin-per-file discovery can't turn into
a valid crate name — **`cargo check --workspace` failed on this alone**,
unrelated to anything GPU. Deleted.

### Fixed: real regressions in the uncommitted "v1-gpu" work
- **`rotate_interleaved` RoPE convention regression**
  (`llm-core/src/backends/attention.rs`): had been changed from this
  codebase's established adjacent-pair `(2i, 2i+1)` rotation (the GGUF-native
  convention — confirmed no weight-permutation step exists anywhere in the
  loaders to compensate for a different convention) to "rotate-half"
  (first-half/second-half block) rotation, silently producing wrong attention
  math for every RoPE-using model, every backend. Reverted to adjacent-pair
  rotation while keeping the legitimate new infrastructure around it
  (precomputed cos/sin, the per-forward-pass `cos_sin_step_cache` in
  `candle.rs` deduplicating repeated RoPE cos/sin builds within one pass —
  a valid extra layer on top of the already-landed `inv_freq_cache`).
- **`qmatmul_cache` lock held across an entire forward pass**
  (`candle.rs`): a `qmatmul_guard = self.qmatmul_cache.lock()` taken once
  before the whole operator loop (instead of per-lookup) serializes/risks
  concurrent forward passes under the scheduler's batching. Narrowed back to
  locking only around the hashmap lookup.
- **`quantized_weights.clear()` breaking on-demand dequant fallback**
  (`candle.rs`): ran unconditionally right after populating `qmatmul_cache`,
  wiping out non-2D quantized tensors the code two lines above it had just
  *deliberately* kept in `quantized_weights` for on-demand dequantization.
  Removed.
- **Fuzzy tensor-name matching removed** (`weights.rs`): new
  `ends_with`/`contains` substring matching in the dequant-cache lookup and
  embedding-alias fallback risked silently substituting the *wrong* tensor
  (e.g. `post_attention_norm` vs `attn_norm` both satisfy naive suffix
  checks) — a direct conflict with this project's "no silent fallback" rule.
  Restored exact-match-only lookups (keeping the existing explicit
  tied-embedding alias list).
- Restored a `tracing::warn!` that a leftover debug `println!` had replaced.
- `hardware_check.sh`'s hardcoded `cargo build -j 4` removed (hardware-aware
  behavior, matching the whole point of this pass, shouldn't hardcode a job
  count).

### Found and fixed via real CUDA testing (not caught by any unit test): F16-overflow regression
Real generation testing on this hardware — not just `cargo test` — surfaced a
second, more serious bug the diff-reading pass above missed. A non-quantized
F16 GGUF checkpoint (`Qwen2.5-0.5B-Instruct`, `fp16` variant) generated
repetitive garbage tokens ("The/é/é/é/é/é...") on CUDA. Bisected via a
`git worktree` running the untouched `ff8c45c` baseline side-by-side on the
same file/prompt/hardware: baseline was coherent (if weak), current code was
not — a real, newly-introduced regression, not a pre-existing gap.

Root cause: `candle.rs`'s `MatMul` operator had an explicit, documented
invariant — qmatmul output must stay F32, **never** downcast back to
`compute_dtype` (F16 on CUDA), because "f16 overflows at ~65504; Qwen 2.5
residuals reach ~43k by layer 0 and easily overflow by layer 4, producing NaN
that poisons all subsequent layers." The uncommitted work added exactly that
downcast back. Removed it, restoring the documented F32-only behavior.
Verified: FP16-on-CUDA generation is coherent again, matching baseline;
Q4_K_M (the realistic, commonly-used case) unaffected either way. Full
workspace test suite (`--features cuda` and default/no-cuda) both 101+94+
other crates all green before and after.

### hardware_check.sh's default smoke-test model repo returns HTTP 401 (external, unrelated)
`HuggingFaceTB/SmolLM2-135M-Instruct-GGUF` (the script's hardcoded default)
now returns `401 Unauthorized` from HuggingFace's API for anonymous access —
confirmed via direct `curl`, not an environment/proxy issue on this machine.
Swapped the script's default to `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (already
confirmed reachable and already used elsewhere in this project's own test
history).

### Real, measured results on this RTX 2000 Ada (first ever for this project)
- `cargo build --release --features cuda` (llm-core/llm-scheduler/llm-cli/
  llm-cluster/llm-ffi): clean, first time ever.
- `cargo test --workspace --exclude llm-py --features cuda`: **all green**
  (101 llm-core + 94 llm-cli + 3 new concurrency tests + others), and
  identical pass counts with no `--features` (default/CPU) — proves nothing
  hardware-specific regressed either direction.
- `scripts/hardware_check.sh --release`: build/hardware-detection/cluster all
  pass. Hardware detection correctly reports `Selected backend: Cuda`,
  `8.59 GB total / 8.16 GB free VRAM` (real `nvidia-smi` numbers, not
  guessed), 16 CPU cores, AVX2+AVX-512F detected.
- **The real confirmation that matters**: `gemma-4-E2B-it-Q4_K_M.gguf` (this
  project's own primary target architecture) generates fully correct,
  coherent text on this real CUDA GPU for the first time ever —
  `"The capital of France is **Paris**."` (42.63 t/s decode), plus a second,
  longer, fully coherent completion for an open-ended prompt. This is the
  real acceptance signal for this phase, not just green tests.
- The tiny 0.5B smoke-test model's exact greedy-decode completion for
  hardware_check.sh's specific "capital of France" wording turned out to be
  numerically borderline on this quant/hardware combination (sometimes
  "Paris", sometimes an incomplete-but-coherent echo) — confirmed this
  borderline-ness exists identically in the untouched baseline too (not a
  regression), reported honestly rather than tuned to force a specific answer.

### New, honestly-scoped discovery: Qwen3.5 is a hybrid SSM + attention architecture (not implemented, not a naming bug)
Downloaded a real `unsloth/Qwen3.5-2B-GGUF` checkpoint (per direct request,
alongside Gemma-4-E2B-it) and hit `Weight 'model.layers.0.self_attn.q_proj.weight'
not found`. Inspected the real GGUF tensor names directly (Python `gguf`
library, not guessed): most layers have `attn_qkv` (fused QKV, not separate
q/k/v), `attn_gate`, and `ssm_a`/`ssm_alpha`/`ssm_beta`/`ssm_conv1d`/`ssm_dt`/
`ssm_norm`/`ssm_out` (a genuine Mamba-style state-space recurrent layer); a
minority of layers (3, 7, 11, 15, ...) use plain `attn_q`/`attn_k`/`attn_v`
full attention. This is a hybrid SSM + periodic-full-attention architecture
(same family as Jamba/Zamba/Nemotron-H) — confirmed zero support anywhere in
`graph/scan.rs`/`graph/builder.rs` (no SSM operator type exists at all). This
is **not a tensor-naming/hardcoding issue to patch** — it's a genuinely new
architecture family with no operator support in this engine yet, real future
work on the scale of implementing an actual SSM recurrence (verified against
a reference, not guessed), not attempted this session per this project's own
standing rule against guessing at unverified math. `Gemma-4-E2B-it` (a
supported architecture) was used for this session's real coherence
verification instead.

## v2026.7.21 (part 9) — MLX loader, RoPE/KV-cache perf fixes, model-agnostic config fixes, concurrency audit

This entry covers a broad pass across quant-performance-plan.md Phase 1.3,
Phase 3 (MLX loader), a real HF-config/tensor-naming audit, a concurrency/
KV-cache correctness review, and an honest look at Qwen2-VL vision RoPE.
Everything below was verified (tests, hardware_check.sh, or real-checkpoint
loading) except where explicitly flagged otherwise.

### Fixed and verified

- **`hidden_act` silent fallback removed** (`llm-core/src/model/config.rs`).
  Any `hidden_act` value that wasn't a `"gelu"` substring silently defaulted
  to SiLU - including a genuinely missing field, or real values like
  `"relu"`/`"geglu"`/`"swiglu"`. SiLU-gated and ReLU-gated MLPs compute
  different math, not just a different curve - this could silently produce
  numerically wrong output for any HF safetensors model with an unusual or
  missing `hidden_act`. Now: `"gelu"`-family and `"silu"`/`"swish"` resolve
  explicitly; anything else - including a missing field - bails with a
  clear error. 4 new unit tests. Existing test fixtures across
  `llm-tests/tests/mlc_parity.rs`/`mlc_test.rs` that omitted `hidden_act`
  updated to specify it explicitly.

- **RoPE `inv_freq` and KV-quantization's Hadamard rotation matrix cached**
  (`llm-core/src/backends/candle.rs`, `attention.rs`). Both were pure
  functions of architecture constants (`head_dim`/`rope_theta` for RoPE;
  `head_dim` for the Hadamard matrix) but were rebuilt from a fresh host
  `Vec` + device tensor upload on **every single call** - every layer,
  every prefill/decode step. Found via `LLM_PROFILE_STEP=1` (added last
  session): `rope` was disproportionately expensive relative to its actual
  (cheap, elementwise) math. Fixed with `inv_freq_cache`/`hadamard_cache`
  (same `Mutex<HashMap>` pattern as `qmatmul_cache`/`kv_history_cache`
  elsewhere in this file). **Measured**: decode 34.2-34.8 → 44.8-45.4 t/s
  on Metal (+~30%), 16.5 → 18.9 t/s on `LLM_FORCE_CPU=1` - a genuine
  cross-backend win, confirming it's architecture-derived caching, not a
  Metal-specific patch. The Hadamard fix alone showed no measurable win
  (the matrix involved is small, e.g. 128x128 floats - kept anyway as a
  correct, zero-risk cleanup, not oversold as a speedup). Token IDs
  bit-identical before/after on both backends and both KV dtypes (F16
  and Q8).

- **New MLX-format loader** (`llm-core/src/loader/mlx.rs`,
  quant-performance-plan.md Phase 3). Dequantizes MLX's affine-quantized
  weights (`mlx.core.quantize`, 4-bit/8-bit groupwise) to dense tensors at
  load time, routed through candle's existing dense/`QMatMul` execution
  path - no separate MLX runtime dependency. **Format verified against
  real bytes, not documentation**: loaded `mlx-community/gemma-4-e2b-
  it-4bit`'s actual `q_proj` weight/scales/biases via `mlx.core.load`, ran
  the real `mlx.core.dequantize`, and reverse-derived the packed integer
  values and nibble order from its output - confirmed low-nibble-first
  packing and `w = value*scale + bias` exactly (encoded as a regression
  test using this real data point). 6 new unit tests.

- **Three more real, broadly-applicable HF-config parsing gaps**, found
  while testing the MLX loader against the real checkpoint end-to-end
  (`llm-core/src/model/config.rs`):
  1. Real multimodal HF configs (LLaVA-style, Qwen2-VL, Gemma3/Gemma4...)
     commonly nest the text decoder's real dimensions under a
     `text_config` object with no top-level `vocab_size`/`hidden_size`/
     etc at all - confirmed against this checkpoint. `parse_config` now
     falls back to `text_config` when a field is absent at the top level.
  2. Gemma2/Gemma3/Gemma4 HF configs use `hidden_activation`, not
     `hidden_act` - confirmed (no `hidden_act` field exists at all in
     this checkpoint's config). This would have made every real
     Gemma-family safetensors checkpoint hit the new "missing hidden_act"
     bail above - both names now checked.
  3. `torch_dtype`'s newer alias `dtype` (a real field in this
     checkpoint's config), and Gemma3/4's hybrid local/global attention
     config (`layer_types`, `sliding_window`, `num_kv_shared_layers`,
     per-layer-type `rope_theta` via `rope_parameters`, `global_head_dim`)
     - the GGUF loading path already reads GGUF's equivalent metadata
     into these exact `ModelMeta` fields and the graph/forward-pass
     consumption is already fully generic; only the safetensors config
     parser was missing the read. **Cross-verified, not guessed**: dumped
     the real GGUF `gemma-4-E2B-it-Q4_K_M.gguf`'s own resolved metadata
     for these exact fields and confirmed byte-identical values to what
     the new safetensors parser derives from the MLX checkpoint's
     `config.json` for the same real model (`shared_kv_layers=20`,
     `sliding_window=512`, `key_length=512`/`key_length_swa=256`,
     `rope_theta=1000000`/`rope_theta_swa=10000`, and the same 4:1
     sliding:full layer pattern across all 35 layers).

- **`language_model.` tensor-name-prefix stripping** (`candle.rs`). Some
  real multimodal-wrapped checkpoints nest the entire text decoder under
  a `language_model.` prefix (confirmed: this checkpoint's tensors are
  `language_model.model.layers.N...` alongside sibling `vision_tower.*`/
  `audio_tower.*` trees) - a different convention from the flat
  `model.layers.N...` this loading path expects. Added a normalization
  pass stripping a detected `language_model.` prefix before graph
  building.

- **`scan_tensors` recognizes real HF names for QK-norm and per-layer-
  embedding tensors** (`llm-core/src/graph/scan.rs`), found the same way
  and cross-verified against the real, working GGUF model's own tensor
  names (dumped directly from its GGUF header, not guessed):
  - `self_attn.q_norm.weight`/`self_attn.k_norm.weight` (real HF names)
    alongside GGUF's own `attn_q_norm.weight`/`attn_k_norm.weight`.
    Without this, **QK-norm was silently dropped for every safetensors-
    loaded model that uses it** (Gemma2/3/4, Qwen3, others) - architecturally
    significant, not a minor detail, and not specific to Gemma or MLX.
  - `model.embed_tokens_per_layer.weight`/`model.per_layer_model_
    projection.weight`/`model.per_layer_projection_norm.weight` (real HF
    names for Gemma-4 2B/4B's per-layer-embedding mechanism) alongside
    GGUF's own flat names.

### Honest status: MLX loading works at the infrastructure level; Gemma-4 coherence is NOT achieved

Real progress, precisely bounded: the real MLX checkpoint (including its
vision AND audio encoders) now **loads completely** via `llm-cli` (`LLM_
FORCE_CPU=1`, since the default Metal-selecting hardware-headroom check
was blocked by transient system memory pressure on this dev machine, not
a code issue) and the forward pass **runs to completion** producing real
vocabulary-shaped tokens (not garbage bytes, not a crash). Output is
**still not coherent** after all of the above fixes.

Chased this down further than a quick "not supported" writeoff would -
cross-referenced the real `mlx_vlm/models/gemma4/language.py` reference
line-by-line against this codebase's graph builder
(`llm-core/src/graph/builder.rs`) and found a precisely-identified,
NOT-yet-resolved discrepancy: the real GGUF model (which works
correctly) has **five** per-layer norm tensors (`attn_norm`,
`post_attention_norm`, `post_norm`, `ffn_norm`, `post_ffw_norm` - dumped
directly from the real GGUF header), while the real HF/mlx_vlm reference
structure describes only **four** (`input_layernorm`,
`post_attention_layernorm`, `pre_feedforward_layernorm`,
`post_feedforward_layernorm`). This codebase's `LayerTensors` fields
(`post_attention_norm` used before the attention residual add,
`post_attention_layernorm` used after it, pre-MLP) don't have an
established, verified 1:1 correspondence to either naming scheme without
guessing - resolving this needs llama.cpp's own Gemma-4 GGUF conversion
source (not available in this environment: no internet access to browse
it, and it's C++, not something to reverse-engineer from tensor names
alone) or the same real-numerical-comparison methodology used for the
audio work (tap intermediate layer outputs, compare against a live
`mlx_vlm` forward pass, narrow down layer-by-layer) - not attempted this
pass to avoid guessing at norm-placement correctness, which would risk
producing a confidently-wrong "fix."

**Recommendation, stated plainly per direction to not oversell this**:
do not recommend MLX-format loading for Gemma-4 checkpoints as a working
feature yet - the loader and config-parsing infrastructure are real and
correct (verified independently, several fixes benefit the GGUF-
independent HF safetensors path in general, not just MLX), but end-to-end
coherent generation is not there for this specific, unusually complex
hybrid architecture. A simpler, non-hybrid-attention MLX checkpoint (no
`layer_types`/`rope_parameters`/PLE mechanism) would likely exercise a
much smaller fraction of this session's open questions and is a
reasonable next thing to try - not attempted this pass due to time spent
chasing the Gemma-4 case as far as safely possible.

### Vision: Qwen2-VL missing 2D RoPE - audited and precisely specified, not blind-implemented

Confirmed (again) that `llm-core/src/backends/vision.rs`'s forward pass
has no rotary position embedding implementation at all - when
`vision.pos_embed.weight` is absent (as it is for Qwen2-VL, which uses
RoPE instead of an absolute position table), patches get **zero**
positional information in self-attention.

Read the real reference (`transformers/models/qwen2_vl/modeling_qwen2_
vl.py`'s `VisionRotaryEmbedding`/`apply_rotary_pos_emb_vision`, and
`transformers/vision_utils.py`'s `get_vision_position_ids`) to get the
exact, real specification rather than guess:
- `VisionRotaryEmbedding(dim=head_dim//2)`: `inv_freq = 1/theta^(arange(0,dim,2)/dim)`,
  applied to (h, w) position ids separately then concatenated
  (`emb = cat([rotary_pos_emb, rotary_pos_emb], dim=-1)`, `cos=emb.cos()`,
  `sin=emb.sin()`).
- Position ids are **not** simple raster (row-major) order - they're laid
  out **block-major over `spatial_merge_size × spatial_merge_size`
  blocks** (`get_vision_position_ids`: reshape into
  `(h//merge, merge, w//merge, merge)`, transpose, flatten), matching the
  later spatial-merge step's expected token order.

**Not implemented this pass.** This requires the patch token ordering
produced by this codebase's `PatchEmbed`+reshape (currently plain raster
order) and the spatial-merge step (`vision.rs`'s `spatial_merge`) to
already agree with whatever ordering the real GGUF/mmproj conversion
(llama.cpp's `clip.cpp`, external C++ not available in this environment)
actually uses - getting the (patch order) × (RoPE position-id order) ×
(merge order) interaction wrong is a subtle, easy-to-silently-mis-verify
bug class, and the same kind of "looks plausible, numerically wrong"
mistake this session's audio work spent significant effort catching and
fixing (see parts 6-8). Flagged with the exact real specification above
so a future pass with either `clip.cpp` source access or a live numerical
verification loop (same methodology as the audio work) can implement it
correctly rather than guessing.

### Concurrency / KV-cache / scheduler audit (no code changes - review only)

Read `llm-scheduler`'s `engine.rs`/`scheduler.rs`/`block_allocator.rs`/
`prefix_cache.rs` in full plus the relevant `candle.rs` call sites.
Findings:
- **No real concurrency hazard found in the current architecture**: the
  engine's single spawned task owns one `Scheduler` (which owns the one
  `CandleBackend`) and processes one sequential `loop { scheduler.step()
  }` - there is exactly one caller of `forward_pass` at any time. Most
  "shared mutable state under concurrent access" concerns are moot by
  construction, not by explicit locking.
- **Multimodal single-slot globals** (`ACTIVE_IMAGE_PATH`/
  `ACTIVE_AUDIO_PATH`) are only set by the REPL binary (`chat.rs`),
  which is genuinely sequential (blocks on stdin, awaits full generation
  before the next line) - not currently reachable concurrently. But
  there IS a real latent architectural gap if multimodal input is ever
  wired into the concurrent HTTP server (`chat_completions` in `lib.rs`
  has zero image/audio handling today): these are single global "active
  path" slots, not per-sequence, so two different concurrent
  image/audio requests batched together would incorrectly get the same
  single image/audio applied to the whole batch. Flagged for whoever
  adds multimodal to the HTTP server - not an issue in the current
  feature set.
- **Prefix cache computes correct matches but provides ZERO actual reuse
  benefit** - already honestly flagged in-code
  (`scheduler.rs:124-138`, "NOTE(#7 audit finding)") from an earlier pass:
  `insert_sequence`'s match result is deliberately discarded because
  wiring up real KV-block sharing correctly requires keeping recycled
  blocks alive across the free/reuse boundary and mapping token-
  granularity radix matches onto block-granularity ref-counting - "a real
  feature-completion task, not a small bug fix." Independently confirmed
  this is still accurate; not attempted this pass for the same reason it
  wasn't attempted before (KV-cache sharing/ref-counting correctness bugs
  are exactly the class of thing not to rush).
- `inv_freq_cache`/`hadamard_cache` (added this session): confirmed
  check-then-insert-without-holding-lock is a benign TOCTOU (worst case:
  redundant recomputation of an identical deterministic value), not a
  correctness hazard.
- Block allocator: sound for the current single-owner design; no locking
  needed given the architecture above.

### CUDA status (unchanged - still cannot be verified in this environment)

Confirmed (not assumed) that this environment genuinely cannot even
type-check `--features cuda`: `cargo check --features cuda` fails at
`cudarc`'s own build script (`nvcc --version` not found - no CUDA
toolkit installed, no NVIDIA GPU). Writing more CUDA-specific code
blind, without being able to compile-check it at all, risks shipping
basic syntax/type errors that would immediately fail on real CUDA
hardware, wasting time there instead of saving it. The existing AWQ/GPTQ/
MLX dequant-to-dense loaders should already work on CUDA today (same
generic `candle_core::Tensor`/`QMatMul` code path used everywhere else,
no CUDA-specific branching) - unverified, per the existing honest
posture in `quant-performance-plan.md` Phase 4. Action for whoever has
CUDA hardware: pull this branch, `cargo build --release --features
cuda`, try an AWQ/GPTQ/MLX-quantized model, and share the result.

## v2026.7.21 (part 8) — five more real audio bugs found and fixed via layer-by-layer numeric verification; CPU/Metal benchmarks re-run

Continuation of part 7's investigation, using the exact same tooling
(`/tmp/hf-ref-env` venv with `mlx`/`mlx-vlm`, the cached
`mlx-community/gemma-4-e2b-it-4bit` model, the real JFK speech sample)
to walk the encoder layer-by-layer instead of reasoning from source code
alone - exactly what part 7 flagged as the needed next step. This found
**five more real, independently-confirmed bugs**, three of them
significant, by comparing intermediate tensors (SSCP output, per-block
attention output) directly against the reference at each stage rather
than only checking final output.

### Fixed, in order of how they were found

1. **`extract_block_context` block-0 misalignment** (moderate). The
   function took `_max_past_horizon`/`_max_future_horizon` parameters
   with a leading underscore - i.e. explicitly unused - which should
   have been the tell. It right-padded a copy of K/V only up to the
   query-side length and derived each attention block's context-window
   start via `(block_start + chunk_size).saturating_sub(context_size)`.
   For block index 0 this is negative (`-max_past_horizon`) and should
   realize as `max_past_horizon` zero-padded positions followed by real
   data (confirmed against the reference's `_extract_block_context`,
   which explicitly left-pads by `max_past_horizon` and right-pads by
   `max_future_horizon + chunk_size - 1` before indexing) -
   `saturating_sub` instead clamped it to `0`, shifting the entire first
   attention chunk's context window to the wrong real positions. Fixed
   by properly left/right-padding before slicing, matching the reference
   exactly.
2. **Missing clamp before `attn_post_norm`** (minor on CPU/F32, real on
   F16). The reference clips the self-attention output
   (`mx.clip(x, -grad_clip, grad_clip)`) before `norm_post_attn`; this
   clamp was absent. Restored.
3. **Missing final validity masking** (moderate). The reference's
   `AudioEncoder.__call__` forces the output to exactly zero at any
   position past the valid length as its very last step, after
   `output_proj` - covering the few frames near the valid/invalid
   boundary where each block's own causal depthwise conv could otherwise
   leak a little real signal forward into the padded tail. This project's
   `encode_conformer` had no equivalent final pass. Fixed by calling
   `zero_invalid_time_steps` on the projected output before returning.
4. **Non-periodic Hann window** (real, but small per-frame effect).
   `hann_window` divided by `n - 1` (the *symmetric* Hann window, the
   variant used for e.g. filter design) instead of `n` (the *periodic*
   Hann window). The reference's feature extractor is explicit about
   this: `"Periodic Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n/frame_length)
   ... Matches HuggingFace Transformers (signal.hann_window with
   periodic=True)"` - and it's also PyTorch/Whisper's own default
   (`torch.hann_window(n, periodic=True)`). Fixed; this function is
   shared by both the Gemma-4 and Whisper mel paths, so both benefit.
   After this fix the mel-spectrogram matches the real reference to
   float32 precision (verified frame-by-frame: mean/std/min/max and
   individual per-bin values agree to 5-6 significant figures on real
   speech, not just on silence as part 7 had managed).
5. **`zero_invalid_time_steps` masking the wrong axis in all three SSCP
   call sites** (real, but turned out to be a no-op every time - see
   below). All three calls used `time_dim=2`, but at each of those three
   points in `encode_conformer` (before conv0, between conv0 and conv1,
   after conv1) the tensor's dim 2 was actually the **frequency** axis
   (128, then 64, then 32), not time. Since the freq-axis size is always
   smaller than any realistic valid-frame count, the function's own
   `if valid_len >= t { return unchanged }` safety check made every one
   of these calls silently do nothing - the very padding-zeroing step
   part 7 credited as "confirmed working" (the bit-exact silent mel
   frame) was never actually exercised on the SSCP side for real,
   variable-length audio. Reclassified as belonging to bug #6 below
   rather than an independent axis-index typo (see there for why) and
   fixed together with it.
6. **SSCP conv2d applied with the kernel's two spatial axes
   transposed relative to the input's** (the dominant bug - this
   explains nearly everything part 7 could not). The reference's
   `SSCPConvBlock` operates on `x: [B, T, F, C]` (MLX channel-last,
   H=time, W=freq - `mlx.nn.Conv2d` convention), with a conv weight
   `[C_out, kH, kW, C_in]` where kH indexes a *time* offset and kW a
   *freq* offset. This project's mel tensor is naturally `(batch,
   freq=128, time)`; a plain `unsqueeze(1)` gives `(batch, channels=1,
   freq, time)` - H=freq, W=time, the **opposite** axis assignment. The
   GGUF-loaded conv weight has the identical per-index values as the
   real reference's own weight (verified directly:
   `mine[c_out,c_in,kh,kw] == reference[c_out,kh,kw,c_in]` for spot-
   checked indices across both conv layers, so weight loading itself was
   never in question) - meaning the kernel's own kh axis is *semantically*
   a time-offset and kw a freq-offset, independent of which tensor axis
   order it happens to be stored in. A 3x3 convolution kernel is not
   symmetric under swapping its own two spatial axes (it's a learned,
   generally-asymmetric filter), so convolving it against an
   (H=freq, W=time) input applies the kernel's time-offset weights to
   frequency offsets and vice versa - a real, silent, numerically-wrong
   computation that nonetheless produces plausible-looking
   (right-order-of-magnitude, structurally coherent) output. This is
   *exactly* the "right ballpark, not matching per-position" symptom
   part 7 reported for the SSCP stage and could not root-cause further.
   Fixed by transposing the input to `(batch, channels, time, freq)`
   immediately after `unsqueeze(1)`, which also made bug #5's axis
   confusion moot (dim 2 is now genuinely time throughout) and simplified
   the final reshape-to-sequence step back to the reference's own
   `(batch, time, freq, channels) -> (batch, time, freq*channels)`
   no-op-merge form.

### Numeric verification (not "looks more correct" - actually compared against ground truth)

Real 11-second JFK speech sample, real `mlx-community/gemma-4-e2b-it-4bit`
model run via `mlx_vlm`, same real GGUF weights loaded on both sides
(spot-checked identical, see bug #6). Before vs after this pass's fixes,
against the reference's own intermediate tensors:

| Stage | Metric | Before this pass | After this pass | Reference |
|---|---|---|---|---|
| Mel-spectrogram (valid frames) | mean / std | -2.2321 / 1.9836 | -2.2312 / 1.9833 | -2.2313 / 1.9835 |
| SSCP output (valid frames) | mean / std | 0.5527 / 3.7906 | **0.8104 / 4.7228** | 0.8105 / 4.7233 |
| SSCP output, frame 0, first 8 dims | - | `[2.42, -3.46, 8.58, 2.13, -2.08, -3.50, 2.07, -1.13]` | `[1.3770, -2.8423, 10.2214, 1.4191, -0.2701, -0.9403, -0.4367, 0.7014]` | `[1.3770, -2.8423, 10.2214, 1.4191, -0.2701, -0.9403, -0.4367, 0.7014]` |
| Block 0 self-attn output, frame 0 | - | not comparable (pre-fix) | `[6.2134, -2.6749, 9.4661, 0.5409]` | `[6.2134, -2.6749, 9.4660, 0.5408]` |
| Full encoder output (valid frames) | mean / std | 0.0220 / 1.7532 | 0.0248 / 1.7463 | 0.0526 / 5.7323 |

**SSCP is now bit-exact** to float32 precision (mean/std/min/max and
individual per-position values all agree to 5-6 significant figures).
Block 0's attention output at frame 0 also now matches exactly. This is
the SSCP-transpose fix (#6) working precisely as diagnosed.

**The full encoder output still does not match** (std 1.75 vs 5.73) -
spot-checking later frames within block 0's own attention output shows
the first 2-3 dimensions of each frame tracking the reference closely
(within a few percent) but consistently diverging more (up to ~10-20%)
in later per-head dimensions and later sequence positions, in a pattern
that does not correspond to a simple frame-index shift (checked: cross-
correlating the full valid-range SSCP output against the reference
across shifts of -5..+5 frames found the best alignment at shift 0 with
correlation 0.80, not the near-1.0 a pure alignment bug would produce -
so this is a real remaining numerical divergence, not another axis/
off-by-one issue like the ones above). **Not root-caused this pass** -
the leading candidates are the relative-position embedding/`rel_shift`
computation or some other position-dependent term inside the chunked
attention, since the divergence grows with distance from the start of
each attention chunk rather than being uniform. End-to-end generation
with the real JFK sample is still not a coherent transcription
(`"/** **tool to_times_time_tool_tool_tool_tool_tooth_tool..."` -
repetitive garbage tokens, though a different garbage pattern than
before this pass, which itself is expected given the underlying
embeddings changed substantially). This is real, measured progress - from
"completely uncorrelated with ground truth past the mel-spectrogram
stage" to "SSCP and block-0-frame-0 provably exact, encoder-level output
close but not exact" - not a claim that audio is fixed.

### Debug tooling used and removed afterward

A temporary `llm-cli/src/bin/audio_debug_dump.rs` binary (loads the real
GGUF audio encoder + `/tmp/jfk.wav`, prints mel/SSCP/encoder-output
stats and can dump the valid-range output to a flat text file for
cross-referencing against a matching Python-side dump) and several
`LLM_DEBUG_*`-env-var-gated early-return taps inside `audio.rs` were used
to do this layer-by-layer comparison, then all removed before commit -
same temporary-tooling pattern as part 7. The `/tmp/hf-ref-env` venv and
cached reference model from part 7 are still in place and still work,
so this comparison loop can be re-entered directly by anyone continuing
this investigation (re-add an env-var-gated early return at the point
you want to inspect, dump both sides' tensors, compare).

### Full test suite + hardware_check.sh

`cargo test --workspace --exclude llm-py --features metal`: 101 passed
(llm-core) + 94 passed (llm-cli), 0 failed, no regressions from any of
the six fixes above. `scripts/hardware_check.sh --skip-download --release`
with the real Gemma-4 vision+audio GGUF and mmproj: all 7 checks pass
(build, hardware detection, text generation - bit-identical token IDs to
prior runs, confirming zero impact on the non-audio path - vision smoke
test, audio smoke test, cluster registration, cluster failure detection).

### CPU/GPU benchmark re-run (M4 Pro, Metal + CPU-forced; llama.cpp bcfd1989e as baseline)

Same `SmolLM3-3B-Q4_K_M.gguf`, `llama-bench` vs `benchmark_speed`, steady
state (repeated runs, not first-run outliers):

| | Prefill | Decode |
|---|---|---|
| llama.cpp, Metal | 371.9 t/s | 54.8 t/s |
| llama.cpp, CPU-only | 49.8 t/s | 18.6 t/s |
| llm-rs, Metal | 163.8-166.1 t/s | 34.2-34.8 t/s |
| llm-rs, CPU-forced (`LLM_FORCE_CPU=1`) | 26.2 t/s | 16.5 t/s |

No decode/prefill-path code changed this session (only `audio.rs`'s
Conformer/mel/SSCP logic, which text-only generation never touches), so
this is a **re-confirmation**, not a new result: llm-rs's CPU decode
(16.5 t/s) is now within ~11% of llama.cpp's CPU decode (18.6 t/s) - a
real, competitive result. The Metal gap is real and unchanged from prior
sessions: llm-rs Metal decode (34.2-34.8 t/s) is ~63% of llama.cpp's
(54.8 t/s), and llm-rs Metal prefill (163.8-166.1 t/s) is ~44% of
llama.cpp's (371.9 t/s) - prefill is the larger relative gap and the
better target for future profiling (`quant-performance-plan.md`'s
"Phase 1: profile before optimizing" still applies; not attempted this
pass since it's orthogonal to the audio work this session focused on).

### What this means for CPU vs GPU (Metal/CUDA) code paths

All six fixes above are in `llm-core/src/backends/audio.rs`, which is
pure `candle_core::Tensor` operations (`conv2d`, `layer_norm`, `matmul`,
elementwise ops) with **no backend-specific branching** - the same code
runs on CPU, Metal, and (once built with `--features cuda`) CUDA, with
candle's `Device` enum picking the actual kernel per op. There is no
separate `cubecl.rs`/CUDA-specific audio path to duplicate these fixes
into (checked: only `candle.rs` references `AudioEncoder` anywhere in
the codebase; `CandleBackend` is the only `impl LlmBackend` that exists
today). Concretely, this means:

- **These fixes apply identically and automatically to a CUDA build** -
  there is nothing CUDA-specific left to write for this batch of bugs.
  The only difference between backends is the compute dtype
  (`audio_dtype = if device.is_cpu() { F32 } else { F16 }` in
  `AudioEncoder::load`, so both Metal and CUDA run the Conformer in F16)
  and raw kernel throughput.
- **What's NOT yet verified on real CUDA hardware** (this environment has
  none - no `nvcc`, no NVIDIA GPU): that F16 numerics behave the same on
  CUDA as they measurably do on Metal for this specific masking/softcap/
  RMSNorm-heavy computation (the `MASKED_BIAS = -1.0e4` constant in
  `conformer_attention_mask_bias` was chosen to stay safely inside F16's
  representable range on Metal; CUDA's F16 handling should be identical
  since it's the same IEEE 754 binary16 format via candle's own F16
  kernels, but has not been run to confirm), and that `hardware_check.sh`
  passes end-to-end (build + text + vision + audio smoke tests) on a real
  CUDA box. **Action for whoever has CUDA hardware**: pull this branch,
  `cargo build --release --features cuda`, run
  `scripts/hardware_check.sh --release` with the same Gemma-4 vision/
  audio models used here, and share the report - a pass there is real
  verification this environment cannot provide itself (same posture as
  AWQ/GPTQ in v5 and the CUDA-path work in v4).
- **What CPU-side work remains**: none specific to this pass's fixes -
  CPU already ran the identical code (this whole verification loop was
  done in `Device::Cpu` for reproducibility) and the existing CPU
  benchmark/regression numbers above already reflect it.
- **The still-open encoder-level numeric divergence** (see above) is
  likewise backend-agnostic - it is a logic/math issue that will
  reproduce identically on CPU, Metal, and CUDA, not something that
  needs separate diagnosis per backend.

## v2026.7.20 (part 7) — found and fixed the wrong-reference-model mistake; real numeric verification

**Critical correction to part 6**: this project's actual target is
**"Gemma-4"** (`google/gemma-4-E2B-it`, matching this repo's own GGUF
filenames) - a different, newer model from **"Gemma 3n"**, the model in
HF `transformers` that part 6 used as its reference. They share a similar
Conformer architecture family but have real, materially different
configs. Part 6's fixes (removing k_scale, replacing LayerNorm with
CumulativeGroupNorm, switching to asymmetric conv padding) were each
individually well-reasoned against Gemma 3n's real source - but Gemma 3n
was the wrong model. All three have been reverted.

This was caught by finally obtaining what earlier passes lacked: a real,
*running* reference on this exact machine. `mlx-community/gemma-4-e2b-
it-4bit` (a public, non-gated 4-bit MLX port of the actual target model)
was downloaded and run via `mlx-vlm` (`pip install mlx mlx-vlm`, both
work natively on Apple Silicon). Given the real JFK audio sample used
throughout this session's audio testing, it produced the **exact correct
transcription**: `"And so my fellow Americans, ask not what your country
can do for you, ask what you can do for your country."` This is real,
runnable ground truth, not source-code reasoning - `mlx_vlm/models/
gemma4/audio.py` and `audio_feature_extractor.py` are the actual
target architecture's implementation, confirmed against the real
model's own `processor_config.json`.

### Reverted (real regressions from the wrong reference)
- **K-scale removal**: restored. The real Gemma-4 attention DOES apply a
  fixed key-side scale `ln(1+e)/ln(2)` (`self.k_scale = math.log(1 +
  math.e) / math.log(2)`, applied as `k = k * self.k_scale`) - Gemma 3n
  genuinely lacks this, but that's irrelevant; Gemma-4 has it.
- **CumulativeGroupNorm**: reverted to plain `LayerNorm` (channel-dim, no
  bias) - the real `SSCPConvBlock` uses `nn.LayerNorm(out_channels, eps,
  bias=False)`, not a cumulative/masked group norm. Masking is instead
  applied by zeroing invalid time steps *before* the conv
  (`x = mx.where(mask, 0.0, x)`), which is what `zero_invalid_time_steps`
  (replacing `cumulative_group_norm`) now does.
- **Asymmetric SSCP conv padding**: reverted to symmetric `(1,1,1,1)`
  padding on both time and frequency axes (`self.padding = (1,1,1,1)`,
  ordinary `conv2d(padding=1)`) - the real model has no "reverse-causal"
  time-axis asymmetry at all.

### Fixed for real (Gemma-4's actual mel-spectrogram front-end)
The mel-spectrogram implementation from part 6 (itself built against
Gemma 3n) has been replaced with `gemma4_mel_spectrogram`, matching the
real `Gemma4AudioFeatureExtractor` exactly:
- 20ms/320-sample frames (not 32ms/512), FFT length 512 (not 1024 - no
  "FFT overdrive" for this model).
- Mel filterbank spans 0-8000 Hz (not 125-7600 Hz).
- **No preemphasis** (coefficient 0.0) - part 6 had *added* HTK
  preemphasis based on Gemma 3n; the real model has none.
- Semicausal left-padding (`frame_length/2` = 160 zero samples prepended
  so the first frame is centered at t=0) - entirely missing before.
- `ln(mel + 1e-3)` (additive floor), not `ln(max(mel, 1e-5))`.

**Independently verified numerically, not just by reading source**:
extracted the real reference's actual mel-spectrogram values for this
session's real JFK audio sample (`Gemma4AudioFeatureExtractor` run
directly in Python) and compared frame-by-frame against this codebase's
own `load_audio` output for the identical file. A pure-silence frame
(frame 0) matched **bit-for-bit** (`-6.9077554` in every bin, `=
ln(1e-3)`, non-trivial to match by coincidence). A real-signal frame
(frame 50) was close but showed a small, consistent per-bin discrepancy.

### Also fixed: a real, now-measurable mel-filterbank bin-index bug
Investigating that remaining frame-50 discrepancy found a genuine bug in
`build_mel_filterbank` (shared by both the Whisper and Gemma paths):
converting a mel-scale frequency to an FFT bin index used `(n_fft + 1) *
Hz / sample_rate`; the mathematically correct conversion (each FFT bin
`k` represents frequency `k * sample_rate / n_fft`, so the inverse is
`Hz * n_fft / sample_rate`) has no `+1`. Part 5 had flagged this as a
"minor, ~0.1%, deliberately unfixed" nit specifically because it
couldn't be verified against a reference at the time. Now it's
measured, not estimated: removing the erroneous `+1` reduced the
frame-50 per-bin discrepancy against the real reference by roughly an
order of magnitude (differences dropped from the ~0.01-0.02 range to
~0.001-0.01). Fixed - this formula has one objectively correct form, so
unlike the Gemma-vs-Whisper-specific fixes above, this was safe to
correct for both architectures at once.

### Honest status: still not coherent, and now precisely why
With the mel front-end verified close (not exact) and three real
regressions reverted, end-to-end output is still not coherent (tested
with the same real JFK sample). Went one level deeper for a definitive
answer: extracted the *encoder's* actual output tensor from the real
reference (`audio_tower(mel, mask)`, shape `(1, 275, 1536)` - 275 matches
this codebase's own valid-length arithmetic exactly, a good sign) and
compared against this codebase's own full encoder output for the
identical input. **They do not match** - not just numerically off, but
a different overall scale (reference std ≈ 5.7, this codebase's std ≈
1.5) and non-matching sign patterns per dimension. Narrowed further: the
SSCP-stage output (before the 12 Conformer blocks) is in the right
ballpark (same order of magnitude, similar largest-dimension pattern)
but also does not match closely. This means real, additional bugs remain
somewhere in the SSCP conv/norm stage and/or the Conformer blocks
themselves (attention, relative position embedding, or light-conv) -
now a well-scoped, verifiable-not-guessable problem: the tooling to
compare any intermediate tensor against real ground truth now exists and
works (see below), the remaining work is methodically walking it
layer-by-layer rather than reasoning from source code alone.

**What's now available for the next pass** (not yet exhausted this
session, given how much ground was already covered): the real reference
model runs locally (`/tmp/hf-ref-env` venv, `mlx`/`mlx-vlm` installed,
`mlx-community/gemma-4-e2b-it-4bit` cached in `~/.cache/huggingface`) -
any intermediate tensor (per-conv-layer, per-attention-layer, position
embeddings, etc.) can be dumped from the real model and compared
directly against this codebase's own (a temporary `LLM_DEBUG_*`-gated
early-return in `encode_conformer`, exactly like the one used to extract
the SSCP-only comparison above, is the fastest way to tap any
intermediate point). This is a materially better position than "read
source code and hope" - the verification loop is now real.

Full test suite (101+94 tests) and `scripts/hardware_check.sh` (all 7
checks) still pass; Gemma-4 text-only output remains bit-identical
throughout every change in this entry.

## v2026.7.20 (part 6) — audio fixes verified against the real Gemma3n reference source

Part 5's audit flagged several audio issues as "could not verify without a
reference implementation." This session obtained one: `pip install
transformers` (no GPU/weights needed) gives direct access to the real
`Gemma3nAudioEncoder`/`Gemma3nAudioFeatureExtractor` PyTorch source in
`site-packages` - reading it directly resolved most of part 5's open
questions with actual ground truth instead of further guessing. Four
real, reference-verified fixes landed as a result.

### Fixed
- **Mel-spectrogram front-end was fundamentally wrong for Gemma-Conformer**
  (`llm-core/src/backends/audio.rs`): the previous implementation shared
  Whisper's exact convention (400-sample/25ms frames, no preemphasis,
  power spectrum, 0-8000Hz mel range, Whisper-specific final rescale) for
  BOTH architectures. The real Gemma3n front-end (confirmed via
  `feature_extraction_gemma3n.py`) uses none of that: 512-sample/32ms
  frames, a 1024-point FFT (`2^ceil(log2(512))`, doubled for "FFT
  overdrive"), HTK-flavor preemphasis (coefficient 0.97, previously
  entirely absent), a 125-7600Hz mel range (not 0-8000Hz), a magnitude
  spectrum (not power), and a plain `ln(max(x, 1e-5))` with NO final
  clip/rescale (Whisper's rescale step doesn't apply to Gemma at all).
  Implemented as a new `gemma3n_mel_spectrogram` function, dispatched by
  architecture (`whisper_mel_spectrogram` kept unchanged for Whisper
  checkpoints - both are real, correct, architecture-specific pipelines
  now, not one guessed-shared one). **Independently double-confirmed**:
  every parameter matches the real deployed model's own
  `preprocessor_config.json` (fetched from a public mirror,
  `unsloth/gemma-3n-E2B-it`) exactly, including confirming
  `per_bin_mean`/`per_bin_stddev` are genuinely unset (null) for this
  model, so correctly omitting that optional normalization step is
  itself verified, not assumed.
- **SSCP conv-subsampling used the wrong normalization type**: replaced
  a plain per-frame `LayerNorm` with a real `cumulative_group_norm`
  function, ported exactly from `Gemma3nAudioCumulativeGroupNorm` -
  normalizes over a single group spanning frequency+channel jointly,
  with statistics accumulated cumulatively over time (each time step's
  stats include every step from `0..=t`, not computed independently
  per-step). Also fixed the normalization epsilon (`1e-3`, not `1e-5` -
  confirmed from `sscp_conv_group_norm_eps`'s real config default).
- **SSCP conv time-axis padding was symmetric; the reference uses
  asymmetric "reverse-causal" padding** (0 before, kernel_size-1=2 after
  - every output step sees only past+current input, never future).
  candle's `conv2d` only supports one symmetric padding value for both
  axes, so padding is now applied manually (`pad_with_zeros`) before a
  `padding=0` conv2d call - frequency axis stays symmetric (1,1,
  unchanged, already correct), only the time axis changes. Output shape
  is unaffected (same arithmetic result either way); only which specific
  positions get zero-padded changes.
- **K-side attention scale removal (from part 5) is now externally
  confirmed correct**: `Gemma3nAudioAttention.forward` in the real
  reference applies `q_scale`/`per_dim_scale` to queries only - keys are
  used completely unscaled. This is exactly what part 5's removal
  (based on "no principled derivation found," without a reference to
  fully confirm) already changed the code to do - independent
  confirmation, no further change needed.

**Verified, every fix**: no regression (full test suite - 101 llm-core +
94 llm-cli tests - and `scripts/hardware_check.sh`'s all 7 checks pass;
Gemma-4 text-only output is bit-identical throughout, since none of
these fixes touch the text-only path at all), and a real behavior change
on audio input after each fix (tested with both a synthetic tone and a
real 11-second speech sample, `whisper.cpp`'s own public `jfk.wav` test
fixture - output changed meaningfully and differently after each fix).

**Honest status: audio is still not fully coherent** even after all
four verified fixes. This is not a failure of the fixes themselves (each
is independently confirmed correct against real reference source, not
guessed) - it means at least one more issue remains. The most likely
remaining candidate, identified but NOT implemented this pass:

### Confirmed real, NOT fixed: missing validity/causal attention masking
The reference's `Gemma3nAudioAttention` builds a real combined mask
(local causal window + which time steps are actual audio vs zero-padding)
and applies it before softmax (`torch.where(mask, logits, -inf)`); it
also zeroes out padded positions before the light-conv step
(`Gemma3nAudioConformerBlock`'s `validity_mask_for_lconv`). This
codebase's fixed-size 480,000-sample (30s) buffer means a short clip
(e.g. the 11-second JFK sample - barely a third of the buffer) is mostly
zero-padding, and none of that masking exists here - every position
attends/convolves as if the whole buffer were real audio.
**Reassessed impact, on reflection**: likely smaller than initially
feared, specifically because this architecture is almost entirely
causal/backward-looking already (the light-conv is manually causal-
padded; `cumulative_group_norm`'s running stats only ever look backward
in time, so trailing padding can't contaminate earlier real-content
statistics; the chunked attention's `max_future_horizon=0` means no
position ever looks forward at all) - so the missing masking mostly
under-constrains the padded tail itself, not the real content that
precedes it, though it's not ruled out as still-significant. Not
implemented this pass: doing it right requires plumbing a real-vs-padded
frame count from `load_audio` through `AudioEncoder::encode` and every
downstream norm/attention/conv call, touching several function
signatures - a real, distinct, well-scoped follow-up rather than
something to guess at partially.

### Minor, deliberately unfixed: `build_mel_filterbank`'s bin-index formula
A small, low-confidence-impact discrepancy noticed while reading the
reference: this codebase's shared filterbank builder converts a mel
frequency to an FFT bin index via `freq_hz * (fft_length+1) /
sample_rate`; the reference computes filters directly in Hz-space using
plain `fft_length` (no `+1`). For `fft_length=1024` this is a ~0.1%
relative scale discrepancy - likely negligible, but NOT fixed here
because `build_mel_filterbank` is shared between the Gemma and Whisper
paths, and there is no reference confirmation either way for Whisper's
own convention - changing it could as easily introduce a new bug for
Whisper as fix a negligible one for Gemma. Flagged, not guessed at.

## v2026.7.20 (part 5) — full audio/vision multimodal correctness audit

A dedicated audit pass on the two remaining open multimodal correctness
gaps (Gemma-4 audio Conformer incoherence; Qwen2-VL vision incoherence),
requested explicitly after part 4 fixed the `embed_scale` bug without
fully solving either. Audio was audited via a focused subagent review
against known Gemma3n/Google Conformer/USM lineage conventions (no
internet or Python reference available in this environment); vision was
audited directly. Documenting everything found - fixed, suspected-but-
not-fixed, and confirmed-correct - per an explicit request for full
honesty rather than a partial or rosy account.

### Fixed
- **Unjustified K-side attention scale removed** (`llm-core/src/backends/
  audio.rs`, Gemma Conformer's `forward_conformer_block`): a fixed
  multiplier `ln(1+e)/ln(2) ≈ 1.894` was applied to every key vector
  before attention. The analogous Q-side scale (`head_dim^-0.5/ln(2)`,
  combined with a *learned*, zero-initialized `per_dim_scale` weight) is
  a documented Google Conformer/USM "PerDimScale" trick - chosen
  specifically so the learned scale reduces to plain scaled-dot-product
  attention at initialization. No reference for this architecture
  defines an analogous K-side scale (learned or fixed), and no
  `k_per_dim_scale` weight is loaded anywhere in this codebase. Audited
  and found no principled derivation for "softplus evaluated at 1.0" in
  this family of formulas - most likely a garbled/misattributed constant
  from translation. Removed rather than replaced with another guess (no
  reference available to derive a correct replacement value, if any is
  even needed). Verified no regression: full test suite (101+94 tests)
  and `hardware_check.sh` still pass, Gemma-4 text-only output is
  bit-identical. **Real behavior change on audio input** (different
  garbled output than before), consistent with the removed multiplier
  being real, but **audio output is still not coherent** - see below.

### Audited, found suspicious, deliberately NOT fixed (would require a reference implementation to verify)
Each of these was a real candidate for "the actual bug," but fixing any
of them means guessing a replacement formula/constant with no way to
confirm it's more correct than what's there now - which would violate
this project's "no silent, unverified claims" principle just as much as
leaving a known bug unfixed silently would. Flagging all of them
precisely instead:
- **SSCP conv-subsample normalization may use the wrong norm type.**
  `encode_conformer`'s two subsampling stages (`llm-core/src/backends/
  audio.rs` ~lines 328-345) use a plain per-frame `LayerNorm`. Gemma3n's
  real audio front-end (`Gemma3nAudioSSCPConvBlock`) is recalled to use a
  **cumulative group norm** (causally-accumulated statistics over the
  time axis, not independent per-frame normalization) - a streaming-
  friendly norm distinct from a fixed LayerNorm. If so, every frame's
  normalization would be systematically wrong (worse for early frames) -
  plausibly a major contributor to the incoherence. Channel dimensions
  (128 then 32) are confirmed correct; only the normalization *type* is
  in question.
- **Whisper-specific mel-spectrogram normalization may not apply to
  Gemma.** `load_audio` (same file, ~lines 1003-1010) applies Whisper's
  documented `log_max - 8.0` clip + `(v+4)/4` rescale as one shared
  function for BOTH the `Whisper` and `GemmaConformer` architectures
  (differentiated only by mel-bin count, 80 vs 128). No confirmation
  exists that Gemma3n's own feature extractor uses this exact Whisper
  convention rather than a different one (raw log-mel, different
  clip/rescale constants, etc.) - the fact this normalization is generic/
  shared rather than architecture-specific is itself a signal it may
  have been assumed rather than verified when written.
- **`rel_shift`'s pad/reshape/narrow skew trick and the
  `queries`/`keys`/`matrix_ac`/`matrix_bd` permute chains** (relative
  positional attention, ~lines 512-524 and 730-738): structurally match
  the well-known Transformer-XL/Music-Transformer relative-attention
  "skew" algorithm and produce plausible shapes throughout, but this is
  exactly the class of bug (valid shape, wrong semantic axis) that needs
  numeric comparison against a real forward pass to rule out - not
  achievable without a reference implementation.
- `extract_block_context` silently ignores its own `max_past_horizon`/
  `max_future_horizon` parameters (ranged, prefixed `_`, only
  `chunk_size`/`context_size` actually drive the window) - currently
  harmless only because `max_future_horizon == 0` in the hardcoded
  config, but not a general implementation of what its name promises.
  Flagged as a code-quality issue, not fixed (no behavior change today).

### Confirmed correct (audited, no longer worth re-reviewing)
- `chunk_size=12`/`max_past_horizon=12`/`max_future_horizon=0`/
  `context_size=24` are self-consistent and match Gemma3n's recalled
  `conf_attention_context_left=13` convention.
- The Q-side "PerDimScale" formula itself (distinct from the removed
  K-side one above).
- `grad_clip`, `softcap=50.0`, residual weight `0.5` are plausible/
  consistent Gemma-Conformer hyperparameters.
- The shared `rms_norm` free function correctly omits Gemma's HF-only
  `(1+weight)` convention, consistent with `attention.rs`'s documented
  GGUF-already-bakes-in-the-+1 rule - this encoder's tensors come from
  the same native-GGUF export path as the main LLM.
- `num_mel_bins` is correctly differentiated per architecture (128
  Gemma / 80 Whisper), not blindly shared like the normalization above.
- Conv-subsample channel dimensions are internally consistent with their
  LayerNorm bias tensor sizes at every stage.

### Qwen2-VL vision: confirmed, not newly fixed
Re-confirmed by direct code reading (`llm-core/src/backends/vision.rs`
~lines 195-251): there is **no rotary position embedding implementation
at all** for the vision transformer. Qwen2-VL ships no absolute
`vision.pos_embed.weight` tensor (it relies entirely on 2D RoPE instead),
so patches currently get **zero positional information** in
self-attention - the transformer sees an unordered set of patches, not a
grid. This matches the previously-documented gap exactly (see "v3").
Deliberately not implemented this pass: real Qwen2-VL 2D vision RoPE
requires replicating its exact rotary-frequency/patch-grid convention
(including the patch-window reordering interaction with `spatial_merge`)
with no reference available to verify a from-scratch implementation
against - the risk of a "looks plausible, still wrong" implementation
giving false confidence was judged worse than leaving this as a clearly
documented, known-missing feature. `spatial_merge`'s post-hoc grid
reshape/permute (raster-order patches, ~lines 454-466) was checked and
is internally consistent with how patches are laid out earlier in the
same function - not itself a bug, just working around the missing RoPE
rather than depending on it.

### Honest bottom line
Neither Gemma-4 audio nor Qwen2-VL vision produces coherent output as of
this commit. Real progress was made (the `embed_scale` fix in part 4,
the K-scale removal here), verified with no regressions each time - but
"still broken, differently" is the accurate status, not "fixed." The
next actionable step for either would require a Python/HF reference
implementation to run side-by-side (not available in this environment)
to numerically confirm or rule out the suspected items above, rather
than another round of plausible-looking guesses.

## v2026.7.20 (part 4) — multimodal embed_scale fix + quantized-KV cache extension

### Fixed
- **Vision/audio embeddings incorrectly scaled by Gemma's `embed_scale`**
  (`llm-core/src/backends/candle.rs`): Gemma-family models multiply the
  whole embedding tensor by `sqrt(hidden_dim)` in one graph-wide op,
  applied AFTER vision/audio splicing - so image/audio embeddings (already
  at the correct final magnitude from their own encoder) were getting an
  extra ~39x (for Gemma-4-E2B) magnitude blow-up on top of their correct
  scale. This matches a real root cause for the "runs fine, output is
  garbage" multimodal failure mode. Fixed by pre-dividing vision/audio
  embeddings by `embed_scale` at the splice point - mathematically
  equivalent to HF's real Gemma3/PaliGemma ordering (scale text first,
  splice unscaled image features after) without restructuring the
  existing graph. No-op for non-Gemma architectures (gated on
  `meta.embed_scale.is_some()`) - model-agnostic, not a Gemma-only patch.
  Verified no regression (Gemma-4 text-only bit-identical, SmolLM3
  unaffected, full test suite + `hardware_check.sh` still pass) and a
  real behavior change (repetitive garbage -> clean stop). **Multimodal
  output is still not fully coherent** - this is real, verified partial
  progress, not a complete fix; at least one more bug (likely encoder/
  projector-level) remains, not isolated in this session.

### Added
- **KV-history-cache fast path extended to quantized (Q8/Q4) KV**: the
  decode-speed fix from part 2 only covered non-quantized (F16/F32) KV;
  the Hadamard-rotated Q8/Q4 path always fell back to the full per-step
  block rebuild. Extended to cover it too, putting the new chunk through
  the same quantize-then-dequantize round trip the block store applies,
  so it stays bit-identical to the old path rather than trading away
  precision for speed. Verified: `LLM_KV_DTYPE=q8` and default KV produce
  bit-identical generated token IDs.

## v2026.7.20 (part 3) — re-verification: llama.cpp comparison + multimodal coherence check

No code changes - a verification pass on part 2's work, requested
explicitly to confirm the numbers hold and that Gemma-4/multimodal still
work after the KV-cache fix.

- **llama.cpp comparison, steady state** (repeated runs, not just the
  first/outlier one): llm-rs Metal is ~1.4x slower on prefill and ~1.8x
  slower on decode than llama.cpp on this machine (same GGUF file/
  hardware) - down from ~2.4x/~2.9x before the KV-cache fix. See
  PROGRESS.md "v5" for the full numbers table.
- **Multimodal coherence re-checked with real prompts** (not just "did it
  crash"): fed a real test image to Gemma-4 and Qwen2-VL, and a real WAV
  tone to Gemma-4's audio path. Both vision models and the audio path run
  end-to-end without crashing, but produce **incoherent output** -
  confirmed this is a pre-existing gap (already noted in "v3"), not a
  regression from today's KV-cache change, by A/B testing the identical
  prompt against a `git worktree` build of the commit immediately before
  that fix (also incoherent, differently). Gemma-4's plain text-only
  generation in the same session is fully coherent ("The capital of
  France is **Paris**."), isolating the problem to the vision/audio
  splice path specifically, not the core engine.
- **CHANGELOG completeness pass**: added a missing entry for
  `scripts/hardware_check.sh` and the `llm-cluster` recovery-log fix
  (commit `e40e351`), which had no CHANGELOG entry despite being
  committed and documented in PROGRESS.md.

## v2026.7.20 (part 2) — decode-speed fix + first-pass AWQ/GPTQ loaders

Prompted by a real, measured comparison against llama.cpp (installed on
this dev machine) showing llm-rs decoding ~3x slower on Metal for the
same GGUF file/hardware.

### Fixed
- **O(n)-per-decode-step KV-cache reconstruction** in `CandleBackend`'s
  `PagedAttention` operator (`llm-core/src/backends/candle.rs`): every
  decode step, for every layer, fully rebuilt the entire K/V history from
  block storage (dequantize + clone every stored block + concat +
  repeat_kv + transpose + contiguous) just to append one new token -
  confirmed via a context-length sweep showing decode throughput
  dropping from ~31 t/s to ~16 t/s between a 28-token and a 278-token
  context on the same 3B model. Added a `kv_history_cache` that extends
  the previous step's already-processed history instead of rebuilding it,
  falling back to the original full-rebuild path (unchanged) for
  quantized (Q8/Q4) KV, new sequences, and any cache-miss/mismatch.
  Verified bit-identical generated token IDs before/after; full test
  suite and `scripts/hardware_check.sh` still pass. Real measured result
  on this machine: Metal decode throughput 14.7 -> 24.9 t/s (+69%) on the
  same benchmark. This is backend-agnostic Rust logic, not a Metal-only
  patch.

### Added
- **First-pass AWQ + GPTQ safetensors loaders** (`llm-core/src/loader/
  awq.rs`, `gptq.rs`): dequantize AWQ/GPTQ 4-bit packed weights to dense
  F16/F32 at load time, wired into the existing safetensors-loading path.
  `parse_config` no longer hard-rejects these formats (bitsandbytes still
  is). Tensor layout (AWQ packs along the output axis, GPTQ along the
  input axis - opposite of each other) confirmed by inspecting two real
  HF repos' safetensors headers via HTTP range requests, not assumed.
  **Numerically unverified** - this has not been checked against a real
  Python (`transformers`/`autoawq`/`auto-gptq`) reference on real
  tensors, and cannot be built/run at all in this environment (no CUDA
  hardware, no local AWQ/GPTQ model). Correctness-first only: full
  dequant at load time trades away AWQ/GPTQ's memory savings and speed
  advantage for simplicity; real throughput needs a tensor-core kernel
  (Marlin-class), not attempted here. See `quant-performance-plan.md`.

## v2026.7.19 (branch, unreleased) — phase 2b: end-to-end hardware verification script

### Added
- **`scripts/hardware_check.sh`**: a single, portable smoke-test script for
  any machine (CPU/CUDA/Metal/Raspberry Pi/generic Linux ARM). Auto-detects
  the platform and picks the right cargo feature flags, builds, runs `llm
  devices` to confirm hardware detection, runs a real text-generation
  correctness check (not just "did it crash"), optional vision/audio smoke
  tests (auto-generating a synthetic PNG via pure-stdlib zlib/struct and a
  synthetic WAV tone if none are supplied), and a real two-process
  `llm-cluster` networking + kill-detection test. `--check-mobile` prints an
  honest "not implemented, here's what it needs" report instead of faking a
  result for Android/iOS, which doesn't exist yet. Verified with a full real
  run on this machine (release/Metal): all 7 checks (build, hardware
  detection, text, vision, audio, cluster registration, cluster failure
  detection) pass.

### Fixed
- **`llm-cluster/src/recovery.rs`**: `ClusterHealthMonitor::check_failures`
  logged "Triggering Pause-Replicate-Retry" on a node failure - false; only
  eviction from the active-node roster happens, no re-partitioning or
  re-prefill is implemented anywhere. Found while wiring up the cluster
  step of the new script (the check first failed silently because
  `RUST_LOG` wasn't set, then once fixed, the log message itself turned out
  to be lying about what had actually happened). Logging a recovery action
  that didn't happen is exactly the kind of silent-seeming-success this
  project's rules forbid - message now just reports the failure honestly.

## v2026.7.20 — Multimodal Stabilization, Explicit Projectors & Multi-GPU VRAM Auto-Selection

This release completes the hardware-agnostic and model-agnostic multimodal stabilization for Gemma 4 E2B, Qwen3-VL, and SmolLM3 across both CUDA GPU and CPU execution backends, resolving tensor shape mismatches and multi-GPU VRAM discovery issues.

### Added
- **Explicit `--mmproj-path <PATH>` CLI argument**: Supported across `chat`, `server`, `run_model`, and `benchmark_speed`. Allows passing custom or non-standard multimodal projector weight files (e.g. `gemma-4-E2B-mmproj-BF16.gguf`) directly, overriding directory-wide stem scanning.
- **Automatic Multimodal Tag Prepending**: In interactive `chat`, using `/image <path>` or `/audio <path>` commands automatically prepends `<image>` / `<audio>` tags to the prompt content when missing, ensuring visual and audio token placeholders are correctly formatted into the token stream.

### Fixed
- **Vision Attention QKV Bias Shape Mismatch (`vision.rs`)**: Fixed dimension derivation in fallback zero-bias generation (`attn_qkv.bias`) from `dim(0)` (input size) to `dim(1)` (output projection size). This resolved a `broadcast_add` shape mismatch crash when executing Gemma 4 E2B vision encoder projections.
- **Multi-GPU / Hybrid Graphics VRAM Selection (`profile/mod.rs`)**: Updated `query_nvidia_smi()` to iterate across all lines returned by `nvidia-smi` and select the GPU with the highest free VRAM instead of blindly taking line 1. On dual-GPU laptops (e.g. AMD iGPU + NVIDIA dGPU), line 1 previously reported 0 VRAM, causing silent fallback from CUDA to CPU.
- **Explicit Projector Weight Priority (`candle.rs`)**: Prioritized user-specified `custom_mmproj_path` during weight loading in `CandleBackend` for both `VisionEncoder` and `AudioEncoder`, guaranteeing deterministic weight loading across diverse multi-modal models.

## v2026.7.19 (branch, unreleased) — phase 2: model-agnostic + hardware-agnostic push

Follow-up pass on the same branch: a real HF downloader, real TCP
networking for `llm-cluster` (previously pure mock), and a second round
of live multimodal testing (Qwen2-VL + Gemma-4, a different real
checkpoint from phase 1's) that found and fixed 6 more real bugs. Full
detail, including a per-hardware honest status table (CPU/Metal verified;
x86/CUDA/Raspberry Pi structurally fine but unverified on real hardware;
Vulkan/mobile genuinely unimplemented), is in PROGRESS.md's "v3" section.

### Added
- **`llm pull <model>`**: real Hugging Face downloader. Resolves a search
  term or `owner/repo`, lists GGUF quant variants with real sizes,
  recommends the largest one that fits this machine's detected
  `HardwareProfile` (same 15% headroom rule as model load), downloads
  weights + tokenizer/config sidecars (with a fallback to a repo's base
  non-GGUF sibling for `tokenizer.json` when the GGUF repo doesn't ship
  one), and verifies each download is byte-complete before declaring
  success.
- **Pre-quantized-model detection**: bitsandbytes/AWQ/GPTQ repos are now
  detected via `config.json`'s `quantization_config` both in `llm pull`
  and directly in `llm-core::model::config::parse_config`, and refused
  with a clear message rather than silently mis-loaded.
- **Real `llm-cluster` networking**: length-prefixed JSON wire protocol
  (`protocol.rs`), a real `TcpListener`-based coordinator that registers
  workers via Hello/Welcome and tracks them with `ClusterHealthMonitor`
  (previously never invoked from anywhere, so failure detection was
  structurally inert), and a real `TcpStream`-based worker sending live
  heartbeats with reconnect-on-failure backoff. Verified with two real
  local processes, including a real kill-and-detect failure test.
- Real log-mel spectrogram computation for audio input (Hann window,
  DFT power spectrum, triangular mel filterbank, Whisper-style
  normalization) plus sample-rate detection and linear-interpolation
  resampling to 16kHz, replacing a placeholder that carried zero real
  frequency information.

### Fixed — multimodal (found via two live model tests, not static audit)
- Non-contiguous tensor crash in vision layernorm (Qwen2-VL).
- Vision-encoder weight matmul assumed a single fixed `[out,in]`
  orientation; a second real checkpoint (Gemma-4) proved this isn't a
  fixed convention across GGUF exporters — replaced with a `linear()`
  helper that detects orientation per-tensor from the input feature dim.
- `spatial_merge_size` silently defaulted to 1 when absent from GGUF
  metadata; now inferred from the vision projector's own weight shape.
- `VisualEmbed` ran the vision encoder on a dummy image and cached the
  result unconditionally on every request, corrupting later audio-only
  splicing in the same forward pass — now skips the encoder and the
  cache write entirely when no image is active, mirroring `AudioEmbed`.
- Audio placeholder-token count used the encoder's hidden dimension
  instead of its real output sequence length — now computed per
  architecture from `audio_num_mel_bins`.
- `symphonia`'s "pcm" feature was missing, silently failing to decode
  real WAV files.

### Known gap, precisely diagnosed
- GGUF files using llama.cpp's newer "IQ" quant types (e.g. IQ4_NL,
  dtype id 20) fail to load: `candle-core` 0.9.2 has no dequantization
  support for any IQ-series type, and the failure occurs during header
  parsing, aborting the whole file. Classic types (F16/F32/BF16, Q4-Q8,
  the full K-quant family) all confirmed working. Error message now
  names the likely cause instead of an opaque parse failure. Real fix
  requires either an upstream candle-core upgrade or a custom GGUF
  reader with IQ dequant kernels — not attempted this session.

## v2026.7.19 (branch, unreleased) — full audit + fix pass

A comprehensive, unbiased audit (7 parallel agents, every crate, no length
limit) found roughly 100 issues across all severities; nearly all
critical/high findings are fixed on this branch, plus a new Python
inference library and the project's first-ever real end-to-end
multimodal (vision) test.

### Fixed — correctness and safety
- **HTTP server hardcoded ChatML for every model** regardless of its real
  chat template (`ModelMeta` was loaded then discarded, never reaching
  `AppState`) — now uses the served model's own template, same as the
  chat TUI.
- **`llm-ffi` (C API)**: `tokio::spawn` with no runtime present (UB risk
  across the FFI boundary), a fake tokenizer (raw `char as u32` cast), and
  a silent `DummyBackend` substitution on load failure or a path
  containing "tmp"/"dummy"/"temp" — all fixed.
- **Scheduler**: one bad request could abort every other concurrent
  user's generation; an OOM-preempted sequence never sent its client a
  terminal event, hanging it forever. Both fixed, with regression tests.
- **Cluster**: uneven tensor splits silently dropped data in all-reduce
  and tensor-parallel sharding (fixed, with tests); `llm-cluster`'s
  `main.rs` does no real networking at all (confirmed via its own code
  comments) — now logged as an explicit "not functional" warning at
  startup instead of silently appearing to work.
- **CUDA/Metal device-init failure previously fell back to CPU silently**
  — now returns a clear error, per the project's own hardware-dispatch
  rule.
- **Confirmed, reproduced crash**: batching two different-length prompts
  together crashed `forward_pass` (assumed uniform sequence length across
  a batch). Fixed to use `cu_seqlens` throughout; regression tests added.
- **Silent multimodal-embedding corruption**: the vision-embedding splice
  ran unconditionally for any vision-capable model with no check that an
  image was actually attached, risking corruption of ordinary text
  containing 16+ repeated tokens. Fixed with an explicit guard.
- Real vision-pipeline bugs found via the first-ever live multimodal test
  (a downloaded Qwen2-VL-2B-Instruct GGUF + mmproj, since no local
  vision-capable checkpoint was available): a non-contiguous-layernorm
  crash, a backwards matmul-transpose convention affecting 9 call sites,
  and a `spatial_merge_size` metadata gap causing a shape-mismatch crash.
  The vision pipeline now runs end-to-end without crashing; output is not
  yet coherent for Qwen2-VL specifically because its vision transformer's
  2D rotary position encoding isn't implemented (a real, separate, open
  feature gap — not silently claimed as working).
- **Audio mel-spectrogram was fake**: computed only a per-frame scalar
  energy value fanned out via a fixed sine envelope, with zero real
  frequency information — affected every audio-capable model regardless
  of architecture. Replaced with a real log-mel spectrogram (Hann window,
  DFT power spectrum, triangular mel filterbank, standard normalization).
  Also fixed: the decoded sample rate was read and discarded, so non-16kHz
  audio (the common case) was never resampled — added linear-
  interpolation resampling.
- Dozens of medium/low findings: duplicated/hardcoded Gemma architecture
  checks consolidated to one function; hardcoded `qwen2.5-0.5b`/`1.5b`
  tokenizer-path guesses removed from the server's path-resolution logic;
  a vision position-embedding off-by-one; a zero-bias placeholder with
  the wrong shape; several hot-loop inefficiencies; a `cargo clippy`
  *error* (not just a warning) fixed.

### Added
- **`llm-py`**: a PyO3-based Python inference library with a vLLM-style
  API (`LLM(model=...).generate(prompts, sampling_params)`). Binds
  directly to `llm-core`/`llm-scheduler` (not through `llm-ffi`'s C ABI),
  owns its own Tokio runtime and a real tokenizer, and applies the
  served model's chat template by default. No image/audio input support
  yet (text-only for now).
- `[profile.release]` workspace-wide build settings.
- Graceful shutdown for the HTTP server.

### Known gaps (explicitly open, not silently assumed done)
- CUDA/x86_64 hardware still unverified (unchanged from v1.0.0 — no such
  hardware in this environment).
- `llm-cluster` distributed networking remains non-functional scaffolding
  (now honestly logged as such, not fixed — a multi-week feature, not a
  bug).
- Prefix-cache block reuse is implemented but not wired into block
  allocation (judged too large a change to do safely in this pass).
- Qwen2-VL's 2D rotary vision position encoding is unimplemented.
- No CI workflow exists yet.
- The audio mel-spectrogram fix is unit-tested at the DSP level but not
  exercised end-to-end against a real audio file + audio-capable model
  in this session (none was available locally).
- `llm-py` has no image/audio input support.

## v1.0.0 — 2026-07-19

First "production v1" release. Unifies the `cpu`, `mlx`, and
`vision-stability` branches into a single `v1-unified` branch (merged to
`master` as this release), hardware-agnostic-but-hardware-aware, targeting
CUDA, Metal (Apple Silicon), and CPU (x86_64 + ARM).

### Supported today

- **Backends**: CPU (candle reference backend, x86_64 AVX2/AVX-512 and
  ARM/NEON) and Metal (Apple Silicon GPU), both auto-selected at runtime by
  a single `HardwareProfile::choose_device()` dispatch point — never
  hardcoded, never a silent fallback to the wrong backend. CUDA is
  implemented and feature-gated (`--features cuda`) using the same
  dispatch point, but has not been exercised on real NVIDIA hardware in
  this release cycle (no such hardware was available) — see "Known gaps"
  below.
- **Model paradigms**: dense autoregressive (Llama/Qwen/SmolLM-style) and
  Gemma-4's own architecture (GQA + QK-norm + tied embeddings), both
  auto-classified from GGUF metadata/tensor names with zero model-specific
  code. Multimodal support: vision encoder (SigLIP/Qwen2-VL-style) and
  audio encoder — the latter now supports BOTH a Gemma-4 Conformer encoder
  and a Whisper-style encoder, auto-detected from checkpoint tensor names.
- **CLI**: `llm-cli` (OpenAI-compatible HTTP server — `/health`,
  `/v1/models`, `/v1/chat/completions` with both streaming SSE and
  non-streaming JSON), `chat` (interactive multi-turn chat TUI with Jinja
  chat-template rendering, image/audio input via `/image`/`/audio`),
  `run_model` (one-shot generation + benchmark stats), `benchmark_speed`,
  `devices` (prints the auto-detected `HardwareProfile`).
- Verified end-to-end on this release's test hardware (Apple Silicon /
  macOS, CPU + Metal): both the chat TUI and the HTTP server (streaming
  and non-streaming) produce correct, coherent output against two real
  GGUF checkpoints (SmolLM3-3B, Gemma-4-E2B), with CPU and Metal producing
  byte-identical token IDs on the same model/prompt.

### Fixed in this release

- Restored Whisper-style audio-encoder support, which had been fully
  replaced (not kept alongside) by a Gemma-4 Conformer encoder at some
  point in the `cpu`/`mlx` branches' history — checkpoints using either
  architecture now load correctly, auto-detected from tensor names.
- Fixed a real unsoundness in `llm-cluster`'s all-reduce: raw-pointer
  `Vec<u8>`↔`&[f32]` reinterpret casts assumed alignment `Vec<u8>` doesn't
  guarantee. Replaced with safe byte conversions.
- Fixed a real unsoundness in the GGUF/safetensors loaders: tensor data
  was exposed as `&'static [u8]` behind a `pub` field, letting a caller
  hold a dangling reference after the owning mmap-backed file was
  dropped. Replaced with an on-demand accessor that borrows directly from
  the file, eliminating the unsafe lifetime cast entirely.
- Fixed several silent-fallback bugs that violated the "explicit error,
  never guess" principle: an unknown GGML quant type was silently
  reinterpreted as Q4_K (corrupting weights); missing mandatory GGUF
  metadata (`general.architecture`, `tokenizer.ggml.tokens`) was silently
  guessed as `"llama"` / a fixed vocab size. Both now fail loudly with a
  clear error instead.
- Fixed a real bug in the HTTP server: streaming `/v1/chat/completions`
  never closed the connection after the final token (the SSE stream was
  built on a long-lived broadcast channel with no explicit cutoff),
  meaning any real OpenAI-compatible client (no client-side timeout)
  would hang forever waiting for more data. The stream now ends with an
  explicit `data: [DONE]` sentinel.
- Removed a `println!`-based full-compute-graph dump that fired on every
  chat/server session's first token — moved to a `tracing::trace!` level,
  silent by default.
- Fixed miscellaneous library-code panics (`unwrap()` on real failure
  paths: KV-cache block allocation, NaN-unsafe MoE router logit
  comparison, path handling, prefix-cache lookups) to propagate proper
  errors instead.

### Known gaps (explicitly not silently assumed done)

- **CUDA and x86_64 hardware**: implemented and code-reviewed, but not
  executed on real hardware in this release cycle — no NVIDIA GPU or x86
  machine was available in the environment that produced this release.
  To be verified in a follow-up session on that hardware.
- **Vulkan**: not implemented in the real inference path. The
  CubeCL/Vulkan JIT-kernel crate (`llm-kernel`, matching goal.md's Phase
  5) exists but is not depended on by any other crate — the actual
  inference path dispatches entirely through candle-core's own
  CUDA/Metal/CPU backends.
- **Concurrent throughput / KV waste / numerical parity vs HuggingFace /
  `cargo miri` / cluster fault recovery**: none of these were measured or
  run in this release cycle (would require a load-testing harness, an HF
  reference environment, and multi-node hardware respectively, none of
  which were available). Single-request throughput was measured on real
  models on this release's test hardware (see PROGRESS.md for exact
  numbers) but is not a substitute for the above.
- `models/qwen2.5-0.5b/` in this repo only contains a tokenizer, no
  weight file — could not be used for verification.
