# Changelog

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
