# Progress

## Current task
v1.0.0 shipped. Branch **v2026.7.19** (off master) has since done: a full
7-agent audit and fix pass (see "v2" below), a real HF downloader with
hardware-aware quant recommendation, real TCP networking for
`llm-cluster` (was pure mock), and two rounds of live multimodal testing
(Qwen2-VL + Gemma-4 vision/audio) that found and fixed 6 more real bugs
no prior audit could have caught, since none of them ran a real forward
pass before this session. See "v3 — Model-agnostic/hardware-agnostic
push" below for the full account, including a precise, honest breakdown
of what's still open (CUDA/x86/Vulkan/mobile/Raspberry Pi/physical USB —
none of which this environment can verify — and exactly what each would
need). A separate CUDA/CPU machine then applied more real-hardware fixes
(explicit `--mmproj-path`, multi-GPU VRAM selection, vision bias-shape
guards — see CHANGELOG's v2026.7.20 entry, commit `c7ece93`); those
changes were re-verified end-to-end on this Mac (Metal + CPU) via
`scripts/hardware_check.sh` with no regressions — see "v4" below. Since
then: a real, measured GPU-throughput investigation (llama.cpp is
installed on this machine, used as a direct comparison baseline), which
found and fixed a genuine O(n)-per-decode-step KV-cache reconstruction
bug (real, hardware-agnostic decode throughput improvement, verified
bit-identical output before/after), plus first-pass AWQ/GPTQ safetensors
loaders (dequantize-at-load-time, correctness-first, numerically
**unverified** — needs the GPU machine and a Python reference to confirm).
See "v5" below and `quant-performance-plan.md` for the full plan and
honest status. Re-ran the llama.cpp comparison and multimodal (vision +
audio, Gemma-4 and Qwen2-VL) tests after that work to confirm the fix
holds and nothing regressed — see "v5" update below. Then: found and
fixed a real, model-agnostic multimodal magnitude bug (Gemma's
`embed_scale` was being applied to vision/audio embeddings that should
never see it), and extended the KV-history-cache fast path to also cover
quantized (Q8/Q4, Hadamard-rotated) KV, verified bit-identical to the
old slow path. Multimodal output is measurably less broken but **still
not coherent** - reported honestly as partial progress, not a fix. Then
did a full, dedicated audit of both remaining gaps (audio Conformer via
a focused subagent review, Qwen2-VL vision directly): found and removed
one more real, unjustified constant in the audio attention math
(verified no regression, real behavior change, still not coherent), and
documented several more suspected-but-unverifiable issues precisely
rather than guessing at fixes for them. Then obtained a real reference:
`pip install transformers` gives direct access to the actual
`modeling_gemma3n.py`/`feature_extraction_gemma3n.py` source (no
weights/GPU needed) - reading it directly resolved most of the
previously "could not verify" items with real ground truth. Four more
reference-verified fixes landed: the mel-spectrogram front-end was
completely wrong for Gemma (independently double-confirmed against the
real deployed model's own `preprocessor_config.json`), the SSCP conv
stages used the wrong normalization type (plain LayerNorm instead of a
real cumulative group norm) and the wrong (symmetric, not reverse-
causal) time-axis padding, and the earlier k_scale removal is now
externally confirmed correct by the reference. All four verified with no
regression and real behavior changes on both a synthetic tone and a real
speech sample. **Audio is still not fully coherent** even after all of
this - reported honestly; the most likely remaining cause (missing
validity/causal attention masking for zero-padded buffer positions) is
identified and scoped but not implemented this pass. See CHANGELOG's
"part 6" entry for the complete writeup. **Then implemented that
masking** (real reference-ported causal+validity mask for the chunked
attention, plus lconv1d masking), verified no regression - but the
resulting output got noticeably SHORTER (near-immediate stop), not more
coherent, which turned out to be a critical clue: **"part 6" had used
the wrong reference model entirely**. This project's actual target,
"Gemma-4," is a different, newer model from "Gemma 3n" (the HF
`transformers` model part 6 read). Caught this by finally getting a
REAL, RUNNING reference on this machine
(`mlx-community/gemma-4-e2b-it-4bit` via `mlx-vlm`, both work natively on
Apple Silicon) - it transcribed a real JFK speech sample perfectly,
proving the weights/architecture work and giving actual ground truth to
compare against, not just source code. Reverted three real regressions
(k_scale removal, CumulativeGroupNorm, asymmetric conv padding - all
individually well-reasoned against the WRONG reference), rewrote the
mel-spectrogram front-end to match Gemma-4's real (and much simpler,
no-preemphasis) config, and found+fixed a real mel-filterbank bin-index
bug along the way (numerically confirmed by comparing frame-by-frame
against the real reference's own mel output - one frame matched
bit-for-bit). **Audio is still not coherent** - but for the first time,
precisely why is known rather than guessed: the SSCP-stage and full
encoder outputs were directly compared against the real reference's own
intermediate tensors and do not match, narrowing the remaining bug to a
specific, verifiable location instead of an open-ended search. See
CHANGELOG's "part 7" entry for the complete writeup and what tooling now
exists for the next pass. Not yet merged to master.

**Then did exactly that** - walked the encoder layer-by-layer against the
same real reference tooling instead of reasoning from source alone, and
found five more real bugs, three significant: a block-0 context-window
misalignment in `extract_block_context` (an unused-parameter smell that
turned out to hide a real `saturating_sub` clamping bug), a missing
clamp before `attn_post_norm`, a missing final zero-masking pass after
`output_proj`, a non-periodic (symmetric instead of periodic) Hann
window, and - the dominant one - **the SSCP conv2d was convolving its
kernel against the input with the two spatial axes (time, freq)
transposed relative to the reference**, silently applying the kernel's
time-offset weights to frequency offsets and vice versa. That last fix
alone took the SSCP stage from "right ballpark, not matching" to
**numerically bit-exact** against the real reference (verified frame-by-
frame). The full encoder output still doesn't match exactly (std 1.75 vs
reference's 5.73) even though SSCP and block-0-frame-0 attention output
are now provably exact - the remaining divergence grows with distance
from the start of each attention chunk, pointing at the relative-
position-embedding math as the next suspect, but this wasn't root-caused
this pass. End-to-end audio is **still not coherent**. Also re-ran the
full test suite (195 tests, 0 failures), `hardware_check.sh` (7/7 pass),
and the llama.cpp comparison benchmark on both Metal and CPU-forced -
llm-rs's CPU decode throughput (16.5 t/s) is now within ~11% of
llama.cpp's CPU decode (18.6 t/s), a genuinely competitive result; the
Metal gap (especially prefill, ~44% of llama.cpp's) remains and is a
good next profiling target. Confirmed these audio fixes are backend-
agnostic (pure `candle_core::Tensor` ops, no CUDA-specific code path
exists to separately patch) and wrote an explicit CUDA-hardware
verification checklist for whoever has that hardware. See CHANGELOG's
"part 8" entry for the complete writeup, numeric before/after table, and
full benchmark data.

## v5 — Real GPU-throughput investigation + AWQ/GPTQ loaders, 2026-07-20

### Measured, not assumed: llm-rs vs llama.cpp head-to-head
llama.cpp is installed on this machine (Homebrew, build bcfd1989e) - used
it as a real comparison baseline rather than guessing. Same GGUF file
(`SmolLM3-3B-Q4_K_M`), same M4 Pro, `llama-bench` vs `llm-cli`'s
`benchmark_speed`:

| | Prefill | Decode |
|---|---|---|
| llama.cpp, Metal | 530 t/s | 43 t/s |
| llama.cpp, CPU-only | 55 t/s | 22 t/s |
| llm-rs, Metal (before fix) | 222 t/s | 14.7 t/s |
| llm-rs, Metal (after fix, see below) | 427 t/s | **24.9 t/s** |
| llm-rs, CPU-forced | 20-25 t/s | ~9.3-9.8 t/s (fix gave no measurable change here - see why below) |

### Root cause found and fixed: O(n) full KV-history rebuild every decode step
`llm-core/src/backends/candle.rs`'s `PagedAttention` operator rebuilt the
**entire** K/V history from block storage on **every single decode
step, for every layer**: dequantize each stored block, clone it, concat
all of them into one tensor, then re-run `repeat_kv`+transpose+
`contiguous` over the whole thing - all just to append ONE new token.
Confirmed empirically before fixing anything: a context-length sweep
(28->278 tokens) showed decode throughput dropping from ~31 t/s to ~16
t/s on the same 3B model, far more than a model this size should
degrade over such a small range - the signature of accidentally-
quadratic total decode cost, not normal attention scaling.

Fixed with a new `kv_history_cache: Mutex<HashMap<(SeqId, usize), (usize, Tensor, Tensor)>>`
field on `CandleBackend`: caches the already-repeated/transposed/
contiguous full K/V history per (sequence, source-layer - keyed by
source layer so KV-shared layers like Gemma's local/global sharing reuse
one entry instead of duplicating it), and each step either (a) reuses it
directly if a sharing layer already extended it this same call, (b)
appends just the new token(s) onto it if it's this layer's own turn, or
(c) falls back to the original full-rebuild path unchanged whenever
neither holds (new sequence, evicted/reused seq_id, or quantized Q8/Q4
KV dtype - the Hadamard-rotation + lossy quantization round trip that
path applies isn't replicated by the fast path, so it's excluded
entirely for safety). Cache entries are cleared in `clear_sequence` so a
reused `seq_id` can never read stale history.

**Correctness verified, not assumed**: same prompt, same greedy
sampling, same model - generated token IDs are **bit-identical** before
and after this change (`[128002, 271, 128003, 198, 791, 6864, 315, 9822,
374, 12366, 13, 128012]` both times). Full existing test suite (101
llm-core + 94 llm-cli tests) still passes, and the full
`scripts/hardware_check.sh` run (build/detect/text/vision/audio/cluster)
is still all-green after the change - including the Gemma-4 vision/audio
smoke test, which exercises this same sliding-window/KV-shared code path.

**Why CPU didn't measurably benefit**: this fix removes redundant CPU-
side tensor-copy overhead per decode step. On Metal, the GPU compute
itself is fast, so that CPU-side overhead was a large fraction of total
decode time - removing it is a big win. On CPU, compute is already the
bottleneck (the CPU is doing both the "extra" copy work AND the real
matmul work), so removing the copy overhead barely moves total time.
This is consistent with the same fix very likely mattering even more on
CUDA (also GPU-compute-fast, CPU-copy-bound), though that's not
verified here (no CUDA hardware in this environment).

This is real, hardware-agnostic engine improvement - not a Metal-only
patch - and is exactly the kind of fix `quant-performance-plan.md`'s
"Phase 1: profile before optimizing" step was meant to surface.

### First-pass AWQ + GPTQ safetensors loaders (correctness-first, UNVERIFIED numerically)
New `llm-core/src/loader/awq.rs` and `gptq.rs`: dequantize AWQ/GPTQ
packed 4-bit safetensors weights to dense F16/F32 at load time, wired
into `CandleBackend::load_weights`'s existing safetensors-loading path
(detects `quantization_config.quant_method` in `config.json`, groups the
per-linear `qweight`/`qzeros`/`scales`(+`g_idx` for GPTQ) tensors, and
dequantizes each into a plain `{base}.weight` tensor - everything else
loads through the existing dense-weight path unchanged). `parse_config`
no longer hard-rejects `awq`/`gptq` (bitsandbytes remains rejected - its
packed layout isn't implemented at all). `llm pull`'s pre-download
warning updated to match.

**Tensor layout confirmed by direct inspection** of two real HF repos'
safetensors headers (`TheBloke/Llama-2-7B-AWQ` and `TheBloke/Llama-2-7B-
Chat-GPTQ`, via HTTP range requests - no full download needed), not
assumed from memory: AWQ packs 8 int4 values per int32 along the
**output** axis; GPTQ packs along the **input** axis (opposite of each
other) - a real, easy-to-get-backwards detail now backed by real header
data. Round-trip unit tests (`awq_nibble_order_round_trips`,
`gptq_sequential_order_round_trips`) verify the bit-unpacking logic is
internally self-consistent.

**Explicitly NOT verified**: the actual numerical dequantization has
**not** been checked against a real Python (`transformers`/`autoawq`/
`auto-gptq`) reference computing the same real tensor - AWQ's
documented GEMM-kernel nibble interleave order (`[0,2,4,6,1,3,5,7]`)
and GPTQ's documented zero-point `+1` offset are implemented per public
documentation of those formats, but neither has been confirmed against
ground truth. This is real code, not a stub, but it must not be trusted
for production inference until that check happens - see
`quant-performance-plan.md` phase 4.1's acceptance criteria for exactly
what that check involves. This also intentionally trades away AWQ/GPTQ's
memory savings (full dequant to F16 at load time) for correctness/
simplicity; real throughput AND memory benefits need phase 4.3's
tensor-core kernel work, not attempted here.

Cannot be built or tested end-to-end in this environment at all (no
`nvcc`, no NVIDIA GPU, and no local AWQ/GPTQ model on disk to load) -
this is scaffolding for the GPU machine to build on, exactly like the
CUDA-path work documented in "v4".

### Re-run: llama.cpp comparison (steady state) + real multimodal re-check
Re-ran everything after the KV-cache fix + AWQ/GPTQ loaders landed, to
confirm the numbers hold and nothing regressed - including on Gemma-4
specifically, since that's this project's own primary target
architecture.

**llama.cpp vs llm-rs, steady state** (same `SmolLM3-3B-Q4_K_M.gguf`,
same M4 Pro; first-run numbers are consistently an outlier on this
machine - warm/repeated runs are the representative figures):

| | Prefill | Decode |
|---|---|---|
| llama.cpp, Metal | ~588 t/s | ~57 t/s |
| llama.cpp, CPU-only | ~57 t/s | ~22 t/s |
| llm-rs, Metal | ~430 t/s | ~31 t/s |
| llm-rs, CPU-forced | ~29 t/s | ~9.9 t/s |

llm-rs Metal is now ~1.4x slower on prefill and ~1.8x slower on decode
than llama.cpp on this machine (was ~2.4x / ~2.9x before the KV-cache
fix) - real, repeated-run-confirmed improvement, though both engines'
absolute numbers vary run-to-run on this machine (thermal/system state),
so treat the ratio as the meaningful comparison, not either single
number in isolation.

**Multimodal re-check, with actual coherence checked (not just
crash-free)**: fed a real solid-color test image (clear expected answer:
"red") to both vision-capable models, and a real WAV tone to Gemma-4's
audio path, via `chat`'s `/image`/`/audio` commands:
- **Gemma-4 vision**: ran without crashing, but output is **not
  coherent** ("covering this / word / ing this / ing this ..." -
  repetitive, non-answering degenerate text).
- **Gemma-4 audio**: same - ran without crashing, output not coherent
  (repetitive tokens, some stray non-English tokens - consistent with a
  confused model given a pure sine tone with no linguistic content, but
  still not a real answer).
- **Qwen2-VL vision**: also ran without crashing, output not coherent
  (near-nonsense token soup) - consistent with its already-documented
  missing 2D vision RoPE.
- **Gemma-4 TEXT-ONLY generation (no image/audio), same session**: fully
  coherent - "The capital of France is **Paris**." This isolates the
  problem precisely: the core engine (including today's KV-cache fix) is
  correct; the incoherence is specific to the vision/audio embedding-
  splice path, not a general regression.
- **Verified this is pre-existing, not a regression from today's
  KV-cache change**: built a `git worktree` at the commit immediately
  before that fix (`a7f2887`) and ran the identical Gemma-4 vision prompt
  against it - it also produced incoherent (different, but equally
  non-answering) output. This matches what "v3" already documented
  ("output is not yet fully coherent for either" [vision or audio]) -
  today's testing re-confirms that gap still exists with a real
  descriptive prompt (stronger than the earlier crash-only smoke test),
  but does not newly introduce or worsen it.

**Honest status**: vision/audio pipelines run end-to-end without
crashing on real files, on real production-family models, and don't
corrupt unrelated text/audio-only requests (fixed earlier this branch) -
but neither produces a *correct* answer yet. This is real, open,
pre-existing work, distinct from anything fixed in this session - the
root cause (encoder correctness? projector correctness? something in the
splice/positional-encoding path?) is not yet isolated for Gemma-4
specifically (Qwen2-VL's cause - missing 2D RoPE - is already known).

### Found and fixed a real, model-agnostic multimodal bug: `embed_scale` applied to vision/audio embeddings
Root-caused the Gemma-4 incoherence (not just observed it): Gemma-family
models multiply the ENTIRE embedding tensor by `sqrt(hidden_dim)`
(`embed_scale`) in a single graph-wide `Operator::Scale`, applied in
`graph/builder.rs` AFTER text embedding, vision splice, AND audio splice
all happen (`Embed -> VisualEmbed+Splice -> AudioEmbed+Splice -> Scale`).
That means vision/audio embeddings - which come from a separate encoder
already producing values at the correct final magnitude - were getting
multiplied by `sqrt(1536) ≈ 39.2` (for Gemma-4-E2B) on top of their
already-correct scale, an ~39x magnitude blow-up at exactly the image/
audio token positions. This matches HF's real Gemma3/PaliGemma reference
behavior, which avoids this by scaling text embeddings BEFORE the
image-feature scatter (image features themselves are never scaled).

Fixed by pre-dividing vision/audio embeddings by `embed_scale` right at
the splice point (`SpliceTensors`/`SpliceAudioTensors` in
`llm-core/src/backends/candle.rs`), so the later uniform multiply brings
them back to the encoder's intended scale - mathematically equivalent to
HF's actual order of operations, without restructuring the existing
graph. Gated purely on `meta.embed_scale.is_some()` (only true for Gemma-
family models), so this is a no-op for Qwen2-VL/any non-Gemma
architecture - fully model-agnostic, not a Gemma-specific hack.

**Verified real and correct, but NOT a complete fix**:
- Confirmed no regression: Gemma-4 text-only generation is bit-identical
  before/after (`The capital of France is **Paris**.`); SmolLM3 (non-
  Gemma, `embed_scale: None`) unaffected; full test suite (101+94 tests)
  and `hardware_check.sh` (all 7 checks) still pass.
- Measured real behavior change: Gemma-4 vision went from repetitive
  garbage tokens ("covering this / ing this / ing this...") to a clean,
  short stop (no hallucinated babbling) - a real, less-severe failure
  mode, consistent with the magnitude corruption being real and now
  fixed. Tested with both a flat-color and a two-color test image.
- **Still not coherent**: neither vision nor audio produces an actually
  *correct* description yet. Audio in particular still shows garbled
  output, not just silence - meaning at least one more bug remains,
  likely in the vision/audio encoder or projector itself rather than
  this embedding-scale path (which is now verified sound). Not isolated
  in this session - would need either a numerical reference comparison
  against HF's Gemma3/PaliGemma vision-tower output, or further encoder-
  level debugging with more time than this pass allowed. Reporting this
  honestly as real, verified partial progress, not a solved bug.

### Extended the KV-history-cache fast path to quantized (Q8/Q4) KV
The KV-cache optimization from earlier in this section only covered
non-quantized (F16/F32) KV - the Q8/Q4 path (Hadamard-rotated before
quantization) always fell back to the full block-rebuild every step.
Confirmed `generate_hadamard_orthogonal`'s rotation matrix is a pure,
deterministic function of `head_dim` (a fixed Walsh-Hadamard
construction, no randomness) - safe to cache across calls, since the
same rotation is always reproduced identically.

Extended the fast path to cover this case too: the newly-computed
chunk's K/V is put through the *exact same* quantize-then-dequantize
round trip the block store itself applies (via the existing `quantize`/
`dequantize` functions) before being appended to the cached history, so
the result stays bit-identical to reading the same chunk back from
`gpu_kv_cache` - no precision is traded away for the speed gain.

**Verified**: `LLM_KV_DTYPE=q8` and default (F16) KV now produce
bit-identical generated token IDs for the same prompt/model
(`[128002, 271, 128003, 198, 791, 6864, 315, 9822, 374, 12366, 13,
128012]`, both cases). Real measured decode speed with the fast path
active: ~19 t/s on Metal for `LLM_KV_DTYPE=q8` (no "before" number exists
for direct before/after comparison on this specific path, since it
wasn't benchmarked prior to this extension - but it shares the identical
architectural fix already proven on the F16 path).

## v4 — Re-verification of GPU/CPU-machine fixes on this Mac, 2026-07-20
Commit `c7ece93` (authored on a separate CUDA/CPU machine) added an
explicit `--mmproj-path` CLI flag, fixed a multi-GPU VRAM-selection bug
in `query_nvidia_smi` (was always reading the first `nvidia-smi` line,
which is wrong on dual-GPU laptops where line 1 can be an iGPU reporting
0 free VRAM), fixed a vision patch-embed bias shape mismatch, added an
`LLM_FORCE_CPU` override, and added CUDA-tensor-load-failure-falls-back-
to-CPU handling plus a real bug fix in the VRAM-budget size estimate
(dequantized tensors are F32 = 4 bytes/element, was calculating as if F16
= 2 bytes/element, undercounting VRAM usage by 2x on CUDA).

Reviewed the full diff and re-ran, on this machine (Apple Silicon):
- `cargo build --release --features metal` and plain `cargo build
  --release` (CPU-only, no feature flags) both compile clean.
- `cargo test --workspace --exclude llm-py --features metal`: 94/94
  passed, no regressions.
- A real forced-CPU generation via the new `LLM_FORCE_CPU=1` env var
  (bypassing Metal entirely) produced correct output ("The capital of
  France is Paris.") — proves the CPU path specifically, not just that
  it compiles.
- Full `scripts/hardware_check.sh --release` run (build, hardware
  detection, text generation, vision, audio, cluster registration,
  cluster failure-detection): **all 7 checks pass** on Metal, same as
  before this commit landed.

**Honestly still true**: none of this verifies CUDA itself — no NVIDIA
GPU or CUDA toolchain (`nvcc`) exists in this environment, so the CUDA-
specific code paths (the VRAM-selection fix, the load-failure-falls-
back-to-CPU path, the F32 size-estimate fix) were reviewed by reading the
diff only, not executed. They are additive and gated behind
`self.device.is_cuda()` / non-CPU-device checks, so they cannot affect
the Metal/CPU paths that WERE verified above — but "reviewed, looks
correct" is not the same claim as "ran on a CUDA GPU and confirmed."

## v3 — Model-agnostic / hardware-agnostic push, 2026-07-19

### Real multimodal bugs found via TWO live model tests (not just audit)
Continued from "v2"'s audit findings: downloaded real mmproj-paired
checkpoints (none existed locally) and ran real images/audio through
them, since no prior test in this project's history had ever exercised a
real forward pass through the vision or audio pipeline end-to-end.

**Test 1 — Qwen2-VL-2B-Instruct** (ggml-org GGUF + Q8_0 mmproj):
1. Non-contiguous `layer_norm` crash (`x` permuted but never made
   contiguous before `candle_nn::ops::layer_norm`; harmless for models
   with an absolute position-embedding table, but Qwen2-VL has none).
2. Wrong matmul-transpose convention: `vision.rs` assumed all per-layer
   weights load as `[out, in]` (standard PyTorch), but this file's
   weights load as `[in, out]`.
3. `spatial_merge_size` silently defaulting to 1 (metadata key absent
   from this real file) instead of the required 2, causing a shape
   mismatch against the projector.
4. A defensive bias-shape check added for a genuinely internally-
   inconsistent bias tensor in this specific export.

**Test 2 — Gemma-4-E2B-it** (unsloth GGUF + matching mmproj-BF16, the
project's own primary target architecture, `has_vision_encoder` AND
`has_audio_encoder` both true):
5. The `[in,out]` vs `[out,in]` fix from Test 1 **broke this file** -
   confirmed these are genuinely different per-file conventions (not a
   fixed property of the code), likely a byproduct of each file's own
   quantization/export tooling (Q8_0 vs BF16). Replaced the fixed
   assumption with `linear()`, a small helper that auto-detects
   orientation per-tensor by comparing each weight axis against the
   actual input feature dimension — works for both conventions.
6. `Operator::VisualEmbed` ran the vision encoder on a dummy zero image
   and cached the result UNCONDITIONALLY on every request to any
   vision-capable model, even pure-text/audio-only ones. That cache
   write is exactly what the "is an image actually active" splice guard
   checks — so the dummy encode on the first op of a forward pass made
   every later op in the SAME pass believe a real image was preloaded.
   Confirmed via a real `/audio`-only request: the dummy vision
   embedding got spliced into the audio placeholder token run, crashing
   with a length mismatch unrelated to audio at all. Fixed by mirroring
   `AudioEmbed`'s already-correct pattern (skip the encoder AND the
   cache write entirely when no image is active).
7. `chat.rs` computed the audio placeholder-token count from
   `meta.audio_embedding_length` — the encoder's HIDDEN dimension (1024),
   not a sequence length — silently inserting the wrong placeholder
   count for every audio request. Since `load_audio` always produces a
   fixed 3000 mel frames, each architecture's real output length is a
   fixed constant from its conv-subsampling factor (750 for Gemma-
   Conformer's 4x, 1500 for Whisper's 2x); now computed from the
   already-correct `audio_num_mel_bins` field instead.
8. `symphonia`'s "pcm" feature was missing (`"wav"` only enables the
   container-format parser, not the PCM codec most real WAV files
   actually use inside it) — a real WAV test file failed to decode
   entirely, silently falling back to zeros and masking bugs 6-7 during
   initial testing.

**Result**: both vision and audio now run the FULL pipeline end-to-end
without crashing on real files, on a real production-family model, for
the first time in this project's history. Output is not yet fully
coherent for either (Qwen2-VL needs unimplemented 2D rotary position
encoding for its vision transformer — confirmed via direct GGUF
inspection that no such tensor/metadata exists to read; the audio test
used a synthetic pure-tone WAV with no linguistic content, and there may
be residual Conformer correctness gaps not yet isolated). This is real,
honestly-reported remaining work, distinct from the crash-level bugs
fixed above — do not read "runs without crashing" as "fully correct."

Also found and fixed, unrelated to the above: the audio mel-spectrogram
in `load_audio` was ENTIRELY FAKE (a per-frame scalar RMS energy value
fanned out across mel bins via a fixed sine envelope — zero real
frequency information, affecting every audio-capable model regardless of
architecture). Replaced with a real log-mel spectrogram (Hann window,
direct DFT power spectrum, proper triangular mel filterbank, Whisper's
standard normalization) plus sample-rate detection + linear-
interpolation resampling (the decoded sample rate was previously read
and discarded, so non-16kHz audio — the common case — was never
resampled). New tests prove a 1kHz and 4kHz tone now peak in genuinely
different mel bins, which the old placeholder could never do.

### `llm pull`: real HF downloader with hardware-aware recommendation
New binary implementing goal.md's `llm pull <model>` contract:
- Resolves a bare search term (HF search API, results sorted by download
  count so the canonical repo wins over obscure forks — the raw
  relevance order does NOT reliably do this, confirmed by search
  surfacing a 33-download fork ahead of a 164k-download official repo)
  or an explicit `owner/repo`.
- Lists every GGUF quant variant with real sizes (from HF's tree API),
  recommending the largest one that fits this machine's detected
  `HardwareProfile` free VRAM/RAM with the same 15% headroom
  `choose_device` uses at load time.
- Detects (but does not implement dequantization for) bitsandbytes/AWQ/
  GPTQ-quantized native-safetensors repos via `config.json`'s
  `quantization_config`, with a clear message rather than a silent wrong
  load — also enforced as a defense-in-depth check directly in
  `llm-core/src/model/config.rs::parse_config`, so a model obtained any
  other way (not just via `pull`) hits the same clear error.
- Downloads the chosen file(s) plus tokenizer/config sidecars, with a
  fallback that fetches `tokenizer.json` from the likely base (non-GGUF)
  repo when a GGUF-only repo doesn't ship one itself (common for
  official quantization repos — confirmed: `Qwen/Qwen2.5-0.5B-Instruct-
  GGUF` ships no tokenizer.json at all).
- Verifies download completeness (received bytes == content-length)
  before declaring success and removing partial files otherwise — found
  via this session's own testing: an interrupted download previously
  looked, at load time, EXACTLY like a confusing model-compatibility bug
  ("Failed to read tensor X"), when it was actually just a truncated
  file. This cost real debugging time before the actual cause was found;
  now it can't happen silently again.

**Verified against 4 real HF repos**:
- `Qwen/Qwen2.5-0.5B-Instruct-GGUF`: Q4_K_M and Q8_0 both download and
  generate correctly ("The capital of France is Paris" / "France").
  Q2_K genuinely fails to load — see below, a real and precisely-
  diagnosed gap, not a downloader bug (confirmed via independent SHA256
  verification against HF's LFS hash that the download itself is
  byte-perfect).
- `HuggingFaceTB/SmolLM2-135M-Instruct`: a **native HF safetensors repo,
  no GGUF at all**. This is the first time the HF-safetensors (non-GGUF)
  loading path has been exercised end-to-end in this project's history
  (a prior audit flagged it as untested). It worked correctly on the
  first real try: "The capital of France is called Paris." — real
  evidence that model-agnosticism holds across both major weight
  formats, not just GGUF.
- `TheBloke/Llama-2-7B-AWQ`: correctly detected and refused with a clear
  message (not attempted to load).

### A genuine, precisely-diagnosed GGUF compatibility gap: "IQ" quant types
`Qwen2.5-0.5B-Instruct-GGUF`'s "Q2_K" file mixes in llama.cpp's newer
IQ4_NL "importance quantization" format (GGML dtype id 20) for most
weight tensors — a real, common technique (upgrading precision for
sensitive tensors even within an overall low-bit quant scheme), not a
corrupt or unusual file. Confirmed via direct inspection (Python `gguf`
library) of the exact dtype ids used, then confirmed by grepping
`candle-core 0.9.2`'s own source: it has **zero dequantization support
for any IQ-series type** (IQ1_S/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ4_NL/IQ3_S/
IQ2_S/IQ4_XS/IQ1_M). Worse, this failure happens while candle-core is
still parsing the file's HEADER (building its tensor-info table), which
aborts the ENTIRE file the moment it hits one unrecognized dtype id — so
it can't be worked around per-tensor at the point where we load weights;
properly supporting it would mean replacing candle-core's GGUF reader
with a custom one for such files, a genuinely large architecture change
that was not attempted here (this session's own `llm-core/src/loader/
gguf.rs` already exists and was hardened earlier, but is not wired into
the real inference path and doesn't implement IQ dequant either).
Classic quant types (F16/F32/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, and
the whole K-quant family Q2_K..Q8_K) all confirmed working. Fixed the
IMMEDIATELY actionable part: replaced the opaque "Failed to read GGUF
content" error with one that names the likely cause (an IQ-series type)
and suggests trying a different quant of the same model.
**This is exactly the kind of gap that should be reported precisely, not
hidden or hand-waved as "model-agnostic, done" — it isn't, for this one
(increasingly common) quantization family, until someone implements IQ
dequant kernels or wires in a replacement GGUF reader.**

### Real TCP networking for `llm-cluster` (was pure mock scaffolding)
`main.rs`'s Coordinator/Worker previously did no networking at all —
confirmed by its own code comments, and flagged in the "v2" audit.
Replaced with: a real length-prefixed JSON wire protocol
(`protocol.rs`), a real `TcpListener`-based coordinator that registers
workers via a Hello/Welcome handshake carrying their actual profiled
`NodeCapability`, and a real `TcpStream`-based worker that sends actual
periodic heartbeats and reconnects with backoff on failure.
`ClusterHealthMonitor::record_heartbeat` — previously never called from
anywhere, so failure detection was structurally inert regardless of what
happened on the network — now fires from real received messages.

**Verified with two real local processes** on `127.0.0.1:9123`: both
workers registered with correct profiled capabilities, heartbeats flowed
continuously, and killing one worker's process was correctly detected
(both via immediate disconnect AND the heartbeat-timeout sweep) within
the configured window, evicting it from the active-node list.

**Scope, stated plainly**: this is real transport + registration +
failure detection, not the full goal.md Pause-Replicate-Retry story.
`analyzer.rs`/`collective.rs`/`tensor_parallel.rs` (layer partitioning,
all-reduce, tensor-parallel sharding) still are not invoked from this
networking layer or from real inference — a node failure is detected and
evicted from the roster, but nothing re-partitions work onto survivors
or re-prefills in-flight sequences. The coordinator now logs this
explicitly on every failure event so it's never mistaken for full
recovery.

### Hardware portability: what's actually verified vs. genuinely open
Stated as precisely and honestly as possible, hardware by hardware:

| Hardware | Status | What's actually true |
|---|---|---|
| **CPU, ARM (this machine: Apple Silicon)** | ✅ Verified | Real models generate correctly; NEON detected correctly by `HardwareProfile`. |
| **Metal (Apple Silicon GPU)** | ✅ Verified | Real models generate correctly, numerically matches CPU on the same prompt (byte-identical token IDs, confirmed in the v1.0.0 pass). |
| **CPU, x86_64** | ⬜ Not verified on real hardware | Code is generically portable (candle-core supports x86_64 AVX2/AVX-512 like any other target); `CpuSimdCaps::detect()` is properly `#[cfg(target_arch = "x86_64")]`-gated. Cross-compiling to `x86_64-unknown-linux-gnu` from this Mac was attempted and failed for environmental reasons (no cross C-toolchain in this sandbox for `tokenizers`' `onig`/`esaxx-rs` build scripts) — inconclusive, not a code-level finding either way. Needs real x86 hardware to verify. |
| **CUDA** | ⬜ Not verified on real hardware | No NVIDIA GPU available in this environment, unchanged from the v1.0.0 pass. Code reviewed statically: `HardwareProfile` gates the whole path behind `candle_core::utils::cuda_is_available()`, falls back to CPU with a clear warning (not silently) if `nvidia-smi`/the CUDA driver API can't be queried, and an earlier hardening pass fixed the previous silent-CPU-fallback-on-init-failure bug. Needs real CUDA hardware to actually run and verify numerically. |
| **Vulkan** | ❌ Not implemented at all, and a real gap to close (not just untested) | `candle-core` itself has **no Vulkan backend** — only CUDA/Metal/Accelerate/MKL. The only Vulkan-adjacent code in this repo is `llm-kernel` (CubeCL/`cubecl-wgpu`), confirmed dead/unwired since the v1.0.0 audit — it implements a handful of standalone kernels (GEMV, attention, RMSNorm, RoPE, SiLU) but nothing resembling a full `LlmBackend` implementation, and nothing in `llm-core` calls it. **What real Vulkan support would require**: implementing an entire second backend on top of `cubecl-wgpu` — quantized weight loading/dequantization, the full transformer forward pass (attention with paged KV, RMSNorm, RoPE, SwiGLU/gated MLP, sampling), wired through the `LlmBackend` trait exactly like `CandleBackend` — realistically a multi-week project, not a bug fix. No Vulkan SDK is installed in this environment either, so even a minimal wgpu-Vulkan smoke test isn't possible here right now. |
| **Raspberry Pi (aarch64 Linux)** | ⬜ Not verified, but no known code-level blocker | Grepped the entire `llm-core`/`llm-cli`/`llm-cluster` source: the *only* `target_os`-gated code anywhere is the Metal VRAM query (correctly macOS-only) — nothing else assumes macOS, so there's no known reason this wouldn't build and run on Linux aarch64 via the plain CPU backend. The existing RAM-aware `choose_device` OOM guard (refuses to load a model that won't fit, with a clear error, rather than letting the OS OOM-killer take down the process) is exactly the kind of defensive behavior a memory-constrained device like a Pi needs, and it's already there. What's genuinely unverified: actual cross-compilation to `aarch64-unknown-linux-gnu`/`aarch64-unknown-linux-musl` was not exercised in this session (blocked here by the same cross-toolchain gap as the x86_64 attempt above — this is an environment limitation of this sandbox, not evidence about the Pi itself), and no physical Pi (or any ARM Linux box) was available to actually run a build on. |
| **Mobile (Android/iOS)** | ❌ Not implemented, real scope needed | No JNI (Android) or Swift/Objective-C (iOS) bindings exist. `llm-ffi`'s C ABI (now fixed this session: real Tokio runtime, real tokenizer, `catch_unwind` at the boundary) is the right foundation to bind FROM, but nobody has built the Android `.aar`/iOS `.xcframework` packaging, verified `candle-core` actually builds for `aarch64-linux-android`/`aarch64-apple-ios` targets, or confirmed Metal works through iOS's (technically compatible, but unverified in this specific stack) Metal implementation. This is a genuine, multi-week mobile-packaging project on top of a now-more-solid FFI base, not a quick addition. |
| **Physical USB / multi-machine networking** | ⬜ Transport code exists and is generically correct; physically untested | `llm-cluster`'s new TCP protocol (see above) is plain `std`/`tokio` TCP sockets with no interface-specific code, so it should work unmodified over a USB cable in RNDIS/CDC-ECM gadget mode (which presents to the OS as an ordinary network interface with its own IP — goal.md's own description of how this is meant to work). This was verified over `127.0.0.1` in this session because only one machine was available; **no physical two-machine USB link, and no actual distributed inference (tensor-parallel/pipeline execution split across the registered nodes) was exercised**, since that layer (`analyzer.rs`/`collective.rs`/`tensor_parallel.rs`) still isn't wired to real inference at all (see above). |

The honest summary: CPU+Metal on Apple Silicon is the only combination
with real, verified, numerically-checked evidence behind it. Every other
row above is either "should work, structurally no reason it wouldn't,
but genuinely never run" (x86 CPU, CUDA, Raspberry Pi) or "does not
exist yet and needs real engineering time to build" (Vulkan, mobile
packaging, actual distributed multi-node inference execution).

## v2 — Full audit + fix pass, branch v2026.7.19, 2026-07-19

### The audit
Ran 7 parallel adversarial audit agents (no length limits, instructed to
hide nothing) covering: llm-core (split across 4 sub-passes: candle.rs;
vision/audio/multimodal/weights/attention; loader/quantization/profile/
backend; model/graph/sampler/tokenizer/types/conv_template),
llm-scheduler+llm-cluster, llm-cli+llm-ffi+build-config, and a
cross-cutting vision/hardcoding/goal.md-promise sweep. Combined findings:
~100 distinct issues across all severities. Highest-severity themes:

- **`llm-cli`'s HTTP server hardcoded ChatML for every model** —
  `ModelMeta` was loaded then discarded in `main.rs`, never reaching
  `AppState`, so `/v1/chat/completions` formatted every request the same
  way regardless of the served model's actual chat template. The chat TUI
  did this correctly; the production HTTP API did not. **Fixed.**
- **`llm-ffi` (C API): `tokio::spawn` with no runtime present** (panics/UB
  across the FFI boundary for any real C/Python caller), a **fake
  tokenizer** (raw `char as u32` cast, not `LlmTokenizer`), and a **silent
  `DummyBackend` substitution** on load failure or on any model path
  containing "tmp"/"dummy"/"temp". **All fixed.**
- **`llm-scheduler`: one bad request could kill every other concurrent
  user's generation** (a single sampling error aborted the whole batch),
  and an **OOM-preempted sequence never got a terminal event**, hanging
  its client forever. **Both fixed**, with new regression tests.
- **`llm-cluster`: uneven tensor splits silently dropped data** in both
  the all-reduce and tensor-parallel sharding code (any length not evenly
  divisible by world_size lost its remainder elements with no error), and
  **the whole crate's `main.rs` does no real networking at all** (mock
  coordinator/worker, confirmed via its own code comments) — the latter
  is now logged as an explicit "not functional" warning at startup instead
  of silently looking like it works. Data-loss bugs **fixed** with tests;
  the "build a real distributed cluster" gap is a feature, not a bug, and
  stays open.
- **`llm-core`: CUDA/Metal device-init failure silently fell back to
  CPU** instead of erroring — a direct violation of CLAUDE.md's own
  hardware-dispatch rule. **Fixed** (now propagates a clear `Err`).
- **A confirmed, reproduced mixed-length-batch crash**: found live via the
  new Python bindings (`llm.generate()` with two different-length
  prompts in one batch crashed with a reshape/matmul shape mismatch).
  `forward_pass` assumed every sequence in a batch had equal length; now
  correctly uses `cu_seqlens` throughout (packed/varlen layout), with new
  regression tests. **Fixed.**
- **Silent multimodal-embedding corruption on ordinary text**: the vision
  embedding splice ran unconditionally for any vision-capable model, with
  no check that an image was actually attached — a 16+-token repeated
  run in ordinary text (padding, repeated punctuation) could trigger a
  silent splice of zero-valued "image" embeddings over real text. **Fixed**
  (mirrors the audio path's existing explicit guard).
- Dozens more medium/low findings (hardcoded Gemma arch-name checks
  consolidated to one function instead of duplicated in 2 files;
  hardcoded `qwen2.5-0.5b`/`1.5b` tokenizer-path guesses removed;
  vision.rs position-embedding off-by-one; zero-bias placeholder wrong
  shape; hot-loop inefficiencies; `[profile.release]` added; graceful
  shutdown wired up; etc.) — see individual commit messages on
  v2026.7.19 for the full itemized list.

### Execution
Dispatched 3 parallel background agents (isolated git worktrees) to fix
llm-core, llm-scheduler+llm-cluster, and llm-cli+llm-ffi independently
(disjoint file ownership to avoid conflicts). One agent (llm-core, the
largest scope) hit a session/API limit mid-task; resumed it in its
existing worktree (preserving all uncommitted progress) with a follow-up
agent rather than restarting. Merged all three branches back into
v2026.7.19 — two clean merges, one with two small, mechanically-resolved
conflicts (`llm-scheduler/scheduler.rs`, `llm-cli/bin/chat.rs` — both
crates' fixes touched the same lines for different reasons; kept both
sets of changes). Fixed follow-on breakage in `llm-tests` fixtures that
none of the three agents' scopes covered (constructor signatures changed
by two different agents independently). **Full workspace test suite
after every merge/fix round: 252 passed, 0 failed.**

### Python inference library (`llm-py`)
New crate, PyO3-based, vLLM-style API:
```python
from llm_rs import LLM, SamplingParams
llm = LLM(model="./model.gguf")
outputs = llm.generate(["Hello!"], SamplingParams(temperature=0.7, max_tokens=64))
```
Binds directly to `llm-core`/`llm-scheduler` (not through `llm-ffi`'s C
ABI) so it owns its own Tokio runtime and a real `LlmTokenizer`,
sidestepping both FFI bugs above by construction rather than needing them
fixed first. `SamplingParams` validates at construction (bad values raise
`ValueError` immediately, not silently clamped). `generate()` applies the
served model's own chat template by default (wired to the same shared
`render_prompt`/`manual_format` functions the HTTP server and chat TUI
use — confirmed via a real Gemma-4 test: without this wiring, output was
degenerate garbage; with it, correct). Built with `maturin`, verified
importable and working against Python 3.12 + pyo3 0.22. Real end-to-end
tests: SmolLM3 and Gemma-4 text generation (correct output, matches CLI);
mixed-length batch (2 different-length prompts, previously crashed — now
correct, non-cross-contaminated output for both).
**Known gap**: no image/audio input support yet in the Python API (the
chat TUI's `/image`/`/audio` commands have no Python-side equivalent) —
`generate()` is text-only for now.

### First-ever real multimodal (vision) test — found 3 more real bugs
No mmproj-capable checkpoint was available locally (the only local
vision-capable model, gemma-4-E2B, ships without its mmproj file in this
environment), so downloaded a real `Qwen2-VL-2B-Instruct` GGUF + mmproj
pair (ggml-org, ~1.7GB) and ran a real image through `/image` in the chat
TUI — the first time any real image has ever been run through this
vision pipeline in the project's history. Found and fixed, in order of
discovery:
1. **Non-contiguous `layer_norm` crash**: `x` was permuted but never made
   contiguous before `candle_nn::ops::layer_norm`; harmless for models
   with an absolute position-embedding table (whose `broadcast_add`
   happens to produce a contiguous tensor first) but Qwen2-VL has none,
   so `x` reached `layer_norm` still permuted. **Fixed.**
2. **Wrong matmul transpose convention**: `vision.rs` called `.t()` on
   every per-layer attn/ffn weight assuming the standard PyTorch `[out,
   in]` layout, but this GGUF-loading path actually produces `[in, out]`
   for these tensors (confirmed empirically via a debug print against
   real loaded tensors) — removed the incorrect transpose at 9 call
   sites, and fixed the QKV-fusion logic (wrong concat axis, wrong
   dimension read for zero-bias sizing) that had the same wrong
   assumption baked in. Left `.t()` in place for the `mm.0`/`mm.2`
   projector weights, which DO load as `[out, in]` — a confirmed, real
   difference in convention between per-layer and projector-level
   tensors in this export, not an inconsistency to "fix away." **Fixed.**
3. **`spatial_merge_size` silently defaulting to 1 for Qwen2-VL**:
   `clip.vision.spatial_merge_size` is simply absent from this real
   mmproj file's metadata (confirmed via direct GGUF inspection), so the
   hardcoded default of 1 was always wrong for this architecture
   (needs 2), causing a matmul shape mismatch against the projector.
   Now derives the merge factor from the projector's own weight shape
   (`hidden_dim * merge_size² == projector_input_dim`) when metadata
   doesn't provide it — model-agnostic, no architecture-name hardcoding,
   works for any file regardless of whether it happens to export this
   metadata key. **Fixed.**
4. Also added a defensive bias-shape check: this specific mmproj export's
   `ffn_up.bias`/`ffn_down.bias` are internally inconsistent with their
   own weights' output dimensions (confirmed via direct inspection with
   the `gguf` Python library — an apparent bug in the exporter itself,
   not our code). Rather than crash or guess which bias belongs where,
   skip a mismatched bias with a warning.

**Result**: the vision pipeline now runs end-to-end without crashing for
a real Qwen2-VL-2B-Instruct model — confirmed exercising the full
graph-splice/attention/allocation path for the first time ever. **Output
is not yet coherent for this architecture**: Qwen2-VL's vision
transformer needs 2D rotary position encoding for patches, which this
file doesn't implement at all (confirmed: no such tensor or metadata
exists to read, this is a genuine unimplemented feature, not a bug to
silently paper over). This is real, scoped, honestly-reported remaining
work — not claimed as done.

### Fake audio mel-spectrogram — fixed (found in the audit, not live-tested)
Per the audit, `audio.rs::load_audio` computed only a per-frame scalar RMS
energy value fanned out across mel bins via a fixed `sin(bin/n * pi)`
envelope — real-looking but carrying zero frequency information,
identical in shape for any two inputs of the same loudness. This affected
every audio-capable model equally (Whisper-style or Gemma-Conformer),
since the bug was in shared feature extraction. Replaced with a real
log-mel spectrogram (Hann-windowed frames, direct DFT power spectrum,
proper triangular mel filterbank, Whisper's standard normalization).
Also fixed: the decoded audio's real sample rate was read and discarded,
so any non-16kHz file (the common case) was fed unresampled into a
16kHz-assuming pipeline — added linear-interpolation resampling. New
tests prove a 1kHz and 4kHz tone now peak in genuinely different mel
bins. **Not live-tested against a real audio file + audio-capable
checkpoint** in this session (no such local model available, and the
multimodal-download budget for this session went to the vision test
instead) — the fix is unit-tested at the DSP level but not exercised
end-to-end through a real audio encoder. Flagging honestly, not claiming
full verification.

### Also fixed while auditing
- A `cargo clippy` **error** (not just a warning) in `vision.rs` (a
  provably-zero multiplication in the image-normalization loop — harmless
  numerically, but a real clippy blocker). `cargo clippy` now reports
  zero errors workspace-wide (style warnings remain, not addressed).
- `[profile.release]` added to the workspace (lto=thin, codegen-units=1,
  panic=unwind kept for `llm-ffi`'s `catch_unwind` requirement).
- Graceful shutdown wired into the HTTP server.
- A minimal `.github/workflows/ci.yml` — explicitly deferred by the
  cli+ffi agent (out of its assigned file scope) — **still not done**,
  flagging as open.

### What's still honestly open after this pass
- CUDA and x86_64: still unverified on real hardware (unchanged from the
  v1.0.0 gap — no such hardware in this environment).
- `llm-cluster`'s distributed networking: still non-functional scaffolding
  (now honestly logged as such at startup instead of looking like it
  works) — building a real implementation is a multi-week feature, not a
  bug fix, and was correctly out of scope for this pass.
- Prefix-cache block reuse: still computed but not wired into actual
  block allocation (the scheduler agent judged wiring it correctly — with
  proper ref-counting across recycled sequences — too large a change for
  this pass; left a loud, accurate comment instead of a misleading one).
- Qwen2-VL vision positional encoding (2D-RoPE): not implemented (see
  above).
- No CI workflow exists yet.
- `llm-py` has no image/audio input support yet.
- Real audio end-to-end test not performed (no local model available).
- v2026.7.19 not yet merged to master (holding for explicit go-ahead,
  same as the v1.0.0 release-push pattern earlier).

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
