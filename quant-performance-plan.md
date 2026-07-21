# llm-rs: 4-bit Quantization + GPU Throughput Plan

Triggered by a real, measured finding: on this dev machine (M4 Pro), llm-rs's
Metal backend is ~2.4x slower on prefill and ~2.9x slower on decode than
llama.cpp on the **identical** GGUF file. Benchmarked 2026-07-20 with
`llama-bench` (Homebrew llama.cpp, build bcfd1989e) vs `llm-cli`'s
`benchmark_speed` binary, same model (`SmolLM3-3B-Q4_K_M.gguf`), same machine:

| | Prefill | Decode | Metal-vs-own-CPU |
|---|---|---|---|
| llama.cpp, Metal (`-ngl 999`) | 530 t/s | 43 t/s | 9.6x / 2.0x |
| llama.cpp, CPU-only (`-ngl 0`) | 55 t/s | 22 t/s | — |
| llm-rs, Metal | 222 t/s | 14.7 t/s | 8.8x / 1.5x |
| llm-rs, `LLM_FORCE_CPU=1` | 25 t/s | 9.8 t/s | — |

Two separate problems, not one:
1. **llm-rs is ~2.2-2.4x slower than llama.cpp on CPU too.** Prefill scaling
   from CPU→Metal is nearly proportionate to llama.cpp's own (8.8x vs 9.6x),
   which means a big chunk of the "GPU isn't helping much" feeling is
   actually "the whole engine has more per-token overhead than llama.cpp,
   independent of backend." This is a profiling problem, not a kernel one.
2. **Decode-specific Metal speedup (1.5x) undershoots llama.cpp's own (2.0x)**
   on the same hardware/model. Prefill (batched matmul) scales fine; decode
   (batch=1, matrix-*vector* kernel path) doesn't. This is a real, isolated
   Metal-kernel-tuning gap.

Separately: on CUDA, this codebase (via candle-core) has **zero** AWQ/GPTQ
support and no Marlin-class tensor-core INT4 kernel of any kind — confirmed
by research, not assumed. GGUF is the only format llm-rs can run today, on
any backend.

Constraint for this plan, per explicit direction: **Metal work stays inside
candle** (no separate MLX C++/Swift runtime dependency — this is a
hardware-agnostic engine, not "GGUF here, a different runtime there").

Do phases in order. Each ends with an acceptance test. Phase 4 (CUDA) cannot
be built or verified in this environment at all (no `nvcc`, no NVIDIA GPU) —
see that phase's own note before starting it.

---

## PHASE 1 — Find out where the per-token overhead actually is (do first, cheap, no new kernels)

Don't guess between "Rust-side overhead" and "kernel is slow" — measure.
This determines which of Phase 2/3 is worth doing at all.

**UPDATE 2026-07-20**: found the dominant issue via direct code reading
(a context-length sweep pointed at it, then `PagedAttention`'s block-
gather loop confirmed it) rather than the wall-clock/Metal-trace
instrumentation below — a real O(n)-per-decode-step KV-history rebuild,
fixed (see PROGRESS.md "v5", CHANGELOG). Measured result: Metal decode
14.7 → 24.9 t/s (+69%) on the same benchmark; CPU-forced decode
unchanged (~9.3-9.8 t/s either way — consistent with CPU already being
compute-bound rather than copy-bound, see PROGRESS.md's explanation).
llm-rs Metal decode is still behind llama.cpp's 43 t/s on the same file —
steps 1.1/1.2 below (proper per-stage wall-clock/GPU-trace profiling) are
still worth doing to find what's left, now that the biggest single item
is gone.

- [ ] **1.1 — Wall-clock instrument one decode step end-to-end.**
  Add `tracing::debug_span!`/manual `Instant::now()` timing around the
  stages inside `CandleBackend::forward_pass` (llm-core/src/backends/candle.rs)
  for a single-token decode call: KV-cache tensor construction/copy, the
  actual matmul/attention calls, sampling (llm-core/src/sampler.rs), and
  anything llm-scheduler does per step (llm-scheduler/src/engine.rs,
  scheduler.rs). Run with `--max-new-tokens 64` on the same SmolLM3 model
  used for the benchmark above, print per-stage microsecond totals.
  Acceptance: a table showing what fraction of each ~14.7 t/s decode step
  (≈68ms/token) is kernel time vs everything else.

- [ ] **1.2 — Cross-check with Metal's own GPU trace.**
  Attach Xcode's Metal System Trace (or `MTL_CAPTURE_ENABLED=1` +
  `MTLCaptureManager`) to a running `chat`/`benchmark_speed` process during
  decode. This shows actual GPU-side kernel dispatch time vs CPU-side
  encoding/submission overhead — the thing step 1.1 can't see from the Rust
  side alone.
  Acceptance: confirms or refutes whether GPU kernel execution itself is the
  bottleneck, or whether it's Rust-side buffer prep/dispatch overhead between
  kernel calls.

- [ ] **1.3 — If overhead is Rust-side (allocations, `.contiguous()` copies,
  redundant tensor construction per step):** fix those directly — this
  category of bug has already shown up repeatedly in this codebase (several
  `.contiguous()`/cache-related fixes landed during the vision/audio work).
  Look specifically at whether KV-cache blocks are being reallocated or
  copied per-step instead of written in place, and whether sampling
  (top-k/top-p/repetition penalty) is doing anything O(vocab) more than once
  per step unnecessarily.
  Acceptance: re-run the same benchmark; decode t/s improves without having
  touched any kernel code.

---

## PHASE 2 — Metal kernel tuning, staying inside candle (no MLX runtime)

Only pursue this once Phase 1 shows the gap is genuinely kernel-side, not
Rust-side overhead that Phase 1.3 already fixed.

- [x] **2.1 — Check whether upgrading `candle-core`/`candle-nn` past the
  currently-pinned 0.9.2 (Cargo.toml: `candle-core = { version = "0.9" }`)
  already closes some of the gap.**
  **DONE 2026-07-21 — tried, no measured benefit, reverted.** Bumped to
  0.11.0, full workspace built clean (also transitively updated
  `tokenizers` 0.22.2, `safetensors` 0.8.0), full test suite passed, and
  SmolLM3 text-only output was bit-identical to 0.9.2. But 5 repeated
  Metal decode benchmarks on the same prompt/model landed at ~22-28 t/s,
  overlapping the 0.9.2 baseline's own noisy ~25-31 t/s range - no clear
  win, and given this machine's real run-to-run variance (confirmed
  repeatedly throughout this session), that's not enough signal to
  justify the risk of a major dependency bump (larger diff surface,
  unverified CUDA-path implications since this environment can't test
  CUDA at all). Reverted `Cargo.toml`/`Cargo.lock` back to 0.9.2. If
  revisited, do it with a CUDA-capable machine in the loop so the CUDA
  side of the bump gets real signal too, not just Metal's inconclusive
  result. Candle 0.9.2 (released Jan 2026) does
  postdate the big Metal matrix-*matrix* kernel fix (PR #2615, merged
  2025-07-18, ~6x prompt-processing speedup upstream) — so prefill should
  already reflect that. But 0.10.0 (2026-03-31) and 0.11.0 (2026-06-26) are
  newer and may carry further Metal quantized-matvec (decode-path) tuning
  that 0.9.2 doesn't have. Diff candle's `candle-metal-kernels` crate source
  between 0.9.2 and 0.11.0 specifically for the quantized matvec kernels
  (the ones actually exercised at batch=1/decode).
  Acceptance: either (a) a version bump measurably improves decode t/s on
  the same benchmark — do it, re-run the full `hardware_check.sh` and the
  existing test suite to confirm no regressions, or (b) confirm nothing
  relevant changed and rule this out, on to 2.2.

- [ ] **2.2 — If still behind after 2.1, profile candle's specific
  quantized-matvec Metal kernel against ggml's** (the one llama.cpp/ggml
  uses — note llama-bench's own Metal init log on this machine reports
  `simdgroup reduction = true`, `simdgroup matrix mul = true`, which is
  exactly the kind of Apple-GPU-specific tuning a matvec kernel benefits
  from). If ggml's kernel has SIMD-group-level optimizations candle's port
  doesn't, this becomes a **small, well-scoped Metal shading language
  patch** to candle's own kernel (not a new class of kernel, not a new
  runtime) — realistically a single `.metal` file change plus whatever
  Rust-side dispatch tweak is needed to invoke it. Consider whether this is
  worth upstreaming to candle directly vs. carrying as a local vendored
  patch.
  Acceptance: decode t/s on the benchmark closes meaningfully toward
  llama.cpp's ratio (not necessarily matching it exactly - llama.cpp has
  years of Apple-specific tuning behind it).

---

## PHASE 3 — MLX-format weight loading, via candle (not the MLX runtime)

Loads a real `mlx-community`-style quantized checkpoint's *weights*, but
executes them through candle's own kernels — matches "one engine, one
`LlmBackend` trait" rather than bolting on Apple's MLX as a second runtime.

- [ ] **3.1 — New loader module `llm-core/src/loader/mlx.rs`.** MLX's affine
  quantization (confirmed via `mlx.core.quantize` docs) packs 4-bit values
  8-per-32-bit-word, with a scale+bias per group (`group_size` default 64,
  32 for higher quality) stored as sibling tensors per weight. Parse this
  layout directly from the repo's `.safetensors` + its quantization config
  (`config.json`'s `quantization` block in MLX-published repos) — this is
  the same shape of work as the existing GGUF/safetensors loaders already
  in `llm-core/src/loader/`, not new kernel research.
  Acceptance: unit test loading a small synthetic MLX-quantized fixture
  (generate one locally with `pip install mlx mlx-lm` — not installed on
  this machine yet, needs adding) and confirming the parsed scale/bias/
  packed-int4 tensors match known values.

- [ ] **3.2 — Re-pack parsed MLX weights into a form candle's existing
  quantized-matmul path can consume** — either dequantize to F32/F16 (reuse
  today's dequant-then-QMatMul pipeline in `candle.rs`, correctness-first)
  or, if group_size=64 affine quant maps cleanly onto candle's own block
  quant representation, convert directly into a `QTensor` and skip the
  dequant round-trip. Start with the dequant path (lower risk, matches
  existing GGUF-non-CUDA-fallback pattern in the codebase already).
  Acceptance: an MLX-published model (e.g. from `mlx-community/`) loads and
  generates correct, coherent text on this Mac.

- [ ] **3.3 — Wire `has_mlx_format` (or similar) detection into
  `llm-core/src/model/config.rs`** alongside the existing bnb/awq/gptq
  `quantization_config` detection, so an MLX repo is recognized at load
  time the same way those are.
  Acceptance: loading an MLX repo on non-Apple hardware gives the same kind
  of clear, explicit error the AWQ/GPTQ detection already gives on non-CUDA
  hardware — no silent wrong-format load.

---

## PHASE 4 — AWQ + GPTQ on CUDA (currently 100% unimplemented — confirmed, not assumed)

**UPDATE 2026-07-20**: 4.1's loaders and a first pass at 4.2's dequant
path are written (`llm-core/src/loader/awq.rs`, `gptq.rs`, wired into
`CandleBackend::load_weights`). Tensor layout confirmed against two real
HF repos' safetensors headers (HTTP range request, no full download).
Bit-unpacking logic has round-trip unit tests and is internally
consistent, but the actual dequantized VALUES have **not** been checked
against a Python (`transformers`/`autoawq`/`auto-gptq`) reference on a
real tensor — do that first, on the GPU machine, before trusting this
for anything real. Also note: this dequantizes fully to F16 at load time
(not per-tile), so it currently gives up AWQ/GPTQ's memory savings too,
not just speed — 4.3's kernel work is still what actually delivers the
format's real benefits.

**Cannot be built or tested in this environment.** No `nvcc`, no NVIDIA GPU,
confirmed this session. This phase's code can only be written here as
scaffolding; it must be compiled, debugged, and perf-tuned on the separate
GPU/CPU machine already used for commit `c7ece93` (or a cloud CUDA
environment, if one becomes available) — same pattern as this session's
existing CUDA-path work, which was reviewed by reading the diff only, never
executed here. Flag every step of this phase honestly as "written, not
verified here" until that happens.

- [ ] **4.1 — Loaders: `llm-core/src/loader/awq.rs`, `gptq.rs`.** Parse the
  real tensor layout from `config.json`'s `quantization_config.quant_method`
  (already detected — currently only to *refuse* loading, in
  `llm-core/src/model/config.rs` — this phase changes that from refuse to
  route-to-a-real-loader for `awq`/`gptq` specifically, `bnb` stays refused
  for now, out of scope): `qweight` (int32, 8x int4 packed), `qzeros`,
  `scales`, `group_size` (typically 128), and for GPTQ optionally `g_idx`
  (act-order variant — support the common non-act-order case first, add
  act-order as a follow-up once the basic path works).
  Acceptance (on the GPU machine): a known AWQ/GPTQ repo's tensors parse
  into the expected shapes/dtypes, checked against the same tensors read by
  Python (`safetensors` + `transformers`) as ground truth.

- [ ] **4.2 — MVP: dequant-then-cuBLAS (correctness first, not peak
  speed).** Dequantize AWQ/GPTQ weights to F16 (on load, or per-tile), run
  through candle's existing CUDA F16 matmul path. This is intentionally
  *not* Marlin-class throughput — it's the fastest way to get a real,
  correct, end-to-end AWQ/GPTQ model generating text on CUDA, matching this
  project's "no silent fallback, model-agnostic" rule (an AWQ model should
  at minimum *run*, even before it's fast).
  Acceptance (on the GPU machine): a real AWQ or GPTQ model loads and
  generates correct output end-to-end via `llm-cli`.

- [ ] **4.3 — Real throughput: a Marlin-class INT4 tensor-core GEMM
  kernel.** This is the actual fix for "AWQ/GPTQ should be fast," and it is
  a genuinely large, separate piece of engineering — not a variant of 4.2.
  Two paths, pick one after 4.2 ships and the team/user has bandwidth to
  commit real time:
  - **(a) Write one from scratch** via `cudarc` (candle's existing CUDA
    glue) or `llm-kernel`'s CubeCL groundwork (currently unwired dead code -
    a handful of standalone kernels, nothing resembling a full backend).
    Highest effort/risk: this reproduces multi-month, expert-tuned
    tensor-core kernel research (`mma.sync` int4 paths, per-architecture
    tile tuning) from scratch.
  - **(b) Vendor a proven, existing Marlin kernel** (e.g. the IST-DASLab/
    vLLM lineage, MIT/Apache-2.0 licensed CUDA/PTX source) behind a small
    `extern "C"` FFI shim, compiled via `build.rs` + `nvcc`, called through
    `cudarc`'s raw-pointer APIs. Lower risk (reuses tuned, battle-tested
    code instead of re-deriving it), but is the project's first non-Rust
    (CUDA C++) build dependency — confirm the user is fine with that before
    starting, and double-check the exact license terms of whichever kernel
    source gets vendored.
  Recommendation: **(b)**, once 4.2 is proven correct. Flag this choice to
  the user explicitly before starting - it's a real project-shape decision
  (vendoring foreign kernel source vs. writing one in-house), not a code
  detail.
  Acceptance (on the GPU machine): AWQ/GPTQ decode throughput approaches
  the ~700+ t/s class of numbers vLLM+Marlin gets on comparable hardware,
  not just "faster than 4.2's dequant path."

- [ ] **4.4 — Hardware-dispatch compliance.** AWQ/GPTQ's packed-int4 layout
  has no sane CPU or Metal execution path without dequantizing everything
  up front (which defeats the point). Loading one when
  `HardwareProfile.backend != Cuda` must hit the existing "missing
  kernel = explicit Err, never silent fallback" rule from CLAUDE.md - bail
  with a message naming the actual constraint ("AWQ/GPTQ requires CUDA;
  this machine's backend is Metal/CPU - use a GGUF or MLX quant of this
  model instead").
  Acceptance: loading an AWQ repo on this Mac produces that exact kind of
  clear error, not a crash or a silent CPU fallback.

---

## PHASE 5 — Downloader: hardware-aware format recommendation

Ties phases 3+4 back into what the user actually asked for two turns ago:
"if the model with the machine's best format is found, suggest it."

- [ ] **5.1 — Extend `llm-cli/src/bin/pull.rs`'s search/list logic** to also
  look for AWQ repos and MLX-format repos (not just GGUF variants) for a
  given model name, surfacing them in `--list` output alongside existing
  GGUF variants.
  Acceptance: `llm pull <model> --list` shows GGUF, AWQ, and MLX variants
  when they exist, each labeled with which backend it targets.

- [ ] **5.2 — Recommendation logic changes from "biggest GGUF that fits" to
  "best FORMAT for this `HardwareProfile.backend` first, then best-fitting
  quant within that format."** Cuda → prefer AWQ (once Phase 4 ships) >
  GGUF. Metal → prefer MLX (once Phase 3 ships) > GGUF. CPU-only → GGUF
  K-quants (unchanged - already the right answer there).
  Acceptance: running `llm pull` for a model that has both a GGUF and an
  MLX repo, on this Mac, recommends the MLX one once 3.x ships (and
  honestly labels it "not yet loadable, falling back to GGUF" if run before
  Phase 3 ships).

---

## Suggested order of attack

1. **Phase 1** now - cheap, no new kernels, tells us how much of the
   problem is even worth chasing with kernel work at all.
2. **Phase 2** - stays entirely on this Mac, fully testable here, directly
   answers the original question ("why isn't GPU faster").
3. **Phase 3** - also fully testable here (once an MLX fixture/model is
   available), no CUDA hardware needed, matches "hardware-agnostic, one
   engine" philosophy.
4. **Phase 4** - the big one, and the one that actually needs "100% not
   written" AWQ/GPTQ kernels built. Start 4.1/4.2 as soon as there's GPU-
   machine time to build and test on; do NOT attempt 4.3 until 4.2 is
   proven correct end-to-end.
5. **Phase 5** - small, ties everything together for the user-facing
   `llm pull` experience; do last since it depends on 3/4 existing.
