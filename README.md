# llm-rs (AgnosticEngine)

A Rust LLM inference runtime aiming for one codebase that runs on CUDA,
Metal/Apple Silicon, and CPU (x86 + ARM), auto-detects the model architecture
from its config/tensor names, and dispatches to the right hardware backend at
runtime — no per-model or per-hardware special-casing outside a few explicit,
documented exceptions.

This README describes what's actually implemented and verified today, not
the aspirational end state in `goal.md`. Where the two differ, this file
follows the code and `PROGRESS.md`/`CHANGELOG.md` (which record what was
actually built and tested, including honest gaps). See `CLAUDE.md` for the
contributor-facing rules this project holds itself to.

## Status at a glance

- **CPU** (x86 AVX2/AVX-512, ARM NEON): verified.
- **Metal** (Apple Silicon): verified, numerically matches CPU output.
- **CUDA** (NVIDIA): verified (RTX 2000 Ada, 8GB VRAM) — real, coherent
  generation confirmed for GGUF, plain HF safetensors, AWQ, and GPTQ models.
- **Vulkan**: not implemented.
- Text generation across Llama/Qwen2/Qwen2.5/Qwen3/Gemma-3/Gemma-4 families:
  verified with real checkpoints, real prompts, real generated text — not
  just unit tests.
- Multimodal (vision + audio): vision works for Gemma-4; Qwen2-VL vision is
  missing 2D RoPE for its encoder. Audio is **not yet fully coherent** on
  either architecture — a known, tracked gap, not silently claimed as done.
- Qwen3.5 ("Gated DeltaNet" hybrid SSM+attention) is **not implemented** —
  researched and precisely scoped in `PROGRESS.md` for a future session.

If you're deciding whether this is ready for your use case, read
`PROGRESS.md` and `CHANGELOG.md` before trusting a claim not repeated here —
this project's own convention is to log honestly, including regressions and
things that only real hardware testing (not unit tests) caught.

## Crate layout

| Crate | Role |
|---|---|
| `llm-core` | The engine: `LlmBackend` trait, model loaders (GGUF/safetensors/AWQ/GPTQ/MLX), the Candle-based backend (`backends/candle.rs`), hardware detection (`profile/`), tokenizer, sampler, chat templates. |
| `llm-scheduler` | Serving engine: paged KV-cache block allocator, prefix cache, continuous-batching scheduler, the `ServingEngine` background task. |
| `llm-cli` | CLI binaries and an Axum-based OpenAI-compatible HTTP server (SSE streaming). |
| `llm-cluster` | Multi-node distributed inference (TCP-based today, not the USB/Zenoh mesh `goal.md` describes). |
| `llm-kernel` | CubeCL GPU-kernel scaffolding. Not currently wired into the real inference path — see "Why no custom GPU kernels yet" below. |
| `llm-ffi` | C-ABI bindings for embedding llm-rs in non-Rust hosts. |
| `llm-py` | PyO3 Python bindings with a vLLM-style `LLM`/`generate()` API. Has its own `README.md`/build flow (see below). |
| `llm-tests` | Cross-crate integration/regression tests. |

### Why no custom GPU kernels yet

`llm-kernel` (CubeCL-based attention/gemv/rmsnorm/rope/silu kernels) exists
but isn't used by real inference — everything currently dispatches through
Candle's own CUDA/Metal/CPU device backends. A CubeCL attention kernel was
investigated and rejected with real measurements: at this project's current
per-operator-per-layer graph granularity, the interop tax between Candle and
CubeCL measured 12.1x versus Candle's own equivalent op — enough to erase
the benefit before any kernel math runs. See `CHANGELOG.md`'s "part 13"
entry for the full writeup. Performance work instead targets Candle's own
kernel ecosystem (e.g. switching to its fused `rms_norm` kernel gave a real,
measured 52% decode throughput improvement on Gemma-4).

## Supported model formats

- **GGUF** — the primary format; the full GGML quant family (Q4_K_M, Q8_0,
  K-quants, IQ-quants, F16/BF16) via `candle_core::quantized`.
- **HF safetensors** — plain (unquantized) checkpoints.
- **AWQ** and **GPTQ** — 4-bit packed safetensors, dequantized to dense F16
  at load time (correctness-first; this does not yet capture the formats'
  memory-bandwidth benefits — see `quant-performance-plan.md` phase 4.3 for
  the real tensor-core-kernel work that would). **CUDA-only**: loading an
  AWQ/GPTQ model on a non-CUDA `HardwareProfile` fails with a clear error
  rather than silently degrading.
- **MLX affine-quantized safetensors** — e.g. `mlx-community/*` checkpoints,
  dequantized at load and routed through the normal execution path.

## Hardware detection

`HardwareProfile::get()` (`llm-core/src/profile/`) runs once at startup and
picks CUDA, Metal, or CPU based on what's actually present (GPU VRAM/unified
memory, system RAM, CPU core count and SIMD capability) — never a
compile-time-only choice. A missing kernel or unsupported combination is an
explicit error, never a silent fallback to a slower or different backend
(set `LLM_ALLOW_CPU_FALLBACK=1` to opt into a CPU fallback instead of erroring).

Useful environment variables:

| Variable | Effect |
|---|---|
| `LLM_FORCE_CUDA` / `LLM_FORCE_CPU` | Force backend selection, skipping auto-detection. |
| `LLM_EXPLICIT_DEQUANTIZE` | Same as `--explicit-dequantize`: dequantize GGUF weights eagerly instead of on-demand. |
| `LLM_USE_VRAM_EMBEDDINGS` | Same as `--use-vram-embeddings`: keep embedding tables on GPU instead of system RAM. |
| `LLM_KV_DTYPE` | KV-cache storage dtype (`f16`, `q8`, `q4`). |
| `LLM_ALLOW_CPU_FALLBACK` | Opt into a silent CPU fallback instead of erroring on a backend/kernel mismatch. |
| `LLM_PROFILE_STEP=1` | Print a per-operator-category timing breakdown for one forward pass (matmul/norm/rope/attention/...) — used throughout this project's own perf work. |

## Building

No `build.rs`/special setup required beyond a working Rust toolchain (and,
for GPU features, a working CUDA toolkit or macOS with Metal).

```bash
# CPU only (default)
cargo build --release --workspace --exclude llm-py

# CUDA
cargo build --release --features cuda --workspace --exclude llm-py

# Metal (macOS)
cargo build --release --features metal --workspace --exclude llm-py
```

`llm-py` (Python bindings) builds separately via `maturin` — see
`llm-py/README.md`.

## Quick start

Download a model (public HF repos only; picks a quant matching your
detected hardware if you don't specify one):

```bash
cargo run --release --bin pull -- Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

Chat with it interactively (supports inline `/image <path>` and
`/audio <path>` for multimodal models):

```bash
cargo run --release --bin chat -- \
  --model-path <path/to/model.gguf> \
  --tokenizer-path <path/to/tokenizer.json>
```

One-shot generation for scripting:

```bash
cargo run --release --bin run_model -- \
  --model-path <path/to/model.gguf> \
  --tokenizer-path <path/to/tokenizer.json> \
  --prompt "Explain the theory of relativity in 2 sentences."
```

Run an OpenAI-compatible HTTP server:

```bash
cargo run --release --bin llm-cli -- \
  --model-path <path/to/model.gguf> \
  --tokenizer-path <path/to/tokenizer.json> \
  --port 8080
```

Check what hardware was detected without loading a model:

```bash
cargo run --release --bin devices
```

Benchmark decode/prefill throughput:

```bash
cargo run --release --bin benchmark_speed -- \
  --model-path <path/to/model.gguf> \
  --tokenizer-path <path/to/tokenizer.json>
```

All binaries share `--explicit-dequantize`, `--use-vram-embeddings`, and
`--mmproj-path <path>` (for a model's separate multimodal projector file)
where applicable.

## Testing

```bash
cargo test --workspace --exclude llm-py
cargo test --workspace --exclude llm-py --features cuda   # on CUDA hardware
```

`scripts/hardware_check.sh` is this project's end-to-end verification
script — it builds with the right feature flag, confirms hardware detection,
runs a real text-generation smoke test (downloading a model via `pull` if
needed), optionally exercises vision/audio multimodal paths, and writes a
timestamped report under `hwcheck-results/`. It exists because this
project's own history shows unit tests alone have repeatedly missed real
bugs that only a live forward pass against a real checkpoint caught — see
`PROGRESS.md` for several concrete examples.

## Contributing / project conventions

See `CLAUDE.md` for the rules this codebase holds itself to: one
`HardwareProfile` decision point (no hardcoded backend branching), one
`LlmBackend` trait, no silent fallback on a missing kernel, no `unwrap()` in
library code, and no `unsafe` without a `// SAFETY:` comment. `PROGRESS.md`
is the living record of what's been built, verified, and found broken —
read it before assuming a feature works as described elsewhere.

## License

No license file is currently present in this repository (the `llm-py`
Python package's `pyproject.toml` names MIT, but there is no corresponding
top-level `LICENSE` backing that yet). Treat this repository as unlicensed
until that's resolved.
