# Changelog

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
