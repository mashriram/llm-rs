# Changelog

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
