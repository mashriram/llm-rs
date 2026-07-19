# llm-rs: Full Execution Plan (CPU → GPU → Universal Hardware)

This is a blindly-followable, ordered task list built directly from a file-by-file audit of the `vision-stability` and `cpu` branches. Every item references the actual file/line where the problem lives. Do them in order — later phases assume earlier ones are done. Each phase ends with an acceptance test; do not start the next phase until the current one's acceptance test passes.

---

## PHASE 0 — Stop the bleeding (do this before anything else)

These are things that will actively break builds or crash a running server. Fix first, no judgment calls needed.

- [x] **0.1 — Remove absolute machine-specific paths.**
  `vision-stability` branch's `llm-core/Cargo.toml` has `[[bin]]` entries pointing at `/home/mukundan/.gemini/antigravity/brain/<uuid>/scratch/...`. This will not build on any machine but the original author's. Confirm the `cpu` branch's relative-path version (`../scratch/...`) is what gets merged forward, and grep the whole repo for `/home/` to make sure no other absolute paths snuck in anywhere (Cargo.toml, build.rs, tests).
  ```
  grep -rn "/home/" --include="*.toml" --include="*.rs" .
  ```
  Acceptance: `grep` returns nothing outside of comments/log strings.

- [x] **0.2 — Replace poisoning `Mutex` with a non-poisoning lock everywhere in `candle.rs`.**
  Every `.lock().unwrap()` in `llm-core/src/backends/candle.rs` (image/audio path cache, dequant cache, qmatmul cache, visual embeddings cache — around lines 1244–1676) will permanently wedge the server if any thread ever panics while holding the lock. Switch `std::sync::Mutex` → `parking_lot::Mutex` (drop-in API, doesn't poison, slightly faster) across the crate.
  - Add `parking_lot = "0.12"` to `llm-core/Cargo.toml`.
  - `sed -i 's/std::sync::Mutex/parking_lot::Mutex/' llm-core/src/backends/candle.rs llm-core/src/backends/mod.rs` then remove now-unneeded `.unwrap()` after `.lock()` calls (parking_lot's `lock()` returns the guard directly, not a `Result`).
  Acceptance: `cargo build` succeeds with zero `.lock().unwrap()` occurrences left in `llm-core`.

- [x] **0.3 — Make the GGUF byte parser fail gracefully instead of panicking.**
  `llm-core/src/loader/gguf.rs` lines ~66, 75, 84, 93, 119, 123 do `try_into().unwrap()` on raw byte slices read from disk. A truncated or corrupted GGUF file (which end users *will* upload) currently panics the whole process instead of returning an error. Replace every one with:
  ```rust
  self.data[self.pos..self.pos+4].try_into()
      .map_err(|_| anyhow!("GGUF file truncated at offset {}: expected 4 bytes", self.pos))?
  ```
  and propagate the `Result` up (the function should already return `Result<T>`; if it currently doesn't, change its signature).
  Acceptance: feed the loader a hand-truncated `.gguf` file (`head -c 100 model.gguf > truncated.gguf`) and confirm `llm serve truncated.gguf` prints a clean error and exits, instead of a Rust panic/backtrace.

- [x] **0.4 — Fix the VRAM headroom comment/code mismatch.**
  `llm-core/src/profile/mod.rs`, `choose_device()`: docstring says "15% safety headroom," code does `estimated_bytes as f64 * 1.01` (1%, not 15%). Decide the real number you want (1.15 is safer for KV-cache growth headroom) and make the comment match the code exactly:
  ```rust
  let required = (estimated_bytes as f64 * 1.15) as u64; // 15% headroom for KV cache growth
  ```
  Acceptance: no docstring in the repo states a percentage that doesn't match the literal in the code below it (grep for `%` in comments near `.choose_device`, `.query_free_vram`, and any other budget-calculation function and verify each).

- [x] **0.5 — Audit the remaining ~30 `.unwrap()` calls in `llm-core/src/sampler.rs` and `llm-core/src/graph/builder.rs`.**
  `sampler.rs` (`partial_cmp(&a.0).unwrap()`) will panic on NaN logits — which quantized models *do* occasionally produce on malformed input. Replace with `.unwrap_or(std::cmp::Ordering::Equal)` and log a warning once (not per-token) if a NaN is ever seen, since repeated NaNs indicate a real numerical bug upstream that silent-ignoring would hide.
  `graph/builder.rs` lines 52–54 (`per_layer_token_embd.clone().unwrap()` etc.) assume optional per-layer projection tensors are always present once one is found — verify this invariant is actually guaranteed by the calling code, and if so replace `.unwrap()` with `.expect("invariant: per_layer_* group means all three fields are populated together — see call site in X")` so a future violation fails with a message instead of a bare panic location.
  Acceptance: `grep -rn "\.unwrap()" llm-core/src --include=*.rs | grep -v "tests\|#\[test\]\|conv_template.rs"` returns 0 lines outside sampler.rs/builder.rs's now-documented `.expect()`s.

---

## PHASE 1 — CPU correctness (the actual "cpu branch" work, finished properly)

The current `cpu` branch fixed the vision encoder's dtype handling but left the root cause of "vision instability" unaddressed: a hardcoded patch count. Do these in order.

- [x] **1.1 — Kill the hardcoded `576`-patch image placeholder.**
  `llm-cli/src/bin/chat.rs`, in the vision-encoder block: `"<|image_pad|>".repeat(576)`. This must become a function of the *actual* image passed in:
  1. Compute `(h, w)` the vision encoder will use for a given input image at whatever resize/patchify step happens before `VisionEncoder::forward`.
  2. Compute `num_patches = h * w` post-`spatial_merge_size` exactly the way `vision.rs::forward` computes it (reuse the same formula, don't re-derive it — extract it into a shared helper `fn compute_num_patches(img_h, img_w, patch_size, spatial_merge_size) -> usize` in `vision.rs` and call it from both places).
  3. Replace the literal `576` with this computed value.
  Acceptance: run the same chat session with two different image resolutions and confirm the number of `<|image_pad|>` tokens inserted differs and matches the vision encoder's actual output embedding count (add a debug assert: `assert_eq!(pad_count, vision_embeds.dim(1)?)` in debug builds).

- [x] **1.2 — Confirm the `spatial_merge_size` default change (2→1) is per-architecture, not global.**
  `vision.rs` line ~39 changed the fallback default from `2` to `1`. Trace every model family you actually test against (Qwen2-VL, Gemma-4/SigLIP variants, whatever else) and confirm each one's `config.json`/GGUF metadata explicitly overrides this value. If any model relies on the bare default, verify `1` is correct for it specifically — don't leave it as "we changed a magic number and errors went away," because that's exactly the kind of fix that silently breaks a *different* model that depended on the old default of `2`.
  Add an explicit table as a code comment above the default:
  ```rust
  // Known spatial_merge_size by family (verified against upstream configs):
  //   Qwen2-VL family: 2
  //   Gemma-4 vision (SigLIP-derived): 1
  //   default when metadata absent: 1 (safe: produces MORE tokens than needed, never fewer)
  ```
  Acceptance: a short integration test per known vision-model family asserting the merge size read from metadata matches the documented value.

- [x] **1.3 — Extend the CPU dtype fix (F32-on-CPU) beyond the vision encoder into the text decoder path.**
  The `cpu` branch only changed `vision_dtype` in `vision.rs`. Check `candle.rs`'s main decoder forward pass for the same F16-hardcoded-regardless-of-device pattern — grep:
  ```
  grep -n "DType::F16" llm-core/src/backends/candle.rs
  ```
  For every hit, apply the same `if device.is_cpu() { F32 } else { F16 }` pattern (or better, a single `self.compute_dtype` field set once at load time from `HardwareProfile`/device, instead of repeating the ternary at every call site).
  Acceptance: `llm serve <model>.gguf --device cpu` produces output that numerically matches (`cargo test --test parity`, tolerance 1e-3) the GPU path on the same model, for both text-only and vision-enabled models.

- [x] **1.4 — Add real CPU SIMD dispatch (the spec promised this; nothing exists yet).**
  Candle already has AVX2/NEON kernels internally for F32 ops, so you likely get *some* SIMD for free through Candle. What's missing per the spec's Phase 3 is *detecting and reporting* what's available so users/logs know why CPU inference is slow on old hardware. Add to `llm-core/src/profile/mod.rs`:
  ```rust
  pub struct CpuSimdCaps {
      pub avx2: bool,
      pub avx512f: bool,
      pub neon: bool,
  }
  pub fn detect_cpu_simd() -> CpuSimdCaps {
      CpuSimdCaps {
          avx2: is_x86_feature_detected!("avx2"),
          avx512f: is_x86_feature_detected!("avx512f"),
          neon: cfg!(target_arch = "aarch64"),
      }
  }
  ```
  Log this once at startup (`llm devices` command and server boot log). This is diagnostic, not a new kernel path — don't build custom SIMD kernels yet, that's premature given Candle already handles it; just surface what's active so bug reports are debuggable.
  Acceptance: `llm devices` prints CPU SIMD capability alongside the existing CUDA/Metal detection.

- [x] **1.5 — CPU memory-budget check (currently only VRAM is budgeted).**
  `profile/mod.rs::choose_device` only guards against exceeding VRAM. Add the equivalent guard for pure-CPU runs: compare `estimated_bytes` against `system_ram_free_bytes` with the same 15%-headroom logic, and refuse to load (clean error, not an OOM-kill) if a model won't fit in RAM.
  Acceptance: attempt to load a model sized larger than available RAM on a constrained container; confirm a clean `anyhow` error instead of the OS OOM-killing the process.

**Phase 1 acceptance gate:** parity test suite passes on CPU for at least one text-only model and one vision-enabled model, at both a small and large image resolution, with zero panics on a 30-minute fuzz run feeding truncated/malformed GGUF and image files.

---

## PHASE 2 — GPU (CUDA) correctness and completeness

The CUDA path is currently a thinner, less-tested twin of the CPU path — it works, but several of the spec's stated GPU guarantees aren't actually implemented.

- [x] **2.1 — Verify the Metal fallback path is actually tested, not just compiled.**
  `candle.rs::load_weights` already attempts `Device::new_metal(0)` when CUDA isn't available (line ~626) — this exists in the code today but there's no evidence anywhere in the test suite that it's ever been run against a real Apple GPU. Before doing anything else GPU-related, get this path onto real Apple Silicon hardware and run the parity suite against it. If it fails, fix it before layering more platform work on top of an unverified foundation.
  Acceptance: `cargo test --test parity` green on an actual M-series Mac using the Metal device path (not CPU fallback).

- [x] **2.2 — Replace the fake VRAM query with a real one.**
  `profile/mod.rs::detect()`: when `nvidia-smi` fails, the code currently *assumes* 8GB total / 6GB free (`"Assuming default 8GB VRAM"`). This is a silent correctness landmine — a 24GB card with a broken `nvidia-smi` (common in containers without the right mounts) will be told it only has 8GB and refuse to load models it could actually run, while a 4GB card in the same broken state will be told it has 8GB and OOM. Replace the guess with:
  1. First try `nvidia-smi` (existing).
  2. If that fails, try `candle_core::cuda_backend`'s device properties query directly (CUDA driver API total/free mem) instead of shelling out — this avoids the whole "assume 8GB" problem since it doesn't depend on `nvidia-smi` being on PATH.
  3. Only if *both* fail, refuse to select CUDA at all and fall back to CPU with a clear warning — do not guess a number.
  Acceptance: run in a container with `nvidia-smi` deliberately removed from PATH; confirm the profiler either gets a correct VRAM figure via the driver API or falls back to CPU, never silently guesses.

- [ ] **2.3 — Decide the fate of `llm-kernel` (CubeCL JIT kernels) — do not leave it orphaned.**
  Right now `llm-kernel` (gemv, attention, rope, rmsnorm, silu — all real CubeCL code) is depended on by nothing. This is actively misleading: anyone reading the workspace `Cargo.toml` believes custom JIT kernels power inference; they don't. Go with below implement properly 
  - **Option  (recommended if you want the ">25% vs llama.cpp" claim to be true):** Wire `llm-kernel` in as an *optional* backend behind a feature flag (`cubecl-backend`), implementing the same `LlmBackend` trait as `CandleBackend`. Start with just `q8_gemv` replacing Candle's quantized matmul on the decode hot path, parity-test it against Candle's output, and only then benchmark it. Do not attempt all 9 kernels at once — one kernel, proven correct and faster, before the next.
  

- [ ] **2.4 — Multi-GPU: currently unimplemented, spec doesn't even claim it, but it's the natural Phase 2 GPU item.**
  Nothing in the current code selects among multiple CUDA devices (`Device::new_cuda(0)` is hardcoded to device 0 everywhere it's called). Before doing anything distributed (Phase 6 cluster mesh), get single-node multi-GPU tensor parallelism working using the existing `llm-cluster/src/tensor_parallel.rs` sharding math (which is real, just unused) — this validates that code path locally before you ever add a network in between.
  Acceptance: a model too large for one GPU's VRAM but fitting across two loads and runs correctly, sharded via `shard_col_parallel`/`shard_row_parallel`, on a single machine with 2 GPUs.

- [ ] **2.5 — KV-cache quantization (Rotor rotation) — currently spec-only, not implemented anywhere in the repo.**
  If long-context GPU throughput matters to you, this is real, scoped work: implement the orthogonal-rotation-before-INT4-packing scheme the spec describes for the KV cache in `llm-scheduler/src/block_allocator.rs`. Do this only after Phase 2.1–2.4 are solid; it's a targeted optimization, not a correctness fix, so it should not jump the queue.
  Acceptance: 4K+ token context requests show measurably lower perplexity at INT4 KV quantization with rotation vs. without, on the same model/prompt.

**Phase 2 acceptance gate:** CUDA and Metal both pass the full parity suite; `llm-kernel` is either wired in and proven, or explicitly retired; VRAM detection never silently guesses.

---

## PHASE 3 — Cross-cutting hygiene (do alongside Phases 1–2, not after)

- [ ] **3.1 — Get a Rust toolchain into actual CI.** There is currently no evidence this repo is compiled in CI (no toolchain was even available in the audit sandbox). Minimum bar: `cargo check --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo test --workspace`, and `cargo miri test -p llm-core` (per your own Agent Rule about memory safety) on every push, for at least CPU + CUDA feature combinations.
- [ ] **3.2 — Split `candle.rs`.** At ~1900 lines doing loading, weight caching, KV cache, attention, sampling glue, and multimodal fusion, this file is your biggest ongoing maintenance risk. Break it into `weights.rs` (loading/caching), `attention.rs` (KV cache + attention forward), `multimodal.rs` (image/audio state), keeping `candle.rs` as the thin `LlmBackend` trait impl that calls into the others. Do this after Phase 1/2 fixes land (don't refactor and fix bugs in the same diff).
- [ ] **3.3 — Extend `docs/mlc-contracts.md`-style invariant capture to your own scheduler and sampler.** You've already done the hard reverse-engineering work against mlc-llm's engine; do the equivalent pass writing down what *your* `scheduler.rs`, `prefix_cache.rs`, and `sampler.rs` actually guarantee today, then diff that against the mlc-llm invariants you already documented. Gaps found there are real prioritized work items, not guesses.

---

## PHASE 4 — Universal hardware expansion (macOS/MLX, Android, Qualcomm, WASM)

Do not start this phase until Phases 0–3 are green. Expanding to 4 more platforms on top of an unstable 2-platform (CPU/CUDA) foundation multiplies your bug surface for no benefit — every platform-specific backend inherits every unresolved bug in the shared code above it.

### 4.1 — macOS / Apple Silicon — MLX backend
Metal-via-Candle already exists (2.1) but is unverified; MLX is a *separate, additional* backend option, not a replacement, since MLX gets you Apple's unified-memory-aware kernels that Candle's Metal backend doesn't have.
- [ ] Add an `llm-mlx` crate behind a `mlx` feature flag, implementing `LlmBackend`, using the `mlx-rs` bindings (or FFI directly to Apple's `mlx` C++/Swift if bindings are immature — check current state before committing).
- [ ] Reuse the exact same GGUF/SafeTensors loader from `llm-ingest`/`llm-core` — do not write a second model loader. MLX backends should only differ at the "run this op on this device" layer.
- [ ] Parity-test against the Candle-Metal path and Candle-CPU path on the same M-series machine before trusting MLX numbers.
- [ ] Only after parity: benchmark MLX vs Candle-Metal for your target model sizes; keep whichever wins as default, keep both behind flags either way (some users will want Candle for x86 Mac / Metal driver quirks).
Acceptance: same parity suite green on MLX backend; `llm devices` correctly reports `mlx` as available and its measured (not assumed) unified memory size.

### 4.2 — Android — CPU baseline first, then NPU
- [ ] **Baseline:** cross-compile the existing CPU (Candle) backend for `aarch64-linux-android` via `cargo-ndk`. This alone gets you a working, if slow, Android target with zero new backend code — do this before any NPU work, since it's your fallback and your correctness baseline on ARM.
- [ ] Verify NEON is actually engaged (tie into the `CpuSimdCaps` from 1.4 — `neon: cfg!(target_arch = "aarch64")` should read `true` on-device; confirm via a log line in a real APK/JNI test harness, not just the build).
- [ ] Expose the runtime via JNI (`llm-ffi` crate already exists and is a sane place for this — check what it currently does before adding to it, don't duplicate).
- [ ] **NPU tier (Qualcomm Hexagon / QNN):** this is a genuinely separate backend, not a flag on the CPU one. Add an `llm-qnn` crate implementing `LlmBackend` against Qualcomm's QNN SDK, gated behind a `qnn` feature, targeting INT8/INT4 quantized models specifically (Hexagon NPUs are quantization-first hardware — don't try to run F16/F32 models through it).
- [ ] Do NOT attempt a generic "Android NPU" abstraction spanning Qualcomm/MediaTek/Samsung NPUs in one crate. Ship Qualcomm QNN first, prove it end-to-end on one real device family, and only generalize once you have two working NPU backends to abstract over — a single-example abstraction is usually wrong.
Acceptance: CPU-baseline parity suite passes cross-compiled on an actual Android device (not just an emulator); QNN backend passes parity on at least one quantized model on one real Qualcomm-NPU device before calling it supported.

### 4.3 — Windows / generic GPU — Vulkan
- [ ] Candle has an experimental Vulkan path via `wgpu`; evaluate its current maturity directly (test it) rather than trusting the spec's claim that Vulkan support is a checkbox — this is exactly the kind of gap the earlier audit found in `llm-cluster` and `llm-kernel`, so verify before promising it.
- [ ] If Candle's Vulkan path is too immature, this becomes an `llm-kernel`(CubeCL)-via-`wgpu` job instead — CubeCL already supports a wgpu target, which is the actual reason to finish Phase 2.3 Option A rather than deleting `llm-kernel`. Cross-reference that decision now, before committing to Vulkan work, since it changes which crate does the work.
Acceptance: parity suite green on at least one non-NVIDIA, non-Apple GPU via Vulkan.

### 4.4 — Browser / WASM (lowest priority, mention only for completeness)
- [ ] Only pursue this once 4.1–4.3 are stable. Candle has WASM support for small CPU models already; this is a packaging exercise (wasm-bindgen + a size-constrained model), not new inference logic. Do not build custom kernels for this target — reuse whatever CPU path already exists.

---

## Order-of-operations summary (the one-paragraph version)

Phase 0 (crash fixes) → Phase 1 (CPU correctness, especially the image-token-count bug) → Phase 2 (CUDA/Metal correctness + decide `llm-kernel`'s fate) → Phase 3 (CI + file-splitting, ongoing) → Phase 4 (MLX, then Android CPU baseline, then Qualcomm QNN, then Vulkan, then WASM last). Each phase's acceptance gate must pass before the next phase starts. Do not let platform breadth (Phase 4) get ahead of platform depth (Phases 0–2) — a five-platform inference runtime that's subtly wrong on all five is worse than a two-platform one that's provably correct.
