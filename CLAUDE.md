# AgnosticEngine — Project Context

## What this is
Rust LLM inference runtime. Target: ONE branch, v1-unified, that runs on
CUDA, Metal/MLX (Apple Silicon, ARM), and CPU (x86+ARM), and auto-classifies
any HF/GGUF model (autoregressive/MoE/diffusion/multimodal). Production v1,
not a demo. Spec: goal.md. Task breakdown: v1-plan.md.

## Hardware-aware dispatch rule (read this before touching backend code)
- ONE HardwareProfile detected at runtime picks the backend. No compile-time
  "only builds for CUDA" branching in the final binary — all backends are
  compiled in (feature-gated is fine for build size, but the SELECTED one at
  runtime is a HardwareProfile decision, never hardcoded).
- ONE LlmBackend trait. candle.rs (CPU), cubecl.rs (CUDA/Vulkan JIT), and the
  MLX backend all implement it. Scheduler/router code must never match on
  "which backend am I" — that defeats the abstraction.
- Missing kernel on a given backend = explicit Err, never a silent fallback
  to a different (slower/wrong) backend.

## Rules for this repo
- Base branch: v1-unified, created from cpu.
- One task from PROGRESS.md at a time. Do not start the next task in the
  same session unless explicitly told to continue.
- After finishing a task: update PROGRESS.md, commit with a clear message,
  and STOP. Report what changed and what's next.
- Large output (build logs, diffs, test results) goes to a file on disk
  (`/tmp/...` or scratchpad), not into context. Read/report only the
  relevant slice.
- No `unwrap()` in library code. No `unsafe` without a `// SAFETY:` comment.
- No silent fallback: if a backend/kernel/model type isn't supported, error
  clearly — don't quietly degrade.
- One canonical ModelMeta / KvDtype / LlmBackend definition — no more
  per-branch drift.

## Actual crate layout (differs from goal.md's ae-* naming — real crates use llm-* prefix)
- llm-core: LlmBackend trait, types, backends/ (candle etc), model/, graph/, loader/, profile/
- llm-kernel: attention, gemv, rmsnorm, rope, silu
- llm-scheduler: block_allocator, engine, prefix_cache, scheduler
- llm-cluster: analyzer, collective, moe, pipeline, profiler, recovery, tensor_parallel
- llm-cli: server/CLI entry point
- llm-ffi: FFI bindings

## Current phase
See PROGRESS.md — "Current task" is always the source of truth.
