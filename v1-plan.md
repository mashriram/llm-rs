# AgnosticEngine → v1: Claude Code Execution Plan

**Goal:** one branch, `v1-unified`, created from `cpu` and merged with `mlx` and `vision-stability` (plus any other branches you have), that is:

- **Hardware-agnostic, hardware-aware** — runs on CUDA, Apple Silicon/MLX, plain CPU (x86 + ARM), and whatever else `ae-profile` can detect. Not "pick a backend at compile time" — one binary that detects hardware at startup and dispatches to the right backend, per your own architecture doc.
- **Model-agnostic** — any HF/GGUF model auto-classifies through the paradigm router (autoregressive / MoE / diffusion / multimodal), no per-model code.
- **Production v1** — not a demo. Tests pass, no silent fallbacks, error handling is real, CLI works end to end, benchmarks meet the acceptance criteria in your spec.

Done with Claude Code, structured so no session runs out of context mid-task — every task is small, self-contained, and checkpointed to disk so a cutoff never loses work.

---

## 0. What "hardware-agnostic but hardware-aware" means operationally

This is worth being explicit about before Claude Code touches anything, because it's easy to accidentally build "three separate binaries" instead of "one binary, three code paths":

- **One `HardwareProfile` struct, populated at runtime** (per Phase 3 of your spec) — it detects CUDA / Metal / Vulkan / CPU SIMD and picks `best_backend` automatically. This already exists in your `ae-profile` design; the merge must preserve it as *the* dispatch point, not something any branch bypasses.
- **One `LlmBackend` trait, multiple implementations** — `candle.rs` (CPU reference), `cubecl.rs` (CUDA/Vulkan via JIT), and an MLX implementation for Apple Silicon. All three implement the same trait. The scheduler and paradigm router never know which one they're talking to.
- **No `#[cfg(feature = "cuda")]`-only code paths that silently no-op on other hardware.** If a kernel isn't implemented for a backend yet, it must return a clear `Err`, not fall through to a slower/wrong path silently. This matters a lot for "production" — silent fallback is how you ship something that looks like it works on your Mac and then breaks in prod on a CUDA box.
- **Model-agnostic is a Phase 2 concern, not new work** — your paradigm router already classifies MoE/diffusion/multimodal/autoregressive from tensor names + config. The merge just needs to make sure this classifier isn't accidentally hardcoded to assumptions from one branch (e.g. vision-stability may have baked in assumptions about which backend runs the vision encoder — that needs to become backend-agnostic too).

Put this section (or a link to it) at the top of `CLAUDE.md` so every session is anchored on it.

---

## 1. The anti-token-exhaustion setup (do this once, first)

Create two files at the repo root. These make everything below resumable across sessions.

### `CLAUDE.md`

```markdown
# AgnosticEngine — Project Context

## What this is
Rust LLM inference runtime. Target: ONE branch, v1-unified, that runs on
CUDA, Metal/MLX (Apple Silicon, ARM), and CPU (x86+ARM), and auto-classifies
any HF/GGUF model (autoregressive/MoE/diffusion/multimodal). Production v1,
not a demo.

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
  (`/tmp/...`), not into context. Read/report only the relevant slice.
- No `unwrap()` in library code. No `unsafe` without a `// SAFETY:` comment.
- No silent fallback: if a backend/kernel/model type isn't supported, error
  clearly — don't quietly degrade.
- One canonical ModelMeta / KvDtype / LlmBackend definition — no more
  per-branch drift.

## Current phase
See PROGRESS.md — "Current task" is always the source of truth.
```

### `PROGRESS.md`

```markdown
# Progress

## Current task
STEP 0 — not started

## Completed
(nothing yet)

## Known blockers / open questions
(none yet)

## Struct conflict log
(populated during Task 1)

## Backend dispatch audit
(populated during Task 2 — where does each branch decide which backend to
use? does anything hardcode it instead of asking HardwareProfile?)
```

First message to Claude Code, verbatim:

> Read CLAUDE.md and PROGRESS.md. Confirm you've read them and tell me what "Current task" says. Don't do anything else yet.

---

## 2. Why this won't stall mid-task

1. **Small tasks.** Each task below is scoped to ~20–40 min of work — one branch's compile fixes, one dispatch audit, one benchmark run. Never "merge everything and make it all work" as one task.
2. **Externalize big output.** Full `cargo build`/`git diff`/test logs go to files; Claude Code reads slices, not the whole thing, into context.
3. **Commit-and-stop.** Every task ends in a commit + PROGRESS.md update *before* Claude Code reports back. Worst case on a hard cutoff: you lose the last chat message, never actual work.

Resuming after any interruption is just: "continue" — it re-reads PROGRESS.md.

---

## 3. The task list

Hand these to Claude Code **one at a time**, review the stop-and-report points, then move on.

### Task 1 — Branch audit (read-only)

```
Task 1: Audit cpu, mlx, vision-stability (and list any other branches you
find — don't assume these are the only three) without merging anything.

1. `git branch -a` to confirm the full branch list. Report anything besides
   cpu/mlx/vision-stability that exists.
2. Diff ModelMeta, KvDtype, and the LlmBackend trait definitions specifically
   across all branches (grep for the type names first, diff just those
   files — not a full repo diff). Redirect full output to
   /tmp/branch-audit.txt, read only the relevant hunks.
3. For each branch, find where it decides which backend to run on (search
   for backend selection / device selection code). Log in PROGRESS.md under
   "Backend dispatch audit": does it go through HardwareProfile, or is
   something hardcoded/branch-specific?
4. Write the struct conflict log in PROGRESS.md: fields that differ or are
   missing across branches, backends that exist only on one branch.
5. Do NOT modify code. Audit only.
6. Update PROGRESS.md current task to "Task 2 — create v1-unified + merge
   canonical types", commit the PROGRESS.md update, stop.
```

### Task 2 — Create `v1-unified` + canonical types

```
Task 2: Create the target branch and unify the type layer.

1. `git checkout cpu && git checkout -b v1-unified`.
2. Using the struct conflict log from Task 1, define ONE canonical
   ModelMeta, KvDtype, and LlmBackend trait — a superset of all branches'
   fields. Backend-specific fields become Option<T> with a comment on which
   backend populates them.
3. Define/confirm ONE HardwareProfile-driven dispatch point (per CLAUDE.md's
   hardware-aware rule) — if none of the branches actually have this wired
   correctly yet, this is where you build it, not defer it.
4. Do not port mlx or vision-stability backend code yet — that's Tasks 3-4.
   This task only lands the type layer + dispatch point on v1-unified.
5. `cargo check -p ae-core` (or equivalent). Long errors → file, summarize
   categories only.
6. Commit to v1-unified. Update PROGRESS.md: Task 2 done, list every
   Option<T> decision and the dispatch point design for my review, set
   current task to "Task 3 — merge cpu backend".
7. Stop. I want to review the Option<T> decisions and dispatch design before
   you port any backend.
```

### Task 3 — Merge `cpu` backend into `v1-unified`

```
Task 3: Get the CPU (candle) backend fully working against the canonical
types and dispatch point, directly on v1-unified.

1. Stay on v1-unified. Fix compile errors in the CPU backend against the
   canonical types.
2. Confirm it registers itself with the HardwareProfile dispatch point
   correctly (not hardcoded as "the" backend).
3. `cargo test` for this backend — redirect full output, report pass/fail
   counts + only real failures.
4. Any genuine behavior change (not just a rename) → stop, log under
   "Known blockers" in PROGRESS.md, report to me — don't decide alone.
5. Commit. Update PROGRESS.md, set current task to "Task 4 — merge mlx
   backend", stop.
```

### Task 4 — Merge `mlx` backend into `v1-unified`

```
Task 4: Same as Task 3, for the MLX (Apple Silicon/ARM) backend, merged
directly into v1-unified (not a separate branch).

1. Pull the MLX backend implementation from the mlx branch into v1-unified,
   fix it against canonical types.
2. Confirm it registers with HardwareProfile dispatch (Metal/ARM detection
   → MLX backend selected automatically, not manually flagged).
3. Same test/log discipline as Task 3.
4. If MLX needs a canonical-type field that doesn't exist: log it, stop for
   review, don't add ad hoc.
5. Commit, update PROGRESS.md, set current task to "Task 5 — merge
   vision-stability", stop.
```

### Task 5 — Merge `vision-stability` (multimodal) into `v1-unified`

```
Task 5: Merge multimodal support (has_audio_encoder etc.) directly into
v1-unified.

1. Port vision-stability's multimodal fields/paradigm-router logic onto the
   canonical types.
2. Check specifically: does vision-stability assume any particular backend
   runs the vision encoder? If so, generalize it — the vision encoder must
   run on whichever backend HardwareProfile selected, same as the text
   decoder.
3. Same test/log discipline.
4. Commit, update PROGRESS.md, set current task to "Task 6 — cross-backend
   parity pass", stop.
```

### Task 6 — Cross-backend parity + any other branches

```
Task 6: Now that CPU, MLX, and multimodal are all on v1-unified:

1. If Task 1 found other branches beyond cpu/mlx/vision-stability, merge
   them now, same discipline as Tasks 3-5 (one at a time, compile + test +
   commit + stop between each if there's more than one).
2. Run the numerical parity test (per your spec's acceptance criteria)
   across every backend that's implemented so far, against the HuggingFace
   reference. Tolerance 1e-3. Redirect output, report pass/fail per backend.
3. Any backend that fails parity: log under "Known blockers", do not merge
   it as "working" — mark it explicitly as WIP in PROGRESS.md instead of
   silently shipping a broken backend.
4. Commit, update PROGRESS.md, set current task to "Task 7 — production
   hardening pass", stop.
```

### Task 7 — Production hardening pass

```
Task 7: v1-unified should now be functionally merged. Harden it.

1. Grep for `unwrap()` in library code and `unsafe` without a `// SAFETY:`
   comment (per CLAUDE.md rules) across the WHOLE merged codebase — earlier
   tasks only checked the files they touched. Fix or log every hit.
2. Grep for any remaining hardcoded backend selection (bypassing
   HardwareProfile) anywhere in the merged code, not just the parts touched
   so far. This is the most likely place old branch habits survive a merge.
3. Confirm no silent fallback paths exist: every "backend doesn't support X"
   case returns Err with a clear message.
4. Run the full test suite (all backends, all crates). Redirect output,
   report failures only.
5. Commit, update PROGRESS.md, set current task to "Task 8 — CLI + end to
   end verification", stop.
```

### Task 8 — CLI + end-to-end verification

```
Task 8: Verify the user-facing contract works, per your spec's CLI section.

1. `llm serve <a small real model>` on whatever hardware this session is
   running on — confirm it starts, HardwareProfile correctly detects and
   selects this machine's backend, and /v1/chat/completions responds.
2. If you have access to more than one hardware type across sessions (e.g.
   you run this on a CUDA box in one session, Apple Silicon in another),
   repeat this task per hardware type — log which hardware has been
   verified in PROGRESS.md under a new "Hardware verification matrix"
   section, and which hasn't yet.
3. Run `llm devices` and confirm the reported profile matches the actual
   machine.
4. Commit, update PROGRESS.md, set current task to "Task 9 — benchmark
   against acceptance criteria", stop.
```

### Task 9 — Benchmark against your spec's acceptance criteria

```
Task 9: Check v1-unified against the Verification Matrix in the spec doc
(day-zero model support, quantized GEMV speed, concurrent throughput, KV
waste, numerical parity, memory safety, fault recovery where implemented).

1. Run whichever of these benchmarks are feasible on this machine's
   hardware. Redirect raw output to file, report just the metric vs target.
2. Anything that misses target: log under "Known blockers" with the actual
   number, don't round up or soften it in PROGRESS.md.
3. Commit, update PROGRESS.md, set current task to "Task 10 — tag v1", stop.
```

### Task 10 — Tag v1

```
Task 10: Only after Tasks 1-9 are clean (or every remaining gap is
explicitly logged as a known, accepted limitation — not silently missing):

1. Write a short CHANGELOG entry summarizing what v1-unified supports:
   which backends, which model paradigms, what's verified vs still WIP.
2. Merge v1-unified into main (create main from it if main doesn't exist).
3. Tag `v1.0.0`.
4. Update PROGRESS.md: mark this phase COMPLETE.
5. Stop. This is your review point before calling it production.
```

---

## 4. Session hygiene checklist

- **Start of every session:** "Read CLAUDE.md and PROGRESS.md, confirm current task, execute it."
- **Long output → file, not chat context.** Say this explicitly if Claude Code doesn't do it on its own for a given command.
- **One task per session** unless you say "continue."
- **Review the stop points** — Task 2's Option<T>/dispatch design, and any "Known blockers" entries — these are exactly the decisions that silently compound if you don't look at them.
- **If context runs low mid-task:** tell Claude Code to commit whatever's in a working state, log exactly where it stopped in PROGRESS.md, and stop. Because tasks are scoped small, this is a normal exit, not a failure.
- **Hardware verification matrix** (Task 8) will only ever cover hardware you actually run a session on — if you only have access to CUDA and Apple Silicon, CPU-only/other-Vulkan-GPU verification stays open. That's fine — log it as an open gap in PROGRESS.md rather than assuming it works.

---

## 5. Why this ordering

- Types + dispatch point before any backend merge: every backend depends on `ModelMeta`/`KvDtype`/`HardwareProfile`, so nailing that first turns each backend port into a mechanical compile-fix, not a design decision made three separate times.
- CPU first: your most complete branch, least unfinished code to fight through while validating the canonical types.
- MLX next, multimodal last: multimodal fields are additive on top of a working backend layer — doing it last stops multimodal assumptions from leaking into the base merge.
- Hardening (Task 7) happens *after* functional merge, not interleaved — mixing "make it compile" with "make it safe" across three separate backend ports means the same grep-for-`unwrap()` work happens three times instead of once, cleanly, at the end.
- Benchmarks and the v1 tag are last on purpose: nothing gets called "production" until it's measured against your own spec's numbers, not just "it compiled and the demo ran."