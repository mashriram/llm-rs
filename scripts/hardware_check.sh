#!/usr/bin/env bash
# llm-rs end-to-end hardware verification.
#
# Run this on ANY machine (this Mac, a CUDA box, a Raspberry Pi, an x86
# server) to get an honest pass/fail report of what llm-rs actually does on
# that hardware: build with the right backend feature, detect the hardware
# profile, run a real text generation, optionally run vision/audio and
# cluster-networking smoke tests.
#
# It does NOT and CANNOT test mobile (Android/iOS) — that packaging doesn't
# exist yet (see PROGRESS.md's "v3" hardware table). Passing --check-mobile
# only prints what's missing so you know exactly what's left to build,
# rather than silently skipping the topic.
#
# Usage:
#   ./scripts/hardware_check.sh [options]
#
# Options:
#   --model-dir DIR        Directory containing (or to download) the text
#                           test model. Default: ./models/hwcheck
#   --skip-download         Don't call `llm pull`; --model-dir must already
#                           contain model.gguf + tokenizer.json (see below).
#   --vision-model PATH     GGUF file with a vision-capable model, to run the
#                           vision smoke test. Needs --vision-mmproj too.
#   --vision-mmproj PATH    Matching mmproj GGUF for --vision-model.
#   --vision-image PATH     Image file to feed the vision smoke test.
#                           Default: auto-generates a synthetic PNG.
#   --audio-model PATH      GGUF file with an audio-capable model.
#   --audio-mmproj PATH     Matching mmproj GGUF for --audio-model.
#   --audio-file PATH       WAV file for the audio smoke test.
#                           Default: auto-generates a synthetic tone.
#   --skip-cluster          Skip the llm-cluster networking smoke test.
#   --check-mobile          Print the honest mobile-readiness report and exit.
#   --release               Build in release mode (default: debug, faster to
#                           build, slower to run — fine for a smoke test).
#   -h, --help              Show this help.
#
# Exit code is 0 only if every step that ran passed. Skipped steps (no
# vision/audio model given, --skip-cluster) do not count as failures — the
# final report says explicitly what ran vs what was skipped.
#
# Example — text-only smoke test on this machine, downloading a tiny model:
#   ./scripts/hardware_check.sh
#
# Example — full multimodal + cluster check with local models already on disk:
#   ./scripts/hardware_check.sh --skip-download \
#     --model-dir ./models/gemma \
#     --vision-model ./models/gemma/gemma-4-E2B-it-Q4_K_M.gguf \
#     --vision-mmproj ./models/gemma/gemma-4-E2B-mmproj-BF16.gguf \
#     --audio-model ./models/gemma/gemma-4-E2B-it-Q4_K_M.gguf \
#     --audio-mmproj ./models/gemma/gemma-4-E2B-mmproj-BF16.gguf

set -uo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

MODEL_DIR="./models/hwcheck"
SKIP_DOWNLOAD=0
VISION_MODEL=""
VISION_MMPROJ=""
VISION_IMAGE=""
AUDIO_MODEL=""
AUDIO_MMPROJ=""
AUDIO_FILE=""
SKIP_CLUSTER=0
CHECK_MOBILE=0
PROFILE="debug"
CARGO_PROFILE_FLAG=""

print_help() {
  sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --vision-model) VISION_MODEL="$2"; shift 2 ;;
    --vision-mmproj) VISION_MMPROJ="$2"; shift 2 ;;
    --vision-image) VISION_IMAGE="$2"; shift 2 ;;
    --audio-model) AUDIO_MODEL="$2"; shift 2 ;;
    --audio-mmproj) AUDIO_MMPROJ="$2"; shift 2 ;;
    --audio-file) AUDIO_FILE="$2"; shift 2 ;;
    --skip-cluster) SKIP_CLUSTER=1; shift ;;
    --check-mobile) CHECK_MOBILE=1; shift ;;
    --release) PROFILE="release"; CARGO_PROFILE_FLAG="--release"; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run)"
REPORT_DIR="$REPO_ROOT/hwcheck-results/$TS"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
: > "$SUMMARY"

PASS=0
FAIL=0
SKIP=0

log()  { echo "[hwcheck] $*"; }
note() { echo "$*" >> "$SUMMARY"; }

step_pass() { PASS=$((PASS+1)); note "PASS  - $1"; log "PASS  - $1"; }
step_fail() { FAIL=$((FAIL+1)); note "FAIL  - $1"; log "FAIL  - $1"; }
step_skip() { SKIP=$((SKIP+1)); note "SKIP  - $1"; log "SKIP  - $1"; }

# ---------------------------------------------------------------------------
# Mobile readiness report (honest "not implemented" path, not a fake test)
# ---------------------------------------------------------------------------

print_mobile_report() {
  cat <<'EOF'
=== Mobile (Android/iOS) readiness — honest status ===

NOT IMPLEMENTED. There is no way for this script (or any script) to give
you a real pass/fail here yet, because the packaging layer doesn't exist:

  - No JNI bindings wrapping llm-ffi for Android.
  - No Swift/Objective-C bridging header/module for iOS.
  - candle-core has never been confirmed to build for the
    aarch64-linux-android or aarch64-apple-ios targets in this project.
  - No .aar (Android) or .xcframework (iOS) packaging exists.

What DOES already exist and is the right foundation to build on:
  - llm-ffi: a real C ABI (fixed this session — real Tokio runtime, real
    tokenizer, catch_unwind at the boundary) that a JNI or Swift layer
    would bind to.
  - The CPU backend itself is generic ARM code (NEON-aware, verified on
    Apple Silicon in this same script's CPU/Metal path) — the CPU inference
    path itself is not the blocker; the mobile OS packaging is.

To actually verify mobile, someone needs to, in order:
  1. `rustup target add aarch64-linux-android aarch64-apple-ios` and confirm
     candle-core + tokenizers (onig/esaxx-rs C deps) cross-compile clean.
  2. Write a minimal JNI shim (Android) / Swift wrapper (iOS) calling
     llm-ffi's existing C functions.
  3. Package + run on a real device or emulator/simulator with a real
     small model, and confirm it doesn't OOM (mobile RAM budgets are far
     tighter than this dev machine's).

This is a multi-week project, not a missing test flag.
EOF
}

if [[ "$CHECK_MOBILE" -eq 1 ]]; then
  print_mobile_report | tee "$REPORT_DIR/mobile_report.txt"
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Detect platform
# ---------------------------------------------------------------------------

OS="$(uname -s)"
ARCH="$(uname -m)"
HAS_NVCC=0; command -v nvcc >/dev/null 2>&1 && HAS_NVCC=1
HAS_NVIDIA_SMI=0; command -v nvidia-smi >/dev/null 2>&1 && HAS_NVIDIA_SMI=1
IS_MACOS_ARM=0; [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]] && IS_MACOS_ARM=1
IS_LINUX_ARM=0; [[ "$OS" == "Linux" && ( "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ) ]] && IS_LINUX_ARM=1

log "Platform: OS=$OS ARCH=$ARCH  nvcc=$HAS_NVCC nvidia-smi=$HAS_NVIDIA_SMI  macOS-arm64=$IS_MACOS_ARM  linux-arm=$IS_LINUX_ARM"
note "Platform: OS=$OS ARCH=$ARCH  nvcc=$HAS_NVCC nvidia-smi=$HAS_NVIDIA_SMI  macOS-arm64=$IS_MACOS_ARM  linux-arm=$IS_LINUX_ARM"

FEATURE_FLAG=""
BACKEND_LABEL="cpu-only"
if [[ "$IS_MACOS_ARM" -eq 1 ]]; then
  FEATURE_FLAG="--features metal"
  BACKEND_LABEL="metal"
elif [[ "$HAS_NVCC" -eq 1 ]]; then
  FEATURE_FLAG="--features cuda"
  BACKEND_LABEL="cuda"
else
  log "No CUDA toolkit (nvcc) and not macOS/arm64 — building CPU-only. This is \
also the expected path for Raspberry Pi / generic Linux ARM: no code here is \
macOS-specific outside the Metal feature, so the same CPU build should run \
there, though this script has only been confirmed to actually PASS on \
macOS/Apple Silicon and CPU-only Linux/x86 so far."
fi
note "Backend feature selected: $BACKEND_LABEL ($FEATURE_FLAG)"

if [[ "$HAS_NVIDIA_SMI" -eq 1 && "$HAS_NVCC" -eq 0 ]]; then
  log "WARNING: nvidia-smi found but nvcc not found — an NVIDIA GPU is \
present but the CUDA toolchain isn't, so this build will be CPU-only and \
will NOT exercise the GPU. Install the CUDA toolkit to test the cuda backend."
fi

# ---------------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------------

log "Building llm-cli + llm-cluster ($PROFILE, $BACKEND_LABEL)..."
BUILD_LOG="$REPORT_DIR/build.log"
if cargo build $CARGO_PROFILE_FLAG $FEATURE_FLAG \
    -p llm-cli --bin devices --bin chat --bin pull --bin run_model \
    -p llm-cluster --bin llm-cluster \
    > "$BUILD_LOG" 2>&1; then
  step_pass "build ($BACKEND_LABEL, $PROFILE)"
else
  step_fail "build ($BACKEND_LABEL, $PROFILE) — see $BUILD_LOG"
  log "Build failed, aborting remaining steps. Full log: $BUILD_LOG"
  tail -40 "$BUILD_LOG"
  echo "----" >> "$SUMMARY"
  echo "Report: $REPORT_DIR"
  exit 1
fi

BIN_DIR="$REPO_ROOT/target/$PROFILE"
DEVICES_BIN="$BIN_DIR/devices"
CHAT_BIN="$BIN_DIR/chat"
PULL_BIN="$BIN_DIR/pull"
RUN_MODEL_BIN="$BIN_DIR/run_model"
CLUSTER_BIN="$BIN_DIR/llm-cluster"

# ---------------------------------------------------------------------------
# 3. Hardware profile detection
# ---------------------------------------------------------------------------

log "Running hardware detection (llm devices)..."
DEVICES_LOG="$REPORT_DIR/devices.txt"
if "$DEVICES_BIN" > "$DEVICES_LOG" 2>&1; then
  cat "$DEVICES_LOG"
  SELECTED_BACKEND="$(grep 'Selected backend' "$DEVICES_LOG" | awk '{print $NF}')"
  step_pass "hardware detection (selected backend: $SELECTED_BACKEND)"
else
  step_fail "hardware detection — see $DEVICES_LOG"
fi

# ---------------------------------------------------------------------------
# 4. Text generation smoke test
# ---------------------------------------------------------------------------

mkdir -p "$MODEL_DIR"
MODEL_GGUF=""
TOKENIZER_JSON=""

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  log "Downloading a small test model into $MODEL_DIR (llm pull)..."
  PULL_LOG="$REPORT_DIR/pull.txt"
  if "$PULL_BIN" "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF" \
      --output-dir "$MODEL_DIR" > "$PULL_LOG" 2>&1; then
    step_pass "llm pull (text test model)"
  else
    step_fail "llm pull (text test model) — see $PULL_LOG"
    log "Could not download a test model automatically. Re-run with \
--skip-download --model-dir <dir containing a .gguf + tokenizer.json> to \
supply one manually."
  fi
fi

MODEL_GGUF="$(find "$MODEL_DIR" -maxdepth 1 -iname '*.gguf' ! -iname '*mmproj*' | head -1)"
TOKENIZER_JSON="$(find "$MODEL_DIR" -maxdepth 1 -iname 'tokenizer.json' | head -1)"

if [[ -n "$MODEL_GGUF" && -n "$TOKENIZER_JSON" ]]; then
  log "Running text generation smoke test with $MODEL_GGUF ..."
  GEN_LOG="$REPORT_DIR/generate.txt"
  if "$RUN_MODEL_BIN" --model-path "$MODEL_GGUF" --tokenizer-path "$TOKENIZER_JSON" \
      --prompt "The capital of France is" --max-new-tokens 16 \
      > "$GEN_LOG" 2>&1; then
    if grep -qi 'paris' "$GEN_LOG"; then
      step_pass "text generation ($BACKEND_LABEL) — correct output"
    else
      step_fail "text generation ($BACKEND_LABEL) — ran but output looked wrong, check $GEN_LOG"
    fi
  else
    step_fail "text generation ($BACKEND_LABEL) — crashed, see $GEN_LOG"
  fi
else
  step_skip "text generation — no model+tokenizer found in $MODEL_DIR"
fi

# ---------------------------------------------------------------------------
# 5. Vision smoke test (opt-in: needs --vision-model + --vision-mmproj)
# ---------------------------------------------------------------------------

if [[ -n "$VISION_MODEL" && -n "$VISION_MMPROJ" ]]; then
  if [[ "$(dirname "$VISION_MODEL")" != "$(dirname "$VISION_MMPROJ")" ]]; then
    log "WARNING: llm-rs auto-discovers the mmproj file from the SAME \
directory as --model-path (see find_mmproj_path in candle.rs) — it is not \
passed as a separate flag. $VISION_MMPROJ is in a different directory than \
$VISION_MODEL, so it will NOT be picked up. Copy or symlink it alongside \
the model file."
  fi
  if [[ -z "$VISION_IMAGE" ]]; then
    VISION_IMAGE="$REPORT_DIR/synthetic_test_image.png"
    log "No --vision-image given, generating a synthetic PNG at $VISION_IMAGE"
    # Pure stdlib (zlib+struct) PNG writer -- no PIL/Pillow dependency, since
    # it's frequently not installed (confirmed absent on this dev machine).
    python3 - "$VISION_IMAGE" <<'PY' 2>/dev/null || true
import sys, struct, zlib

path = sys.argv[1]
w, h = 224, 224
row = bytes([120, 160, 200]) * w
raw = b"".join(b"\x00" + row for _ in range(h))  # filter byte 0 (None) per row

def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data)))

png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(raw, 9))
png += chunk(b"IEND", b"")

with open(path, "wb") as f:
    f.write(png)
PY
  fi
  if [[ -f "$VISION_IMAGE" ]]; then
    log "Running vision smoke test with $VISION_MODEL / $VISION_MMPROJ ..."
    VISION_LOG="$REPORT_DIR/vision.txt"
    if printf '/image %s Describe this image in one sentence.\n/quit\n' "$VISION_IMAGE" | \
        "$CHAT_BIN" --model-path "$VISION_MODEL" --tokenizer-path "$TOKENIZER_JSON" \
        --max-new-tokens 32 > "$VISION_LOG" 2>&1; then
      step_pass "vision smoke test (ran without crashing — NOT a correctness check, see PROGRESS.md)"
    else
      step_fail "vision smoke test — crashed, see $VISION_LOG"
    fi
  else
    step_skip "vision smoke test — could not produce a test image (need PIL/python3, or pass --vision-image)"
  fi
else
  step_skip "vision smoke test — no --vision-model/--vision-mmproj given"
fi

# ---------------------------------------------------------------------------
# 6. Audio smoke test (opt-in: needs --audio-model + --audio-mmproj)
# ---------------------------------------------------------------------------

if [[ -n "$AUDIO_MODEL" && -n "$AUDIO_MMPROJ" ]]; then
  if [[ "$(dirname "$AUDIO_MODEL")" != "$(dirname "$AUDIO_MMPROJ")" ]]; then
    log "WARNING: llm-rs auto-discovers the mmproj file from the SAME \
directory as --model-path — it is not passed as a separate flag. \
$AUDIO_MMPROJ is in a different directory than $AUDIO_MODEL, so it will \
NOT be picked up. Copy or symlink it alongside the model file."
  fi
  if [[ -z "$AUDIO_FILE" ]]; then
    AUDIO_FILE="$REPORT_DIR/synthetic_tone.wav"
    log "No --audio-file given, generating a synthetic tone at $AUDIO_FILE"
    python3 - "$AUDIO_FILE" <<'PY' 2>/dev/null || true
import sys, wave, struct, math
rate = 16000
duration = 2.0
freq = 440.0
n = int(rate * duration)
with wave.open(sys.argv[1], "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    for i in range(n):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / rate))
        w.writeframes(struct.pack("<h", v))
PY
  fi
  if [[ -f "$AUDIO_FILE" ]]; then
    log "Running audio smoke test with $AUDIO_MODEL / $AUDIO_MMPROJ ..."
    AUDIO_LOG="$REPORT_DIR/audio.txt"
    if printf '/audio %s What do you hear?\n/quit\n' "$AUDIO_FILE" | \
        "$CHAT_BIN" --model-path "$AUDIO_MODEL" --tokenizer-path "$TOKENIZER_JSON" \
        --max-new-tokens 32 > "$AUDIO_LOG" 2>&1; then
      step_pass "audio smoke test (ran without crashing — NOT a correctness check, see PROGRESS.md)"
    else
      step_fail "audio smoke test — crashed, see $AUDIO_LOG"
    fi
  else
    step_skip "audio smoke test — could not produce a test tone (need python3, or pass --audio-file)"
  fi
else
  step_skip "audio smoke test — no --audio-model/--audio-mmproj given"
fi

# ---------------------------------------------------------------------------
# 7. Cluster networking smoke test (real TCP, real processes, real kill)
# ---------------------------------------------------------------------------

if [[ "$SKIP_CLUSTER" -eq 0 ]]; then
  log "Running llm-cluster networking smoke test (real TCP, 2 local processes)..."
  CLUSTER_PORT=19123
  COORD_LOG="$REPORT_DIR/cluster_coordinator.txt"
  WORKER_LOG="$REPORT_DIR/cluster_worker.txt"

  # RUST_LOG=info is required: tracing_subscriber's EnvFilter defaults to
  # ERROR-only with no RUST_LOG set, which would silently hide the
  # "registered"/"failed (missed heartbeat" lines this check greps for.
  RUST_LOG=info "$CLUSTER_BIN" coordinator --listen-addr "127.0.0.1:$CLUSTER_PORT" \
    --heartbeat-timeout-secs 3 > "$COORD_LOG" 2>&1 &
  COORD_PID=$!
  sleep 1

  RUST_LOG=info "$CLUSTER_BIN" worker --coordinator-addr "127.0.0.1:$CLUSTER_PORT" \
    --node-id hwcheck-worker --heartbeat-interval-secs 1 > "$WORKER_LOG" 2>&1 &
  WORKER_PID=$!
  sleep 2

  if grep -q "registered" "$COORD_LOG" 2>/dev/null; then
    step_pass "cluster: worker registered with coordinator"
  else
    step_fail "cluster: worker never registered — see $COORD_LOG"
  fi

  kill -9 "$WORKER_PID" 2>/dev/null
  sleep 5

  if grep -qi "failed (missed heartbeat" "$COORD_LOG" 2>/dev/null; then
    step_pass "cluster: killed worker correctly detected as failed"
  else
    step_fail "cluster: killed worker was NOT detected as failed within timeout — see $COORD_LOG"
  fi

  kill -9 "$COORD_PID" 2>/dev/null
  wait "$COORD_PID" 2>/dev/null
  wait "$WORKER_PID" 2>/dev/null
else
  step_skip "cluster networking smoke test (--skip-cluster given)"
fi

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------

echo ""
echo "=================================================================="
echo " llm-rs hardware check — $BACKEND_LABEL on $OS/$ARCH"
echo "=================================================================="
cat "$SUMMARY"
echo "------------------------------------------------------------------"
echo " $PASS passed, $FAIL failed, $SKIP skipped"
echo " Full logs: $REPORT_DIR"
echo "=================================================================="
echo ""
echo "Reminder: mobile (Android/iOS) is not tested by this script because"
echo "it isn't implemented yet — run with --check-mobile for the honest"
echo "readiness report. CPU/Metal have real verified passes behind them"
echo "(see PROGRESS.md); CUDA/ARM-Linux correctness depends on this script"
echo "actually being run on that hardware — a pass here on a real CUDA box"
echo "or a real Raspberry Pi IS the verification that was previously"
echo "missing, so please share the report if you run it on either."

[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
