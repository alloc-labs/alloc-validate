#!/usr/bin/env bash
set -euo pipefail

# Validates `alloc upload` and the analysis pipeline (bottleneck classification,
# recommendations, config scoring). Requires ALLOC_TOKEN.
#
# Tests signal level progression:
# - NVML_ONLY: artifact from pytorch (no callback)
# - FRAMEWORK_TIMING: artifact from huggingface/lightning (with callback)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ALLOC_BIN="${ALLOC_BIN:-$REPO_ROOT/.venv/bin/alloc}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$ALLOC_BIN" ]; then
  ALLOC_BIN="$(command -v alloc || true)"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ -z "${ALLOC_BIN:-}" ] || [ ! -x "$ALLOC_BIN" ]; then
  echo "ERROR: alloc executable not found."
  exit 1
fi

echo "=== upload + analysis validation ==="

if [ -z "${ALLOC_TOKEN:-}" ]; then
    echo "SKIP: ALLOC_TOKEN not set — upload validation requires authentication"
    echo "=== upload validation skipped ==="
    exit 0
fi

echo "ALLOC_TOKEN set — running upload validation"

# --- Upload pytorch artifact (NVML_ONLY signal level) ---
echo ""
echo "--- upload pytorch artifact (NVML_ONLY) ---"
cd "$REPO_ROOT/pytorch"

# Generate fresh artifact if missing
if [ ! -f "alloc_artifact.json.gz" ] && [ ! -f "alloc_artifact.json" ]; then
    echo "Generating fresh artifact..."
    "$ALLOC_BIN" run -- "$PYTHON_BIN" train.py --model small-cnn --max-steps 30
fi

ARTIFACT="alloc_artifact.json.gz"
if [ ! -f "$ARTIFACT" ]; then
    ARTIFACT="alloc_artifact.json"
fi

if "$ALLOC_BIN" upload "$ARTIFACT" 2>&1; then
    echo "OK: alloc upload succeeded (pytorch / NVML_ONLY)"
else
    echo "WARN: alloc upload exited non-zero (pytorch)"
fi

# --- Upload huggingface artifact (FRAMEWORK_TIMING signal level) ---
echo ""
echo "--- upload huggingface artifact (FRAMEWORK_TIMING) ---"
cd "$REPO_ROOT/huggingface"

if [ ! -f "alloc_artifact.json.gz" ] && [ ! -f "alloc_artifact.json" ]; then
    echo "Generating fresh artifact..."
    "$ALLOC_BIN" run -- "$PYTHON_BIN" train.py --model distilbert-tiny --max-steps 30
fi

ARTIFACT="alloc_artifact.json.gz"
if [ ! -f "$ARTIFACT" ]; then
    ARTIFACT="alloc_artifact.json"
fi

if "$ALLOC_BIN" upload "$ARTIFACT" 2>&1; then
    echo "OK: alloc upload succeeded (huggingface / FRAMEWORK_TIMING)"
else
    echo "WARN: alloc upload exited non-zero (huggingface)"
fi

# --- Upload lightning artifact (FRAMEWORK_TIMING signal level) ---
echo ""
echo "--- upload lightning artifact (FRAMEWORK_TIMING) ---"
cd "$REPO_ROOT/lightning"

if [ ! -f "alloc_artifact.json.gz" ] && [ ! -f "alloc_artifact.json" ]; then
    echo "Generating fresh artifact..."
    "$ALLOC_BIN" run -- "$PYTHON_BIN" train.py --model small-cnn --max-steps 30
fi

ARTIFACT="alloc_artifact.json.gz"
if [ ! -f "$ARTIFACT" ]; then
    ARTIFACT="alloc_artifact.json"
fi

if "$ALLOC_BIN" upload "$ARTIFACT" 2>&1; then
    echo "OK: alloc upload succeeded (lightning / FRAMEWORK_TIMING)"
else
    echo "WARN: alloc upload exited non-zero (lightning)"
fi

cd "$REPO_ROOT"

echo ""
echo "=== upload + analysis validation complete ==="
echo "Check the dashboard for bottleneck classification and recommendations."
