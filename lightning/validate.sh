#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$WORKLOAD_DIR/.." && pwd)"

MODEL="${1:-small-cnn}"

ALLOC_BIN="${ALLOC_BIN:-$REPO_ROOT/.venv/bin/alloc}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$ALLOC_BIN" ]; then
  ALLOC_BIN="$(command -v alloc || true)"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ -z "${ALLOC_BIN:-}" ] || [ ! -x "$ALLOC_BIN" ]; then
  echo "ERROR: alloc executable not found. Install deps with: .venv/bin/pip install -e '.[all]'"
  exit 1
fi
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python executable not found."
  exit 1
fi

echo "=== lightning validation (model=$MODEL) ==="

# Determine auth tier
if [ -n "${ALLOC_TOKEN:-}" ]; then
    TIER="full"
    echo "ALLOC_TOKEN set — running full validation"
else
    TIER="free"
    echo "ALLOC_TOKEN not set — running free-tier validation"
fi

# --- CLI smoke tests (version, whoami) ---
echo ""
echo "--- CLI smoke tests ---"
"$ALLOC_BIN" version
echo "OK: alloc version"

if "$ALLOC_BIN" whoami --json > /dev/null 2>&1; then
    echo "OK: alloc whoami --json"
else
    echo "WARN: alloc whoami --json exited non-zero"
fi

# --- alloc run (probe) ---
echo ""
echo "--- alloc run ---"
cd "$WORKLOAD_DIR"
"$ALLOC_BIN" run -- "$PYTHON_BIN" train.py --model "$MODEL"

# Check artifact
ARTIFACT="alloc_artifact.json.gz"
if [ ! -f "$ARTIFACT" ]; then
    echo "WARN: artifact $ARTIFACT not found, checking for plain JSON..."
    ARTIFACT="alloc_artifact.json"
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/check_artifact.py" \
    --artifact "$ARTIFACT" \
    --schema "$WORKLOAD_DIR/expected/schema.json" \
    --tier "$TIER"

# Check for callback sidecar (Lightning callback writes .alloc_callback.json)
echo ""
echo "--- callback sidecar check ---"
if [ -f ".alloc_callback.json" ]; then
    echo "OK: .alloc_callback.json sidecar found"
    SIDECAR=".alloc_callback.json" "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["SIDECAR"]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        for key in ["step_count", "step_time_ms_p50", "step_time_ms_p90", "samples_per_sec"]:
            if key in data:
                print(f"  {key}: {data[key]}")
        print("OK: callback sidecar has expected structure")
    else:
        print("WARN: callback sidecar is not a dict")
except (json.JSONDecodeError, FileNotFoundError):
    print("WARN: callback sidecar is not parseable JSON")
PY
else
    echo "INFO: .alloc_callback.json not found (expected on CPU-only runs without callback)"
fi

# --- alloc ghost (static VRAM analysis) ---
echo ""
echo "--- alloc ghost ---"
if "$ALLOC_BIN" ghost train.py; then
    echo "OK: alloc ghost completed"
else
    echo "WARN: alloc ghost exited non-zero (may not support this script yet)"
fi

echo "=== lightning validation complete ==="
