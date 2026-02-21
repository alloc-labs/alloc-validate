#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$WORKLOAD_DIR/.." && pwd)"

MODEL="${1:-distilbert-tiny}"

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

echo "=== huggingface validation (model=$MODEL) ==="

# Determine auth tier
if [ -n "${ALLOC_TOKEN:-}" ]; then
    TIER="full"
    echo "ALLOC_TOKEN set — running full validation"
else
    TIER="free"
    echo "ALLOC_TOKEN not set — running free-tier validation"
fi

# Run training under alloc
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

# Check for callback sidecar (HF callback writes .alloc_callback.json)
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

echo "=== huggingface validation complete ==="
