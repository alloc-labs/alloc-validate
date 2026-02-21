#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$WORKLOAD_DIR/.." && pwd)"

STRATEGY="${1:-ddp}"

ALLOC_BIN="${ALLOC_BIN:-$REPO_ROOT/.venv/bin/alloc}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$REPO_ROOT/.venv/bin/torchrun}"

if [ ! -x "$ALLOC_BIN" ]; then
  ALLOC_BIN="$(command -v alloc || true)"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ ! -x "$TORCHRUN_BIN" ]; then
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
# Verify torchrun actually works (may be installed but broken, e.g. missing importlib_metadata on Python 3.9)
TORCHRUN_OK=false
if [ -n "${TORCHRUN_BIN:-}" ] && [ -x "$TORCHRUN_BIN" ]; then
  if "$TORCHRUN_BIN" --help > /dev/null 2>&1; then
    TORCHRUN_OK=true
  else
    echo "WARN: torchrun found but not functional — falling back to python"
  fi
fi
if [ -z "${ALLOC_BIN:-}" ] || [ ! -x "$ALLOC_BIN" ]; then
  echo "ERROR: alloc executable not found. Install deps with: .venv/bin/pip install -e '.[all]'"
  exit 1
fi
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python executable not found."
  exit 1
fi

echo "=== distributed validation (strategy=$STRATEGY) ==="

# Determine auth tier
if [ -n "${ALLOC_TOKEN:-}" ]; then
    TIER="full"
    echo "ALLOC_TOKEN set — running full validation"
else
    TIER="free"
    echo "ALLOC_TOKEN not set — running free-tier validation"
fi

cd "$WORKLOAD_DIR"

# --- CLI smoke tests ---
echo ""
echo "--- CLI smoke tests ---"
"$ALLOC_BIN" version
echo "OK: alloc version"

if "$ALLOC_BIN" whoami --json > /dev/null 2>&1; then
    echo "OK: alloc whoami --json"
else
    echo "WARN: alloc whoami --json exited non-zero"
fi

# --- Detect GPU count for nproc ---
NUM_GPUS=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
echo "Detected GPUs: $NUM_GPUS"

# --- Run distributed training ---
echo ""
echo "--- alloc run (strategy=$STRATEGY) ---"

TRAIN_SCRIPT="train_${STRATEGY}.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: $TRAIN_SCRIPT not found in $WORKLOAD_DIR"
    exit 1
fi

# Clean old artifacts
rm -f alloc_artifact.json.gz alloc_artifact.json

if [ "$NUM_GPUS" -ge 2 ] && [ "$TORCHRUN_OK" = true ]; then
    # Multi-GPU: use torchrun
    NPROC="$NUM_GPUS"
    if [ "$NPROC" -gt 4 ]; then
        NPROC=4
    fi
    echo "Running with torchrun --nproc_per_node=$NPROC"
    "$ALLOC_BIN" run -- "$TORCHRUN_BIN" --nproc_per_node="$NPROC" "$TRAIN_SCRIPT" --model small --max-steps 5
elif [ "$TORCHRUN_OK" = true ]; then
    # CPU or single GPU: torchrun with 1 process
    echo "Running with torchrun --nproc_per_node=1 (single process)"
    "$ALLOC_BIN" run -- "$TORCHRUN_BIN" --nproc_per_node=1 "$TRAIN_SCRIPT" --model small --max-steps 5
else
    # No working torchrun: direct python fallback
    echo "Running with python directly (no torchrun)"
    "$ALLOC_BIN" run -- "$PYTHON_BIN" "$TRAIN_SCRIPT" --model small --max-steps 5
fi

# Check artifact
ARTIFACT="alloc_artifact.json.gz"
if [ ! -f "$ARTIFACT" ]; then
    echo "WARN: artifact $ARTIFACT not found, checking for plain JSON..."
    ARTIFACT="alloc_artifact.json"
fi

if [ -f "$ARTIFACT" ]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/check_artifact.py" \
        --artifact "$ARTIFACT" \
        --schema "$WORKLOAD_DIR/expected/schema.json" \
        --tier "$TIER"

    # On GPU, check for topology fields (advisory, not required on CPU)
    if [ "$NUM_GPUS" -ge 2 ]; then
        echo ""
        echo "--- topology field check (GPU) ---"
        "$PYTHON_BIN" - "$ARTIFACT" <<'PY'
import gzip
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
try:
    raw = gzip.decompress(raw)
except gzip.BadGzipFile:
    pass
data = json.loads(raw)

probe = data.get("probe", {})
topology_fields = {
    "strategy": probe.get("strategy"),
    "dp_degree": probe.get("dp_degree"),
    "tp_degree": probe.get("tp_degree"),
    "pp_degree": probe.get("pp_degree"),
    "num_nodes": probe.get("num_nodes"),
    "gpus_per_node": probe.get("gpus_per_node"),
    "interconnect_type": probe.get("interconnect_type"),
}

found = {k: v for k, v in topology_fields.items() if v is not None}
missing = [k for k, v in topology_fields.items() if v is None]

if found:
    print(f"OK: topology fields present: {found}")
else:
    print("WARN: no topology fields found in artifact (alloc may not emit these yet)")

if missing:
    print(f"INFO: topology fields not present: {missing}")
PY
    fi
else
    echo "WARN: no artifact found — alloc run may have failed silently"
fi

# --- alloc ghost ---
echo ""
echo "--- alloc ghost ---"
if "$ALLOC_BIN" ghost "$TRAIN_SCRIPT"; then
    echo "OK: alloc ghost completed for $TRAIN_SCRIPT"
else
    echo "WARN: alloc ghost exited non-zero (may not support this script format yet)"
fi

echo "=== distributed validation complete (strategy=$STRATEGY) ==="
