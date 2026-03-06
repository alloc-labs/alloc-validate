#!/usr/bin/env bash
set -euo pipefail

# Local dual-L4 stress validation for alloc CLI.
# Purpose: exercise calibrate/full modes, multi-GPU (2x) process discovery,
# distributed DDP/FSDP paths, callback timing path, and optional upload.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ALLOC_BIN="${ALLOC_BIN:-$REPO_ROOT/.venv/bin/alloc}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$REPO_ROOT/.venv/bin/torchrun}"
TORCHRUN_CMD=()

if [ ! -x "$ALLOC_BIN" ]; then
  ALLOC_BIN="$(command -v alloc || true)"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ ! -x "$TORCHRUN_BIN" ]; then
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi

if [ -z "${ALLOC_BIN:-}" ] || [ ! -x "$ALLOC_BIN" ]; then
  echo "ERROR: alloc executable not found. Run: python3 bootstrap.py"
  exit 1
fi
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python executable not found."
  exit 1
fi
if [ -n "${TORCHRUN_BIN:-}" ] && [ -x "$TORCHRUN_BIN" ]; then
  TORCHRUN_CMD=("$TORCHRUN_BIN")
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD=("$(command -v torchrun)")
elif "$PYTHON_BIN" -m torch.distributed.run --help >/dev/null 2>&1; then
  TORCHRUN_CMD=("$PYTHON_BIN" "-m" "torch.distributed.run")
else
  echo "ERROR: torchrun launcher not found."
  echo "Install extras with: .venv/bin/pip install -e '.[all]'"
  echo "or ensure 'python -m torch.distributed.run' is available."
  exit 1
fi

if ! "${TORCHRUN_CMD[@]}" --help >/dev/null 2>&1; then
  echo "ERROR: torchrun is installed but not functional on this environment."
  exit 1
fi

cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$REPO_ROOT/baselines/dual-l4-local-$STAMP"
mkdir -p "$OUT_DIR"

echo "=== alloc local dual-L4 stress ==="
echo "repo: $REPO_ROOT"
echo "output: $OUT_DIR"
echo "launcher: ${TORCHRUN_CMD[*]}"

"$ALLOC_BIN" version || true
"$ALLOC_BIN" whoami --json || true

echo ""
echo "--- GPU sanity ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "WARN: nvidia-smi not found"
fi

GPU_COUNT="$($PYTHON_BIN - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

echo "Detected CUDA GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 2 ]; then
  echo "ERROR: dual-L4 stress requires >=2 visible GPUs"
  exit 1
fi

echo ""
echo "--- T0 pre-flight sizing (ghost + optional scan) ---"
"$ALLOC_BIN" ghost distributed/train_ddp.py --param-count-b 7 --json > "$OUT_DIR/ghost_7b_reference.json"

set +e
"$ALLOC_BIN" scan --model llama-3-8b --gpu L4-24GB --num-gpus 2 --json > "$OUT_DIR/scan_l4x2_llama3_8b.json"
SCAN_RC=$?
set -e
if [ "$SCAN_RC" -eq 0 ]; then
  echo "OK: scan recommendation captured"
else
  echo "SKIP: alloc scan unavailable (network/API/etc.), continuing local GPU stress lanes"
fi

echo ""
echo "--- T1 calibrate path (default mode; should stop early when stable) ---"
"$ALLOC_BIN" run \
  --out "$OUT_DIR/ddp_calibrate_medium.json.gz" \
  -- "${TORCHRUN_CMD[@]}" --nproc_per_node=2 distributed/train_ddp.py --model medium --batch-size 16 --max-steps 200

echo ""
echo "--- T2 full monitoring DDP stress (2 GPUs) ---"
set +e
"$ALLOC_BIN" run --full \
  --out "$OUT_DIR/ddp_full_large.json.gz" \
  -- "${TORCHRUN_CMD[@]}" --nproc_per_node=2 distributed/train_ddp.py --model large --batch-size 8 --max-steps 30
T2_RC=$?
set -e
if [ "$T2_RC" -ne 0 ]; then
  echo "WARN: T2 DDP full lane failed (likely memory pressure). Continuing."
fi

echo ""
echo "--- T3 full monitoring FSDP stress (2 GPUs) ---"
set +e
"$ALLOC_BIN" run --full \
  --out "$OUT_DIR/fsdp_full_medium.json.gz" \
  -- "${TORCHRUN_CMD[@]}" --nproc_per_node=2 distributed/train_fsdp.py --model medium --batch-size 8 --max-steps 25
T3_RC=$?
set -e
if [ "$T3_RC" -ne 0 ]; then
  echo "WARN: T3 FSDP lane failed. Continuing."
fi

echo ""
echo "--- T4 callback path (HF) ---"
set +e
"$ALLOC_BIN" run --full \
  --out "$OUT_DIR/hf_full_gpt2_tiny.json.gz" \
  -- "$PYTHON_BIN" huggingface/train.py --model gpt2-tiny --batch-size 16 --max-steps 60
T4_RC=$?
set -e
if [ "$T4_RC" -ne 0 ]; then
  echo "WARN: T4 HF callback lane failed. Continuing."
fi

echo ""
echo "--- T5 high-pressure lane (expect OOM OR >=85% VRAM on at least one GPU) ---"
set +e
"$ALLOC_BIN" run --full \
  --out "$OUT_DIR/ddp_pressure_xl.json.gz" \
  -- "${TORCHRUN_CMD[@]}" --nproc_per_node=2 distributed/train_ddp.py --model xl --batch-size 32 --max-steps 10 \
  > "$OUT_DIR/ddp_pressure_xl.log" 2>&1
PRESSURE_RC=$?
set -e

if [ "$PRESSURE_RC" -ne 0 ]; then
  echo "OK: pressure lane exited non-zero (likely OOM/high-memory failure), as expected for stress testing"
else
  "$PYTHON_BIN" - <<PY
import gzip
import json
from pathlib import Path

artifact = Path("$OUT_DIR/ddp_pressure_xl.json.gz")
if not artifact.exists():
    print("WARN: pressure lane succeeded but no artifact found")
else:
    with gzip.open(artifact, "rt", encoding="utf-8") as f:
        data = json.load(f)
    probe = data.get("probe", {})
    hw = data.get("hardware", {})
    peak = probe.get("peak_vram_mb")
    total = hw.get("gpu_total_vram_mb")
    if peak and total:
        pct = (peak / total) * 100
        print(f"Pressure lane peak VRAM: {pct:.1f}%")
        if pct < 85:
            print("WARN: pressure lane ran but did not reach expected high VRAM pressure (<85%)")
    else:
        print("WARN: pressure lane artifact missing peak/total VRAM fields")
PY
fi

echo ""
echo "--- Artifact schema checks ---"
check_artifact_if_present() {
  local artifact="$1"
  local schema="$2"
  if [ -f "$artifact" ]; then
    "$PYTHON_BIN" scripts/check_artifact.py --artifact "$artifact" --schema "$schema" --tier free
  else
    echo "SKIP: missing artifact $(basename "$artifact")"
  fi
}

check_artifact_if_present "$OUT_DIR/ddp_calibrate_medium.json.gz" "distributed/expected/schema.json"
check_artifact_if_present "$OUT_DIR/ddp_full_large.json.gz" "distributed/expected/schema.json"
check_artifact_if_present "$OUT_DIR/fsdp_full_medium.json.gz" "distributed/expected/schema.json"
check_artifact_if_present "$OUT_DIR/hf_full_gpt2_tiny.json.gz" "huggingface/expected/schema.json"

echo ""
echo "--- Optional upload smoke (only if ALLOC_TOKEN is set) ---"
if [ -n "${ALLOC_TOKEN:-}" ]; then
  if [ -f "$OUT_DIR/ddp_full_large.json.gz" ]; then
    "$ALLOC_BIN" upload "$OUT_DIR/ddp_full_large.json.gz"
    echo "OK: upload smoke completed"
  else
    echo "SKIP: ALLOC_TOKEN is set but ddp_full_large artifact is missing"
  fi
else
  echo "SKIP: ALLOC_TOKEN not set"
fi

echo ""
echo "=== dual-L4 stress complete ==="
echo "Artifacts: $OUT_DIR"
echo "Lane exit codes: T2=${T2_RC:-0} T3=${T3_RC:-0} T4=${T4_RC:-0} T5=$PRESSURE_RC"
echo "Expected outcomes:"
echo "  - ghost pre-flight JSON exists (VRAM decomposition baseline)"
echo "  - scan JSON exists when API/network is available (otherwise explicit SKIP)"
echo "  - calibrate run stops early with artifact"
echo "  - full runs complete and include non-zero GPU metrics"
echo "  - pressure lane either fails due memory pressure or reports >=85% peak VRAM"
echo "  - DDP/FSDP artifacts pass distributed schema"
echo "  - HF artifact includes callback timing fields"
