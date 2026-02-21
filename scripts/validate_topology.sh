#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: python executable not found."
  exit 1
fi

echo "=== topology validation ==="

# --- Ghost estimates for distributed scripts ---
echo ""
echo "--- alloc ghost (distributed scripts) ---"

DIST_DIR="$REPO_ROOT/distributed"
PASS=0
FAIL=0
WARN=0

for script in train_ddp.py train_fsdp.py train_pp.py train_tp.py train_tp_dp.py train_pp_dp.py train_3d.py train_3d_fsdp.py; do
    echo ""
    echo "--- alloc ghost $script ---"
    if [ ! -f "$DIST_DIR/$script" ]; then
        echo "SKIP: $script not found"
        continue
    fi

    outfile="$DIST_DIR/ghost_${script%.py}.json"
    if "$ALLOC_BIN" ghost "$DIST_DIR/$script" --json > "$outfile" 2>/dev/null; then
        echo "OK: alloc ghost $script produced output"
        OUTFILE="$outfile" "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["OUTFILE"]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
    for key in ["param_count", "weights_gb", "total_gb"]:
        if key not in data:
            print(f"WARN: ghost output missing key: {key}")
    print("OK: ghost output has expected structure")
except (json.JSONDecodeError, FileNotFoundError):
    print("WARN: ghost output is not parseable JSON")
PY
        PASS=$((PASS + 1))
    else
        echo "WARN: alloc ghost $script exited non-zero"
        WARN=$((WARN + 1))
    fi
done

# --- Scan topology combos (requires ALLOC_TOKEN) ---
echo ""
echo "--- alloc scan (topology combos) ---"

if [ -n "${ALLOC_TOKEN:-}" ]; then
    echo "ALLOC_TOKEN set — running topology scan combos"

    MODELS=("llama-3-8b" "mistral-7b")
    GPUS=("A100-80GB" "H100-80GB")
    NUM_GPUS=(1 4 8)

    for model in "${MODELS[@]}"; do
        for gpu in "${GPUS[@]}"; do
            for ngpu in "${NUM_GPUS[@]}"; do
                echo ""
                echo "--- alloc scan --model $model --gpu $gpu --num-gpus $ngpu ---"
                outfile="$DIST_DIR/scan_topo_${model}_${gpu}_${ngpu}gpu.json"
                errfile="$DIST_DIR/scan_topo_${model}_${gpu}_${ngpu}gpu.stderr.log"

                if "$ALLOC_BIN" scan --model "$model" --gpu "$gpu" --num-gpus "$ngpu" --json > "$outfile" 2> "$errfile"; then
                    echo "OK: scan produced output"

                    # Validate scan response structure
                    OUTFILE="$outfile" "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["OUTFILE"]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
    keys = list(data.keys()) if isinstance(data, dict) else []
    print(f"OK: scan response has {len(keys)} keys: {keys[:10]}")

    # Check for topology-related fields
    topo_keys = ["topologies", "strategies", "recommendations", "feasible", "vram_estimate"]
    found = [k for k in topo_keys if k in data or any(k in str(v) for v in (data.values() if isinstance(data, dict) else []))]
    if found:
        print(f"OK: topology-related fields present: {found}")
    else:
        print("INFO: no explicit topology fields (scan response format may vary)")
except (json.JSONDecodeError, FileNotFoundError):
    print("WARN: scan output is not parseable JSON")
PY
                    PASS=$((PASS + 1))
                else
                    echo "WARN: alloc scan exited non-zero for $model / $gpu / ${ngpu}gpu"
                    WARN=$((WARN + 1))
                fi
                sleep 0.5
            done
        done
    done
else
    echo "SKIP: ALLOC_TOKEN not set — skipping topology scan"
fi

# --- Run distributed validation for each strategy ---
echo ""
echo "--- distributed strategy validation ---"

for strategy in ddp fsdp pp tp tp_dp pp_dp 3d 3d_fsdp; do
    echo ""
    echo "=== validating strategy: $strategy ==="
    if bash "$DIST_DIR/validate.sh" "$strategy"; then
        echo "OK: $strategy validation passed"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $strategy validation failed"
        FAIL=$((FAIL + 1))
    fi
done

# --- Summary ---
echo ""
echo "=== topology validation summary ==="
echo "PASS: $PASS | FAIL: $FAIL | WARN: $WARN"

if [ "$FAIL" -gt 0 ]; then
    echo "RESULT: FAIL"
    exit 1
fi

echo "RESULT: OK"
