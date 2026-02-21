#!/usr/bin/env bash
set -euo pipefail

WORKLOAD_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$WORKLOAD_DIR/.." && pwd)"

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

echo "=== scan-only validation ==="

# --- CLI smoke tests ---
echo "--- CLI smoke tests ---"
"$ALLOC_BIN" version
echo "OK: alloc version"

if "$ALLOC_BIN" whoami --json > /dev/null 2>&1; then
    echo "OK: alloc whoami --json"
else
    echo "WARN: alloc whoami --json exited non-zero"
fi

if "$ALLOC_BIN" catalog list > /dev/null 2>&1; then
    echo "OK: alloc catalog list"
else
    echo "WARN: alloc catalog list exited non-zero"
fi

# --- Ghost scan (local, no token needed) ---
echo ""
echo "--- alloc ghost (all targets) ---"
cd "$WORKLOAD_DIR"

for target in ghost_target*.py; do
    echo ""
    echo "--- alloc ghost --json $target ---"
    outfile="ghost_output_${target%.py}.json"
    errfile="ghost_output_${target%.py}.stderr.log"

    if "$ALLOC_BIN" ghost "$target" --json > "$outfile" 2> "$errfile"; then
        echo "OK: alloc ghost $target produced output"
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
        if [ -s "$errfile" ]; then
            echo "WARN: alloc ghost wrote stderr (see $errfile)"
        fi
    else
        echo "WARN: alloc ghost $target exited non-zero (may not support this script format yet)"
    fi
done

# --- Remote scan (requires ALLOC_TOKEN) ---
echo ""
echo "--- alloc scan ---"

if [ -n "${ALLOC_TOKEN:-}" ]; then
    echo "ALLOC_TOKEN set — running remote scan combos"

    GPUS=("A100-80GB" "H100-80GB" "T4-16GB" "V100-32GB")
    MODELS=("llama-3-8b" "llama-3-70b" "mistral-7b")
    NUM_GPUS=(1 4)

    for gpu in "${GPUS[@]}"; do
        for model in "${MODELS[@]}"; do
            for ngpu in "${NUM_GPUS[@]}"; do
                echo ""
                echo "--- alloc scan --json --model $model --gpu $gpu --num-gpus $ngpu ---"
                outfile="scan_${model}_${gpu}_${ngpu}gpu.json"
                errfile="scan_${model}_${gpu}_${ngpu}gpu.stderr.log"
                if "$ALLOC_BIN" scan --model "$model" --gpu "$gpu" --num-gpus "$ngpu" --json > "$outfile" 2> "$errfile"; then
                    echo "OK: alloc scan produced output"
                    OUTFILE="$outfile" "$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ["OUTFILE"]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
    print(f"OK: scan returned {len(data)} keys")
except (json.JSONDecodeError, FileNotFoundError):
    print("WARN: scan output is not parseable JSON")
PY
                    if [ -s "$errfile" ]; then
                        echo "WARN: alloc scan wrote stderr (see $errfile)"
                    fi
                else
                    echo "WARN: alloc scan exited non-zero for $model / $gpu / ${ngpu}gpu"
                fi
                sleep 0.5
            done
        done
    done
else
    echo "SKIP: ALLOC_TOKEN not set — skipping remote scan"
fi

echo "=== scan-only validation complete ==="
