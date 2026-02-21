#!/usr/bin/env bash
set -euo pipefail

# Record a GPU test baseline for comparison across releases.
# Saves matrix JSON output and individual artifacts with date/host stamps.
#
# Usage:
#   bash scripts/record_baseline.sh              # after running make matrix
#   bash scripts/record_baseline.sh --run-first   # runs matrix then records

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d)_$(hostname -s)"
BASELINE_DIR="$REPO_ROOT/baselines"
ARTIFACT_DIR="$BASELINE_DIR/artifacts"

mkdir -p "$BASELINE_DIR" "$ARTIFACT_DIR"

# Optionally run matrix first
if [ "${1:-}" = "--run-first" ]; then
    echo "Running matrix..."
    cd "$REPO_ROOT"
    make matrix-quick
fi

# Save matrix JSON output
echo "Recording baseline: $STAMP"

cd "$REPO_ROOT"
ALLOC_BIN="${ALLOC_BIN:-$REPO_ROOT/.venv/bin/alloc}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

MATRIX_FILE="$BASELINE_DIR/${STAMP}_matrix.json"
ALLOC_BIN="$ALLOC_BIN" PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" scripts/run_matrix.py --quick --json > "$MATRIX_FILE" 2>/dev/null || true

if [ -f "$MATRIX_FILE" ] && [ -s "$MATRIX_FILE" ]; then
    echo "OK: matrix results saved to $MATRIX_FILE"
else
    echo "WARN: matrix JSON output was empty"
fi

# Save individual artifacts
for workload in pytorch huggingface lightning ray; do
    for ext in alloc_artifact.json.gz alloc_artifact.json; do
        src="$REPO_ROOT/$workload/$ext"
        if [ -f "$src" ]; then
            dest="$ARTIFACT_DIR/${workload}_${STAMP}.${ext##*.}"
            cp "$src" "$dest"
            echo "OK: saved $workload artifact → $dest"
            break
        fi
    done
done

# Save callback sidecars
for workload in huggingface lightning; do
    src="$REPO_ROOT/$workload/.alloc_callback.json"
    if [ -f "$src" ]; then
        dest="$ARTIFACT_DIR/${workload}_callback_${STAMP}.json"
        cp "$src" "$dest"
        echo "OK: saved $workload callback sidecar → $dest"
    fi
done

echo ""
echo "Baseline recorded: $STAMP"
echo "Files in $BASELINE_DIR:"
ls -la "$BASELINE_DIR"/*.json 2>/dev/null || echo "  (no matrix files yet)"
echo "Artifacts in $ARTIFACT_DIR:"
ls -la "$ARTIFACT_DIR"/ 2>/dev/null || echo "  (no artifacts yet)"
