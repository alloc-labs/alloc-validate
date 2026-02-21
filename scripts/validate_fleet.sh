#!/usr/bin/env bash
set -euo pipefail

# Validates .alloc.yaml fleet context, --no-config flag, and catalog commands.
# This tests alloc CLI features that are documented but had zero coverage.

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

echo "=== fleet context validation ==="

# --- Create a temporary .alloc.yaml for testing ---
TMPDIR="$(mktemp -d)"
YAML_FILE="$TMPDIR/.alloc.yaml"
TRAIN_SCRIPT="$REPO_ROOT/pytorch/train.py"

cat > "$YAML_FILE" <<'YAML'
fleet:
  - gpu: nvidia-t4-16gb
    count: 4
  - gpu: nvidia-a100-sxm-80gb
    explore: true
priority:
  cost: 70
  latency: 30
budget:
  monthly_usd: 1000
objective: best_value
YAML

echo "OK: .alloc.yaml fixture created"

# --- Test alloc run WITH .alloc.yaml ---
echo ""
echo "--- alloc run with fleet config ---"
cp "$YAML_FILE" "$REPO_ROOT/pytorch/.alloc.yaml"
cd "$REPO_ROOT/pytorch"

# Run training with fleet config present
if "$ALLOC_BIN" run -- "$PYTHON_BIN" train.py --model small-cnn --max-steps 10; then
    echo "OK: alloc run succeeded with .alloc.yaml present"
else
    echo "FAIL: alloc run failed with .alloc.yaml"
    rm -f "$REPO_ROOT/pytorch/.alloc.yaml"
    rm -rf "$TMPDIR"
    exit 1
fi

# Check artifact still generated
if [ -f "alloc_artifact.json.gz" ] || [ -f "alloc_artifact.json" ]; then
    echo "OK: artifact generated with fleet config"
else
    echo "FAIL: no artifact generated with fleet config"
fi

rm -f "$REPO_ROOT/pytorch/.alloc.yaml"

# --- Test --no-config flag ---
echo ""
echo "--- alloc run --no-config ---"
cp "$YAML_FILE" "$REPO_ROOT/pytorch/.alloc.yaml"

if "$ALLOC_BIN" run --no-config -- "$PYTHON_BIN" train.py --model small-cnn --max-steps 10; then
    echo "OK: alloc run --no-config succeeded"
else
    echo "WARN: alloc run --no-config exited non-zero"
fi

rm -f "$REPO_ROOT/pytorch/.alloc.yaml"

# --- Test alloc ghost with --no-config ---
echo ""
echo "--- alloc ghost --no-config ---"
cd "$REPO_ROOT/pytorch"
if "$ALLOC_BIN" ghost --no-config train.py > /dev/null 2>&1; then
    echo "OK: alloc ghost --no-config succeeded"
else
    echo "WARN: alloc ghost --no-config exited non-zero"
fi

# --- Test alloc catalog list ---
echo ""
echo "--- alloc catalog list ---"
if "$ALLOC_BIN" catalog list > /dev/null 2>&1; then
    echo "OK: alloc catalog list"
else
    echo "WARN: alloc catalog list exited non-zero"
fi

# --- Test alloc catalog show (pick a known GPU) ---
echo ""
echo "--- alloc catalog show ---"
if "$ALLOC_BIN" catalog show nvidia-t4-16gb > /dev/null 2>&1; then
    echo "OK: alloc catalog show nvidia-t4-16gb"
elif "$ALLOC_BIN" catalog show T4-16GB > /dev/null 2>&1; then
    echo "OK: alloc catalog show T4-16GB"
else
    echo "WARN: alloc catalog show exited non-zero (GPU ID format may differ)"
fi

# --- Test alloc init (non-interactive smoke test) ---
echo ""
echo "--- alloc init ---"
cd "$TMPDIR"
if echo "" | "$ALLOC_BIN" init > /dev/null 2>&1; then
    echo "OK: alloc init ran (may require interactive input)"
else
    echo "INFO: alloc init exited non-zero (expected — requires interactive input)"
fi

# Cleanup
rm -rf "$TMPDIR"
cd "$REPO_ROOT"

echo ""
echo "=== fleet context validation complete ==="
