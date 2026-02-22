#!/usr/bin/env python3
"""Eval recommendations harness for alloc ghost and scan.

Runs `alloc ghost` and optionally `alloc scan` for large model configs and
evaluates whether alloc's recommendations are sensible for the model's
characteristics. Compares VRAM estimates against expected ranges.

Usage:
    python scripts/eval_recommendations.py              # ghost-only (no token needed)
    python scripts/eval_recommendations.py --full        # ghost + scan (requires ALLOC_TOKEN)
    python scripts/eval_recommendations.py --json        # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_alloc_raw = os.environ.get("ALLOC_BIN", "alloc")
_python_raw = os.environ.get("PYTHON_BIN", "python")
ALLOC_BIN = str(Path(_alloc_raw).absolute()) if "/" in _alloc_raw else _alloc_raw
PYTHON_BIN = str(Path(_python_raw).absolute()) if "/" in _python_raw else _python_raw

# Models to evaluate (GPU-only large configs)
EVAL_MODELS = ["7b", "13b", "30b", "70b"]

# Expected parameter count ranges (approximate, in billions)
EXPECTED_PARAMS: dict[str, tuple[float, float]] = {
    "7b": (5.0, 10.0),
    "13b": (10.0, 18.0),
    "30b": (25.0, 40.0),
    "70b": (55.0, 85.0),
}

# Expected VRAM ranges for weights only (GB, fp16 default)
EXPECTED_WEIGHTS_GB: dict[str, tuple[float, float]] = {
    "7b": (10.0, 30.0),
    "13b": (20.0, 50.0),
    "30b": (45.0, 120.0),
    "70b": (100.0, 260.0),
}

# Approximate param count in billions for --param-count-b flag
# (alloc ghost can't extract our synthetic models, so we hint the size)
PARAM_COUNT_B: dict[str, float] = {
    "7b": 7.0,
    "13b": 13.0,
    "30b": 30.0,
    "70b": 70.0,
}

# Map synthetic model names to alloc catalog names for scan
SCAN_MODEL_MAP = {
    "7b": "llama-3-8b",
    "13b": "llama-3-8b",
    "30b": "llama-3-70b",
    "70b": "llama-3-70b",
}

EVAL_GPUS = ["A100-80GB", "H100-80GB", "T4-16GB"]
EVAL_NUM_GPUS = [1, 4, 8]


def run_command(cmd: list[str], cwd: Path, timeout: int = 120) -> tuple[bool, str, str]:
    """Run a command, return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def eval_ghost(model: str) -> dict:
    """Run alloc ghost on a distributed script and evaluate the output."""
    script = REPO_ROOT / "distributed" / "train_ddp.py"
    param_b = PARAM_COUNT_B.get(model, 7.0)
    cmd = [ALLOC_BIN, "ghost", str(script), "--param-count-b", str(param_b), "--json"]

    ok, stdout, stderr = run_command(cmd, cwd=REPO_ROOT)

    result: dict = {
        "model": model,
        "ghost_ok": ok,
        "param_count": None,
        "weights_gb": None,
        "total_gb": None,
        "param_check": "SKIP",
        "vram_check": "SKIP",
        "notes": "",
    }

    if not ok:
        result["notes"] = f"ghost failed: {stderr.strip()[:200]}"
        return result

    # Parse JSON output — ghost outputs pretty-printed multi-line JSON
    ghost_data = None
    # Try parsing the entire stdout first (may have non-JSON preamble)
    try:
        ghost_data = json.loads(stdout)
    except json.JSONDecodeError:
        # Find the first complete JSON object in stdout using raw_decode
        decoder = json.JSONDecoder()
        brace_start = stdout.find("{")
        if brace_start >= 0:
            try:
                ghost_data, _ = decoder.raw_decode(stdout, brace_start)
            except json.JSONDecodeError:
                pass

    if ghost_data is None:
        result["notes"] = "ghost output not parseable as JSON"
        return result

    param_count = ghost_data.get("param_count")
    weights_gb = ghost_data.get("weights_gb")
    total_gb = ghost_data.get("total_gb")

    result["param_count"] = param_count
    result["weights_gb"] = weights_gb
    result["total_gb"] = total_gb

    # Validate param count against expected range
    if param_count is not None and model in EXPECTED_PARAMS:
        lo, hi = EXPECTED_PARAMS[model]
        params_b = param_count / 1e9
        if lo <= params_b <= hi:
            result["param_check"] = "PASS"
        else:
            result["param_check"] = "WARN"
            result["notes"] += f"params {params_b:.1f}B outside expected [{lo}, {hi}]B; "

    # Validate VRAM estimate against expected range
    if weights_gb is not None and model in EXPECTED_WEIGHTS_GB:
        lo, hi = EXPECTED_WEIGHTS_GB[model]
        if lo <= weights_gb <= hi:
            result["vram_check"] = "PASS"
        else:
            result["vram_check"] = "WARN"
            result["notes"] += f"weights {weights_gb:.1f}GB outside expected [{lo}, {hi}]GB; "

    return result


def eval_scan(model: str, gpu: str, num_gpus: int) -> dict:
    """Run alloc scan and evaluate the topology recommendation."""
    scan_model = SCAN_MODEL_MAP.get(model, model)
    cmd = [
        ALLOC_BIN, "scan",
        "--model", scan_model,
        "--gpu", gpu,
        "--num-gpus", str(num_gpus),
        "--json",
    ]

    ok, stdout, stderr = run_command(cmd, cwd=REPO_ROOT)

    result: dict = {
        "model": model,
        "scan_model": scan_model,
        "gpu": gpu,
        "num_gpus": num_gpus,
        "scan_ok": ok,
        "has_recommendations": False,
        "recommendation_keys": [],
        "feasibility": "UNKNOWN",
        "notes": "",
    }

    if not ok:
        result["notes"] = f"scan failed: {stderr.strip()[:200]}"
        return result

    # Parse JSON output — scan outputs pretty-printed multi-line JSON
    scan_data = None
    try:
        scan_data = json.loads(stdout)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        brace_start = stdout.find("{")
        if brace_start >= 0:
            try:
                scan_data, _ = decoder.raw_decode(stdout, brace_start)
            except json.JSONDecodeError:
                pass

    if scan_data is None:
        result["notes"] = "scan output not parseable as JSON"
        return result

    result["recommendation_keys"] = sorted(scan_data.keys()) if isinstance(scan_data, dict) else []

    # Check if scan returned any meaningful content
    result["has_recommendations"] = len(result["recommendation_keys"]) > 0

    # Feasibility assessment based on model size and GPU config
    if model == "70b" and num_gpus == 1 and "T4" in gpu:
        # 70b on single T4 should not recommend single-GPU
        recs = scan_data.get("recommendations", []) if isinstance(scan_data, dict) else []
        if isinstance(recs, list) and any("single" in str(r).lower() for r in recs):
            result["feasibility"] = "SUSPECT"
            result["notes"] += "70b on 1xT4 shouldn't recommend single-GPU; "
        else:
            result["feasibility"] = "OK"
    else:
        result["feasibility"] = "OK"

    return result


def print_ghost_table(results: list[dict]) -> None:
    """Print ghost evaluation results as ASCII table."""
    headers = ["Model", "Ghost", "Params", "Weights GB", "Total GB", "Param Check", "VRAM Check", "Notes"]
    rows = []
    for r in results:
        params_str = f"{r['param_count'] / 1e9:.1f}B" if r["param_count"] else "-"
        weights_str = f"{r['weights_gb']:.1f}" if r["weights_gb"] else "-"
        total_str = f"{r['total_gb']:.1f}" if r["total_gb"] else "-"
        rows.append([
            r["model"],
            "OK" if r["ghost_ok"] else "FAIL",
            params_str,
            weights_str,
            total_str,
            r["param_check"],
            r["vram_check"],
            r["notes"][:60] if r["notes"] else "-",
        ])

    _print_ascii_table(headers, rows)


def print_scan_table(results: list[dict]) -> None:
    """Print scan evaluation results as ASCII table."""
    headers = ["Model", "GPU", "GPUs", "Scan", "Has Recs", "Feasibility", "Notes"]
    rows = []
    for r in results:
        rows.append([
            r["model"],
            r["gpu"],
            str(r["num_gpus"]),
            "OK" if r["scan_ok"] else "FAIL",
            "Y" if r["has_recommendations"] else "N",
            r["feasibility"],
            r["notes"][:60] if r["notes"] else "-",
        ])

    _print_ascii_table(headers, rows)


def _print_ascii_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print an ASCII table (matching run_matrix.py style)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(cells, widths))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate alloc ghost/scan recommendations for large models")
    parser.add_argument("--full", action="store_true", help="Run scan evaluation too (requires ALLOC_TOKEN)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable JSON output")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=EVAL_MODELS,
        default=EVAL_MODELS,
        help="Models to evaluate",
    )
    args = parser.parse_args()

    has_token = bool(os.environ.get("ALLOC_TOKEN"))
    total_start = time.monotonic()

    # --- Ghost evaluation ---
    print("=== Ghost Evaluation ===")
    print()

    ghost_results = []
    for model in args.models:
        if not args.json_output:
            print(f"Evaluating ghost for model '{model}'...", flush=True)
        result = eval_ghost(model)
        ghost_results.append(result)

    # --- Scan evaluation (requires ALLOC_TOKEN) ---
    scan_results = []
    if args.full:
        if not has_token:
            print("\nERROR: --full requires ALLOC_TOKEN to be set")
            return 1

        print()
        print("=== Scan Evaluation ===")
        print()

        for model in args.models:
            for gpu in EVAL_GPUS:
                for ngpu in EVAL_NUM_GPUS:
                    if not args.json_output:
                        print(f"Evaluating scan for {model} on {gpu} x{ngpu}...", flush=True)
                    result = eval_scan(model, gpu, ngpu)
                    scan_results.append(result)
                    time.sleep(0.5)

    total_time = time.monotonic() - total_start

    # --- Output ---
    if args.json_output:
        output = {
            "ghost_results": ghost_results,
            "scan_results": scan_results,
            "total_time": total_time,
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        print("--- Ghost Results ---")
        print_ghost_table(ghost_results)

        if scan_results:
            print()
            print("--- Scan Results ---")
            print_scan_table(scan_results)

        # Summary
        ghost_pass = sum(1 for r in ghost_results if r["ghost_ok"])
        ghost_total = len(ghost_results)
        scan_pass = sum(1 for r in scan_results if r["scan_ok"])
        scan_total = len(scan_results)

        print()
        print(f"Ghost: {ghost_pass}/{ghost_total} OK | Scan: {scan_pass}/{scan_total} OK | Time: {total_time:.1f}s")

    # Fail if any ghost calls failed
    if any(not r["ghost_ok"] for r in ghost_results):
        return 1
    if any(not r["scan_ok"] for r in scan_results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
