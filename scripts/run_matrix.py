#!/usr/bin/env python3
"""Matrix test runner for alloc-validate.

Runs all model/GPU combos across frameworks and prints a results table.

Usage:
    python scripts/run_matrix.py                   # full matrix
    python scripts/run_matrix.py --quick            # 1 model per framework
    python scripts/run_matrix.py --framework pytorch
    python scripts/run_matrix.py --json             # machine-readable output
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Resolve binary paths from env vars (set by Makefile) or fall back to bare commands.
# Use absolute() not resolve() to avoid following symlinks (venv python -> system python).
_alloc_raw = os.environ.get("ALLOC_BIN", "alloc")
_python_raw = os.environ.get("PYTHON_BIN", "python")
ALLOC_BIN = str(Path(_alloc_raw).absolute()) if "/" in _alloc_raw else _alloc_raw
PYTHON_BIN = str(Path(_python_raw).absolute()) if "/" in _python_raw else _python_raw

PYTORCH_MODELS = ["small-cnn", "medium-cnn", "large-cnn", "mlp-small", "mlp-large"]
HF_MODELS = ["distilbert-tiny", "distilbert-small", "gpt2-tiny", "bert-tiny"]
LIGHTNING_MODELS = ["small-cnn", "medium-cnn", "mlp"]
RAY_MODELS = ["small-cnn", "medium-cnn", "mlp"]
DISTRIBUTED_STRATEGIES = ["ddp", "fsdp", "pp", "tp"]
HYBRID_STRATEGIES = ["tp_dp", "pp_dp", "3d", "3d_fsdp"]
ALL_DISTRIBUTED_STRATEGIES = DISTRIBUTED_STRATEGIES + HYBRID_STRATEGIES
DISTRIBUTED_MODELS = ["small", "medium"]
GHOST_DIR = REPO_ROOT / "scan-only"

SCAN_GPUS = ["A100-80GB", "H100-80GB", "T4-16GB", "V100-32GB"]
SCAN_MODELS = ["llama-3-8b", "llama-3-70b", "mistral-7b"]
SCAN_NUM_GPUS = [1, 4]

# NOTE: This is a test matrix default, not a product ceiling.
# Override with: ALLOC_VALIDATE_SCAN_NUM_GPUS="1,2,4,8,16"
_scan_env = os.environ.get("ALLOC_VALIDATE_SCAN_NUM_GPUS", "").strip()
if _scan_env:
    try:
        SCAN_NUM_GPUS = [int(x.strip()) for x in _scan_env.split(",") if x.strip()]
    except Exception:
        pass


def load_artifact_keys(path: Path) -> Optional[list[str]]:
    """Load artifact and return sorted top-level keys, or None on failure."""
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass
        data = json.loads(raw)
        return sorted(data.keys())
    except (json.JSONDecodeError, OSError):
        return None


def run_command(cmd: list[str], cwd: Path, timeout: int = 300) -> tuple[bool, float, str]:
    """Run a command, return (success, duration_seconds, output)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        output = result.stdout + result.stderr
        return result.returncode == 0, duration, output
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        return False, duration, "TIMEOUT"
    except Exception as e:
        duration = time.monotonic() - start
        return False, duration, str(e)


def run_distributed_combo(
    strategy: str,
    model: str,
    max_steps: int,
    num_gpus: int = 1,
) -> dict:
    """Run alloc run torchrun train_{strategy}.py for a distributed combo."""
    workdir = REPO_ROOT / "distributed"
    script = f"train_{strategy}.py"

    # Check if torchrun is functional (may be installed but broken on Python 3.9)
    # Derive default torchrun from ALLOC_BIN's directory (same venv bin/)
    default_torchrun = str(Path(ALLOC_BIN).parent / "torchrun") if os.sep in ALLOC_BIN else "torchrun"
    torchrun_bin = os.environ.get("TORCHRUN_BIN", default_torchrun)
    torchrun_ok = False
    try:
        subprocess.run(
            [torchrun_bin, "--help"], capture_output=True, timeout=10,
        )
        torchrun_ok = True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    if torchrun_ok:
        cmd = [
            ALLOC_BIN, "run", "--", torchrun_bin, "--nproc_per_node", "1",
            script, "--model", model, "--max-steps", str(max_steps),
        ]
    else:
        cmd = [
            ALLOC_BIN, "run", "--", PYTHON_BIN, script,
            "--model", model, "--max-steps", str(max_steps),
        ]

    # Clean old artifact
    for ext in ("alloc_artifact.json.gz", "alloc_artifact.json"):
        artifact_path = workdir / ext
        if artifact_path.exists():
            artifact_path.unlink()

    ok, duration, output = run_command(cmd, cwd=workdir)

    # Find artifact
    keys: list[str] = []
    artifact_path = workdir / "alloc_artifact.json.gz"
    if not artifact_path.exists():
        artifact_path = workdir / "alloc_artifact.json"
    loaded_keys = load_artifact_keys(artifact_path)
    if loaded_keys is not None:
        keys = loaded_keys

    status = "PASS" if ok and keys else "FAIL"
    return {
        "framework": f"distributed/{strategy}",
        "model": model,
        "gpus": num_gpus,
        "status": status,
        "duration": duration,
        "keys": keys,
    }


def run_training_combo(
    framework: str,
    model: str,
    max_steps: int,
    num_gpus: int = 1,
) -> dict:
    """Run alloc run python train.py for a given framework/model combo."""
    if framework == "pytorch":
        workdir = REPO_ROOT / "pytorch"
    elif framework == "huggingface":
        workdir = REPO_ROOT / "huggingface"
    elif framework == "lightning":
        workdir = REPO_ROOT / "lightning"
    elif framework == "ray":
        workdir = REPO_ROOT / "ray"
    else:
        return {"framework": framework, "model": model, "gpus": num_gpus,
                "status": "SKIP", "duration": 0.0, "keys": []}

    cmd = [
        ALLOC_BIN, "run", "--", PYTHON_BIN, "train.py",
        "--model", model,
        "--max-steps", str(max_steps),
    ]

    # Clean old artifact
    for ext in ("alloc_artifact.json.gz", "alloc_artifact.json"):
        artifact_path = workdir / ext
        if artifact_path.exists():
            artifact_path.unlink()

    ok, duration, output = run_command(cmd, cwd=workdir)

    # Find artifact
    keys: list[str] = []
    artifact_path = workdir / "alloc_artifact.json.gz"
    if not artifact_path.exists():
        artifact_path = workdir / "alloc_artifact.json"
    loaded_keys = load_artifact_keys(artifact_path)
    if loaded_keys is not None:
        keys = loaded_keys

    status = "PASS" if ok and keys else "FAIL"
    return {
        "framework": framework,
        "model": model,
        "gpus": num_gpus,
        "status": status,
        "duration": duration,
        "keys": keys,
    }


def run_ghost_combo(target_file: str) -> dict:
    """Run alloc ghost on a target file."""
    cmd = [ALLOC_BIN, "ghost", target_file]
    ok, duration, output = run_command(cmd, cwd=GHOST_DIR)
    status = "PASS" if ok else "FAIL"
    return {
        "framework": "ghost",
        "model": target_file,
        "gpus": "-",
        "status": status,
        "duration": duration,
        "keys": [],
    }


def run_scan_combo(model: str, gpu: str, num_gpus: int) -> dict:
    """Run alloc scan for a model/GPU combo."""
    cmd = [ALLOC_BIN, "scan", "--model", model, "--gpu", gpu, "--num-gpus", str(num_gpus)]
    ok, duration, output = run_command(cmd, cwd=GHOST_DIR)
    status = "PASS" if ok else "FAIL"
    return {
        "framework": "scan",
        "model": f"{model}",
        "gpus": f"{gpu}/{num_gpus}",
        "status": status,
        "duration": duration,
        "keys": [],
    }


def discover_ghost_targets() -> list[str]:
    """Find all ghost_target*.py files in scan-only/."""
    targets = sorted(p.name for p in GHOST_DIR.glob("ghost_target*.py"))
    return targets


def print_table(results: list[dict]) -> None:
    """Print an ASCII results table."""
    headers = ["Framework", "Model", "GPUs", "Status", "Duration", "Artifact Keys"]
    rows = []
    for r in results:
        keys_str = ", ".join(r["keys"]) if r["keys"] else "-"
        rows.append([
            r["framework"],
            r["model"],
            str(r["gpus"]),
            r["status"],
            f"{r['duration']:.1f}s",
            keys_str,
        ])

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for cell, w in zip(cells, widths):
            parts.append(cell.ljust(w))
        return " | ".join(parts)

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def print_summary(results: list[dict], total_time: float) -> None:
    """Print pass/fail/skip summary."""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    total = len(results)
    print(f"\nTotal: {total} | PASS: {passed} | FAIL: {failed} | SKIP: {skipped} | Time: {total_time:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run alloc-validate model/GPU matrix")
    parser.add_argument(
        "--framework",
        choices=["pytorch", "huggingface", "lightning", "ray", "scan-only", "distributed", "all"],
        default="all",
        help="Which framework to test (default: all)",
    )
    parser.add_argument("--quick", action="store_true", help="1 model per framework for fast iteration")
    parser.add_argument("--max-steps", type=int, default=30, help="Max training steps per run")
    parser.add_argument("--include-multi-gpu", action="store_true", help="Add num_gpus=2,4 variations")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Machine-readable JSON output")
    args = parser.parse_args()

    has_token = bool(os.environ.get("ALLOC_TOKEN"))
    results: list[dict] = []
    total_start = time.monotonic()

    run_pytorch = args.framework in ("pytorch", "all")
    run_hf = args.framework in ("huggingface", "all")
    run_lightning = args.framework in ("lightning", "all")
    run_ray = args.framework in ("ray", "all")
    run_scan = args.framework in ("scan-only", "all")
    run_distributed = args.framework in ("distributed", "all")

    gpu_counts = [1]
    if args.include_multi_gpu:
        gpu_counts.extend([2, 4])

    # --- PyTorch ---
    if run_pytorch:
        models = PYTORCH_MODELS[:1] if args.quick else PYTORCH_MODELS
        for model in models:
            for ngpu in gpu_counts:
                if not args.json_output:
                    print(f"Running pytorch/{model} (gpus={ngpu})...", flush=True)
                r = run_training_combo("pytorch", model, args.max_steps, ngpu)
                results.append(r)

    # --- HuggingFace ---
    if run_hf:
        models = HF_MODELS[:1] if args.quick else HF_MODELS
        for model in models:
            for ngpu in gpu_counts:
                if not args.json_output:
                    print(f"Running huggingface/{model} (gpus={ngpu})...", flush=True)
                r = run_training_combo("huggingface", model, args.max_steps, ngpu)
                results.append(r)

    # --- Lightning ---
    if run_lightning:
        models = LIGHTNING_MODELS[:1] if args.quick else LIGHTNING_MODELS
        for model in models:
            for ngpu in gpu_counts:
                if not args.json_output:
                    print(f"Running lightning/{model} (gpus={ngpu})...", flush=True)
                r = run_training_combo("lightning", model, args.max_steps, ngpu)
                results.append(r)

    # --- Ray ---
    if run_ray:
        models = RAY_MODELS[:1] if args.quick else RAY_MODELS
        for model in models:
            for ngpu in gpu_counts:
                if not args.json_output:
                    print(f"Running ray/{model} (gpus={ngpu})...", flush=True)
                r = run_training_combo("ray", model, args.max_steps, ngpu)
                results.append(r)

    # --- Distributed ---
    if run_distributed:
        strategies = ALL_DISTRIBUTED_STRATEGIES[:1] if args.quick else ALL_DISTRIBUTED_STRATEGIES
        models = DISTRIBUTED_MODELS[:1] if args.quick else DISTRIBUTED_MODELS
        for strategy in strategies:
            for model in models:
                if not args.json_output:
                    print(f"Running distributed/{strategy}/{model}...", flush=True)
                r = run_distributed_combo(strategy, model, args.max_steps)
                results.append(r)

    # --- Ghost ---
    if run_scan:
        targets = discover_ghost_targets()
        if args.quick:
            targets = targets[:1]
        for target in targets:
            if not args.json_output:
                print(f"Running ghost/{target}...", flush=True)
            r = run_ghost_combo(target)
            results.append(r)

    # --- Scan (requires ALLOC_TOKEN) ---
    if run_scan and has_token:
        scan_models = SCAN_MODELS[:1] if args.quick else SCAN_MODELS
        scan_gpus = SCAN_GPUS[:1] if args.quick else SCAN_GPUS
        scan_ngpus = SCAN_NUM_GPUS[:1] if args.quick else SCAN_NUM_GPUS
        for model in scan_models:
            for gpu in scan_gpus:
                for ngpu in scan_ngpus:
                    if not args.json_output:
                        print(f"Running scan/{model}/{gpu}/{ngpu}gpu...", flush=True)
                    r = run_scan_combo(model, gpu, ngpu)
                    results.append(r)
                    time.sleep(0.5)
    elif run_scan and not has_token:
        results.append({
            "framework": "scan",
            "model": "(all)",
            "gpus": "-",
            "status": "SKIP",
            "duration": 0.0,
            "keys": [],
        })

    total_time = time.monotonic() - total_start

    if args.json_output:
        output = {"results": results, "total_time": total_time}
        print(json.dumps(output, indent=2))
    else:
        print()
        print_table(results)
        print_summary(results, total_time)

    failed = any(r["status"] == "FAIL" for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
