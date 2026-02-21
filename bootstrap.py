#!/usr/bin/env python3
"""alloc-validate setup script.

Creates a venv, installs all dependencies, verifies the installation,
detects local GPUs, and optionally bootstraps GCP for remote GPU testing.

Usage:
    python3 bootstrap.py              # install deps + detect GPUs
    python3 bootstrap.py --gcp        # also set up GCP auth + quota check
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- ANSI colors (same palette as setup-gcp.sh) ---

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
BOLD = "\033[1m"
NC = "\033[0m"

VENV_DIR = Path(".venv")
VENV_BIN = VENV_DIR / "bin"
VENV_PIP = VENV_BIN / "pip"
VENV_PYTHON = VENV_BIN / "python"
VENV_ALLOC = VENV_BIN / "alloc"

SCRIPT_DIR = Path(__file__).resolve().parent


def ok(msg: str) -> None:
    print(f"  {GREEN}OK:{NC} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN:{NC} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}ERROR:{NC} {msg}")


def header(step: int, total: int, title: str) -> None:
    print(f"\n{BLUE}{BOLD}[{step}/{total}]{NC} {BOLD}{title}{NC}")


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


# ------------------------------------------------------------------
# Steps
# ------------------------------------------------------------------


def check_python():
    v = sys.version_info
    if v < (3, 9):
        # Try to find a suitable Python and re-exec
        for candidate in ("python3.12", "python3.11", "python3.10", "python3.9"):
            path = shutil.which(candidate)
            if path:
                print("  Found %s, re-launching..." % candidate)
                os.execv(path, [path] + sys.argv)
        # No suitable Python found
        fail("Python 3.9+ required, got %d.%d.%d" % (v.major, v.minor, v.micro))
        print("  Install Python 3.9+ or use Docker instead:")
        print("    make docker-build && make docker-test")
        sys.exit(1)


def create_venv(total: int) -> None:
    header(1, total, "Creating virtual environment...")
    if VENV_PYTHON.exists():
        # Verify the venv is functional
        r = run([str(VENV_PYTHON), "-c", "import sys; print(sys.prefix)"])
        if r.returncode == 0:
            ok(f"{VENV_DIR} (already exists)")
            return
        warn(f"{VENV_DIR} exists but is broken — recreating")
        shutil.rmtree(VENV_DIR)

    import venv
    venv.create(str(VENV_DIR), with_pip=True)
    ok(str(VENV_DIR))


def install_deps(total: int) -> None:
    header(2, total, "Installing dependencies...")

    r = run(
        [str(VENV_PIP), "install", "-e", ".[all]"],
        cwd=str(SCRIPT_DIR),
    )
    if r.returncode != 0:
        fail("pip install failed")
        print(r.stderr[-2000:] if len(r.stderr) > 2000 else r.stderr)
        sys.exit(1)

    # Report key packages
    alloc_ver = _pkg_version("alloc")
    if alloc_ver:
        ok(f"alloc v{alloc_ver}")
    else:
        warn("alloc installed but version unknown")

    extras = []
    for pkg in ("torch", "transformers", "lightning", "ray"):
        v = _pkg_version(pkg)
        if v:
            extras.append(pkg)
    if extras:
        ok(", ".join(extras))


def _pkg_version(pkg):
    r = run([
        str(VENV_PYTHON), "-c",
        f"import importlib.metadata; print(importlib.metadata.version('{pkg}'))",
    ])
    if r.returncode == 0:
        return r.stdout.strip()
    return None


def verify_install(total: int) -> None:
    header(3, total, "Verifying installation...")

    # alloc version
    r = run([str(VENV_ALLOC), "version"])
    if r.returncode == 0:
        ver = r.stdout.strip()
        ok(f"alloc version → {ver}")
    else:
        warn("alloc version command failed (may still work)")

    # torch import
    r = run([str(VENV_PYTHON), "-c", "import torch"])
    if r.returncode == 0:
        ok("torch import")
    else:
        warn("torch import failed — GPU workloads will not run")


def detect_gpus(total: int) -> int:
    header(4, total, "Detecting GPUs...")

    r = run([
        str(VENV_PYTHON), "-c",
        "import torch, json; "
        "n = torch.cuda.device_count(); "
        "gpus = [{"
        "'name': torch.cuda.get_device_name(i), "
        "'vram_mb': torch.cuda.get_device_properties(i).total_mem // (1024*1024)"
        "} for i in range(n)]; "
        "print(json.dumps({'count': n, 'gpus': gpus}))",
    ])

    if r.returncode != 0 or not r.stdout.strip():
        print("  No CUDA GPUs detected (CPU-only mode)")
        return 0

    try:
        data = json.loads(r.stdout.strip())
    except json.JSONDecodeError:
        print("  No CUDA GPUs detected (CPU-only mode)")
        return 0

    count = data.get("count", 0)
    if count == 0:
        print("  No CUDA GPUs detected (CPU-only mode)")
        return 0

    for gpu in data.get("gpus", []):
        name = gpu.get("name", "unknown")
        vram = gpu.get("vram_mb", 0)
        ok(f"{name} ({vram} MB VRAM)")
    ok(f"{count} GPU(s) available")
    return count


def setup_gcp(total: int) -> None:
    header(4, total, "GCP GPU setup...")

    script = SCRIPT_DIR / "scripts" / "gpu" / "setup-gcp.sh"
    if not script.exists():
        fail(f"GCP setup script not found: {script}")
        sys.exit(1)

    # Delegate to existing bash script (inherits stdio for interactive prompts)
    result = subprocess.run(["bash", str(script)])
    if result.returncode != 0:
        fail("GCP setup failed — see output above")
        sys.exit(1)


def print_summary(gpu_count: int, gcp: bool) -> None:
    print(f"\n{GREEN}{BOLD}Setup complete!{NC}\n")
    print("  Next steps:")
    print("    source .venv/bin/activate")
    print("    make validate-free          # run all workloads (CPU)")
    print("    make matrix-quick           # quick smoke test")

    print("")
    print("  Authentication (optional, unlocks full-tier features):")
    print("    alloc login --browser       # opens browser for Google/Microsoft sign-in")
    print("    alloc login --token <tok>   # paste a token directly")

    if gpu_count > 0:
        print("")
        print("  GPU workloads:")
        print("    make validate-full          # full validation (requires ALLOC_TOKEN)")
        print("    make matrix                 # full model/GPU variation matrix")

    if not gcp:
        print("")
        print("  For GPU testing:")
        print("    python3 bootstrap.py --gcp   # set up GCP + check GPU quota")
        print("    See GPU_TESTING.md for cloud GPU options")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up alloc-validate environment")
    parser.add_argument(
        "--gcp", action="store_true",
        help="Also set up GCP authentication and check GPU quota",
    )
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    check_python()

    total = 4

    create_venv(total)
    install_deps(total)
    verify_install(total)

    if args.gcp:
        setup_gcp(total)
        gpu_count = 0  # GCP path — local GPU count not relevant
    else:
        gpu_count = detect_gpus(total)

    print_summary(gpu_count, args.gcp)


if __name__ == "__main__":
    main()
