#!/usr/bin/env python3
"""alloc-validate setup script.

Creates a reproducible Python environment, installs all dependencies, verifies
the installation, detects local GPUs, and optionally bootstraps GCP for remote
GPU testing.

Usage:
    python3 bootstrap.py                    # auto-select Python, install deps
    python3 bootstrap.py --python python3.10
    python3 bootstrap.py --gcp              # also set up GCP auth + quota check
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

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
MIN_PYTHON = (3, 9)
PYTHON_CANDIDATES = (
    "python3.12",
    "python3.11",
    "python3.10",
    "python3.9",
    "python3",
    "python",
)

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


def _resolve_python(binary: str) -> Optional[str]:
    """Resolve a Python executable from PATH or absolute path."""
    p = Path(binary)
    if p.is_absolute() and p.exists():
        return str(p)
    return shutil.which(binary)


def _probe_python(python_bin: str) -> Optional[dict]:
    """Return version/ssl probe for an interpreter, or None if unusable."""
    code = (
        "import json, sys\n"
        "ssl_ok = True\n"
        "try:\n"
        "    import ssl  # noqa: F401\n"
        "except Exception:\n"
        "    ssl_ok = False\n"
        "print(json.dumps({"
        "'major': sys.version_info[0], "
        "'minor': sys.version_info[1], "
        "'micro': sys.version_info[2], "
        "'ssl_ok': ssl_ok"
        "}))"
    )
    result = run([python_bin, "-c", code])
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None


def _python_is_supported(probe: Optional[dict]) -> bool:
    if not probe:
        return False
    version_ok = (probe["major"], probe["minor"]) >= MIN_PYTHON
    return version_ok and bool(probe.get("ssl_ok"))


def _reexec_with(python_bin: str) -> None:
    """Re-launch bootstrap with a different interpreter."""
    print(f"  Using interpreter: {python_bin}")
    os.execv(python_bin, [python_bin] + sys.argv)


def check_python(requested_python: Optional[str]) -> None:
    """Ensure we run under a supported interpreter (>=3.9 + SSL)."""
    current = str(Path(sys.executable).resolve())
    current_probe = _probe_python(current)

    # Optional override path from flag or env.
    override = requested_python or os.environ.get("ALLOC_VALIDATE_PYTHON")
    if override:
        override_path = _resolve_python(override)
        if not override_path:
            fail(f"Requested interpreter not found: {override}")
            print("  Tip: use an absolute path or install the requested Python.")
            sys.exit(1)
        override_probe = _probe_python(override_path)
        if not _python_is_supported(override_probe):
            if override_probe:
                fail(
                    "Requested interpreter is unsupported: "
                    f"{override_probe['major']}.{override_probe['minor']}.{override_probe['micro']} "
                    f"(ssl_ok={override_probe['ssl_ok']})"
                )
            else:
                fail(f"Requested interpreter is not runnable: {override_path}")
            print("  Need Python 3.9+ with SSL support enabled.")
            sys.exit(1)

        if Path(override_path).resolve() != Path(current):
            _reexec_with(override_path)
        return

    # Current interpreter already good.
    if _python_is_supported(current_probe):
        return

    # Auto-find a better interpreter and re-exec.
    for candidate in PYTHON_CANDIDATES:
        path = _resolve_python(candidate)
        if not path:
            continue
        if Path(path).resolve() == Path(current):
            continue
        probe = _probe_python(path)
        if _python_is_supported(probe):
            print(
                "  Current Python is unsupported, re-launching with "
                f"{candidate} ({probe['major']}.{probe['minor']}.{probe['micro']})..."
            )
            _reexec_with(path)

    # No suitable Python found.
    if current_probe:
        fail(
            "Python 3.9+ with SSL support required, got "
            f"{current_probe['major']}.{current_probe['minor']}.{current_probe['micro']} "
            f"(ssl_ok={current_probe['ssl_ok']})"
        )
    else:
        fail(f"Could not probe current interpreter: {current}")
    print("  Install Python 3.9+ (or pyenv) and retry.")
    print("  Example: pyenv install 3.10.12 && pyenv local 3.10.12")
    print("  Or override interpreter explicitly:")
    print("    python3 bootstrap.py --python python3.10")
    sys.exit(1)


def _venv_is_healthy() -> bool:
    """A healthy venv has python + pip available."""
    if not VENV_PYTHON.exists():
        return False
    pip_check = run([str(VENV_PYTHON), "-m", "pip", "--version"])
    return pip_check.returncode == 0


def _create_venv_with_virtualenv() -> None:
    """Fallback for environments missing ensurepip/python-venv."""
    pip_check = run([sys.executable, "-m", "pip", "--version"])
    if pip_check.returncode != 0:
        fail("Current Python lacks pip; cannot install virtualenv fallback.")
        print("  Install pip for this interpreter and retry.")
        print("  Ubuntu/Debian: sudo apt-get install -y python3-pip")
        sys.exit(1)

    install_virtualenv = run(
        [sys.executable, "-m", "pip", "install", "--user", "virtualenv"],
    )
    if install_virtualenv.returncode != 0:
        fail("Failed to install virtualenv fallback.")
        print(install_virtualenv.stderr[-2000:] if install_virtualenv.stderr else "")
        sys.exit(1)

    create = run([sys.executable, "-m", "virtualenv", str(VENV_DIR)])
    if create.returncode != 0:
        fail("virtualenv fallback failed to create .venv")
        print(create.stderr[-2000:] if create.stderr else "")
        sys.exit(1)


def create_venv(total: int) -> None:
    header(1, total, "Creating virtual environment...")
    if _venv_is_healthy():
        ok(f"{VENV_DIR} (already exists)")
        return

    if VENV_DIR.exists():
        warn(f"{VENV_DIR} exists but is broken — recreating")
        shutil.rmtree(VENV_DIR)

    import venv
    try:
        venv.create(str(VENV_DIR), with_pip=True)
    except Exception as exc:
        warn(f"stdlib venv creation failed ({exc.__class__.__name__}); trying virtualenv fallback")
        _create_venv_with_virtualenv()

    if not _venv_is_healthy():
        warn("venv exists but pip is missing; trying virtualenv fallback")
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        _create_venv_with_virtualenv()

    if not _venv_is_healthy():
        fail("Virtual environment creation failed (pip unavailable in .venv)")
        sys.exit(1)

    ok(f"{VENV_DIR} (python={sys.version_info.major}.{sys.version_info.minor})")


def install_deps(total: int) -> None:
    header(2, total, "Installing dependencies...")

    prep = run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=str(SCRIPT_DIR),
    )
    if prep.returncode != 0:
        fail("pip bootstrap failed")
        print(prep.stderr[-2000:] if len(prep.stderr) > 2000 else prep.stderr)
        sys.exit(1)

    r = run(
        [str(VENV_PYTHON), "-m", "pip", "install", "-e", ".[all]"],
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
        "--python",
        help="Python interpreter to use (e.g. python3.10 or /usr/bin/python3.10)",
    )
    parser.add_argument(
        "--gcp", action="store_true",
        help="Also set up GCP authentication and check GPU quota",
    )
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)
    check_python(args.python)

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
