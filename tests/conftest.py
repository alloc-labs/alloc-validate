"""Shared fixtures for the alloc-validate pytest suite."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_alloc(*args: str, cwd: str | Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run an alloc CLI command and return the CompletedProcess."""
    return subprocess.run(
        ["alloc", *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        timeout=timeout,
    )


def run_python(*args: str, cwd: str | Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a python command and return the CompletedProcess."""
    return subprocess.run(
        ["python", *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        timeout=timeout,
    )


@pytest.fixture
def alloc_bin() -> str:
    path = shutil.which("alloc")
    assert path, "alloc CLI not found on PATH"
    return path


@pytest.fixture
def has_token() -> bool:
    return bool(os.environ.get("ALLOC_TOKEN"))


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """A temporary working directory for tests that produce artifacts."""
    return tmp_path


def parse_diagnose_json(result: subprocess.CompletedProcess) -> dict:
    """Parse alloc diagnose --json output."""
    assert result.returncode == 0, f"alloc diagnose failed: {result.stderr}"
    return json.loads(result.stdout)


def findings_rule_ids(data: dict) -> set[str]:
    """Extract rule IDs from diagnose JSON output."""
    return {f["rule_id"] for f in data.get("findings", [])}
