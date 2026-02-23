"""Tests for alloc run — wraps training and produces artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.conftest import ROOT, run_alloc, run_python

# These tests actually run training scripts — they're slower (~30s each)


def _find_artifact(workdir: Path) -> Path | None:
    """Find the most recent alloc_artifact.json.gz in a directory."""
    candidates = list(workdir.glob("alloc_artifact*.json.gz"))
    return candidates[0] if candidates else None


def _load_artifact(path: Path) -> dict:
    """Load a gzipped JSON artifact."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


class TestRunPyTorch:
    def test_run_produces_artifact(self, tmp_workdir: Path) -> None:
        r = run_alloc(
            "run", "--", "python", str(ROOT / "pytorch" / "train.py"),
            "--max-steps", "20",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"alloc run failed: {r.stderr}"
        artifact = _find_artifact(tmp_workdir)
        assert artifact is not None, "No artifact produced"
        data = _load_artifact(artifact)
        assert "probe" in data
        assert "hardware" in data

    def test_run_json_output(self, tmp_workdir: Path) -> None:
        r = run_alloc(
            "run", "--json", "--", "python", str(ROOT / "pytorch" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"alloc run --json failed: {r.stderr}"
        # JSON is appended after the training script's stdout output.
        # Extract the JSON object that starts with '{'.
        stdout = r.stdout
        json_start = stdout.rfind("\n{")
        if json_start == -1:
            json_start = stdout.find("{")
        assert json_start != -1, f"No JSON object found in stdout: {stdout[:200]}"
        data = json.loads(stdout[json_start:])
        assert isinstance(data, dict)


class TestRunHuggingFace:
    def test_run_produces_artifact(self, tmp_workdir: Path) -> None:
        r = run_alloc(
            "run", "--", "python", str(ROOT / "huggingface" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"alloc run failed: {r.stderr}"
        artifact = _find_artifact(tmp_workdir)
        assert artifact is not None, "No artifact produced"


class TestRunLightning:
    def test_run_produces_artifact(self, tmp_workdir: Path) -> None:
        r = run_alloc(
            "run", "--", "python", str(ROOT / "lightning" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"alloc run failed: {r.stderr}"
        artifact = _find_artifact(tmp_workdir)
        assert artifact is not None, "No artifact produced"


class TestRunRay:
    def test_run_produces_artifact(self, tmp_workdir: Path) -> None:
        r = run_alloc(
            "run", "--", "python", str(ROOT / "ray" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"alloc run failed: {r.stderr}"
        artifact = _find_artifact(tmp_workdir)
        assert artifact is not None, "No artifact produced"
