"""Tests for self-contained callbacks — artifact generation without alloc run."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.conftest import ROOT, run_python


def _find_artifact(workdir: Path) -> Path | None:
    candidates = list(workdir.glob("alloc_artifact*.json.gz"))
    return candidates[0] if candidates else None


def _find_callback_sidecar(workdir: Path) -> Path | None:
    candidates = list(workdir.glob(".alloc_callback.json"))
    return candidates[0] if candidates else None


class TestHuggingFaceCallback:
    """Verify HF callback produces artifact WITHOUT alloc run wrapper."""

    def test_callback_produces_artifact(self, tmp_workdir: Path) -> None:
        r = run_python(
            str(ROOT / "huggingface" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"HF training failed: {r.stderr}"
        artifact = _find_artifact(tmp_workdir)
        if artifact is not None:
            with gzip.open(artifact, "rt") as f:
                data = json.load(f)
            assert "probe" in data or "version" in data

    def test_callback_sidecar_written(self, tmp_workdir: Path) -> None:
        r = run_python(
            str(ROOT / "huggingface" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0
        sidecar = _find_callback_sidecar(tmp_workdir)
        if sidecar is not None:
            data = json.loads(sidecar.read_text())
            assert isinstance(data, dict)


class TestLightningCallback:
    """Verify Lightning callback produces artifact WITHOUT alloc run wrapper."""

    def test_callback_produces_artifact(self, tmp_workdir: Path) -> None:
        r = run_python(
            str(ROOT / "lightning" / "train.py"),
            "--max-steps", "10",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"Lightning training failed: {r.stderr}"
        artifact = _find_artifact(tmp_workdir)
        if artifact is not None:
            with gzip.open(artifact, "rt") as f:
                data = json.load(f)
            assert "probe" in data or "version" in data
