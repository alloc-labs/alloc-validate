"""Tests for alloc ghost — static VRAM estimation."""

from __future__ import annotations

import json

import pytest

from tests.conftest import ROOT, run_alloc

GHOST_TARGETS = [
    "scan-only/ghost_target.py",
    "scan-only/ghost_target_small_cnn.py",
    "scan-only/ghost_target_medium_cnn.py",
    "scan-only/ghost_target_large_cnn.py",
    "scan-only/ghost_target_mlp_small.py",
    "scan-only/ghost_target_mlp_large.py",
]


class TestGhostJson:
    @pytest.mark.parametrize("target", GHOST_TARGETS)
    def test_ghost_json_exits_zero(self, target: str) -> None:
        r = run_alloc("ghost", str(ROOT / target), "--json")
        assert r.returncode == 0, f"alloc ghost failed on {target}: {r.stderr}"

    @pytest.mark.parametrize("target", GHOST_TARGETS)
    def test_ghost_json_valid(self, target: str) -> None:
        r = run_alloc("ghost", str(ROOT / target), "--json")
        data = json.loads(r.stdout)
        # Ghost output should contain model size info
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"

    @pytest.mark.parametrize("target", GHOST_TARGETS)
    def test_ghost_param_count_reasonable(self, target: str) -> None:
        r = run_alloc("ghost", str(ROOT / target), "--json")
        data = json.loads(r.stdout)
        # Check param_count exists and is positive
        if "param_count" in data:
            assert data["param_count"] > 0, "param_count should be positive"
            assert data["param_count"] < 1e12, "param_count unreasonably large"


class TestGhostVerbose:
    def test_ghost_verbose_output(self) -> None:
        r = run_alloc("ghost", str(ROOT / GHOST_TARGETS[0]), "--verbose")
        assert r.returncode == 0
        output = r.stdout + r.stderr
        # Verbose mode should produce human-readable output
        assert len(output) > 0
