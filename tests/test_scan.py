"""Tests for alloc scan — remote API queries (requires ALLOC_TOKEN)."""

from __future__ import annotations

import json
import os

import pytest

from tests.conftest import run_alloc

needs_token = pytest.mark.skipif(
    not os.environ.get("ALLOC_TOKEN"),
    reason="ALLOC_TOKEN not set — skipping scan tests",
)


@needs_token
class TestScan:
    def test_scan_json(self) -> None:
        r = run_alloc(
            "scan", "--model", "llama-3-8b", "--gpu", "A100-80GB", "--json",
        )
        assert r.returncode == 0, f"alloc scan failed: {r.stderr}"
        data = json.loads(r.stdout)
        assert isinstance(data, dict)
