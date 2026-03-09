"""Tests for P2.5 — alloc merge command and per-rank artifact handling.

Creates synthetic rank artifacts and tests the merge workflow:
  - alloc merge auto-discovers rank artifacts
  - Merged artifact has straggler ratio
  - Merged artifact preserves per-rank data
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.conftest import ROOT, run_alloc


def _write_rank_artifact(
    path: Path,
    rank: int,
    peak_vram_mb: float,
    step_time_p50: float,
    gpu_name: str = "NVIDIA A100-SXM4-80GB",
) -> None:
    """Write a minimal rank artifact for merge testing."""
    data = {
        "version": "0.0.4",
        "timestamp": "2026-02-23T12:00:00Z",
        "probe": {
            "peak_vram_mb": peak_vram_mb,
            "avg_gpu_util": 75.0,
            "duration_seconds": 120.0,
            "stop_reason": "calibrate_exit",
            "step_count": 50,
            "step_time_ms_p50": step_time_p50,
            "step_time_ms_p90": step_time_p50 * 1.2,
            "samples_per_sec": 1000.0 / step_time_p50 * 32,
            "step_times_raw": [step_time_p50 + (i % 5) for i in range(50)],
            "source": "callback",
            "is_distributed": True,
            "rank": rank,
            "world_size": 4,
        },
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_total_vram_mb": 81920,
            "num_gpus_detected": 4,
        },
        "context": {},
    }
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        json.dump(data, f)


class TestMergeCommand:
    """Test alloc merge CLI command."""

    def test_merge_rank_artifacts(self, tmp_workdir: Path) -> None:
        """alloc merge should combine rank artifacts."""
        # Write 4 rank artifacts with varying VRAM and step times
        for rank in range(4):
            peak = 30000 + rank * 1000  # 30000, 31000, 32000, 33000
            step_time = 100.0 + rank * 10  # 100, 110, 120, 130
            path = tmp_workdir / f"alloc_artifact_rank{rank}.json.gz"
            _write_rank_artifact(path, rank, peak, step_time)

        r = run_alloc(
            "merge",
            "--json",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0, f"alloc merge failed: {r.stderr}"
        merged = json.loads(r.stdout)
        assert merged["gpu_count"] == 4
        assert merged["peak_vram_mb"] == 33000.0  # max across ranks

    def test_merge_detects_straggler(self, tmp_workdir: Path) -> None:
        """Merge should compute straggler ratio from step time variance."""
        # Rank 3 is 50% slower → straggler_ratio = 1.5
        for rank in range(4):
            step_time = 100.0 if rank < 3 else 150.0
            path = tmp_workdir / f"alloc_artifact_rank{rank}.json.gz"
            _write_rank_artifact(path, rank, 30000, step_time)

        r = run_alloc("merge", "--json", cwd=tmp_workdir)
        assert r.returncode == 0
        merged = json.loads(r.stdout)
        ratio = merged.get("straggler_ratio")
        assert ratio is not None, "Straggler ratio should be computed"
        assert abs(ratio - 1.5) < 0.01, f"Expected ~1.5, got {ratio}"

    def test_merge_no_straggler(self, tmp_workdir: Path) -> None:
        """Identical ranks should have straggler_ratio = 1.0."""
        for rank in range(4):
            path = tmp_workdir / f"alloc_artifact_rank{rank}.json.gz"
            _write_rank_artifact(path, rank, 30000, 100.0)

        r = run_alloc("merge", "--json", cwd=tmp_workdir)
        assert r.returncode == 0
        merged = json.loads(r.stdout)
        ratio = merged.get("straggler_ratio")
        assert ratio is not None
        assert abs(ratio - 1.0) < 0.01

    def test_merge_writes_output_file(self, tmp_workdir: Path) -> None:
        """alloc merge should write a merged artifact file."""
        for rank in range(2):
            path = tmp_workdir / f"alloc_artifact_rank{rank}.json.gz"
            _write_rank_artifact(path, rank, 30000, 100.0)

        output_path = tmp_workdir / "merged.json.gz"
        r = run_alloc(
            "merge",
            "--output", str(output_path),
            cwd=tmp_workdir,
        )
        assert r.returncode == 0
        assert output_path.exists(), "Merged artifact file should be created"

        # Verify it's a valid gzipped JSON
        with gzip.open(str(output_path), "rt") as f:
            data = json.load(f)
        assert "probe" in data or "version" in data

    def test_merge_no_artifacts_exits_nonzero(self, tmp_workdir: Path) -> None:
        """alloc merge with no rank artifacts should exit with error."""
        r = run_alloc("merge", cwd=tmp_workdir)
        assert r.returncode != 0

    def test_merge_explicit_paths(self, tmp_workdir: Path) -> None:
        """alloc merge should accept explicit artifact paths."""
        for rank in range(2):
            path = tmp_workdir / f"alloc_artifact_rank{rank}.json.gz"
            _write_rank_artifact(path, rank, 30000 + rank * 1000, 100.0)

        r = run_alloc(
            "merge",
            str(tmp_workdir / "alloc_artifact_rank0.json.gz"),
            str(tmp_workdir / "alloc_artifact_rank1.json.gz"),
            "--json",
            cwd=tmp_workdir,
        )
        assert r.returncode == 0
        merged = json.loads(r.stdout)
        assert merged["gpu_count"] == 2

    def test_merge_preserves_per_rank_data(self, tmp_workdir: Path) -> None:
        """Merged artifact should have per-rank VRAM and step times."""
        for rank in range(4):
            path = tmp_workdir / f"alloc_artifact_rank{rank}.json.gz"
            _write_rank_artifact(path, rank, 30000 + rank * 1000, 100.0 + rank)

        r = run_alloc("merge", "--json", cwd=tmp_workdir)
        assert r.returncode == 0
        merged = json.loads(r.stdout)
        per_rank = merged.get("per_rank_peak_vram_mb")
        assert per_rank is not None
        assert len(per_rank) == 4
        assert per_rank == [30000.0, 31000.0, 32000.0, 33000.0]
