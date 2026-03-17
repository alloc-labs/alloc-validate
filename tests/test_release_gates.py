"""Opt-in release-gate checks for real regressions seen on Cruise.

These tests are intentionally skipped by default. They are meant to be run
explicitly on a machine with the right network/GPU environment via:

    make test-release-gates
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT, run_alloc, run_python


pytestmark = pytest.mark.skipif(
    os.environ.get("ALLOC_VALIDATE_RELEASE_GATES") != "1",
    reason="Set ALLOC_VALIDATE_RELEASE_GATES=1 to run release-gate checks",
)


def _extract_json_object(output: str) -> dict:
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for idx, ch in enumerate(output):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(output[idx:])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise AssertionError(
        f"No JSON object found in output. Last decode error: {last_error}. Output: {output[:400]}"
    )


def _write_cli_config(home: Path, payload: dict[str, str]) -> None:
    cfg = home / ".alloc" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(payload))


def _find_artifact(workdir: Path) -> Path:
    candidates = sorted(workdir.glob("alloc_artifact*.json.gz"))
    assert candidates, f"No alloc artifact found in {workdir}"
    return candidates[-1]


def _load_artifact(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _gpu_count() -> int:
    r = run_python("-c", "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
    if r.returncode != 0:
        return 0
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def _torchrun_available() -> bool:
    r = run_python("-m", "torch.distributed.run", "--help", timeout=30)
    return r.returncode == 0


def _run_ddp_artifact(workdir: Path) -> dict:
    r = run_alloc(
        "run",
        "--",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        str(ROOT / "distributed" / "train_ddp.py"),
        "--model",
        "medium",
        "--batch-size",
        "16",
        "--max-steps",
        "40",
        cwd=workdir,
        timeout=420,
    )
    assert r.returncode == 0, f"alloc run failed: {r.stdout}\n{r.stderr}"
    return _load_artifact(_find_artifact(workdir))


def _run_fsdp_artifact(workdir: Path):
    r = run_alloc(
        "run",
        "--",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        str(ROOT / "distributed" / "train_fsdp.py"),
        "--model",
        "medium",
        "--batch-size",
        "8",
        "--max-steps",
        "25",
        cwd=workdir,
        timeout=420,
    )
    assert r.returncode == 0, f"alloc run failed: {r.stdout}\n{r.stderr}"
    return r, _load_artifact(_find_artifact(workdir))


class TestAuthAndScanReleaseGates:
    def test_headless_browser_login_refuses_cleanly(self, tmp_path: Path) -> None:
        env = {
            "HOME": str(tmp_path),
            "SSH_TTY": "/dev/pts/99",
            "DISPLAY": "",
            "WAYLAND_DISPLAY": "",
        }
        r = run_alloc("login", "--browser", env=env)
        output = r.stdout + r.stderr
        assert r.returncode != 0
        assert "browser login won't work here" in output.lower()
        assert "alloc login --method token --token" in output

    def test_scan_stale_token_falls_back_to_public_scan(self, tmp_path: Path) -> None:
        env = {"HOME": str(tmp_path)}
        _write_cli_config(tmp_path, {"token": "bad-token"})

        r = run_alloc(
            "scan",
            "--model",
            "llama-3-8b",
            "--gpu",
            "A100-80GB",
            "--json",
            env=env,
            timeout=120,
        )
        assert r.returncode == 0, f"alloc scan failed: {r.stdout}\n{r.stderr}"
        output = r.stdout + r.stderr
        assert "falling back to public scan" in output.lower()
        data = _extract_json_object(r.stdout)
        assert "vram_breakdown" in data
        assert "strategy_verdict" in data

    def test_whoami_stale_token_reports_expired_not_logged_in(self, tmp_path: Path) -> None:
        env = {"HOME": str(tmp_path)}
        _write_cli_config(tmp_path, {"token": "bad-token"})

        r = run_alloc("whoami", "--json", env=env, timeout=60)
        assert r.returncode == 0, f"alloc whoami failed: {r.stdout}\n{r.stderr}"
        data = json.loads(r.stdout)
        assert data["logged_in"] is False
        assert data["token_status"] == "expired"


class TestDistributedReleaseGates:
    @pytest.mark.skipif(_gpu_count() < 1, reason="Requires at least 1 GPU")
    def test_gpu_run_does_not_leak_alloc_dependency_warnings(self, tmp_path: Path) -> None:
        r = run_alloc(
            "run",
            "--",
            sys.executable,
            str(ROOT / "pytorch" / "train.py"),
            "--model",
            "small-cnn",
            cwd=tmp_path,
            timeout=240,
        )
        assert r.returncode == 0, f"alloc run failed: {r.stdout}\n{r.stderr}"
        combined = f"{r.stdout}\n{r.stderr}"
        assert "The pynvml package is deprecated" not in combined
        assert "site-packages/torch/cuda/__init__.py" not in combined

    def test_distributed_ghost_returns_structured_unsupported(self) -> None:
        r = run_alloc("ghost", str(ROOT / "distributed" / "train_ddp.py"), "--json", timeout=120)
        assert r.returncode != 0
        data = _extract_json_object(r.stdout)
        assert data["error"] == "distributed_entrypoint"
        assert data["supported"] is False
        assert "distributed runtime" in data["detail"].lower()

    @pytest.mark.skipif(_gpu_count() < 2, reason="Requires at least 2 GPUs")
    @pytest.mark.skipif(not _torchrun_available(), reason="python -m torch.distributed.run is unavailable")
    def test_ddp_artifact_contains_distributed_fields(self, tmp_path: Path) -> None:
        data = _run_ddp_artifact(tmp_path)
        probe = data.get("probe", {})
        hardware = data.get("hardware", {})

        assert hardware.get("num_gpus_detected", 0) >= 2
        assert probe.get("gpus_per_node", 0) >= 2
        assert probe.get("strategy") == "ddp"
        assert probe.get("dp_degree") == 2
        assert isinstance(probe.get("process_map"), list) and len(probe["process_map"]) >= 2
        per_gpu = probe.get("per_gpu_peak_vram_mb") or []
        assert len(per_gpu) >= 2
        per_rank = probe.get("per_rank_peak_vram_mb") or []
        assert len(per_rank) >= 2

    @pytest.mark.skipif(_gpu_count() < 2, reason="Requires at least 2 GPUs")
    @pytest.mark.skipif(not _torchrun_available(), reason="python -m torch.distributed.run is unavailable")
    def test_fsdp_artifact_is_not_mislabeled_as_ddp(self, tmp_path: Path) -> None:
        result, data = _run_fsdp_artifact(tmp_path)
        probe = data.get("probe", {})
        strategy = probe.get("strategy")

        if strategy is None:
            reasons = (
                probe.get("degradation_reasons")
                or data.get("degradation_reasons")
                or data.get("missing_signals")
                or []
            )
            assert reasons, (
                "FSDP strategy may be unknown, but the artifact must explain the missing "
                f"topology signal.\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
            return

        assert strategy == "fsdp", (
            "Real FSDP runs must not be mislabeled as DDP.\n"
            f"strategy={strategy!r}, detection_method={probe.get('strategy_detection_method')!r}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
