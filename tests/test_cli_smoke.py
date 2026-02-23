"""Smoke tests for core alloc CLI commands."""

from __future__ import annotations

from tests.conftest import run_alloc


class TestVersion:
    def test_version_exits_zero(self) -> None:
        r = run_alloc("version")
        assert r.returncode == 0

    def test_version_contains_string(self) -> None:
        r = run_alloc("version")
        # stdout or stderr — alloc may print version to either
        output = r.stdout + r.stderr
        assert "alloc" in output.lower() or "0." in output


class TestHelp:
    def test_help_exits_zero(self) -> None:
        r = run_alloc("--help")
        assert r.returncode == 0

    def test_help_shows_commands(self) -> None:
        r = run_alloc("--help")
        output = r.stdout + r.stderr
        assert "run" in output.lower()


class TestWhoami:
    def test_whoami_json_exits_zero(self) -> None:
        r = run_alloc("whoami", "--json")
        assert r.returncode == 0


class TestCatalog:
    def test_catalog_list_exits_zero(self) -> None:
        r = run_alloc("catalog", "list")
        assert r.returncode == 0

    def test_catalog_list_contains_known_gpus(self) -> None:
        r = run_alloc("catalog", "list")
        output = r.stdout + r.stderr
        # At least one of these should appear
        assert any(gpu in output for gpu in ["H100", "A100", "L4", "T4"])

    def test_catalog_show_gpu(self) -> None:
        r = run_alloc("catalog", "show", "H100-80GB")
        assert r.returncode == 0
        output = r.stdout + r.stderr
        assert "H100" in output
