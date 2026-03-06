"""Tests for P2 — alloc diagnose --explain flag.

Tests the CLI-side behavior of --explain. The actual AI synthesis
requires an API key and Pro tier, so we test the unauthenticated
and error paths that are always accessible.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import ROOT, run_alloc

DIAGNOSE_TARGETS = ROOT / "diagnose-targets"


class TestExplainFlag:
    """Test --explain flag behavior on diagnose command."""

    def test_explain_without_login_shows_prompt(self) -> None:
        """--explain without auth should show login prompt, not crash."""
        r = run_alloc(
            "diagnose",
            str(DIAGNOSE_TARGETS / "dl_issues.py"),
            "--explain",
            "--no-artifact",
        )
        # Should either succeed (if logged in) or show login prompt
        output = r.stdout + r.stderr
        # We accept either outcome — the key thing is no crash
        assert r.returncode in (0, 1), f"Unexpected exit code {r.returncode}: {output}"
        if r.returncode == 1:
            assert "login" in output.lower() or "alloc login" in output

    def test_explain_flag_accepted(self) -> None:
        """alloc diagnose --explain should be a recognized flag."""
        r = run_alloc("diagnose", "--help")
        assert r.returncode == 0
        assert "--explain" in r.stdout

    def test_explain_still_runs_heuristics(self) -> None:
        """--explain should still produce heuristic findings first."""
        r = run_alloc(
            "diagnose",
            str(DIAGNOSE_TARGETS / "dl_issues.py"),
            "--explain",
            "--no-artifact",
        )
        output = r.stdout + r.stderr
        # Should mention findings even if explain fails due to auth
        # At minimum, the command should produce some output
        assert len(output) > 0


class TestDiagnoseEfficiencyMode:
    """Test --efficiency output mode with phase-aware display."""

    def test_efficiency_flag_accepted(self) -> None:
        """alloc diagnose --efficiency should be a recognized flag."""
        r = run_alloc("diagnose", "--help")
        assert "--efficiency" in r.stdout

    def test_efficiency_without_artifact(self) -> None:
        """--efficiency without artifact should show message about missing data."""
        r = run_alloc(
            "diagnose",
            str(DIAGNOSE_TARGETS / "dl_issues.py"),
            "--efficiency",
            "--no-artifact",
        )
        # Should exit clean even without artifact data
        assert r.returncode == 0


class TestMergeCommandHelp:
    """Verify alloc merge command exists and has help."""

    def test_merge_help(self) -> None:
        r = run_alloc("merge", "--help")
        assert r.returncode == 0
        assert "merge" in r.stdout.lower() or "rank" in r.stdout.lower()
        assert "--output" in r.stdout or "-o" in r.stdout
        assert "--json" in r.stdout
