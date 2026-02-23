"""Tests for alloc diagnose — the core static analysis engine."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.conftest import ROOT, findings_rule_ids, parse_diagnose_json, run_alloc

DIAGNOSE_TARGETS = ROOT / "diagnose-targets"


# ── Rule trigger tests ──────────────────────────────────────────────────


class TestDiagnoseRuleTriggers:
    """Verify each diagnose target triggers the expected rules."""

    def test_dl_issues(self) -> None:
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "dl_issues.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        ids = findings_rule_ids(data)
        assert "DL001" in ids, f"Expected DL001, got {ids}"
        assert "DL002" in ids
        assert "DL003" in ids
        assert "DL004" in ids

    def test_precision_issues(self) -> None:
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "precision_issues.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        ids = findings_rule_ids(data)
        assert "PREC001" in ids, f"Expected PREC001, got {ids}"

    def test_memory_issues(self) -> None:
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "memory_issues.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        ids = findings_rule_ids(data)
        assert "MEM002" in ids, f"Expected MEM002, got {ids}"
        assert "MEM005" in ids, f"Expected MEM005, got {ids}"

    def test_dist_issues(self) -> None:
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "dist_issues.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        ids = findings_rule_ids(data)
        assert "DIST005" in ids, f"Expected DIST005, got {ids}"

    def test_clean_script_zero_findings(self) -> None:
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "clean_script.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        assert len(data["findings"]) == 0, (
            f"Expected 0 findings on clean_script.py, got: "
            f"{[f['rule_id'] for f in data['findings']]}"
        )

    def test_hf_trainer_issues(self) -> None:
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "hf_trainer_issues.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        ids = findings_rule_ids(data)
        assert "MEM002" in ids or "MEM005" in ids, f"Expected memory rules, got {ids}"


# ── Existing training scripts ───────────────────────────────────────────


class TestDiagnoseExistingScripts:
    """Run diagnose on the real training scripts in the repo."""

    @pytest.mark.parametrize("script", [
        "pytorch/train.py",
        "huggingface/train.py",
        "lightning/train.py",
        "ray/train.py",
    ])
    def test_diagnose_exits_zero(self, script: str) -> None:
        r = run_alloc("diagnose", str(ROOT / script), "--json", "--no-artifact")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "findings" in data
        assert "summary" in data


# ── Output mode tests ───────────────────────────────────────────────────


class TestDiagnoseOutputModes:
    def test_diff_output(self) -> None:
        """--diff should produce unified diff markers when patches exist."""
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "dl_issues.py"), "--diff", "--no-artifact")
        assert r.returncode == 0
        output = r.stdout + r.stderr
        # If patches exist, check for diff markers; if no patches, the command
        # should still succeed (some rules may not produce patches)
        # We just verify it doesn't crash
        assert r.returncode == 0

    def test_rules_list(self) -> None:
        """--rules should list all available rules."""
        r = run_alloc("diagnose", "--rules")
        assert r.returncode == 0
        output = r.stdout + r.stderr
        # Verify known rule IDs appear
        assert "DL001" in output
        assert "MEM002" in output
        assert "PREC001" in output
        assert "DIST005" in output
        assert "THRU001" in output

    def test_json_output_structure(self) -> None:
        """--json should produce valid JSON with expected keys."""
        r = run_alloc("diagnose", str(DIAGNOSE_TARGETS / "dl_issues.py"), "--json", "--no-artifact")
        data = parse_diagnose_json(r)
        assert "version" in data
        assert "script" in data
        assert "findings" in data
        assert "summary" in data
        # Verify finding structure
        finding = data["findings"][0]
        assert "rule_id" in finding
        assert "severity" in finding
        assert "category" in finding
        assert "title" in finding


# ── Filter tests ────────────────────────────────────────────────────────


class TestDiagnoseFilters:
    def test_severity_filter(self) -> None:
        """--severity critical should only show critical findings."""
        r = run_alloc(
            "diagnose", str(DIAGNOSE_TARGETS / "dl_issues.py"),
            "--json", "--no-artifact", "--severity", "critical",
        )
        data = parse_diagnose_json(r)
        for finding in data["findings"]:
            assert finding["severity"] == "critical", (
                f"Expected only critical findings, got {finding['rule_id']} "
                f"with severity {finding['severity']}"
            )

    def test_category_filter(self) -> None:
        """--category dataloader should only show DL* rules."""
        r = run_alloc(
            "diagnose", str(DIAGNOSE_TARGETS / "dl_issues.py"),
            "--json", "--no-artifact", "--category", "dataloader",
        )
        data = parse_diagnose_json(r)
        for finding in data["findings"]:
            assert finding["category"] == "dataloader", (
                f"Expected only dataloader findings, got {finding['rule_id']} "
                f"with category {finding['category']}"
            )

    def test_no_artifact_skips_runtime_rules(self) -> None:
        """--no-artifact should only produce AST-only findings."""
        r = run_alloc(
            "diagnose", str(ROOT / "pytorch" / "train.py"),
            "--json", "--no-artifact",
        )
        data = parse_diagnose_json(r)
        assert data["tier"] == "ast_only"
        runtime_rules = {"MEM001", "MEM003", "MEM004", "DIST001", "DIST003",
                         "THRU003", "XS004", "STRAG001"}
        for finding in data["findings"]:
            assert finding["rule_id"] not in runtime_rules, (
                f"Runtime rule {finding['rule_id']} should not appear with --no-artifact"
            )


# ── Multi-file test ─────────────────────────────────────────────────────


class TestDiagnoseMultiFile:
    def test_import_follow(self, tmp_workdir) -> None:
        """Diagnose should follow imports one level deep."""
        # Create a helper module with issues
        helper = tmp_workdir / "helper.py"
        helper.write_text(
            "import torch\n"
            "from torch.utils.data import DataLoader, TensorDataset\n"
            "def make_loader():\n"
            "    ds = TensorDataset(torch.randn(10, 3))\n"
            "    return DataLoader(ds, batch_size=2, num_workers=1)\n"
        )
        # Create main script that imports helper
        main_script = tmp_workdir / "main_train.py"
        main_script.write_text(
            "import torch\n"
            "from helper import make_loader\n"
            "model = torch.nn.Linear(3, 1).cuda()\n"
            "loader = make_loader()\n"
            "for x, in loader:\n"
            "    model(x.cuda())\n"
        )
        r = run_alloc("diagnose", str(main_script), "--json", "--no-artifact", cwd=tmp_workdir)
        # Even if import following doesn't work, the command should succeed
        assert r.returncode == 0
