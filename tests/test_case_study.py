"""Tests for case-study/ — before/after optimization scripts.

Validates that alloc diagnose catches deliberate issues in before.py
and produces zero findings on the optimized after.py.
"""

from __future__ import annotations

from tests.conftest import ROOT, findings_rule_ids, parse_diagnose_json, run_alloc

CASE_STUDY = ROOT / "case-study"


class TestCaseStudyBefore:
    """before.py has deliberate issues — diagnose should catch them all."""

    def _diagnose_before(self) -> dict:
        r = run_alloc(
            "diagnose", str(CASE_STUDY / "before.py"), "--json", "--no-artifact",
        )
        return parse_diagnose_json(r)

    def test_finds_dataloader_issues(self) -> None:
        ids = findings_rule_ids(self._diagnose_before())
        assert "DL001" in ids, "Should flag num_workers=2 as too low"
        assert "DL002" in ids, "Should flag missing pin_memory"
        assert "DL003" in ids, "Should flag missing persistent_workers"
        assert "DL004" in ids, "Should flag missing prefetch_factor"

    def test_finds_no_mixed_precision(self) -> None:
        ids = findings_rule_ids(self._diagnose_before())
        assert "MEM002" in ids, "Should flag fp32 training without mixed precision"

    def test_finds_dataparallel(self) -> None:
        ids = findings_rule_ids(self._diagnose_before())
        assert "DIST005" in ids, "Should flag nn.DataParallel over DDP"

    def test_finds_no_torch_compile(self) -> None:
        ids = findings_rule_ids(self._diagnose_before())
        assert "MEM005" in ids, "Should flag missing torch.compile"

    def test_finds_no_cudnn_benchmark(self) -> None:
        ids = findings_rule_ids(self._diagnose_before())
        assert "THRU001" in ids, "Should flag missing cudnn.benchmark"

    def test_total_finding_count(self) -> None:
        data = self._diagnose_before()
        assert data["summary"]["total"] >= 5, (
            f"Expected at least 5 findings on before.py, got {data['summary']['total']}"
        )


class TestCaseStudyAfter:
    """after.py has all fixes applied — diagnose should find nothing."""

    def test_zero_findings(self) -> None:
        r = run_alloc(
            "diagnose", str(CASE_STUDY / "after.py"), "--json", "--no-artifact",
        )
        data = parse_diagnose_json(r)
        assert len(data["findings"]) == 0, (
            f"Expected 0 findings on after.py, got: "
            f"{[f['rule_id'] for f in data['findings']]}"
        )


class TestCaseStudyComparison:
    """Cross-script comparison tests."""

    def test_before_has_more_findings_than_after(self) -> None:
        before = run_alloc(
            "diagnose", str(CASE_STUDY / "before.py"), "--json", "--no-artifact",
        )
        after = run_alloc(
            "diagnose", str(CASE_STUDY / "after.py"), "--json", "--no-artifact",
        )
        before_data = parse_diagnose_json(before)
        after_data = parse_diagnose_json(after)
        assert before_data["summary"]["total"] > after_data["summary"]["total"]

    def test_before_has_critical_findings(self) -> None:
        data = parse_diagnose_json(run_alloc(
            "diagnose", str(CASE_STUDY / "before.py"), "--json", "--no-artifact",
        ))
        assert data["summary"]["critical"] >= 1, "before.py should have at least 1 critical finding"

    def test_diff_output_has_patches(self) -> None:
        """before.py with --diff should produce actionable patches."""
        r = run_alloc(
            "diagnose", str(CASE_STUDY / "before.py"), "--diff", "--no-artifact",
        )
        assert r.returncode == 0
        output = r.stdout + r.stderr
        # If patches are available, check for diff markers
        if "No code patches" not in output:
            assert any(marker in output for marker in ["---", "+++", "@@", "diff"]), (
                "Expected diff markers in --diff output"
            )
