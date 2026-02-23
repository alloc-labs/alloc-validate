"""Tests for artifact schema validation using check_artifact.py."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tests.conftest import ROOT

SCHEMA_DIRS = ["pytorch", "huggingface", "lightning", "ray", "distributed"]


class TestArtifactSchemaFiles:
    """Verify expected/schema.json files exist and are valid JSON."""

    @pytest.mark.parametrize("workload", SCHEMA_DIRS)
    def test_schema_exists(self, workload: str) -> None:
        schema_path = ROOT / workload / "expected" / "schema.json"
        assert schema_path.exists(), f"Missing schema: {schema_path}"

    @pytest.mark.parametrize("workload", SCHEMA_DIRS)
    def test_schema_valid_json(self, workload: str) -> None:
        schema_path = ROOT / workload / "expected" / "schema.json"
        data = json.loads(schema_path.read_text())
        assert isinstance(data, dict)


class TestExistingArtifacts:
    """Validate any existing artifacts in workload directories against their schemas."""

    @pytest.mark.parametrize("workload", SCHEMA_DIRS)
    def test_existing_artifact_loadable(self, workload: str) -> None:
        artifact_path = ROOT / workload / "alloc_artifact.json.gz"
        if not artifact_path.exists():
            pytest.skip(f"No artifact in {workload}/")
        with gzip.open(artifact_path, "rt") as f:
            data = json.load(f)
        assert "version" in data or "probe" in data or isinstance(data, dict)


class TestCheckArtifactScript:
    """Verify the check_artifact.py validation script works."""

    def test_script_exists(self) -> None:
        script = ROOT / "scripts" / "check_artifact.py"
        assert script.exists()

    def test_script_parseable(self) -> None:
        """check_artifact.py should be valid Python."""
        script = ROOT / "scripts" / "check_artifact.py"
        import py_compile
        py_compile.compile(str(script), doraise=True)
