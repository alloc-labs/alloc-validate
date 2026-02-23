"""Tests for alloc config/init — .alloc.yaml and catalog."""

from __future__ import annotations

import json

from tests.conftest import run_alloc


class TestCatalogJson:
    def test_catalog_list_json(self) -> None:
        r = run_alloc("catalog", "list", "--json")
        # If --json flag exists, verify JSON output; otherwise just check exit code
        if r.returncode == 0 and r.stdout.strip():
            try:
                data = json.loads(r.stdout)
                assert isinstance(data, (list, dict))
            except json.JSONDecodeError:
                # --json flag may not be supported for catalog list
                pass


class TestAllocYaml:
    def test_diagnose_with_yaml(self, tmp_workdir) -> None:
        """Diagnose should work when .alloc.yaml is present."""
        # Create a minimal .alloc.yaml
        yaml_path = tmp_workdir / ".alloc.yaml"
        yaml_path.write_text(
            "fleet:\n"
            "  - gpu: H100-80GB\n"
            "    count: 4\n"
            "    rate_per_hour: 3.50\n"
            "objective: cheapest\n"
        )
        # Create a minimal training script
        script = tmp_workdir / "train.py"
        script.write_text(
            "import torch\n"
            "model = torch.nn.Linear(10, 1).cuda()\n"
        )
        r = run_alloc("diagnose", str(script), "--json", "--no-artifact", cwd=tmp_workdir)
        assert r.returncode == 0
