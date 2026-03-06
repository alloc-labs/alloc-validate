"""Repository hygiene checks to prevent drift and credential-risk regressions."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import ROOT


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_no_invalid_make_matrix_flag_usage() -> None:
    """`make matrix --include-multi-gpu` is invalid and must never reappear."""
    for path in [
        "GPU_TESTING.md",
        "scripts/gpu/launch-gcp-4xl4.sh",
        "scripts/gpu/launch-aws-4xt4.sh",
    ]:
        assert "make matrix --include-multi-gpu" not in _read(path), path


def test_multi_gpu_launchers_use_matrix_multi_target() -> None:
    for path in [
        "scripts/gpu/launch-gcp-4xl4.sh",
        "scripts/gpu/launch-aws-4xt4.sh",
    ]:
        assert "make matrix-multi" in _read(path), path


def test_launchers_do_not_embed_alloc_token_in_bootstrap() -> None:
    """Prevent leaking ALLOC_TOKEN via metadata/user-data scripts."""
    banned = "export ALLOC_TOKEN='$ALLOC_TOKEN'"
    for path in [
        "scripts/gpu/launch-gcp-l4.sh",
        "scripts/gpu/launch-gcp-4xl4.sh",
        "scripts/gpu/launch-aws-t4.sh",
        "scripts/gpu/launch-aws-4xt4.sh",
    ]:
        assert banned not in _read(path), path


def test_readme_scan_section_is_not_auth_only() -> None:
    text = _read("README.md")
    assert "Requires authentication" not in text
    assert "Works without login" in text


def test_gitignore_has_private_context_guards() -> None:
    """Keep local founder context/docs out of the public repo by default."""
    text = _read(".gitignore")
    expected = [
        "CLAUDE_WORK_CONTEXT.md",
        ".private/",
        "*.local.md",
    ]
    for pattern in expected:
        assert pattern in text, pattern
