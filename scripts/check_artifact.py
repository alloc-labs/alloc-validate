#!/usr/bin/env python3
"""Validate an alloc artifact against an expected schema.

The artifact is the gzipped JSON dict produced by `alloc run`.
This validator supports both:
- legacy top-level key schemas (list form), and
- nested contract schemas (object form with keys/paths/non_null/types).

Usage:
    python scripts/check_artifact.py \
        --artifact alloc_artifact.json.gz \
        --schema pytorch/expected/schema.json \
        --tier free
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any


def load_artifact(path: Path) -> dict:
    """Load a JSON artifact, decompressing gzip if needed."""
    raw = path.read_bytes()
    try:
        raw = gzip.decompress(raw)
    except (gzip.BadGzipFile, EOFError, OSError):
        pass  # plain JSON
    return json.loads(raw)


def load_schema(path: Path) -> dict:
    """Load the expected schema definition."""
    return json.loads(path.read_text())


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _path_get(data: dict[str, Any], path: str) -> tuple[bool, Any]:
    cur: Any = data
    for segment in path.split("."):
        if not isinstance(cur, dict) or segment not in cur:
            return False, None
        cur = cur[segment]
    return True, cur


def _normalize_section(section: Any) -> dict[str, Any]:
    """Normalize schema section into keys/paths/non_null/types.

    Supported formats:
      1) Legacy list: ["version", "timestamp", ...]
      2) Rich object:
         {
           "keys": [...],
           "paths": [...],
           "non_null": [...],
           "types": {"probe.samples": "array"}
         }
    """
    if section is None:
        return {"keys": [], "paths": [], "non_null": [], "types": {}}

    if isinstance(section, list):
        return {
            "keys": [str(x) for x in section],
            "paths": [],
            "non_null": [],
            "types": {},
        }

    if isinstance(section, dict):
        keys = [str(x) for x in section.get("keys", [])]
        paths = [str(x) for x in section.get("paths", [])]
        non_null = [str(x) for x in section.get("non_null", [])]
        types = section.get("types", {})
        if not isinstance(types, dict):
            raise ValueError("schema section 'types' must be an object")
        normalized_types = {str(k): str(v) for k, v in types.items()}
        return {
            "keys": keys,
            "paths": paths,
            "non_null": non_null,
            "types": normalized_types,
        }

    raise ValueError("schema section must be a list or object")


def _check_types(value: Any, expected: str) -> bool:
    expected = expected.lower()
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported expected type: {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate alloc artifact")
    parser.add_argument("--artifact", required=True, type=Path, help="Path to artifact file")
    parser.add_argument("--schema", required=True, type=Path, help="Path to expected schema JSON")
    parser.add_argument(
        "--tier",
        choices=["free", "full"],
        default="free",
        help="Auth tier: 'free' checks base keys, 'full' checks all keys",
    )
    args = parser.parse_args()

    if not args.artifact.exists():
        print(f"FAIL: artifact not found: {args.artifact}")
        return 1

    if not args.schema.exists():
        print(f"FAIL: schema not found: {args.schema}")
        return 1

    artifact = load_artifact(args.artifact)
    schema = load_schema(args.schema)

    try:
        free = _normalize_section(schema.get("free", []))
        full = _normalize_section(schema.get("full", [])) if args.tier == "full" else _normalize_section(None)
    except ValueError as exc:
        print(f"FAIL: invalid schema: {exc}")
        return 1

    keys_to_check = _unique(free["keys"] + full["keys"])
    paths_to_check = _unique(free["paths"] + full["paths"])
    non_null_to_check = _unique(free["non_null"] + full["non_null"])
    types_to_check = {**free["types"], **full["types"]}

    missing_keys = [key for key in keys_to_check if key not in artifact]

    missing_paths: list[str] = []
    for path in paths_to_check:
        exists, _ = _path_get(artifact, path)
        if not exists:
            missing_paths.append(path)

    null_failures: list[str] = []
    for target in non_null_to_check:
        if "." in target:
            exists, value = _path_get(artifact, target)
            if exists and value is None:
                null_failures.append(target)
        else:
            if target in artifact and artifact[target] is None:
                null_failures.append(target)

    type_failures: list[str] = []
    for target, expected in types_to_check.items():
        if "." in target:
            exists, value = _path_get(artifact, target)
            if not exists:
                continue
        else:
            if target not in artifact:
                continue
            value = artifact[target]

        try:
            ok = _check_types(value, expected)
        except ValueError as exc:
            print(f"FAIL: invalid schema type for {target}: {exc}")
            return 1

        if not ok:
            actual = type(value).__name__
            type_failures.append(f"{target} expected={expected} actual={actual}")

    if missing_keys or missing_paths or null_failures or type_failures:
        print(f"FAIL: artifact contract validation failed ({args.tier} tier):")
        if missing_keys:
            print("- missing top-level keys:")
            for key in missing_keys:
                print(f"  - {key}")
        if missing_paths:
            print("- missing nested paths:")
            for path in missing_paths:
                print(f"  - {path}")
        if null_failures:
            print("- null disallowed:")
            for path in null_failures:
                print(f"  - {path}")
        if type_failures:
            print("- type mismatches:")
            for msg in type_failures:
                print(f"  - {msg}")
        print(f"\nArtifact keys present: {sorted(artifact.keys())}")
        return 1

    total_checks = len(keys_to_check) + len(paths_to_check) + len(non_null_to_check) + len(types_to_check)
    print(f"OK: artifact passes schema validation ({args.tier} tier, {total_checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
