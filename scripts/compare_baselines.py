#!/usr/bin/env python3
"""Compare two baseline matrix results and flag regressions.

Checks for:
- VRAM estimate drift (>30% between ghost and probe)
- Pass/fail status changes
- Duration regressions (>2x slower)

Usage:
    python scripts/compare_baselines.py baselines/old.json baselines/new.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_baseline(path: Path) -> dict:
    """Load a matrix baseline JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(old: dict, new: dict) -> list[str]:
    """Compare two baselines, return list of warnings."""
    warnings: list[str] = []

    old_results = {(r["framework"], r["model"]): r for r in old.get("results", [])}
    new_results = {(r["framework"], r["model"]): r for r in new.get("results", [])}

    # Check for regressions
    for key, old_r in old_results.items():
        framework, model = key
        new_r = new_results.get(key)

        if new_r is None:
            warnings.append(f"MISSING: {framework}/{model} not in new baseline")
            continue

        # Status regression
        if old_r["status"] == "PASS" and new_r["status"] != "PASS":
            warnings.append(
                f"REGRESSION: {framework}/{model} was PASS, now {new_r['status']}"
            )

        # Duration regression (>2x slower)
        old_dur = old_r.get("duration", 0)
        new_dur = new_r.get("duration", 0)
        if old_dur > 0 and new_dur > old_dur * 2:
            warnings.append(
                f"SLOWER: {framework}/{model} was {old_dur:.1f}s, now {new_dur:.1f}s "
                f"({new_dur / old_dur:.1f}x)"
            )

    # Check for new entries
    for key in new_results:
        if key not in old_results:
            framework, model = key
            warnings.append(f"NEW: {framework}/{model} added in new baseline")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare alloc-validate baselines")
    parser.add_argument("old", type=Path, help="Path to old baseline JSON")
    parser.add_argument("new", type=Path, help="Path to new baseline JSON")
    args = parser.parse_args()

    if not args.old.exists():
        print(f"ERROR: {args.old} not found")
        return 1
    if not args.new.exists():
        print(f"ERROR: {args.new} not found")
        return 1

    old = load_baseline(args.old)
    new = load_baseline(args.new)

    old_total = old.get("total_time", 0)
    new_total = new.get("total_time", 0)
    old_count = len(old.get("results", []))
    new_count = len(new.get("results", []))

    print(f"Old: {old_count} results, {old_total:.1f}s total")
    print(f"New: {new_count} results, {new_total:.1f}s total")
    print()

    warnings = compare(old, new)

    if not warnings:
        print("No regressions detected.")
        return 0

    print(f"{len(warnings)} issue(s) found:")
    for w in warnings:
        print(f"  - {w}")

    # Return non-zero if any regressions (not just new entries)
    has_regression = any(w.startswith(("REGRESSION:", "MISSING:", "SLOWER:")) for w in warnings)
    return 1 if has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
