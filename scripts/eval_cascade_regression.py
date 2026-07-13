#!/usr/bin/env python3
"""Compare cascade labeling against blessed golden partitions (regression harness).

Exit code 1 when any fixture gate fails. Bless is manual-only via --bless.

Run:
  python scripts/eval_cascade_regression.py
  python scripts/eval_cascade_regression.py --fixture 96260B_rear
  python scripts/eval_cascade_regression.py --bless 96260B_rear --reason "intentional relabel"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cascade.cascade_regression import (  # noqa: E402
    bless_regression_fixture,
    format_partition_diff_report,
    list_regression_fixture_ids,
    load_regression_fixture_by_id,
    load_partition,
    partition_diff_to_dict,
    run_config_drift_lines,
    run_fixture_regression,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Cascade labeling golden regression compare (or manual bless)",
    )
    ap.add_argument(
        "--fixture",
        action="append",
        dest="fixtures",
        metavar="FIXTURE_ID",
        help="run one fixture (repeatable; default: all fixtures in eval/regression/fixtures/)",
    )
    ap.add_argument(
        "--bless",
        metavar="FIXTURE_ID",
        help="bless a single fixture (writes golden; requires --reason)",
    )
    ap.add_argument(
        "--reason",
        help="required with --bless (min 10 characters)",
    )
    ap.add_argument(
        "--mute-stock",
        action="store_true",
        help="hide stock attribution section (display only; never changes exit code)",
    )
    ap.add_argument(
        "--mute-features",
        action="store_true",
        help="hide feature-attributed section (display only; never changes exit code)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit partition diff as JSON after text report",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.bless:
        if args.fixtures:
            print("error: --bless cannot be combined with --fixture", file=sys.stderr)
            return 2
        if not args.reason:
            print("error: --bless requires --reason", file=sys.stderr)
            return 2
        try:
            fixture = load_regression_fixture_by_id(args.bless, repo_root=REPO_ROOT)
            bless_regression_fixture(fixture, reason=args.reason, repo_root=REPO_ROOT)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"blessed {fixture.golden_path}")
        return 0

    fixture_ids = args.fixtures or list_regression_fixture_ids(repo_root=REPO_ROOT)
    if not fixture_ids:
        print("error: no fixtures found under eval/regression/fixtures/", file=sys.stderr)
        return 2

    exit_code = 0
    json_payload: list[dict] = []

    for fixture_id in fixture_ids:
        try:
            fixture = load_regression_fixture_by_id(fixture_id, repo_root=REPO_ROOT)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if not fixture.step_exists:
            print(f"error: missing STEP for {fixture_id}: {fixture.step_path}", file=sys.stderr)
            exit_code = 2
            continue

        if not fixture.golden_path.is_file():
            print(
                f"error: missing golden for {fixture_id}: {fixture.golden_path}",
                file=sys.stderr,
            )
            exit_code = 2
            continue

        try:
            golden = load_partition(fixture.golden_path)
            for line in run_config_drift_lines(golden, fixture):
                print(line)
            diff = run_fixture_regression(fixture)
        except Exception as exc:
            print(f"error [{fixture_id}]: {exc}", file=sys.stderr)
            exit_code = 2
            continue

        print(format_partition_diff_report(
            diff,
            mute_stock=args.mute_stock,
            mute_features=args.mute_features,
        ))
        print("")
        if args.json:
            json_payload.append(partition_diff_to_dict(diff))
        if not diff.gate_passed:
            exit_code = 1

    if args.json and json_payload:
        print(json.dumps(json_payload, indent=2))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
