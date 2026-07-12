#!/usr/bin/env python3
"""Attach per-wall B-rep sizing params to a cascade feature graph (post-pass copy).

Usage:
    python scripts/enrich_wall_geometry.py \\
        pipeline_out/96260B_rear/feature_graph_cascade.json \\
        "96260B_REAR_XR004_PCD PLATE.stp copy" \\
        --out pipeline_out/96260B_rear/feature_graph_cascade_enriched.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_bootstrap  # noqa: F401,E402

from wall_geometry import (  # noqa: E402
    enrich_graph_wall_geometry,
    print_verification_report,
    verify_wall_geometry,
    write_enriched_graph,
)


def _load_graph(path: Path) -> dict:
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _default_out_path(graph_path: Path) -> Path:
    stem = graph_path.stem
    if stem.endswith("_cascade"):
        return graph_path.with_name(f"{stem}_enriched.json")
    return graph_path.with_name(f"{stem}_wall_enriched.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich cascade wall nodes with depth/clearance/lateral B-rep params",
    )
    parser.add_argument("graph", type=Path, help="feature_graph_cascade.json input")
    parser.add_argument("step", type=Path, help="matching STEP file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: <graph_stem>_enriched.json beside input)",
    )
    parser.add_argument(
        "--graph-npz",
        type=Path,
        default=None,
        help="edge graph NPZ for concave adjacency (auto-resolved when omitted)",
    )
    parser.add_argument(
        "--opening-axis",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="override opening axis unit vector (default: auto from pocket wall axes)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="print verification table and sanity gates (default: on)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="skip verification report",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    graph_path = args.graph if args.graph.is_absolute() else REPO_ROOT / args.graph
    step_path = args.step if args.step.is_absolute() else REPO_ROOT / args.step
    if not graph_path.is_file():
        parser.error(f"graph not found: {graph_path}")
    if not step_path.is_file():
        parser.error(f"STEP not found: {step_path}")

    graph = _load_graph(graph_path)
    enriched = enrich_graph_wall_geometry(
        graph,
        step_path,
        graph_npz=args.graph_npz,
        opening_axis=args.opening_axis,
    )

    out_path = args.out or _default_out_path(graph_path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    write_enriched_graph(enriched, out_path)

    n_walls = sum(1 for n in enriched["nodes"] if n.get("class_name") == "wall")
    print(f"wrote {out_path}  ({n_walls} wall nodes enriched)")

    if args.verify:
        report = verify_wall_geometry(enriched)
        print_verification_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
