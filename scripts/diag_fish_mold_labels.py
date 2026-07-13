#!/usr/bin/env python3
"""TEMPORARY read-only diagnostic: fish_mold face STOCK/CUT + labeled_by path.

Print-only; does not modify classification or viewer behavior.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP = ROOT / "fixtures/step/fish_mold.stp"
CASCADE_GRAPH = ROOT / "pipeline_out/fish_mold_cascade/feature_graph_cascade.json"


def _cascade_face_ids(graph: dict) -> set[int]:
    ids: set[int] = set()
    for node in graph.get("nodes", []):
        for face_id in node.get("face_ids", []):
            ids.add(int(face_id))
    return ids


def _is_cavity_face(record) -> bool:
    """Interior non-envelope CUT face (existing classifier signals)."""
    if record.label != "CUT":
        return False
    if record.envelope_coincident.coincident:
        return False
    return record.convex_summary.concave_edges > 0


def main() -> int:
    from cascade.stock_cut_classification import classify_report

    if not STEP.is_file():
        print(f"STEP missing: {STEP}", file=sys.stderr)
        return 1

    cascade_ids: set[int] | None = None
    if CASCADE_GRAPH.is_file():
        with open(CASCADE_GRAPH) as fh:
            cascade_ids = _cascade_face_ids(json.load(fh))
    else:
        print(f"# warning: cascade graph missing: {CASCADE_GRAPH}", file=sys.stderr)

    records = {
        r.face_id: r
        for r in classify_report(STEP, classifier="new", cascade_face_ids=cascade_ids)
    }

    print("face_id | final_label | labeled_by | is_cavity_face")
    print("--------+-------------+------------+---------------")

    gate_cavity = 0
    cascade_cavity = 0
    unclaimed_cavity = 0

    for face_id in sorted(records):
        rec = records[face_id]
        cavity = _is_cavity_face(rec)
        print(
            f"{face_id:7d} | {rec.label:11s} | {rec.labeled_by:10s} | {str(cavity):5s}"
        )
        if cavity:
            if rec.labeled_by == "pre_cascade_gate":
                gate_cavity += 1
            elif rec.labeled_by == "convexity_cascade":
                cascade_cavity += 1
            else:
                unclaimed_cavity += 1

    print()
    print("# cavity-face counts by labeled_by")
    print(f"pre_cascade_gate:                 {gate_cavity}")
    print(f"convexity_cascade:                {cascade_cavity}")
    print(f"convexity_cascade_pool_unclaimed: {unclaimed_cavity}")
    print(f"total cavity faces:               {gate_cavity + cascade_cavity + unclaimed_cavity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
