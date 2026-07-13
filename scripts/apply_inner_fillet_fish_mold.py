#!/usr/bin/env python3
"""Apply inner_fillet label to fish_mold faces 90 and 159 (direct graph edit)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brep.feature_graph import write_feature_graph
from cascade.inner_fillet_detection import (
    REFERENCE_FACE_ID_FISH_MOLD,
    InnerFilletDetectionConfig,
    apply_inner_fillets,
    derive_thresholds_from_reference_face,
    matching_face_ids,
    print_derived_thresholds,
)

# Universal criteria: the geometric predicate must select exactly this set on
# fish_mold (and, guarded below, nothing on either 96260B panel). Face 90 is a
# geometric twin of reference face 159 in a different shell; both are inner
# fillets once the (off-by-default) shell gate is dropped.
EXPECTED_FISH_MATCHES = sorted([90, REFERENCE_FACE_ID_FISH_MOLD])

FROZEN_GRAPHS = [
    ROOT / "pipeline_out/96260B_front/feature_graph_cascade.json",
    ROOT / "pipeline_out/96260B_rear/feature_graph_cascade.json",
]
TARGET_GRAPH = ROOT / "pipeline_out/fish_mold_cascade/feature_graph_cascade.json"
FISH_STEP = ROOT / "fixtures/step/fish_mold.stp"
FISH_NPZ = ROOT / "pipeline_out/fish_mold/graph.npz"
PANEL_NPZ = {
    "rear": ROOT / "pipeline_out/96260B_plate/graph.npz",
    "front": ROOT / "pipeline_out/96260B_front/graph.npz",
}
PANEL_STEPS = {
    "rear": ROOT / "fixtures/step/96260B_rear.stp",
    "front": ROOT / "fixtures/step/96260B_front.stp",
}


def _graph_fingerprint(path: Path) -> str:
    return path.read_text()


def main() -> int:
    npz = np.load(FISH_NPZ)
    thresholds = derive_thresholds_from_reference_face(
        FISH_STEP,
        REFERENCE_FACE_ID_FISH_MOLD,
        npz["edge_index"],
        npz["edge_attr"],
    )
    print_derived_thresholds(thresholds)
    config = InnerFilletDetectionConfig(thresholds=thresholds)

    frozen_before = {p: _graph_fingerprint(p) for p in FROZEN_GRAPHS}

    for panel in ("front", "rear"):
        g = json.loads(FROZEN_GRAPHS[1 if panel == "rear" else 0].read_text())
        stock = set(int(i) for i in g.get("stock_face_ids", []))
        npz_p = np.load(PANEL_NPZ[panel])
        matches = matching_face_ids(
            PANEL_STEPS[panel],
            npz_p["edge_index"],
            npz_p["edge_attr"],
            config=config,
            stock_face_ids=stock,
        )
        print(f"Validation {panel}: {len(matches)} match(es) {matches}")
        if matches:
            print("STOP - predicate matches on frozen 96260B panel; no labels applied.")
            return 1

    graph = json.loads(TARGET_GRAPH.read_text())

    # Detection-driven (not hardcoded to a face id): run the universal geometric
    # predicate over fish_mold and apply inner_fillet to whatever it matches.
    fish_stock = set(int(i) for i in graph.get("stock_face_ids", []))
    result = apply_inner_fillets(
        graph, FISH_STEP, npz["edge_index"], npz["edge_attr"],
        config=config, stock_face_ids=fish_stock,
    )
    print(f"\nfish_mold matches: {result['matches']}")
    if result["matches"] != EXPECTED_FISH_MATCHES:
        print(
            f"STOP - fish_mold inner_fillet matches {result['matches']} != "
            f"expected {EXPECTED_FISH_MATCHES}; refusing edit."
        )
        return 1
    if not result["applied"]:
        print(f"Already applied (skipped {result['skipped']}); nothing to do.")
        return 0
    print(f"Applied: {result['applied']}")
    write_feature_graph(str(TARGET_GRAPH), graph)

    from feature_graph_viewer.build import DEFAULT_TEMPLATE, build_viewer

    build_viewer(
        part_id="fish_mold",
        graph_path=TARGET_GRAPH,
        step_path=FISH_STEP,
        output_path=TARGET_GRAPH.parent / "viewer.html",
        template_path=DEFAULT_TEMPLATE,
        open_browser=False,
    )

    for p in FROZEN_GRAPHS:
        unchanged = frozen_before[p] == _graph_fingerprint(p)
        print(f"Frozen graph unchanged ({p.parent.name}): {unchanged}")
        if not unchanged:
            return 1

    rear_text = FROZEN_GRAPHS[1].read_text()
    if "inner_fillet" in rear_text:
        print("ERROR: inner_fillet found in 96260B_rear graph")
        return 1
    print("96260B_rear: no inner_fillet present")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
