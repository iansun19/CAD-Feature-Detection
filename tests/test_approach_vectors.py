"""Tests for per-feature 3-axis tool-approach directions (Stage 3, step 2).

Runs the live cascade on 96260B front/rear, so it requires the OCC-enabled env
(conda ``mlcad``); mirrors tests/test_lobe_tier_boundary.py's harness.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from cascade.eval_cascade import build_cascade_feature_graph
from brep.feature_params import analyze_step, load_step_faces
from cascade.pocket_detection import PocketDetectionConfig, resolve_pocket_setup_for_run
from run_cascade import _load_edges, run_cascade

FRONT_STEP = Path("fixtures/step/96260B_front.stp")
FRONT_NPZ = Path("pipeline_out/96260B_front/graph.npz")
REAR_STEP = Path("fixtures/step/96260B_rear.stp")
REAR_NPZ = Path("pipeline_out/96260B_plate/graph.npz")


def _build_graph(step: Path, side: str, npz: Path):
    config = PocketDetectionConfig(
        setup=resolve_pocket_setup_for_run(step, machining_side=side),
    )
    edge_index, edge_attr = _load_edges(npz, step)
    faces = analyze_step(step)
    _, pk, hl, cx, fl, of, wl, pr, rs, *_ = run_cascade(
        step, edge_index, edge_attr, pocket_config=config,
    )
    graph = build_cascade_feature_graph(
        f"96260B_{side}", len(faces), pk, hl, cx, fl, of, wl, pr, rs, edge_index,
        faces=faces, opening_axis=pk.opening_axis,
    )
    return graph, np.asarray(pk.opening_axis, dtype=float)


class ApproachVectorTests(unittest.TestCase):
    def _check_partition(self, step: Path, side: str, npz: Path) -> None:
        graph, opening_axis = _build_graph(step, side, npz)
        z = opening_axis / np.linalg.norm(opening_axis)

        self.assertEqual(graph["schema_version"], 3)
        frame = graph["approach_frame"]
        self.assertIn("z", frame)
        self.assertIn("part_axis_top", frame)
        np.testing.assert_allclose(frame["z"], z, atol=1e-4)

        n_resolved = 0
        for node in graph["nodes"]:
            self.assertIn("approach", node, f"node {node['feature_id']} missing approach")
            ap = node["approach"]
            self.assertIn(ap["setup_dir"], ("+Z", "-Z", None))
            self.assertEqual(ap["reachable_3axis"], ap["setup_dir"] is not None)

            params = node.get("params") or {}
            is_hole = isinstance(params.get("axis"), dict)
            is_pocket = params.get("opening_axis") is not None

            if ap["reachable_3axis"]:
                n_resolved += 1
                axis = np.asarray(ap["axis"], dtype=float)
                # Resolved features must be (near-)parallel to the opening axis.
                self.assertGreater(abs(float(np.dot(axis, z))), 0.9)
                # Outward axis sign must match the reported setup direction.
                self.assertEqual(
                    "+Z" if float(np.dot(axis, z)) >= 0 else "-Z", ap["setup_dir"],
                )

            # Pockets open along the opening axis -> always +Z.
            if is_pocket:
                self.assertEqual(ap["setup_dir"], "+Z", node["class_name"])
            # Bores in this plate family run along the opening axis.
            if is_hole:
                self.assertTrue(ap["reachable_3axis"], f"hole {node['feature_id']}")

        self.assertGreater(n_resolved, 0)

    def test_front_approach_vectors(self):
        self._check_partition(FRONT_STEP, "front", FRONT_NPZ)

    def test_rear_approach_vectors(self):
        self._check_partition(REAR_STEP, "rear", REAR_NPZ)


if __name__ == "__main__":
    unittest.main()
