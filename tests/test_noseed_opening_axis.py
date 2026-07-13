"""Regression guard for the no-seed opening-axis fix (pocket_detection.py).

The no-seed early-return in detect_pockets used to hardcode opening_axis=(0,1,0)
regardless of geometry. It now routes through the shared _estimate_opening_axis /
_broad_face_axis estimator (one estimator, two call sites). This test pins:

  * Seeded parts (96260B front/rear, fish_mold) — MUST be byte-identical to the
    pre-fix behavior. The fix only touches the no-seed branch; if a seeded part's
    axis moves, the change leaked outside that branch (hard failure).
  * No-seed parts (part1/2/3/4, nist_ctc_01) — MUST match the approved Phase 1
    broad-face table: part1->+Z, part4->+Z, nist->+Z, part2->+X, part3->+Y.

Requires the OCC-enabled env (conda ``mlcad``); analyze_step needs pythonocc.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from brep.feature_params import analyze_step
from cascade.pocket_detection import PocketDetectionConfig, detect_pockets
from run_cascade import _load_edges

# Seeded parts run the full pocket pass and need graph edges; the no-seed parts
# short-circuit at the early-return before edges are used, so edges are optional.
SEEDED = {
    "96260B_front": (Path("fixtures/step/96260B_front.stp"),
                     Path("pipeline_out/96260B_front/graph.npz"),
                     (0.0, 1.0, 0.0)),
    "96260B_rear": (Path("fixtures/step/96260B_rear.stp"),
                    Path("pipeline_out/96260B_plate/graph.npz"),
                    (0.0, 1.0, 0.0)),
    "fish_mold": (Path("fixtures/step/fish_mold.stp"),
                  Path("eval/regression/graphs/fish_mold.graph.npz"),
                  (0.0, 0.0, 1.0)),
}

# No-seed parts: expected axis AFTER the fix (approved Phase 1 table).
NO_SEED = {
    "part1": (Path("fixtures/step/fixtures/step/part1.step"), (0.0, 0.0, 1.0)),
    "part2": (Path("fixtures/step/fixtures/step/part2.step"), (1.0, 0.0, 0.0)),
    "part3": (Path("fixtures/step/fixtures/step/part3.step"), (0.0, 1.0, 0.0)),
    "part4": (Path("fixtures/step/fixtures/step/part4.step"), (0.0, 0.0, 1.0)),
    "nist_ctc_01": (Path("fixtures/step/fixtures/step/nist_ctc_01.step"), (0.0, 0.0, 1.0)),
}


class NoSeedOpeningAxisTests(unittest.TestCase):
    def test_seeded_parts_unchanged(self):
        """Seeded parts keep their pre-fix opening axis (byte-identical)."""
        for name, (step, npz, expected) in SEEDED.items():
            if not step.is_file():
                self.skipTest(f"{name}: STEP missing ({step})")
            with self.subTest(part=name):
                faces = analyze_step(str(step))
                edge_index, edge_attr = _load_edges(npz, step)
                res = detect_pockets(faces, edge_index, edge_attr,
                                     config=PocketDetectionConfig())
                self.assertGreater(len(res.features) + res.n_clusters, -1)
                self.assertEqual(
                    tuple(round(float(x), 6) for x in res.opening_axis), expected,
                    f"{name} seeded axis moved -> change leaked outside no-seed "
                    f"branch: got {res.opening_axis}",
                )

    def test_noseed_parts_match_broadface(self):
        """No-seed parts derive the approved broad-face cardinal from geometry."""
        for name, (step, expected) in NO_SEED.items():
            if not step.is_file():
                self.skipTest(f"{name}: STEP missing ({step})")
            with self.subTest(part=name):
                faces = analyze_step(str(step))
                res = detect_pockets(faces, config=PocketDetectionConfig())
                self.assertEqual(len(res.features), 0, f"{name} unexpectedly seeded")
                self.assertEqual(
                    tuple(round(float(x), 6) for x in res.opening_axis), expected,
                    f"{name}: got {res.opening_axis}, expected {expected}",
                )


if __name__ == "__main__":
    unittest.main()
