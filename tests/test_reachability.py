"""Golden expectations for verified 3-axis reachability (Stage 3, step 4a).

Runs the live cascade on 96260B front/rear, so it needs the OCC-enabled env
(conda ``mlcad``); mirrors tests/test_lobe_tier_boundary.py's harness. The
expectations here were derived from the geometry and confirmed against the
part's known machining (front + rear plates, every feature actually cut):
nothing non-wall is unreachable, pockets/holes resolve to +Z, the blind hole is
one-sided, the through-hole is two-sided, and walls are exempt (lateral access).
"""
from __future__ import annotations

import unittest
from pathlib import Path

from cascade.eval_cascade import build_cascade_feature_graph
from brep.feature_params import analyze_step, load_step_faces
from cascade.pocket_detection import PocketDetectionConfig, resolve_pocket_setup_for_run
from cascade.reachability import annotate_reachability
from run_cascade import _load_edges, run_cascade
from brep.step_ingest import load_step_shape

FRONT = ("fixtures/step/96260B_front.stp", "front", "pipeline_out/96260B_front/graph.npz")
REAR = ("fixtures/step/96260B_rear.stp", "back", "pipeline_out/96260B_plate/graph.npz")


def _graph(step: str, side: str, npz: str):
    sp = Path(step)
    cfg = PocketDetectionConfig(setup=resolve_pocket_setup_for_run(sp, machining_side=side))
    ei, ea = _load_edges(Path(npz), sp)
    faces = analyze_step(sp)
    _, pk, hl, cx, fl, of, wl, pr, rs, *_ = run_cascade(sp, ei, ea, pocket_config=cfg)
    g = build_cascade_feature_graph(
        f"96260B_{side}", len(faces), pk, hl, cx, fl, of, wl, pr, rs, ei,
        faces=faces, opening_axis=pk.opening_axis,
    )
    occ = load_step_faces(sp)
    shape, _ = load_step_shape(str(sp))
    annotate_reachability(g["nodes"], occ_faces=occ, shape=shape, opening_axis=pk.opening_axis)
    return g


class ReachabilityGoldenTests(unittest.TestCase):
    def _check(self, spec):
        g = _graph(*spec)
        nodes = g["nodes"]

        for n in nodes:
            self.assertIn("reachability", n["approach"], n["feature_id"])

        # Ground truth: every non-wall feature is reachable from some Z setup.
        for n in nodes:
            r = n["approach"]["reachability"]
            if n["class_name"] == "wall":
                self.assertTrue(r.get("exempt"), f"wall {n['feature_id']} not exempt")
                continue
            self.assertTrue(
                r["reachable_dirs"],
                f"{n['class_name']} {n['feature_id']} unreachable: {r}",
            )

        # Pockets are machined from the opening side -> +Z reachable.
        for n in nodes:
            if (n.get("params") or {}).get("opening_axis") is not None:
                self.assertIn("+Z", n["approach"]["reachability"]["reachable_dirs"])

        by_class = {}
        for n in nodes:
            by_class.setdefault(n["class_name"], []).append(n)

        # Blind hole: one-sided (+Z only). Through hole: two-sided.
        for bh in by_class.get("filleted_blind_hole", []):
            self.assertEqual(bh["approach"]["reachability"]["reachable_dirs"], ["+Z"])
        for th in by_class.get("through_hole", []):
            dirs = th["approach"]["reachability"]["reachable_dirs"]
            self.assertIn("+Z", dirs)
            self.assertIn("-Z", dirs)

    def test_front_reachability(self):
        self._check(FRONT)

    def test_rear_reachability(self):
        self._check(REAR)


if __name__ == "__main__":
    unittest.main()
