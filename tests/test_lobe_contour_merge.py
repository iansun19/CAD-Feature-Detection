"""Tests for lobe cap contour fragment merge pass."""
from __future__ import annotations

import unittest
from pathlib import Path

from cascade.eval_cascade import build_cascade_feature_graph
from brep.feature_params import analyze_step, load_step_faces
from cascade.approach_vectors import annotate_approach_vectors
from cascade.reachability import annotate_reachability
from cascade.lobe_contour_merge import merge_lobe_contour_fragments
from run_cascade import _load_edges, run_cascade
from cascade.pocket_detection import PocketDetectionConfig, resolve_pocket_setup_for_run
from brep.step_ingest import load_step_shape

ROOT = Path(__file__).resolve().parent.parent
REAR_STEP = ROOT / "fixtures/step/96260B_rear.stp"
# Committed, git-tracked reference graph pinned by the rear regression fixture
# (eval/regression/fixtures/96260B_rear.yaml). The former target
# pipeline_out/96260B_rear/graph.npz was a never-tracked pipeline artifact that
# the graph regen orphaned; this canonical copy (348 faces) cannot drift.
REAR_NPZ = ROOT / "eval/regression/graphs/96260B_rear.graph.npz"


def _rear_graph(*, merge: bool):
    cfg = PocketDetectionConfig(
        setup=resolve_pocket_setup_for_run(REAR_STEP, machining_side="back"),
    )
    ei, ea = _load_edges(REAR_NPZ, REAR_STEP)
    faces = analyze_step(REAR_STEP)
    _, pk, hl, cx, fl, of, wl, pr, rs, *_ = run_cascade(
        REAR_STEP, ei, ea, pocket_config=cfg,
    )
    graph = build_cascade_feature_graph(
        "96260B_rear", len(faces), pk, hl, cx, fl, of, wl, pr, rs, ei,
    )
    occ = load_step_faces(REAR_STEP)
    shape, _ = load_step_shape(str(REAR_STEP))
    annotate_approach_vectors(graph["nodes"], faces=faces, opening_axis=pk.opening_axis)
    annotate_reachability(
        graph["nodes"], occ_faces=occ, shape=shape, opening_axis=pk.opening_axis,
    )
    merged, report = merge_lobe_contour_fragments(
        graph, faces, ei, ea,
        opening_axis=pk.opening_axis,
        occ_faces=occ,
        enabled=merge,
        prefer_rear=True,
    )
    return merged, report


class LobeContourMergeTests(unittest.TestCase):
    # The merge is obsoleted by the current segmentation, not broken. The cascade
    # now emits the lobe cap as ~7 band contour_surface nodes directly, but each
    # band straddles two adjacent lobes, so the single-lobe candidate filter
    # (lobe_contour_merge.py:100, requires len(lobes)==1) matches nothing. These
    # tests pin that no-op: the pass still runs and still detects all 7 lobes,
    # but finds zero merge candidates on the current graph. If the cascade ever
    # regresses to per-lobe over-segmentation, the candidate count goes positive
    # and this file goes red -- telling us the merge is needed again.

    @unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP required")
    def test_rear_merge_finds_no_candidates_on_current_segmentation(self):
        graph, report = _rear_graph(merge=True)
        self.assertTrue(report.enabled)
        # Lobe detection still works: all 7 filleted lobes are found.
        self.assertEqual(report.n_lobes, 7)
        # ...but no cap fragment maps to a single lobe, so nothing merges.
        self.assertEqual(report.candidates_before, 0)
        self.assertEqual(len(report.clusters), 0)
        self.assertEqual(report.nodes_removed, 0)
        debris_nodes = [
            n for n in graph["nodes"]
            if (n.get("params") or {}).get("lobe_contour_merge_kind") == "debris"
        ]
        self.assertEqual(len(debris_nodes), 0)

    @unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP required")
    def test_merge_is_noop_enabled_or_disabled(self):
        # With the current segmentation the enabled pass changes nothing, so the
        # node set is identical whether the merge runs or not.
        graph_on, report_on = _rear_graph(merge=True)
        graph_off, report_off = _rear_graph(merge=False)
        self.assertEqual(report_off.nodes_removed, 0)
        self.assertEqual(report_on.nodes_removed, 0)
        self.assertEqual(len(graph_off["nodes"]), report_off.nodes_before)
        self.assertEqual(len(graph_on["nodes"]), len(graph_off["nodes"]))


if __name__ == "__main__":
    unittest.main()
