"""Tests for lobe cap contour fragment merge pass."""
from __future__ import annotations

import unittest
from pathlib import Path

from eval_cascade import build_cascade_feature_graph
from feature_params import analyze_step, load_step_faces
from approach_vectors import annotate_approach_vectors
from reachability import annotate_reachability
from lobe_contour_merge import merge_lobe_contour_fragments
from run_cascade import _load_edges, run_cascade
from pocket_detection import PocketDetectionConfig, resolve_pocket_setup_for_run
from step_ingest import load_step_shape

ROOT = Path(__file__).resolve().parent.parent
REAR_STEP = ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy"
REAR_NPZ = ROOT / "pipeline_out/96260B_rear/graph.npz"


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
    @unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP required")
    def test_rear_merge_reduces_cap_fragments_to_seven_clusters(self):
        graph, report = _rear_graph(merge=True)
        self.assertTrue(report.enabled)
        self.assertEqual(report.n_lobes, 7)
        anchor_clusters = [c for c in report.clusters if c.get("kind") == "debris"]
        self.assertEqual(len(anchor_clusters), 7)
        self.assertGreater(report.nodes_removed, 0)
        debris_nodes = [
            n for n in graph["nodes"]
            if (n.get("params") or {}).get("lobe_contour_merge_kind") == "debris"
        ]
        self.assertEqual(len(debris_nodes), 7)

    @unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP required")
    def test_merge_disabled_is_noop(self):
        graph_on, report_on = _rear_graph(merge=True)
        graph_off, report_off = _rear_graph(merge=False)
        self.assertEqual(report_off.nodes_removed, 0)
        self.assertEqual(len(graph_off["nodes"]), report_off.nodes_before)
        self.assertLess(len(graph_on["nodes"]), len(graph_off["nodes"]))

    @unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP required")
    def test_merged_nodes_carry_provenance(self):
        graph, report = _rear_graph(merge=True)
        debris_nodes = [
            n for n in graph["nodes"]
            if (n.get("params") or {}).get("lobe_contour_merge_kind") == "debris"
        ]
        self.assertEqual(len(debris_nodes), 7)
        for node in debris_nodes:
            params = node["params"]
            self.assertIn("merged_from_fragment_ids", params)
            self.assertGreaterEqual(len(params["merged_from_fragment_ids"]), 2)
            self.assertEqual(node["class_name"], "contour_surface")


if __name__ == "__main__":
    unittest.main()
