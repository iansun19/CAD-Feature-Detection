"""Tests for inner_fillet_detection (fish_mold face 159 reference)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

from inner_fillet_detection import (
    REFERENCE_FACE_ID_FISH_MOLD,
    InnerFilletDetectionConfig,
    apply_inner_fillet_manual_override,
    apply_inner_fillets,
    derive_thresholds_from_reference_face,
    matching_face_ids,
)

FISH_STEP = ROOT / "fish mold.stp"
FISH_NPZ = ROOT / "pipeline_out/fish_mold/graph.npz"
FISH_GRAPH = ROOT / "pipeline_out/fish_mold_cascade/feature_graph_cascade.json"
REAR_GRAPH = ROOT / "pipeline_out/96260B_rear/feature_graph_cascade.json"
FRONT_GRAPH = ROOT / "pipeline_out/96260B_front/feature_graph_cascade.json"


class InnerFilletFishMoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        npz = np.load(FISH_NPZ)
        cls.edge_index = npz["edge_index"]
        cls.edge_attr = npz["edge_attr"]
        cls.thresholds = derive_thresholds_from_reference_face(
            FISH_STEP,
            REFERENCE_FACE_ID_FISH_MOLD,
            cls.edge_index,
            cls.edge_attr,
        )
        cls.config = InnerFilletDetectionConfig(thresholds=cls.thresholds)

    def test_reference_face_metrics(self) -> None:
        th = self.thresholds
        self.assertEqual(th.reference_face_id, 159)
        self.assertEqual(th.reference_surface_type, "cylinder")
        self.assertAlmostEqual(th.reference_radius_mm, 3.175, places=2)
        self.assertAlmostEqual(th.max_radius_mm, 3.175 * 1.5, places=2)
        self.assertEqual(th.core_shell_id, 1)
        self.assertEqual(th.tangent_concave_parent_count, 4)
        self.assertEqual(th.reference_convex_edges, 4)
        self.assertEqual(th.reference_concave_edges, 0)

    def test_fish_mold_matches_only_reference_face(self) -> None:
        # Universal geometry-only detection: no STOCK seeding. The predicate
        # selects exactly {159} on fish_mold with an empty stock set.
        matches = matching_face_ids(
            FISH_STEP,
            self.edge_index,
            self.edge_attr,
            config=self.config,
            stock_face_ids=set(),
        )
        self.assertEqual(matches, [159])

    def test_96260B_panels_zero_matches(self) -> None:
        # Discriminative on geometry alone: zero matches on either panel even
        # with an empty stock set (i.e. not relying on STOCK gating).
        for panel, step, npz_path in (
            ("rear", ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy",
             ROOT / "pipeline_out/96260B_plate/graph.npz"),
            ("front", ROOT / "96260B_FRONT_XR004_PCD PLATE.stp copy",
             ROOT / "pipeline_out/96260B_front/graph.npz"),
        ):
            npz = np.load(npz_path)
            matches = matching_face_ids(
                step,
                npz["edge_index"],
                npz["edge_attr"],
                config=self.config,
                stock_face_ids=set(),
            )
            self.assertEqual(matches, [], f"{panel} should have 0 predicate matches")

    def test_96260B_rear_has_no_inner_fillet(self) -> None:
        g = json.loads(REAR_GRAPH.read_text())
        classes = {n.get("class_name") for n in g.get("nodes", [])}
        self.assertNotIn("inner_fillet", classes)

    def test_universal_driver_detects_and_reclaims(self) -> None:
        # apply_inner_fillets is detection-driven (no hardcoded face id): on a
        # graph where the cascade absorbed 159 into a pocket, it detects {159}
        # geometrically and reclaims the face into its own inner_fillet node.
        graph = {
            "nodes": [
                {
                    "feature_id": 9,
                    "class_name": "pocket",
                    "face_ids": [159, 187, 188],
                    "n_faces": 3,
                    "params": {"face_indices": [159, 187, 188], "n_faces": 3},
                },
            ],
            "stock_face_ids": [],
        }
        result = apply_inner_fillets(
            graph, FISH_STEP, self.edge_index, self.edge_attr,
            config=self.config, stock_face_ids=set(),
        )
        self.assertEqual(result["matches"], [159])
        self.assertEqual([a["face_id"] for a in result["applied"]], [159])

        pocket = next(n for n in graph["nodes"] if n["feature_id"] == 9)
        self.assertEqual(pocket["face_ids"], [187, 188])
        fillet = next(n for n in graph["nodes"] if n["class_name"] == "inner_fillet")
        self.assertEqual(fillet["face_ids"], [159])

        # Idempotent: a second pass matches 159 but applies nothing.
        again = apply_inner_fillets(
            graph, FISH_STEP, self.edge_index, self.edge_attr,
            config=self.config, stock_face_ids=set(),
        )
        self.assertEqual(again["applied"], [])
        self.assertEqual(again["skipped"], [159])


class InnerFilletManualOverrideTests(unittest.TestCase):
    """Reclaim path: face 159 is absorbed into a pocket once fish_mold is
    ungated (no STOCK), so the override must detach it from its owner."""

    def _pocket_graph(self) -> dict:
        return {
            "nodes": [
                {
                    "feature_id": 9,
                    "class_id": 3,
                    "class_name": "pocket",
                    "face_ids": [159, 187, 188],
                    "n_faces": 3,
                    "params": {"face_indices": [159, 187, 188], "n_faces": 3},
                },
            ],
            "stock_face_ids": [],
        }

    def test_reclaims_face_from_pocket(self) -> None:
        g = self._pocket_graph()
        report = apply_inner_fillet_manual_override(g, 159)
        self.assertEqual(report["before_feature_id"], 9)

        pocket = next(n for n in g["nodes"] if n["feature_id"] == 9)
        self.assertEqual(pocket["face_ids"], [187, 188])
        self.assertEqual(pocket["n_faces"], 2)
        self.assertEqual(pocket["params"]["face_indices"], [187, 188])
        self.assertEqual(pocket["params"]["n_faces"], 2)

        fillet = next(n for n in g["nodes"] if n["class_name"] == "inner_fillet")
        self.assertEqual(fillet["face_ids"], [159])
        self.assertEqual(report["new_feature_id"], fillet["feature_id"])

    def test_reapply_on_inner_fillet_raises(self) -> None:
        g = self._pocket_graph()
        apply_inner_fillet_manual_override(g, 159)
        with self.assertRaises(ValueError):
            apply_inner_fillet_manual_override(g, 159)

    def test_unclaimed_face_still_supported(self) -> None:
        g = {"nodes": [], "stock_face_ids": []}
        report = apply_inner_fillet_manual_override(g, 159)
        self.assertEqual(report["before_feature_id"], -1)
        self.assertEqual(len(g["nodes"]), 1)
        self.assertEqual(g["nodes"][0]["class_name"], "inner_fillet")


if __name__ == "__main__":
    unittest.main()
