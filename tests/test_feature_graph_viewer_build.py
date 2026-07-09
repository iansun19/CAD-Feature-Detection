"""Tests for feature graph viewer payload build."""
from __future__ import annotations

import unittest

from feature_graph_viewer.build import assign_coincident_depth_tiers, coincident_face_partners


class CoincidentFacePartnerTests(unittest.TestCase):
    def test_maps_stock_to_coincident_feature(self) -> None:
        faces = [
            {
                "face_index": 0,
                "normal": [0.0, -1.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
                "triangles": [],
            },
            {
                "face_index": 1,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
                "triangles": [],
            },
        ]
        face_to_feature = {0: -1, 1: 5}
        # Feature face 1 stays visible; unassigned stock 0 is hidden.
        partners = coincident_face_partners(faces, face_to_feature)
        self.assertEqual(partners, {0: 1})

    def test_maps_stock_labeled_feature_to_coincident_partner(self) -> None:
        faces = [
            {
                "face_index": 0,
                "normal": [0.0, 0.0, -1.0],
                "centroid": [0.0, 0.0, 0.0],
                "stock_cut_label": "STOCK",
                "triangles": [],
            },
            {
                "face_index": 1,
                "normal": [0.0, 0.0, 1.0],
                "centroid": [0.0, 0.0, 0.0],
                "stock_cut_label": "CUT",
                "triangles": [],
            },
        ]
        face_to_feature = {0: 87, 1: 28}
        # Both carry feature ids, but 0 is STOCK per the gate, so 1 (CUT) wins.
        partners = coincident_face_partners(faces, face_to_feature)
        self.assertEqual(partners, {0: 1})

    def test_ignores_separated_faces(self) -> None:
        faces = [
            {
                "face_index": 0,
                "normal": [0.0, -1.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
                "triangles": [],
            },
            {
                "face_index": 1,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 10.0, 0.0],
                "triangles": [],
            },
        ]
        face_to_feature = {0: -1, 1: 5}
        partners = coincident_face_partners(faces, face_to_feature)
        self.assertEqual(partners, {})

    def test_feature_feature_cluster_keeps_one_visible(self) -> None:
        # Three coincident feature faces (double-shell overlap): exactly two are
        # hidden and both point at the same visible representative.
        faces = [
            {
                "face_index": 2,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
                "triangles": [],
            },
            {
                "face_index": 5,
                "normal": [0.0, -1.0, 0.0],
                "centroid": [0.0, 0.01, 0.0],
                "triangles": [],
            },
            {
                "face_index": 9,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.02, 0.0],
                "triangles": [],
            },
        ]
        face_to_feature = {2: 3, 5: 4, 9: 7}
        partners = coincident_face_partners(faces, face_to_feature)
        # Lowest index (2) is the representative; 5 and 9 are hidden.
        self.assertEqual(partners, {5: 2, 9: 2})

    def test_does_not_chain_offset_parallel_shells(self) -> None:
        # Parallel faces stepped along the normal must stay visible; only true
        # coplanar duplicates collapse (fixes union-find over-clustering).
        faces = [
            {
                "face_index": 29,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
                "triangles": [],
            },
            {
                "face_index": 30,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.1, 0.0],
                "triangles": [],
            },
            {
                "face_index": 31,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.2, 0.0],
                "triangles": [],
            },
            {
                "face_index": 32,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.3, 0.0],
                "triangles": [],
            },
            {
                "face_index": 33,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.535, 0.0],
                "triangles": [],
            },
        ]
        face_to_feature = {i: i for i in (29, 30, 31, 32, 33)}
        partners = coincident_face_partners(faces, face_to_feature)
        self.assertEqual(partners, {})


class CoincidentDepthTierTests(unittest.TestCase):
    def test_assigns_tiers_without_hiding(self) -> None:
        faces = [
            {
                "face_index": 2,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
                "triangles": [],
            },
            {
                "face_index": 5,
                "normal": [0.0, -1.0, 0.0],
                "centroid": [0.0, 0.01, 0.0],
                "triangles": [],
            },
            {
                "face_index": 9,
                "normal": [0.0, 1.0, 0.0],
                "centroid": [0.0, 0.02, 0.0],
                "triangles": [],
            },
        ]
        face_to_feature = {2: 3, 5: 4, 9: 7}
        assign_coincident_depth_tiers(faces, face_to_feature)
        by_index = {f["face_index"]: f for f in faces}
        self.assertEqual(by_index[2]["depth_tier"], 0)
        self.assertEqual(by_index[5]["depth_tier"], 1)
        self.assertEqual(by_index[9]["depth_tier"], 2)
        self.assertEqual(by_index[5]["coincident_partner"], 2)
        self.assertNotIn("viewer_pick_only", by_index[5])


if __name__ == "__main__":
    unittest.main()
