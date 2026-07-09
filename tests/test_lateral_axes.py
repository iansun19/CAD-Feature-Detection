"""Tests for the provisional lateral (±X/±Y/±Z) approach path (lateral_axes).

Two tiers:
  * Candidacy (pure numpy, synthetic FaceGeom/nodes) — runs in the repo .venv.
  * Reachability + transform-equivalence (OCC) — needs the mlcad env; a small
    synthetic box+side-hole solid is built in-code, no real part data.

Run pure only:  python -m unittest tests.test_lateral_axes
Run all (OCC):  <mlcad-python> -m unittest tests.test_lateral_axes
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from axis_frames import rotation_to_plus_z, transform_points, transform_vector
from lateral_axes import (
    annotate_lateral_candidates,
    lateral_candidates_for_node,
)

try:  # OCC only present in the mlcad conda env
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCC.Extend.TopologyUtils import TopologyExplorer

    HAS_OCC = True
except Exception:  # pragma: no cover
    HAS_OCC = False


@dataclass
class _Face:
    index: int
    normal: np.ndarray
    area: float = 1.0


class CandidacyTests(unittest.TestCase):
    def test_side_flat_gets_lateral_candidate(self):
        # A flat whose normal is +X should be a candidate for +X approach only.
        node = {"feature_id": "f1", "class_name": "flat", "face_ids": [0],
                "params": {"normal": [1.0, 0.0, 0.0]}}
        summary = lateral_candidates_for_node(
            node, z=np.array([0.0, 0.0, 1.0]), normals={}, areas={}
        )
        dirs = {c["dir"] for c in summary["candidates"]}
        self.assertEqual(dirs, {"+X"})
        self.assertEqual(summary["nearest_cardinal"], "+X")

    def test_axial_pocket_gets_z_candidate(self):
        node = {"feature_id": "f2", "class_name": "open_pocket", "face_ids": [0],
                "params": {"opening_axis": [0.0, 0.0, 1.0]}}
        summary = lateral_candidates_for_node(
            node, z=np.array([0.0, 0.0, 1.0]), normals={}, areas={}
        )
        self.assertEqual({c["dir"] for c in summary["candidates"]}, {"+Z"})

    def test_oblique_axis_has_no_cardinal_candidate(self):
        node = {"feature_id": "f3", "class_name": "flat", "face_ids": [0],
                "params": {"normal": [1.0, 1.0, 0.0]}}  # 45° between +X and +Y
        summary = lateral_candidates_for_node(
            node, z=np.array([0.0, 0.0, 1.0]), normals={}, areas={}
        )
        self.assertEqual(summary["candidates"], [])

    def test_annotate_counts_lateral_features(self):
        nodes = [
            {"feature_id": "f1", "class_name": "flat", "face_ids": [0],
             "params": {"normal": [1.0, 0.0, 0.0]}},          # lateral (+X)
            {"feature_id": "f2", "class_name": "open_pocket", "face_ids": [1],
             "params": {"opening_axis": [0.0, 0.0, 1.0]}},     # axial only
        ]
        faces = [_Face(0, np.array([1.0, 0.0, 0.0])), _Face(1, np.array([0.0, 0.0, 1.0]))]
        summary = annotate_lateral_candidates(
            nodes, faces=faces, opening_axis=[0.0, 0.0, 1.0]
        )
        self.assertTrue(summary["provisional"])
        self.assertEqual(summary["n_features_with_lateral_candidate"], 1)
        for n in nodes:
            self.assertIn("lateral_candidates", n["approach"])


@dataclass
class _Setup:
    setup_id: str
    opening_axis: str
    orientation: str | None = None


@dataclass
class _Op:
    op_id: str
    setup_id: str
    tool_id: str
    operation: str
    feature_refs: tuple = ()


class OrientationSequencingTests(unittest.TestCase):
    """Task 6/7: op approach comes from its setup orientation (not hardcoded Z),
    and consecutive ops in differently-oriented setups cost an approach change."""

    def test_approach_vector_follows_setup_orientation(self):
        from lateral_axes import approach_vector_for_setup

        np.testing.assert_allclose(
            approach_vector_for_setup(_Setup("s", "+Z", orientation="+X")),
            [1.0, 0.0, 0.0],
        )
        # Falls back to opening_axis when no explicit orientation.
        np.testing.assert_allclose(
            approach_vector_for_setup(_Setup("s", "-Y")), [0.0, -1.0, 0.0]
        )

    def test_orientation_change_incurs_approach_cost(self):
        from score_sequence import score_sequence

        ops = [
            _Op("o1", "top", "t1", "drill"),
            _Op("o2", "side", "t1", "drill"),  # same tool, different orientation
        ]
        # setup 'top' evaluated from +Z, 'side' from +X — a real orientation flip.
        approach = {"top": "+Z", "side": "+X"}
        scored = score_sequence(ops, setup_approach=approach)
        self.assertEqual(scored.approach_changes, 1)
        self.assertEqual(scored.setup_changes, 1)
        # No cost when orientations match.
        same = score_sequence(ops, setup_approach={"top": "+Z", "side": "+Z"})
        self.assertEqual(same.approach_changes, 0)


def _box_with_side_hole():
    """A 100x100x40 block with a Ø12 hole bored through it along X (a side hole).

    The hole opens on the ±X faces (lateral), not the top ±Z face.
    """
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 100.0, 40.0).Shape()
    # Cylinder axis along +X, centred at (y=50, z=20), long enough to pierce both.
    ax = gp_Ax2(gp_Pnt(-5.0, 50.0, 20.0), gp_Dir(1.0, 0.0, 0.0))
    drill = BRepPrimAPI_MakeCylinder(ax, 6.0, 110.0).Shape()
    return BRepAlgoAPI_Cut(box, drill).Shape()


@unittest.skipUnless(HAS_OCC, "requires OCC (mlcad env)")
class ReachabilityOCCTests(unittest.TestCase):
    def setUp(self):
        self.shape = _box_with_side_hole()
        self.occ_faces = list(TopologyExplorer(self.shape).faces())
        # Identify the cylindrical hole faces (through hole along X).
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Cylinder

        self.hole_face_ids = []
        for i, f in enumerate(self.occ_faces):
            s = BRepAdaptor_Surface(f, True)
            if s.GetType() == GeomAbs_Cylinder:
                self.hole_face_ids.append(i)
        self.assertTrue(self.hole_face_ids, "expected a cylindrical hole face")

    def test_side_hole_reachable_laterally_not_axially(self):
        from lateral_axes import annotate_lateral_reachability

        node = {"feature_id": "hole", "class_name": "through_hole",
                "face_ids": self.hole_face_ids, "params": {}}
        annotate_lateral_reachability(
            [node], occ_faces=self.occ_faces, shape=self.shape
        )
        lat = node["approach"]["lateral"]
        self.assertTrue(lat["provisional"])
        dirs = set(lat["reachable_dirs"])
        # A through hole along X is open on both ±X ends...
        self.assertIn("+X", dirs)
        self.assertIn("-X", dirs)
        # ...and blocked from the top/side faces of the solid block.
        self.assertNotIn("+Z", dirs)
        self.assertNotIn("+Y", dirs)

    def test_reframe_matches_direct_and_shares_transform(self):
        """Exactness of transform-and-reuse + the stock/fixture invariant.

        Reframing the feature points AND the part solid by the SAME cardinal->+Z
        rotation and testing a +Z corridor must give the identical verdict to
        testing the cardinal corridor directly. This is the concrete guard
        against the "rotated feature vs. stale-frame solid" bug: only when both
        are reframed together does the result match.
        """
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.gp import gp_Trsf, gp_Mat

        from reachability import _Corridor, _direction_result, _feature_boundary_points

        occ_by_index = {i: f for i, f in enumerate(self.occ_faces)}
        pts = _feature_boundary_points(occ_by_index, self.hole_face_ids)
        all_pts = _feature_boundary_points(occ_by_index, list(occ_by_index))
        ray_len = 2.0 * float(np.linalg.norm(all_pts.max(0) - all_pts.min(0))) + 10.0
        node = {"class_name": "through_hole"}

        corridor = _Corridor(self.shape)
        for label in ("+X", "-X", "+Y", "+Z"):
            direct = _direction_result(
                corridor, pts, np.asarray(__import__("axis_frames").CARDINAL_VECTORS[label]),
                ray_len, node,
            )["occluded"]

            # Reframe BOTH feature points and the solid by the same R.
            R = rotation_to_plus_z(label)
            m = gp_Mat(*[float(x) for x in R.reshape(-1)])
            trsf = gp_Trsf()
            trsf.SetValues(*[float(x) for x in np.hstack([R, np.zeros((3, 1))]).reshape(-1)])
            shape_r = BRepBuilderAPI_Transform(self.shape, trsf, True).Shape()
            corridor_r = _Corridor(shape_r)
            pts_r = transform_points(R, pts)
            reframed = _direction_result(
                corridor_r, pts_r, np.array([0.0, 0.0, 1.0]), ray_len, node
            )["occluded"]

            self.assertEqual(
                direct, reframed,
                f"{label}: reframe-to-+Z disagreed with direct cardinal test",
            )


if __name__ == "__main__":
    unittest.main()
