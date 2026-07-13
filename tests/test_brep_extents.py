"""Tests for B-rep machined extent helpers (requires pythonocc)."""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from OCC.Core.STEPControl import STEPControl_Reader  # noqa: F401
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

from brep.feature_params import export_cam_params, HAS_OCC as FP_HAS_OCC  # noqa: E402


@unittest.skipUnless(HAS_OCC and FP_HAS_OCC, "pythonocc-core not installed")
class TestBrepExtents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_29000 = os.path.join(
            ROOT, "MFCAD++_dataset", "step", "test", "29000.step",
        )
        cls.graph_29000 = os.path.join(ROOT, "fixtures/graphs/fixtures/graphs/29000_feature_graph.json")
        if not os.path.isfile(cls.step_29000):
            raise unittest.SkipTest("missing 29000.step")
        if not os.path.isfile(cls.graph_29000):
            raise unittest.SkipTest("missing fixtures/graphs/fixtures/graphs/29000_feature_graph.json")

    def test_single_face_aabb_nonzero(self):
        """Regression: old bbox_* was zero on single-face nodes."""
        from brep.brep_extents import feature_aabb, resolve_occ_faces

        face_ids = [2]
        _, resolved = resolve_occ_faces(self.step_29000, face_ids, expected_n_faces=26)
        occ_map = {fid: f for fid, f in resolved}
        aabb = feature_aabb(occ_map, face_ids)
        spans = [aabb.max_corner[i] - aabb.min_corner[i] for i in range(3)]
        self.assertTrue(any(s > 1e-6 for s in spans), f"AABB spans all zero: {spans}")
        self.assertEqual(aabb.zero_width_axes, [])

    def test_valid_pocket_has_depth_and_floor(self):
        from brep.brep_extents import pocket_machined_extents
        from brep.feature_params import analyze_step

        records = analyze_step(self.step_29000)
        face_ids = [13, 14, 15, 16]
        cam_params = {"analytic_surfaces": []}
        out, errors, blend = pocket_machined_extents(
            self.step_29000, face_ids, records, cam_params, expected_n_faces=26,
        )
        self.assertFalse(blend)
        self.assertIsNotNone(out, errors)
        assert out is not None
        self.assertGreater(out["depth"]["value"], 0.0)
        self.assertIn("floor_face_index", out)
        self.assertIn("anchor", out)
        self.assertEqual(out["anchor"]["type"], "floor_plane_point")

    def test_single_plane_pocket_no_floor(self):
        from brep.brep_extents import pocket_machined_extents
        from brep.feature_params import analyze_step

        records = analyze_step(self.step_29000)
        out, errors, blend = pocket_machined_extents(
            self.step_29000, [25], records, {}, expected_n_faces=26,
        )
        self.assertIsNone(out)
        self.assertTrue(any("floor" in e or "wall" in e for e in errors))

    def test_multi_cone_chamfer_legs_and_angles(self):
        from brep.brep_extents import chamfer_machined_extents
        from brep.feature_params import analyze_step

        records = analyze_step(self.step_29000)
        face_ids = [17, 18, 19, 20, 21, 22, 23, 24]
        out, errors, blend = chamfer_machined_extents(
            self.step_29000, face_ids, records, {}, expected_n_faces=26,
        )
        self.assertFalse(blend)
        self.assertIsNotNone(out, errors)
        assert out is not None
        self.assertEqual(len(out["chamfer_legs"]), 4)
        self.assertEqual(sorted(out["semi_angles_deg"]), [45.0, 67.5])
        for leg in out["chamfer_legs"]:
            self.assertIn("leg_axial", leg)
            self.assertIn("leg_radial", leg)

    def test_index_mismatch_raises(self):
        from brep.brep_extents import FaceIndexError, resolve_occ_faces

        with self.assertRaises(FaceIndexError):
            resolve_occ_faces(self.step_29000, [0], expected_n_faces=99)

    def test_cam_export_index_params_only(self):
        with open(self.graph_29000) as f:
            graph = json.load(f)
        # Use graph without pre-enriched heuristic params
        for node in graph.get("nodes", []):
            node.pop("params", None)
        cam = export_cam_params(graph, self.step_29000, enrich_if_needed=True)

        self.assertEqual(len(cam["nodes"]), 5)
        pocket = next(n for n in cam["nodes"] if n["feature_id"] == 2)
        params = pocket["params"]
        self.assertIn("analytic_surfaces", params)
        self.assertIn("surface_counts", params)
        self.assertNotIn("machined_extents", params)
        self.assertNotIn("step_height", params)
        self.assertNotIn("bbox_depth", params)
        self.assertNotIn("validation", pocket)
        self.assertNotIn("cam_ready", pocket)

        for node in cam["nodes"]:
            for key in (
                "step_height", "floor_face_index", "depth_along_floor_normal",
                "bbox_depth", "n_walls", "semi_angle_deg",
            ):
                self.assertNotIn(key, node.get("params", {}))


@unittest.skipUnless(HAS_OCC and FP_HAS_OCC, "pythonocc-core not installed")
class TestHoleExtentsSynthetic(unittest.TestCase):
    """Hole depth tests on OCC primitives (no labeled hole fixture in repo)."""

    @classmethod
    def setUpClass(cls):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IFSelect import IFSelect_RetDone

        cls.tmp_dir = os.path.join(ROOT, "tests", "_tmp_brep_extents")
        os.makedirs(cls.tmp_dir, exist_ok=True)

        ax = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        through = BRepPrimAPI_MakeCylinder(ax, 5.0, 20.0).Shape()
        cls.through_step = os.path.join(cls.tmp_dir, "through_cyl.step")
        w = STEPControl_Writer()
        w.Transfer(through, STEPControl_AsIs)
        if w.Write(cls.through_step) != IFSelect_RetDone:
            raise unittest.SkipTest("could not write synthetic STEP")

        ax2 = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        blind = BRepPrimAPI_MakeCylinder(ax2, 5.0, 15.0).Shape()
        cls.blind_step = os.path.join(cls.tmp_dir, "blind_cyl.step")
        w2 = STEPControl_Writer()
        w2.Transfer(blind, STEPControl_AsIs)
        if w2.Write(cls.blind_step) != IFSelect_RetDone:
            raise unittest.SkipTest("could not write synthetic STEP")

    def test_through_hole_depth(self):
        from brep.brep_extents import hole_machined_extents
        from brep.feature_params import analyze_step

        records = analyze_step(self.through_step)
        cyl_idx = next(i for i, r in enumerate(records) if r.surface_type == "cylinder")
        cam = {
            "primary_cylinder_face_index": cyl_idx,
            "radius": records[cyl_idx].radius,
        }
        out, errors, blend = hole_machined_extents(
            self.through_step, [cyl_idx], records, cam, class_id=0,
        )
        self.assertFalse(blend, errors)
        self.assertIsNotNone(out, errors)
        assert out is not None
        self.assertAlmostEqual(out["depth"]["value"], 20.0, places=2)
        self.assertEqual(out["through_or_blind"], "through")
        self.assertIn("anchor", out)

    def test_blind_hole_classified(self):
        from brep.brep_extents import hole_machined_extents
        from brep.feature_params import analyze_step

        records = analyze_step(self.blind_step)
        cyl_idx = next(i for i, r in enumerate(records) if r.surface_type == "cylinder")
        cam = {"primary_cylinder_face_index": cyl_idx}
        out, errors, blend = hole_machined_extents(
            self.blind_step, [cyl_idx], records, cam, class_id=5,
        )
        self.assertIsNotNone(out, errors)
        assert out is not None
        self.assertAlmostEqual(out["depth"]["value"], 15.0, places=2)
        self.assertEqual(out["through_or_blind"], "blind")


class TestBsplineChamferNoWidth(unittest.TestCase):
    def test_bspline_flags_blend_without_width(self):
        from brep.feature_params import analyze_step
        from brep.brep_extents import chamfer_machined_extents

        if not HAS_OCC:
            self.skipTest("pythonocc not installed")

        step = os.path.join(ROOT, "MFCAD++_dataset", "step", "test", "49960.step")
        if not os.path.isfile(step):
            self.skipTest("no bspline STEP fixture (49960.step)")

        records = analyze_step(step)
        bs_idx = next(
            i for i, r in enumerate(records) if r.surface_type in ("bspline", "bezier")
        )
        out, errors, blend = chamfer_machined_extents(
            step, [bs_idx], records, {}, expected_n_faces=len(records),
        )
        self.assertTrue(blend)
        self.assertIsNone(out)
        self.assertTrue(any("requires_blend" in e for e in errors))
        self.assertFalse(any("width" in str(out) for out in [out] if out))


if __name__ == "__main__":
    unittest.main()
