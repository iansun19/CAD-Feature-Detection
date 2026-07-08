"""Unit tests for setup_descriptor.py (no pythonocc required)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pocket_detection import (  # noqa: E402
    PocketFeature,
    PocketSetupConfig,
    apply_explicit_setup_access_to_pocket,
    classify_pocket_open_closed,
)
from setup_descriptor import (  # noqa: E402
    OpeningAxisLowConfidenceError,
    OpeningAxisSpec,
    SetupDescriptorError,
    SetupScope,
    auto_detect_opening_axis,
    default_setup_descriptor,
    load_setup_descriptor,
    parse_setup_descriptor,
    resolve_opening_axis,
    resolve_setup_entry,
    to_pocket_setup_config,
    _parse_setup_scope,
)


def _minimal_pocket_feature(feature_id: int = 1) -> PocketFeature:
    return PocketFeature(
        feature_id=feature_id,
        kind="pocket",
        subtype="blind_pocket",
        face_indices={1},
        centroid_uv=(0.0, 0.0),
        opening_axis=(0.0, 1.0, 0.0),
        wall_diameters={},
        wall_count=1,
        floor_count=0,
        sphere_count=0,
        step_plane_count=0,
        released_faces=[],
        released_by_type={},
        depth_below_top_mm=None,
        fillet_radius_mm=1.0,
        surface_3d=False,
        blend_ring={},
        blend_face_indices=[],
        template_match=False,
    )


class TestOpeningAxis(unittest.TestCase):
    def test_explicit_requires_vector(self) -> None:
        with self.assertRaises(SetupDescriptorError):
            OpeningAxisSpec(mode="explicit")

    def test_auto_low_confidence_errors(self) -> None:
        spec = OpeningAxisSpec(mode="auto", min_confidence=0.85)
        with self.assertRaises(OpeningAxisLowConfidenceError):
            resolve_opening_axis(spec, [], setup_id="test")

    def test_auto_agreeing_axes_high_confidence(self) -> None:
        dirs = [[0, 1, 0], [0, 0.99, 0.01], [0, 1, 0]]
        axis, conf = auto_detect_opening_axis(dirs)
        self.assertGreater(conf, 0.99)
        self.assertAlmostEqual(axis[1], 1.0, places=3)

    def test_explicit_vector_normalized(self) -> None:
        spec = OpeningAxisSpec(mode="explicit", vector=(0.0, 2.0, 0.0))
        axis, conf = resolve_opening_axis(spec, [])
        self.assertAlmostEqual(axis, (0.0, 1.0, 0.0))
        self.assertEqual(conf, 1.0)


class TestParseDescriptor(unittest.TestCase):
    def test_defaults_only_descriptor(self) -> None:
        desc = parse_setup_descriptor({"part_id": "widget"})
        resolved = resolve_setup_entry(desc)
        self.assertEqual(resolved.setup_id, "default")
        self.assertEqual(resolved.pocket_access, "unknown")
        self.assertEqual(resolved.opening_axis.mode, "auto")

    def test_96260B_example_file(self) -> None:
        path = os.path.join(ROOT, "eval/gt/96260B_setup.yaml")
        desc = load_setup_descriptor(path)
        self.assertEqual(desc.part_id, "96260B")
        front = resolve_setup_entry(
            desc, step_path="96260B_FRONT_XR004_PCD PLATE.stp copy"
        )
        rear = resolve_setup_entry(
            desc, step_path="96260B_REAR_XR004_PCD PLATE.stp copy"
        )
        self.assertEqual(front.setup_id, "front")
        self.assertEqual(front.pocket_access, "open")
        self.assertEqual(rear.pocket_access, "closed")
        self.assertEqual(to_pocket_setup_config(front).machining_side, "front")
        self.assertEqual(to_pocket_setup_config(rear).machining_side, "back")

    def test_per_seed_face_override(self) -> None:
        desc = parse_setup_descriptor({
            "part_id": "x",
            "setups": {
                "a": {
                    "setup_id": "a",
                    "pockets": {"by_seed_face": {12: "open", 99: "closed"}},
                },
            },
        })
        entry = desc.setups["a"]
        self.assertEqual(entry.pockets_by_seed_face[12], "open")


class TestArbitrarySetupAccess(unittest.TestCase):
    def test_op10_top_pocket_access_open_no_warning(self) -> None:
        desc = parse_setup_descriptor({
            "part_id": "widget",
            "setups": {
                "op10_top": {
                    "setup_id": "op10_top",
                    "pocket_access": "open",
                },
            },
        })
        resolved = resolve_setup_entry(desc, setup_id="op10_top")
        self.assertEqual(resolved.setup_id, "op10_top")
        self.assertEqual(resolved.pocket_access, "open")
        self.assertIsNone(resolved.machining_side)

        config = to_pocket_setup_config(resolved)
        self.assertEqual(config.resolved_access(), ("open", True))

        feat = _minimal_pocket_feature()
        with self.assertNoLogs("pocket_detection", level="WARNING"):
            classify_pocket_open_closed(feat, config)
        self.assertEqual(feat.access, "open")

    def test_op20_left_machining_side_raises(self) -> None:
        with self.assertRaises(SetupDescriptorError) as ctx:
            parse_setup_descriptor({
                "part_id": "widget",
                "setups": {
                    "op20_left": {
                        "setup_id": "op20_left",
                        "machining_side": "left",
                        "pocket_access": "open",
                    },
                },
            })
        msg = str(ctx.exception)
        self.assertIn("machining_side", msg)
        self.assertIn("front", msg)
        self.assertIn("back", msg)
        self.assertIn("pocket_access", msg)

    def test_parse_scope_variants(self) -> None:
        self.assertTrue(_parse_setup_scope("full", where="t").is_full)
        facing = _parse_setup_scope(["facing"], where="t")
        self.assertTrue(facing.stock_boundary_only)
        explicit = _parse_setup_scope(
            {"classes": ["wall"], "feature_ids": ["12"]},
            where="t",
        )
        self.assertEqual(explicit.classes, ("wall",))
        self.assertEqual(explicit.feature_ids, ("12",))

    def test_96260B_front_back_unchanged(self) -> None:
        path = os.path.join(ROOT, "eval/gt/96260B_setup.yaml")
        desc = load_setup_descriptor(path)
        front = resolve_setup_entry(
            desc, step_path="96260B_FRONT_XR004_PCD PLATE.stp copy"
        )
        rear = resolve_setup_entry(
            desc, step_path="96260B_REAR_XR004_PCD PLATE.stp copy"
        )
        self.assertEqual(front.pocket_access, "open")
        self.assertEqual(rear.pocket_access, "closed")
        self.assertTrue(front.scope.stock_boundary_only)
        self.assertTrue(rear.scope.is_full)
        self.assertEqual(to_pocket_setup_config(front).resolved_access(), ("open", True))
        self.assertEqual(to_pocket_setup_config(rear).resolved_access(), ("closed", True))


class TestExplicitSetupAccessRelabel(unittest.TestCase):
    def test_no_pocket_access_preserves_geometry_band_label(self) -> None:
        feat = _minimal_pocket_feature()
        feat.access = "open"
        feat.toolpath_class = "filleted_open_pocket"
        setup = PocketSetupConfig(machining_side="back")
        self.assertFalse(apply_explicit_setup_access_to_pocket(feat, setup))
        self.assertEqual(feat.access, "open")
        self.assertEqual(feat.toolpath_class, "filleted_open_pocket")
        self.assertEqual(feat.template_deviation, {})

    def test_explicit_pocket_access_closed_skips_lobe_tier_band(self) -> None:
        feat = _minimal_pocket_feature()
        feat.access = "open"
        feat.toolpath_class = "filleted_open_pocket"
        feat.template_deviation = {"lobe_tier": True}
        setup = PocketSetupConfig(machining_side="back", pocket_access="closed")
        self.assertFalse(apply_explicit_setup_access_to_pocket(feat, setup))
        self.assertEqual(feat.access, "open")
        self.assertEqual(feat.toolpath_class, "filleted_open_pocket")

    def test_explicit_pocket_access_closed_relabels_spatial_pocket(self) -> None:
        feat = _minimal_pocket_feature()
        feat.access = "open"
        feat.toolpath_class = "filleted_open_pocket"
        setup = PocketSetupConfig(machining_side="back", pocket_access="closed")
        self.assertTrue(apply_explicit_setup_access_to_pocket(feat, setup))
        self.assertEqual(feat.access, "closed")
        self.assertEqual(feat.toolpath_class, "filleted_pocket")

    def test_explicit_pocket_access_open_relabels(self) -> None:
        feat = _minimal_pocket_feature()
        feat.access = "closed"
        feat.toolpath_class = "filleted_pocket"
        setup = PocketSetupConfig(pocket_access="open")
        self.assertTrue(apply_explicit_setup_access_to_pocket(feat, setup))
        self.assertEqual(feat.access, "open")
        self.assertEqual(feat.toolpath_class, "filleted_open_pocket")


if __name__ == "__main__":
    unittest.main()
