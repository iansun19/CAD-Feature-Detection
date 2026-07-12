"""Per-setup opening-axis resolution and reachability-dir mapping."""
from __future__ import annotations

import unittest
from pathlib import Path

from machining_context import (
    MachiningContext,
    OpeningAxisResolutionError,
    SetupContext,
    Stock,
    build_setup_context,
    resolve_opening_axis_vector_for_planning,
)
from planner import (
    SetupApproachAxisError,
    _primary_setup_approach_dir,
    _reachability_dir_for_setup,
    _resolve_setup_opening_axis_vector,
    cascade_node_to_feature,
    filter_features_for_setup_by_reachability,
)
from setup_descriptor import (
    OpeningAxisSpec,
    PartSetupDescriptor,
    ResolvedSetup,
    SetupDefaults,
    SetupEntry,
    resolve_setup_entry,
)

ROOT = Path(__file__).resolve().parent.parent
REAR_GRAPH = ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"


def _context(
    *,
    setup_id: str = "rear",
    opening_axis: str = "+Y",
    opening_axis_vector: tuple[float, float, float] = (0.0, 1.0, 0.0),
    machining_side: str | None = "back",
) -> MachiningContext:
    return MachiningContext(
        part_id="96260B",
        feature_graph_ref=str(REAR_GRAPH),
        stock=Stock(bbox_min=(0.0, 0.0, 0.0), bbox_max=(1.0, 1.0, 1.0)),
        setups=[
            SetupContext(
                setup_id=setup_id,
                opening_axis=opening_axis,
                opening_axis_vector=opening_axis_vector,
                machining_side=machining_side,
            ),
        ],
        tools=[],
    )


def _reach_node(reachable_dirs: list[str]) -> dict:
    return {
        "feature_id": 1,
        "class_name": "contour_surface",
        "approach": {
            "reachability": {
                "verified": True,
                "reachable_dirs": reachable_dirs,
            },
        },
    }


class TestSetupAxisResolution(unittest.TestCase):
    def test_plus_y_setup_resolves_to_plus_y_not_plus_z(self) -> None:
        ctx = _context(machining_side="back")
        self.assertEqual(_primary_setup_approach_dir(ctx), "+Y")
        self.assertNotEqual(_primary_setup_approach_dir(ctx), "+Z")

    def test_missing_descriptor_axis_raises(self) -> None:
        descriptor = PartSetupDescriptor(
            part_id="test",
            setups={
                "rear": SetupEntry(
                    setup_id="rear",
                    opening_axis=OpeningAxisSpec(mode="auto"),
                    machining_side="back",
                ),
            },
        )
        resolved = resolve_setup_entry(descriptor, setup_id="rear")
        with self.assertRaises(OpeningAxisResolutionError):
            resolve_opening_axis_vector_for_planning(resolved, {})

    def test_zero_descriptor_axis_raises(self) -> None:
        descriptor = PartSetupDescriptor(
            part_id="test",
            setups={
                "rear": SetupEntry(
                    setup_id="rear",
                    opening_axis=OpeningAxisSpec(mode="auto"),
                    machining_side="back",
                ),
            },
        )
        resolved = resolve_setup_entry(descriptor, setup_id="rear")
        with self.assertRaises(OpeningAxisResolutionError):
            resolve_opening_axis_vector_for_planning(
                resolved,
                {"approach_frame": {"z": [0.0, 0.0, 0.0]}},
            )

    def test_back_plus_y_maps_minus_z_reachability(self) -> None:
        ctx = _context(machining_side="back")
        self.assertEqual(_reachability_dir_for_setup(ctx), "-Z")

    def test_front_plus_y_maps_plus_z_reachability(self) -> None:
        ctx = _context(setup_id="front", machining_side="front")
        self.assertEqual(_reachability_dir_for_setup(ctx), "+Z")

    def test_minus_z_feature_in_back_plus_y_scope(self) -> None:
        ctx = _context(machining_side="back")
        node = _reach_node(["-Z"])
        feat = cascade_node_to_feature(node)
        kept, dropped, info = filter_features_for_setup_by_reachability(
            [feat],
            {"1": node},
            {},
            ctx,
        )
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(info["opening_axis"], "+Y")
        self.assertEqual(info["reachability_dir"], "-Z")

    def test_minus_z_feature_out_of_front_plus_y_scope(self) -> None:
        ctx = _context(setup_id="front", machining_side="front")
        node = _reach_node(["-Z"])
        feat = cascade_node_to_feature(node)
        kept, dropped, _ = filter_features_for_setup_by_reachability(
            [feat],
            {"1": node},
            {},
            ctx,
        )
        self.assertEqual(len(kept), 0)
        self.assertEqual(dropped, 1)

    def test_plus_z_feature_out_of_back_plus_y_scope(self) -> None:
        ctx = _context(machining_side="back")
        node = _reach_node(["+Z"])
        feat = cascade_node_to_feature(node)
        kept, dropped, _ = filter_features_for_setup_by_reachability(
            [feat],
            {"1": node},
            {},
            ctx,
        )
        self.assertEqual(len(kept), 0)
        self.assertEqual(dropped, 1)

    def test_non_unit_axis_vector_raises(self) -> None:
        ctx = _context(opening_axis_vector=(0.0, 2.0, 0.0))
        with self.assertRaises(SetupApproachAxisError):
            _resolve_setup_opening_axis_vector(ctx)

    def test_missing_machining_side_raises(self) -> None:
        ctx = _context(machining_side=None)
        with self.assertRaises(SetupApproachAxisError):
            _reachability_dir_for_setup(ctx)

    def test_build_setup_context_from_generated_descriptor(self) -> None:
        if not REAR_GRAPH.is_file():
            self.skipTest(f"missing {REAR_GRAPH}")
        import json

        graph = json.loads(REAR_GRAPH.read_text())
        descriptor = PartSetupDescriptor(
            part_id="96260B",
            setups={
                "rear": SetupEntry(
                    setup_id="rear",
                    opening_axis=OpeningAxisSpec(
                        mode="explicit",
                        vector=(0.0, 1.0, 0.0),
                    ),
                    machining_side="back",
                ),
            },
        )
        resolved = resolve_setup_entry(descriptor, setup_id="rear")
        setup = build_setup_context(resolved, graph)
        self.assertEqual(setup.opening_axis, "+Y")
        self.assertEqual(setup.opening_axis_vector, (0.0, 1.0, 0.0))
        self.assertEqual(setup.machining_side, "back")


if __name__ == "__main__":
    unittest.main()
