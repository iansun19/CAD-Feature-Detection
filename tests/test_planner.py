"""Unit tests for planner.py (no pythonocc required)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cam_plan_schema import CamPlan, MachiningParameters, PocketAccess, Setup, load_cam_plan  # noqa: E402
from machining_context import (  # noqa: E402
    MachiningContext,
    SetupContext,
    SetupScopeSpec,
    Stock,
    Tool,
    ToolPreset,
    default_tool_library,
    load_machining_context,
    load_tool_library,
)
from planner import (  # noqa: E402
    UNRESOLVED_TOOL_ID,
    OpSpec,
    SetupPlanInput,
    _tool_fits_op,
    _tool_material_rank,
    assign_parameters,
    build_precedence,
    cascade_node_to_feature,
    filter_planner_features,
    filter_features_for_setup,
    filter_features_for_setup_by_reachability,
    _reachability_dir_for_setup,
    group_coaxial_holes,
    group_operations_by_tool_strategy,
    identify_facing_feature_ids,
    map_feature_to_operations,
    plan,
    plan_multi_setups,
    resolve_preset,
    select_tool,
    sequence,
    _assign_op_ids,
)

SAMPLE_LIBRARY = Path(ROOT) / "tests" / "fixtures" / "Aluminum_Sample_Library__Inch_.json"
DRILL_LIBRARY = Path(ROOT) / "tool_libraries" / "Kennametal_Standard_Drills__Inch_.json"

CASCADE_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
CONTEXT_PATH = Path(ROOT) / "examples" / "machining_context_96260B.json"
PLAN_PATH = Path(ROOT) / "examples" / "cam_plan_96260B.json"


def _minimal_context(**overrides) -> MachiningContext:
    base = {
        "part_id": "test_part",
        "feature_graph_ref": "test/feature_graph_cascade.json",
        "stock": Stock(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(10.0, 10.0, 10.0),
        ),
        "setups": [
            SetupContext(
                setup_id="primary",
                opening_axis="+Y",
                opening_axis_vector=(0.0, 1.0, 0.0),
                machining_side="back",
                pocket_access={"1": "closed", "2": "open"},
            ),
        ],
        "tools": default_tool_library(),
    }
    base.update(overrides)
    return MachiningContext(**base)


class TestCascadeAdapter(unittest.TestCase):
    def test_rename_and_geometry_extraction(self) -> None:
        node = {
            "feature_id": 15,
            "class_name": "through_hole",
            "params": {
                "nominal_diameter": 81.28,
                "depth": 25.3,
                "axis": {"point": [0, 0, 0], "direction": [0, 1, 0]},
            },
        }
        feat = cascade_node_to_feature(node)
        self.assertEqual(feat.feature_id, "15")
        self.assertEqual(feat.feature_type, "through_hole")
        self.assertAlmostEqual(feat.diameter_mm, 81.28)
        self.assertAlmostEqual(feat.depth_mm, 25.3)


class TestFeatureFilter(unittest.TestCase):
    def test_wall_features_are_kept(self) -> None:
        features = [
            cascade_node_to_feature({"feature_id": 0, "class_name": "wall", "params": {}}),
            cascade_node_to_feature({"feature_id": 1, "class_name": "filleted_pocket", "params": {}}),
            cascade_node_to_feature({"feature_id": 2, "class_name": "through_hole", "params": {}}),
            cascade_node_to_feature({"feature_id": 99, "class_name": "unknown_thing", "params": {}}),
        ]
        kept, dropped = filter_planner_features(features)
        self.assertEqual(dropped, 1)
        self.assertEqual({f.feature_id for f in kept}, {"0", "1", "2"})


class TestCoaxialStack(unittest.TestCase):
    def test_shared_axis_groups_into_one_drill_op(self) -> None:
        holes = [
            cascade_node_to_feature(
                {
                    "feature_id": 14,
                    "class_name": "filleted_blind_hole",
                    "params": {
                        "nominal_diameter": 101.0,
                        "depth": 2.0,
                        "axis": {"point": [0, 0, 0], "direction": [0, 1, 0]},
                    },
                }
            ),
            cascade_node_to_feature(
                {
                    "feature_id": 15,
                    "class_name": "through_hole",
                    "params": {
                        "nominal_diameter": 81.0,
                        "depth": 25.0,
                        "axis": {"point": [0, 0, 0], "direction": [0, 1, 0]},
                    },
                }
            ),
        ]
        groups = group_coaxial_holes(holes)
        self.assertEqual(len(groups), 1)
        self.assertEqual({f.feature_id for f in groups[0]}, {"14", "15"})

        ctx = _minimal_context()
        ops = map_feature_to_operations(groups[0], ctx)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].operation, "helix_bore")
        self.assertEqual(ops[0].tool_type_needed, "endmill")
        self.assertEqual(sorted(ops[0].feature_refs), ["14", "15"])


class TestAccessStrategyMapping(unittest.TestCase):
    def test_closed_and_open_pocket_strategies(self) -> None:
        ctx = _minimal_context()
        closed = cascade_node_to_feature(
            {"feature_id": 1, "class_name": "filleted_pocket", "params": {"depth_below_top_mm": 5.0}}
        )
        open_pocket = cascade_node_to_feature(
            {"feature_id": 2, "class_name": "open_pocket", "params": {"depth_below_top_mm": 5.0}}
        )
        closed_ops = map_feature_to_operations(closed, ctx)
        open_ops = map_feature_to_operations(open_pocket, ctx)
        self.assertEqual(closed_ops[0].operation, "pocket")
        self.assertEqual(open_ops[0].operation, "dynamic_mill_2d")
        self.assertEqual(open_ops[1].operation, "contour_2d")


class TestWidenedFeatureMapping(unittest.TestCase):
    def test_wall_surface_fillet_and_face_ops(self) -> None:
        ctx = _minimal_context()
        wall = cascade_node_to_feature({"feature_id": 22, "class_name": "wall", "params": {}})
        surface = cascade_node_to_feature({"feature_id": 37, "class_name": "contour_surface", "params": {}})
        fillet = cascade_node_to_feature({"feature_id": 20, "class_name": "outer_fillet", "params": {}})
        face = cascade_node_to_feature({"feature_id": 18, "class_name": "flat", "params": {}})

        wall_ops = map_feature_to_operations(wall, ctx)
        self.assertEqual(wall_ops[0].operation, "contour_2d")

        surface_ops = map_feature_to_operations(surface, ctx)
        self.assertEqual(surface_ops[0].operation, "constant_scallop")
        self.assertEqual(surface_ops[0].tool_type_needed, "ball_endmill")

        fillet_ops = map_feature_to_operations(fillet, ctx)
        self.assertEqual(fillet_ops[0].operation, "pencil")

        face_ops = map_feature_to_operations(face, ctx)
        self.assertEqual(face_ops[0].operation, "raster")

    def test_axisymmetric_surface_maps_to_radial_spiral(self) -> None:
        ctx = _minimal_context()
        # single shared axis + revolved-surface-dominated histogram -> round -> radial_spiral
        dome = cascade_node_to_feature({
            "feature_id": 102, "class_name": "contour_surface",
            "params": {"n_distinct_axes": 1, "surface_type_histogram": {"torus": 2, "sphere": 2, "cone": 2}},
        })
        # freeform bspline (no shared axis) -> constant_scallop
        freeform = cascade_node_to_feature({
            "feature_id": 41, "class_name": "contour_surface",
            "params": {"n_distinct_axes": 0, "surface_type_histogram": {"bspline": 1}},
        })
        # single axis but bspline-dominated -> stays constant_scallop
        blend = cascade_node_to_feature({
            "feature_id": 39, "class_name": "contour_surface",
            "params": {"n_distinct_axes": 1, "surface_type_histogram": {"bspline": 5, "torus": 1}},
        })

        self.assertEqual(map_feature_to_operations(dome, ctx)[0].operation, "radial_spiral")
        self.assertEqual(map_feature_to_operations(freeform, ctx)[0].operation, "constant_scallop")
        self.assertEqual(map_feature_to_operations(blend, ctx)[0].operation, "constant_scallop")

    def test_mixed_slope_surface_maps_to_steep_shallow(self) -> None:
        ctx = _minimal_context()
        # surface spanning both steep and shallow bands -> steep_shallow, even when
        # it is also a body of revolution (mixed slope wins over roundness)
        mixed_round = cascade_node_to_feature({
            "feature_id": 36, "class_name": "contour_surface",
            "params": {"n_distinct_axes": 1, "surface_type_histogram": {"cone": 1, "torus": 1}},
            "slope_profile": {"mixed": True, "steep_fraction": 0.54, "shallow_fraction": 0.46},
        })
        # round but single-band (not mixed) -> radial_spiral
        round_uniform = cascade_node_to_feature({
            "feature_id": 102, "class_name": "contour_surface",
            "params": {"n_distinct_axes": 1, "surface_type_histogram": {"torus": 1}},
            "slope_profile": {"mixed": False, "steep_fraction": 1.0, "shallow_fraction": 0.0},
        })
        self.assertEqual(map_feature_to_operations(mixed_round, ctx)[0].operation, "steep_shallow")
        self.assertEqual(map_feature_to_operations(mixed_round, ctx)[0].tool_type_needed, "ball_endmill")
        self.assertEqual(map_feature_to_operations(round_uniform, ctx)[0].operation, "radial_spiral")

    def test_area_roughing_selected_for_steep_3d_pocket(self) -> None:
        # Synthetic fixture: no repo part has a 3D_surface=True pocket (the only
        # 3D_surface features are contour_surfaces, which are finished not roughed),
        # so exercise the optirough-vs-area_roughing split directly here.
        ctx = _minimal_context()

        def rough_op(steep_fraction: float, three_d: bool = True) -> str:
            node = {
                "feature_id": 1,
                "class_name": "filleted_pocket",
                "params": {"3D_surface": three_d, "depth": 10.0},
                "slope_profile": {
                    "steep_fraction": steep_fraction,
                    "shallow_fraction": 1.0 - steep_fraction,
                    "mixed": False,
                },
            }
            feat = cascade_node_to_feature(node)
            return map_feature_to_operations(feat, ctx)[0].operation

        # steep-dominated 3D content -> Z-level area_roughing; shallow -> adaptive optirough
        self.assertEqual(rough_op(0.80), "area_roughing")
        self.assertEqual(rough_op(0.20), "optirough")
        self.assertEqual(rough_op(0.50), "area_roughing")  # threshold is >=
        # a 2.5D pocket ignores slope entirely (3D gate closed) -> neither 3D rough op
        self.assertNotIn(rough_op(0.90, three_d=False), {"area_roughing", "optirough"})

    def test_rest_roughing_triggers_when_rough_tool_exceeds_fillet(self) -> None:
        import types
        import planner as _p

        setup = "rear"
        fillet = 3.81
        rough = OpSpec(
            op_id="OP010", feature_refs=["1"], feature_type="filleted_pocket",
            setup_id=setup, operation="dynamic_mill_2d", tool_id="BIG",
            tool_type_needed="endmill", fillet_radius_mm=fillet,
        )
        finish = OpSpec(
            op_id="OP020", feature_refs=["1"], feature_type="filleted_pocket",
            setup_id=setup, operation="contour_2d", tool_id="SMALL",
            tool_type_needed="endmill", fillet_radius_mm=fillet,
        )
        # BIG rough tool (r=6.0) leaves >0.5mm uncut in a 3.81mm corner -> rest_roughing;
        # SMALL finish tool (r=3.0) reaches the corner -> no rest_finish.
        tool_lookup = {
            "BIG": types.SimpleNamespace(diameter_mm=12.0),
            "SMALL": types.SimpleNamespace(diameter_mm=6.0),
        }
        rest = _p._rest_machining_ops([rough, finish], tool_lookup)
        ops = {r.operation for r in rest}
        self.assertIn("rest_roughing", ops)
        self.assertNotIn("rest_finish", ops)
        rr = next(r for r in rest if r.operation == "rest_roughing")
        self.assertEqual(rr.fillet_radius_mm, fillet)
        self.assertEqual(rr.feature_refs, ["1"])

        # When the rough tool already fits the corner (r=3.0 <= 3.81), nothing fires.
        tool_lookup["BIG"] = types.SimpleNamespace(diameter_mm=6.0)
        self.assertEqual(_p._rest_machining_ops([rough, finish], tool_lookup), [])

    def test_chamfer_feature_maps_to_chamfer_op(self) -> None:
        ctx = _minimal_context()
        cham = cascade_node_to_feature({
            "feature_id": 7, "class_name": "chamfer",
            "params": {"has_chamfer": True, "chamfer_size_mm": 0.5, "chamfer_angle_deg": 45.0},
        })
        ops = map_feature_to_operations(cham, ctx)
        self.assertEqual([o.operation for o in ops], ["chamfer"])
        self.assertEqual(ops[0].tool_type_needed, "chamfer_mill")
        self.assertEqual(ops[0].lateral_extent_mm, 0.5)

    def test_engrave_spec_parses_from_descriptor(self) -> None:
        from setup_descriptor import parse_setup_descriptor
        desc = parse_setup_descriptor({
            "part_id": "p",
            "setups": {"front": {"engrave": [
                {"text": "XR004", "target": {"feature_id": "21"}, "depth_mm": 0.2},
            ]}},
        })
        specs = desc.setups["front"].engrave
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].text, "XR004")
        self.assertEqual(specs[0].target_feature_id, "21")
        self.assertEqual(specs[0].depth_mm, 0.2)
        # malformed target (both / neither) is rejected, not silently dropped
        from setup_descriptor import SetupDescriptorError
        with self.assertRaises(SetupDescriptorError):
            parse_setup_descriptor({"setups": {"s": {"engrave": [{"text": "x"}]}}})

    def test_declared_engraving_emits_op_with_source(self) -> None:
        ctx = load_machining_context(CONTEXT_PATH)
        ctx.setups[0].engrave = [
            {"text": "XR004", "target": {"feature_id": "17"}, "depth_mm": 0.2}
        ]
        cam = plan(CASCADE_PATH, ctx, seq_search="beam", seq_beam_width=5)
        eng = [o for o in cam.operations if o.operation == "engraving"]
        self.assertEqual(len(eng), 1)
        self.assertEqual(eng[0].feature_refs, ["17"])
        self.assertEqual(eng[0].attributes["text"], "XR004")
        self.assertEqual(eng[0].attributes["source"], "explicit_spec")
        self.assertEqual(eng[0].attributes["depth_mm"], 0.2)
        self.assertEqual(cam.operations[-1].operation, "deburr")  # engraving before deburr
        flags = cam.metadata["planner_stats"].get("review_flags", [])
        self.assertFalse(any(f.get("source") == "engrave_unresolved" for f in flags))

    def test_unresolved_engraving_is_flagged_not_fabricated(self) -> None:
        ctx = load_machining_context(CONTEXT_PATH)
        ctx.setups[0].engrave = [
            {"text": "X", "target": {"feature_id": "999999"}, "depth_mm": 0.1}
        ]
        cam = plan(CASCADE_PATH, ctx, seq_search="beam", seq_beam_width=5)
        self.assertEqual([o for o in cam.operations if o.operation == "engraving"], [])
        flags = cam.metadata["planner_stats"].get("review_flags", [])
        self.assertTrue(any(f.get("source") == "engrave_unresolved" for f in flags))

    def test_deburr_is_appended_last_over_whole_setup(self) -> None:
        ctx = load_machining_context(CONTEXT_PATH)
        cam = plan(CASCADE_PATH, ctx, seq_search="beam", seq_beam_width=5)
        deburrs = [op for op in cam.operations if op.operation == "deburr"]
        self.assertEqual(len(deburrs), 1, "exactly one whole-setup deburr op")
        self.assertEqual(cam.operations[-1].operation, "deburr", "deburr sequences last")
        # whole-setup: references more than one feature
        self.assertGreater(len(deburrs[0].feature_refs), 1)

    def test_front_stock_flat_maps_to_facing(self) -> None:
        ctx = _minimal_context(
            setups=[
                SetupContext(
                    setup_id="front",
                    opening_axis="+Y",
                    opening_axis_vector=(0.0, 1.0, 0.0),
                    machining_side="front",
                    pocket_access={},
                ),
            ],
        )
        large_flat = cascade_node_to_feature(
            {
                "feature_id": 18,
                "class_name": "flat",
                "params": {"area": 7875.0},
            }
        )
        small_flat = cascade_node_to_feature(
            {
                "feature_id": 19,
                "class_name": "flat",
                "params": {"area": 844.0},
            }
        )
        facing_ids = identify_facing_feature_ids([large_flat, small_flat], "front")
        self.assertEqual(facing_ids, frozenset({"18"}))

        facing_ops = map_feature_to_operations(
            large_flat,
            ctx,
            facing_feature_ids=facing_ids,
        )
        self.assertEqual(facing_ops[0].operation, "facing")
        self.assertEqual(facing_ops[0].tool_type_needed, "face_mill")

        floor_ops = map_feature_to_operations(
            small_flat,
            ctx,
            facing_feature_ids=facing_ids,
        )
        self.assertEqual(floor_ops[0].operation, "raster")

    def test_facing_selects_face_mill_near_shop_diameter(self) -> None:
        tools = [
            Tool(
                tool_id="FM_SMALL",
                tool_type="face_mill",
                diameter_mm=25.4,
                flute_length_mm=10.0,
                presets=[
                    ToolPreset(
                        preset_name="AluWrought_Face_Rough_Starred",
                        preset_material="aluminum",
                        spindle_rpm=12000.0,
                        feed_mm_per_min=3556.0,
                        stepdown_mm=5.08,
                        stepover_mm=35.56,
                    ),
                ],
            ),
            Tool(
                tool_id="FM_SHOP",
                tool_type="face_mill",
                diameter_mm=38.1,
                flute_length_mm=12.0,
                presets=[
                    ToolPreset(
                        preset_name="AluWrought_Face_Rough_Starred",
                        preset_material="aluminum",
                        spindle_rpm=12000.0,
                        feed_mm_per_min=3556.0,
                    ),
                ],
            ),
        ]
        op = OpSpec(
            operation="facing",
            tool_type_needed="face_mill",
        )
        chosen = select_tool(op, tools, material="aluminum")
        self.assertEqual(chosen, "FM_SHOP")
        tool = next(item for item in tools if item.tool_id == chosen)
        self.assertAlmostEqual(tool.diameter_mm, 38.1, places=1)
        preset = resolve_preset(tool, op, "aluminum")
        self.assertIsNotNone(preset)
        assert preset is not None
        self.assertIn("Face", preset.preset_name)

    def test_small_hole_still_drills(self) -> None:
        ctx = _minimal_context()
        hole = cascade_node_to_feature(
            {
                "feature_id": 9,
                "class_name": "through_hole",
                "params": {"nominal_diameter": 6.0, "depth": 10.0},
            }
        )
        ops = map_feature_to_operations(hole, ctx)
        self.assertEqual(ops[0].operation, "drill")
        self.assertEqual(select_tool(ops[0], ctx.tools), "T02")


class TestToolSelection(unittest.TestCase):
    def test_smallest_fitting_drill_selected(self) -> None:
        op = OpSpec(
            tool_type_needed="drill",
            diameter_mm=6.0,
            depth_mm=20.0,
        )
        self.assertEqual(select_tool(op, default_tool_library()), "T02")

    def test_open_finish_prefers_larger_tool_not_catalog_minimum(self) -> None:
        tools = [
            Tool(
                tool_id="MICRO",
                tool_type="bullnose_endmill",
                diameter_mm=0.2,
                flute_length_mm=10.0,
            ),
            Tool(
                tool_id="MID",
                tool_type="bullnose_endmill",
                diameter_mm=12.7,
                flute_length_mm=30.0,
            ),
            Tool(
                tool_id="LARGE",
                tool_type="bullnose_endmill",
                diameter_mm=19.05,
                flute_length_mm=30.0,
            ),
        ]
        op = OpSpec(
            operation="raster",
            tool_type_needed="endmill",
            depth_mm=5.0,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "MID")

    def test_open_finish_respects_lateral_extent(self) -> None:
        tools = [
            Tool(tool_id="BIG", tool_type="endmill", diameter_mm=20.0, flute_length_mm=30.0),
            Tool(tool_id="OK", tool_type="endmill", diameter_mm=10.0, flute_length_mm=30.0),
        ]
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
            depth_mm=5.0,
            lateral_extent_mm=12.0,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "OK")

    def test_finish_contour_respects_fillet_radius_cap(self) -> None:
        tools = [
            Tool(tool_id="TOO_BIG", tool_type="bullnose_endmill", diameter_mm=16.0, flute_length_mm=30.0),
            Tool(tool_id="OK", tool_type="bullnose_endmill", diameter_mm=12.0, flute_length_mm=30.0),
        ]
        op = OpSpec(
            operation="contour_2d",
            tool_type_needed="endmill",
            depth_mm=5.0,
            fillet_radius_mm=6.35,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "OK")

    def test_oversized_hole_is_unresolved(self) -> None:
        op = OpSpec(
            tool_type_needed="drill",
            diameter_mm=81.28,
            depth_mm=25.0,
        )
        self.assertEqual(select_tool(op, default_tool_library()), UNRESOLVED_TOOL_ID)

    @unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library missing")
    def test_bore_prefers_larger_endmill_in_sane_range(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        op = OpSpec(
            operation="helix_bore",
            tool_type_needed="endmill",
            depth_mm=25.0,
            diameter_mm=101.7,
        )
        chosen_id = select_tool(op, tools)
        chosen = next(t for t in tools if t.tool_id == chosen_id)
        self.assertGreaterEqual(chosen.diameter_mm, 9.525)
        self.assertLessEqual(chosen.diameter_mm, 25.4)
        self.assertGreater(chosen.diameter_mm, 9.525)


class TestPrecedence(unittest.TestCase):
    def test_drill_before_tap_and_rough_before_finish(self) -> None:
        ctx = _minimal_context()
        tapped = cascade_node_to_feature(
            {
                "feature_id": 9,
                "class_name": "through_hole",
                "params": {"is_tapped": True, "nominal_diameter": 3.0, "depth": 10.0},
            }
        )
        ops = map_feature_to_operations(tapped, ctx)
        ops[0].op_id = "OP010"
        ops[1].op_id = "OP020"

        pocket = cascade_node_to_feature(
            {"feature_id": 1, "class_name": "filleted_pocket", "params": {"depth_below_top_mm": 5.0}}
        )
        pocket_ops = map_feature_to_operations(pocket, ctx)
        pocket_ops[0].op_id = "OP030"
        pocket_ops[1].op_id = "OP040"

        all_ops = ops + pocket_ops
        precedence = build_precedence(all_ops)
        self.assertIn("OP010", precedence["OP020"])
        self.assertIn("OP030", precedence["OP040"])

    def test_grouped_finish_depends_on_overlapping_rough(self) -> None:
        rough = OpSpec(
            op_id="OP010",
            feature_refs=["1", "2", "3"],
            operation="pocket",
        )
        finish = OpSpec(
            op_id="OP020",
            feature_refs=["2", "4"],
            operation="contour_2d",
        )
        precedence = build_precedence([rough, finish])
        self.assertIn("OP010", precedence["OP020"])


class TestBatchGrouping(unittest.TestCase):
    def test_same_tool_and_operation_type_merge(self) -> None:
        shared_params = assign_parameters(
            OpSpec(operation="pocket", tool_type_needed="endmill"),
            default_tool_library()[2],
            _minimal_context(material="aluminum"),
        )
        ops = [
            OpSpec(
                op_id="OP010",
                feature_refs=["1"],
                feature_type="filleted_pocket",
                setup_id="rear",
                operation="pocket",
                tool_id="T03",
                tool_type_needed="endmill",
                parameters=shared_params,
            ),
            OpSpec(
                op_id="OP020",
                feature_refs=["2"],
                feature_type="filleted_pocket",
                setup_id="rear",
                operation="pocket",
                tool_id="T03",
                tool_type_needed="endmill",
                parameters=shared_params,
            ),
        ]
        grouped, splits = group_operations_by_tool_strategy(ops, default_tool_library())
        self.assertEqual(splits, 0)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].feature_refs, ["1", "2"])

    def test_different_operation_type_do_not_merge(self) -> None:
        params = assign_parameters(
            OpSpec(operation="pocket", tool_type_needed="endmill"),
            default_tool_library()[2],
            _minimal_context(material="aluminum"),
        )
        finish_params = assign_parameters(
            OpSpec(operation="contour_2d", tool_type_needed="endmill"),
            default_tool_library()[2],
            _minimal_context(material="aluminum"),
        )
        ops = [
            OpSpec(
                op_id="OP010",
                feature_refs=["1"],
                setup_id="rear",
                operation="pocket",
                tool_id="T03",
                tool_type_needed="endmill",
                parameters=params,
            ),
            OpSpec(
                op_id="OP020",
                feature_refs=["2"],
                setup_id="rear",
                operation="contour_2d",
                tool_id="T03",
                tool_type_needed="endmill",
                parameters=finish_params,
            ),
        ]
        grouped, _ = group_operations_by_tool_strategy(ops, default_tool_library())
        self.assertEqual(len(grouped), 2)

    def test_different_tool_do_not_merge(self) -> None:
        params = assign_parameters(
            OpSpec(operation="pocket", tool_type_needed="endmill"),
            default_tool_library()[2],
            _minimal_context(material="aluminum"),
        )
        ops = [
            OpSpec(
                op_id="OP010",
                feature_refs=["1"],
                setup_id="rear",
                operation="pocket",
                tool_id="T03",
                tool_type_needed="endmill",
                parameters=params,
            ),
            OpSpec(
                op_id="OP020",
                feature_refs=["2"],
                setup_id="rear",
                operation="pocket",
                tool_id="T04",
                tool_type_needed="endmill",
                parameters=params,
            ),
        ]
        grouped, _ = group_operations_by_tool_strategy(ops, default_tool_library())
        self.assertEqual(len(grouped), 2)

    def test_reachability_splits_tight_clearance_feature(self) -> None:
        params = assign_parameters(
            OpSpec(operation="contour_2d", tool_type_needed="endmill"),
            default_tool_library()[2],
            _minimal_context(material="aluminum"),
        )
        ops = [
            OpSpec(
                op_id="OP010",
                feature_refs=["1"],
                setup_id="rear",
                operation="contour_2d",
                tool_id="T03",
                tool_type_needed="endmill",
                lateral_extent_mm=20.0,
                parameters=params,
            ),
            OpSpec(
                op_id="OP020",
                feature_refs=["2"],
                setup_id="rear",
                operation="contour_2d",
                tool_id="T03",
                tool_type_needed="endmill",
                lateral_extent_mm=1.0,
                parameters=params,
            ),
        ]
        grouped, splits = group_operations_by_tool_strategy(ops, default_tool_library())
        self.assertGreaterEqual(splits, 1)
        self.assertGreaterEqual(len(grouped), 2)
        self.assertLess(len(grouped[0].feature_refs), 2)

    def test_precedence_preserved_after_grouping(self) -> None:
        rough_a = OpSpec(
            op_id="OP010",
            feature_refs=["1"],
            setup_id="rear",
            operation="pocket",
            tool_id="T03",
            tool_type_needed="endmill",
        )
        rough_b = OpSpec(
            op_id="OP020",
            feature_refs=["2"],
            setup_id="rear",
            operation="pocket",
            tool_id="T03",
            tool_type_needed="endmill",
        )
        finish_a = OpSpec(
            op_id="OP030",
            feature_refs=["1"],
            setup_id="rear",
            operation="contour_2d",
            tool_id="T03",
            tool_type_needed="endmill",
        )
        finish_b = OpSpec(
            op_id="OP040",
            feature_refs=["2"],
            setup_id="rear",
            operation="contour_2d",
            tool_id="T03",
            tool_type_needed="endmill",
        )
        grouped = group_operations_by_tool_strategy([rough_a, rough_b, finish_a, finish_b], default_tool_library())[0]
        self.assertEqual(len(grouped), 2)
        _assign_op_ids(grouped)
        precedence = build_precedence(grouped)
        rough_group = next(op for op in grouped if op.operation == "pocket")
        finish_group = next(op for op in grouped if op.operation == "contour_2d")
        self.assertIn(rough_group.op_id, precedence[finish_group.op_id])
        ordered = sequence(grouped, precedence, tool_lookup=_tool_by_id(default_tool_library()))
        rough_idx = next(i for i, op in enumerate(ordered) if op.op_id == rough_group.op_id)
        finish_idx = next(i for i, op in enumerate(ordered) if op.op_id == finish_group.op_id)
        self.assertLess(rough_idx, finish_idx)


def _tool_by_id(tools):
    return {tool.tool_id: tool for tool in tools}


def _mill_tools(
    *,
    endmill_id: str = "EM_6",
    bullnose_id: str = "BN_6",
    ball_id: str = "BALL_6",
    diameter_mm: float = 6.0,
    flute_length_mm: float = 25.0,
) -> list[Tool]:
    finish_preset = ToolPreset(
        preset_name="AluWrought_Finish",
        preset_material="aluminum",
        spindle_rpm=10000.0,
        feed_mm_per_min=800.0,
        stepdown_mm=0.5,
        stepover_mm=0.3,
    )
    return [
        Tool(
            tool_id=endmill_id,
            tool_type="endmill",
            diameter_mm=diameter_mm,
            flute_length_mm=flute_length_mm,
            presets=[finish_preset],
        ),
        Tool(
            tool_id=bullnose_id,
            tool_type="bullnose_endmill",
            diameter_mm=diameter_mm,
            flute_length_mm=flute_length_mm,
            corner_radius_mm=0.762,
            presets=[finish_preset],
        ),
        Tool(
            tool_id=ball_id,
            tool_type="ball_endmill",
            diameter_mm=diameter_mm,
            flute_length_mm=flute_length_mm,
            presets=[finish_preset],
        ),
    ]


class TestBullnoseFinishingSelection(unittest.TestCase):
    def test_floor_finish_prefers_bullnose_over_endmill(self) -> None:
        tools = _mill_tools()
        op = OpSpec(
            operation="raster",
            tool_type_needed="endmill",
            depth_mm=5.0,
            lateral_extent_mm=20.0,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "BN_6")
        chosen_tool = next(t for t in tools if t.tool_id == chosen)
        self.assertEqual(chosen_tool.tool_type, "bullnose_endmill")
        self.assertTrue(_tool_fits_op(chosen_tool, op))

    def test_finish_contour_prefers_bullnose_over_endmill(self) -> None:
        tools = _mill_tools()
        op = OpSpec(
            operation="contour_2d",
            tool_type_needed="endmill",
            depth_mm=5.0,
            lateral_extent_mm=20.0,
        )
        self.assertEqual(select_tool(op, tools), "BN_6")

    def test_falls_back_to_endmill_when_no_bullnose_fits(self) -> None:
        tools = [
            Tool(
                tool_id="BN_12",
                tool_type="bullnose_endmill",
                diameter_mm=12.0,
                flute_length_mm=10.0,
            ),
            Tool(
                tool_id="EM_6",
                tool_type="endmill",
                diameter_mm=6.0,
                flute_length_mm=25.0,
            ),
        ]
        op = OpSpec(
            operation="contour_2d",
            tool_type_needed="endmill",
            depth_mm=20.0,
            lateral_extent_mm=8.0,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "EM_6")
        self.assertNotEqual(chosen, UNRESOLVED_TOOL_ID)

    def test_surface_finish_keeps_ball_primary(self) -> None:
        tools = _mill_tools()
        op = OpSpec(
            operation="constant_scallop",
            tool_type_needed="ball_endmill",
            depth_mm=5.0,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "BALL_6")

    def test_surface_finish_bullnose_before_endmill_when_no_ball(self) -> None:
        tools = [
            Tool(
                tool_id="BN_6",
                tool_type="bullnose_endmill",
                diameter_mm=6.0,
                flute_length_mm=25.0,
            ),
            Tool(
                tool_id="EM_6",
                tool_type="endmill",
                diameter_mm=6.0,
                flute_length_mm=25.0,
            ),
        ]
        op = OpSpec(
            operation="constant_scallop",
            tool_type_needed="ball_endmill",
            depth_mm=5.0,
        )
        self.assertEqual(select_tool(op, tools), "BN_6")

    def test_roughing_still_uses_endmill(self) -> None:
        tools = _mill_tools()
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
            depth_mm=5.0,
            lateral_extent_mm=20.0,
        )
        chosen = select_tool(op, tools)
        self.assertEqual(chosen, "EM_6")
        chosen_tool = next(t for t in tools if t.tool_id == chosen)
        self.assertEqual(chosen_tool.tool_type, "endmill")

    @unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library missing")
    def test_bullnose_finish_uses_toolpath_preset(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        bullnose = next(t for t in tools if t.tool_type == "bullnose_endmill")
        ctx = _minimal_context(material="aluminum", tools=tools)
        op = OpSpec(
            operation="raster",
            tool_type_needed="endmill",
            depth_mm=5.0,
        )
        params = assign_parameters(op, bullnose, ctx)
        self.assertEqual(params.param_source, "toolpath_preset")


class TestSequence(unittest.TestCase):
    def test_topological_order_respects_edges(self) -> None:
        ops = [
            OpSpec(op_id="OP010", feature_refs=["1"], operation="drill"),
            OpSpec(op_id="OP020", feature_refs=["1"], operation="drill"),
        ]
        precedence = {"OP010": [], "OP020": ["OP010"]}
        ordered = sequence(ops, precedence)
        self.assertEqual([op.op_id for op in ordered], ["OP010", "OP020"])

    def test_cycle_detection(self) -> None:
        ops = [
            OpSpec(op_id="OP010"),
            OpSpec(op_id="OP020"),
        ]
        precedence = {"OP010": ["OP020"], "OP020": ["OP010"]}
        with self.assertRaises(ValueError) as ctx:
            sequence(ops, precedence)
        self.assertIn("cycle", str(ctx.exception).lower())


@unittest.skipUnless(DRILL_LIBRARY.is_file(), "Kennametal drill library missing")
class TestAssignParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.drill_tools = load_tool_library(DRILL_LIBRARY)
        cls.drill_tool = next(
            t for t in cls.drill_tools if any(p.preset_name == "LowCSteel_Drill" for p in t.presets)
        )
        cls.drill_preset = next(
            p for p in cls.drill_tool.presets if p.preset_name == "AluWrought_Drill"
        )

    def test_drill_uses_plunge_not_v_f(self) -> None:
        ctx = _minimal_context(material="aluminum")
        op = OpSpec(operation="drill", tool_type_needed="drill")
        params = assign_parameters(op, self.drill_tool, ctx)

        self.assertEqual(params.param_source, "toolpath_preset")
        self.assertAlmostEqual(params.plunge_mm_per_min, self.drill_preset.plunge_mm_per_min, places=1)
        self.assertAlmostEqual(params.spindle_rpm, self.drill_preset.spindle_rpm, places=1)
        self.assertIsNone(params.stepdown_mm)
        self.assertIsNone(params.stepover_mm)
        self.assertIsNotNone(params.feed_mm_per_min)
        if self.drill_preset.feed_per_rev_mm and self.drill_preset.spindle_rpm:
            expected_feed = self.drill_preset.feed_per_rev_mm * self.drill_preset.spindle_rpm
            self.assertAlmostEqual(params.feed_mm_per_min, expected_feed, places=1)
        self.assertNotAlmostEqual(
            params.plunge_mm_per_min,
            self.drill_preset.feed_mm_per_min or -1.0,
        )

    @unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library missing")
    def test_pocket_uses_feed_and_stepover(self) -> None:
        endmill = load_tool_library(SAMPLE_LIBRARY)[0]
        preset = next(p for p in endmill.presets if p.preset_name == "AluWrought_Adaptive_Rough")
        ctx = _minimal_context(material="aluminum")
        op = OpSpec(operation="pocket", tool_type_needed="endmill")
        params = assign_parameters(op, endmill, ctx)

        self.assertEqual(params.param_source, "toolpath_preset")
        self.assertAlmostEqual(params.feed_mm_per_min, preset.feed_mm_per_min, places=1)
        self.assertAlmostEqual(params.stepover_mm, preset.stepover_mm, places=2)
        self.assertAlmostEqual(params.stepdown_mm, preset.stepdown_mm, places=2)
        self.assertAlmostEqual(params.spindle_rpm, preset.spindle_rpm, places=1)

    @unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library missing")
    def test_surface_finish_selects_ball_endmill(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        ball_tools = [t for t in tools if t.tool_type == "ball_endmill"]
        self.assertTrue(ball_tools)
        op = OpSpec(
            operation="constant_scallop",
            tool_type_needed="ball_endmill",
            depth_mm=5.0,
        )
        chosen = select_tool(op, tools)
        self.assertNotEqual(chosen, UNRESOLVED_TOOL_ID)
        chosen_tool = next(t for t in tools if t.tool_id == chosen)
        self.assertEqual(chosen_tool.tool_type, "ball_endmill")

    @unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library missing")
    def test_finish_prefers_finish_preset_over_rough(self) -> None:
        endmill = load_tool_library(SAMPLE_LIBRARY)[0]
        ctx = _minimal_context(material="aluminum")
        rough = OpSpec(operation="pocket", tool_type_needed="endmill")
        finish = OpSpec(operation="contour_2d", tool_type_needed="endmill")
        rough_preset = resolve_preset(endmill, rough, "aluminum")
        finish_preset = resolve_preset(endmill, finish, "aluminum")
        self.assertIsNotNone(rough_preset)
        self.assertIsNotNone(finish_preset)
        assert rough_preset is not None and finish_preset is not None
        self.assertIn("Rough", rough_preset.preset_name)
        self.assertIn("Finish", finish_preset.preset_name)

    def test_material_prefers_matching_preset(self) -> None:
        tool = Tool(
            tool_id="T_material",
            tool_type="endmill",
            diameter_mm=6.0,
            presets=[
                ToolPreset(
                    preset_name="Steel_Adaptive_Rough",
                    preset_material="steel",
                    spindle_rpm=8000.0,
                    feed_mm_per_min=500.0,
                    stepdown_mm=1.0,
                    stepover_mm=2.0,
                ),
                ToolPreset(
                    preset_name="AluWrought_Adaptive_Rough",
                    preset_material="aluminum",
                    spindle_rpm=12000.0,
                    feed_mm_per_min=700.0,
                    stepdown_mm=0.5,
                    stepover_mm=1.5,
                ),
            ],
        )
        op = OpSpec(operation="pocket", tool_type_needed="endmill")
        chosen = resolve_preset(tool, op, "aluminum")
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.preset_material, "aluminum")
        self.assertEqual(chosen.spindle_rpm, 12000.0)

    def test_missing_preset_falls_back_without_nulls(self) -> None:
        bare_tool = Tool(tool_id="T_bare", tool_type="endmill", diameter_mm=6.0, presets=[])
        ctx = _minimal_context(material="aluminum")
        op = OpSpec(operation="pocket", tool_type_needed="endmill")
        params = assign_parameters(op, bare_tool, ctx)

        self.assertEqual(params.param_source, "handbook_default")
        self.assertIsNotNone(params.spindle_rpm)
        self.assertIsNotNone(params.feed_mm_per_min)
        self.assertIsNotNone(params.plunge_mm_per_min)
        self.assertIsNotNone(params.stepdown_mm)
        self.assertIsNotNone(params.stepover_mm)

    def test_mixed_param_source_when_preset_incomplete(self) -> None:
        partial = Tool(
            tool_id="T_partial",
            tool_type="endmill",
            diameter_mm=6.0,
            presets=[
                ToolPreset(
                    preset_name="AluWrought_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=12000.0,
                ),
            ],
        )
        ctx = _minimal_context(material="aluminum")
        op = OpSpec(operation="pocket", tool_type_needed="endmill")
        params = assign_parameters(op, partial, ctx)

        self.assertEqual(params.param_source, "mixed")
        self.assertEqual(params.spindle_rpm, 12000.0)
        self.assertIsNotNone(params.feed_mm_per_min)
        self.assertIsNotNone(params.stepover_mm)

    def test_strategy_preset_beats_iso_material_category(self) -> None:
        tool = Tool(
            tool_id="T_skookum_like",
            tool_type="endmill",
            diameter_mm=12.7,
            presets=[
                ToolPreset(
                    preset_name="N - Aluminum & Aluminum Alloys",
                    preset_material="all",
                    spindle_rpm=10313.0,
                    feed_mm_per_min=2095.0,
                    plunge_mm_per_min=523.0,
                ),
                ToolPreset(
                    preset_name="LowCSteel_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=465.0,
                    plunge_mm_per_min=100.0,
                    stepdown_mm=2.54,
                    stepover_mm=9.5,
                ),
            ],
        )
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
        )
        chosen = resolve_preset(tool, op, "aluminum")
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.preset_name, "LowCSteel_Adaptive_Rough")

    def test_iso_category_preset_only_when_no_strategy_preset(self) -> None:
        tool = Tool(
            tool_id="T_iso_only",
            tool_type="endmill",
            diameter_mm=6.0,
            presets=[
                ToolPreset(
                    preset_name="N - Aluminum & Aluminum Alloys",
                    preset_material="all",
                    spindle_rpm=9000.0,
                    feed_mm_per_min=1500.0,
                ),
            ],
        )
        op = OpSpec(operation="pocket", tool_type_needed="endmill")
        chosen = resolve_preset(tool, op, "aluminum")
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen.preset_name, "N - Aluminum & Aluminum Alloys")

    def test_floor_wall_surface_ops_select_strategy_presets(self) -> None:
        tool = Tool(
            tool_id="T_finish_set",
            tool_type="bullnose_endmill",
            diameter_mm=12.7,
            presets=[
                ToolPreset(
                    preset_name="N - Aluminum & Aluminum Alloys",
                    preset_material="all",
                    spindle_rpm=10313.0,
                    feed_mm_per_min=2095.0,
                    plunge_mm_per_min=523.0,
                ),
                ToolPreset(
                    preset_name="LowCSteel_Floor_Finish",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=349.0,
                    stepdown_mm=3.0,
                    stepover_mm=11.43,
                    plunge_mm_per_min=80.0,
                ),
                ToolPreset(
                    preset_name="LowCSteel_Wall_Finish",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=349.0,
                    stepdown_mm=3.0,
                    stepover_mm=11.43,
                    plunge_mm_per_min=80.0,
                ),
                ToolPreset(
                    preset_name="LowCSteel_Surface",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=349.0,
                    stepover_mm=0.4,
                    plunge_mm_per_min=80.0,
                ),
            ],
        )
        floor_op = OpSpec(
            operation="raster",
            tool_type_needed="bullnose_endmill",
        )
        wall_op = OpSpec(
            operation="contour_2d",
            feature_type="wall",
            tool_type_needed="bullnose_endmill",
        )
        surface_op = OpSpec(
            operation="constant_scallop",
            tool_type_needed="bullnose_endmill",
        )
        fillet_op = OpSpec(
            operation="pencil",
            tool_type_needed="bullnose_endmill",
        )
        self.assertEqual(
            resolve_preset(tool, floor_op, "aluminum").preset_name,
            "LowCSteel_Floor_Finish",
        )
        self.assertEqual(
            resolve_preset(tool, wall_op, "aluminum").preset_name,
            "LowCSteel_Wall_Finish",
        )
        self.assertEqual(
            resolve_preset(tool, surface_op, "aluminum").preset_name,
            "LowCSteel_Surface",
        )
        self.assertEqual(
            resolve_preset(tool, fillet_op, "aluminum").preset_name,
            "LowCSteel_Surface",
        )

    def test_strategy_preset_yields_toolpath_preset_not_mixed(self) -> None:
        tool = Tool(
            tool_id="T_full_strategy",
            tool_type="endmill",
            diameter_mm=12.7,
            presets=[
                ToolPreset(
                    preset_name="N - Aluminum & Aluminum Alloys",
                    preset_material="all",
                    spindle_rpm=10313.0,
                    feed_mm_per_min=2095.0,
                    plunge_mm_per_min=523.0,
                ),
                ToolPreset(
                    preset_name="AluWrought_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=7639.0,
                    feed_mm_per_min=373.0,
                    plunge_mm_per_min=93.0,
                    stepdown_mm=31.75,
                    stepover_mm=1.55,
                ),
            ],
        )
        ctx = _minimal_context(material="aluminum")
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
        )
        params = assign_parameters(op, tool, ctx)
        self.assertEqual(params.param_source, "toolpath_preset")
        self.assertAlmostEqual(params.stepdown_mm, 31.75, places=2)
        self.assertAlmostEqual(params.stepover_mm, 1.55, places=2)

    def test_aluminum_job_prefers_aluwrought_tool_over_ferrous(self) -> None:
        ferrous = Tool(
            tool_id="ferrous::1",
            tool_type="endmill",
            diameter_mm=12.7,
            source="supabase:Skookum_Tools_General_Purpose_Ferrous_Inch",
            presets=[
                ToolPreset(
                    preset_name="LowCSteel_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=465.0,
                    stepdown_mm=2.0,
                    stepover_mm=1.0,
                ),
            ],
        )
        aluminum = Tool(
            tool_id="nonferrous::1",
            tool_type="endmill",
            diameter_mm=12.7,
            source="supabase:Skookum_Tools_General_Purpose_Non_Ferrous_Inch",
            presets=[
                ToolPreset(
                    preset_name="AluWrought_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=7639.0,
                    feed_mm_per_min=373.0,
                    stepdown_mm=31.75,
                    stepover_mm=1.55,
                ),
            ],
        )
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
            lateral_extent_mm=50.0,
        )
        chosen = select_tool(op, [ferrous, aluminum], material="aluminum")
        self.assertEqual(chosen, "nonferrous::1")

    def test_aluminum_job_falls_back_to_ferrous_when_only_option(self) -> None:
        ferrous = Tool(
            tool_id="ferrous::only",
            tool_type="endmill",
            diameter_mm=12.7,
            source="supabase:Skookum_Tools_Ultra_High_Performance_Inch",
            presets=[
                ToolPreset(
                    preset_name="LowCSteel_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=465.0,
                ),
            ],
        )
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
            lateral_extent_mm=50.0,
        )
        chosen = select_tool(op, [ferrous], material="aluminum")
        self.assertEqual(chosen, "ferrous::only")
        self.assertGreaterEqual(_tool_material_rank(ferrous, "aluminum"), 2)

    def test_aluminum_feeds_from_aluwrought_not_lowcsteel(self) -> None:
        ferrous = Tool(
            tool_id="ferrous::1",
            tool_type="endmill",
            diameter_mm=12.7,
            source="supabase:Skookum_Tools_General_Purpose_Ferrous_Inch",
            presets=[
                ToolPreset(
                    preset_name="LowCSteel_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=4583.0,
                    feed_mm_per_min=465.0,
                    plunge_mm_per_min=100.0,
                    stepdown_mm=2.0,
                    stepover_mm=1.0,
                ),
            ],
        )
        aluminum = Tool(
            tool_id="nonferrous::1",
            tool_type="endmill",
            diameter_mm=12.7,
            source="supabase:Skookum_Tools_General_Purpose_Non_Ferrous_Inch",
            presets=[
                ToolPreset(
                    preset_name="AluWrought_Adaptive_Rough",
                    preset_material="all",
                    spindle_rpm=7639.0,
                    feed_mm_per_min=3730.0,
                    plunge_mm_per_min=930.0,
                    stepdown_mm=31.75,
                    stepover_mm=1.55,
                ),
            ],
        )
        ctx = _minimal_context(material="aluminum", tools=[ferrous, aluminum])
        op = OpSpec(
            operation="pocket",
            tool_type_needed="endmill",
            lateral_extent_mm=50.0,
        )
        op.tool_id = select_tool(op, ctx.tools, material="aluminum")
        tool = next(t for t in ctx.tools if t.tool_id == op.tool_id)
        params = assign_parameters(op, tool, ctx)
        self.assertEqual(op.tool_id, "nonferrous::1")
        self.assertEqual(params.param_source, "toolpath_preset")
        self.assertAlmostEqual(params.feed_mm_per_min, 3730.0, places=1)
        self.assertNotAlmostEqual(params.feed_mm_per_min, 465.0, places=0)


class TestSetupScopeFilter(unittest.TestCase):
    def test_facing_scope_keeps_envelope_stock_flat_only(self) -> None:
        envelope_faces = frozenset({273})
        large_flat = cascade_node_to_feature(
            {
                "feature_id": 18,
                "class_name": "flat",
                "params": {"area": 7875.0, "face_indices": [273]},
            }
        )
        small_flat = cascade_node_to_feature(
            {
                "feature_id": 19,
                "class_name": "flat",
                "params": {"area": 844.0, "face_indices": [97]},
            }
        )
        pocket = cascade_node_to_feature(
            {
                "feature_id": 0,
                "class_name": "filleted_open_pocket",
                "params": {"face_indices": [80, 81]},
            }
        )
        ctx = _minimal_context(
            setups=[
                SetupContext(
                    setup_id="front",
                    opening_axis="+Y",
                    opening_axis_vector=(0.0, 1.0, 0.0),
                    machining_side="front",
                    pocket_access={},
                    scope=SetupScopeSpec(mode="filtered", classes=["facing"]),
                ),
            ],
        )
        kept, dropped, info = filter_features_for_setup(
            [large_flat, small_flat, pocket],
            ctx,
            envelope_faces=envelope_faces,
            use_reachability=False,
        )
        self.assertEqual(dropped, 2)
        self.assertEqual([f.feature_id for f in kept], ["18"])
        self.assertEqual(info["scope_mode"], "filtered")

    def test_full_scope_keeps_all_features(self) -> None:
        feat = cascade_node_to_feature(
            {"feature_id": 1, "class_name": "wall", "params": {}},
        )
        ctx = _minimal_context()
        kept, dropped, info = filter_features_for_setup(
            [feat],
            ctx,
            envelope_faces=frozenset(),
            use_reachability=False,
        )
        self.assertEqual(dropped, 0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(info["scope_mode"], "full")


class TestMultiSetupPlanning(unittest.TestCase):
    def test_two_graphs_emit_two_setups_and_setup_tagged_ops(self) -> None:
        self.assertTrue(CASCADE_PATH.is_file(), f"missing {CASCADE_PATH}")
        front_graph = Path(ROOT) / "pipeline_out" / "96260B_front" / "feature_graph_cascade.json"
        self.assertTrue(front_graph.is_file(), f"missing {front_graph}")

        from machining_context import build_context_v0

        setup_yaml = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
        rear_ctx = build_context_v0(
            Path(ROOT) / "96260B_REAR_XR004_PCD PLATE.stp copy",
            setup_yaml,
            CASCADE_PATH,
            setup_id="rear",
            material="aluminum",
            tool_source="hardcoded",
            setups_source="authored",
        )
        front_ctx = build_context_v0(
            Path(ROOT) / "96260B_FRONT_XR004_PCD PLATE.stp copy",
            setup_yaml,
            front_graph,
            setup_id="front",
            material="aluminum",
            tool_source="hardcoded",
            setups_source="authored",
        )
        cam_plan = plan_multi_setups(
            [
                SetupPlanInput(CASCADE_PATH, rear_ctx),
                SetupPlanInput(front_graph, front_ctx),
            ],
            setup_order=("rear", "front"),
            source_part="96260B",
        )
        setup_ids = {setup.setup_id for setup in cam_plan.setups}
        self.assertEqual(setup_ids, {"rear", "front"})
        op_setup_ids = {op.setup_id for op in cam_plan.operations}
        self.assertTrue(op_setup_ids.issubset(setup_ids))
        self.assertIn("rear", op_setup_ids)
        self.assertIn("front", op_setup_ids)

        rear_ops = [op for op in cam_plan.operations if op.setup_id == "rear"]
        front_ops = [op for op in cam_plan.operations if op.setup_id == "front"]
        self.assertEqual(len(rear_ops), 8)
        self.assertEqual(len(front_ops), 13)
        # each setup ends with its whole-setup deburr pass
        self.assertEqual(rear_ops[-1].operation, "deburr")
        self.assertEqual(front_ops[-1].operation, "deburr")
        # front's 3.81mm-fillet pockets trigger rest roughing (rough tool too big for
        # the corner); rear's 6.35mm fillets do not
        self.assertTrue(any(op.operation == "rest_roughing" for op in front_ops))
        self.assertFalse(any("rest" in op.operation for op in rear_ops))

        per_setup = cam_plan.metadata["planner_stats"]["per_setup"]
        self.assertEqual(per_setup["rear"]["scope_mode"], "reachability")
        self.assertEqual(per_setup["rear"]["opening_axis"], "+Y")
        self.assertEqual(per_setup["rear"]["reachability_dir"], "-Z")
        self.assertEqual(per_setup["front"]["reachability_dir"], "+Z")
        self.assertEqual(per_setup["rear"]["features_kept"], 36)
        self.assertEqual(per_setup["front"]["features_kept"], 39)

    def test_cross_setup_sequence_and_precedence(self) -> None:
        rear_ops = [
            OpSpec(
                op_id="OP010",
                feature_refs=["1"],
                setup_id="rear",
                operation="pocket",
                tool_id="T05",
                tool_type_needed="endmill",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
            OpSpec(
                op_id="OP020",
                feature_refs=["2"],
                setup_id="rear",
                operation="contour_2d",
                tool_id="T05",
                tool_type_needed="endmill",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
        ]
        front_ops = [
            OpSpec(
                op_id="OP010",
                feature_refs=["18"],
                setup_id="front",
                operation="facing",
                tool_id="FM_SHOP",
                tool_type_needed="face_mill",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
        ]

        from planner import _apply_cross_setup_precedence, _reassign_global_op_ids

        merged = list(rear_ops) + list(front_ops)
        _reassign_global_op_ids(merged)
        _apply_cross_setup_precedence(merged, ("rear", "front"))

        rear_indices = [i for i, op in enumerate(merged) if op.setup_id == "rear"]
        front_indices = [i for i, op in enumerate(merged) if op.setup_id == "front"]
        self.assertLess(max(rear_indices), min(front_indices))
        boundary_id = merged[rear_indices[-1]].op_id
        for op in merged:
            if op.setup_id == "front":
                self.assertIn(boundary_id, op.depends_on)


class TestPlanIntegration(unittest.TestCase):
    def test_full_plan_validates_as_cam_plan(self) -> None:
        self.assertTrue(CASCADE_PATH.is_file(), f"missing {CASCADE_PATH}")
        self.assertTrue(CONTEXT_PATH.is_file(), f"missing {CONTEXT_PATH}")
        context = load_machining_context(CONTEXT_PATH)
        cam_plan = plan(CASCADE_PATH, context)
        self.assertIsInstance(cam_plan, CamPlan)
        self.assertGreater(len(cam_plan.operations), 0)
        self.assertEqual(cam_plan.setups[0].setup_id, "rear")

        stats = cam_plan.metadata["planner_stats"]
        # 62 = 105 raw cascade nodes minus 43 merged by lobe-contour merge on export.
        self.assertEqual(stats["nodes_in"], 62)
        self.assertEqual(stats["features_kept"], 36)
        self.assertEqual(stats["features_dropped"], 0)
        self.assertEqual(stats["reachability_dir"], "-Z")

        coaxial_bore = [
            op for op in cam_plan.operations
            if op.operation == "helix_bore" and "15" in op.feature_refs
        ]
        self.assertEqual(len(coaxial_bore), 1)
        self.assertEqual(coaxial_bore[0].feature_refs, ["15"])

        wall_ops = [op for op in cam_plan.operations if op.operation == "contour_2d"]
        surface_ops = [op for op in cam_plan.operations if op.operation == "constant_scallop"]
        self.assertGreaterEqual(len(wall_ops), 1)
        self.assertGreaterEqual(len(surface_ops), 1)

        stats = cam_plan.metadata["planner_stats"]
        self.assertGreaterEqual(stats.get("operations_out"), 5)
        # grew with the surface-finish split: +deburr, +radial_spiral, +steep_shallow
        self.assertLessEqual(stats.get("operations_out"), 8)

        tool_lookup = {tool.tool_id: tool for tool in context.tools}
        open_finish_types = {
            "raster",
            "contour_2d",
            "waterline",
            "constant_scallop",
            "pencil",
        }
        for op in cam_plan.operations:
            if op.operation not in open_finish_types:
                continue
            tool = tool_lookup.get(op.tool_id)
            if tool is None:
                continue
            self.assertGreaterEqual(
                tool.diameter_mm,
                3.0,
                msg=f"{op.op_id} {op.operation} picked sub-3mm tool {tool.tool_id}",
            )

        self.assertGreaterEqual(stats.get("operations_before_grouping", 0), 30)
        self.assertLess(stats["operations_out"], stats["operations_before_grouping"])
        self.assertGreaterEqual(stats["operations_out"], 5)
        self.assertLess(stats["operations_out"], 10)

        multi_feature_ops = [op for op in cam_plan.operations if len(op.feature_refs) > 1]
        self.assertGreater(len(multi_feature_ops), 0)

    def test_generated_example_file_validates(self) -> None:
        self.assertTrue(PLAN_PATH.is_file(), f"missing {PLAN_PATH}; run planner.py first")
        plan_doc = load_cam_plan(PLAN_PATH)
        if len(plan_doc.setups) > 1:
            self.assertEqual(plan_doc.source_part, "96260B")
            refs = plan_doc.metadata.get("feature_graph_refs", {})
            self.assertIn("rear", refs)
            self.assertIn("front", refs)
        else:
            self.assertEqual(
                plan_doc.feature_graph_ref,
                "pipeline_out/96260B_rear/feature_graph_cascade.json",
            )
            self.assertEqual(plan_doc.source_part, load_machining_context(CONTEXT_PATH).part_id)


def _reach_node(fid, cls, dirs, *, verified=True, exempt=False, has_reach=True):
    """Synthetic cascade node carrying a step-4a approach.reachability block."""
    approach = {"setup_dir": (dirs[0] if dirs else None), "reachable_3axis": bool(dirs)}
    if has_reach:
        reach = {"verified": verified, "reachable_dirs": list(dirs)}
        if exempt:
            reach["exempt"] = True
        approach["reachability"] = reach
    return {"feature_id": fid, "class_name": cls, "params": {}, "approach": approach}


class TestReachabilityIntegration(unittest.TestCase):
    """Lock-in: the planner consumes step-4a reachability to scope features.

    Guards the step-2/3/4 -> planner integration so it cannot silently regress
    back to the class-only scope filter.
    """

    def _front_ctx(self) -> MachiningContext:
        return _minimal_context(setups=[SetupContext(
            setup_id="front", opening_axis="+Y",
            opening_axis_vector=(0.0, 1.0, 0.0), machining_side="front",
        )])

    def test_reachability_dir_maps_front_plus_z_back_minus_z(self) -> None:
        self.assertEqual(_reachability_dir_for_setup(self._front_ctx()), "+Z")
        self.assertEqual(_reachability_dir_for_setup(_minimal_context()), "-Z")  # back

    def test_reachability_filter_keeps_and_drops(self) -> None:
        nodes = [
            _reach_node(0, "filleted_pocket", ["+Z"]),               # keep (this side)
            _reach_node(1, "filleted_pocket", ["-Z"]),               # drop (other side)
            _reach_node(2, "through_hole", ["+Z", "-Z"]),            # keep (two-sided)
            _reach_node(3, "filleted_pocket", [], has_reach=False),  # drop (missing)
            _reach_node(4, "wall", ["+Z"], exempt=True),             # keep (exempt w/ dirs)
            _reach_node(5, "wall", [], exempt=True),                 # drop (exempt, no dirs)
        ]
        feats = [cascade_node_to_feature(n) for n in nodes]
        nodes_by_id = {str(n["feature_id"]): n for n in nodes}
        kept, dropped, info = filter_features_for_setup_by_reachability(
            feats, nodes_by_id, {"nodes": nodes}, self._front_ctx(),
        )
        self.assertEqual({f.feature_id for f in kept}, {"0", "2", "4"})
        self.assertEqual(dropped, 3)
        self.assertEqual(info["scope_mode"], "reachability")
        self.assertEqual(info["reachability_dir"], "+Z")
        self.assertEqual(info["missing_reachability"], 1)

    def test_dispatcher_routes_to_reachability_when_present(self) -> None:
        nodes = [
            _reach_node(0, "filleted_pocket", ["+Z"]),
            _reach_node(1, "filleted_pocket", ["-Z"]),
        ]
        feats = [cascade_node_to_feature(n) for n in nodes]
        nodes_by_id = {str(n["feature_id"]): n for n in nodes}
        kept, _dropped, info = filter_features_for_setup(
            feats, self._front_ctx(), envelope_faces=frozenset(),
            nodes_by_id=nodes_by_id, graph={"nodes": nodes},
        )
        self.assertEqual(info["scope_mode"], "reachability")
        self.assertEqual({f.feature_id for f in kept}, {"0"})

    def test_dispatcher_falls_back_to_class_scope_without_reachability(self) -> None:
        plain = [{"feature_id": 0, "class_name": "filleted_pocket", "params": {}}]
        feats = [cascade_node_to_feature(n) for n in plain]
        kept, _dropped, info = filter_features_for_setup(
            feats, self._front_ctx(), envelope_faces=frozenset(),
            nodes_by_id={"0": plain[0]}, graph={"nodes": plain},
        )
        self.assertEqual(info["scope_mode"], "full")
        self.assertEqual({f.feature_id for f in kept}, {"0"})


if __name__ == "__main__":
    unittest.main()
