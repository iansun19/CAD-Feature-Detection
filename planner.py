"""planner.py - v0 vertical slice: feature graph + machining context -> CamPlan.

Rule-based planner covering holes, pockets, walls, surfaces, fillets, profiles, and
faces. Ops are batched by (tool_id, operation, setup_id) before sequencing.

Operations are drawn from the single flat canonical bank in operation_bank.py; the
former operation_type/strategy split is retired. Which bank op a feature maps to is
geometry-driven (see map_feature_to_operations): the 3D_surface flag splits 2.5D
vs 3D roughing/finishing, pocket access splits closed vs open clearing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cam_plan_schema import (
    CamPlan,
    MachiningParameters,
    Operation,
    PocketAccess,
    Setup,
    ToolRef,
    write_cam_plan,
)
from operation_bank import FINISH_PHASE_OPS, ROUGH_PHASE_OPS
from operation_bank import Operation as BankOp
from sequence_search import SeqSearchStrategy, search_sequence
from machining_context import (
    MachiningContext,
    SetupScopeSpec,
    Tool,
    ToolPreset,
    load_feature_graph,
    vector_to_opening_axis_label,
)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent

UNRESOLVED_TOOL_ID = "UNRESOLVED"

# Recognizer class names observed on 96260B (+ hole/pocket families from task brief).
HOLE_CLASSES = frozenset({
    "through_hole",
    "hole",
    "blind_hole",
    "filleted_blind_hole",
})
POCKET_CLASSES = frozenset({
    "filleted_pocket",
    "filleted_open_pocket",
    "open_pocket",
    "pocket",
    "blind_pocket",
})
WALL_CLASSES = frozenset({"wall"})
SURFACE_CLASSES = frozenset({"contour_surface"})
FILLET_CLASSES = frozenset({"outer_fillet", "fillet"})
PROFILE_CLASSES = frozenset({"profile"})
FACE_CLASSES = frozenset({"flat", "face"})
CHAMFER_CLASSES = frozenset({"chamfer"})

PLANNER_FEATURE_CLASSES = (
    HOLE_CLASSES | POCKET_CLASSES | WALL_CLASSES | SURFACE_CLASSES
    | FILLET_CLASSES | PROFILE_CLASSES | FACE_CLASSES | CHAMFER_CLASSES
)

# --- Canonical operation bank groupings (operation_bank.Operation values) ---
# Roughing / clearing ops -> phase 0; finish ops -> phase 1. Sourced from the bank so
# the planner and the sequence scorer share one classification. helix_bore/facing run
# in the rough phase (the former "bore"/"facing" were rough-phase ops).
ROUGH_OPS = ROUGH_PHASE_OPS
FINISH_OPS = FINISH_PHASE_OPS
# Hole ops that plunge a drill/tap tool along an axis (no open tool sizing).
HOLE_OPS = frozenset({BankOp.DRILL, BankOp.CHIP_BREAK_DRILL, BankOp.THREAD_MILL})
# Ops whose tool is capped at 2*fillet (tool must fit the internal corner radius).
# Rest ops are included so they resolve a smaller-than-prior tool that reaches the fillet.
FILLET_CAP_OPS = frozenset({
    BankOp.CONTOUR_2D,
    BankOp.WATERLINE,
    BankOp.PENCIL,
    BankOp.REST_ROUGHING,
    BankOp.REST_FINISH,
})
# Deep-hole peck threshold: depth/diameter above this -> chip-break drilling.
CHIP_BREAK_DEPTH_RATIO = 4.0
# Rest machining trigger: emit a rest op only when the prior tool leaves at least this
# much uncut radius in a feature's internal corner (prior_tool_radius - fillet_radius).
REST_UNCUT_MARGIN_MM = 0.5

# 3D roughing split: a 3D-surface feature whose steep-face area fraction (slope_profile)
# reaches this is roughed with Z-level area_roughing; below it, adaptive optirough.
# Tunable; UNVERIFIED on real data (no 96260B pocket has 3D_surface=True) -- see
# _rough_operation and test_area_roughing_selected_for_steep_3d_pocket.
AREA_ROUGH_STEEP_FRACTION = 0.5

# Finish-phase tie-break rank (floor -> wall -> surface -> fillet -> bore -> hole).
# Computed per op because contour_2d/waterline serve both floor (pocket) and wall
# roles; the sub-role comes from the source feature class. See _finish_order().
_SURFACE_FINISH_ORDER_OPS = frozenset({
    BankOp.CONSTANT_SCALLOP,
    BankOp.RADIAL_SPIRAL,
    BankOp.STEEP_SHALLOW,
})

_PARAM_TOLERANCE = 1e-3
BORE_MIN_DIA_MM = 0.375 * 25.4
BORE_MAX_DIA_MM = 1.0 * 25.4

# Open-feature sizing: prefer rigid mid-size tools (typical shop 1/4"-1/2"), not catalog minimum.
OPEN_DEFAULT_MIN_DIA_MM = 0.25 * 25.4  # 6.35 mm
OPEN_DEFAULT_MAX_DIA_MM = 0.5 * 25.4  # 12.7 mm

_AXIS_TOLERANCE = 1e-3

# Ops without a hole/bore diameter cap: pick largest fitting tool up to extent/default band.
# = all milling roughing + finishing ops (excludes holes, helix_bore, facing).
OPEN_MILLING_OPS = (ROUGH_OPS | FINISH_OPS) - {BankOp.FACING, BankOp.HELIX_BORE}

# Finish ops where the shop prefers bullnose over plain endmill (floor/wall/fillet).
# constant_scallop (freeform ball finish) is handled by _SURFACE_FINISH_TOOL_TYPES first.
_BULLNOSE_PREFERRED_OPS = frozenset({
    BankOp.RASTER,
    BankOp.CONTOUR_2D,
    BankOp.WATERLINE,
    BankOp.PENCIL,
})

# Tool-type precedence for finishing selection (fit checked before advancing to next type):
#   - surface_finish: ball primary for 3D contour; bullnose before endmill as fallback
#   - floor/wall/fillet finish: bullnose before endmill
#   - roughing / drill / tap / bore: unchanged (single required type)
_SURFACE_FINISH_TOOL_TYPES: tuple[str, ...] = ("ball_endmill", "bullnose_endmill", "endmill")
_FINISHING_TOOL_TYPES: tuple[str, ...] = ("bullnose_endmill", "endmill")
_MILLING_TOOL_TYPES = frozenset({"endmill", "ball_endmill", "bullnose_endmill", "face_mill"})

# Shop 96260B setup-2 facing uses a 1.5" face mill.
FACE_MILL_PREFERRED_DIA_MM = 1.5 * 25.4
FACE_MILL_MIN_DIA_MM = 1.0 * 25.4
FACE_MILL_MAX_DIA_MM = 2.0 * 25.4

# Setups whose stock-boundary flat (envelope STOCK face) maps to facing, not floor_finish.
_FACING_SETUP_IDS = frozenset({"front", "setup_2"})

STOCK_BOUNDARY_SCOPE_TOKENS = frozenset({"facing", "stock_face"})

_SCOPE_CLASS_GROUPS: dict[str, frozenset[str]] = {
    "hole": HOLE_CLASSES,
    "holes": HOLE_CLASSES,
    "pocket": POCKET_CLASSES,
    "pockets": POCKET_CLASSES,
    "wall": WALL_CLASSES,
    "walls": WALL_CLASSES,
    "surface": SURFACE_CLASSES,
    "contour_surface": SURFACE_CLASSES,
    "fillet": FILLET_CLASSES,
    "profile": PROFILE_CLASSES,
    "flat": FACE_CLASSES,
    "face": FACE_CLASSES,
}


@dataclass(frozen=True)
class SetupPlanInput:
    """One feature graph + single-setup context slice for multi-setup planning."""

    feature_graph_path: Path
    context: MachiningContext


@dataclass
class _SetupPlanSlice:
    """Internal per-setup planner output before cross-setup merge."""

    setup: Setup
    grouped_ops: list[OpSpec]
    precedence: dict[str, list[str]]
    plan_tools: list[ToolRef]
    stats: dict[str, Any]
    feature_graph_ref: str


@dataclass(frozen=True)
class PlannerFeature:
    """Internal planner feature; sole adapter output from cascade nodes."""

    feature_id: str
    feature_type: str
    diameter_mm: float | None = None
    depth_mm: float | None = None
    lateral_extent_mm: float | None = None
    fillet_radius_mm: float | None = None
    axis_point: tuple[float, float, float] | None = None
    axis_direction: tuple[float, float, float] | None = None
    is_tapped: bool = False
    is_threaded: bool = False  # non-tap threading (thread-milled); tap wins if both set
    slope_mixed: bool = False  # surface spans both steep and shallow bands (slope_profile)
    steep_fraction: float = 0.0  # area fraction of steep faces (slope_profile), for 3D roughing
    raw_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OpSpec:
    """Internal operation before CamPlan emission."""

    op_id: str = ""
    feature_refs: list[str] = field(default_factory=list)
    feature_type: str = ""
    setup_id: str = ""
    operation: str = ""
    tool_id: str = ""
    tool_type_needed: str = ""
    diameter_mm: float | None = None
    depth_mm: float | None = None
    lateral_extent_mm: float | None = None
    fillet_radius_mm: float | None = None
    access: PocketAccess | None = None
    depends_on: list[str] = field(default_factory=list)
    parameters: MachiningParameters | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


def _vec3(raw: Any) -> tuple[float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _float_param(params: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = params.get(key)
        if raw is not None:
            return float(raw)
    return None


def _lateral_extent_mm(params: Mapping[str, Any]) -> float | None:
    """Smallest positive bbox dimension; conservative cap on tool diameter."""
    dims: list[float] = []
    for key in (
        "bbox_width",
        "bbox_length",
        "bbox_depth",
        "bbox_size_x",
        "bbox_size_y",
        "bbox_size_z",
    ):
        value = _float_param(params, key)
        if value is not None and value > 1e-3:
            dims.append(value)
    return min(dims) if dims else None


def _bounded_hole_diameter(feature_type: str, diameter: float | None) -> float | None:
    """Hole/bore diameter is a real tool-sizing constraint; exterior OD is not."""
    if diameter is None or feature_type not in HOLE_CLASSES:
        return None
    return diameter


def cascade_node_to_feature(node: Mapping[str, Any]) -> PlannerFeature:
    """Map one cascade graph node to a PlannerFeature (boundary adapter)."""
    params = node.get("params") or {}
    feature_id = str(node["feature_id"])
    feature_type = str(node.get("class_name", ""))

    diameter = _float_param(params, "nominal_diameter", "diameter_mm")
    if diameter is None:
        radius = _float_param(params, "radius", "radius_mm")
        if radius is not None:
            diameter = radius * 2.0

    bounded_diameter = _bounded_hole_diameter(feature_type, diameter)
    if feature_type in WALL_CLASSES and diameter is not None:
        # Wall nominal_diameter is exterior OD, not a pocket clearance cap.
        diameter = None

    depth = _float_param(params, "depth", "depth_mm", "depth_below_top_mm")
    lateral_extent = _lateral_extent_mm(params)
    fillet_radius = None
    if feature_type not in HOLE_CLASSES:
        fillet_radius = _float_param(params, "fillet_radius_mm")

    axis = params.get("axis") or {}
    axis_point = _vec3(axis.get("point"))
    axis_direction = _vec3(axis.get("direction"))

    # Tapping is one specific kind of threading -> drill (tap cycle). Any other
    # threading -> thread mill. Tap-specific flags win when both are present.
    is_tapped = any(params.get(key) for key in ("is_tapped", "tapped"))
    is_threaded = (not is_tapped) and any(
        params.get(key) for key in ("threaded", "has_thread", "is_threaded")
    )

    slope = node.get("slope_profile") or {}
    slope_mixed = bool(slope.get("mixed"))
    steep_fraction = float(slope.get("steep_fraction") or 0.0)

    return PlannerFeature(
        feature_id=feature_id,
        feature_type=feature_type,
        diameter_mm=bounded_diameter if bounded_diameter is not None else diameter,
        depth_mm=depth,
        lateral_extent_mm=lateral_extent,
        fillet_radius_mm=fillet_radius,
        axis_point=axis_point,
        axis_direction=axis_direction,
        is_tapped=bool(is_tapped),
        is_threaded=bool(is_threaded),
        slope_mixed=slope_mixed,
        steep_fraction=steep_fraction,
        raw_params=params,
    )


def _feature_face_indices(feature: PlannerFeature) -> frozenset[int]:
    raw = feature.raw_params.get("face_indices") or feature.raw_params.get("face_ids") or []
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(int(i) for i in raw)


def _resolve_envelope_stock_faces(
    graph: Mapping[str, Any],
    context: MachiningContext,
) -> frozenset[int]:
    """Envelope-coincident STOCK face ids from graph cache or STEP classification."""
    cached = graph.get("envelope_stock_face_ids")
    if isinstance(cached, list):
        return frozenset(int(i) for i in cached)

    step_ref = context.setups[0].source_step_file
    if not step_ref:
        return frozenset()

    step_path = Path(step_ref)
    if not step_path.is_file():
        step_path = REPO_ROOT / step_ref
    if not step_path.is_file():
        return frozenset()

    try:
        from stock_cut_classification import envelope_stock_face_ids

        return frozenset(envelope_stock_face_ids(step_path))
    except ImportError:
        logger.warning(
            "setup %s: stock_cut_classification unavailable; "
            "envelope stock faces unknown",
            context.setups[0].setup_id,
        )
        return frozenset()


_REACHABILITY_FRAME_DIRS = frozenset({"+Z", "-Z"})


class SetupApproachAxisError(ValueError):
    """Raised when a setup lacks a descriptor-sourced opening axis for reachability scoping."""


def _resolve_setup_opening_axis_vector(context: MachiningContext) -> tuple[float, float, float]:
    """Return the setup's unit opening-axis vector (descriptor-sourced, fail-loud)."""
    setup = context.setups[0]
    vec = setup.opening_axis_vector
    if vec is None:
        raise SetupApproachAxisError(
            f"setup {setup.setup_id!r}: opening_axis_vector is missing; "
            "build context from the setup descriptor"
        )
    norm = float(sum(v * v for v in vec) ** 0.5)
    if norm <= 1e-12:
        raise SetupApproachAxisError(
            f"setup {setup.setup_id!r}: opening_axis_vector is zero: {vec!r}"
        )
    if abs(norm - 1.0) > 1e-6:
        raise SetupApproachAxisError(
            f"setup {setup.setup_id!r}: opening_axis_vector is not unit length "
            f"(norm={norm:.6f}): {vec!r}"
        )
    if vec in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) and setup.opening_axis in _REACHABILITY_FRAME_DIRS:
        raise SetupApproachAxisError(
            f"setup {setup.setup_id!r}: opening_axis_vector looks like a reachability "
            f"frame literal, not a descriptor axis: {vec!r}"
        )
    return vec


def _primary_setup_approach_dir(context: MachiningContext) -> str:
    """Discrete opening-axis label from the setup descriptor (e.g. '+Y', not '+Z')."""
    return vector_to_opening_axis_label(_resolve_setup_opening_axis_vector(context))


def _reachability_dir_for_setup(context: MachiningContext) -> str:
    """Map descriptor opening axis + machining_side to a reachability frame token.

    Reachability annotates along approach_frame Z = opening_axis:
      ``+Z`` = along +opening_axis, ``-Z`` = along -opening_axis.
    Front setups approach from +opening_axis; back setups from -opening_axis.
    """
    setup = context.setups[0]
    _resolve_setup_opening_axis_vector(context)
    side = setup.machining_side
    if side not in ("front", "back"):
        raise SetupApproachAxisError(
            f"setup {setup.setup_id!r}: machining_side must be 'front' or 'back' "
            f"for reachability scoping, got {side!r}"
        )
    return "+Z" if side == "front" else "-Z"


def _graph_has_verified_reachability(graph: Mapping[str, Any]) -> bool:
    for node in graph.get("nodes", []):
        approach = node.get("approach") or {}
        reach = approach.get("reachability")
        if isinstance(reach, Mapping) and reach.get("verified"):
            return True
    return False


def _feature_reachable_for_setup(
    node: Mapping[str, Any],
    setup_approach_dir: str,
) -> bool:
    """True when step-4a reachability includes this setup's approach direction."""
    approach = node.get("approach") or {}
    reach = approach.get("reachability")
    if not isinstance(reach, Mapping):
        return False

    if reach.get("exempt"):
        dirs = reach.get("reachable_dirs") or []
        return bool(dirs)

    reachable_dirs = reach.get("reachable_dirs") or []
    return setup_approach_dir in reachable_dirs


def _feature_in_setup_scope(
    feature: PlannerFeature,
    scope: SetupScopeSpec,
    envelope_faces: frozenset[int],
) -> bool:
    if scope.is_full:
        return True

    if scope.feature_ids and feature.feature_id in scope.feature_ids:
        return True

    if not scope.classes:
        return False

    if scope.stock_boundary_only:
        if feature.feature_type not in FACE_CLASSES:
            return False
        faces = _feature_face_indices(feature)
        if not faces or not envelope_faces:
            return False
        return faces.issubset(envelope_faces)

    allowed_types: set[str] = set()
    for token in scope.classes:
        group = _SCOPE_CLASS_GROUPS.get(token)
        if group is not None:
            allowed_types |= set(group)
        else:
            allowed_types.add(token)
    return feature.feature_type in allowed_types


def filter_features_for_setup_by_class(
    features: Sequence[PlannerFeature],
    context: MachiningContext,
    *,
    envelope_faces: frozenset[int],
) -> tuple[list[PlannerFeature], int, dict[str, Any]]:
    """Drop features outside the setup's declared class scope (legacy filter)."""
    setup = context.setups[0]
    scope = setup.scope
    if scope.is_full:
        return list(features), 0, {
            "scope_mode": "full",
            "scope_classes": [],
        }

    kept: list[PlannerFeature] = []
    dropped = 0
    for feat in features:
        if _feature_in_setup_scope(feat, scope, envelope_faces):
            kept.append(feat)
        else:
            dropped += 1
            logger.info(
                "setup %s: dropped out-of-scope feature_id=%s class_name=%s",
                setup.setup_id,
                feat.feature_id,
                feat.feature_type,
            )

    logger.info(
        "setup %s: %d features out of class scope, kept %d",
        setup.setup_id,
        dropped,
        len(kept),
    )
    return kept, dropped, {
        "scope_mode": "filtered",
        "scope_classes": list(scope.classes),
        "scope_feature_ids": list(scope.feature_ids),
        "envelope_stock_faces": sorted(envelope_faces),
    }


def filter_features_for_setup_by_reachability(
    features: Sequence[PlannerFeature],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, Any],
    context: MachiningContext,
) -> tuple[list[PlannerFeature], int, dict[str, Any]]:
    """Keep features verified reachable from this setup's approach direction."""
    setup = context.setups[0]
    opening_axis_label = _primary_setup_approach_dir(context)
    opening_axis_vector = _resolve_setup_opening_axis_vector(context)
    reachability_dir = _reachability_dir_for_setup(context)
    kept: list[PlannerFeature] = []
    dropped = 0
    missing_reachability = 0

    for feat in features:
        node = nodes_by_id.get(feat.feature_id)
        if node is None:
            dropped += 1
            continue
        approach = node.get("approach") or {}
        if approach.get("reachability") is None:
            missing_reachability += 1
            dropped += 1
            logger.warning(
                "setup %s: feature_id=%s lacks verified reachability; dropped",
                setup.setup_id,
                feat.feature_id,
            )
            continue
        if _feature_reachable_for_setup(node, reachability_dir):
            kept.append(feat)
        else:
            dropped += 1
            logger.info(
                "setup %s: dropped unreachable feature_id=%s class_name=%s",
                setup.setup_id,
                feat.feature_id,
                feat.feature_type,
            )

    if missing_reachability:
        logger.warning(
            "setup %s: %d features missing reachability (export cascade with step 4a)",
            setup.setup_id,
            missing_reachability,
        )

    logger.info(
        "setup %s: %d features unreachable, kept %d "
        "(opening_axis=%s reachability_dir=%s)",
        setup.setup_id,
        dropped,
        len(kept),
        opening_axis_label,
        reachability_dir,
    )
    return kept, dropped, {
        "scope_mode": "reachability",
        "opening_axis": opening_axis_label,
        "opening_axis_vector": list(opening_axis_vector),
        "reachability_dir": reachability_dir,
        "setup_approach_dir": opening_axis_label,
        "missing_reachability": missing_reachability,
    }


def filter_features_for_setup(
    features: Sequence[PlannerFeature],
    context: MachiningContext,
    *,
    envelope_faces: frozenset[int],
    nodes_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    graph: Mapping[str, Any] | None = None,
    use_reachability: bool = True,
) -> tuple[list[PlannerFeature], int, dict[str, Any]]:
    """Drop features outside the setup before planning."""
    if use_reachability and graph is not None and nodes_by_id is not None:
        if _graph_has_verified_reachability(graph):
            return filter_features_for_setup_by_reachability(
                features, nodes_by_id, graph, context,
            )
        logger.warning(
            "setup %s: no verified reachability on graph; falling back to class scope",
            context.setups[0].setup_id,
        )
    return filter_features_for_setup_by_class(
        features, context, envelope_faces=envelope_faces,
    )


def _assigned_feature_ids(
    features: Sequence[PlannerFeature],
    context: MachiningContext,
    *,
    envelope_faces: frozenset[int],
    nodes_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    graph: Mapping[str, Any] | None = None,
    use_reachability: bool,
) -> set[str]:
    kept, _, _ = filter_features_for_setup(
        features,
        context,
        envelope_faces=envelope_faces,
        nodes_by_id=nodes_by_id,
        graph=graph,
        use_reachability=use_reachability,
    )
    return {f.feature_id for f in kept}


def print_scope_assignment_diff(
    *,
    setup_id: str,
    features: Sequence[PlannerFeature],
    context: MachiningContext,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    graph: Mapping[str, Any],
    envelope_faces: frozenset[int],
) -> list[str]:
    """Print class-scope vs reachability assignment; return feature ids lost in new mode."""
    class_ids = _assigned_feature_ids(
        features,
        context,
        envelope_faces=envelope_faces,
        use_reachability=False,
    )
    reach_ids = _assigned_feature_ids(
        features,
        context,
        envelope_faces=envelope_faces,
        nodes_by_id=nodes_by_id,
        graph=graph,
        use_reachability=True,
    )

    print(f"\n=== setup {setup_id}: scope assignment diff ===")
    opening_axis = _primary_setup_approach_dir(context)
    reach_dir = _reachability_dir_for_setup(context)
    axis_vec = _resolve_setup_opening_axis_vector(context)
    print(f"  opening axis (descriptor): {opening_axis}  vector={list(axis_vec)}")
    print(f"  reachability filter dir:   {reach_dir}  (+/- along opening axis)")
    print(f"  class filter:        {len(class_ids)} features")
    print(f"  reachability filter: {len(reach_ids)} features")
    only_class = sorted(class_ids - reach_ids, key=lambda x: (len(x), x))
    only_reach = sorted(reach_ids - class_ids, key=lambda x: (len(x), x))
    if only_class:
        print(f"  class-only (dropped by reachability): {only_class}")
    if only_reach:
        print(f"  reachability-only (added vs class):   {only_reach}")
    if not only_class and not only_reach:
        print("  (identical assignment)")

    lost_all = sorted(
        {f.feature_id for f in features} - reach_ids,
        key=lambda x: (len(x), x),
    )
    if lost_all:
        print(f"  WARNING reachability gap (no setup assignment): {lost_all}")
    return lost_all


def filter_planner_features(
    features: Sequence[PlannerFeature],
) -> tuple[list[PlannerFeature], int]:
    """Keep machinable feature families; log dropped recognizer classes."""
    kept: list[PlannerFeature] = []
    dropped = 0
    for feat in features:
        if feat.feature_type in PLANNER_FEATURE_CLASSES:
            kept.append(feat)
        else:
            dropped += 1
            logger.info(
                "dropped feature_id=%s class_name=%s (not in planner slice)",
                feat.feature_id,
                feat.feature_type,
            )
    return kept, dropped


def _axis_key(feat: PlannerFeature) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if feat.axis_point is None or feat.axis_direction is None:
        return None
    point = tuple(round(v, 3) for v in feat.axis_point)
    direction = tuple(round(v, 3) for v in feat.axis_direction)
    norm = sum(v * v for v in direction) ** 0.5
    if norm <= _AXIS_TOLERANCE:
        return None
    direction = tuple(v / norm for v in direction)
    return point, direction


def group_coaxial_holes(holes: Sequence[PlannerFeature]) -> list[list[PlannerFeature]]:
    """Group hole features sharing the same axis into coaxial stacks."""
    buckets: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[PlannerFeature]] = {}
    singletons: list[list[PlannerFeature]] = []

    for hole in holes:
        key = _axis_key(hole)
        if key is None:
            singletons.append([hole])
            continue
        buckets.setdefault(key, []).append(hole)

    groups = [sorted(group, key=lambda f: f.feature_id) for group in buckets.values()]
    groups.extend(singletons)
    groups.sort(key=lambda g: g[0].feature_id)
    return groups


def _open_feature_geometry(feature: PlannerFeature) -> tuple[float | None, float | None]:
    return feature.lateral_extent_mm, feature.fillet_radius_mm


def _pocket_access_label(feature_id: str, context: MachiningContext) -> PocketAccess:
    setup = context.setups[0]
    raw = setup.pocket_access.get(feature_id, "unknown")
    try:
        return PocketAccess(str(raw).lower())
    except ValueError:
        return PocketAccess.UNKNOWN


def _flat_area_mm(feature: PlannerFeature) -> float:
    return _float_param(feature.raw_params, "area") or 0.0


def identify_facing_feature_ids(
    features: Sequence[PlannerFeature],
    setup_id: str,
    *,
    envelope_faces: frozenset[int] | None = None,
) -> frozenset[str]:
    """Return stock-boundary flat feature_ids that map to facing (face_mill).

    Prefers envelope-coincident STOCK faces from the classifier over area-only
    heuristics when envelope_faces is supplied.
    """
    if setup_id not in _FACING_SETUP_IDS:
        return frozenset()
    flats = [feat for feat in features if feat.feature_type in FACE_CLASSES]
    if not flats:
        return frozenset()
    if envelope_faces:
        envelope_flats = [
            feat
            for feat in flats
            if _feature_face_indices(feat)
            and _feature_face_indices(feat).issubset(envelope_faces)
        ]
        if envelope_flats:
            flats = envelope_flats
    best = max(flats, key=_flat_area_mm)
    if _flat_area_mm(best) <= 0.0:
        return frozenset()
    return frozenset({best.feature_id})


def _is_3d_surface(feature: PlannerFeature) -> bool:
    """True when the feature's floor/walls are sculpted (needs a 3D toolpath).

    Sourced from the cascade ``3D_surface`` flag; absent -> prismatic (2.5D).
    """
    return bool(feature.raw_params.get("3D_surface"))


_REVOLVED_SURFACE_TYPES = ("cone", "torus", "sphere", "cylinder")


def _is_axisymmetric_surface(feature: PlannerFeature) -> bool:
    """True for a body-of-revolution surface (round boss/pocket/dome) -> radial_spiral.

    Uses the cascade ``n_distinct_axes`` (1 = faces share a single axis) plus a
    ``surface_type_histogram`` dominated by surfaces of revolution, so incidental
    single-axis bspline blends stay freeform (constant_scallop).
    """
    params = feature.raw_params
    if int(params.get("n_distinct_axes") or 0) != 1:
        return False
    hist = params.get("surface_type_histogram") or {}
    total = sum(hist.values())
    if total <= 0:
        return False
    revolved = sum(hist.get(k, 0) for k in _REVOLVED_SURFACE_TYPES)
    return revolved * 2 >= total  # revolved surfaces are the majority


def _surface_finish_operation(feature: PlannerFeature) -> BankOp:
    """Pick a 3D surface-finish op:

    - spans both steep and shallow bands -> steep_shallow (waterline+raster split);
    - else round/axisymmetric (uniform revolution) -> radial_spiral;
    - else freeform -> constant_scallop.

    Mixed slope wins over roundness: a surface that spans both bands needs the split
    even when it is a body of revolution (a single radial pass smears the finish).
    """
    if feature.slope_mixed:
        return BankOp.STEEP_SHALLOW
    if _is_axisymmetric_surface(feature):
        return BankOp.RADIAL_SPIRAL
    return BankOp.CONSTANT_SCALLOP


def _deburr_op(feature_refs: Sequence[str], setup_id: str) -> OpSpec:
    """Whole-setup deburr: auto-breaks all model edges accessible in this setup.

    Not tied to one recognizer feature -- references every in-scope feature and needs
    a chamfer/deburr tool (UNRESOLVED until such a tool exists in the library). Sorts
    last via _finish_order (auxiliary, rank 99).
    """
    return OpSpec(
        feature_refs=list(feature_refs),
        feature_type="part",
        setup_id=setup_id,
        operation=BankOp.DEBURR,
        tool_type_needed="chamfer_mill",
        access=None,
    )


def _resolve_engrave_target(
    spec: Mapping[str, Any],
    features: Sequence[PlannerFeature],
    facing_feature_ids: frozenset[str],
) -> str | None:
    """Resolve a declared engrave target to an in-scope feature_id, or None.

    `target.feature_id` -> that feature if present; `target.datum` -> the setup's
    datum flat (the same face the facing op uses). Never fabricates a reference.
    """
    target = spec.get("target") or {}
    fid = target.get("feature_id")
    if fid is not None:
        return str(fid) if str(fid) in {f.feature_id for f in features} else None
    if target.get("datum") is not None:
        return next(iter(sorted(facing_feature_ids)), None)
    return None


def _engraving_ops(
    context: MachiningContext,
    features: Sequence[PlannerFeature],
    facing_feature_ids: frozenset[str],
    setup_id: str,
) -> tuple[list[OpSpec], list[Mapping[str, Any]]]:
    """Build ENGRAVING ops from declared specs; return (ops, unresolved_specs).

    Declared (explicit process input), never inferred from geometry. Text/depth flow
    straight to the CAM op via attributes; source is tagged explicit_spec. A spec whose
    target can't be resolved is NOT fabricated into an op -- it's returned for a flag.
    """
    ops: list[OpSpec] = []
    unresolved: list[Mapping[str, Any]] = []
    for spec in (context.setups[0].engrave or []):
        target = _resolve_engrave_target(spec, features, facing_feature_ids)
        if target is None:
            unresolved.append(spec)
            continue
        ops.append(OpSpec(
            feature_refs=[target],
            feature_type="engraving",
            setup_id=setup_id,
            operation=BankOp.ENGRAVING,
            tool_type_needed="engraver",
            attributes={
                "text": spec.get("text"),
                "depth_mm": spec.get("depth_mm"),
                "source": "explicit_spec",
            },
            access=None,
        ))
    return ops, unresolved


def _op_tool_radius_mm(op: OpSpec, tool_lookup: Mapping[str, Tool]) -> float | None:
    """Radius of the tool assigned to ``op`` (None if unresolved)."""
    tool = tool_lookup.get(op.tool_id)
    if tool is None or not tool.diameter_mm:
        return None
    return tool.diameter_mm / 2.0


def _rest_op(base: OpSpec, operation: BankOp, fillet_radius_mm: float) -> OpSpec:
    """A rest-machining op cloning a base op's geometry but capped to the fillet.

    Carries the fillet radius so FILLET_CAP_OPS sizing resolves a tool that reaches
    the corner (necessarily smaller than the prior op's tool, which triggered it).
    The leftover-stock shape and toolpath are Mastercam's rest-strategy job at encode
    time; this op only declares that a rest pass is needed here.
    """
    return OpSpec(
        feature_refs=list(base.feature_refs),
        feature_type=base.feature_type,
        setup_id=base.setup_id,
        operation=operation,
        tool_type_needed="endmill",
        depth_mm=base.depth_mm,
        lateral_extent_mm=base.lateral_extent_mm,
        fillet_radius_mm=fillet_radius_mm,
        access=base.access,
    )


def _rest_machining_ops(
    op_specs: Sequence[OpSpec],
    tool_lookup: Mapping[str, Tool],
) -> list[OpSpec]:
    """Per-feature rest-machining trigger (Track C, radius heuristic, no stock model).

    For each feature, compare the tool actually assigned to its rough/finish op against
    the feature's tightest internal fillet radius. When a tool leaves more than
    REST_UNCUT_MARGIN_MM of uncut radius in the corner, emit a rest op with a
    fillet-capped (smaller) tool:
      * roughing tool too big + a finish follows -> rest_roughing (between them);
      * finish tool still too big               -> rest_finish (after it).
    No leftover-stock geometry is computed -- Mastercam rest-machines from the tool
    sequence at encode time.
    """
    by_feature: dict[tuple[str, ...], list[OpSpec]] = {}
    for op in op_specs:
        by_feature.setdefault(tuple(op.feature_refs), []).append(op)

    rest_ops: list[OpSpec] = []
    for ops in by_feature.values():
        fillets = [o.fillet_radius_mm for o in ops if o.fillet_radius_mm]
        if not fillets:
            continue
        fillet = min(fillets)  # tightest corner limits the tool

        rough = next(
            (o for o in ops if o.operation in ROUGH_OPS and o.tool_type_needed == "endmill"),
            None,
        )
        finish = next(
            (o for o in ops if o.operation in (BankOp.CONTOUR_2D, BankOp.WATERLINE)),
            None,
        )

        if rough is not None and finish is not None:
            r = _op_tool_radius_mm(rough, tool_lookup)
            if r is not None and r - fillet > REST_UNCUT_MARGIN_MM:
                rest_ops.append(_rest_op(rough, BankOp.REST_ROUGHING, fillet))

        if finish is not None:
            r = _op_tool_radius_mm(finish, tool_lookup)
            if r is not None and r - fillet > REST_UNCUT_MARGIN_MM:
                rest_ops.append(_rest_op(finish, BankOp.REST_FINISH, fillet))

    return rest_ops


def _rough_operation(feature: PlannerFeature, access: PocketAccess) -> BankOp:
    """Geometry-driven roughing/clearing op for a pocket-class feature.

    3D sculpted -> steep-dominated content gets Z-level area_roughing, shallow/adaptive
    content gets optirough (steep_fraction from the slope pass); prismatic closed region
    -> pocket; prismatic open region -> 2D dynamic mill.

    NOTE: the 3D branch is UNVERIFIED on real data -- no 96260B pocket has 3D_surface=True
    (the only 3D_surface features are contour_surfaces, which are finished not roughed).
    Covered only by a synthetic fixture (test_area_roughing_selected_for_steep_3d_pocket)
    until a part with real freeform-pocket content exists.
    """
    if _is_3d_surface(feature):
        if feature.steep_fraction >= AREA_ROUGH_STEEP_FRACTION:
            return BankOp.AREA_ROUGHING
        return BankOp.OPTIROUGH
    if access == PocketAccess.CLOSED:
        return BankOp.POCKET
    return BankOp.DYNAMIC_MILL_2D


def _wall_contour_operation(feature: PlannerFeature) -> BankOp:
    """Geometry-driven wall/contour finish: waterline for 3D, 2D contour otherwise."""
    return BankOp.WATERLINE if _is_3d_surface(feature) else BankOp.CONTOUR_2D


def _drill_operation(depth_mm: float | None, diameter_mm: float | None) -> BankOp:
    """Plain drill vs chip-break drill by depth/diameter ratio."""
    if (
        depth_mm is not None
        and diameter_mm is not None
        and diameter_mm > 0
        and depth_mm / diameter_mm > CHIP_BREAK_DEPTH_RATIO
    ):
        return BankOp.CHIP_BREAK_DRILL
    return BankOp.DRILL


def _needs_bore_instead_of_drill(
    op_spec: OpSpec,
    tools: Sequence[Tool],
    material: str | None = None,
) -> bool:
    """True when no drill fits the hole (large diameter / depth)."""
    if op_spec.tool_type_needed != "drill":
        return False
    return select_tool(op_spec, tools, material=material) == UNRESOLVED_TOOL_ID


def map_feature_to_operations(
    feature: PlannerFeature | Sequence[PlannerFeature],
    context: MachiningContext,
    *,
    facing_feature_ids: frozenset[str] = frozenset(),
) -> list[OpSpec]:
    """Geometry-driven mapping from feature(s) to internal OpSpec list.

    Every emitted OpSpec.operation is a value from the canonical bank
    (operation_bank.Operation). Roughing/finishing op choice is 3D-vs-2.5D driven
    (see _rough_operation / _wall_contour_operation); hole cycle (peck vs tap) is
    carried by tool_type_needed, not the operation name.
    """
    setup_id = context.setups[0].setup_id

    if isinstance(feature, PlannerFeature):
        features = [feature]
    else:
        features = list(feature)

    if not features:
        return []

    primary = features[0]
    if primary.feature_type in HOLE_CLASSES:
        refs = [f.feature_id for f in features]
        feature_type = primary.feature_type
        for f in features:
            if f.feature_type == "through_hole":
                feature_type = "through_hole"
                break

        max_diameter = max((f.diameter_mm or 0.0) for f in features)
        max_depth = max((f.depth_mm or 0.0) for f in features)
        tapped = any(f.is_tapped for f in features)
        thread_milled = (not tapped) and any(f.is_threaded for f in features)
        dia = max_diameter if max_diameter > 0 else None
        depth = max_depth if max_depth > 0 else None

        drill_spec = OpSpec(
            feature_refs=refs,
            feature_type=feature_type,
            setup_id=setup_id,
            operation=_drill_operation(depth, dia),
            tool_type_needed="drill",
            diameter_mm=dia,
            depth_mm=depth,
            access=None,
        )
        if _needs_bore_instead_of_drill(drill_spec, context.tools, material=context.material):
            ops = [
                OpSpec(
                    feature_refs=refs,
                    feature_type=feature_type,
                    setup_id=setup_id,
                    operation=BankOp.HELIX_BORE,
                    tool_type_needed="endmill",
                    diameter_mm=dia,
                    depth_mm=depth,
                    access=None,
                ),
            ]
        else:
            ops = [drill_spec]

        if tapped:
            # Tapping is a drill-op cycle (per the bank); the tap tool distinguishes
            # it from the pilot drill via tool_type_needed.
            ops.append(
                OpSpec(
                    feature_refs=refs,
                    feature_type=feature_type,
                    setup_id=setup_id,
                    operation=BankOp.DRILL,
                    tool_type_needed="tap",
                    diameter_mm=dia,
                    depth_mm=depth,
                    access=None,
                ),
            )
        elif thread_milled:
            # Any non-tap threading is interpolated with a thread mill after the
            # pilot drill/bore.
            ops.append(
                OpSpec(
                    feature_refs=refs,
                    feature_type=feature_type,
                    setup_id=setup_id,
                    operation=BankOp.THREAD_MILL,
                    tool_type_needed="thread_mill",
                    diameter_mm=dia,
                    depth_mm=depth,
                    access=None,
                ),
            )
        return ops

    if primary.feature_type in POCKET_CLASSES:
        access = _pocket_access_label(primary.feature_id, context)
        depth = primary.depth_mm
        lateral, fillet = _open_feature_geometry(primary)
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=_rough_operation(primary, access),
                tool_type_needed="endmill",
                depth_mm=depth,
                lateral_extent_mm=lateral,
                fillet_radius_mm=fillet,
                access=access,
            ),
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=_wall_contour_operation(primary),
                tool_type_needed="endmill",
                depth_mm=depth,
                lateral_extent_mm=lateral,
                fillet_radius_mm=fillet,
                access=access,
            ),
        ]

    if primary.feature_type in WALL_CLASSES:
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=_wall_contour_operation(primary),
                tool_type_needed="endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=primary.lateral_extent_mm,
                fillet_radius_mm=primary.fillet_radius_mm,
                access=None,
            ),
        ]

    if primary.feature_type in SURFACE_CLASSES:
        lateral, _ = _open_feature_geometry(primary)
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=_surface_finish_operation(primary),
                tool_type_needed="ball_endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=lateral,
                access=None,
            ),
        ]

    if primary.feature_type in FILLET_CLASSES:
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=BankOp.PENCIL,
                tool_type_needed="endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=primary.lateral_extent_mm,
                fillet_radius_mm=primary.fillet_radius_mm,
                access=None,
            ),
        ]

    if primary.feature_type in PROFILE_CLASSES:
        lateral, fillet = _open_feature_geometry(primary)
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=_rough_operation(primary, PocketAccess.OPEN),
                tool_type_needed="endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=lateral,
                fillet_radius_mm=fillet,
                access=None,
            ),
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=_wall_contour_operation(primary),
                tool_type_needed="endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=lateral,
                fillet_radius_mm=fillet,
                access=None,
            ),
        ]

    if primary.feature_type in FACE_CLASSES:
        lateral, _ = _open_feature_geometry(primary)
        if primary.feature_id in facing_feature_ids:
            return [
                OpSpec(
                    feature_refs=[primary.feature_id],
                    feature_type=primary.feature_type,
                    setup_id=setup_id,
                    operation=BankOp.FACING,
                    tool_type_needed="face_mill",
                    depth_mm=primary.depth_mm,
                    lateral_extent_mm=lateral,
                    access=None,
                ),
            ]
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=BankOp.RASTER,
                tool_type_needed="endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=lateral,
                access=None,
            ),
        ]

    if primary.feature_type in CHAMFER_CLASSES:
        # Edge-break: cut the bevel with a chamfer mill (UNRESOLVED until such a tool
        # exists in the library, like deburr). Sequenced late (auxiliary).
        return [
            OpSpec(
                feature_refs=[primary.feature_id],
                feature_type=primary.feature_type,
                setup_id=setup_id,
                operation=BankOp.CHAMFER,
                tool_type_needed="chamfer_mill",
                lateral_extent_mm=_float_param(primary.raw_params, "chamfer_size_mm"),
                access=None,
            ),
        ]

    return []


def _uses_default_open_band(op_spec: OpSpec) -> bool:
    """True when no geometry-derived diameter cap is available."""
    if op_spec.lateral_extent_mm is not None:
        return False
    if (
        op_spec.fillet_radius_mm is not None
        and op_spec.operation in FILLET_CAP_OPS
    ):
        return False
    return op_spec.operation in OPEN_MILLING_OPS


def _max_open_tool_diameter_mm(op_spec: OpSpec) -> float | None:
    """Upper bound on tool diameter for open milling ops."""
    if op_spec.operation not in OPEN_MILLING_OPS:
        return None

    caps: list[float] = []
    if op_spec.lateral_extent_mm is not None:
        caps.append(op_spec.lateral_extent_mm)
    if (
        op_spec.fillet_radius_mm is not None
        and op_spec.operation in FILLET_CAP_OPS
    ):
        caps.append(2.0 * op_spec.fillet_radius_mm)

    if caps:
        return min(caps)
    return OPEN_DEFAULT_MAX_DIA_MM


def _tool_fits_op(tool: Tool, op_spec: OpSpec) -> bool:
    """True when ``tool`` can cut ``op_spec`` (depth reach + clearance when known)."""
    if op_spec.tool_type_needed == "drill":
        if op_spec.diameter_mm is not None and tool.diameter_mm + 1e-6 < op_spec.diameter_mm:
            return False
        if op_spec.depth_mm is not None and tool.max_depth_mm is not None:
            if tool.max_depth_mm + 1e-6 < op_spec.depth_mm:
                return False
        return True

    if tool.tool_type in _MILLING_TOOL_TYPES or op_spec.tool_type_needed in _MILLING_TOOL_TYPES:
        if op_spec.depth_mm is not None and tool.flute_length_mm is not None:
            if tool.flute_length_mm + 1e-6 < op_spec.depth_mm:
                return False
        if op_spec.operation == BankOp.HELIX_BORE and op_spec.diameter_mm is not None:
            if tool.diameter_mm > op_spec.diameter_mm + 1e-6:
                return False
        max_dia = _max_open_tool_diameter_mm(op_spec)
        if max_dia is not None and tool.diameter_mm > max_dia + 1e-6:
            return False
        return True

    return True


def _tool_type_precedence(op_spec: OpSpec) -> tuple[str, ...]:
    """Return tool types to try in order for ``op_spec`` (fit before type preference)."""
    if op_spec.operation in _SURFACE_FINISH_ORDER_OPS:
        return _SURFACE_FINISH_TOOL_TYPES
    if op_spec.operation in _BULLNOSE_PREFERRED_OPS:
        return _FINISHING_TOOL_TYPES
    return (op_spec.tool_type_needed,)


def _source_library_name(tool: Tool) -> str:
    """Extract catalog library name from tool provenance."""
    source = tool.source or ""
    if source.startswith("supabase:"):
        return source.removeprefix("supabase:")
    if source.startswith("fusion_library:"):
        return source.removeprefix("fusion_library:")
    if "::" in tool.tool_id:
        return tool.tool_id.split("::", 1)[0]
    return ""


_ISO_MATERIAL_CATEGORY_PREFIXES = frozenset({"n", "p", "m", "k", "s", "h"})


def _is_iso_material_category_preset(preset_name: str) -> bool:
    """True for Fusion ISO material-group presets (``N - Aluminum...``, ``P - ...``, etc.)."""
    name = preset_name.strip()
    if len(name) < 4 or name[1:3] != " -":
        return False
    return name[0].lower() in _ISO_MATERIAL_CATEGORY_PREFIXES


_MATERIAL_PRESET_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "aluminum": ("aluwrought",),
    "steel": ("lowcsteel",),
    "stainless": ("stainlesssteel",),
    "cast_iron": ("castiron",),
    "titanium": ("titanium",),
}

_WRONG_MATERIAL_PRESET_HINTS: dict[str, tuple[str, ...]] = {
    "aluminum": ("lowcsteel", "stainlesssteel", "castiron"),
    "steel": ("aluwrought",),
    "stainless": ("aluwrought", "lowcsteel"),
}

_LIBRARY_MATERIAL_HINTS: dict[str, tuple[str, ...]] = {
    "aluminum": ("non_ferrous", "aluminum"),
    "steel": ("ferrous", "steel"),
    "stainless": ("stainless",),
}


def _tool_material_rank(tool: Tool, material: str | None) -> int:
    """Rank how well a tool matches the workpiece material (lower is better).

    0 = carries workpiece-material presets or an appropriate library
    1 = neutral (generic ISO / ``all`` presets only)
    2 = predominantly wrong-material presets (e.g. LowCSteel_* for aluminum)
    """
    if material is None:
        return 0

    target = material.strip().lower()
    if any(p.preset_material == target for p in tool.presets):
        return 0

    name_hints = _MATERIAL_PRESET_NAME_HINTS.get(target, ())
    wrong_hints = _WRONG_MATERIAL_PRESET_HINTS.get(target, ())
    lib_name = _source_library_name(tool).lower()
    lib_hints = _LIBRARY_MATERIAL_HINTS.get(target, ())

    has_good_preset = any(
        any(hint in preset.preset_name.lower() for hint in name_hints)
        and not _is_iso_material_category_preset(preset.preset_name)
        for preset in tool.presets
    )
    has_wrong_preset = any(
        any(hint in preset.preset_name.lower() for hint in wrong_hints)
        for preset in tool.presets
    )
    lib_matches = any(hint in lib_name for hint in lib_hints)
    lib_mismatch = (
        target == "aluminum"
        and "ferrous" in lib_name
        and "non_ferrous" not in lib_name
    ) or (
        target == "steel"
        and "non_ferrous" in lib_name
    )

    if has_good_preset or lib_matches:
        return 0
    if lib_mismatch or (has_wrong_preset and not has_good_preset):
        return 2
    return 1


def _material_selection_score(tool: Tool, material: str | None) -> int:
    """Higher score = stronger material affinity (for max() tie-breaks)."""
    return 2 - _tool_material_rank(tool, material)


def _open_tool_sort_key(tool: Tool, material: str | None) -> tuple[float, float, float]:
    """Sort key for open-feature tool pick (higher = preferred)."""
    return (
        float(_material_selection_score(tool, material)),
        tool.diameter_mm,
        tool.flute_length_mm or 0.0,
    )


def _select_largest_open_tool(
    fitting: Sequence[Tool],
    op_spec: OpSpec,
    material: str | None = None,
) -> Tool | None:
    """Pick the largest fitting tool for open features (bounded -> largest within cap).

    When geometry is missing, prefer the default shop band (1/4"-1/2") instead of
    the catalog minimum.
    """
    pool = list(fitting)
    if not pool:
        return None

    if _uses_default_open_band(op_spec):
        in_band = [
            tool
            for tool in pool
            if OPEN_DEFAULT_MIN_DIA_MM - 1e-6 <= tool.diameter_mm <= OPEN_DEFAULT_MAX_DIA_MM + 1e-6
        ]
        if in_band:
            pool = in_band

    return max(pool, key=lambda item: _open_tool_sort_key(item, material))


def _select_facing_tool_id(
    op_spec: OpSpec,
    tools: Sequence[Tool],
    material: str | None = None,
) -> str | None:
    """Pick a face mill near the shop 1.5" facing diameter."""
    candidates = [tool for tool in tools if tool.tool_type == "face_mill"]
    if not candidates:
        return None
    fitting = [tool for tool in candidates if _tool_fits_op(tool, op_spec)]
    if not fitting:
        return None
    in_band = [
        tool
        for tool in fitting
        if FACE_MILL_MIN_DIA_MM - 1e-6 <= tool.diameter_mm <= FACE_MILL_MAX_DIA_MM + 1e-6
    ]
    pool = in_band if in_band else fitting
    return min(
        pool,
        key=lambda tool: (
            abs(tool.diameter_mm - FACE_MILL_PREFERRED_DIA_MM),
            -_material_selection_score(tool, material),
            -tool.diameter_mm,
        ),
    ).tool_id


def _select_fitting_tool_id(
    tool_type: str,
    op_spec: OpSpec,
    tools: Sequence[Tool],
    material: str | None = None,
) -> str | None:
    """Pick a fitting tool of ``tool_type`` using bounded vs open sizing rules."""
    if op_spec.operation == BankOp.FACING and tool_type == "face_mill":
        return _select_facing_tool_id(op_spec, tools, material=material)

    candidates = [t for t in tools if t.tool_type == tool_type]
    if not candidates:
        return None

    fitting = [tool for tool in candidates if _tool_fits_op(tool, op_spec)]
    if not fitting:
        return None

    if op_spec.operation == BankOp.HELIX_BORE:
        in_range = [
            tool
            for tool in fitting
            if BORE_MIN_DIA_MM - 1e-6 <= tool.diameter_mm <= BORE_MAX_DIA_MM + 1e-6
        ]
        pool = in_range if in_range else fitting
        return max(
            pool,
            key=lambda item: _open_tool_sort_key(item, material),
        ).tool_id

    if op_spec.tool_type_needed in ("drill", "tap"):
        fitting.sort(key=lambda t: (_tool_material_rank(t, material), t.diameter_mm))
        return fitting[0].tool_id

    if op_spec.operation in OPEN_MILLING_OPS:
        chosen = _select_largest_open_tool(fitting, op_spec, material=material)
        return chosen.tool_id if chosen is not None else None

    fitting.sort(key=lambda t: (_tool_material_rank(t, material), t.diameter_mm))
    return fitting[0].tool_id


def select_tool(
    op_spec: OpSpec,
    tools: Sequence[Tool],
    material: str | None = None,
) -> str:
    """Pick a fitting tool; UNRESOLVED if none fit across the precedence list.

    Sizing rules (fit checked before advancing to the next tool type):
      - Bounded holes/drills/taps: smallest tool whose diameter meets the hole floor.
      - Bounded bores: largest endmill in the shop helical-bore band that fits the hole.
      - Open milling (pockets, floors, walls, surfaces, fillets): largest fitting tool up
        to lateral extent / fillet-radius caps, or the default 1/4"-1/2" band when geometry
        is missing. Never fall back to the catalog minimum for open features.
    """
    for tool_type in _tool_type_precedence(op_spec):
        chosen = _select_fitting_tool_id(tool_type, op_spec, tools, material=material)
        if chosen is not None:
            return chosen

    logger.warning(
        "no fitting tool for feature_refs=%s types=%s diameter=%s depth=%s",
        op_spec.feature_refs,
        _tool_type_precedence(op_spec),
        op_spec.diameter_mm,
        op_spec.depth_mm,
    )
    return UNRESOLVED_TOOL_ID


def _apply_tool_selection(
    op: OpSpec,
    tools: Sequence[Tool],
    tool_lookup: Mapping[str, Tool],
    material: str | None = None,
) -> None:
    """Assign ``tool_id`` and sync ``tool_type_needed`` to the selected catalog tool."""
    op.tool_id = select_tool(op, tools, material=material)
    tool = tool_lookup.get(op.tool_id)
    if tool is not None:
        op.tool_type_needed = tool.tool_type
        if material is not None and _tool_material_rank(tool, material) >= 2:
            logger.warning(
                "wrong-material tool fallback feature_refs=%s tool=%s material=%s library=%s",
                op.feature_refs,
                op.tool_id,
                material,
                _source_library_name(tool),
            )


def _batch_role(op: OpSpec) -> str:
    """Sub-role that keeps distinct finishes from over-merging under the flat vocab.

    contour_2d/waterline serve both a wall/profile role and a pocket-floor role; before
    the operation_type/strategy collapse these were separate ops with different feeds,
    so keep them in separate batches (avoids a wall+floor merge with mismatched params).
    """
    if op.operation in (BankOp.CONTOUR_2D, BankOp.WATERLINE):
        return "wall" if op.feature_type in (WALL_CLASSES | PROFILE_CLASSES) else "floor"
    if op.operation == BankOp.ENGRAVING:
        # Distinct engravings carry distinct text/target -> never merge them.
        return f"engrave:{sorted(op.feature_refs)}:{op.attributes.get('text')}"
    return ""


def _batch_key(op: OpSpec) -> tuple[str, str, str, str]:
    return (op.tool_id, op.operation, op.setup_id, _batch_role(op))


def _batch_probe_op(members: Sequence[OpSpec]) -> OpSpec:
    """Aggregate depth/clearance constraints for reachability on a candidate batch."""
    primary = members[0]
    depths = [member.depth_mm for member in members if member.depth_mm is not None]
    lateral_caps = [
        member.lateral_extent_mm
        for member in members
        if member.lateral_extent_mm is not None
        and member.operation in OPEN_MILLING_OPS
    ]
    fillet_radii = [
        member.fillet_radius_mm
        for member in members
        if member.fillet_radius_mm is not None
        and member.operation in FILLET_CAP_OPS
    ]
    return OpSpec(
        tool_type_needed=primary.tool_type_needed,
        operation=primary.operation,
        depth_mm=max(depths) if depths else None,
        lateral_extent_mm=min(lateral_caps) if lateral_caps else None,
        fillet_radius_mm=min(fillet_radii) if fillet_radii else None,
    )


def _split_members_by_reachability(
    tool: Tool,
    members: list[OpSpec],
) -> tuple[list[OpSpec], list[OpSpec]]:
    """Peel members off until the batch tool reaches deepest / fits tightest feature."""
    split_out: list[OpSpec] = []
    remaining = list(members)
    while len(remaining) > 1:
        probe = _batch_probe_op(remaining)
        if _tool_fits_op(tool, probe):
            return remaining, split_out
        deepest = max(remaining, key=lambda item: item.depth_mm or 0.0)
        remaining.remove(deepest)
        split_out.append(deepest)
        logger.info(
            "reachability split feature_refs=%s from batch tool=%s (depth=%s flute=%s)",
            deepest.feature_refs,
            tool.tool_id,
            probe.depth_mm,
            tool.flute_length_mm,
        )
    return remaining, split_out + remaining


def _merge_member_group(members: Sequence[OpSpec], tool: Tool | None) -> OpSpec:
    primary = members[0]
    merged_refs = sorted({ref for member in members for ref in member.feature_refs}, key=int)
    feature_types = {member.feature_type for member in members}
    feature_type = primary.feature_type if len(feature_types) == 1 else "batched"
    accesses = {member.access for member in members if member.access is not None}
    access = primary.access if len(accesses) <= 1 else None

    label = (
        f"{tool.tool_id}/{primary.operation}"
        if tool is not None
        else f"{primary.tool_id}/{primary.operation}"
    )
    if primary.parameters is not None:
        for member in members[1:]:
            if member.parameters is None:
                continue
            _params_consistent(primary.parameters, member.parameters, context=label)

    depths = [member.depth_mm for member in members if member.depth_mm is not None]
    lateral_caps = [
        member.lateral_extent_mm
        for member in members
        if member.lateral_extent_mm is not None
        and member.operation in OPEN_MILLING_OPS
    ]
    fillet_radii = [
        member.fillet_radius_mm
        for member in members
        if member.fillet_radius_mm is not None
        and member.operation in FILLET_CAP_OPS
    ]
    return OpSpec(
        feature_refs=merged_refs,
        feature_type=feature_type,
        setup_id=primary.setup_id,
        operation=primary.operation,
        tool_id=primary.tool_id,
        tool_type_needed=primary.tool_type_needed,
        lateral_extent_mm=min(lateral_caps) if lateral_caps else primary.lateral_extent_mm,
        fillet_radius_mm=min(fillet_radii) if fillet_radii else primary.fillet_radius_mm,
        depth_mm=max(depths) if depths else primary.depth_mm,
        access=access,
        parameters=primary.parameters,
        attributes=dict(primary.attributes),
    )


def group_operations_by_tool_strategy(
    op_specs: Sequence[OpSpec],
    tools: Sequence[Tool],
    context: MachiningContext | None = None,
) -> tuple[list[OpSpec], int]:
    """Collapse ops sharing (tool_id, operation, setup_id); split on reachability."""
    tool_lookup = _tool_by_id(tools)
    pending = list(op_specs)
    grouped: list[OpSpec] = []
    split_count = 0

    while pending:
        buckets: dict[tuple[str, str, str, str], list[OpSpec]] = {}
        for op in pending:
            buckets.setdefault(_batch_key(op), []).append(op)
        pending = []

        for members in buckets.values():
            tool = tool_lookup.get(members[0].tool_id)
            if tool is None:
                grouped.append(_merge_member_group(members, tool))
                continue
            if len(members) == 1:
                grouped.append(_merge_member_group(members, tool))
                continue

            kept, peeled = _split_members_by_reachability(tool, list(members))
            if kept:
                grouped.append(_merge_member_group(kept, tool))
            for member in peeled:
                split_count += 1
                prior_tool = member.tool_id
                _apply_tool_selection(member, tools, tool_lookup)
                if context is not None and member.tool_id != prior_tool:
                    member.parameters = assign_parameters(
                        member,
                        tool_lookup.get(member.tool_id),
                        context,
                    )
                pending.append(member)

    grouped.sort(
        key=lambda op: (
            op.operation,
            op.tool_id,
            int(op.feature_refs[0]) if op.feature_refs else 0,
        )
    )
    return grouped, split_count


def _refs_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    return bool(set(left) & set(right))


def _params_consistent(
    primary: MachiningParameters,
    other: MachiningParameters,
    *,
    context: str,
) -> bool:
    fields = (
        "spindle_rpm",
        "feed_mm_per_min",
        "plunge_mm_per_min",
        "stepdown_mm",
        "stepover_mm",
    )
    for field in fields:
        left = getattr(primary, field)
        right = getattr(other, field)
        if left is None and right is None:
            continue
        if left is None or right is None:
            logger.warning(
                "batch param mismatch (%s): %s None vs %s",
                context,
                field,
                left,
                right,
            )
            return False
        if abs(left - right) > _PARAM_TOLERANCE:
            logger.warning(
                "batch param mismatch (%s): %s %.4f vs %.4f",
                context,
                field,
                left,
                right,
            )
            return False
    return True


def build_precedence(op_specs: Sequence[OpSpec]) -> dict[str, list[str]]:
    """Hardcoded precedence edges between op_ids.

    Rules:
      - drill before tap on the same feature_ref set
      - rough (pocket_mill) before finish (finish_contour) on the same feature_ref
      - TODO(v0): datum-first ordering across features
      - TODO(v0): larger-before-smaller hole ordering on shared axes
    """
    by_id = {op.op_id: op for op in op_specs}
    precedence: dict[str, list[str]] = {op.op_id: [] for op in op_specs}

    def refs_key(op: OpSpec) -> tuple[str, ...]:
        return tuple(sorted(op.feature_refs))

    for op in op_specs:
        # A thread op (tapping with a tap tool, or thread milling) follows the
        # pilot hole-making op (drill/chip-break or helix bore) on the same refs.
        if op.tool_type_needed == "tap" or op.operation == BankOp.THREAD_MILL:
            for other in op_specs:
                pilot = other.tool_type_needed == "drill" or other.operation == BankOp.HELIX_BORE
                if (
                    pilot
                    and refs_key(other) == refs_key(op)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        # Every finish op follows any roughing op that overlaps its features.
        # (Subsumes the former finish_contour<-pocket_mill and wall_finish<-rough rules.)
        if op.operation in FINISH_OPS:
            for other in op_specs:
                if (
                    other.operation in ROUGH_OPS
                    and _refs_overlap(other.feature_refs, op.feature_refs)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        # A rest op follows the coarser pass it cleans up after, on the same feature(s):
        # rest_roughing after the main roughing (then the main finish follows it via the
        # FINISH<-ROUGH rule); rest_finish after the main contour finish.
        if op.operation == BankOp.REST_ROUGHING:
            for other in op_specs:
                if (
                    other.operation in ROUGH_OPS
                    and other.operation != BankOp.REST_ROUGHING
                    and _refs_overlap(other.feature_refs, op.feature_refs)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        if op.operation == BankOp.REST_FINISH:
            for other in op_specs:
                if (
                    other.operation in (BankOp.CONTOUR_2D, BankOp.WATERLINE)
                    and _refs_overlap(other.feature_refs, op.feature_refs)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        # Whole-setup deburr runs after every other op in its setup (edge-break last),
        # so the sequencer/beam search cannot float it before a machining op.
        if op.operation == BankOp.DEBURR:
            for other in op_specs:
                if (
                    other.op_id != op.op_id
                    and other.setup_id == op.setup_id
                    and other.operation != BankOp.DEBURR
                ):
                    precedence[op.op_id].append(other.op_id)

    # Preserve only valid op_ids and dedupe while keeping order.
    for op_id, deps in list(precedence.items()):
        seen: set[str] = set()
        cleaned: list[str] = []
        for dep in deps:
            if dep in by_id and dep not in seen and dep != op_id:
                cleaned.append(dep)
                seen.add(dep)
        precedence[op_id] = cleaned

    return precedence


def _operation_phase(op: OpSpec) -> int:
    return 0 if op.operation in ROUGH_OPS else 1


def _finish_order(op: OpSpec) -> int:
    """Finish-phase tie-break rank: floor -> wall -> surface -> fillet -> hole."""
    o = op.operation
    if o == BankOp.RASTER:
        return 0
    if o in (BankOp.CONTOUR_2D, BankOp.WATERLINE):
        # pocket/floor contour finish before wall/profile contour finish
        return 1 if op.feature_type in (WALL_CLASSES | PROFILE_CLASSES) else 0
    if o in _SURFACE_FINISH_ORDER_OPS:
        return 2
    if o == BankOp.PENCIL:
        return 3
    if o == BankOp.THREAD_MILL:
        return 7
    if op.tool_type_needed == "tap":
        return 6
    if o in (BankOp.DRILL, BankOp.CHIP_BREAK_DRILL):
        return 5
    return 99


def _sequence_tie_break_key(op: OpSpec, tool_lookup: Mapping[str, Tool]) -> tuple[Any, ...]:
    tool = tool_lookup.get(op.tool_id)
    diameter = tool.diameter_mm if tool is not None else 0.0
    phase = _operation_phase(op)
    if phase == 0:
        return (phase, -diameter, str(op.operation), op.tool_id)
    finish_rank = _finish_order(op)
    return (phase, finish_rank, -diameter, str(op.operation), op.tool_id)


def sequence(
    op_specs: Sequence[OpSpec],
    precedence: Mapping[str, Sequence[str]],
    *,
    tool_lookup: Mapping[str, Tool] | None = None,
) -> list[OpSpec]:
    """Topological sort with shop-ish tie-break (rough/coarse first, then finish)."""
    by_id = {op.op_id: op for op in op_specs}
    in_degree = {op.op_id: 0 for op in op_specs}
    dependents: dict[str, list[str]] = {op.op_id: [] for op in op_specs}
    tools = tool_lookup or {}

    for op_id, deps in precedence.items():
        for dep in deps:
            if dep not in by_id or op_id not in by_id:
                continue
            in_degree[op_id] += 1
            dependents[dep].append(op_id)

    def sort_ready(ids: list[str]) -> None:
        ids.sort(key=lambda op_id: _sequence_tie_break_key(by_id[op_id], tools))

    ready = [op_id for op_id, deg in in_degree.items() if deg == 0]
    sort_ready(ready)
    ordered_ids: list[str] = []

    while ready:
        current = ready.pop(0)
        ordered_ids.append(current)
        for child in dependents[current]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
        sort_ready(ready)

    if len(ordered_ids) != len(op_specs):
        remaining = set(by_id) - set(ordered_ids)
        cycle_nodes = _find_cycle_nodes(precedence, remaining)
        raise ValueError(
            "precedence graph has a cycle involving ops: "
            + ", ".join(sorted(cycle_nodes))
        )

    return [by_id[op_id] for op_id in ordered_ids]


def _find_cycle_nodes(
    precedence: Mapping[str, Sequence[str]],
    start_nodes: set[str],
) -> set[str]:
    """Return op_ids participating in a cycle (best-effort for error messaging)."""
    adjacency: dict[str, list[str]] = {node: list(precedence.get(node, [])) for node in start_nodes}
    for deps in precedence.values():
        for dep in deps:
            adjacency.setdefault(dep, [])

    visited: set[str] = set()
    stack: set[str] = set()
    cycle: set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        stack.add(node)
        for dep in adjacency.get(node, []):
            if dep not in visited:
                if dfs(dep):
                    cycle.add(node)
                    return True
            elif dep in stack:
                cycle.add(node)
                cycle.add(dep)
                return True
        stack.remove(node)
        return False

    for node in sorted(start_nodes):
        if node not in visited and dfs(node):
            break
    return cycle or start_nodes


def _handbook_parameters(op_spec: OpSpec, diameter: float) -> MachiningParameters:
    """Conservative handbook-ish defaults keyed by tool type + diameter."""
    if op_spec.tool_type_needed == "drill":
        rpm = max(800.0, 12000.0 / max(diameter, 0.5))
        feed = diameter * 60.0
        plunge = feed * 0.5
        return MachiningParameters(
            spindle_rpm=round(rpm, 1),
            feed_mm_per_min=round(feed, 1),
            plunge_mm_per_min=round(plunge, 1),
            stepdown_mm=None,
            stepover_mm=None,
            coolant=None,
            param_source="handbook_default",
        )

    if op_spec.tool_type_needed == "tap":
        rpm = max(200.0, 3000.0 / max(diameter, 0.5))
        feed = diameter * 20.0
        return MachiningParameters(
            spindle_rpm=round(rpm, 1),
            feed_mm_per_min=round(feed, 1),
            plunge_mm_per_min=round(feed, 1),
            stepdown_mm=None,
            stepover_mm=None,
            coolant=None,
            param_source="handbook_default",
        )

    if op_spec.operation in (BankOp.CONTOUR_2D, BankOp.WATERLINE) and (
        op_spec.feature_type in POCKET_CLASSES
    ):
        rpm = max(6000.0, 24000.0 / max(diameter, 0.5))
        stepover = round(diameter * 0.15, 3)
        return MachiningParameters(
            spindle_rpm=round(rpm, 1),
            feed_mm_per_min=round(diameter * 120.0, 1),
            plunge_mm_per_min=round(diameter * 40.0, 1),
            stepdown_mm=round(diameter * 0.05, 3),
            stepover_mm=stepover,
            coolant=None,
            param_source="handbook_default",
        )

    rpm = max(5000.0, 20000.0 / max(diameter, 0.5))
    stepdown = round(min(diameter * 0.5, 3.0), 3)
    stepover = round(diameter * 0.4, 3)
    return MachiningParameters(
        spindle_rpm=round(rpm, 1),
        feed_mm_per_min=round(diameter * 100.0, 1),
        plunge_mm_per_min=round(diameter * 35.0, 1),
        stepdown_mm=stepdown,
        stepover_mm=stepover,
        coolant=None,
        param_source="handbook_default",
    )


def _material_match_rank(preset_material: str | None, work_material: str | None) -> int:
    """Rank preset material affinity (lower is better).

    When work_material is set: exact category match, then ``all``, then anything else.
    When work_material is None: prefer ``all``, then any other preset.
    """
    if work_material is None:
        if preset_material in (None, "all"):
            return 0
        return 1

    target = work_material.strip().lower()
    if preset_material == target:
        return 0
    if preset_material == "all":
        return 1
    if preset_material is None:
        return 2
    return 3


def _tool_has_strategy_preset(tool: Tool, op_spec: OpSpec) -> bool:
    """Return True when the tool carries a named strategy preset for this operation."""
    return any(
        not _is_iso_material_category_preset(p.preset_name)
        and _strategy_match_rank(p.preset_name, op_spec) == 0
        for p in tool.presets
    )


def _iso_category_rank(preset_name: str, *, has_strategy_preset: bool) -> int:
    """Deprefer ISO material-group presets when a strategy-specific preset exists."""
    if has_strategy_preset and _is_iso_material_category_preset(preset_name):
        return 1
    return 0


def _preset_name_material_rank(preset_name: str, work_material: str | None) -> int:
    """Tie-breaker when preset_material is ``all``: match material tokens in preset names."""
    if work_material is None:
        return 0

    hints_by_material: dict[str, tuple[str, ...]] = {
        "aluminum": ("alu",),
        "steel": ("lowcsteel",),
        "stainless": ("stainless",),
        "cast_iron": ("castiron",),
        "titanium": ("titanium",),
    }
    hints = hints_by_material.get(work_material.strip().lower(), ())
    name = preset_name.lower()
    if hints and any(token in name for token in hints):
        return 0
    if hints:
        return 1
    return 0


def _strategy_match_rank(preset_name: str, op_spec: OpSpec) -> int:
    """Rank preset name affinity to the operation (lower is better).

    Substring heuristics on Fusion preset names, keyed by the canonical bank op
    plus tool_type and source feature class (contour_2d/waterline serve both a
    pocket-floor and a wall role, distinguished by feature_type).
    """
    name = preset_name.lower()
    op = op_spec.operation
    tool_type = op_spec.tool_type_needed

    if tool_type == "drill":
        return 0 if "drill" in name else 2

    if tool_type == "tap":
        return 0 if "tap" in name else 2

    if op == BankOp.THREAD_MILL:
        return 0 if "thread" in name else 2

    if op == BankOp.HELIX_BORE:
        if "bore" in name or "adaptive" in name or "rough" in name:
            return 0
        return 2

    if op == BankOp.FACING:
        if "face" in name and "rough" in name:
            return 0
        if "face" in name:
            return 1
        return 2

    if op in _SURFACE_FINISH_ORDER_OPS:  # freeform ball finish (constant_scallop, ...)
        if "surface" in name:
            return 0
        if "finish" in name:
            return 1
        return 2

    if op == BankOp.PENCIL:  # fillet finish
        if "surface" in name or "fillet" in name:
            return 0
        return 2

    if op in (BankOp.CONTOUR_2D, BankOp.WATERLINE):
        if op_spec.feature_type in (WALL_CLASSES | PROFILE_CLASSES):  # wall contour
            if "wall" in name:
                return 0
            if "finish" in name:
                return 1
            return 2
        # pocket-floor contour finish
        if "floor" in name:
            return 0
        finish_tokens = ("finish", "floor", "wall", "contour", "surface")
        if any(token in name for token in finish_tokens):
            return 1
        return 2

    if op == BankOp.RASTER:  # flat floor finish
        if "floor" in name:
            return 0
        finish_tokens = ("finish", "floor", "wall", "contour", "surface")
        if any(token in name for token in finish_tokens):
            return 1
        return 2

    # roughing default (pocket, dynamic_mill_2d, optirough, area_roughing, rest_roughing)
    if "rough" in name or "adaptive" in name or "traditional" in name:
        return 0
    if "finish" in name or "floor" in name or "wall" in name:
        return 2
    return 1


def resolve_preset(
    tool: Tool | None,
    op_spec: OpSpec,
    material: str | None,
) -> ToolPreset | None:
    """Pick the best tool preset for an operation, or None if the tool has none."""
    if tool is None or not tool.presets:
        return None

    has_strategy_preset = _tool_has_strategy_preset(tool, op_spec)

    ranked = sorted(
        tool.presets,
        key=lambda preset: (
            _material_match_rank(preset.preset_material, material),
            _strategy_match_rank(preset.preset_name, op_spec),
            _iso_category_rank(
                preset.preset_name,
                has_strategy_preset=has_strategy_preset,
            ),
            _preset_name_material_rank(preset.preset_name, material),
            preset.preset_name,
        ),
    )
    return ranked[0]


def _param_source_from_fields(preset_fields: set[str], handbook_fields: set[str]) -> str:
    if preset_fields and not handbook_fields:
        return "toolpath_preset"
    if handbook_fields and not preset_fields:
        return "handbook_default"
    return "mixed"


def _pick_float(
    field: str,
    preset_value: float | None,
    handbook_value: float | None,
    *,
    preset_fields: set[str],
    handbook_fields: set[str],
) -> float | None:
    if preset_value is not None:
        preset_fields.add(field)
        return preset_value
    if handbook_value is not None:
        handbook_fields.add(field)
    return handbook_value


def _pick_optional(
    field: str,
    preset_value: str | None,
    handbook_value: str | None,
    *,
    preset_fields: set[str],
    handbook_fields: set[str],
) -> str | None:
    if preset_value is not None:
        preset_fields.add(field)
        return preset_value
    if handbook_value is not None:
        handbook_fields.add(field)
    return handbook_value


def assign_parameters(
    op_spec: OpSpec,
    tool: Tool | None,
    context: MachiningContext,
) -> MachiningParameters:
    """Assign cutting parameters from tool presets with handbook fallback."""
    material = context.material
    diameter = tool.diameter_mm if tool is not None else (op_spec.diameter_mm or 6.0)
    handbook = _handbook_parameters(op_spec, diameter)
    preset = resolve_preset(tool, op_spec, material)

    preset_fields: set[str] = set()
    handbook_fields: set[str] = set()

    if op_spec.tool_type_needed == "drill":
        rpm = _pick_float(
            "spindle_rpm",
            preset.spindle_rpm if preset else None,
            handbook.spindle_rpm,
            preset_fields=preset_fields,
            handbook_fields=handbook_fields,
        )
        plunge = _pick_float(
            "plunge_mm_per_min",
            preset.plunge_mm_per_min if preset else None,
            handbook.plunge_mm_per_min,
            preset_fields=preset_fields,
            handbook_fields=handbook_fields,
        )
        feed = handbook.feed_mm_per_min
        if preset is not None and preset.feed_per_rev_mm is not None and rpm is not None:
            feed = preset.feed_per_rev_mm * rpm
            preset_fields.add("feed_mm_per_min")
        elif preset is not None and preset.feed_mm_per_min is not None:
            feed = preset.feed_mm_per_min
            preset_fields.add("feed_mm_per_min")
        else:
            handbook_fields.add("feed_mm_per_min")

        coolant = _pick_optional(
            "coolant",
            preset.coolant if preset else None,
            handbook.coolant,
            preset_fields=preset_fields,
            handbook_fields=handbook_fields,
        )
        return MachiningParameters(
            spindle_rpm=round(rpm, 1) if rpm is not None else None,
            feed_mm_per_min=round(feed, 1) if feed is not None else None,
            plunge_mm_per_min=round(plunge, 1) if plunge is not None else None,
            stepdown_mm=None,
            stepover_mm=None,
            coolant=coolant,
            param_source=_param_source_from_fields(preset_fields, handbook_fields),
        )

    if op_spec.tool_type_needed == "tap":
        rpm = _pick_float(
            "spindle_rpm",
            preset.spindle_rpm if preset else None,
            handbook.spindle_rpm,
            preset_fields=preset_fields,
            handbook_fields=handbook_fields,
        )
        feed = handbook.feed_mm_per_min
        if preset is not None and preset.feed_mm_per_min is not None:
            feed = preset.feed_mm_per_min
            preset_fields.add("feed_mm_per_min")
        elif preset is not None and preset.feed_per_rev_mm is not None and rpm is not None:
            feed = preset.feed_per_rev_mm * rpm
            preset_fields.add("feed_mm_per_min")
        elif preset is not None and preset.plunge_mm_per_min is not None:
            feed = preset.plunge_mm_per_min
            preset_fields.add("feed_mm_per_min")
        else:
            handbook_fields.add("feed_mm_per_min")

        plunge = _pick_float(
            "plunge_mm_per_min",
            preset.plunge_mm_per_min if preset else None,
            handbook.plunge_mm_per_min,
            preset_fields=preset_fields,
            handbook_fields=handbook_fields,
        )
        coolant = _pick_optional(
            "coolant",
            preset.coolant if preset else None,
            handbook.coolant,
            preset_fields=preset_fields,
            handbook_fields=handbook_fields,
        )
        return MachiningParameters(
            spindle_rpm=round(rpm, 1) if rpm is not None else None,
            feed_mm_per_min=round(feed, 1) if feed is not None else None,
            plunge_mm_per_min=round(plunge, 1) if plunge is not None else None,
            stepdown_mm=None,
            stepover_mm=None,
            coolant=coolant,
            param_source=_param_source_from_fields(preset_fields, handbook_fields),
        )

    rpm = _pick_float(
        "spindle_rpm",
        preset.spindle_rpm if preset else None,
        handbook.spindle_rpm,
        preset_fields=preset_fields,
        handbook_fields=handbook_fields,
    )
    feed = _pick_float(
        "feed_mm_per_min",
        preset.feed_mm_per_min if preset else None,
        handbook.feed_mm_per_min,
        preset_fields=preset_fields,
        handbook_fields=handbook_fields,
    )
    plunge = _pick_float(
        "plunge_mm_per_min",
        preset.plunge_mm_per_min if preset else None,
        handbook.plunge_mm_per_min,
        preset_fields=preset_fields,
        handbook_fields=handbook_fields,
    )
    stepdown = _pick_float(
        "stepdown_mm",
        preset.stepdown_mm if preset else None,
        handbook.stepdown_mm,
        preset_fields=preset_fields,
        handbook_fields=handbook_fields,
    )
    stepover = _pick_float(
        "stepover_mm",
        preset.stepover_mm if preset else None,
        handbook.stepover_mm,
        preset_fields=preset_fields,
        handbook_fields=handbook_fields,
    )
    coolant = _pick_optional(
        "coolant",
        preset.coolant if preset else None,
        handbook.coolant,
        preset_fields=preset_fields,
        handbook_fields=handbook_fields,
    )
    return MachiningParameters(
        spindle_rpm=round(rpm, 1) if rpm is not None else None,
        feed_mm_per_min=round(feed, 1) if feed is not None else None,
        plunge_mm_per_min=round(plunge, 1) if plunge is not None else None,
        stepdown_mm=round(stepdown, 3) if stepdown is not None else None,
        stepover_mm=round(stepover, 3) if stepover is not None else None,
        coolant=coolant,
        param_source=_param_source_from_fields(preset_fields, handbook_fields),
    )


def _assign_op_ids(op_specs: list[OpSpec]) -> None:
    for idx, op in enumerate(op_specs, start=1):
        op.op_id = f"OP{idx * 10:03d}"


def _tool_by_id(tools: Sequence[Tool]) -> dict[str, Tool]:
    return {t.tool_id: t for t in tools}


def _tool_ref_from_context(tool: Tool) -> ToolRef:
    return ToolRef(
        tool_id=tool.tool_id,
        tool_type=tool.tool_type,
        diameter_mm=tool.diameter_mm,
        flute_length_mm=tool.flute_length_mm,
        max_depth_mm=tool.max_depth_mm,
        corner_radius_mm=tool.corner_radius_mm,
        source=tool.source,
    )


def _unresolved_tool_ref() -> ToolRef:
    return ToolRef(
        tool_id=UNRESOLVED_TOOL_ID,
        tool_type="unknown",
        diameter_mm=0.1,
        flute_length_mm=None,
        max_depth_mm=None,
        source="planner_v0",
    )


def _plan_one_setup(
    feature_graph_path: str | Path,
    context: MachiningContext,
) -> _SetupPlanSlice:
    """Run the v0 per-setup pipeline: adapter -> ops -> sequence."""
    if len(context.setups) != 1:
        raise ValueError("each setup slice requires exactly one SetupContext")

    graph = load_feature_graph(feature_graph_path)
    nodes = graph.get("nodes", [])
    total_nodes = len(nodes)

    all_features = [cascade_node_to_feature(node) for node in nodes]
    features, dropped = filter_planner_features(all_features)
    nodes_by_id = {str(node["feature_id"]): node for node in nodes}
    envelope_faces = _resolve_envelope_stock_faces(graph, context)
    features, scope_dropped, scope_info = filter_features_for_setup(
        features,
        context,
        envelope_faces=envelope_faces,
        nodes_by_id=nodes_by_id,
        graph=graph,
        use_reachability=True,
    )
    setup_id = context.setups[0].setup_id
    facing_feature_ids = identify_facing_feature_ids(
        features,
        setup_id,
        envelope_faces=envelope_faces or None,
    )

    holes = [f for f in features if f.feature_type in HOLE_CLASSES]
    pockets = [f for f in features if f.feature_type in POCKET_CLASSES]
    other_features = [
        f for f in features
        if f.feature_type not in HOLE_CLASSES and f.feature_type not in POCKET_CLASSES
    ]

    op_specs: list[OpSpec] = []
    for stack in group_coaxial_holes(holes):
        op_specs.extend(
            map_feature_to_operations(
                stack,
                context,
                facing_feature_ids=facing_feature_ids,
            )
        )
    for pocket in sorted(pockets, key=lambda f: f.feature_id):
        op_specs.extend(
            map_feature_to_operations(
                pocket,
                context,
                facing_feature_ids=facing_feature_ids,
            )
        )
    for feat in sorted(other_features, key=lambda f: f.feature_id):
        op_specs.extend(
            map_feature_to_operations(
                feat,
                context,
                facing_feature_ids=facing_feature_ids,
            )
        )

    # Declared engraving (explicit process input): emit an ENGRAVING op per resolved
    # marking spec. Never inferred from geometry; unresolved targets are surfaced as
    # review flags below, not fabricated into ops.
    engrave_ops, engrave_unresolved = _engraving_ops(
        context, features, facing_feature_ids, setup_id
    )
    op_specs.extend(engrave_ops)

    # Whole-setup deburr: one auto edge-break pass over all in-scope features,
    # sequenced last. Emitted only when the setup has machined features.
    if op_specs:
        deburr_refs = sorted({f.feature_id for f in features}, key=int)
        op_specs.append(_deburr_op(deburr_refs, setup_id))

    _assign_op_ids(op_specs)

    tool_lookup = _tool_by_id(context.tools)
    for op in op_specs:
        _apply_tool_selection(op, context.tools, tool_lookup, material=context.material)

    # Rest machining (Track C): compare each feature's assigned rough/finish tool
    # against its fillet radius; inject rest ops (with their own smaller tool) where a
    # prior tool cannot reach the corner. Runs post-tool-selection so radii are known.
    rest_ops = _rest_machining_ops(op_specs, tool_lookup)
    if rest_ops:
        for op in rest_ops:
            _apply_tool_selection(op, context.tools, tool_lookup, material=context.material)
        op_specs.extend(rest_ops)
        _assign_op_ids(op_specs)

    for op in op_specs:
        tool = tool_lookup.get(op.tool_id)
        op.parameters = assign_parameters(op, tool, context)

    ops_before_grouping = len(op_specs)
    grouped_ops, reachability_splits = group_operations_by_tool_strategy(
        op_specs,
        context.tools,
        context,
    )
    _assign_op_ids(grouped_ops)

    precedence = build_precedence(grouped_ops)
    for op in grouped_ops:
        op.depends_on = list(precedence.get(op.op_id, []))

    # Residual/coverage flag (fail-safe, never silent): surface in-scope features that
    # got no shaping op, and declared engravings whose target didn't resolve. These are
    # flags for post-hoc review -- NOT ops, and they never auto-commit anything.
    _AUX_OPS = {BankOp.DEBURR, BankOp.ENGRAVING}
    machined_refs = {
        ref for op in grouped_ops if op.operation not in _AUX_OPS for ref in op.feature_refs
    }
    review_flags: list[dict[str, Any]] = [
        {
            "feature_id": f.feature_id,
            "class_name": f.feature_type,
            "source": "unclassified_residual",
            "confidence": "low",
            "note": "recognized in-scope feature produced no machining operation",
        }
        for f in features
        if f.feature_id not in machined_refs
    ]
    review_flags += [
        {
            "text": s.get("text"),
            "target": s.get("target"),
            "source": "engrave_unresolved",
            "confidence": "low",
            "note": "declared engraving target did not resolve to an in-scope feature",
        }
        for s in engrave_unresolved
    ]

    param_source_counts: dict[str, int] = {
        "toolpath_preset": 0,
        "handbook_default": 0,
        "mixed": 0,
    }

    for op in grouped_ops:
        source = (op.parameters or MachiningParameters(param_source="handbook_default")).param_source
        param_source_counts[source] = param_source_counts.get(source, 0) + 1

    wrong_material_ops = 0
    if context.material is not None:
        for op in grouped_ops:
            tool = tool_lookup.get(op.tool_id)
            if tool is not None and _tool_material_rank(tool, context.material) >= 2:
                wrong_material_ops += 1

    used_tool_ids = {op.tool_id for op in grouped_ops}
    plan_tools: list[ToolRef] = []
    for tool in context.tools:
        if tool.tool_id in used_tool_ids:
            plan_tools.append(_tool_ref_from_context(tool))
    if UNRESOLVED_TOOL_ID in used_tool_ids:
        plan_tools.append(_unresolved_tool_ref())

    setup_ctx = context.setups[0]
    setup = Setup(
        setup_id=setup_ctx.setup_id,
        opening_axis=setup_ctx.opening_axis,
        orientation=getattr(setup_ctx, "orientation", None),
        orientation_provisional=bool(getattr(setup_ctx, "orientation_provisional", False)),
        fixture=setup_ctx.fixture,
        notes=None,
    )

    stats = {
        "setup_id": setup_ctx.setup_id,
        "feature_graph_ref": context.feature_graph_ref,
        "nodes_in": total_nodes,
        "features_kept": len(features),
        "features_dropped": dropped,
        "features_scope_dropped": scope_dropped,
        "facing_feature_ids": sorted(facing_feature_ids),
        "operations_before_grouping": ops_before_grouping,
        "reachability_splits": reachability_splits,
        "operations_out": len(grouped_ops),
        "tools_used": len(used_tool_ids - {UNRESOLVED_TOOL_ID}),
        "unresolved_ops": sum(1 for op in grouped_ops if op.tool_id == UNRESOLVED_TOOL_ID),
        "params_toolpath_preset": param_source_counts.get("toolpath_preset", 0),
        "params_handbook_default": param_source_counts.get("handbook_default", 0),
        "params_mixed": param_source_counts.get("mixed", 0),
        "wrong_material_tool_ops": wrong_material_ops,
        "review_flags": review_flags,
        **scope_info,
    }

    return _SetupPlanSlice(
        setup=setup,
        grouped_ops=grouped_ops,
        precedence=precedence,
        plan_tools=plan_tools,
        stats=stats,
        feature_graph_ref=context.feature_graph_ref,
    )


def _reassign_global_op_ids(merged_ops: list[OpSpec]) -> None:
    """Assign contiguous op_ids across setups and remap depends_on edges.

    Per-setup slices each number their ops from OP010, so raw op_ids collide across
    setups. All pre-merge depends_on edges are intra-setup (cross-setup edges are
    added afterward with global ids), so the remap is keyed by (setup_id, old_id) to
    disambiguate the collision -- otherwise a slice's deps can be remapped onto
    another setup's ops and fabricate a cycle.
    """
    new_ids = [f"OP{idx * 10:03d}" for idx in range(1, len(merged_ops) + 1)]
    old_to_new: dict[tuple[str, str], str] = {
        (op.setup_id, op.op_id): new_id for op, new_id in zip(merged_ops, new_ids)
    }
    for op, new_id in zip(merged_ops, new_ids):
        op.depends_on = [
            old_to_new[(op.setup_id, dep)]
            for dep in op.depends_on
            if (op.setup_id, dep) in old_to_new
        ]
        op.op_id = new_id


def _apply_cross_setup_precedence(
    merged_ops: Sequence[OpSpec],
    setup_order: Sequence[str],
) -> None:
    """Later setups depend on the last op of the prior setup (fixture boundary)."""
    indices_by_setup: dict[str, list[int]] = {sid: [] for sid in setup_order}
    for idx, op in enumerate(merged_ops):
        indices_by_setup.setdefault(op.setup_id, []).append(idx)

    for idx in range(1, len(setup_order)):
        earlier_sid = setup_order[idx - 1]
        later_sid = setup_order[idx]
        earlier_indices = indices_by_setup.get(earlier_sid, [])
        later_indices = indices_by_setup.get(later_sid, [])
        if not earlier_indices or not later_indices:
            continue
        boundary_op_id = merged_ops[earlier_indices[-1]].op_id
        for op_idx in later_indices:
            op = merged_ops[op_idx]
            if boundary_op_id not in op.depends_on:
                op.depends_on.append(boundary_op_id)


def _merge_tool_catalog(slices: Sequence[_SetupPlanSlice]) -> list[ToolRef]:
    merged: list[ToolRef] = []
    seen: set[str] = set()
    for slice_ in slices:
        for tool in slice_.plan_tools:
            if tool.tool_id in seen:
                continue
            merged.append(tool)
            seen.add(tool.tool_id)
    return merged


def _setup_approach_map(setups: Sequence[Setup]) -> dict[str, str]:
    """Map setup_id -> the discrete approach direction the scorer penalizes changes
    between. Prefers a setup's general cardinal ``orientation`` (lateral path) when
    present, falling back to the calibrated ``opening_axis`` label. This is what
    makes consecutive ops in setups of different orientations incur an
    approach-change cost (see score_sequence)."""
    return {
        setup.setup_id: (getattr(setup, "orientation", None) or setup.opening_axis)
        for setup in setups
    }


def _run_sequence_search(
    ops: Sequence[OpSpec],
    precedence: Mapping[str, Sequence[str]],
    *,
    seq_search: SeqSearchStrategy,
    beam_width: int,
    tool_lookup: Mapping[str, Tool],
    setup_approach: Mapping[str, str] | None = None,
) -> tuple[list[OpSpec], dict[str, Any]]:
    result = search_sequence(
        ops,
        precedence,
        strategy=seq_search,
        beam_width=beam_width,
        setup_approach=setup_approach,
        tie_break_key=lambda op: _sequence_tie_break_key(op, tool_lookup),
        topo_sort_fn=sequence,
        tool_lookup=tool_lookup,
    )
    metadata = {
        "strategy": result.strategy,
        **result.score.as_metadata(),
    }
    return list(result.ordered), metadata


def _merge_for_cross_setup_boundary(
    slices: Sequence[_SetupPlanSlice],
    *,
    tool_lookup: Mapping[str, Tool],
) -> list[OpSpec]:
    """Concatenate per-setup topo orders so fixture boundary ops stay at setup tails."""
    merged: list[OpSpec] = []
    for slice_ in slices:
        per_setup_ordered = sequence(
            slice_.grouped_ops,
            slice_.precedence,
            tool_lookup=tool_lookup,
        )
        merged.extend(per_setup_ordered)
    return merged


def _merge_tool_lookup_from_inputs(
    setup_inputs: Sequence[SetupPlanInput],
) -> dict[str, Tool]:
    merged: dict[str, Tool] = {}
    for item in setup_inputs:
        for tool in item.context.tools:
            merged[tool.tool_id] = tool
    return merged


def _slice_to_operations(ordered: Sequence[OpSpec]) -> list[Operation]:
    return [
        Operation(
            op_id=op.op_id,
            sequence_index=idx,
            feature_refs=op.feature_refs,
            feature_type=op.feature_type,
            setup_id=op.setup_id,
            operation=str(op.operation),
            tool_id=op.tool_id,
            parameters=op.parameters or MachiningParameters(param_source="handbook_default"),
            depends_on=op.depends_on,
            access=op.access,
            attributes=dict(op.attributes),
        )
        for idx, op in enumerate(ordered)
    ]


def _aggregate_setup_stats(slices: Sequence[_SetupPlanSlice]) -> dict[str, Any]:
    totals = {
        "nodes_in": 0,
        "features_kept": 0,
        "features_dropped": 0,
        "features_scope_dropped": 0,
        "operations_before_grouping": 0,
        "reachability_splits": 0,
        "operations_out": 0,
        "tools_used": 0,
        "unresolved_ops": 0,
        "params_toolpath_preset": 0,
        "params_handbook_default": 0,
        "params_mixed": 0,
        "wrong_material_tool_ops": 0,
    }
    per_setup: dict[str, dict[str, Any]] = {}
    for slice_ in slices:
        per_setup[slice_.setup.setup_id] = dict(slice_.stats)
        for key in totals:
            totals[key] += int(slice_.stats.get(key, 0))
    totals["setups"] = len(slices)
    totals["per_setup"] = per_setup
    totals["review_flags"] = [
        {**flag, "setup_id": slice_.setup.setup_id}
        for slice_ in slices
        for flag in slice_.stats.get("review_flags", [])
    ]
    return totals


def plan_multi_setups(
    setup_inputs: Sequence[SetupPlanInput],
    *,
    setup_order: Sequence[str],
    source_part: str,
    seq_search: SeqSearchStrategy = "beam",
    seq_beam_width: int = 5,
) -> CamPlan:
    """Plan multiple setup slices and merge into one CamPlan.

    Each input runs the existing per-setup pipeline unchanged. Cross-setup ordering
    runs earlier setups to completion before later ones; the first op of each later
    setup depends on the last op of the prior setup.
    """
    if not setup_inputs:
        raise ValueError("setup_inputs must not be empty")
    if not setup_order:
        raise ValueError("setup_order must not be empty")

    slices = [
        _plan_one_setup(item.feature_graph_path, item.context)
        for item in setup_inputs
    ]
    by_setup_id = {slice_.setup.setup_id: slice_ for slice_ in slices}
    missing = [sid for sid in setup_order if sid not in by_setup_id]
    if missing:
        raise ValueError(f"setup_order references unknown setup_id(s): {missing}")

    ordered_slices = [by_setup_id[sid] for sid in setup_order]
    tool_lookup = _merge_tool_lookup_from_inputs(setup_inputs)
    merged_ops = _merge_for_cross_setup_boundary(ordered_slices, tool_lookup=tool_lookup)
    _reassign_global_op_ids(merged_ops)
    _apply_cross_setup_precedence(merged_ops, setup_order)

    precedence = {op.op_id: list(op.depends_on) for op in merged_ops}
    setups = [slice_.setup for slice_ in ordered_slices]
    setup_approach = _setup_approach_map(setups)
    ordered, sequence_score = _run_sequence_search(
        merged_ops,
        precedence,
        seq_search=seq_search,
        beam_width=seq_beam_width,
        tool_lookup=tool_lookup,
        setup_approach=setup_approach,
    )
    for op in ordered:
        op.depends_on = list(precedence.get(op.op_id, []))

    operations = _slice_to_operations(ordered)
    plan_tools = _merge_tool_catalog(ordered_slices)

    primary_ref = ordered_slices[0].feature_graph_ref
    graph_refs = {
        slice_.setup.setup_id: slice_.feature_graph_ref for slice_ in ordered_slices
    }

    cam_plan = CamPlan(
        source_part=source_part,
        feature_graph_ref=primary_ref,
        setups=setups,
        operations=operations,
        tools=plan_tools,
        remaining_material=None,
        metadata={
            "generator": "planner.plan_multi_setups",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "setup_order": list(setup_order),
            "feature_graph_refs": graph_refs,
            "planner_stats": _aggregate_setup_stats(ordered_slices),
            "sequence_score": sequence_score,
        },
    )
    CamPlan.model_validate(cam_plan.model_dump())
    return cam_plan


def plan(
    feature_graph_path: str | Path,
    context: MachiningContext,
    *,
    seq_search: SeqSearchStrategy = "beam",
    seq_beam_width: int = 5,
) -> CamPlan:
    """Orchestrate adapter -> ops -> sequence -> CamPlan (single setup)."""
    if len(context.setups) != 1:
        raise ValueError("v0 planner supports exactly one setup")

    slice_ = _plan_one_setup(feature_graph_path, context)
    tool_lookup = _tool_by_id(context.tools)
    setup_approach = _setup_approach_map([slice_.setup])
    ordered, sequence_score = _run_sequence_search(
        slice_.grouped_ops,
        slice_.precedence,
        seq_search=seq_search,
        beam_width=seq_beam_width,
        tool_lookup=tool_lookup,
        setup_approach=setup_approach,
    )
    for op in ordered:
        op.depends_on = list(slice_.precedence.get(op.op_id, []))
    operations = _slice_to_operations(ordered)

    cam_plan = CamPlan(
        source_part=context.part_id,
        feature_graph_ref=slice_.feature_graph_ref,
        setups=[slice_.setup],
        operations=operations,
        tools=slice_.plan_tools,
        remaining_material=None,
        metadata={
            "generator": "planner.plan",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "planner_stats": slice_.stats,
            "sequence_score": sequence_score,
        },
    )

    # Trigger cross-field validation (also runs in CamPlan constructor).
    CamPlan.model_validate(cam_plan.model_dump())
    return cam_plan


def _print_summary(plan: CamPlan) -> None:
    stats = plan.metadata.get("planner_stats", {})
    print(
        f"Planner summary: {stats.get('nodes_in', '?')} features in -> "
        f"{stats.get('features_kept', '?')} kept ("
        f"{stats.get('features_dropped', '?')} filtered) -> "
        f"{stats.get('operations_before_grouping', '?')} pre-batch -> "
        f"{stats.get('operations_out', len(plan.operations))} operations out, "
        f"{stats.get('tools_used', '?')} tools used, "
        f"{stats.get('unresolved_ops', 0)} UNRESOLVED, "
        f"sequence length {len(plan.operations)}, "
        f"preset params {stats.get('params_toolpath_preset', 0)}, "
        f"handbook fallback {stats.get('params_handbook_default', 0)}, "
        f"mixed {stats.get('params_mixed', 0)}"
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run v0 planner vertical slice.")
    parser.add_argument(
        "--feature-graph",
        type=Path,
        default=None,
        help="Single-setup feature graph (default: 96260B rear).",
    )
    parser.add_argument(
        "--multi-setup",
        action="store_true",
        help="Plan 96260B rear + front setups into one CamPlan.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "examples" / "cam_plan_96260B.json",
    )
    parser.add_argument(
        "--material",
        type=str,
        default="aluminum",
        help="Workpiece material for preset selection (default: aluminum).",
    )
    parser.add_argument(
        "--tool-source",
        choices=("hardcoded", "directory", "supabase"),
        default="supabase",
        help="Tool catalog source (default: supabase).",
    )
    parser.add_argument(
        "--setups",
        choices=("authored", "generated"),
        default="generated",
        help="Setup descriptor source: hand-authored YAML or cascade-generated (default: generated).",
    )
    parser.add_argument(
        "--setup-yaml",
        type=Path,
        default=REPO_ROOT / "eval" / "gt" / "96260B_setup.yaml",
        help="Hand-authored setup descriptor YAML (used when --setups=authored).",
    )
    parser.add_argument(
        "--generated-setup-yaml",
        type=Path,
        default=None,
        help="Generated setup descriptor YAML (optional override for --setups=generated).",
    )
    parser.add_argument(
        "--scope-diff",
        action="store_true",
        default=True,
        help="Print class-scope vs reachability assignment diff before planning (default: on).",
    )
    parser.add_argument(
        "--no-scope-diff",
        action="store_false",
        dest="scope_diff",
        help="Skip scope assignment diff output.",
    )
    parser.add_argument(
        "--seq-search",
        choices=("none", "greedy", "beam"),
        default="beam",
        help="Operation sequencing strategy (default: beam).",
    )
    parser.add_argument(
        "--seq-beam-width",
        type=int,
        default=5,
        help="Beam width for --seq-search beam (default: 5).",
    )
    args = parser.parse_args()

    from machining_context import build_context_v0

    ctx_kwargs = {
        "material": args.material,
        "tool_source": args.tool_source,
        "setups_source": args.setups,
        "generated_descriptor_path": args.generated_setup_yaml,
    }

    if args.multi_setup:
        rear_graph = REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
        front_graph = REPO_ROOT / "pipeline_out" / "96260B_front" / "feature_graph_cascade.json"
        rear_step = REPO_ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy"
        front_step = REPO_ROOT / "96260B_FRONT_XR004_PCD PLATE.stp copy"

        print("\n=== setups used ===")
        print(f"  source: {args.setups}")
        if args.setups == "authored":
            print(f"  path:   {args.setup_yaml}")
        else:
            gen_path = args.generated_setup_yaml or (
                REPO_ROOT / "pipeline_out" / "96260B" / "setup_descriptor.yaml"
            )
            print(f"  path:   {gen_path}")

        rear_ctx = build_context_v0(
            rear_step,
            args.setup_yaml,
            rear_graph,
            setup_id="rear",
            **ctx_kwargs,
        )
        front_ctx = build_context_v0(
            front_step,
            args.setup_yaml,
            front_graph,
            setup_id="front",
            **ctx_kwargs,
        )

        if args.scope_diff:
            for graph_path, ctx in ((rear_graph, rear_ctx), (front_graph, front_ctx)):
                graph = load_feature_graph(graph_path)
                nodes = graph.get("nodes", [])
                all_features = [cascade_node_to_feature(n) for n in nodes]
                features, _ = filter_planner_features(all_features)
                nodes_by_id = {str(n["feature_id"]): n for n in nodes}
                envelope_faces = _resolve_envelope_stock_faces(graph, ctx)
                print_scope_assignment_diff(
                    setup_id=ctx.setups[0].setup_id,
                    features=features,
                    context=ctx,
                    nodes_by_id=nodes_by_id,
                    graph=graph,
                    envelope_faces=envelope_faces,
                )

        cam_plan = plan_multi_setups(
            [
                SetupPlanInput(rear_graph, rear_ctx),
                SetupPlanInput(front_graph, front_ctx),
            ],
            setup_order=("rear", "front"),
            source_part="96260B",
            seq_search=args.seq_search,
            seq_beam_width=args.seq_beam_width,
        )
    else:
        feature_graph = args.feature_graph or (
            REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
        )
        ctx = build_context_v0(
            REPO_ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy",
            args.setup_yaml,
            feature_graph,
            setup_id="rear",
            **ctx_kwargs,
        )
        if args.scope_diff:
            graph = load_feature_graph(feature_graph)
            nodes = graph.get("nodes", [])
            all_features = [cascade_node_to_feature(n) for n in nodes]
            features, _ = filter_planner_features(all_features)
            nodes_by_id = {str(n["feature_id"]): n for n in nodes}
            envelope_faces = _resolve_envelope_stock_faces(graph, ctx)
            print_scope_assignment_diff(
                setup_id=ctx.setups[0].setup_id,
                features=features,
                context=ctx,
                nodes_by_id=nodes_by_id,
                graph=graph,
                envelope_faces=envelope_faces,
            )
        cam_plan = plan(
            feature_graph,
            ctx,
            seq_search=args.seq_search,
            seq_beam_width=args.seq_beam_width,
        )
    write_cam_plan(args.out, cam_plan)
    print(f"Wrote {args.out}")
    _print_summary(cam_plan)
