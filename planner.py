"""planner.py - v0 vertical slice: feature graph + machining context -> CamPlan.

Rule-based planner covering holes, pockets, walls, surfaces, fillets, profiles, and
faces. Ops are batched by (tool_id, strategy, setup_id) before sequencing.
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
from machining_context import MachiningContext, SetupScopeSpec, Tool, ToolPreset, load_feature_graph

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

PLANNER_FEATURE_CLASSES = (
    HOLE_CLASSES | POCKET_CLASSES | WALL_CLASSES | SURFACE_CLASSES
    | FILLET_CLASSES | PROFILE_CLASSES | FACE_CLASSES
)

ROUGH_OPERATION_TYPES = frozenset({"pocket_mill", "adaptive_rough", "rough", "bore", "facing"})
FINISH_OPERATION_TYPES = frozenset({
    "finish_contour",
    "finish",
    "wall_finish",
    "floor_finish",
    "surface_finish",
    "fillet_finish",
})

# Tie-break ordering after precedence is satisfied (documented shop approximation):
#   1) rough ops before finish (hard precedence edges)
#   2) rough ops: descending tool diameter (coarse before fine)
#   3) finish ops: floor -> wall -> surface -> fillet, then descending tool diameter
FINISH_STRATEGY_ORDER: dict[str, int] = {
    "facing": -1,
    "finishing_floor": 0,
    "finishing_wall": 1,
    "finishing": 2,
    "finishing_fillet": 3,
    "helical_bore": 4,
    "peck_drill": 5,
    "rigid_tap": 6,
}

_PARAM_TOLERANCE = 1e-3
BORE_MIN_DIA_MM = 0.375 * 25.4
BORE_MAX_DIA_MM = 1.0 * 25.4

# Open-feature sizing: prefer rigid mid-size tools (typical shop 1/4"-1/2"), not catalog minimum.
OPEN_DEFAULT_MIN_DIA_MM = 0.25 * 25.4  # 6.35 mm
OPEN_DEFAULT_MAX_DIA_MM = 0.5 * 25.4  # 12.7 mm

_AXIS_TOLERANCE = 1e-3

# Ops without a hole/bore diameter cap: pick largest fitting tool up to extent/default band.
OPEN_MILLING_OPERATION_TYPES = frozenset({
    "pocket_mill",
    "adaptive_rough",
    "finish_contour",
    "floor_finish",
    "wall_finish",
    "surface_finish",
    "fillet_finish",
})

# Finishing strategies where the shop prefers bullnose over plain endmill (floor/wall/fillet).
_BULLNOSE_PREFERRED_STRATEGIES = frozenset({
    "finishing_floor",
    "finishing_wall",
    "finishing_fillet",
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
    ordered: list[OpSpec]
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
    raw_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OpSpec:
    """Internal operation before CamPlan emission."""

    op_id: str = ""
    feature_refs: list[str] = field(default_factory=list)
    feature_type: str = ""
    setup_id: str = ""
    operation_type: str = ""
    strategy: str = ""
    tool_id: str = ""
    tool_type_needed: str = ""
    diameter_mm: float | None = None
    depth_mm: float | None = None
    lateral_extent_mm: float | None = None
    fillet_radius_mm: float | None = None
    access: PocketAccess | None = None
    depends_on: list[str] = field(default_factory=list)
    parameters: MachiningParameters | None = None


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

    is_tapped = any(
        params.get(key) for key in ("is_tapped", "tapped", "threaded", "has_thread")
    )

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


def filter_features_for_setup(
    features: Sequence[PlannerFeature],
    context: MachiningContext,
    *,
    envelope_faces: frozenset[int],
) -> tuple[list[PlannerFeature], int, dict[str, Any]]:
    """Drop features outside the setup's declared scope before planning."""
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
        "setup %s: %d features out of scope, kept %d",
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


def _rough_strategy(access: PocketAccess) -> str:
    return "roughing"


def _finish_strategy(access: PocketAccess) -> str:
    return "finishing_floor"


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
    """Lookup-table mapping from feature(s) to internal OpSpec list."""
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

        drill_spec = OpSpec(
            feature_refs=refs,
            feature_type=feature_type,
            setup_id=setup_id,
            operation_type="drill",
            strategy="peck_drill",
            tool_type_needed="drill",
            diameter_mm=max_diameter if max_diameter > 0 else None,
            depth_mm=max_depth if max_depth > 0 else None,
            access=None,
        )
        if _needs_bore_instead_of_drill(drill_spec, context.tools, material=context.material):
            ops = [
                OpSpec(
                    feature_refs=refs,
                    feature_type=feature_type,
                    setup_id=setup_id,
                    operation_type="bore",
                    strategy="helical_bore",
                    tool_type_needed="endmill",
                    diameter_mm=max_diameter if max_diameter > 0 else None,
                    depth_mm=max_depth if max_depth > 0 else None,
                    access=None,
                ),
            ]
        else:
            ops = [drill_spec]

        if tapped:
            ops.append(
                OpSpec(
                    feature_refs=refs,
                    feature_type=feature_type,
                    setup_id=setup_id,
                    operation_type="tap",
                    strategy="rigid_tap",
                    tool_type_needed="tap",
                    diameter_mm=max_diameter if max_diameter > 0 else None,
                    depth_mm=max_depth if max_depth > 0 else None,
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
                operation_type="pocket_mill",
                strategy=_rough_strategy(access),
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
                operation_type="finish_contour",
                strategy=_finish_strategy(access),
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
                operation_type="wall_finish",
                strategy="finishing_wall",
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
                operation_type="surface_finish",
                strategy="finishing",
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
                operation_type="fillet_finish",
                strategy="finishing_fillet",
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
                operation_type="adaptive_rough",
                strategy="roughing",
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
                operation_type="wall_finish",
                strategy="finishing_wall",
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
                    operation_type="facing",
                    strategy="facing",
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
                operation_type="floor_finish",
                strategy="finishing_floor",
                tool_type_needed="endmill",
                depth_mm=primary.depth_mm,
                lateral_extent_mm=lateral,
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
        and op_spec.operation_type in ("wall_finish", "fillet_finish", "finish_contour")
    ):
        return False
    return op_spec.operation_type in OPEN_MILLING_OPERATION_TYPES


def _max_open_tool_diameter_mm(op_spec: OpSpec) -> float | None:
    """Upper bound on tool diameter for open milling ops."""
    if op_spec.operation_type not in OPEN_MILLING_OPERATION_TYPES:
        return None

    caps: list[float] = []
    if op_spec.lateral_extent_mm is not None:
        caps.append(op_spec.lateral_extent_mm)
    if (
        op_spec.fillet_radius_mm is not None
        and op_spec.operation_type in ("wall_finish", "fillet_finish", "finish_contour")
    ):
        caps.append(2.0 * op_spec.fillet_radius_mm)

    if caps:
        return min(caps)
    return OPEN_DEFAULT_MAX_DIA_MM


def _tool_fits_op(tool: Tool, op_spec: OpSpec) -> bool:
    """True when ``tool`` can cut ``op_spec`` (depth reach + clearance when known)."""
    if op_spec.tool_type_needed == "drill" or op_spec.operation_type == "drill":
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
        if op_spec.operation_type == "bore" and op_spec.diameter_mm is not None:
            if tool.diameter_mm > op_spec.diameter_mm + 1e-6:
                return False
        max_dia = _max_open_tool_diameter_mm(op_spec)
        if max_dia is not None and tool.diameter_mm > max_dia + 1e-6:
            return False
        return True

    return True


def _tool_type_precedence(op_spec: OpSpec) -> tuple[str, ...]:
    """Return tool types to try in order for ``op_spec`` (fit before type preference)."""
    if op_spec.operation_type == "surface_finish":
        return _SURFACE_FINISH_TOOL_TYPES
    if op_spec.strategy in _BULLNOSE_PREFERRED_STRATEGIES:
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
    if op_spec.operation_type == "facing" and tool_type == "face_mill":
        return _select_facing_tool_id(op_spec, tools, material=material)

    candidates = [t for t in tools if t.tool_type == tool_type]
    if not candidates:
        return None

    fitting = [tool for tool in candidates if _tool_fits_op(tool, op_spec)]
    if not fitting:
        return None

    if op_spec.operation_type == "bore" or op_spec.strategy == "helical_bore":
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

    if (
        op_spec.tool_type_needed == "drill"
        or op_spec.operation_type in ("drill", "tap")
    ):
        fitting.sort(key=lambda t: (_tool_material_rank(t, material), t.diameter_mm))
        return fitting[0].tool_id

    if op_spec.operation_type in OPEN_MILLING_OPERATION_TYPES:
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


def _batch_key(op: OpSpec) -> tuple[str, str, str]:
    return (op.tool_id, op.operation_type, op.setup_id)


def _batch_probe_op(members: Sequence[OpSpec]) -> OpSpec:
    """Aggregate depth/clearance constraints for reachability on a candidate batch."""
    primary = members[0]
    depths = [member.depth_mm for member in members if member.depth_mm is not None]
    lateral_caps = [
        member.lateral_extent_mm
        for member in members
        if member.lateral_extent_mm is not None
        and member.operation_type in OPEN_MILLING_OPERATION_TYPES
    ]
    fillet_radii = [
        member.fillet_radius_mm
        for member in members
        if member.fillet_radius_mm is not None
        and member.operation_type in ("wall_finish", "fillet_finish", "finish_contour")
    ]
    return OpSpec(
        tool_type_needed=primary.tool_type_needed,
        operation_type=primary.operation_type,
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
        f"{tool.tool_id}/{primary.operation_type}"
        if tool is not None
        else f"{primary.tool_id}/{primary.operation_type}"
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
        and member.operation_type in OPEN_MILLING_OPERATION_TYPES
    ]
    fillet_radii = [
        member.fillet_radius_mm
        for member in members
        if member.fillet_radius_mm is not None
        and member.operation_type in ("wall_finish", "fillet_finish", "finish_contour")
    ]
    return OpSpec(
        feature_refs=merged_refs,
        feature_type=feature_type,
        setup_id=primary.setup_id,
        operation_type=primary.operation_type,
        strategy=primary.strategy,
        tool_id=primary.tool_id,
        tool_type_needed=primary.tool_type_needed,
        lateral_extent_mm=min(lateral_caps) if lateral_caps else primary.lateral_extent_mm,
        fillet_radius_mm=min(fillet_radii) if fillet_radii else primary.fillet_radius_mm,
        depth_mm=max(depths) if depths else primary.depth_mm,
        access=access,
        parameters=primary.parameters,
    )


def group_operations_by_tool_strategy(
    op_specs: Sequence[OpSpec],
    tools: Sequence[Tool],
    context: MachiningContext | None = None,
) -> tuple[list[OpSpec], int]:
    """Collapse ops sharing (tool_id, operation_type, setup_id); split on reachability."""
    tool_lookup = _tool_by_id(tools)
    pending = list(op_specs)
    grouped: list[OpSpec] = []
    split_count = 0

    while pending:
        buckets: dict[tuple[str, str, str], list[OpSpec]] = {}
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
            op.strategy,
            op.operation_type,
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
        if op.operation_type == "tap":
            for other in op_specs:
                if (
                    other.operation_type == "drill"
                    and refs_key(other) == refs_key(op)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        if op.operation_type in FINISH_OPERATION_TYPES:
            for other in op_specs:
                if (
                    other.operation_type in ROUGH_OPERATION_TYPES
                    and _refs_overlap(other.feature_refs, op.feature_refs)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        if op.operation_type == "finish_contour":
            for other in op_specs:
                if (
                    other.operation_type == "pocket_mill"
                    and _refs_overlap(other.feature_refs, op.feature_refs)
                    and other.op_id != op.op_id
                ):
                    precedence[op.op_id].append(other.op_id)

        if op.operation_type == "wall_finish":
            for other in op_specs:
                if (
                    other.operation_type in ("adaptive_rough", "pocket_mill")
                    and _refs_overlap(other.feature_refs, op.feature_refs)
                    and other.op_id != op.op_id
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
    if op.operation_type in ROUGH_OPERATION_TYPES or op.strategy == "roughing":
        return 0
    return 1


def _sequence_tie_break_key(op: OpSpec, tool_lookup: Mapping[str, Tool]) -> tuple[Any, ...]:
    tool = tool_lookup.get(op.tool_id)
    diameter = tool.diameter_mm if tool is not None else 0.0
    phase = _operation_phase(op)
    if phase == 0:
        return (phase, -diameter, op.strategy, op.tool_id, op.operation_type)
    finish_rank = FINISH_STRATEGY_ORDER.get(op.strategy, 99)
    return (phase, finish_rank, -diameter, op.strategy, op.tool_id, op.operation_type)


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
    if op_spec.tool_type_needed == "drill" or op_spec.operation_type == "drill":
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

    if op_spec.operation_type == "tap":
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

    if op_spec.operation_type == "finish_contour":
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

    Simple substring heuristics on Fusion preset names, e.g. ``*_Drill`` for holes,
    ``*_Rough`` / ``Adaptive`` for pocket roughing, ``*_Finish`` for finishing.
    """
    name = preset_name.lower()
    op = op_spec.operation_type
    tool_type = op_spec.tool_type_needed

    if tool_type == "drill" or op == "drill":
        if "drill" in name:
            return 0
        return 2

    if tool_type == "tap" or op == "tap":
        if "tap" in name:
            return 0
        return 2

    if op == "finish_contour" or op == "floor_finish":
        if op_spec.strategy == "finishing_floor" and "floor" in name:
            return 0
        if op == "floor_finish" and "floor" in name:
            return 0
        finish_tokens = ("finish", "floor", "wall", "contour", "surface")
        if any(token in name for token in finish_tokens):
            return 1
        if "rough" in name or "adaptive" in name:
            return 2
        return 2

    if op == "wall_finish" or op_spec.strategy == "finishing_wall":
        if "wall" in name:
            return 0
        if "finish" in name:
            return 1
        return 2

    if op == "surface_finish" or op_spec.strategy == "finishing":
        if "surface" in name:
            return 0
        if "finish" in name:
            return 1
        return 2

    if op == "fillet_finish" or op_spec.strategy == "finishing_fillet":
        if "surface" in name or "fillet" in name:
            return 0
        return 2

    if op == "facing" or op_spec.strategy == "facing":
        if "face" in name and "rough" in name:
            return 0
        if "face" in name:
            return 1
        return 2

    if op == "bore" or op_spec.strategy == "helical_bore":
        if "bore" in name or "adaptive" in name or "rough" in name:
            return 0
        return 2

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

    if op_spec.tool_type_needed == "drill" or op_spec.operation_type == "drill":
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

    if op_spec.operation_type == "tap":
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
    envelope_faces = _resolve_envelope_stock_faces(graph, context)
    features, scope_dropped, scope_info = filter_features_for_setup(
        features,
        context,
        envelope_faces=envelope_faces,
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

    _assign_op_ids(op_specs)

    tool_lookup = _tool_by_id(context.tools)
    for op in op_specs:
        _apply_tool_selection(op, context.tools, tool_lookup, material=context.material)

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

    ordered = sequence(grouped_ops, precedence, tool_lookup=tool_lookup)

    param_source_counts: dict[str, int] = {
        "toolpath_preset": 0,
        "handbook_default": 0,
        "mixed": 0,
    }

    for op in ordered:
        source = (op.parameters or MachiningParameters(param_source="handbook_default")).param_source
        param_source_counts[source] = param_source_counts.get(source, 0) + 1

    wrong_material_ops = 0
    if context.material is not None:
        for op in ordered:
            tool = tool_lookup.get(op.tool_id)
            if tool is not None and _tool_material_rank(tool, context.material) >= 2:
                wrong_material_ops += 1

    used_tool_ids = {op.tool_id for op in ordered}
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
        "operations_out": len(ordered),
        "tools_used": len(used_tool_ids - {UNRESOLVED_TOOL_ID}),
        "unresolved_ops": sum(1 for op in ordered if op.tool_id == UNRESOLVED_TOOL_ID),
        "params_toolpath_preset": param_source_counts.get("toolpath_preset", 0),
        "params_handbook_default": param_source_counts.get("handbook_default", 0),
        "params_mixed": param_source_counts.get("mixed", 0),
        "wrong_material_tool_ops": wrong_material_ops,
        **scope_info,
    }

    return _SetupPlanSlice(
        setup=setup,
        ordered=ordered,
        plan_tools=plan_tools,
        stats=stats,
        feature_graph_ref=context.feature_graph_ref,
    )


def _reassign_global_op_ids(merged_ops: list[OpSpec]) -> None:
    """Assign contiguous op_ids across setups and remap depends_on edges."""
    old_to_new: dict[str, str] = {}
    for idx, op in enumerate(merged_ops, start=1):
        new_id = f"OP{idx * 10:03d}"
        old_to_new[op.op_id] = new_id
        op.op_id = new_id
    for op in merged_ops:
        op.depends_on = [
            old_to_new[dep] for dep in op.depends_on if dep in old_to_new
        ]


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


def _slice_to_operations(ordered: Sequence[OpSpec]) -> list[Operation]:
    return [
        Operation(
            op_id=op.op_id,
            sequence_index=idx,
            feature_refs=op.feature_refs,
            feature_type=op.feature_type,
            setup_id=op.setup_id,
            operation_type=op.operation_type,
            strategy=op.strategy,
            tool_id=op.tool_id,
            parameters=op.parameters or MachiningParameters(param_source="handbook_default"),
            depends_on=op.depends_on,
            access=op.access,
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
    return totals


def plan_multi_setups(
    setup_inputs: Sequence[SetupPlanInput],
    *,
    setup_order: Sequence[str],
    source_part: str,
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
    merged_ops: list[OpSpec] = []
    for slice_ in ordered_slices:
        merged_ops.extend(slice_.ordered)
    _reassign_global_op_ids(merged_ops)
    _apply_cross_setup_precedence(merged_ops, setup_order)

    setups = [slice_.setup for slice_ in ordered_slices]
    operations = _slice_to_operations(merged_ops)
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
        },
    )
    CamPlan.model_validate(cam_plan.model_dump())
    return cam_plan


def plan(feature_graph_path: str | Path, context: MachiningContext) -> CamPlan:
    """Orchestrate adapter -> ops -> sequence -> CamPlan (single setup)."""
    if len(context.setups) != 1:
        raise ValueError("v0 planner supports exactly one setup")

    slice_ = _plan_one_setup(feature_graph_path, context)
    operations = _slice_to_operations(slice_.ordered)

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
    args = parser.parse_args()

    from machining_context import build_context_v0

    setup_yaml = REPO_ROOT / "eval" / "gt" / "96260B_setup.yaml"

    if args.multi_setup:
        rear_graph = REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
        front_graph = REPO_ROOT / "pipeline_out" / "96260B_front" / "feature_graph_cascade.json"
        rear_step = REPO_ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy"
        front_step = REPO_ROOT / "96260B_FRONT_XR004_PCD PLATE.stp copy"
        rear_ctx = build_context_v0(
            rear_step,
            setup_yaml,
            rear_graph,
            setup_id="rear",
            material=args.material,
            tool_source=args.tool_source,
        )
        front_ctx = build_context_v0(
            front_step,
            setup_yaml,
            front_graph,
            setup_id="front",
            material=args.material,
            tool_source=args.tool_source,
        )
        cam_plan = plan_multi_setups(
            [
                SetupPlanInput(rear_graph, rear_ctx),
                SetupPlanInput(front_graph, front_ctx),
            ],
            setup_order=("rear", "front"),
            source_part="96260B",
        )
    else:
        feature_graph = args.feature_graph or (
            REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
        )
        ctx = build_context_v0(
            REPO_ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy",
            setup_yaml,
            feature_graph,
            setup_id="rear",
            material=args.material,
            tool_source=args.tool_source,
        )
        cam_plan = plan(feature_graph, ctx)
    write_cam_plan(args.out, cam_plan)
    print(f"Wrote {args.out}")
    _print_summary(cam_plan)
