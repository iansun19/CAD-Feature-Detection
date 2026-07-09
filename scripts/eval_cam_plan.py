#!/usr/bin/env python3
"""Compare an emitted CamPlan against Toolpath ground-truth YAML.

Read-only measurement harness - reports coverage, op-type, tool, sequence, and
parameter divergence. Does not pass/fail; surfaces where the plan differs from shop.

Prefer ``scripts/plan_sanity_report.py`` for the standing sanity check (gates + CI exit).
This script remains useful for per-feature GT matching and aggregate scorecard detail.

Run:
  python scripts/eval_cam_plan.py
  python scripts/eval_cam_plan.py examples/cam_plan_96260B.json eval/gt/96260B_rear_shop_program.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cam_plan_schema import CamPlan, load_cam_plan  # noqa: E402

INCH_TO_MM = 25.4
DIAMETER_TOL_MM = 0.75
ROUGH_OP_TYPES = frozenset({"pocket_mill", "adaptive_rough", "rough", "bore", "slot_mill"})
FINISH_OP_TYPES = frozenset({"finish_contour", "finish", "contour", "wall_finish", "floor_finish"})
HOLE_OP_TYPES = frozenset({"drill", "bore", "peck_drill", "helical_bore", "tap", "spot_drill"})

POCKET_CLASSES = frozenset({
    "filleted_pocket",
    "filleted_open_pocket",
    "open_pocket",
    "pocket",
    "blind_pocket",
})
HOLE_CLASSES = frozenset({
    "through_hole",
    "hole",
    "blind_hole",
    "filleted_blind_hole",
})

SHOP_PROGRAM_GT = REPO_ROOT / "eval" / "gt" / "96260B_rear_shop_program.yaml"

SHOP_STRATEGIES = frozenset({
    "roughing",
    "finishing_floor",
    "finishing_wall",
    "finishing",
    "finishing_fillet",
    "facing",
})

SHOP_FEATURE_CATEGORIES = frozenset({
    "Face",
    "Contour Surface",
    "Wall",
    "Filleted Pocket",
    "Filleted Blind Hole",
    "Through Hole",
    "Profile",
    "Outer Fillet",
})

CASCADE_TO_SHOP_CATEGORY: dict[str, str] = {
    "flat": "Face",
    "contour_surface": "Contour Surface",
    "wall": "Wall",
    "filleted_pocket": "Filleted Pocket",
    "filleted_open_pocket": "Filleted Pocket",
    "open_pocket": "Filleted Pocket",
    "pocket": "Filleted Pocket",
    "blind_pocket": "Filleted Pocket",
    "filleted_blind_hole": "Filleted Blind Hole",
    "through_hole": "Through Hole",
    "hole": "Through Hole",
    "blind_hole": "Filleted Blind Hole",
    "profile": "Profile",
    "outer_fillet": "Outer Fillet",
}

STRATEGY_TO_OP_TYPE: dict[str, str] = {
    "roughing": "adaptive_rough",
    "finishing_floor": "floor_finish",
    "finishing_wall": "wall_finish",
    "finishing": "surface_finish",
    "finishing_fillet": "fillet_finish",
    "facing": "facing",
}

# Legacy planner strategy tags -> shop strategy vocabulary (aggregate coverage).
EMITTED_STRATEGY_TO_SHOP: dict[str, str] = {
    "spiral": "roughing",
    "open_side": "roughing",
    "plunge_roughing": "roughing",
    "contour": "finishing",
    "helical_bore": "finishing_wall",
}

# Eval tool-type buckets (match by type + diameter, not label string).
EVAL_TOOL_TYPE_ALIASES: dict[str, str] = {
    "bullnose_endmill": "bullnose",
    "bullnose": "bullnose",
    "ball_endmill": "ball",
    "ball": "ball",
    "face_mill": "face_mill",
    "endmill": "endmill",
    "drill": "drill",
    "tap": "tap",
}

# Documented shop divergences when GT YAML lacks an ``operations`` block yet.
# Keys: part_id -> feature_refs tuple -> override fields.
_SHOP_OP_OVERRIDES: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {
    "96260B_rear": {
        ("14", "15"): {
            "operation_type": "bore",
            "strategy": "helical_bore",
            "tool_type": "endmill",
            "notes": (
                "Toolpath bores/helical-mills the central 101.7 mm region; "
                "not a peck-drill cycle."
            ),
        },
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required (pip install pyyaml).") from exc
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    text = "".join(
        ch
        if (unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")
        else " "
        for ch in text
    )
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"GT yaml root must be a mapping: {path}")
    return data


def _resolve_path(raw: str, anchor: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (REPO_ROOT, anchor.parent):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return (REPO_ROOT / path).resolve()


def _refs_key(feature_refs: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(str(ref) for ref in feature_refs))


def _to_mm(value: Any, unit: str | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if unit in ("inch", "inches", "in"):
        return numeric * INCH_TO_MM
    return numeric


def _op_phase(operation_type: str) -> str:
    op = operation_type.strip().lower()
    if op in ROUGH_OP_TYPES:
        return "rough"
    if op in FINISH_OP_TYPES:
        return "finish"
    if op in HOLE_OP_TYPES:
        return "hole"
    return "other"


@dataclass(frozen=True)
class GtOperation:
    feature_refs: tuple[str, ...]
    operation_type: str
    strategy: str | None = None
    tool_type: str | None = None
    diameter_mm: float | None = None
    sequence: int | None = None
    spindle_rpm: float | None = None
    feed_mm_per_min: float | None = None
    plunge_mm_per_min: float | None = None
    notes: str | None = None
    source: str = "gt_yaml"
    gt_class: str | None = None
    feature_count: int | None = None
    feature_categories: tuple[str, ...] = ()
    setup_index: int | None = None


@dataclass(frozen=True)
class EmittedOperation:
    op_id: str
    feature_refs: tuple[str, ...]
    operation_type: str
    strategy: str
    tool_id: str
    tool_type: str | None
    tool_diameter_mm: float | None
    sequence_index: int
    spindle_rpm: float | None
    feed_mm_per_min: float | None
    plunge_mm_per_min: float | None
    feature_type: str


@dataclass
class GtSchemaReport:
    path: Path
    part_id: str | None
    has_counts: bool
    has_features: bool
    has_operations: bool
    operation_source: str
    feature_entries: int
    count_classes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def inspect_gt_schema(gt: Mapping[str, Any], path: Path) -> GtSchemaReport:
    """Summarize which GT fields are present (adapt to actual YAML, don't assume)."""
    counts = gt.get("counts")
    features = gt.get("features")
    operations = gt.get("operations") or gt.get("shop_operations")

    report = GtSchemaReport(
        path=path,
        part_id=str(gt.get("part_id")) if gt.get("part_id") is not None else None,
        has_counts=isinstance(counts, Mapping),
        has_features=isinstance(features, list),
        has_operations=isinstance(operations, list) and bool(operations),
        operation_source="none",
        feature_entries=len(features) if isinstance(features, list) else 0,
        count_classes=dict(counts) if isinstance(counts, Mapping) else {},
    )

    if report.has_operations:
        report.operation_source = "operations_block"
    elif isinstance(gt.get("setups"), list) and any(
        isinstance(setup, Mapping) and isinstance(setup.get("operations"), list) and setup["operations"]
        for setup in gt["setups"]
    ):
        report.has_operations = True
        report.operation_source = "shop_program_setups"
    elif isinstance(features, list) and any(
        isinstance(item, Mapping)
        and (item.get("operations") or item.get("shop_operation_type") or item.get("operation_type"))
        for item in features
    ):
        report.operation_source = "per_feature_operations"
        report.notes.append("Per-feature operation hints found under features[].")
    else:
        report.operation_source = "inferred_from_features_graph"
        report.notes.append(
            "GT YAML has counts/features only (cascade-eval shape). "
            "Shop operations inferred from feature graph + diameter-matched features "
            "+ documented shop overrides."
        )
        if report.has_counts:
            report.notes.append(
                "counts block is instance-level Toolpath feature totals (not operations)."
            )

    return report


def _category_from_feature_name(name: str) -> str | None:
    """Parse Toolpath display name prefix, e.g. 'Face #4' -> 'Face'."""
    match = re.match(r"^([A-Za-z][A-Za-z ]*?)\s*#", name.strip())
    if not match:
        return None
    return match.group(1).strip()


def _categories_from_features_shown(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    categories: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        category = _category_from_feature_name(item)
        if category and category not in seen:
            categories.append(category)
            seen.add(category)
    return tuple(categories)


def _normalize_eval_tool_type(tool_type: str | None) -> str | None:
    if tool_type is None:
        return None
    normalized = tool_type.strip().lower()
    return EVAL_TOOL_TYPE_ALIASES.get(normalized, normalized)


def _shop_strategy_from_emitted(op: EmittedOperation) -> str | None:
    if op.strategy in SHOP_STRATEGIES:
        return op.strategy
    mapped = EMITTED_STRATEGY_TO_SHOP.get(op.strategy)
    if mapped is not None:
        return mapped
    if op.operation_type in ROUGH_OP_TYPES:
        return "roughing"
    if op.operation_type == "floor_finish":
        return "finishing_floor"
    if op.operation_type == "wall_finish":
        return "finishing_wall"
    if op.operation_type in ("surface_finish", "finish_contour", "finish"):
        return "finishing"
    if op.operation_type == "fillet_finish":
        return "finishing_fillet"
    if op.operation_type == "facing":
        return "facing"
    return None


def _emitted_shop_strategies(emitted_ops: Sequence[EmittedOperation]) -> set[str]:
    found: set[str] = set()
    for op in emitted_ops:
        shop_strategy = _shop_strategy_from_emitted(op)
        if shop_strategy is not None:
            found.add(shop_strategy)
    return found


def _emitted_shop_categories(
    emitted_ops: Sequence[EmittedOperation],
    graph: Mapping[str, Any] | None = None,
) -> set[str]:
    categories: set[str] = set()
    nodes_by_id: dict[str, str] = {}
    if graph is not None:
        for node in graph.get("nodes", []):
            if isinstance(node, Mapping) and node.get("feature_id") is not None:
                nodes_by_id[str(node["feature_id"])] = str(node.get("class_name", ""))

    for op in emitted_ops:
        category = CASCADE_TO_SHOP_CATEGORY.get(op.feature_type)
        if category is not None:
            categories.add(category)
        for ref in op.feature_refs:
            node_class = nodes_by_id.get(ref)
            if node_class:
                mapped = CASCADE_TO_SHOP_CATEGORY.get(node_class)
                if mapped is not None:
                    categories.add(mapped)
    return categories


def _shop_categories_from_gt(gt: Mapping[str, Any], gt_ops: Sequence[GtOperation]) -> set[str]:
    categories: set[str] = set()
    summary = gt.get("summary")
    if isinstance(summary, Mapping):
        raw = summary.get("feature_categories_machined")
        if isinstance(raw, list):
            categories.update(str(item) for item in raw)
    for op in gt_ops:
        categories.update(op.feature_categories)
    return categories


def _parse_shop_program_operation(
    raw: Mapping[str, Any],
    *,
    setup_index: int | None,
    source: str,
) -> GtOperation | None:
    strategy = raw.get("strategy")
    if not strategy:
        return None

    op_type = raw.get("operation_type") or raw.get("op_type")
    if not op_type:
        op_type = STRATEGY_TO_OP_TYPE.get(str(strategy), str(strategy))

    diameter = _to_mm(raw.get("tool_diameter_in") or raw.get("diameter_in"), "inches")
    if diameter is None:
        diameter = _to_mm(raw.get("diameter_mm"), raw.get("diameter_unit"))

    feed = _to_mm(raw.get("feed_in_min"), "inches")
    if feed is None:
        feed = _to_mm(raw.get("feed_mm_per_min"), None)

    params = raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else raw
    spindle = params.get("spindle_rpm")
    plunge = params.get("plunge_mm_per_min") or params.get("plunge_in_min")

    feature_count = raw.get("feature_count")
    if feature_count is not None:
        feature_count = int(feature_count)

    return GtOperation(
        feature_refs=(),
        operation_type=str(op_type),
        strategy=str(strategy),
        tool_type=str(raw["tool_type"]) if raw.get("tool_type") is not None else None,
        diameter_mm=diameter,
        sequence=int(raw["index"]) if raw.get("index") is not None else None,
        spindle_rpm=float(spindle) if spindle is not None else None,
        feed_mm_per_min=float(feed) if feed is not None else None,
        plunge_mm_per_min=float(_to_mm(plunge, "inches") or plunge)
        if plunge is not None
        else None,
        notes=str(raw["notes"]) if raw.get("notes") is not None else None,
        source=source,
        gt_class=None,
        feature_count=feature_count,
        feature_categories=_categories_from_features_shown(raw.get("features_shown")),
        setup_index=setup_index,
    )


def _parse_gt_operation(raw: Mapping[str, Any], *, source: str) -> GtOperation | None:
    refs_raw = raw.get("feature_refs") or raw.get("feature_ref")
    if refs_raw is None and raw.get("feature_id") is not None:
        refs_raw = [raw["feature_id"]]
    if not refs_raw:
        return None

    if isinstance(refs_raw, (str, int)):
        refs = (str(refs_raw),)
    else:
        refs = tuple(sorted(str(ref) for ref in refs_raw))

    op_type = raw.get("operation_type") or raw.get("op_type")
    if not op_type:
        return None

    tool = raw.get("tool") if isinstance(raw.get("tool"), Mapping) else {}
    params = raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else raw

    diameter = _to_mm(
        raw.get("diameter_mm") or tool.get("diameter_mm") or raw.get("diameter"),
        raw.get("diameter_unit") or tool.get("diameter_unit"),
    )
    if diameter is None:
        diameter = _to_mm(raw.get("diameter_in") or tool.get("diameter_in"), "inches")

    return GtOperation(
        feature_refs=refs,
        operation_type=str(op_type),
        strategy=str(raw["strategy"]) if raw.get("strategy") is not None else None,
        tool_type=str(tool.get("tool_type") or raw.get("tool_type"))
        if (tool.get("tool_type") or raw.get("tool_type"))
        else None,
        diameter_mm=diameter,
        sequence=int(raw["sequence"]) if raw.get("sequence") is not None else None,
        spindle_rpm=float(params["spindle_rpm"]) if params.get("spindle_rpm") is not None else None,
        feed_mm_per_min=(
            float(params["feed_mm_per_min"]) if params.get("feed_mm_per_min") is not None else None
        ),
        plunge_mm_per_min=(
            float(params["plunge_mm_per_min"])
            if params.get("plunge_mm_per_min") is not None
            else None
        ),
        notes=str(raw["notes"]) if raw.get("notes") is not None else None,
        source=source,
        gt_class=str(raw.get("class") or raw.get("feature_type"))
        if raw.get("class") or raw.get("feature_type")
        else None,
    )


def _load_operations_from_yaml(gt: Mapping[str, Any]) -> list[GtOperation]:
    ops: list[GtOperation] = []
    for key in ("operations", "shop_operations"):
        raw_ops = gt.get(key)
        if not isinstance(raw_ops, list):
            continue
        for item in raw_ops:
            if isinstance(item, Mapping):
                parsed = _parse_gt_operation(item, source="gt_yaml")
                if parsed is not None:
                    ops.append(parsed)
    if ops:
        return ops

    setups = gt.get("setups")
    if isinstance(setups, list):
        for setup in setups:
            if not isinstance(setup, Mapping):
                continue
            setup_index = int(setup["setup"]) if setup.get("setup") is not None else None
            raw_ops = setup.get("operations")
            if not isinstance(raw_ops, list):
                continue
            for item in raw_ops:
                if isinstance(item, Mapping):
                    parsed = _parse_shop_program_operation(
                        item,
                        setup_index=setup_index,
                        source="shop_program",
                    )
                    if parsed is not None:
                        ops.append(parsed)
    if ops:
        return ops

    features = gt.get("features")
    if isinstance(features, list):
        for item in features:
            if not isinstance(item, Mapping):
                continue
            nested = item.get("operations")
            if isinstance(nested, list):
                for raw in nested:
                    if isinstance(raw, Mapping):
                        parsed = _parse_gt_operation(
                            {**raw, "class": item.get("class")},
                            source="gt_yaml_features",
                        )
                        if parsed is not None:
                            ops.append(parsed)
                continue
            shop_type = item.get("shop_operation_type") or item.get("operation_type")
            if shop_type and item.get("feature_id") is not None:
                parsed = _parse_gt_operation(
                    {
                        "feature_id": item["feature_id"],
                        "operation_type": shop_type,
                        "class": item.get("class"),
                        "tool": item.get("tool"),
                        "parameters": item.get("parameters"),
                        "diameter_mm": item.get("diameter_mm"),
                        "diameter_in": item.get("diameter"),
                        "diameter_unit": "inches" if item.get("diameter") is not None else None,
                    },
                    source="gt_yaml_features",
                )
                if parsed is not None:
                    ops.append(parsed)
    return ops


def _graph_nodes(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, Mapping)]


def _node_diameter_mm(node: Mapping[str, Any]) -> float | None:
    params = node.get("params") or {}
    for key in ("nominal_diameter", "diameter_mm"):
        if params.get(key) is not None:
            return float(params[key])
    if params.get("radius") is not None:
        return float(params["radius"]) * 2.0
    if params.get("radius_mm") is not None:
        return float(params["radius_mm"]) * 2.0
    return None


def _match_gt_feature_to_nodes(
    gt_feature: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Link a GT features[] entry to cascade feature_id(s) by class + diameter."""
    gt_class = str(gt_feature.get("class", ""))
    gt_diameter_mm = _to_mm(gt_feature.get("diameter_mm"), gt_feature.get("diameter_unit"))
    if gt_diameter_mm is None and gt_feature.get("diameter") is not None:
        gt_diameter_mm = float(gt_feature["diameter"]) * INCH_TO_MM

    if gt_feature.get("feature_id") is not None:
        return [str(gt_feature["feature_id"])]

    matches: list[str] = []
    for node in nodes:
        if str(node.get("class_name", "")) != gt_class:
            continue
        node_diameter = _node_diameter_mm(node)
        if gt_diameter_mm is not None and node_diameter is not None:
            if abs(node_diameter - gt_diameter_mm) > DIAMETER_TOL_MM:
                continue
        matches.append(str(node["feature_id"]))
    return matches


def _group_coaxial_hole_refs(
    feature_ids: Sequence[str],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, ...]]:
    """Group hole feature_ids sharing the same axis (planner coaxial stack)."""
    holes = [fid for fid in feature_ids if fid in nodes_by_id]
    if not holes:
        return []

    def axis_key(fid: str) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        params = nodes_by_id[fid].get("params") or {}
        axis = params.get("axis") or {}
        point = axis.get("point")
        direction = axis.get("direction")
        if not isinstance(point, (list, tuple)) or not isinstance(direction, (list, tuple)):
            return None
        if len(point) != 3 or len(direction) != 3:
            return None
        p = tuple(round(float(v), 3) for v in point)
        d = tuple(round(float(v), 3) for v in direction)
        norm = sum(v * v for v in d) ** 0.5
        if norm <= 1e-9:
            return None
        d = tuple(v / norm for v in d)
        return p, d

    buckets: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[str]] = {}
    singletons: list[tuple[str, ...]] = []
    for fid in sorted(holes, key=int):
        key = axis_key(fid)
        if key is None:
            singletons.append((fid,))
            continue
        buckets.setdefault(key, []).append(fid)

    groups = [tuple(sorted(group, key=int)) for group in buckets.values()]
    groups.extend(singletons)
    groups.sort(key=lambda g: int(g[0]))
    return groups


def infer_shop_operations(
    gt: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    part_id: str | None,
) -> list[GtOperation]:
    """Build shop-reference operations when GT YAML lacks an operations block."""
    nodes = _graph_nodes(graph)
    nodes_by_id = {str(node["feature_id"]): node for node in nodes}
    ops: list[GtOperation] = []
    overrides = _SHOP_OP_OVERRIDES.get(part_id or "", {})

    hole_ids: list[str] = []
    gt_features = gt.get("features")
    if isinstance(gt_features, list):
        for item in gt_features:
            if not isinstance(item, Mapping):
                continue
            gt_class = str(item.get("class", ""))
            matched = _match_gt_feature_to_nodes(item, nodes)
            if gt_class in HOLE_CLASSES:
                hole_ids.extend(matched)
            elif gt_class in POCKET_CLASSES:
                for fid in matched:
                    ops.extend(
                        [
                            GtOperation(
                                feature_refs=(fid,),
                                operation_type="pocket_mill",
                                source="inferred",
                                gt_class=gt_class,
                            ),
                            GtOperation(
                                feature_refs=(fid,),
                                operation_type="finish_contour",
                                source="inferred",
                                gt_class=gt_class,
                            ),
                        ]
                    )

    if not hole_ids:
        hole_ids = [
            str(node["feature_id"])
            for node in nodes
            if str(node.get("class_name", "")) in HOLE_CLASSES
        ]

    for group in _group_coaxial_hole_refs(hole_ids, nodes_by_id):
        override = overrides.get(group, {})
        op_type = str(override.get("operation_type", "drill"))
        diameters = [
            d
            for fid in group
            if (d := _node_diameter_mm(nodes_by_id[fid])) is not None
        ]
        max_diameter = max(diameters) if diameters else None
        ops.append(
            GtOperation(
                feature_refs=group,
                operation_type=op_type,
                strategy=str(override["strategy"]) if override.get("strategy") else None,
                tool_type=str(override["tool_type"]) if override.get("tool_type") else None,
                diameter_mm=max_diameter,
                notes=str(override["notes"]) if override.get("notes") else None,
                source="inferred_override" if override else "inferred",
                gt_class=str(nodes_by_id[group[0]].get("class_name", "")),
            )
        )

    pocket_nodes = [
        node
        for node in nodes
        if str(node.get("class_name", "")) in POCKET_CLASSES
        and str(node["feature_id"]) not in hole_ids
    ]
    for node in sorted(pocket_nodes, key=lambda n: int(n["feature_id"])):
        fid = str(node["feature_id"])
        gt_class = str(node.get("class_name", ""))
        if any(op.feature_refs == (fid,) for op in ops):
            continue
        ops.extend(
            [
                GtOperation(
                    feature_refs=(fid,),
                    operation_type="pocket_mill",
                    source="inferred",
                    gt_class=gt_class,
                ),
                GtOperation(
                    feature_refs=(fid,),
                    operation_type="finish_contour",
                    source="inferred",
                    gt_class=gt_class,
                ),
            ]
        )

    if not ops:
        for node in nodes:
            cls = str(node.get("class_name", ""))
            fid = str(node["feature_id"])
            if cls in POCKET_CLASSES:
                ops.extend(
                    [
                        GtOperation(
                            feature_refs=(fid,),
                            operation_type="pocket_mill",
                            source="inferred",
                            gt_class=cls,
                        ),
                        GtOperation(
                            feature_refs=(fid,),
                            operation_type="finish_contour",
                            source="inferred",
                            gt_class=cls,
                        ),
                    ]
                )
            elif cls in HOLE_CLASSES:
                group = (fid,)
                override = overrides.get(group, {})
                ops.append(
                    GtOperation(
                        feature_refs=group,
                        operation_type=str(override.get("operation_type", "drill")),
                        strategy=str(override["strategy"]) if override.get("strategy") else None,
                        tool_type=str(override["tool_type"]) if override.get("tool_type") else None,
                        diameter_mm=_node_diameter_mm(node),
                        notes=str(override["notes"]) if override.get("notes") else None,
                        source="inferred_override" if override else "inferred",
                        gt_class=cls,
                    )
                )

    return ops


def load_gt_operations(
    gt: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    part_id: str | None,
) -> tuple[list[GtOperation], str]:
    yaml_ops = _load_operations_from_yaml(gt)
    if yaml_ops:
        if any(op.source == "shop_program" for op in yaml_ops):
            return yaml_ops, "shop_program"
        return yaml_ops, "gt_yaml"
    inferred = infer_shop_operations(gt, graph, part_id=part_id)
    return inferred, "inferred"


# Canonical operation-bank slug -> legacy (operation_type, strategy) eval vocabulary.
# The planner collapsed operation_type+strategy into a single ``operation`` field;
# the eval scorecard still reasons in the old shop-ish vocabulary, so map back here.
_BANK_TO_EVAL_OPTYPE: dict[str, tuple[str, str]] = {
    "helix_bore": ("bore", "helical_bore"),
    "drill": ("drill", "peck_drill"),
    "chip_break_drill": ("drill", "peck_drill"),
    "thread_mill": ("tap", "rigid_tap"),
    "optirough": ("pocket_mill", "roughing"),
    "area_roughing": ("pocket_mill", "roughing"),
    "rest_roughing": ("pocket_mill", "roughing"),
    "pocket": ("pocket_mill", "roughing"),
    "dynamic_mill_2d": ("pocket_mill", "roughing"),
    "facing": ("facing", "facing"),
    "raster": ("floor_finish", "finishing_floor"),
    "constant_scallop": ("surface_finish", "finishing"),
    "radial_spiral": ("surface_finish", "finishing"),
    "steep_shallow": ("surface_finish", "finishing"),
    "rest_finish": ("surface_finish", "finishing"),
    "pencil": ("fillet_finish", "finishing_fillet"),
}
_WALL_FEATURE_CLASSES = frozenset({"wall", "profile"})


def _emitted_optype_strategy(operation: str, feature_type: str) -> tuple[str, str]:
    """Map a canonical bank op back to the eval's legacy (operation_type, strategy)."""
    if operation in ("contour_2d", "waterline"):
        if feature_type in _WALL_FEATURE_CLASSES:
            return ("wall_finish", "finishing_wall")
        return ("finish_contour", "finishing_floor")
    return _BANK_TO_EVAL_OPTYPE.get(operation, (operation, operation))


def load_emitted_operations(plan: CamPlan) -> list[EmittedOperation]:
    tool_lookup = {tool.tool_id: tool for tool in plan.tools}
    emitted: list[EmittedOperation] = []
    for op in plan.operations:
        tool = tool_lookup.get(op.tool_id)
        params = op.parameters
        op_type, op_strategy = _emitted_optype_strategy(op.operation, op.feature_type)
        emitted.append(
            EmittedOperation(
                op_id=op.op_id,
                feature_refs=_refs_key(op.feature_refs),
                operation_type=op_type,
                strategy=op_strategy,
                tool_id=op.tool_id,
                tool_type=tool.tool_type if tool is not None else None,
                tool_diameter_mm=tool.diameter_mm if tool is not None else None,
                sequence_index=op.sequence_index,
                spindle_rpm=params.spindle_rpm,
                feed_mm_per_min=params.feed_mm_per_min,
                plunge_mm_per_min=params.plunge_mm_per_min,
                feature_type=op.feature_type,
            )
        )
    return emitted


@dataclass
class OpMatch:
    gt: GtOperation
    emitted: EmittedOperation | None
    match_kind: str


def _match_operations(
    gt_ops: Sequence[GtOperation],
    emitted_ops: Sequence[EmittedOperation],
) -> tuple[list[OpMatch], list[EmittedOperation]]:
    """Match GT ops to emitted ops by feature_refs + operation phase."""
    emitted_by_refs: dict[tuple[str, ...], list[EmittedOperation]] = defaultdict(list)
    for op in emitted_ops:
        emitted_by_refs[op.feature_refs].append(op)

    used_emitted: set[str] = set()
    matches: list[OpMatch] = []

    for gt in gt_ops:
        candidates = emitted_by_refs.get(gt.feature_refs, [])
        gt_phase = _op_phase(gt.operation_type)

        chosen: EmittedOperation | None = None
        for cand in candidates:
            if cand.op_id in used_emitted:
                continue
            if _op_phase(cand.operation_type) == gt_phase:
                chosen = cand
                break

        if chosen is None:
            for cand in candidates:
                if cand.op_id not in used_emitted:
                    chosen = cand
                    break

        if chosen is not None:
            used_emitted.add(chosen.op_id)
            if (
                chosen.operation_type == gt.operation_type
                or _op_phase(chosen.operation_type) == gt_phase
            ):
                kind = "matched"
            else:
                kind = "phase_matched_type_mismatch"
            matches.append(OpMatch(gt=gt, emitted=chosen, match_kind=kind))
        else:
            matches.append(OpMatch(gt=gt, emitted=None, match_kind="missing"))

    extras = [op for op in emitted_ops if op.op_id not in used_emitted]
    return matches, extras


def _format_refs(refs: Sequence[str]) -> str:
    return ",".join(refs)


def _tool_agrees(gt: GtOperation, emitted: EmittedOperation) -> tuple[bool, str]:
    if emitted.tool_id == "UNRESOLVED":
        return False, "UNRESOLVED tool"

    gt_type = _normalize_eval_tool_type(gt.tool_type)
    emitted_type = _normalize_eval_tool_type(emitted.tool_type)
    if gt_type and emitted_type:
        if gt_type != emitted_type:
            # endmill vs bullnose are distinct in shop programs; don't conflate.
            return False, f"type {emitted_type} vs GT {gt_type}"

    if gt.diameter_mm is not None and emitted.tool_diameter_mm is not None:
        delta = abs(emitted.tool_diameter_mm - gt.diameter_mm)
        if delta > max(DIAMETER_TOL_MM, gt.diameter_mm * 0.05):
            return (
                False,
                f"dia {emitted.tool_diameter_mm:.3f} mm vs GT dia {gt.diameter_mm:.3f} mm "
                f"(delta {delta:.2f} mm)",
            )
        return True, f"dia {emitted.tool_diameter_mm:.3f} mm ~ GT dia {gt.diameter_mm:.3f} mm"

    if emitted_type:
        return True, (
            f"{emitted_type} dia {emitted.tool_diameter_mm or 0:.3f} mm "
            f"(GT tool unspecified)"
        )
    return False, "no tool info"


def _sequence_pairs(matches: Sequence[OpMatch]) -> list[tuple[OpMatch, OpMatch]]:
    pairs: list[tuple[OpMatch, OpMatch]] = []
    by_refs: dict[tuple[str, ...], list[OpMatch]] = defaultdict(list)
    for match in matches:
        if match.emitted is not None:
            by_refs[match.gt.feature_refs].append(match)

    for ref_matches in by_refs.values():
        rough = next((m for m in ref_matches if _op_phase(m.gt.operation_type) == "rough"), None)
        finish = next((m for m in ref_matches if _op_phase(m.gt.operation_type) == "finish"), None)
        if rough and finish:
            pairs.append((rough, finish))
    return pairs


def build_aggregate_scorecard(
    gt: Mapping[str, Any],
    gt_ops: Sequence[GtOperation],
    emitted_ops: Sequence[EmittedOperation],
    *,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate coverage vs shop program (no per-feature id matching)."""
    shop_strategies = {op.strategy for op in gt_ops if op.strategy}
    emitted_strategies = _emitted_shop_strategies(emitted_ops)
    strategy_covered = shop_strategies & emitted_strategies
    strategy_missed = sorted(shop_strategies - emitted_strategies)

    shop_categories = _shop_categories_from_gt(gt, gt_ops)
    emitted_categories = _emitted_shop_categories(emitted_ops, graph)
    category_covered = shop_categories & emitted_categories
    category_missed = sorted(shop_categories - emitted_categories)

    shop_tool_types = {
        normalized
        for op in gt_ops
        if op.tool_type and (normalized := _normalize_eval_tool_type(op.tool_type))
    }
    emitted_tool_types = {
        normalized
        for op in emitted_ops
        if op.tool_type and (normalized := _normalize_eval_tool_type(op.tool_type))
    }
    tool_covered = shop_tool_types & emitted_tool_types
    tool_missed = sorted(shop_tool_types - emitted_tool_types)
    tool_extra = sorted(emitted_tool_types - shop_tool_types)

    shop_feature_instances = sum(op.feature_count or 0 for op in gt_ops)

    feed_rows: list[dict[str, Any]] = []
    for strategy in sorted(strategy_covered):
        shop_feeds = [
            op.feed_mm_per_min for op in gt_ops if op.strategy == strategy and op.feed_mm_per_min
        ]
        emitted_feeds = [
            op.feed_mm_per_min
            for op in emitted_ops
            if _shop_strategy_from_emitted(op) == strategy and op.feed_mm_per_min
        ]
        shop_rpms = [
            op.spindle_rpm for op in gt_ops if op.strategy == strategy and op.spindle_rpm
        ]
        emitted_rpms = [
            op.spindle_rpm
            for op in emitted_ops
            if _shop_strategy_from_emitted(op) == strategy and op.spindle_rpm
        ]
        if not shop_feeds or not emitted_feeds:
            continue
        row: dict[str, Any] = {
            "strategy": strategy,
            "shop_ops": len(shop_feeds),
            "emitted_ops": len(emitted_feeds),
            "feed_ratio_emitted_over_shop": round(
                statistics.median(emitted_feeds) / statistics.median(shop_feeds),
                3,
            ),
        }
        if shop_rpms and emitted_rpms:
            row["rpm_ratio_emitted_over_shop"] = round(
                statistics.median(emitted_rpms) / statistics.median(shop_rpms),
                3,
            )
        feed_rows.append(row)

    return {
        "op_count": {
            "shop": len(gt_ops),
            "emitted": len(emitted_ops),
            "gap_emitted_minus_shop": len(emitted_ops) - len(gt_ops),
        },
        "shop_feature_instances": shop_feature_instances,
        "strategy": {
            "shop": sorted(shop_strategies),
            "emitted": sorted(emitted_strategies),
            "covered": sorted(strategy_covered),
            "missed": strategy_missed,
        },
        "feature_category": {
            "shop": sorted(shop_categories),
            "emitted": sorted(emitted_categories),
            "covered": sorted(category_covered),
            "missed": category_missed,
        },
        "tool_type": {
            "shop": sorted(shop_tool_types),
            "emitted": sorted(emitted_tool_types),
            "overlap": sorted(tool_covered),
            "shop_only": tool_missed,
            "emitted_only": tool_extra,
        },
        "feeds": feed_rows,
    }


def build_scorecard(
    plan: CamPlan,
    gt: Mapping[str, Any],
    gt_path: Path,
    graph: Mapping[str, Any],
) -> dict[str, Any]:
    schema = inspect_gt_schema(gt, gt_path)
    part_id = schema.part_id
    gt_ops, gt_op_source = load_gt_operations(gt, graph, part_id=part_id)
    emitted_ops = load_emitted_operations(plan)
    matches, extras = _match_operations(gt_ops, emitted_ops)

    matched = [m for m in matches if m.emitted is not None]
    missing = [m for m in matches if m.emitted is None]
    type_matches = [
        m
        for m in matched
        if m.emitted is not None and m.emitted.operation_type == m.gt.operation_type
    ]
    type_mismatches = [
        m
        for m in matched
        if m.emitted is not None and m.emitted.operation_type != m.gt.operation_type
    ]

    tool_checks: list[tuple[OpMatch, bool, str]] = []
    for match in matched:
        assert match.emitted is not None
        ok, detail = _tool_agrees(match.gt, match.emitted)
        tool_checks.append((match, ok, detail))

    tool_agree = sum(1 for _, ok, _ in tool_checks if ok)
    tool_total = len(tool_checks)

    seq_pairs = _sequence_pairs(matches)
    seq_agree = 0
    seq_inversions: list[str] = []
    for rough, finish in seq_pairs:
        assert rough.emitted is not None and finish.emitted is not None
        if rough.emitted.sequence_index < finish.emitted.sequence_index:
            seq_agree += 1
        else:
            seq_inversions.append(
                f"feature {_format_refs(rough.gt.feature_refs)}: "
                f"{rough.emitted.op_id}({rough.emitted.operation_type}) index "
                f"{rough.emitted.sequence_index} not before "
                f"{finish.emitted.op_id}({finish.emitted.operation_type}) "
                f"index {finish.emitted.sequence_index}"
            )

    param_rows: list[dict[str, Any]] = []
    for match in matched:
        assert match.emitted is not None
        gt = match.gt
        emitted = match.emitted
        row: dict[str, Any] = {
            "feature_refs": _format_refs(gt.feature_refs),
            "operation_type": emitted.operation_type,
            "gt_operation_type": gt.operation_type,
        }
        if gt.spindle_rpm and emitted.spindle_rpm:
            row["rpm_ratio_emitted_over_gt"] = round(emitted.spindle_rpm / gt.spindle_rpm, 3)
        if gt.feed_mm_per_min and emitted.feed_mm_per_min:
            row["feed_ratio_emitted_over_gt"] = round(
                emitted.feed_mm_per_min / gt.feed_mm_per_min,
                3,
            )
        if gt.plunge_mm_per_min and emitted.plunge_mm_per_min:
            row["plunge_ratio_emitted_over_gt"] = round(
                emitted.plunge_mm_per_min / gt.plunge_mm_per_min,
                3,
            )
        if len(row) > 3:
            param_rows.append(row)

    aggregate = None
    if gt_op_source == "shop_program":
        aggregate = build_aggregate_scorecard(gt, gt_ops, emitted_ops, graph=graph)

    return {
        "plan_path": str(plan.source_part),
        "gt_path": str(gt_path),
        "setup_id": plan.setups[0].setup_id if plan.setups else None,
        "schema": schema,
        "gt_operation_source": gt_op_source,
        "counts": {
            "gt_operations": len(gt_ops),
            "emitted_operations": len(emitted_ops),
            "matched": len(matched),
            "missing_from_plan": len(missing),
            "extra_in_plan": len(extras),
            "op_type_agreements": len(type_matches),
            "op_type_mismatches": len(type_mismatches),
            "tool_agreements": tool_agree,
            "tool_checks": tool_total,
            "sequence_pairs_checked": len(seq_pairs),
            "sequence_pairs_agree": seq_agree,
            "param_comparisons": len(param_rows),
        },
        "missing": missing,
        "extras": extras,
        "type_mismatches": type_mismatches,
        "tool_checks": tool_checks,
        "sequence_inversions": seq_inversions,
        "param_rows": param_rows,
        "gt_ops": gt_ops,
        "emitted_ops": emitted_ops,
        "aggregate": aggregate,
    }


def _pick_gt_yaml(plan: CamPlan, gt_arg: Path | None) -> Path:
    if gt_arg is not None:
        return gt_arg.resolve()

    setup_id = plan.setups[0].setup_id if plan.setups else ""
    if SHOP_PROGRAM_GT.is_file() and (
        len(plan.setups) > 1
        or not setup_id
        or "rear" in setup_id
    ):
        return SHOP_PROGRAM_GT

    candidates = sorted((REPO_ROOT / "eval" / "gt").glob("96260B_*.yaml"))
    if setup_id:
        for path in candidates:
            if setup_id in path.stem and path != SHOP_PROGRAM_GT:
                return path
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No eval/gt/96260B_*.yaml ground-truth file found.")


def _print_aggregate_scorecard(aggregate: Mapping[str, Any]) -> None:
    op = aggregate["op_count"]
    print()
    print("=== AGGREGATE SCORECARD (primary for shop program GT) ===")
    print()
    print("--- 1. Op count ---")
    print(
        f"Shop ops: {op['shop']}   Emitted: {op['emitted']}   "
        f"Gap (ours - shop): {op['gap_emitted_minus_shop']:+d}"
    )
    print(
        f"Shop feature-instances batched across ops: "
        f"{aggregate.get('shop_feature_instances', '?')}"
    )
    print("  Grouping is applied; remaining gap vs shop is multi-pass tool sizing / setup-2 facing.")

    strategy = aggregate["strategy"]
    print()
    print("--- 2. Strategy coverage ---")
    print(f"Shop strategies:    {', '.join(strategy['shop'])}")
    print(f"Emitted (mapped):   {', '.join(strategy['emitted']) or '(none)'}")
    print(f"Covered:            {', '.join(strategy['covered']) or '(none)'}")
    if strategy["missed"]:
        print(f"MISSED:             {', '.join(strategy['missed'])}")
    else:
        print("MISSED:             (none)")

    category = aggregate["feature_category"]
    print()
    print("--- 3. Feature-category coverage ---")
    print(f"Shop categories:    {', '.join(category['shop'])}")
    print(f"Emitted categories: {', '.join(category['emitted']) or '(none)'}")
    print(f"Covered:            {', '.join(category['covered']) or '(none)'}")
    if category["missed"]:
        print(f"MISSED:             {', '.join(category['missed'])}")
    else:
        print("MISSED:             (none)")

    tool = aggregate["tool_type"]
    print()
    print("--- 4. Tool-type coverage (type+diameter; labels may be truncated) ---")
    print(f"Shop tool types:    {', '.join(tool['shop'])}")
    print(f"Emitted tool types: {', '.join(tool['emitted']) or '(none)'}")
    print(f"Overlap:            {', '.join(tool['overlap']) or '(none)'}")
    if tool["shop_only"]:
        print(f"Shop-only gaps:     {', '.join(tool['shop_only'])}")
    else:
        print("Shop-only gaps:     (none)")
    if tool["emitted_only"]:
        print(f"Emitted-only:       {', '.join(tool['emitted_only'])}")

    print()
    print("--- 5. Feeds (informational; not pass/fail) ---")
    feed_rows: list[dict[str, Any]] = aggregate.get("feeds", [])
    if feed_rows:
        for row in feed_rows:
            parts = [
                f"{row['strategy']}: feed ratio {row['feed_ratio_emitted_over_shop']}",
                f"({row['emitted_ops']} emitted vs {row['shop_ops']} shop ops)",
            ]
            if "rpm_ratio_emitted_over_shop" in row:
                parts.append(f"rpm ratio {row['rpm_ratio_emitted_over_shop']}")
            print("  - " + " ".join(parts))
    else:
        print("  No overlapping strategies with feed data to compare.")


def print_scorecard(scorecard: Mapping[str, Any]) -> None:
    schema: GtSchemaReport = scorecard["schema"]
    counts = scorecard["counts"]
    aggregate: dict[str, Any] | None = scorecard.get("aggregate")

    print("=" * 72)
    print("CamPlan vs Toolpath ground truth")
    print("=" * 72)
    print(f"Setup:           {scorecard.get('setup_id')}")
    print(f"GT file:         {scorecard['gt_path']}")
    print(f"GT part_id:      {schema.part_id or scorecard.get('gt_part')}")
    print(f"GT schema:       counts={schema.has_counts} features={schema.has_features} "
          f"operations={schema.has_operations}")
    print(f"GT op source:    {scorecard['gt_operation_source']}")
    for note in schema.notes:
        print(f"  note: {note}")

    if aggregate is not None:
        _print_aggregate_scorecard(aggregate)
        print()
        print("--- Per-feature matching (unreliable for shop program GT) ---")

    print()
    print("--- Operation coverage (per-feature refs) ---")
    print(
        f"GT ops: {counts['gt_operations']}   Emitted: {counts['emitted_operations']}   "
        f"Matched: {counts['matched']}   Missing: {counts['missing_from_plan']}   "
        f"Extra: {counts['extra_in_plan']}"
    )

    missing: list[OpMatch] = scorecard["missing"]
    if missing:
        print("Missing from emitted plan:")
        for match in missing:
            gt = match.gt
            dia = f" dia={gt.diameter_mm:.1f}mm" if gt.diameter_mm else ""
            print(
                f"  - feature {_format_refs(gt.feature_refs)}: "
                f"GT={gt.operation_type}{dia} ({gt.source})"
            )
    else:
        print("Missing from emitted plan: (none)")

    extras: list[EmittedOperation] = scorecard["extras"]
    if extras:
        print("Extra in emitted plan (not in GT reference):")
        for op in extras:
            print(
                f"  - {op.op_id} feature {_format_refs(op.feature_refs)}: "
                f"{op.operation_type}/{op.strategy}"
            )
    else:
        print("Extra in emitted plan: (none)")

    print()
    print("--- 2. Op-type agreement ---")
    denom = counts["matched"] or 1
    rate = counts["op_type_agreements"] / denom
    print(
        f"Exact type match: {counts['op_type_agreements']}/{counts['matched']} "
        f"({rate:.0%} of matched ops)"
    )
    type_mismatches: list[OpMatch] = scorecard["type_mismatches"]
    if type_mismatches:
        print("Op-type mismatches:")
        for match in type_mismatches:
            assert match.emitted is not None
            gt = match.gt
            emitted = match.emitted
            dia = f" dia={gt.diameter_mm:.1f}mm" if gt.diameter_mm else ""
            print(
                f"  - feature {_format_refs(gt.feature_refs)}: "
                f"GT={gt.operation_type}{dia}, "
                f"ours={emitted.operation_type} ({emitted.op_id}, tool={emitted.tool_id})"
            )
            if gt.notes:
                print(f"      GT note: {gt.notes}")
    else:
        print("Op-type mismatches: (none)")

    print()
    print("--- 3. Tool agreement ---")
    tool_total = counts["tool_checks"] or 1
    tool_rate = counts["tool_agreements"] / tool_total
    print(
        f"Tool type/diameter agree: {counts['tool_agreements']}/{counts['tool_checks']} "
        f"({tool_rate:.0%} of matched ops with tool checks)"
    )
    tool_checks: list[tuple[OpMatch, bool, str]] = scorecard["tool_checks"]
    for match, ok, detail in tool_checks:
        if ok:
            continue
        assert match.emitted is not None
        print(
            f"  - feature {_format_refs(match.gt.feature_refs)} "
            f"{match.emitted.operation_type}: {detail}"
        )
    if counts["tool_agreements"] == counts["tool_checks"]:
        print("Tool mismatches: (none - GT tool specs mostly unspecified)")

    print()
    print("--- 4. Sequence agreement ---")
    seq_total = counts["sequence_pairs_checked"] or 1
    seq_rate = counts["sequence_pairs_agree"] / seq_total
    print(
        f"Rough-before-finish pairs: {counts['sequence_pairs_agree']}/"
        f"{counts['sequence_pairs_checked']} ({seq_rate:.0%})"
    )
    inversions: list[str] = scorecard["sequence_inversions"]
    if inversions:
        print("Sequence inversions:")
        for line in inversions:
            print(f"  - {line}")
    else:
        print("Sequence inversions: (none)")

    print()
    print("--- 5. Parameters (informational) ---")
    param_rows: list[dict[str, Any]] = scorecard["param_rows"]
    if param_rows:
        for row in param_rows:
            parts = [f"feature {row['feature_refs']} {row['operation_type']}"]
            if "rpm_ratio_emitted_over_gt" in row:
                parts.append(f"rpm-{row['rpm_ratio_emitted_over_gt']}")
            if "feed_ratio_emitted_over_gt" in row:
                parts.append(f"feed-{row['feed_ratio_emitted_over_gt']}")
            if "plunge_ratio_emitted_over_gt" in row:
                parts.append(f"plunge-{row['plunge_ratio_emitted_over_gt']}")
            print("  - " + " ".join(parts))
    else:
        print("No GT feed/speed values to compare (GT YAML lacks operation parameters).")
        stats = (
            scorecard["emitted_ops"][0]
            if scorecard["emitted_ops"]
            else None
        )
        if scorecard["emitted_ops"]:
            preset_ops = sum(
                1
                for op in scorecard["emitted_ops"]
                if op.op_id != "UNRESOLVED"
            )
            print(
                f"Emitted plan has feeds/speeds on {len(scorecard['emitted_ops'])} ops "
                f"(see cam_plan metadata for param_source breakdown)."
            )

    planner_stats = {}
    if isinstance(scorecard.get("plan_metadata"), Mapping):
        planner_stats = scorecard["plan_metadata"].get("planner_stats", {})
    if planner_stats:
        print(
            f"Emitted param_source: preset={planner_stats.get('params_toolpath_preset', 0)} "
            f"handbook={planner_stats.get('params_handbook_default', 0)} "
            f"mixed={planner_stats.get('params_mixed', 0)}"
        )

    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CamPlan against Toolpath GT YAML.")
    parser.add_argument(
        "plan",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "examples" / "cam_plan_96260B.json",
    )
    parser.add_argument(
        "gt",
        nargs="?",
        type=Path,
        default=None,
        help="Ground-truth YAML (default: auto-pick eval/gt/96260B_<setup>.yaml).",
    )
    args = parser.parse_args(argv)

    plan_path = _resolve_path(str(args.plan), REPO_ROOT)
    plan = load_cam_plan(plan_path)
    gt_path = _pick_gt_yaml(plan, args.gt)
    gt = _load_yaml(gt_path)

    graph_path = _resolve_path(plan.feature_graph_ref, gt_path)
    with graph_path.open(encoding="utf-8") as fh:
        graph = json.load(fh)

    scorecard = build_scorecard(plan, gt, gt_path, graph)
    scorecard["plan_metadata"] = plan.metadata
    print_scorecard(scorecard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
