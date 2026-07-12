#!/usr/bin/env python3
"""Diagnose param_source=mixed on a CamPlan: preset coverage vs matcher misses.

Read-only diagnostic - replays resolve_preset / assign_parameters for each operation,
classifies sparse_tool vs matcher_miss, and compares feeds/rpm to shop GT.

Prefer ``scripts/plan_sanity_report.py`` for the standing sanity check (gates + CI exit).
This script remains a preset-sparsity / matcher_miss drill-down.

Run:
  python scripts/diagnose_preset_coverage.py
  python scripts/diagnose_preset_coverage.py examples/cam_plan_96260B.json \\
      --shop eval/gt/96260B_rear_shop_program.yaml --material aluminum
"""
from __future__ import annotations

import argparse
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_bootstrap  # noqa: F401, E402 - loads .env

from cam_plan_schema import CamPlan, Operation, load_cam_plan  # noqa: E402
from machining_context import (  # noqa: E402
    Tool,
    _library_name_from_path,
    load_tool_library,
)
from planner import (  # noqa: E402
    OpSpec,
    _handbook_parameters,
    _material_match_rank,
    _preset_name_material_rank,
    _strategy_match_rank,
    assign_parameters,
    resolve_preset,
)
from tool_store import create_supabase_client, row_to_tool  # noqa: E402

DEFAULT_PLAN = REPO_ROOT / "examples" / "cam_plan_96260B.json"
DEFAULT_SHOP = REPO_ROOT / "eval" / "gt" / "96260B_rear_shop_program.yaml"
LOCAL_LIBS_DIR = REPO_ROOT / "local_tool_libraries"
INCH_TO_MM = 25.4

SHOP_STRATEGIES = frozenset({
    "roughing",
    "finishing_floor",
    "finishing_wall",
    "finishing",
    "finishing_fillet",
    "facing",
})

ROUGH_OP_TYPES = frozenset({"pocket_mill", "adaptive_rough", "rough", "bore", "slot_mill"})

# Canonical operation-bank slug -> legacy (operation_type, strategy) vocabulary. The
# planner collapsed operation_type+strategy into one ``operation`` field; this diagnostic
# still reasons in the old shop-ish vocabulary, so map back at the schema boundary.
_BANK_TO_LEGACY_OPTYPE: dict[str, tuple[str, str]] = {
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


def _legacy_optype_strategy(op: Operation) -> tuple[str, str]:
    """Map a schema Operation's canonical bank op back to (operation_type, strategy)."""
    operation = op.operation
    if operation in ("contour_2d", "waterline"):
        if op.feature_type in _WALL_FEATURE_CLASSES:
            return ("wall_finish", "finishing_wall")
        return ("finish_contour", "finishing_floor")
    return _BANK_TO_LEGACY_OPTYPE.get(operation, (operation, operation))

MILLING_PARAM_FIELDS = (
    "spindle_rpm",
    "feed_mm_per_min",
    "plunge_mm_per_min",
    "stepdown_mm",
    "stepover_mm",
    "coolant",
)


@dataclass
class PresetRankRow:
    preset_name: str
    preset_material: str | None
    material_rank: int
    name_material_rank: int
    strategy_rank: int
    populated_fields: tuple[str, ...]
    completeness: int = 0


@dataclass
class OpDiagnosis:
    op_id: str
    operation_type: str
    strategy: str
    param_source: str
    tool_id: str
    tool_type: str | None
    diameter_mm: float | None
    source_library: str | None
    preset_count: int
    preset_catalog: list[tuple[str, str | None]]
    match_path: str
    chosen_preset: str | None
    should_have_preset: str | None
    preset_fields: list[str]
    handbook_fields: list[str]
    classification: str
    classification_detail: str
    feed_mm_per_min: float | None = None
    spindle_rpm: float | None = None
    shop_strategy: str | None = None
    shop_feed_mm_per_min: float | None = None
    shop_spindle_rpm: float | None = None
    feed_ratio_over_shop: float | None = None
    rpm_ratio_over_shop: float | None = None


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


def _source_library_from_tool_id(tool_id: str) -> str | None:
    if "::" not in tool_id:
        return None
    return tool_id.split("::", 1)[0]


def _source_library_from_tool(tool: Tool) -> str | None:
    source = tool.source or ""
    if source.startswith("supabase:"):
        return source.removeprefix("supabase:")
    if source.startswith("fusion_library:"):
        return source.removeprefix("fusion_library:")
    return _source_library_from_tool_id(tool.tool_id)


def _local_library_path(source_library: str) -> Path | None:
    """Find a local Fusion JSON file whose sanitized stem matches source_library."""
    if not LOCAL_LIBS_DIR.is_dir():
        return None
    for path in sorted(LOCAL_LIBS_DIR.glob("*.json")):
        if _library_name_from_path(path) == source_library:
            return path
    nested = LOCAL_LIBS_DIR / "1. AI CAM Tools"
    if nested.is_dir():
        for path in sorted(nested.glob("*.json")):
            if _library_name_from_path(path) == source_library:
                return path
    return None


def _fetch_tools_from_supabase(tool_ids: Sequence[str]) -> dict[str, Tool]:
    if not tool_ids:
        return {}
    try:
        client = create_supabase_client()
    except Exception as exc:
        print(f"Supabase unavailable ({exc}); falling back to local libraries.")
        return {}

    by_id: dict[str, Tool] = {}
    chunk_size = 50
    ids = list(dict.fromkeys(tool_ids))
    try:
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            response = client.table("tools").select("*").in_("tool_id", chunk).execute()
            for row in response.data or []:
                tool = row_to_tool(row)
                by_id[tool.tool_id] = tool
    except Exception as exc:
        print(f"Supabase fetch failed ({exc}); falling back to local libraries.")
        return {}
    return by_id


def _load_tools_local(tool_ids: Sequence[str], material: str | None) -> dict[str, Tool]:
    by_id: dict[str, Tool] = {}
    libraries: dict[str, Path] = {}
    for tool_id in tool_ids:
        lib_name = _source_library_from_tool_id(tool_id)
        if lib_name is None:
            continue
        if lib_name not in libraries:
            path = _local_library_path(lib_name)
            if path is not None:
                libraries[lib_name] = path

    for lib_name, path in libraries.items():
        for tool in load_tool_library(path, material=material):
            if tool.tool_id in tool_ids:
                by_id[tool.tool_id] = tool
    return by_id


def _load_plan_tools(plan: CamPlan, material: str | None) -> tuple[dict[str, Tool], str]:
    tool_ids = [op.tool_id for op in plan.operations if op.tool_id]
    from_supabase = _fetch_tools_from_supabase(tool_ids)
    if from_supabase and len(from_supabase) == len(set(tool_ids)):
        return from_supabase, "supabase"

    from_local = _load_tools_local(tool_ids, material)
    if from_local:
        missing = sorted(set(tool_ids) - set(from_local))
        if missing:
            print(f"Warning: {len(missing)} tool(s) missing from local libraries.")
        return from_local, "local_tool_libraries"

    return from_supabase or from_local, "partial"


def _operation_to_op_spec(op: Operation, tool: Tool | None) -> OpSpec:
    tool_type = tool.tool_type if tool is not None else "endmill"
    return OpSpec(
        op_id=op.op_id,
        feature_refs=list(op.feature_refs),
        feature_type=op.feature_type,
        setup_id=op.setup_id,
        operation=op.operation,
        tool_id=op.tool_id,
        tool_type_needed=tool_type,
    )


def _preset_value_for_field(preset: Any, field_name: str, rpm: float | None) -> Any:
    if field_name == "feed_mm_per_min":
        if preset.feed_mm_per_min is not None:
            return preset.feed_mm_per_min
        if preset.feed_per_rev_mm is not None and rpm is not None:
            return preset.feed_per_rev_mm * rpm
        return None
    return getattr(preset, field_name, None)


def _fields_populated_on_preset(
    preset: Any,
    op_spec: OpSpec,
    *,
    rpm: float | None = None,
) -> tuple[str, ...]:
    effective_rpm = rpm if rpm is not None else preset.spindle_rpm
    fields: list[str] = []
    if op_spec.tool_type_needed == "drill" or op_spec.operation == "drill":
        candidates = ("spindle_rpm", "feed_mm_per_min", "plunge_mm_per_min", "coolant")
    elif op_spec.tool_type_needed == "tap":
        candidates = ("spindle_rpm", "feed_mm_per_min", "plunge_mm_per_min", "coolant")
    else:
        candidates = MILLING_PARAM_FIELDS

    for name in candidates:
        if _preset_value_for_field(preset, name, effective_rpm) is not None:
            fields.append(name)
    return tuple(fields)


def _required_fields(op_spec: OpSpec) -> tuple[str, ...]:
    if op_spec.tool_type_needed == "drill" or op_spec.operation == "drill":
        return ("spindle_rpm", "feed_mm_per_min", "plunge_mm_per_min")
    if op_spec.tool_type_needed == "tap":
        return ("spindle_rpm", "feed_mm_per_min", "plunge_mm_per_min")
    return MILLING_PARAM_FIELDS[:-1]  # coolant optional for completeness check


def _rank_presets(
    tool: Tool,
    op_spec: OpSpec,
    material: str | None,
) -> list[PresetRankRow]:
    rows: list[PresetRankRow] = []
    for preset in tool.presets:
        rows.append(
            PresetRankRow(
                preset_name=preset.preset_name,
                preset_material=preset.preset_material,
                material_rank=_material_match_rank(preset.preset_material, material),
                name_material_rank=_preset_name_material_rank(preset.preset_name, material),
                strategy_rank=_strategy_match_rank(preset.preset_name, op_spec),
                populated_fields=_fields_populated_on_preset(preset, op_spec),
            )
        )
    rows.sort(
        key=lambda row: (
            row.material_rank,
            row.strategy_rank,
            row.name_material_rank,
            row.preset_name,
        )
    )
    return rows


def _explain_match_path(
    ranked: Sequence[PresetRankRow],
    chosen: PresetRankRow | None,
) -> str:
    if chosen is None:
        return "no presets on tool -> None"
    if not ranked:
        return "empty preset list"

    top = ranked[0]
    if top.preset_name != chosen.preset_name:
        return (
            f"sorted rank #1 is {top.preset_name!r} but resolve_preset returned "
            f"{chosen.preset_name!r} (unexpected)"
        )

    parts = [
        f"material_rank={chosen.material_rank}",
        f"strategy_rank={chosen.strategy_rank}",
        f"name_material_rank={chosen.name_material_rank}",
    ]
    if chosen.name_material_rank == 0 and chosen.strategy_rank > 0:
        parts.append("aluminum token in preset name beat strategy-specific preset")
    elif chosen.strategy_rank == 0:
        parts.append("best strategy affinity among tied presets")
    elif chosen.strategy_rank > 0:
        parts.append("no strategy-rank-0 preset won sort; fell through to generic")
    return "; ".join(parts)


def _best_strategy_preset(
    tool: Tool,
    op_spec: OpSpec,
    material: str | None,
) -> PresetRankRow | None:
    """Preset with strategy_rank==0 and acceptable material, most complete."""
    candidates: list[PresetRankRow] = []
    required = _required_fields(op_spec)
    for preset in tool.presets:
        mat_rank = _material_match_rank(preset.preset_material, material)
        strat_rank = _strategy_match_rank(preset.preset_name, op_spec)
        if strat_rank != 0 or mat_rank > 1:
            continue
        populated = _fields_populated_on_preset(preset, op_spec)
        completeness = sum(1 for f in required if f in populated)
        candidates.append(
            PresetRankRow(
                preset_name=preset.preset_name,
                preset_material=preset.preset_material,
                material_rank=mat_rank,
                name_material_rank=_preset_name_material_rank(preset.preset_name, material),
                strategy_rank=strat_rank,
                populated_fields=populated,
                completeness=completeness,
            )
        )

    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            -row.completeness,
            row.material_rank,
            row.name_material_rank,
            row.preset_name,
        )
    )
    return candidates[0]


def _field_provenance(
    op_spec: OpSpec,
    tool: Tool | None,
    material: str | None,
) -> tuple[list[str], list[str], Any | None]:
    """Return (preset_fields, handbook_fields, chosen_preset)."""
    diameter = tool.diameter_mm if tool is not None else 6.0
    handbook = _handbook_parameters(op_spec, diameter)
    preset = resolve_preset(tool, op_spec, material)

    preset_fields: set[str] = set()
    handbook_fields: set[str] = set()

    def pick(field: str, preset_val: Any, handbook_val: Any) -> None:
        if preset_val is not None:
            preset_fields.add(field)
        elif handbook_val is not None:
            handbook_fields.add(field)

    if op_spec.tool_type_needed == "drill" or op_spec.operation == "drill":
        rpm = preset.spindle_rpm if preset and preset.spindle_rpm else handbook.spindle_rpm
        pick("spindle_rpm", preset.spindle_rpm if preset else None, handbook.spindle_rpm)
        pick(
            "plunge_mm_per_min",
            preset.plunge_mm_per_min if preset else None,
            handbook.plunge_mm_per_min,
        )
        if preset and preset.feed_per_rev_mm is not None and rpm is not None:
            pick("feed_mm_per_min", preset.feed_per_rev_mm * rpm, handbook.feed_mm_per_min)
        else:
            pick(
                "feed_mm_per_min",
                preset.feed_mm_per_min if preset else None,
                handbook.feed_mm_per_min,
            )
        pick("coolant", preset.coolant if preset else None, handbook.coolant)
        return sorted(preset_fields), sorted(handbook_fields), preset

    if op_spec.tool_type_needed == "tap":
        rpm = preset.spindle_rpm if preset and preset.spindle_rpm else handbook.spindle_rpm
        pick("spindle_rpm", preset.spindle_rpm if preset else None, handbook.spindle_rpm)
        feed_from_preset = None
        if preset:
            if preset.feed_mm_per_min is not None:
                feed_from_preset = preset.feed_mm_per_min
            elif preset.feed_per_rev_mm is not None and rpm is not None:
                feed_from_preset = preset.feed_per_rev_mm * rpm
            elif preset.plunge_mm_per_min is not None:
                feed_from_preset = preset.plunge_mm_per_min
        pick("feed_mm_per_min", feed_from_preset, handbook.feed_mm_per_min)
        pick(
            "plunge_mm_per_min",
            preset.plunge_mm_per_min if preset else None,
            handbook.plunge_mm_per_min,
        )
        pick("coolant", preset.coolant if preset else None, handbook.coolant)
        return sorted(preset_fields), sorted(handbook_fields), preset

    pick("spindle_rpm", preset.spindle_rpm if preset else None, handbook.spindle_rpm)
    pick("feed_mm_per_min", preset.feed_mm_per_min if preset else None, handbook.feed_mm_per_min)
    pick(
        "plunge_mm_per_min",
        preset.plunge_mm_per_min if preset else None,
        handbook.plunge_mm_per_min,
    )
    pick("stepdown_mm", preset.stepdown_mm if preset else None, handbook.stepdown_mm)
    pick("stepover_mm", preset.stepover_mm if preset else None, handbook.stepover_mm)
    pick("coolant", preset.coolant if preset else None, handbook.coolant)
    return sorted(preset_fields), sorted(handbook_fields), preset


def _classify_op(
    tool: Tool | None,
    op_spec: OpSpec,
    material: str | None,
    chosen_name: str | None,
    preset_fields: Sequence[str],
    handbook_fields: Sequence[str],
) -> tuple[str, str]:
    if tool is None or not tool.presets:
        return (
            "sparse_tool",
            "tool missing or has zero presets after material filter",
        )

    required = _required_fields(op_spec)
    best_strat = _best_strategy_preset(tool, op_spec, material)

    if best_strat is None:
        if len(tool.presets) <= 2:
            return (
                "sparse_tool",
                f"only {len(tool.presets)} preset(s) and none match strategy affinity rank 0",
            )
        return (
            "sparse_tool",
            "no strategy-plausible preset (strategy_rank=0) on this tool",
        )

    best_complete = all(f in best_strat.populated_fields for f in required)

    if chosen_name != best_strat.preset_name:
        missing_if_best = [f for f in required if f not in preset_fields]
        missing_if_best_named = [
            f for f in required if f not in best_strat.populated_fields
        ]
        return (
            "matcher_miss",
            (
                f"resolve_preset chose {chosen_name!r} but {best_strat.preset_name!r} "
                f"has strategy_rank=0 and fields {list(best_strat.populated_fields)} "
                f"(chosen missing {missing_if_best}; best-strategy missing "
                f"{missing_if_best_named})"
            ),
        )

    if handbook_fields and not best_complete:
        missing = [f for f in required if f in handbook_fields]
        return (
            "sparse_tool",
            (
                f"matcher picked best strategy preset {chosen_name!r} but it lacks "
                f"{missing} - mixed is honest"
            ),
        )

    if handbook_fields:
        return (
            "sparse_tool",
            f"chosen preset {chosen_name!r} incomplete; handbook filled {list(handbook_fields)}",
        )

    return ("sparse_tool", "full preset coverage (unexpected if param_source is mixed)")


def _shop_strategy_from_op(op: Operation) -> str | None:
    operation_type, strategy = _legacy_optype_strategy(op)
    if strategy in SHOP_STRATEGIES:
        return strategy
    if operation_type in ROUGH_OP_TYPES:
        return "roughing"
    if operation_type == "floor_finish":
        return "finishing_floor"
    if operation_type == "wall_finish":
        return "finishing_wall"
    if operation_type in ("surface_finish", "finish_contour", "finish"):
        return "finishing"
    if operation_type == "fillet_finish":
        return "finishing_fillet"
    if operation_type == "facing":
        return "facing"
    if strategy == "helical_bore":
        return "finishing_wall"
    return None


def _shop_ops_by_strategy(shop_yaml: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    setups = shop_yaml.get("setups") or []
    if not isinstance(setups, list):
        return by_strategy
    for setup in setups:
        if not isinstance(setup, Mapping):
            continue
        ops = setup.get("operations") or []
        if not isinstance(ops, list):
            continue
        for op in ops:
            if not isinstance(op, Mapping):
                continue
            strategy = op.get("strategy")
            if isinstance(strategy, str):
                by_strategy.setdefault(strategy, []).append(dict(op))
    return by_strategy


def _shop_median_feed_rpm(ops: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    feeds: list[float] = []
    rpms: list[float] = []
    for op in ops:
        feed_in = op.get("feed_in_min")
        rpm = op.get("spindle_rpm")
        if feed_in is not None:
            feeds.append(float(feed_in) * INCH_TO_MM)
        if rpm is not None:
            rpms.append(float(rpm))
    feed_med = statistics.median(feeds) if feeds else None
    rpm_med = statistics.median(rpms) if rpms else None
    return feed_med, rpm_med


def diagnose_plan(
    plan: CamPlan,
    *,
    material: str | None,
    tools: Mapping[str, Tool],
    shop_yaml: Mapping[str, Any] | None,
) -> list[OpDiagnosis]:
    from machining_context import MachiningContext, SetupContext, Stock

    ctx = MachiningContext(
        part_id=plan.source_part or "plan",
        feature_graph_ref=plan.feature_graph_ref,
        stock=Stock(bbox_min=(0.0, 0.0, 0.0), bbox_max=(1.0, 1.0, 1.0)),
        setups=[SetupContext(
            setup_id="diag",
            opening_axis="+Y",
            opening_axis_vector=(0.0, 1.0, 0.0),
            machining_side="front",
        )],
        tools=list(tools.values()),
        material=material,
    )

    shop_by_strategy = _shop_ops_by_strategy(shop_yaml) if shop_yaml else {}
    shop_medians: dict[str, tuple[float | None, float | None]] = {
        strategy: _shop_median_feed_rpm(ops)
        for strategy, ops in shop_by_strategy.items()
    }

    results: list[OpDiagnosis] = []
    for op in plan.operations:
        tool = tools.get(op.tool_id)
        op_spec = _operation_to_op_spec(op, tool)
        ranked = _rank_presets(tool, op_spec, material) if tool else []
        chosen_preset = resolve_preset(tool, op_spec, material)
        chosen_row = next(
            (row for row in ranked if row.preset_name == chosen_preset.preset_name),
            ranked[0] if ranked and chosen_preset else None,
        ) if chosen_preset and ranked else None

        preset_fields, handbook_fields, _ = _field_provenance(op_spec, tool, material)
        params = assign_parameters(op_spec, tool, ctx)
        classification, detail = _classify_op(
            tool,
            op_spec,
            material,
            chosen_preset.preset_name if chosen_preset else None,
            preset_fields,
            handbook_fields,
        )

        best_strat = _best_strategy_preset(tool, op_spec, material) if tool else None
        shop_strategy = _shop_strategy_from_op(op)
        shop_feed, shop_rpm = (
            shop_medians.get(shop_strategy, (None, None))
            if shop_strategy
            else (None, None)
        )
        feed_ratio = (
            round(params.feed_mm_per_min / shop_feed, 3)
            if params.feed_mm_per_min and shop_feed
            else None
        )
        rpm_ratio = (
            round(params.spindle_rpm / shop_rpm, 3)
            if params.spindle_rpm and shop_rpm
            else None
        )

        preset_catalog = [
            (p.preset_name, p.preset_material)
            for p in (tool.presets if tool else [])
        ]

        results.append(
            OpDiagnosis(
                op_id=op.op_id,
                operation_type=_legacy_optype_strategy(op)[0],
                strategy=_legacy_optype_strategy(op)[1],
                param_source=params.param_source,
                tool_id=op.tool_id,
                tool_type=tool.tool_type if tool else None,
                diameter_mm=tool.diameter_mm if tool else None,
                source_library=_source_library_from_tool(tool) if tool else None,
                preset_count=len(tool.presets) if tool else 0,
                preset_catalog=preset_catalog,
                match_path=_explain_match_path(ranked, chosen_row),
                chosen_preset=chosen_preset.preset_name if chosen_preset else None,
                should_have_preset=best_strat.preset_name if best_strat else None,
                preset_fields=list(preset_fields),
                handbook_fields=list(handbook_fields),
                classification=classification,
                classification_detail=detail,
                feed_mm_per_min=params.feed_mm_per_min,
                spindle_rpm=params.spindle_rpm,
                shop_strategy=shop_strategy,
                shop_feed_mm_per_min=shop_feed,
                shop_spindle_rpm=shop_rpm,
                feed_ratio_over_shop=feed_ratio,
                rpm_ratio_over_shop=rpm_ratio,
            )
        )
    return results


def _print_report(
    diagnoses: Sequence[OpDiagnosis],
    *,
    plan_path: Path,
    tool_source: str,
    material: str | None,
) -> None:
    matcher = sum(1 for d in diagnoses if d.classification == "matcher_miss")
    sparse = sum(1 for d in diagnoses if d.classification == "sparse_tool")

    print("=" * 88)
    print("PRESET COVERAGE DIAGNOSIS")
    print(f"  plan:     {plan_path}")
    print(f"  material: {material}")
    print(f"  tools:    {tool_source}")
    print(f"  ops:      {len(diagnoses)}")
    print("=" * 88)

    header = (
        f"{'op':<7} {'type/strategy':<28} {'param_src':<18} "
        f"{'class':<14} {'presets':>7} {'chosen':<32}"
    )
    print(header)
    print("-" * len(header))

    for d in diagnoses:
        type_str = f"{d.operation_type}/{d.strategy}"[:28]
        chosen = (d.chosen_preset or "None")[:32]
        print(
            f"{d.op_id:<7} {type_str:<28} {d.param_source:<18} "
            f"{d.classification:<14} {d.preset_count:>7} {chosen:<32}"
        )

    print("\n" + "=" * 88)
    print("PER-OP DETAIL")
    print("=" * 88)

    for d in diagnoses:
        dia_in = d.diameter_mm / INCH_TO_MM if d.diameter_mm else None
        print(f"\n--- {d.op_id} {d.operation_type} / {d.strategy} ---")
        print(f"  param_source (plan): {d.param_source}")
        print(f"  classification:      {d.classification} - {d.classification_detail}")
        print(
            f"  tool library: {d.source_library}  |  preset: {d.chosen_preset!r}  "
            f"(param_source={d.param_source})"
        )
        print(f"  preset count on tool: {d.preset_count}")
        if d.preset_catalog:
            print("  preset names (material):")
            for name, mat in d.preset_catalog:
                print(f"    - {name}  [{mat}]")
        print(f"  resolve_preset -> {d.chosen_preset!r}")
        print(f"  match path: {d.match_path}")
        if d.should_have_preset and d.should_have_preset != d.chosen_preset:
            print(f"  should have matched: {d.should_have_preset!r}")
        print(f"  fields from preset:   {d.preset_fields or '(none)'}")
        print(f"  fields from handbook: {d.handbook_fields or '(none)'}")
        if d.shop_strategy and d.shop_feed_mm_per_min and d.feed_mm_per_min is not None:
            print(
                f"  shop compare ({d.shop_strategy}): "
                f"feed {d.feed_mm_per_min:.1f} vs shop median {d.shop_feed_mm_per_min:.1f} mm/min "
                f"(ratio {d.feed_ratio_over_shop}) | "
                f"rpm {d.spindle_rpm:.1f} vs shop {d.shop_spindle_rpm:.1f} "
                f"(ratio {d.rpm_ratio_over_shop})"
            )

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"  matcher_miss: {matcher}")
    print(f"  sparse_tool:  {sparse}")

    if matcher:
        print("\n  Actionable matcher misses:")
        for d in diagnoses:
            if d.classification != "matcher_miss":
                continue
            print(
                f"    {d.op_id}: chose {d.chosen_preset!r} instead of "
                f"{d.should_have_preset!r} - {d.match_path}"
            )
    else:
        print("\n  All ops classified sparse_tool: param_source=mixed is honest; no matcher fix needed.")

    print("\n  Feed/RPM vs shop (median by strategy):")
    strategies_seen: set[str] = set()
    for d in diagnoses:
        if not d.shop_strategy or d.shop_strategy in strategies_seen:
            continue
        strategies_seen.add(d.shop_strategy)
        same = [x for x in diagnoses if x.shop_strategy == d.shop_strategy]
        feeds = [x.feed_ratio_over_shop for x in same if x.feed_ratio_over_shop]
        rpms = [x.rpm_ratio_over_shop for x in same if x.rpm_ratio_over_shop]
        if feeds:
            feed_str = f"feed ratio median={statistics.median(feeds):.3f} (ops={len(feeds)})"
        else:
            feed_str = "feed ratio n/a"
        if rpms:
            rpm_str = f"rpm ratio median={statistics.median(rpms):.3f}"
        else:
            rpm_str = "rpm ratio n/a"
        print(f"    {d.shop_strategy}: {feed_str}; {rpm_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose param_source=mixed: preset sparsity vs matcher miss.",
    )
    parser.add_argument(
        "plan",
        nargs="?",
        type=Path,
        default=DEFAULT_PLAN,
        help=f"CamPlan JSON (default: {DEFAULT_PLAN.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--shop",
        type=Path,
        default=DEFAULT_SHOP,
        help="Shop GT YAML for feed/rpm comparison",
    )
    parser.add_argument(
        "--material",
        default="aluminum",
        help="Workpiece material for preset selection (default: aluminum)",
    )
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else (REPO_ROOT / args.plan)
    shop_path = args.shop if args.shop.is_absolute() else (REPO_ROOT / args.shop)

    plan = load_cam_plan(plan_path)
    tools, tool_source = _load_plan_tools(plan, args.material)
    shop_yaml = _load_yaml(shop_path) if shop_path.is_file() else None
    if shop_yaml is None:
        print(f"Warning: shop GT not found at {shop_path}; skipping feed comparison.")

    diagnoses = diagnose_plan(
        plan,
        material=args.material,
        tools=tools,
        shop_yaml=shop_yaml,
    )
    _print_report(
        diagnoses,
        plan_path=plan_path,
        tool_source=tool_source,
        material=args.material,
    )


if __name__ == "__main__":
    main()
