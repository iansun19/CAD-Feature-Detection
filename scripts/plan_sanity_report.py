#!/usr/bin/env python3
"""Standing plan sanity report - single entry point for "did the plan come out sane?"

Combines preset replay (diagnose_preset_coverage), shop-GT feed comparison
(eval_cam_plan), and explicit sanity gates that catch validates-but-wrong bugs
(micro-tools, wrong-material presets, matcher misses, feed drift).

Run after every planner change:
  python scripts/plan_sanity_report.py
  python scripts/plan_sanity_report.py examples/cam_plan_96260B.json \\
      --shop eval/gt/96260B_rear_shop_program.yaml --material aluminum

Older scripts remain as thin wrappers / detail views:
  diagnose_preset_coverage.py - preset sparsity vs matcher_miss drill-down
  eval_cam_plan.py            - per-feature GT matching (unreliable for shop GT)

Exit code: 0 when no hard sanity gates fire; non-zero otherwise (CI gate).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_bootstrap  # noqa: F401, E402

from cam_plan_schema import CamPlan, Operation, load_cam_plan  # noqa: E402
from machining_context import Tool  # noqa: E402
from planner import (  # noqa: E402
    _material_match_rank,
    _tool_has_strategy_preset,
    resolve_preset,
)
from scripts.diagnose_preset_coverage import (  # noqa: E402
    ROUGH_OP_TYPES,
    OpDiagnosis,
    _best_strategy_preset,
    _legacy_optype_strategy,
    _load_plan_tools,
    _load_yaml,
    _operation_to_op_spec,
    _shop_strategy_from_op,
    _source_library_from_tool,
    diagnose_plan,
)
from scripts.eval_cam_plan import (  # noqa: E402
    GtOperation,
    build_aggregate_scorecard,
    load_emitted_operations,
    load_gt_operations,
    _resolve_path,
)

DEFAULT_PLAN = REPO_ROOT / "examples" / "cam_plan_96260B.json"
DEFAULT_SHOP = REPO_ROOT / "eval" / "gt" / "96260B_rear_shop_program.yaml"
COVERAGE_EXPECTATIONS_DIR = REPO_ROOT / "examples" / "coverage_expectations"

OPEN_FEATURE_TYPES = frozenset({"open_pocket", "filleted_open_pocket"})
FINISH_OP_TYPES = frozenset({
    "floor_finish",
    "wall_finish",
    "finish_contour",
    "surface_finish",
    "fillet_finish",
    "finish",
})
BORE_OP_TYPES = frozenset({"bore", "helical_bore"})
MILLING_TOOL_TYPES = frozenset({
    "endmill",
    "bullnose_endmill",
    "bullnose",
    "ball_endmill",
    "ball",
    "face_mill",
})
DRILL_TOOL_TYPES = frozenset({"drill"})
TAP_TOOL_TYPES = frozenset({"tap"})

MICRO_TOOL_MM = 3.0
BORE_DIA_MIN_MM = 6.0
BORE_DIA_MAX_MM = 30.0
FEED_OK_MIN = 0.5
FEED_OK_MAX = 1.5
FEED_HARD_MIN = 0.5
FEED_HARD_MAX = 2.0
HOMOLOG_OVERLAP_THRESHOLD = 0.5
SCOPED_FACING_STRATEGIES = frozenset({"facing"})


class FlagSeverity(StrEnum):
    HARD = "hard"
    WARN = "warn"


@dataclass
class SanityFlag:
    gate: str
    severity: FlagSeverity
    op_id: str
    message: str


@dataclass
class OpRow:
    op_id: str
    strategy: str
    operation_type: str
    tool_type: str | None
    tool_diameter_mm: float | None
    source_library: str | None
    chosen_preset: str | None
    preset_material: str | None
    param_source: str
    spindle_rpm: float | None
    feed_mm_per_min: float | None
    stepdown_mm: float | None
    stepover_mm: float | None
    feature_type: str


@dataclass
class FeedCompareRow:
    strategy: str
    op_id: str
    our_feed: float
    our_diameter_mm: float | None
    shop_feed: float
    shop_diameter_mm: float | None
    shop_op_index: int | None
    feed_ratio: float
    rpm_ratio: float | None
    compare_mode: str  # "size_matched" | "median" | "single"


@dataclass
class SetupCoverageVerdict:
    setup_id: str
    matches: bool
    emitted_strategies: list[str]
    expected_present: list[str]
    expected_absent: list[str]
    missing: list[str]
    unexpected: list[str]
    explained_absent: list[str]
    message: str


@dataclass
class SanityReport:
    plan_path: Path
    shop_path: Path | None
    material: str | None
    tool_source: str
    op_rows: list[OpRow] = field(default_factory=list)
    flags: list[SanityFlag] = field(default_factory=list)
    feed_rows: list[FeedCompareRow] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    diagnoses: list[OpDiagnosis] = field(default_factory=list)


def _tool_type_ok_for_op(
    tool_type: str | None,
    operation_type: str,
    strategy: str,
) -> bool:
    if tool_type is None:
        return True
    tt = tool_type.strip().lower()
    op = operation_type.strip().lower()
    strat = strategy.strip().lower()

    if op in ("drill", "peck_drill", "spot_drill"):
        return tt in DRILL_TOOL_TYPES
    if op == "tap":
        return tt in TAP_TOOL_TYPES
    if op in BORE_OP_TYPES or strat == "helical_bore":
        return tt in MILLING_TOOL_TYPES
    if op in ROUGH_OP_TYPES | FINISH_OP_TYPES | {"facing", "slot_mill"}:
        return tt in MILLING_TOOL_TYPES
    return True


def _is_open_feature_finish(op: Operation) -> bool:
    return (
        op.feature_type in OPEN_FEATURE_TYPES
        and _legacy_optype_strategy(op)[0] in FINISH_OP_TYPES
    )


# Shop strategy -> feature category for size-matched feed compare (when GT has categories).
STRATEGY_SHOP_CATEGORY: dict[str, str] = {
    "finishing_wall": "Wall",
    "finishing_floor": "Face",
    "facing": "Face",
}


def _shop_op_matches_category(op: GtOperation, category: str | None) -> bool:
    if not category:
        return True
    if category in op.feature_categories:
        return True
    return False


def _size_matched_shop_op(
    shop_ops: Sequence[GtOperation],
    strategy: str,
    diameter_mm: float | None,
    *,
    prefer_category: str | None = None,
) -> GtOperation | None:
    candidates = [
        op for op in shop_ops
        if op.strategy == strategy and op.feed_mm_per_min is not None
    ]
    if prefer_category:
        category_matches = [
            op for op in candidates if _shop_op_matches_category(op, prefer_category)
        ]
        if category_matches:
            candidates = category_matches
    if not candidates:
        return None
    if diameter_mm is None:
        return candidates[0]
    return min(
        candidates,
        key=lambda op: abs((op.diameter_mm or 0.0) - diameter_mm),
    )


def _shop_median_feed_rpm(
    shop_ops: Sequence[GtOperation],
    strategy: str,
) -> tuple[float | None, float | None]:
    feeds: list[float] = []
    rpms: list[float] = []
    for op in shop_ops:
        if op.strategy != strategy:
            continue
        if op.feed_mm_per_min is not None:
            feeds.append(op.feed_mm_per_min)
        if op.spindle_rpm is not None:
            rpms.append(op.spindle_rpm)
    return (
        statistics.median(feeds) if feeds else None,
        statistics.median(rpms) if rpms else None,
    )


def _has_same_material_strategy_preset(
    tool: Tool | None,
    op_spec: Any,
    material: str | None,
) -> bool:
    if tool is None:
        return False
    best = _best_strategy_preset(tool, op_spec, material)
    if best is None:
        return False
    return best.material_rank == 0


def _build_op_rows(
    plan: CamPlan,
    tools: Mapping[str, Tool],
    material: str | None,
) -> list[OpRow]:
    rows: list[OpRow] = []
    for op in plan.operations:
        tool = tools.get(op.tool_id)
        op_spec = _operation_to_op_spec(op, tool)
        preset = resolve_preset(tool, op_spec, material)
        params = op.parameters
        rows.append(
            OpRow(
                op_id=op.op_id,
                strategy=_legacy_optype_strategy(op)[1],
                operation_type=_legacy_optype_strategy(op)[0],
                tool_type=tool.tool_type if tool else None,
                tool_diameter_mm=tool.diameter_mm if tool else None,
                source_library=_source_library_from_tool(tool) if tool else None,
                chosen_preset=preset.preset_name if preset else None,
                preset_material=preset.preset_material if preset else None,
                param_source=params.param_source,
                spindle_rpm=params.spindle_rpm,
                feed_mm_per_min=params.feed_mm_per_min,
                stepdown_mm=params.stepdown_mm,
                stepover_mm=params.stepover_mm,
                feature_type=op.feature_type,
            )
        )
    return rows


def _feed_flag_severity(ratio: float) -> FlagSeverity | None:
    if FEED_OK_MIN <= ratio <= FEED_OK_MAX:
        return None
    if ratio < FEED_HARD_MIN or ratio > FEED_HARD_MAX:
        return FlagSeverity.HARD
    return FlagSeverity.WARN


def run_sanity_gates(
    plan: CamPlan,
    tools: Mapping[str, Tool],
    material: str | None,
    shop_ops: Sequence[GtOperation],
) -> list[SanityFlag]:
    flags: list[SanityFlag] = []

    for op in plan.operations:
        tool = tools.get(op.tool_id)
        op_spec = _operation_to_op_spec(op, tool)
        preset = resolve_preset(tool, op_spec, material)
        params = op.parameters
        dia = tool.diameter_mm if tool else None
        op_type, op_strategy = _legacy_optype_strategy(op)

        # Gate: micro-tool on open feature
        if _is_open_feature_finish(op) and dia is not None and dia < MICRO_TOOL_MM:
            flags.append(
                SanityFlag(
                    gate="micro_tool_open_feature",
                    severity=FlagSeverity.HARD,
                    op_id=op.op_id,
                    message=(
                        f"open-feature finish uses {dia:.2f} mm tool "
                        f"(<{MICRO_TOOL_MM} mm threshold)"
                    ),
                )
            )

        # Gate: tool type vs strategy/operation mismatch
        if not _tool_type_ok_for_op(
            tool.tool_type if tool else None,
            op_type,
            op_strategy,
        ):
            flags.append(
                SanityFlag(
                    gate="tool_type_mismatch",
                    severity=FlagSeverity.WARN,
                    op_id=op.op_id,
                    message=(
                        f"{tool.tool_type if tool else '?'} on "
                        f"{op_type}/{op_strategy}"
                    ),
                )
            )

        # Gate: bore/large-hole diameter band
        is_bore = (
            op_type in BORE_OP_TYPES
            or op_strategy == "helical_bore"
        )
        if is_bore and dia is not None and (dia < BORE_DIA_MIN_MM or dia > BORE_DIA_MAX_MM):
            flags.append(
                SanityFlag(
                    gate="bore_tool_diameter",
                    severity=FlagSeverity.WARN,
                    op_id=op.op_id,
                    message=(
                        f"bore tool diameter {dia:.2f} mm outside "
                        f"[{BORE_DIA_MIN_MM}, {BORE_DIA_MAX_MM}] mm band"
                    ),
                )
            )

        # Gate: handbook_default when strategy preset exists on tool
        if params.param_source == "handbook_default" and _tool_has_strategy_preset(
            tool, op_spec
        ) if tool else False:
            flags.append(
                SanityFlag(
                    gate="preset_available_not_used",
                    severity=FlagSeverity.HARD,
                    op_id=op.op_id,
                    message=(
                        "param_source=handbook_default but tool carries a "
                        "matching strategy preset"
                    ),
                )
            )

        # Gate: wrong-material preset when same-material preset was available
        if preset is not None and material is not None:
            chosen_rank = _material_match_rank(preset.preset_material, material)
            if chosen_rank > 1 and _has_same_material_strategy_preset(tool, op_spec, material):
                flags.append(
                    SanityFlag(
                        gate="wrong_material_preset",
                        severity=FlagSeverity.HARD,
                        op_id=op.op_id,
                        message=(
                            f"preset {preset.preset_name!r} material="
                            f"{preset.preset_material!r} on {material} job; "
                            f"same-material strategy preset exists on tool"
                        ),
                    )
                )

        # Gate: feed ratio vs size-matched shop op
        shop_strategy = _shop_strategy_from_op(op)
        if shop_strategy and params.feed_mm_per_min:
            prefer_cat = STRATEGY_SHOP_CATEGORY.get(shop_strategy or "")
            matched = _size_matched_shop_op(
                shop_ops, shop_strategy, dia, prefer_category=prefer_cat,
            )
            if matched and matched.feed_mm_per_min:
                ratio = params.feed_mm_per_min / matched.feed_mm_per_min
                sev = _feed_flag_severity(ratio)
                if sev is not None:
                    shop_dia = matched.diameter_mm
                    shop_idx = matched.sequence
                    flags.append(
                        SanityFlag(
                            gate="feed_off",
                            severity=sev,
                            op_id=op.op_id,
                            message=(
                                f"feed {params.feed_mm_per_min:.1f} mm/min vs shop "
                                f"{matched.feed_mm_per_min:.1f} mm/min "
                                f"(ratio {ratio:.3f}, size-matched shop op "
                                f"#{shop_idx} {shop_dia:.2f} mm dia "
                                f"vs our {dia:.2f} mm)"
                                if dia and shop_dia
                                else (
                                    f"feed ratio {ratio:.3f} vs size-matched shop "
                                    f"op #{shop_idx}"
                                )
                            ),
                        )
                    )

    return flags


def _setup_op_pairs(plan: CamPlan) -> dict[str, list[tuple[str, str, str]]]:
    """Map setup_id -> [(op_id, feature_id, operation_type), ...]."""
    by_setup: dict[str, list[tuple[str, str, str]]] = {}
    for op in plan.operations:
        operation_type = _legacy_optype_strategy(op)[0]
        for feature_id in op.feature_refs:
            by_setup.setdefault(op.setup_id, []).append(
                (op.op_id, feature_id, operation_type)
            )
    return by_setup


def check_homolog_overlap(plan: CamPlan) -> list[SanityFlag]:
    """Hard gate: homolog feature_ids machined with duplicate op types across setups."""
    by_setup = _setup_op_pairs(plan)
    if len(by_setup) < 2:
        return []

    pair_setups: dict[tuple[str, str], set[str]] = {}
    for setup_id, entries in by_setup.items():
        for _, feature_id, operation_type in entries:
            pair_setups.setdefault((feature_id, operation_type), set()).add(setup_id)

    flags: list[SanityFlag] = []
    for setup_id, entries in by_setup.items():
        if not entries:
            continue
        overlap = sum(
            1
            for _, feature_id, operation_type in entries
            if len(pair_setups.get((feature_id, operation_type), set())) > 1
        )
        ratio = overlap / len(entries)
        if ratio > HOMOLOG_OVERLAP_THRESHOLD:
            flags.append(
                SanityFlag(
                    gate="homolog_overlap",
                    severity=FlagSeverity.HARD,
                    op_id=setup_id,
                    message=(
                        f"setup {setup_id}: {ratio:.0%} of (feature_id, op_type) pairs "
                        f"duplicate another setup ({overlap}/{len(entries)} pairs) "
                        f"- split-panel over-machining signature"
                    ),
                )
            )
    return flags


def check_per_setup_op_counts(
    plan: CamPlan,
    shop_yaml: Mapping[str, Any] | None,
) -> list[SanityFlag]:
    """Warn when per-setup emitted op counts diverge from shop setup totals."""
    if shop_yaml is None:
        return []

    summary = shop_yaml.get("summary") or {}
    shop_counts: dict[str, int | None] = {
        "rear": summary.get("setup1_operation_count"),
        "front": summary.get("setup2_operation_count"),
    }
    emitted_counts = {
        setup.setup_id: sum(1 for op in plan.operations if op.setup_id == setup.setup_id)
        for setup in plan.setups
    }

    flags: list[SanityFlag] = []
    for setup_id, emitted in emitted_counts.items():
        shop_total = shop_counts.get(setup_id)
        if shop_total is None:
            continue
        if emitted == shop_total:
            continue
        severity = (
            FlagSeverity.HARD
            if setup_id == "front" and emitted > shop_total
            else FlagSeverity.WARN
        )
        flags.append(
            SanityFlag(
                gate="per_setup_op_count",
                severity=severity,
                op_id=setup_id,
                message=(
                    f"setup {setup_id}: emitted {emitted} ops vs shop setup "
                    f"{shop_total} ops"
                ),
            )
        )
    return flags


def check_setup_strategy_scope(plan: CamPlan) -> list[SanityFlag]:
    """Hard gate when a scoped setup emits strategies outside its declared scope."""
    per_setup = (plan.metadata.get("planner_stats") or {}).get("per_setup") or {}
    flags: list[SanityFlag] = []

    for setup in plan.setups:
        stats = per_setup.get(setup.setup_id) or {}
        if stats.get("scope_mode") != "filtered":
            continue
        scope_classes = stats.get("scope_classes") or []
        if not any(token in ("facing", "stock_face") for token in scope_classes):
            continue

        setup_ops = [op for op in plan.operations if op.setup_id == setup.setup_id]
        strategies = {_legacy_optype_strategy(op)[1] for op in setup_ops}
        extra = strategies - SCOPED_FACING_STRATEGIES
        if extra:
            flags.append(
                SanityFlag(
                    gate="setup_strategy_scope",
                    severity=FlagSeverity.HARD,
                    op_id=setup.setup_id,
                    message=(
                        f"setup {setup.setup_id} scope={scope_classes!r} but emitted "
                        f"strategies {sorted(strategies)!r}"
                    ),
                )
            )
    return flags


def _load_setup_graphs(
    plan: CamPlan,
    plan_path: Path,
    primary_graph: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    graphs: dict[str, Mapping[str, Any]] = {}
    refs = plan.metadata.get("feature_graph_refs") or {}
    if isinstance(refs, dict):
        for setup_id, ref in refs.items():
            graph_path = _resolve_path(str(ref), plan_path)
            if graph_path.is_file():
                with graph_path.open(encoding="utf-8") as fh:
                    graphs[setup_id] = json.load(fh)
    if primary_graph is not None and plan.setups:
        graphs.setdefault(plan.setups[0].setup_id, primary_graph)
    return graphs


def _envelope_faces_for_graph(
    graph: Mapping[str, Any],
    setup_id: str,
    plan: CamPlan,
) -> frozenset[int]:
    cached = graph.get("envelope_stock_face_ids")
    if isinstance(cached, list):
        return frozenset(int(i) for i in cached)

    per_setup = (plan.metadata.get("planner_stats") or {}).get("per_setup") or {}
    stats_faces = (per_setup.get(setup_id) or {}).get("envelope_stock_faces")
    if isinstance(stats_faces, list):
        return frozenset(int(i) for i in stats_faces)

    return frozenset()


def _feature_refs_on_envelope_stock(
    feature_refs: Sequence[str],
    graph: Mapping[str, Any],
    envelope_faces: frozenset[int],
) -> bool:
    if not envelope_faces:
        return False
    nodes = {
        str(node.get("feature_id")): node
        for node in graph.get("nodes", [])
    }
    for feature_id in feature_refs:
        node = nodes.get(str(feature_id))
        if node is None:
            return False
        params = node.get("params") or {}
        raw = params.get("face_indices") or params.get("face_ids") or []
        faces = frozenset(int(i) for i in raw)
        if not faces or not faces.issubset(envelope_faces):
            return False
    return True


def check_facing_stock_boundary(
    plan: CamPlan,
    plan_path: Path,
    primary_graph: Mapping[str, Any] | None,
) -> list[SanityFlag]:
    """Hard gate: facing ops must target envelope-coincident STOCK faces only."""
    graphs = _load_setup_graphs(plan, plan_path, primary_graph)
    flags: list[SanityFlag] = []
    for op in plan.operations:
        if _legacy_optype_strategy(op)[0] != "facing":
            continue
        graph = graphs.get(op.setup_id)
        if graph is None:
            continue
        envelope_faces = _envelope_faces_for_graph(graph, op.setup_id, plan)
        if not _feature_refs_on_envelope_stock(op.feature_refs, graph, envelope_faces):
            flags.append(
                SanityFlag(
                    gate="facing_stock_boundary",
                    severity=FlagSeverity.HARD,
                    op_id=op.op_id,
                    message=(
                        f"facing op targets feature_refs={list(op.feature_refs)!r} "
                        f"not on envelope STOCK faces "
                        f"(expected subset of {sorted(envelope_faces)!r})"
                    ),
                )
            )
    return flags


def run_multi_setup_gates(
    plan: CamPlan,
    *,
    plan_path: Path,
    shop_yaml: Mapping[str, Any] | None,
    primary_graph: Mapping[str, Any] | None,
) -> list[SanityFlag]:
    flags: list[SanityFlag] = []
    flags.extend(check_homolog_overlap(plan))
    flags.extend(check_per_setup_op_counts(plan, shop_yaml))
    flags.extend(check_setup_strategy_scope(plan))
    flags.extend(check_facing_stock_boundary(plan, plan_path, primary_graph))
    return flags


def _part_id_from_plan(plan: CamPlan) -> str:
    return str(plan.source_part or "").strip()


def load_coverage_expectations(part_id: str) -> dict[str, Any] | None:
    """Load per-part coverage manifest from examples/coverage_expectations/."""
    if not part_id:
        return None
    path = COVERAGE_EXPECTATIONS_DIR / f"{part_id}.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _emitted_shop_strategies_by_setup(plan: CamPlan) -> dict[str, set[str]]:
    by_setup: dict[str, set[str]] = {}
    for op in plan.operations:
        shop_strategy = _shop_strategy_from_op(op)
        if shop_strategy:
            by_setup.setdefault(op.setup_id, set()).add(shop_strategy)
    return by_setup


def _setup_expectations_from_manifest(
    manifest: Mapping[str, Any],
    setup_id: str,
) -> tuple[set[str], dict[str, str]]:
    setups = manifest.get("setups") or {}
    setup_spec = setups.get(setup_id) or {}
    present_block = setup_spec.get("expected_present") or {}
    absent_block = setup_spec.get("expected_absent") or {}

    expected_present = {
        str(name)
        for name in (present_block.get("strategies") or [])
    }
    expected_absent: dict[str, str] = {}
    for entry in absent_block.get("strategies") or []:
        if isinstance(entry, str):
            expected_absent[entry] = "expected absent per manifest"
        elif isinstance(entry, Mapping):
            name = str(entry.get("name") or "")
            if name:
                expected_absent[name] = str(
                    entry.get("reason") or "expected absent per manifest"
                )
    return expected_present, expected_absent


def _format_setup_coverage_message(
    setup_id: str,
    *,
    matches: bool,
    present: Sequence[str],
    expected_present: Sequence[str],
    expected_absent: Mapping[str, str],
    missing: Sequence[str],
    unexpected: Sequence[str],
    explained_absent: Sequence[str],
) -> str:
    if matches:
        present_str = ", ".join(present) or "(none)"
        if explained_absent and len(explained_absent) == 1:
            name = explained_absent[0]
            return (
                f"setup {setup_id}: coverage MATCHES expectations "
                f"({len(present)} present: {present_str}; "
                f"{name} expected-absent)"
            )
        if explained_absent and set(explained_absent) == set(expected_absent.keys()):
            sample_reason = next(iter(expected_absent.values()), "")
            return (
                f"setup {setup_id}: coverage MATCHES expectations "
                f"({present_str} present, rest expected-absent"
                + (f": {sample_reason}" if sample_reason else "")
                + ")"
            )
        if explained_absent:
            absent_bits = ", ".join(
                f"{name} expected-absent"
                for name in explained_absent
            )
            return (
                f"setup {setup_id}: coverage MATCHES expectations "
                f"({len(present)} present: {present_str}; {absent_bits})"
            )
        return (
            f"setup {setup_id}: coverage MATCHES expectations "
            f"({len(present)} present: {present_str})"
        )

    parts: list[str] = [f"setup {setup_id}: coverage FAIL"]
    detail_parts: list[str] = []
    if missing:
        detail_parts.append(
            f"{len(missing)} expected missing ({', '.join(missing)})"
        )
    if unexpected:
        detail_parts.append(
            f"{len(unexpected)} unexpected strategies ({', '.join(unexpected)})"
        )
    if detail_parts:
        return f"{parts[0]} ({'; '.join(detail_parts)})"
    return parts[0]


def check_coverage_expectations(
    plan: CamPlan,
    manifest: Mapping[str, Any] | None,
    *,
    part_id: str,
) -> tuple[list[SetupCoverageVerdict], list[SanityFlag]]:
    """Compare per-setup emitted shop strategies against manifest expectations."""
    emitted_by_setup = _emitted_shop_strategies_by_setup(plan)
    flags: list[SanityFlag] = []
    verdicts: list[SetupCoverageVerdict] = []

    if manifest is None:
        flags.append(
            SanityFlag(
                gate="coverage_expectations",
                severity=FlagSeverity.WARN,
                op_id=part_id or "(unknown)",
                message=f"no coverage expectations declared for {part_id or '(unknown part)'}",
            )
        )
        return verdicts, flags

    manifest_setups = manifest.get("setups") or {}
    setup_ids = sorted(
        set(manifest_setups.keys()) | {setup.setup_id for setup in plan.setups}
    )

    for setup_id in setup_ids:
        if setup_id not in manifest_setups:
            continue
        expected_present, expected_absent = _setup_expectations_from_manifest(
            manifest, setup_id,
        )
        emitted = emitted_by_setup.get(setup_id, set())
        missing = sorted(expected_present - emitted)
        unexpected = sorted(
            (emitted - expected_present)
            | (emitted & set(expected_absent.keys()))
        )
        explained_absent = sorted(set(expected_absent.keys()) - emitted)
        matches = not missing and not unexpected

        for strategy in missing:
            flags.append(
                SanityFlag(
                    gate="coverage_expectations",
                    severity=FlagSeverity.HARD,
                    op_id=setup_id,
                    message=(
                        f"setup {setup_id}: expected coverage missing "
                        f"strategy {strategy!r}"
                    ),
                )
            )
        for strategy in unexpected:
            reason = expected_absent.get(strategy)
            detail = f" ({reason})" if reason else ""
            flags.append(
                SanityFlag(
                    gate="coverage_expectations",
                    severity=FlagSeverity.WARN,
                    op_id=setup_id,
                    message=(
                        f"setup {setup_id}: unexpected strategy "
                        f"{strategy!r}{detail}"
                    ),
                )
            )

        present = sorted(emitted & expected_present)
        verdicts.append(
            SetupCoverageVerdict(
                setup_id=setup_id,
                matches=matches,
                emitted_strategies=sorted(emitted),
                expected_present=sorted(expected_present),
                expected_absent=sorted(expected_absent.keys()),
                missing=missing,
                unexpected=unexpected,
                explained_absent=explained_absent,
                message=_format_setup_coverage_message(
                    setup_id,
                    matches=matches,
                    present=present,
                    expected_present=sorted(expected_present),
                    expected_absent=expected_absent,
                    missing=missing,
                    unexpected=unexpected,
                    explained_absent=explained_absent,
                ),
            )
        )

    return verdicts, flags


def _coverage_verdict_pass(verdicts: Sequence[SetupCoverageVerdict]) -> bool:
    return bool(verdicts) and all(v.matches for v in verdicts)


def _build_feed_compare_rows(
    plan: CamPlan,
    tools: Mapping[str, Tool],
    shop_ops: Sequence[GtOperation],
) -> list[FeedCompareRow]:
    rows: list[FeedCompareRow] = []
    strategies_seen: set[str] = set()

    for op in plan.operations:
        shop_strategy = _shop_strategy_from_op(op)
        if not shop_strategy or not op.parameters.feed_mm_per_min:
            continue
        if shop_strategy in strategies_seen:
            continue
        strategies_seen.add(shop_strategy)

        tool = tools.get(op.tool_id)
        dia = tool.diameter_mm if tool else None
        our_feed = op.parameters.feed_mm_per_min
        our_rpm = op.parameters.spindle_rpm

        shop_for_strategy = [s for s in shop_ops if s.strategy == shop_strategy]
        multi_tool = len({s.diameter_mm for s in shop_for_strategy if s.diameter_mm}) > 1
        prefer_cat = STRATEGY_SHOP_CATEGORY.get(shop_strategy)

        matched = _size_matched_shop_op(
            shop_ops, shop_strategy, dia, prefer_category=prefer_cat,
        )
        med_feed, med_rpm = _shop_median_feed_rpm(shop_ops, shop_strategy)

        if matched and matched.feed_mm_per_min:
            shop_feed = matched.feed_mm_per_min
            shop_rpm = matched.spindle_rpm
            mode = "size_matched" if multi_tool else "single"
            rows.append(
                FeedCompareRow(
                    strategy=shop_strategy,
                    op_id=op.op_id,
                    our_feed=our_feed,
                    our_diameter_mm=dia,
                    shop_feed=shop_feed,
                    shop_diameter_mm=matched.diameter_mm,
                    shop_op_index=matched.sequence,
                    feed_ratio=round(our_feed / shop_feed, 3),
                    rpm_ratio=(
                        round(our_rpm / shop_rpm, 3)
                        if our_rpm and shop_rpm
                        else None
                    ),
                    compare_mode=mode,
                )
            )

        if multi_tool and med_feed:
            rows.append(
                FeedCompareRow(
                    strategy=shop_strategy,
                    op_id=op.op_id,
                    our_feed=our_feed,
                    our_diameter_mm=dia,
                    shop_feed=med_feed,
                    shop_diameter_mm=None,
                    shop_op_index=None,
                    feed_ratio=round(our_feed / med_feed, 3),
                    rpm_ratio=(
                        round(our_rpm / med_rpm, 3)
                        if our_rpm and med_rpm
                        else None
                    ),
                    compare_mode="median",
                )
            )

    return rows


def build_sanity_report(
    plan: CamPlan,
    *,
    plan_path: Path,
    shop_yaml: Mapping[str, Any] | None,
    shop_path: Path | None,
    material: str | None,
    graph: Mapping[str, Any] | None = None,
) -> SanityReport:
    tools, tool_source = _load_plan_tools(plan, material)
    diagnoses = diagnose_plan(
        plan,
        material=material,
        tools=tools,
        shop_yaml=shop_yaml,
    )
    op_rows = _build_op_rows(plan, tools, material)

    shop_ops: list[GtOperation] = []
    coverage: dict[str, Any] = {}
    if shop_yaml is not None:
        shop_ops, gt_source = load_gt_operations(
            shop_yaml,
            graph or {"nodes": []},
            part_id=str(shop_yaml.get("part")) if shop_yaml.get("part") else None,
        )
        if gt_source == "shop_program":
            emitted = load_emitted_operations(plan)
            aggregate = build_aggregate_scorecard(
                shop_yaml, shop_ops, emitted, graph=graph,
            )
            per_setup_counts = {
                setup.setup_id: sum(
                    1 for op in plan.operations if op.setup_id == setup.setup_id
                )
                for setup in plan.setups
            }
            per_setup_strategies = {
                setup.setup_id: sorted(
                    {_legacy_optype_strategy(op)[1] for op in plan.operations if op.setup_id == setup.setup_id}
                )
                for setup in plan.setups
            }
            cat = aggregate["feature_category"]
            strat = aggregate["strategy"]
            tool = aggregate["tool_type"]
            coverage = {
                "categories": f"{len(cat['covered'])}/{len(cat['shop'])}",
                "categories_covered": cat["covered"],
                "categories_missed": cat["missed"],
                "strategies": f"{len(strat['covered'])}/{len(strat['shop'])}",
                "strategies_covered": strat["covered"],
                "strategies_missed": strat["missed"],
                "tool_types_overlap": tool["overlap"],
                "tool_types_shop_only": tool["shop_only"],
                "per_setup_op_counts": per_setup_counts,
                "per_setup_strategies": per_setup_strategies,
            }

    flags = run_sanity_gates(plan, tools, material, shop_ops)
    flags.extend(
        run_multi_setup_gates(
            plan,
            plan_path=plan_path,
            shop_yaml=shop_yaml,
            primary_graph=graph,
        )
    )

    part_id = _part_id_from_plan(plan)
    manifest = load_coverage_expectations(part_id)
    setup_verdicts, coverage_flags = check_coverage_expectations(
        plan, manifest, part_id=part_id,
    )
    flags.extend(coverage_flags)
    if setup_verdicts:
        coverage["setup_verdicts"] = [
            {
                "setup_id": v.setup_id,
                "matches": v.matches,
                "message": v.message,
                "emitted_strategies": v.emitted_strategies,
                "missing": v.missing,
                "unexpected": v.unexpected,
                "explained_absent": v.explained_absent,
            }
            for v in setup_verdicts
        ]
        coverage["verdict"] = (
            "PASS" if _coverage_verdict_pass(setup_verdicts) else "FAIL"
        )
    elif manifest is None and part_id:
        coverage["verdict"] = "UNKNOWN"

    feed_rows = _build_feed_compare_rows(plan, tools, shop_ops)

    return SanityReport(
        plan_path=plan_path,
        shop_path=shop_path,
        material=material,
        tool_source=tool_source,
        op_rows=op_rows,
        flags=flags,
        feed_rows=feed_rows,
        coverage=coverage,
        diagnoses=list(diagnoses),
    )


def _hard_flags(flags: Sequence[SanityFlag]) -> list[SanityFlag]:
    return [f for f in flags if f.severity == FlagSeverity.HARD]


def print_sanity_report(report: SanityReport) -> None:
    print("=" * 100)
    print("PLAN SANITY REPORT")
    print(f"  plan:     {report.plan_path}")
    print(f"  shop GT:  {report.shop_path or '(none)'}")
    print(f"  material: {report.material}")
    print(f"  tools:    {report.tool_source}")
    print("=" * 100)

    # Section 1: per-op table
    print("\n--- 1. PER-OP TABLE ---")
    header = (
        f"{'op':<7} {'strategy':<18} {'op_type':<16} {'tool_type':<14} "
        f"{'dia_mm':>7} {'library':<28} {'preset':<22} {'p_mat':<10} "
        f"{'src':<18} {'rpm':>8} {'feed':>8} {'stepdn':>7} {'stepov':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in report.op_rows:
        lib = (row.source_library or "")[:28]
        preset = (row.chosen_preset or "None")[:22]
        pmat = (row.preset_material or "-")[:10]
        dia = f"{row.tool_diameter_mm:.2f}" if row.tool_diameter_mm else "-"
        rpm = f"{row.spindle_rpm:.0f}" if row.spindle_rpm else "-"
        feed = f"{row.feed_mm_per_min:.0f}" if row.feed_mm_per_min else "-"
        sdn = f"{row.stepdown_mm:.1f}" if row.stepdown_mm else "-"
        sov = f"{row.stepover_mm:.2f}" if row.stepover_mm else "-"
        print(
            f"{row.op_id:<7} {row.strategy:<18} {row.operation_type:<16} "
            f"{(row.tool_type or '-'):<14} {dia:>7} {lib:<28} {preset:<22} "
            f"{pmat:<10} {row.param_source:<18} {rpm:>8} {feed:>8} {sdn:>7} {sov:>7}"
        )

    # Section 2: sanity gates
    print("\n--- 2. SANITY GATES ---")
    if not report.flags:
        print("  (no flags)")
    else:
        for flag in report.flags:
            tag = flag.severity.upper()
            print(f"  [{tag}] {flag.gate} @ {flag.op_id}: {flag.message}")

    # Section 3: feed/rpm ratio table
    print("\n--- 3. FEED/RPM vs SHOP (per strategy) ---")
    if not report.feed_rows:
        print("  (no shop feed data)")
    else:
        fheader = (
            f"{'strategy':<18} {'op':<7} {'mode':<14} {'our_feed':>9} "
            f"{'shop_feed':>9} {'ratio':>7} {'rpm_r':>7} "
            f"{'our_dia':>8} {'shop_dia':>8} {'shop_op':>8}"
        )
        print(fheader)
        print("-" * len(fheader))
        for row in report.feed_rows:
            shop_dia = f"{row.shop_diameter_mm:.2f}" if row.shop_diameter_mm else "-"
            our_dia = f"{row.our_diameter_mm:.2f}" if row.our_diameter_mm else "-"
            shop_op = str(row.shop_op_index) if row.shop_op_index else "-"
            rpm_r = f"{row.rpm_ratio:.3f}" if row.rpm_ratio is not None else "-"
            print(
                f"{row.strategy:<18} {row.op_id:<7} {row.compare_mode:<14} "
                f"{row.our_feed:>9.1f} {row.shop_feed:>9.1f} {row.feed_ratio:>7.3f} "
                f"{rpm_r:>7} {our_dia:>8} {shop_dia:>8} {shop_op:>8}"
            )

    # Section 4: coverage
    print("\n--- 4. COVERAGE ---")
    cov = report.coverage
    if cov:
        setup_verdicts = cov.get("setup_verdicts") or []
        if setup_verdicts:
            for entry in setup_verdicts:
                print(f"  {entry['message']}")
            verdict = cov.get("verdict")
            if verdict:
                print(f"  overall coverage verdict: {verdict}")
        elif cov.get("verdict") == "UNKNOWN":
            part_id = "(unknown)"
            for flag in report.flags:
                if flag.gate == "coverage_expectations":
                    part_id = flag.op_id
                    break
            print(
                f"  (no coverage expectations manifest - using aggregate counts only)"
            )
        print(
            f"  feature categories: {cov.get('categories', '?')}  "
            f"({', '.join(cov.get('categories_covered', []))})"
        )
        if cov.get("categories_missed"):
            print(f"  categories missed:  {', '.join(cov['categories_missed'])}")
        print(
            f"  strategies (aggregate summary): {cov.get('strategies', '?')}  "
            f"({', '.join(cov.get('strategies_covered', []))})"
        )
        if cov.get("strategies_missed"):
            print(f"  strategies missed:  {', '.join(cov['strategies_missed'])}")
        overlap = cov.get("tool_types_overlap") or []
        shop_only = cov.get("tool_types_shop_only") or []
        print(f"  tool-type overlap:  {', '.join(overlap) or '(none)'}")
        if shop_only:
            print(f"  shop-only gaps:     {', '.join(shop_only)}")
        per_setup_counts = cov.get("per_setup_op_counts") or {}
        if per_setup_counts:
            print(
                "  per-setup op counts: "
                + ", ".join(f"{k}={v}" for k, v in sorted(per_setup_counts.items()))
            )
        per_setup_strategies = cov.get("per_setup_strategies") or {}
        if per_setup_strategies:
            for setup_id, strategies in sorted(per_setup_strategies.items()):
                print(
                    f"  setup {setup_id} strategies: {', '.join(strategies) or '(none)'}"
                )
    else:
        print("  (shop GT not loaded - coverage skipped)")

    # Section 5: summary
    hard = _hard_flags(report.flags)
    warn = [f for f in report.flags if f.severity == FlagSeverity.WARN]
    by_gate: dict[str, int] = {}
    for flag in report.flags:
        by_gate[flag.gate] = by_gate.get(flag.gate, 0) + 1

    print("\n--- 5. SUMMARY ---")
    print(f"  total flags: {len(report.flags)}  (hard={len(hard)}, warn={len(warn)})")
    if by_gate:
        print("  by gate: " + ", ".join(f"{k}={v}" for k, v in sorted(by_gate.items())))
    if hard:
        print("  HARD gates fired - exit non-zero:")
        for flag in hard:
            print(f"    - {flag.gate} @ {flag.op_id}: {flag.message}")
    else:
        print("  No hard gates - plan passes sanity check.")
    if warn:
        print(f"  Warnings ({len(warn)}):")
        for flag in warn:
            print(f"    - {flag.gate} @ {flag.op_id}: {flag.message}")

    print("=" * 100)


def exit_code_for_report(report: SanityReport) -> int:
    if _hard_flags(report.flags):
        return 1
    if report.coverage.get("verdict") == "FAIL":
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan sanity report: preset replay, shop feed compare, sanity gates.",
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
        help="Shop GT YAML for feed/coverage comparison",
    )
    parser.add_argument(
        "--material",
        default="aluminum",
        help="Workpiece material for preset selection (default: aluminum)",
    )
    args = parser.parse_args(argv)

    plan_path = args.plan if args.plan.is_absolute() else (REPO_ROOT / args.plan)
    shop_path = args.shop if args.shop.is_absolute() else (REPO_ROOT / args.shop)

    plan = load_cam_plan(plan_path)
    shop_yaml = _load_yaml(shop_path) if shop_path.is_file() else None
    if shop_yaml is None:
        print(f"Warning: shop GT not found at {shop_path}; feed/coverage gates limited.")

    graph: Mapping[str, Any] | None = None
    graph_path = _resolve_path(plan.feature_graph_ref, plan_path)
    if graph_path.is_file():
        with graph_path.open(encoding="utf-8") as fh:
            graph = json.load(fh)

    report = build_sanity_report(
        plan,
        plan_path=plan_path,
        shop_yaml=shop_yaml,
        shop_path=shop_path if shop_yaml else None,
        material=args.material,
        graph=graph,
    )
    print_sanity_report(report)
    return exit_code_for_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
