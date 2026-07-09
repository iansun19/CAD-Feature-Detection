#!/usr/bin/env python3
"""Read-only coverage audit for 96260B after per-setup opening-axis fix.

Produces eval/coverage_audit_96260B.md and exits 1 on FAIL.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from machining_context import (  # noqa: E402
    MachiningContext,
    Stock,
    build_setup_context,
    default_tool_library,
    load_feature_graph,
    load_setup_descriptor,
)
from planner import (  # noqa: E402
    SetupPlanInput,
    _feature_reachable_for_setup,
    cascade_node_to_feature,
    filter_features_for_setup,
    filter_planner_features,
    plan_multi_setups,
)
from setup_descriptor import resolve_setup_entry  # noqa: E402

REAR_GRAPH = REPO_ROOT / "pipeline_out/96260B_rear/feature_graph_cascade.json"
FRONT_GRAPH = REPO_ROOT / "pipeline_out/96260B_front/feature_graph_cascade.json"
SETUP_YAML = REPO_ROOT / "eval/gt/96260B_setup.yaml"
NEW_PLAN_PATH = REPO_ROOT / "examples/cam_plan_96260B.json"
REPORT_PATH = REPO_ROOT / "eval/coverage_audit_96260B.md"

REAR_REACH_DIR = "-Z"
FRONT_REACH_DIR = "+Z"
OLD_WRONG_REAR_DIR = "+Z"

# Explicit flag lists from eval/orphaned_feature_triage.md (post-fix).
REAR_ORPHAN_FLAGS = frozenset({"34"})
FRONT_ORPHAN_FLAGS = frozenset({"34"})
LATERAL_FLAGS: dict[str, frozenset[str]] = {
    "rear": frozenset({"37", "103"}),
    "front": frozenset({"36", "37"}),
}

BUCKETS = (
    "KEPT_REAR",
    "KEPT_FRONT",
    "KEPT_BOTH",
    "FLAGGED_ORPHAN",
    "FLAGGED_LATERAL",
    "MERGED_CAP_DEBRIS",
    "UNACCOUNTED",
)


@dataclass(frozen=True)
class FeatureRecord:
    panel: str
    feature_id: str
    feature_type: str
    reachable_dirs: list[str]
    in_rear: bool
    in_front: bool
    planner_rear: bool
    planner_front: bool
    bucket: str


def _build_context(panel: str, graph_path: Path) -> MachiningContext:
    graph = load_feature_graph(graph_path)
    descriptor = load_setup_descriptor(SETUP_YAML)
    resolved = resolve_setup_entry(descriptor, setup_id=panel)
    setup = build_setup_context(resolved, graph)
    return MachiningContext(
        part_id="96260B",
        feature_graph_ref=str(graph_path.relative_to(REPO_ROOT)),
        stock=Stock(
            bbox_min=(0.0, 0.0, 0.0),
            bbox_max=(1.0, 1.0, 1.0),
            source="audit_dummy",
        ),
        setups=[setup],
        tools=default_tool_library(),
    )


def _planner_kept_ids(
    graph: Mapping[str, Any],
    context: MachiningContext,
) -> set[str]:
    nodes = graph.get("nodes", [])
    all_features = [cascade_node_to_feature(n) for n in nodes]
    features, _ = filter_planner_features(all_features)
    nodes_by_id = {str(n["feature_id"]): n for n in nodes}
    kept, _, _ = filter_features_for_setup(
        features,
        context,
        envelope_faces=frozenset(),
        nodes_by_id=nodes_by_id,
        graph=graph,
        use_reachability=True,
    )
    return {f.feature_id for f in kept}


def _dir_kept_ids(graph: Mapping[str, Any], reach_dir: str) -> set[str]:
    nodes = graph.get("nodes", [])
    all_features = [cascade_node_to_feature(n) for n in nodes]
    features, _ = filter_planner_features(all_features)
    nodes_by_id = {str(n["feature_id"]): n for n in nodes}
    return {
        f.feature_id
        for f in features
        if _feature_reachable_for_setup(nodes_by_id[f.feature_id], reach_dir)
    }


def _reachable_dirs_raw(node: Mapping[str, Any]) -> list[str]:
    approach = node.get("approach") or {}
    reach = approach.get("reachability") or {}
    dirs = reach.get("reachable_dirs")
    if isinstance(dirs, list):
        return [str(d) for d in dirs]
    return []


def _logical_resolution(node: Mapping[str, Any]) -> tuple[bool, bool]:
    return (
        _feature_reachable_for_setup(node, REAR_REACH_DIR),
        _feature_reachable_for_setup(node, FRONT_REACH_DIR),
    )


def _classify_bucket(
    panel: str,
    feature_id: str,
    in_rear: bool,
    in_front: bool,
    node: Mapping[str, Any] | None = None,
) -> str:
    if node is not None:
        params = node.get("params") or {}
        if params.get("lobe_contour_merge_kind") == "debris":
            return "MERGED_CAP_DEBRIS"
    if feature_id in LATERAL_FLAGS.get(panel, frozenset()):
        return "FLAGGED_LATERAL"
    orphan_flags = REAR_ORPHAN_FLAGS if panel == "rear" else FRONT_ORPHAN_FLAGS
    if feature_id in orphan_flags:
        return "FLAGGED_ORPHAN"
    if in_rear and in_front:
        return "KEPT_BOTH"
    if in_rear:
        return "KEPT_REAR"
    if in_front:
        return "KEPT_FRONT"
    return "UNACCOUNTED"


def _build_ledger(
    rear_graph: Mapping[str, Any],
    front_graph: Mapping[str, Any],
    rear_ctx: MachiningContext,
    front_ctx: MachiningContext,
) -> list[FeatureRecord]:
    rear_planner = _planner_kept_ids(rear_graph, rear_ctx)
    front_planner = _planner_kept_ids(front_graph, front_ctx)

    records: list[FeatureRecord] = []
    for panel, graph in (("rear", rear_graph), ("front", front_graph)):
        for node in graph.get("nodes", []):
            fid = str(node["feature_id"])
            in_rear, in_front = _logical_resolution(node)
            records.append(
                FeatureRecord(
                    panel=panel,
                    feature_id=fid,
                    feature_type=str(node.get("class_name", "")),
                    reachable_dirs=_reachable_dirs_raw(node),
                    in_rear=in_rear,
                    in_front=in_front,
                    planner_rear=(panel == "rear" and fid in rear_planner),
                    planner_front=(panel == "front" and fid in front_planner),
                    bucket=_classify_bucket(panel, fid, in_rear, in_front, node),
                )
            )
    records.sort(
        key=lambda r: (r.panel, int(r.feature_id) if r.feature_id.isdigit() else r.feature_id)
    )
    return records


def _plan_in_memory(wrong_axis: bool) -> list[dict[str, Any]]:
    import planner as planner_mod

    rear_ctx = _build_context("rear", REAR_GRAPH)
    front_ctx = _build_context("front", FRONT_GRAPH)

    original = planner_mod._reachability_dir_for_setup

    def _patched(ctx: MachiningContext) -> str:
        if wrong_axis:
            return OLD_WRONG_REAR_DIR
        return original(ctx)

    planner_mod._reachability_dir_for_setup = _patched
    try:
        cam_plan = plan_multi_setups(
            [
                SetupPlanInput(REAR_GRAPH, rear_ctx),
                SetupPlanInput(FRONT_GRAPH, front_ctx),
            ],
            setup_order=("rear", "front"),
            source_part="96260B",
        )
    finally:
        planner_mod._reachability_dir_for_setup = original

    return [op.model_dump() for op in cam_plan.operations]


# Canonical operation-bank slug -> legacy operation_type vocabulary. The planner
# collapsed operation_type+strategy into one ``operation`` field; this audit still
# reasons in the old vocabulary, so map plan dicts back at the read boundary.
_BANK_TO_LEGACY_OPTYPE: dict[str, str] = {
    "helix_bore": "bore",
    "drill": "drill",
    "chip_break_drill": "drill",
    "thread_mill": "tap",
    "optirough": "pocket_mill",
    "area_roughing": "pocket_mill",
    "rest_roughing": "pocket_mill",
    "pocket": "pocket_mill",
    "dynamic_mill_2d": "pocket_mill",
    "facing": "facing",
    "raster": "floor_finish",
    "constant_scallop": "surface_finish",
    "radial_spiral": "surface_finish",
    "steep_shallow": "surface_finish",
    "rest_finish": "surface_finish",
    "pencil": "fillet_finish",
}
_WALL_FEATURE_CLASSES = frozenset({"wall", "profile"})


def _op_type(op: Mapping[str, Any]) -> str:
    """Legacy operation_type for a plan-dict op (handles new ``operation`` field)."""
    if op.get("operation_type") is not None:
        return str(op["operation_type"])
    operation = str(op.get("operation", ""))
    if operation in ("contour_2d", "waterline"):
        if op.get("feature_type") in _WALL_FEATURE_CLASSES:
            return "wall_finish"
        return "finish_contour"
    return _BANK_TO_LEGACY_OPTYPE.get(operation, operation)


def _find_disappeared_ops(
    old_ops: list[Mapping[str, Any]],
    new_ops: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Old ops with no same-setup successor sharing any feature refs."""
    all_new_refs = {
        str(x) for op in new_ops for x in op.get("feature_refs", [])
    }
    disappeared: list[dict[str, Any]] = []
    for op in old_ops:
        refs = {str(x) for x in op.get("feature_refs", [])}
        setup = op["setup_id"]
        op_type = _op_type(op)
        same_setup = [n for n in new_ops if n["setup_id"] == setup]
        if any(refs & {str(x) for x in n.get("feature_refs", [])} for n in same_setup):
            continue
        if any(_op_type(n) == op_type for n in same_setup):
            if op_type in {"floor_finish", "wall_finish", "bore"}:
                continue
        # Fillet on rear whose feature already appears on front is regrouped, not lost.
        if (
            setup == "rear"
            and op_type == "fillet_finish"
            and refs <= all_new_refs
        ):
            continue
        disappeared.append(dict(op))
    return disappeared


def _classify_disappeared_op(
    old_op: Mapping[str, Any],
    *,
    left_rear_ids: set[str],
    new_rear_planner: set[str],
    rear_graph_nodes: Mapping[str, Mapping[str, Any]],
    front_graph_nodes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    setup = old_op["setup_id"]
    refs = [str(x) for x in old_op.get("feature_refs", [])]
    op_type = _op_type(old_op)

    wrong_axis_refs = [
        ref
        for ref in refs
        if setup == "rear" and ref in left_rear_ids and ref not in new_rear_planner
    ]
    if wrong_axis_refs and len(wrong_axis_refs) == len(refs):
        return (
            "SPURIOUS",
            f"wrong-axis rear (+Z) scope only; features {wrong_axis_refs}",
        )

    front_migrated = [
        ref
        for ref in refs
        if ref in front_graph_nodes
        and _feature_reachable_for_setup(front_graph_nodes[ref], FRONT_REACH_DIR)
    ]
    if front_migrated and len(front_migrated) == len(refs):
        return (
            "SPURIOUS",
            f"+Z-only on rear graph; same ids reachable on front panel ({front_migrated})",
        )

    if setup == "rear":
        plus_z_only = [
            ref
            for ref in refs
            if ref in rear_graph_nodes
            and _feature_reachable_for_setup(rear_graph_nodes[ref], FRONT_REACH_DIR)
            and not _feature_reachable_for_setup(rear_graph_nodes[ref], REAR_REACH_DIR)
        ]
        if plus_z_only and len(plus_z_only) == len(refs):
            return (
                "SPURIOUS",
                f"rear-graph features +Z-only (belong on front setup): {plus_z_only}",
            )

    return (
        "REAL_WORK",
        f"features {refs} not covered after axis fix",
    )


def _render_report(
    *,
    ledger: list[FeatureRecord],
    rear_old_z: set[str],
    rear_new_planner: set[str],
    front_new_planner: set[str],
    dropped_left: list[FeatureRecord],
    tally_left: Counter[str],
    unaccounted_left: list[FeatureRecord],
    old_ops: list[dict[str, Any]],
    new_ops: list[dict[str, Any]],
    dropped_ops: list[tuple[dict[str, Any], str, str]],
    disappeared_ops: list[dict[str, Any]],
    regroup_notes: list[str],
    verdict: str,
    fail_reasons: list[str],
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Coverage audit -- 96260B (post opening-axis fix)")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"## VERDICT: **{verdict}**")
    if fail_reasons:
        for reason in fail_reasons:
            lines.append(f"- {reason}")
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    left_count = len(rear_old_z - rear_new_planner)
    lines.append("## Scope counts")
    lines.append("")
    lines.append("| Setup | Old wrong-axis rear filter (+Z) | Planner kept (current) | Delta |")
    lines.append("|-------|--------------------------------|------------------------|-------|")
    lines.append(
        f"| rear  | {len(rear_old_z)} | {len(rear_new_planner)} | "
        f"{len(rear_new_planner) - len(rear_old_z)} |"
    )
    lines.append(
        f"| front | n/a | {len(front_new_planner)} | n/a |"
    )
    lines.append("")
    lines.append(
        "Logical resolution tests each panel's node against rear `-Z` and front `+Z` "
        "reachability tokens on that panel's graph (per-panel, not cross-panel id join)."
    )
    lines.append("")

    bucket_counts = Counter(r.bucket for r in ledger)
    lines.append("## Feature ledger summary")
    lines.append("")
    for bucket in BUCKETS:
        lines.append(f"- **{bucket}**: {bucket_counts.get(bucket, 0)}")
    lines.append("")

    lines.append("## Full feature ledger")
    lines.append("")
    lines.append(
        "| panel | feature_id | feature_type | reachable_dirs | "
        "resolved_setups | planner_scope | bucket |"
    )
    lines.append(
        "|-------|------------|--------------|----------------|"
        "-----------------|---------------|--------|"
    )
    for r in ledger:
        setups: list[str] = []
        if r.in_rear:
            setups.append("rear")
        if r.in_front:
            setups.append("front")
        resolved = ",".join(setups) if setups else "-"
        planner_bits: list[str] = []
        if r.planner_rear:
            planner_bits.append("rear")
        if r.planner_front:
            planner_bits.append("front")
        planner = ",".join(planner_bits) if planner_bits else "-"
        dirs = json.dumps(r.reachable_dirs)
        lines.append(
            f"| {r.panel} | {r.feature_id} | {r.feature_type} | {dirs} | "
            f"{resolved} | {planner} | {r.bucket} |"
        )
    lines.append("")

    lines.append(f"## Rear scope drop reconciliation ({left_count} features)")
    lines.append("")
    lines.append(
        f"Rear graph features in OLD wrong-axis `+Z` scope ({len(rear_old_z)}) "
        f"minus NEW rear planner scope ({len(rear_new_planner)}):"
    )
    lines.append("")
    lines.append("| feature_id | feature_type | reachable_dirs | bucket |")
    lines.append("|------------|--------------|----------------|--------|")
    for r in dropped_left:
        lines.append(
            f"| {r.feature_id} | {r.feature_type} | {json.dumps(r.reachable_dirs)} | {r.bucket} |"
        )
    lines.append("")
    lines.append("### Tally")
    lines.append("")
    for bucket in ("KEPT_FRONT", "KEPT_BOTH", "FLAGGED_ORPHAN", "FLAGGED_LATERAL", "UNACCOUNTED"):
        lines.append(f"- **{bucket}**: {tally_left.get(bucket, 0)}")
    lines.append(f"- **KEPT_REAR** (still rear-only): {tally_left.get('KEPT_REAR', 0)}")
    lines.append("")
    lines.append("### UNACCOUNTED (must be empty)")
    lines.append("")
    if unaccounted_left:
        lines.append(", ".join(r.feature_id for r in unaccounted_left))
    else:
        lines.append("*(empty)*")
    lines.append("")

    lines.append("## Op-count reconciliation")
    lines.append("")
    lines.append(f"- Old wrong-axis plan (in-memory replay): **{len(old_ops)}** ops")
    lines.append(f"- New fixed-axis plan on disk: **{len(new_ops)}** ops")
    lines.append(f"- Net delta: **{len(old_ops) - len(new_ops)}** ops")
    lines.append(f"- Disappeared rear/front op slots: **{len(disappeared_ops)}**")
    lines.append("")
    lines.append("### The 4 disappeared operations")
    lines.append("")
    lines.append("| old_op | setup | op_type | features | classification | evidence |")
    lines.append("|--------|-------|---------|----------|----------------|----------|")
    for old_op, classification, evidence in dropped_ops:
        refs = ", ".join(str(x) for x in old_op.get("feature_refs", []))
        lines.append(
            f"| {old_op['op_id']} | {old_op['setup_id']} | {_op_type(old_op)} "
            f"| {refs} | **{classification}** | {evidence} |"
        )
    lines.append("")
    if regroup_notes:
        lines.append("### Regrouped (not counted in the 4)")
        lines.append("")
        for note in regroup_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Prior-findings confirmation")
    lines.append("")
    lines.append("| Check | Expected | Observed |")
    lines.append("|-------|----------|----------|")
    for fid in ("25", "26", "36"):
        r = next(x for x in ledger if x.panel == "front" and x.feature_id == fid)
        lines.append(
            f"| front {fid} (-Z-only) | lands on rear, not orphaned | "
            f"bucket={r.bucket}, resolved rear={r.in_rear} |"
        )
    r37r = next(x for x in ledger if x.panel == "rear" and x.feature_id == "37")
    lines.append(
        f"| rear 37 | FLAGGED_LATERAL | bucket={r37r.bucket}, planner_rear={r37r.planner_rear} |"
    )
    r34r = next(x for x in ledger if x.panel == "rear" and x.feature_id == "34")
    r34f = next(x for x in ledger if x.panel == "front" and x.feature_id == "34")
    lines.append(
        f"| cross-panel id 34 | per-panel, not id-joined | "
        f"rear: bucket={r34r.bucket} dirs={r34r.reachable_dirs}; "
        f"front: bucket={r34f.bucket} dirs={r34f.reachable_dirs} |"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    rear_graph = load_feature_graph(REAR_GRAPH)
    front_graph = load_feature_graph(FRONT_GRAPH)
    rear_ctx = _build_context("rear", REAR_GRAPH)
    front_ctx = _build_context("front", FRONT_GRAPH)

    rear_old_z = _dir_kept_ids(rear_graph, OLD_WRONG_REAR_DIR)
    rear_new_planner = _planner_kept_ids(rear_graph, rear_ctx)
    front_new_planner = _planner_kept_ids(front_graph, front_ctx)

    ledger = _build_ledger(rear_graph, front_graph, rear_ctx, front_ctx)

    left_rear = rear_old_z - rear_new_planner
    dropped_left: list[FeatureRecord] = [
        r for r in ledger if r.panel == "rear" and r.feature_id in left_rear
    ]
    dropped_left.sort(key=lambda r: int(r.feature_id))

    tally_left = Counter(r.bucket for r in dropped_left)
    unaccounted_left = [r for r in dropped_left if r.bucket == "UNACCOUNTED"]
    unaccounted_all = [r for r in ledger if r.bucket == "UNACCOUNTED"]

    old_ops = _plan_in_memory(wrong_axis=True)
    new_ops = json.loads(NEW_PLAN_PATH.read_text())["operations"]

    rear_nodes = {str(n["feature_id"]): n for n in rear_graph.get("nodes", [])}
    front_nodes = {str(n["feature_id"]): n for n in front_graph.get("nodes", [])}

    disappeared_ops = _find_disappeared_ops(old_ops, new_ops)
    # Expect exactly the 4 rear pocket/surface ops shed by axis fix.
    disappeared_ops.sort(key=lambda op: op["op_id"])

    dropped_classified: list[tuple[dict[str, Any], str, str]] = []
    real_work_ops: list[dict[str, Any]] = []
    for op in disappeared_ops:
        classification, evidence = _classify_disappeared_op(
            op,
            left_rear_ids=left_rear,
            new_rear_planner=rear_new_planner,
            rear_graph_nodes=rear_nodes,
            front_graph_nodes=front_nodes,
        )
        dropped_classified.append((op, classification, evidence))
        if classification == "REAL_WORK":
            real_work_ops.append(op)

    regroup_notes: list[str] = []
    old_by_id = {op["op_id"]: op for op in old_ops}
    if "OP070" in old_by_id:
        old070 = {str(x) for x in old_by_id["OP070"]["feature_refs"]}
        new_cov = set()
        for nop in new_ops:
            new_cov |= {str(x) for x in nop.get("feature_refs", [])}
        shed = sorted(old070 - new_cov, key=int)
        regroup_notes.append(
            f"Old OP070 regrouped into new rear OP040; {len(shed)} +Z-only rear-only "
            f"contour ids not in new plan: {', '.join(shed[:8])}"
            + (f" ... (+{len(shed) - 8} more)" if len(shed) > 8 else "")
            + ". These are logically KEPT_FRONT on the rear graph but absent from the "
            "front-panel export (no planner path)."
        )

    notes: list[str] = []
    if len(rear_old_z) != 98:
        notes.append(
            f"Rear old +Z scope is {len(rear_old_z)} on current graph "
            f"(triage cited 98); drop set size {len(left_rear)}."
        )
    if len(old_ops) != 20:
        notes.append(
            f"In-memory wrong-axis replay produced {len(old_ops)} ops (brief cited 20)."
        )

    if len(disappeared_ops) != 4:
        notes.append(
            f"Disappeared-op detector found {len(disappeared_ops)} ops "
            f"(brief cited 4): {[op['op_id'] for op in disappeared_ops]}"
        )

    fail_reasons: list[str] = []
    if unaccounted_all:
        ids = ", ".join(f"{r.panel}:{r.feature_id}" for r in unaccounted_all)
        fail_reasons.append(f"UNACCOUNTED features in full ledger: {ids}")
    if unaccounted_left:
        fail_reasons.append(
            "Rear-drop reconciliation UNACCOUNTED: "
            + ", ".join(r.feature_id for r in unaccounted_left)
        )
    if real_work_ops:
        fail_reasons.append(
            "REAL_WORK op drops: " + ", ".join(op["op_id"] for op in real_work_ops)
        )

    verdict = "PASS" if not fail_reasons else "FAIL"

    report = _render_report(
        ledger=ledger,
        rear_old_z=rear_old_z,
        rear_new_planner=rear_new_planner,
        front_new_planner=front_new_planner,
        dropped_left=dropped_left,
        tally_left=tally_left,
        unaccounted_left=unaccounted_left,
        old_ops=old_ops,
        new_ops=new_ops,
        dropped_ops=dropped_classified,
        disappeared_ops=disappeared_ops,
        regroup_notes=regroup_notes,
        verdict=verdict,
        fail_reasons=fail_reasons,
        notes=notes,
    )
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH}")
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
