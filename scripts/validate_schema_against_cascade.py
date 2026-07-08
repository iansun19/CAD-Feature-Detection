#!/usr/bin/env python3
"""Validate cam_plan_schema assumptions against a real feature_graph_cascade.json.

Read-only conformance check: inspect cascade JSON, run assertions, smoke-test
CamPlan construction with real feature ids. No LLM, no planner logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cam_plan_schema import (  # noqa: E402
    CamPlan,
    MachiningParameters,
    Operation,
    PocketAccess,
    Setup,
    ToolRef,
)

DEFAULT_CASCADE = REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"

# Pocket classes where open/closed access is operationally relevant.
ACCESS_RELEVANT_POCKET_CLASSES = frozenset({
    "filleted_pocket",
    "filleted_open_pocket",
    "pocket",
    "open_pocket",
    "blind_pocket",
})

FEATURE_ID_KEY = "feature_id"
FEATURE_TYPE_KEY = "class_name"  # cascade nodes; maps to Operation.feature_type


class CheckResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.detail = ""

    def ok(self, detail: str = "") -> None:
        self.passed = True
        self.detail = detail

    def fail(self, detail: str) -> None:
        self.passed = False
        self.detail = detail


def load_cascade(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def inspect_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph["nodes"]
    edges = graph.get("edges", [])

    id_types: Counter[type] = Counter()
    id_values: list[Any] = []
    class_names: set[str] = set()
    missing_class_name: list[int] = []
    null_class_name: list[int] = []

    for node in nodes:
        raw_id = node.get(FEATURE_ID_KEY)
        id_types[type(raw_id)] += 1
        id_values.append(raw_id)

        cn = node.get(FEATURE_TYPE_KEY)
        if FEATURE_TYPE_KEY not in node:
            missing_class_name.append(raw_id)
        elif cn is None:
            null_class_name.append(raw_id)
        else:
            class_names.add(str(cn))

    features_with_edges: set[int] = set()
    for edge in edges:
        features_with_edges.add(int(edge["source"]))
        features_with_edges.add(int(edge["target"]))

    return {
        "nodes": nodes,
        "edges": edges,
        "id_types": id_types,
        "id_values": id_values,
        "class_names": class_names,
        "missing_class_name": missing_class_name,
        "null_class_name": null_class_name,
        "n_features": len(nodes),
        "n_with_edges": len(features_with_edges),
        "n_edges": len(edges),
    }


def check_feature_ids_as_str(id_values: list[Any]) -> CheckResult:
    check = CheckResult("feature_id -> str (lossless, no collisions)")
    if not id_values:
        check.fail("graph has zero nodes")
        return check

    bad_types = {type(v) for v in id_values} - {int, str}
    if bad_types:
        check.fail(f"unexpected id types: {sorted(t.__name__ for t in bad_types)}")
        return check

    str_ids = [str(v) for v in id_values]
    if len(str_ids) != len(set(str_ids)):
        dupes = [s for s, c in Counter(str_ids).items() if c > 1]
        check.fail(f"str(id) collisions: {dupes[:10]}")
        return check

    for raw, sid in zip(id_values, str_ids):
        if isinstance(raw, int):
            try:
                if int(sid) != raw:
                    check.fail(f"round-trip failed for id {raw!r} -> {sid!r}")
                    return check
            except ValueError:
                check.fail(f"str(id) not parseable as int for raw id {raw!r}")
                return check

    dominant = Counter(type(v) for v in id_values).most_common(1)[0][0].__name__
    check.ok(
        f"all {len(id_values)} ids stringify uniquely; JSON loads as {dominant}"
    )
    return check


def check_feature_types_present(info: dict[str, Any]) -> CheckResult:
    check = CheckResult(f"{FEATURE_TYPE_KEY} present and non-null (open string field)")
    problems: list[str] = []
    if info["missing_class_name"]:
        problems.append(f"{len(info['missing_class_name'])} nodes missing {FEATURE_TYPE_KEY!r}")
    if info["null_class_name"]:
        problems.append(f"{len(info['null_class_name'])} nodes with null {FEATURE_TYPE_KEY!r}")
    if problems:
        check.fail("; ".join(problems))
    else:
        check.ok(f"{len(info['class_names'])} distinct values, all non-null")
    return check


def report_pocket_access_gap(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Informational only - CamPlan.access comes from setup descriptor, not cascade."""
    access_relevant: list[dict[str, Any]] = []
    with_params_access: list[dict[str, Any]] = []
    without_params_access: list[dict[str, Any]] = []

    for node in nodes:
        cn = node.get(FEATURE_TYPE_KEY, "")
        if cn not in ACCESS_RELEVANT_POCKET_CLASSES:
            continue
        access_relevant.append(node)
        params = node.get("params") or {}
        access_val = params.get("access")
        if access_val is not None:
            with_params_access.append(node)
        else:
            without_params_access.append(node)

    return {
        "access_relevant": access_relevant,
        "with_params_access": with_params_access,
        "without_params_access": without_params_access,
    }


def pick_smoke_features(nodes: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pick 1-2 real nodes for CamPlan smoke test (hole + pocket when available)."""
    hole_classes = ("through_hole", "filleted_blind_hole")
    pocket_classes = tuple(sorted(ACCESS_RELEVANT_POCKET_CLASSES))

    by_class: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_class.setdefault(node[FEATURE_TYPE_KEY], []).append(node)

    first: dict[str, Any] | None = None
    second: dict[str, Any] | None = None

    for cls in hole_classes:
        if cls in by_class:
            first = by_class[cls][0]
            break
    if first is None:
        first = nodes[0]

    for cls in pocket_classes:
        if cls in by_class:
            candidate = by_class[cls][0]
            if candidate[FEATURE_ID_KEY] != first[FEATURE_ID_KEY]:
                second = candidate
                break
    if second is None and len(nodes) > 1:
        for node in nodes:
            if node[FEATURE_ID_KEY] != first[FEATURE_ID_KEY]:
                second = node
                break

    return first, second


def build_smoke_cam_plan(
    cascade_path: Path,
    graph: dict[str, Any],
    node_a: dict[str, Any],
    node_b: dict[str, Any] | None,
) -> CamPlan:
    part_id = graph.get("part_id") or cascade_path.stem
    setup_id = "primary"

    ops: list[Operation] = [
        Operation(
            op_id="OP010",
            sequence_index=0,
            feature_refs=[str(node_a[FEATURE_ID_KEY])],
            feature_type=str(node_a[FEATURE_TYPE_KEY]),
            setup_id=setup_id,
            operation_type="drill" if "hole" in str(node_a[FEATURE_TYPE_KEY]) else "pocket_mill",
            strategy="peck_drill" if "hole" in str(node_a[FEATURE_TYPE_KEY]) else "spiral",
            tool_id="T01",
            parameters=MachiningParameters(
                spindle_rpm=3000.0,
                feed_mm_per_min=180.0,
                param_source="handbook_default",
            ),
            depends_on=[],
            access=None,
        ),
    ]

    if node_b is not None:
        access = None
        if node_b[FEATURE_TYPE_KEY] in ACCESS_RELEVANT_POCKET_CLASSES:
            access = PocketAccess.UNKNOWN
        ops.append(
            Operation(
                op_id="OP020",
                sequence_index=1,
                feature_refs=[str(node_b[FEATURE_ID_KEY])],
                feature_type=str(node_b[FEATURE_TYPE_KEY]),
                setup_id=setup_id,
                operation_type="pocket_mill",
                strategy="spiral",
                tool_id="T02",
                parameters=MachiningParameters(
                    spindle_rpm=8000.0,
                    feed_mm_per_min=1200.0,
                    stepdown_mm=2.0,
                    param_source="conservative_heuristic",
                ),
                depends_on=["OP010"],
                access=access,
            ),
        )

    return CamPlan(
        source_part=str(part_id),
        feature_graph_ref=str(cascade_path.relative_to(REPO_ROOT)),
        setups=[
            Setup(
                setup_id=setup_id,
                opening_axis="+Y",
                fixture=None,
                notes="Smoke test setup from validate_schema_against_cascade.py",
            ),
        ],
        tools=[
            ToolRef(
                tool_id="T01",
                tool_type="drill",
                diameter_mm=3.2,
                source="hardcoded_v0",
            ),
            ToolRef(
                tool_id="T02",
                tool_type="endmill",
                diameter_mm=6.0,
                source="hardcoded_v0",
            ),
        ],
        operations=ops,
        remaining_material=None,
        metadata={"generator": "validate_schema_against_cascade.py"},
    )


def check_smoke_cam_plan(plan: CamPlan) -> CheckResult:
    check = CheckResult("CamPlan smoke test (real feature_refs + cross-field FK validation)")
    try:
        CamPlan.model_validate(plan.model_dump(mode="json"))
        refs = [ref for op in plan.operations for ref in op.feature_refs]
        check.ok(f"validated plan with feature_refs={refs}")
    except Exception as exc:
        check.fail(str(exc))
    return check


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate cam_plan_schema against feature_graph_cascade.json",
    )
    parser.add_argument(
        "cascade_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CASCADE,
        help=f"path to feature_graph_cascade.json (default: {DEFAULT_CASCADE.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)

    cascade_path = args.cascade_path.resolve()
    if not cascade_path.is_file():
        print(f"ERROR: cascade file not found: {cascade_path}", file=sys.stderr)
        return 2

    graph = load_cascade(cascade_path)
    info = inspect_graph(graph)
    nodes = info["nodes"]

    print_section("CASCADE INSPECTION")
    print(f"File:           {cascade_path}")
    print(f"part_id:        {graph.get('part_id')!r}")
    print(f"schema_version: {graph.get('schema_version')!r}")
    print(f"source:         {graph.get('source')!r}")
    print()
    print(f"Feature id key (feature_graph.py): {FEATURE_ID_KEY!r}")
    print("Python types of node[feature_id] after json.load:")
    for py_type, count in sorted(info["id_types"].items(), key=lambda x: x[0].__name__):
        print(f"  {py_type.__name__}: {count}")
    print()
    print(f"Feature type key in cascade nodes: {FEATURE_TYPE_KEY!r}")
    print(f"  (maps to CamPlan Operation.feature_type - not enum-locked)")
    print()
    print(f"Distinct {FEATURE_TYPE_KEY} values ({len(info['class_names'])}):")
    for name in sorted(info["class_names"]):
        count = sum(1 for n in nodes if n.get(FEATURE_TYPE_KEY) == name)
        print(f"  {name}: {count}")
    print()
    print(f"Total features:              {info['n_features']}")
    print(f"Adjacency edges:           {info['n_edges']}")
    print(f"Features with >=1 edge:    {info['n_with_edges']}")
    print(f"Features with no edges:    {info['n_features'] - info['n_with_edges']}")

    print_section("CONFORMANCE CHECKS")
    checks: list[CheckResult] = []

    id_check = check_feature_ids_as_str(info["id_values"])
    checks.append(id_check)
    print(f"[{'PASS' if id_check.passed else 'FAIL'}] {id_check.name}")
    if id_check.detail:
        print(f"       {id_check.detail}")

    type_check = check_feature_types_present(info)
    checks.append(type_check)
    print(f"[{'PASS' if type_check.passed else 'FAIL'}] {type_check.name}")
    if type_check.detail:
        print(f"       {type_check.detail}")

    access_report = report_pocket_access_gap(nodes)
    print()
    print("[INFO] Pocket access gap (informational - does not fail gate)")
    print("       CamPlan Operation.access is sourced from setup descriptor YAML,")
    print("       not from feature_graph_cascade.json.")
    print(f"       Access-relevant pocket nodes: {len(access_report['access_relevant'])}")
    print(f"       With params.access in cascade: {len(access_report['with_params_access'])}")
    print(f"       Without params.access:         {len(access_report['without_params_access'])}")
    if access_report["without_params_access"]:
        print("       Nodes missing params.access:")
        for node in access_report["without_params_access"]:
            print(
                f"         feature_id={node[FEATURE_ID_KEY]!r} "
                f"{FEATURE_TYPE_KEY}={node[FEATURE_TYPE_KEY]!r}"
            )
    if access_report["with_params_access"]:
        access_vals = Counter(
            (node.get("params") or {}).get("access") for node in access_report["with_params_access"]
        )
        print(f"       params.access values where present: {dict(access_vals)}")

    node_a, node_b = pick_smoke_features(nodes)
    print()
    print("Smoke-test feature picks:")
    print(
        f"  OP010 -> feature_id={node_a[FEATURE_ID_KEY]!r} "
        f"({node_a[FEATURE_TYPE_KEY]!r}) -> feature_refs={[str(node_a[FEATURE_ID_KEY])]!r}"
    )
    if node_b is not None:
        print(
            f"  OP020 -> feature_id={node_b[FEATURE_ID_KEY]!r} "
            f"({node_b[FEATURE_TYPE_KEY]!r}) -> feature_refs={[str(node_b[FEATURE_ID_KEY])]!r}"
        )

    smoke_plan = build_smoke_cam_plan(cascade_path, graph, node_a, node_b)
    smoke_check = check_smoke_cam_plan(smoke_plan)
    checks.append(smoke_check)
    print()
    print(f"[{'PASS' if smoke_check.passed else 'FAIL'}] {smoke_check.name}")
    if smoke_check.detail:
        print(f"       {smoke_check.detail}")

    print_section("SUMMARY")
    n_pass = sum(1 for c in checks if c.passed)
    n_fail = sum(1 for c in checks if not c.passed)
    print(f"Checks: {n_pass} passed, {n_fail} failed (of {len(checks)} gated assertions)")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}")

    dominant_id_type = info["id_types"].most_common(1)[0][0].__name__
    print()
    print("feature_refs type recommendation:")
    if id_check.passed and dominant_id_type == "int":
        print(
            "  KEEP list[str]. Cascade stores feature_id as JSON integers (0..N-1); "
            "str(id) is lossless and unambiguous. Serializing ids as strings in "
            "CamPlan keeps JSON schema stable and matches the schema docstring "
            "(feature_id serialized as string). No need for list[int]."
        )
    elif id_check.passed and dominant_id_type == "str":
        print("  list[str] matches native cascade id type.")
    else:
        print(
            "  REVIEW REQUIRED: id stringification check failed; see failures above "
            "before wiring planner code."
        )

    if n_fail:
        print()
        print("RESULT: FAIL - one or more conformance assertions failed.")
        return 1

    print()
    print("RESULT: PASS - schema assumptions match cascade data for gated checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
