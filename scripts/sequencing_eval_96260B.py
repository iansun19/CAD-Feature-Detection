#!/usr/bin/env python3
"""Emit score breakdowns for none/greedy/beam on 96260B."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cam_plan_schema import load_cam_plan  # noqa: E402
from machining_context import build_context_v0, load_feature_graph  # noqa: E402
from planner import SetupPlanInput, plan_multi_setups  # noqa: E402
from sequence_search import validate_sequence_precedence  # noqa: E402


def machined_faces(plan, root: Path) -> set[int]:
    refs = plan.metadata.get("feature_graph_refs", {})
    faces: set[int] = set()
    for sid, ref in refs.items():
        graph = load_feature_graph(root / ref)
        by_id = {str(node["feature_id"]): node for node in graph["nodes"]}
        for op in plan.operations:
            if op.setup_id != sid:
                continue
            for fid in op.feature_refs:
                node = by_id.get(fid)
                if node:
                    faces.update(node.get("face_ids", []))
    return faces


def work_signature(plan) -> set[tuple]:
    return {
        (
            tuple(sorted(op.feature_refs)),
            op.setup_id,
            op.operation,
            op.tool_id,
        )
        for op in plan.operations
    }


def main() -> int:
    rear_graph = REPO_ROOT / "pipeline_out/96260B_rear/feature_graph_cascade.json"
    front_graph = REPO_ROOT / "pipeline_out/96260B_front/feature_graph_cascade.json"
    setup_yaml = REPO_ROOT / "eval/gt/96260B_setup.yaml"
    ctx_kwargs = {
        "material": "aluminum",
        "tool_source": "hardcoded",
        "setups_source": "authored",
    }
    rear_ctx = build_context_v0(
        REPO_ROOT / "96260B_REAR_XR004_PCD PLATE.stp copy",
        setup_yaml,
        rear_graph,
        setup_id="rear",
        **ctx_kwargs,
    )
    front_ctx = build_context_v0(
        REPO_ROOT / "96260B_FRONT_XR004_PCD PLATE.stp copy",
        setup_yaml,
        front_graph,
        setup_id="front",
        **ctx_kwargs,
    )
    inputs = [
        SetupPlanInput(rear_graph, rear_ctx),
        SetupPlanInput(front_graph, front_ctx),
    ]

    plans = {}
    for mode in ("none", "greedy", "beam"):
        plan = plan_multi_setups(
            inputs,
            setup_order=("rear", "front"),
            source_part="96260B",
            seq_search=mode,
        )
        precedence = {op.op_id: list(op.depends_on) for op in plan.operations}
        validate_sequence_precedence(plan.operations, precedence)
        plans[mode] = plan
        score = plan.metadata["sequence_score"]
        print(f"=== {mode} ===")
        print(f"  total: {score['total']}")
        print(f"  setup_changes: {score['setup_changes']}")
        print(f"  tool_changes: {score['tool_changes']}")
        print(f"  weighted: {score['weighted']}")
        print(f"  ops: {len(plan.operations)}")
        print(f"  machined_faces: {len(machined_faces(plan, REPO_ROOT))}")

    baseline_faces = machined_faces(plans["none"], REPO_ROOT)
    baseline_work = work_signature(plans["none"])
    for mode in ("greedy", "beam"):
        assert work_signature(plans[mode]) == baseline_work
        assert machined_faces(plans[mode], REPO_ROOT) == baseline_faces

    none_total = plans["none"].metadata["sequence_score"]["total"]
    greedy_total = plans["greedy"].metadata["sequence_score"]["total"]
    beam_total = plans["beam"].metadata["sequence_score"]["total"]
    assert greedy_total <= none_total + 1e-9
    assert beam_total <= greedy_total + 1e-9

    example = load_cam_plan(REPO_ROOT / "examples/cam_plan_96260B.json")
    print("example_plan_faces:", len(machined_faces(example, REPO_ROOT)))
    print("work_signature_unchanged:", work_signature(plans["none"]) == work_signature(example))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
