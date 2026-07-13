#!/usr/bin/env python3
"""End-to-end smoke test for the PROVISIONAL lateral (±X/±Y/±Z) approach path.

The one real part in this repo (96260B) is a flat plate with no side-access
features, so there is nothing real to exercise the lateral path on. This script
builds a *synthetic* OCC solid — a block with a hole bored through a side face —
and runs it through the whole lateral chain, confirming a sane CamPlan comes out
the far end without crashing:

    synthetic solid
      -> lateral candidacy (which cardinals could reach each feature)
      -> verified lateral reachability (six-cardinal collision test)
      -> setup descriptor with a general cardinal orientation ("+X"), YAML round-trip
      -> orientation-aware sequencing (setup/approach-change cost)
      -> validated CamPlan

Requires the OCC env:
    env -u VIRTUAL_ENV -u PYTHONPATH \
        /Users/iansun19/miniconda3/envs/mlcad/bin/python scripts/smoke_lateral_axes.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

from schema.cam_plan_schema import CamPlan, MachiningParameters, Operation, Setup, ToolRef
from cascade.lateral_axes import (
    annotate_lateral_candidates,
    annotate_lateral_reachability,
    approach_vector_for_setup,
)
from planning.score_sequence import score_sequence
from planning.sequence_search import search_sequence
from cascade.setup_descriptor import (
    OpeningAxisSpec,
    PartSetupDescriptor,
    SetupDefaults,
    SetupEntry,
    dump_setup_descriptor,
    load_setup_descriptor,
)


@dataclass
class _FaceGeom:
    index: int
    normal: np.ndarray
    area: float


def build_solid():
    """100x100x40 block with a Ø12 hole bored through the side along +X."""
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 100.0, 40.0).Shape()
    ax = gp_Ax2(gp_Pnt(-5.0, 50.0, 20.0), gp_Dir(1.0, 0.0, 0.0))
    drill = BRepPrimAPI_MakeCylinder(ax, 6.0, 110.0).Shape()
    return BRepAlgoAPI_Cut(box, drill).Shape()


def classify_faces(shape):
    faces = list(TopologyExplorer(shape).faces())
    geoms: list[_FaceGeom] = []
    hole_ids: list[int] = []
    top_ids: list[int] = []
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from brep.brep_extents import plane_normal_occ

    for i, f in enumerate(faces):
        s = BRepAdaptor_Surface(f, True)
        props = GProp_GProps()
        brepgprop.SurfaceProperties(f, props)
        area = float(props.Mass())
        if s.GetType() == GeomAbs_Cylinder:
            d = s.Cylinder().Axis().Direction()
            normal = np.array([d.X(), d.Y(), d.Z()])
            hole_ids.append(i)
        elif s.GetType() == GeomAbs_Plane:
            normal = plane_normal_occ(f)
            if abs(normal[2]) > 0.9 and props.CentreOfMass().Z() > 20.0:
                top_ids.append(i)
        else:
            normal = np.array([0.0, 0.0, 1.0])
        geoms.append(_FaceGeom(i, normal, area))
    return faces, geoms, hole_ids, top_ids


def build_graph(hole_ids, top_ids):
    """Minimal hand-built feature graph (cascade recognition needs the trained
    model + graph tensors, which a synthetic solid does not have — that front end
    is not what the lateral path is testing)."""
    return {
        "nodes": [
            {"feature_id": "1", "class_name": "through_hole", "face_ids": hole_ids,
             "params": {"axis": {"direction": [1.0, 0.0, 0.0]}}},
            {"feature_id": "2", "class_name": "flat", "face_ids": top_ids,
             "params": {"normal": [0.0, 0.0, 1.0]}},
        ]
    }


def main() -> int:
    print("=" * 72)
    print("LATERAL-AXIS SMOKE TEST (provisional path) — synthetic block + side hole")
    print("=" * 72)

    shape = build_solid()
    faces, geoms, hole_ids, top_ids = classify_faces(shape)
    print(f"\nsolid: {len(faces)} faces | hole faces={hole_ids} top faces={top_ids}")
    assert hole_ids and top_ids, "expected a side hole and a top face"

    graph = build_graph(hole_ids, top_ids)

    # 1) Candidacy (pure geometry).
    cand = annotate_lateral_candidates(
        graph["nodes"], faces=geoms, opening_axis=[0.0, 0.0, 1.0]
    )
    print(f"\n[1] candidacy: {cand}")
    for n in graph["nodes"]:
        lc = n["approach"]["lateral_candidates"]
        print(f"    feature {n['feature_id']} ({n['class_name']}): "
              f"candidates={[c['dir'] for c in lc['candidates']]}")

    # 2) Verified reachability over six cardinals (OCC collision).
    reach = annotate_lateral_reachability(graph["nodes"], occ_faces=faces, shape=shape)
    print(f"\n[2] reachability summary: {reach}")
    for n in graph["nodes"]:
        lat = n["approach"]["lateral"]
        print(f"    feature {n['feature_id']} ({n['class_name']}): "
              f"reachable_dirs={lat['reachable_dirs']} provisional={lat['provisional']}")

    hole = next(n for n in graph["nodes"] if n["class_name"] == "through_hole")
    hole_dirs = set(hole["approach"]["lateral"]["reachable_dirs"])
    assert {"+X", "-X"} <= hole_dirs, f"side hole should open ±X, got {hole_dirs}"
    assert "+Z" not in hole_dirs, "side hole must NOT be reachable from the top (+Z)"
    print("\n    ✓ side hole reachable ±X (lateral), not +Z — as expected")

    # 3) Setup descriptor with a general cardinal orientation, YAML round-trip.
    desc = PartSetupDescriptor(
        part_id="synthetic_block",
        defaults=SetupDefaults(opening_axis=OpeningAxisSpec(mode="auto")),
        setups={
            "top": SetupEntry(
                setup_id="top", opening_axis=OpeningAxisSpec(
                    mode="explicit", vector=(0.0, 0.0, 1.0)),
                orientation="+Z"),
            "side": SetupEntry(
                setup_id="side", opening_axis=OpeningAxisSpec(
                    mode="explicit", vector=(0.0, 0.0, 1.0)),
                orientation="+X"),  # PROVISIONAL lateral orientation
        },
    )
    out = ROOT / "pipeline_out" / "synthetic_block"
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / "setup_descriptor.yaml"
    dump_setup_descriptor(desc, yaml_path, header="Synthetic lateral-axis smoke test.")
    reloaded = load_setup_descriptor(yaml_path)
    assert reloaded.setups["side"].orientation == "+X", "orientation lost on round-trip"
    print(f"\n[3] setup descriptor round-trips with orientation "
          f"top={reloaded.setups['top'].orientation} "
          f"side={reloaded.setups['side'].orientation} -> {yaml_path}")

    # 4) Orientation-aware sequencing.
    setups = [Setup(setup_id="top", opening_axis="+Z", orientation="+Z"),
              Setup(setup_id="side", opening_axis="+Z", orientation="+X",
                    orientation_provisional=True)]
    approach_map = {s.setup_id: (s.orientation or s.opening_axis) for s in setups}

    @dataclass
    class _Op:
        op_id: str
        setup_id: str
        tool_id: str
        operation: str
        feature_refs: tuple

    ops = [
        _Op("op_face", "top", "T1", "facing", ("2",)),
        _Op("op_drill", "side", "T2", "drill", ("1",)),
    ]
    precedence = {"op_drill": ["op_face"]}
    result = search_sequence(ops, precedence, strategy="beam", setup_approach=approach_map)
    print(f"\n[4] sequenced: {[o.op_id for o in result.ordered]}")
    print(f"    approach vectors: "
          f"top={approach_vector_for_setup(setups[0]).tolist()} "
          f"side={approach_vector_for_setup(setups[1]).tolist()}")
    print(f"    score: setup_changes={result.score.setup_changes} "
          f"approach_changes={result.score.approach_changes} total={result.score.total}")
    assert result.score.approach_changes == 1, "orientation flip should cost an approach change"

    # 5) Emit and validate a CamPlan carrying the provisional orientation.
    plan = CamPlan(
        source_part="synthetic_block",
        feature_graph_ref="synthetic:in_memory",
        setups=setups,
        tools=[
            ToolRef(tool_id="T1", tool_type="face_mill", diameter_mm=50.0, source="hardcoded_v0"),
            ToolRef(tool_id="T2", tool_type="drill", diameter_mm=12.0, source="hardcoded_v0"),
        ],
        operations=[
            Operation(op_id=o.op_id, sequence_index=i, feature_refs=list(o.feature_refs),
                      feature_type="flat" if o.operation == "facing" else "through_hole",
                      setup_id=o.setup_id, operation=o.operation, tool_id=o.tool_id,
                      parameters=MachiningParameters(param_source="conservative_heuristic"),
                      depends_on=[] if o.op_id == "op_face" else ["op_face"])
            for i, o in enumerate(result.ordered)
        ],
        metadata={"note": "PROVISIONAL lateral-axis smoke test; unvalidated path"},
    )
    plan_path = out / "cam_plan.json"
    from schema.cam_plan_schema import write_cam_plan
    write_cam_plan(plan_path, plan)
    side_setup = next(s for s in plan.setups if s.setup_id == "side")
    assert side_setup.orientation == "+X" and side_setup.orientation_provisional
    print(f"\n[5] ✓ valid CamPlan emitted -> {plan_path}")
    print(f"    side setup orientation={side_setup.orientation} "
          f"provisional={side_setup.orientation_provisional}")

    print("\n" + "=" * 72)
    print("SMOKE TEST PASSED — lateral path produced a sane CamPlan without crashing")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
