#!/usr/bin/env python
"""
PROBE — ownership trace for face 97 (Toolpath's true flat) on 96260B_front.

Established by probe_identify_toolpath_flat.py:
  - face 97 = Toolpath's flat (depth 4.450 mm = 0.1752 in, +Y normal, 844 mm2)
  - face 97 is NOT in the flat bucket -> claimed upstream OR filtered out
  - the flat bucket instead holds 7 pocket step-floors (~23.6 mm) + faces 273/277

Two questions, both answerable from the cascade run alone (no Toolpath input):

  Q1. WHO claims face 97?  Walk every pass's claimed_faces in cascade order
      (pockets -> holes -> flats -> outer_fillets -> residual) and report the
      first owner. That owner is where the fix lives.

  Q2. WHY does the flat pass admit 273/277 and the 7 step-planes?  Dump the
      flat pass's own admission signals for those faces so we can see what
      predicate let them in (and would have let 97 in, had it been free).

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_who_claims_97.py --part 96260B_front
"""
from __future__ import annotations

import argparse
from pathlib import Path

from feature_params import analyze_step
from flats_detection import (
    FlatDetectionConfig,
    _group_planes,
    _is_central_stack_endcap,
    _qualifies_as_standalone_flat,
)
from hole_detection import FaceGraph
from run_cascade import _load_edges, run_cascade

TARGET = 97
STEP_PLANES = [112, 139, 164, 189, 214, 239, 262]
BIG = [273, 277]

PARTS: dict[str, dict[str, str]] = {
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
    },
    "96260B_plate": {
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_plate/graph.npz",
    },
}


def _feature_owner(target: int, features, *, label: str) -> str | None:
    for feat in features:
        if target in feat.face_indices:
            extra = ""
            if hasattr(feat, "kind"):
                extra = f" kind={feat.kind}"
            if hasattr(feat, "nominal_diameter"):
                extra += f" dia={feat.nominal_diameter:.2f} mm"
            if hasattr(feat, "area"):
                extra += f" area={feat.area:.1f} mm²"
            return f"{label} feature {feat.feature_id}{extra}  faces={sorted(feat.face_indices)}"
    return None


def _flat_admission_row(
    face_idx: int,
    *,
    by_index: dict,
    flat_pool: set[int],
    hole_claimed: set[int],
    graph: FaceGraph,
    groups_map: dict,
    flat_result,
    config: FlatDetectionConfig,
) -> dict:
    fg = by_index[face_idx]
    group_ids = next((ids for ids in groups_map.values() if face_idx in ids), None)
    group_area = (
        sum(float(by_index[i].area) for i in group_ids) if group_ids else float("nan")
    )
    in_pool = face_idx in flat_pool
    qualifies = reason = endcap = None
    if group_ids and in_pool:
        qualifies, reason = _qualifies_as_standalone_flat(
            group_ids, by_index, graph, hole_claimed, config,
        )
        endcap = _is_central_stack_endcap(
            face_idx, fg, graph, hole_claimed, by_index, config,
        )
    flat_feat = next(
        (f.feature_id for f in flat_result.features if face_idx in f.face_indices),
        None,
    )
    return {
        "face": face_idx,
        "in_flat_pool": in_pool,
        "surf": fg.surface_type,
        "area": float(fg.area),
        "group": group_ids,
        "group_area": group_area,
        "min_area_ok": group_area >= config.min_flat_area_mm2 if group_ids else False,
        "endcap": endcap,
        "qualifies": qualifies,
        "reason": reason,
        "deferred": flat_result.deferred_reasons.get(face_idx),
        "flat_feature": flat_feat,
        "in_bucket": face_idx in flat_result.claimed_faces,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="96260B_front")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--step", default=None)
    ap.add_argument("--graph-npz", default=None)
    args = ap.parse_args()

    cfg = PARTS.get(args.part, {})
    step_path = args.step or cfg.get("step") or f"{args.part}.step"
    graph_npz = args.graph_npz or cfg.get("graph_npz")

    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _faces, pocket_result, hole_result, coaxial_result, flat_result, outer_fillet_result, profile_result, residual_result = (
        run_cascade(step_path, edge_index, edge_attr)
    )
    by_index = {f.index: f for f in analyze_step(step_path)}

    passes = [
        ("pockets", pocket_result),
        ("holes", hole_result),
        ("coaxial_stack", coaxial_result),
        ("flats", flat_result),
        ("outer_fillets", outer_fillet_result),
        ("profile", profile_result),
        ("residual", residual_result),
    ]

    print(f"part={args.part}   tracing ownership of face {args.target}\n")
    owner = None
    for name, r in passes:
        claimed = set(r.claimed_faces)
        has = args.target in claimed
        print(f"  {name:>14}: {len(claimed):>3} faces claimed   "
              f"{'<== CLAIMS ' + str(args.target) if has else ''}")
        if has and owner is None:
            owner = name

    print(f"\n  first owner of face {args.target}: {owner or 'UNCLAIMED / filtered'}")
    if owner == "pockets":
        detail = _feature_owner(args.target, pocket_result.features, label="pocket")
    elif owner == "holes":
        detail = _feature_owner(args.target, hole_result.features, label="hole")
    elif owner == "coaxial_stack":
        detail = _feature_owner(args.target, coaxial_result.features, label="coaxial")
    elif owner == "flats":
        detail = _feature_owner(args.target, flat_result.features, label="flat")
    elif owner == "outer_fillets":
        detail = _feature_owner(args.target, outer_fillet_result.features, label="outer_fillet")
    elif owner == "residual":
        detail = _feature_owner(args.target, residual_result.features, label="residual")
    else:
        detail = None
    if detail:
        print(f"  detail: {detail}")

    # Q2: flat-pass admission signals for bucket faces + target.
    print("\n--- flat-pass admission for the 9 bucket faces + target 97 ---")
    config = FlatDetectionConfig()
    flat_pool = set(hole_result.remaining_faces)
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(by_index))
    plane_ids = sorted(i for i in flat_pool if by_index[i].surface_type == "plane")
    groups_map = _group_planes(plane_ids, by_index, config)

    watch: list[tuple[str, list[int]]] = [
        ("target", [args.target]),
        ("step-planes", STEP_PLANES),
        ("big", BIG),
    ]
    hdr = (
        f"{'grp':>11} {'face':>5} {'pool':>5} {'bucket':>7} "
        f"{'area':>8} {'grp_area':>9} {'min_ok':>7} {'endcap':>7}  admission"
    )
    print(hdr)
    print("-" * len(hdr))
    for grp_name, face_list in watch:
        for i in face_list:
            row = _flat_admission_row(
                i,
                by_index=by_index,
                flat_pool=flat_pool,
                hole_claimed=set(hole_result.claimed_faces),
                graph=graph,
                groups_map=groups_map,
                flat_result=flat_result,
                config=config,
            )
            if not row["in_flat_pool"]:
                admission = "NOT in flat pool (claimed upstream)"
            elif row["in_bucket"]:
                admission = f"claimed flat {row['flat_feature']}: {row['reason']}"
            elif row["deferred"]:
                admission = f"deferred: {row['deferred']}"
            else:
                admission = row["reason"] or "—"
            print(
                f"{grp_name:>11} {i:>5} "
                f"{'yes' if row['in_flat_pool'] else 'no':>5} "
                f"{'yes' if row['in_bucket'] else 'no':>7} "
                f"{row['area']:>8.1f} "
                f"{row['group_area']:>9.1f} "
                f"{'yes' if row['min_area_ok'] else 'no':>7} "
                f"{('yes' if row['endcap'] else 'no') if row['endcap'] is not None else '—':>7}  "
                f"{admission}"
            )
            if row["group"] and len(row["group"]) > 1:
                print(f"{'':>11}       coplanar group: {row['group']}")

    print("\n--- read ---")
    print("If pockets own 97 -> boundary bug on the POCKET side (over-reach onto")
    print("   top flat ring). Fix pocket boundary, not the flat pass.")
    print("If holes own 97 -> central-bore pass over-reaching near +Y ring.")
    print("If UNCLAIMED/filtered -> a flat-pass exclusion kills 97; find the predicate.")
    print("Either way: the 7 step-planes should be POCKET-owned, and 273/277")
    print("belong in residual/contour -- the flat pass is admitting on the wrong axis.")


if __name__ == "__main__":
    main()
