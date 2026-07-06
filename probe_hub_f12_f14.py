#!/usr/bin/env python
"""
probe_hub_f12_f14.py — Hub face-level diagnostic + regression gates (96260B_front).

Maps F12/F14 residual hub geometry: depth tiers, adjacency, cascade ownership.
Exits non-zero when wired hub classes regress (flat 97, through hole, coaxial hub).

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_hub_f12_f14.py
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_hub_f12_f14.py --part 96260B_front
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from coaxial_stack_detection import (
    REFERENCE_CONTOUR_FACES_FRONT,
    REFERENCE_HUB_FLAT_FACES_FRONT,
    REFERENCE_OPEN_POCKET_FACES_FRONT,
    validate_coaxial_stack,
)
from feature_params import analyze_step, load_step_faces
from hole_detection import FaceGraph, _unit
from run_cascade import _load_edges, run_cascade

try:
    from brep_extents import collect_boundary_points
except ImportError:
    collect_boundary_points = None  # type: ignore[misc, assignment]

MM_PER_IN = 25.4
Y_AXIS = np.array([0.0, 1.0, 0.0])

F12 = [273, 274, 281, 282, 283]
F14 = [275, 276]
CONTEXT = [93, 94, 95, 96, 97, 105, 277, 278, 279, 280, 287, 296, 298]
HUB = sorted(set(F12 + F14 + [277, 278, 296]))

PARTS: dict[str, dict[str, str]] = {
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
        "graph_json": "pipeline_out/96260B_front/feature_graph_cascade.json",
    },
}


def top_y(step_path: str, n: int) -> float:
    if collect_boundary_points is not None:
        occ = load_step_faces(step_path)
        ys: list[float] = []
        for i in range(n):
            pts = collect_boundary_points(occ[i])
            ys.extend(pts[:, 1].tolist())
        return max(ys)
    faces = analyze_step(step_path)
    return max(float(f.centroid[1]) for f in faces)


def radial_uv(centroid, axis_pt=(0, 0, 0), axis_dir=Y_AXIS) -> float:
    c = np.asarray(centroid, float) - np.asarray(axis_pt, float)
    d = _unit(np.asarray(axis_dir, float))
    return float(np.linalg.norm(c - np.dot(c, d) * d))


def load_feature_map(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    out: dict[int, dict] = {}
    for node in data["nodes"]:
        info = {
            "feature_id": node["feature_id"],
            "class_name": node["class_name"],
            "class_id": node["class_id"],
        }
        for fid in node["face_ids"]:
            out[int(fid)] = info
    return out


def run_probe(part: str) -> int:
    cfg = PARTS[part]
    step_path = cfg["step"]
    graph_npz = cfg["graph_npz"]
    graph_json = Path(cfg.get("graph_json", ""))

    faces = analyze_step(step_path)
    by_index = {f.index: f for f in faces}
    n = len(faces)
    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n)

    _, pk, hl, cx, fl, of_r, _wl, _pr, rs = run_cascade(step_path, edge_index, edge_attr)
    top = top_y(step_path, n)

    print("=" * 88)
    print(f"HUB PROBE — {part}")
    print("=" * 88)
    print(f"STEP: {step_path}")
    print(f"part top Y = {top:.4f} mm")
    print()

    hdr = (
        f"{'face':>5} {'surf':>8} {'area':>9} {'depth_in':>8} {'rad_uv':>8}  "
        f"coaxial_owner"
    )
    print(hdr)
    print("-" * len(hdr))
    for fid in sorted(set(HUB + [97, 280, 298])):
        if fid not in by_index:
            continue
        fg = by_index[fid]
        depth_in = (top - float(fg.centroid[1])) / MM_PER_IN
        owner = "—"
        for feat in cx.features:
            if fid in feat.face_indices:
                owner = feat.toolpath_class
                break
        if owner == "—" and fid in fl.claimed_faces:
            owner = "flat"
        if owner == "—" and fid in hl.claimed_faces:
            owner = hl.features[0].kind if len(hl.features) == 1 else "hole"
        for feat in hl.features:
            if fid in feat.face_indices:
                owner = feat.kind
        if owner == "—" and fid in of_r.claimed_faces:
            owner = "outer_fillet(perimeter)"
        if owner == "—" and fid in cx.hub_flat_faces:
            owner = "flat(hub-deferred)"
        tag = " [F12]" if fid in F12 else (" [F14]" if fid in F14 else "")
        print(
            f"{fid:>5} {fg.surface_type:>8} {fg.area:>9.1f} {depth_in:>8.4f} "
            f"{radial_uv(fg.centroid):>8.2f}  {owner}{tag}"
        )

    print("\n--- adjacency (F12 ∪ F14) ---")
    hub_set = set(HUB)
    for fid in HUB:
        fg = by_index[fid]
        print(f"\nface {fid} ({fg.surface_type}):")
        for nb in sorted(graph.neighbors.get(fid, set())):
            if nb not in by_index:
                continue
            nfg = by_index[nb]
            kind = graph.edge_kind(fid, nb) or "?"
            marker = " [hub]" if nb in hub_set else ""
            print(
                f"  -> {nb:>3} {kind:>7} {nfg.surface_type:<8}{marker}"
            )

    print("\n--- regression gates ---")
    ok = True

    if 97 not in fl.claimed_faces:
        print("  [FAIL] face 97 not in flat bucket")
        ok = False
    else:
        print("  [ok] face 97 in flat bucket")

    for fid in (280, 298):
        if fid not in hl.claimed_faces:
            print(f"  [FAIL] through-hole face {fid} not claimed by hole pass")
            ok = False
        else:
            print(f"  [ok] through-hole face {fid} claimed")

    if part == "96260B_front":
        cx_report = validate_coaxial_stack(
            cx,
            expected_hub_flat_faces=REFERENCE_HUB_FLAT_FACES_FRONT,
            expected_open_pocket_faces=REFERENCE_OPEN_POCKET_FACES_FRONT,
            expected_contour_faces=REFERENCE_CONTOUR_FACES_FRONT,
            forbidden_faces=(97, 280, 298),
        )
        print(cx_report.render())
        ok = ok and cx_report.ok

        residual_hub = hub_set & rs.claimed_faces
        if residual_hub:
            print(f"  [FAIL] hub faces still in residual: {sorted(residual_hub)}")
            ok = False
        else:
            print("  [ok] no hub faces left in residual")

    print(f"\nprobe_hub: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Hub F12/F14 diagnostic + regression")
    ap.add_argument("--part", default="96260B_front", choices=sorted(PARTS))
    args = ap.parse_args()
    raise SystemExit(run_probe(args.part))


if __name__ == "__main__":
    main()
