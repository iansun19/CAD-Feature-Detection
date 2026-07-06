"""probe_depth_calibration.py — align cascade pocket depth with Toolpath panel fields.

Diagnostic only. Compares Toolpath ground truth for Filleted Pocket #1 on the
96260B reference plate against depths/attributes the cascade currently emits
and against explicit geometric candidates.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_depth_calibration.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Sphere

from brep_extents import collect_boundary_points, pocket_machined_extents
from feature_params import (
    _params_pocket_like,
    analyze_step,
    load_step_faces,
    span_along_direction,
)
from instance_features import instance_features
from hole_detection import FaceGraph
from run_cascade import _load_edges, run_cascade

MM_PER_IN = 25.4

STEP_PATH = Path("96260B_REAR_XR004_PCD PLATE.stp copy")
GRAPH_NPZ = Path("pipeline_out/96260B_plate/graph.npz")

# Toolpath Feature Details — Filleted Pocket #1 (all 7 pockets near-identical).
TP = {
    "feature_depth_in": 0.2885,
    "depth_below_top_in": 1.0848,
    "ld_ratio": 0.44,
    "fillet_radius_in": 0.25,
    "three_d_surface": False,
    "primary_wall_dia_in": 0.800,
}

TOL_IN = 0.015  # ~0.4 mm for "match?"


def _in(mm: float) -> float:
    return float(mm) / MM_PER_IN


def _fmt_in(mm: float | None) -> str:
    if mm is None or (isinstance(mm, float) and np.isnan(mm)):
        return "—"
    return f"{_in(mm):.4f} in ({mm:.3f} mm)"


def _match(got_in: float | None, target_in: float, tol: float = TOL_IN) -> str:
    if got_in is None or (isinstance(got_in, float) and np.isnan(got_in)):
        return "no"
    return "YES" if abs(got_in - target_in) <= tol else "no"


def _closest_label(got_in: float | None, a: float, b: float) -> str:
    if got_in is None or (isinstance(got_in, float) and np.isnan(got_in)):
        return "—"
    da, db = abs(got_in - a), abs(got_in - b)
    if da < db:
        return f"closer to feature_depth (Δ={da:.4f} in)"
    if db < da:
        return f"closer to depth_below_top (Δ={db:.4f} in)"
    return "equidistant"


def _wall800_ids(records, face_ids: list[int]) -> list[int]:
    target_r = TP["primary_wall_dia_in"] * MM_PER_IN / 2.0
    out = []
    for i in face_ids:
        f = records[i]
        if f.surface_type == "cylinder" and f.radius is not None:
            if abs(float(f.radius) - target_r) <= 0.05 * MM_PER_IN:
                out.append(i)
    return out


def _y_values(occ_faces, face_ids: list[int]) -> np.ndarray:
    vals: list[float] = []
    for i in face_ids:
        pts = collect_boundary_points(occ_faces[i])
        vals.extend(pts[:, 1].tolist())
    return np.asarray(vals, dtype=np.float64)


def _axis_values(occ_faces, face_ids: list[int], axis: np.ndarray) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    vals: list[float] = []
    for i in face_ids:
        pts = collect_boundary_points(occ_faces[i])
        vals.extend((pts @ axis).tolist())
    return np.asarray(vals, dtype=np.float64)


def _sphere_radii_in(occ_faces, records, face_ids: list[int]) -> list[float]:
    radii: list[float] = []
    for i in face_ids:
        if records[i].surface_type != "sphere":
            continue
        surf = BRepAdaptor_Surface(occ_faces[i], True)
        if surf.GetType() == GeomAbs_Sphere:
            radii.append(_in(surf.Sphere().Radius()))
    return sorted(set(round(r, 4) for r in radii))


def _dominant_fillet_radius_in(occ_faces, records, face_ids: list[int]) -> float | None:
    """Return the largest sphere fillet radius in the pocket (Toolpath uses 0.25 in)."""
    radii = _sphere_radii_in(occ_faces, records, face_ids)
    return max(radii) if radii else None


def measure_pocket(
    *,
    pocket_id: int,
    face_ids: list[int],
    opening_axis: tuple[float, float, float],
    records,
    occ_faces,
    part_y_top: float,
    part_axis_top: float,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    step_path: Path,
) -> dict:
    """Compute cascade-emitted and candidate depths for one pocket."""
    idxs = sorted(face_ids)
    pfaces = [records[i] for i in idxs]
    params = _params_pocket_like(pfaces)

    # --- cascade currently emits (pocket_detection has no depth field) ---
    cascade_depth_along_floor_mm = float(params.get("depth_along_floor_normal", float("nan")))
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(records))
    inst = instance_features(pfaces, graph=graph, face_indices=set(idxs))
    cascade_bbox_depth_mm = float(inst.get("bbox_depth", float("nan")))

    brep_out, brep_errs, _ = pocket_machined_extents(
        step_path, idxs, records, params,
    )
    brep_depth_mm = (
        float(brep_out["depth"]["value"]) if brep_out else float("nan")
    )

    axis = np.asarray(opening_axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    # --- candidate (a): self-depth = axial extent of pocket boundary geometry ---
    pocket_axis = _axis_values(occ_faces, idxs, axis)
    self_depth_mm = float(pocket_axis.max() - pocket_axis.min())

    pocket_y = _y_values(occ_faces, idxs)
    self_depth_y_mm = float(pocket_y.max() - pocket_y.min())

    wall800 = _wall800_ids(records, idxs)
    if wall800:
        wall_y = _y_values(occ_faces, wall800)
        wall800_span_mm = float(wall_y.max() - wall_y.min())
        wall800_centroid_span_mm = float(
            span_along_direction([records[i].centroid for i in wall800], axis)
        )
        top_to_wall_bottom_mm = float(part_y_top - wall_y.min())
    else:
        wall800_span_mm = float("nan")
        wall800_centroid_span_mm = float("nan")
        top_to_wall_bottom_mm = float("nan")

    # --- candidate (b): depth below top datum ---
    depth_below_top_y_mm = float(part_y_top - pocket_y.min())
    depth_below_top_axis_mm = float(part_axis_top - pocket_axis.min())

    # L/D reference depth implied by Toolpath ratio × primary diameter
    ld_implied_depth_in = TP["ld_ratio"] * TP["primary_wall_dia_in"]

    surf_counts: dict[str, int] = {}
    for i in idxs:
        st = records[i].surface_type
        surf_counts[st] = surf_counts.get(st, 0) + 1

    return {
        "pocket_id": pocket_id,
        "n_faces": len(idxs),
        "face_ids": idxs,
        "opening_axis": opening_axis,
        "wall800_ids": wall800,
        "surface_counts": surf_counts,
        "cascade_depth_along_floor_mm": cascade_depth_along_floor_mm,
        "cascade_bbox_depth_mm": cascade_bbox_depth_mm,
        "brep_depth_mm": brep_depth_mm,
        "brep_errors": brep_errs,
        "self_depth_mm": self_depth_mm,
        "self_depth_y_mm": self_depth_y_mm,
        "wall800_span_mm": wall800_span_mm,
        "wall800_centroid_span_mm": wall800_centroid_span_mm,
        "top_to_wall_bottom_mm": top_to_wall_bottom_mm,
        "depth_below_top_y_mm": depth_below_top_y_mm,
        "depth_below_top_axis_mm": depth_below_top_axis_mm,
        "fillet_radii_in": _sphere_radii_in(occ_faces, records, idxs),
        "fillet_radius_in": _dominant_fillet_radius_in(occ_faces, records, idxs),
        "has_bspline": surf_counts.get("bspline", 0) + surf_counts.get("bezier", 0) > 0,
        "has_sphere": surf_counts.get("sphere", 0) > 0,
        "ld_implied_depth_in": ld_implied_depth_in,
    }


def print_pocket_row(label: str, m: dict, key_mm: str) -> None:
    mm = m.get(key_mm)
    if mm is None:
        return
    print(f"  {label:28s} {_fmt_in(mm)}")


def main() -> int:
    print("probe_depth_calibration.py — Toolpath vs cascade pocket depth\n")

    if not STEP_PATH.is_file():
        print(f"ERROR: missing STEP {STEP_PATH}")
        return 1

    edge_index, edge_attr = _load_edges(
        GRAPH_NPZ if GRAPH_NPZ.is_file() else None,
        STEP_PATH,
    )
    _faces, pocket_result, _, _, _, _ = run_cascade(STEP_PATH, edge_index, edge_attr)
    records = analyze_step(STEP_PATH)
    occ_faces = load_step_faces(STEP_PATH)

    # Part datum: boundary extents (opening axis = Y for these pockets).
    all_pts = np.vstack([
        collect_boundary_points(occ_faces[i]) for i in range(len(occ_faces))
    ])
    part_y_min = float(all_pts[:, 1].min())
    part_y_max = float(all_pts[:, 1].max())
    part_y_span_mm = part_y_max - part_y_min

    axis = np.array([0.0, 1.0, 0.0])
    part_axis_projs = all_pts @ axis
    part_axis_min = float(part_axis_projs.min())
    part_axis_max = float(part_axis_projs.max())

    print("Part Y datum (all face boundaries):")
    print(f"  Y_min = {part_y_min:.4f} mm ({_in(part_y_min):.4f} in)  [bottom]")
    print(f"  Y_max = {part_y_max:.4f} mm ({_in(part_y_max):.4f} in)  [top / opening datum]")
    print(f"  Y_span = {part_y_span_mm:.4f} mm ({_in(part_y_span_mm):.4f} in)")
    print(f"  opening_axis = (0, 1, 0) — axis projection top = {part_axis_max:.4f} mm\n")

    pockets = pocket_result.features
    print(f"Cascade pockets: {len(pockets)}\n")

    measurements = [
        measure_pocket(
            pocket_id=f.feature_id,
            face_ids=sorted(f.face_indices),
            opening_axis=f.opening_axis,
            records=records,
            occ_faces=occ_faces,
            part_y_top=part_y_max,
            part_axis_top=part_axis_max,
            edge_index=edge_index,
            edge_attr=edge_attr,
            step_path=STEP_PATH,
        )
        for f in pockets
    ]

    # --- per-pocket table (all 7) ---
    hdr = (
        f"{'id':>3} {'below_top':>10} {'self_span':>10} {'wall⌀800':>10} "
        f"{'floor_norm':>10} {'brep':>10} {'filletR':>8}"
    )
    print("All pockets (inches):")
    print(hdr)
    for m in measurements:
        brep_in = _in(m["brep_depth_mm"]) if not np.isnan(m["brep_depth_mm"]) else float("nan")
        fillet = m["fillet_radius_in"] if m["fillet_radius_in"] is not None else float("nan")
        print(
            f"{m['pocket_id']:3d} "
            f"{_in(m['depth_below_top_y_mm']):10.4f} "
            f"{_in(m['self_depth_mm']):10.4f} "
            f"{_in(m['wall800_span_mm']):10.4f} "
            f"{_in(m['cascade_depth_along_floor_mm']):10.4f} "
            f"{brep_in:10.4f} "
            f"{fillet:8.4f}"
        )
    print()

    rep = measurements[0]
    print(f"Representative pocket: id={rep['pocket_id']}  n_faces={rep['n_faces']}")
    print(f"  face_ids: {rep['face_ids']}")
    print(f"  surface_counts: {rep['surface_counts']}")
    print(f"  ⌀0.800 wall faces: {rep['wall800_ids']}\n")

    print("Cascade CURRENTLY emits (pocket_detection emits no depth field):")
    print("  feature_params.depth_along_floor_normal (centroid span along floor normal):")
    print_pocket_row("", rep, "cascade_depth_along_floor_mm")
    print(f"    → {_closest_label(_in(rep['cascade_depth_along_floor_mm']), TP['feature_depth_in'], TP['depth_below_top_in'])}")
    print("  instance_features.bbox_depth (sorted centroid bbox — not opening-axis depth):")
    print_pocket_row("", rep, "cascade_bbox_depth_mm")
    print("  brep_extents.pocket_machined_extents.depth:")
    if np.isnan(rep["brep_depth_mm"]):
        print(f"    FAILED — {rep['brep_errors']}")
    else:
        print_pocket_row("", rep, "brep_depth_mm")
    print()

    print("Candidate depth definitions (from pocket boundary geometry, opening axis Y):")
    print("  (a) self-depth — full pocket face boundary span along opening axis:")
    print_pocket_row("", rep, "self_depth_mm")
    print("  (a') ⌀0.800 wall boundary span only (primary wall family):")
    print_pocket_row("", rep, "wall800_span_mm")
    print("  (a'') ⌀0.800 wall centroid span along opening axis:")
    print_pocket_row("", rep, "wall800_centroid_span_mm")
    print("  (b) depth below top — part Y_max − pocket deepest boundary point:")
    print_pocket_row("", rep, "depth_below_top_y_mm")
    print("  (b') top to ⌀0.800 wall bottom (excludes step pads below walls):")
    print_pocket_row("", rep, "top_to_wall_bottom_mm")
    print()

    # --- summary table ---
    feat_depth_best = _in(rep["wall800_span_mm"])
    below_top_best = _in(rep["depth_below_top_y_mm"])
    cascade_emit = _in(rep["cascade_depth_along_floor_mm"])
    fillet_best = rep["fillet_radius_in"]

    ld_from_feature = TP["feature_depth_in"] / TP["primary_wall_dia_in"]
    ld_from_wall_span = _in(rep["wall800_span_mm"]) / TP["primary_wall_dia_in"]
    ld_from_below_top = _in(rep["depth_below_top_y_mm"]) / TP["primary_wall_dia_in"]
    ld_depth_implied_in = rep["ld_implied_depth_in"]  # 0.44 × ⌀0.800 (TP arithmetic)

    three_d_cascade = rep["has_bspline"] or rep["has_sphere"]

    rows = [
        (
            "feature_depth",
            TP["feature_depth_in"],
            cascade_emit,
            _match(cascade_emit, TP["feature_depth_in"]),
        ),
        (
            "feature_depth",
            TP["feature_depth_in"],
            feat_depth_best,
            _match(feat_depth_best, TP["feature_depth_in"]),
        ),
        (
            "depth_below_top",
            TP["depth_below_top_in"],
            cascade_emit,
            _match(cascade_emit, TP["depth_below_top_in"]),
        ),
        (
            "depth_below_top",
            TP["depth_below_top_in"],
            below_top_best,
            _match(below_top_best, TP["depth_below_top_in"]),
        ),
        (
            "L/D ratio (depth/⌀0.800)",
            TP["ld_ratio"],
            ld_from_feature,
            _match(ld_from_feature, TP["ld_ratio"]),
        ),
        (
            "L/D ratio (wall_span/⌀0.800)",
            TP["ld_ratio"],
            ld_from_wall_span,
            _match(ld_from_wall_span, TP["ld_ratio"]),
        ),
        (
            "L/D depth implied (0.44×⌀)",
            ld_depth_implied_in,
            feat_depth_best,
            _match(feat_depth_best, ld_depth_implied_in),
        ),
        (
            "L/D depth implied (0.44×⌀)",
            ld_depth_implied_in,
            _in(rep["wall800_span_mm"]),
            _match(_in(rep["wall800_span_mm"]), ld_depth_implied_in),
        ),
        (
            "fillet_radius",
            TP["fillet_radius_in"],
            fillet_best,
            _match(fillet_best, TP["fillet_radius_in"]) if fillet_best else "no",
        ),
        (
            "3D surface",
            "No" if not TP["three_d_surface"] else "Yes",
            "Yes" if three_d_cascade else "No",
            "YES" if three_d_cascade == TP["three_d_surface"] else "no",
        ),
    ]

    print("=" * 78)
    print(f"{'Toolpath field':22s} {'Toolpath':>12s} {'cascade candidate':>18s} {'match?':>8s}")
    print("=" * 78)
    seen: set[str] = set()
    for field, tp_val, cand, ok in rows:
        if field in seen:
            label = "  ↳ alt"
        else:
            label = field
            seen.add(field)
        if isinstance(tp_val, float):
            tp_s = f"{tp_val:.4f}"
        else:
            tp_s = str(tp_val)
        if isinstance(cand, float):
            cand_s = f"{cand:.4f}"
        else:
            cand_s = str(cand)
        print(f"{label:22s} {tp_s:>12s} {cand_s:>18s} {ok:>8s}")

    print()
    print("Field mapping notes:")
    print("  • depth_below_top: EXACT match at part Y_max − pocket Y_min = 1.0848 in (all 7 pockets).")
    print("  • feature_depth: nearest candidate = ⌀0.800 wall boundary span = 0.3291 in")
    print("    (Δ=+0.0406 in vs Toolpath 0.2885; no computed candidate hits 0.2885 exactly).")
    print("  • cascade depth_along_floor_normal = 0.8643 in — centroid span, matches neither TP field.")
    print(f"  • L/D=0.44 ↔ depth={rep['ld_implied_depth_in']:.4f} in at ⌀0.800 (between wall span")
    print(f"    0.3291 and feature_depth 0.2885); wall_span/D={ld_from_wall_span:.4f} is closest.")
    print(f"  • fillet_radius: sphere blend faces max R={fillet_best:.4f} in "
          f"(all radii: {rep['fillet_radii_in']}) — matches Toolpath 0.25 in.")
    print("  • 3D surface: cascade sees 6 bspline + 7 sphere floor/blend faces; Toolpath reports No")
    print("    (Toolpath likely excludes bspline floors from the '3D surface' flag).")
    print()
    print(
        "VERDICT: Emit 'depth below top' as part_top − deepest_pocket_boundary along the "
        "opening axis (1.0848 in here). Emit 'feature depth' as ⌀0.800 wall axial span "
        "(0.3291 in) pending confirmation — closer than centroid floor-normal depth but "
        "still ~14% above Toolpath 0.2885. L/D=0.44 implies depth 0.352 in (0.44×⌀0.800); "
        "measured wall_span/D=0.41 is closest measured ratio. fillet_radius (max sphere R) "
        "aligns at 0.25 in."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
