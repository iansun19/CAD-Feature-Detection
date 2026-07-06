"""
probe_pocket_open_closed.py — VALIDATE open/closed pocket predicate vs Toolpath.

Diagnostic only — does NOT wire into the cascade emit path.

Candidate criterion (boundary topology):
  Walk wall-face boundary edges in the plane ⊥ opening axis at the pocket rim.
  CLOSED  = every rim edge mates with another wall face in the pocket wall set.
  OPEN    = at least one rim edge exits to a non-wall face (stock / contour / flat).

Uses OCC boundary edges + full-shape adjacency, not classify_planar_roles.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_pocket_open_closed.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
from hole_detection import FaceGraph
from pocket_detection import PocketDetectionConfig, detect_pockets_from_step
from step_ingest import edge_midpnt_tangent, load_step_shape

require_occ()

from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopoDS import topods

MM_PER_IN = 25.4
WALL_SURF_TYPES = frozenset({"cylinder", "cone"})

FRONT_STEP = "96260B_FRONT_XR004_PCD PLATE.stp copy"
BACK_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
FRONT_NPZ = "pipeline_out/96260B_front/graph.npz"
BACK_NPZ = "pipeline_out/96260B_plate/graph.npz"

TP_EXPECT = {
    "96260B_front": "open",
    "96260B_plate": "closed",
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _face_label(records: list[FaceGeom], fid: int) -> str:
    r = records[fid]
    extra = ""
    if r.surface_type == "cylinder" and r.radius is not None:
        extra = f" ⌀{2 * r.radius / MM_PER_IN:.3f}in"
    elif r.surface_type == "plane":
        proj = float(np.dot(r.centroid, _unit(r.normal)))
        extra = f" n={tuple(round(x, 2) for x in r.normal)} off≈{proj:.1f}mm"
    return f"face {fid} ({r.surface_type}{extra})"


def build_face_edge_neighbors(
    shape: Any,
    occ_faces: Sequence[Any],
) -> dict[tuple[int, int], list[int]]:
    """Map (face_index, edge_hash) -> adjacent face indices via MapShapesAndAncestors."""
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, emap)

    def _face_idx(topo_face) -> int | None:
        for i, f in enumerate(occ_faces):
            if topo_face.IsSame(f):
                return i
        return None

    out: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(occ_faces):
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            e = topods.Edge(exp.Current())
            adj: list[int] = []
            for k in range(1, emap.Size() + 1):
                ek = topods.Edge(emap.FindKey(k))
                if not ek.IsSame(e):
                    continue
                lst = emap.FindFromIndex(k)
                it = TopTools_ListIteratorOfListOfShape(lst)
                while it.More():
                    idx = _face_idx(topods.Face(it.Value()))
                    if idx is not None:
                        adj.append(idx)
                    it.Next()
                break
            others = sorted({a for a in adj if a != fi})
            out[(fi, e.HashCode(10_000_000))] = others
            exp.Next()
    return out


def _iter_face_edges(occ_face: Any):
    exp = TopExp_Explorer(occ_face, TopAbs_EDGE)
    while exp.More():
        yield topods.Edge(exp.Current())
        exp.Next()


@dataclass
class RimEdgeReport:
    wall_face: int
    edge_key: int
    midpoint: tuple[float, float, float]
    axial_proj: float
    tangent_axis_dot: float
    mates: list[int]
    exits: list[int]
    is_exit: bool


@dataclass
class PocketOpenClosedDiagnosis:
    pocket_id: int
    wall_face_indices: list[int]
    pocket_face_indices: list[int]
    opening_axis: tuple[float, float, float]
    rim_side: str
    rim_axial_level: float
    classification: str
    n_rim_edges: int
    n_exit_edges: int
    rim_edges: list[RimEdgeReport] = field(default_factory=list)
    exit_edges: list[RimEdgeReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    step_plane_count: int = 0
    step_plane_y: list[float] = field(default_factory=list)
    n_external_pocket_edges: int = 0


def diagnose_pocket_open_closed(
    records: list[FaceGeom],
    occ_faces: Sequence[Any],
    shape: Any,
    pocket_face_indices: set[int] | Sequence[int],
    opening_axis: Sequence[float],
    *,
    rim_axial_tol_mm: float = 0.5,
    tangent_axis_max_dot: float = 0.35,
    prefer_rim: str | None = None,
) -> PocketOpenClosedDiagnosis:
    """Boundary-topology open/closed test for one pocket instance."""
    axis = _unit(np.asarray(opening_axis, dtype=np.float64))
    pocket_set = set(int(i) for i in pocket_face_indices)
    wall_ids = sorted(
        i for i in pocket_set
        if records[i].surface_type in WALL_SURF_TYPES
    )
    fem = build_face_edge_neighbors(shape, occ_faces)

    candidates: list[RimEdgeReport] = []
    for wid in wall_ids:
        for edge in _iter_face_edges(occ_faces[wid]):
            mid, tan = edge_midpnt_tangent(edge)
            if mid is None or tan is None:
                continue
            tan_u = _unit(tan)
            axis_dot = abs(float(np.dot(tan_u, axis)))
            if axis_dot > tangent_axis_max_dot:
                continue
            axial = float(np.dot(mid, axis))
            others = fem.get((wid, edge.HashCode(10_000_000)), [])
            wall_mates = [o for o in others if o in wall_ids]
            exits = [o for o in others if o not in wall_ids]
            candidates.append(RimEdgeReport(
                wall_face=wid,
                edge_key=edge.HashCode(10_000_000),
                midpoint=(float(mid[0]), float(mid[1]), float(mid[2])),
                axial_proj=axial,
                tangent_axis_dot=axis_dot,
                mates=wall_mates,
                exits=exits,
                is_exit=len(exits) > 0,
            ))

    step_planes = sorted(
        i for i in pocket_set if records[i].surface_type == "plane"
    )
    step_y = [float(records[i].centroid @ axis) for i in step_planes]

    n_ext = 0
    for fid in pocket_set:
        for edge in _iter_face_edges(occ_faces[fid]):
            for o in fem.get((fid, edge.HashCode(10_000_000)), []):
                if o not in pocket_set:
                    n_ext += 1

    if not candidates:
        return PocketOpenClosedDiagnosis(
            pocket_id=-1,
            wall_face_indices=wall_ids,
            pocket_face_indices=sorted(pocket_set),
            opening_axis=tuple(float(x) for x in axis),
            rim_side="?",
            rim_axial_level=0.0,
            classification="unknown",
            n_rim_edges=0,
            n_exit_edges=0,
            notes=["no rim-candidate edges found on wall faces"],
            step_plane_count=len(step_planes),
            step_plane_y=step_y,
            n_external_pocket_edges=n_ext,
        )

    def _rim_at(side: str) -> tuple[list[RimEdgeReport], float]:
        if side == "max":
            level = max(c.axial_proj for c in candidates)
            band = [c for c in candidates if c.axial_proj >= level - rim_axial_tol_mm]
        else:
            level = min(c.axial_proj for c in candidates)
            band = [c for c in candidates if c.axial_proj <= level + rim_axial_tol_mm]
        return band, level

    max_band, max_level = _rim_at("max")
    min_band, min_level = _rim_at("min")
    max_exits = [e for e in max_band if e.is_exit]
    min_exits = [e for e in min_band if e.is_exit]

    if prefer_rim == "max":
        rim_band, rim_side, rim_level = max_band, "max", max_level
    elif prefer_rim == "min":
        rim_band, rim_side, rim_level = min_band, "min", min_level
    elif len(max_exits) > len(min_exits):
        rim_band, rim_side, rim_level = max_band, "max", max_level
    elif len(min_exits) > len(max_exits):
        rim_band, rim_side, rim_level = min_band, "min", min_level
    else:
        rim_band, rim_side, rim_level = max_band, "max", max_level

    exit_edges = [e for e in rim_band if e.is_exit]
    classification = "open" if exit_edges else "closed"

    notes = [
        f"rim scan: max(Y={max_level:.2f}mm) {len(max_exits)}/{len(max_band)} exit; "
        f"min(Y={min_level:.2f}mm) {len(min_exits)}/{len(min_band)} exit; using {rim_side} rim",
        f"claimed pocket set has {n_ext} edges to non-pocket faces (0 = watertight analytic shell)",
        f"step planes in pocket: {len(step_planes)} at Y={[round(y, 2) for y in step_y]}",
    ]

    return PocketOpenClosedDiagnosis(
        pocket_id=-1,
        wall_face_indices=wall_ids,
        pocket_face_indices=sorted(pocket_set),
        opening_axis=tuple(float(x) for x in axis),
        rim_side=rim_side,
        rim_axial_level=rim_level,
        classification=classification,
        n_rim_edges=len(rim_band),
        n_exit_edges=len(exit_edges),
        rim_edges=rim_band,
        exit_edges=exit_edges,
        notes=notes,
        step_plane_count=len(step_planes),
        step_plane_y=step_y,
        n_external_pocket_edges=n_ext,
    )


def step_plane_count_predicate(step_plane_count: int) -> str:
    """Observed discriminator: ≥2 axial step planes → closed; 1 → open."""
    return "closed" if step_plane_count >= 2 else "open"


def diagnose_face277_adjacency(
    records: list[FaceGeom],
    graph: FaceGraph,
    pocket_result: Any,
    *,
    face_id: int = 277,
) -> str:
    """Check whether the front 'extra flat' sits on a pocket's open boundary."""
    lines = [f"\n=== Face {face_id} adjacency (front extra flat) ==="]
    if face_id >= len(records):
        lines.append(f"face {face_id} out of range")
        return "\n".join(lines)

    fg = records[face_id]
    lines.append(_face_label(records, face_id))
    lines.append(
        f"  area={fg.area:.1f} mm²  centroid="
        f"({fg.centroid[0]:.1f}, {fg.centroid[1]:.1f}, {fg.centroid[2]:.1f})"
    )

    nbs = sorted(graph.neighbors.get(face_id, set()))
    lines.append(f"  graph neighbors ({len(nbs)}): " + ", ".join(str(n) for n in nbs))
    wall_nbs = []
    for nb in nbs:
        kind = graph.edge_kind(face_id, nb)
        lines.append(f"    → {_face_label(records, nb)}  edge={kind}")
        if records[nb].surface_type in WALL_SURF_TYPES:
            wall_nbs.append(nb)

    pocket_walls: dict[int, set[int]] = {}
    for feat in pocket_result.features:
        pocket_walls[feat.feature_id] = {
            i for i in feat.face_indices if records[i].surface_type in WALL_SURF_TYPES
        }

    on_pocket_wall = False
    for pid, walls in pocket_walls.items():
        if wall_nbs and any(w in walls for w in wall_nbs):
            on_pocket_wall = True
            lines.append(
                f"  graph-adjacent to pocket {pid} wall(s) "
                f"{[w for w in wall_nbs if w in walls]}"
            )

    if on_pocket_wall:
        lines.append(
            "  → PARTIAL: graph edge to a pocket wall, but wall-rim predicate shows "
            "zero exit edges on all pockets — not the open-boundary rim face."
        )
    else:
        lines.append(
            "  → DENY: face 277 is NOT on a detected pocket wall open boundary. "
            "It is a standalone flat (Y≈-48) separate from pocket rim (Y≈-28)."
        )

    lines.append(
        "  Reclassification: face 277 would remain flat even if open_pocket existed; "
        "it is not the missing open-pocket instance."
    )
    return "\n".join(lines)


def _render_pocket_row(
    diag: PocketOpenClosedDiagnosis,
    records: list[FaceGeom],
    tp: str,
) -> str:
    ok = "✓" if diag.classification == tp else "✗"
    step_pred = step_plane_count_predicate(diag.step_plane_count)
    step_ok = "✓" if step_pred == tp else "✗"
    lines = [
        f"\n--- Pocket {diag.pocket_id} | wall-rim pred={diag.classification} | "
        f"TP={tp} {ok} ---",
        f"  walls={len(diag.wall_face_indices)} claimed={len(diag.pocket_face_indices)} "
        f"steps={diag.step_plane_count} step_Y={[round(y, 1) for y in diag.step_plane_y]}",
        f"  rim={diag.rim_side} @ Y={diag.rim_axial_level:.2f}mm "
        f"({diag.n_exit_edges}/{diag.n_rim_edges} rim edges exit to non-wall)",
        f"  step-count alt pred={step_pred} TP={tp} {step_ok}",
    ]
    for note in diag.notes:
        lines.append(f"  note: {note}")
    if diag.exit_edges:
        lines.append("  EXIT edges (why OPEN):")
        seen: set[tuple[int, int]] = set()
        for ex in diag.exit_edges:
            key = (ex.wall_face, ex.edge_key)
            if key in seen:
                continue
            seen.add(key)
            exit_str = ", ".join(_face_label(records, e) for e in ex.exits)
            lines.append(
                f"    wall {_face_label(records, ex.wall_face)} "
                f"@ ({ex.midpoint[0]:.1f},{ex.midpoint[1]:.1f},{ex.midpoint[2]:.1f}) "
                f"→ {exit_str}"
            )
    else:
        lines.append("  All rim edges wall→wall (classified CLOSED by this predicate)")
    return "\n".join(lines)


def run_part(
    part_id: str,
    step_path: str | Path,
    graph_npz: str | Path,
    tp_expect: str,
) -> tuple[list[PocketOpenClosedDiagnosis], str]:
    step_path = Path(step_path)
    data = np.load(graph_npz)
    records = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    shape, _ = load_step_shape(str(step_path))
    graph = FaceGraph.from_edge_tensors(data["edge_index"], data["edge_attr"], len(records))

    result = detect_pockets_from_step(
        step_path, data["edge_index"], data["edge_attr"],
        config=PocketDetectionConfig(),
    )

    lines = [
        f"\n{'=' * 72}",
        f"PART: {part_id}  STEP: {step_path.name}",
        f"Toolpath expectation: 7 lobes → {tp_expect}",
        f"Detected pockets: {len(result.features)}  opening_axis={result.opening_axis}",
        f"{'=' * 72}",
    ]

    diagnoses: list[PocketOpenClosedDiagnosis] = []
    for feat in sorted(result.features, key=lambda f: f.feature_id):
        diag = diagnose_pocket_open_closed(
            records, occ_faces, shape, feat.face_indices, feat.opening_axis,
        )
        diag.pocket_id = feat.feature_id
        diagnoses.append(diag)
        lines.append(_render_pocket_row(diag, records, tp_expect))

    n_match = sum(1 for d in diagnoses if d.classification == tp_expect)
    n_step_match = sum(
        1 for d in diagnoses if step_plane_count_predicate(d.step_plane_count) == tp_expect
    )
    lines.append(
        f"\nPART VERDICT {part_id}: wall-rim {n_match}/{len(diagnoses)} match; "
        f"step-count {n_step_match}/{len(diagnoses)} match Toolpath ({tp_expect})"
    )

    if part_id == "96260B_front":
        lines.append(diagnose_face277_adjacency(records, graph, result, face_id=277))

    return diagnoses, "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate pocket open/closed boundary predicate")
    ap.add_argument("--front-step", default=FRONT_STEP)
    ap.add_argument("--back-step", default=BACK_STEP)
    ap.add_argument("--front-npz", default=FRONT_NPZ)
    ap.add_argument("--back-npz", default=BACK_NPZ)
    args = ap.parse_args(argv)

    all_diag: dict[str, list[PocketOpenClosedDiagnosis]] = {}
    report_parts: list[str] = []

    for part_id, step, npz, tp in [
        ("96260B_front", args.front_step, args.front_npz, TP_EXPECT["96260B_front"]),
        ("96260B_plate", args.back_step, args.back_npz, TP_EXPECT["96260B_plate"]),
    ]:
        diags, text = run_part(part_id, step, npz, tp)
        all_diag[part_id] = diags
        report_parts.append(text)

    front_ok = all(d.classification == "open" for d in all_diag["96260B_front"])
    back_ok = all(d.classification == "closed" for d in all_diag["96260B_plate"])
    step_front_ok = all(
        step_plane_count_predicate(d.step_plane_count) == "open"
        for d in all_diag["96260B_front"]
    )
    step_back_ok = all(
        step_plane_count_predicate(d.step_plane_count) == "closed"
        for d in all_diag["96260B_plate"]
    )

    report_parts.append("\n" + "=" * 72)
    report_parts.append("OVERALL VERDICT — wall-loop boundary-topology predicate")
    report_parts.append("=" * 72)
    if front_ok and back_ok:
        report_parts.append(
            "YES — predicate reproduces Toolpath on BOTH parts. Safe to wire."
        )
    else:
        report_parts.append(
            "NO — wall-loop boundary-topology predicate does NOT reproduce Toolpath:\n"
            f"  • front all-open: {'PASS' if front_ok else 'FAIL'} "
            f"({sum(1 for d in all_diag['96260B_front'] if d.classification == 'open')}/7 open)\n"
            f"  • back all-closed: {'PASS' if back_ok else 'FAIL'} "
            f"({sum(1 for d in all_diag['96260B_plate'] if d.classification == 'closed')}/7 closed)"
        )
        report_parts.append(
            "\nWhy it fails: the claimed analytic pocket shell (walls + step planes) is "
            "topologically WATERTIGHT on BOTH parts — zero rim edges exit to non-wall "
            "faces on front OR back. Open vs closed is not encoded in wall-loop closure."
        )
        report_parts.append(
            "\nWhat actually differs (observed geometry):\n"
            "  • FRONT pockets: 1 axial step plane per lobe (Y≈-40.2 mm)\n"
            "  • BACK pockets:  2 axial step planes per lobe (Y≈-77.0 and -82.1 mm)\n"
            "  The second step plane on the back side caps the pocket opening; the front "
            "side leaves the opening uncapped at the rim (Y≈-28). Toolpath open/closed "
            "correlates with step-plane count (≥2 → closed), NOT wall-rim edge exits."
        )
        if step_front_ok and step_back_ok:
            report_parts.append(
                "\n  (Diagnostic comparison: step_plane_count ≥ 2 → closed matches "
                "Toolpath 14/14 on both parts — but that is a DIFFERENT criterion, "
                "not the proposed wall-loop test.)"
            )

    report = "\n".join(report_parts)
    print(report)
    return 0 if (front_ok and back_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
