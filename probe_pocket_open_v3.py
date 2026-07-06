"""
probe_pocket_open_v3.py — VALIDATE cap-agnostic gateway open/closed pocket rule.

Diagnostic only — does NOT wire into the cascade emit path.

Rule (cap-agnostic, gateway-based):
  For each pocket:
    1. Inboard wall = per-pocket ⌀2.867in concave cylinder (bore-facing wall).
    2. Pocket neighborhood N = claimed analytic faces + released sculpted floors.
    3. Gateway = inboard-wall boundary edge(s) toward central cavity (typically
       inboard wall → shared central cone, e.g. face 251→91 FRONT).
    4. Scan N plus 1-hop adjacent contour for CAP faces across the bore-facing exit:
       any non-wall face in the inboard radial band (between inboard wall and outer
       walls) at the deep axial end of the pocket (Y ≤ deepest wall Y + tol).
       Cap surface types: plane, bspline, bezier, sphere (not wall cylinders/cones;
       torus blend ring excluded — fillet transition, not a bore cap).
    5. OPEN  = gateway exists AND no cap face blocks the bore-facing exit.
    6. CLOSED = at least one cap face in the gateway corridor.

Does NOT use step_plane_count or cap surface type (step vs sculpted). Step count is
reported for comparison only.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_pocket_open_v3.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
from hole_detection import HoleDetectionConfig, detect_holes
from pocket_detection import PocketDetectionConfig, PocketFeature, detect_pockets_from_step
from probe_pocket_open_v2 import (
    CENTRAL_HOLE_DIA_IN,
    FRONT_NPZ,
    FRONT_STEP,
    BACK_NPZ,
    BACK_STEP,
    INBOARD_UV_MARGIN_MM,
    INBOARD_WALL_DIA_IN,
    TP_EXPECT,
    _dia_in,
    _face_label,
    _inboard_wall_ids,
    _near_dia,
    _project_uv,
    _unit,
    build_face_adjacency,
    identify_central_bore_faces,
)
from step_ingest import load_step_shape

require_occ()

MM_PER_IN = 25.4
WALL_DIA_IN = (0.800, 0.500, 3.453, 2.867)
CAP_SURFACE_TYPES = frozenset({"plane", "bspline", "bezier", "sphere"})
DEEP_AXIAL_TOL_MM = 1.0
GATEWAY_UV_TOL_MM = 8.0


def _is_pocket_wall(records: list[FaceGeom], fid: int) -> bool:
    fg = records[fid]
    if fg.surface_type not in ("cylinder", "cone"):
        return False
    return any(_near_dia(_dia_in(fg), t) for t in WALL_DIA_IN)


@dataclass
class GatewayEdge:
    inboard_wall: int
    outside_face: int
    outside_type: str
    outside_uv: float
    outside_y: float
    primary: bool


@dataclass
class CapFace:
    face_id: int
    surface_type: str
    uv_radius: float
    axial_y: float
    source: str  # claimed | released | contour
    in_claimed: bool


@dataclass
class RejectedCandidate:
    face_id: int
    surface_type: str
    reason: str


@dataclass
class PocketGatewayDiagnosis:
    pocket_id: int
    inboard_wall_ids: list[int]
    gateway_edges: list[GatewayEdge]
    cap_faces: list[CapFace]
    rejected_near_caps: list[RejectedCandidate]
    classification: str
    inboard_uv: float
    outer_uv: float
    deep_wall_y: float
    inboard_band: tuple[float, float]
    step_plane_count: int
    notes: list[str] = field(default_factory=list)


def _face_source(fid: int, claimed: set[int], released: set[int]) -> str:
    if fid in claimed:
        return "claimed"
    if fid in released:
        return "released"
    return "contour"


def _gateway_edges(
    records: list[FaceGeom],
    inboard_ids: list[int],
    neighborhood: set[int],
    adj: dict[int, set[int]],
    axis: np.ndarray,
    inboard_uv: float,
) -> list[GatewayEdge]:
    edges: list[GatewayEdge] = []
    min_cone_uv = float("inf")
    cone_candidates: list[GatewayEdge] = []

    for wid in inboard_ids:
        w_uv = float(np.linalg.norm(_project_uv(records[wid].centroid, axis)))
        for nb in sorted(adj.get(wid, set())):
            if nb in neighborhood:
                continue
            nb_uv = float(np.linalg.norm(_project_uv(records[nb].centroid, axis)))
            nb_y = float(records[nb].centroid @ axis)
            st = records[nb].surface_type
            ge = GatewayEdge(
                inboard_wall=wid,
                outside_face=nb,
                outside_type=st,
                outside_uv=nb_uv,
                outside_y=nb_y,
                primary=False,
            )
            edges.append(ge)
            if st in ("cone", "cylinder") and nb_uv < w_uv + GATEWAY_UV_TOL_MM:
                cone_candidates.append(ge)
                min_cone_uv = min(min_cone_uv, nb_uv)

    primary_uv = min_cone_uv if cone_candidates else None
    for ge in edges:
        if primary_uv is not None and ge.outside_type in ("cone", "cylinder"):
            ge.primary = abs(ge.outside_uv - primary_uv) < 0.5
    return edges


def _scan_caps_and_rejects(
    records: list[FaceGeom],
    scan_set: set[int],
    claimed: set[int],
    released: set[int],
    inboard_ids: list[int],
    axis: np.ndarray,
    inboard_uv: float,
    outer_uv: float,
    deep_wall_y: float,
) -> tuple[list[CapFace], list[RejectedCandidate]]:
    lo = inboard_uv + INBOARD_UV_MARGIN_MM
    hi = outer_uv - INBOARD_UV_MARGIN_MM
    caps: list[CapFace] = []
    rejected: list[RejectedCandidate] = []

    for fid in sorted(scan_set):
        fg = records[fid]
        st = fg.surface_type
        if _is_pocket_wall(records, fid):
            continue
        uv = float(np.linalg.norm(_project_uv(fg.centroid, axis)))
        y = float(fg.centroid @ axis)

        if not (lo < uv < hi):
            if st in CAP_SURFACE_TYPES and fid in (claimed | released):
                rejected.append(RejectedCandidate(
                    fid, st, f"uv_r={uv:.1f} outside inboard band ({lo:.1f},{hi:.1f})"
                ))
            continue

        if st not in CAP_SURFACE_TYPES:
            if st == "torus":
                rejected.append(RejectedCandidate(
                    fid, st, f"blend ring, not bore cap (uv_r={uv:.1f} Y={y:.1f})"
                ))
            continue

        src = _face_source(fid, claimed, released)
        if y > deep_wall_y + DEEP_AXIAL_TOL_MM:
            rejected.append(RejectedCandidate(
                fid, st,
                f"rim/floor blend not deep cap (Y={y:.1f} > deep_wall={deep_wall_y:.1f})",
            ))
            continue

        caps.append(CapFace(
            face_id=fid,
            surface_type=st,
            uv_radius=uv,
            axial_y=y,
            source=src,
            in_claimed=fid in claimed,
        ))

    return caps, rejected


def diagnose_pocket_gateway(
    records: list[FaceGeom],
    feat: PocketFeature,
    adj: dict[int, set[int]],
    opening_axis: np.ndarray,
) -> PocketGatewayDiagnosis:
    axis = _unit(opening_axis)
    claimed = set(feat.face_indices)
    released = set(feat.released_faces)
    neighborhood = claimed | released

    contour: set[int] = set()
    for fid in neighborhood:
        for nb in adj.get(fid, set()):
            if nb not in neighborhood:
                contour.add(nb)
    scan_set = neighborhood | contour

    inboard_ids = _inboard_wall_ids(records, claimed)
    walls = [i for i in claimed if records[i].surface_type in ("cylinder", "cone")]
    inboard_uv = min(
        float(np.linalg.norm(_project_uv(records[i].centroid, axis)))
        for i in inboard_ids
    )
    outer_uv = max(
        float(np.linalg.norm(_project_uv(records[i].centroid, axis)))
        for i in walls
    )
    deep_wall_y = min(float(records[w].centroid @ axis) for w in walls)

    gateways = _gateway_edges(records, inboard_ids, neighborhood, adj, axis, inboard_uv)
    caps, rejected = _scan_caps_and_rejects(
        records, scan_set, claimed, released, inboard_ids,
        axis, inboard_uv, outer_uv, deep_wall_y,
    )

    steps = [i for i in claimed if records[i].surface_type == "plane"]
    has_gateway = any(g.primary for g in gateways) or bool(gateways)

    if caps:
        classification = "closed"
    elif has_gateway:
        classification = "open"
    else:
        classification = "unknown"

    cap_types = sorted({c.surface_type for c in caps})
    cap_sources = sorted({c.source for c in caps})
    notes = [
        f"neighborhood={len(neighborhood)} (claimed={len(claimed)}, released={len(released)}), "
        f"contour scan +{len(contour)}",
        f"inboard wall(s)={inboard_ids} uv_r={inboard_uv:.1f}; outer uv_r={outer_uv:.1f}; "
        f"deep_wall_Y={deep_wall_y:.1f}; band=({inboard_uv + INBOARD_UV_MARGIN_MM:.1f}, "
        f"{outer_uv - INBOARD_UV_MARGIN_MM:.1f})",
        f"gateway edges={len(gateways)}, primary cones="
        f"{sorted({g.outside_face for g in gateways if g.primary})}",
    ]
    if caps:
        notes.append(
            f"CAP across bore-facing exit: {len(caps)} face(s) types={cap_types} "
            f"sources={cap_sources}"
        )
    else:
        notes.append("no cap in gateway corridor → OPEN")

    return PocketGatewayDiagnosis(
        pocket_id=feat.feature_id,
        inboard_wall_ids=inboard_ids,
        gateway_edges=gateways,
        cap_faces=caps,
        rejected_near_caps=rejected,
        classification=classification,
        inboard_uv=inboard_uv,
        outer_uv=outer_uv,
        deep_wall_y=deep_wall_y,
        inboard_band=(inboard_uv + INBOARD_UV_MARGIN_MM, outer_uv - INBOARD_UV_MARGIN_MM),
        step_plane_count=len(steps),
        notes=notes,
    )


def step_plane_count_predicate(n: int) -> str:
    return "closed" if n >= 2 else "open"


def _render_pocket(diag: PocketGatewayDiagnosis, records: list[FaceGeom], tp: str) -> str:
    ok = "✓" if diag.classification == tp else "✗"
    step_alt = step_plane_count_predicate(diag.step_plane_count)
    lines = [
        f"\n--- Pocket {diag.pocket_id} | pred={diag.classification} | TP={tp} {ok} ---",
    ]
    for note in diag.notes:
        lines.append(f"  note: {note}")

    prim = [g for g in diag.gateway_edges if g.primary]
    lines.append(f"  GATEWAY ({len(prim)} primary bore-facing exit(s)):")
    if prim:
        for g in prim:
            lines.append(
                f"    {_face_label(records, g.inboard_wall)} → "
                f"{_face_label(records, g.outside_face)}  "
                f"(uv_r={g.outside_uv:.1f} Y={g.outside_y:.1f})"
            )
    else:
        for g in diag.gateway_edges[:4]:
            lines.append(
                f"    {_face_label(records, g.inboard_wall)} → "
                f"{_face_label(records, g.outside_face)}"
            )

    if diag.classification == "closed":
        lines.append(f"  CAP faces blocking bore-facing exit ({len(diag.cap_faces)}):")
        for c in diag.cap_faces:
            lines.append(
                f"    {_face_label(records, c.face_id)}  source={c.source} "
                f"claimed={c.in_claimed} uv_r={c.uv_radius:.1f} Y={c.axial_y:.1f}"
            )
    else:
        lines.append("  CAP faces: none (exit open to central cavity via gateway)")

    rim_blends = [r for r in diag.rejected_near_caps if "rim/floor blend" in r.reason]
    if rim_blends:
        lines.append(f"  Near-cap candidates rejected ({len(rim_blends)}):")
        for r in rim_blends[:5]:
            lines.append(f"    {_face_label(records, r.face_id)}: {r.reason}")
        if len(rim_blends) > 5:
            lines.append(f"    ... +{len(rim_blends) - 5} more")

    lines.append(
        f"  step-count alt={step_alt} (comparison ONLY, not criterion)"
    )
    return "\n".join(lines)


def run_part(
    part_id: str,
    step_path: str | Path,
    graph_npz: str | Path,
    tp_expect: str,
) -> tuple[list[PocketGatewayDiagnosis], str]:
    step_path = Path(step_path)
    data = np.load(graph_npz)
    records = analyze_step(step_path)
    adj = build_face_adjacency(step_path, len(records))

    pocket_result = detect_pockets_from_step(
        step_path, data["edge_index"], data["edge_attr"],
        config=PocketDetectionConfig(),
    )
    axis = np.array(pocket_result.opening_axis, float)

    hole_result = detect_holes(
        records, data["edge_index"], data["edge_attr"],
        occ_faces=load_step_faces(step_path),
        candidate_faces=pocket_result.remaining_faces,
        config=HoleDetectionConfig(max_hole_diameter_mm=150.0),
    )
    hole_faces = set()
    for hf in hole_result.features:
        hole_faces |= hf.face_indices
    bore_faces, bore_detail = identify_central_bore_faces(records, hole_faces, axis, adj)

    lines = [
        f"\n{'=' * 72}",
        f"PART: {part_id}  STEP: {step_path.name}",
        f"Toolpath expectation: 7 lobes → {tp_expect}",
        f"Detected pockets: {len(pocket_result.features)}  "
        f"opening_axis={tuple(round(x, 3) for x in axis)}",
        f"Central bore: {bore_detail}",
        f"{'=' * 72}",
    ]

    diagnoses: list[PocketGatewayDiagnosis] = []
    for feat in sorted(pocket_result.features, key=lambda f: f.feature_id):
        diag = diagnose_pocket_gateway(records, feat, adj, axis)
        diagnoses.append(diag)
        lines.append(_render_pocket(diag, records, tp_expect))

    n_match = sum(1 for d in diagnoses if d.classification == tp_expect)
    lines.append(
        f"\nPART SUMMARY {part_id}: gateway-cap rule {n_match}/{len(diagnoses)} match TP ({tp_expect})"
    )
    return diagnoses, "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate cap-agnostic gateway open/closed rule")
    ap.add_argument("--front-step", default=FRONT_STEP)
    ap.add_argument("--back-step", default=BACK_STEP)
    ap.add_argument("--front-npz", default=FRONT_NPZ)
    ap.add_argument("--back-npz", default=BACK_NPZ)
    args = ap.parse_args(argv)

    all_diag: dict[str, list[PocketGatewayDiagnosis]] = {}
    report: list[str] = []

    for part_id, step, npz, tp in [
        ("96260B_front", args.front_step, args.front_npz, TP_EXPECT["96260B_front"]),
        ("96260B_plate", args.back_step, args.back_npz, TP_EXPECT["96260B_plate"]),
    ]:
        diags, text = run_part(part_id, step, npz, tp)
        all_diag[part_id] = diags
        report.append(text)

    front_ok = all(d.classification == "open" for d in all_diag["96260B_front"])
    back_ok = all(d.classification == "closed" for d in all_diag["96260B_plate"])

    # Cap type breakdown on BACK
    back_caps = [c for d in all_diag["96260B_plate"] for c in d.cap_faces]
    back_cap_types = sorted({c.surface_type for c in back_caps})
    back_cap_sources = sorted({c.source for c in back_caps})
    back_released_caps = [c for c in back_caps if c.source == "released"]
    back_contour_caps = [c for c in back_caps if c.source == "contour"]

    step_agrees = all(
        d.classification == step_plane_count_predicate(d.step_plane_count)
        for d in all_diag["96260B_front"] + all_diag["96260B_plate"]
    )
    cap_driven = all(
        (d.classification == "closed") == bool(d.cap_faces)
        for d in all_diag["96260B_front"] + all_diag["96260B_plate"]
    )

    report.append("\n" + "=" * 72)
    report.append("REPRODUCTION TABLE")
    report.append("=" * 72)
    report.append(
        f"{'part':<16} {'pkt':>3} {'pred':>8} {'TP':>8} {'ok':>3} "
        f"{'gateway':>12} {'cap_faces':>20} {'cap_types':>12} {'step#':>5}"
    )
    for part_id, tp in TP_EXPECT.items():
        for d in all_diag[part_id]:
            gw = sorted({g.outside_face for g in d.gateway_edges if g.primary})
            caps = ",".join(str(c.face_id) for c in d.cap_faces) or "—"
            ctypes = ",".join(sorted({c.surface_type for c in d.cap_faces})) or "—"
            report.append(
                f"{part_id:<16} {d.pocket_id:>3} {d.classification:>8} {tp:>8} "
                f"{'✓' if d.classification == tp else '✗':>3} "
                f"{str(gw):>12} {caps:>20} {ctypes:>12} {d.step_plane_count:>5}"
            )

    report.append("\n" + "=" * 72)
    report.append("BACK CAP DECOMPOSITION (check #2)")
    report.append("=" * 72)
    report.append(f"Cap surface types on BACK: {back_cap_types}")
    report.append(f"Cap sources on BACK: {back_cap_sources}")
    report.append(f"Released sculpted caps on BACK: {len(back_released_caps)} "
                  f"{[(c.face_id, c.surface_type) for c in back_released_caps]}")
    report.append(f"Contour caps on BACK: {len(back_contour_caps)}")
    if back_cap_types == ["plane"] and not back_released_caps:
        report.append(
            "On THIS part the bore-facing cap is ONLY the inboard step plane (claimed). "
            "Released bspline/sphere floors at the rim (Y≈-68) do NOT cap the exit — "
            "they are rejected as rim blends (Y > deep_wall_Y). The rule is cap-agnostic "
            "and would catch a sculpted cap at the deep inboard band if present."
        )

    report.append("\n" + "=" * 72)
    report.append("CRITERION INDEPENDENCE (check #3)")
    report.append("=" * 72)
    report.append(f"Classification driven by cap_faces presence: {cap_driven}")
    report.append(f"Agrees with step_plane_count on all 14: {step_agrees}")
    report.append(
        "Open/closed is NOT decided by counting steps — it is decided by whether a "
        "non-wall cap face (any of plane/bspline/bezier/sphere) sits in the inboard "
        "radial band at the deep end of the pocket. On this 2-part set, step-count "
        "agrees only because BACK's extra step IS that deep inboard cap; FRONT has "
        "no deep inboard cap regardless of step count."
    )

    report.append("\n" + "=" * 72)
    report.append("OVERALL VERDICT")
    report.append("=" * 72)
    if front_ok and back_ok and cap_driven:
        report.append(
            "VALIDATED — cap-agnostic gateway rule reproduces FRONT 7/7 open, BACK 7/7 closed.\n\n"
            "Rule to wire:\n"
            "  1. inboard_wall = ⌀2.867in concave cylinder in claimed pocket.\n"
            "  2. N = claimed + released sculpted floors; scan_set = N ∪ 1-hop contour.\n"
            "  3. deep_wall_Y = min axial Y of claimed pocket walls.\n"
            "  4. inboard band = (inboard_wall_uv + margin, outer_wall_uv − margin).\n"
            "  5. cap = face in scan_set, not a pocket wall, type ∈ {plane,bspline,bezier,sphere},\n"
            "     in inboard band, axial Y ≤ deep_wall_Y + tol.\n"
            "  6. gateway = inboard_wall → outside cone/cylinder toward central cavity.\n"
            "  7. CLOSED if any cap; OPEN if no cap and gateway exists.\n"
            "  Do NOT use step_plane_count."
        )
    else:
        report.append(
            f"NOT VALIDATED: FRONT {'PASS' if front_ok else 'FAIL'}, "
            f"BACK {'PASS' if back_ok else 'FAIL'}"
        )

    text = "\n".join(report)
    print(text)
    return 0 if (front_ok and back_ok and cap_driven) else 1


if __name__ == "__main__":
    raise SystemExit(main())
