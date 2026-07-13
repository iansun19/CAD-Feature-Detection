"""countersink_detection.py — new cascade feature: the countersunk through-hole.

countersink = a coaxial CONE that widens the MOUTH of a through-bore so a
flush-seating fastener head sits at/below the surface. Its geometric signature
is a coaxial stack:

    conical entry (flares from bore radius UP to R_cone at the surface)
        ↕
    smaller coaxial bore cylinder (R_bore) continuing THROUGH the part

Unlike a counterbore, the widening entry is a CONE (tapered), not a flat-bottomed
cylindrical enlargement — so there is NO flat annular shoulder plane between two
coaxial cylinders. That difference is exactly what ``counterbore_detection``
requires and this pass forbids: the two features are mutually exclusive.

WHY THIS RUNS ON hole_result.features (post-hole, like counterbore)
-------------------------------------------------------------------
The hole pass ALREADY groups each cone+bore cluster into ONE ``through_hole``
feature carrying every face, and already sets ``is_countersink`` (a coaxial cone
is present) and ``is_counterbore`` (a stepped second bore radius is present). The
cone is NOT lost — it is claimed and grouped correctly; only the emitted LABEL is
wrong (no ``countersink`` toolpath class existed). So this pass does not re-detect
or reorder anything: it reclassifies qualifying hole features and the graph
builder emits them as ``countersink`` nodes, suppressing the absorbed
``through_hole`` features (identical mechanism to counterbore_detection). No face
moves; no downstream pass sees a different pool.

Consuming the POST-POCKET hole set is load-bearing for golden safety. fish_mold
has 5 coned "through holes" when the hole pass is run on the RAW face pool, but
in the real cascade POCKETS claim those faces first, so they never reach the hole
pass — they are absent from ``hole_result.features`` and cannot qualify here.

THE PREDICATE (relative/derived quantities only — no ids, no part gate, no mm)
    (1) kind == "through_hole"     — open both ends (excludes blind & drill-tip
                                     cones, e.g. part1's drilled_blind_hole).
    (2) is_countersink             — >=1 coaxial cone in the wall cluster.
    (3) not is_counterbore         — a single bore radius (rejects the flat-bottom
                                     cylindrical counterbore look-alike).
    (4) cone flares WIDER than the bore: max cone boundary radius > bore radius
                                     — the "conical WIDENING entry" test; rejects
                                     a narrowing drill-tip / degenerate cone.
The cone HALF-ANGLE is deliberately NOT used as a discriminator: Part3 and
fish_mold cones are both 45 deg, so it separates nothing.

Clauses 1-3 are jointly sufficient on the current corpus (empty on all five
frozen parts, exactly 10 groups on Part3). Clause 4 is an INDEPENDENT forward
robustness guard — NOT load-bearing on today's frozen set — kept so a future
non-widening cone (drill tip / degenerate flare) cannot be mislabeled, and as a
backstop should pass order ever change so fish_mold's degenerate cones reach the
hole pass (their flare does exceed the bore, so clause 4 alone would not catch
them; only the pocket-first ordering does — see note above).

Validated selection (over hole_result.features in the FULL cascade):
    96260B_front : {}   96260B_rear : {}   fish_mold : {}   part1 : {}   part2 : {}
    part3        : 10 countersinks (40 faces) — the target groups, exactly.

KNOWN LIMITATIONS (documented, not present on Part3)
  * A countersink over a BLIND hole reads kind blind_hole/drilled_blind_hole and
    is rejected by clause 1 (stays a blind hole).
  * A chamfered through-hole is geometrically a shallow countersink and WOULD be
    labeled countersink here (a chamfer is a shallow countersink); no magnitude
    split is attempted (would need a magic threshold).
  * A countersink whose bore was claimed elsewhere leaves a lone cone, which the
    hole pass drops as a chamfer (needs >=1 cylinder) — so it is not detected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from brep.feature_params import FaceGeom


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CountersinkDetectionConfig:
    # clause 4: cone must flare strictly wider than the bore. A tiny relative
    # epsilon (fraction of the bore radius) keeps a flush no-flare cone out; it is
    # relative to feature geometry, not an absolute mm.
    flare_margin_frac: float = 0.0


# ---------------------------------------------------------------------------
# Enriched record — the predicate operates on these only, so the self-test can
# build them directly without OCC.
# ---------------------------------------------------------------------------
@dataclass
class _ECS:
    feature_id: int
    kind: str
    is_countersink: bool
    is_counterbore: bool
    bore_radius: float
    cone_flare_radius: float          # max cone boundary radius (widest point)
    face_indices: set[int]
    cone_face_indices: list[int]
    cylinder_face_indices: list[int]
    axis_point: list[float]
    axis_direction: list[float]
    half_angle_deg: float | None


def _qualifies(rec: _ECS, cfg: CountersinkDetectionConfig) -> bool:
    """Pure geometry-only countersink predicate (OCC-free)."""
    if rec.kind != "through_hole":            # (1)
        return False
    if not rec.is_countersink:                # (2)
        return False
    if rec.is_counterbore:                    # (3)
        return False
    if not rec.cone_face_indices:             # (2, defensive)
        return False
    # (4) cone flares strictly wider than the bore
    if rec.bore_radius <= 0.0:
        return False
    return rec.cone_flare_radius > rec.bore_radius * (1.0 + cfg.flare_margin_frac)


# ---------------------------------------------------------------------------
# OCC flare measurement (boundary max radius of a cone face about the hole axis)
# ---------------------------------------------------------------------------
def _cone_flare_radius(
    feat: Any,
    faces: Sequence[FaceGeom],
    occ_faces: Sequence[Any] | None,
) -> float:
    """Max radial distance of any cone-face boundary point from the hole axis.

    With OCC faces this is the physical widest radius of the conical entry. In
    the OCC-free path (self-test) it falls back to the ``radius`` attribute on the
    cone FaceGeom stub (which the self-test sets to the intended flare).
    """
    if not feat.cone_face_indices:
        return 0.0
    if occ_faces is None:
        vals = [float(faces[c].radius) for c in feat.cone_face_indices
                if faces[c].radius is not None]
        return max(vals) if vals else 0.0
    from brep.brep_extents import collect_boundary_points

    axis_pt = np.asarray(feat.axis.point, dtype=np.float64)
    axis_dir = np.asarray(feat.axis.direction, dtype=np.float64)
    n = float(np.linalg.norm(axis_dir))
    if n > 1e-12:
        axis_dir = axis_dir / n
    best = 0.0
    for c in feat.cone_face_indices:
        pts = collect_boundary_points(occ_faces[c])
        v = pts - axis_pt
        radial = np.linalg.norm(v - (v @ axis_dir)[:, None] * axis_dir, axis=1)
        if radial.size:
            best = max(best, float(radial.max()))
    return best


def _half_angle_deg(feat: Any, faces: Sequence[FaceGeom]) -> float | None:
    for c in feat.cone_face_indices:
        sa = getattr(faces[c], "semi_angle_rad", None)
        if sa is not None:
            return round(float(np.degrees(sa)), 3)
    return None


# ---------------------------------------------------------------------------
# Feature / result dataclasses (mirror counterbore_detection)
# ---------------------------------------------------------------------------
@dataclass
class CountersinkFeature:
    feature_id: int
    face_indices: set[int]
    area: float
    bore_radius_mm: float
    cone_flare_radius_mm: float
    half_angle_deg: float | None
    axis_point: list[float]
    axis_direction: list[float]
    cone_face_indices: list[int]
    cylinder_face_indices: list[int]
    kind: str = "countersink"
    toolpath_class: str = "countersink"

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def flare_ratio(self) -> float:
        return self.cone_flare_radius_mm / self.bore_radius_mm if self.bore_radius_mm else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(int(i) for i in self.face_indices),
            "n_faces": self.n_faces,
            "area": round(float(self.area), 3),
            "bore_radius_mm": round(float(self.bore_radius_mm), 6),
            "nominal_diameter": round(float(self.bore_radius_mm) * 2.0, 6),
            "cone_flare_radius_mm": round(float(self.cone_flare_radius_mm), 6),
            "flare_ratio": round(float(self.flare_ratio), 4),
            "half_angle_deg": self.half_angle_deg,
            "axis": {"point": self.axis_point, "direction": self.axis_direction},
            "cone_face_indices": sorted(self.cone_face_indices),
            "cylinder_face_indices": sorted(self.cylinder_face_indices),
            "is_countersink": True,
            "is_counterbore": False,
        }


@dataclass
class CountersinkDetectionResult:
    features: list[CountersinkFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int

    def summary(self) -> str:
        return (
            f"{len(self.features)} countersink(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


def detect_countersinks(
    hole_result: Any,
    faces: Sequence[FaceGeom],
    *,
    occ_faces: Sequence[Any] | None = None,
    candidate_faces: set[int] | frozenset[int] | None = None,
    config: CountersinkDetectionConfig | None = None,
) -> CountersinkDetectionResult:
    """Reclassify qualifying ``through_hole`` hole features as countersinks.

    Operates on ``hole_result.features`` (the POST-POCKET hole set): each feature
    already carries all its faces + ``is_countersink``/``is_counterbore`` flags.
    A feature that passes the geometry-only predicate becomes ONE countersink; its
    faces are not moved (the graph builder suppresses the absorbed hole feature).
    Pure-geometry: claims nothing on parts without a widening conical through-bore.
    """
    n_faces = len(faces)
    pool = (
        set(range(n_faces))
        if candidate_faces is None
        else {int(i) for i in candidate_faces}
    )
    cfg = config or CountersinkDetectionConfig()

    features: list[CountersinkFeature] = []
    claimed: set[int] = set()
    for feat in hole_result.features:
        flare = _cone_flare_radius(feat, faces, occ_faces)
        rec = _ECS(
            feature_id=feat.feature_id,
            kind=feat.kind,
            is_countersink=bool(feat.is_countersink),
            is_counterbore=bool(feat.is_counterbore),
            bore_radius=float(feat.radius or 0.0),
            cone_flare_radius=float(flare),
            face_indices={int(i) for i in feat.face_indices},
            cone_face_indices=list(feat.cone_face_indices),
            cylinder_face_indices=list(feat.cylinder_face_indices),
            axis_point=[float(x) for x in feat.axis.point],
            axis_direction=[float(x) for x in feat.axis.direction],
            half_angle_deg=_half_angle_deg(feat, faces),
        )
        if not _qualifies(rec, cfg):
            continue
        fids = rec.face_indices
        if not fids <= pool or (fids & claimed):
            continue
        area = float(sum(float(faces[i].area) for i in fids))
        features.append(CountersinkFeature(
            feature_id=len(features),
            face_indices=set(fids),
            area=area,
            bore_radius_mm=rec.bore_radius,
            cone_flare_radius_mm=rec.cone_flare_radius,
            half_angle_deg=rec.half_angle_deg,
            axis_point=rec.axis_point,
            axis_direction=rec.axis_direction,
            cone_face_indices=rec.cone_face_indices,
            cylinder_face_indices=rec.cylinder_face_indices,
        ))
        claimed |= fids

    return CountersinkDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=pool - claimed,
        n_faces=n_faces,
    )


def render_table(result: CountersinkDetectionResult) -> str:
    if not result.features:
        return "  (no countersinks)"
    lines = ["  #  faces                                bore_r  cone_r  ratio  half°"]
    for f in result.features:
        lines.append(
            f"  {f.feature_id:<2} {str(sorted(f.face_indices)):<36} "
            f"{f.bore_radius_mm:<7.3f} {f.cone_flare_radius_mm:<7.3f} "
            f"{f.flare_ratio:<6.3f} {f.half_angle_deg if f.half_angle_deg is not None else '-'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OCC-free self-test — exercises the predicate + wrapper directly (no cascade,
# no OCC). Builds fake HoleFeature-like + FaceGeom-like stubs.
# ---------------------------------------------------------------------------
def _run_selftest() -> bool:
    from types import SimpleNamespace as NS

    passed = True

    def check(name: str, cond: bool) -> None:
        nonlocal passed
        passed = passed and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    def rec(**kw) -> _ECS:
        base = dict(
            feature_id=0, kind="through_hole", is_countersink=True,
            is_counterbore=False, bore_radius=2.25, cone_flare_radius=4.7,
            face_indices={0, 1, 44, 45}, cone_face_indices=[0, 45],
            cylinder_face_indices=[1, 44], axis_point=[0, 0, 0],
            axis_direction=[0, 1, 0], half_angle_deg=45.0,
        )
        base.update(kw)
        return _ECS(**base)

    cfg = CountersinkDetectionConfig()

    # --- predicate-level cases (OCC-free, direct) ---
    check("Part3 group (through + cone flare>bore) -> countersink",
          _qualifies(rec(), cfg))
    check("counterbore look-alike (is_counterbore) -> rejected",
          not _qualifies(rec(is_counterbore=True), cfg))
    check("plain through-hole (no cone) -> rejected",
          not _qualifies(rec(is_countersink=False, cone_face_indices=[]), cfg))
    check("blind countersink (kind=blind_hole) -> rejected",
          not _qualifies(rec(kind="blind_hole"), cfg))
    check("drill-tip blind (kind=drilled_blind_hole) -> rejected",
          not _qualifies(rec(kind="drilled_blind_hole"), cfg))
    check("non-widening cone (flare<=bore) -> rejected",
          not _qualifies(rec(cone_flare_radius=2.25), cfg))

    # --- wrapper-level: the 10 Part3-like groups + look-alikes, OCC-free ---
    # FaceGeom stubs: cones carry radius=flare (OCC-free flare fallback), bore
    # cylinders carry the bore radius; semi_angle_rad on cones for half-angle.
    N = 60

    def fg(stype, area, radius=None, sa=None):
        return NS(surface_type=stype, area=area, radius=radius,
                  semi_angle_rad=sa, centroid=np.zeros(3), normal=np.zeros(3))

    faces = [fg("plane", 100.0) for _ in range(N)]

    GROUPS = [
        [0, 1, 44, 45], [2, 3, 42, 43], [4, 5, 40, 41], [6, 7, 38, 39],
        [8, 9, 36, 37], [10, 11, 34, 35], [12, 13, 32, 33], [14, 15, 30, 31],
        [16, 17, 28, 29], [18, 19, 26, 27],
    ]

    class _Axis:
        def __init__(self):
            self.point = [0.0, 0.0, 0.0]
            self.direction = [0.0, 1.0, 0.0]

    def hole(fid, grp, *, kind="through_hole", is_cs=True, is_cb=False,
             bore=2.25, flare=4.7):
        cones = [grp[0], grp[3]]
        cyls = [grp[1], grp[2]]
        for c in cones:
            faces[c] = fg("cone", 37.83, radius=flare, sa=np.radians(45.0))
        for c in cyls:
            faces[c] = fg("cylinder", 39.23, radius=bore)
        return NS(feature_id=fid, kind=kind, is_countersink=is_cs,
                  is_counterbore=is_cb, radius=bore, face_indices=list(grp),
                  cone_face_indices=cones, cylinder_face_indices=cyls, axis=_Axis())

    hole_features = [hole(i, g) for i, g in enumerate(GROUPS)]
    hole_result = NS(features=hole_features)
    res = detect_countersinks(hole_result, faces, occ_faces=None)
    got = sorted(sorted(f.face_indices) for f in res.features)
    check("10 Part3-like groups -> 10 countersinks (40 faces)",
          len(res.features) == 10 and len(res.claimed_faces) == 40
          and got == sorted(sorted(g) for g in GROUPS))
    check("every countersink kind/class == 'countersink'",
          all(f.kind == "countersink" and f.toolpath_class == "countersink"
              for f in res.features))
    check("flare_ratio ~2.089 on small groups",
          any(abs(f.flare_ratio - 2.089) < 0.01 for f in res.features))

    # counterbore look-alike (flat-bottom, is_counterbore) must be rejected
    cb = hole(0, [0, 1, 44, 45], is_cb=True)
    r_cb = detect_countersinks(NS(features=[cb]), faces, occ_faces=None)
    check("counterbore look-alike part -> 0 countersinks", len(r_cb.features) == 0)

    # plain through-hole (no cone) must be rejected
    ph = NS(feature_id=0, kind="through_hole", is_countersink=False,
            is_counterbore=False, radius=3.0, face_indices=[5, 6],
            cone_face_indices=[], cylinder_face_indices=[5, 6], axis=_Axis())
    r_ph = detect_countersinks(NS(features=[ph]), faces, occ_faces=None)
    check("plain through-hole part -> 0 countersinks", len(r_ph.features) == 0)

    # empty part -> no countersink
    r_empty = detect_countersinks(NS(features=[]), faces, occ_faces=None)
    check("empty part -> 0 countersinks", len(r_empty.features) == 0)

    print(f"\nself-test: {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Countersink detection pass")
    ap.add_argument("--selftest", action="store_true", help="run OCC-free self-test")
    ap.add_argument("step", nargs="?", default=None, help="STEP file to scan")
    ap.add_argument("--graph-npz", default=None)
    args = ap.parse_args(argv)

    if args.selftest or not args.step:
        return 0 if _run_selftest() else 1

    from pathlib import Path

    from brep.feature_params import analyze_step, load_step_faces, require_occ
    from cascade.hole_detection import HoleDetectionConfig, detect_holes
    from run_cascade import _load_edges

    require_occ()
    faces = analyze_step(args.step)
    occ = load_step_faces(args.step)
    ei, ea = _load_edges(Path(args.graph_npz) if args.graph_npz else None, Path(args.step))
    hl = detect_holes(faces, ei, ea, occ_faces=occ,
                      candidate_faces=set(range(len(faces))),
                      config=HoleDetectionConfig(max_hole_diameter_mm=150.0))
    res = detect_countersinks(hl, faces, occ_faces=occ)
    print(f"STEP: {args.step}")
    print(render_table(res))
    print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
