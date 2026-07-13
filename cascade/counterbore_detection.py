"""
counterbore_detection.py — new cascade feature: the counterbored through-hole.

counterbore = a flat-bottomed cylindrical recess that enlarges the MOUTH of a
through-bore so a fastener sits flush/below the surface. Its geometric signature
is a coaxial stack:

    enlarged mouth cylinder (R_big)
        ↕  (optional coaxial fillet toruses on the shoulder)
    flat annular shoulder plane  (normal ∥ bore axis)   <-- the recess floor
        ↕
    smaller coaxial bore cylinder (R_small) continuing THROUGH the part

The predicate is SHOULDER-CENTRIC and uses derived/relative quantities only
(diameter ratio, shoulder flatness/normal-to-axis, coaxiality, through-ness) —
no face-id lists and no part gate. It is provably DISJOINT from conical features:
it only ever inspects cylinders, planes, and toruses, and REQUIRES a flat plane
shoulder normal to the axis. A countersink (conical enlargement) and a drill-tip
blind hole (conical cap) contain no such plane-between-two-coaxial-cylinders
structure, so neither can ever match — see the self-test's cone guard.

Validated selection (geometry-only, run over every face of each fixture):
    96260B_front : {93,94,95,285,286,280,298}      (ratio 1.252)
    96260B_rear  : {108,109,110,334,335,329,347}   (ratio 1.252)
    part2        : {1,2,19,20,21}                    (ratio 1.818)
    fish_mold    : {} (no qualifying structure; only cones)
    part1        : {} (no qualifying structure; only cones)

Runs AFTER inner_fillet and BEFORE pockets in the cascade so the whole coaxial
stack is claimed as ONE counterbore before the pocket pass can grab the enlarged
mouth as a filleted_blind_hole and the hole pass can grab the bore as a separate
through_hole. Claims nothing on parts without the structure, leaving the
downstream pool — and hence the partition — unchanged there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brep.feature_params import FaceGeom


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CounterboreDetectionConfig:
    axis_parallel_dot: float = 0.99   # |a·b| for parallel axes / normal∥axis
    coaxial_line_tol_mm: float = 1.0  # perp distance between coaxial axis lines
    ratio_min: float = 1.05           # R_big must exceed R_small by ≥5%
    ratio_max: float = 6.0            # sanity ceiling on the enlargement ratio
    shoulder_margin_mm: float = 1.0   # slack on shoulder-centroid radial distance
    # A far-end neighbour of the bore counts as a blind CAP (⇒ not through) only
    # when it is a plane normal to the axis whose area is small relative to the
    # shoulder (a genuine floor, not the opposite exterior face of the part).
    cap_area_ratio: float = 2.0


# ---------------------------------------------------------------------------
# Enriched face record (real OR fake) — the predicate operates on these only,
# so the self-test can build them directly without OCC.
# ---------------------------------------------------------------------------
@dataclass
class _EFace:
    index: int
    surface_type: str
    area: float
    centroid: np.ndarray
    normal: np.ndarray
    radius: float | None = None
    axis_dir: np.ndarray | None = None
    axis_pt: np.ndarray | None = None


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _line_perp_dist(direction: np.ndarray, line_pt: np.ndarray, q: np.ndarray) -> float:
    """Perpendicular distance of point ``q`` from the line (line_pt, direction)."""
    v = q - line_pt
    return float(np.linalg.norm(v - np.dot(v, direction) * direction))


def _adjacency(edge_index: np.ndarray) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {}
    ei = np.asarray(edge_index)
    for a, b in zip(ei[0], ei[1]):
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))
    return adj


def _axis_point(occ_face) -> np.ndarray | None:
    """Axis location for a cylinder/cone/torus face, else None (OCC)."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Torus,
    )

    surf = BRepAdaptor_Surface(occ_face, True)
    st = surf.GetType()
    loc = None
    if st == GeomAbs_Cylinder:
        loc = surf.Cylinder().Axis().Location()
    elif st == GeomAbs_Cone:
        loc = surf.Cone().Axis().Location()
    elif st == GeomAbs_Torus:
        loc = surf.Torus().Axis().Location()
    if loc is None:
        return None
    return np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)


def _enrich(faces: Sequence[FaceGeom], occ_faces: Sequence[Any] | None) -> list[_EFace]:
    out: list[_EFace] = []
    for i, g in enumerate(faces):
        axis_dir = None if g.axis is None else _unit(np.asarray(g.axis, dtype=float))
        axis_pt = None
        if occ_faces is not None and axis_dir is not None:
            axis_pt = _axis_point(occ_faces[i])
        out.append(_EFace(
            index=i,
            surface_type=g.surface_type,
            area=float(g.area),
            centroid=np.asarray(g.centroid, dtype=float),
            normal=_unit(np.asarray(g.normal, dtype=float)),
            radius=g.radius,
            axis_dir=axis_dir,
            axis_pt=axis_pt,
        ))
    return out


# ---------------------------------------------------------------------------
# Core geometry-only predicate (OCC-free once faces are enriched)
# ---------------------------------------------------------------------------
def _detect_core(
    ef: Sequence[_EFace],
    adj: dict[int, set[int]],
    cfg: CounterboreDetectionConfig,
) -> list[dict[str, Any]]:
    """Return one dict per counterbore: faces, shoulder, r_big, r_small, axis."""
    results: list[dict[str, Any]] = []
    planes = [f for f in ef if f.surface_type == "plane"]

    def parallel(a: np.ndarray, b: np.ndarray) -> bool:
        return abs(float(np.dot(a, b))) >= cfg.axis_parallel_dot

    def torus_between(pid: int, cid: int) -> int | None:
        """A single coaxial fillet torus hop from plane ``pid`` to cylinder ``cid``."""
        for nb in adj.get(pid, set()):
            if ef[nb].surface_type == "torus" and cid in adj.get(nb, set()):
                return nb
        return None

    for p in planes:
        axis = p.normal
        # attached coaxial cylinders: direct neighbour of the shoulder, or via
        # one coaxial fillet torus adjacent to the shoulder.
        attached: dict[int, int | None] = {}
        for c in ef:
            if c.surface_type != "cylinder" or c.axis_dir is None or c.radius is None:
                continue
            if not parallel(c.axis_dir, axis):
                continue
            line_pt = c.axis_pt if c.axis_pt is not None else c.centroid
            if _line_perp_dist(axis, line_pt, p.centroid) > c.radius + cfg.shoulder_margin_mm:
                continue
            ci = c.index
            if ci in adj.get(p.index, set()):
                attached[ci] = None
            else:
                tb = torus_between(p.index, ci)
                if tb is not None:
                    attached[ci] = tb
        if len(attached) < 2:
            continue

        radii = sorted({round(ef[ci].radius, 3) for ci in attached})  # type: ignore[arg-type]
        if len(radii) != 2:
            continue
        r_small, r_big = radii[0], radii[-1]
        ratio = r_big / r_small
        if not (cfg.ratio_min <= ratio <= cfg.ratio_max):
            continue

        big_ids = {ci for ci in attached if abs(ef[ci].radius - r_big) < 1e-3}  # type: ignore[operator]
        small_ids = {ci for ci in attached if abs(ef[ci].radius - r_small) < 1e-3}  # type: ignore[operator]

        def aproj(f: _EFace) -> float:
            return float(np.dot(f.centroid, axis))

        sh_pos = aproj(p)
        big_pos = float(np.mean([aproj(ef[ci]) for ci in big_ids]))
        small_pos = float(np.mean([aproj(ef[ci]) for ci in small_ids]))
        # shoulder must sit axially BETWEEN mouth (R_big) and bore (R_small):
        # the two radius groups lie on opposite axial sides of the shoulder.
        if (big_pos - sh_pos) * (small_pos - sh_pos) >= 0:
            continue

        # absorb coaxial fillet toruses on the SHOULDER (mouth↔shoulder blend):
        # coaxial toruses directly adjacent to the shoulder plane. Excludes the
        # mouth-rim fillets (mouth↔top face), a separate edge-break.
        toruses: set[int] = set()
        for ti in adj.get(p.index, set()):
            t = ef[ti]
            if t.surface_type != "torus" or t.axis_dir is None:
                continue
            if not parallel(t.axis_dir, axis):
                continue
            line_pt = t.axis_pt if t.axis_pt is not None else t.centroid
            if _line_perp_dist(axis, line_pt, p.centroid) > r_big + cfg.shoulder_margin_mm:
                continue
            toruses.add(ti)

        feat_ids = {p.index} | set(attached) | toruses

        # through-ness: the bore group opens at its FAR end to a face OUTSIDE the
        # feature, and is NOT capped by a small floor plane normal to the axis.
        bore_ext: set[int] = set()
        for sid in small_ids:
            bore_ext |= (adj.get(sid, set()) - feat_ids)
        capped = False
        for nb in bore_ext:
            f = ef[nb]
            if f.surface_type == "plane" and parallel(f.normal, axis) \
                    and f.area < p.area * cfg.cap_area_ratio:
                capped = True
                break
        if not bore_ext or capped:
            continue

        # axis POINT: take a coaxial cylinder's axis line point (bore group
        # preferred), falling back to the shoulder centroid when OCC axis
        # points are absent (self-test path). Emitted as a {point, direction}
        # dict to match every other feature's axis shape (the planner reads
        # axis.get("point")/axis.get("direction")); direction stays the
        # shoulder normal (parallel to the bore axis).
        axis_pt = None
        for ci in sorted(small_ids) + sorted(big_ids):
            if ef[ci].axis_pt is not None:
                axis_pt = ef[ci].axis_pt
                break
        if axis_pt is None:
            axis_pt = p.centroid

        results.append({
            "faces": set(feat_ids),
            "shoulder": p.index,
            "r_big": r_big,
            "r_small": r_small,
            "ratio": round(ratio, 4),
            "axis": {
                "point": [round(float(x), 6) for x in axis_pt],
                "direction": [round(float(x), 6) for x in axis],
            },
            "area": float(sum(ef[i].area for i in feat_ids)),
        })
    return results


# ---------------------------------------------------------------------------
# Feature / result dataclasses (mirror inner_fillet_detection)
# ---------------------------------------------------------------------------
@dataclass
class CounterboreFeature:
    feature_id: int
    face_indices: set[int]
    area: float
    shoulder_face_id: int
    counterbore_radius_mm: float   # R_big (enlarged mouth)
    bore_radius_mm: float          # R_small (through bore)
    axis: dict[str, list[float]]   # {"point": [...], "direction": [...]}
    kind: str = "counterbore"
    toolpath_class: str = "counterbore"

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def enlargement_ratio(self) -> float:
        return self.counterbore_radius_mm / self.bore_radius_mm if self.bore_radius_mm else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "area": round(float(self.area), 3),
            "shoulder_face_id": self.shoulder_face_id,
            "counterbore_radius_mm": round(float(self.counterbore_radius_mm), 6),
            "bore_radius_mm": round(float(self.bore_radius_mm), 6),
            "enlargement_ratio": round(float(self.enlargement_ratio), 4),
            "axis": self.axis,
        }


@dataclass
class CounterboreDetectionResult:
    features: list[CounterboreFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int

    def summary(self) -> str:
        return (
            f"{len(self.features)} counterbore(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


def detect_counterbores(
    step_path: str | Path,
    faces: Sequence[FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    occ_faces: Sequence[Any] | None = None,
    candidate_faces: set[int] | frozenset[int] | None = None,
    config: CounterboreDetectionConfig | None = None,
) -> CounterboreDetectionResult:
    """Cascade pass: geometry-only counterbore detection over a face pool.

    Each qualifying coaxial stack (enlarged mouth + flat shoulder + through bore
    + shoulder fillets) becomes ONE counterbore instance and is removed from the
    pool. Pure-geometry: claims nothing on parts without the structure.

    ``occ_faces`` (from ``feature_params.load_step_faces``) is required to read
    exact cylinder/torus axis locations for the coaxiality test; when omitted the
    face centroid is used as a fallback axis point (looser, self-test path).
    """
    n_faces = len(faces)
    pool = (
        set(range(n_faces))
        if candidate_faces is None
        else {int(i) for i in candidate_faces}
    )
    cfg = config or CounterboreDetectionConfig()

    ef = _enrich(faces, occ_faces)
    adj = _adjacency(edge_index)
    raw = _detect_core(ef, adj, cfg)

    features: list[CounterboreFeature] = []
    claimed: set[int] = set()
    for r in raw:
        fids = r["faces"]
        # only claim when the WHOLE stack is inside the candidate pool (it runs
        # early, so this holds; the guard prevents partial claims otherwise).
        if not fids <= pool:
            continue
        if fids & claimed:
            continue
        features.append(CounterboreFeature(
            feature_id=len(features),
            face_indices=set(fids),
            area=float(r["area"]),
            shoulder_face_id=int(r["shoulder"]),
            counterbore_radius_mm=float(r["r_big"]),
            bore_radius_mm=float(r["r_small"]),
            axis=dict(r["axis"]),
        ))
        claimed |= fids

    return CounterboreDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=pool - claimed,
        n_faces=n_faces,
    )


def reconcile_counterbores_with_holes(
    cb_result: CounterboreDetectionResult,
    hole_result: Any,
) -> CounterboreDetectionResult:
    """Validate geometric counterbores against the hole pass (non-mutating).

    The hole pass splits a counterbored through-hole into a filleted_blind_hole
    (enlarged mouth + flat shoulder + shoulder fillets) and a separate
    through_hole (the bore). A geometrically-detected counterbore is KEPT only
    when its face set is EXACTLY the union of a subset of hole features — i.e.
    the hole pass genuinely produced this split. Neither ``hole_result`` nor its
    ``claimed_faces`` are modified: the actual suppression of the absorbed hole
    features happens in ``build_cascade_feature_graph`` when a counterbore result
    is supplied, keeping every other graph-builder caller backward-compatible.

    A geometric counterbore NOT cleanly covered by hole features is dropped
    (cascade labeling left unchanged) rather than force-merged, to avoid
    corrupting the partition. Returns a result carrying only the validated
    counterbores.
    """
    import logging

    log = logging.getLogger("counterbore_detection")
    kept: list[CounterboreFeature] = []
    claimed: set[int] = set()
    for cb in cb_result.features:
        cbset = set(cb.face_indices)
        covering = [
            h for h in hole_result.features
            if set(h.face_indices) <= cbset
        ]
        union: set[int] = set()
        for h in covering:
            union |= set(h.face_indices)
        if covering and union == cbset:
            cb.feature_id = len(kept)
            kept.append(cb)
            claimed |= cbset
        else:
            log.warning(
                "counterbore at shoulder %d (faces %s) not cleanly covered by "
                "hole features (got %s); leaving cascade labeling unchanged",
                cb.shoulder_face_id, sorted(cbset), sorted(union),
            )
    return CounterboreDetectionResult(
        features=kept,
        claimed_faces=claimed,
        remaining_faces=set(),
        n_faces=cb_result.n_faces,
    )


def render_table(result: CounterboreDetectionResult) -> str:
    if not result.features:
        return "  (no counterbores)"
    lines = ["  #  faces                                shoulder  R_big   R_small  ratio"]
    for f in result.features:
        lines.append(
            f"  {f.feature_id:<2} {str(sorted(f.face_indices)):<36} "
            f"{f.shoulder_face_id:<8} {f.counterbore_radius_mm:<7.3f} "
            f"{f.bore_radius_mm:<8.3f} {f.enlargement_ratio:.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OCC-free self-test
# ---------------------------------------------------------------------------
def _face(index, stype, centroid, normal, *, radius=None, axis_dir=None,
          axis_pt=None, area=1.0) -> _EFace:
    return _EFace(
        index=index, surface_type=stype, area=area,
        centroid=np.asarray(centroid, float), normal=_unit(np.asarray(normal, float)),
        radius=radius,
        axis_dir=None if axis_dir is None else _unit(np.asarray(axis_dir, float)),
        axis_pt=None if axis_pt is None else np.asarray(axis_pt, float),
    )


def _adj_from_pairs(pairs, n) -> dict[int, set[int]]:
    src, dst = [], []
    for u, v in pairs:
        src += [u, v]
        dst += [v, u]
    return _adjacency(np.array([src, dst], dtype=np.int64))


def _run_selftest() -> bool:
    z = [0.0, 0.0, 1.0]
    cfg = CounterboreDetectionConfig()
    passed = True

    def check(name: str, cond: bool) -> None:
        nonlocal passed
        passed = passed and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # ---- Case 1: canonical counterbore (front/rear geometry, WITH fillets) ----
    # mouth cyls (R_big=50.876) above shoulder plane, bore cyls (R_small=40.64)
    # below, two shoulder fillet toruses, bore opens to a large exterior plane.
    axis_pt = [0.0, 0.0, 0.0]
    ef = [
        _face(0, "plane", [0, 0, -29.5], z, area=2701.0),                 # shoulder
        _face(1, "torus", [30, 0, -28.7], [0, 0, 0], axis_dir=z, axis_pt=[0, 0, -28.7], area=190.0),
        _face(2, "torus", [-30, 0, -28.7], [0, 0, 0], axis_dir=z, axis_pt=[0, 0, -28.7], area=190.0),
        _face(3, "cylinder", [-20, 25, -27.8], [1, 0, 0], radius=50.876, axis_dir=z, axis_pt=axis_pt, area=296.0),
        _face(4, "cylinder", [20, -25, -27.8], [1, 0, 0], radius=50.876, axis_dir=z, axis_pt=axis_pt, area=296.0),
        _face(5, "cylinder", [-25, 0, -42.2], [1, 0, 0], radius=40.640, axis_dir=z, axis_pt=axis_pt, area=3236.0),
        _face(6, "cylinder", [25, 0, -42.2], [1, 0, 0], radius=40.640, axis_dir=z, axis_pt=axis_pt, area=3236.0),
        _face(7, "plane", [0, 0, -70.0], z, area=7875.0),                 # far exterior face
        _face(8, "plane", [0, 0, 0.0], z, area=7875.0),                   # top / mouth exterior
    ]
    adj = _adj_from_pairs(
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4),   # shoulder↔fillets↔mouth
         (0, 5), (0, 6),                                    # shoulder↔bore
         (3, 8), (4, 8),                                    # mouth↔top face
         (5, 7), (6, 7)],                                   # bore↔far exterior (through)
        9,
    )
    r = _detect_core(ef, adj, cfg)
    ok1 = (len(r) == 1 and r[0]["faces"] == {0, 1, 2, 3, 4, 5, 6}
           and r[0]["shoulder"] == 0)
    check("counterbore w/ fillets → 1 feature, 7 faces (2 fillets absorbed)", ok1)

    # ---- Case 2: counterbore WITHOUT fillets (part2 geometry) ----
    ef2 = [
        _face(0, "cylinder", [10, 5, 0], [0, -1, 0], radius=5.0, axis_dir=[1, 0, 0], axis_pt=[0, 0, 0], area=86.0),
        _face(1, "cylinder", [10, -5, 0], [0, 1, 0], radius=5.0, axis_dir=[1, 0, 0], axis_pt=[0, 0, 0], area=86.0),
        _face(2, "plane", [7, 0, 0], [1, 0, 0], area=54.0),               # shoulder
        _face(3, "cylinder", [-3, 2.75, 0], [0, -1, 0], radius=2.75, axis_dir=[1, 0, 0], axis_pt=[0, 0, 0], area=171.0),
        _face(4, "cylinder", [-3, -2.75, 0], [0, 1, 0], radius=2.75, axis_dir=[1, 0, 0], axis_pt=[0, 0, 0], area=171.0),
        _face(5, "plane", [-20, 0, 0], [1, 0, 0], area=4467.0),           # far exterior
        _face(6, "plane", [20, 0, 0], [1, 0, 0], area=4467.0),            # top / mouth exterior
    ]
    adj2 = _adj_from_pairs(
        [(2, 0), (2, 1), (2, 3), (2, 4), (0, 6), (1, 6), (3, 5), (4, 5)], 7,
    )
    r2 = _detect_core(ef2, adj2, cfg)
    ok2 = len(r2) == 1 and r2[0]["faces"] == {0, 1, 2, 3, 4} and r2[0]["shoulder"] == 2
    check("counterbore w/o fillets (part2) → 1 feature, 5 faces", ok2)

    # ---- Case 3: countersink (conical enlargement) must NOT be a counterbore ----
    # cone replaces the mouth cylinder; there is NO flat plane shoulder between two
    # coaxial cylinders. Guards fish_mold countersinks + part1 drill-tip cones.
    ef3 = [
        _face(0, "cone", [10, 0, 5], [-1, 0, 0], radius=6.0, axis_dir=z, axis_pt=[0, 0, 0], area=200.0),
        _face(1, "cylinder", [-3, 0, -5], [1, 0, 0], radius=3.0, axis_dir=z, axis_pt=[0, 0, 0], area=100.0),
        _face(2, "plane", [0, 0, 10], z, area=500.0),   # opening plane the cone flares into
        _face(3, "plane", [0, 0, -20], z, area=5000.0),  # far exterior
    ]
    adj3 = _adj_from_pairs([(0, 1), (0, 2), (1, 3)], 4)
    r3 = _detect_core(ef3, adj3, cfg)
    check("countersink (cone enlargement) → NOT a counterbore", len(r3) == 0)

    # ---- Case 4: plain through-hole (single bore, no enlargement) stays through ----
    ef4 = [
        _face(0, "cylinder", [3, 0, 0], [-1, 0, 0], radius=3.0, axis_dir=z, axis_pt=[0, 0, 0], area=200.0),
        _face(1, "plane", [0, 0, 10], z, area=5000.0),
        _face(2, "plane", [0, 0, -10], z, area=5000.0),
    ]
    adj4 = _adj_from_pairs([(0, 1), (0, 2)], 3)
    r4 = _detect_core(ef4, adj4, cfg)
    check("plain through-hole (single radius) → NOT a counterbore", len(r4) == 0)

    # ---- Case 5: blind counterbore-shaped recess (bore capped by small floor) ----
    # two coaxial radii + flat shoulder, but the smaller cyl bottoms on a SMALL
    # floor plane (not through) → rejected by the through-ness clause.
    ef5 = [
        _face(0, "plane", [0, 0, -5], z, area=200.0),                    # shoulder
        _face(1, "cylinder", [8, 0, -2], [-1, 0, 0], radius=8.0, axis_dir=z, axis_pt=[0, 0, 0], area=100.0),
        _face(2, "cylinder", [4, 0, -8], [-1, 0, 0], radius=4.0, axis_dir=z, axis_pt=[0, 0, 0], area=100.0),
        _face(3, "plane", [0, 0, -12], z, area=50.0),                    # small blind floor
        _face(4, "plane", [0, 0, 0], z, area=5000.0),                    # top exterior
    ]
    adj5 = _adj_from_pairs([(0, 1), (0, 2), (1, 4), (2, 3)], 5)
    r5 = _detect_core(ef5, adj5, cfg)
    check("blind recess (bore capped by small floor) → NOT a counterbore", len(r5) == 0)

    # ---- Case 6: empty geometry → no counterbore ----
    r6 = _detect_core([], {}, cfg)
    check("empty part → no counterbore", len(r6) == 0)

    # ---- Case 7: full detect_counterbores wrapper builds the feature record ----
    from brep.feature_params import FaceGeom  # noqa

    class _FG:
        def __init__(self, e):
            self.surface_type = e.surface_type
            self.area = e.area
            self.centroid = e.centroid
            self.normal = e.normal
            self.radius = e.radius
            self.axis = e.axis_dir

    fgs = [_FG(e) for e in ef]
    # rebuild edge_index from adj for the wrapper
    src, dst = [], []
    for a, nbs in adj.items():
        for b in nbs:
            src.append(a)
            dst.append(b)
    ei = np.array([src, dst], dtype=np.int64)
    # supply axis points by faking occ via a monkeypatched _enrich path: pass
    # occ_faces=None so centroid fallback is used — but our test axis_pt matters
    # for coaxiality, so instead call _detect_core-equivalent through the wrapper
    # with a pre-enriched shortcut is not exposed; validate feature dataclass here.
    res = detect_counterbores("<selftest>", fgs, ei, None, occ_faces=None)
    feat_ok = (len(res.features) == 1 and res.features[0].n_faces == 7
               and res.features[0].kind == "counterbore"
               and res.features[0].toolpath_class == "counterbore"
               and abs(res.features[0].enlargement_ratio - 1.252) < 0.01)
    check("detect_counterbores wrapper → 1 counterbore feature (ratio 1.252)", feat_ok)

    print(f"\nself-test: {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Counterbore detection pass")
    ap.add_argument("--selftest", action="store_true", help="run OCC-free self-test")
    ap.add_argument("step", nargs="?", default=None, help="STEP file to scan")
    ap.add_argument("--graph-npz", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.selftest or not args.step:
        return 0 if _run_selftest() else 1

    from brep.feature_params import analyze_step, load_step_faces, require_occ
    from run_cascade import _load_edges

    require_occ()
    faces = analyze_step(args.step)
    occ = load_step_faces(args.step)
    ei, ea = _load_edges(args.graph_npz, Path(args.step))
    res = detect_counterbores(args.step, faces, ei, ea, occ_faces=occ)
    print(f"STEP: {args.step}")
    print(render_table(res))
    print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
