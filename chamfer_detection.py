"""Chamfer (oblique-bevel ring) recognizer.

A chamfer is an angled bevel that bridges two levels. On a real part it is not a
single planar facet but a *closed band* of oblique faces running around a step:
the straight runs are planes tilted ~45deg to the part axis, and the rounded
corners are cones coaxial with the part axis at the same oblique half-angle. So
the recognizer is a template match on that band as a whole:

  1. OBLIQUE-BEVEL face (relative to a part axis, so no absolute-mm magic):
       * a plane whose normal is 25..65deg off the axis, OR
       * a cone whose axis is parallel (<= ~10deg) to the part axis with a
         semi-angle in 25..65deg.
     The 25..65 band is the same bevel band the old recognizer used; it excludes
     ~0deg tangent blends (fillets) and ~90deg step walls / axis-aligned flats.

  2. GROUP the oblique faces into connected components by their mutual B-rep
     edges. Distinct steps land in distinct components (they are separated by the
     non-oblique walls/floors between them), so the axis-band separation is
     emergent from connectivity -- no Z constant.

  3. ACCEPT a component as a chamfer iff BOTH:
       a. it is a CLOSED LOOP -- every member has induced degree exactly 2 within
          the component (a bevel band is a ring; this also drops degenerate lone
          oblique planes), AND
       b. REACHABILITY GATE -- at least one member face is reachable (non-occluded)
          from a setup approach direction. This rejects buried internal edge-breaks
          (e.g. a bevel ring around occluded through-holes) while keeping genuine,
          machinable chamfers. The gate reuses the cascade's swept-tool corridor
          test (reachability.make_axis_reachability_probe).

The part axis is derived from the part's own geometry (the normal of the largest
planar face -- the envelope cap of a plate is perpendicular to its thickness
axis), NOT from the pocket pass's opening_axis, which can be a mis-detected
aggregate.

CAVEATS:
  * ``chamfer_size_mm`` is a coarse footprint proxy (sqrt of total band area),
    NOT the true chamfer leg length (edge lengths are not available here).
  * Two mirror chamfers on a symmetric two-sided part are BOTH real (one per
    setup); this recognizer correctly returns both. Selecting one per setup is a
    setup-scoped decision, out of scope here.
  * 96260B has no plane+cone oblique-bevel loop about its part axis (its real lip
    chamfer is a cone+torus band handled elsewhere), so this recognizer finds
    zero there -- the downstream partition is unchanged on those panels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

# Bevel band on the angle between an oblique face and the part axis (degrees). A
# 45deg chamfer on a square step puts its planes' normals at 45deg and its corner
# cones at a 45deg semi-angle; 30/60 chamfers stay inside 25..65. Below the band
# -> tangent blend (fillet); above -> step wall / axis-aligned flat.
CHAMFER_MIN_ANGLE_DEG = 25.0
CHAMFER_MAX_ANGLE_DEG = 65.0
# A cone counts as coaxial with the part axis when its axis is within this of
# parallel (bevel cones are surfaces of revolution about the approach axis).
CONE_AXIS_PARALLEL_DEG = 10.0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _angle_to_axis_deg(vec: Any, axis: np.ndarray) -> float | None:
    """Angle (deg) between ``vec`` and ``axis``, folded to [0, 90]."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    c = abs(float(np.dot(v / n, axis)))
    return math.degrees(math.acos(min(1.0, c)))


def derive_part_axis(faces: Sequence[Any]) -> np.ndarray | None:
    """Part principal axis = normal of the largest planar face.

    On a plate the biggest flat face is an envelope cap whose normal is the
    thickness (approach) axis; geometry-derived and independent of the pocket
    pass's opening_axis. Returns a unit vector, or None if there is no usable
    planar face.
    """
    best = None
    best_area = -1.0
    for f in faces:
        if getattr(f, "surface_type", None) != "plane":
            continue
        n = getattr(f, "normal", None)
        if n is None:
            continue
        if float(np.linalg.norm(np.asarray(n, dtype=np.float64))) < 1e-9:
            continue
        area = float(getattr(f, "area", 0.0) or 0.0)
        if area > best_area:
            best_area = area
            best = n
    if best is None:
        return None
    return _unit(np.asarray(best, dtype=np.float64))


def is_oblique_bevel_face(
    f: Any,
    axis: np.ndarray,
    *,
    min_angle_deg: float = CHAMFER_MIN_ANGLE_DEG,
    max_angle_deg: float = CHAMFER_MAX_ANGLE_DEG,
    cone_axis_parallel_deg: float = CONE_AXIS_PARALLEL_DEG,
) -> bool:
    """True if ``f`` is an oblique bevel face relative to ``axis`` (see module doc)."""
    st = getattr(f, "surface_type", None)
    if st == "plane":
        a = _angle_to_axis_deg(getattr(f, "normal", None), axis)
        return a is not None and min_angle_deg <= a <= max_angle_deg
    if st == "cone":
        cax = getattr(f, "axis", None)
        semi = getattr(f, "semi_angle_rad", None)
        if cax is None or semi is None:
            return False
        para = _angle_to_axis_deg(cax, axis)
        if para is None or para > cone_axis_parallel_deg:
            return False
        semi_deg = math.degrees(float(semi))
        return min_angle_deg <= semi_deg <= max_angle_deg
    return False


@dataclass
class ChamferFeature:
    """One recognized chamfer -- a closed oblique-bevel ring."""

    face_indices_: frozenset[int]
    bevel_angle_deg: float
    area_mm2: float
    n_planes: int
    n_cones: int
    reachable_dirs: list[str] = field(default_factory=list)

    @property
    def face_indices(self) -> frozenset[int]:
        return self.face_indices_

    @property
    def toolpath_class(self) -> str:
        return "chamfer"

    @property
    def chamfer_size_mm(self) -> float:
        # Coarse footprint proxy; NOT the true chamfer leg length (see module note).
        return round(math.sqrt(self.area_mm2), 3) if self.area_mm2 > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": min(self.face_indices_) if self.face_indices_ else -1,
            "chamfer_angle_deg": round(self.bevel_angle_deg, 2),
            "chamfer_size_mm": self.chamfer_size_mm,
            "area_mm2": round(self.area_mm2, 4),
            "n_planes": self.n_planes,
            "n_cones": self.n_cones,
            "reachable_dirs": list(self.reachable_dirs),
            "has_chamfer": True,
            "toolpath_class": "chamfer",
        }


@dataclass
class ChamferResult:
    features: list[ChamferFeature] = field(default_factory=list)
    reference_axis: tuple[float, float, float] | None = None
    remaining_faces: set[int] = field(default_factory=set)

    @property
    def claimed_faces(self) -> frozenset[int]:
        out: set[int] = set()
        for feat in self.features:
            out |= set(feat.face_indices_)
        return frozenset(out)


def _oblique_adjacency(
    oblique: set[int], edge_index: np.ndarray
) -> dict[int, set[int]]:
    """Undirected neighbour sets restricted to the oblique face set (any edge kind)."""
    adj: dict[int, set[int]] = {i: set() for i in oblique}
    ei = np.asarray(edge_index)
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i == j or i not in oblique or j not in oblique:
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _connected_components(adj: dict[int, set[int]]) -> list[set[int]]:
    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in adj:
        if start in seen:
            continue
        comp: set[int] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def detect_chamfers(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray | None = None,
    *,
    candidate_faces: Sequence[int] | None = None,
    exclude_faces: frozenset[int] = frozenset(),
    reference_axis: Sequence[float] | None = None,
    reachability_probe: Callable[[Sequence[int]], Sequence[str]] | None = None,
    reachability_probe_factory: Callable[[np.ndarray], Callable[[Sequence[int]], Sequence[str]]] | None = None,
    min_angle_deg: float = CHAMFER_MIN_ANGLE_DEG,
    max_angle_deg: float = CHAMFER_MAX_ANGLE_DEG,
) -> ChamferResult:
    """Recognize closed oblique-bevel rings (chamfers), gated by reachability.

    ``candidate_faces`` scopes the search to the unclaimed pool (falls back to all
    faces minus ``exclude_faces``). The reachability gate is load-bearing: a loop
    is claimed only if a probe reports at least one reachable direction. The probe
    is supplied ready (``reachability_probe``) or built lazily from the derived
    axis (``reachability_probe_factory``) so the OCC shape is loaded only when
    loop candidates actually exist. With neither probe, no loop can be confirmed
    reachable, so nothing is claimed (safe default).
    """
    by_index = {int(getattr(f, "index", i)): f for i, f in enumerate(faces)}

    if candidate_faces is not None:
        pool = {int(i) for i in candidate_faces} - set(exclude_faces)
    else:
        pool = set(by_index) - set(exclude_faces)

    axis = (
        _unit(np.asarray(reference_axis, dtype=np.float64))
        if reference_axis is not None
        else derive_part_axis(faces)
    )
    result = ChamferResult(
        reference_axis=None if axis is None else tuple(float(x) for x in axis),
        remaining_faces=set(pool),
    )
    if axis is None:
        return result

    # 1. oblique-bevel faces within the pool
    oblique = {
        i
        for i in pool
        if i in by_index
        and is_oblique_bevel_face(
            by_index[i], axis,
            min_angle_deg=min_angle_deg, max_angle_deg=max_angle_deg,
        )
    }
    if not oblique:
        return result

    # 2. group into connected components by mutual edges
    adj = _oblique_adjacency(oblique, edge_index)
    components = _connected_components(adj)

    # 3a. loop filter: every member has induced degree exactly 2
    loops = [
        comp
        for comp in components
        if len(comp) >= 3 and all(len(adj[i] & comp) == 2 for i in comp)
    ]
    if not loops:
        return result

    # 3b. reachability gate (lazy probe build once loops exist)
    probe = reachability_probe
    if probe is None and reachability_probe_factory is not None:
        probe = reachability_probe_factory(axis)
    if probe is None:
        # Cannot confirm reachability -> claim nothing (gate is load-bearing).
        return result

    for comp in loops:
        dirs = list(probe(sorted(comp)))
        if not dirs:
            continue  # fully occluded -> not a machinable chamfer (rejects band C)
        members = sorted(comp)
        planes = [by_index[i] for i in members if getattr(by_index[i], "surface_type", None) == "plane"]
        cones = [by_index[i] for i in members if getattr(by_index[i], "surface_type", None) == "cone"]
        angs: list[float] = []
        for f in planes:
            a = _angle_to_axis_deg(getattr(f, "normal", None), axis)
            if a is not None:
                angs.append(a)
        for f in cones:
            semi = getattr(f, "semi_angle_rad", None)
            if semi is not None:
                angs.append(math.degrees(float(semi)))
        area = float(sum(float(getattr(by_index[i], "area", 0.0) or 0.0) for i in members))
        result.features.append(
            ChamferFeature(
                face_indices_=frozenset(members),
                bevel_angle_deg=float(np.mean(angs)) if angs else 0.0,
                area_mm2=area,
                n_planes=len(planes),
                n_cones=len(cones),
                reachable_dirs=dirs,
            )
        )

    result.remaining_faces = set(pool) - set(result.claimed_faces)
    return result


# ---------------------------------------------------------------------------
# Self-tests (OCC-free): synthetic faces + hand-built edge graphs + fake probe.
# ---------------------------------------------------------------------------
def _selftest() -> int:
    from types import SimpleNamespace as NS

    Z = np.array([0.0, 0.0, 1.0])

    def plane(idx, normal, area=50.0):
        return NS(index=idx, surface_type="plane", normal=np.asarray(normal, float),
                  axis=None, radius=None, semi_angle_rad=None, area=area,
                  centroid=np.zeros(3))

    def cone(idx, axis=(0, 0, 1), semi_deg=45.0, area=12.0):
        return NS(index=idx, surface_type="cone", normal=np.zeros(3),
                  axis=np.asarray(axis, float), radius=0.0,
                  semi_angle_rad=math.radians(semi_deg), area=area,
                  centroid=np.zeros(3))

    def cap(idx, normal, area):  # large envelope cap -> derives the axis
        return NS(index=idx, surface_type="plane", normal=np.asarray(normal, float),
                  axis=None, radius=None, semi_angle_rad=None, area=area,
                  centroid=np.zeros(3))

    def wall(idx, normal):  # 90deg step wall (not oblique)
        return plane(idx, normal, area=200.0)

    def floor(idx, normal):  # 0deg floor (not oblique)
        return plane(idx, normal, area=300.0)

    def ei(edges):
        if not edges:
            return np.zeros((2, 0), dtype=int)
        a = [e[0] for e in edges] + [e[1] for e in edges]
        b = [e[1] for e in edges] + [e[0] for e in edges]
        return np.array([a, b], dtype=int)

    n45 = _unit(np.array([0.0, 1.0, -1.0]))   # plane normal 45deg to Z
    n45b = _unit(np.array([1.0, 0.0, -1.0]))
    n45c = _unit(np.array([0.0, -1.0, -1.0]))
    n45d = _unit(np.array([-1.0, 0.0, -1.0]))

    ok = True

    def check(name, cond):
        nonlocal ok
        status = "ok" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{status}] {name}")

    reach_all = lambda ids: ["+Z", "-Z"]
    reach_none = lambda ids: []

    # (a) reachable plane+cone closed 45deg ring -> one chamfer
    faces_a = [
        cap(0, (0, 0, 1), 4000.0), cap(1, (0, 0, -1), 4000.0),  # envelope caps
        plane(10, n45), cone(11), plane(12, n45b), cone(13),
        plane(14, n45c), cone(15), plane(16, n45d), cone(17),
    ]
    ring = [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 10)]
    r = detect_chamfers(faces_a, ei(ring), None,
                        candidate_faces=[10, 11, 12, 13, 14, 15, 16, 17],
                        reachability_probe=reach_all)
    check("(a) reachable plane+cone ring -> 1 chamfer of 8 faces",
          len(r.features) == 1 and len(r.features[0].face_indices_) == 8)
    check("(a) axis derived as +/-Z from largest cap",
          r.reference_axis is not None and abs(abs(r.reference_axis[2]) - 1.0) < 1e-6)

    # (b) same ring but fully occluded -> NOT claimed (guards band C)
    r = detect_chamfers(faces_a, ei(ring), None,
                        candidate_faces=[10, 11, 12, 13, 14, 15, 16, 17],
                        reachability_probe=reach_none)
    check("(b) occluded ring -> 0 chamfers", len(r.features) == 0)

    # (c) a 90deg wall and 0deg floor adjacent to the ring -> NOT pulled in
    faces_c = faces_a + [wall(20, (0, 1, 0)), floor(21, (0, 0, 1))]
    ring_c = ring + [(10, 20), (12, 21)]
    r = detect_chamfers(faces_c, ei(ring_c), None,
                        candidate_faces=[10, 11, 12, 13, 14, 15, 16, 17, 20, 21],
                        reachability_probe=reach_all)
    claimed = set(r.features[0].face_indices_) if r.features else set()
    check("(c) wall(20)/floor(21) excluded, ring intact",
          len(r.features) == 1 and claimed == {10, 11, 12, 13, 14, 15, 16, 17})

    # (d) a lone oblique plane -> NOT claimed (loop filter)
    faces_d = [cap(0, (0, 0, 1), 4000.0), cap(1, (0, 0, -1), 4000.0), plane(10, n45)]
    r = detect_chamfers(faces_d, ei([]), None, candidate_faces=[10],
                        reachability_probe=reach_all)
    check("(d) lone oblique plane -> 0 chamfers", len(r.features) == 0)

    # (e) no oblique-to-axis faces -> empty (guards 96260B)
    faces_e = [cap(0, (0, 0, 1), 4000.0), cap(1, (0, 0, -1), 4000.0),
               wall(10, (1, 0, 0)), wall(11, (0, 1, 0)), floor(12, (0, 0, 1))]
    ring_e = [(10, 11), (11, 12), (12, 10)]
    r = detect_chamfers(faces_e, ei(ring_e), None,
                        candidate_faces=[10, 11, 12], reachability_probe=reach_all)
    check("(e) no oblique faces -> 0 chamfers", len(r.features) == 0)

    # (f) loop candidates exist but no probe supplied -> claim nothing (gate)
    r = detect_chamfers(faces_a, ei(ring), None,
                        candidate_faces=[10, 11, 12, 13, 14, 15, 16, 17])
    check("(f) no reachability probe -> 0 chamfers (gate load-bearing)",
          len(r.features) == 0)

    print("self-test:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
