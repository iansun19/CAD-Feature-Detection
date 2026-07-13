"""
prismatic_profile_detection.py — swept 2.5D outer-contour profile bands.

A *prismatic profile* is the outer side-wall band of a 2.5D prismatic region:
a closed loop of planar walls joined by rounded corner fillets, swept along an
extrusion axis. Geometrically it is a **closed adjacency cycle of vertical faces
about a common axis, built from planar walls and ISOLATED convex corner-fillet
cylinders** (no two corner fillets touch). A fully rounded perimeter alternates
plane <-> cylinder; a perimeter with mixed sharp and rounded corners carries
some wall <-> wall junctions and is equally valid. On a split part (e.g. a
two-half mold block) the same contour appears as two coaxial rings whose walls
are pairwise coplanar; those are merged back into one profile.

Discriminator (all geometry-derived — no face lists, no part gate, no absolute
feature-size constant):

  1. axis  : discovered as a shared cylinder axis in the candidate pool (not the
             part opening axis — a profile's extrusion axis can differ).
  2. member: planar faces with normal PERPENDICULAR to axis (walls) and
             cylindrical faces with axis PARALLEL to axis (corner fillets).
             Cones are excluded — a tapered bevel is not a vertical wall.
  3. cycle : in the vertical-face adjacency subgraph, a connected component in
             which every member has degree exactly 2 (a single closed loop) and
             no internal edge joins two cylinders (corner fillets stay isolated).
             Walls may be adjacent (sharp corners).
  4. merge : rings that are coaxial and share >= 2 coplanar wall planes are the
             split halves of one profile -> merged.

This intentionally rejects (a) interior through-hole rings — those carry two
adjacent cylinders, so a corner fillet would not be isolated; (b) all-planar
cycles and (c) opposing-cylinder profiles (the 96260B hub-step kind, handled by
profile_detection.py), both filtered by the min-wall / min-corner-fillet counts.
On 96260B no qualifying wall+corner-fillet cycle exists, so this pass claims
nothing there and leaves the partition byte-identical.

Units are mm. numpy only.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brep.feature_params import FaceGeom
from cascade.hole_detection import FaceGraph, _unit

logger = logging.getLogger("prismatic_profile_detection")


@dataclass
class PrismaticProfileConfig:
    # cylinder axis parallel to profile axis
    axis_parallel_tol_deg: float = 5.0
    # plane normal perpendicular to profile axis
    normal_perp_tol_deg: float = 5.0
    # coplanar-wall merge (two rings are one split profile)
    coplanar_normal_tol_deg: float = 2.0
    coplanar_offset_tol_mm: float = 0.05
    coplanar_walls_to_merge: int = 2
    # a profile band needs at least this many walls and corner fillets
    min_walls: int = 2
    min_corner_fillets: int = 2
    units: str = "mm"


@dataclass
class PrismaticProfileFeature:
    feature_id: int
    face_indices: set[int]
    axis: tuple[float, float, float]
    n_rings: int
    n_walls: int
    n_corner_fillets: int
    axial_span_mm: float
    fillet_radius_mm: float | None = None
    kind: str = "prismatic_profile"

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def toolpath_class(self) -> str:
        return "profile"

    # display-compat with profile_detection.render_table (never exercised on
    # 96260B, present so a mixed profile list renders without AttributeError)
    @property
    def nominal_diameter_mm(self) -> float:
        return float(2.0 * self.fillet_radius_mm) if self.fillet_radius_mm else 0.0

    @property
    def radial_mm(self) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "profile_axis": [round(float(x), 4) for x in self.axis],
            "n_rings": self.n_rings,
            "n_walls": self.n_walls,
            "n_corner_fillets": self.n_corner_fillets,
            "axial_span_mm": round(float(self.axial_span_mm), 3),
        }
        if self.fillet_radius_mm is not None:
            d["fillet_radius_mm"] = round(float(self.fillet_radius_mm), 3)
        return d


@dataclass
class PrismaticProfileResult:
    features: list[PrismaticProfileFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    pool_size: int = 0
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} prismatic profile(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


def _is_wall_plane(f: FaceGeom, axis: np.ndarray, tol_deg: float) -> bool:
    if f.surface_type != "plane" or f.normal is None:
        return False
    dot = abs(float(np.dot(_unit(np.asarray(f.normal, float)), axis)))
    return dot < np.sin(np.radians(tol_deg))


def _is_corner_cylinder(f: FaceGeom, axis: np.ndarray, tol_deg: float) -> bool:
    if f.surface_type != "cylinder" or f.axis is None or f.radius is None:
        return False
    dot = abs(float(np.dot(_unit(np.asarray(f.axis, float)), axis)))
    return dot > np.cos(np.radians(tol_deg))


def _candidate_axes(
    pool: set[int],
    by_index: dict[int, FaceGeom],
    tol_deg: float,
) -> list[np.ndarray]:
    """Distinct (unsigned) cylinder-axis directions present in the pool."""
    cos_tol = np.cos(np.radians(tol_deg))
    axes: list[np.ndarray] = []
    for fid in sorted(pool):
        f = by_index[fid]
        if f.surface_type != "cylinder" or f.axis is None:
            continue
        a = _unit(np.asarray(f.axis, float))
        if not np.isfinite(a).all() or np.linalg.norm(a) < 0.5:
            continue
        if any(abs(float(np.dot(a, b))) > cos_tol for b in axes):
            continue
        axes.append(a)
    return axes


def _vertical_faces(
    pool: set[int],
    by_index: dict[int, FaceGeom],
    axis: np.ndarray,
    config: PrismaticProfileConfig,
) -> tuple[set[int], dict[int, str]]:
    """Return (vertical faces, role map) for a candidate axis."""
    vset: set[int] = set()
    role: dict[int, str] = {}
    for fid in pool:
        f = by_index[fid]
        if _is_wall_plane(f, axis, config.normal_perp_tol_deg):
            vset.add(fid)
            role[fid] = "wall"
        elif _is_corner_cylinder(f, axis, config.axis_parallel_tol_deg):
            vset.add(fid)
            role[fid] = "cylinder"
    return vset, role


def _alternating_cycles(
    vset: set[int],
    role: dict[int, str],
    graph: FaceGraph,
    config: PrismaticProfileConfig,
) -> list[set[int]]:
    """Connected components that are single closed cycles of vertical walls and
    isolated corner-fillet cylinders.

    The corner fillets must stay ISOLATED (no internal edge joins two
    cylinders) — this still rejects interior through-hole rings, which carry
    adjacent cylinders. Walls MAY be adjacent, so a prism perimeter with mixed
    sharp and rounded corners (some wall<->wall junctions) still qualifies. A
    fully rounded perimeter is the strict-alternation special case."""
    # degree within the vertical-face subgraph
    deg = {
        fid: sum(1 for nb in graph.neighbors.get(fid, ()) if nb in vset)
        for fid in vset
    }
    seen: set[int] = set()
    rings: list[set[int]] = []
    for start in sorted(vset):
        if start in seen:
            continue
        comp: set[int] = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in graph.neighbors.get(u, ()):
                if v in vset and v not in comp:
                    stack.append(v)
        seen |= comp

        if len(comp) < 4:
            continue
        if not all(deg[n] == 2 for n in comp):
            continue  # not a single simple cycle
        # corner fillets stay isolated: no internal edge joins two cylinders
        # (rejects interior through-hole rings). Adjacent walls (sharp corners)
        # are allowed.
        corner_isolated = True
        for u in comp:
            if role[u] != "cylinder":
                continue
            for v in graph.neighbors.get(u, ()):
                if v in comp and role[v] == "cylinder":
                    corner_isolated = False
                    break
            if not corner_isolated:
                break
        if not corner_isolated:
            continue
        n_walls = sum(1 for n in comp if role[n] == "wall")
        n_cyl = sum(1 for n in comp if role[n] == "cylinder")
        if n_walls < config.min_walls or n_cyl < config.min_corner_fillets:
            continue
        rings.append(comp)
    return rings


def _plane_of(f: FaceGeom) -> tuple[np.ndarray, float]:
    n = _unit(np.asarray(f.normal, float))
    offset = float(np.dot(n, np.asarray(f.centroid, float)))
    return n, offset


def _coplanar(
    fa: FaceGeom, fb: FaceGeom, config: PrismaticProfileConfig
) -> bool:
    na, oa = _plane_of(fa)
    nb, ob = _plane_of(fb)
    dot = float(np.dot(na, nb))
    if abs(dot) < np.cos(np.radians(config.coplanar_normal_tol_deg)):
        return False
    # align offset sign with normal orientation
    ob_aligned = ob if dot > 0 else -ob
    return abs(oa - ob_aligned) <= config.coplanar_offset_tol_mm


def _shared_coplanar_walls(
    ring_a: set[int],
    ring_b: set[int],
    role: dict[int, str],
    by_index: dict[int, FaceGeom],
    config: PrismaticProfileConfig,
) -> int:
    walls_a = [i for i in ring_a if role.get(i) == "wall"]
    walls_b = [i for i in ring_b if role.get(i) == "wall"]
    count = 0
    used_b: set[int] = set()
    for a in walls_a:
        for b in walls_b:
            if b in used_b:
                continue
            if _coplanar(by_index[a], by_index[b], config):
                used_b.add(b)
                count += 1
                break
    return count


def _merge_rings(
    rings: list[set[int]],
    role: dict[int, str],
    by_index: dict[int, FaceGeom],
    config: PrismaticProfileConfig,
) -> list[list[int]]:
    """Union-find over rings: merge coaxial rings sharing coplanar walls."""
    parent = list(range(len(rings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = _shared_coplanar_walls(
                rings[i], rings[j], role, by_index, config
            )
            if shared >= config.coplanar_walls_to_merge:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(rings)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def detect_prismatic_profiles(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    config: PrismaticProfileConfig | None = None,
) -> PrismaticProfileResult:
    """Claim closed alternating wall/corner-fillet profile bands."""
    config = config or PrismaticProfileConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        int(i) for i in candidate_faces if 0 <= int(i) < n_faces
    }
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    # 1. discover candidate extrusion axes, gather alternating rings per axis
    rings: list[set[int]] = []
    ring_axis: list[np.ndarray] = []
    ring_role: dict[int, str] = {}
    seen_rings: list[frozenset[int]] = []
    for axis in _candidate_axes(pool, by_index, config.axis_parallel_tol_deg):
        vset, role = _vertical_faces(pool, by_index, axis, config)
        for comp in _alternating_cycles(vset, role, graph, config):
            key = frozenset(comp)
            if key in seen_rings:
                continue
            seen_rings.append(key)
            rings.append(comp)
            ring_axis.append(axis)
            for fid in comp:
                ring_role[fid] = role[fid]

    features: list[PrismaticProfileFeature] = []
    claimed: set[int] = set()
    if rings:
        # 2. merge coaxial rings sharing coplanar walls into single profiles
        groups = _merge_rings(rings, ring_role, by_index, config)
        for feat_id, group in enumerate(groups):
            member_faces: set[int] = set()
            for ri in group:
                member_faces |= rings[ri]
            if member_faces & claimed:  # a face may not belong to two profiles
                member_faces -= claimed
                if not member_faces:
                    continue
            axis = ring_axis[group[0]]
            walls = [i for i in member_faces if ring_role.get(i) == "wall"]
            cyls = [i for i in member_faces if ring_role.get(i) == "cylinder"]
            axials = [
                float(np.dot(np.asarray(by_index[i].centroid, float), axis))
                for i in member_faces
            ]
            radii = [
                float(by_index[i].radius)
                for i in cyls
                if by_index[i].radius is not None
            ]
            feat = PrismaticProfileFeature(
                feature_id=feat_id,
                face_indices=set(member_faces),
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                n_rings=len(group),
                n_walls=len(walls),
                n_corner_fillets=len(cyls),
                axial_span_mm=(max(axials) - min(axials)) if axials else 0.0,
                fillet_radius_mm=min(radii) if radii else None,
            )
            features.append(feat)
            claimed |= member_faces
            logger.info(
                "prismatic profile %d: %d faces (%d walls, %d fillets, %d rings) "
                "axis=%s",
                feat_id, feat.n_faces, feat.n_walls, feat.n_corner_fillets,
                feat.n_rings, feat.axis,
            )

    return PrismaticProfileResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=pool - claimed,
        n_faces=n_faces,
        pool_size=len(pool),
        units=config.units,
    )


def render_table(result: PrismaticProfileResult) -> str:
    header = (
        f"{'id':>3}  {'#f':>3}  {'walls':>5}  {'fillets':>7}  "
        f"{'rings':>5}  {'ax_span':>8}  {'R_mm':>6}  faces"
    )
    lines = [header, "-" * len(header)]
    for feat in result.features:
        face_s = ",".join(str(i) for i in sorted(feat.face_indices))
        r = f"{feat.fillet_radius_mm:.2f}" if feat.fillet_radius_mm else "-"
        lines.append(
            f"{feat.feature_id:>3}  {feat.n_faces:>3}  {feat.n_walls:>5}  "
            f"{feat.n_corner_fillets:>7}  {feat.n_rings:>5}  "
            f"{feat.axial_span_mm:>8.1f}  {r:>6}  {face_s}"
        )
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


DEFAULT_STEP = "fixtures/step/fish_mold.stp"
DEFAULT_GRAPH_NPZ = "pipeline_out/fish_mold/graph.npz"


# ---------------------------------------------------------------------------
# OCC-free self-tests for the relaxed corner-isolated cycle rule
# ---------------------------------------------------------------------------
def _mk_face(index, stype, centroid, normal=(0, 0, 0), radius=None, axis=None):
    return FaceGeom(
        index=index,
        surface_type=stype,
        area=1.0,
        centroid=np.asarray(centroid, dtype=float),
        normal=np.asarray(normal, dtype=float),
        radius=radius,
        axis=None if axis is None else np.asarray(axis, dtype=float),
    )


def _mk_edges(pairs):
    """Undirected adjacency (convexity irrelevant to this pass -> all smooth)."""
    ei = np.array([[u for u, _v in pairs], [v for _u, v in pairs]], dtype=np.int64)
    ea = np.zeros((len(pairs), 4), dtype=np.float32)
    ea[:, 2] = 1.0  # smooth one-hot
    return ei, ea


def _selftest() -> int:
    """Exercise Predicate B (relaxed corner-isolated cycle) with synthetic
    perimeter loops about Z. numpy only, no OCC."""
    fails: list[str] = []
    W = lambda i, n: _mk_face(i, "plane", (i, i, 0), normal=n)          # noqa: E731
    C = lambda i: _mk_face(i, "cylinder", (i, i, 0), radius=3.0, axis=(0, 0, 1))  # noqa: E731

    def run(faces, pairs):
        ei, ea = _mk_edges(pairs)
        return detect_prismatic_profiles(faces, ei, ea)

    # Part1-like: 4 walls + 2 fillets, TWO sharp wall<->wall corners.
    # loop: wall+X - wall+Y - cyl - wall-X - cyl - wall-Y - (back to wall+X)
    p1 = [W(0, (1, 0, 0)), W(1, (0, 1, 0)), C(2), W(3, (-1, 0, 0)), C(4), W(5, (0, -1, 0))]
    r1 = run(p1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    if len(r1.features) != 1 or r1.claimed_faces != {0, 1, 2, 3, 4, 5}:
        fails.append(f"part1-like mixed loop not one 6-face profile: "
                     f"{[sorted(f.face_indices) for f in r1.features]}")

    # Part2-like: 4 walls + 4 fillets, fully alternating (strict special case).
    p2 = [W(0, (1, 0, 0)), C(1), W(2, (0, 1, 0)), C(3),
          W(4, (-1, 0, 0)), C(5), W(6, (0, -1, 0)), C(7)]
    r2 = run(p2, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)])
    if len(r2.features) != 1 or r2.claimed_faces != set(range(8)):
        fails.append(f"part2-like alternating loop not one 8-face profile: "
                     f"{[sorted(f.face_indices) for f in r2.features]}")

    # Look-alike REJECTED: interior through-hole ring = adjacent cylinders.
    r3 = run([C(0), C(1), C(2), C(3)], [(0, 1), (1, 2), (2, 3), (3, 0)])
    if r3.features:
        fails.append(f"adjacent-cylinder ring wrongly claimed: "
                     f"{[sorted(f.face_indices) for f in r3.features]}")

    # EMPTY on a part with none of the target geometry: all-planar loop, no fillet.
    r4 = run([W(0, (1, 0, 0)), W(1, (0, 1, 0)), W(2, (-1, 0, 0)), W(3, (0, -1, 0))],
             [(0, 1), (1, 2), (2, 3), (3, 0)])
    if r4.features:
        fails.append(f"all-planar loop wrongly claimed: "
                     f"{[sorted(f.face_indices) for f in r4.features]}")

    # Part4-like: the exterior boundary the hole pass fragments. A single closed
    # loop of walls interleaved with ISOLATED vertical cylinders of MIXED radii —
    # corner fillets (R3), semicircular scallop notches (R2.75) and a large
    # boundary arc (R9.5). Each cylinder is flanked by walls, so all corner
    # fillets stay isolated and the whole contour is ONE profile. Radius plays no
    # role: Predicate B keys on axis + isolation, not feature size. This is the
    # geometry that, when holes ran first, was split into through_hole (scallops/
    # arc) + flat + wall + contour.
    Cr = lambda i, r: _mk_face(i, "cylinder", (i, i, 0), radius=r, axis=(0, 0, 1))  # noqa: E731
    p5 = [W(0, (1, 0, 0)), Cr(1, 3.0), W(2, (0, 1, 0)), Cr(3, 2.75),
          W(4, (0, 1, 0)), Cr(5, 9.5), W(6, (-1, 0, 0)), Cr(7, 3.0),
          W(8, (0, -1, 0)), Cr(9, 2.75)]
    r5 = run(p5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                  (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
    if len(r5.features) != 1 or r5.claimed_faces != set(range(10)):
        fails.append(f"Part4-like scallop/arc perimeter not one 10-face profile: "
                     f"{[sorted(f.face_indices) for f in r5.features]}")

    # Idempotent: a part whose exterior is ALREADY exactly one clean profile
    # loop is claimed once, unchanged, with no extra faces (re-running the pass
    # on the same input yields the identical claim).
    r5b = run(p5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                   (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
    if r5b.claimed_faces != r5.claimed_faces or len(r5b.features) != 1:
        fails.append("Part4-like profile not idempotent on re-run")

    # Interior contour_surface REJECTED: a lone curved interior face (bspline
    # floor / sculpt patch) with no wall+fillet cycle must not be claimed.
    r6 = run([_mk_face(0, "bspline", (0, 0, 5), normal=(0, 0, 1)),
              _mk_face(1, "plane", (0, 0, 0), normal=(0, 0, 1))],
             [(0, 1)])
    if r6.features:
        fails.append(f"interior contour face wrongly claimed: "
                     f"{[sorted(f.face_indices) for f in r6.features]}")

    if fails:
        print("prismatic_profile Predicate-B selftest: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("prismatic_profile Predicate-B selftest: PASS "
          "(part1 sharp-corner + part2 alternating + Part4 scallop/arc perimeter "
          "accepted, idempotent; through-hole ring + all-planar loop + interior "
          "contour rejected)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prismatic profile-band detection")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ))
    ap.add_argument("--selftest", action="store_true",
                    help="run OCC-free relaxed-cycle unit checks and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    step_path = Path(args.step)
    if not step_path.is_file():
        print(f"STEP not found: {step_path}")
        return 2

    from brep.feature_params import analyze_step
    from run_cascade import _load_edges

    faces = analyze_step(step_path)
    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    result = detect_prismatic_profiles(faces, edge_index, edge_attr)
    print(f"STEP: {step_path}  faces: {len(faces)}")
    print(render_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
