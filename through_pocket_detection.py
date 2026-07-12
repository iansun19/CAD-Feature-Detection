"""
through_pocket_detection.py — closed side-wall cavities that cut through the part.

A *through pocket* is a closed-profile cavity (side walls forming a ring) that is
open on BOTH of the part's opposite faces — like a through-hole, but with a
non-circular, multi-wall profile rather than a single round drilled bore. The
canonical example is an obround/stadium slot milled straight through a plate:
two straight side walls joined by rounded ends, opening onto both plate faces
with no floor.

Discriminator (all geometry-derived — no face-ID lists, no part gate, no
absolute-mm constant; every threshold is an angle band, a convexity label, or a
ratio to the ring's own geometry):

  1. caps   : ONE shared pair of planar faces (C1, C2) with collinear normals,
              offset along that normal (the two opposite open faces of the part).
              EVERY ring member is joined to BOTH caps by a CONVEX edge -> the
              cavity emerges onto both faces (open both ends), and the ring's
              axial centre lies between the caps. A blind pocket fails this: its
              walls meet an interior FLOOR at a CONCAVE edge, so they are not
              convex-adjacent to a second opposite cap.
  2. loop   : in the adjacency subgraph induced on the shared-cap ring, a
              connected component in which every member has degree exactly 2 — a
              single closed wall loop (the closed profile).
  3. profile: the loop is NOT a plain round bore. It carries either >= 1 planar
              side wall OR >= 2 distinct cylinder axis-lines. A through-hole
              (one coaxial cylinder pair split into arcs) has zero planar walls
              and a single axis-line, so it fails here and stays a through-hole.
  4. cavity : every member's outward (from-solid) surface normal points TOWARD
              the loop centroid -> the loop encloses a void that was cut into the
              part. The part's outer perimeter (a profile/contour ring) has its
              walls facing OUTWARD and is rejected here, keeping its existing
              profile/flat labels untouched.

Conditions 1-3 alone also match the part's outer contour ring; condition 4 is
the load-bearing separator (cavity vs. perimeter). On parts with no such cavity
(96260B front/rear, fish_mold, part1) the qualifying set is empty, so this pass
claims nothing and leaves the partition byte-identical.

Units are mm. numpy only.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hole_detection import FaceGraph, _unit

logger = logging.getLogger("through_pocket_detection")


@dataclass
class ThroughPocketConfig:
    # cap-pair normals collinear (parallel or anti-parallel); the two open faces
    # are perpendicular to the through-axis so their normals share its line.
    cap_normal_collinear_tol_deg: float = 3.0
    # the cap centroids must be offset ALONG that shared normal (distinct planes
    # facing opposite ways), not merely parallel co-planar faces.
    cap_offset_collinear_tol_deg: float = 5.0
    # a loop face is "inward" when its outward normal points toward the loop
    # centroid; require a strictly positive projection for EVERY member.
    inward_cos_min: float = 1.0e-3
    # two cylinder walls share an axis-line when their axis points sit within this
    # fraction of the mean wall radius of each other (ratio to feature geometry,
    # not an absolute mm) — used only for the planar-wall-free profile branch.
    axis_line_merge_radius_frac: float = 0.5
    units: str = "mm"


@dataclass
class ThroughPocketFeature:
    feature_id: int
    face_indices: set[int]
    axis: tuple[float, float, float]
    cap_face_ids: tuple[int, int]
    n_walls: int
    n_cyl: int
    n_distinct_axis_lines: int
    kind: str = "through_pocket"

    # --- pocket-feature duck-type: this feature is folded into the pocket pass'
    # feature list (see run_cascade), which several consumers iterate expecting a
    # PocketFeature (the discrete-instance scorer, the reporting table, the graph
    # builder). These fields carry the "not a sculpted/blind pocket" answer so
    # those consumers read sane values. A through pocket has no floor, no sculpted
    # seeds, and no analytic depth/fillet template. ---
    subtype: str = "through_pocket"
    depth_below_top_mm: float | None = None
    fillet_radius_mm: float | None = None
    surface_3d: bool = False
    wall_diameters: dict[str, int] = field(default_factory=dict)
    floor_count: int = 0
    sphere_count: int = 0
    step_plane_count: int = 0
    template_match: bool = True
    released_faces: list[int] = field(default_factory=list)
    centroid_uv: tuple[float, float] = (0.0, 0.0)

    @property
    def wall_count(self) -> int:
        return self.n_walls

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def toolpath_class(self) -> str:
        return "through_pocket"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "subtype": self.subtype,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "through_axis": [round(float(x), 4) for x in self.axis],
            "cap_face_ids": [int(self.cap_face_ids[0]), int(self.cap_face_ids[1])],
            "n_walls": self.n_walls,
            "n_corner_cyl": self.n_cyl,
            "n_distinct_axis_lines": self.n_distinct_axis_lines,
        }


@dataclass
class ThroughPocketResult:
    features: list[ThroughPocketFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    pool_size: int = 0
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} through pocket(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


def _planar_faces(by_index: dict[int, Any]) -> list[int]:
    return [
        i for i, f in by_index.items()
        if f.surface_type == "plane" and f.normal is not None
    ]


def _cap_pairs(
    planes: Sequence[int],
    by_index: dict[int, Any],
    config: ThroughPocketConfig,
) -> list[tuple[int, int, np.ndarray]]:
    """Planar face pairs whose normals are collinear AND whose centroids are
    offset along that normal (opposite open ends). Returns (c1, c2, axis)."""
    collinear = np.cos(np.radians(config.cap_normal_collinear_tol_deg))
    offset_collinear = np.cos(np.radians(config.cap_offset_collinear_tol_deg))
    out: list[tuple[int, int, np.ndarray]] = []
    for a, b in combinations(sorted(planes), 2):
        na = _unit(np.asarray(by_index[a].normal, float))
        nb = _unit(np.asarray(by_index[b].normal, float))
        if abs(float(np.dot(na, nb))) < collinear:
            continue
        d = np.asarray(by_index[b].centroid, float) - np.asarray(by_index[a].centroid, float)
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            continue
        if abs(float(np.dot(d / dn, na))) < offset_collinear:
            continue  # parallel but co-planar / laterally shifted, not opposite ends
        out.append((a, b, na))
    return out


def _shared_cap_ring(
    c1: int,
    c2: int,
    pool: set[int],
    graph: FaceGraph,
) -> set[int]:
    """Pool faces joined to BOTH caps by a convex edge (open on both ends)."""
    return {
        f for f in pool
        if f not in (c1, c2)
        and graph.edge_kind(f, c1) == "convex"
        and graph.edge_kind(f, c2) == "convex"
    }


def _single_cycles(members: set[int], graph: FaceGraph) -> list[set[int]]:
    """Connected components of `members` (edges within `members`) that are each a
    single simple cycle: every node has degree exactly 2 and the component is
    connected."""
    seen: set[int] = set()
    cycles: list[set[int]] = []
    for start in sorted(members):
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
                if v in members and v not in comp:
                    stack.append(v)
        seen |= comp
        if len(comp) < 3:
            continue
        if all(
            sum(1 for v in graph.neighbors.get(u, ()) if v in comp) == 2
            for u in comp
        ):
            cycles.append(comp)
    return cycles


def _distinct_axis_lines(
    cyls: Sequence[int],
    by_index: dict[int, Any],
    axis: np.ndarray,
    config: ThroughPocketConfig,
) -> int:
    """Count distinct cylinder axis-lines among `cyls`. A concave wall's outward
    normal points toward its own axis, so the axis point ~ centroid + r*normal;
    projected perpendicular to the through-axis, coincident points are one line.
    Merge tolerance is a fraction of the mean wall radius (feature-relative)."""
    pts: list[np.ndarray] = []
    radii: list[float] = []
    for i in cyls:
        f = by_index[i]
        if f.radius is None or f.normal is None:
            continue
        n = _unit(np.asarray(f.normal, float))
        p = np.asarray(f.centroid, float) + float(f.radius) * n
        p = p - float(np.dot(p, axis)) * axis  # drop the through-axis component
        pts.append(p)
        radii.append(float(f.radius))
    if not pts:
        return 0
    tol = config.axis_line_merge_radius_frac * (float(np.mean(radii)) if radii else 1.0)
    lines: list[np.ndarray] = []
    for p in pts:
        if any(float(np.linalg.norm(p - q)) <= tol for q in lines):
            continue
        lines.append(p)
    return len(lines)


def detect_through_pockets(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    config: ThroughPocketConfig | None = None,
) -> ThroughPocketResult:
    """Claim closed side-wall cavities open on both faces (through pockets)."""
    config = config or ThroughPocketConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        int(i) for i in candidate_faces if 0 <= int(i) < n_faces
    }
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    # Caps may already be claimed by an earlier pass (the outer plate faces
    # become flats), so search cap candidates over ALL faces; ring members must
    # be unclaimed (in the pool).
    planes = _planar_faces(by_index)

    features: list[ThroughPocketFeature] = []
    claimed: set[int] = set()
    seen: set[frozenset[int]] = set()

    for c1, c2, axis in _cap_pairs(planes, by_index, config):
        ax_c1 = float(np.dot(np.asarray(by_index[c1].centroid, float), axis))
        ax_c2 = float(np.dot(np.asarray(by_index[c2].centroid, float), axis))
        ax_lo, ax_hi = min(ax_c1, ax_c2), max(ax_c1, ax_c2)

        ring_pool = _shared_cap_ring(c1, c2, pool, graph)
        if len(ring_pool) < 3:
            continue

        for comp in _single_cycles(ring_pool, graph):
            key = frozenset(comp)
            if key in seen or comp & claimed:
                continue

            centroids = {i: np.asarray(by_index[i].centroid, float) for i in comp}
            ring_centroid = np.mean(list(centroids.values()), axis=0)

            # open on both faces: the loop sits axially BETWEEN the two caps.
            ax_ring = float(np.dot(ring_centroid, axis))
            if not (ax_lo - 1e-6 <= ax_ring <= ax_hi + 1e-6):
                continue

            # (4) cavity, not perimeter: every wall normal points inward.
            inward = True
            for i in comp:
                nrm = by_index[i].normal
                if nrm is None:
                    inward = False
                    break
                to_c = ring_centroid - centroids[i]
                dn = float(np.linalg.norm(to_c))
                if dn < 1e-9:
                    inward = False
                    break
                if float(np.dot(_unit(np.asarray(nrm, float)), to_c / dn)) <= config.inward_cos_min:
                    inward = False
                    break
            if not inward:
                continue

            # (3) non-round profile: planar wall present, or >= 2 axis-lines.
            planar_walls = [i for i in comp if by_index[i].surface_type == "plane"]
            cyls = [i for i in comp if by_index[i].surface_type == "cylinder"]
            n_lines = _distinct_axis_lines(cyls, by_index, axis, config)
            if not (len(planar_walls) >= 1 or n_lines >= 2):
                continue

            seen.add(key)
            # (u, v) of the ring centroid in the plane perpendicular to the
            # through-axis — display/reporting only.
            e1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
            if float(np.linalg.norm(e1)) < 1e-6:
                e1 = np.cross(axis, np.array([0.0, 1.0, 0.0]))
            e1 = _unit(e1)
            e2 = _unit(np.cross(axis, e1))
            feat = ThroughPocketFeature(
                feature_id=len(features),
                face_indices=set(comp),
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                cap_face_ids=(int(c1), int(c2)),
                n_walls=len(planar_walls),
                n_cyl=len(cyls),
                n_distinct_axis_lines=n_lines,
                centroid_uv=(float(np.dot(ring_centroid, e1)),
                             float(np.dot(ring_centroid, e2))),
            )
            features.append(feat)
            claimed |= set(comp)
            logger.info(
                "through pocket %d: %d faces (%d walls, %d cyl, %d axis-lines) "
                "caps=(%d,%d) axis=%s",
                feat.feature_id, feat.n_faces, feat.n_walls, feat.n_cyl,
                feat.n_distinct_axis_lines, c1, c2, feat.axis,
            )

    return ThroughPocketResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=pool - claimed,
        n_faces=n_faces,
        pool_size=len(pool),
        units=config.units,
    )


def render_table(result: ThroughPocketResult) -> str:
    header = (
        f"{'id':>3}  {'#f':>3}  {'walls':>5}  {'cyl':>4}  "
        f"{'lines':>5}  {'caps':>9}  faces"
    )
    lines = [header, "-" * len(header)]
    for feat in result.features:
        face_s = ",".join(str(i) for i in sorted(feat.face_indices))
        caps = f"{feat.cap_face_ids[0]},{feat.cap_face_ids[1]}"
        lines.append(
            f"{feat.feature_id:>3}  {feat.n_faces:>3}  {feat.n_walls:>5}  "
            f"{feat.n_cyl:>4}  {feat.n_distinct_axis_lines:>5}  {caps:>9}  {face_s}"
        )
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OCC-free self-tests (synthetic geometry; no STEP / no pythonocc required)
# ---------------------------------------------------------------------------
def _selftest() -> int:
    from types import SimpleNamespace as NS

    def plane(idx, centroid, normal, area=100.0):
        return NS(index=idx, surface_type="plane",
                  normal=_unit(np.asarray(normal, float)), axis=None,
                  radius=None, semi_angle_rad=None, area=area,
                  centroid=np.asarray(centroid, float))

    def cyl(idx, centroid, normal, radius=4.5, area=180.0):
        return NS(index=idx, surface_type="cylinder",
                  normal=_unit(np.asarray(normal, float)),
                  axis=np.array([1.0, 0.0, 0.0]), radius=radius,
                  semi_angle_rad=None, area=area,
                  centroid=np.asarray(centroid, float))

    def edges(triples):
        """triples: (u, v, convexity in {'concave','convex','smooth'})."""
        if not triples:
            return np.zeros((2, 0), int), np.zeros((0, 4), np.float32)
        cmap = {"concave": 0, "convex": 1, "smooth": 2}
        ei = [[], []]
        ea = []
        for u, v, c in triples:
            row = [0.0, 0.0, 0.0, 1.0]
            row[cmap[c]] = 1.0
            for a, b in ((u, v), (v, u)):
                ei[0].append(a)
                ei[1].append(b)
                ea.append(list(row))
        return np.asarray(ei, int), np.asarray(ea, np.float32)

    def inward_normal(centroid, center):
        return _unit(np.asarray(center, float) - np.asarray(centroid, float))

    # Every scenario uses CONTIGUOUS face indices (0..n-1), as the real cascade
    # does: caps are 0 and 1, ring members follow. Two opposite plate caps sit
    # along X with both normals reported +X, mirroring the real mid-normal sign
    # quirk on part2's f5/f10.
    def caps_at(x=12.5):
        return [plane(0, [-x, 0, 0], [1, 0, 0], area=5000.0),
                plane(1, [x, 0, 0], [1, 0, 0], area=5000.0)]
    CAP1, CAP2 = 0, 1

    def stadium(base, center_y):
        """5-face stadium slot: 2 planar walls + 3 arc faces, inward normals.
        Face indices are base..base+4. Returns (faces, ring_edges, ring_ids)."""
        c = np.array([0.0, center_y, 0.0])
        w1, w2, e_top, e_mid, e_bot = (base + k for k in range(5))
        fs = [
            plane(w1, [0, center_y + 4.5, 0], inward_normal([0, center_y + 4.5, 0], c)),
            plane(w2, [0, center_y - 4.5, 0], inward_normal([0, center_y - 4.5, 0], c)),
            cyl(e_top, [0, center_y + 2.0, 7.0], inward_normal([0, center_y + 2.0, 7.0], c)),
            cyl(e_mid, [0, center_y - 2.0, 7.0], inward_normal([0, center_y - 2.0, 7.0], c)),
            cyl(e_bot, [0, center_y, -7.0], inward_normal([0, center_y, -7.0], c)),
        ]
        ring = [(w1, e_top, "concave"), (e_top, e_mid, "smooth"),
                (e_mid, w2, "convex"), (w2, e_bot, "concave"),
                (e_bot, w1, "convex")]
        cap_edges = ([(i, CAP1, "convex") for i in range(base, base + 5)]
                     + [(i, CAP2, "convex") for i in range(base, base + 5)])
        return fs, ring + cap_edges, set(range(base, base + 5))

    ok = True

    def check(name, cond):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # (a) two stadium slots -> exactly two through pockets, one per slot.
    fA, eA, idA = stadium(2, 60.0)
    fB, eB, idB = stadium(7, -60.0)
    faces = caps_at() + fA + fB
    r = detect_through_pockets(faces, *edges(eA + eB),
                               candidate_faces=idA | idB)
    got = sorted(sorted(f.face_indices) for f in r.features)
    check("(a) two stadium slots -> 2 through pockets", len(r.features) == 2)
    check("(a) slot A claimed as ONE through pocket", sorted(idA) in got)
    check("(a) slot B claimed as ONE through pocket", sorted(idB) in got)
    check("(a) each pocket has planar side walls", all(f.n_walls >= 1 for f in r.features))

    # (b) round through-hole: a single bore split into 3 coaxial arc faces
    #     (one axis-line, no planar wall). Must be REJECTED -> stays a hole.
    #     Guards part1's genuine through-hole {0,15}.
    c0 = np.array([0.0, 0.0, 0.0])
    arcs = [cyl(i, p, inward_normal(p, c0)) for i, p in zip(
        (2, 3, 4), ([0, 3, 3], [0, -3, 3], [0, 0, -4.5]))]
    ring_b = [(2, 3, "smooth"), (3, 4, "smooth"), (4, 2, "smooth")]
    cap_b = ([(i, CAP1, "convex") for i in (2, 3, 4)]
             + [(i, CAP2, "convex") for i in (2, 3, 4)])
    r = detect_through_pockets(caps_at() + arcs, *edges(ring_b + cap_b),
                               candidate_faces={2, 3, 4})
    check("(b) round through-hole (1 axis-line, no wall) -> rejected", len(r.features) == 0)

    # (c) profile branch: an all-cylinder cavity with TWO distinct axis-lines and
    #     NO planar wall -> ACCEPTED (exercises the >=2-axis-lines OR-branch).
    quads = [cyl(i, p, inward_normal(p, c0)) for i, p in zip(
        (2, 3, 4, 5), ([0, 8, 3], [0, -8, 3], [0, -8, -3], [0, 8, -3]))]
    ring_c = [(2, 3, "smooth"), (3, 4, "smooth"), (4, 5, "smooth"), (5, 2, "smooth")]
    cap_c = ([(i, CAP1, "convex") for i in (2, 3, 4, 5)]
             + [(i, CAP2, "convex") for i in (2, 3, 4, 5)])
    r = detect_through_pockets(caps_at() + quads, *edges(ring_c + cap_c),
                               candidate_faces={2, 3, 4, 5})
    check("(c) all-cyl cavity, 2 axis-lines, no wall -> 1 through pocket",
          len(r.features) == 1)
    check("(c) claimed via the >=2-distinct-axis-lines branch",
          bool(r.features) and r.features[0].n_walls == 0
          and r.features[0].n_distinct_axis_lines >= 2)

    # (d) outer contour: same stadium ring but walls face OUTWARD (normals away
    #     from ring centroid). Must be REJECTED by condition 4, and its faces
    #     must remain UNCLAIMED so profile/flat passes keep their labels.
    fO, eO, idO = stadium(2, 60.0)
    for f in fO:  # flip every ring normal outward
        f.normal = -f.normal
    r = detect_through_pockets(caps_at() + fO, *edges(eO), candidate_faces=idO)
    check("(d) outward-facing contour ring -> rejected", len(r.features) == 0)
    check("(d) rejected contour faces stay unclaimed (labels preserved)",
          not (idO & r.claimed_faces) and idO <= r.remaining_faces)

    # (e) blind pocket with a floor: walls open onto ONE cap only; the far end is
    #     a FLOOR plane joined to the walls by CONCAVE edges (no second opposite
    #     open cap). Must be REJECTED (not open on both faces).
    fBl, eBl, idBl = stadium(2, 60.0)
    FLOOR = 7
    floor = plane(FLOOR, [0, 60, -9], [0, 0, 1], area=300.0)
    eBl_blind = [(u, v, c) for (u, v, c) in eBl if CAP2 not in (u, v)]
    eBl_blind += [(i, FLOOR, "concave") for i in idBl]
    r = detect_through_pockets(caps_at() + fBl + [floor], *edges(eBl_blind),
                               candidate_faces=idBl | {FLOOR})
    check("(e) blind pocket with floor (open one end) -> rejected",
          len(r.features) == 0)

    # (f) a plain block with no cavity -> empty (guards the four goldens).
    block = [plane(0, [-12.5, 0, 0], [1, 0, 0], 5000.0),
             plane(1, [12.5, 0, 0], [1, 0, 0], 5000.0),
             plane(2, [0, 10, 0], [0, 1, 0], 4000.0),
             plane(3, [0, -10, 0], [0, 1, 0], 4000.0)]
    be = [(2, 0, "convex"), (2, 1, "convex"), (3, 0, "convex"), (3, 1, "convex")]
    r = detect_through_pockets(block, *edges(be), candidate_faces={2, 3})
    check("(f) plain block -> 0 through pockets", len(r.features) == 0)

    print("self-test:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


DEFAULT_STEP = "part2.step"
DEFAULT_GRAPH_NPZ = "pipeline_out/part2/graph.npz"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Through-pocket detection")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ))
    ap.add_argument("--selftest", action="store_true",
                    help="run OCC-free synthetic self-tests and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.selftest:
        return _selftest()

    from feature_params import analyze_step
    from run_cascade import _load_edges

    step_path = Path(args.step)
    if not step_path.is_file():
        print(f"STEP not found: {step_path}")
        return 2
    faces = analyze_step(step_path)
    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    result = detect_through_pockets(faces, edge_index, edge_attr)
    print(f"STEP: {step_path}  faces: {len(faces)}")
    print(render_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
