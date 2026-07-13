"""
hole_detection.py — Stage 1 of the CAM feature-grouping cascade: HOLE DETECTION.

Where this fits
---------------
OCCT ingest (step_ingest.extract_brep_from_step / feature_params.analyze_step)
produces ~348 granular B-rep faces plus a face-adjacency graph whose edges carry
a convexity label (concave / convex / smooth) and a dihedral angle. A single
machining feature spans several of those faces, so before the GNN labels
features we group faces into feature INSTANCES with a cascade of geometric
passes:

    holes  ->  pockets  ->  flats  ->  contour surfaces  ->  outer fillets

Each pass consumes a *shared pool* of still-unclaimed face indices, CLAIMS the
faces that belong to its feature type, and hands the remaining faces to the next
pass. This module implements ONLY the first pass (holes) and the pool handoff:
`detect_holes(...)` returns the detected `HoleFeature`s plus the set of faces it
did NOT claim, ready for the (separate) pocket pass.

What counts as a hole (geometry)
--------------------------------
A hole is a family of *coaxial* cylindrical/conical faces (the wall) plus any
tangent fillets/chamfers and, for blind holes, a planar floor perpendicular to
the axis:
  * through  : cylindrical wall(s), no capping floor.
  * blind    : cylindrical wall + planar floor ~perp to axis (often via a fillet).
  * counterbore / countersink : two or more coaxial cylinders (different radii),
    or a cylinder + coaxial cone. These MUST group into ONE hole feature.

Integration points (wired to this repo)
---------------------------------------
  * Per-face geometry ....... feature_params.analyze_step()  -> list[FaceGeom]
                              (surface_type, radius, axis DIRECTION, area,
                              centroid, outward normal), TopologyExplorer order.
  * OCC faces (axis LINE) ... feature_params.load_step_faces() -> [TopoDS_Face]
                              FaceGeom only stores the axis *direction*; the axis
                              *location* (needed for colinearity) is read here
                              from BRepAdaptor_Surface.Cylinder()/Cone().Axis().
  * Face graph + convexity .. edge_index [2,E] + edge_attr [E,4] with columns
                              [concave, convex, smooth] one-hot + cos(dihedral),
                              exactly as produced by step_ingest.model_to_pyg and
                              cached in pipeline_out/<part>/graph.npz. Adjacency
                              is NOT recomputed here; `FaceGraph` just adapts it.

Units
-----
OCCT normalises STEP geometry to millimetres regardless of the file's declared
unit, so all lengths here are in **mm**. `detect_step_units()` sniffs the STEP
header for a CONVERSION_BASED_UNIT (e.g. INCH) purely so the run can log the
conversion; the reference plate is authored in inches (INCH = 25.4 mm).

No external dependencies beyond what the pipeline already uses (pythonocc-core,
numpy). Union-find is inlined; scipy is intentionally NOT pulled in.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger("hole_detection")

# Convexity code layout of edge_attr[:, :3] (see step_ingest.model_to_pyg /
# dataset.build_edge_features: "0=concave,1=convex,2=smooth").
CONVEXITY_NAMES = ("concave", "convex", "smooth")

MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HoleDetectionConfig:
    """Tolerances and guards. All linear tolerances are in model units (mm)."""

    # Two candidate faces are coaxial when their axis directions are parallel
    # within this angle AND their axis lines are colinear within the linear tol.
    axis_angular_tol_deg: float = 1.0
    # Colinearity tol = max(axis_colinear_tol * min_radius, abs_colinear_floor).
    axis_colinear_tol: float = 0.05
    abs_colinear_floor: float = 1e-3

    # A planar face is a hole floor when its normal is within this angle of the
    # axis direction (i.e. the plane is ~perpendicular to the axis).
    floor_perp_tol_deg: float = 10.0
    # A cap plane must sit within this fraction of the wall's axial span from an
    # END of that span to count as a blind floor (prevents mid-span or opening
    # planes from being read as floors).
    cap_end_tol_frac: float = 0.2
    # A neighbour edge counts as "tangent/smooth" when the angle between the two
    # face normals is <= this (the graph already buckets these as `smooth`; this
    # is the numeric fallback when cos(dihedral) is inspected directly).
    tangent_smooth_tol_deg: float = 8.0

    # Diameter CEILING (mm), applied AFTER the interior test: clusters whose
    # nominal diameter exceeds this are NOT holes (they are the central bore /
    # structural region). Their faces are deferred to the bore/contour pass
    # (added to remaining_faces, never claimed) and logged. Default ≈ 6 in.
    max_hole_diameter_mm: float = 150.0
    # Log any candidate cylinder whose diameter is >= this, so large bores /
    # the part's main body can be spotted and the ceiling tuned. Default: off.
    warn_diameter: float | None = None

    # A cylindrical/conical face qualifies as a hole WALL only when it is an
    # interior (concave) surface — its outward solid normal points toward the
    # axis. This rejects outer bodies / bosses (convex cylinders). Set False to
    # cluster every coaxial cylinder regardless of orientation.
    require_interior_wall: bool = True

    # When set, dump per-cluster face-id lists + depths for the diameter family
    # nearest this value (mm), to confirm each cluster has distinct faces.
    debug_depth_diameter: float | None = None

    # A planar floor must be geometrically consistent with the bore it caps.
    # Reject when the floor centroid's radial distance to the bore axis exceeds
    # this multiple of the bore radius, or when the floor area exceeds this
    # multiple of the bore cross-section (π R²). Both are dimensionless ratios
    # derived from the candidate hole — no absolute-mm constants.
    max_floor_centroid_dist_frac: float = 2.0
    max_floor_area_frac: float = 9.0

    # Model units label, for logging/reporting only (OCCT already emits mm).
    units: str = "mm"


# ---------------------------------------------------------------------------
# Geometry containers
# ---------------------------------------------------------------------------
@dataclass
class Axis:
    """An infinite axis line: a point on it plus a unit direction."""

    point: np.ndarray  # (3,)
    direction: np.ndarray  # (3,), unit

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "point": [round(float(x), 6) for x in self.point],
            "direction": [round(float(x), 6) for x in self.direction],
        }


@dataclass
class Candidate:
    """A cylinder/cone face that may be part of a hole wall."""

    face_index: int
    kind: str  # "cylinder" | "cone"
    axis: Axis
    radius: float | None  # cylinder radius, or cone reference radius
    interior: bool  # outward normal points toward the axis (concave wall)


@dataclass
class HoleFeature:
    feature_id: int
    kind: str  # "through_hole" | "blind_hole" | "drilled_blind_hole"
    face_indices: set[int]
    axis: Axis
    nominal_diameter: float
    depth: float | None
    is_counterbore: bool
    is_countersink: bool
    # Debug / provenance (not required by the contract, handy downstream):
    radius: float = 0.0
    cylinder_face_indices: list[int] = field(default_factory=list)
    cone_face_indices: list[int] = field(default_factory=list)
    floor_face_indices: list[int] = field(default_factory=list)
    blend_face_indices: list[int] = field(default_factory=list)
    n_distinct_radii: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "face_indices": sorted(self.face_indices),
            "n_faces": len(self.face_indices),
            "axis": self.axis.as_dict(),
            "nominal_diameter": round(float(self.nominal_diameter), 6),
            "radius": round(float(self.radius), 6),
            "depth": None if self.depth is None else round(float(self.depth), 6),
            "is_counterbore": self.is_counterbore,
            "is_countersink": self.is_countersink,
            "n_distinct_radii": self.n_distinct_radii,
            "cylinder_face_indices": sorted(self.cylinder_face_indices),
            "cone_face_indices": sorted(self.cone_face_indices),
            "floor_face_indices": sorted(self.floor_face_indices),
            "blend_face_indices": sorted(self.blend_face_indices),
        }


@dataclass
class HoleDetectionResult:
    features: list[HoleFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    # Opening-tier torus fillets released from blind-hole bodies (Toolpath outer
    # fillet role) — NOT claimed here; consumed by the outer-fillet pass.
    deferred_feature_fillets: set[int] = field(default_factory=set)
    deferred_feature_fillet_groups: list[set[int]] = field(default_factory=list)
    # Diagnostics for tuning / next-pass handoff.
    skipped_bspline_faces: list[int] = field(default_factory=list)
    rejected_convex_faces: list[int] = field(default_factory=list)
    oversize_faces: list[int] = field(default_factory=list)
    degenerate_faces: list[int] = field(default_factory=list)
    candidate_diag: list[dict[str, Any]] = field(default_factory=list)
    units: str = "mm"

    def summary(self) -> str:
        n_blind = sum(1 for f in self.features if f.kind in ("blind_hole", "drilled_blind_hole"))
        n_through = sum(1 for f in self.features if f.kind == "through_hole")
        return (
            f"{len(self.features)} holes ({n_blind} blind, {n_through} through); "
            f"claimed {len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain for next pass"
        )


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _axes_parallel(d1: np.ndarray, d2: np.ndarray, ang_tol_deg: float) -> bool:
    """+d and -d are treated as the same axis (compare |dot|)."""
    c = abs(float(np.dot(_unit(d1), _unit(d2))))
    return c >= math.cos(math.radians(ang_tol_deg))


def _line_line_distance(a: Axis, b: Axis) -> float:
    """Perpendicular distance between two ~parallel axis lines."""
    d = _unit(a.direction + math.copysign(1.0, float(np.dot(a.direction, b.direction))) * b.direction)
    if float(np.linalg.norm(d)) < 1e-9:
        d = _unit(a.direction)
    delta = np.asarray(b.point, dtype=np.float64) - np.asarray(a.point, dtype=np.float64)
    perp = delta - float(np.dot(delta, d)) * d
    return float(np.linalg.norm(perp))


# ---------------------------------------------------------------------------
# Inlined union-find (connected components over the coaxial relation)
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, keys: Iterable[int]):
        self.parent = {k: k for k in keys}

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for k in self.parent:
            out.setdefault(self.find(k), []).append(k)
        return list(out.values())


# ---------------------------------------------------------------------------
# Face graph adapter (reuses existing convexity; does NOT recompute adjacency)
# ---------------------------------------------------------------------------
class FaceGraph:
    """Undirected face adjacency with per-pair convexity, keyed by face index.

    Built from the `edge_index`/`edge_attr` tensors already produced by
    step_ingest.model_to_pyg (and cached in graph.npz). The index space must be
    the SAME as the FaceGeom list (TopologyExplorer order). For the reference
    part both are 348 faces with no zero-area faces filtered, so they align.
    """

    def __init__(self, n_faces: int):
        self.n_faces = n_faces
        self.neighbors: dict[int, set[int]] = {i: set() for i in range(n_faces)}
        self._convexity: dict[tuple[int, int], str] = {}
        self._cos: dict[tuple[int, int], float] = {}

    @staticmethod
    def _key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    @classmethod
    def from_edge_tensors(
        cls,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        n_faces: int,
    ) -> "FaceGraph":
        g = cls(n_faces)
        edge_index = np.asarray(edge_index)
        edge_attr = np.asarray(edge_attr)
        if edge_index.size == 0:
            return g
        for k in range(edge_index.shape[1]):
            u, v = int(edge_index[0, k]), int(edge_index[1, k])
            if u == v or u >= n_faces or v >= n_faces:
                continue
            key = cls._key(u, v)
            g.neighbors[u].add(v)
            g.neighbors[v].add(u)
            row = edge_attr[k]
            cid = int(np.argmax(row[:3])) if row.shape[0] >= 3 else 2
            g._convexity[key] = CONVEXITY_NAMES[cid]
            if row.shape[0] >= 4:
                g._cos[key] = float(row[3])
        return g

    def edge_kind(self, u: int, v: int) -> str | None:
        return self._convexity.get(self._key(u, v))

    def edge_cos(self, u: int, v: int) -> float | None:
        return self._cos.get(self._key(u, v))


def induced_subgraph_components(
    face_ids: set[int] | Sequence[int],
    graph: FaceGraph,
) -> list[list[int]]:
    """Connected components of the subgraph induced on *face_ids* by B-rep adjacency."""
    fset = set(face_ids)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for s in sorted(fset):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph.neighbors.get(u, ()) & fset:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return sorted(comps, key=lambda c: (-len(c), c))


# ---------------------------------------------------------------------------
# OCC-dependent extraction (only touched when OCC faces are supplied)
# ---------------------------------------------------------------------------
def _axis_from_occ_face(occ_face) -> tuple[Axis, float | None, str] | None:
    """Read axis (location + direction) and radius for a cylinder/cone face."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder

    surf = BRepAdaptor_Surface(occ_face, True)
    st = surf.GetType()
    if st == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        ax = cyl.Axis()
        loc, d = ax.Location(), ax.Direction()
        point = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
        direction = _unit(np.array([d.X(), d.Y(), d.Z()], dtype=np.float64))
        return Axis(point, direction), float(cyl.Radius()), "cylinder"
    if st == GeomAbs_Cone:
        cone = surf.Cone()
        ax = cone.Axis()
        loc, d = ax.Location(), ax.Direction()
        point = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
        direction = _unit(np.array([d.X(), d.Y(), d.Z()], dtype=np.float64))
        return Axis(point, direction), float(cone.RefRadius()), "cone"
    return None


def _axial_length(occ_face, axis: Axis) -> float:
    """Extent of a face's boundary sampled along the axis direction."""
    # Wired to brep_extents.collect_boundary_points (boundary vertices + edge
    # curve samples) — the repo's existing "real geometry, not centroid" sampler.
    from brep.brep_extents import collect_boundary_points

    pts = collect_boundary_points(occ_face)
    projs = (pts - axis.point) @ axis.direction
    return float(projs.max() - projs.min())


def _cluster_axial_extent(occ_faces_by_index: dict[int, Any], face_ids: Sequence[int], axis: Axis) -> float | None:
    from brep.brep_extents import collect_boundary_points

    projs: list[float] = []
    for fid in face_ids:
        face = occ_faces_by_index.get(fid)
        if face is None:
            continue
        pts = collect_boundary_points(face)
        projs.extend((pts - axis.point) @ axis.direction)
    if not projs:
        return None
    return float(max(projs) - min(projs))


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------
def _interior_wall_sign(centroid: np.ndarray, normal: np.ndarray, axis: Axis) -> float:
    """Signed projection of the outward normal onto the outward radial direction.

    radial_outward = component of (centroid - axis) perpendicular to the axis.
    `face_mid_normal` (step_ingest) returns the OUTWARD-from-material normal
    (flipped on TopAbs_REVERSED), so a hole wall (void toward the axis) yields a
    NEGATIVE sign and a boss/outer body a POSITIVE sign. Returns +0.0 when the
    centroid sits on the axis (degenerate) so callers treat it as non-interior.
    """
    delta = np.asarray(centroid, dtype=np.float64) - axis.point
    radial = delta - float(np.dot(delta, axis.direction)) * axis.direction
    ru = _unit(radial)
    n = _unit(np.asarray(normal, dtype=np.float64))
    if float(np.linalg.norm(ru)) < 1e-9 or float(np.linalg.norm(n)) < 1e-9:
        return 0.0
    return float(np.dot(n, ru))


def _interior_sign_occ(occ_face, axis: Axis, n_samples: int = 5) -> tuple[float, float, int]:
    """Robust interior sign from several outward normals sampled across a face.

    A single mid-UV normal can be undefined or unrepresentative on a partial-arc
    wall, which is how a convex outer body can slip past a mid-normal test. This
    samples an n×n UV grid, computes the OUTWARD normal at each valid point
    (orientation-corrected like step_ingest.face_mid_normal), and returns
    (mean_sign, fraction_negative, n_valid) of normal·radial_outward.
    mean_sign < 0 ⇒ interior/hole wall; > 0 ⇒ boss/outer body.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_REVERSED

    umin, umax, vmin, vmax = breptools.UVBounds(occ_face)
    surf = BRepAdaptor_Surface(occ_face, True)
    flip = occ_face.Orientation() == TopAbs_REVERSED
    signs: list[float] = []
    for u in np.linspace(umin, umax, n_samples):
        for v in np.linspace(vmin, vmax, n_samples):
            props = BRepLProp_SLProps(surf, float(u), float(v), 1, 1e-6)
            if not props.IsNormalDefined():
                continue
            dnorm = props.Normal()
            nrm = np.array([dnorm.X(), dnorm.Y(), dnorm.Z()], dtype=np.float64)
            if flip:
                nrm = -nrm
            pv = props.Value()
            p = np.array([pv.X(), pv.Y(), pv.Z()], dtype=np.float64)
            delta = p - axis.point
            radial = delta - float(np.dot(delta, axis.direction)) * axis.direction
            ru = _unit(radial)
            nu = _unit(nrm)
            if float(np.linalg.norm(ru)) < 1e-9 or float(np.linalg.norm(nu)) < 1e-9:
                continue
            signs.append(float(np.dot(nu, ru)))
    if not signs:
        return 0.0, 0.0, 0
    arr = np.asarray(signs)
    return float(arr.mean()), float((arr < 0).mean()), int(arr.size)


def _collect_candidates(
    faces: Sequence[Any],
    occ_faces: Sequence[Any] | None,
    config: HoleDetectionConfig,
    result_bins: dict[str, Any],
    candidate_faces: set[int] | None = None,
) -> list[Candidate]:
    """Collect cylinder/cone faces as hole-wall candidates.

    `faces` are FaceGeom records (surface_type, radius, axis direction, centroid,
    normal). `occ_faces` (same index order) supply the axis LOCATION that
    FaceGeom lacks; when absent, the direction-only axis is used with the face
    centroid as its point (degraded — used by the OCC-free self-test only).

    When `candidate_faces` is given (the reordered cascade feeds the pocket
    pass's remaining_faces here), only those face indices are eligible — faces a
    prior pass already claimed are never re-considered. This restricts the INPUT
    source only; the wall/floor/coaxial LOGIC below is unchanged.
    """
    candidates: list[Candidate] = []
    for fg in faces:
        st = getattr(fg, "surface_type", None)
        idx = int(fg.index)
        if candidate_faces is not None and idx not in candidate_faces:
            continue
        if st in ("bspline", "bezier"):
            # "Cylindrical-ish" B-splines don't type as GeomAbs_Cylinder; a later
            # pass handles them. Log and skip.
            result_bins["skipped_bspline_faces"].append(idx)
            continue
        if st not in ("cylinder", "cone"):
            continue

        radius = getattr(fg, "radius", None)
        if occ_faces is not None:
            occ_info = _axis_from_occ_face(occ_faces[idx])
            if occ_info is None:
                continue
            axis, occ_radius, _kind = occ_info
            if radius is None:
                radius = occ_radius
            # Robust multi-sample interior sign from the OCC face.
            sign, frac_neg, n_valid = _interior_sign_occ(occ_faces[idx], axis)
        else:
            direction = getattr(fg, "axis", None)
            if direction is None:
                continue
            # FaceGeom has no axis location; the OCC-free path uses an explicit
            # axis_location when supplied (self-test), else the centroid.
            point = getattr(fg, "axis_location", None)
            if point is None:
                point = fg.centroid
            axis = Axis(np.asarray(point, dtype=np.float64), _unit(direction))
            sign = _interior_wall_sign(
                np.asarray(fg.centroid, dtype=np.float64),
                np.asarray(fg.normal, dtype=np.float64),
                axis,
            )
            frac_neg, n_valid = (1.0 if sign < 0 else 0.0), 1

        interior = sign < 0.0
        diam = None if radius is None else 2.0 * float(radius)
        # A real cylinder wall has radius > 0; a zero/None cylinder radius is a
        # degenerate kernel read and is not a hole candidate. Cones legitimately
        # report RefRadius 0 (apex reference), so they are NOT treated as
        # degenerate here — they only matter as countersink attachments.
        degenerate = st == "cylinder" and (radius is None or radius <= 1e-9)
        accepted = (not degenerate) and (interior or not config.require_interior_wall)
        result_bins["candidate_diag"].append({
            "face": idx, "kind": st, "radius": radius, "diameter": diam,
            "interior_sign": round(sign, 4), "frac_negative": round(frac_neg, 3),
            "n_samples": n_valid, "accepted": bool(accepted), "degenerate": degenerate,
        })

        if degenerate:
            result_bins["degenerate_faces"].append(idx)
            continue

        if config.require_interior_wall and not interior:
            # Outward normal points AWAY from the axis on average ⇒ convex/outer
            # body, not a hole wall. Left in the pool for later passes.
            result_bins["rejected_convex_faces"].append(idx)
            continue

        if diam is not None and config.warn_diameter is not None and diam >= config.warn_diameter:
            logger.warning(
                "candidate face %d has large diameter %.3f %s (>= warn_diameter %.3f); "
                "check it is a real hole and not the part body/main bore",
                idx, diam, config.units, config.warn_diameter,
            )

        candidates.append(Candidate(
            face_index=idx,
            kind=st,
            axis=axis,
            radius=radius,
            interior=interior,
        ))
    return candidates


# ---------------------------------------------------------------------------
# Coaxial clustering
# ---------------------------------------------------------------------------
def _coaxial(a: Candidate, b: Candidate, config: HoleDetectionConfig) -> bool:
    if not _axes_parallel(a.axis.direction, b.axis.direction, config.axis_angular_tol_deg):
        return False
    radii = [r for r in (a.radius, b.radius) if r is not None and r > 0]
    min_r = min(radii) if radii else 0.0
    lin_tol = max(config.axis_colinear_tol * min_r, config.abs_colinear_floor)
    return _line_line_distance(a.axis, b.axis) <= lin_tol


def _connectivity_components(graph: FaceGraph, candidate_faces: set[int]) -> dict[int, int]:
    """Connected-component id per face over *traversable* edges only.

    An edge is traversable when it is SMOOTH or CONCAVE (so a walk can chain
    through fillets/shoulders/floors of a single hole), OR it directly joins two
    candidate faces (partial-arc seams / abutting coaxial cylinders, whatever the
    convexity). Hole-rim edges (wall↔flat top/bottom) are CONVEX, so this walk
    does NOT bridge two distinct holes that merely open onto a shared planar
    face — which is exactly what let concentric holes over-merge before.
    """
    uf = _UnionFind(range(graph.n_faces))
    for u, nbrs in graph.neighbors.items():
        for v in nbrs:
            if v <= u:
                continue
            kind = graph.edge_kind(u, v)
            traversable = kind in ("smooth", "concave") or (
                u in candidate_faces and v in candidate_faces
            )
            if traversable:
                uf.union(u, v)
    return {f: uf.find(f) for f in range(graph.n_faces)}


def _cluster_candidates(
    candidates: list[Candidate],
    graph: FaceGraph,
    config: HoleDetectionConfig,
) -> list[list[Candidate]]:
    """Group candidates that are BOTH coaxial AND connected in the face graph.

    Coaxiality alone fuses every feature that happens to share the part's centre
    axis (concentric holes). A merge therefore additionally requires the two
    faces to lie in the same connected component of the traversable-edge graph:
    two coaxial cylinders that do not touch in the graph are DIFFERENT holes.
    """
    comp = _connectivity_components(graph, {c.face_index for c in candidates})
    uf = _UnionFind(range(len(candidates)))
    for i in range(len(candidates)):
        ci = candidates[i]
        for j in range(i + 1, len(candidates)):
            cj = candidates[j]
            if comp.get(ci.face_index) != comp.get(cj.face_index):
                continue  # not connected in the graph -> different holes
            if _coaxial(ci, cj, config):
                uf.union(i, j)
    return [[candidates[i] for i in grp] for grp in uf.groups()]


def _log_interior_families(diag: list[dict[str, Any]], config: HoleDetectionConfig) -> None:
    """Log the interior-wall verdict per diameter family (convention check).

    For each distinct candidate diameter, reports face count, the mean measured
    normal·(axis→point) sign, and whether the family was ACCEPTED (interior /
    hole wall, sign < 0) or REJECTED (convex / outer body, sign > 0). Lets the
    caller confirm the normal sign convention and that big outer-body families
    (e.g. ⌀280.416 / ⌀525.780) are rejected.
    """
    if not diag:
        return
    families: dict[float, list[dict[str, Any]]] = {}
    for d in diag:
        key = round(d["diameter"], 3) if d["diameter"] is not None else -1.0
        families.setdefault(key, []).append(d)

    logger.info(
        "interior-wall verdict by diameter family (sign<0 ⇒ hole wall, >0 ⇒ outer body):"
    )
    for diam in sorted(families):
        rows = families[diam]
        signs = [r["interior_sign"] for r in rows]
        mean_sign = float(np.mean(signs))
        n_acc = sum(1 for r in rows if r["accepted"])
        verdict = "ACCEPT(interior)" if mean_sign < 0 else "REJECT(convex/outer)"
        if not config.require_interior_wall:
            verdict = "ACCEPT(filter off)"
        in_val = f"{diam / MM_PER_INCH:.3f}in" if diam >= 0 else "n/a"
        logger.info(
            "  ⌀%9.3f mm (%8s)  faces=%2d  mean_sign=%+.3f  accepted=%d/%d  -> %s",
            diam, in_val, len(rows), mean_sign, n_acc, len(rows), verdict,
        )


def _log_degenerate_families(diag: list[dict[str, Any]]) -> None:
    """Dump every candidate face whose diameter is ~0 or None (per-face detail).

    A ⌀0.000 family is almost always cones (RefRadius reads 0 at the apex) or a
    failed radius extraction; list face index, surface_type and radius so the
    caller can see what they actually are.
    """
    rows = [d for d in diag if d["diameter"] is None or abs(d["diameter"]) < 1e-6]
    if not rows:
        return
    logger.info("degenerate / ⌀0.000 candidate faces (%d):", len(rows))
    for d in sorted(rows, key=lambda r: r["face"]):
        logger.info(
            "  face %3d  type=%-8s radius=%s  degenerate=%s",
            d["face"], d["kind"],
            "None" if d["radius"] is None else f"{d['radius']:.6f}",
            d.get("degenerate", False),
        )


def _log_depth_debug(features: list["HoleFeature"], config: HoleDetectionConfig) -> None:
    """Dump per-cluster face-id lists + depths for the target diameter family.

    Confirms each cluster of a given diameter has DISTINCT faces (and therefore
    an independently computed depth). Asserts the face sets are distinct.
    """
    target = config.debug_depth_diameter
    if target is None:
        return
    sel = [f for f in features if abs(f.nominal_diameter - target) <= 1.0]
    logger.info(
        "depth debug for ⌀~%.3f mm family: %d cluster(s)", target, len(sel),
    )
    face_sets: list[frozenset[int]] = []
    for f in sel:
        fs = frozenset(f.face_indices)
        logger.info(
            "  feature %d: depth=%s  faces=%s",
            f.feature_id,
            "None" if f.depth is None else f"{f.depth:.3f}",
            sorted(f.face_indices),
        )
        face_sets.append(fs)
    # Distinct clusters must own distinct face sets (else depth would alias).
    assert len(face_sets) == len(set(face_sets)), (
        "depth-debug: two clusters share the same face set — depth would alias"
    )
    depths = {round(f.depth, 3) for f in sel if f.depth is not None}
    if len(sel) > 1 and len(depths) == 1:
        logger.info(
            "  note: all %d clusters share depth %s — faces are distinct, so this "
            "is genuine identical geometry, not a family-aliasing bug",
            len(sel), next(iter(depths)),
        )


# ---------------------------------------------------------------------------
# Attach floors + fillets/chamfers via the existing convexity graph
# ---------------------------------------------------------------------------
_BLEND_TYPES = ("torus", "bspline", "bezier")
_CURVED_BORE_TYPES = frozenset({"cylinder", "cone", "torus", "bspline", "bezier"})


def _is_top_opening_ring_plane(
    plane_idx: int,
    cluster_ids: set[int],
    blend_ids: set[int],
    faces: Sequence[Any],
    graph: FaceGraph,
    occ_faces_by_index: dict[int, Any] | None,
    axis: Axis,
    config: HoleDetectionConfig,
) -> bool:
    """True when a plane sits opening-ward of all curved bore geometry and meets
    it via a convex opening-rim edge (top flat ring), not a blind cap/floor.

    Relative to the coaxial group's own curved faces — no part-specific depth
    constants. Keeps entry-tier hole rings (concave fillet + recess wall bounded)
    while releasing the mouth flat that only kisses the entry fillet convexly.
    """
    group_curved = {
        i for i in (cluster_ids | blend_ids)
        if getattr(faces[i], "surface_type", None) in _CURVED_BORE_TYPES
    }
    if not group_curved:
        return False

    curved_pos = [
        _axial_position(i, faces, occ_faces_by_index, axis) for i in group_curved
    ]
    max_curved = max(curved_pos)
    span = max(curved_pos) - min(curved_pos)
    tol = max(config.abs_colinear_floor, 0.02 * max(span, 1e-9))

    plane_pos = _axial_position(plane_idx, faces, occ_faces_by_index, axis)
    if plane_pos <= max_curved + tol:
        return False

    for nb in graph.neighbors.get(plane_idx, ()):
        if nb not in group_curved:
            continue
        if graph.edge_kind(plane_idx, nb) == "convex":
            return True
    return False


def _is_opening_tier_blend(
    blend_idx: int,
    cluster_ids: set[int],
    body_floor_ids: set[int],
    faces: Sequence[Any],
    graph: FaceGraph,
    occ_faces_by_index: dict[int, Any] | None,
    axis: Axis,
    config: HoleDetectionConfig,
) -> bool:
    """True when a blend sits at the opening cap of a coaxial stack.

    Toolpath "outer fillet" at a stack: opening-ward of the wall cylinders and
    adjacent to the opening-ring flat (or any plane opening-ward of the walls).
    Deeper side fillets (wall↔floor) stay in the blind-hole body.
    """
    if getattr(faces[blend_idx], "surface_type", None) not in _BLEND_TYPES:
        return False

    wall_curved = {
        i for i in cluster_ids
        if getattr(faces[i], "surface_type", None) in _CURVED_BORE_TYPES
    }
    if not wall_curved:
        return False

    curved_pos = [
        _axial_position(i, faces, occ_faces_by_index, axis) for i in wall_curved
    ]
    max_curved = max(curved_pos)
    span = max(curved_pos) - min(curved_pos)
    tol = max(config.abs_colinear_floor, 0.02 * max(span, 1e-9))

    blend_pos = _axial_position(blend_idx, faces, occ_faces_by_index, axis)
    if blend_pos < max_curved - tol:
        return False

    for nb in graph.neighbors.get(blend_idx, ()):
        nb_st = getattr(faces[nb], "surface_type", None)
        if nb_st != "plane":
            continue
        if _is_top_opening_ring_plane(
            nb, cluster_ids, set(), faces, graph,
            occ_faces_by_index, axis, config,
        ):
            return True
        plane_pos = _axial_position(nb, faces, occ_faces_by_index, axis)
        if plane_pos > max_curved + tol:
            return True
    return False


def _split_hole_stack_tiers(
    cluster_ids: set[int],
    floor_ids: set[int],
    blend_ids: set[int],
    faces: Sequence[Any],
    graph: FaceGraph,
    occ_faces_by_index: dict[int, Any] | None,
    axis: Axis,
    config: HoleDetectionConfig,
) -> tuple[set[int], set[int], set[int]]:
    """Return (body_floors, body_blends, boundary_fillets) for one coaxial stack."""
    boundary: set[int] = set()
    body_blends: set[int] = set()
    for bid in blend_ids:
        if _is_opening_tier_blend(
            bid, cluster_ids, floor_ids, faces, graph,
            occ_faces_by_index, axis, config,
        ):
            boundary.add(bid)
        else:
            body_blends.add(bid)
    return set(floor_ids), body_blends, boundary


def _axial_position(
    face_index: int,
    faces: Sequence[Any],
    occ_faces_by_index: dict[int, Any] | None,
    axis: Axis,
) -> float:
    """Mean axial coordinate of a face along the axis (OCC boundary else centroid)."""
    if occ_faces_by_index is not None and occ_faces_by_index.get(face_index) is not None:
        from brep.brep_extents import collect_boundary_points

        pts = collect_boundary_points(occ_faces_by_index[face_index])
        return float(np.mean((pts - axis.point) @ axis.direction))
    centroid = np.asarray(faces[face_index].centroid, dtype=np.float64)
    return float(np.dot(centroid - axis.point, axis.direction))


def _radial_dist_to_axis(
    point: Sequence[float],
    axis: Axis,
) -> float:
    """Perpendicular distance from a point to the bore axis line."""
    delta = np.asarray(point, dtype=np.float64) - np.asarray(axis.point, dtype=np.float64)
    d = _unit(axis.direction)
    perp = delta - float(np.dot(delta, d)) * d
    return float(np.linalg.norm(perp))


def _floor_matches_bore(
    face_index: int,
    faces: Sequence[Any],
    axis: Axis,
    bore_radius: float,
    config: HoleDetectionConfig,
) -> bool:
    """True when a planar face is a geometrically plausible blind-hole floor."""
    if bore_radius <= 1e-9:
        return False
    fg = faces[face_index]
    area = getattr(fg, "area", None)
    centroid = np.asarray(getattr(fg, "centroid", (0.0, 0.0, 0.0)), dtype=np.float64)
    radial = _radial_dist_to_axis(centroid, axis)
    if radial > config.max_floor_centroid_dist_frac * bore_radius:
        return False
    if area is not None and float(area) > 0.0:
        bore_area = math.pi * bore_radius * bore_radius
        if float(area) > config.max_floor_area_frac * bore_area:
            return False
    return True


def _wall_axial_span(
    occ_faces_by_index: dict[int, Any] | None,
    wall_face_ids: Sequence[int],
    faces: Sequence[Any],
    axis: Axis,
) -> tuple[float, float] | None:
    """(min, max) axial coordinate of the wall faces' boundary along the axis."""
    projs: list[float] = []
    if occ_faces_by_index is not None:
        from brep.brep_extents import collect_boundary_points

        for fid in wall_face_ids:
            face = occ_faces_by_index.get(fid)
            if face is None:
                continue
            pts = collect_boundary_points(face)
            projs.extend((pts - axis.point) @ axis.direction)
    if not projs:
        return None
    return float(min(projs)), float(max(projs))


def _axial_face_extent(
    face_index: int,
    faces: Sequence[Any],
    occ_faces_by_index: dict[int, Any] | None,
    axis: Axis,
) -> tuple[float, float]:
    """(min, max) axial coordinate of ONE face along the axis.

    Uses the OCC boundary sampler when the face is available; falls back to the
    face centroid (a single point, so min == max) in the OCC-free path.
    """
    if occ_faces_by_index is not None and occ_faces_by_index.get(face_index) is not None:
        from brep.brep_extents import collect_boundary_points

        pts = collect_boundary_points(occ_faces_by_index[face_index])
        projs = (pts - axis.point) @ axis.direction
        return float(projs.min()), float(projs.max())
    p = _axial_position(face_index, faces, occ_faces_by_index, axis)
    return p, p


def _has_drill_tip_cap(
    cyls: list[Candidate],
    cones: list[Candidate],
    cluster_ids: set[int],
    faces: Sequence[Any],
    graph: FaceGraph,
    occ_faces_by_index: dict[int, Any] | None,
    axis: Axis,
    config: HoleDetectionConfig,
) -> bool:
    """True when a coaxial cone caps the bore as a DRILL POINT (blind), as opposed
    to a countersink that merely chamfers an opening (through).

    A blind drilled hole terminates in a conical point (the twist-drill tip): a
    cone, coaxial with the wall (guaranteed by cluster membership), whose apex is
    buried in the stock. It differs from a countersink cone in two locally
    measurable ways, and a cone must satisfy BOTH to be read as a cap:

      * it EXTENDS the bore axially OUTWARD past the cylinder-wall span on one end
        (its outer boundary lies beyond that end by more than a small fraction of
        the wall span) — a mid-span cone is a step, not a cap; and
      * that end is CLOSED — the cone has NO planar neighbour, and NO convex-rim
        opening plane sits within a small fraction of the wall span of the cone's
        outer extreme. A countersink cone instead FLARES into an opening plane
        (planar neighbour and/or a convex opening at its outer end).

    All tolerances are fractions of THIS cluster's own cylinder span — no absolute
    lengths, no face-id lists, no part gate. The wall span is measured from the
    CYLINDERS only (not cones), so a capping cone can register as "beyond" it.
    """
    if not cones or not cyls:
        return False

    cyl_ext = [
        _axial_face_extent(c.face_index, faces, occ_faces_by_index, axis)
        for c in cyls
    ]
    wmin = min(e[0] for e in cyl_ext)
    wmax = max(e[1] for e in cyl_ext)
    span = max(wmax - wmin, 1e-9)
    out_tol = max(config.cap_end_tol_frac * 0.1 * span, config.abs_colinear_floor)
    open_tol = config.cap_end_tol_frac * span

    for cone in cones:
        ci = cone.face_index
        cmin, cmax = _axial_face_extent(ci, faces, occ_faces_by_index, axis)
        beyond_hi = cmax > wmax + out_tol
        beyond_lo = cmin < wmin - out_tol
        if not (beyond_hi or beyond_lo):
            continue  # cone sits within the wall span — a step, not an end cap

        # A drill-tip cone is buried in material: it abuts no planar face. A
        # countersink cone flares into the opening plane it chamfers.
        if any(
            getattr(faces[nb], "surface_type", None) == "plane"
            for nb in graph.neighbors.get(ci, ())
        ):
            continue

        # No exterior opening (convex-rim plane) at the cone's OUTER end.
        ext = cmax if beyond_hi else cmin
        opening_at_end = False
        for f in cluster_ids:
            for nb in graph.neighbors.get(f, ()):
                if getattr(faces[nb], "surface_type", None) != "plane":
                    continue
                if graph.edge_kind(f, nb) != "convex":
                    continue
                pos = _axial_position(nb, faces, occ_faces_by_index, axis)
                if abs(pos - ext) <= open_tol:
                    opening_at_end = True
                    break
            if opening_at_end:
                break
        if opening_at_end:
            continue

        return True
    return False


def _attach_faces(
    cluster: list[Candidate],
    faces: Sequence[Any],
    graph: FaceGraph,
    axis: Axis,
    config: HoleDetectionConfig,
    *,
    occ_faces_by_index: dict[int, Any] | None = None,
    wall_span: tuple[float, float] | None = None,
    candidate_faces: set[int] | None = None,
    bore_radius: float = 0.0,
    claimed_floor_faces: set[int] | None = None,
) -> tuple[set[int], set[int]]:
    """Return (floor_face_ids, blend_face_ids) attached to a coaxial cluster.

    Floors (blind caps): a planar neighbour is a cap ONLY when ALL hold:
      * it is joined to a wall by a CONCAVE or SMOOTH edge — a hole-rim/opening
        that the wall passes THROUGH is a CONVEX edge and is never a cap;
      * its normal is ~parallel to the axis (plane ⟂ axis);
      * it sits at an axial END of the wall span (within cap_end_tol_frac);
      * its centroid and area are consistent with the bore radius (see config);
      * it is not already the floor of another hole feature (single ownership).
    Blends: torus/bspline/bezier faces reachable from cluster faces across SMOOTH
    or CONCAVE edges (traversed transitively for multi-face fillet rings).
    """
    cluster_ids = {c.face_index for c in cluster}
    cos_floor = math.cos(math.radians(config.floor_perp_tol_deg))

    end_tol = None
    if wall_span is not None:
        span = max(wall_span[1] - wall_span[0], 1e-9)
        end_tol = max(config.cap_end_tol_frac * span, config.abs_colinear_floor)

    floor_ids: set[int] = set()
    blend_ids: set[int] = set()
    visited = set(cluster_ids)
    queue: list[int] = list(cluster_ids)

    while queue:
        f = queue.pop()
        for nb in graph.neighbors.get(f, ()):
            if nb in visited or nb in cluster_ids:
                continue
            if candidate_faces is not None and nb not in candidate_faces:
                continue  # a face a prior pass claimed is not attachable here
            fg = faces[nb]
            st = getattr(fg, "surface_type", None)
            kind = graph.edge_kind(f, nb)
            if st == "plane":
                # Convex rim = the plate face the hole opens through, not a cap.
                if kind not in ("concave", "smooth"):
                    continue
                n = _unit(np.asarray(fg.normal, dtype=np.float64))
                if abs(float(np.dot(n, axis.direction))) < cos_floor:
                    continue  # not perpendicular to the axis
                if end_tol is not None:
                    pos = _axial_position(nb, faces, occ_faces_by_index, axis)
                    # At/BEYOND an end (a blind floor can sit just past the wall
                    # bottom, below the fillet); only a MID-span plane is excluded.
                    at_end = (
                        pos <= wall_span[0] + end_tol
                        or pos >= wall_span[1] - end_tol
                    )
                    if not at_end:
                        continue  # mid-span perpendicular plane is not a floor
                if _is_top_opening_ring_plane(
                    nb, cluster_ids, blend_ids, faces, graph,
                    occ_faces_by_index, axis, config,
                ):
                    continue  # top opening flat — leave for the flat pass
                if claimed_floor_faces is not None and nb in claimed_floor_faces:
                    continue  # single ownership — already another hole's floor
                if not _floor_matches_bore(nb, faces, axis, bore_radius, config):
                    continue  # geometrically inconsistent with this bore
                floor_ids.add(nb)
                visited.add(nb)  # cap terminates the walk
                continue
            if st in _BLEND_TYPES and kind in ("smooth", "concave"):
                blend_ids.add(nb)
                visited.add(nb)
                queue.append(nb)  # keep walking through blend chains
    return floor_ids, blend_ids


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _distinct_radii(cyls: list[Candidate], tol: float = 1e-3) -> int:
    vals = sorted(c.radius for c in cyls if c.radius is not None)
    if not vals:
        return 0
    distinct = [vals[0]]
    for r in vals[1:]:
        if abs(r - distinct[-1]) > tol:
            distinct.append(r)
    return len(distinct)


def _primary_cylinder(
    cyls: list[Candidate],
    occ_faces_by_index: dict[int, Any] | None,
) -> Candidate | None:
    """Nominal radius comes from the longest cylinder (fallback: any cylinder)."""
    if not cyls:
        return None
    if occ_faces_by_index is not None:
        def length(c: Candidate) -> float:
            face = occ_faces_by_index.get(c.face_index)
            if face is None:
                return 0.0
            try:
                return _axial_length(face, c.axis)
            except Exception:  # noqa: BLE001 - degenerate boundary sampling
                return 0.0
        return max(cyls, key=length)
    return cyls[0]


def _classify_cluster(
    feature_id: int,
    cluster: list[Candidate],
    faces: Sequence[Any],
    graph: FaceGraph,
    occ_faces_by_index: dict[int, Any] | None,
    config: HoleDetectionConfig,
    candidate_faces: set[int] | None = None,
    claimed_floor_faces: set[int] | None = None,
) -> tuple[HoleFeature | None, set[int]]:
    cyls = [c for c in cluster if c.kind == "cylinder"]
    cones = [c for c in cluster if c.kind == "cone"]
    # A hole needs at least one cylindrical wall. A lone coaxial cone with no
    # cylinder is a chamfer, not a hole — leave it for a later pass.
    if not cyls:
        return None, set()

    primary = _primary_cylinder(cyls, occ_faces_by_index)
    axis = primary.axis if primary is not None else cluster[0].axis
    radius = float(primary.radius) if primary and primary.radius else 0.0
    nominal_diameter = 2.0 * radius

    # Diameter CEILING (after the interior test): a large concave region is the
    # central bore / structural pocket, not a drilled hole. Defer its faces to
    # the bore/contour pass (caller keeps them in remaining_faces).
    if nominal_diameter > config.max_hole_diameter_mm:
        logger.info(
            "ceiling: cluster faces %s ⌀%.3f mm (%.3f in) > max_hole_diameter_mm "
            "%.1f; deferred to bore/contour pass (not claimed)",
            sorted(c.face_index for c in cluster), nominal_diameter,
            nominal_diameter / MM_PER_INCH, config.max_hole_diameter_mm,
        )
        return None, set()

    # Wall axial span (this cluster only) gates cap-at-end floor detection.
    wall_face_ids = [c.face_index for c in cyls + cones]
    wall_span = _wall_axial_span(occ_faces_by_index, wall_face_ids, faces, axis)

    floor_ids, blend_ids = _attach_faces(
        cluster, faces, graph, axis, config,
        occ_faces_by_index=occ_faces_by_index, wall_span=wall_span,
        candidate_faces=candidate_faces,
        bore_radius=radius,
        claimed_floor_faces=claimed_floor_faces,
    )

    cluster_ids = {c.face_index for c in cluster}
    body_floors, body_blends, boundary_fillets = _split_hole_stack_tiers(
        cluster_ids, floor_ids, blend_ids, faces, graph,
        occ_faces_by_index, axis, config,
    )
    if boundary_fillets:
        logger.info(
            "stack tier: cluster %s releases opening-tier fillets %s to outer-fillet pass",
            sorted(cluster_ids), sorted(boundary_fillets),
        )

    # Blind ⇔ one end of the wall span is capped. A planar floor gives a
    # filleted/flat-bottom blind ("blind_hole"); a coaxial drill-tip cone with no
    # opening beyond it gives a drilled blind ("drilled_blind_hole"). A wall open
    # at both ends (no qualifying cap) is a through hole.
    if body_floors:
        kind = "blind_hole"
    elif _has_drill_tip_cap(
        cyls, cones, cluster_ids, faces, graph, occ_faces_by_index, axis, config,
    ):
        kind = "drilled_blind_hole"
    else:
        kind = "through_hole"

    n_distinct = _distinct_radii(cyls)
    is_counterbore = n_distinct > 1
    is_countersink = len(cones) > 0

    face_ids: set[int] = cluster_ids | body_floors | body_blends

    depth = None
    if occ_faces_by_index is not None:
        # Depth uses ONLY this cluster's own wall + cap faces, projected onto
        # THIS cluster's axis — never a diameter family. axis.point cancels in
        # the (max - min) range, so it is this instance's axial extent.
        depth_faces = sorted(set(wall_face_ids) | body_floors)
        depth = _cluster_axial_extent(occ_faces_by_index, depth_faces, axis)

    feat = HoleFeature(
        feature_id=feature_id,
        kind=kind,
        face_indices=face_ids,
        axis=axis,
        nominal_diameter=nominal_diameter,
        depth=depth,
        is_counterbore=is_counterbore,
        is_countersink=is_countersink,
        radius=radius,
        cylinder_face_indices=[c.face_index for c in cyls],
        cone_face_indices=[c.face_index for c in cones],
        floor_face_indices=sorted(body_floors),
        blend_face_indices=sorted(body_blends),
        n_distinct_radii=n_distinct,
    )
    return feat, boundary_fillets


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_holes(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    occ_faces: Sequence[Any] | None = None,
    candidate_faces: set[int] | None = None,
    config: HoleDetectionConfig | None = None,
) -> HoleDetectionResult:
    """Run the hole pass.

    Parameters
    ----------
    faces:
        FaceGeom records from feature_params.analyze_step(step_path), in
        TopologyExplorer order (index i == face index i in the graph).
    edge_index, edge_attr:
        Face-adjacency tensors from step_ingest.model_to_pyg / graph.npz, columns
        [concave, convex, smooth] one-hot + cos(dihedral). Adjacency is reused,
        not recomputed.
    occ_faces:
        TopoDS_Face list from feature_params.load_step_faces(step_path), same
        index order. Required for real axis-line colinearity and depth. When
        omitted (self-test), a degraded direction-only path is used.
    candidate_faces:
        The still-unclaimed face indices this pass may consume. In the reordered
        cascade the POCKET pass runs first and its remaining_faces are passed
        here, so the hole pass sees a clean residual (no pocket walls). Default =
        every face (standalone use). This restricts the INPUT source only; the
        coaxial-clustering / floor-attachment LOGIC is unchanged.
    """
    config = config or HoleDetectionConfig()
    n_faces = len(faces)
    pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }

    if occ_faces is not None and len(occ_faces) != n_faces:
        # Index-space mismatch would silently corrupt face ids.
        raise ValueError(
            f"occ_faces ({len(occ_faces)}) and faces ({n_faces}) differ; "
            "they must both be TopologyExplorer order for the same shape"
        )
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)
    if graph.n_faces != n_faces:
        logger.warning(
            "face graph built for %d faces but %d FaceGeom records supplied; "
            "check zero-area filtering did not shift indices",
            graph.n_faces, n_faces,
        )

    result_bins: dict[str, Any] = {
        "skipped_bspline_faces": [],
        "rejected_convex_faces": [],
        "oversize_faces": [],
        "degenerate_faces": [],
        "candidate_diag": [],
    }
    candidates = _collect_candidates(faces, occ_faces, config, result_bins, pool)
    _log_interior_families(result_bins["candidate_diag"], config)
    _log_degenerate_families(result_bins["candidate_diag"])
    clusters = _cluster_candidates(candidates, graph, config)

    occ_map = None
    if occ_faces is not None:
        occ_map = {i: occ_faces[i] for i in range(n_faces)}

    features: list[HoleFeature] = []
    deferred_feature_fillets: set[int] = set()
    deferred_feature_fillet_groups: list[set[int]] = []
    claimed_floor_faces: set[int] = set()
    next_id = 0
    for cluster in clusters:
        feat, boundary = _classify_cluster(
            next_id, cluster, faces, graph, occ_map, config, pool,
            claimed_floor_faces=claimed_floor_faces,
        )
        if feat is None:
            if all(c.kind == "cone" for c in cluster):
                pass  # cone-only cluster -> chamfer, handled elsewhere
            else:
                # Oversize (guarded) clusters: keep faces in the pool.
                result_bins["oversize_faces"].extend(c.face_index for c in cluster)
            continue
        if boundary:
            deferred_feature_fillet_groups.append(boundary)
            deferred_feature_fillets |= boundary
        features.append(feat)
        claimed_floor_faces |= set(feat.floor_face_indices)
        next_id += 1

    _log_depth_debug(features, config)

    claimed: set[int] = set()
    for f in features:
        claimed |= f.face_indices
    remaining = pool - claimed

    if result_bins["skipped_bspline_faces"]:
        logger.info(
            "skipped %d bspline/bezier candidate faces (deferred to later pass): %s",
            len(result_bins["skipped_bspline_faces"]), result_bins["skipped_bspline_faces"],
        )
    if config.require_interior_wall and result_bins["rejected_convex_faces"]:
        logger.info(
            "rejected %d convex (outer/boss) cylinder/cone faces",
            len(result_bins["rejected_convex_faces"]),
        )
    if result_bins["degenerate_faces"]:
        logger.info(
            "skipped %d degenerate-radius cylinder faces: %s",
            len(result_bins["degenerate_faces"]), result_bins["degenerate_faces"],
        )
    if result_bins["oversize_faces"]:
        logger.info(
            "ceiling deferred %d face(s) above ⌀%.1f mm to bore/contour pass",
            len(result_bins["oversize_faces"]), config.max_hole_diameter_mm,
        )

    return HoleDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        deferred_feature_fillets=deferred_feature_fillets,
        deferred_feature_fillet_groups=deferred_feature_fillet_groups,
        skipped_bspline_faces=result_bins["skipped_bspline_faces"],
        rejected_convex_faces=result_bins["rejected_convex_faces"],
        oversize_faces=result_bins["oversize_faces"],
        degenerate_faces=result_bins["degenerate_faces"],
        candidate_diag=result_bins["candidate_diag"],
        units=config.units,
    )


def detect_holes_from_step(
    step_path: str | Path,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    config: HoleDetectionConfig | None = None,
) -> HoleDetectionResult:
    """Convenience wrapper: load FaceGeom + OCC faces from a STEP, then detect."""
    # Wired to feature_params: analyze_step -> FaceGeom, load_step_faces -> OCC.
    from brep.feature_params import analyze_step, load_step_faces, require_occ

    require_occ()
    faces = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    return detect_holes(
        faces, edge_index, edge_attr, occ_faces=occ_faces,
        candidate_faces=candidate_faces, config=config,
    )


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
def detect_step_units(step_path: str | Path) -> dict[str, Any]:
    """Sniff the STEP header for the authored length unit (for logging only).

    OCCT converts everything to mm on read, so this just records what the file
    *declared* (e.g. INCH via CONVERSION_BASED_UNIT). Returns the declared unit
    name and its mm factor when found.
    """
    text = Path(step_path).read_text(errors="ignore")
    m = re.search(
        r"CONVERSION_BASED_UNIT\(\s*'([^']+)'", text, flags=re.IGNORECASE,
    )
    declared = m.group(1).upper() if m else None
    factor = None
    if declared:
        fm = re.search(r"LENGTH_MEASURE\(([-0-9.eE+]+)\)", text)
        if fm:
            factor = float(fm.group(1))
    return {
        "declared_unit": declared,
        "declared_mm_factor": factor,
        "occ_working_units": "mm",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ExpectedHole:
    kind: str  # "through_hole" | "blind_hole"
    diameter: float  # in `units`
    units: str = "mm"
    tol: float = 0.75  # matching tolerance in mm

    def diameter_mm(self) -> float:
        return self.diameter * (MM_PER_INCH if self.units == "inch" else 1.0)


@dataclass
class ValidationReport:
    ok: bool
    matches: list[tuple[ExpectedHole, HoleFeature]]
    missing: list[ExpectedHole]
    n_detected: int
    n_extra: int

    def render(self) -> str:
        lines = [f"validation: {'PASS' if self.ok else 'FAIL'}"]
        for exp, feat in self.matches:
            lines.append(
                f"  matched  {exp.kind:<12} ⌀{exp.diameter_mm():.3f} mm "
                f"-> feature {feat.feature_id} ⌀{feat.nominal_diameter:.3f} mm "
                f"({len(feat.face_indices)} faces)"
            )
        for exp in self.missing:
            lines.append(
                f"  MISSING  {exp.kind:<12} ⌀{exp.diameter_mm():.3f} mm "
                f"(±{exp.tol} mm) — no detected hole matched"
            )
        lines.append(
            f"  detected {self.n_detected} holes total "
            f"({self.n_extra} beyond the expected set)"
        )
        return "\n".join(lines)


def validate_against_expected(
    features: Sequence[HoleFeature],
    expected: Sequence[ExpectedHole],
    *,
    exact: bool = False,
) -> ValidationReport:
    """Match expected holes against detected `features`.

    A detected feature matches an expected hole when the kind agrees and the
    nominal diameter is within `tol` (mm). Greedy best-diameter assignment.

    Two modes:
      * subset (default): every expected hole must be present; extra detected
        holes are reported but NOT an error. Used when the hole pass runs on the
        FULL part and intentionally over-detects (e.g. pocket walls).
      * exact (exact=True): additionally require that there are NO extras — the
        detected set must be EXACTLY the expected set. Used in the reordered
        cascade, where the pocket pass has already claimed the pocket walls so
        the residual must yield precisely the real holes (⌀101.752 blind,
        ⌀81.28 through) and nothing else.
    """
    remaining = list(features)
    matches: list[tuple[ExpectedHole, HoleFeature]] = []
    missing: list[ExpectedHole] = []

    for exp in expected:
        target = exp.diameter_mm()
        best = None
        best_err = exp.tol
        for feat in remaining:
            if feat.kind != exp.kind:
                continue
            err = abs(feat.nominal_diameter - target)
            if err <= best_err:
                best, best_err = feat, err
        if best is None:
            missing.append(exp)
        else:
            matches.append((exp, best))
            remaining.remove(best)

    n_extra = max(0, len(features) - len(matches))
    ok = (not missing) and (not exact or n_extra == 0)
    return ValidationReport(
        ok=ok,
        matches=matches,
        missing=missing,
        n_detected=len(features),
        n_extra=n_extra,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_table(result: HoleDetectionResult) -> str:
    header = (
        f"{'id':>3}  {'kind':<12} {'⌀ (mm)':>10} {'⌀ (in)':>8} {'#f':>3} "
        f"{'cbore':>5} {'csink':>5} {'depth':>9}  axis_dir"
    )
    lines = [header, "-" * len(header)]
    for f in sorted(result.features, key=lambda x: x.nominal_diameter, reverse=True):
        d = f.axis.direction
        depth = "-" if f.depth is None else f"{f.depth:.3f}"
        lines.append(
            f"{f.feature_id:>3}  {f.kind:<12} {f.nominal_diameter:>10.3f} "
            f"{f.nominal_diameter / MM_PER_INCH:>8.3f} {len(f.face_indices):>3} "
            f"{'Y' if f.is_counterbore else '·':>5} "
            f"{'Y' if f.is_countersink else '·':>5} {depth:>9}  "
            f"[{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}]"
        )
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (no OCC required) — validates the pure-Python core logic
# ---------------------------------------------------------------------------
@dataclass
class _FakeFace:
    """Minimal FaceGeom stand-in for the OCC-free self-test."""

    index: int
    surface_type: str
    centroid: np.ndarray
    normal: np.ndarray
    radius: float | None = None
    axis: np.ndarray | None = None
    axis_location: np.ndarray | None = None
    area: float | None = None


def _fake_cylinder(index, radius, axis_dir, center, *, interior=True):
    axis_dir = _unit(np.asarray(axis_dir, dtype=np.float64))
    center = np.asarray(center, dtype=np.float64)
    # place the centroid off-axis, with an outward normal pointing toward
    # (interior/hole) or away from (boss) the axis.
    perp = _unit(np.cross(axis_dir, [1.0, 0.0, 0.0]))
    if float(np.linalg.norm(perp)) < 1e-6:
        perp = _unit(np.cross(axis_dir, [0.0, 1.0, 0.0]))
    centroid = center + perp * radius
    normal = -perp if interior else perp
    return _FakeFace(index, "cylinder", centroid, normal, radius=radius,
                     axis=axis_dir, axis_location=center)


def _edges_from_pairs(pairs: list[tuple[int, int, str]], n: int) -> tuple[np.ndarray, np.ndarray]:
    cmap = {"concave": 0, "convex": 1, "smooth": 2}
    src, dst, attr = [], [], []
    for u, v, kind in pairs:
        oh = [0.0, 0.0, 0.0]
        oh[cmap[kind]] = 1.0
        row = oh + [1.0 if kind == "smooth" else 0.0]
        for a, b in ((u, v), (v, u)):
            src.append(a)
            dst.append(b)
            attr.append(row)
    ei = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)
    ea = np.array(attr, dtype=np.float32) if attr else np.zeros((0, 4), np.float32)
    return ei, ea


def _run_selftest() -> bool:
    z = [0.0, 0.0, 1.0]
    x = [1.0, 0.0, 0.0]
    passed = True

    def check(name: str, cond: bool) -> None:
        nonlocal passed
        passed = passed and cond
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # Case 1: counterbore = two coaxial cylinders (different radii) + shoulder
    # plane, connected by a convex shoulder edge -> ONE through hole, cbore flag.
    faces = [
        _fake_cylinder(0, 4.0, z, [0, 0, 0]),
        _fake_cylinder(1, 8.0, z, [0, 0, 0]),   # coaxial, bigger radius
        _FakeFace(2, "plane", np.array([6.0, 0, 5.0]), np.array(z, float)),  # shoulder
    ]
    ei, ea = _edges_from_pairs([(0, 1, "convex"), (1, 2, "convex"), (0, 2, "convex")], 3)
    r = detect_holes(faces, ei, ea)
    check("counterbore groups 2 cylinders into 1 feature", len(r.features) == 1)
    check("counterbore flagged is_counterbore", r.features and r.features[0].is_counterbore)

    # Case 2: partial-arc wall = two coaxial half-cylinders same radius -> one hole.
    faces = [
        _fake_cylinder(0, 5.0, z, [0, 0, 0]),
        _fake_cylinder(1, 5.0, z, [1e-4, 0, 0]),  # same axis within floor tol
    ]
    ei, ea = _edges_from_pairs([(0, 1, "smooth")], 2)
    r = detect_holes(faces, ei, ea)
    check("partial arcs merge into 1 through hole", len(r.features) == 1 and r.features[0].kind == "through_hole")

    # Case 3: blind hole = cylinder + perpendicular floor plane -> blind.
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),
        _FakeFace(1, "plane", np.array([0, 0, -5.0]), np.array(z, float)),  # floor ⟂ axis
    ]
    ei, ea = _edges_from_pairs([(0, 1, "concave")], 2)
    r = detect_holes(faces, ei, ea)
    check("cylinder + perpendicular floor -> blind hole", len(r.features) == 1 and r.features[0].kind == "blind_hole")

    # Case 4: filleted blind hole = cylinder + torus fillet (smooth) + floor.
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),
        _FakeFace(1, "torus", np.array([2.0, 0, -4.0]), np.array(z, float)),  # fillet
        _FakeFace(2, "plane", np.array([0, 0, -5.0]), np.array(z, float)),   # floor
    ]
    ei, ea = _edges_from_pairs([(0, 1, "smooth"), (1, 2, "smooth")], 3)
    r = detect_holes(faces, ei, ea)
    ok4 = len(r.features) == 1 and r.features[0].kind == "blind_hole" and 1 in r.features[0].face_indices
    check("filleted blind hole attaches torus fillet + floor", ok4)

    # Case 5: countersink = cylinder + coaxial cone -> one hole, csink flag.
    # cone centroid off +x axis, interior normal points back toward axis (-x).
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),
        _FakeFace(1, "cone", np.array([4.0, 0, 5.0]), np.array([-1.0, 0.0, 0.0]),
                  radius=6.0, axis=np.array(z, float),
                  axis_location=np.array([0.0, 0.0, 0.0])),
    ]
    ei, ea = _edges_from_pairs([(0, 1, "convex")], 2)
    r = detect_holes(faces, ei, ea)
    check("countersink (cylinder + coaxial cone) flagged is_countersink",
          len(r.features) == 1 and r.features[0].is_countersink)

    # Case 6: outer boss (convex cylinder) rejected as a hole wall.
    faces = [_fake_cylinder(0, 10.0, z, [0, 0, 0], interior=False)]
    ei, ea = _edges_from_pairs([], 1)
    r = detect_holes(faces, ei, ea)
    check("convex boss cylinder rejected (0 holes)", len(r.features) == 0)

    # Case 7: two DIFFERENT axes stay separate (colinearity guard).
    faces = [
        _fake_cylinder(0, 4.0, z, [0, 0, 0]),
        _fake_cylinder(1, 4.0, z, [50.0, 0, 0]),  # parallel but far off-axis
    ]
    ei, ea = _edges_from_pairs([], 2)
    r = detect_holes(faces, ei, ea)
    check("non-colinear parallel cylinders -> 2 separate holes", len(r.features) == 2)

    # Case 7b (BUG 1 regression): two CONCENTRIC cylinders on the SAME axis but
    # NOT connected in the face graph are DIFFERENT holes, not one blob.
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),    # inner, ⌀6
        _fake_cylinder(1, 6.0, z, [0, 0, 0]),    # outer/concentric, ⌀12, no shared edge
    ]
    ei, ea = _edges_from_pairs([], 2)  # coaxial but graph-disconnected
    r = detect_holes(faces, ei, ea)
    check("coaxial but graph-disconnected cylinders -> 2 separate holes",
          len(r.features) == 2)

    # Case 7c: same two coaxial cylinders, now joined by a SMOOTH blend chain
    # (cylinder—torus—cylinder) -> ONE counterbore feature.
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),
        _FakeFace(1, "torus", np.array([4.5, 0, 2.0]), np.array([-1.0, 0.0, 0.0]),
                  axis=np.array(z, float)),
        _fake_cylinder(2, 6.0, z, [0, 0, 0]),
    ]
    ei, ea = _edges_from_pairs([(0, 1, "smooth"), (1, 2, "smooth")], 3)
    r = detect_holes(faces, ei, ea)
    check("coaxial cylinders joined by smooth blend -> 1 counterbore",
          len(r.features) == 1 and r.features[0].is_counterbore)

    # Case 8: bspline candidate skipped and logged.
    faces = [_FakeFace(0, "bspline", np.array([1.0, 0, 0]), np.array(x, float), radius=3.0)]
    ei, ea = _edges_from_pairs([], 1)
    r = detect_holes(faces, ei, ea)
    check("bspline face skipped (deferred to later pass)", len(r.features) == 0 and r.skipped_bspline_faces == [0])

    # Case 9 (BUG A): through hole whose wall meets top/bottom planes via CONVEX
    # rim edges must stay THROUGH (rims are openings, not floors) and not attach
    # those planes.
    faces = [
        _fake_cylinder(0, 4.0, z, [0, 0, 0]),
        _FakeFace(1, "plane", np.array([0, 0, 5.0]), np.array(z, float)),   # top opening
        _FakeFace(2, "plane", np.array([0, 0, -5.0]), np.array(z, float)),  # bottom opening
    ]
    ei, ea = _edges_from_pairs([(0, 1, "convex"), (0, 2, "convex")], 3)
    r = detect_holes(faces, ei, ea)
    ok9 = (len(r.features) == 1 and r.features[0].kind == "through_hole"
           and r.features[0].face_indices == {0})
    check("convex-rim opening planes -> through hole, planes not attached", ok9)

    # Case 10 (BUG C): a large concave bore above the diameter ceiling is deferred
    # to remaining_faces, not claimed as a hole.
    faces = [_fake_cylinder(0, 100.0, z, [0, 0, 0])]  # ⌀200 mm > 150 ceiling
    ei, ea = _edges_from_pairs([], 1)
    r = detect_holes(faces, ei, ea, config=HoleDetectionConfig(max_hole_diameter_mm=150.0))
    ok10 = len(r.features) == 0 and 0 in r.remaining_faces and 0 in r.oversize_faces
    check("bore above diameter ceiling deferred to remaining_faces", ok10)

    # Case 11: a large shared deck plane concave-adjacent to two shallow holes
    # must NOT be annexed as a floor (geometric mismatch + single ownership).
    deck_area = 50.0 * 50.0  # >> 9 * pi * R^2 for R=3
    faces = [
        _fake_cylinder(0, 3.0, z, [10.0, 0.0, 0.0]),
        _fake_cylinder(1, 3.0, z, [40.0, 0.0, 0.0]),
        _FakeFace(2, "plane", np.array([25.0, 0.0, -2.0]), np.array(z, float),
                  area=deck_area),
    ]
    ei, ea = _edges_from_pairs([(0, 2, "concave"), (1, 2, "concave")], 3)
    r = detect_holes(faces, ei, ea)
    ok11 = (
        len(r.features) == 2
        and all(f.kind == "through_hole" for f in r.features)
        and 2 not in r.claimed_faces
        and all(2 not in f.floor_face_indices for f in r.features)
    )
    check("shared deck plane not annexed as blind-hole floor", ok11)

    # Case 12 (part1 target): a coaxial drill-tip cone that extends the bore past
    # the cylinder wall and is buried (no plane neighbour, no opening beyond it)
    # caps a blind DRILLED hole even with NO planar floor.
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),                              # wall, axial ~0
        _FakeFace(1, "plane", np.array([0, 0, 5.0]), np.array(z, float)),  # single mouth (opening)
        _FakeFace(2, "cone", np.array([1.5, 0, -6.0]), np.array([-1.0, 0.0, 0.0]),
                  radius=3.0, axis=np.array(z, float),
                  axis_location=np.array([0.0, 0.0, 0.0])),                # drill tip beyond bottom
    ]
    ei, ea = _edges_from_pairs([(0, 1, "convex"), (0, 2, "concave")], 3)
    r = detect_holes(faces, ei, ea)
    ok12 = (len(r.features) == 1 and r.features[0].kind == "drilled_blind_hole"
            and r.features[0].face_indices == {0, 2})
    check("coaxial drill-tip cone (no floor) -> drilled_blind_hole", ok12)

    # Case 13 (fish_mold guard): a countersink cone that FLARES into an opening
    # plane at each end keeps the hole THROUGH — the cone is a chamfer, not a cap.
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),
        _FakeFace(1, "plane", np.array([0, 0, 5.0]), np.array(z, float)),    # top opening
        _FakeFace(2, "plane", np.array([0, 0, -5.0]), np.array(z, float)),   # bottom opening
        _FakeFace(3, "cone", np.array([4.0, 0, 4.5]), np.array([-1.0, 0.0, 0.0]),
                  radius=3.0, axis=np.array(z, float),
                  axis_location=np.array([0.0, 0.0, 0.0])),                  # countersink at top plane
    ]
    ei, ea = _edges_from_pairs(
        [(0, 1, "convex"), (0, 2, "convex"), (0, 3, "concave"), (3, 1, "convex")], 4,
    )
    r = detect_holes(faces, ei, ea)
    ok13 = (len(r.features) == 1 and r.features[0].kind == "through_hole"
            and r.features[0].is_countersink)
    check("countersink cone flaring into opening plane -> stays through_hole", ok13)

    # Case 14 (golden guard): a plain through hole with NO cone stays through and
    # yields no drilled-blind (empty on parts lacking the target geometry).
    faces = [
        _fake_cylinder(0, 4.0, z, [0, 0, 0]),
        _FakeFace(1, "plane", np.array([0, 0, 5.0]), np.array(z, float)),
        _FakeFace(2, "plane", np.array([0, 0, -5.0]), np.array(z, float)),
    ]
    ei, ea = _edges_from_pairs([(0, 1, "convex"), (0, 2, "convex")], 3)
    r = detect_holes(faces, ei, ea)
    ok14 = (len(r.features) == 1 and r.features[0].kind == "through_hole"
            and not any(f.kind == "drilled_blind_hole" for f in r.features))
    check("no-cone through hole -> no drilled_blind_hole (golden guard)", ok14)

    # Validation subset check.
    exp = [ExpectedHole("blind_hole", 6.0), ExpectedHole("through_hole", 8.0)]
    faces = [
        _fake_cylinder(0, 3.0, z, [0, 0, 0]),
        _FakeFace(1, "plane", np.array([0, 0, -5.0]), np.array(z, float)),
        _fake_cylinder(2, 4.0, z, [100.0, 0, 0]),
    ]
    ei, ea = _edges_from_pairs([(0, 1, "concave")], 3)
    r = detect_holes(faces, ei, ea)
    report = validate_against_expected(r.features, exp)
    check("validate_against_expected subset match passes", report.ok)

    print(f"\nself-test: {'ALL PASS' if passed else 'FAILURES PRESENT'}")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_STEP = "fixtures/step/96260B_rear.stp"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"


def _load_edges(graph_npz: Path | None, step_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the face graph from a cached npz, else recompute via step_ingest."""
    if graph_npz is not None and graph_npz.is_file():
        d = np.load(graph_npz)
        return d["edge_index"], d["edge_attr"]
    # Wired to step_ingest.ingest_step_to_pyg (same tensors the GNN consumes).
    from brep.step_ingest import ingest_step_to_pyg

    _x, edge_index, edge_attr, _stats = ingest_step_to_pyg(str(step_path))
    return edge_index, edge_attr


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hole-detection pass (cascade stage 1)")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP, help="reference STEP file")
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ),
                    help="cached face graph (edge_index/edge_attr); recomputed if absent")
    ap.add_argument("--max-diameter", type=float, default=150.0,
                    help="diameter ceiling (mm): clusters above this are deferred to "
                         "the bore/contour pass, not claimed as holes (default 150 ≈ 6in)")
    ap.add_argument("--warn-diameter", type=float, default=None,
                    help="log candidate cylinders at/above this diameter (mm)")
    ap.add_argument("--no-interior-filter", action="store_true",
                    help="cluster every coaxial cylinder, incl. convex bosses")
    ap.add_argument("--debug-depth-diameter", type=float, default=None,
                    help="dump per-cluster face-ids + depths for this diameter family (mm)")
    ap.add_argument("--selftest", action="store_true",
                    help="run OCC-free logic self-test and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.selftest:
        return 0 if _run_selftest() else 1

    step_path = Path(args.step)
    if not step_path.is_file():
        print(f"STEP not found: {step_path}")
        return 2

    units = detect_step_units(step_path)
    print(f"STEP: {step_path}")
    print(
        f"units: declared={units['declared_unit']} "
        f"(mm factor {units['declared_mm_factor']}), "
        f"OCC working units = mm  -> all diameters below are mm"
    )

    config = HoleDetectionConfig(
        max_hole_diameter_mm=args.max_diameter,
        warn_diameter=args.warn_diameter,
        require_interior_wall=not args.no_interior_filter,
        debug_depth_diameter=args.debug_depth_diameter,
    )
    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    result = detect_holes_from_step(step_path, edge_index, edge_attr, config=config)

    print()
    print(render_table(result))

    # Reference part ground truth (Toolpath): 1 filleted blind ⌀4.006in,
    # 1 through ⌀3.2in. Subset match (this pass may over-detect other cylinders).
    expected = [
        ExpectedHole("blind_hole", 4.006, units="inch", tol=0.75),
        ExpectedHole("through_hole", 3.2, units="inch", tol=0.75),
    ]
    print()
    report = validate_against_expected(result.features, expected)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
