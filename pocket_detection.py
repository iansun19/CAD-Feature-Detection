"""
pocket_detection.py — Cascade stage: SCULPTED-POCKET DETECTION (spatial).

Where this fits
---------------
The CAM feature-grouping cascade groups the ~348 granular B-rep faces into
machining-feature INSTANCES with a sequence of geometric passes that share one
pool of unclaimed faces. The cascade principle is *specific-first*: the more
specific recognizer runs first and claims its faces; the generic one takes the
residual.

A sculpted pocket (a family of walls + bspline floors + spheres + a torus/cone
blend ring, all spatially co-located) is MORE SPECIFIC than "a coaxial concave
cylinder = a hole". So pockets run BEFORE holes:

    pockets  ->  holes  ->  flats  ->  contour  ->  outer fillets

This module implements the pocket pass. It CLAIMS each pocket's filleted region:
interior walls + step planes + cluster-assigned sculpted floors (bspline/bezier/
sphere seeds from spatial clustering). Only sculpted faces that are NOT assigned
to any pocket cluster, plus non-wall-adjacent torus, are released to contour.

Why this ordering matters (the bug it fixes)
---------------------------------------------
Geometrically a pocket wall is a concave cylinder, so a hole recognizer grabs it
as a false-positive "hole". On the reference plate the hole pass alone reports
65 holes — 56 pocket walls + 7 per-pocket ⌀2.867in bores + the 2 genuine central
holes. Running pockets first removes those 63 pocket faces from hole candidacy so
the hole pass sees a clean residual and reports exactly the 2 real holes.

How a pocket is grouped (settled by four diagnostic probes — do NOT re-litigate)
--------------------------------------------------------------------------------
* A pocket is a SPATIAL cluster, not a coaxial one. Group by generic Euclidean
  proximity of face centroids in the plane perpendicular to the opening axis.
* Distance clustering == angular clustering (100% agreement, 0/147 faces differ)
  on the reference part -> use generic proximity, NO ring/polar assumption.
* The floors/spheres/blend do NOT graph-connect within a pocket (the whole blend
  structure is ONE 205-face component across all pockets), so grouping MUST be
  spatial, never edge-walking.

Wall attachment (spatial, principled — not a hardcoded diameter list)
---------------------------------------------------------------------
A pocket WALL is an INTERIOR (concave) cylinder/cone whose projected centroid
lies close to that pocket's floor signature. On the reference part every wall
family (⌀0.800/0.500/3.453/2.867 in) sits within ~23 mm of a pocket floor, while
the two genuine central holes (⌀4.006 blind, ⌀3.200 through) sit >=37 mm from any
floor — a clean separating gap. Convex cylinders (the ⌀6.370in stray bosses at
faces 328/346) are rejected by the interior test, never claimed. So walls are
gated by (interior sign < 0) AND (near a pocket floor), with the diameter
families used only as a reference template for reporting/validation.

NOTE on the "21-face template": the earlier probes restricted their pool to the
three wall families ⌀0.800/0.500/3.453 and reported a uniform 21-face template.
That pool OMITTED the 7th per-pocket concave cylinder (the ⌀2.867in bore at faces
116/148/178/208/238/268/296). Left unclaimed, those 7 concave bores leak into the
hole pass as false-positive holes (9 holes instead of 2). They are genuine pocket
faces (one per pocket, ~23 mm from the pocket floor, Toolpath counts them as
pocket not hole), so this pass claims them too.

Durable contract (do NOT gate on face count alone)
--------------------------------------------------
The reference rear plate's filleted pocket template is ~24 claimed faces per pocket:
{⌀0.800×4, ⌀0.500×2, ⌀3.453×2, ⌀2.867×1} walls=9 + 2 step planes + 6 bspline
+ 7 sphere cluster-assigned floor seeds. The open front plate has fewer seeds
(4+5) and 1 step band but the same structural contract.
The generalizable contract is the STRUCTURAL gate: interior/concave +
near-or-adjacent-to-pocket + wall/step-sized, grouped by (X,Z) proximity +
adjacency grow from already-claimed faces. A different part will have a different
count; validation on other parts must check structure, not ==11.

Integration points (reused, verified against this repo)
-------------------------------------------------------
* Per-face geometry ...... feature_params.analyze_step()  -> list[FaceGeom]
* OCC faces (interior test) feature_params.load_step_faces() -> [TopoDS_Face]
* Interior-sign + axis .... hole_detection._interior_sign_occ / _axis_from_occ_face
* Union-find .............. hole_detection._UnionFind (shared, not re-implemented)

Units are mm (OCCT normalises STEP geometry to mm); ⌀in = 2r/25.4.
No dependency beyond numpy + the existing pythonocc usage. No sklearn.
"""
from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

# Reuse hole_detection's shared primitives (union-find + interior/axis reads) so
# the two passes stay consistent and the union-find is not re-implemented.
from hole_detection import (
    FaceGraph,
    _UnionFind,
    _axis_from_occ_face,
    _interior_sign_occ,
    _interior_wall_sign,
    _unit,
    induced_subgraph_components,
)

logger = logging.getLogger("pocket_detection")

MM_PER_INCH = 25.4

# Pocket classes that must be one B-rep-connected component at emit time.
POCKET_CONNECTIVITY_CLASSES = frozenset({
    "filleted_pocket",
    "filleted_open_pocket",
    "open_pocket",
})


class PocketConnectivityError(ValueError):
    """Emitted pocket feature spans multiple B-rep-connected components."""


# Surface types that form the reliable pocket "floor" signature (the spatial
# anchor of a pocket cluster). Spheres are the ball-nose floor blends.
# Spatial cluster seeds; cluster-assigned seeds are retained in the pocket claim.
FLOOR_TYPES = ("bspline", "bezier")
SPHERE_TYPES = ("sphere",)
SCULPTED_FLOOR_TYPES = FLOOR_TYPES + SPHERE_TYPES
# Cylinder/cone are wall candidates; everything else (torus/plane/large blend
# cylinders/near-apex cones) is blend ring — characterised, never claimed.
WALL_SURF_TYPES = ("cylinder", "cone")
FILLET_TYPES = frozenset({"torus"})
BLEND_TRAVERSABLE = frozenset({"torus", "sphere", "bspline", "bezier"})
CONTOUR_TYPES = frozenset({"plane", "cylinder", "cone"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PocketSetupConfig:
    """Setup-scoped filleted-pocket access (open vs closed).

    Open/closed cannot be recovered from single-STEP lobe geometry on split plate
    panels — see ``eval/pocket_openclosed_findings.md``.  Supply ``machining_side``
    or ``pocket_access`` explicitly per run / GT yaml.

    Resolution order:
      1. ``pocket_access`` if set (``"open"`` | ``"closed"``)
      2. ``machining_side``: ``front`` → open, ``back`` → closed
      3. Fallback: ``closed`` (emit ``filleted_pocket``) with a warning — never
         infer access from geometry.
    """

    machining_side: Literal["front", "back"] | None = None
    pocket_access: Literal["open", "closed"] | None = None

    def resolved_access(self) -> tuple[str, bool]:
        """Return ``(access, from_explicit_setup)``."""
        if self.pocket_access is not None:
            return self.pocket_access, True
        if self.machining_side == "front":
            return "open", True
        if self.machining_side == "back":
            return "closed", True
        return "closed", False


@dataclass
class PocketDetectionConfig:
    """Tolerances and guards. Linear tolerances are in model units (mm)."""

    # Opening axis: None -> auto-detect from the shared wall-axis direction; else
    # an explicit unit 3-vector (e.g. (0,1,0)).
    opening_axis: tuple[float, float, float] | None = None
    # Auto-detect fallback guard. The mean-of-interior-wall-axes estimator is only
    # trustworthy when the walls actually share a direction. On freeform cavities
    # (fish_mold) the walls scatter and their mean drifts off-cardinal. When the
    # wall-axis coherence (mean-resultant length after sign-folding, 1.0 = perfectly
    # aligned) drops below this AND the mean disagrees with the broad-face estimator
    # on the dominant cardinal, fall back to the broad-face axis (largest total
    # planar-face area by cardinal — the physical plate/mold opening). Threshold sits
    # in the measured gap between fish_mold (R=0.61, needs fallback) and the
    # prismatic parts that must stay on the mean (part1 R=0.83; 96260B/part2-4 R=1.0).
    opening_axis_coherence_min: float = 0.72
    # Snap the derived opening axis to the nearest cardinal only when within this
    # angular tolerance; otherwise keep the derived vector. A no-op on the current
    # corpus (coherent parts are already exact cardinals; the broad-face fallback is
    # exactly cardinal) — present to clean sub-tolerance mesh noise.
    opening_axis_snap_tol_deg: float = 5.0

    # --- pocket grouping ---
    # "region_grow": B-rep region grow from floor seeds through walls/fillets/step
    # planes via smooth/concave edges only; gateway trim on open pockets.
    # "spatial": legacy single-linkage floor-seed clustering + spatial wall attach.
    # Default stays spatial until region_grow passes the rear-plate regression gate.
    floor_seed_grouping: Literal["region_grow", "spatial"] = "spatial"

    # --- proximity clustering of floor seeds -> pockets (spatial path only) ---
    # Knee detection on the single-linkage merge heights: trust the knee only if
    # the height ratio (break-K / within-K) exceeds this.
    min_knee_ratio: float = 1.3
    # Fallback absolute merge distance when no knee clears min_knee_ratio, tied
    # to feature scale: fallback_dist = k_scale * median floor-cluster diameter.
    k_scale: float = 1.5
    # Optional hard overrides (skip knee logic entirely).
    hard_k: int | None = None
    hard_threshold: float | None = None
    # Only consider cluster counts up to this many when searching for the knee.
    max_clusters: int = 32

    # --- wall attachment ---
    # A wall is an interior cyl/cone whose projected centroid lies within this
    # distance of a pocket-cluster floor. None -> adaptive: attach_frac * (min
    # inter-pocket centroid distance), which lands cleanly in the wall/central-
    # hole gap on the reference part.
    wall_attach_dist: float | None = None
    attach_frac: float = 0.5
    # Require walls to be interior (concave). Rejects convex outer bosses
    # (e.g. the ⌀6.370in faces 328/346). Set False to attach any near cyl/cone.
    require_interior_wall: bool = True
    # Wall-diameter band (mm): a wall is a wall-sized concave cylinder/cone. This
    # separates true walls from the pocket BLEND RING, which is characteristically
    # either very large-radius (gentle fillet/blend cylinders ⌀5.060/11.040/20.700
    # in on the ref part, i.e. >= ~128 mm) or near-apex cones (RefRadius ~0). The
    # reference walls span ⌀12.7–87.7 mm, with the next larger concave surface at
    # ⌀128.5 mm — a clean gap. Together with the interior + proximity gates this
    # keeps diameter from being the SOLE gate. Set either bound to None to disable.
    wall_dia_min_mm: float | None = 3.0
    wall_dia_max_mm: float | None = 100.0

    # --- reference template (for reporting + validation only, NOT a gate) ---
    # Wall diameter families (inches) and their per-pocket counts on the ref part.
    wall_families_in: tuple[tuple[float, int], ...] = (
        (0.800, 4), (0.500, 2), (3.453, 2), (2.867, 1),
    )
    dia_tol_in: float = 0.02
    template_floor_count: int = 6
    template_sphere_count: int = 7
    # Reference-part validation only — NOT a claim gate (see module docstring).
    template_step_plane_count: int = 2
    # Expected claimed faces per pocket on the reference part (analytic only).
    template_analytic_face_count: int = 11

    # --- membership grow (step planes + any remaining pocket-interior faces) ---
    # A grow-candidate plane must have its normal ~parallel to the opening axis
    # (step floors/steps in a Y-opening pocket). Rejects central-stack end caps
    # whose normals are perpendicular to the opening axis (e.g. ±X caps on the ref
    # plate) without hardcoding face indices.
    step_plane_normal_tol_deg: float = 10.0

    units: str = "mm"

    def wall_attach_or_default(self, min_inter_centroid: float) -> float:
        if self.wall_attach_dist is not None:
            return self.wall_attach_dist
        return self.attach_frac * min_inter_centroid

    setup: PocketSetupConfig = field(default_factory=PocketSetupConfig)


# ---------------------------------------------------------------------------
# Geometry containers
# ---------------------------------------------------------------------------
@dataclass
class PocketFeature:
    feature_id: int
    kind: str  # always "pocket"
    subtype: str  # "through_pocket" | "blind_pocket"
    face_indices: set[int]  # CLAIMED analytic set (walls + step planes)
    centroid_uv: tuple[float, float]  # (u,v) in the plane perpendicular to opening
    opening_axis: tuple[float, float, float]
    wall_diameters: dict[str, int]  # "⌀0.800in" -> count
    wall_count: int
    floor_count: int  # sculpted floors in cluster (released, not claimed)
    sphere_count: int  # sphere blends in cluster (released, not claimed)
    step_plane_count: int
    released_faces: list[int]  # bspline/bezier/sphere (+ optional torus) shed to contour
    released_by_type: dict[str, list[int]]
    depth_below_top_mm: float | None
    fillet_radius_mm: float | None
    surface_3d: bool  # False for analytic pockets (Toolpath "3D Surface: No")
    blend_ring: dict[str, int]  # surface-type composition of nearby unclaimed faces
    blend_face_indices: list[int]
    template_match: bool
    template_deviation: dict[str, Any] = field(default_factory=dict)
    access: str = "closed"  # "open" | "closed"
    inboard_wall_index: int | None = None
    gateway_face_indices: list[int] = field(default_factory=list)
    cap_face_indices: list[int] = field(default_factory=list)
    toolpath_class: str = "filleted_pocket"

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "subtype": self.subtype,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "centroid_uv": [round(float(x), 4) for x in self.centroid_uv],
            "opening_axis": [round(float(x), 4) for x in self.opening_axis],
            "wall_diameters": self.wall_diameters,
            "wall_count": self.wall_count,
            "floor_count": self.floor_count,
            "sphere_count": self.sphere_count,
            "step_plane_count": self.step_plane_count,
            "released_faces": sorted(self.released_faces),
            "released_by_type": {
                k: sorted(v) for k, v in sorted(self.released_by_type.items())
            },
            "depth_below_top_mm": (
                round(self.depth_below_top_mm, 4)
                if self.depth_below_top_mm is not None else None
            ),
            "fillet_radius_mm": (
                round(self.fillet_radius_mm, 4)
                if self.fillet_radius_mm is not None else None
            ),
            "3D_surface": self.surface_3d,
            "blend_ring": self.blend_ring,
            "blend_face_indices": sorted(self.blend_face_indices),
            "template_match": self.template_match,
            "template_deviation": self.template_deviation,
            "access": self.access,
            "inboard_wall_index": self.inboard_wall_index,
            "gateway_face_indices": sorted(self.gateway_face_indices),
            "cap_face_indices": sorted(self.cap_face_indices),
            "toolpath_class": self.toolpath_class,
        }


def assert_pocket_feature_connected(feat: PocketFeature, graph: FaceGraph) -> None:
    """Invariant: every emitted pocket is one connected component under face adjacency."""
    if feat.toolpath_class not in POCKET_CONNECTIVITY_CLASSES:
        return
    comps = induced_subgraph_components(feat.face_indices, graph)
    if len(comps) <= 1:
        return
    parts = ", ".join(f"comp[{i}]={c}" for i, c in enumerate(comps))
    raise PocketConnectivityError(
        f"pocket feature_id={feat.feature_id} class={feat.toolpath_class!r} "
        f"has {len(comps)} disconnected components (expected 1): {parts}"
    )


@dataclass
class PocketDetectionResult:
    features: list[PocketFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    opening_axis: tuple[float, float, float]
    n_clusters: int
    wall_attach_dist: float
    # Diagnostics / handoff.
    excluded_convex_faces: list[int] = field(default_factory=list)
    blend_by_dia_faces: list[int] = field(default_factory=list)
    unattached_wall_faces: list[int] = field(default_factory=list)
    nonconforming_clusters: list[int] = field(default_factory=list)
    grown_faces: list[int] = field(default_factory=list)
    floor_absorbed_faces: list[int] = field(default_factory=list)
    grow_conflicts: list[dict[str, Any]] = field(default_factory=list)
    # Faces shed by R1 structure validation; hole pass should defer these to contour.
    structureless_released_faces: list[int] = field(default_factory=list)
    units: str = "mm"

    def summary(self) -> str:
        n_through = sum(1 for f in self.features if f.subtype == "through_pocket")
        n_blind = sum(1 for f in self.features if f.subtype == "blind_pocket")
        return (
            f"{len(self.features)} pockets ({n_through} through, {n_blind} blind); "
            f"claimed {len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain for the hole pass"
        )


# ---------------------------------------------------------------------------
# Candidate description
# ---------------------------------------------------------------------------
@dataclass
class _Wall:
    face_index: int
    kind: str  # cylinder | cone
    diameter_mm: float | None
    axis_dir: np.ndarray  # unit direction
    uv: np.ndarray  # (2,) projected centroid


def _project_uv(centroid: np.ndarray, opening_axis: np.ndarray) -> np.ndarray:
    """Drop the opening-axis component; return the 2D in-plane coordinate.

    Builds an orthonormal (e1,e2) basis of the plane perpendicular to the opening
    axis so the projection is a proper 2D coordinate regardless of axis
    orientation (not a naive coordinate drop).
    """
    a = _unit(opening_axis)
    # pick a helper not parallel to a
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(helper - np.dot(helper, a) * a)
    e2 = np.cross(a, e1)
    c = np.asarray(centroid, dtype=np.float64)
    return np.array([float(np.dot(c, e1)), float(np.dot(c, e2))])


# ---------------------------------------------------------------------------
# Opening-axis estimation
# ---------------------------------------------------------------------------
def _wall_axis_coherence(axes: Sequence[np.ndarray]) -> tuple[float, np.ndarray]:
    """(coherence, mean_unit_axis) for a bundle of interior-wall axis directions.

    Coherence is the mean-resultant length after sign-folding each axis about the
    dominant global component (axes are undirected, so +d == -d): 1.0 means the
    walls share one direction, ~0 means they scatter. The mean is the current
    heuristic's estimate; the coherence tells you whether to trust it.
    """
    A = np.array(axes, dtype=np.float64)
    dom = int(np.argmax(np.abs(A).sum(axis=0)))
    A = A * np.sign(A[:, dom] + 1e-12)[:, None]
    mean = A.mean(axis=0)
    coherence = float(np.linalg.norm(mean))
    return coherence, _unit(mean)


def _broad_face_axis(faces: Sequence[Any]) -> np.ndarray | None:
    """Dominant cardinal by total planar-face area — the physical opening axis.

    A plate/mold opens along the normal of its broad faces, not along its cavity
    blend-wall axes. Sums planar-face area (OCC world-frame normals) into the three
    cardinal buckets and returns the unit vector of the largest. Returns None when
    there are no planar faces to measure. Uses every face: the opening axis is a
    global part property, independent of which faces the cascade has claimed.
    """
    area = np.zeros(3, dtype=np.float64)
    for f in faces:
        if getattr(f, "surface_type", None) != "plane":
            continue
        n = _unit(np.asarray(f.normal, dtype=np.float64))
        area[int(np.argmax(np.abs(n)))] += float(f.area)
    if not area.any():
        return None
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmax(area))] = 1.0
    return axis


def _dominant_cardinal(axis: np.ndarray) -> int:
    return int(np.argmax(np.abs(np.asarray(axis, dtype=np.float64))))


def _snap_to_cardinal(axis: np.ndarray, tol_deg: float) -> np.ndarray:
    """Snap to the nearest signed cardinal iff within tol_deg; else return as-is."""
    a = _unit(np.asarray(axis, dtype=np.float64))
    k = _dominant_cardinal(a)
    if abs(a[k]) >= math.cos(math.radians(tol_deg)):
        snapped = np.zeros(3, dtype=np.float64)
        snapped[k] = 1.0 if a[k] >= 0.0 else -1.0
        return snapped
    return a


def _estimate_opening_axis(
    axis_accum: Sequence[np.ndarray],
    faces: Sequence[Any],
    config: PocketDetectionConfig,
) -> np.ndarray:
    """Auto-detect the opening axis from interior-wall directions, with a
    broad-face fallback when the walls are too scattered to trust their mean.

    See :attr:`PocketDetectionConfig.opening_axis_coherence_min` for the guard.
    """
    if not axis_accum:
        broad = _broad_face_axis(faces)
        if broad is None:
            logger.warning(
                "opening axis: no interior-wall seeds AND no planar faces to "
                "measure -> cannot derive axis from geometry; defaulting to "
                "[0, 1, 0]. This part's opening axis is UNDETERMINED."
            )
            return np.array([0.0, 1.0, 0.0])
        # snap (no-op on clean cardinals from _broad_face_axis, future-proofs
        # any derived vector) and keep the positive-dominant sign convention.
        return _snap_to_cardinal(broad, config.opening_axis_snap_tol_deg)

    coherence, mean_axis = _wall_axis_coherence(axis_accum)
    broad_axis = _broad_face_axis(faces)
    # Fall back to broad-face only when the wall mean is BOTH untrustworthy (low
    # coherence) AND points at a different cardinal than the physical broad face.
    # The coherence gate keeps moderately-coherent prismatic parts on their mean;
    # the disagreement gate keeps low-coherence-but-consistent parts (mean already
    # agrees with the broad face) on their mean. Only a scattered mean that also
    # contradicts the broad face (fish_mold) is overridden.
    if (
        broad_axis is not None
        and coherence < config.opening_axis_coherence_min
        and _dominant_cardinal(mean_axis) != _dominant_cardinal(broad_axis)
    ):
        logger.info(
            "opening axis: wall coherence %.3f < %.2f and mean cardinal %d != "
            "broad-face cardinal %d -> broad-face fallback",
            coherence, config.opening_axis_coherence_min,
            _dominant_cardinal(mean_axis), _dominant_cardinal(broad_axis),
        )
        axis = broad_axis
    else:
        axis = mean_axis
    return _snap_to_cardinal(axis, config.opening_axis_snap_tol_deg)


# ---------------------------------------------------------------------------
# Single-linkage proximity clustering (numpy only) with knee detection
# ---------------------------------------------------------------------------
def _single_linkage(points: np.ndarray) -> tuple[list[float], list[tuple[int, int]]]:
    """Return the ascending merge heights (length n-1) of single-linkage.

    heights[i] is the edge distance of the (i+1)-th merge (the merge that reduced
    the component count from n-i to n-i-1). Also returns the merged index pairs.
    """
    n = len(points)
    if n <= 1:
        return [], []
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu, ju])
    uf = _UnionFind(range(n))
    heights: list[float] = []
    pairs: list[tuple[int, int]] = []
    for o in order:
        a, b = int(iu[o]), int(ju[o])
        if uf.find(a) != uf.find(b):
            uf.union(a, b)
            heights.append(float(dist[a, b]))
            pairs.append((a, b))
        if len(heights) == n - 1:
            break
    return heights, pairs


def _labels_at_k(points: np.ndarray, k: int) -> np.ndarray:
    """Single-linkage cluster labels with exactly k components (k>=1)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int)
    if k >= n:
        return np.arange(n)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu, ju])
    uf = _UnionFind(range(n))
    ncomp = n
    for o in order:
        if ncomp <= k:
            break
        a, b = int(iu[o]), int(ju[o])
        if uf.find(a) != uf.find(b):
            uf.union(a, b)
            ncomp -= 1
    roots: dict[int, int] = {}
    out = np.empty(n, dtype=int)
    for i in range(n):
        r = uf.find(i)
        out[i] = roots.setdefault(r, len(roots))
    return out


def _choose_k(
    heights: list[float],
    n_points: int,
    config: PocketDetectionConfig,
) -> tuple[int, str]:
    """Pick the cluster count K from the single-linkage merge heights.

    Knee = the K whose "break height" (merge that would drop K -> K-1) most
    exceeds the largest within-K merge height. Trust it only if that ratio clears
    min_knee_ratio; otherwise fall back to an absolute distance threshold and
    report that the knee was not trusted.
    """
    if config.hard_k is not None:
        return max(1, min(config.hard_k, n_points)), "hard_k override"
    if n_points <= 1:
        return n_points, "single point"
    if not heights:
        return 1, "no merges"

    # heights is ascending; height to break K clusters = heights[n-K] (0-based),
    # within-K max height = heights[n-K-1]. Search K in [2, kmax].
    kmax = min(config.max_clusters, n_points - 1)
    best_k, best_ratio = 1, 0.0
    for k in range(2, kmax + 1):
        break_idx = n_points - k       # merge that goes K -> K-1
        within_idx = n_points - k - 1  # last merge inside K clusters
        if break_idx >= len(heights) or within_idx < 0:
            continue
        within = heights[within_idx]
        brk = heights[break_idx]
        ratio = brk / within if within > 1e-9 else float("inf")
        if ratio > best_ratio:
            best_ratio, best_k = ratio, k

    if best_ratio >= config.min_knee_ratio:
        return best_k, f"knee (gap ratio {best_ratio:.2f} >= {config.min_knee_ratio})"

    # Fallback: absolute threshold tied to feature scale. Use k_scale * median
    # within-neighbour merge height as a distance cut on the linkage.
    med = float(np.median(heights))
    thresh = config.hard_threshold if config.hard_threshold is not None else config.k_scale * med
    k = 1 + sum(1 for h in heights if h > thresh)
    return max(1, k), (
        f"knee not trusted (best ratio {best_ratio:.2f} < {config.min_knee_ratio}); "
        f"fallback threshold {thresh:.2f} mm -> {k} clusters"
    )


def _pocket_region_traversable(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> bool:
    """Faces that may belong to a pocket region (walls, fillets, floors, step planes)."""
    fg = by_index[face_idx]
    st = getattr(fg, "surface_type", None)
    if st in SCULPTED_FLOOR_TYPES:
        return True
    if st in FILLET_TYPES:
        return True
    if st in WALL_SURF_TYPES:
        info = _wall_interior_and_axis(fg, occ_map.get(face_idx) if occ_map else None)
        if info is None:
            return False
        is_interior, _axis_dir, diameter = info
        if config.require_interior_wall and not is_interior:
            return False
        d = 0.0 if diameter is None else float(diameter)
        if config.wall_dia_min_mm is not None and d < config.wall_dia_min_mm:
            return False
        if config.wall_dia_max_mm is not None and d > config.wall_dia_max_mm:
            return False
        return True
    if st == "plane":
        return _is_axial_plane(fg, opening_axis, config)
    return False


def _pocket_region_edge_ok(graph: FaceGraph, u: int, v: int) -> bool:
    """True when two pocket-region faces may be unioned (smooth/concave only)."""
    kind = graph.edge_kind(u, v)
    return kind in ("smooth", "concave")


def _merge_overlapping_region_groups(groups: list[set[int]]) -> list[set[int]]:
    if not groups:
        return []
    uf = _UnionFind(list(range(len(groups))))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i] & groups[j]:
                uf.union(i, j)
    merged: dict[int, set[int]] = defaultdict(set)
    for idx, comp in enumerate(groups):
        merged[uf.find(idx)] |= comp
    return list(merged.values())


def _attach_sculpted_to_wall_regions(
    groups: list[set[int]],
    graph: FaceGraph,
    by_index: dict[int, Any],
) -> list[set[int]]:
    """Assign sculpted floor seeds to wall-bearing regions via any-edge contact."""
    if not groups:
        return groups
    groups = [set(g) for g in groups]
    face_group: dict[int, int] = {}
    for gi, comp in enumerate(groups):
        for f in comp:
            face_group[f] = gi
    changed = True
    while changed:
        changed = False
        for f, fg in by_index.items():
            if fg.surface_type not in SCULPTED_FLOOR_TYPES or f in face_group:
                continue
            nbr_groups = {face_group[nb] for nb in graph.neighbors.get(f, ()) if nb in face_group}
            if not nbr_groups:
                continue
            gi = min(nbr_groups) if len(nbr_groups) > 1 else next(iter(nbr_groups))
            groups[gi].add(f)
            face_group[f] = gi
            changed = True
    return groups


def _region_grow_pocket_groups(
    seed_ids: Sequence[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> list[set[int]]:
    """Connected pocket regions: floor seeds grown through walls/fillets/steps.

    Region grow uses smooth/concave edges only. Sculpted floor patches separated
    by convex rims are attached afterward via any-edge neighbor assignment.
    """
    seed_set = set(seed_ids)
    visited: set[int] = set()
    groups: list[set[int]] = []

    for seed in sorted(seed_ids):
        if seed in visited:
            continue
        q: deque[int] = deque([seed])
        comp: set[int] = {seed}
        while q:
            u = q.popleft()
            for v in graph.neighbors.get(u, ()):
                if v in comp:
                    continue
                if not _pocket_region_traversable(
                    v, by_index, occ_map, opening_axis, config,
                ):
                    continue
                if not _pocket_region_edge_ok(graph, u, v):
                    continue
                comp.add(v)
                q.append(v)
        visited |= comp
        groups.append(comp)

    groups = _merge_overlapping_region_groups(groups)
    groups = [
        g for g in groups
        if any(by_index[i].surface_type in WALL_SURF_TYPES for i in g)
    ]
    groups = _attach_sculpted_to_wall_regions(groups, graph, by_index)
    return groups


def _cluster_from_region(
    comp: set[int],
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
) -> dict[str, Any]:
    members = sorted(comp)
    floor_ids = [i for i in members if by_index[i].surface_type in FLOOR_TYPES]
    sphere_ids = [i for i in members if by_index[i].surface_type in SPHERE_TYPES]
    wall_ids = [i for i in members if by_index[i].surface_type in WALL_SURF_TYPES]
    step_ids = [i for i in members if by_index[i].surface_type == "plane"]
    walls: list[_Wall] = []
    for i in wall_ids:
        info = _wall_interior_and_axis(by_index[i], occ_map.get(i) if occ_map else None)
        if info is None:
            continue
        _is_interior, axis_dir, diameter = info
        walls.append(_Wall(
            i, by_index[i].surface_type, diameter, axis_dir,
            uv=_project_uv(by_index[i].centroid, opening_axis),
        ))
    seed_members = floor_ids + sphere_ids
    pts = np.array([
        _project_uv(by_index[i].centroid, opening_axis) for i in seed_members
    ]) if seed_members else np.zeros((0, 2))
    centroid = pts.mean(axis=0) if len(pts) else np.zeros(2)
    return {
        "seed_ids": seed_members,
        "seed_uv": pts,
        "centroid": centroid,
        "floor_ids": floor_ids,
        "sphere_ids": sphere_ids,
        "wall_ids": wall_ids,
        "walls": walls,
        "step_plane_ids": step_ids,
    }


def _trim_open_pocket_gateway_bleed(
    feat: PocketFeature,
    graph: FaceGraph,
) -> set[int]:
    """Remove pocket faces reachable from gateway exits via smooth/concave paths."""
    if feat.access != "open" or not feat.gateway_face_indices:
        return set()
    exterior: set[int] = set(feat.gateway_face_indices)
    q: deque[int] = deque(feat.gateway_face_indices)
    while q:
        u = q.popleft()
        for v in graph.neighbors.get(u, ()):
            if v in exterior:
                continue
            if not _pocket_region_edge_ok(graph, u, v):
                continue
            exterior.add(v)
            q.append(v)
    return set(feat.face_indices) & exterior


# ---------------------------------------------------------------------------
# Interior sign (reuse hole_detection, OCC or OCC-free)
# ---------------------------------------------------------------------------
def _wall_interior_and_axis(
    fg: Any,
    occ_face: Any | None,
) -> tuple[bool, np.ndarray, float | None] | None:
    """(is_interior, axis_dir_unit, diameter_mm) for a cyl/cone face, or None."""
    st = getattr(fg, "surface_type", None)
    if st not in WALL_SURF_TYPES:
        return None
    radius = getattr(fg, "radius", None)
    if occ_face is not None:
        info = _axis_from_occ_face(occ_face)
        if info is None:
            return None
        axis, occ_radius, _kind = info
        if radius is None:
            radius = occ_radius
        sign, _frac, _n = _interior_sign_occ(occ_face, axis)
        axis_dir = axis.direction
    else:
        direction = getattr(fg, "axis", None)
        if direction is None:
            return None
        axis_dir = _unit(np.asarray(direction, dtype=np.float64))
        # OCC-free interior test needs an axis location; use the centroid as a
        # degraded anchor (self-test path only).
        from hole_detection import Axis
        axis = Axis(np.asarray(fg.centroid, dtype=np.float64), axis_dir)
        sign = _interior_wall_sign(
            np.asarray(fg.centroid, dtype=np.float64),
            np.asarray(fg.normal, dtype=np.float64),
            axis,
        )
    diameter = None if radius is None else 2.0 * float(radius)
    return (sign < 0.0), axis_dir, diameter


def _pocket_footprint(
    face_indices: Sequence[int],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> tuple[np.ndarray, float]:
    """(centroid_uv, footprint_radius) from a pocket's claimed face set."""
    if not face_indices:
        return np.zeros(2), 0.0
    pts = np.array([
        _project_uv(np.asarray(by_index[i].centroid, dtype=np.float64), opening_axis)
        for i in face_indices
    ])
    centroid = pts.mean(axis=0)
    spread = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1)).max())
    return centroid, spread


def _normal_parallel_to_axis(
    normal: Sequence[float],
    axis: np.ndarray,
    tol_deg: float,
) -> bool:
    n = _unit(np.asarray(normal, dtype=np.float64))
    a = _unit(axis)
    cos_tol = math.cos(math.radians(tol_deg))
    return abs(float(np.dot(n, a))) >= cos_tol


def _is_axial_plane(
    fg: Any,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> bool:
    if getattr(fg, "surface_type", None) != "plane":
        return False
    return _normal_parallel_to_axis(
        fg.normal, opening_axis, config.step_plane_normal_tol_deg,
    )


def _depth_tiers_from_gaps(
    face_ids: Sequence[int],
    depth_fn,
) -> list[list[int]]:
    """Cluster face ids by depth using the largest natural gaps."""
    if not face_ids:
        return []
    items = sorted((depth_fn(i), i) for i in face_ids)
    if len(items) == 1:
        return [[items[0][1]]]
    gaps = [
        (items[k + 1][0] - items[k][0], k)
        for k in range(len(items) - 1)
        if items[k + 1][0] - items[k][0] > 1e-6
    ]
    if not gaps:
        return [[i for _, i in items]]
    gaps.sort(key=lambda x: -x[0])
    split_indices = sorted(idx for _, idx in gaps[: min(2, len(gaps))])
    tiers: list[list[int]] = []
    start = 0
    for split_idx in split_indices:
        chunk = items[start: split_idx + 1]
        tiers.append([i for _, i in chunk])
        start = split_idx + 1
    tail = items[start:]
    tiers.append([i for _, i in tail])
    return tiers


def _footprints_containing_face(
    face_idx: int,
    pocket_meta: Sequence[dict[str, Any]],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> list[int]:
    """Pocket ids whose wall-loop footprint contains the face centroid (uv)."""
    uv = _project_uv(np.asarray(by_index[face_idx].centroid, dtype=np.float64), opening_axis)
    hits: list[int] = []
    for pk in pocket_meta:
        dist = float(np.linalg.norm(uv - pk["centroid_uv"]))
        if dist <= pk["footprint_radius"] + 1e-6:
            hits.append(int(pk["feature_id"]))
    return hits


def _torus_borders_pocket_walls(
    torus_id: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pocket_walls: set[int],
) -> bool:
    for w in graph.neighbors.get(torus_id, ()):
        if w in pocket_walls and by_index[w].surface_type in WALL_SURF_TYPES:
            return True
        if by_index[w].surface_type not in FILLET_TYPES:
            continue
        for w2 in graph.neighbors.get(w, ()):
            if w2 in pocket_walls and by_index[w2].surface_type in WALL_SURF_TYPES:
                return True
    return False


def _local_fillet_walls_for_pocket(
    step_face: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pocket_walls: set[int],
    *,
    max_hops: int = 3,
) -> set[int]:
    """Walls of `pocket_walls` reachable from step_face through fillet/blend hops."""
    reached: set[int] = set()
    q: deque[tuple[int, list[int], int]] = deque([(step_face, [step_face], 0)])
    seen_paths: set[tuple[int, ...]] = {(step_face,)}

    while q:
        u, path, depth = q.popleft()
        if depth >= max_hops:
            continue
        for v in graph.neighbors.get(u, ()):
            st = by_index[v].surface_type
            if v in pocket_walls and st in WALL_SURF_TYPES:
                reached.add(v)
                continue
            if st in FILLET_TYPES or (depth > 0 and st in BLEND_TRAVERSABLE):
                key = tuple(path + [v])
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                q.append((v, path + [v], depth + 1))
    return reached


def _floor_signature_passes(
    face_idx: int,
    feat: PocketFeature,
    *,
    by_index: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> bool:
    """Structural pocket-floor signature (i)-(iv) relative to one pocket instance."""
    fg = by_index[face_idx]
    if not _is_axial_plane(fg, opening_axis, config):
        return False

    axis = _unit(np.asarray(feat.opening_axis, dtype=np.float64))
    pocket_walls = {
        i for i in feat.face_indices if by_index[i].surface_type in WALL_SURF_TYPES
    }
    if not pocket_walls:
        return False

    reached_walls = _local_fillet_walls_for_pocket(
        face_idx, graph, by_index, pocket_walls,
    )
    intermediate_faces: set[int] = set()
    q2: deque[tuple[int, list[int], int]] = deque([(face_idx, [face_idx], 0)])
    seen2: set[tuple[int, ...]] = {(face_idx,)}
    while q2:
        u, path, depth = q2.popleft()
        if depth >= 3:
            continue
        for v in graph.neighbors.get(u, ()):
            st = by_index[v].surface_type
            if v in pocket_walls and st in WALL_SURF_TYPES:
                continue
            if st in FILLET_TYPES or (depth > 0 and st in BLEND_TRAVERSABLE):
                key = tuple(path + [v])
                if key in seen2:
                    continue
                seen2.add(key)
                intermediate_faces.add(v)
                q2.append((v, path + [v], depth + 1))

    # (i) enclosed by this pocket's wall loop via local fillet chain
    test_i = (
        bool(reached_walls)
        and reached_walls <= pocket_walls
        and all(
            by_index[p].surface_type not in CONTOUR_TYPES or p in feat.face_indices
            for p in intermediate_faces
        )
    )
    if not test_i:
        return False

    # (ii) normal parallel to pocket opening axis
    if not _normal_parallel_to_axis(fg.normal, axis, config.step_plane_normal_tol_deg):
        return False

    # (iii) at the closed axial end of this pocket's plane tiers
    pocket_planes = [
        i for i in feat.face_indices if by_index[i].surface_type == "plane"
    ] + [face_idx]
    axials = {i: _axial_y(by_index, i, axis) for i in pocket_planes}
    axial_tiers = _depth_tiers_from_gaps(
        list(axials.keys()),
        lambda i: -axials[i],
    )
    deepest_tier = set(axial_tiers[-1]) if axial_tiers else set()
    mouth_axial = max(axials[i] for i in feat.face_indices if i in axials)
    step_axial = axials[face_idx]
    if face_idx not in deepest_tier or step_axial > mouth_axial + 1e-6:
        return False

    # (iv) concave step->fillet edges onto this pocket's wall fillet ring
    pocket_fillet_edges: list[tuple[int, str]] = []
    for nb in graph.neighbors.get(face_idx, ()):
        if by_index[nb].surface_type not in FILLET_TYPES:
            continue
        ek = graph.edge_kind(face_idx, nb) or "?"
        if ek != "concave":
            continue
        if _torus_borders_pocket_walls(nb, graph, by_index, pocket_walls):
            pocket_fillet_edges.append((nb, ek))
    if not pocket_fillet_edges:
        return False

    return True


def _absorb_pocket_floors(
    features: list[PocketFeature],
    candidate_faces: set[int],
    claimed: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> list[int]:
    """Claim pocket floors by footprint containment + floor signature (not hop grow)."""
    feat_by_id = {f.feature_id: f for f in features}
    pocket_meta: list[dict[str, Any]] = []
    for f in features:
        c_uv, fp_r = _pocket_footprint(sorted(f.face_indices), by_index, opening_axis)
        pocket_meta.append({
            "feature_id": f.feature_id,
            "centroid_uv": c_uv,
            "footprint_radius": fp_r,
            "claimed": set(f.face_indices),
        })

    absorbed: list[int] = []
    for face_idx in sorted(candidate_faces - claimed):
        if not _is_axial_plane(by_index[face_idx], opening_axis, config):
            continue
        containing = _footprints_containing_face(
            face_idx, pocket_meta, by_index, opening_axis,
        )
        if len(containing) != 1:
            if len(containing) > 1:
                logger.warning(
                    "floor absorption skip: face %d inside %d pocket footprints %s",
                    face_idx, len(containing), containing,
                )
            continue
        pid = containing[0]
        feat = feat_by_id[pid]
        if not _floor_signature_passes(
            face_idx, feat, by_index=by_index, graph=graph,
            opening_axis=opening_axis, config=config,
        ):
            continue
        feat.face_indices.add(face_idx)
        feat.step_plane_count += 1
        for pk in pocket_meta:
            if pk["feature_id"] == pid:
                pk["claimed"].add(face_idx)
                c_uv, fp_r = _pocket_footprint(sorted(pk["claimed"]), by_index, opening_axis)
                pk["centroid_uv"] = c_uv
                pk["footprint_radius"] = fp_r
                break
        claimed.add(face_idx)
        absorbed.append(face_idx)

    if absorbed:
        logger.info(
            "floor absorption: claimed %d pocket floor plane(s) by footprint+signature: %s",
            len(absorbed), sorted(absorbed),
        )
    return absorbed


def _is_grow_candidate(
    fg: Any,
    occ_face: Any | None,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> bool:
    """True when an unclaimed face may be grown into a pocket (structural gate).

    Rejects blend-ring types, already-claimed seed types, and planes whose
    normals are not ~parallel to the opening axis (filters central-stack end caps
    without hardcoding indices). Cyl/cone candidates reuse the wall interior +
    diameter-band gates.
    """
    st = getattr(fg, "surface_type", None)
    if st == "torus":
        return False
    if st in FLOOR_TYPES or st in SPHERE_TYPES:
        return False
    if st == "plane":
        n = _unit(np.asarray(fg.normal, dtype=np.float64))
        a = _unit(opening_axis)
        cos_tol = math.cos(math.radians(config.step_plane_normal_tol_deg))
        return abs(float(np.dot(n, a))) >= cos_tol
    if st in WALL_SURF_TYPES:
        info = _wall_interior_and_axis(fg, occ_face)
        if info is None:
            return False
        is_interior, _axis_dir, diameter = info
        if config.require_interior_wall and not is_interior:
            return False
        d = 0.0 if diameter is None else float(diameter)
        if config.wall_dia_min_mm is not None and d < config.wall_dia_min_mm:
            return False
        if config.wall_dia_max_mm is not None and d > config.wall_dia_max_mm:
            return False
        return True
    return False


def _pocket_candidates_for_face(
    face_idx: int,
    pocket_meta: Sequence[dict[str, Any]],
    graph: FaceGraph,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> list[tuple[int, float, bool]]:
    """Pockets that may own `face_idx`: inside footprint OR adjacent to claimed.

    Returns [(feature_id, uv_distance, is_adjacent), ...].
    """
    uv = _project_uv(np.asarray(by_index[face_idx].centroid, dtype=np.float64), opening_axis)
    out: list[tuple[int, float, bool]] = []
    nbrs = graph.neighbors.get(face_idx, set())
    for pk in pocket_meta:
        dist = float(np.linalg.norm(uv - pk["centroid_uv"]))
        inside = dist <= pk["footprint_radius"]
        adjacent = bool(nbrs & pk["claimed"])
        if inside or adjacent:
            out.append((pk["feature_id"], dist, adjacent))
    return out


def _pick_pocket_assignment(
    candidates: Sequence[tuple[int, float, bool]],
) -> tuple[int | None, bool]:
    """Resolve 1:1 pocket assignment; return (feature_id, is_conflict)."""
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return candidates[0][0], False
    adjacent = [c for c in candidates if c[2]]
    pool = adjacent if adjacent else list(candidates)
    if len({c[0] for c in pool}) == 1:
        return pool[0][0], len(candidates) > 1
    best = min(pool, key=lambda c: (0 if c[2] else 1, c[1]))
    return best[0], True


def _grow_pocket_membership(
    features: list[PocketFeature],
    candidate_faces: set[int],
    claimed: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
    occ_map: dict[int, Any] | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Grow each pocket by claiming interior faces inside footprint or adjacent.

    General rule (not part-specific): a face belongs to a pocket when it is
    spatially inside that pocket's (X,Z) footprint OR graph-adjacent to a face
    already claimed by that pocket. Assignment is 1:1 per face; conflicts are
    logged rather than silently merged.
    """
    feat_by_id = {f.feature_id: f for f in features}
    pocket_meta: list[dict[str, Any]] = []
    for f in features:
        c_uv, fp_r = _pocket_footprint(sorted(f.face_indices), by_index, opening_axis)
        pocket_meta.append({
            "feature_id": f.feature_id,
            "centroid_uv": c_uv,
            "footprint_radius": fp_r,
            "claimed": set(f.face_indices),
        })

    grown: list[int] = []
    conflicts: list[dict[str, Any]] = []
    unclaimed = sorted(candidate_faces - claimed)

    for face_idx in unclaimed:
        fg = by_index[face_idx]
        if not _is_grow_candidate(fg, occ_map.get(face_idx) if occ_map else None,
                                  opening_axis, config):
            continue
        cands = _pocket_candidates_for_face(
            face_idx, pocket_meta, graph, by_index, opening_axis,
        )
        if not cands:
            continue
        assign_id, is_conflict = _pick_pocket_assignment(cands)
        if assign_id is None:
            continue
        if is_conflict:
            entry = {
                "face": face_idx,
                "candidates": [
                    {"pocket": pid, "dist_mm": round(dist, 3), "adjacent": adj}
                    for pid, dist, adj in cands
                ],
                "assigned_pocket": assign_id,
            }
            conflicts.append(entry)
            logger.warning(
                "grow conflict: face %d matches pockets %s -> assigned pocket %d",
                face_idx, [c[0] for c in cands], assign_id,
            )
        feat = feat_by_id[assign_id]
        feat.face_indices.add(face_idx)
        feat.step_plane_count += int(fg.surface_type == "plane")
        for pk in pocket_meta:
            if pk["feature_id"] == assign_id:
                pk["claimed"].add(face_idx)
                break
        claimed.add(face_idx)
        grown.append(face_idx)

    if grown:
        logger.info(
            "membership grow: claimed %d additional pocket faces (step planes + interior): %s",
            len(grown), sorted(grown),
        )
        for feat in features:
            _refresh_template_match(feat, config)
    return grown, conflicts


def _refresh_template_match(feat: PocketFeature, config: PocketDetectionConfig) -> None:
    """Re-evaluate reference template after the grow pass mutates face counts."""
    expected_walls = {f"⌀{d:.3f}in": c for d, c in config.wall_families_in}
    deviation: dict[str, Any] = {}
    for lab, cnt in expected_walls.items():
        got = feat.wall_diameters.get(lab, 0)
        if got != cnt:
            deviation.setdefault("walls", {})[lab] = {"expected": cnt, "got": got}
    extra_walls = {lab: c for lab, c in feat.wall_diameters.items() if lab not in expected_walls}
    if extra_walls:
        deviation["extra_walls"] = extra_walls
    # floor/sphere counts are cluster metadata (released); validate seed signature.
    if feat.floor_count != config.template_floor_count:
        deviation["floor_seed_count"] = {
            "expected": config.template_floor_count, "got": feat.floor_count,
        }
    if feat.sphere_count != config.template_sphere_count:
        deviation["sphere_seed_count"] = {
            "expected": config.template_sphere_count, "got": feat.sphere_count,
        }
    if feat.step_plane_count != config.template_step_plane_count:
        deviation["step_plane_count"] = {
            "expected": config.template_step_plane_count, "got": feat.step_plane_count,
        }
    claimed_3d = any(
        i in feat.face_indices for i in feat.released_faces
    ) or feat.surface_3d
    if claimed_3d:
        deviation["3D_surface"] = {"expected": False, "got": True}
    feat.template_deviation = deviation
    feat.template_match = not deviation


def _part_axis_top(
    faces: Sequence[Any],
    opening_axis: np.ndarray,
    occ_map: dict[int, Any] | None,
) -> float:
    """Maximum boundary (or centroid) projection along the opening axis."""
    axis = _unit(opening_axis)
    vals: list[float] = []
    if occ_map is not None:
        from brep_extents import collect_boundary_points
        for i, occ_face in occ_map.items():
            pts = collect_boundary_points(occ_face)
            vals.extend((pts @ axis).tolist())
    else:
        for f in faces:
            vals.append(float(np.dot(np.asarray(f.centroid, dtype=np.float64), axis)))
    return float(max(vals)) if vals else 0.0


def _depth_below_top_mm(
    claimed_ids: Sequence[int],
    opening_axis: np.ndarray,
    part_axis_top: float,
    occ_map: dict[int, Any] | None,
    by_index: dict[int, Any],
) -> float | None:
    """part_top − deepest claimed pocket boundary along the opening axis."""
    if not claimed_ids:
        return None
    axis = _unit(opening_axis)
    min_proj = float("inf")
    if occ_map is not None:
        from brep_extents import collect_boundary_points
        for i in claimed_ids:
            pts = collect_boundary_points(occ_map[i])
            min_proj = min(min_proj, float((pts @ axis).min()))
    else:
        for i in claimed_ids:
            c = np.asarray(by_index[i].centroid, dtype=np.float64)
            min_proj = min(min_proj, float(np.dot(c, axis)))
    if not math.isfinite(min_proj):
        return None
    return part_axis_top - min_proj


def _fillet_radius_mm(
    wall_ids: Sequence[int],
    cluster_face_ids: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    occ_map: dict[int, Any] | None = None,
) -> float | None:
    """Max sphere/torus blend radius among fillet faces bordering pocket walls."""

    def _blend_radius(fg: Any, face_idx: int) -> float | None:
        st = fg.surface_type
        if st == "sphere":
            if fg.radius is not None:
                return float(fg.radius)
            if occ_map is not None:
                from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
                from OCC.Core.GeomAbs import GeomAbs_Sphere
                surf = BRepAdaptor_Surface(occ_map[face_idx], True)
                if surf.GetType() == GeomAbs_Sphere:
                    return float(surf.Sphere().Radius())
        elif st == "torus" and fg.torus_minor_r is not None:
            return float(fg.torus_minor_r)
        return None

    radii: list[float] = []
    for wid in wall_ids:
        for nb in graph.neighbors.get(wid, set()):
            if nb not in cluster_face_ids:
                continue
            r = _blend_radius(by_index[nb], nb)
            if r is not None:
                radii.append(r)
    if not radii:
        for i in cluster_face_ids:
            r = _blend_radius(by_index[i], i)
            if r is not None:
                radii.append(r)
    return max(radii) if radii else None


def _release_sculpted_floors(
    feat: PocketFeature,
    floor_ids: list[int],
    sphere_ids: list[int],
    wall_ids: list[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    claimed: set[int],
) -> None:
    """Release unassigned sculpted faces and stray torus; keep cluster-assigned floors."""
    from collections import defaultdict

    released: dict[str, set[int]] = defaultdict(set)
    to_release: set[int] = set()
    cluster_sculpted = set(floor_ids) | set(sphere_ids)

    # bspline/bezier/sphere cluster seeds stay claimed (filleted pocket floor).

    # Torus: keep only if graph-adjacent to a claimed wall; else release to contour.
    cluster_ids = cluster_sculpted | set(wall_ids) | set(feat.face_indices)
    for i in sorted(cluster_ids):
        if by_index[i].surface_type != "torus" or i not in claimed:
            continue
        borders_wall = bool(graph.neighbors.get(i, set()) & set(wall_ids))
        if borders_wall:
            continue
        to_release.add(i)
        released["torus"].add(i)

    # Safety: release sculpted only if it leaked into face_indices without cluster membership.
    for i in list(feat.face_indices):
        st = by_index[i].surface_type
        if st in SCULPTED_FLOOR_TYPES and i not in cluster_sculpted:
            to_release.add(i)
            released[st].add(i)

    feat.face_indices -= to_release
    claimed -= to_release
    feat.released_faces = sorted(to_release)
    feat.released_by_type = {k: sorted(v) for k, v in released.items()}
    feat.surface_3d = False

    if to_release:
        by_type = ", ".join(
            f"{k}={sorted(v)}" for k, v in sorted(feat.released_by_type.items())
        )
        logger.info(
            "pocket %d released non-pocket sculpted to contour (%d faces): %s",
            feat.feature_id, len(to_release), by_type,
        )


def _pocket_passes_r1_structure(
    feat: PocketFeature,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> bool:
    """True when a pocket has fillet blends or walled+axial-floor structure (R1)."""
    if feat.fillet_radius_mm is not None and feat.fillet_radius_mm > 0:
        return True
    wall_count = sum(
        1 for i in feat.face_indices if by_index[i].surface_type in WALL_SURF_TYPES
    )
    if wall_count < 1:
        return False
    return any(
        _is_axial_plane(by_index[i], opening_axis, config)
        for i in feat.face_indices
    )


def _release_structureless_pockets(
    features: list[PocketFeature],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
    claimed: set[int],
) -> list[int]:
    """Release pockets that fail R1; faces flow to residual as contour_surface."""
    from collections import defaultdict

    released_all: list[int] = []
    for feat in features:
        if not feat.face_indices:
            continue
        if _pocket_passes_r1_structure(feat, by_index, opening_axis, config):
            continue
        to_release = set(feat.face_indices)
        merged_by_type: dict[str, set[int]] = defaultdict(set)
        for k, v in feat.released_by_type.items():
            merged_by_type[k].update(v)
        merged_by_type["structureless"].update(to_release)

        feat.released_faces = sorted(set(feat.released_faces) | to_release)
        feat.released_by_type = {k: sorted(v) for k, v in sorted(merged_by_type.items())}
        feat.face_indices -= to_release
        claimed -= to_release
        released_all.extend(sorted(to_release))
        logger.info(
            "pocket %d released structureless sculpted band to contour (%d faces): %s",
            feat.feature_id, len(to_release), sorted(to_release),
        )

    if released_all:
        features[:] = [f for f in features if f.face_indices]
    return released_all


# ---------------------------------------------------------------------------
# Open / closed pocket access (setup-scoped — not geometry-inferred)
# ---------------------------------------------------------------------------
def _axial_y(by_index: dict[int, Any], face_id: int, opening_axis: np.ndarray) -> float:
    return float(by_index[face_id].centroid @ opening_axis)


def pocket_toolpath_class(feat: PocketFeature, *, setup: PocketSetupConfig) -> str:
    """Map pocket geometry + setup access to Toolpath emit class."""
    has_fillet = feat.fillet_radius_mm is not None and feat.fillet_radius_mm > 0
    access, _explicit = setup.resolved_access()
    is_open = access == "open"
    if has_fillet and is_open:
        return "filleted_open_pocket"
    if has_fillet:
        return "filleted_pocket"
    if is_open:
        return "open_pocket"
    return "pocket"


def pocket_config_from_setup_dict(data: dict[str, Any] | None) -> PocketSetupConfig:
    """Build ``PocketSetupConfig`` from GT yaml ``setup:`` block or top-level keys."""
    data = data or {}
    nested = data.get("setup") if isinstance(data.get("setup"), dict) else {}
    side = nested.get("machining_side") or data.get("machining_side")
    access = nested.get("pocket_access") or data.get("pocket_access")
    if side is not None and side not in ("front", "back"):
        raise ValueError(f"machining_side must be 'front' or 'back', got {side!r}")
    if access is not None and access not in ("open", "closed"):
        raise ValueError(f"pocket_access must be 'open' or 'closed', got {access!r}")
    return PocketSetupConfig(
        machining_side=side,
        pocket_access=access,
    )


def infer_machining_side_from_step(step_path: str | Path) -> Literal["front", "back"] | None:
    """Best-effort filename hint for split panel STEPs (not a geometry inference)."""
    name = Path(step_path).name.upper()
    if "FRONT" in name:
        return "front"
    if "REAR" in name or "BACK" in name:
        return "back"
    return None


def resolve_pocket_setup_for_run(
    step_path: str | Path,
    *,
    machining_side: Literal["front", "back"] | None = None,
    pocket_access: Literal["open", "closed"] | None = None,
) -> PocketSetupConfig:
    """CLI / run_cascade helper: explicit args beat filename hint."""
    if pocket_access is not None or machining_side is not None:
        return PocketSetupConfig(
            machining_side=machining_side,
            pocket_access=pocket_access,
        )
    inferred = infer_machining_side_from_step(step_path)
    if inferred is not None:
        return PocketSetupConfig(machining_side=inferred)
    return PocketSetupConfig()


def classify_pocket_open_closed(
    feat: PocketFeature,
    setup: PocketSetupConfig,
) -> None:
    """Apply setup-scoped pocket access; no geometry cap detection."""
    access, explicit = setup.resolved_access()
    if not explicit:
        logger.warning(
            "pocket %d: no machining_side/pocket_access in setup config — "
            "defaulting access to %r (filleted_pocket when filleted). "
            "See eval/pocket_openclosed_findings.md.",
            feat.feature_id, access,
        )
    feat.access = access
    feat.inboard_wall_index = None
    feat.gateway_face_indices = []
    feat.cap_face_indices = []
    feat.toolpath_class = pocket_toolpath_class(feat, setup=setup)


def apply_explicit_setup_access_to_pocket(
    feat: PocketFeature,
    setup: PocketSetupConfig,
) -> bool:
    """Relabel pocket access/class when setup declares ``pocket_access`` explicitly.

    Returns True when ``feat.toolpath_class`` / ``feat.access`` were overwritten;
    False when ``pocket_access`` is unset (geometry-band labels preserved).

    Lobe-tier split features keep their open/closed band emit class — setup
    ``pocket_access`` applies to spatial pockets only, not mouth/deep tier bands.
    """
    if setup.pocket_access is None:
        return False
    if feat.template_deviation.get("lobe_tier"):
        return False
    feat.access = setup.pocket_access
    feat.inboard_wall_index = None
    feat.gateway_face_indices = []
    feat.cap_face_indices = []
    feat.toolpath_class = pocket_toolpath_class(feat, setup=setup)
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_pockets(
    faces: Sequence[Any],
    edge_index: np.ndarray | None = None,
    edge_attr: np.ndarray | None = None,
    *,
    occ_faces: Sequence[Any] | None = None,
    candidate_faces: set[int] | None = None,
    config: PocketDetectionConfig | None = None,
) -> PocketDetectionResult:
    """Run the pocket pass over `candidate_faces` (default: all faces).

    Parameters
    ----------
    faces:
        FaceGeom records from feature_params.analyze_step, TopologyExplorer order.
    edge_index, edge_attr:
        Accepted for interface symmetry with the hole pass; the pocket grouping
        is purely spatial, but edge_index is used for the post-cluster membership
        GROW step (adjacency to already-claimed faces — valid and distinct from
        using the graph to GROUP pockets, which fails on this part).
    occ_faces:
        TopoDS_Face list (same order) for the robust multi-sample interior test.
        When omitted, a degraded centroid-normal interior test is used.
    candidate_faces:
        The still-unclaimed face indices this pass may consume. Default = every
        face (the pocket pass runs FIRST in the reordered cascade, on the full
        pool). Faces outside this set are never touched.
    """
    config = config or PocketDetectionConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    if candidate_faces is None:
        candidate_faces = set(range(n_faces))
    candidate_faces = {i for i in candidate_faces if 0 <= i < n_faces}

    occ_map = None
    if occ_faces is not None:
        if len(occ_faces) != n_faces:
            raise ValueError(
                f"occ_faces ({len(occ_faces)}) and faces ({n_faces}) differ; "
                "both must be TopologyExplorer order for the same shape"
            )
        occ_map = {i: occ_faces[i] for i in range(n_faces)}

    # --- floor seeds + sphere seeds (the reliable pocket signature) ---
    floor_ids = [i for i in candidate_faces
                 if by_index[i].surface_type in FLOOR_TYPES]
    sphere_ids = [i for i in candidate_faces
                  if by_index[i].surface_type in SPHERE_TYPES]
    seed_ids = sorted(floor_ids + sphere_ids)
    if not seed_ids:
        # No pocket seeds, but the opening axis is a global part property (used by
        # downstream flats/wall/setup logic), so derive it from geometry instead
        # of hardcoding a blind Y. Honor an explicit config override for parity
        # with the normal (seeded) path below; otherwise use the shared broad-face
        # estimator via _estimate_opening_axis (empty axis_accum -> broad-face).
        if config.opening_axis is not None:
            noseed_axis = _unit(np.asarray(config.opening_axis, dtype=np.float64))
        else:
            noseed_axis = _estimate_opening_axis([], faces, config)
        logger.info(
            "no bspline/sphere floor seeds in candidate pool -> 0 pockets "
            "(opening axis %s derived from geometry)", noseed_axis.tolist(),
        )
        return PocketDetectionResult(
            features=[], claimed_faces=set(), remaining_faces=set(candidate_faces),
            n_faces=n_faces,
            opening_axis=tuple(float(x) for x in noseed_axis), n_clusters=0,
            wall_attach_dist=0.0, units=config.units,
        )

    # --- interior cyl/cone wall candidates (+ convex rejects) ---
    walls: list[_Wall] = []
    excluded_convex: list[int] = []
    blend_by_dia: list[int] = []
    axis_accum: list[np.ndarray] = []
    for i in sorted(candidate_faces):
        info = _wall_interior_and_axis(by_index[i], occ_map[i] if occ_map else None)
        if info is None:
            continue
        is_interior, axis_dir, diameter = info
        if config.require_interior_wall and not is_interior:
            excluded_convex.append(i)
            continue
        # wall-diameter band: reject blend-ring surfaces (very large-radius blend
        # cylinders / near-apex cones) that are also concave and near a floor.
        d = 0.0 if diameter is None else float(diameter)
        if config.wall_dia_min_mm is not None and d < config.wall_dia_min_mm:
            blend_by_dia.append(i)
            continue
        if config.wall_dia_max_mm is not None and d > config.wall_dia_max_mm:
            blend_by_dia.append(i)
            continue
        walls.append(_Wall(i, by_index[i].surface_type, diameter, axis_dir,
                           uv=np.zeros(2)))  # uv filled after axis known
        axis_accum.append(axis_dir)

    # --- opening axis (auto-detect from the shared wall-axis direction, with a
    #     broad-face fallback when the walls scatter; see _estimate_opening_axis) ---
    if config.opening_axis is not None:
        opening_axis = _unit(np.asarray(config.opening_axis, dtype=np.float64))
    else:
        opening_axis = _estimate_opening_axis(axis_accum, faces, config)

    # project seeds + walls into the plane perpendicular to the opening axis
    seed_uv = np.array([_project_uv(by_index[i].centroid, opening_axis) for i in seed_ids])
    for w in walls:
        w.uv = _project_uv(by_index[w.face_index].centroid, opening_axis)

    graph = FaceGraph.from_edge_tensors(
        edge_index if edge_index is not None else np.zeros((2, 0), dtype=np.int64),
        edge_attr if edge_attr is not None else np.zeros((0, 4), dtype=np.float32),
        n_faces,
    )

    # --- group floor seeds into pocket seed clusters ---
    seed_id_to_j = {sid: j for j, sid in enumerate(seed_ids)}
    attach_dist = 0.0
    if config.floor_seed_grouping == "region_grow":
        region_groups = _region_grow_pocket_groups(
            seed_ids, graph, by_index, occ_map, opening_axis, config,
        )
        n_clusters = len(region_groups)
        logger.info(
            "floor-seed region grow: %d seeds -> %d pocket regions",
            len(seed_ids), n_clusters,
        )
        clusters: dict[int, dict[str, Any]] = {}
        for lab, comp in enumerate(region_groups):
            clusters[lab] = _cluster_from_region(
                comp, by_index, occ_map, opening_axis,
            )
    else:
        heights, _pairs = _single_linkage(seed_uv)
        k, why = _choose_k(heights, len(seed_ids), config)
        labels = _labels_at_k(seed_uv, k)
        n_clusters = int(len(np.unique(labels)))
        seed_groups = [
            [seed_ids[j] for j in range(len(seed_ids)) if labels[j] == lab]
            for lab in range(n_clusters)
        ]
        logger.info("floor-seed clustering: %d seeds -> %d clusters [%s]",
                    len(seed_ids), n_clusters, why)

        clusters = {}
        for lab, members in enumerate(seed_groups):
            j_idx = [seed_id_to_j[m] for m in members]
            pts = seed_uv[j_idx]
            clusters[lab] = {
                "seed_ids": members,
                "seed_uv": pts,
                "centroid": pts.mean(axis=0),
                "floor_ids": [i for i in members if by_index[i].surface_type in FLOOR_TYPES],
                "sphere_ids": [i for i in members if by_index[i].surface_type in SPHERE_TYPES],
            }

        # --- adaptive wall-attach distance from inter-pocket spacing ---
        if n_clusters >= 2:
            cents = np.array([clusters[l]["centroid"] for l in range(n_clusters)])
            cd = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2))
            np.fill_diagonal(cd, np.inf)
            min_inter = float(cd.min())
        else:
            pts = clusters[0]["seed_uv"]
            c = clusters[0]["centroid"]
            min_inter = 2.0 * float(np.sqrt(((pts - c) ** 2).sum(axis=1)).max() + 1e-9)
        attach_dist = config.wall_attach_or_default(min_inter)
        logger.info("min inter-pocket centroid = %.2f mm -> wall_attach_dist = %.2f mm",
                    min_inter, attach_dist)

        # --- attach each wall to the nearest cluster whose floors it is close to ---
        unattached: list[int] = []
        for w in walls:
            best_lab, best_d = None, float("inf")
            for lab in range(n_clusters):
                fpts = clusters[lab]["seed_uv"]
                d = float(np.sqrt(((fpts - w.uv) ** 2).sum(axis=1)).min())
                if d < best_d:
                    best_d, best_lab = d, lab
            if best_lab is not None and best_d <= attach_dist:
                clusters[best_lab].setdefault("wall_ids", []).append(w.face_index)
                clusters[best_lab].setdefault("walls", []).append(w)
            else:
                unattached.append(w.face_index)

    if config.floor_seed_grouping == "region_grow":
        unattached = [
            w.face_index for w in walls
            if not any(w.face_index in clusters[l].get("wall_ids", []) for l in clusters)
        ]

    # --- build PocketFeatures + claim faces ---
    features: list[PocketFeature] = []
    nonconforming: list[int] = []
    claimed: set[int] = set()
    # order clusters by (u,v) for stable ids
    order = sorted(range(n_clusters), key=lambda l: tuple(clusters[l]["centroid"]))
    for new_id, lab in enumerate(order):
        cl = clusters[lab]
        cl_walls: list[_Wall] = cl.get("walls", [])
        wall_ids = cl.get("wall_ids", [])
        floor_ids_c = cl["floor_ids"]
        sphere_ids_c = cl["sphere_ids"]

        feat = _build_feature(
            new_id, cl, cl_walls, wall_ids, floor_ids_c, sphere_ids_c,
            by_index, opening_axis, config,
            step_plane_ids=cl.get("step_plane_ids"),
        )
        if not feat.template_match:
            nonconforming.append(new_id)
            logger.info(
                "pocket %d NONCONFORMING (claimed anyway; review): %s",
                new_id, feat.template_deviation,
            )
        features.append(feat)
        claimed |= feat.face_indices

    # --- grow membership (spatial path only; region_grow claims in one pass) ---
    grown: list[int] = []
    grow_conflicts: list[dict[str, Any]] = []
    floor_absorbed: list[int] = []
    if config.floor_seed_grouping == "spatial":
        grown, grow_conflicts = _grow_pocket_membership(
            features, candidate_faces, claimed, by_index, graph,
            opening_axis, config, occ_map,
        )
        floor_absorbed = _absorb_pocket_floors(
            features, candidate_faces, claimed, by_index, graph,
            opening_axis, config,
        )
    if floor_absorbed:
        for feat in features:
            _refresh_template_match(feat, config)

    part_top = _part_axis_top(faces, opening_axis, occ_map)
    for feat in features:
        # Recover cluster seed lists from the original build (stored as counts).
        lab = order[feat.feature_id] if feat.feature_id < len(order) else None
        cl = clusters[lab] if lab is not None else {}
        floor_ids_c = cl.get("floor_ids", [])
        sphere_ids_c = cl.get("sphere_ids", [])
        wall_ids_c = [w.face_index for w in cl.get("walls", [])]

        _release_sculpted_floors(
            feat, floor_ids_c, sphere_ids_c, wall_ids_c,
            by_index, graph, claimed,
        )
        cluster_ids = (
            set(wall_ids_c) | set(floor_ids_c) | set(sphere_ids_c) | feat.face_indices
        )
        feat.depth_below_top_mm = _depth_below_top_mm(
            sorted(feat.face_indices), opening_axis, part_top, occ_map, by_index,
        )
        feat.fillet_radius_mm = _fillet_radius_mm(
            wall_ids_c, cluster_ids, by_index, graph, occ_map,
        )
        _refresh_template_match(feat, config)
        classify_pocket_open_closed(feat, config.setup)
        if config.floor_seed_grouping == "region_grow":
            bleed = _trim_open_pocket_gateway_bleed(feat, graph)
            if bleed:
                feat.face_indices -= bleed
                claimed -= bleed
                logger.info(
                    "pocket %d gateway trim removed %d exterior bleed face(s): %s",
                    feat.feature_id, len(bleed), sorted(bleed),
                )
                _refresh_template_match(feat, config)
                classify_pocket_open_closed(feat, config.setup)

    structureless_released = _release_structureless_pockets(
        features, by_index, opening_axis, config, claimed,
    )

    remaining = set(candidate_faces) - claimed

    if excluded_convex:
        logger.info("rejected %d convex (outer/boss) cyl/cone faces (not pocket walls): %s",
                    len(excluded_convex), sorted(excluded_convex))
    if blend_by_dia:
        logger.info("deferred %d concave blend-ring faces outside the wall-diameter "
                    "band [%s, %s] mm (not claimed): %s",
                    len(blend_by_dia), config.wall_dia_min_mm, config.wall_dia_max_mm,
                    sorted(blend_by_dia))
    if unattached:
        logger.info("%d interior cyl/cone faces too far from any pocket floor "
                    "(> %.2f mm) -> left for the hole pass: %s",
                    len(unattached), attach_dist, sorted(unattached))

    return PocketDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        opening_axis=tuple(round(float(x), 4) for x in opening_axis),
        n_clusters=n_clusters,
        wall_attach_dist=attach_dist,
        excluded_convex_faces=sorted(excluded_convex),
        blend_by_dia_faces=sorted(blend_by_dia),
        unattached_wall_faces=sorted(unattached),
        nonconforming_clusters=nonconforming,
        grown_faces=sorted(grown),
        floor_absorbed_faces=sorted(floor_absorbed),
        grow_conflicts=grow_conflicts,
        structureless_released_faces=sorted(structureless_released),
        units=config.units,
    )


def _pocket_feature_from_lobe_tier(
    feature_id: int,
    face_indices: set[int],
    *,
    lobe_centroid_uv: tuple[float, float],
    opening_axis: tuple[float, float, float],
    access: str,
    toolpath_class: str,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    part_top: float,
) -> PocketFeature:
    """Build a PocketFeature from a lobe tier face set (open or closed band)."""
    axis = np.asarray(opening_axis, dtype=np.float64)
    wall_ids = [
        i for i in face_indices if by_index[i].surface_type in WALL_SURF_TYPES
    ]
    floor_ids = [i for i in face_indices if by_index[i].surface_type in FLOOR_TYPES]
    sphere_ids = [i for i in face_indices if by_index[i].surface_type in SPHERE_TYPES]
    step_ids = [i for i in face_indices if by_index[i].surface_type == "plane"]

    wall_diameters: dict[str, int] = {}
    for i in wall_ids:
        info = _wall_interior_and_axis(by_index[i], occ_map.get(i) if occ_map else None)
        if info is None:
            continue
        lab = _dia_family_label(info[2])
        wall_diameters[lab] = wall_diameters.get(lab, 0) + 1

    return PocketFeature(
        feature_id=feature_id,
        kind="pocket",
        subtype="blind_pocket",
        face_indices=set(face_indices),
        centroid_uv=lobe_centroid_uv,
        opening_axis=opening_axis,
        wall_diameters=wall_diameters,
        wall_count=len(wall_ids),
        floor_count=len(floor_ids),
        sphere_count=len(sphere_ids),
        step_plane_count=len(step_ids),
        released_faces=[],
        released_by_type={},
        depth_below_top_mm=_depth_below_top_mm(
            sorted(face_indices), axis, part_top, occ_map, by_index,
        ),
        fillet_radius_mm=_fillet_radius_mm(
            wall_ids, face_indices, by_index, graph, occ_map,
        ),
        surface_3d=False,
        blend_ring={},
        blend_face_indices=[],
        template_match=True,
        template_deviation={"lobe_tier": True},
        access=access,
        toolpath_class=toolpath_class,
    )


def apply_filleted_lobe_tiers_to_result(
    result: PocketDetectionResult,
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    occ_faces: Sequence[Any] | None,
    config: PocketDetectionConfig,
    *,
    n_lobes_hint: int = 7,
    min_lobes: int = 6,
) -> PocketDetectionResult:
    """Replace spatial filleted-pocket clusters with lobe tier open/closed features."""
    from lobe_tier_detection import (
        LobeTierConfig,
        detect_filleted_lobe_tiers,
        _is_exterior_contour_fillet,
    )

    lobe = detect_filleted_lobe_tiers(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces,
        opening_axis=result.opening_axis,
        n_lobes_hint=n_lobes_hint,
        config=LobeTierConfig(),
    )
    if len(lobe.lobes) < min_lobes:
        logger.info(
            "lobe tier split: %d lobes (< %d) — keeping spatial pockets",
            len(lobe.lobes), min_lobes,
        )
        return result

    pool_all: set[int] = set()
    for lob in lobe.lobes:
        pool_all |= lob.pool_faces
    if not pool_all:
        return result

    by_index = {int(f.index): f for f in faces}
    occ_map = None
    if occ_faces is not None:
        occ_map = {i: occ_faces[i] for i in range(len(faces))}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))
    opening_axis = np.asarray(result.opening_axis, dtype=np.float64)
    part_top = _part_axis_top(faces, opening_axis, occ_map)

    kept: list[PocketFeature] = []
    released_from_spatial: set[int] = set()
    for feat in result.features:
        if feat.face_indices & pool_all:
            released_from_spatial |= feat.face_indices
            continue
        kept.append(feat)

    released_exterior: set[int] = set()
    for feat in kept:
        exterior = {
            i for i in feat.face_indices
            if _is_exterior_contour_fillet(i, graph, by_index)
        }
        if not exterior:
            continue
        feat.face_indices -= exterior
        released_exterior |= exterior
    kept = [f for f in kept if f.face_indices]

    next_id = max((f.feature_id for f in kept), default=-1) + 1
    new_claimed: set[int] = set()
    for feat in kept:
        new_claimed |= feat.face_indices

    tier_features: list[PocketFeature] = []
    for lob in sorted(lobe.lobes, key=lambda lb: lb.centroid_uv):
        for access, face_set, tp_class in (
            ("open", lob.open_faces, "filleted_open_pocket"),
            ("closed", lob.closed_faces, "filleted_pocket"),
        ):
            if not face_set:
                continue
            feat = _pocket_feature_from_lobe_tier(
                next_id, face_set,
                lobe_centroid_uv=lob.centroid_uv,
                opening_axis=result.opening_axis,
                access=access,
                toolpath_class=tp_class,
                by_index=by_index,
                occ_map=occ_map,
                graph=graph,
                part_top=part_top,
            )
            apply_explicit_setup_access_to_pocket(feat, config.setup)
            assert_pocket_feature_connected(feat, graph)
            tier_features.append(feat)
            new_claimed |= face_set
            next_id += 1

    logger.info(
        "lobe tier split: dropped %d spatial pocket(s), added %d tier features "
        "(%d lobes; open=%d closed=%d faces)",
        len(result.features) - len(kept),
        len(tier_features),
        len(lobe.lobes),
        len(lobe.all_open_faces()),
        len(lobe.all_closed_faces()),
    )

    remaining = set(result.remaining_faces) | released_from_spatial | released_exterior
    remaining -= new_claimed

    return PocketDetectionResult(
        features=kept + tier_features,
        claimed_faces=new_claimed,
        remaining_faces=remaining,
        n_faces=result.n_faces,
        opening_axis=result.opening_axis,
        n_clusters=result.n_clusters,
        wall_attach_dist=result.wall_attach_dist,
        excluded_convex_faces=result.excluded_convex_faces,
        blend_by_dia_faces=result.blend_by_dia_faces,
        unattached_wall_faces=result.unattached_wall_faces,
        nonconforming_clusters=result.nonconforming_clusters,
        grown_faces=result.grown_faces,
        floor_absorbed_faces=result.floor_absorbed_faces,
        grow_conflicts=result.grow_conflicts,
        structureless_released_faces=list(result.structureless_released_faces),
        units=result.units,
    )


def _dia_family_label(diameter_mm: float | None) -> str:
    if diameter_mm is None:
        return "⌀?"
    return f"⌀{diameter_mm / MM_PER_INCH:.3f}in"


def _build_feature(
    feature_id: int,
    cl: dict[str, Any],
    cl_walls: list[_Wall],
    wall_ids: list[int],
    floor_ids: list[int],
    sphere_ids: list[int],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
    step_plane_ids: list[int] | None = None,
) -> PocketFeature:
    step_plane_ids = step_plane_ids or []
    # Include sculpted seeds for the grow pass (footprint + adjacency); cluster-
    # assigned seeds are retained as filleted floor in _release_sculpted_floors.
    face_indices = (
        set(wall_ids) | set(floor_ids) | set(sphere_ids) | set(step_plane_ids)
    )

    # wall diameter families
    wall_diameters: dict[str, int] = {}
    for w in cl_walls:
        lab = _dia_family_label(w.diameter_mm)
        wall_diameters[lab] = wall_diameters.get(lab, 0) + 1

    # through vs blind: does the pocket have walls opening on BOTH sides of the
    # opening axis? (through pockets have +axis and -axis wall centroids)
    a = _unit(opening_axis)
    sides = set()
    for w in cl_walls:
        proj = float(np.dot(np.asarray(by_index[w.face_index].centroid, float), a))
        # relative to the cluster's mean axial position
        sides.add(1 if proj >= 0 else -1)
    # more robust: split walls by their axial coordinate around the median
    axials = [float(np.dot(np.asarray(by_index[w.face_index].centroid, float), a))
              for w in cl_walls]
    subtype = "through_pocket"
    if axials:
        med = float(np.median(axials))
        has_pos = any(x > med + 1e-6 for x in axials)
        has_neg = any(x < med - 1e-6 for x in axials)
        subtype = "through_pocket" if (has_pos and has_neg) else "blind_pocket"

    # template match (reference: wall_families_in + floor/sphere counts)
    expected_walls = {f"⌀{d:.3f}in": c for d, c in config.wall_families_in}
    deviation: dict[str, Any] = {}
    for lab, cnt in expected_walls.items():
        got = wall_diameters.get(lab, 0)
        if got != cnt:
            deviation.setdefault("walls", {})[lab] = {"expected": cnt, "got": got}
    extra_walls = {lab: c for lab, c in wall_diameters.items() if lab not in expected_walls}
    if extra_walls:
        deviation["extra_walls"] = extra_walls
    if len(floor_ids) != config.template_floor_count:
        deviation["floor_seed_count"] = {
            "expected": config.template_floor_count, "got": len(floor_ids),
        }
    if len(sphere_ids) != config.template_sphere_count:
        deviation["sphere_seed_count"] = {
            "expected": config.template_sphere_count, "got": len(sphere_ids),
        }
    n_step = len(step_plane_ids)
    if n_step != config.template_step_plane_count:
        deviation["step_plane_count"] = {
            "expected": config.template_step_plane_count, "got": n_step,
        }
    template_match = not deviation

    centroid_uv = tuple(float(x) for x in cl["centroid"])
    return PocketFeature(
        feature_id=feature_id,
        kind="pocket",
        subtype=subtype,
        face_indices=face_indices,
        centroid_uv=centroid_uv,
        opening_axis=tuple(float(x) for x in a),
        wall_diameters=wall_diameters,
        wall_count=len(wall_ids),
        floor_count=len(floor_ids),
        sphere_count=len(sphere_ids),
        step_plane_count=len(step_plane_ids),
        released_faces=[],
        released_by_type={},
        depth_below_top_mm=None,
        fillet_radius_mm=None,
        surface_3d=False,
        blend_ring={},          # filled by attach_blend_ring (optional, reporting)
        blend_face_indices=[],
        template_match=template_match,
        template_deviation=deviation,
    )


def detect_pockets_from_step(
    step_path: str | Path,
    edge_index: np.ndarray | None = None,
    edge_attr: np.ndarray | None = None,
    *,
    candidate_faces: set[int] | None = None,
    config: PocketDetectionConfig | None = None,
) -> PocketDetectionResult:
    """Convenience wrapper: load FaceGeom + OCC faces from a STEP, then detect."""
    from feature_params import analyze_step, load_step_faces, require_occ

    require_occ()
    faces = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    return detect_pockets(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=candidate_faces, config=config,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class PocketValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"pocket validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_pockets(
    result: PocketDetectionResult,
    *,
    expected_pockets: int = 7,
    expected_faces_per_pocket: int = 24,
    expected_floor_seeds: int = 6,
    expected_sphere_seeds: int = 7,
    expected_released_per_pocket: int = 0,
    expected_step_planes: int = 14,
    depth_below_top_in: float = 1.0848,
    fillet_radius_in: float = 0.25,
    depth_tol_in: float = 0.015,
    fillet_tol_in: float = 0.015,
    excluded_faces: Sequence[int] = (328, 346),
    central_hole_faces: Sequence[int] = (110, 335, 329, 347),
) -> PocketValidationReport:
    """Assert the reference-part filleted pocket contract (see module docstring).

    expected_faces_per_pocket and expected_step_planes are PART-SPECIFIC checks
    for the reference rear plate only. Other parts should validate structure, not counts.
    """
    checks: list[tuple[str, bool, str]] = []

    n = len(result.features)
    checks.append((f"exactly {expected_pockets} pockets", n == expected_pockets,
                   f"got {n}"))

    face_counts = [f.n_faces for f in result.features]
    uniform_faces = len(set(face_counts)) == 1
    checks.append((
        f"uniform ~{expected_faces_per_pocket}-face filleted template across pockets",
        uniform_faces and (not face_counts or face_counts[0] == expected_faces_per_pocket),
        f"face counts={face_counts}",
    ))

    # uniform composition signature (walls + steps + cluster-assigned sculpted floors)
    sigs = set()
    for f in result.features:
        sigs.add((
            tuple(sorted(f.wall_diameters.items())),
            f.floor_count, f.sphere_count, f.step_plane_count,
        ))
    uniform = len(sigs) == 1
    checks.append(("uniform template across pockets", uniform,
                   f"{len(sigs)} distinct signature(s)"
                   + (f"; walls={dict(sorted(next(iter(sigs))[0]))}, "
                      f"floor_seeds={next(iter(sigs))[1]}, sphere_seeds={next(iter(sigs))[2]}, "
                      f"step_planes={next(iter(sigs))[3]}"
                      if uniform else "")))

    all_match = all(f.template_match for f in result.features)
    checks.append(("every pocket matches reference template", all_match,
                   f"{sum(f.template_match for f in result.features)}/{n} match"))

    all_analytic = all(not f.surface_3d for f in result.features)
    checks.append(("every pocket is analytic (3D_surface=False)", all_analytic,
                   f"{sum(not f.surface_3d for f in result.features)}/{n} analytic"))

    both_signs = all(f.subtype == "through_pocket" for f in result.features)
    checks.append(("every pocket is through (both wall signs)", both_signs,
                   f"{sum(f.subtype=='through_pocket' for f in result.features)}/{n} through"))

    checks.append((f"cluster has {expected_floor_seeds} bspline floor seeds per pocket",
                   all(f.floor_count == expected_floor_seeds for f in result.features),
                   f"counts={[f.floor_count for f in result.features]}"))
    checks.append((f"cluster has {expected_sphere_seeds} sphere seeds per pocket",
                   all(f.sphere_count == expected_sphere_seeds for f in result.features),
                   f"counts={[f.sphere_count for f in result.features]}"))

    n_released = sum(len(f.released_faces) for f in result.features)
    checks.append((
        f"released ≤{expected_released_per_pocket} non-cluster sculpted per pocket",
        all(len(f.released_faces) <= expected_released_per_pocket for f in result.features),
        f"total released={n_released}",
    ))
    cluster_sculpted_retained = all(
        f.n_faces == f.wall_count + f.step_plane_count + f.floor_count + f.sphere_count
        for f in result.features
    )
    checks.append((
        "cluster-assigned sculpted floors retained in pocket claims",
        cluster_sculpted_retained,
        "ok" if cluster_sculpted_retained else "missing cluster seeds in claim",
    ))
    no_overlap = all(
        not (set(f.released_faces) & f.face_indices)
        for f in result.features
    )
    checks.append(("released faces disjoint from claimed set", no_overlap,
                   "ok" if no_overlap else "leaked"))

    n_step = sum(f.step_plane_count for f in result.features)
    checks.append((f"all {expected_step_planes} step planes assigned (grow pass)",
                   n_step == expected_step_planes, f"got {n_step}"))

    grow_ok = len(result.grown_faces) == expected_step_planes
    checks.append(("grow pass claimed expected step-plane set", grow_ok,
                   f"grown={sorted(result.grown_faces)}"))

    depths_in = [
        f.depth_below_top_mm / MM_PER_INCH
        for f in result.features
        if f.depth_below_top_mm is not None
    ]
    depth_ok = (
        len(depths_in) == n
        and all(abs(d - depth_below_top_in) <= depth_tol_in for d in depths_in)
    )
    checks.append((
        f"depth_below_top ≈ {depth_below_top_in} in on all pockets",
        depth_ok,
        f"values={[round(d, 4) for d in depths_in]}" if depths_in else "missing",
    ))

    fillets_in = [
        f.fillet_radius_mm / MM_PER_INCH
        for f in result.features
        if f.fillet_radius_mm is not None
    ]
    fillet_ok = (
        len(fillets_in) == n
        and all(abs(r - fillet_radius_in) <= fillet_tol_in for r in fillets_in)
    )
    checks.append((
        f"fillet_radius ≈ {fillet_radius_in} in on all pockets",
        fillet_ok,
        f"values={[round(r, 4) for r in fillets_in]}" if fillets_in else "missing",
    ))

    excl_ok = all(i not in result.claimed_faces for i in excluded_faces)
    checks.append((f"stray faces {list(excluded_faces)} NOT claimed", excl_ok,
                   "excluded" if excl_ok else
                   f"leaked: {[i for i in excluded_faces if i in result.claimed_faces]}"))

    central_ok = all(i not in result.claimed_faces for i in central_hole_faces)
    checks.append((f"central-hole walls {list(central_hole_faces)} NOT claimed", central_ok,
                   "excluded" if central_ok else
                   f"leaked: {[i for i in central_hole_faces if i in result.claimed_faces]}"))

    ok = all(c[1] for c in checks)
    return PocketValidationReport(ok=ok, checks=checks)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_table(result: PocketDetectionResult) -> str:
    header = (f"{'id':>3}  {'subtype':<14} {'#f':>3} {'3D':>3} {'rel':>3} "
              f"{'walls':>5} {'seedF':>5} {'seedS':>5} {'step':>4} {'tmpl':>4}  "
              f"{'depth_in':>8} {'filR_in':>7}  {'centroid(u,v)':>18}  wall_diameters")
    lines = [header, "-" * len(header)]
    for f in result.features:
        wd = " ".join(f"{k}×{v}" for k, v in sorted(f.wall_diameters.items()))
        depth_in = (
            f"{f.depth_below_top_mm / MM_PER_INCH:.4f}"
            if f.depth_below_top_mm is not None else "—"
        )
        fil_in = (
            f"{f.fillet_radius_mm / MM_PER_INCH:.4f}"
            if f.fillet_radius_mm is not None else "—"
        )
        lines.append(
            f"{f.feature_id:>3}  {f.subtype:<14} {f.n_faces:>3} "
            f"{'Y' if f.surface_3d else 'N':>3} {len(f.released_faces):>3} "
            f"{f.wall_count:>5} {f.floor_count:>5} {f.sphere_count:>5} "
            f"{f.step_plane_count:>4} {'Y' if f.template_match else 'N':>4}  "
            f"{depth_in:>8} {fil_in:>7}  "
            f"({f.centroid_uv[0]:+7.2f},{f.centroid_uv[1]:+7.2f})  {wd}"
        )
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (OCC-free) — pure-Python clustering + attachment logic
# ---------------------------------------------------------------------------
@dataclass
class _FakeFace:
    index: int
    surface_type: str
    centroid: np.ndarray
    normal: np.ndarray
    radius: float | None = None
    axis: np.ndarray | None = None


def _run_selftest() -> bool:
    """OCC-free test of the spatial logic: clustering (knee), wall attachment by
    proximity, the wall-diameter band, and central-hole isolation.

    The interior/convex sign test needs a real OCC surface, so this test disables
    require_interior_wall and exercises only the geometry-independent paths (that
    filter is covered end-to-end by the reference part).
    """
    passed = True

    def check(name, cond):
        nonlocal passed
        passed = passed and cond
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    z = np.array([0.0, 0.0, 1.0])  # opening axis
    faces: list[_FakeFace] = []
    idx = 0

    def add(st, xy, r=None):
        nonlocal idx
        x, y = xy
        f = _FakeFace(idx, st, np.array([x, y, 0.0]), np.array([0.0, 0.0, 1.0]),
                      radius=r, axis=z.copy())
        faces.append(f)
        idx += 1
        return f.index

    # Two well-separated pockets (~120 mm apart), each with a 6-bspline + 7-sphere
    # floor signature and near walls. Plus a far central "hole" cylinder and one
    # oversized (blend) cylinder near a pocket.
    near_walls: list[int] = []
    for cx in (-60.0, 60.0):
        for dx in range(6):
            add("bspline", (cx + dx * 0.8 - 2, 0.0))
        for dx in range(7):
            add("sphere", (cx + dx * 0.8 - 2.4, 3.0))
        for dx in (-6, 6):
            near_walls.append(add("cylinder", (cx + dx, 1.0), r=6.0))  # ⌀12 mm
        near_walls.append(add("cylinder", (cx, -9.0), r=30.0))          # ⌀60 mm bore
        add("plane", (cx, -5.0))  # axial step floor (R1 keep predicate)
    # oversized blend cylinder near pocket 0 -> excluded by wall-diameter band
    blend = add("cylinder", (-60.0, 8.0), r=90.0)                        # ⌀180 mm
    # central hole cylinder far (~60 mm) from any floor -> unattached
    central = add("cylinder", (0.0, 0.0), r=25.0)                        # ⌀50 mm

    cfg = PocketDetectionConfig(
        opening_axis=(0.0, 0.0, 1.0), require_interior_wall=False,
        wall_families_in=(), template_floor_count=6, template_sphere_count=7,
        wall_dia_max_mm=100.0, wall_dia_min_mm=3.0, wall_attach_dist=20.0,
    )
    r = detect_pockets(faces, None, None, occ_faces=None, config=cfg)
    check("finds 2 pockets (knee)", len(r.features) == 2)
    check("each pocket claims its 3 near walls",
          all(f.wall_count == 3 for f in r.features))
    check("oversized blend cylinder excluded by diameter band",
          blend in r.blend_by_dia_faces and blend not in r.claimed_faces)
    check("central-far cylinder NOT claimed", central not in r.claimed_faces)
    check("central-far cylinder reported unattached", central in r.unattached_wall_faces)
    check("sculpted cluster floors retained in pocket claims",
          all(i in r.claimed_faces for i in range(len(faces))
              if faces[i].surface_type in SCULPTED_FLOOR_TYPES))
    check("floor/sphere seeds still clustered",
          sum(f.floor_count for f in r.features) == 12
          and sum(f.sphere_count for f in r.features) == 14)
    check("no cluster sculpted released to contour",
          all(len(f.released_faces) == 0 for f in r.features))

    # Case 11: R1 structure validation (OCC-free)
    def _case11_faces() -> tuple[list[_FakeFace], Any]:
        ff: list[_FakeFace] = []
        nxt = 0
        ax = np.array([0.0, 0.0, 1.0])

        def put(st: str, xy: tuple[float, float], *, r: float | None = None) -> int:
            nonlocal nxt
            x, y = xy
            f = _FakeFace(
                nxt, st, np.array([x, y, 0.0]), ax.copy(),
                radius=r, axis=ax.copy(),
            )
            ff.append(f)
            nxt += 1
            return f.index

        return ff, put

    cfg11 = PocketDetectionConfig(
        opening_axis=(0.0, 0.0, 1.0),
        require_interior_wall=False,
        wall_families_in=(),
        template_floor_count=1,
        template_sphere_count=0,
        wall_attach_dist=8.0,
        wall_dia_max_mm=100.0,
        wall_dia_min_mm=3.0,
    )

    # 11a: sculpted bspline band — no walls, no axial floor, no fillet → released
    band_faces, put = _case11_faces()
    for dx in range(3):
        put("bspline", (dx * 1.5, 0.0))
    r_band = detect_pockets(
        band_faces, None, None, occ_faces=None,
        config=replace(cfg11, hard_k=3),
    )
    check("Case11a structureless band not claimed", len(r_band.claimed_faces) == 0)
    check("Case11a structureless band has zero kept pockets", len(r_band.features) == 0)

    # 11b: walled + axial floor pocket → kept
    kept_faces, put = _case11_faces()
    put("bspline", (0.0, 0.0))
    put("cylinder", (3.0, 0.0), r=5.0)
    put("plane", (1.0, 0.0))
    r_kept = detect_pockets(
        kept_faces, None, None, occ_faces=None, config=replace(cfg11, hard_k=1),
    )
    check("Case11b walled+floored pocket kept", len(r_kept.features) == 1)
    check("Case11b walled+floored faces claimed", len(r_kept.claimed_faces) == 3)

    # 11c: filleted pocket with no walls (sphere blend only) → kept via fillet branch
    fillet_faces, put = _case11_faces()
    put("sphere", (0.0, 0.0), r=2.54)
    put("bspline", (1.0, 0.0))
    r_fillet = detect_pockets(
        fillet_faces, None, None, occ_faces=None,
        config=replace(cfg11, hard_k=1),
    )
    check("Case11c filleted no-wall pocket kept", len(r_fillet.features) == 1)
    check("Case11c fillet radius detected",
          r_fillet.features[0].fillet_radius_mm is not None
          and r_fillet.features[0].fillet_radius_mm > 0)

    print(f"\nself-test: {'ALL PASS' if passed else 'FAILURES PRESENT'}")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pocket-detection pass (spatial)")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ),
                    help="cached face graph (accepted for symmetry; grouping is spatial)")
    ap.add_argument("--chain-after-holes", action="store_true",
                    help="run hole_detection first and feed its remaining_faces as "
                         "this pass's candidate pool (LEGACY order; the reordered "
                         "cascade runs pockets FIRST — see run_cascade.py)")
    ap.add_argument("--wall-attach-dist", type=float, default=None,
                    help="override wall-attach distance (mm); default is adaptive")
    ap.add_argument("--selftest", action="store_true")
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

    edge_index = edge_attr = None
    if args.graph_npz.is_file():
        d = np.load(args.graph_npz)
        edge_index, edge_attr = d["edge_index"], d["edge_attr"]

    config = PocketDetectionConfig(wall_attach_dist=args.wall_attach_dist)

    candidate_faces = None
    if args.chain_after_holes:
        from hole_detection import detect_holes_from_step
        hres = detect_holes_from_step(step_path, edge_index, edge_attr)
        candidate_faces = set(hres.remaining_faces)
        print(f"[--chain-after-holes] hole pass left {len(candidate_faces)} faces")

    result = detect_pockets_from_step(
        step_path, edge_index, edge_attr,
        candidate_faces=candidate_faces, config=config,
    )

    print(f"STEP: {step_path}")
    print(f"opening axis: {result.opening_axis}  clusters: {result.n_clusters}  "
          f"wall_attach_dist: {result.wall_attach_dist:.2f} mm")
    print()
    print(render_table(result))
    print()
    report = validate_pockets(result)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
