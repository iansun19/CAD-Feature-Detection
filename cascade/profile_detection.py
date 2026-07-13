"""
profile_detection.py — Cascade stage 7: hub step profile cylinders.

Toolpath profile on the reference plate is a single opposing vertical-cylinder
pair (faces 279, 297): two convex-adjacent cylinders with the same diameter,
parallel to the opening axis, facing each other (outward normals oppose). They
sit inboard of the exterior wall band and are not concave into pocket/hole
interior.

Units are mm. numpy only.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cascade.exterior_boundary import (
    ExteriorBoundaryConfig,
    compute_exterior_envelope,
    face_diameter_mm,
    interior_claimed,
    radial_mm,
    axial_mm,
    axis_parallel,
)
from brep.feature_params import FaceGeom
from cascade.hole_detection import FaceGraph, _unit

logger = logging.getLogger("profile_detection")

# Reference-part regression (96260B_front) — oracles only, not algorithm input.
REFERENCE_PROFILE_FACES_FRONT = frozenset({279, 297})
REFERENCE_PROFILE_INSTANCES_FRONT = 1


@dataclass
class ProfileDetectionConfig:
    opening_axis_parallel_tol_deg: float = 15.0
    opposing_normal_dot_max: float = -0.95
    diameter_match_abs_tol_mm: float = 0.5
    radial_match_abs_tol_mm: float = 1.0
    # Profiles sit inboard of the OD wall band (reference pair at ~66% of R_ext).
    max_exterior_radial_fraction: float = 0.95
    units: str = "mm"


@dataclass
class ProfileFeature:
    feature_id: int
    kind: str
    face_indices: set[int]
    nominal_diameter_mm: float
    radial_mm: float
    axial_span_mm: float

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def toolpath_class(self) -> str:
        return "profile"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "nominal_diameter_mm": round(float(self.nominal_diameter_mm), 3),
            "radial_mm": round(float(self.radial_mm), 3),
            "axial_span_mm": round(float(self.axial_span_mm), 3),
        }


@dataclass
class ProfileDetectionResult:
    features: list[ProfileFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    pool_size: int = 0
    exterior_radius_mm: float = 0.0
    exterior_diameter_mm: float = 0.0
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} profile(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain for residual pass"
        )


def _normals_oppose(
    fa: FaceGeom,
    fb: FaceGeom,
    *,
    dot_max: float,
) -> bool:
    if fa.normal is None or fb.normal is None:
        return False
    na = _unit(np.asarray(fa.normal, dtype=float))
    nb = _unit(np.asarray(fb.normal, dtype=float))
    return float(np.dot(na, nb)) <= dot_max


def _is_vertical_cylinder(
    f: FaceGeom,
    opening_axis: np.ndarray,
    config: ProfileDetectionConfig,
) -> bool:
    if f.surface_type != "cylinder" or f.radius is None:
        return False
    return axis_parallel(f.axis, opening_axis, config.opening_axis_parallel_tol_deg)


def _concave_into_interior(
    fid: int,
    graph: FaceGraph,
    interior: set[int],
) -> bool:
    for nb in graph.neighbors.get(fid, ()):
        if nb in interior and graph.edge_kind(fid, nb) == "concave":
            return True
    return False


def find_opposing_profile_pairs(
    pool: set[int],
    *,
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    interior: set[int],
    r_exterior: float,
    config: ProfileDetectionConfig,
) -> list[tuple[int, int]]:
    """Return sorted (a, b) pairs satisfying the profile-cylinder rule."""
    radial_ceiling = r_exterior * config.max_exterior_radial_fraction
    cyls = [
        fid for fid in pool
        if _is_vertical_cylinder(by_index[fid], opening_axis, config)
    ]
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for a in cyls:
        fa = by_index[a]
        da = face_diameter_mm(fa)
        if da is None:
            continue
        ra = radial_mm(fa.centroid, opening_axis)
        if ra > radial_ceiling:
            continue
        if _concave_into_interior(a, graph, interior):
            continue

        for b in graph.neighbors.get(a, ()):
            if b not in pool or b <= a:
                continue
            if (a, b) in seen:
                continue
            if graph.edge_kind(a, b) != "convex":
                continue
            fb = by_index[b]
            if not _is_vertical_cylinder(fb, opening_axis, config):
                continue
            db = face_diameter_mm(fb)
            if db is None or abs(da - db) > config.diameter_match_abs_tol_mm:
                continue
            rb = radial_mm(fb.centroid, opening_axis)
            if abs(ra - rb) > config.radial_match_abs_tol_mm:
                continue
            if rb > radial_ceiling:
                continue
            if _concave_into_interior(b, graph, interior):
                continue
            if not _normals_oppose(fa, fb, dot_max=config.opposing_normal_dot_max):
                continue
            seen.add((a, b))
            pairs.append((a, b))

    return pairs


def detect_profiles(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    pocket_claimed_faces: set[int] | None = None,
    hole_claimed_faces: set[int] | None = None,
    hub_open_pocket_faces: set[int] | None = None,
    wall_claimed_faces: set[int] | None = None,
    wall_seed_faces: set[int] | None = None,
    opening_axis: np.ndarray | None = None,
    config: ProfileDetectionConfig | None = None,
) -> ProfileDetectionResult:
    """Claim opposing inboard vertical-cylinder profile pair(s)."""
    del wall_seed_faces  # wall pass runs first; profile rule is independent of seeds
    config = config or ProfileDetectionConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }
    pocket_claimed = set(pocket_claimed_faces or ())
    hole_claimed = set(hole_claimed_faces or ())
    hub_open = set(hub_open_pocket_faces or ())
    wall_claimed = set(wall_claimed_faces or ())
    interior = {
        i for i in range(n_faces)
        if interior_claimed(
            i,
            pocket_claimed=pocket_claimed,
            hole_claimed=hole_claimed,
            hub_open_pocket_faces=hub_open,
        )
    }

    if opening_axis is None:
        opening_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    opening_axis = np.asarray(opening_axis, dtype=float)

    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)
    env_config = ExteriorBoundaryConfig()
    r_ext, d_ext = compute_exterior_envelope(
        pool | wall_claimed, by_index, opening_axis, env_config,
    )

    pairs = find_opposing_profile_pairs(
        pool,
        by_index=by_index,
        graph=graph,
        opening_axis=opening_axis,
        interior=interior,
        r_exterior=r_ext,
        config=config,
    )
    logger.info(
        "profile opposing-cylinder pairs: %d (R_ext=%.1f mm)",
        len(pairs), r_ext,
    )

    features: list[ProfileFeature] = []
    claimed: set[int] = set()
    for fid, (a, b) in enumerate(pairs):
        face_set = {a, b}
        fa, fb = by_index[a], by_index[b]
        nom_d = face_diameter_mm(fa) or 0.0
        axial = [
            axial_mm(by_index[i].centroid, opening_axis)
            for i in face_set
        ]
        feat = ProfileFeature(
            feature_id=fid,
            kind="profile",
            face_indices=face_set,
            nominal_diameter_mm=float(nom_d),
            radial_mm=max(radial_mm(fa.centroid, opening_axis),
                          radial_mm(fb.centroid, opening_axis)),
            axial_span_mm=float(max(axial) - min(axial)) if axial else 0.0,
        )
        features.append(feat)
        claimed |= face_set
        logger.info(
            "profile %d: faces=%s D=%.1f mm R=%.1f mm",
            fid, sorted(face_set), nom_d, feat.radial_mm,
        )

    remaining = pool - claimed
    return ProfileDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        pool_size=len(pool),
        exterior_radius_mm=r_ext,
        exterior_diameter_mm=float(d_ext or 0.0),
        units=config.units,
    )


@dataclass
class ProfileValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"profile validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_profiles(
    result: ProfileDetectionResult,
    *,
    expected_instances: int = REFERENCE_PROFILE_INSTANCES_FRONT,
    expected_faces: Sequence[int] | None = None,
    forbidden_faces: Sequence[int] | None = None,
) -> ProfileValidationReport:
    if expected_faces is None:
        expected_faces = sorted(REFERENCE_PROFILE_FACES_FRONT)
    expected_set = set(expected_faces)
    forbidden = set(forbidden_faces or ())

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        f"instances == {expected_instances}",
        len(result.features) == expected_instances,
        f"got {len(result.features)}",
    ))
    two_face_instances = all(len(f.face_indices) == 2 for f in result.features)
    checks.append((
        "two faces per profile instance",
        two_face_instances or not result.features,
        "ok" if two_face_instances or not result.features else "unexpected face count",
    ))
    claimed = set()
    overlap = False
    for feat in result.features:
        if claimed & feat.face_indices:
            overlap = True
        claimed |= feat.face_indices
    checks.append((
        "disjoint instances",
        not overlap,
        "overlap" if overlap else f"{len(result.features)} instance(s)",
    ))
    missing = sorted(expected_set - claimed)
    extra = sorted(claimed - expected_set)
    checks.append((
        "expected faces claimed",
        not missing,
        f"missing {missing}" if missing else f"{len(expected_set)} faces",
    ))
    if expected_set:
        checks.append((
            "no extra faces in profile",
            not extra,
            f"extra {extra}" if extra else "exact match",
        ))
    bad = sorted(claimed & forbidden)
    checks.append((
        "forbidden faces excluded",
        not bad,
        f"forbidden present {bad}" if bad else "ok",
    ))
    ok = all(c[1] for c in checks)
    return ProfileValidationReport(ok=ok, checks=checks)


def render_table(result: ProfileDetectionResult) -> str:
    header = f"{'id':>3}  {'#f':>3}  {'D_mm':>8}  {'R_mm':>8}  {'ax_span':>8}  faces"
    lines = [header, "-" * len(header)]
    for feat in result.features:
        face_s = ",".join(str(i) for i in sorted(feat.face_indices))
        lines.append(
            f"{feat.feature_id:>3}  {feat.n_faces:>3}  "
            f"{feat.nominal_diameter_mm:>8.1f}  {feat.radial_mm:>8.1f}  "
            f"{feat.axial_span_mm:>8.1f}  {face_s}"
        )
    lines.append("")
    lines.append(
        f"envelope: R={result.exterior_radius_mm:.1f} mm "
        f"D={result.exterior_diameter_mm:.1f} mm  pool={result.pool_size}"
    )
    lines.append(result.summary())
    return "\n".join(lines)


DEFAULT_STEP = "fixtures/step/96260B_front.stp"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_front/graph.npz"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile detection pass (cascade stage 7)")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    step_path = Path(args.step)
    if not step_path.is_file():
        print(f"STEP not found: {step_path}")
        return 2

    from cascade.pocket_detection import PocketDetectionConfig, pocket_config_from_setup_dict
    from run_cascade import _load_edges, run_cascade

    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    setup = pocket_config_from_setup_dict({"setup": {"machining_side": "front"}})
    _, pk, hl, cx, fl, of, wl, pr, _rs = run_cascade(
        step_path, edge_index, edge_attr,
        pocket_config=PocketDetectionConfig(setup=setup),
    )

    print(f"STEP: {step_path}")
    print(f"input pool: {len(wl.remaining_faces)} faces (after wall pass)")
    print()
    print(render_table(pr))
    print()
    print(validate_profiles(pr).render())
    return 0 if validate_profiles(pr).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
