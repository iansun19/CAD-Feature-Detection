"""
wall_detection.py — Cascade stage 6: exterior notch wall segments.

Claims vertical OD cylinder slivers at the part's maximum exterior diameter —
one Toolpath wall instance per seed (reference plate: 14 faces).

Runs before profile so lobe sculpt can grow inboard from the same seeds without
double-claiming wall faces.

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
    axial_mm,
    face_diameter_mm,
    find_exterior_wall_seeds,
    interior_claimed,
    radial_mm,
)
from brep.feature_params import FaceGeom
from cascade.hole_detection import FaceGraph

logger = logging.getLogger("wall_detection")

# Reference-part regression (96260B_front) — oracles only, not algorithm input.
REFERENCE_WALL_FACES_FRONT = frozenset({
    4, 11, 19, 24, 32, 37, 45, 50, 58, 63, 71, 76, 82, 87,
})
REFERENCE_WALL_INSTANCES_FRONT = 14


@dataclass
class WallDetectionConfig(ExteriorBoundaryConfig):
    pass


@dataclass
class WallFeature:
    feature_id: int
    kind: str
    face_indices: set[int]
    nominal_diameter_mm: float
    radial_mm: float

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def toolpath_class(self) -> str:
        return "wall"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "nominal_diameter_mm": round(float(self.nominal_diameter_mm), 3),
            "radial_mm": round(float(self.radial_mm), 3),
        }


@dataclass
class WallDetectionResult:
    features: list[WallFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    seed_faces: set[int]
    n_faces: int
    pool_size: int = 0
    exterior_radius_mm: float = 0.0
    exterior_diameter_mm: float = 0.0
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} wall(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


def detect_walls(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    pocket_claimed_faces: set[int] | None = None,
    hole_claimed_faces: set[int] | None = None,
    hub_open_pocket_faces: set[int] | None = None,
    opening_axis: np.ndarray | None = None,
    config: WallDetectionConfig | None = None,
) -> WallDetectionResult:
    """Claim exterior OD notch wall segments (one instance per seed face)."""
    config = config or WallDetectionConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }
    pocket_claimed = set(pocket_claimed_faces or ())
    hole_claimed = set(hole_claimed_faces or ())
    hub_open = set(hub_open_pocket_faces or ())
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
    seeds, r_ext, d_ext = find_exterior_wall_seeds(
        pool, by_index, graph, opening_axis, interior, config,
    )
    logger.info(
        "wall envelope R=%.1f mm D=%.1f mm; %d seeds",
        r_ext, d_ext or 0.0, len(seeds),
    )

    features: list[WallFeature] = []
    claimed: set[int] = set()
    for fid, seed in enumerate(sorted(seeds)):
        f = by_index[seed]
        d = face_diameter_mm(f) or (d_ext or 0.0)
        feat = WallFeature(
            feature_id=fid,
            kind="wall",
            face_indices={seed},
            nominal_diameter_mm=float(d),
            radial_mm=radial_mm(f.centroid, opening_axis),
        )
        features.append(feat)
        claimed.add(seed)

    remaining = pool - claimed
    return WallDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        seed_faces=seeds,
        n_faces=n_faces,
        pool_size=len(pool),
        exterior_radius_mm=r_ext,
        exterior_diameter_mm=float(d_ext or 0.0),
        units=config.units,
    )


@dataclass
class WallValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"wall validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_walls(
    result: WallDetectionResult,
    *,
    expected_instances: int = REFERENCE_WALL_INSTANCES_FRONT,
    expected_faces: Sequence[int] | None = None,
    forbidden_faces: Sequence[int] | None = None,
) -> WallValidationReport:
    if expected_faces is None:
        expected_faces = sorted(REFERENCE_WALL_FACES_FRONT)
    expected_set = set(expected_faces)
    forbidden = set(forbidden_faces or ())

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        f"instances == {expected_instances}",
        len(result.features) == expected_instances,
        f"got {len(result.features)}",
    ))
    one_face_each = all(len(f.face_indices) == 1 for f in result.features)
    checks.append((
        "one face per wall instance",
        one_face_each,
        "ok" if one_face_each else "multi-face wall(s) found",
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
            "exact face set",
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
    return WallValidationReport(ok=ok, checks=checks)


def render_table(result: WallDetectionResult) -> str:
    header = f"{'id':>3}  {'face':>4}  {'D_mm':>8}  {'R_mm':>8}"
    lines = [header, "-" * len(header)]
    for feat in sorted(result.features, key=lambda f: min(f.face_indices)):
        face = min(feat.face_indices)
        lines.append(
            f"{feat.feature_id:>3}  {face:>4}  "
            f"{feat.nominal_diameter_mm:>8.1f}  {feat.radial_mm:>8.1f}"
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
    ap = argparse.ArgumentParser(description="Wall detection pass (cascade stage 6)")
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
    _, pk, hl, cx, fl, of, wl, _pr, _rs = run_cascade(
        step_path, edge_index, edge_attr,
        pocket_config=PocketDetectionConfig(setup=setup),
    )

    print(f"STEP: {step_path}")
    print(f"input pool: {len(of.remaining_faces)} faces")
    print()
    print(render_table(wl))
    print()
    print(validate_walls(wl).render())
    return 0 if validate_walls(wl).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
