"""
outer_fillet_detection.py — Cascade stage 4: outer-fillet detection.

Three roles (Toolpath-aligned):
  A) Stack-boundary fillets — opening-tier torus blends released by the hole pass
     at coaxial-stack caps (convex/concave to the opening flat).
  B) Structural exterior fillets — blend on the part exterior at opening tier
     (convex to large plane, smooth to cylinder, not pocket-enclosed).
  C) Hub-perimeter fillets — torus band outside the coaxial open-pocket floor
     radius (smooth to exterior cylinder, adjacent to hub pocket floor/walls).

Pass B skips candidates deeper than the opening tier. Pass C targets the hub
perimeter band using ``hub_perimeter_context`` from coaxial_stack_detection.
"""
from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from coaxial_stack_detection import HubPerimeterFilletContext
from feature_params import FaceGeom

logger = logging.getLogger("outer_fillet_detection")

CONVEXITY_NAMES = ("concave", "convex", "smooth")

# Blend pool in post-flats residual; torus is primary on reference parts.
DEFAULT_BLEND_SURFACE_TYPES = frozenset(
    {"torus", "cylinder", "cone", "sphere", "bspline", "bezier"},
)
# Rule (b): larger exterior neighbors for convex adjacency test.
DEFAULT_EXTERIOR_NEIGHBOR_TYPES = frozenset({"plane", "cylinder", "cone"})


@dataclass
class OuterFilletDetectionConfig:
    """Relational outer-fillet contract — no absolute R/area gates."""

    blend_surface_types: frozenset[str] = DEFAULT_BLEND_SURFACE_TYPES
    exterior_neighbor_types: frozenset[str] = DEFAULT_EXTERIOR_NEIGHBOR_TYPES
    smooth_neighbor_type: str = "cylinder"
    # Structural (part-exterior) fillets must sit within this margin of the
    # shallowest opening-tier reference depth on the part (from claimed flats or
    # stack-boundary fillets). Rejects back-side / deep-hub false positives.
    opening_tier_depth_margin_mm: float = 6.0
    opening_axis_parallel_tol_deg: float = 15.0
    units: str = "mm"


@dataclass
class OuterFilletFeature:
    feature_id: int
    kind: str  # always "outer_fillet"
    face_indices: set[int]
    area: float

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    @property
    def toolpath_class(self) -> str:
        return "outer_fillet"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "area": round(float(self.area), 3),
        }


@dataclass
class OuterFilletDetectionResult:
    features: list[OuterFilletFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    pool_size: int = 0
    hit_faces: list[int] = field(default_factory=list)
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} outer fillet(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain for residual pass"
        )


class _UF:
    def __init__(self, nodes: set[int]):
        self.p = {n: n for n in nodes}

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def components(self) -> list[list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for n in self.p:
            groups[self.find(n)].append(n)
        return sorted((sorted(v) for v in groups.values()), key=len, reverse=True)


def _edge_convexity(row: np.ndarray) -> str:
    cid = int(np.argmax(row[:3])) if row.shape[0] >= 3 else 2
    return CONVEXITY_NAMES[cid]


def _face_edges(
    fid: int,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == fid:
            out.append((_edge_convexity(edge_attr[k]), v))
        elif v == fid:
            out.append((_edge_convexity(edge_attr[k]), u))
    return out


def _rule_b_larger_exterior_convex(
    fid: int,
    *,
    by_index: dict[int, FaceGeom],
    pocket_claimed: set[int],
    edges: list[tuple[str, int]],
    exterior_types: frozenset[str],
) -> bool:
    fa = by_index[fid]
    for conv, nb in edges:
        if conv != "convex":
            continue
        fb = by_index[nb]
        if fb.area <= fa.area:
            continue
        if nb in pocket_claimed:
            continue
        if fb.surface_type in exterior_types:
            return True
    return False


def _rule_c_smooth_profile_cylinder(
    fid: int,
    *,
    by_index: dict[int, FaceGeom],
    pocket_claimed: set[int],
    edges: list[tuple[str, int]],
    smooth_neighbor_type: str,
) -> bool:
    fa = by_index[fid]
    for conv, nb in edges:
        if conv != "smooth":
            continue
        fb = by_index[nb]
        if fb.surface_type != smooth_neighbor_type:
            continue
        if fb.area <= fa.area:
            continue
        if nb in pocket_claimed:
            continue
        return True
    return False


def _rule_d_exterior_not_pocket_enclosed(
    *,
    pocket_claimed: set[int],
    edges: list[tuple[str, int]],
) -> bool:
    return not any(conv == "concave" and nb in pocket_claimed for conv, nb in edges)


def _structural_hits(
    *,
    pool: set[int],
    pocket_claimed: set[int],
    by_index: dict[int, FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    config: OuterFilletDetectionConfig,
) -> tuple[list[int], list[list[int]]]:
    blend_pool = sorted(
        i for i in pool if by_index[i].surface_type in config.blend_surface_types
    )
    hits: list[int] = []
    for fid in blend_pool:
        edges = _face_edges(fid, edge_index, edge_attr)
        if not _rule_b_larger_exterior_convex(
            fid,
            by_index=by_index,
            pocket_claimed=pocket_claimed,
            edges=edges,
            exterior_types=config.exterior_neighbor_types,
        ):
            continue
        if not _rule_c_smooth_profile_cylinder(
            fid,
            by_index=by_index,
            pocket_claimed=pocket_claimed,
            edges=edges,
            smooth_neighbor_type=config.smooth_neighbor_type,
        ):
            continue
        if not _rule_d_exterior_not_pocket_enclosed(
            pocket_claimed=pocket_claimed, edges=edges,
        ):
            continue
        hits.append(fid)

    hitset = set(hits)
    uf = _UF(hitset)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u not in hitset or v not in hitset:
            continue
        if _edge_convexity(edge_attr[k]) == "convex":
            uf.union(u, v)
    return hits, uf.components()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _face_depth_below_top_mm(
    face_idx: int,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
    part_axis_top: float,
) -> float:
    axis = _unit(opening_axis)
    c = np.asarray(by_index[face_idx].centroid, dtype=np.float64)
    return part_axis_top - float(np.dot(c, axis))


def _normal_parallel_opening(
    normal: np.ndarray | None,
    opening_axis: np.ndarray,
    tol_deg: float,
) -> bool:
    if normal is None:
        return False
    n = _unit(np.asarray(normal, dtype=float))
    a = _unit(opening_axis)
    return abs(float(np.dot(n, a))) >= math.cos(math.radians(tol_deg))


def _opening_tier_reference_depth_mm(
    *,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray | None,
    part_axis_top: float | None,
    flat_claimed_faces: set[int],
    stack_boundary_fillets: set[int],
) -> float | None:
    """Shallowest depth among opening-side flats and stack-boundary fillets."""
    if opening_axis is None or part_axis_top is None:
        return None
    depths: list[float] = []
    for idx in flat_claimed_faces:
        fg = by_index.get(idx)
        if fg is None or fg.surface_type != "plane":
            continue
        if _normal_parallel_opening(fg.normal, opening_axis, 15.0):
            depths.append(_face_depth_below_top_mm(idx, by_index, opening_axis, part_axis_top))
    for idx in stack_boundary_fillets:
        if idx in by_index:
            depths.append(
                _face_depth_below_top_mm(idx, by_index, opening_axis, part_axis_top)
            )
    return min(depths) if depths else None


def _hub_perimeter_fillet_hits(
    *,
    pool: set[int],
    hub_open_pocket_faces: set[int],
    pocket_claimed: set[int],
    by_index: dict[int, FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    hub_ctx: HubPerimeterFilletContext,
    config: OuterFilletDetectionConfig,
) -> tuple[list[int], list[list[int]]]:
    """Pass C — torus blends on the hub perimeter outside the open-pocket floor radius."""
    axis_point = np.asarray(hub_ctx.axis_point, dtype=float)
    axis_dir = np.asarray(hub_ctx.axis_direction, dtype=float)
    min_rad = hub_ctx.open_pocket_floor_radial_mm + hub_ctx.perimeter_radial_margin_mm
    anchor = hub_open_pocket_faces | pocket_claimed

    def _radial(fid: int) -> float:
        c = np.asarray(by_index[fid].centroid, dtype=float) - axis_point
        d = _unit(axis_dir)
        return float(np.linalg.norm(c - np.dot(c, d) * d))

    hits: list[int] = []
    for fid in sorted(pool):
        if by_index[fid].surface_type != "torus":
            continue
        if _radial(fid) <= min_rad:
            continue
        edges = _face_edges(fid, edge_index, edge_attr)
        if not _rule_c_smooth_profile_cylinder(
            fid,
            by_index=by_index,
            pocket_claimed=pocket_claimed,
            edges=edges,
            smooth_neighbor_type=config.smooth_neighbor_type,
        ):
            continue
        if not _rule_d_exterior_not_pocket_enclosed(
            pocket_claimed=pocket_claimed, edges=edges,
        ):
            continue
        if not any(nb in anchor for nb in (nb for _, nb in edges)):
            continue
        hits.append(fid)

    hitset = set(hits)
    uf = _UF(hitset)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u not in hitset or v not in hitset:
            continue
        kind = _edge_convexity(edge_attr[k])
        if kind in ("convex", "smooth"):
            uf.union(u, v)
    return hits, uf.components()


def detect_outer_fillets(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    pocket_claimed_faces: set[int] | None = None,
    stack_boundary_fillet_groups: list[set[int]] | None = None,
    flat_claimed_faces: set[int] | None = None,
    hub_open_pocket_faces: set[int] | None = None,
    hub_perimeter_context: HubPerimeterFilletContext | None = None,
    opening_axis: Sequence[float] | None = None,
    part_axis_top: float | None = None,
    excluded_faces: set[int] | None = None,
    config: OuterFilletDetectionConfig | None = None,
) -> OuterFilletDetectionResult:
    """Claim outer fillets: stack-boundary (Toolpath) first, then opening-tier structural."""
    config = config or OuterFilletDetectionConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    input_pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }
    pool = input_pool - (excluded_faces or set())
    pocket_claimed = pocket_claimed_faces or set()
    stack_boundary: set[int] = set()
    for grp in stack_boundary_fillet_groups or []:
        stack_boundary |= {i for i in grp if i in pool}
    flat_claimed = flat_claimed_faces or set()
    axis_arr = (
        _unit(np.asarray(opening_axis, dtype=float))
        if opening_axis is not None else None
    )

    features: list[OuterFilletFeature] = []
    claimed: set[int] = set()
    next_id = 0

    # Pass A — coaxial-stack boundary fillets released by the hole pass (one instance per stack).
    groups = stack_boundary_fillet_groups or []
    if not groups and stack_boundary:
        groups = [stack_boundary]
    for comp in groups:
        comp_set = {i for i in comp if i in pool}
        if not comp_set:
            continue
        total_area = sum(float(by_index[i].area) for i in comp_set)
        features.append(OuterFilletFeature(
            feature_id=next_id,
            kind="outer_fillet",
            face_indices=comp_set,
            area=total_area,
        ))
        claimed |= comp_set
        next_id += 1
        logger.info(
            "stack-boundary outer fillet %d: faces=%s area=%.1f mm²",
            next_id - 1, sorted(comp_set), total_area,
        )

    structural_pool = pool - claimed
    ref_depth = _opening_tier_reference_depth_mm(
        by_index=by_index,
        opening_axis=axis_arr,
        part_axis_top=part_axis_top,
        flat_claimed_faces=flat_claimed,
        stack_boundary_fillets=stack_boundary,
    )

    hits, components = _structural_hits(
        pool=structural_pool,
        pocket_claimed=pocket_claimed,
        by_index=by_index,
        edge_index=edge_index,
        edge_attr=edge_attr,
        config=config,
    )

    # Pass B — part-exterior structural fillets at opening tier only.
    for comp in components:
        comp_set = set(comp)
        if ref_depth is not None and axis_arr is not None and part_axis_top is not None:
            comp_depths = [
                _face_depth_below_top_mm(i, by_index, axis_arr, part_axis_top)
                for i in comp_set
            ]
            if min(comp_depths) > ref_depth + config.opening_tier_depth_margin_mm:
                logger.info(
                    "skip deep structural outer fillet (below opening tier): faces=%s "
                    "depths=%s ref=%.2f mm",
                    sorted(comp_set),
                    [round(d, 2) for d in comp_depths],
                    ref_depth,
                )
                continue
        total_area = sum(float(by_index[i].area) for i in comp_set)
        features.append(OuterFilletFeature(
            feature_id=next_id,
            kind="outer_fillet",
            face_indices=comp_set,
            area=total_area,
        ))
        claimed |= comp_set
        next_id += 1
        logger.info(
            "structural outer fillet %d: faces=%s area=%.1f mm²",
            next_id - 1, sorted(comp_set), total_area,
        )

    # Pass C — hub-perimeter torus band (outside coaxial open-pocket floor radius).
    hub_pool = input_pool - claimed
    if hub_perimeter_context is not None and hub_open_pocket_faces:
        hub_hits, hub_components = _hub_perimeter_fillet_hits(
            pool=hub_pool,
            hub_open_pocket_faces=hub_open_pocket_faces,
            pocket_claimed=pocket_claimed,
            by_index=by_index,
            edge_index=edge_index,
            edge_attr=edge_attr,
            hub_ctx=hub_perimeter_context,
            config=config,
        )
        for comp in hub_components:
            comp_set = set(comp)
            total_area = sum(float(by_index[i].area) for i in comp_set)
            features.append(OuterFilletFeature(
                feature_id=next_id,
                kind="outer_fillet",
                face_indices=comp_set,
                area=total_area,
            ))
            claimed |= comp_set
            next_id += 1
            logger.info(
                "hub-perimeter outer fillet %d: faces=%s area=%.1f mm²",
                next_id - 1, sorted(comp_set), total_area,
            )

    remaining = input_pool - claimed
    return OuterFilletDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        pool_size=len(pool),
        hit_faces=hits,
        units=config.units,
    )


def detect_outer_fillets_from_step(
    step_path: str | Path,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    pocket_claimed_faces: set[int] | None = None,
    config: OuterFilletDetectionConfig | None = None,
) -> OuterFilletDetectionResult:
    from feature_params import analyze_step, require_occ

    require_occ()
    faces = analyze_step(step_path)
    return detect_outer_fillets(
        faces, edge_index, edge_attr,
        candidate_faces=candidate_faces,
        pocket_claimed_faces=pocket_claimed_faces,
        config=config,
    )


@dataclass
class OuterFilletValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"outer fillet validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_outer_fillets(
    result: OuterFilletDetectionResult,
    *,
    expected_instances: int = 1,
    expected_faces: Sequence[int] | None = None,
) -> OuterFilletValidationReport:
    checks: list[tuple[str, bool, str]] = []
    n = len(result.features)
    checks.append((
        f"exactly {expected_instances} outer fillet(s)",
        n == expected_instances,
        f"got {n}",
    ))
    if expected_faces is not None:
        exp_set = set(expected_faces)
        got = set(result.claimed_faces)
        checks.append((
            f"claimed faces are {sorted(expected_faces)}",
            got == exp_set,
            f"got {sorted(got)}",
        ))
    checks.append((
        "remaining pool excludes claimed faces",
        not (result.claimed_faces & result.remaining_faces),
        f"overlap {sorted(result.claimed_faces & result.remaining_faces)}",
    ))
    ok = all(c[1] for c in checks)
    return OuterFilletValidationReport(ok=ok, checks=checks)


def render_table(result: OuterFilletDetectionResult) -> str:
    header = f"{'id':>3}  {'#f':>3} {'area (mm²)':>12}  faces"
    lines = [header, "-" * len(header)]
    for f in sorted(result.features, key=lambda x: -x.area):
        lines.append(
            f"{f.feature_id:>3}  {f.n_faces:>3} {f.area:>12.1f}  "
            f"{sorted(f.face_indices)}"
        )
    lines.append("")
    lines.append(f"blend hits (pre-group): {result.hit_faces}")
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


DEFAULT_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
REFERENCE_OUTER_FILLET_FACES_REAR = (111, 336)
REFERENCE_OUTER_FILLET_FACES_FRONT = (96, 287)
REFERENCE_HUB_OUTER_FILLET_FACES_FRONT = (278, 296)
REFERENCE_HUB_OUTER_FILLET_FACES_REAR = (327, 345)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Outer-fillet pass (cascade stage 4)")
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

    from run_cascade import _load_edges, run_cascade

    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    _, pk, hl, _, fl, of, _, _, _ = run_cascade(step_path, edge_index, edge_attr)
    result = of

    expected_faces = (
        REFERENCE_OUTER_FILLET_FACES_FRONT
        if "FRONT" in step_path.name.upper()
        else REFERENCE_OUTER_FILLET_FACES_REAR
    )

    print(f"STEP: {step_path}")
    print(f"input pool: {len(fl.remaining_faces)} faces (flat pass residual)")
    print()
    print(render_table(result))
    print()
    print(validate_outer_fillets(
        result, expected_instances=1, expected_faces=expected_faces,
    ).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
