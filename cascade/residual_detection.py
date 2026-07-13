"""
residual_detection.py — Cascade stage 7: coarse residual grouping.

After pockets, holes, flats, outer fillets, and profile, every still-unclaimed face
lands in exactly one ``contour_surface`` instance. Walls are NOT separated here —
the GNN may refine those labels downstream.

  * CONNECT across convex AND smooth internal edges
  * CUT across concave internal edges (feature separator, same as pocket/hole)

Units are mm. numpy only.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cascade.hole_detection import FaceGraph
from brep.instance_features import instance_features

logger = logging.getLogger("residual_detection")

CONVEXITY_NAMES = ("concave", "convex", "smooth")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ResidualDetectionConfig:
    units: str = "mm"


# ---------------------------------------------------------------------------
# Feature containers
# ---------------------------------------------------------------------------
@dataclass
class ResidualCandidateFeature:
    feature_id: int
    kind: str  # always "contour_surface"
    face_indices: set[int]
    toolpath_class: str  # always "contour_surface"
    params: dict[str, Any]

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            **self.params,
        }


@dataclass
class ResidualDetectionResult:
    features: list[ResidualCandidateFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    n_convex_only_components: int = 0
    largest_convex_only_size: int = 0
    n_concave_cut_components: int = 0
    component_sizes: list[int] = field(default_factory=list)
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} residual candidates; claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


# ---------------------------------------------------------------------------
# Union-find (inlined)
# ---------------------------------------------------------------------------
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

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for n in self.p:
            groups[self.find(n)].append(n)
        return dict(groups)


def _edge_convexity(row: np.ndarray) -> str:
    cid = int(np.argmax(row[:3])) if row.shape[0] >= 3 else 2
    return CONVEXITY_NAMES[cid]


def _count_convex_only_components(
    pool: set[int],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> tuple[int, int]:
    """Return (n_components, largest_size) for convex-only connectivity."""
    uf = _UF(pool)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u not in pool or v not in pool:
            continue
        if _edge_convexity(edge_attr[k]) == "convex":
            uf.union(u, v)
    sizes = sorted((len(c) for c in uf.components().values()), reverse=True)
    return len(sizes), (sizes[0] if sizes else 0)


def _group_concave_cut(
    pool: set[int],
    graph: FaceGraph,
) -> list[list[int]]:
    """Union-find over pool: union on convex+smooth internal edges only."""
    uf = _UF(pool)
    for f in pool:
        for nb in graph.neighbors.get(f, set()):
            if nb not in pool or nb <= f:
                continue
            kind = graph.edge_kind(f, nb)
            if kind in ("convex", "smooth"):
                uf.union(f, nb)
    return sorted(uf.components().values(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_residual_candidates(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    config: ResidualDetectionConfig | None = None,
) -> ResidualDetectionResult:
    """Group the flat-pass residual into coarse candidate instances."""
    config = config or ResidualDetectionConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }

    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    n_convex, largest_convex = _count_convex_only_components(pool, edge_index, edge_attr)
    components = _group_concave_cut(pool, graph)
    n_cut = len(components)

    logger.info(
        "concave-cut grouping: %d components (convex-only would be %d, largest %d)",
        n_cut, n_convex, largest_convex,
    )

    features: list[ResidualCandidateFeature] = []
    claimed: set[int] = set()
    for fid, comp in enumerate(components):
        comp_set = set(comp)
        comp_faces = [by_index[i] for i in comp]
        params = instance_features(comp_faces, graph=graph, face_indices=comp_set)
        feat = ResidualCandidateFeature(
            feature_id=fid,
            kind="contour_surface",
            face_indices=comp_set,
            toolpath_class="contour_surface",
            params=params,
        )
        features.append(feat)
        claimed |= comp_set
        logger.info(
            "candidate %d: n_faces=%d hist=%s area=%.1f mm²",
            fid, feat.n_faces, params["surface_type_histogram"], params["total_area"],
        )

    remaining = pool - claimed
    return ResidualDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        n_convex_only_components=n_convex,
        largest_convex_only_size=largest_convex,
        n_concave_cut_components=n_cut,
        component_sizes=[len(c) for c in components],
        units=config.units,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# Post-profile pass residual pool size (profile output -> residual pass input).
# STOCK gate removed (2026-07-10): every face enters the cascade, so the residual
# pass is the terminal partition owner and must claim its whole input pool. On the
# rear reference part the profile pass now claims the stray ⌀6.370 bosses {328, 346}
# (previously left to residual), so the residual pool is 102, not the old 160.
REFERENCE_RESIDUAL_POOL_REAR = 102
# Post-profile pass residual pool size on the front reference part. STOCK gate
# removed (2026-07-10): every face enters the cascade, so the residual pass owns
# the terminal partition and claims its whole 74-face input pool (was 123 under
# the old stock-gated partition).
REFERENCE_RESIDUAL_POOL_FRONT = 74
REFERENCE_SANITY_FACES_REAR = {
    # {328, 346} moved to the profile pass; residual now owns only the zero-cones.
    "zero-cones": (121, 137, 153),
}
REFERENCE_SANITY_FACES_FRONT = {
    "perimeter torus": (105,),
}


@dataclass
class ResidualValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"residual validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_residual(
    result: ResidualDetectionResult,
    *,
    expected_pool_size: int = REFERENCE_RESIDUAL_POOL_REAR,
    expected_total_faces: int = 348,
    sanity_faces: dict[str, Sequence[int]] | None = None,
) -> ResidualValidationReport:
    """Reference-part contract: claim entire pool; key deferred faces present."""
    if sanity_faces is None:
        sanity_faces = {
            # {328, 346} now claimed by the profile pass; residual owns the zero-cones.
            "zero-cones": (121, 137, 153),
        }

    checks: list[tuple[str, bool, str]] = []
    n_claimed = len(result.claimed_faces)
    n_remain = len(result.remaining_faces)

    checks.append((
        f"claimed == {expected_pool_size}",
        n_claimed == expected_pool_size,
        f"got {n_claimed}",
    ))
    checks.append((
        "remaining == 0",
        n_remain == 0,
        f"got {n_remain}",
    ))
    checks.append((
        f"concave-cut splits convex blob ({result.largest_convex_only_size} faces)",
        result.n_concave_cut_components > 1,
        f"{result.n_concave_cut_components} components "
        f"(convex-only: {result.n_convex_only_components}, "
        f"largest {result.largest_convex_only_size})",
    ))

    all_claimed = set()
    overlap = False
    for feat in result.features:
        if all_claimed & feat.face_indices:
            overlap = True
            break
        all_claimed |= feat.face_indices
    checks.append((
        "single ownership (no overlapping instances)",
        not overlap and all_claimed == result.claimed_faces,
        "overlap detected" if overlap else f"{len(result.features)} disjoint instances",
    ))

    for label, idxs in sanity_faces.items():
        missing = [i for i in idxs if i not in result.claimed_faces]
        checks.append((
            f"{label} {list(idxs)} claimed",
            not missing,
            f"missing {missing}" if missing else "all present",
        ))

    ok = all(c[1] for c in checks)
    return ResidualValidationReport(ok=ok, checks=checks)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_table(result: ResidualDetectionResult) -> str:
    header = (
        f"{'id':>3}  {'#f':>3}  {'area':>10}  {'axes':>4}  "
        f"{'concave':>7} {'convex':>7} {'smooth':>6}  surface histogram"
    )
    lines = [header, "-" * len(header)]
    for f in sorted(result.features, key=lambda x: -x.n_faces):
        p = f.params
        ec = p.get("internal_edge_convexity", {})
        hist = ", ".join(
            f"{k}×{v}" for k, v in sorted(
                p.get("surface_type_histogram", {}).items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        lines.append(
            f"{f.feature_id:>3}  {f.n_faces:>3}  {p['total_area']:>10.1f}  "
            f"{p['n_distinct_axes']:>4}  "
            f"{ec.get('concave', 0):>7} {ec.get('convex', 0):>7} "
            f"{ec.get('smooth', 0):>6}  {hist}"
        )
    lines.append("")
    lines.append(
        f"concave-cut: {result.n_concave_cut_components} components, "
        f"sizes={result.component_sizes[:12]}"
        f"{'...' if len(result.component_sizes) > 12 else ''}"
    )
    lines.append(
        f"convex-only (reference): {result.n_convex_only_components} components, "
        f"largest={result.largest_convex_only_size}"
    )
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_STEP = "fixtures/step/96260B_rear.stp"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Residual grouping pass (cascade stage 4)")
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
    _, pk, hl, _, fl, of, wl, pr, rs = run_cascade(step_path, edge_index, edge_attr)
    result = rs

    print(f"STEP: {step_path}")
    print(f"input pool: {len(pr.remaining_faces)} faces (profile pass residual)")
    print()
    print(render_table(result))
    print()
    print(validate_residual(result).render())
    return 0 if validate_residual(result).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
