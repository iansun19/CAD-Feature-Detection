"""probe_contour_census.py — census the post-flats residual before contour/fillet/profile.

Run: /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_contour_census.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from feature_params import analyze_step
from run_cascade import _load_edges, run_cascade

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
MM_PER_IN = 25.4
CONVEXITY_NAMES = ("concave", "convex", "smooth")

# Deferred faces that SHOULD remain in the contour residual.
EXPECTED_DEFERRED = {
    "central end caps": [330, 332],
    "stray ⌀6.370 bosses": [328, 346],
    "zero-cones": [121, 137, 153],
}

# Red-flag pocket-wall diameters (inches) — must NOT appear in residual.
POCKET_WALL_DIA_IN = (0.800, 0.500, 3.453)
# Red-flag central-hole diameters (inches) — must NOT appear in residual.
CENTRAL_HOLE_DIA_IN = (4.006, 3.200)
DIA_TOL_IN = 0.02


def dia_in(f) -> float | None:
    return (2.0 * f.radius) / MM_PER_IN if f.radius else None


def near_dia(d: float | None, target: float, tol: float = DIA_TOL_IN) -> bool:
    return d is not None and abs(d - target) < tol


def edge_convexity(row: np.ndarray) -> str:
    cid = int(np.argmax(row[:3])) if row.shape[0] >= 3 else 2
    return CONVEXITY_NAMES[cid]


class _UF:
    """Inlined union-find over arbitrary hashable nodes."""

    def __init__(self, nodes):
        self.p = {n: n for n in nodes}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def components(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for n in self.p:
            groups[self.find(n)].append(n)
        return dict(groups)


def surface_hist(idxs: list[int], by_index: dict) -> dict[str, int]:
    c: Counter[str] = Counter()
    for i in idxs:
        c[by_index[i].surface_type] += 1
    return dict(c)


def fmt_hist(d: dict, *, key_fmt=str) -> str:
    return ", ".join(f"{key_fmt(k)}={v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# Load cascade handoff: post-flats residual
# ---------------------------------------------------------------------------
faces = analyze_step(STEP_PATH)
by_index = {f.index: f for f in faces}
edge_index, edge_attr = _load_edges(Path(GRAPH_NPZ), Path(STEP_PATH))
_, pk, hl, fl, _, _ = run_cascade(STEP_PATH, edge_index, edge_attr)
residual = set(int(i) for i in fl.remaining_faces)

print("=" * 78)
print("POST-FLATS RESIDUAL CENSUS (contour / outer fillet / profile preview)")
print("=" * 78)
print(f"STEP: {STEP_PATH}")
print(f"Toolpath targets in this pool: contour ~43, outer fillets 1, profile 1")

# ---------------------------------------------------------------------------
# Block 1 — residual size + surface-type + cyl/cone diameter histogram
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK 1 — RESIDUAL SIZE + SURFACE TYPES + CYL/CONE DIAMETERS")
print("=" * 78)
print(f"residual size: {len(residual)} faces")

stype_hist = Counter(by_index[i].surface_type for i in residual)
print(f"surface-type histogram: {dict(stype_hist)}")

dia_hist: Counter[str] = Counter()
for i in sorted(residual):
    f = by_index[i]
    if f.surface_type not in ("cylinder", "cone"):
        continue
    d = dia_in(f)
    if d is None:
        dia_hist["n/a"] += 1
    elif abs(d) < 1e-4:
        dia_hist["⌀0.000in"] += 1
    else:
        dia_hist[f"⌀{d:.3f}in"] += 1
print(f"cyl/cone diameter histogram ({sum(dia_hist.values())} faces):")
for k, v in dia_hist.most_common():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# Block 2 — internal-residual edge convexity histogram
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK 2 — INTERNAL-RESIDUAL EDGE CONVEXITY")
print("=" * 78)
conv_hist: Counter[str] = Counter()
internal_edges = 0
for k in range(edge_index.shape[1]):
    u, v = int(edge_index[0, k]), int(edge_index[1, k])
    if u not in residual or v not in residual:
        continue
    internal_edges += 1
    conv_hist[edge_convexity(edge_attr[k])] += 1
print(f"internal edges (both endpoints in residual): {internal_edges}")
print(
    f"convexity: concave={conv_hist['concave']}, "
    f"convex={conv_hist['convex']}, smooth={conv_hist['smooth']}"
)

# ---------------------------------------------------------------------------
# Block 3 — SMOOTH-connected components (contour grouping preview)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK 3 — SMOOTH-CONNECTED COMPONENTS (contour tangency preview)")
print("=" * 78)

uf_smooth = _UF(residual)
for k in range(edge_index.shape[1]):
    u, v = int(edge_index[0, k]), int(edge_index[1, k])
    if u not in residual or v not in residual:
        continue
    if edge_convexity(edge_attr[k]) == "smooth":
        uf_smooth.union(u, v)

smooth_comps = sorted(uf_smooth.components().values(), key=len, reverse=True)
print(f"component count: {len(smooth_comps)}")
print(f"sorted sizes: {[len(c) for c in smooth_comps]}")
print(f"\nlargest ~{min(8, len(smooth_comps))} components:")
for ci, comp in enumerate(smooth_comps[:8]):
    comp_sorted = sorted(comp)
    comp_stypes = surface_hist(comp_sorted, by_index)
    print(
        f"  comp {ci}: size={len(comp)}  composition={comp_stypes}  "
        f"faces={comp_sorted}"
    )

# ---------------------------------------------------------------------------
# Block 4 — CONVEX-connected components (outer-fillet candidates)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK 4 — CONVEX-CONNECTED COMPONENTS (outer-fillet candidates)")
print("=" * 78)

uf_convex = _UF(residual)
for k in range(edge_index.shape[1]):
    u, v = int(edge_index[0, k]), int(edge_index[1, k])
    if u not in residual or v not in residual:
        continue
    if edge_convexity(edge_attr[k]) == "convex":
        uf_convex.union(u, v)

convex_comps = sorted(uf_convex.components().values(), key=len, reverse=True)
print(f"component count: {len(convex_comps)}")
print(f"sorted sizes: {[len(c) for c in convex_comps]}")
for ci, comp in enumerate(convex_comps[:8]):
    comp_sorted = sorted(comp)
    comp_stypes = surface_hist(comp_sorted, by_index)
    print(
        f"  comp {ci}: size={len(comp)}  composition={comp_stypes}  "
        f"faces={comp_sorted}"
    )

# ---------------------------------------------------------------------------
# Block 5 — sanity / leak checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK 5 — SANITY / LEAK CHECKS")
print("=" * 78)

red_flags: list[str] = []

print("expected deferred faces (SHOULD be in residual):")
for label, idxs in EXPECTED_DEFERRED.items():
    present = [i for i in idxs if i in residual]
    missing = [i for i in idxs if i not in residual]
    status = "OK" if not missing else "MISSING"
    print(f"  {label}: present={present}  missing={missing}  [{status}]")
    if missing:
        red_flags.append(f"expected-deferred missing: {label} {missing}")

print("\nred-flag scan (must NOT be in residual):")
pocket_wall_hits: list[int] = []
for i in sorted(residual):
    f = by_index[i]
    if f.surface_type not in ("cylinder", "cone"):
        continue
    d = dia_in(f)
    if any(near_dia(d, t) for t in POCKET_WALL_DIA_IN):
        pocket_wall_hits.append(i)
if pocket_wall_hits:
    print(f"  *** POCKET WALL LEAK (⌀0.800/0.500/3.453in): {pocket_wall_hits} ***")
    red_flags.append(f"pocket-wall leak: {pocket_wall_hits}")
else:
    print("  pocket walls (⌀0.800/0.500/3.453in): none  [OK]")

central_hole_hits: list[int] = []
for i in sorted(residual):
    f = by_index[i]
    if f.surface_type not in ("cylinder", "cone"):
        continue
    d = dia_in(f)
    if any(near_dia(d, t) for t in CENTRAL_HOLE_DIA_IN):
        central_hole_hits.append(i)
if central_hole_hits:
    print(
        f"  *** CENTRAL HOLE LEAK (⌀101.752/⌀81.28 mm): {central_hole_hits} ***"
    )
    red_flags.append(f"central-hole leak: {central_hole_hits}")
else:
    print("  central holes (⌀101.752/⌀81.28 mm): none  [OK]")

pocket_floor_hits = sorted(
    i
    for i in residual
    if by_index[i].surface_type in ("bspline", "bezier", "sphere")
)
if pocket_floor_hits:
    types = Counter(by_index[i].surface_type for i in pocket_floor_hits)
    print(f"  *** POCKET FLOOR/SPHERE LEAK: {pocket_floor_hits} ({dict(types)}) ***")
    red_flags.append(f"pocket bspline/sphere leak: {pocket_floor_hits}")
else:
    print("  pocket bspline floors / spheres: none  [OK]")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)

biggest_smooth = len(smooth_comps[0]) if smooth_comps else 0
n_smooth = len(smooth_comps)
extras = sum(len(c) for c in smooth_comps[1:])
frag_ratio = extras / len(residual) if residual else 0.0

if n_smooth == 1:
    contour_read = "single tangency blob — contour groups easily"
elif biggest_smooth >= 0.75 * len(residual):
    contour_read = (
        f"one dominant tangency blob ({biggest_smooth}/{len(residual)} faces) "
        f"+ {n_smooth - 1} small extras — contour likely OK"
    )
elif n_smooth <= 4 and frag_ratio < 0.25:
    contour_read = (
        f"moderate fragmentation ({n_smooth} smooth components, "
        f"{biggest_smooth}-face main blob) — tangency may suffice with small extras"
    )
else:
    contour_read = (
        f"HEAVY FRAGMENTATION ({n_smooth} smooth components, largest "
        f"{biggest_smooth}/{len(residual)} faces) — tangency alone may not "
        f"group the skin"
    )

if red_flags:
    leak_read = f"UPSTREAM LEAKS ({len(red_flags)}) — fix before building contour"
    for rf in red_flags:
        print(f"  RED FLAG: {rf}")
else:
    leak_read = "no upstream leaks detected"

print(
    f"\n{contour_read}; {leak_read}. "
    f"Residual={len(residual)} faces vs Toolpath contour~43 / outer fillet 1 / profile 1; "
    f"convex components={len(convex_comps)} (Toolpath outer fillet target=1)."
)
