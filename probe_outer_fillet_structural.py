"""probe_outer_fillet_structural.py — reduce outer-fillet signature to structural core.

Tests whether convexity + relative adjacency (no absolute R or area gates) isolates
the known outer-fillet chains on both reference parts.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_outer_fillet_structural.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from feature_params import FaceGeom, analyze_step
from run_cascade import _load_edges, run_cascade

CONVEXITY_NAMES = ("concave", "convex", "smooth")
BLEND_TYPES = frozenset({"torus", "cylinder", "cone", "sphere", "bspline", "bezier"})

PARTS = (
    {
        "part_id": "96260B_front",
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
        "known_faces": {96, 287},
    },
    {
        "part_id": "96260B_plate",
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_plate/graph.npz",
        "known_faces": {111, 336},
    },
)


def edge_convexity(row: np.ndarray) -> str:
    cid = int(np.argmax(row[:3])) if row.shape[0] >= 3 else 2
    return CONVEXITY_NAMES[cid]


class _UF:
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

    def components(self) -> list[list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for n in self.p:
            groups[self.find(n)].append(n)
        return [sorted(v) for v in groups.values()]


def _face_edges(
    fid: int, edge_index: np.ndarray, edge_attr: np.ndarray,
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == fid:
            out.append((edge_convexity(edge_attr[k]), v))
        elif v == fid:
            out.append((edge_convexity(edge_attr[k]), u))
    return out


def rule_b_larger_exterior_convex(
    fid: int,
    *,
    by_index: dict[int, FaceGeom],
    pocket_claimed: set[int],
    edges: list[tuple[str, int]],
) -> bool:
    """Convex edge to a larger non-pocket exterior face (plane/cyl/cone)."""
    fa = by_index[fid]
    for conv, nb in edges:
        if conv != "convex":
            continue
        fb = by_index[nb]
        if fb.area <= fa.area:
            continue
        if nb in pocket_claimed:
            continue
        if fb.surface_type in ("plane", "cylinder", "cone"):
            return True
    return False


def rule_c_smooth_profile_cylinder(
    fid: int,
    *,
    by_index: dict[int, FaceGeom],
    pocket_claimed: set[int],
    edges: list[tuple[str, int]],
) -> bool:
    """Smooth edge to a larger non-pocket cylinder (profile / exterior wall)."""
    fa = by_index[fid]
    for conv, nb in edges:
        if conv != "smooth":
            continue
        fb = by_index[nb]
        if fb.surface_type != "cylinder":
            continue
        if fb.area <= fa.area:
            continue
        if nb in pocket_claimed:
            continue
        return True
    return False


def rule_d_exterior_not_pocket_enclosed(
    fid: int,
    *,
    pocket_claimed: set[int],
    edges: list[tuple[str, int]],
) -> bool:
    """No concave edge into pocket-claimed interior (not enclosed by pocket walls)."""
    return not any(conv == "concave" and nb in pocket_claimed for conv, nb in edges)


def structural_pass(
    *,
    residual: set[int],
    pocket_claimed: set[int],
    by_index: dict[int, FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    active: frozenset[str],
    blend_types: frozenset[str] = BLEND_TYPES,
) -> tuple[list[int], list[list[int]], dict[int, dict[str, bool]]]:
    """Return face hits, convex-connected components among hits, per-face rule flags."""
    pool = sorted(i for i in residual if by_index[i].surface_type in blend_types)
    flags: dict[int, dict[str, bool]] = {}
    hits: list[int] = []

    for fid in pool:
        edges = _face_edges(fid, edge_index, edge_attr)
        f = {
            "b": rule_b_larger_exterior_convex(
                fid, by_index=by_index, pocket_claimed=pocket_claimed, edges=edges,
            ),
            "c": rule_c_smooth_profile_cylinder(
                fid, by_index=by_index, pocket_claimed=pocket_claimed, edges=edges,
            ),
            "d": rule_d_exterior_not_pocket_enclosed(
                fid, pocket_claimed=pocket_claimed, edges=edges,
            ),
        }
        flags[fid] = f
        if all(f[r] for r in active):
            hits.append(fid)

    hitset = set(hits)
    uf = _UF(hitset)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u not in hitset or v not in hitset:
            continue
        if edge_convexity(edge_attr[k]) == "convex":
            uf.union(u, v)
    comps = sorted(uf.components(), key=len, reverse=True)
    return hits, comps, flags


def diagnose_part(
    part_id: str,
    step_path: str,
    graph_npz: str,
    known_faces: set[int],
) -> dict:
    faces = analyze_step(step_path)
    by_index = {f.index: f for f in faces}
    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _, pk, _, fl, _, _ = run_cascade(step_path, edge_index, edge_attr)
    residual = set(int(i) for i in fl.remaining_faces)
    pocket_claimed = pk.claimed_faces

    print("\n" + "=" * 78)
    print(f"PART: {part_id}")
    print(f"known outer-fillet faces: {sorted(known_faces)}")
    print(f"post-flats residual: {len(residual)} faces")
    print("=" * 78)

    active = frozenset("bcd")
    hits, comps, flags = structural_pass(
        residual=residual,
        pocket_claimed=pocket_claimed,
        by_index=by_index,
        edge_index=edge_index,
        edge_attr=edge_attr,
        active=active,
    )

    print("\n--- RULE DEFINITIONS (no absolute R / area constants) ---")
    print("  (b) convex edge → larger non-pocket plane/cylinder/cone (relative area)")
    print("  (c) smooth edge → larger non-pocket cylinder (profile wall, relative area)")
    print("  (d) no concave edge → pocket-claimed face (exterior, not pocket-enclosed)")
    print("  (a) CONVEX-connected components among hits (grouping step)")

    print(f"\n--- SELECTION: (a)+(b)+(c)+(d) on blend faces in residual ---")
    print(f"  selected faces ({len(hits)}): {hits}")
    print(f"  convex-connected components: {comps}")

    exact = set(hits) == known_faces
    extra = sorted(set(hits) - known_faces)
    miss = sorted(known_faces - set(hits))
    print(f"  exact match: {exact}")
    if extra:
        print(f"  OVER-select extra: {extra}")
    if miss:
        print(f"  UNDER-select miss: {miss}")

    print("\n--- PER-FACE RULE TRUTH (known + sample pocket-blend rejects) ---")
    torus_residual = sorted(i for i in residual if by_index[i].surface_type == "torus")
    pocket_rejects = [
        i for i in torus_residual
        if i not in known_faces and any(not flags[i][r] for r in "bcd")
    ][:5]
    show = sorted(known_faces | set(pocket_rejects))
    for fid in show:
        f = by_index[fid]
        flg = flags[fid]
        tag = "KNOWN" if fid in known_faces else "reject"
        print(
            f"  [{tag}] face {fid} {f.surface_type} area={f.area:.0f}  "
            f"b={flg['b']} c={flg['c']} d={flg['d']}"
        )

    print("\n--- ABLATION (which rules are necessary?) ---")
    for rule_set in ("b", "c", "d", "bc", "bd", "cd", "bcd"):
        h, _, _ = structural_pass(
            residual=residual,
            pocket_claimed=pocket_claimed,
            by_index=by_index,
            edge_index=edge_index,
            edge_attr=edge_attr,
            active=frozenset(rule_set),
        )
        mark = "OK" if set(h) == known_faces else f"n={len(h)}"
        print(f"  rules={rule_set:3s}  hits={h}  [{mark}]")

    return {
        "part_id": part_id,
        "hits": hits,
        "comps": comps,
        "exact": exact,
        "known_faces": known_faces,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--front-step", default=PARTS[0]["step"])
    ap.add_argument("--front-npz", default=PARTS[0]["graph_npz"])
    ap.add_argument("--back-step", default=PARTS[1]["step"])
    ap.add_argument("--back-npz", default=PARTS[1]["graph_npz"])
    args = ap.parse_args()

    configs = [
        {
            "part_id": PARTS[0]["part_id"],
            "step_path": args.front_step,
            "graph_npz": args.front_npz,
            "known_faces": PARTS[0]["known_faces"],
        },
        {
            "part_id": PARTS[1]["part_id"],
            "step_path": args.back_step,
            "graph_npz": args.back_npz,
            "known_faces": PARTS[1]["known_faces"],
        },
    ]

    print("=" * 78)
    print("OUTER-FILLET STRUCTURAL SIGNATURE REDUCTION (diagnostic)")
    print("=" * 78)
    print("Pool: blend faces in post-flats residual. No R==0.15 or area==2463 gates.")

    results = [diagnose_part(**cfg) for cfg in configs]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    all_exact = all(r["exact"] for r in results)
    print(f"\n1. Does (a)+(b)+(c)+(d) alone isolate exactly the known 2-face chain?")
    for r in results:
        print(
            f"   {r['part_id']}: {'YES' if r['exact'] else 'NO'} — "
            f"selected {r['hits']}  component {r['comps']}"
        )
    print(f"   BOTH parts: {'YES' if all_exact else 'NO'}")

    print(f"\n2. Over-selection / relative tie-breaker needed?")
    if all_exact:
        print(
            "   NO extra faces with (a)+(b)+(c)+(d). Pocket-blend torus leaks are rejected by:"
        )
        print(
            "     • (b) pocket step planes are SMALLER than the blend face → fails relative-area test"
        )
        print(
            "     • (c) pocket wall cylinders are SMALLER than the blend face (or pocket-claimed)"
        )
        print(
            "     • (d) pocket interior fillets have CONCAVE edges to pocket-claimed faces"
        )
        print("   No radius cluster or absolute-area tie-breaker required on these parts.")
    else:
        print("   YES — see OVER-select lists above; add relative discriminator.")

    print(f"\n3. FINAL STRUCTURAL SIGNATURE (durable contract):")
    print(
        """
   Pool:   blend faces in post-flats residual (torus primary; all blend types equivalent here)
   Per-face gates (ALL required):
     (b) ∃ convex edge to neighbor with area > self, surface ∈ {plane,cyl,cone},
         neighbor ∉ pocket_claimed
     (c) ∃ smooth edge to cylinder with area > self, neighbor ∉ pocket_claimed
     (d) no concave edge to any pocket_claimed face
   Group:  (a) union-find on convex edges among passing faces → one instance per component

   Upstream dependency: pocket_claimed from pocket pass (class label, not a geometry constant).
   Relative sizing (neighbor.area > self.area) replaces absolute area/R thresholds.
"""
    )
    print("   Selected faces:")
    for r in results:
        print(f"     {r['part_id']}: {r['hits']}")


if __name__ == "__main__":
    main()
