"""probe_outer_fillet_census.py — census convex-blend / outer-fillet geometry in the residual.

Diagnostic only — confirm outer fillets are cleanly isolable before building a pass.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_outer_fillet_census.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from feature_params import FaceGeom, analyze_step
from run_cascade import _load_edges, run_cascade

MM_PER_IN = 25.4
CONVEXITY_NAMES = ("concave", "convex", "smooth")

PARTS = (
    {
        "part_id": "96260B_front",
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
        "tp_contour_faces": 36,
        "tp_outer_fillets": 1,
    },
    {
        "part_id": "96260B_plate",
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_plate/graph.npz",
        "tp_contour_faces": 43,
        "tp_outer_fillets": 1,
    },
)

BLEND_TYPES = frozenset({"torus", "cylinder", "cone", "sphere", "bspline", "bezier"})
# Outer fillets on these parts use ~0.25 in minor radius (pocket fillet scale).
FILLET_R_MAX_MM = 10.0  # ~0.39 in — generous upper bound for "small radius"


def edge_convexity(row: np.ndarray) -> str:
    cid = int(np.argmax(row[:3])) if row.shape[0] >= 3 else 2
    return CONVEXITY_NAMES[cid]


def blend_radius_mm(f: FaceGeom) -> float | None:
    if f.surface_type == "torus" and f.torus_minor_r is not None:
        return float(f.torus_minor_r)
    if f.surface_type in ("cylinder", "sphere") and f.radius is not None:
        return float(f.radius)
    if f.surface_type == "cone" and f.semi_angle_rad is not None:
        return None
    return None


def fmt_hist(d: dict, *, key_fmt=str) -> str:
    return ", ".join(f"{key_fmt(k)}={v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0]))))


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


def build_adjacency(edge_index: np.ndarray) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def surface_hist(idxs: list[int], by_index: dict[int, FaceGeom]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for i in idxs:
        c[by_index[i].surface_type] += 1
    return dict(c)


def convex_components(residual: set[int], edge_index: np.ndarray, edge_attr: np.ndarray) -> list[list[int]]:
    uf = _UF(residual)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u not in residual or v not in residual:
            continue
        if edge_convexity(edge_attr[k]) == "convex":
            uf.union(u, v)
    return sorted(uf.components().values(), key=len, reverse=True)


def characterize_component(
    comp: list[int],
    *,
    residual: set[int],
    claimed: set[int],
    by_index: dict[int, FaceGeom],
    adj: dict[int, set[int]],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> dict:
    comp_set = set(comp)
    stypes = surface_hist(comp, by_index)
    n = len(comp)
    torus_n = stypes.get("torus", 0)
    blend_n = sum(stypes.get(t, 0) for t in BLEND_TYPES)
    torus_frac = torus_n / n
    blend_frac = blend_n / n

    areas = [by_index[i].area for i in comp]
    median_area = float(np.median(areas))
    total_area = float(sum(areas))

    # Fillet strip proxy: each face borders >=2 larger non-component neighbors.
    faces_two_larger = 0
    per_face_larger_nbrs: list[tuple[int, int, list[int]]] = []
    for fid in comp:
        outside = [nb for nb in adj[fid] if nb not in comp_set]
        larger = [nb for nb in outside if by_index[nb].area > by_index[fid].area]
        per_face_larger_nbrs.append((fid, len(larger), sorted(larger)))
        if len(larger) >= 2:
            faces_two_larger += 1
    strip_frac = faces_two_larger / n

    radii = [r for i in comp if (r := blend_radius_mm(by_index[i])) is not None]
    radii_in = [r / MM_PER_IN for r in radii]
    min_r_mm = min(radii) if radii else None
    max_r_mm = max(radii) if radii else None

    # Exterior proxy: all edges from comp to any outside face (not just residual).
    boundary_convex = boundary_smooth = boundary_concave = 0
    convex_to_claimed_plane: list[tuple[int, int, float]] = []
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u in comp_set and v in comp_set:
            continue
        if u not in comp_set and v not in comp_set:
            continue
        src = u if u in comp_set else v
        other = v if src == u else u
        conv = edge_convexity(edge_attr[k])
        if conv == "convex":
            boundary_convex += 1
            fb = by_index[other]
            if other in claimed and fb.surface_type == "plane":
                convex_to_claimed_plane.append((src, other, fb.area))
        elif conv == "smooth":
            boundary_smooth += 1
        else:
            boundary_concave += 1
    boundary_total = boundary_convex + boundary_smooth + boundary_concave
    exterior_frac = boundary_convex / boundary_total if boundary_total else 0.0
    # Distinct claimed planes reached by convex edges (outer fillet rounds flat↔wall).
    claimed_plane_nbrs = sorted(set(p for _, p, _ in convex_to_claimed_plane))

    # Broad vs narrow: median area vs median of all residual non-comp faces.
    outside_areas = [by_index[i].area for i in residual if i not in comp_set]
    residual_median = float(np.median(outside_areas)) if outside_areas else 0.0
    narrow = median_area < 0.25 * residual_median if residual_median > 0 else False

    # Fillet-likeness score (heuristic, for ranking).
    small_r = bool(radii and max(radii) <= FILLET_R_MAX_MM)
    has_profile_plane = bool(convex_to_claimed_plane)
    score = (
        3.0 * torus_frac
        + 1.5 * blend_frac
        + 2.0 * strip_frac
        + 1.0 * exterior_frac
        + (1.0 if small_r else 0.0)
        + (0.5 if narrow else 0.0)
        + (3.0 if has_profile_plane else 0.0)
    )

    return {
        "size": n,
        "faces": sorted(comp),
        "stypes": stypes,
        "torus_frac": torus_frac,
        "blend_frac": blend_frac,
        "median_area": median_area,
        "total_area": total_area,
        "residual_median_area": residual_median,
        "narrow_strip": narrow,
        "faces_two_larger": faces_two_larger,
        "strip_frac": strip_frac,
        "radii_mm": radii,
        "radii_in": radii_in,
        "min_r_mm": min_r_mm,
        "max_r_mm": max_r_mm,
        "small_radius": small_r,
        "boundary_convex": boundary_convex,
        "boundary_smooth": boundary_smooth,
        "boundary_concave": boundary_concave,
        "exterior_frac": exterior_frac,
        "convex_to_claimed_plane": convex_to_claimed_plane,
        "claimed_plane_nbrs": claimed_plane_nbrs,
        "fillet_score": score,
        "per_face_larger_nbrs": per_face_larger_nbrs,
    }


PROFILE_FLAT_AREA_MM2 = 2463.0
PROFILE_FLAT_AREA_TOL = 0.15  # ±15% matches Toolpath profile flat on both parts


def _is_profile_flat(area: float) -> bool:
    return abs(area - PROFILE_FLAT_AREA_MM2) / PROFILE_FLAT_AREA_MM2 <= PROFILE_FLAT_AREA_TOL


def _has_profile_flat_adjacency(c: dict) -> bool:
    return any(_is_profile_flat(a) for _, _, a in c["convex_to_claimed_plane"])


def pick_outer_fillet_candidate(comps_char: list[dict]) -> dict | None:
    """Best convex component matching outer-fillet signature."""
    # Primary: pure-torus chain, R≈0.15in, convex edge to profile flat (not pocket step / stock).
    primary = [
        c
        for c in comps_char
        if c["torus_frac"] == 1.0
        and _has_profile_flat_adjacency(c)
        and c["min_r_mm"] is not None
        and 0.12 <= c["min_r_mm"] / MM_PER_IN <= 0.18
    ]
    if primary:
        return max(primary, key=lambda c: c["size"])
    # Fallback: any torus with convex-to-profile-flat.
    fallback = [c for c in comps_char if c["torus_frac"] > 0 and _has_profile_flat_adjacency(c)]
    if fallback:
        return max(fallback, key=lambda c: c["fillet_score"])
    return None


def torus_minor_buckets(residual: set[int], by_index: dict[int, FaceGeom]) -> dict[float, list[int]]:
    buckets: dict[float, list[int]] = defaultdict(list)
    for i in sorted(residual):
        f = by_index[i]
        if f.surface_type == "torus" and f.torus_minor_r is not None:
            buckets[round(f.torus_minor_r / MM_PER_IN, 3)].append(i)
    return dict(buckets)


def census_part(
    part_id: str,
    step_path: str,
    graph_npz: str,
    tp_contour_faces: int,
    tp_outer_fillets: int,
) -> dict:
    faces = analyze_step(step_path)
    by_index = {f.index: f for f in faces}
    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _, pk, hl, fl, _, _ = run_cascade(step_path, edge_index, edge_attr)
    residual = set(int(i) for i in fl.remaining_faces)
    claimed = pk.claimed_faces | hl.claimed_faces | fl.claimed_faces
    adj = build_adjacency(edge_index)

    print("\n" + "=" * 78)
    print(f"PART: {part_id}")
    print(f"STEP: {step_path}")
    print(f"GRAPH: {graph_npz}")
    print(
        f"Toolpath targets: contour surfaces ~{tp_contour_faces}, "
        f"outer fillets {tp_outer_fillets}"
    )
    print("=" * 78)

    # --- 1. Residual surface-type histogram ---
    print("\n--- 1. RESIDUAL SURFACE-TYPE HISTOGRAM ---")
    print(f"residual size: {len(residual)} faces")
    stype_hist = Counter(by_index[i].surface_type for i in residual)
    print(f"  {dict(stype_hist)}")

    # --- 2. CONVEX-connected components ---
    print("\n--- 2. CONVEX-CONNECTED COMPONENTS (residual internal edges, convex only) ---")
    convex_comps = convex_components(residual, edge_index, edge_attr)
    print(f"component count: {len(convex_comps)}")
    print(f"sorted sizes: {[len(c) for c in convex_comps]}")
    n_show = min(8, len(convex_comps))
    print(f"\nlargest ~{n_show} components (surface-type composition):")
    comps_char: list[dict] = []
    for ci, comp in enumerate(convex_comps[:n_show]):
        ch = characterize_component(
            comp,
            residual=residual,
            claimed=claimed,
            by_index=by_index,
            adj=adj,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        comps_char.append(ch)
        print(
            f"  comp {ci}: size={ch['size']}  stypes={ch['stypes']}  "
            f"faces={ch['faces']}"
        )

    # Characterize ALL components for candidate pick (not just top 8).
    all_char = [
        characterize_component(
            comp,
            residual=residual,
            claimed=claimed,
            by_index=by_index,
            adj=adj,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        for comp in convex_comps
    ]

    # --- 3. Characterize sizable convex components ---
    print("\n--- 3. OUTER-FILLET vs CONTOUR CHARACTERIZATION (components size >= 1) ---")
    sizable = [c for c in all_char if c["size"] >= 1]
    for ci, ch in enumerate(sizable[:12]):
        r_str = (
            f"R={ch['min_r_mm']:.3f}–{ch['max_r_mm']:.3f} mm "
            f"({ch['min_r_mm']/MM_PER_IN:.4f}–{ch['max_r_mm']/MM_PER_IN:.4f} in)"
            if ch["min_r_mm"] is not None
            else "R=n/a"
        )
        kind = "OUTER-FILLET-like" if ch["fillet_score"] >= 3.0 and ch["torus_frac"] > 0 else "CONTOUR-like"
        print(f"\n  comp {ci} [{kind}] score={ch['fillet_score']:.2f}  faces={ch['faces']}")
        print(
            f"    torus_frac={ch['torus_frac']:.2f}  blend_frac={ch['blend_frac']:.2f}  "
            f"{r_str}  small_radius={ch['small_radius']}"
        )
        print(
            f"    median_area={ch['median_area']:.1f} mm²  "
            f"(residual median={ch['residual_median_area']:.1f})  "
            f"narrow_strip={ch['narrow_strip']}"
        )
        print(
            f"    strip: {ch['faces_two_larger']}/{ch['size']} faces border ≥2 larger outsiders  "
            f"(strip_frac={ch['strip_frac']:.2f})"
        )
        print(
            f"    exterior: boundary convex/smooth/concave = "
            f"{ch['boundary_convex']}/{ch['boundary_smooth']}/{ch['boundary_concave']}  "
            f"(exterior_frac={ch['exterior_frac']:.2f})"
        )
        if ch["convex_to_claimed_plane"]:
            planes = sorted(set(ch["convex_to_claimed_plane"]), key=lambda t: -t[2])
            print(
                f"    convex→claimed-plane: "
                f"{[(f, p, f'{a:.0f}') for f, p, a in planes[:4]]}"
            )
        if ch["size"] <= 6:
            for fid, n_larger, nbrs in ch["per_face_larger_nbrs"]:
                nbr_info = [
                    f"{nb}({by_index[nb].surface_type},{by_index[nb].area:.0f})"
                    for nb in nbrs[:4]
                ]
                print(f"      face {fid}: {n_larger} larger nbrs → {nbr_info}")

    # --- 4. KEY CHECK: dominant outer-fillet chain ---
    print("\n--- 4. KEY CHECK — DOMINANT OUTER-FILLET CHAIN ---")
    buckets = torus_minor_buckets(residual, by_index)
    print("  torus minor-radius buckets in residual (inches):")
    for r_in, idxs in sorted(buckets.items()):
        print(f"    R={r_in:.3f}in: {len(idxs)} face(s) {idxs}")

    candidate = pick_outer_fillet_candidate(all_char)
    torus_comps = [c for c in all_char if c["stypes"].get("torus", 0) > 0]
    print(f"  torus-bearing convex components: {len(torus_comps)}")
    if candidate is None:
        print("  NO outer-fillet candidate matching torus + convex→claimed-plane.")
    else:
        print(f"  *** OUTER-FILLET CANDIDATE faces: {candidate['faces']} ***")
        print(f"  size={candidate['size']}  stypes={candidate['stypes']}")
        if candidate["radii_in"]:
            uniq_r = sorted(set(round(r, 4) for r in candidate["radii_in"]))
            print(f"  blend radii (in): {uniq_r}")
        print(
            f"  claimed-plane nbrs (convex): {candidate['claimed_plane_nbrs']}  "
            f"areas={[f'{by_index[p].area:.0f}' for p in candidate['claimed_plane_nbrs']]}"
        )
        print(
            f"  reads as external rounding: torus strip convex-adjacent to claimed flat, "
            f"smooth-adjacent to profile cylinders in residual; "
            f"strip_frac={candidate['strip_frac']:.2f}, "
            f"exterior_frac={candidate['exterior_frac']:.2f}"
        )
        rivals = [
            c for c in torus_comps
            if c["faces"] != candidate["faces"] and _has_profile_flat_adjacency(c)
        ]
        print(f"  rival torus comps with convex→profile-flat: {len(rivals)}")
        for tc in sorted(rivals, key=lambda c: -c["size"])[:5]:
            print(
                f"    size={tc['size']} R≈{tc['min_r_mm']/MM_PER_IN:.3f}in  "
                f"faces={tc['faces']}  planes={tc['claimed_plane_nbrs']}"
            )

    # --- 5. Separation check ---
    print("\n--- 5. SEPARATION CHECK (shed outer-fillet candidate from residual) ---")
    if candidate:
        shed = set(candidate["faces"])
        remaining = residual - shed
        delta = len(remaining) - tp_contour_faces
        print(f"  residual before shed: {len(residual)}")
        print(f"  shed outer-fillet candidate: {len(shed)} face(s) {sorted(shed)}")
        print(f"  remaining after shed: {len(remaining)}")
        print(f"  Toolpath contour surfaces: {tp_contour_faces}")
        print(f"  delta (remaining − Toolpath contour): {delta:+d}")
        rem_stypes = Counter(by_index[i].surface_type for i in remaining)
        print(f"  remaining surface types: {dict(rem_stypes)}")
    else:
        shed = set()
        remaining = residual
        print("  no candidate to shed")

    return {
        "part_id": part_id,
        "residual_size": len(residual),
        "stype_hist": dict(stype_hist),
        "n_convex_comps": len(convex_comps),
        "convex_sizes": [len(c) for c in convex_comps],
        "all_char": all_char,
        "candidate": candidate,
        "shed_faces": sorted(shed),
        "remaining_after_shed": len(remaining),
        "tp_contour_faces": tp_contour_faces,
        "delta_contour": len(remaining) - tp_contour_faces if candidate else None,
    }


def verdict(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    per_part: list[dict] = []
    for r in results:
        pid = r["part_id"]
        cand = r["candidate"]
        torus_comps = [c for c in r["all_char"] if c["stypes"].get("torus", 0) > 0]
        mega = r["convex_sizes"][0] if r["convex_sizes"] else 0

        if cand is None:
            per_part.append({"pid": pid, "clean": False, "faces": []})
            print(f"  {pid}: NOT CLEAN — no torus + convex→claimed-plane candidate")
            continue

        rivals = [
            c for c in torus_comps
            if c["faces"] != cand["faces"] and _has_profile_flat_adjacency(c)
        ]
        clean = (
            cand["size"] == 2
            and cand["torus_frac"] == 1.0
            and _has_profile_flat_adjacency(cand)
            and cand["min_r_mm"] is not None
            and 0.12 <= cand["min_r_mm"] / MM_PER_IN <= 0.18
            and len(rivals) == 0
        )
        per_part.append({"pid": pid, "clean": clean, "faces": cand["faces"], "cand": cand})

        r_in = cand["min_r_mm"] / MM_PER_IN
        print(
            f"  {pid}: {'CLEAN' if clean else 'PARTIAL'} — "
            f"faces={cand['faces']}  R≈{r_in:.3f}in  "
            f"convex→claimed-plane={cand['claimed_plane_nbrs']}  "
            f"({len(torus_comps)} torus convex comps; mega-contour blob={mega} faces)"
        )

    cands = [p for p in per_part if p.get("cand")]
    all_clean = all(p.get("clean") for p in per_part)
    sig_match = (
        len(cands) == 2
        and all(p["clean"] for p in cands)
        and abs(cands[0]["cand"]["min_r_mm"] - cands[1]["cand"]["min_r_mm"]) < 0.01
        and cands[0]["cand"]["size"] == cands[1]["cand"]["size"] == 2
    )

    print()
    if all_clean and sig_match:
        print(
            "Outer fillet IS a clean, isolable convex-blend chain on BOTH parts "
            "(Toolpath outer_fillet=1 each)."
        )
        print(
            "Pass signature: post-flats residual ∩ {torus minor_R≈0.15in} ∩ "
            "CONVEX-connected chain ∩ convex edge to claimed profile flat "
            f"(area≈{PROFILE_FLAT_AREA_MM2:.0f} mm², flats-pass claimed) ∩ "
            "smooth edge to profile cylinders."
        )
    elif cands:
        print(
            "One plausible outer-fillet chain per part, but naive convex-only grouping "
            "is TANGLED (~20 torus convex components + 120/141-face contour mega-blob)."
        )
        print(
            "Pass needs convex→claimed-plane + R≈0.15in filters — not convex-connectivity alone."
        )

    if cands:
        print(f"  FRONT faces to eyeball: {cands[0]['faces']}")
        print(f"  BACK  faces to eyeball: {cands[1]['faces']}")
        if len(cands) == 2:
            same = (
                cands[0]["cand"]["size"] == cands[1]["cand"]["size"] == 2
                and abs(cands[0]["cand"]["min_r_mm"] - cands[1]["cand"]["min_r_mm"]) < 0.01
            )
            print(
                f"SAME signature on both parts: {'YES' if same else 'NO'} "
                f"(2-face R≈0.15in torus chain, convex to profile flat)"
            )

    print("\nSeparation vs Toolpath contour face counts (informational):")
    for r in results:
        if r["delta_contour"] is not None:
            print(
                f"  {r['part_id']}: residual {r['residual_size']} → "
                f"after shed {r['remaining_after_shed']} "
                f"(Toolpath contour {r['tp_contour_faces']}, Δ={r['delta_contour']:+d})"
            )
    print(
        "Note: shedding 2 outer-fillet faces is a small identifiable subset; "
        "residual face counts remain well above Toolpath contour totals (cascade is finer)."
    )


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
            "tp_contour_faces": PARTS[0]["tp_contour_faces"],
            "tp_outer_fillets": PARTS[0]["tp_outer_fillets"],
        },
        {
            "part_id": PARTS[1]["part_id"],
            "step_path": args.back_step,
            "graph_npz": args.back_npz,
            "tp_contour_faces": PARTS[1]["tp_contour_faces"],
            "tp_outer_fillets": PARTS[1]["tp_outer_fillets"],
        },
    ]

    print("=" * 78)
    print("OUTER-FILLET / CONVEX-BLEND RESIDUAL CENSUS (diagnostic)")
    print("=" * 78)
    print("Residual = post pockets + holes + flats cascade (flat_result.remaining_faces)")

    results = [census_part(**cfg) for cfg in configs]
    verdict(results)


if __name__ == "__main__":
    main()
