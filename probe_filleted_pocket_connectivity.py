"""Diagnostic: filleted_pocket spatial connectivity (read-only)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from feature_params import analyze_step

GRAPH_NPZ = Path("pipeline_out/96260B_plate/graph.npz")
STEP_REAR = "96260B_REAR_XR004_PCD PLATE.stp copy"
STEP_FRONT = "96260B_FRONT_XR004_PCD PLATE.stp copy"
GRAPH_FRONT = Path("pipeline_out/96260B_front/graph.npz")

POCKET_CLASS_NAMES = {
    "filleted_pocket",
    "filleted_open_pocket",
    "pocket",
    "open_pocket",
    "blind_pocket",
}


def load_adjacency(npz_path: Path) -> dict[int, set[int]]:
    data = np.load(npz_path)
    ei = data["edge_index"]
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in zip(ei[0], ei[1]):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def induced_components(face_ids: list[int], adj: dict[int, set[int]]) -> list[list[int]]:
    """Connected components of subgraph induced on face_ids."""
    fset = set(face_ids)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for s in face_ids:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u] & fset:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return sorted(comps, key=lambda c: (-len(c), c))


def centroid_of_faces(face_ids: list[int], centroids: dict[int, np.ndarray]) -> np.ndarray:
    pts = np.array([centroids[f] for f in face_ids if f in centroids])
    return pts.mean(axis=0) if len(pts) else np.full(3, np.nan)


def pocket_features(cascade_path: Path) -> list[dict]:
    with open(cascade_path) as f:
        data = json.load(f)
    out = []
    for node in data["nodes"]:
        cn = node.get("class_name", "")
        kind = (node.get("params") or {}).get("kind")
        if cn in POCKET_CLASS_NAMES or kind == "pocket":
            out.append(node)
    return out


def scope_table(
    part_id: str,
    cascade_path: Path,
    adj: dict[int, set[int]],
) -> list[dict]:
    rows = []
    for node in pocket_features(cascade_path):
        fids = node["face_ids"]
        comps = induced_components(fids, adj)
        rows.append({
            "part": part_id,
            "feature_id": node["feature_id"],
            "class_name": node["class_name"],
            "n_faces": len(fids),
            "n_components": len(comps),
            "component_sizes": [len(c) for c in comps],
            "face_ids": fids,
            "components": comps,
        })
    return rows


def print_feature_detail(
    label: str,
    node: dict,
    adj: dict[int, set[int]],
    centroids: dict[int, np.ndarray],
) -> None:
    fids = node["face_ids"]
    comps = induced_components(fids, adj)
    print(f"\n{'=' * 72}")
    print(f"{label}  feature_id={node['feature_id']}  class={node['class_name']}")
    print(f"face_ids ({len(fids)}): {fids}")
    print(f"component_count: {len(comps)}")
    for i, comp in enumerate(comps):
        c = centroid_of_faces(comp, centroids)
        print(
            f"  comp[{i}] size={len(comp)}  faces={comp}  "
            f"centroid=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
        )
    fset = set(fids)
    n_edges = sum(1 for u in fids for v in adj[u] & fset if u < v)
    print(f"induced_edges: {n_edges}")
    if len(comps) >= 2:
        main = comps[0]
        main_set = set(main)
        for comp in comps[1:]:
            print(f"\n  minority comp faces={comp}")
            for f in comp:
                nbrs_in_feat = sorted(adj[f] & fset)
                nbrs_in_main = sorted(adj[f] & main_set)
                print(
                    f"    face {f}: neighbors_in_feature={nbrs_in_feat}  "
                    f"neighbors_in_main_comp={nbrs_in_main}"
                )


def main() -> None:
    if not GRAPH_NPZ.exists():
        print(f"MISSING: {GRAPH_NPZ}")
        return
    adj_rear = load_adjacency(GRAPH_NPZ)
    faces_rear = analyze_step(STEP_REAR)
    centroids_rear = {f.index: np.array(f.centroid) for f in faces_rear}

    cascade_rear = Path("pipeline_out/96260B_rear/feature_graph_cascade.json")
    with open(cascade_rear) as f:
        rear_data = json.load(f)

    feat1 = next(n for n in rear_data["nodes"] if n["feature_id"] == 1)
    print_feature_detail("OFFENDING (rear filleted_pocket)", feat1, adj_rear, centroids_rear)

    print(f"\n{'=' * 72}")
    print("SCOPE TABLE - 96260B_rear pocket features")
    rear_rows = scope_table("96260B_rear", cascade_rear, adj_rear)
    multi = []
    print(f"{'part':<14} {'fid':>4} {'class':<22} {'faces':>5} {'#comp':>5}  sizes")
    for r in rear_rows:
        flag = " *** MULTI" if r["n_components"] > 1 else ""
        print(
            f"{r['part']:<14} {r['feature_id']:>4} {r['class_name']:<22} "
            f"{r['n_faces']:>5} {r['n_components']:>5}  {r['component_sizes']}{flag}"
        )
        if r["n_components"] > 1:
            multi.append(r)

    cascade_front = Path("pipeline_out/96260B_front/feature_graph_cascade.json")
    if not GRAPH_FRONT.exists():
        print(f"\nMISSING front graph: {GRAPH_FRONT}")
    else:
        adj_front = load_adjacency(GRAPH_FRONT)
        print(f"\n{'=' * 72}")
        print("SCOPE TABLE - 96260B_front pocket features")
        front_rows = scope_table("96260B_front", cascade_front, adj_front)
        for r in front_rows:
            flag = " *** MULTI" if r["n_components"] > 1 else ""
            print(
                f"{r['part']:<14} {r['feature_id']:>4} {r['class_name']:<22} "
                f"{r['n_faces']:>5} {r['n_components']:>5}  {r['component_sizes']}{flag}"
            )
            if r["n_components"] > 1:
                multi.append(r)

    print(f"\nTotal multi-component pocket features: {len(multi)}")
    for r in multi:
        print(f"  {r['part']} fid={r['feature_id']} {r['class_name']} sizes={r['component_sizes']}")

    print(f"\n{'=' * 72}")
    print("STEP 4 - sub-pairs 195/201 vs 196/199 (separate features?)")
    for node in rear_data["nodes"]:
        fset = set(node["face_ids"])
        hits = fset & {195, 196, 199, 201}
        if hits:
            comps = induced_components(node["face_ids"], adj_rear)
            print(
                f"  feature_id={node['feature_id']} class={node['class_name']} "
                f"faces={sorted(fset)} hits={sorted(hits)} n_comp={len(comps)} "
                f"sizes={[len(c) for c in comps]}"
            )


if __name__ == "__main__":
    main()
