"""DIAGNOSIS ONLY — open-pocket over-labeling on part1/part2.

Per-face inventory + adjacency + convexity for the disputed faces, plus a
golden-safety probe skeleton. No code edits to detectors.
"""
import sys
import numpy as np

from brep.feature_params import analyze_step
from brep.step_ingest import ingest_step_to_pyg

PARTS = {
    "part1": "fixtures/step/fixtures/step/part1.step",
    "part2": "fixtures/step/fixtures/step/part2.step",
}
TARGETS = {
    "part1": [5, 6, 7, 8, 9, 10, 16, 17],
    "part2": [5, 6, 7, 8, 9, 10, 23, 24, 25, 26],
}
GT_PROFILE = {
    "part1": [6, 8, 9, 10, 16, 17],
    "part2": [8, 25, 24, 7, 26, 6, 23, 9],
}
GT_FLATS = {
    "part1": [5, 7],
    "part2": [5, 10],
}


def vec(v):
    return "[" + ",".join(f"{x:+.3f}" for x in v) + "]"


def build_adj(edge_index, edge_attr, n):
    """Return dict[frozenset(i,j)] -> (convexity_label, cos_dihedral)."""
    adj = {}
    ei = edge_index
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i == j:
            continue
        key = frozenset((i, j))
        conc, conv, smooth, cosd = edge_attr[k]
        if conc > 0.5:
            lab = "concave"
        elif conv > 0.5:
            lab = "convex"
        else:
            lab = "smooth"
        adj[key] = (lab, float(cosd))
    return adj


def neighbors(adj, f):
    out = []
    for key, (lab, cosd) in adj.items():
        if f in key:
            other = (set(key) - {f}).pop()
            out.append((other, lab, cosd))
    return sorted(out)


def connected(faces, adj):
    """Is the face set connected via any edge? Return components."""
    faces = set(faces)
    seen = set()
    comps = []
    for start in faces:
        if start in seen:
            continue
        stack = [start]
        comp = set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            seen.add(x)
            for key in adj:
                if x in key:
                    o = (set(key) - {x}).pop()
                    if o in faces and o not in comp:
                        stack.append(o)
        comps.append(sorted(comp))
    return comps


def main():
    for part, step in PARTS.items():
        print("=" * 78)
        print(f"PART {part}  ({step})")
        print("=" * 78)
        faces = analyze_step(step)
        n = len(faces)
        _x, ei, ea, _stats = ingest_step_to_pyg(step)
        adj = build_adj(ei, ea, n)

        # part scale
        cents = np.stack([f.centroid for f in faces])
        bbox = cents.max(0) - cents.min(0)
        areas = np.array([f.area for f in faces])
        print(f"n_faces={n}  bbox={vec(bbox)}  median_area={np.median(areas):.2f}")
        print()

        tgt = TARGETS[part]
        print(f"--- per-face inventory (target faces {tgt}) ---")
        for fi in tgt:
            f = faces[fi]
            role = ("FLAT-gt" if fi in GT_FLATS[part]
                    else "PROFILE-gt" if fi in GT_PROFILE[part] else "?")
            ax = vec(f.axis) if f.axis is not None else "----"
            nm = vec(f.normal)
            r = f"r={f.radius:.2f}" if f.radius else ""
            print(f" f{fi:>2} {role:<10} {f.surface_type:<9} A={f.area:>8.2f} "
                  f"n={nm} ax={ax} c={vec(f.centroid)} {r}")
        print()

        print(f"--- adjacency of target faces (neighbor, convexity, cos) ---")
        for fi in tgt:
            nb = [(o, lab, round(c, 2)) for o, lab, c in neighbors(adj, fi)]
            print(f" f{fi:>2}: {nb}")
        print()

        print(f"--- GT profile set {GT_PROFILE[part]} connectivity ---")
        comps = connected(GT_PROFILE[part], adj)
        print(f"   components: {comps}  (connected={len(comps)==1})")
        # closed loop? each face degree within set
        prof = set(GT_PROFILE[part])
        for fi in GT_PROFILE[part]:
            deg = sum(1 for key in adj if fi in key and (set(key) - {fi}).pop() in prof)
            print(f"   f{fi} internal-degree={deg}")
        print()

        print(f"--- GT flats {GT_FLATS[part]} adjacency into profile? ---")
        for fi in GT_FLATS[part]:
            nb = neighbors(adj, fi)
            into_prof = [(o, lab) for o, lab, c in nb if o in prof]
            print(f"   f{fi}: neighbors-in-profile={into_prof}")
        print()


if __name__ == "__main__":
    sys.exit(main())
