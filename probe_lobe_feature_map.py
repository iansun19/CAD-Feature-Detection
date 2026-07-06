"""Map cascade filleted_pocket features to lobe tier closed sets."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from feature_params import analyze_step
from lobe_tier_detection import LobeTierConfig, detect_filleted_lobe_tiers

STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH = Path("pipeline_out/96260B_plate/graph.npz")

faces = analyze_step(STEP)
data = np.load(GRAPH)
lobe = detect_filleted_lobe_tiers(faces, data["edge_index"], data["edge_attr"], opening_axis=[0,1,0], config=LobeTierConfig())

cascade = json.load(open("pipeline_out/96260B_rear/feature_graph_cascade.json"))
pockets = [n for n in cascade["nodes"] if n["class_name"] == "filleted_pocket"]

print("Lobe closed sets:")
for lob in lobe.lobes:
    print(f"  lobe {lob.lobe_id}: n={len(lob.closed_faces)} {sorted(lob.closed_faces)}")

print("\nCascade filleted_pocket features:")
for n in pockets:
    print(f"  fid={n['feature_id']} n={len(n['face_ids'])} {n['face_ids']}")

print("\nBest overlap fid -> lobe:")
for n in pockets:
    fset = set(n["face_ids"])
    best = max(lobe.lobes, key=lambda lb: len(fset & lb.closed_faces))
    overlap = fset & best.closed_faces
    extra = fset - best.closed_faces
    missing = best.closed_faces - fset
    print(f"  fid={n['feature_id']} best_lobe={best.lobe_id} overlap={len(overlap)} extra={sorted(extra)} missing={sorted(missing)}")
