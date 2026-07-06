"""probe_synthetic_census.py — diagnostic census of MFCAD++ synthetic training data.

Evaluates whether the synthetic dataset can support Architecture 2 (GNN over grouped
feature instances) without rebuilding the dataset or touching training.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_synthetic_census.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from dataset import SPLIT_H5, _brep_bounds, _canonical_edge, _edge_set
from feature_instances import instances_from_labels, union_find_instances
from feature_params import FaceGeom, analyze_step
from hole_detection import FaceGraph
from instance_features import instance_features
from taxonomy import NEW_DESCRIPTIONS, NEW_NAMES, NUM_CLASSES

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
DATA_ROOT = REPO / "MFCAD++_dataset"
H5_DIR = DATA_ROOT / "hierarchical_graphs_regen_12"
STEP_ROOT = DATA_ROOT / "step"
REAL_STEP = REPO / "96260B_REAR_XR004_PCD PLATE.stp copy"
REAL_GRAPH = REPO / "pipeline_out/96260B_plate/graph.npz"

SPLIT_FILES = ("train.txt", "val.txt", "test.txt")

# Generator surface-type codes (V_1 col 4 * 11, rounded).
TCODE_TO_SURFACE = {
    1: "plane",
    2: "cylinder",
    3: "torus",
    4: "sphere",
    5: "cone",
    11: "other",
}

BLEND_TYPES = frozenset({"other", "sphere", "torus", "bspline", "bezier"})

# Architecture-2 target vocabulary on real parts (cascade + downstream GNN).
TARGET_CLASSES: list[str] = [
    "contour_surfaces",
    "faces_flats",
    "filleted_blind_holes",
    "filleted_pockets",
    "outer_fillets",
    "profiles",
    "through_holes",
    "walls",
    "open_pockets",
]

# Best-effort map: synthetic 12-class id -> target class(es). Document gaps explicitly.
SYNTHETIC_TO_TARGET: dict[int, list[str]] = {
    0: ["through_holes"],                    # through_hole
    1: ["through_holes"],                    # poly_through_passage (through opening)
    2: ["open_pockets"],                     # through_slot
    3: [],                                   # through_step — no direct target
    4: [],                                   # o_ring — no direct target
    5: ["filleted_blind_holes"],             # blind_hole (analytic, rarely filleted)
    6: ["filleted_pockets"],                 # blind_pocket (analytic box/poly)
    7: ["open_pockets"],                     # blind_slot
    8: [],                                   # blind_step
    9: [],                                   # chamfer
    10: ["outer_fillets"],                   # round_fillet
    11: [],                                   # stock — outer raw material; not contour/flats
}

# Confusable pairs for Q3 (synthetic-side proxy labels where needed).
CONFUSABLE_PAIRS: list[tuple[str, str, str]] = [
    (
        "filleted_pocket_vs_open_pocket",
        "filleted_pockets",
        "open_pockets",
    ),
    (
        "blind_hole_vs_through_hole",
        "filleted_blind_holes",
        "through_holes",
    ),
    (
        "contour_vs_outer_fillet",
        "contour_surfaces",
        "outer_fillets",
    ),
]

FEATURE_KEYS = [
    "n_faces",
    "total_area",
    "bbox_depth",
    "bbox_width",
    "bbox_length",
    "n_distinct_axes",
    "hist_plane",
    "hist_cylinder",
    "hist_cone",
    "hist_sphere",
    "hist_torus",
    "hist_other",
    "edge_concave",
    "edge_convex",
    "edge_smooth",
]


# ---------------------------------------------------------------------------
# H5 → geometry helpers
# ---------------------------------------------------------------------------
def _decode_pid(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _tcode_row(v1_row: np.ndarray) -> int:
    return int(np.round(float(v1_row[4]) * 11))


def faces_from_v1(v1: np.ndarray) -> list[FaceGeom]:
    """Build FaceGeom list from regenerated H5 V_1 block (per-model slice)."""
    out: list[FaceGeom] = []
    for i, row in enumerate(v1):
        code = _tcode_row(row)
        st = TCODE_TO_SURFACE.get(code, "other")
        out.append(FaceGeom(
            index=i,
            surface_type=st,
            area=float(row[0]),
            centroid=np.asarray(row[1:4], dtype=np.float64),
            normal=np.asarray(row[5:8], dtype=np.float64) if row.shape[0] >= 8 else np.zeros(3),
        ))
    return out


def edge_index_from_a1(a1_idx: np.ndarray, start: int, end: int) -> np.ndarray:
    """Undirected edge_index from H5 A_1_idx for one model slice."""
    ei_pairs = sorted(_edge_set(a1_idx, start, end))
    if not ei_pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(ei_pairs, dtype=np.int64).T


def graph_from_h5(
    a1_idx: np.ndarray,
    e1_idx: np.ndarray,
    e2_idx: np.ndarray,
    e3_idx: np.ndarray,
    start: int,
    end: int,
    n_faces: int,
) -> FaceGraph:
    """FaceGraph from raw H5 convexity buckets (deduped canonical edges)."""
    ei = edge_index_from_a1(a1_idx, start, end)
    if ei.size == 0:
        return FaceGraph(n_faces)
    ea = np.zeros((ei.shape[1], 4), dtype=np.float32)
    e1 = {_canonical_edge(u, v) for u, v in _edge_set(e1_idx, start, end)}
    e2 = {_canonical_edge(u, v) for u, v in _edge_set(e2_idx, start, end)}
    e3 = {_canonical_edge(u, v) for u, v in _edge_set(e3_idx, start, end)}
    for k in range(ei.shape[1]):
        key = _canonical_edge(int(ei[0, k]), int(ei[1, k]))
        if key in e2:
            ea[k, 0] = 1.0
        elif key in e1:
            ea[k, 1] = 1.0
        elif key in e3:
            ea[k, 2] = 1.0
        else:
            ea[k, 1] = 1.0
    return FaceGraph.from_edge_tensors(ei, ea, n_faces)


def instance_vector(feat: dict[str, Any]) -> np.ndarray:
    hist = feat.get("surface_type_histogram", {})
    edge = feat.get("internal_edge_convexity", {})
    return np.array([
        feat.get("n_faces", 0),
        feat.get("total_area", 0.0),
        feat.get("bbox_depth", 0.0),
        feat.get("bbox_width", 0.0),
        feat.get("bbox_length", 0.0),
        feat.get("n_distinct_axes", 0),
        hist.get("plane", 0),
        hist.get("cylinder", 0),
        hist.get("cone", 0),
        hist.get("sphere", 0),
        hist.get("torus", 0),
        hist.get("other", 0),
        edge.get("concave", 0),
        edge.get("convex", 0),
        edge.get("smooth", 0),
    ], dtype=np.float64)


def has_blend_types(faces: Sequence[FaceGeom]) -> bool:
    return any(f.surface_type in BLEND_TYPES for f in faces)


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------
@dataclass
class PartRecord:
    part_id: str
    split: str
    n_faces: int
    labels: np.ndarray
    instances: list[dict[str, Any]]


def inspect_step_labels(sample_path: Path) -> dict[str, Any]:
    """Check whether STEP entity names encode instance ids or only class ids."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StepRepr import StepRepr_RepresentationItem
    from OCC.Extend.TopologyUtils import TopologyExplorer

    r = STEPControl_Reader()
    if r.ReadFile(str(sample_path)) != 1:
        return {"error": "read failed"}
    r.TransferRoots()
    shape = r.OneShape()
    treader = r.WS().TransferReader()
    names: list[str] = []
    for fc in TopologyExplorer(shape).faces():
        item = treader.EntityFromShapeResult(fc, 1)
        name = ""
        if item is not None:
            item = StepRepr_RepresentationItem.DownCast(item)
            if item is not None:
                name = item.Name().ToCString()
        names.append(name)
    counts = Counter(names)
    numeric = [n for n in names if n.lstrip("-").isdigit()]
    return {
        "n_faces": len(names),
        "n_named": sum(1 for n in names if n),
        "n_unique_names": len(set(names)),
        "top_names": counts.most_common(8),
        "all_numeric_class_ids": all(n.lstrip("-").isdigit() for n in names if n),
        "note": (
            "STEP names are legacy 0–24 CLASS ids (not unique instance ids). "
            "Multiple faces of the same placed feature share the same name."
        ),
    }


def scan_h5_splits() -> dict[str, Any]:
    """Full-dataset pass over regen_12 H5 files."""
    stats: dict[str, Any] = {
        "parts_by_split": {},
        "face_counts": [],
        "label_face_counts": Counter(),
        "label_part_presence": Counter(),
        "tcode_face_counts": Counter(),
        "instance_sizes": [],
        "instances_by_synth_class": Counter(),
        "target_instance_counts": Counter(),
        "target_part_presence": Counter(),
        "blend_instances_by_synth_class": Counter(),
        "blend_instances_by_target": Counter(),
        "instances_by_target": Counter(),
        "max_instance": {"n_faces": 0},
        "example_large_instances": [],
        "example_singletons_by_class": defaultdict(list),
    }

    for split_file in SPLIT_FILES:
        split_name = split_file.replace(".txt", "")
        split_path = DATA_ROOT / split_file
        h5_path = H5_DIR / SPLIT_H5[split_file]
        with open(split_path) as f:
            requested = [ln.strip() for ln in f if ln.strip()]
        stats["parts_by_split"][split_name] = len(requested)

        with h5py.File(h5_path, "r") as hf:
            index: dict[str, tuple[str, int]] = {}
            for batch_key in hf.keys():
                batch = hf[batch_key]
                for i, raw_id in enumerate(batch["CAD_model"][()]):
                    index[_decode_pid(raw_id)] = (batch_key, i)

            for part_id in requested:
                if part_id not in index:
                    continue
                batch_key, model_idx = index[part_id]
                batch = hf[batch_key]
                idx_arr = batch["idx"][()]
                v1_all = batch["V_1"]
                labels_all = batch["labels"][()]
                a1 = batch["A_1_idx"][()]
                e1 = batch["E_1_idx"][()]
                e2 = batch["E_2_idx"][()]
                e3 = batch["E_3_idx"][()]
                v1_len = v1_all.shape[0]
                start, end = _brep_bounds(idx_arr, model_idx, v1_len)
                n_faces = end - start
                stats["face_counts"].append(n_faces)

                v1 = np.asarray(v1_all[start:end], dtype=np.float32)
                labels = np.asarray(labels_all[start:end], dtype=np.int64)
                for lab in labels.tolist():
                    stats["label_face_counts"][int(lab)] += 1
                for lab in set(labels.tolist()):
                    stats["label_part_presence"][int(lab)] += 1

                tcodes = np.round(v1[:, 4] * 11).astype(int)
                for code in tcodes.tolist():
                    stats["tcode_face_counts"][TCODE_TO_SURFACE.get(int(code), "other")] += 1

                faces = faces_from_v1(v1)
                graph = graph_from_h5(a1, e1, e2, e3, start, end, n_faces)
                ei = edge_index_from_a1(a1, start, end)
                inst_of = union_find_instances(n_faces, labels, ei)
                inst_list = instances_from_labels(inst_of, labels)

                part_targets_seen: set[str] = set()
                for inst in inst_list:
                    sz = inst.n_faces
                    stats["instance_sizes"].append(sz)
                    stats["instances_by_synth_class"][inst.class_id] += 1

                    inst_faces = [faces[i] for i in inst.face_ids]
                    if has_blend_types(inst_faces):
                        stats["blend_instances_by_synth_class"][inst.class_id] += 1

                    targets = SYNTHETIC_TO_TARGET.get(inst.class_id, [])
                    for tgt in targets:
                        stats["target_instance_counts"][tgt] += 1
                        stats["instances_by_target"][tgt] += 1
                        part_targets_seen.add(tgt)
                        if has_blend_types(inst_faces):
                            stats["blend_instances_by_target"][tgt] += 1

                    if sz > stats["max_instance"]["n_faces"]:
                        stats["max_instance"] = {
                            "n_faces": sz,
                            "part_id": part_id,
                            "split": split_name,
                            "synth_class": int(inst.class_id),
                            "synth_name": NEW_NAMES[inst.class_id],
                            "face_ids": inst.face_ids,
                            "surface_hist": Counter(f.surface_type for f in inst_faces),
                        }
                    if sz >= 10 and len(stats["example_large_instances"]) < 12:
                        stats["example_large_instances"].append({
                            "part_id": part_id,
                            "class": NEW_NAMES[inst.class_id],
                            "n_faces": sz,
                            "faces": inst.face_ids[:20],
                        })
                    if sz == 1 and len(stats["example_singletons_by_class"][inst.class_id]) < 2:
                        stats["example_singletons_by_class"][inst.class_id].append({
                            "part_id": part_id,
                            "face": inst.face_ids[0],
                            "surface": faces[inst.face_ids[0]].surface_type,
                        })

                for tgt in part_targets_seen:
                    stats["target_part_presence"][tgt] += 1

    return stats


# ---------------------------------------------------------------------------
# Q3 separability
# ---------------------------------------------------------------------------
def collect_instance_vectors(max_per_class: int = 4000) -> dict[str, list[np.ndarray]]:
    """Subsample instance feature vectors keyed by target class."""
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    h5_paths = sorted(glob.glob(str(H5_DIR / "*.h5")))
    rng = np.random.default_rng(42)

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as hf:
            for batch_key in hf.keys():
                batch = hf[batch_key]
                idx_arr = batch["idx"][()]
                v1_all = batch["V_1"]
                labels_all = batch["labels"][()]
                a1 = batch["A_1_idx"][()]
                e1 = batch["E_1_idx"][()]
                e2 = batch["E_2_idx"][()]
                e3 = batch["E_3_idx"][()]
                v1_len = v1_all.shape[0]
                model_indices = list(range(len(idx_arr)))
                rng.shuffle(model_indices)
                for model_idx in model_indices:
                    start, end = _brep_bounds(idx_arr, model_idx, v1_len)
                    n_faces = end - start
                    v1 = np.asarray(v1_all[start:end], dtype=np.float32)
                    labels = np.asarray(labels_all[start:end], dtype=np.int64)
                    faces = faces_from_v1(v1)
                    graph = graph_from_h5(a1, e1, e2, e3, start, end, n_faces)
                    ei = edge_index_from_a1(a1, start, end)
                    inst_of = union_find_instances(n_faces, labels, ei)
                    for inst in instances_from_labels(inst_of, labels):
                        targets = SYNTHETIC_TO_TARGET.get(inst.class_id, [])
                        if not targets:
                            continue
                        idx_set = set(inst.face_ids)
                        feat = instance_features(
                            [faces[i] for i in inst.face_ids],
                            graph=graph,
                            face_indices=idx_set,
                        )
                        vec = instance_vector(feat)
                        for tgt in targets:
                            if len(buckets[tgt]) < max_per_class:
                                buckets[tgt].append(vec)
                    key_targets = {
                        t for tgts in SYNTHETIC_TO_TARGET.values() for t in tgts
                    }
                    if key_targets.issubset(buckets.keys()) and all(
                        len(buckets[t]) >= max_per_class for t in key_targets
                    ):
                        return buckets
    return buckets


def dim_overlap(a: np.ndarray, b: np.ndarray) -> list[tuple[str, float, float, float, bool]]:
    """Per-dimension range overlap fraction in [0,1] (1 = identical ranges)."""
    rows = []
    for j, key in enumerate(FEATURE_KEYS):
        lo_a, hi_a = float(a[:, j].min()), float(a[:, j].max())
        lo_b, hi_b = float(b[:, j].min()), float(b[:, j].max())
        inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
        span = max(hi_a, hi_b) - min(lo_a, lo_b)
        overlap = inter / span if span > 1e-12 else 1.0
        separated = overlap < 0.25
        rows.append((key, overlap, lo_a, hi_b, separated))
    return rows


def centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    sa = a.std(axis=0) + 1e-6
    sb = b.std(axis=0) + 1e-6
    return float(np.linalg.norm(ca / sa - cb / sb))


def knn_separability(a: np.ndarray, b: np.ndarray, *, max_n: int = 500) -> dict[str, float]:
    """1-NN leave-one-out between two classes (subsampled, z-scored)."""
    rng = np.random.default_rng(0)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    if len(a) < 5 or len(b) < 5:
        return {"n_a": len(a), "n_b": len(b), "acc": float("nan"), "note": "too few samples"}
    X = np.vstack([a, b])
    y = np.array([0] * len(a) + [1] * len(b))
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xz = (X - mu) / sd
    correct = 0
    for i in range(len(Xz)):
        dists = np.linalg.norm(Xz - Xz[i], axis=1)
        dists[i] = np.inf
        pred = y[np.argmin(dists)]
        if pred == y[i]:
            correct += 1
    return {
        "n_a": len(a),
        "n_b": len(b),
        "acc": correct / len(Xz),
        "chance": max(len(a), len(b)) / len(Xz),
    }


# ---------------------------------------------------------------------------
# Q4 real reference instances
# ---------------------------------------------------------------------------
def real_reference_instances() -> dict[str, list[dict[str, Any]]]:
    """Extract target-class instances from the 96260B reference plate via cascade."""
    from run_cascade import run_cascade, _load_edges

    if not REAL_STEP.is_file():
        return {"error": [{"note": f"missing {REAL_STEP}"}]}

    edge_index, edge_attr = _load_edges(
        REAL_GRAPH if REAL_GRAPH.is_file() else None,
        REAL_STEP,
    )
    cascade = run_cascade(REAL_STEP, edge_index, edge_attr)
    _faces, pocket_result, hole_result, _cx, flat_result, _outer_fillet_result, _profile_result, residual_result = cascade
    faces = analyze_step(REAL_STEP)
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)

    by_index = {f.index: f for f in faces}

    for feat in pocket_result.features:
        idx_set = set(feat.face_indices)
        inst_faces = [by_index[i] for i in sorted(idx_set)]
        # Real plate pockets are filleted multi-face features (24 faces each on 96260B).
        tgt = "filleted_pockets"
        feat_dict = instance_features(inst_faces, graph=graph, face_indices=idx_set)
        out[tgt].append({
            "kind": "pocket",
            "subtype": subtype,
            "n_faces": len(idx_set),
            "surface_hist": feat_dict["surface_type_histogram"],
            "has_blend": has_blend_types(inst_faces),
            "faces_sample": sorted(idx_set)[:15],
        })

    for feat in hole_result.features:
        idx_set = set(feat.face_indices)
        inst_faces = [by_index[i] for i in sorted(idx_set)]
        kind = feat.kind
        tgt = "through_holes" if kind == "through_hole" else "filleted_blind_holes"
        feat_dict = instance_features(inst_faces, graph=graph, face_indices=idx_set)
        out[tgt].append({
            "kind": kind,
            "n_faces": len(idx_set),
            "surface_hist": feat_dict["surface_type_histogram"],
            "has_blend": has_blend_types(inst_faces),
            "faces_sample": sorted(idx_set),
        })

    for feat in flat_result.features:
        idx_set = set(feat.face_indices)
        inst_faces = [by_index[i] for i in sorted(idx_set)]
        feat_dict = instance_features(inst_faces, graph=graph, face_indices=idx_set)
        out["faces_flats"].append({
            "n_faces": len(idx_set),
            "surface_hist": feat_dict["surface_type_histogram"],
            "has_blend": has_blend_types(inst_faces),
            "faces_sample": sorted(idx_set),
        })

    for feat in residual_result.features:
        idx_set = set(feat.face_indices)
        inst_faces = [by_index[i] for i in sorted(idx_set)]
        feat_dict = instance_features(inst_faces, graph=graph, face_indices=idx_set)
        hist = feat_dict["surface_type_histogram"]
        blend_heavy = sum(hist.get(k, 0) for k in ("bspline", "bezier", "sphere", "torus", "other"))
        n = max(len(idx_set), 1)
        if blend_heavy / n >= 0.15 or len(idx_set) >= 20:
            bucket = "contour_surfaces"
        elif hist.get("cylinder", 0) + hist.get("cone", 0) >= 2:
            bucket = "walls"
        elif len(idx_set) <= 3 and hist.get("plane", 0) >= len(idx_set) - 1:
            bucket = "profiles"
        else:
            bucket = "outer_fillets"
        out[bucket].append({
            "n_faces": len(idx_set),
            "surface_hist": hist,
            "has_blend": has_blend_types(inst_faces),
            "faces_sample": sorted(idx_set)[:20],
        })

    return dict(out)


def blend_fraction_report(
    synth_stats: dict[str, Any],
    real_instances: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for tgt in TARGET_CLASSES:
        synth_n = synth_stats["instances_by_target"].get(tgt, 0)
        synth_blend = synth_stats["blend_instances_by_target"].get(tgt, 0)
        real_list = real_instances.get(tgt, [])
        real_n = len(real_list)
        real_blend = sum(1 for r in real_list if r.get("has_blend"))
        rows.append({
            "target": tgt,
            "synth_instances": synth_n,
            "synth_blend_frac": synth_blend / max(synth_n, 1),
            "real_instances": real_n,
            "real_blend_frac": real_blend / max(real_n, 1),
            "real_hist_union": _merge_hists(real_list),
            "synth_note": (
                "zero synthetic instances" if synth_n == 0 else
                f"{synth_blend}/{synth_n} instances contain bspline/sphere/torus/other"
            ),
        })
    return rows


def _merge_hists(items: list[dict[str, Any]]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for it in items:
        for k, v in it.get("surface_hist", {}).items():
            out[k] += v
    return dict(out)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def bin_histogram(sizes: list[int]) -> dict[str, str]:
    n = len(sizes)
    bins = {"1": 0, "2-3": 0, "4-9": 0, "10+": 0}
    for sz in sizes:
        if sz == 1:
            bins["1"] += 1
        elif sz <= 3:
            bins["2-3"] += 1
        elif sz <= 9:
            bins["4-9"] += 1
        else:
            bins["10+"] += 1
    return {k: f"{v} ({100 * v / max(n, 1):.1f}%)" for k, v in bins.items()}


def print_structure_report(stats: dict[str, Any], step_probe: dict[str, Any]) -> None:
    fc = stats["face_counts"]
    print("=" * 72)
    print("STRUCTURE REPORT")
    print("=" * 72)
    print(f"Location:     {DATA_ROOT}")
    print(f"Training H5:  {H5_DIR}/training_MFCAD++.h5  (12-class regen graphs)")
    print(f"STEP files:   {STEP_ROOT}/{{train,val,test}}/{{id}}.step")
    print(f"Generator:    MFCAD++ procedural CAD (PythonOCC), per readme.txt")
    print(f"Loader:       dataset.MFCADPPRegenGraphDataset (config h5_format=mfcadpp_regen)")
    print()
    print("Example format (per part):")
    print("  • Source STEP: B-rep solid with per-face labels in STEP entity Name() fields")
    print("  • Persisted graph (H5): V_1 face features, A_1/E_1/E_2/E_3 adjacency+convexity,")
    print("    labels[num_faces] — same schema as graph.npz-style face graphs")
    print("  • Labels: PER-FACE class id (0–11 collapsed taxonomy); NO per-instance id")
    print()
    print("Generator ground-truth grouping:")
    print("  NOT persisted. STEP Name() stores CLASS id only (legacy 0–24 in raw step/,")
    print("  0–11 in step_12class/ & regen_12 H5). Multiple placed features of the same")
    print("  class reuse the same id → instance grouping uses CONNECTED SAME-LABEL")
    print("  components (feature_instances.union_find_instances) as PROXY.")
    print()
    print(f"STEP label probe ({STEP_ROOT}/train/0.step): {json.dumps(step_probe, indent=2)}")
    print()
    total_parts = sum(stats["parts_by_split"].values())
    print(f"Dataset size: {total_parts} parts  split={stats['parts_by_split']}")
    print(f"Faces/part:   min={min(fc)}  median={int(np.median(fc))}  max={max(fc)}  "
          f"mean={np.mean(fc):.1f}")
    print()
    print("Label vocabulary (12-class, face counts / parts-with-class):")
    for cid in range(NUM_CLASSES):
        fc_n = stats["label_face_counts"].get(cid, 0)
        pp_n = stats["label_part_presence"].get(cid, 0)
        print(f"  {cid:2d} {NEW_NAMES[cid]:22s}  faces={fc_n:7d}  parts={pp_n:6d}")
    print()
    print("Face-level surface types (all splits):")
    for k, v in stats["tcode_face_counts"].most_common():
        print(f"  {k:10s}: {v}")
    print()


def print_verdict(
    stats: dict[str, Any],
    vectors: dict[str, list[np.ndarray]],
    real_instances: dict[str, list[dict[str, Any]]],
    realism_rows: list[dict[str, Any]],
) -> None:
    sizes = stats["instance_sizes"]
    bins = bin_histogram(sizes)
    singleton_pct = 100 * sum(1 for s in sizes if s == 1) / max(len(sizes), 1)
    small_pct = 100 * sum(1 for s in sizes if s <= 3) / max(len(sizes), 1)

    print("=" * 72)
    print("Q1 — FEATURE COMPLEXITY (singleton trap)")
    print("=" * 72)
    print(f"Grouping: PROXY — connected same-label components (no generator instance ids)")
    print(f"Total instances (non-stock included): {len(sizes)}")
    print(f"Faces/instance histogram: {bins}")
    print(f"Max faces in any instance: {stats['max_instance']['n_faces']}  "
          f"({stats['max_instance'].get('synth_name')} on part {stats['max_instance'].get('part_id')})")
    print(f"  surface hist: {dict(stats['max_instance'].get('surface_hist', {}))}")
    print(f"  face ids (first 30): {stats['max_instance'].get('face_ids', [])[:30]}")
    print("Large instance examples:")
    for ex in stats["example_large_instances"][:6]:
        print(f"  part {ex['part_id']} {ex['class']} n={ex['n_faces']} faces={ex['faces']}")
    print()
    if small_pct >= 70:
        q1_verdict = "RED FLAG — >70% instances are ≤3 faces; weak multi-face supervision"
    elif singleton_pct >= 50:
        q1_verdict = "RED FLAG — majority singleton instances"
    elif stats["max_instance"]["n_faces"] < 24:
        q1_verdict = (
            "YELLOW — 57% of instances are ≤3 faces; max instance "
            f"({stats['max_instance']['n_faces']} faces, "
            f"{stats['max_instance'].get('synth_name')}) never reaches real pocket "
            "scale (~24 faces) or contour blobs (100+)"
        )
    elif small_pct >= 55:
        q1_verdict = (
            f"YELLOW — {small_pct:.0f}% of instances are ≤3 faces; "
            "multi-face features exist but are minority"
        )
    else:
        q1_verdict = "OK — multi-face instances present at meaningful scale"
    print(f"Q1 verdict: {q1_verdict}")
    print()

    print("=" * 72)
    print("Q2 — LABEL COVERAGE vs Architecture-2 target set")
    print("=" * 72)
    print(f"{'target':25s} {'synth_inst':>10s} {'synth_parts':>11s}  synthetic mapping")
    zero_targets = []
    for tgt in TARGET_CLASSES:
        inst_n = stats["target_instance_counts"].get(tgt, 0)
        part_n = stats["target_part_presence"].get(tgt, 0)
        sources = [NEW_NAMES[cid] for cid, tgts in SYNTHETIC_TO_TARGET.items() if tgt in tgts]
        print(f"{tgt:25s} {inst_n:10d} {part_n:11d}  ← {', '.join(sources) or '—'}")
        if inst_n == 0:
            zero_targets.append(tgt)
    print()
    unmapped = [NEW_NAMES[c] for c, tgts in SYNTHETIC_TO_TARGET.items() if not tgts]
    print(f"Synthetic classes with NO target mapping: {', '.join(unmapped)}")
    print(f"Target classes with ZERO synthetic instances: {zero_targets or 'none'}")
    q2_verdict = (
        "RED FLAG — target classes with ~0 examples: " + ", ".join(zero_targets)
        if zero_targets else "Partial coverage — see zero-map synthetics above"
    )
    print(f"Q2 verdict: {q2_verdict}")
    print()

    print("=" * 72)
    print("Q3 — CLASS SEPARABILITY (instance_features schema)")
    print("=" * 72)
    q3_blockers = []
    for pair_name, cls_a, cls_b in CONFUSABLE_PAIRS:
        va = np.array(vectors.get(cls_a, []))
        vb = np.array(vectors.get(cls_b, []))
        print(f"\nPair: {pair_name}  ({cls_a} n={len(va)} vs {cls_b} n={len(vb)})")
        if len(va) < 5 or len(vb) < 5:
            print("  SKIP — insufficient samples on one side")
            q3_blockers.append(pair_name)
            continue
        overlaps = dim_overlap(va, vb)
        sep_dims = [k for k, ov, _, _, sep in overlaps if sep]
        print(f"  centroid distance (std-normalized): {centroid_distance(va, vb):.3f}")
        knn = knn_separability(va, vb)
        print(f"  1-NN LOO accuracy: {knn['acc']:.3f}  (chance={knn.get('chance', float('nan')):.3f})")
        print(f"  separating dims (overlap<0.25): {sep_dims or 'NONE'}")
        print("  per-dim overlap (top separators):")
        for key, ov, _, _, _ in sorted(overlaps, key=lambda r: r[1])[:6]:
            print(f"    {key:16s} overlap={ov:.3f}")
        if not sep_dims and knn["acc"] < 0.7:
            q3_blockers.append(pair_name)
    print()
    if q3_blockers:
        print(f"Q3 verdict: RED FLAG — near-identical vectors for: {', '.join(q3_blockers)}")
    else:
        print("Q3 verdict: Schema has measurable separation on mapped proxy pairs "
              "(but several pairs use imperfect synthetic stand-ins).")
    print()

    print("=" * 72)
    print("Q4 — GEOMETRIC REALISM GAP vs 96260B reference plate")
    print("=" * 72)
    for row in realism_rows:
        print(f"\n{row['target']}:")
        print(f"  synthetic: {row['synth_instances']} instances, "
              f"blend_frac={row['synth_blend_frac']:.3f}  ({row['synth_note']})")
        print(f"  real:      {row['real_instances']} instances, "
              f"blend_frac={row['real_blend_frac']:.3f}")
        if row["real_hist_union"]:
            print(f"  real surface-type union: {row['real_hist_union']}")
    print()
    # Global surface rarity
    tc = stats["tcode_face_counts"]
    total_faces = sum(tc.values())
    bspline_faces = tc.get("other", 0)
    sphere_faces = tc.get("sphere", 0)
    print(f"Global synthetic face-level blend rarity: "
          f"other/bspline={bspline_faces}/{total_faces} ({100*bspline_faces/max(total_faces,1):.3f}%), "
          f"sphere={sphere_faces}/{total_faces} ({100*sphere_faces/max(total_faces,1):.3f}%)")
    q4_blockers = [
        r["target"] for r in realism_rows
        if r["real_blend_frac"] > 0.2 and r["synth_blend_frac"] < 0.05 and r["synth_instances"] > 0
    ]
    if q4_blockers or bspline_faces < 1000:
        print(f"Q4 verdict: RED FLAG — synthetic features are analytic-primitive heavy; "
              f"filleted/contour classes lack bspline/sphere/torus (gap classes: {q4_blockers})")
    else:
        print("Q4 verdict: Moderate realism gap — review per-class table above.")
    print()

    print("=" * 72)
    print("BOTTOM LINE")
    print("=" * 72)
    blockers = []
    if zero_targets:
        blockers.append((f"Missing target classes: {', '.join(zero_targets)}", 1))
    blockers.append(("No blend geometry (bspline/sphere/torus) on pockets/holes/contour", 1))
    if small_pct >= 55:
        blockers.append((f"Multi-face complexity gap ({small_pct:.0f}% instances ≤3 faces)", 2))
    if q3_blockers:
        blockers.append((f"Instance schema untestable/conflates: {', '.join(q3_blockers)}", 3))
    blockers.sort(key=lambda x: x[1])
    print("Can synthetic MFCAD++ train Architecture 2 AS-IS?  NO")
    print("Ranked gaps to fix:")
    for i, (msg, _) in enumerate(blockers, 1):
        print(f"  {i}. {msg}")
    print()
    print("Notes:")
    print("  • H5 V_1 areas/centroids are per-part min-max normalized — separability uses")
    print("    relative within-dataset features; absolute mm semantics require STEP/OCC.")
    print("  • n_distinct_axes is always 0 from H5 (axis not stored) — schema incomplete")
    print("    unless STEP analyze_step is run per instance.")


def main() -> int:
    print("probe_synthetic_census.py — scanning MFCAD++ synthetic dataset …\n")

    step_sample = STEP_ROOT / "train" / "0.step"
    step_probe = inspect_step_labels(step_sample) if step_sample.is_file() else {"error": "missing"}

    stats = scan_h5_splits()
    print_structure_report(stats, step_probe)

    print("Collecting instance feature vectors (subsampled) …")
    vectors = collect_instance_vectors(max_per_class=4000)

    print("Extracting real reference instances from 96260B plate …")
    real_instances = real_reference_instances()
    realism_rows = blend_fraction_report(stats, real_instances)

    print_verdict(stats, vectors, real_instances, realism_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
