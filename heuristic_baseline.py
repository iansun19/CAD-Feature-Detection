"""
heuristic_baseline.py — adjacency-aware LightGBM baselines vs the GNN.

Baseline 4 (1-hop): per-face row = own 14-d node features + mean/max of neighbors'
node features + mean/max of incident edge features (4-d) + degree.
Baseline 5 (2-hop): Baseline 4 columns + mean/max of each 1-hop *neighbor-aggregate*
column over the face's direct neighbors (chained aggregation, not a deduplicated
2-hop neighborhood — see TWO_HOP_NOTE below).

Feature sources (identical to MFCADPPRegenGraphDataset / cached PyG graphs):
  Node x [N, 14] from build_node_features_regen:
    cols 0-5   surface-type one-hot (plane, cylinder, cone, sphere, torus, other)
    col  6     log-scaled area (per-part standardized)
    cols 7-9   centroid xyz (per-part centered)
    cols 10-12 unit face normal xyz
    col  13    plane-d (signed-log, per-part standardized)
  Edge edge_attr [E, 4] from build_edge_features_regen:
    cols 0-2   convexity one-hot (concave, convex, smooth)
    col  3     cos(dihedral angle)

Split: train.txt / val.txt / test.txt list *part* (CAD model) ids. Every face from
a part stays in one split — same as train.py / evaluate.py / baseline_eval.py.

Edges: regen loader calls make_undirected — each B-rep adjacency appears as both
(u, v) and (v, u). Neighbors of face i = unique endpoints v with an edge (i, v)
in edge_index (one entry per neighbor, no double-count from reverse arcs).

TWO_HOP_NOTE: Baseline 5 is an *approximation*, not a true 2-hop induced subgraph.
For face i we take mean/max over direct neighbors j of j's 1-hop neighbor-aggregate
vector (nb_mean, nb_max, edge_mean, edge_max — 36 dims). Nodes at graph distance 2
from i are included only indirectly through their parent's aggregates; nodes reached
via multiple paths can influence i multiple times; the center face i is excluded.

Usage:
    python heuristic_baseline.py
    python heuristic_baseline.py --run runs_cloud/20260630_042957
    python heuristic_baseline.py --skip-build   # reuse saved tables
    python heuristic_baseline.py --skip-train     # tables only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

from baseline_eval import (
    baseline_majority_class,
    baseline_surface_type_majority,
    latest_run,
    load_cache,
    load_gnn_metrics,
    stack_faces,
)
from dataset import _build_uncached, _cache_path, build_cache
from device import resolve_device, set_seed
from evaluate import load_class_names, per_class_metrics
from taxonomy import NUM_CLASSES

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "lightgbm is required: pip install lightgbm pandas pyarrow"
    ) from exc

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required: pip install pandas") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


SURFACE_NAMES = ("plane", "cylinder", "cone", "sphere", "torus", "other")
CONVEXITY_NAMES = ("concave", "convex", "smooth")

# Interim weak-class set: legacy weak ids remapped via taxonomy.old_to_new and deduped
# (0→9 chamfer; 5,6,7→2 through_slot; 8,9,10→3 through_step; 17,18,19→7 blind_slot).
# TODO: re-derive from a fresh 12-class confusion matrix after the first 12-class eval —
# collapsing may have removed some formerly-weak distinctions.
WEAK_CLASS_IDS = (9, 2, 3, 7)

NODE_DIM = 14
EDGE_DIM = 4
HOP1_NBR_AGG_DIM = NODE_DIM * 2 + EDGE_DIM * 2  # nb_mean + nb_max + edge_mean + edge_max
HOP1_FEAT_DIM = NODE_DIM * 3 + EDGE_DIM * 2 + 1  # self + nb_mean + nb_max + edge_* + degree
HOP2_EXTRA_DIM = HOP1_NBR_AGG_DIM * 2

ZERO_NBR_FALLBACK = "zero_vector"  # documented fallback for degree-0 faces


def node_feature_names(num_surface_types: int = 6) -> list[str]:
    names = [f"surf_{SURFACE_NAMES[i]}" for i in range(num_surface_types)]
    names += ["area", "centroid_x", "centroid_y", "centroid_z",
              "normal_x", "normal_y", "normal_z", "plane_d"]
    assert len(names) == NODE_DIM
    return names


def edge_feature_names() -> list[str]:
    names = [f"conv_{CONVEXITY_NAMES[i]}" for i in range(3)]
    names.append("cos_dihedral")
    assert len(names) == EDGE_DIM
    return names


def hop1_feature_names(num_surface_types: int = 6) -> list[str]:
    nf, ef = node_feature_names(num_surface_types), edge_feature_names()
    names = [f"self_{c}" for c in nf]
    names += [f"nb_mean_{c}" for c in nf]
    names += [f"nb_max_{c}" for c in nf]
    names += [f"edge_mean_{c}" for c in ef]
    names += [f"edge_max_{c}" for c in ef]
    names.append("degree")
    assert len(names) == HOP1_FEAT_DIM
    return names


def hop1_neighbor_agg_names(num_surface_types: int = 6) -> list[str]:
    hop1 = hop1_feature_names(num_surface_types)
    return [c for c in hop1 if c.startswith(("nb_mean_", "nb_max_", "edge_mean_", "edge_max_"))]


def hop2_feature_names(num_surface_types: int = 6) -> list[str]:
    base = hop1_neighbor_agg_names(num_surface_types)
    names = []
    for c in base:
        names.append(f"hop2_mean_{c}")
        names.append(f"hop2_max_{c}")
    assert len(names) == HOP2_EXTRA_DIM
    return names


def feature_schema(num_surface_types: int = 6) -> dict:
    return {
        "node_features": {
            "dim": NODE_DIM,
            "source": "build_node_features_regen(v1, num_surface_types)",
            "columns": node_feature_names(num_surface_types),
        },
        "edge_features": {
            "dim": EDGE_DIM,
            "source": "build_edge_features_regen(convexity_ids, cos_angles)",
            "columns": edge_feature_names(),
        },
        "baseline4_1hop": {
            "dim": HOP1_FEAT_DIM,
            "columns": hop1_feature_names(num_surface_types),
            "zero_neighbor_fallback": ZERO_NBR_FALLBACK,
        },
        "baseline5_2hop_extra": {
            "dim": HOP2_EXTRA_DIM,
            "columns": hop2_feature_names(num_surface_types),
            "note": "chained mean/max over neighbors' 1-hop neighbor-aggregate columns; "
                    "approximation, not deduplicated 2-hop neighborhood",
        },
        "split_granularity": "part (CAD model id); faces from the same part never cross splits",
        "edge_direction": "undirected via make_undirected (both (u,v) and (v,u) stored)",
    }


def load_split_graphs(cfg: dict, split_file: str):
    """Cached PyG graphs + matching part ids (cache row i == dataset.ids[i])."""
    cache_path = _cache_path(cfg, split_file)
    if not os.path.isfile(cache_path):
        print(f"[cache] building {cache_path} (one-time)...", flush=True)
        build_cache(cfg, split_file)
    data_list = torch.load(cache_path, weights_only=False)
    ds = _build_uncached(cfg, split_file)
    part_ids = ds.ids
    if hasattr(ds, "_close_h5"):
        ds._close_h5()
    if len(part_ids) != len(data_list):
        raise RuntimeError(
            f"part id count {len(part_ids)} != cache graphs {len(data_list)} for {split_file}")
    return part_ids, data_list


def _neighbor_sets(num_nodes: int, edge_index: np.ndarray) -> list[set[int]]:
    nbrs = [set() for _ in range(num_nodes)]
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u == v:
            continue
        nbrs[u].add(v)
    return nbrs


def _incident_edge_features(num_nodes: int, edge_index: np.ndarray,
                            edge_attr: np.ndarray) -> list[list[np.ndarray]]:
    inc = [[] for _ in range(num_nodes)]
    for e in range(edge_index.shape[1]):
        u = int(edge_index[0, e])
        inc[u].append(edge_attr[e])
    return inc


def aggregate_1hop_graph(x: np.ndarray, edge_index: np.ndarray,
                         edge_attr: np.ndarray) -> tuple[np.ndarray, int]:
    """1-hop features for one part. Returns (features [N, HOP1_FEAT_DIM], n_zero_degree)."""
    n = x.shape[0]
    nbrs = _neighbor_sets(n, edge_index)
    inc = _incident_edge_features(n, edge_index, edge_attr)
    out = np.zeros((n, HOP1_FEAT_DIM), dtype=np.float32)
    zero_deg = 0

    for i in range(n):
        out[i, :NODE_DIM] = x[i]
        deg = len(nbrs[i])
        out[i, -1] = float(deg)
        if deg == 0:
            zero_deg += 1
            continue
        nb_idx = sorted(nbrs[i])
        nb_x = x[nb_idx]
        out[i, NODE_DIM:NODE_DIM * 2] = nb_x.mean(axis=0)
        out[i, NODE_DIM * 2:NODE_DIM * 3] = nb_x.max(axis=0)
        e_feats = np.stack(inc[i], axis=0)
        e_off = NODE_DIM * 3
        out[i, e_off:e_off + EDGE_DIM] = e_feats.mean(axis=0)
        out[i, e_off + EDGE_DIM:e_off + EDGE_DIM * 2] = e_feats.max(axis=0)
    return out, zero_deg


def aggregate_2hop_graph(hop1: np.ndarray, nbrs: list[set[int]]) -> np.ndarray:
    """2-hop chained aggregates for one part; hop1 is [N, HOP1_FEAT_DIM]."""
    n = hop1.shape[0]
    # indices of neighbor-aggregate columns inside hop1
    agg_start = NODE_DIM
    agg_end = NODE_DIM * 3 + EDGE_DIM * 2
    agg_cols = hop1[:, agg_start:agg_end]
    out = np.zeros((n, HOP2_EXTRA_DIM), dtype=np.float32)

    for i in range(n):
        if not nbrs[i]:
            continue
        nb_vecs = agg_cols[sorted(nbrs[i])]
        # interleave mean/max per source column
        means = nb_vecs.mean(axis=0)
        maxes = nb_vecs.max(axis=0)
        for j in range(HOP1_NBR_AGG_DIM):
            out[i, j * 2] = means[j]
            out[i, j * 2 + 1] = maxes[j]
    return out


def build_split_table(cfg: dict, split_file: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build Baseline 4/5 tables for one split using preallocated arrays."""
    split = os.path.splitext(split_file)[0]
    hop1_names = hop1_feature_names(cfg["num_surface_types"])
    hop2_names = hop2_feature_names(cfg["num_surface_types"])

    part_ids, graphs = load_split_graphs(cfg, split_file)
    n_faces = sum(int(g.num_nodes) for g in graphs)
    n_parts = len(part_ids)

    hop1_arr = np.zeros((n_faces, HOP1_FEAT_DIM), dtype=np.float32)
    hop2_arr = np.zeros((n_faces, HOP2_EXTRA_DIM), dtype=np.float32)
    labels = np.zeros(n_faces, dtype=np.int64)
    face_ids = np.zeros(n_faces, dtype=np.int32)
    part_id_col: list[str] = []

    offset = 0
    zero_deg = 0
    for part_id, g in zip(part_ids, graphs):
        x = g.x.numpy()
        ei = g.edge_index.numpy()
        ea = g.edge_attr.numpy()
        y = g.y.numpy()
        hop1, zdeg = aggregate_1hop_graph(x, ei, ea)
        nbrs = _neighbor_sets(x.shape[0], ei)
        hop2 = aggregate_2hop_graph(hop1, nbrs)

        n = x.shape[0]
        zero_deg += zdeg
        hop1_arr[offset:offset + n] = hop1
        hop2_arr[offset:offset + n] = hop2
        labels[offset:offset + n] = y
        face_ids[offset:offset + n] = np.arange(n, dtype=np.int32)
        part_id_col.extend([part_id] * n)
        offset += n

    del graphs, part_ids

    meta_df = pd.DataFrame({
        "part_id": part_id_col,
        "face_id": face_ids,
        "split": split,
        "label": labels,
    })
    df4 = pd.concat([meta_df, pd.DataFrame(hop1_arr, columns=hop1_names)], axis=1)
    df5 = pd.concat([
        meta_df,
        pd.DataFrame(hop1_arr, columns=hop1_names),
        pd.DataFrame(hop2_arr, columns=hop2_names),
    ], axis=1)

    stats = {
        "split": split,
        "parts": n_parts,
        "faces": n_faces,
        "zero_degree_faces": zero_deg,
    }
    return df4, df5, stats


def build_and_save_tables(cfg: dict, path4: str, path5: str) -> dict:
    """Build tables split-by-split; stream to parquet without holding all rows."""
    stats = {
        "zero_degree_faces": 0,
        "total_faces": 0,
        "parts_per_split": {},
    }
    writer4 = writer5 = None
    out4, out5 = path4 + ".parquet", path5 + ".parquet"
    for old in (out4, out5):
        if os.path.isfile(old):
            os.remove(old)

    for split_file in ("train.txt", "val.txt", "test.txt"):
        df4s, df5s, s = build_split_table(cfg, split_file)
        stats["zero_degree_faces"] += s["zero_degree_faces"]
        stats["total_faces"] += s["faces"]
        stats["parts_per_split"][s["split"]] = s["parts"]
        print(f"  {s['split']}: {s['parts']:,} parts, {s['faces']:,} faces, "
              f"zero-degree={s['zero_degree_faces']:,}", flush=True)

        if pq is not None:
            t4 = pa.Table.from_pandas(df4s, preserve_index=False)
            t5 = pa.Table.from_pandas(df5s, preserve_index=False)
            if writer4 is None:
                writer4 = pq.ParquetWriter(out4, t4.schema)
                writer5 = pq.ParquetWriter(out5, t5.schema)
            writer4.write_table(t4)
            writer5.write_table(t5)
        del df4s, df5s

    if writer4 is not None:
        writer4.close()
        writer5.close()
        return stats

    # CSV fallback: still build per-split then concat one split at a time
    frames4, frames5 = [], []
    for split_file in ("train.txt", "val.txt", "test.txt"):
        df4s, df5s, _ = build_split_table(cfg, split_file)
        frames4.append(df4s)
        frames5.append(df5s)
    save_table(pd.concat(frames4, ignore_index=True), path4)
    save_table(pd.concat(frames5, ignore_index=True), path5)
    return stats


def build_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Legacy in-memory build (small datasets / tests). Prefer build_and_save_tables."""
    dfs4, dfs5 = [], []
    stats = {
        "zero_degree_faces": 0,
        "total_faces": 0,
        "parts_per_split": {},
    }
    for split_file in ("train.txt", "val.txt", "test.txt"):
        df4s, df5s, s = build_split_table(cfg, split_file)
        dfs4.append(df4s)
        dfs5.append(df5s)
        stats["zero_degree_faces"] += s["zero_degree_faces"]
        stats["total_faces"] += s["faces"]
        stats["parts_per_split"][s["split"]] = s["parts"]
        print(f"  {s['split']}: {s['parts']:,} parts, {s['faces']:,} faces, "
              f"zero-degree={s['zero_degree_faces']:,}", flush=True)
    return pd.concat(dfs4, ignore_index=True), pd.concat(dfs5, ignore_index=True), stats


def save_table(df: pd.DataFrame, path_base: str) -> str:
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    parquet_path = path_base + ".parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        csv_path = path_base + ".csv"
        print(f"  parquet write failed ({exc}); falling back to {csv_path}")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_table(path_base: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        p = path_base + ext
        if os.path.isfile(p):
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"no table at {path_base}.{{parquet,csv}}")


def load_table_split(path_base: str, split: str) -> pd.DataFrame:
    """Load one split from a parquet table without reading the full file when possible."""
    p = path_base + ".parquet"
    if os.path.isfile(p):
        return pd.read_parquet(p, filters=[("split", "=", split)])
    df = load_table(path_base)
    return df[df["split"] == split].copy()


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=np.float32)


def train_lightgbm(X_train, y_train, X_val, y_val, num_classes: int = NUM_CLASSES, seed: int = 42):
    assert num_classes == NUM_CLASSES
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "learning_rate": 0.1,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )
    return model


def predict_lightgbm(model, X: np.ndarray) -> np.ndarray:
    prob = model.predict(X, num_iteration=model.best_iteration)
    return prob.argmax(axis=1).astype(np.int64)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                             num_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    precision, recall, f1, support = per_class_metrics(cm)
    acc = (y_true == y_pred).mean()
    present = support > 0
    macro_f1 = f1[present].mean() if present.any() else 0.0
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm: np.ndarray, path: str, num_classes: int):
    header = "true\\pred," + ",".join(str(i) for i in range(num_classes))
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(num_classes):
            f.write(str(i) + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")


def save_importance(model, feature_names: list[str], path: str):
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "gain"])
        for rank, idx in enumerate(order, start=1):
            w.writerow([rank, feature_names[idx], float(imp[idx])])


def print_importance(model, feature_names: list[str], title: str, top_k: int = 25):
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(-imp)[:top_k]
    print(f"\n{'=' * 72}")
    print(f"LightGBM feature importance ({title}) — top {top_k} by gain")
    print(f"{'=' * 72}")
    print(f"{'rank':>4}  {'feature':<40} {'gain':>12}")
    print("-" * 72)
    for rank, idx in enumerate(order, start=1):
        print(f"{rank:>4}  {feature_names[idx]:<40} {imp[idx]:>12.1f}")


def evaluate_baselines_123(cfg: dict, device, knn_k: int = 5,
                         max_train_faces: int | None = 200_000) -> dict[str, dict]:
    """Return per-method metrics dicts with f1 arrays on the test split."""
    num_classes = cfg["num_classes"]
    num_st = cfg["num_surface_types"]
    train_data = load_cache(cfg, "train.txt")
    test_data = load_cache(cfg, "test.txt")
    st_tr, y_tr, X_tr = stack_faces(train_data, num_st)
    st_te, y_te, X_te = stack_faces(test_data, num_st)
    y_test = y_te.numpy()

    out = {}

    maj_id, _, _, _, _ = baseline_majority_class(y_tr, y_te, num_classes)
    preds = np.full(len(y_test), maj_id, dtype=np.int64)
    out["majority_class"] = metrics_from_predictions(y_test, preds, num_classes)

    surface_majority, seen_st, _, _, _, _ = baseline_surface_type_majority(
        st_tr, y_tr, st_te, y_te, num_classes, num_st, global_fallback=maj_id)
    preds = surface_majority[st_te.numpy()]
    out["per_surface_type"] = metrics_from_predictions(y_test, preds, num_classes)

    X_knn, y_knn = X_tr, y_tr
    if max_train_faces and len(y_tr) > max_train_faces:
        rng = torch.Generator().manual_seed(cfg.get("seed", 42))
        idx = torch.randperm(len(y_tr), generator=rng)[:max_train_faces]
        X_knn, y_knn = X_tr[idx], y_tr[idx]
    eps = 1e-8
    X_train_n = X_knn / X_knn.norm(dim=1, keepdim=True).clamp_min(eps)
    X_test_n = X_te / X_te.norm(dim=1, keepdim=True).clamp_min(eps)
    X_train_d = X_train_n.to(device)
    y_train_d = y_knn.to(device)
    knn_preds = []
    batch_size = 8192
    for i in range(0, len(X_test_n), batch_size):
        batch = X_test_n[i:i + batch_size].to(device)
        sim = batch @ X_train_d.T
        nn_idx = sim.topk(knn_k, dim=1).indices
        nn_labels = y_train_d[nn_idx]
        votes = torch.zeros(batch.size(0), num_classes, device=device)
        ones = torch.ones(batch.size(0), knn_k, device=device)
        votes.scatter_add_(1, nn_labels, ones)
        knn_preds.append(votes.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(knn_preds)
    out["knn"] = metrics_from_predictions(y_test, preds, num_classes)
    return out


def load_gnn_full_metrics(run_dir: str, num_classes: int) -> dict:
    path = os.path.join(run_dir, "per_class_metrics.csv")
    cm_path = os.path.join(run_dir, "confusion_matrix.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    f1 = np.zeros(num_classes, dtype=np.float64)
    support = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    for r in rows:
        c = int(r["id"])
        f1[c] = float(r["f1"])
        support[c] = int(r["support"])
        recall[c] = float(r["recall"])
    acc, macro_f1 = load_gnn_metrics(run_dir)
    cm = None
    if os.path.isfile(cm_path):
        with open(cm_path) as f:
            lines = f.readlines()[1:]
        cm = np.array([[int(x) for x in ln.strip().split(",")[1:]] for ln in lines],
                      dtype=np.int64)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm,
    }


def comparison_row(name: str, m: dict, class_names: list[str]) -> dict:
    row = {
        "model": name,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
    }
    for c in WEAK_CLASS_IDS:
        key = f"f1_c{c}_{class_names[c].replace(' ', '_')}"
        row[key] = float(m["f1"][c]) if m.get("f1") is not None else float("nan")
    return row


def write_comparison_table(rows: list[dict], class_names: list[str], out_dir: str):
    csv_path = os.path.join(out_dir, "comparison.csv")
    md_path = os.path.join(out_dir, "comparison.md")

    weak_headers = []
    for c in WEAK_CLASS_IDS:
        weak_headers.append(f"f1_c{c}")

    fieldnames = ["model", "accuracy", "macro_f1"] + weak_headers
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            slim = {"model": row["model"],
                    "accuracy": f"{row['accuracy']:.4f}",
                    "macro_f1": f"{row['macro_f1']:.4f}"}
            for c in WEAK_CLASS_IDS:
                slim[f"f1_c{c}"] = f"{row[f'f1_c{c}_{class_names[c].replace(' ', '_')}']:.4f}"
            w.writerow(slim)

    hdr = "| Model | Accuracy | Macro-F1 |"
    sep = "|---|---:|---:|"
    for c in WEAK_CLASS_IDS:
        hdr += f" F1 c{c} ({class_names[c][:18]}) |"
        sep += "---:|"
    lines = [hdr, sep]
    for row in rows:
        line = f"| {row['model']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |"
        for c in WEAK_CLASS_IDS:
            key = f"f1_c{c}_{class_names[c].replace(' ', '_')}"
            line += f" {row[key]:.4f} |"
        lines.append(line)
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path


def print_comparison(rows: list[dict], class_names: list[str]):
    weak_hdr = "  ".join(f"c{c:>2}" for c in WEAK_CLASS_IDS)
    print(f"\n{'=' * 100}")
    print("COMPARISON (test split)")
    print(f"{'=' * 100}")
    print(f"{'Model':<32} {'Acc':>7} {'MacroF1':>8}  weak-class F1: {weak_hdr}")
    print("-" * 100)
    for row in rows:
        weak_vals = "  ".join(
            f"{row[f'f1_c{c}_{class_names[c].replace(' ', '_')}']:>5.3f}"
            for c in WEAK_CLASS_IDS)
        print(f"{row['model']:<32} {row['accuracy']:>7.4f} {row['macro_f1']:>8.4f}  {weak_vals}")
    print("\nWeak classes:", ", ".join(f"{c}={class_names[c]!r}" for c in WEAK_CLASS_IDS))


def main():
    ap = argparse.ArgumentParser(description="Adjacency + LightGBM heuristic baselines")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="heuristic_baselines",
                    help="output directory for tables, models, metrics")
    ap.add_argument("--run", default=None, help="GNN run dir with per_class_metrics.csv")
    ap.add_argument("--skip-build", action="store_true", help="reuse saved feature tables")
    ap.add_argument("--skip-train", action="store_true", help="build tables only")
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--max-train-faces-knn", type=int, default=200_000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    num_classes = cfg["num_classes"]
    class_names = load_class_names(cfg["data_root"], num_classes)
    device = resolve_device(args.device or cfg.get("device", "auto"))
    os.makedirs(args.out_dir, exist_ok=True)

    schema = feature_schema(cfg["num_surface_types"])
    schema_path = os.path.join(args.out_dir, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print("Feature audit (see feature_schema.json):")
    print(f"  node dim={NODE_DIM}, edge dim={EDGE_DIM}, 1-hop dim={HOP1_FEAT_DIM}, "
          f"2-hop extra dim={HOP2_EXTRA_DIM}")
    print(f"  split: by part id (train.txt / val.txt / test.txt)")
    print(f"  edges: undirected pairs via make_undirected in MFCADPPRegenGraphDataset")
    print(f"  zero-neighbor fallback: {ZERO_NBR_FALLBACK}")

    path4 = os.path.join(args.out_dir, "baseline4_1hop")
    path5 = os.path.join(args.out_dir, "baseline5_2hop")

    if args.skip_build:
        print(f"Using existing tables under {args.out_dir}/")
    else:
        print("\nBuilding feature tables...")
        stats = build_and_save_tables(cfg, path4, path5)
        out4, out5 = path4 + ".parquet", path5 + ".parquet"
        if not os.path.isfile(out4):
            out4 = path4 + ".csv"
            out5 = path5 + ".csv"
        print(f"\nWrote {out4}")
        print(f"Wrote {out5}")
        print(f"Zero-degree faces: {stats['zero_degree_faces']:,} / "
              f"{stats['total_faces']:,} "
              f"({100 * stats['zero_degree_faces'] / max(stats['total_faces'], 1):.4f}%)")
        print(f"Parts per split: {stats['parts_per_split']}")

    hop1_cols = hop1_feature_names(cfg["num_surface_types"])
    hop5_cols = hop1_cols + hop2_feature_names(cfg["num_surface_types"])

    if args.skip_train:
        print("\n--skip-train set; done.")
        return

    def load_xy(path_base: str, feat_cols: list[str]):
        xy = {}
        for split in ("train", "val", "test"):
            df = load_table_split(path_base, split)
            xy[split] = (
                _feature_matrix(df, feat_cols),
                df["label"].to_numpy(dtype=np.int64),
            )
            print(f"  {split}: {len(xy[split][1]):,} faces", flush=True)
        return xy

    results = {}

    for tag, feat_cols, path_base in (
        ("baseline4_1hop", hop1_cols, path4),
        ("baseline5_2hop", hop5_cols, path5),
    ):
        print(f"\nLoading feature matrices for {tag}...")
        xy = load_xy(path_base, feat_cols)
        print(f"Training LightGBM ({tag})...")
        model = train_lightgbm(
            xy["train"][0], xy["train"][1],
            xy["val"][0], xy["val"][1],
            num_classes, cfg.get("seed", 42))
        model_path = os.path.join(args.out_dir, f"lgbm_{tag}.txt")
        model.save_model(model_path)

        imp_path = os.path.join(args.out_dir, f"lgbm_{tag}_importance.csv")
        save_importance(model, feat_cols, imp_path)
        print_importance(model, feat_cols, tag)

        y_pred = predict_lightgbm(model, xy["test"][0])
        m = metrics_from_predictions(xy["test"][1], y_pred, num_classes)
        cm_path = os.path.join(args.out_dir, f"confusion_matrix_{tag}.csv")
        save_confusion_matrix(m["confusion_matrix"], cm_path, num_classes)
        results[tag] = m
        del xy, model
        print(f"  saved model -> {model_path}")
        print(f"  saved importance -> {imp_path}")
        print(f"  test accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")

    print("\nEvaluating existing baselines 1-3 on test...")
    b123 = evaluate_baselines_123(
        cfg, device, knn_k=args.knn_k, max_train_faces=args.max_train_faces_knn)

    run_dir = args.run or latest_run(cfg["out_dir"])
    if run_dir is None:
        cloud = sorted(glob.glob(os.path.join("runs_cloud", "*")))
        cloud = [r for r in cloud if os.path.isfile(os.path.join(r, "per_class_metrics.csv"))]
        run_dir = cloud[-1] if cloud else None

    gnn_metrics = None
    if run_dir and os.path.isfile(os.path.join(run_dir, "per_class_metrics.csv")):
        gnn_metrics = load_gnn_full_metrics(run_dir, num_classes)
        print(f"GNN metrics from {run_dir}")
    else:
        print("WARNING: no GNN run found; comparison table will omit GNN row")

    comp_rows = [
        comparison_row("majority class", b123["majority_class"], class_names),
        comparison_row("per-surface-type majority", b123["per_surface_type"], class_names),
        comparison_row(f"kNN (k={args.knn_k})", b123["knn"], class_names),
        comparison_row("Baseline 4 (1-hop GBM)", results["baseline4_1hop"], class_names),
        comparison_row("Baseline 5 (2-hop GBM)", results["baseline5_2hop"], class_names),
    ]
    if gnn_metrics is not None:
        comp_rows.append(comparison_row("GNN", gnn_metrics, class_names))

    print_comparison(comp_rows, class_names)
    csv_path, md_path = write_comparison_table(comp_rows, class_names, args.out_dir)
    print(f"\nWrote {csv_path}\nWrote {md_path}")
    print(f"\nTwo-hop note: Baseline 5 uses chained mean/max over neighbors' 1-hop "
          f"neighbor-aggregate columns ({HOP1_NBR_AGG_DIM} dims -> {HOP2_EXTRA_DIM} new cols). "
          "This is an approximation, not a deduplicated 2-hop neighborhood excluding self.")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
