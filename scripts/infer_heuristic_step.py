#!/usr/bin/env python3
"""Run Baseline 4 (1-hop LightGBM) heuristic on an arbitrary STEP file + viewer."""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legacy.dataset import _cache_path
from brep.feature_graph import write_feature_graph
from pipeline.core import infer_graph, write_face_predictions
from pipeline.ingest import ingest_step_to_npz
from brep.taxonomy import NUM_CLASSES, NEW_NAMES

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit("lightgbm required: pip install lightgbm") from exc

NODE_DIM = 14
EDGE_DIM = 4
HOP1_FEAT_DIM = NODE_DIM * 3 + EDGE_DIM * 2 + 1


def _neighbor_sets(num_nodes: int, edge_index: np.ndarray) -> list[set[int]]:
    nbrs = [set() for _ in range(num_nodes)]
    for e in range(edge_index.shape[1]):
        u, v = int(edge_index[0, e]), int(edge_index[1, e])
        if u != v:
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


def train_lightgbm(X_train, y_train, X_val, y_val, num_classes: int = NUM_CLASSES,
                   seed: int = 42):
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
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
    return lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )


def _load_split_graphs(cfg: dict, split_file: str):
    path = Path(_cache_path(cfg, split_file))
    if not path.is_file():
        raise SystemExit(f"graph cache missing: {path}")
    return torch.load(path, weights_only=False)


def _stack_hop1(graphs) -> tuple[np.ndarray, np.ndarray]:
    hop1_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for g in graphs:
        x = g.x.numpy()
        ei = g.edge_index.numpy()
        ea = g.edge_attr.numpy()
        hop1, _ = aggregate_1hop_graph(x, ei, ea)
        hop1_parts.append(hop1)
        label_parts.append(g.y.numpy())
    return np.vstack(hop1_parts), np.concatenate(label_parts)


def _predict_probs(model: lgb.Booster, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict(X, num_iteration=model.best_iteration)
    pred = prob.argmax(axis=1).astype(np.int64)
    conf = prob[np.arange(len(pred)), pred].astype(np.float64)
    return pred, conf


def main() -> None:
    ap = argparse.ArgumentParser(description="Heuristic (1-hop LightGBM) STEP inference")
    ap.add_argument("--step", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--open", action="store_true", help="open viewer in browser")
    args = ap.parse_args()

    step_path = args.step.resolve()
    if not step_path.is_file():
        raise SystemExit(f"STEP not found: {step_path}")

    part_id = step_path.stem.replace(" ", "_")
    out_dir = args.out_dir or (ROOT / "pipeline_out" / part_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    print("[heuristic] loading train/val graph cache...")
    train_graphs = _load_split_graphs(cfg, "train.txt")
    val_graphs = _load_split_graphs(cfg, "val.txt")
    X_train, y_train = _stack_hop1(train_graphs)
    X_val, y_val = _stack_hop1(val_graphs)
    del train_graphs, val_graphs
    print(f"[heuristic] train {len(y_train):,} faces, val {len(y_val):,} faces")

    print("[heuristic] training 1-hop LightGBM (Baseline 4)...")
    model = train_lightgbm(X_train, y_train, X_val, y_val, NUM_CLASSES, cfg.get("seed", 42))
    del X_train, y_train, X_val, y_val

    print(f"[heuristic] ingesting {step_path.name}...")
    npz_path = out_dir / "graph.npz"
    x, edge_index, edge_attr = ingest_step_to_npz(step_path, npz_path, cfg)
    hop1, _ = aggregate_1hop_graph(x, edge_index, edge_attr)
    pred, conf = _predict_probs(model, hop1)

    print("[heuristic] building feature graph...")
    graph = infer_graph(pred, conf, edge_index, part_id=part_id)
    graph_path = out_dir / "feature_graph.json"
    pred_path = out_dir / "face_predictions.jsonl"
    write_feature_graph(str(graph_path), graph)
    write_face_predictions(pred_path, pred, conf, edge_index, graph, entity_ids=None)

    counts: dict[str, int] = {}
    for cid in pred:
        name = NEW_NAMES[int(cid)]
        counts[name] = counts.get(name, 0) + 1
    print(f"[heuristic] {graph['n_faces']} faces -> {graph['n_features']} features")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {n}")

    print("[heuristic] building viewer...")
    from feature_graph_viewer.build import DEFAULT_TEMPLATE, build_viewer

    html_path = out_dir / "viewer.html"
    build_viewer(
        part_id=part_id,
        graph_path=graph_path,
        step_path=step_path,
        output_path=html_path,
        template_path=DEFAULT_TEMPLATE,
        open_browser=False,
    )
    print(f"[heuristic] wrote {graph_path}")
    print(f"[heuristic] wrote {pred_path}")
    print(f"[heuristic] wrote {html_path}")

    manifest = {
        "part_id": part_id,
        "method": "heuristic_baseline4_1hop_lightgbm",
        "step_source": str(step_path),
        "class_counts": counts,
        "outputs": {
            "feature_graph": str(graph_path),
            "face_predictions": str(pred_path),
            "viewer": str(html_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.open:
        uri = html_path.resolve().as_uri()
        print(f"[heuristic] opening {uri}")
        webbrowser.open(uri)


if __name__ == "__main__":
    main()
