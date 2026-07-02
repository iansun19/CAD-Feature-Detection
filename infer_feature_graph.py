#!/usr/bin/env python3
"""
infer_feature_graph.py — STEP, H5 part, or Stage-A NPZ → feature_graph.json.

Runs the trained GNN, groups faces into feature instances, and writes a
machine-readable feature graph (nodes = machining features, edges = adjacency).

With --with-params (default when STEP is available), attaches per-feature geometry
from the STEP file (requires pythonocc).

Usage:
    python infer_feature_graph.py --part-id 29000
    python infer_feature_graph.py --part-id 29000 --split test.txt -o out.json
    python infer_feature_graph.py --part-id 29000 --with-params
    python infer_feature_graph.py --npz nist_ctc_01_stage_a.npz -o nist_feature_graph.json
    python infer_feature_graph.py --step path/to/part.step -o out.json   # requires pythonocc
"""
from __future__ import annotations

import env_bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from device import resolve_device
from feature_graph import build_feature_graph, summarize_graph, write_feature_graph
from feature_params import default_step_path, enrich_graph_with_params
from model import BRepGNN
from taxonomy import NUM_CLASSES, OLD_TO_NEW

NUM_OLD = 25
REPO_ROOT = Path(__file__).resolve().parent


def latest_run(out_dir: str) -> str:
    runs = sorted(glob.glob(os.path.join(out_dir, "*")))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "best_model.pt"))]
    if not runs:
        raise SystemExit(f"no run with best_model.pt under {out_dir}/")
    return runs[-1]


def default_run_dir(out_dir: str) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    promoted = os.path.join(root, "runs_cloud_latest")
    if os.path.isfile(os.path.join(promoted, "best_model.pt")):
        return promoted
    return latest_run(out_dir)


def collapse_matrix() -> np.ndarray:
    m = np.zeros((NUM_OLD, NUM_CLASSES), dtype=np.float64)
    for old, new in OLD_TO_NEW.items():
        m[old, new] = 1.0
    return m


def load_model(ckpt: str, cfg: dict, device: torch.device) -> BRepGNN:
    sd = torch.load(ckpt, map_location=device)
    head_w = sd["head.3.weight"].shape[0]
    node_in = sd["input_proj.weight"].shape[1]
    edge_in = sd["edge_proj.weight"].shape[1]
    model = BRepGNN(
        node_in, edge_in, cfg["hidden_dim"], head_w,
        cfg["num_layers"], cfg["dropout"],
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, head_w


@torch.no_grad()
def predict_faces(model, x, edge_index, edge_attr, head_w: int):
    logits = model(x, edge_index, edge_attr)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    if head_w == NUM_CLASSES:
        probs12 = probs
    else:
        probs12 = probs @ collapse_matrix()
    pred = probs12.argmax(1).astype(np.int64)
    conf = probs12[np.arange(len(pred)), pred].astype(np.float64)
    return pred, conf


def infer_from_tensors(model, x, edge_index, edge_attr, head_w, device):
    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    edge_index = torch.from_numpy(np.asarray(edge_index, dtype=np.int64)).to(device)
    edge_attr = torch.from_numpy(np.asarray(edge_attr, dtype=np.float32)).to(device)
    return predict_faces(model, x, edge_index, edge_attr, head_w)


def load_part_from_dataset(cfg: dict, part_id: str, split_file: str):
    from dataset import get_dataset

    split_path = os.path.join(cfg["data_root"], split_file)
    with open(split_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    if part_id not in ids:
        raise SystemExit(f"part id {part_id!r} not in {split_file}")
    ds = get_dataset(cfg, split_file)
    data = ds[ids.index(part_id)]
    if hasattr(ds, "_close_h5"):
        ds._close_h5()
    return data


def load_from_npz(path: str):
    d = np.load(path)
    return {
        "x": d["x"],
        "edge_index": d["edge_index"],
        "edge_attr": d["edge_attr"],
        "entity_ids": d.get("entity_ids"),
        "part_id": os.path.splitext(os.path.basename(path))[0],
    }


def load_from_step(path: str, cfg: dict):
    from pipeline.ingest import require_occ

    require_occ("STEP inference")
    from step_ingest import ingest_step_to_pyg

    x, edge_index, edge_attr, stats = ingest_step_to_pyg(
        path,
        num_surface_types=cfg["num_surface_types"],
        angle_reduce=cfg.get("angle_reduce", "median"),
        include_curvature=cfg.get("include_curvature", False),
    )
    if not stats.success:
        raise SystemExit(f"STEP ingest failed: {stats.error}")
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "entity_ids": None,
        "part_id": os.path.splitext(os.path.basename(path))[0],
    }


def main():
    ap = argparse.ArgumentParser(description="Infer and export a feature graph")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--part-id", help="MFCAD++ part id from a split file")
    src.add_argument("--step", help="path to a STEP file (requires pythonocc)")
    src.add_argument("--npz", help="Stage-A NPZ (x, edge_index, edge_attr, entity_ids)")
    ap.add_argument("--split", default="test.txt", help="split file for --part-id")
    ap.add_argument("--output", "-o", default=None, help="output JSON path")
    ap.add_argument("--run", default=None, help="checkpoint run dir")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--with-params", action=argparse.BooleanOptionalAction, default=None,
        help="attach geometry params from STEP (default: on if STEP file exists)",
    )
    ap.add_argument(
        "--step-dir", default=None,
        help="STEP directory for --part-id (default: MFCAD++_dataset/step/test)",
    )
    ap.add_argument(
        "--step-file", default=None,
        help="explicit STEP path (overrides --step-dir lookup for --part-id / --npz)",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg.get("device", "auto"))
    run_dir = args.run or default_run_dir(cfg["out_dir"])
    ckpt = os.path.join(run_dir, "best_model.pt")

    if args.part_id:
        payload = load_part_from_dataset(cfg, args.part_id, args.split)
        part_id = args.part_id
        x = payload.x.cpu().numpy()
        edge_index = payload.edge_index.cpu().numpy()
        edge_attr = payload.edge_attr.cpu().numpy()
        entity_ids = None
        default_out = f"{part_id}_feature_graph.json"
        resolved_step = (
            Path(args.step_file) if args.step_file
            else default_step_path(part_id, args.step_dir)
        )
    elif args.npz:
        raw = load_from_npz(args.npz)
        part_id = raw["part_id"]
        x, edge_index, edge_attr = raw["x"], raw["edge_index"], raw["edge_attr"]
        entity_ids = raw["entity_ids"]
        default_out = f"{part_id}_feature_graph.json"
        resolved_step = Path(args.step_file) if args.step_file else None
    else:
        raw = load_from_step(args.step, cfg)
        part_id = raw["part_id"]
        x, edge_index, edge_attr = raw["x"], raw["edge_index"], raw["edge_attr"]
        entity_ids = raw["entity_ids"]
        default_out = f"{part_id}_feature_graph.json"
        resolved_step = Path(args.step)

    out_path = args.output or default_out
    print(f"device={device}  ckpt={ckpt}  part={part_id}")

    model, head_w = load_model(ckpt, cfg, device)
    pred, conf = infer_from_tensors(model, x, edge_index, edge_attr, head_w, device)

    graph = build_feature_graph(
        pred, conf, edge_index,
        part_id=part_id,
        entity_ids=entity_ids,
    )

    want_params = args.with_params
    if want_params is None:
        want_params = resolved_step is not None and resolved_step.is_file()

    write_feature_graph(out_path, graph)

    if want_params:
        if resolved_step is None or not resolved_step.is_file():
            print(f"warning: --with-params skipped (STEP not found: {resolved_step})")
        else:
            from pipeline.ingest import require_occ

            require_occ("param enrichment")
            with open(out_path) as f:
                graph = json.load(f)
            enrich_graph_with_params(graph, resolved_step)
            write_feature_graph(out_path, graph)
            print(f"  params from {resolved_step}")

    print(f"wrote {out_path}")
    print(f"  {summarize_graph(graph)}")


if __name__ == "__main__":
    main()
