"""
eval_instances.py — instance-level grouping metrics for a trained B-Rep GNN.

Groups faces into feature instances (connected same-class components), matches
predicted instances to ground-truth instances via face IoU, and reports
detection precision/recall/F1 plus split/merge diagnostics.

Usage:
    python eval_instances.py
    python eval_instances.py --run runs_cloud_latest
    python eval_instances.py --split val.txt --iou 0.5
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import adjusted_rand_score
from torch_geometric.loader import DataLoader

from dataset import get_dataset
from device import resolve_device
from feature_instances import (
    STOCK_CLASS,
    count_split_merge_events,
    instance_prf,
    instances_from_labels,
    match_instances,
    union_find_instances,
)
from model import BRepGNN
from taxonomy import NEW_NAMES


def latest_run(out_dir: str) -> str:
    runs = sorted(glob.glob(os.path.join(out_dir, "*")))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "best_model.pt"))]
    if not runs:
        raise SystemExit(f"no run with best_model.pt under {out_dir}/")
    return runs[-1]


@torch.no_grad()
def evaluate_instances(model, loader, device, num_classes, iou_threshold: float):
    model.eval()

    face_correct = 0
    face_total = 0

    tp = fp = fn = 0
    splits = merges = 0
    gt_inst_total = pred_inst_total = 0

    per_class_gt = np.zeros(num_classes, dtype=np.int64)
    per_class_tp = np.zeros(num_classes, dtype=np.int64)

    ari_scores: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        pred = logits.argmax(1).cpu().numpy()
        conf = F.softmax(logits, dim=1).max(1).values.cpu().numpy()
        y = batch.y.cpu().numpy()
        ei = batch.edge_index.cpu().numpy()

        face_correct += int((pred == y).sum())
        face_total += len(y)

        gt_map = union_find_instances(len(y), y, ei, ignore_class=STOCK_CLASS)
        pred_map = union_find_instances(len(y), pred, ei, ignore_class=STOCK_CLASS)

        gt_list = instances_from_labels(gt_map, y)
        pred_list = instances_from_labels(pred_map, pred, conf=conf)

        gt_inst_total += len(gt_list)
        pred_inst_total += len(pred_list)

        matches, matched_gt, matched_pred = match_instances(
            gt_list, pred_list, iou_threshold=iou_threshold,
        )
        tp += len(matches)
        fp += len(pred_list) - len(matched_pred)
        fn += len(gt_list) - len(matched_gt)

        s, m = count_split_merge_events(gt_list, pred_list, iou_threshold)
        splits += s
        merges += m

        for g in gt_list:
            per_class_gt[g.class_id] += 1
        for match in matches:
            per_class_tp[gt_list[match.gt_idx].class_id] += 1

        # ARI on non-stock faces (cluster ids, not class ids)
        mask = y != STOCK_CLASS
        if mask.sum() >= 2:
            ari_scores.append(float(adjusted_rand_score(
                gt_map[mask], pred_map[mask],
            )))

    face_acc = face_correct / max(face_total, 1)
    prec, rec, f1 = instance_prf(tp, fp, fn)
    mean_ari = float(np.mean(ari_scores)) if ari_scores else 0.0

    per_class_recall = np.divide(
        per_class_tp.astype(np.float64),
        per_class_gt.astype(np.float64),
        out=np.zeros(num_classes, dtype=np.float64),
        where=per_class_gt > 0,
    )

    return {
        "face_acc": face_acc,
        "face_total": face_total,
        "iou_threshold": iou_threshold,
        "instance_tp": tp,
        "instance_fp": fp,
        "instance_fn": fn,
        "instance_precision": prec,
        "instance_recall": rec,
        "instance_f1": f1,
        "gt_instances": gt_inst_total,
        "pred_instances": pred_inst_total,
        "splits": splits,
        "merges": merges,
        "mean_ari": mean_ari,
        "per_class_gt": per_class_gt,
        "per_class_tp": per_class_tp,
        "per_class_recall": per_class_recall,
    }


def main():
    ap = argparse.ArgumentParser(description="Instance-level feature grouping eval")
    ap.add_argument("--run", default=None, help="run dir; default = latest under out_dir")
    ap.add_argument("--split", default="test.txt")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg.get("device", "auto"))
    run_dir = args.run or latest_run(cfg["out_dir"])
    ckpt = os.path.join(run_dir, "best_model.pt")
    print(f"device={device}  run={run_dir}  split={args.split}  iou={args.iou}")

    num_classes = cfg["num_classes"]
    ds = get_dataset(cfg, args.split)
    sample = ds[0]
    node_in, edge_in = sample.x.shape[1], sample.edge_attr.shape[1]
    if hasattr(ds, "_close_h5"):
        ds._close_h5()
    print(f"node_in={node_in} edge_in={edge_in}  samples={len(ds)}")

    model = BRepGNN(node_in, edge_in, cfg["hidden_dim"], num_classes,
                    cfg["num_layers"], cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg.get("num_workers", 0))
    metrics = evaluate_instances(model, loader, device, num_classes, args.iou)

    print(f"\nFace accuracy:           {metrics['face_acc']:.4f}  "
          f"({metrics['face_total']} faces)")
    print(f"Instance P/R/F1 @ IoU {args.iou}:  "
          f"{metrics['instance_precision']:.4f} / "
          f"{metrics['instance_recall']:.4f} / "
          f"{metrics['instance_f1']:.4f}")
    print(f"  TP={metrics['instance_tp']}  FP={metrics['instance_fp']}  "
          f"FN={metrics['instance_fn']}")
    print(f"  GT instances={metrics['gt_instances']}  "
          f"pred instances={metrics['pred_instances']}")
    print(f"Mean ARI (non-stock):    {metrics['mean_ari']:.4f}")
    print(f"Split events:            {metrics['splits']}")
    print(f"Merge events:            {metrics['merges']}")

    print(f"\n{'id':>3} {'class':<24} {'inst_recall':>11} {'gt_inst':>8} {'tp':>6}")
    print("-" * 56)
    order = np.argsort(-metrics["per_class_gt"])
    for c in order:
        if metrics["per_class_gt"][c] == 0:
            continue
        print(f"{c:>3} {NEW_NAMES[c]:<24} "
              f"{metrics['per_class_recall'][c]:>11.4f} "
              f"{int(metrics['per_class_gt'][c]):>8} "
              f"{int(metrics['per_class_tp'][c]):>6}")

    out_path = os.path.join(run_dir, "instance_metrics.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in (
            "face_acc", "iou_threshold", "instance_precision", "instance_recall",
            "instance_f1", "instance_tp", "instance_fp", "instance_fn",
            "gt_instances", "pred_instances", "mean_ari", "splits", "merges",
        ):
            w.writerow([key, metrics[key]])
        w.writerow([])
        w.writerow(["class_id", "class_name", "instance_recall", "gt_instances", "tp"])
        for c in range(num_classes):
            if metrics["per_class_gt"][c] == 0:
                continue
            w.writerow([
                c, NEW_NAMES[c],
                f"{metrics['per_class_recall'][c]:.4f}",
                int(metrics["per_class_gt"][c]),
                int(metrics["per_class_tp"][c]),
            ])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
