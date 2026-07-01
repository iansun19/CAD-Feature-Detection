"""
evaluate.py — per-class diagnostics for a trained B-Rep GNN.

Aggregate accuracy hides everything: an 83% model could be uniformly mediocre or
great-on-common / blind-on-rare. This computes a full confusion matrix plus
per-class precision / recall / F1 / support so you can see WHERE the errors are.

Usage:
    python evaluate.py                      # latest runs/<stamp>/, test split
    python evaluate.py --run runs/2026...   # a specific run dir
    python evaluate.py --split val.txt      # evaluate on val instead of test

Writes into the run dir:
    confusion_matrix.csv     rows = true class, cols = predicted class
    per_class_metrics.csv    precision/recall/f1/support per class
And prints a readable per-class table + overall + macro-F1.
"""

import argparse
import os
import glob

import numpy as np
import yaml
import torch
from torch_geometric.loader import DataLoader

from dataset import get_dataset
from device import resolve_device
from model import BRepGNN
from taxonomy import NEW_NAMES


def load_class_names(data_root, num_classes):
    """Return canonical class names from taxonomy (single source of truth)."""
    return [NEW_NAMES[i] for i in range(num_classes)]


def latest_run(out_dir):
    runs = sorted(glob.glob(os.path.join(out_dir, "*")))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "best_model.pt"))]
    if not runs:
        raise SystemExit(f"no run with best_model.pt under {out_dir}/")
    return runs[-1]


@torch.no_grad()
def collect_confusion(model, loader, device, num_classes):
    """Single pass; accumulate an int64 [C, C] confusion matrix (true x pred)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr).argmax(1)
        t = batch.y.cpu().numpy()
        p = pred.cpu().numpy()
        # vectorized bincount into the flat [C*C] matrix
        idx = t * num_classes + p
        cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes)
    return cm


def per_class_metrics(cm):
    """Return precision, recall, f1, support arrays from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)     # true count per class
    pred_tot = cm.sum(axis=0).astype(np.float64)    # predicted count per class
    precision = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir; default = latest")
    ap.add_argument("--split", default="test.txt", help="split file to evaluate")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg.get("device", "auto"))
    run_dir = args.run or latest_run(cfg["out_dir"])
    ckpt = os.path.join(run_dir, "best_model.pt")
    print(f"device={device}  run={run_dir}  split={args.split}")

    num_classes = cfg["num_classes"]
    names = load_class_names(cfg["data_root"], num_classes)

    ds = get_dataset(cfg, args.split)
    sample = ds[0]
    node_in, edge_in = sample.x.shape[1], sample.edge_attr.shape[1]
    ds._close_h5()
    print(f"node_in={node_in} edge_in={edge_in}  (eval samples={len(ds)})")

    model = BRepGNN(node_in, edge_in, cfg["hidden_dim"], num_classes,
                    cfg["num_layers"], cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    loader = DataLoader(ds, batch_size=cfg["batch_size"],
                        num_workers=cfg["num_workers"])
    cm = collect_confusion(model, loader, device, num_classes)
    precision, recall, f1, support = per_class_metrics(cm)

    total = cm.sum()
    overall_acc = np.diag(cm).sum() / max(total, 1)
    present = support > 0
    macro_f1 = f1[present].mean() if present.any() else 0.0
    # support-weighted accuracy == overall_acc; report balanced (macro) recall too
    macro_recall = recall[present].mean() if present.any() else 0.0

    # ---- printed table ----
    print(f"\n{'id':>3} {'class':<32} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    print("-" * 66)
    order = np.argsort(-support)   # most common first; rare classes at bottom
    for c in order:
        flag = "  <-- weak" if (support[c] > 0 and f1[c] < 0.5) else ""
        print(f"{c:>3} {names[c]:<32} {precision[c]:>6.3f} {recall[c]:>6.3f} "
              f"{f1[c]:>6.3f} {int(support[c]):>8}{flag}")
    print("-" * 66)
    print(f"overall acc = {overall_acc:.4f}   macro-F1 = {macro_f1:.4f}   "
          f"macro-recall = {macro_recall:.4f}   (classes present = {int(present.sum())})")

    # ---- write CSVs into run dir ----
    cm_path = os.path.join(run_dir, "confusion_matrix.csv")
    header = "true\\pred," + ",".join(str(i) for i in range(num_classes))
    with open(cm_path, "w") as f:
        f.write(header + "\n")
        for i in range(num_classes):
            f.write(str(i) + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")

    met_path = os.path.join(run_dir, "per_class_metrics.csv")
    with open(met_path, "w") as f:
        f.write("id,class,precision,recall,f1,support\n")
        for c in range(num_classes):
            f.write(f'{c},"{names[c]}",{precision[c]:.4f},{recall[c]:.4f},'
                    f'{f1[c]:.4f},{int(support[c])}\n')
    print(f"\nwrote {cm_path}\nwrote {met_path}")


if __name__ == "__main__":
    main()
