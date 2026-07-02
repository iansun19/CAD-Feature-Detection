#!/usr/bin/env python3
"""
Stage B (torch env): load the promoted checkpoint from runs_cloud_latest/, run
inference on the Stage-A graph, and emit per-face predictions + honesty checks +
nist_ctc_01_predictions.jsonl.

Handles both head widths: a native 12-class checkpoint is used directly; a legacy
25-class checkpoint is collapsed 25->12 via taxonomy.OLD_TO_NEW (sum softmax mass
per group). The current system model is native 12-class.

Run: .venv/bin/python nist_stage_b_infer.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from feature_graph import build_feature_graph, summarize_graph, write_feature_graph  # noqa: E402
from feature_instances import union_find_instances  # noqa: E402
from model import BRepGNN  # noqa: E402
from taxonomy import NEW_NAMES, NUM_CLASSES, OLD_TO_NEW, validate  # noqa: E402

NPZ = os.path.join(ROOT, "nist_ctc_01_stage_a.npz")
CKPT = os.path.join(ROOT, "runs_cloud_latest", "best_model.pt")  # native 12-class, test_acc 0.9944
OUT_JSONL = os.path.join(ROOT, "nist_ctc_01_predictions.jsonl")
OUT_FEATURE_GRAPH = os.path.join(ROOT, "nist_ctc_01_feature_graph.json")
NUM_OLD = 25
HIDDEN, NUM_LAYERS, DROPOUT = 128, 4, 0.2
SYNTH_TEST_ACC = 0.9945
LOW_CONF = 0.5


def collapse_matrix() -> np.ndarray:
    """[25,12] 0/1 matrix M where M[old,new]=1 iff OLD_TO_NEW[old]==new.
    Collapsed prob over new classes = probs_25 @ M (mass-preserving)."""
    M = np.zeros((NUM_OLD, NUM_CLASSES), dtype=np.float64)
    for old, new in OLD_TO_NEW.items():
        M[old, new] = 1.0
    assert np.allclose(M.sum(1), 1.0), "every old class must map to exactly one new"
    return M


def union_find_clusters(n, pred, edges):
    """Group connected faces sharing the same predicted class. edges: (2,E)."""
    edge_index = np.stack([np.asarray(edges[0]), np.asarray(edges[1])])
    return union_find_instances(n, pred, edge_index, ignore_class=None)


def neighbors(n, edges):
    adj = [[] for _ in range(n)]
    src, dst = edges
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[int(u)].append(int(v))
    return adj


def main():
    validate()
    print(f"[Stage B] loading graph: {NPZ}")
    d = np.load(NPZ)
    x = torch.from_numpy(d["x"]).float()
    edge_index = torch.from_numpy(d["edge_index"]).long()
    edge_attr = torch.from_numpy(d["edge_attr"]).float()
    entity_ids = d["entity_ids"].astype(np.int64)
    N = x.shape[0]
    edges = (d["edge_index"][0], d["edge_index"][1])
    print(f"[Stage B] N={N} node_in={x.shape[1]} edge_in={edge_attr.shape[1]} "
          f"edges={edge_index.shape[1]}")

    # --- load checkpoint; support native 12-class or legacy 25-class heads ---
    sd = torch.load(CKPT, map_location="cpu")
    head_w = sd["head.3.weight"].shape[0]
    node_in = sd["input_proj.weight"].shape[1]
    edge_in = sd["edge_proj.weight"].shape[1]
    print(f"[Stage B] checkpoint head width = {head_w}, "
          f"node_in={node_in} edge_in={edge_in}")
    assert head_w in (NUM_CLASSES, NUM_OLD), (
        f"head width {head_w} is neither 12-class nor 25-class")
    assert node_in == x.shape[1] and edge_in == edge_attr.shape[1], "dim mismatch"

    model = BRepGNN(node_in, edge_in, HIDDEN, head_w, NUM_LAYERS, DROPOUT)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)            # [N,head_w]
        probs = F.softmax(logits, dim=1).cpu().numpy()

    if head_w == NUM_CLASSES:
        print("[Stage B] native 12-class model (direct, no collapse)")
        probs12 = probs                                     # [N,12]
    else:
        print("[Stage B] legacy 25-class model + OLD_TO_NEW collapse")
        probs12 = probs @ collapse_matrix()                 # [N,12], mass-preserving
    assert np.allclose(probs12.sum(1), 1.0, atol=1e-5), "12-class probs not normalized"
    pred = probs12.argmax(1).astype(np.int64)
    conf = probs12[np.arange(N), pred].astype(np.float64)

    # --- clusters + coherence ---
    cluster_of = union_find_clusters(N, pred, edges)
    adj = neighbors(N, edges)

    # ---------- per-face table ----------
    print(f"\n[Stage B] per-face predictions (face_id  entity#N  class  name  conf):")
    records = []
    for i in range(N):
        rec = {
            "face_id": int(i),
            "entity_id": int(entity_ids[i]),
            "class_id": int(pred[i]),
            "class_name": NEW_NAMES[int(pred[i])],
            "confidence": round(float(conf[i]), 4),
            "cluster_id": int(cluster_of[i]),
        }
        records.append(rec)
    for r in records[:20]:
        print(f"   f{r['face_id']:>3} #{r['entity_id']:<4} {r['class_id']:>2} "
              f"{r['class_name']:<20} {r['confidence']:.3f}  clu{r['cluster_id']}")
    print(f"   ... ({N} faces total; full list in {os.path.basename(OUT_JSONL)})")

    # ---------- class histogram ----------
    print(f"\n[Stage B] class histogram (faces per predicted class):")
    counts = np.bincount(pred, minlength=NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if counts[c]:
            cc = conf[pred == c]
            bar = "#" * counts[c]
            print(f"   {c:>2} {NEW_NAMES[c]:<20} {counts[c]:>3}  "
                  f"meanconf={cc.mean():.3f}  {bar}")
    print(f"   [classes with 0 faces: "
          f"{[c for c in range(NUM_CLASSES) if counts[c]==0]}]")

    # ---------- confidence summary ----------
    print(f"\n[Stage B] confidence: mean={conf.mean():.4f}  min={conf.min():.4f}  "
          f"median={np.median(conf):.4f}")
    print(f"   synthetic 25-class test acc was {SYNTH_TEST_ACC}; "
          f"mean-conf drop here = {SYNTH_TEST_ACC - conf.mean():+.4f} "
          f"(expected on harder geometry — reported, not hidden)")

    # ===================== HONESTY CHECKS =====================
    print("\n========== HONESTY CHECKS ==========")

    # (a) collapse check
    top_c = int(counts.argmax())
    top_frac = counts[top_c] / N
    stock_frac = counts[11] / N
    print(f"[collapse] dominant class = {top_c} ({NEW_NAMES[top_c]}) at "
          f"{top_frac:.1%} of faces; stock(11) = {stock_frac:.1%}; "
          f"distinct classes present = {int((counts>0).sum())}/12")
    if top_frac >= 0.95:
        print("  !!! COLLAPSE: >=95% of faces one class — distribution collapsed.")
    elif int((counts > 0).sum()) <= 2:
        print("  !! NEAR-COLLAPSE: only 1-2 classes predicted across the part.")
    else:
        print("  OK: multi-class distribution, not collapsed.")
    if stock_frac == 0:
        print("  note: ZERO faces predicted stock — unusual for a raw-stock part.")

    # (b) confidence check
    low_idx = np.where(conf < LOW_CONF)[0]
    print(f"[confidence] faces below {LOW_CONF} conf: {len(low_idx)}/{N} "
          f"({len(low_idx)/N:.1%})")
    for i in low_idx[:15]:
        print(f"    low: f{i} #{entity_ids[i]} -> {pred[i]} "
              f"{NEW_NAMES[int(pred[i])]} conf={conf[i]:.3f}")
    if len(low_idx) > 15:
        print(f"    ... +{len(low_idx)-15} more (all in jsonl)")

    # (c) cluster-coherence check
    n_clusters = int(cluster_of.max()) + 1
    sizes = np.bincount(cluster_of)
    print(f"[coherence] {n_clusters} connected same-class clusters over {N} faces")
    # clusters per class
    print("  clusters per class (n_clusters | sizes):")
    for c in range(NUM_CLASSES):
        cl = sorted({cluster_of[i] for i in range(N) if pred[i] == c})
        if cl:
            szs = sorted((int(sizes[k]) for k in cl), reverse=True)
            print(f"    {c:>2} {NEW_NAMES[c]:<20} {len(cl):>2} clusters  sizes={szs}")
    # singletons + incoherent lone faces
    singleton_clusters = [k for k in range(n_clusters) if sizes[k] == 1]
    incoherent = []
    for i in range(N):
        if sizes[cluster_of[i]] == 1:  # lone face of its class
            nb = adj[i]
            if nb and all(pred[j] != pred[i] for j in nb):
                incoherent.append(i)
    print(f"  singleton clusters (lone face of its class): {len(singleton_clusters)}")
    print(f"  incoherent lone faces (all neighbors a different class): "
          f"{len(incoherent)}")
    for i in incoherent[:15]:
        nbset = sorted({int(pred[j]) for j in adj[i]})
        print(f"    incoherent: f{i} #{entity_ids[i]} = {pred[i]}"
              f"({NEW_NAMES[int(pred[i])]}) surrounded by classes {nbset}")
    if len(incoherent) > 15:
        print(f"    ... +{len(incoherent)-15} more")
    frac_inc = len(incoherent) / N
    print(f"  >> {frac_inc:.1%} of faces are incoherent singletons. Higher = shakier "
          f"transfer than the 0.9945 synthetic number implies.")

    # NO accuracy / F1 — there is no ground truth in this file.

    # ---------- write jsonl + feature graph ----------
    with open(OUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[Stage B] wrote {OUT_JSONL} ({len(records)} records)")

    graph = build_feature_graph(
        pred, conf, edge_index.cpu().numpy(),
        part_id="nist_ctc_01",
        entity_ids=entity_ids,
    )
    write_feature_graph(OUT_FEATURE_GRAPH, graph)
    print(f"[Stage B] wrote {OUT_FEATURE_GRAPH}")
    print(f"[Stage B]   {summarize_graph(graph)}")


if __name__ == "__main__":
    main()
