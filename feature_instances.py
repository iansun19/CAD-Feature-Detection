"""
feature_instances.py — group B-rep faces into machining-feature instances.

A feature instance is a connected component of faces that share the same class
label in the face-adjacency graph. Used for instance-level eval and (later)
feature-graph export.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STOCK_CLASS = 11


@dataclass
class FeatureInstance:
    instance_id: int
    class_id: int
    face_ids: list[int]
    n_faces: int
    mean_conf: float | None = None


@dataclass
class InstanceMatch:
    gt_idx: int
    pred_idx: int
    iou: float


def union_find_instances(
    n: int,
    labels: np.ndarray,
    edge_index: np.ndarray,
    *,
    ignore_class: int | None = STOCK_CLASS,
) -> np.ndarray:
    """Return per-face compact instance ids; ignored faces get -1.

    Union adjacent faces when labels[u] == labels[v] and the class is not ignored.
    """
    labels = np.asarray(labels, dtype=np.int64)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if edge_index.size:
        src, dst = edge_index[0], edge_index[1]
        for u, v in zip(src.tolist(), dst.tolist()):
            u, v = int(u), int(v)
            if labels[u] != labels[v]:
                continue
            if ignore_class is not None and labels[u] == ignore_class:
                continue
            union(u, v)

    root_to_id: dict[int, int] = {}
    instance_of = np.full(n, -1, dtype=np.int64)
    next_id = 0
    for i in range(n):
        if ignore_class is not None and labels[i] == ignore_class:
            continue
        r = find(i)
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        instance_of[i] = root_to_id[r]
    return instance_of


def instances_from_labels(
    instance_of_face: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray | None = None,
) -> list[FeatureInstance]:
    """Collapse a per-face instance map into a list of FeatureInstance objects."""
    labels = np.asarray(labels, dtype=np.int64)
    buckets: dict[int, list[int]] = {}
    for face_id, inst_id in enumerate(instance_of_face.tolist()):
        if inst_id < 0:
            continue
        buckets.setdefault(int(inst_id), []).append(face_id)

    out: list[FeatureInstance] = []
    for inst_id in sorted(buckets):
        face_ids = buckets[inst_id]
        cls = int(labels[face_ids[0]])
        mean_conf = None
        if conf is not None:
            mean_conf = float(np.mean(conf[face_ids]))
        out.append(FeatureInstance(
            instance_id=inst_id,
            class_id=cls,
            face_ids=face_ids,
            n_faces=len(face_ids),
            mean_conf=mean_conf,
        ))
    return out


def face_iou(faces_a: set[int] | frozenset[int],
             faces_b: set[int] | frozenset[int]) -> float:
    inter = len(faces_a & faces_b)
    if inter == 0:
        return 0.0
    union = len(faces_a | faces_b)
    return inter / union


def match_instances(
    gt_list: list[FeatureInstance],
    pred_list: list[FeatureInstance],
    iou_threshold: float = 0.5,
    require_class_match: bool = True,
) -> tuple[list[InstanceMatch], set[int], set[int]]:
    """Greedy 1:1 matching by IoU (descending). Returns matches and matched indices."""
    gt_faces = [frozenset(g.face_ids) for g in gt_list]
    pred_faces = [frozenset(p.face_ids) for p in pred_list]

    candidates: list[tuple[float, int, int]] = []
    for gi, gf in enumerate(gt_faces):
        for pi, pf in enumerate(pred_faces):
            if require_class_match and gt_list[gi].class_id != pred_list[pi].class_id:
                continue
            iou = face_iou(gf, pf)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda t: t[0], reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[InstanceMatch] = []
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matches.append(InstanceMatch(gt_idx=gi, pred_idx=pi, iou=iou))
        matched_gt.add(gi)
        matched_pred.add(pi)
    return matches, matched_gt, matched_pred


def count_split_merge_events(
    gt_list: list[FeatureInstance],
    pred_list: list[FeatureInstance],
    iou_threshold: float = 0.5,
) -> tuple[int, int]:
    """Count over-segmentation (splits) and under-segmentation (merges) events."""
    gt_faces = [frozenset(g.face_ids) for g in gt_list]
    pred_faces = [frozenset(p.face_ids) for p in pred_list]

    splits = 0
    for gi, gf in enumerate(gt_faces):
        overlaps = sum(
            1 for pi, pf in enumerate(pred_faces)
            if gt_list[gi].class_id == pred_list[pi].class_id
            and face_iou(gf, pf) >= iou_threshold
        )
        if overlaps >= 2:
            splits += 1

    merges = 0
    for pi, pf in enumerate(pred_faces):
        overlaps = sum(
            1 for gi, gf in enumerate(gt_faces)
            if gt_list[gi].class_id == pred_list[pi].class_id
            and face_iou(gf, pf) >= iou_threshold
        )
        if overlaps >= 2:
            merges += 1
    return splits, merges


def instance_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
