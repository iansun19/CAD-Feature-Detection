"""
feature_graph.py — build and serialize machining-feature graphs.

A feature graph has one node per feature instance (connected same-class face
group, stock excluded by default) and undirected edges between instances whose
faces share a B-rep edge.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from brep.feature_instances import (
    STOCK_CLASS,
    FeatureInstance,
    instances_from_labels,
    union_find_instances,
)
from brep.taxonomy import NEW_NAMES

SCHEMA_VERSION = 1


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def feature_adjacency(
    instances: list[FeatureInstance],
    edge_index: np.ndarray,
    n_faces: int,
) -> list[dict[str, Any]]:
    """Undirected feature–feature edges where any face pair shares a B-rep edge."""
    face_to_feature = np.full(n_faces, -1, dtype=np.int64)
    for out_id, inst in enumerate(instances):
        for f in inst.face_ids:
            face_to_feature[f] = out_id

    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    if edge_index.size == 0:
        return out

    src, dst = edge_index[0], edge_index[1]
    for u, v in zip(src.tolist(), dst.tolist()):
        u, v = int(u), int(v)
        fu, fv = int(face_to_feature[u]), int(face_to_feature[v])
        if fu < 0 or fv < 0 or fu == fv:
            continue
        key = _canonical_edge(fu, fv)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": key[0], "target": key[1], "type": "adjacent"})
    return out


def build_feature_graph(
    pred: np.ndarray,
    conf: np.ndarray,
    edge_index: np.ndarray,
    *,
    part_id: str | None = None,
    entity_ids: np.ndarray | None = None,
    ignore_class: int | None = STOCK_CLASS,
) -> dict[str, Any]:
    """Build a feature graph dict from per-face predictions."""
    pred = np.asarray(pred, dtype=np.int64)
    conf = np.asarray(conf, dtype=np.float64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    n_faces = len(pred)

    instance_of = union_find_instances(
        n_faces, pred, edge_index, ignore_class=ignore_class,
    )
    instances = instances_from_labels(instance_of, pred, conf=conf)

    nodes: list[dict[str, Any]] = []
    for feat_id, inst in enumerate(instances):
        node: dict[str, Any] = {
            "feature_id": feat_id,
            "class_id": inst.class_id,
            "class_name": NEW_NAMES[inst.class_id],
            "face_ids": inst.face_ids,
            "n_faces": inst.n_faces,
            "mean_confidence": round(float(inst.mean_conf), 4)
            if inst.mean_conf is not None else None,
        }
        if entity_ids is not None:
            node["entity_ids"] = [int(entity_ids[f]) for f in inst.face_ids]
        nodes.append(node)

    edges = feature_adjacency(instances, edge_index, n_faces)

    graph: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "part_id": part_id,
        "n_faces": n_faces,
        "n_features": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
    return graph


def write_feature_graph(path: str, graph: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
        f.write("\n")


def summarize_graph(graph: dict[str, Any]) -> str:
    """One-line human summary for logging."""
    by_class: dict[str, int] = {}
    for node in graph["nodes"]:
        name = node["class_name"]
        by_class[name] = by_class.get(name, 0) + 1
    parts = [f"{n}×{c}" for c, n in sorted(by_class.items())]
    return (
        f"{graph['n_features']} features, {graph['n_edges']} adjacency edges "
        f"({', '.join(parts)})"
    )
