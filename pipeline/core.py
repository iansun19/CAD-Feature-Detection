"""Core inference + artifact writers for the unified pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from feature_graph import build_feature_graph, summarize_graph, write_feature_graph
from feature_params import strip_heuristic_params
from feature_instances import union_find_instances
from taxonomy import NEW_NAMES


def face_to_feature_map(graph: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for node in graph.get("nodes", []):
        fid = int(node["feature_id"])
        for face_idx in node["face_ids"]:
            mapping[int(face_idx)] = fid
    return mapping


def write_face_predictions(
    path: Path,
    pred: np.ndarray,
    conf: np.ndarray,
    edge_index: np.ndarray,
    graph: dict[str, Any],
    entity_ids: np.ndarray | None = None,
) -> None:
    """One JSONL record per face (predictions + feature instance id)."""
    n = len(pred)
    cluster_of = union_find_instances(n, pred, edge_index, ignore_class=None)
    f2feat = face_to_feature_map(graph)
    # also map stock / unassigned faces -> -1
    full_f2feat = {i: f2feat.get(i, -1) for i in range(n)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            rec: dict[str, Any] = {
                "face_id": i,
                "class_id": int(pred[i]),
                "class_name": NEW_NAMES[int(pred[i])],
                "confidence": round(float(conf[i]), 4),
                "cluster_id": int(cluster_of[i]),
                "feature_id": int(full_f2feat[i]),
            }
            if entity_ids is not None:
                rec["entity_id"] = int(entity_ids[i])
            if "params" in rec:
                rec["params"] = strip_heuristic_params(rec["params"])
            f.write(json.dumps(rec) + "\n")


def build_manifest(
    *,
    part_id: str,
    out_dir: Path,
    step_path: Path | None,
    ckpt: Path,
    graph: dict[str, Any],
    started_at: str,
    finished_at: str,
    params_enabled: bool,
    viewer_enabled: bool,
) -> dict[str, Any]:
    files = {
        "feature_graph": "feature_graph.json",
        "face_predictions": "face_predictions.jsonl",
        "manifest": "manifest.json",
        "log": "pipeline.log",
    }
    if viewer_enabled:
        files["viewer"] = "viewer.html"
    if step_path is not None:
        files["step_source"] = str(step_path)

    return {
        "pipeline_version": 1,
        "part_id": part_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "checkpoint": str(ckpt.resolve()),
        "summary": summarize_graph(graph),
        "schema_version": graph.get("schema_version"),
        "n_faces": graph.get("n_faces"),
        "n_features": graph.get("n_features"),
        "n_edges": graph.get("n_edges"),
        "params_enabled": params_enabled,
        "viewer_enabled": viewer_enabled,
        "outputs": {k: str(out_dir / v) if not v.startswith("/") else v
                    for k, v in files.items()},
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def infer_graph(
    pred: np.ndarray,
    conf: np.ndarray,
    edge_index: np.ndarray,
    *,
    part_id: str,
    entity_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    return build_feature_graph(
        pred, conf, edge_index,
        part_id=part_id,
        entity_ids=entity_ids,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
