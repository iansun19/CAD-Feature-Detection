"""Build a self-contained HTML viewer for feature-graph validation."""
from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from feature_graph_viewer.geometry import triangulate_step_part

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = PKG_DIR / "template.html"

# Distinct palette for feature instances (stock uses STOCK_COLOR).
FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9",
]
STOCK_COLOR = "#d9d9d9"
EDGE_COLOR = "#ff4444"


def load_feature_graph(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_face_predictions(graph_path: Path) -> dict[int, dict] | None:
    """Optional per-face predictions from face_predictions.jsonl alongside the graph."""
    pred_path = graph_path.parent / "face_predictions.jsonl"
    if not pred_path.is_file():
        return None
    out: dict[int, dict] = {}
    for line in pred_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[int(rec["face_id"])] = rec
    return out or None


def face_feature_maps(graph: dict, n_faces: int) -> tuple[dict[int, int], dict[int, str]]:
    """face_index -> feature_id (-1 stock/unassigned), feature_id -> hex color."""
    face_to_feature: dict[int, int] = {i: -1 for i in range(n_faces)}
    feature_colors: dict[int, str] = {}

    for node in graph.get("nodes", []):
        fid = int(node["feature_id"])
        feature_colors[fid] = FEATURE_COLORS[fid % len(FEATURE_COLORS)]
        for face_idx in node["face_ids"]:
            face_to_feature[int(face_idx)] = fid

    return face_to_feature, feature_colors


def build_payload(part_id: str, step_path: Path, graph_path: Path) -> dict:
    graph = load_feature_graph(graph_path)
    face_preds = load_face_predictions(graph_path)
    faces = triangulate_step_part(step_path)
    n_faces = len(faces)

    if graph.get("n_faces") and int(graph["n_faces"]) != n_faces:
        raise ValueError(
            f"Face count mismatch: STEP has {n_faces}, graph says {graph['n_faces']}"
        )

    face_to_feature, feature_colors = face_feature_maps(graph, n_faces)

    feature_meta = []
    for node in graph.get("nodes", []):
        fid = int(node["feature_id"])
        member_centroids = [
            faces[i]["centroid"]
            for i in node["face_ids"]
            if 0 <= int(i) < n_faces
        ]
        if member_centroids:
            cx = sum(c[0] for c in member_centroids) / len(member_centroids)
            cy = sum(c[1] for c in member_centroids) / len(member_centroids)
            cz = sum(c[2] for c in member_centroids) / len(member_centroids)
            centroid = [cx, cy, cz]
        else:
            centroid = [0.0, 0.0, 0.0]
        feature_meta.append({
            "feature_id": fid,
            "class_id": node.get("class_id"),
            "class_name": node.get("class_name"),
            "face_ids": node.get("face_ids", []),
            "n_faces": node.get("n_faces"),
            "mean_confidence": node.get("mean_confidence"),
            "color": feature_colors[fid],
            "centroid": centroid,
            "params": node.get("params"),
        })

    for face in faces:
        idx = face["face_index"]
        feat = face_to_feature[idx]
        face["feature_id"] = feat
        face["color"] = feature_colors[feat] if feat >= 0 else STOCK_COLOR
        if face_preds and idx in face_preds:
            pred = face_preds[idx]
            face["class_id"] = pred.get("class_id")
            face["class_name"] = pred.get("class_name")
            face["confidence"] = pred.get("confidence")

    return {
        "part_id": part_id,
        "n_faces": n_faces,
        "faces": faces,
        "features": feature_meta,
        "edges": graph.get("edges", []),
        "edge_color": EDGE_COLOR,
        "stock_color": STOCK_COLOR,
    }


def render_html(payload: dict, template_path: Path) -> str:
    template = template_path.read_text()
    data_json = json.dumps(payload, separators=(",", ":"))
    return template.replace("__DATA_JSON__", data_json)


def build_viewer(
    part_id: str,
    graph_path: Path,
    step_path: Path | None,
    output_path: Path,
    template_path: Path,
    open_browser: bool = False,
) -> Path:
    if step_path is None:
        step_path = Path("MFCAD++_dataset/step/test") / f"{part_id}.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP not found: {step_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Feature graph not found: {graph_path}")

    payload = build_payload(part_id, step_path, graph_path)
    html = render_html(payload, template_path)
    output_path.write_text(html)

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())

    return output_path
