"""Build a self-contained HTML viewer for feature-graph validation."""
from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from feature_graph_viewer.geometry import triangulate_step_part

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = PKG_DIR / "template.html"

# Distinct palette for feature instances and unassigned faces (no grays).
FEATURE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#46f0f0", "#e6beff", "#ffe119",
]
# Legacy label color for sidebar/docs only; faces never render this flat gray.
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
    """face_index -> feature_id (-1 unassigned), feature_id -> hex color."""
    face_to_feature: dict[int, int] = {i: -1 for i in range(n_faces)}
    feature_colors: dict[int, str] = {}

    for node in graph.get("nodes", []):
        fid = int(node["feature_id"])
        feature_colors[fid] = FEATURE_COLORS[fid % len(FEATURE_COLORS)]
        for face_idx in node["face_ids"]:
            face_to_feature[int(face_idx)] = fid

    return face_to_feature, feature_colors


# Stock faces this close to a same-plane feature face are pick-only in the viewer.
COINCIDENT_FACE_TOL_MM = 0.05
_NORMAL_BUCKET = 0.02


def _plane_key(face: dict) -> tuple[float, float, float, float] | None:
    """Bucket key for coplanar faces (canonical normal + signed plane offset)."""
    import math

    n = face["normal"]
    length = math.sqrt(sum(x * x for x in n))
    if length < 1e-9:
        return None
    n = [x / length for x in n]
    # Opposite normals on the same plane must land in one bucket.
    lead = max(range(3), key=lambda i: abs(n[i]))
    if n[lead] < 0:
        n = [-x for x in n]
    c = face["centroid"]
    d = sum(n[k] * c[k] for k in range(3))
    qd = round(d / COINCIDENT_FACE_TOL_MM) * COINCIDENT_FACE_TOL_MM
    qn = tuple(round(n[k] / _NORMAL_BUCKET) * _NORMAL_BUCKET for k in range(3))
    return (*qn, qd)


def _coincident_plane_buckets(
    faces: list[dict],
) -> dict[tuple[float, float, float, float], list[int]]:
    """Group face indices that lie on the same plane within tolerance."""
    by_index = {int(f["face_index"]): f for f in faces}
    buckets: dict[tuple[float, float, float, float], list[int]] = {}
    for idx in sorted(by_index):
        key = _plane_key(by_index[idx])
        if key is None:
            continue
        buckets.setdefault(key, []).append(idx)
    return buckets


def _stock_like(
    idx: int,
    by_index: dict[int, dict],
    face_to_feature: dict[int, int],
) -> bool:
    return (
        face_to_feature.get(idx, -1) < 0
        or by_index[idx].get("stock_cut_label") == "STOCK"
    )


def coincident_face_partners(
    faces: list[dict],
    face_to_feature: dict[int, int],
) -> dict[int, int]:
    """Map hidden face_index -> visible representative for coincident shells.

    Double-shell molds stack several faces on the same plane. Rendering them
    all z-fights (visible flicker) and makes a single screen spot ambiguous to
    click. We cluster faces that share the same plane (normal + offset within
    tolerance) and keep ONE representative per cluster visible, marking the rest
    pick-only (rendered invisibly). Plane bucketing avoids union-find chains
    that incorrectly collapse offset parallel shells. A real feature face is
    preferred as the representative over bare stock or a face the pre-cascade
    gate still labels STOCK; ties break to the lowest face_index so the choice
    is deterministic across rebuilds.
    """
    by_index = {int(f["face_index"]): f for f in faces}
    partners: dict[int, int] = {}
    for members in _coincident_plane_buckets(faces).values():
        if len(members) < 2:
            continue
        rep = min(members, key=lambda idx: (_stock_like(idx, by_index, face_to_feature), idx))
        for idx in members:
            if idx != rep:
                partners[idx] = rep
    return partners


def assign_coincident_depth_tiers(
    faces: list[dict],
    face_to_feature: dict[int, int],
) -> None:
    """Tag coplanar faces with depth tiers so the viewer can separate them without hiding.

    Tier 0 is the preferred representative (feature over stock, then lowest index).
    Higher tiers nudge along the face normal in the renderer to break z-fighting
    while keeping every face visible.
    """
    by_index = {int(f["face_index"]): f for f in faces}
    for members in _coincident_plane_buckets(faces).values():
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda idx: (_stock_like(idx, by_index, face_to_feature), idx),
        )
        rep = ordered[0]
        for tier, idx in enumerate(ordered):
            face = by_index[idx]
            face["depth_tier"] = tier
            if tier > 0:
                face["coincident_partner"] = rep


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
        face["color"] = (
            feature_colors[feat]
            if feat >= 0
            else FEATURE_COLORS[idx % len(FEATURE_COLORS)]
        )
        if face_preds and idx in face_preds:
            pred = face_preds[idx]
            face["class_id"] = pred.get("class_id")
            face["class_name"] = pred.get("class_name")
            face["confidence"] = pred.get("confidence")
        override = graph.get("face_label_overrides", {}).get(str(idx))
        if override and "labeled_by" in override:
            face["labeled_by"] = override["labeled_by"]

    feature_face_ids = {i for i, fid in face_to_feature.items() if fid >= 0}
    view = analyze_and_orient(faces, feature_meta, feature_face_ids)
    assign_coincident_depth_tiers(faces, face_to_feature)

    return {
        "part_id": part_id,
        "n_faces": n_faces,
        "faces": faces,
        "view": view,
        "features": feature_meta,
        "edges": graph.get("edges", []),
        "edge_color": EDGE_COLOR,
        "stock_color": STOCK_COLOR,
        "graph_source": graph.get("source"),
    }


def _tri_area(a, b, c) -> float:
    import math

    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


# A part is sectioned only when outer stock caps the interior: a flat stock face
# perpendicular to the thin axis that covers most of the footprint (a "lid").
STOCK_LID_COVERAGE = 0.6


def analyze_and_orient(
    faces: list[dict],
    feature_meta: list[dict],
    feature_face_ids: set[int],
) -> dict:
    """Decide whether to half-cut, and reorient geometry so pocket-depth is +Y.

    Reorienting to a Y-up frame keeps OrbitControls stable (no gimbal tumbling)
    and makes the half-cut plane a plain X-normal slice.
    """
    axes = ["X", "Y", "Z"]
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for face in faces:
        for v in face["triangles"]:
            for i in range(3):
                mn[i] = min(mn[i], v[i])
                mx[i] = max(mx[i], v[i])
    if mn[0] == float("inf"):
        return {"section_enabled": False, "section_axis": "X", "section_t": 0.5, "up_axis": "Y"}

    span = [mx[i] - mn[i] for i in range(3)]
    up = min(range(3), key=lambda i: span[i])                    # thinnest = pocket depth
    horizontal = sorted((i for i in range(3) if i != up), key=lambda i: -span[i])
    long_axis, short_axis = horizontal[0], horizontal[1]
    footprint = max(span[long_axis] * span[short_axis], 1e-9)

    # Largest flat face whose normal is (anti)parallel to the thin axis = the
    # candidate "lid". Track its position so we can slice the lid away.
    best_area = 0.0
    best_is_stock = False
    best_up = 0.0
    for face in faces:
        normal = face.get("normal") or [0.0, 0.0, 0.0]
        if abs(normal[up]) < 0.85:
            continue
        tris = face["triangles"]
        area = sum(_tri_area(tris[i], tris[i + 1], tris[i + 2]) for i in range(0, len(tris), 3))
        if area > best_area:
            best_area = area
            best_is_stock = face["face_index"] not in feature_face_ids
            best_up = sum(v[up] for v in tris) / len(tris)

    cut = best_is_stock and (best_area / footprint) > STOCK_LID_COVERAGE
    lid_on_top = (best_up - mn[up]) > 0.5 * span[up]        # lid at +up end?

    # Reorient: long horizontal -> X, thin/depth -> Y, short horizontal -> Z.
    perm = [long_axis, up, short_axis]

    def remap(vec):
        return [vec[perm[0]], vec[perm[1]], vec[perm[2]]]

    for face in faces:
        face["triangles"] = [remap(v) for v in face["triangles"]]
        face["centroid"] = remap(face["centroid"])
        if "normal" in face:
            face["normal"] = remap(face["normal"])
    for meta in feature_meta:
        if meta.get("centroid"):
            meta["centroid"] = remap(meta["centroid"])

    return {
        # Cut along the depth axis (now +Y) so the plane slices through the
        # occluding stock lid; sign keeps the side opposite the lid.
        "section_enabled": cut,
        "section_axis": "Y",
        "section_sign": -1 if lid_on_top else 1,
        "section_t": 0.5,
        "up_axis": "Y",
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
