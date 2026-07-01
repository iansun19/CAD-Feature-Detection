#!/usr/bin/env python3
"""
MFCAD++ Dataset Explorer — interactive browser for 25 machining-feature classes.

Setup (pythonocc-core is NOT on PyPI for macOS — use conda-forge):
  conda activate mfcadstep          # existing env with pythonocc (Python 3.11)
  pip install flask                 # if not already installed
  python dataset_explorer/app.py
  open http://localhost:5000

Or create a fresh env:
  conda create -n explorer python=3.11 -y
  conda activate explorer
  conda install -c conda-forge pythonocc-core flask -y
  python dataset_explorer/app.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_ROOT = REPO_ROOT / "MFCAD++_dataset"
H5_PATH = DATA_ROOT / "hierarchical_graphs_regen" / "test_MFCAD++.h5"
STEP_DIR = DATA_ROOT / "step" / "test"
TEST_SPLIT = DATA_ROOT / "test.txt"
CACHE_DIR = APP_DIR / "cache"

NUM_CLASSES = 25

CLASS_NAMES = [
    "Chamfer",
    "Through hole",
    "Triangular passage",
    "Rectangular passage",
    "6-sided passage",
    "Triangular through slot",
    "Rectangular through slot",
    "Circular through slot",
    "Rectangular through step",
    "2-sided through step",
    "Slanted through step",
    "O-ring",
    "Blind hole",
    "Triangular pocket",
    "Rectangular pocket",
    "6-sided pocket",
    "Circular end pocket",
    "Rectangular blind slot",
    "Vertical circular end blind slot",
    "Horizontal circular end blind slot",
    "Triangular blind step",
    "Circular blind step",
    "Rectangular blind step",
    "Round",
    "Stock",
]

CLASS_DESCRIPTIONS = [
    "Chamfer: narrow planar bevel joining two faces at an oblique angle (~45°).",
    "Through hole: a single cylindrical face passing entirely through the part, open at both ends.",
    "Triangular passage: through opening with a 3-walled (triangular) planar cross-section, open both ends.",
    "Rectangular passage: through opening with 4 planar walls (rectangular section), open both ends.",
    "6-sided passage: through opening with 6 planar walls (hexagonal section), open both ends.",
    "Triangular through slot: triangular-profile channel cut fully across the part, open at both ends and top.",
    "Rectangular through slot: rectangular channel (floor + 2 walls) cut fully across, open at both ends and top.",
    "Circular through slot: through channel with cylindrical (rounded) walls, open at both ends and top.",
    "Rectangular through step: L-shaped shoulder running fully across the part, open at both ends.",
    "2-sided through step: step open on two sides of the part.",
    "Slanted through step: through step whose wall/floor is slanted — planar faces meeting at a non-90° oblique dihedral.",
    "O-ring: a circular ring groove (toroidal/cylindrical channel) recessed into a face.",
    "Blind hole: cylindrical hole that does NOT pass through — a cylinder wall plus a flat bottom.",
    "Triangular pocket: closed blind pocket with a triangular floor (3 walls + floor), open only at the top.",
    "Rectangular pocket: closed blind pocket with a rectangular floor (4 walls + floor), open only at the top.",
    "6-sided pocket: closed blind pocket with a hexagonal floor (6 walls + floor), open only at the top.",
    "Circular end pocket: blind pocket with rounded (cylindrical) ends — curved walls + floor.",
    "Rectangular blind slot: slot closed at one end (floor + 3 walls), open at the other end and the top.",
    "Vertical circular end blind slot: blind slot terminated by a vertically-oriented circular (cylindrical) end.",
    "Horizontal circular end blind slot: blind slot terminated by a horizontally-oriented circular (cylindrical) end.",
    "Triangular blind step: step closed at one end with a triangular profile.",
    "Circular blind step: blind step bounded by a circular/cylindrical wall.",
    "Rectangular blind step: step closed at one end with a rectangular profile.",
    "Round: a fillet — a rounded (cylindrical/toroidal) face blending two faces across a smooth/tangent edge.",
    "Stock: original raw-material outer surface — large planar faces forming the part's outer bounding box, typically convex neighbors.",
]

# Saturated palette tuned for white 3D viewer background (WCAG-friendly contrast).
CLASS_COLORS = [
    "#DC2626",  # 0 Chamfer
    "#16A34A",  # 1 Through hole
    "#CA8A04",  # 2 Triangular passage
    "#2563EB",  # 3 Rectangular passage
    "#EA580C",  # 4 6-sided passage
    "#9333EA",  # 5 Triangular through slot
    "#0891B2",  # 6 Rectangular through slot
    "#DB2777",  # 7 Circular through slot
    "#65A30D",  # 8 Rectangular through step
    "#E11D48",  # 9 2-sided through step
    "#0D9488",  # 10 Slanted through step
    "#7C3AED",  # 11 O-ring
    "#92400E",  # 12 Blind hole
    "#D97706",  # 13 Triangular pocket
    "#991B1B",  # 14 Rectangular pocket
    "#059669",  # 15 6-sided pocket
    "#4D7C0F",  # 16 Circular end pocket
    "#C2410C",  # 17 Rectangular blind slot
    "#1E3A8A",  # 18 Vertical circular end blind slot
    "#7E22CE",  # 19 Horizontal circular end blind slot
    "#0369A1",  # 20 Triangular blind step
    "#BE185D",  # 21 Circular blind step
    "#15803D",  # 22 Rectangular blind step
    "#B45309",  # 23 Round
    "#854D0E",  # 24 Stock (bronze — visible on white, not gray)
]


def _brep_bounds(idx_arr, model_idx, v1_len):
    base = int(idx_arr[0, 0])
    start = int(idx_arr[model_idx, 0]) - base
    if model_idx + 1 < len(idx_arr):
        end = int(idx_arr[model_idx + 1, 0]) - base
    else:
        end = v1_len
    return start, end


def _pid(raw):
    return raw.decode() if isinstance(raw, bytes) else str(raw)


class DataStore:
    """In-memory index built once at startup from test.txt + test H5."""

    def __init__(self):
        self.part_labels: dict[str, np.ndarray] = {}
        self.class_to_parts: dict[int, list[tuple[str, int]]] = {
            i: [] for i in range(NUM_CLASSES)
        }
        self.class_face_totals = [0] * NUM_CLASSES
        self._h5_index: dict[str, tuple[str, int]] = {}
        self._build_h5_index()
        self._build_class_index()

    def _build_h5_index(self):
        with h5py.File(H5_PATH, "r") as f:
            for batch_key in f.keys():
                batch = f[batch_key]
                for i, raw in enumerate(batch["CAD_model"][()]):
                    self._h5_index[_pid(raw)] = (batch_key, i)

    def _labels_for_part(self, part_id: str) -> np.ndarray | None:
        if part_id not in self._h5_index:
            return None
        batch_key, model_idx = self._h5_index[part_id]
        with h5py.File(H5_PATH, "r") as f:
            batch = f[batch_key]
            idx_arr = batch["idx"][()]
            v1_len = batch["V_1"].shape[0]
            start, end = _brep_bounds(idx_arr, model_idx, v1_len)
            return np.asarray(batch["labels"][start:end], dtype=np.int64)

    def _build_class_index(self):
        with open(TEST_SPLIT) as f:
            test_ids = [ln.strip() for ln in f if ln.strip()]
        test_set = set(test_ids)

        per_class_parts: dict[int, dict[str, int]] = {i: {} for i in range(NUM_CLASSES)}

        with h5py.File(H5_PATH, "r") as hf:
            for batch_key in hf.keys():
                batch = hf[batch_key]
                idx_arr = batch["idx"][()]
                labels_all = batch["labels"][()]
                v1_len = batch["V_1"].shape[0]
                for i, raw in enumerate(batch["CAD_model"][()]):
                    pid = _pid(raw)
                    if pid not in test_set:
                        continue
                    start, end = _brep_bounds(idx_arr, i, v1_len)
                    labels = np.asarray(labels_all[start:end], dtype=np.int64)
                    self.part_labels[pid] = labels
                    counts = Counter(int(c) for c in labels if 0 <= int(c) < NUM_CLASSES)
                    for cls, cnt in counts.items():
                        self.class_face_totals[cls] += cnt
                        per_class_parts[cls][pid] = cnt

        for cls in range(NUM_CLASSES):
            items = sorted(per_class_parts[cls].items(), key=lambda x: (-x[1], x[0]))
            self.class_to_parts[cls] = items


def _face_mid_normal(face):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_REVERSED

    umin, umax, vmin, vmax = breptools.UVBounds(face)
    surf = BRepAdaptor_Surface(face, True)
    props = BRepLProp_SLProps(surf, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6)
    if not props.IsNormalDefined():
        return [0.0, 0.0, 1.0]
    n = props.Normal()
    nx, ny, nz = n.X(), n.Y(), n.Z()
    if face.Orientation() == TopAbs_REVERSED:
        nx, ny, nz = -nx, -ny, -nz
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm < 1e-9:
        return [0.0, 0.0, 1.0]
    return [nx / norm, ny / norm, nz / norm]


def _triangulate_face(face, lin_defl=0.5):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location

    mesh = BRepMesh_IncrementalMesh(face, lin_defl, False, 0.5, True)
    mesh.Perform()
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(face, loc)
    if tri is None:
        return []
    trsf = loc.Transformation()
    verts = []
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i).Transformed(trsf)
        verts.append([p.X(), p.Y(), p.Z()])
    triangles = []
    for i in range(1, tri.NbTriangles() + 1):
        n1, n2, n3 = tri.Triangle(i).Get()
        if face.Orientation() == TopAbs_REVERSED:
            n1, n2, n3 = n1, n3, n2
        triangles.extend([verts[n1 - 1], verts[n2 - 1], verts[n3 - 1]])
    return triangles


def parse_part_geometry(part_id: str, labels: np.ndarray) -> dict:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.TopologyUtils import TopologyExplorer

    step_path = STEP_DIR / f"{part_id}.step"
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    faces = list(TopologyExplorer(shape).faces())

    if len(faces) != len(labels):
        raise ValueError(
            f"Face count mismatch for {part_id}: STEP={len(faces)} H5={len(labels)}"
        )

    face_payload = []
    for idx, face in enumerate(faces):
        cls = int(labels[idx])
        cls = cls if 0 <= cls < NUM_CLASSES else 24
        tris = _triangulate_face(face)
        face_payload.append({
            "face_index": idx,
            "class_id": cls,
            "class_name": CLASS_NAMES[cls],
            "triangles": tris,
            "normal": _face_mid_normal(face),
        })

    return {
        "part_id": part_id,
        "n_faces": len(faces),
        "faces": face_payload,
    }


def get_geometry(part_id: str, store: DataStore) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{part_id}.json"
    if cache_path.is_file():
        with open(cache_path) as f:
            return json.load(f)

    labels = store.part_labels.get(part_id)
    if labels is None:
        labels = store._labels_for_part(part_id)
    if labels is None:
        raise KeyError(f"Part {part_id} not in test H5")

    geom = parse_part_geometry(part_id, labels)
    with open(cache_path, "w") as f:
        json.dump(geom, f)
    return geom


def print_color_legend():
    print("\n=== MFCAD++ class color legend ===")
    for i in range(NUM_CLASSES):
        bar = "\033[48;2;{};{};{}m  \033[0m".format(
            int(CLASS_COLORS[i][1:3], 16),
            int(CLASS_COLORS[i][3:5], 16),
            int(CLASS_COLORS[i][5:7], 16),
        )
        print(f"  {bar} {i:2d}  {CLASS_NAMES[i]:<32}  {CLASS_COLORS[i]}")
    print()


def create_app(store: DataStore | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(APP_DIR / "templates"),
        static_folder=str(APP_DIR / "static"),
    )
    if store is None:
        store = DataStore()
    app.config["store"] = store

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/classes")
    def api_classes():
        st = app.config["store"]
        out = []
        for i in range(NUM_CLASSES):
            out.append({
                "id": i,
                "name": CLASS_NAMES[i],
                "description": CLASS_DESCRIPTIONS[i],
                "color": CLASS_COLORS[i],
                "part_count": len(st.class_to_parts[i]),
                "face_count_total": st.class_face_totals[i],
            })
        return jsonify(out)

    @app.route("/api/classes/<int:class_id>/parts")
    def api_class_parts(class_id: int):
        st = app.config["store"]
        if class_id < 0 or class_id >= NUM_CLASSES:
            return jsonify({"error": "invalid class_id"}), 400
        limit = request.args.get("limit", 10, type=int)
        offset = request.args.get("offset", 0, type=int)
        items = st.class_to_parts[class_id]
        slice_ = items[offset: offset + limit]
        return jsonify({
            "class_id": class_id,
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "parts": [
                {"part_id": pid, "face_count": cnt}
                for pid, cnt in slice_
            ],
        })

    @app.route("/api/parts/<part_id>/geometry")
    def api_part_geometry(part_id: str):
        st = app.config["store"]
        try:
            return jsonify(get_geometry(part_id, st))
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/parts/<part_id>/info")
    def api_part_info(part_id: str):
        st = app.config["store"]
        labels = st.part_labels.get(part_id)
        if labels is None:
            return jsonify({"error": "part not found"}), 404
        dist = Counter(int(c) for c in labels)
        return jsonify({
            "part_id": part_id,
            "n_faces": int(len(labels)),
            "class_distribution": {str(k): v for k, v in sorted(dist.items())},
        })

    return app


def main():
    if not DATA_ROOT.is_dir():
        print(f"ERROR: dataset not found at {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)
    if not H5_PATH.is_file():
        print(f"ERROR: H5 not found at {H5_PATH}", file=sys.stderr)
        sys.exit(1)

    print_color_legend()
    print("Building class index from test split…")
    store = DataStore()
    with open(TEST_SPLIT) as f:
        n_test = sum(1 for ln in f if ln.strip())
    print(f"  test parts in split: {n_test}")
    print(f"  parts indexed in H5: {len(store.part_labels)}")
    app = create_app(store)
    print(f"\nStarting server at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
