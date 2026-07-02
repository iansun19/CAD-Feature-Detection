"""STEP face triangulation for the feature-graph viewer (requires pythonocc)."""
from __future__ import annotations

from pathlib import Path


def triangulate_step_part(step_path: Path) -> list[dict]:
    """Return one dict per B-rep face in TopologyExplorer order."""
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from dataset_explorer.app import _face_mid_normal, _triangulate_face

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    faces = list(TopologyExplorer(shape).faces())

    out = []
    for idx, face in enumerate(faces):
        tris = _triangulate_face(face)
        centroid = _triangle_centroid(tris)
        out.append({
            "face_index": idx,
            "triangles": tris,
            "centroid": centroid,
            "normal": _face_mid_normal(face),
        })
    return out


def _triangle_centroid(triangles: list[list[float]]) -> list[float]:
    if not triangles:
        return [0.0, 0.0, 0.0]
    sx = sy = sz = 0.0
    n = 0
    for v in triangles:
        sx += v[0]
        sy += v[1]
        sz += v[2]
        n += 1
    return [sx / n, sy / n, sz / n]
