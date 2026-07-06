"""STEP face triangulation for the feature-graph viewer (requires pythonocc)."""
from __future__ import annotations

from pathlib import Path


def _triangulate_face(face, lin_defl=0.5) -> list[list[float]]:
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


def _face_normal_list(face) -> list[float]:
    from step_ingest import face_mid_normal

    n = face_mid_normal(face)
    return [float(n[0]), float(n[1]), float(n[2])]


def triangulate_step_part(step_path: Path) -> list[dict]:
    """Return one dict per B-rep face in TopologyExplorer order."""
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from step_ingest import load_step_shape

    shape, _ = load_step_shape(str(step_path))
    faces = list(TopologyExplorer(shape).faces())

    out = []
    for idx, face in enumerate(faces):
        tris = _triangulate_face(face)
        centroid = _triangle_centroid(tris)
        out.append({
            "face_index": idx,
            "triangles": tris,
            "centroid": centroid,
            "normal": _face_normal_list(face),
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
