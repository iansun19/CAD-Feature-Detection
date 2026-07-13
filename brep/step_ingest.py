"""
Runtime STEP → B-rep graph extraction (no labels required).

Shared by diag/regen_dataset.py (labeled offline batch) and scripts/ingest_step.py
(unlabeled runtime ingestion). Returns intermediate B-rep arrays and/or PyG tensors.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepTools import breptools
from OCC.Core.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Extend.TopologyUtils import TopologyExplorer

try:
    from OCC.Core.BRepGProp import brepgprop

    def sprops(face):
        p = GProp_GProps()
        brepgprop.SurfaceProperties(face, p)
        return p
except Exception:
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

    def sprops(face):
        p = GProp_GProps()
        brepgprop_SurfaceProperties(face, p)
        return p

logger = logging.getLogger("step_ingest")

# Surface-type codes stored in V_1 col 4 (tcode / 11). Index 11 -> "other" one-hot.
OCC2CODE = {
    GeomAbs_Plane: 1,
    GeomAbs_Cylinder: 2,
    GeomAbs_Torus: 3,
    GeomAbs_Sphere: 4,
    GeomAbs_Cone: 5,
}
OTHER_SURFACE_TYPES = {
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
}
SURFACE_TYPE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Torus: "torus",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Cone: "cone",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_OffsetSurface: "offset",
    GeomAbs_OtherSurface: "other_surface",
}
DEFAULT_MIN_FACE_AREA = 1e-12


class StepIngestError(Exception):
    """STEP file could not be parsed into a usable B-rep graph."""


@dataclass
class StepIngestStats:
    filename: str
    success: bool = False
    error: str | None = None
    face_count: int = 0
    edge_count: int = 0
    skipped_faces: list[tuple[int, str]] = field(default_factory=list)
    skipped_edges: list[tuple[int, str]] = field(default_factory=list)
    surface_type_counts: dict[str, int] = field(default_factory=dict)
    surface_type_other: int = 0
    undefined_normals: int = 0
    zero_area_faces: int = 0
    boundary_edges: int = 0
    non_manifold_edges: int = 0
    self_loop_edges: int = 0
    convexity_undetermined: int = 0
    no_adjacent_faces: int = 0
    partial_skip_boundary: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["skipped_face_count"] = len(self.skipped_faces)
        d["skipped_edge_count"] = len(self.skipped_edges)
        return d

    def csv_fieldnames(self) -> list[str]:
        return [
            "filename", "success", "error", "face_count", "edge_count",
            "skipped_face_count", "skipped_edge_count",
            "surface_type_other", "undefined_normals", "zero_area_faces",
            "boundary_edges", "non_manifold_edges", "self_loop_edges",
            "convexity_undetermined", "no_adjacent_faces", "partial_skip_boundary",
            "surface_type_counts",
        ]

    def csv_row(self) -> dict[str, Any]:
        d = self.to_dict()
        d["surface_type_counts"] = ";".join(
            f"{k}:{v}" for k, v in sorted(self.surface_type_counts.items())
        )
        return {k: d[k] for k in self.csv_fieldnames()}


def _surface_type_code(face) -> tuple[int, str]:
    st = BRepAdaptor_Surface(face, True).GetType()
    name = SURFACE_TYPE_NAMES.get(st, f"geomabs_{int(st)}")
    if st in OCC2CODE:
        return OCC2CODE[st], name
    if st in OTHER_SURFACE_TYPES:
        return 11, name
    # Any future/unlisted GeomAbs value -> explicit "other" bucket (code 11).
    return 11, name


def face_mid_curvature(face) -> tuple[float, float]:
    """(mean, gaussian) curvature at the face UV midpoint, in 1/length units.

    Rotation- and orientation-invariant scalars from the CAD kernel: planes give
    (0, 0), cylinders (1/2r, 0), spheres (1/r, 1/r^2). Returns (0, 0) when the
    surface has no defined curvature (degenerate/undefined patch)."""
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    surf = BRepAdaptor_Surface(face, True)
    p = BRepLProp_SLProps(surf, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 2, 1e-6)
    if not p.IsCurvatureDefined():
        return 0.0, 0.0
    h = float(p.MeanCurvature())
    k = float(p.GaussianCurvature())
    # OCC can emit inf/nan on near-degenerate patches; clamp to a safe zero.
    if not (np.isfinite(h) and np.isfinite(k)):
        return 0.0, 0.0
    return h, k


def face_mid_normal(face) -> np.ndarray:
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    surf = BRepAdaptor_Surface(face, True)
    p = BRepLProp_SLProps(surf, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6)
    if not p.IsNormalDefined():
        return np.array([0.0, 0.0, 0.0])
    d = p.Normal()
    n = np.array([d.X(), d.Y(), d.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        n = -n
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-9 else n


def edge_midpnt_tangent(edge):
    res = BRep_Tool.Curve(edge)
    if res is None or len(res) < 3:
        return None, None
    curve, a, b = res[0], res[1], res[2]
    p = gp_Pnt(0, 0, 0)
    v = gp_Vec(0, 0, 0)
    curve.D1(0.5 * (a + b), p, v)
    return np.array(p.Coord()), np.array(v.Coord())


def normal_on_face_at_point(xyz, face):
    surface = BRep_Tool.Surface(face)
    sas = ShapeAnalysis_Surface(surface)
    uv = sas.ValueOfUV(gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2])), 0.01)
    props = GeomLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    d = props.Normal()
    n = np.array([d.X(), d.Y(), d.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        n = -n
    return n


def load_step_shape(path: str):
    """Load a STEP file; raise StepIngestError on parse/transfer failure."""
    if not os.path.isfile(path):
        raise StepIngestError(f"file not found: {path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise StepIngestError(
            f"STEP ReadFile failed for {os.path.basename(path)} (status={int(status)})"
        )
    n_roots = reader.TransferRoots()
    if n_roots == 0:
        raise StepIngestError(
            f"STEP TransferRoots transferred 0 roots for {os.path.basename(path)}"
        )
    shape = reader.OneShape()
    if shape.IsNull():
        raise StepIngestError(
            f"STEP produced null shape for {os.path.basename(path)}"
        )
    treader = reader.WS().TransferReader()
    # Pin reader: TransferReader loses shape→entity mapping if reader is GC'd.
    treader._step_reader_ref = reader
    return shape, treader


def _read_face_label(treader, face) -> int | None:
    item = treader.EntityFromShapeResult(face, 1)
    name = ""
    if item is not None:
        item = StepRepr_RepresentationItem.DownCast(item)
        if item is not None:
            name = item.Name().ToCString()
    if name == "" or not name.lstrip("-").isdigit():
        return None
    return int(name)


def extract_brep_from_step(
    path: str,
    *,
    require_labels: bool = False,
    require_12class: bool = False,
    stats: StepIngestStats | None = None,
    min_face_area: float = DEFAULT_MIN_FACE_AREA,
) -> tuple[dict, StepIngestStats]:
    """Extract B-rep arrays from one STEP file.

    Returns a model dict (N, area, cent, tcode, normals, plane_d, A, Aval, E1, E2, E3)
    and per-file stats. Raises StepIngestError on unreadable/empty input.
    """
    stats = stats or StepIngestStats(filename=os.path.basename(path))
    shape, treader = load_step_shape(path)
    topo = TopologyExplorer(shape)
    raw_faces = list(topo.faces())
    if not raw_faces:
        raise StepIngestError(
            f"STEP shape has zero faces: {os.path.basename(path)}"
        )

    kept_faces = []
    kept_labels = []
    edge_idx = 0
    for raw_i, face in enumerate(raw_faces):
        gp = sprops(face)
        area = float(gp.Mass())
        if area < min_face_area:
            stats.zero_area_faces += 1
            stats.skipped_faces.append((raw_i, "zero_area"))
            logger.warning(
                "skipped face %d in %s: zero_area (area=%.3e)",
                raw_i, stats.filename, area,
            )
            continue

        tcode, tname = _surface_type_code(face)
        stats.surface_type_counts[tname] = stats.surface_type_counts.get(tname, 0) + 1
        if tcode == 11:
            stats.surface_type_other += 1

        normal = face_mid_normal(face)
        if np.linalg.norm(normal) < 1e-9:
            stats.undefined_normals += 1
            logger.warning(
                "face %d in %s: undefined_normal (using zero vector)",
                raw_i, stats.filename,
            )

        mean_curv, gauss_curv = face_mid_curvature(face)

        c = gp.CentreOfMass()
        label = _read_face_label(treader, face)
        if require_labels:
            if label is None:
                return None, stats  # type: ignore[return-value]
            if require_12class and label > 11:
                raise StepIngestError(
                    f"label {label} > 11 in {os.path.basename(path)} — "
                    "use step_12class/ (legacy step/ still has 0–24 ids)"
                )
            kept_labels.append(label)
        kept_faces.append({
            "face": face,
            "area": area,
            "cent": np.array([c.X(), c.Y(), c.Z()], dtype=np.float64),
            "tcode": float(tcode),
            "normal": normal.astype(np.float64),
            "curv": np.array([mean_curv, gauss_curv], dtype=np.float64),
        })

    if require_labels and len(kept_labels) != len(kept_faces):
        return None, stats  # type: ignore[return-value]

    if not kept_faces:
        raise StepIngestError(
            f"no valid faces after filtering in {os.path.basename(path)} "
            f"(skipped {stats.zero_area_faces} zero-area)"
        )

    N = len(kept_faces)
    fidx = {rec["face"]: i for i, rec in enumerate(kept_faces)}
    area = np.array([r["area"] for r in kept_faces], dtype=np.float64)
    cent = np.stack([r["cent"] for r in kept_faces])
    tcode = np.array([r["tcode"] for r in kept_faces], dtype=np.float64)
    normals = np.stack([r["normal"] for r in kept_faces])
    curv = np.stack([r["curv"] for r in kept_faces])  # [N, 2] (mean, gaussian)
    labels = np.array(kept_labels, dtype=np.int64) if require_labels else None

    A, Aval, E1, E2, E3 = [], [], [], [], []
    for edge in topo.edges():
        ef = list(topo.faces_from_edge(edge))
        ef_valid = [f for f in ef if f in fidx]

        if len(ef) == 0 or len(ef_valid) == 0:
            stats.no_adjacent_faces += 1
            stats.skipped_edges.append((edge_idx, "no_adjacent_faces"))
            logger.warning(
                "skipped edge %d in %s: no_adjacent_faces", edge_idx, stats.filename,
            )
            edge_idx += 1
            continue

        if len(ef) > 2:
            stats.non_manifold_edges += 1
            stats.skipped_edges.append((edge_idx, f"non_manifold_{len(ef)}_faces"))
            logger.warning(
                "skipped edge %d in %s: non_manifold (%d adjacent faces)",
                edge_idx, stats.filename, len(ef),
            )
            edge_idx += 1
            continue

        if len(ef_valid) == 1:
            stats.partial_skip_boundary += int(len(ef) == 2)
            stats.boundary_edges += 1
            i = fidx[ef_valid[0]]
            E3.append((i, i))
            edge_idx += 1
            continue

        i, j = fidx[ef_valid[0]], fidx[ef_valid[1]]
        if i == j:
            stats.self_loop_edges += 1
            E3.append((i, i))
            edge_idx += 1
            continue

        mid, tan = edge_midpnt_tangent(edge)
        ang = np.pi
        sgn = 0
        if mid is not None:
            n0 = normal_on_face_at_point(mid, ef_valid[0])
            n1 = normal_on_face_at_point(mid, ef_valid[1])
            if n0 is not None and n1 is not None:
                cos = np.clip(
                    n0 @ n1 / (np.linalg.norm(n0) * np.linalg.norm(n1)), -1, 1,
                )
                ang = float(np.arccos(cos))
                r = (
                    np.dot(np.cross(n0, n1), tan)
                    if edge.Orientation() == TopAbs_FORWARD
                    else np.dot(np.cross(n1, n0), tan)
                )
                sgn = int(np.sign(r))
            else:
                stats.convexity_undetermined += 1
                logger.warning(
                    "edge %d in %s: convexity_undetermined (normal lookup failed)",
                    edge_idx, stats.filename,
                )
        else:
            stats.convexity_undetermined += 1
            logger.warning(
                "edge %d in %s: convexity_undetermined (no edge curve)",
                edge_idx, stats.filename,
            )

        A.append((i, j))
        Aval.append(ang)
        A.append((j, i))
        Aval.append(ang)
        bucket = E1 if sgn == 1 else (E2 if sgn == -1 else E3)
        bucket.append((i, j))
        bucket.append((j, i))
        edge_idx += 1

    plane_d = np.sum(normals * cent, axis=1)
    model = dict(
        N=N,
        area=area,
        cent=cent,
        tcode=tcode,
        normals=normals,
        curv=curv,
        plane_d=plane_d,
        A=np.array(A, np.int64).reshape(-1, 2),
        Aval=np.array(Aval, np.float32),
        E1=np.array(E1, np.int64).reshape(-1, 2),
        E2=np.array(E2, np.int64).reshape(-1, 2),
        E3=np.array(E3, np.int64).reshape(-1, 2),
    )
    if labels is not None:
        model["labels"] = labels

    stats.face_count = N
    stats.edge_count = len({tuple(sorted(e)) for e in A}) if A else 0
    stats.success = True
    logger.info("ingest ok", extra={"step_ingest": stats.to_dict()})
    return model, stats


def mm01(a):
    a = np.asarray(a, float)
    lo = a.min(0)
    hi = a.max(0)
    return (a - lo) / (hi - lo + 1e-9)


def build_V1(model: dict, *, include_curvature: bool = False) -> np.ndarray:
    cc = mm01(np.column_stack([model["area"], model["cent"]]))
    # cols 0-8: [area,cx,cy,cz]/[0,1]; type/11; nx;ny;nz; plane_d  (released schema)
    cols = [cc, model["tcode"] / 11.0, model["normals"], model["plane_d"]]
    if include_curvature:
        # cols 9-10: mean & gaussian curvature (raw 1/length; normalized in feature build)
        curv = model.get("curv")
        if curv is None:
            curv = np.zeros((model["cent"].shape[0], 2), dtype=np.float64)
        cols.append(curv)
    return np.column_stack(cols).astype(np.float32)


def _canonical_edge(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))


def model_to_pyg(
    model: dict,
    num_surface_types: int = 6,
    angle_reduce: str = "median",
    include_curvature: bool = False,
):
    """Convert a B-rep model dict to (x, edge_index, edge_attr) PyG tensors as numpy."""
    from brep.brep_features import (
        build_edge_features_regen,
        build_node_features_regen,
        make_undirected,
    )

    v1 = build_V1(model, include_curvature=include_curvature)
    x = build_node_features_regen(v1, num_surface_types)

    e1 = {_canonical_edge(u, v) for u, v in model["E1"] if u != v}
    e2 = {_canonical_edge(u, v) for u, v in model["E2"] if u != v}
    e3 = {_canonical_edge(u, v) for u, v in model["E3"] if u != v}

    ang_by_pair: dict[tuple[int, int], list[float]] = {}
    if model["A"].size:
        for (u, v), ang in zip(model["A"].tolist(), model["Aval"].tolist()):
            if u == v:
                continue
            key = _canonical_edge(u, v)
            ang_by_pair.setdefault(key, []).append(float(ang))

    reduce_fn = np.mean if angle_reduce == "mean" else np.median
    pairs, convexity, cos_ang = [], [], []
    for key, angs in ang_by_pair.items():
        if key in e2:
            cid = 0
        elif key in e1:
            cid = 1
        elif key in e3:
            cid = 2
        else:
            # Should not happen when buckets and A are built together; explicit smooth.
            cid = 2
            logger.warning(
                "edge pair %s missing convexity bucket; encoding as smooth", key,
            )
        pairs.append(key)
        convexity.append(cid)
        cos_ang.append(np.cos(float(reduce_fn(angs))))

    if pairs:
        edge_index = np.asarray(pairs, dtype=np.int64).T
        edge_attr = build_edge_features_regen(
            np.asarray(convexity, dtype=np.int64),
            np.asarray(cos_ang, dtype=np.float32),
        )
        edge_index, edge_attr = make_undirected(edge_index, edge_attr)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 4), dtype=np.float32)

    return x, edge_index, edge_attr


def ingest_step_to_pyg(
    path: str,
    num_surface_types: int = 6,
    angle_reduce: str = "median",
    min_face_area: float = DEFAULT_MIN_FACE_AREA,
    include_curvature: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StepIngestStats]:
    """Single-file runtime entry: STEP path -> (x, edge_index, edge_attr), stats."""
    model, stats = extract_brep_from_step(
        path, require_labels=False, min_face_area=min_face_area,
    )
    x, edge_index, edge_attr = model_to_pyg(
        model,
        num_surface_types=num_surface_types,
        angle_reduce=angle_reduce,
        include_curvature=include_curvature,
    )
    return x, edge_index, edge_attr, stats
