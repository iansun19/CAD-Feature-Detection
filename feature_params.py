"""
feature_params.py — B-rep geometry parameters for feature-graph nodes (Phase 3).

Requires pythonocc. Analytic measures (radius, axis, area) come from the CAD
kernel. Derived measures (n_walls, depth) use explicit geometric rules documented
in each helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Torus
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from step_ingest import _surface_type_code, face_mid_normal, load_step_shape, sprops

    HAS_OCC = True
except ImportError:
    HAS_OCC = False

# |n_floor · n_face| below this ⇒ treat planar face as a wall (not floor/opening).
WALL_NORMAL_DOT_MAX = 0.25
# |n_floor · n_face| above this ⇒ parallel to floor (opening/ceiling candidate).
OPENING_NORMAL_DOT_MIN = 0.9


def _r(x: float) -> float:
    return round(float(x), 6)


def _vec3(v) -> list[float]:
    return [_r(v[0]), _r(v[1]), _r(v[2])]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


@dataclass
class FaceGeom:
    index: int
    surface_type: str
    area: float
    centroid: np.ndarray
    normal: np.ndarray
    radius: float | None = None
    axis: np.ndarray | None = None
    semi_angle_rad: float | None = None
    torus_major_r: float | None = None
    torus_minor_r: float | None = None


def require_occ() -> None:
    if not HAS_OCC:
        raise ImportError(
            "feature_params requires pythonocc-core. "
            "Install the unified env: conda env create -f environment.yml && conda activate mlcad"
        )


def load_step_faces(step_path: str | Path) -> list[Any]:
    """OCC faces in TopologyExplorer order (matches STEP / H5 face indices)."""
    require_occ()
    shape, _ = load_step_shape(str(step_path))
    return list(TopologyExplorer(shape).faces())


def analyze_face(face, index: int) -> FaceGeom:
    require_occ()
    gp = sprops(face)
    c = gp.CentreOfMass()
    centroid = np.array([c.X(), c.Y(), c.Z()], dtype=np.float64)
    normal = face_mid_normal(face)
    _tcode, tname = _surface_type_code(face)

    radius = None
    axis = None
    semi_angle_rad = None
    torus_major_r = None
    torus_minor_r = None

    surf = BRepAdaptor_Surface(face, True)
    st = surf.GetType()
    if st == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        radius = float(cyl.Radius())
        d = cyl.Axis().Direction()
        axis = np.array([d.X(), d.Y(), d.Z()], dtype=np.float64)
    elif st == GeomAbs_Cone:
        cone = surf.Cone()
        semi_angle_rad = float(cone.SemiAngle())
        d = cone.Axis().Direction()
        axis = np.array([d.X(), d.Y(), d.Z()], dtype=np.float64)
    elif st == GeomAbs_Torus:
        tor = surf.Torus()
        torus_major_r = float(tor.MajorRadius())
        torus_minor_r = float(tor.MinorRadius())
        d = tor.Axis().Direction()
        axis = np.array([d.X(), d.Y(), d.Z()], dtype=np.float64)
    elif st == GeomAbs_Plane:
        pass

    return FaceGeom(
        index=index,
        surface_type=tname,
        area=float(gp.Mass()),
        centroid=centroid,
        normal=normal,
        radius=radius,
        axis=axis,
        semi_angle_rad=semi_angle_rad,
        torus_major_r=torus_major_r,
        torus_minor_r=torus_minor_r,
    )


def analyze_step(step_path: str | Path) -> list[FaceGeom]:
    faces = load_step_faces(step_path)
    return [analyze_face(f, i) for i, f in enumerate(faces)]


def bbox_params(centroids: list[np.ndarray]) -> dict[str, float]:
    c = np.stack(centroids, axis=0)
    mn = c.min(axis=0)
    mx = c.max(axis=0)
    size = mx - mn
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    dims = sorted([sx, sy, sz])
    return {
        "bbox_size_x": _r(sx),
        "bbox_size_y": _r(sy),
        "bbox_size_z": _r(sz),
        "bbox_depth": _r(dims[0]),
        "bbox_width": _r(dims[1]),
        "bbox_length": _r(dims[2]),
    }


def span_along_direction(centroids: list[np.ndarray], direction: np.ndarray) -> float:
    """Projection span of centroids onto a unit direction (mm)."""
    d = _unit(direction)
    if float(np.linalg.norm(d)) < 1e-12:
        return 0.0
    projs = [float(c @ d) for c in centroids]
    return max(projs) - min(projs)


def pick_floor_plane(planes: list[FaceGeom]) -> FaceGeom | None:
    """Floor candidate = largest-area planar face in the feature."""
    if not planes:
        return None
    return max(planes, key=lambda p: p.area)


def count_planar_walls(planes: list[FaceGeom]) -> int:
    """Legacy wall count: all planar faces minus the largest-area floor."""
    if not planes:
        return 0
    if len(planes) == 1:
        return 1
    return len(planes) - 1


def classify_planar_roles(
    planes: list[FaceGeom],
    floor: FaceGeom | None,
) -> tuple[list[FaceGeom], list[FaceGeom], list[FaceGeom]]:
    """Split planar faces into walls, openings (parallel to floor), and floor."""
    if not planes:
        return [], [], []
    if floor is None:
        floor = pick_floor_plane(planes)
    fn = _unit(floor.normal)
    walls, openings, floor_list = [], [], [floor]
    for p in planes:
        if p.index == floor.index:
            continue
        dot = abs(float(np.dot(_unit(p.normal), fn)))
        if dot >= OPENING_NORMAL_DOT_MIN:
            openings.append(p)
        elif dot <= WALL_NORMAL_DOT_MAX:
            walls.append(p)
        else:
            # Sloped planar face (e.g. chamfer modeled as plane): count as wall.
            walls.append(p)
    return walls, openings, floor_list


def count_walls(
    planes: list[FaceGeom],
    cylinders: list[FaceGeom],
    floor: FaceGeom | None = None,
) -> dict[str, int]:
    """Wall count = perpendicular planar walls + cylindrical faces."""
    planar_walls, openings, floor_faces = classify_planar_roles(planes, floor)
    floor = floor_faces[0] if floor_faces else floor
    return {
        "n_walls": len(planar_walls) + len(cylinders),
        "n_planar_walls": len(planar_walls),
        "n_cylindrical_walls": len(cylinders),
        "n_opening_faces": len(openings),
        "floor_face_index": floor.index if floor else None,
    }


def analytic_surfaces(faces: list[FaceGeom]) -> list[dict[str, Any]]:
    """Per-face OCC analytic parameters (exact kernel values, not fitted)."""
    out: list[dict[str, Any]] = []
    for f in faces:
        entry: dict[str, Any] = {
            "face_index": f.index,
            "surface_type": f.surface_type,
            "area": _r(f.area),
        }
        if f.radius is not None:
            entry["radius"] = _r(f.radius)
            entry["diameter"] = _r(2 * f.radius)
        if f.axis is not None and float(np.linalg.norm(f.axis)) > 1e-12:
            entry["axis"] = _vec3(f.axis)
        if f.semi_angle_rad is not None:
            entry["semi_angle_deg"] = _r(float(np.degrees(f.semi_angle_rad)))
        if f.torus_major_r is not None:
            entry["torus_major_radius"] = _r(f.torus_major_r)
        if f.torus_minor_r is not None:
            entry["torus_minor_radius"] = _r(f.torus_minor_r)
        if f.surface_type == "plane":
            entry["normal"] = _vec3(f.normal)
        out.append(entry)
    return out


def cylindrical_radii(faces: list[FaceGeom]) -> list[dict[str, Any]]:
    """All cylindrical faces with kernel radius (any feature type)."""
    rows = []
    for f in faces:
        if f.surface_type == "cylinder" and f.radius is not None:
            row: dict[str, Any] = {
                "face_index": f.index,
                "radius": _r(f.radius),
                "diameter": _r(2 * f.radius),
                "area": _r(f.area),
            }
            if f.axis is not None:
                row["axis"] = _vec3(f.axis)
            rows.append(row)
    return rows


def _surface_counts(faces: list[FaceGeom]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in faces:
        counts[f.surface_type] = counts.get(f.surface_type, 0) + 1
    return counts


def _common_params(faces: list[FaceGeom]) -> dict[str, Any]:
    return {
        "surface_counts": _surface_counts(faces),
        "total_area": _r(sum(f.area for f in faces)),
        "analytic_surfaces": analytic_surfaces(faces),
        "cylindrical_radii": cylindrical_radii(faces),
    }


def _dominant_cylinder(faces: list[FaceGeom]) -> FaceGeom | None:
    cyls = [f for f in faces if f.surface_type == "cylinder" and f.radius is not None]
    if not cyls:
        return None
    return max(cyls, key=lambda f: f.area)


def _params_hole(faces: list[FaceGeom], *, blind: bool) -> dict[str, Any]:
    cyl = _dominant_cylinder(faces)
    planes = [f for f in faces if f.surface_type == "plane"]
    floor = pick_floor_plane(planes) if planes else None
    out: dict[str, Any] = _common_params(faces)
    if cyl is not None:
        out["radius"] = _r(cyl.radius)
        out["diameter"] = _r(2 * cyl.radius)
        out["axis"] = _vec3(cyl.axis)
        out["primary_cylinder_face_index"] = cyl.index
    centroids = [f.centroid for f in faces]
    if cyl is not None and cyl.axis is not None:
        out["length_along_axis"] = _r(span_along_direction(centroids, cyl.axis))
    if blind and floor is not None:
        out["floor_face_index"] = floor.index
        out["depth_along_floor_normal"] = _r(
            span_along_direction(centroids, floor.normal),
        )
        out.update(count_walls(planes, [f for f in faces if f.surface_type == "cylinder"], floor))
    return out


def _params_pocket_like(faces: list[FaceGeom]) -> dict[str, Any]:
    planes = [f for f in faces if f.surface_type == "plane"]
    cylinders = [f for f in faces if f.surface_type == "cylinder"]
    floor = pick_floor_plane(planes)
    wall_info = count_walls(planes, cylinders, floor)
    centroids = [f.centroid for f in faces]

    out: dict[str, Any] = {
        **_common_params(faces),
        "n_planar_faces": len(planes),
        "n_cylindrical_faces": len(cylinders),
        **wall_info,
    }
    out.update(bbox_params(centroids))
    if floor is not None:
        out["depth_along_floor_normal"] = _r(
            span_along_direction(centroids, floor.normal),
        )
        out["floor_normal"] = _vec3(floor.normal)
    return out


def _params_step(faces: list[FaceGeom]) -> dict[str, Any]:
    planes = [f for f in faces if f.surface_type == "plane"]
    out: dict[str, Any] = {
        **_common_params(faces),
        "n_planar_faces": len(planes),
    }
    out.update(bbox_params([f.centroid for f in faces]))
    if planes:
        floor = pick_floor_plane(planes)
        out["step_height"] = _r(
            span_along_direction([f.centroid for f in faces], floor.normal),
        )
    return out


def _params_chamfer(faces: list[FaceGeom]) -> dict[str, Any]:
    cones = [f for f in faces if f.surface_type == "cone" and f.semi_angle_rad is not None]
    out: dict[str, Any] = {
        **_common_params(faces),
        "n_faces": len(faces),
    }
    if cones:
        angles = sorted({ _r(float(np.degrees(c.semi_angle_rad))) for c in cones })
        out["semi_angle_deg"] = angles[0]
        if len(angles) > 1:
            out["semi_angles_deg"] = angles
    out.update(bbox_params([f.centroid for f in faces]))
    return out


def _params_fillet(faces: list[FaceGeom]) -> dict[str, Any]:
    radii = cylindrical_radii(faces)
    minor = [
        f.torus_minor_r for f in faces
        if f.torus_minor_r is not None
    ]
    out: dict[str, Any] = _common_params(faces)
    if radii:
        rvals = [row["radius"] for row in radii]
        out["fillet_radius"] = min(rvals)
        out["fillet_radii"] = rvals
    if minor:
        out["torus_minor_radius"] = _r(min(minor))
        out["torus_minor_radii"] = [_r(v) for v in sorted(set(minor))]
    return out


def _params_o_ring(faces: list[FaceGeom]) -> dict[str, Any]:
    tori = [f for f in faces if f.surface_type == "torus"]
    out: dict[str, Any] = _common_params(faces)
    if tori:
        t = max(tori, key=lambda f: f.area)
        out["groove_major_radius"] = _r(t.torus_major_r)
        out["groove_minor_radius"] = _r(t.torus_minor_r)
        out["primary_torus_face_index"] = t.index
        if t.axis is not None:
            out["axis"] = _vec3(t.axis)
    out.update(bbox_params([f.centroid for f in faces]))
    return out


def extract_feature_params(class_id: int, face_ids: list[int],
                           records: list[FaceGeom]) -> dict[str, Any]:
    """Class-aware geometry params for one feature instance."""
    faces = [records[i] for i in face_ids if 0 <= i < len(records)]
    if not faces:
        return {}

    if class_id == 0:
        return _params_hole(faces, blind=False)
    if class_id == 5:
        return _params_hole(faces, blind=True)
    if class_id in (1, 2, 6, 7):
        return _params_pocket_like(faces)
    if class_id in (3, 8):
        return _params_step(faces)
    if class_id == 9:
        return _params_chamfer(faces)
    if class_id == 10:
        return _params_fillet(faces)
    if class_id == 4:
        return _params_o_ring(faces)
    return {
        **_common_params(faces),
        **bbox_params([f.centroid for f in faces]),
    }


def enrich_graph_with_params(graph: dict[str, Any],
                             step_path: str | Path) -> dict[str, Any]:
    """Attach a params dict to each node in an existing feature graph."""
    require_occ()
    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP not found: {step_path}")

    records = analyze_step(step_path)
    n_step = len(records)
    n_graph = int(graph.get("n_faces", n_step))
    if n_step != n_graph:
        raise ValueError(
            f"Face count mismatch: STEP has {n_step}, graph has {n_graph}"
        )

    for node in graph.get("nodes", []):
        face_ids = [int(i) for i in node.get("face_ids", [])]
        node["params"] = extract_feature_params(
            int(node["class_id"]), face_ids, records,
        )

    graph["schema_version"] = 2
    graph["params_units"] = "mm"
    graph["params_source"] = str(step_path)
    graph["params_notes"] = (
        "analytic_surfaces/cylindrical_radii/radius/diameter = exact OCC values; "
        "n_walls/depth_along_floor_normal/step_height = derived rules (see feature_params.py)"
    )
    return graph


def default_step_path(part_id: str, step_dir: str | Path | None = None) -> Path:
    root = Path(__file__).resolve().parent
    base = Path(step_dir) if step_dir else root / "MFCAD++_dataset" / "step" / "test"
    return base / f"{part_id}.step"
