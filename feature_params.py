"""
feature_params.py — B-rep geometry parameters for feature-graph nodes (Phase 3).

Requires pythonocc. Analytic measures (radius, axis, area) come from the CAD
kernel. Derived measures (n_walls, depth) use explicit geometric rules documented
in each helper.
"""
from __future__ import annotations

import json
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
        class_id = int(node["class_id"])
        full_params = extract_feature_params(class_id, face_ids, records)
        faces = [records[i] for i in face_ids if 0 <= i < len(records)]
        index_params, _ = split_params_for_cam(full_params, class_id, faces)
        _apply_face_index_validation(
            node, face_ids, n_step, records=records, step_path=step_path,
        )
        node["params"] = index_params

    graph["schema_version"] = 2
    graph["params_units"] = "mm"
    graph["params_source"] = str(step_path)
    graph["params_notes"] = (
        "Index export: label + face_ids + direct OCC reads "
        "(analytic_surfaces, surface_counts, total_area, cylindrical_radii). "
        "Heuristic aggregates available via collect_derived_debug()."
    )
    return graph


def default_step_path(part_id: str, step_dir: str | Path | None = None) -> Path:
    root = Path(__file__).resolve().parent
    base = Path(step_dir) if step_dir else root / "MFCAD++_dataset" / "step" / "test"
    return base / f"{part_id}.step"


# ---------------------------------------------------------------------------
# CAM export profile (schema v2) — kernel-trusted geometry only
# ---------------------------------------------------------------------------

CAM_SCHEMA_VERSION = 2

# Params kept on the index export (direct STEP/OCC reads + tallies only).
INDEX_EXPORT_PARAM_KEYS: frozenset[str] = frozenset({
    "analytic_surfaces", "surface_counts", "total_area", "cylindrical_radii",
})

# Heuristic / cross-face aggregates stripped from the index export.
REMOVED_HEURISTIC_KEYS: frozenset[str] = frozenset({
    "step_height",
    "floor_face_index", "floor_normal",
    "depth_along_floor_normal", "length_along_axis",
    "bbox_size_x", "bbox_size_y", "bbox_size_z",
    "bbox_depth", "bbox_width", "bbox_length",
    "n_walls", "n_planar_walls", "n_cylindrical_walls", "n_opening_faces",
    "semi_angle_deg", "semi_angles_deg",
})

DERIVED_DEBUG_KEYS: frozenset[str] = REMOVED_HEURISTIC_KEYS | frozenset({
    "n_faces", "n_planar_faces", "n_cylindrical_faces",
    "radius", "diameter", "axis", "primary_cylinder_face_index",
    "fillet_radius", "fillet_radii",
    "groove_major_radius", "groove_minor_radius", "primary_torus_face_index",
    "torus_minor_radius", "torus_minor_radii",
    "machined_extents",
})


def brep_extent_along_direction(
    face_ids: list[int],
    direction: np.ndarray,
    step_path: str | Path,
    *,
    expected_n_faces: int | None = None,
) -> Any:
    """OCC B-rep extent along *direction* for indexed faces (not centroid span)."""
    from brep_extents import axis_extent, resolve_occ_faces

    _, resolved = resolve_occ_faces(
        step_path, face_ids, expected_n_faces=expected_n_faces,
    )
    occ_map = {fid: face for fid, face in resolved}
    return axis_extent(occ_map, face_ids, direction, method="brep_extent_along_direction")


def brep_boundary_extents(
    face_ids: list[int],
    step_path: str | Path,
    *,
    expected_n_faces: int | None = None,
) -> dict[str, Any]:
    """OCC axis-aligned envelope extents for the indexed face group."""
    from brep_extents import feature_aabb, resolve_occ_faces

    _, resolved = resolve_occ_faces(
        step_path, face_ids, expected_n_faces=expected_n_faces,
    )
    occ_map = {fid: face for fid, face in resolved}
    return feature_aabb(occ_map, face_ids).to_dict()


def split_params_for_cam(
    full_params: dict[str, Any],
    class_id: int,
    faces: list[FaceGeom],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split full params into index-export params and derived_debug sidecar."""
    derived_debug: dict[str, Any] = {}
    index_params: dict[str, Any] = {}

    for key, value in full_params.items():
        if key in INDEX_EXPORT_PARAM_KEYS:
            index_params[key] = value
        else:
            derived_debug[key] = value

    if "analytic_surfaces" not in index_params and faces:
        index_params["analytic_surfaces"] = analytic_surfaces(faces)
    if "surface_counts" not in index_params and faces:
        index_params["surface_counts"] = _surface_counts(faces)
    if "total_area" not in index_params and faces:
        index_params["total_area"] = _r(sum(f.area for f in faces))
    if "cylindrical_radii" not in index_params and faces:
        index_params["cylindrical_radii"] = cylindrical_radii(faces)

    return index_params, derived_debug


def strip_heuristic_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove heuristic keys from a flat params dict (top-level only)."""
    return {k: v for k, v in params.items() if k not in REMOVED_HEURISTIC_KEYS}


def node_face_count(node: dict[str, Any]) -> int:
    """Resolvable face count from node fields."""
    if "n_faces" in node:
        return int(node["n_faces"])
    face_ids = node.get("face_ids", [])
    if face_ids:
        return len(face_ids)
    counts = node.get("params", {}).get("surface_counts", {})
    if counts:
        return int(sum(counts.values()))
    return 0


def validate_face_indices(
    face_ids: list[int],
    n_faces: int,
    *,
    records: list[FaceGeom] | None = None,
    step_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Face-index integrity check (the only index-export rejection path).

    Returns {valid, errors, invalid_indices}. Raises FaceIndexError only when
    called with raise_on_error=True (used in unit tests).
    """
    from brep_extents import FaceIndexError

    errors: list[str] = []
    invalid_indices: list[int] = []

    if not face_ids:
        errors.append("face_ids is empty")

    seen: set[int] = set()
    for fid in face_ids:
        if fid in seen:
            errors.append(f"duplicate face index {fid}")
            invalid_indices.append(fid)
        seen.add(fid)
        if fid < 0 or fid >= n_faces:
            errors.append(f"face index {fid} out of range [0, {n_faces})")
            invalid_indices.append(fid)

    if step_path is not None and face_ids and not errors:
        try:
            _, resolved = _resolve_occ_faces(step_path, face_ids, expected_n_faces=n_faces)
            if len(resolved) != len(face_ids):
                errors.append(
                    f"partial_face_resolution: {len(resolved)}/{len(face_ids)} faces resolved",
                )
        except FaceIndexError as exc:
            errors.append(str(exc))
            if exc.face_ids:
                invalid_indices.extend(exc.face_ids)

    if records is not None and face_ids and not errors:
        resolved = [records[i] for i in face_ids if 0 <= i < len(records)]
        if len(resolved) != len(face_ids):
            bad = [i for i in face_ids if i < 0 or i >= len(records)]
            errors.append(f"unresolvable face indices: {bad}")
            invalid_indices.extend(bad)
        elif len(resolved) != len(set(face_ids)):
            errors.append("duplicate face indices in face_ids")

    invalid_indices = sorted(set(invalid_indices))
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "invalid_indices": invalid_indices,
    }


def _resolve_occ_faces(step_path, face_ids, *, expected_n_faces):
    from brep_extents import resolve_occ_faces
    return resolve_occ_faces(step_path, face_ids, expected_n_faces=expected_n_faces)


def _apply_face_index_validation(
    node: dict[str, Any],
    face_ids: list[int],
    n_faces: int,
    *,
    records: list[FaceGeom] | None = None,
    step_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate face indices and mark node invalid on failure."""
    from brep_extents import FaceIndexError

    validation = validate_face_indices(
        face_ids, n_faces, records=records, step_path=step_path,
    )
    node.pop("invalid", None)
    node.pop("face_index_error", None)
    if validation["valid"]:
        return validation

    node["invalid"] = True
    node["face_index_error"] = {
        "type": FaceIndexError.__name__,
        "message": "; ".join(validation["errors"]),
        "indices": validation["invalid_indices"],
    }
    return validation


def collect_derived_debug(
    graph: dict[str, Any],
    records: list[FaceGeom] | None = None,
    *,
    step_path: str | Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-feature derived/heuristic params (not consumed by index export)."""
    if records is None and step_path is not None:
        records = analyze_step(step_path)

    out: dict[int, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        face_ids = [int(i) for i in node.get("face_ids", [])]
        class_id = int(node["class_id"])
        if records is not None:
            full = extract_feature_params(class_id, face_ids, records)
        else:
            full = node.get("params", {})
        faces = [records[i] for i in face_ids if 0 <= i < len(records)] if records else []
        _, derived = split_params_for_cam(full, class_id, faces)
        if derived:
            out[int(node["feature_id"])] = derived
    return out


def _filter_cam_edges(edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    safe: list[dict[str, Any]] = []
    warnings: list[str] = []
    for edge in edges:
        etype = edge.get("type", "adjacent")
        if etype != "adjacent":
            warnings.append(
                f"excluded non-brep edge {edge.get('source')}–{edge.get('target')} "
                f"(type={etype!r})"
            )
            continue
        safe.append({
            "source": int(edge["source"]),
            "target": int(edge["target"]),
            "type": "adjacent",
        })
    return safe, warnings


def print_index_export_summary(graph: dict[str, Any]) -> None:
    """Print per-part index export confirmation (node count, stripped fields, failures)."""
    part_id = graph.get("part_id", "?")
    nodes = graph.get("nodes", [])
    print(f"part {part_id}: {len(nodes)} nodes")

    removed_found: list[str] = []
    for node in nodes:
        params = node.get("params", {})
        for key in REMOVED_HEURISTIC_KEYS:
            if key in params:
                removed_found.append(f"feature_id={node['feature_id']}:{key}")
    if removed_found:
        print(f"  WARNING: removed heuristic fields still present: {removed_found}")
    else:
        print("  removed heuristic fields: absent from all nodes")

    invalid = [
        n for n in nodes
        if n.get("invalid") or n.get("face_index_error")
    ]
    if invalid:
        for node in invalid:
            err = node.get("face_index_error", {})
            print(
                f"  face-index failure feature_id={node['feature_id']}: "
                f"{err.get('message', 'invalid')} indices={err.get('indices', [])}",
            )
    else:
        print("  face-index integrity: all nodes pass")

    if nodes:
        sample = nodes[0]
        retained = sorted(sample.get("params", {}).keys())
        top_level = [
            k for k in ("class_name", "face_ids", "mean_confidence", "n_faces")
            if k in sample
        ]
        print(f"  sample node top-level: {top_level}")
        print(f"  sample node params keys: {retained}")


def export_cam_params(
    graph: dict[str, Any],
    step_path: str | Path,
    *,
    enrich_if_needed: bool = True,
) -> dict[str, Any]:
    """
    Emit index-safe feature params JSON (schema v2).

    Heuristic / derived measures are stripped from node params and are available
    separately via collect_derived_debug(); they are not included in this output.
    """
    require_occ()
    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP not found: {step_path}")

    work = json.loads(json.dumps(graph)) if enrich_if_needed else graph
    needs_enrich = enrich_if_needed or not any(
        n.get("params") for n in work.get("nodes", [])
    )
    if needs_enrich:
        enrich_graph_with_params(work, step_path)

    records = analyze_step(step_path)
    n_faces = len(records)
    graph_n_faces = work.get("n_faces")
    if graph_n_faces is not None and int(graph_n_faces) != n_faces:
        raise ValueError(
            f"Face count mismatch: STEP has {n_faces}, graph has {graph_n_faces}"
        )

    index_nodes: list[dict[str, Any]] = []
    for node in work.get("nodes", []):
        face_ids = [int(i) for i in node.get("face_ids", [])]
        class_id = int(node["class_id"])
        faces = [records[i] for i in face_ids if 0 <= i < len(records)]
        full_params = extract_feature_params(class_id, face_ids, records)
        index_params, _derived = split_params_for_cam(full_params, class_id, faces)
        validation = validate_face_indices(
            face_ids, n_faces, records=records, step_path=step_path,
        )
        out_node: dict[str, Any] = {
            "feature_id": int(node["feature_id"]),
            "class_id": class_id,
            "class_name": node["class_name"],
            "face_ids": face_ids,
            "n_faces": node_face_count(node),
            "mean_confidence": node.get("mean_confidence"),
            "params": index_params,
        }
        if not validation["valid"]:
            out_node["invalid"] = True
            out_node["face_index_error"] = {
                "type": "FaceIndexError",
                "message": "; ".join(validation["errors"]),
                "indices": validation["invalid_indices"],
            }
        index_nodes.append(out_node)

    index_edges, edge_warnings = _filter_cam_edges(work.get("edges", []))

    out: dict[str, Any] = {
        "schema_version": CAM_SCHEMA_VERSION,
        "part_id": work.get("part_id"),
        "n_faces": n_faces,
        "params_units": "mm",
        "params_source": str(step_path),
        "nodes": index_nodes,
        "edges": index_edges,
    }
    if edge_warnings:
        out["edge_exclusion_warnings"] = edge_warnings
    return out
