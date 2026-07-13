"""
brep_extents.py — B-rep depth, height, envelope, and anchor geometry from STEP.

All measures use OCC kernel geometry on indexed faces (TopologyExplorer order).
Never uses face centroids for extents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.gp import gp_Pnt
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from brep.feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
    from brep.step_ingest import face_mid_normal, sprops

    HAS_OCC = True
except ImportError:
    HAS_OCC = False

AXIS_ALIGN_MIN = 0.9
WALL_NORMAL_DOT_MAX = 0.25
OPENING_NORMAL_DOT_MIN = 0.9
ZERO_EXTENT_EPS = 1e-9
DEGEN_AREA_EPS = 1e-12


class FaceIndexError(Exception):
    """Face index could not be resolved against STEP TopologyExplorer order."""

    def __init__(
        self,
        message: str,
        *,
        face_ids: list[int] | None = None,
        n_faces: int | None = None,
    ):
        super().__init__(message)
        self.face_ids = face_ids
        self.n_faces = n_faces


@dataclass
class Extent:
    min: float
    max: float
    length: float
    axis: list[float]
    method: str
    face_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(float(self.length), 6),
            "min": round(float(self.min), 6),
            "max": round(float(self.max), 6),
            "axis": [round(float(x), 6) for x in self.axis],
            "method": self.method,
            "faces": list(self.face_indices),
        }


@dataclass
class AABB:
    min_corner: list[float]
    max_corner: list[float]
    method: str
    face_indices: list[int]
    zero_width_axes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": [round(float(x), 6) for x in self.min_corner],
            "max": [round(float(x), 6) for x in self.max_corner],
            "method": self.method,
            "faces": list(self.face_indices),
            "zero_width_axes": list(self.zero_width_axes),
        }


@dataclass
class Anchor:
    point: list[float]
    anchor_type: str
    method: str
    face_indices: list[int]
    axis: list[float] | None = None
    normal: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "point": [round(float(x), 6) for x in self.point],
            "type": self.anchor_type,
            "method": self.method,
            "faces": list(self.face_indices),
        }
        if self.axis is not None:
            out["axis"] = [round(float(x), 6) for x in self.axis]
        if self.normal is not None:
            out["normal"] = [round(float(x), 6) for x in self.normal]
        return out


def _r(x: float) -> float:
    return round(float(x), 6)


def _vec3(v: np.ndarray) -> list[float]:
    return [_r(v[0]), _r(v[1]), _r(v[2])]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def _pnt_to_array(p: gp_Pnt) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)


def resolve_occ_faces(
    step_path: str | Path,
    face_ids: list[int],
    *,
    expected_n_faces: int | None = None,
) -> tuple[list[Any], list[tuple[int, Any]]]:
    """Load STEP faces in TopologyExplorer order and resolve indices."""
    require_occ()
    step_path = Path(step_path)
    occ_faces = load_step_faces(step_path)
    n_faces = len(occ_faces)
    if expected_n_faces is not None and n_faces != expected_n_faces:
        raise FaceIndexError(
            f"face count mismatch: STEP has {n_faces}, expected {expected_n_faces}",
            face_ids=face_ids,
            n_faces=n_faces,
        )
    seen: set[int] = set()
    resolved: list[tuple[int, Any]] = []
    for fid in face_ids:
        if fid in seen:
            raise FaceIndexError(
                f"duplicate face index {fid}",
                face_ids=face_ids,
                n_faces=n_faces,
            )
        if fid < 0 or fid >= n_faces:
            raise FaceIndexError(
                f"face index {fid} out of range [0, {n_faces})",
                face_ids=face_ids,
                n_faces=n_faces,
            )
        seen.add(fid)
        resolved.append((fid, occ_faces[fid]))
    return occ_faces, resolved


def collect_boundary_points(
    occ_face,
    *,
    n_edge_samples: int = 5,
) -> np.ndarray:
    """Sample vertices and edge curve points on a face boundary."""
    points: list[np.ndarray] = []
    exp = TopExp_Explorer(occ_face, TopAbs_VERTEX)
    while exp.More():
        points.append(_pnt_to_array(BRep_Tool.Pnt(exp.Current())))
        exp.Next()

    exp = TopExp_Explorer(occ_face, TopAbs_EDGE)
    while exp.More():
        edge = exp.Current()
        res = BRep_Tool.Curve(edge)
        if res is not None and len(res) >= 3:
            curve, a, b = res[0], float(res[1]), float(res[2])
            if abs(b - a) > 1e-12:
                for t in np.linspace(a, b, n_edge_samples):
                    p = gp_Pnt()
                    curve.D0(t, p)
                    points.append(_pnt_to_array(p))
        exp.Next()

    if not points:
        gp = sprops(occ_face).CentreOfMass()
        points.append(np.array([gp.X(), gp.Y(), gp.Z()], dtype=np.float64))
    return np.stack(points, axis=0)


def plane_normal_occ(occ_face) -> np.ndarray:
    """Outward plane normal from OCC surface + face orientation."""
    return _unit(face_mid_normal(occ_face))


def axis_extent(
    occ_faces_by_index: dict[int, Any],
    face_indices: list[int],
    axis: np.ndarray,
    *,
    origin: np.ndarray | None = None,
    method: str = "boundary_point_projection",
) -> Extent:
    """
    Signed extent of face set along a unit axis using boundary geometry samples.
    """
    axis_u = _unit(np.asarray(axis, dtype=np.float64))
    if float(np.linalg.norm(axis_u)) < 1e-12:
        raise ValueError("axis_extent: degenerate axis")

    origin_v = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    projs: list[float] = []
    for fid in face_indices:
        pts = collect_boundary_points(occ_faces_by_index[fid])
        projs.extend((pts - origin_v) @ axis_u)

    if not projs:
        raise ValueError("axis_extent: no boundary points")

    mn, mx = float(min(projs)), float(max(projs))
    return Extent(
        min=mn,
        max=mx,
        length=mx - mn,
        axis=_vec3(axis_u),
        method=method,
        face_indices=list(face_indices),
    )


def feature_aabb(
    occ_faces_by_index: dict[int, Any],
    face_indices: list[int],
    *,
    optimal: bool = True,
) -> AABB:
    """Axis-aligned bounds from BRepBndLib over indexed faces."""
    box = Bnd_Box()
    for fid in face_indices:
        face = occ_faces_by_index[fid]
        if optimal:
            brepbndlib.AddOptimal(face, box, True, False)
        else:
            brepbndlib.Add(face, box)
    box.SetGap(0.0)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    mn = [xmin, ymin, zmin]
    mx = [xmax, ymax, zmax]
    zero_axes: list[str] = []
    labels = ("x", "y", "z")
    total_area = sum(float(sprops(occ_faces_by_index[fid]).Mass()) for fid in face_indices)
    for i, label in enumerate(labels):
        span = mx[i] - mn[i]
        if span <= ZERO_EXTENT_EPS and total_area > DEGEN_AREA_EPS:
            zero_axes.append(label)
    return AABB(
        min_corner=mn,
        max_corner=mx,
        method="BRepBndLib.AddOptimal" if optimal else "BRepBndLib.Add",
        face_indices=list(face_indices),
        zero_width_axes=zero_axes,
    )


def _faces_share_edge(face_a, face_b) -> bool:
    edges_a = list(TopologyExplorer(face_a).edges())
    edges_b = list(TopologyExplorer(face_b).edges())
    for ea in edges_a:
        for eb in edges_b:
            if ea.IsSame(eb):
                return True
    return False


def _group_adjacency(
    occ_faces_by_index: dict[int, Any],
    face_indices: list[int],
) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {fid: set() for fid in face_indices}
    for i, fi in enumerate(face_indices):
        for fj in face_indices[i + 1:]:
            if _faces_share_edge(occ_faces_by_index[fi], occ_faces_by_index[fj]):
                adj[fi].add(fj)
                adj[fj].add(fi)
    return adj


def _dominant_cylinder(
    records: list[FaceGeom],
    face_indices: list[int],
    cam_params: dict[str, Any],
) -> tuple[int, FaceGeom] | None:
    primary = cam_params.get("primary_cylinder_face_index")
    cyl_ids = [i for i in face_indices if records[i].surface_type == "cylinder"]
    if not cyl_ids:
        return None
    if primary is not None and int(primary) in cyl_ids:
        idx = int(primary)
        return idx, records[idx]
    idx = max(cyl_ids, key=lambda i: records[i].area)
    return idx, records[idx]


def _cap_faces(
    records: list[FaceGeom],
    face_indices: list[int],
    axis: np.ndarray,
) -> list[int]:
    caps: list[int] = []
    for fid in face_indices:
        rec = records[fid]
        if rec.surface_type == "plane":
            if abs(float(np.dot(_unit(rec.normal), axis))) >= AXIS_ALIGN_MIN:
                caps.append(fid)
        elif rec.surface_type == "cone" and rec.axis is not None:
            if abs(float(np.dot(_unit(rec.axis), axis))) >= AXIS_ALIGN_MIN:
                caps.append(fid)
    return caps


def _opening_loop_center(
    occ_faces_by_index: dict[int, Any],
    face_indices: list[int],
    axis: np.ndarray,
    *,
    prefer_max: bool = True,
) -> Anchor:
    """Center of the opening edge loop at the max (or min) projection along axis."""
    axis_u = _unit(axis)
    best_proj = -np.inf if prefer_max else np.inf
    loop_pts: list[np.ndarray] = []

    for fid in face_indices:
        pts = collect_boundary_points(occ_faces_by_index[fid])
        projs = pts @ axis_u
        target = float(np.max(projs) if prefer_max else np.min(projs))
        if prefer_max and target >= best_proj - 1e-6:
            if target > best_proj + 1e-6:
                loop_pts = []
            best_proj = max(best_proj, target)
            mask = projs >= best_proj - 1e-4
            loop_pts.extend(pts[mask])
        elif not prefer_max and target <= best_proj + 1e-6:
            if target < best_proj - 1e-6:
                loop_pts = []
            best_proj = min(best_proj, target)
            mask = projs <= best_proj + 1e-4
            loop_pts.extend(pts[mask])

    if not loop_pts:
        raise ValueError("opening_loop_center: no loop points")

    center = np.mean(np.stack(loop_pts, axis=0), axis=0)
    return Anchor(
        point=_vec3(center),
        anchor_type="opening_loop_center",
        method="boundary_loop_centroid_at_axis_extremum",
        face_indices=list(face_indices),
        axis=_vec3(axis_u),
    )


def hole_machined_extents(
    step_path: str | Path,
    face_indices: list[int],
    records: list[FaceGeom],
    cam_params: dict[str, Any],
    *,
    class_id: int,
    expected_n_faces: int | None = None,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Depth and anchor for through/blind holes."""
    errors: list[str] = []
    try:
        _, resolved = resolve_occ_faces(step_path, face_indices, expected_n_faces=expected_n_faces)
    except FaceIndexError as exc:
        return None, [str(exc)], False

    occ_map = {fid: face for fid, face in resolved}
    dom = _dominant_cylinder(records, face_indices, cam_params)
    if dom is None:
        return None, ["hole_depth: no cylinder face"], False

    cyl_idx, cyl_rec = dom
    axis = _unit(cyl_rec.axis) if cyl_rec.axis is not None else plane_normal_occ(occ_map[cyl_idx])
    cyl_ids = [i for i in face_indices if records[i].surface_type == "cylinder"]

    try:
        depth_ext = axis_extent(
            occ_map, cyl_ids, axis,
            method="cylinder_boundary_extent_along_axis",
        )
    except ValueError as exc:
        return None, [f"hole_depth: {exc}"], False

    caps = _cap_faces(records, face_indices, axis)
    if class_id == 5:
        through_or_blind = "blind"
        if not caps:
            errors.append("hole_depth: blind hole missing cap face")
    elif caps:
        through_or_blind = "blind"
    else:
        through_or_blind = "through"

    try:
        anchor = _opening_loop_center(occ_map, cyl_ids, axis, prefer_max=True)
    except ValueError as exc:
        return None, [f"hole_anchor: {exc}"], False

    out: dict[str, Any] = {
        "depth": depth_ext.to_dict(),
        "through_or_blind": through_or_blind,
        "aabb": feature_aabb(occ_map, face_indices).to_dict(),
        "anchor": anchor.to_dict(),
    }
    return out, errors, False


def find_pocket_floor(
    records: list[FaceGeom],
    face_indices: list[int],
    occ_faces_by_index: dict[int, Any],
) -> tuple[int | None, int | None, list[str]]:
    """
    Floor = planar face bounded by wall loop; opening = anti-parallel partner.
    Returns (floor_id, opening_id, errors).
    """
    errors: list[str] = []
    planes = [i for i in face_indices if records[i].surface_type == "plane"]
    cylinders = [i for i in face_indices if records[i].surface_type == "cylinder"]
    if not planes:
        return None, None, ["pocket_floor: no planar faces"]
    if not cylinders and len(planes) < 2:
        return None, None, ["pocket_floor: no wall faces"]

    adj = _group_adjacency(occ_faces_by_index, face_indices)
    wall_ids = set(cylinders)
    for pid in planes:
        n = _unit(records[pid].normal)
        for cid in cylinders:
            caxis = records[cid].axis
            if caxis is not None and abs(float(np.dot(n, _unit(caxis)))) <= WALL_NORMAL_DOT_MAX:
                wall_ids.add(pid)
        for oid in planes:
            if oid == pid:
                continue
            if abs(float(np.dot(n, _unit(records[oid].normal)))) <= WALL_NORMAL_DOT_MAX:
                wall_ids.add(pid)

    if not wall_ids:
        wall_ids = set(cylinders)

    best_floor: int | None = None
    best_opening: int | None = None
    best_score = -1.0

    for pf in planes:
        n_floor = plane_normal_occ(occ_faces_by_index[pf])
        wall_adj = sum(1 for w in wall_ids if w in adj.get(pf, set()))
        if wall_adj == 0:
            continue

        opening_candidates = [
            op for op in planes
            if op != pf and float(np.dot(n_floor, _unit(records[op].normal))) <= -OPENING_NORMAL_DOT_MIN
        ]
        if not opening_candidates:
            continue

        for op in opening_candidates:
            score = wall_adj + sum(1 for w in wall_ids if w in adj.get(op, set()))
            if score > best_score:
                best_score = score
                best_floor = pf
                best_opening = op

    if best_floor is None:
        return None, None, ["pocket_floor: no floor bounded by wall loop"]

    if best_score < len(wall_ids):
        errors.append("pocket_floor: floor does not close full wall loop")

    return best_floor, best_opening, errors


def pocket_machined_extents(
    step_path: str | Path,
    face_indices: list[int],
    records: list[FaceGeom],
    cam_params: dict[str, Any],
    *,
    expected_n_faces: int | None = None,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    try:
        _, resolved = resolve_occ_faces(step_path, face_indices, expected_n_faces=expected_n_faces)
    except FaceIndexError as exc:
        return None, [str(exc)], False

    occ_map = {fid: face for fid, face in resolved}
    floor_id, opening_id, floor_errors = find_pocket_floor(records, face_indices, occ_map)
    if floor_id is None:
        return None, floor_errors, False

    n_floor = plane_normal_occ(occ_map[floor_id])
    wall_ids = [
        i for i in face_indices
        if i not in (floor_id, opening_id)
        and records[i].surface_type in ("cylinder", "plane")
    ]
    if not wall_ids:
        return None, ["pocket_depth: n_walls=0"], False

    floor_origin = collect_boundary_points(occ_map[floor_id]).mean(axis=0)
    try:
        depth_ext = axis_extent(
            occ_map, wall_ids, n_floor,
            origin=floor_origin,
            method="wall_extent_to_opening_loop",
        )
    except ValueError as exc:
        return None, [f"pocket_depth: {exc}"], False

    floor_pt = collect_boundary_points(occ_map[floor_id]).mean(axis=0)
    anchor = Anchor(
        point=_vec3(floor_pt),
        anchor_type="floor_plane_point",
        method="floor_face_boundary_centroid",
        face_indices=[floor_id],
        normal=_vec3(n_floor),
    )

    out: dict[str, Any] = {
        "depth": depth_ext.to_dict(),
        "floor_face_index": floor_id,
        "opening_face_index": opening_id,
        "aabb": feature_aabb(occ_map, face_indices).to_dict(),
        "anchor": anchor.to_dict(),
    }
    return out, floor_errors, False


def step_machined_extents(
    step_path: str | Path,
    face_indices: list[int],
    records: list[FaceGeom],
    cam_params: dict[str, Any],
    *,
    expected_n_faces: int | None = None,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    try:
        _, resolved = resolve_occ_faces(step_path, face_indices, expected_n_faces=expected_n_faces)
    except FaceIndexError as exc:
        return None, [str(exc)], False

    occ_map = {fid: face for fid, face in resolved}
    planes = [i for i in face_indices if records[i].surface_type == "plane"]
    if len(planes) < 2:
        return None, ["step_height: need at least two planar faces"], False

    best_pair: tuple[int, int] | None = None
    best_height = -1.0
    best_axis: np.ndarray | None = None

    for i, pa in enumerate(planes):
        na = plane_normal_occ(occ_map[pa])
        for pb in planes[i + 1:]:
            nb = plane_normal_occ(occ_map[pb])
            dot = abs(float(np.dot(na, nb)))
            if dot >= OPENING_NORMAL_DOT_MIN:
                axis = na if float(np.dot(na, nb)) > 0 else -na
                try:
                    ext = axis_extent(occ_map, [pa, pb], axis, method="step_plane_separation")
                except ValueError:
                    continue
                if ext.length > best_height:
                    best_height = ext.length
                    best_pair = (pa, pb)
                    best_axis = axis

    if best_pair is None or best_axis is None:
        return None, ["step_height: direction ambiguity between step planes"], False

    height_ext = axis_extent(
        occ_map, list(face_indices), best_axis,
        method="step_height_along_plane_normal",
    )
    ref_plane = max(best_pair, key=lambda i: records[i].area)
    anchor = Anchor(
        point=_vec3(collect_boundary_points(occ_map[ref_plane]).mean(axis=0)),
        anchor_type="step_reference_plane_point",
        method="larger_step_plane_boundary_centroid",
        face_indices=[ref_plane],
        normal=_vec3(best_axis),
    )

    return {
        "height": height_ext.to_dict(),
        "aabb": feature_aabb(occ_map, face_indices).to_dict(),
        "anchor": anchor.to_dict(),
    }, [], False


def _cone_chamfer_legs(
    occ_face,
    face_index: int,
    records: list[FaceGeom],
) -> dict[str, Any]:
    rec = records[face_index]
    if rec.semi_angle_rad is None:
        raise ValueError("missing semi_angle")
    angle_deg = float(np.degrees(rec.semi_angle_rad))
    axis = _unit(rec.axis) if rec.axis is not None else np.array([0.0, 0.0, 1.0])

    pts = collect_boundary_points(occ_face)
    ext_along = pts @ axis
    axial_len = float(ext_along.max() - ext_along.min())

    perp = pts - np.outer(pts @ axis, axis)
    radial_ext = float(np.max(np.linalg.norm(perp, axis=1)))

    tan_a = float(np.tan(rec.semi_angle_rad))
    leg_axial = axial_len
    leg_radial = radial_ext
    if tan_a > 1e-12:
        leg_from_angle = axial_len * tan_a
        leg_radial = max(radial_ext, leg_from_angle)

    return {
        "face_index": face_index,
        "semi_angle_deg": _r(angle_deg),
        "leg_axial": _r(leg_axial),
        "leg_radial": _r(leg_radial),
        "axial_extent": _r(axial_len),
        "method": "cone_boundary_extent_legs",
    }


def chamfer_machined_extents(
    step_path: str | Path,
    face_indices: list[int],
    records: list[FaceGeom],
    cam_params: dict[str, Any],
    *,
    expected_n_faces: int | None = None,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    try:
        _, resolved = resolve_occ_faces(step_path, face_indices, expected_n_faces=expected_n_faces)
    except FaceIndexError as exc:
        return None, [str(exc)], False

    occ_map = {fid: face for fid, face in resolved}
    blends = [i for i in face_indices if records[i].surface_type in ("bspline", "bezier")]
    cones = [i for i in face_indices if records[i].surface_type == "cone"]
    if blends and not cones:
        return None, ["chamfer_width: BSpline/bezier requires_blend_handling"], True

    if not cones:
        return None, ["chamfer_width: no cone faces"], False

    legs: list[dict[str, Any]] = []
    for cid in cones:
        try:
            legs.append(_cone_chamfer_legs(occ_map[cid], cid, records))
        except ValueError as exc:
            return None, [f"chamfer_width: {exc}"], False

    angles = sorted({leg["semi_angle_deg"] for leg in legs})
    ref = legs[0]
    anchor = Anchor(
        point=_vec3(collect_boundary_points(occ_map[cones[0]]).mean(axis=0)),
        anchor_type="chamfer_cone_reference_point",
        method="first_cone_boundary_centroid",
        face_indices=[cones[0]],
    )

    return {
        "chamfer_legs": legs,
        "semi_angles_deg": angles,
        "width": {
            "value": ref["leg_radial"],
            "leg_axial": ref["leg_axial"],
            "leg_radial": ref["leg_radial"],
            "method": "cone_boundary_extent_legs",
            "faces": cones,
        },
        "aabb": feature_aabb(occ_map, face_indices).to_dict(),
        "anchor": anchor.to_dict(),
    }, [], False


def fillet_machined_extents(
    step_path: str | Path,
    face_indices: list[int],
    records: list[FaceGeom],
    cam_params: dict[str, Any],
    *,
    expected_n_faces: int | None = None,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    try:
        _, resolved = resolve_occ_faces(step_path, face_indices, expected_n_faces=expected_n_faces)
    except FaceIndexError as exc:
        return None, [str(exc)], False

    occ_map = {fid: face for fid, face in resolved}
    blends = [i for i in face_indices if records[i].surface_type in ("bspline", "bezier")]
    tori = [i for i in face_indices if records[i].surface_type == "torus"]
    if blends and not tori:
        return None, ["fillet: BSpline/bezier requires_blend_handling"], True

    if not tori:
        return None, ["fillet: no torus face"], False

    tid = max(tori, key=lambda i: records[i].area)
    anchor = Anchor(
        point=_vec3(collect_boundary_points(occ_map[tid]).mean(axis=0)),
        anchor_type="fillet_torus_reference_point",
        method="largest_torus_boundary_centroid",
        face_indices=[tid],
    )
    return {
        "aabb": feature_aabb(occ_map, face_indices).to_dict(),
        "anchor": anchor.to_dict(),
    }, [], False


def compute_machined_extents(
    class_id: int,
    face_indices: list[int],
    step_path: str | Path,
    records: list[FaceGeom],
    cam_params: dict[str, Any],
    *,
    expected_n_faces: int | None = None,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Feature-specific machined extents block for CAM export."""
    if class_id in (0, 5):
        return hole_machined_extents(
            step_path, face_indices, records, cam_params,
            class_id=class_id, expected_n_faces=expected_n_faces,
        )
    if class_id in (1, 2, 6, 7):
        return pocket_machined_extents(
            step_path, face_indices, records, cam_params,
            expected_n_faces=expected_n_faces,
        )
    if class_id in (3, 8):
        return step_machined_extents(
            step_path, face_indices, records, cam_params,
            expected_n_faces=expected_n_faces,
        )
    if class_id == 9:
        return chamfer_machined_extents(
            step_path, face_indices, records, cam_params,
            expected_n_faces=expected_n_faces,
        )
    if class_id == 10:
        return fillet_machined_extents(
            step_path, face_indices, records, cam_params,
            expected_n_faces=expected_n_faces,
        )

    try:
        _, resolved = resolve_occ_faces(step_path, face_indices, expected_n_faces=expected_n_faces)
    except FaceIndexError as exc:
        return None, [str(exc)], False
    occ_map = {fid: face for fid, face in resolved}
    return {
        "aabb": feature_aabb(occ_map, face_indices).to_dict(),
    }, [], False
