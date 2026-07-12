"""wall_geometry.py - post-pass B-rep sizing params for exterior wall nodes.

Additive enrichment only: attaches optional wall_depth_mm, wall_min_clearance_mm,
and wall_lateral_extent_mm to cascade wall node params without touching cascade
detection or existing fields.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent

WALL_DEPTH_MIN_MM = 0.5
WALL_DEPTH_MAX_MM = 100.0
WALL_CLEARANCE_MIN_MM = 0.1
WALL_CLEARANCE_MAX_MM = 50.0

WALL_PARAM_KEYS = (
    "wall_depth_mm",
    "wall_min_clearance_mm",
    "wall_lateral_extent_mm",
)

# Shop reference diameters for 96260B rear wall-split verification (inches -> mm).
SHOP_SMALL_WALL_TOOL_DIA_MM = 0.1875 * 25.4  # op 17 - 3/16 inch
SHOP_LARGE_WALL_TOOL_DIA_MM = 0.375 * 25.4   # op 18/19 - 3/8 inch


@dataclass
class WallGeometryResult:
    wall_depth_mm: float | None = None
    wall_min_clearance_mm: float | None = None
    wall_lateral_extent_mm: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_params(self) -> dict[str, float | None]:
        return {
            "wall_depth_mm": self.wall_depth_mm,
            "wall_min_clearance_mm": self.wall_min_clearance_mm,
            "wall_lateral_extent_mm": self.wall_lateral_extent_mm,
        }


@dataclass
class _WallGeomContext:
    step_path: Path
    faces: list[Any]
    by_index: dict[int, Any]
    occ_map: dict[int, Any]
    face_graph: Any
    cad_ctx: Any
    opening_axis: np.ndarray
    part_axis_top: float


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _round_or_none(value: float | None, *, places: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), places)


def _blend_radius_mm(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any],
) -> float | None:
    """Sphere/torus blend radius for one face (mirrors pocket_detection helper)."""
    fg = by_index[face_idx]
    if fg.surface_type == "sphere":
        if fg.radius is not None:
            return float(fg.radius)
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Sphere

        surf = BRepAdaptor_Surface(occ_map[face_idx], True)
        if surf.GetType() == GeomAbs_Sphere:
            return float(surf.Sphere().Radius())
    elif fg.surface_type == "torus" and fg.torus_minor_r is not None:
        return float(fg.torus_minor_r)
    return None


def _wall_depth_mm(
    face_ids: Sequence[int],
    opening_axis: np.ndarray,
    part_axis_top: float,
    occ_map: dict[int, Any],
    by_index: dict[int, Any],
) -> tuple[float | None, str | None]:
    from pocket_detection import _depth_below_top_mm

    missing = [i for i in face_ids if i not in occ_map and i not in by_index]
    if missing:
        return None, f"missing face indices: {missing}"

    depth = _depth_below_top_mm(
        face_ids, opening_axis, part_axis_top, occ_map, by_index,
    )
    if depth is None:
        return None, "depth_below_top returned None"
    if depth < WALL_DEPTH_MIN_MM or depth > WALL_DEPTH_MAX_MM:
        return depth, (
            f"wall_depth_mm={depth:.4f} outside sanity band "
            f"[{WALL_DEPTH_MIN_MM}, {WALL_DEPTH_MAX_MM}]"
        )
    return depth, None


def _wall_min_clearance_mm(
    face_id: int,
    ctx: _WallGeomContext,
) -> tuple[float | None, str | None]:
    """Minimum blend radius at concave junctions bordering the wall face."""
    radii: list[float] = []
    neighbors = ctx.face_graph.neighbors.get(face_id, set())
    if not neighbors:
        return None, "wall face has no graph neighbors"

    concave_neighbors = [
        nb for nb in neighbors
        if ctx.face_graph.edge_kind(face_id, nb) == "concave"
    ]
    if not concave_neighbors:
        if ctx.cad_ctx is not None:
            summary = _convex_summary_safe(face_id, ctx)
            if summary and summary.concave_edges == 0:
                return None, "no concave edges on wall face"
        return None, "no concave graph neighbors"

    for nb in concave_neighbors:
        corner_faces = {nb}
        for nb2 in ctx.face_graph.neighbors.get(nb, set()):
            kind = ctx.face_graph.edge_kind(nb, nb2)
            if kind in ("concave", "smooth"):
                corner_faces.add(nb2)
        for cid in corner_faces:
            radius = _blend_radius_mm(cid, ctx.by_index, ctx.occ_map)
            if radius is not None:
                radii.append(radius)

    if not radii:
        return None, "concave junctions have no analytic sphere/torus blends"

    clearance = min(radii)
    if clearance < WALL_CLEARANCE_MIN_MM or clearance > WALL_CLEARANCE_MAX_MM:
        return clearance, (
            f"wall_min_clearance_mm={clearance:.4f} outside sanity band "
            f"[{WALL_CLEARANCE_MIN_MM}, {WALL_CLEARANCE_MAX_MM}]"
        )
    return clearance, None


def _convex_summary_safe(face_id: int, ctx: _WallGeomContext) -> Any | None:
    from stock_cut_classification import _convex_summary

    return _convex_summary(face_id, ctx.cad_ctx)


def _opening_axis_index(opening_axis: np.ndarray) -> int:
    """Dominant axis label index (0=X, 1=Y, 2=Z) for AABB lateral filtering."""
    axis = _unit(opening_axis)
    return int(np.argmax(np.abs(axis)))


def _wall_lateral_extent_mm(
    face_ids: Sequence[int],
    opening_axis: np.ndarray,
    occ_map: dict[int, Any],
) -> tuple[float | None, str | None]:
    from brep_extents import feature_aabb

    try:
        aabb = feature_aabb(occ_map, list(face_ids))
    except (ValueError, KeyError) as exc:
        return None, f"feature_aabb failed: {exc}"

    spans = [
        float(aabb.max_corner[i] - aabb.min_corner[i])
        for i in range(3)
    ]
    axis_i = _opening_axis_index(opening_axis)
    lateral = [span for i, span in enumerate(spans) if i != axis_i and span > 1e-6]
    if not lateral:
        return None, "degenerate lateral AABB spans"
    extent = min(lateral)
    if extent <= 0.0:
        return None, "non-positive lateral extent"
    return extent, None


def _collect_interior_wall_axes(
    graph: dict[str, Any],
    by_index: dict[int, Any],
) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    pocket_like = {
        "filleted_pocket",
        "filleted_open_pocket",
        "pocket",
        "open_pocket",
        "blind_pocket",
        "through_hole",
        "filleted_blind_hole",
    }
    for node in graph.get("nodes", []):
        if node.get("class_name") not in pocket_like:
            continue
        for face_id in node.get("face_ids", []):
            fg = by_index.get(int(face_id))
            if fg is None or fg.surface_type != "cylinder" or fg.axis is None:
                continue
            axes.append(np.asarray(fg.axis, dtype=np.float64))
    return axes


def resolve_opening_axis(
    graph: dict[str, Any],
    by_index: dict[int, Any],
    *,
    opening_axis: Sequence[float] | None = None,
) -> np.ndarray:
    if opening_axis is not None:
        return _unit(np.asarray(opening_axis, dtype=np.float64))

    wall_axes = _collect_interior_wall_axes(graph, by_index)
    if wall_axes:
        from setup_descriptor import auto_detect_opening_axis

        axis, _confidence = auto_detect_opening_axis(wall_axes)
        return _unit(np.asarray(axis, dtype=np.float64))

    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def resolve_graph_npz(
    graph: dict[str, Any],
    step_path: Path,
    *,
    graph_npz: str | Path | None = None,
) -> Path | None:
    if graph_npz is not None:
        path = Path(graph_npz)
        return path if path.is_file() else None

    part_id = str(graph.get("part_id", ""))
    candidates = [
        REPO_ROOT / "pipeline_out" / part_id / "graph.npz",
        REPO_ROOT / "pipeline_out" / "96260B_plate" / "graph.npz",
    ]
    gt_path = REPO_ROOT / "eval" / "gt" / f"{part_id}.yaml"
    if gt_path.is_file():
        try:
            import yaml

            meta = yaml.safe_load(gt_path.read_text())
            if isinstance(meta, dict) and meta.get("graph_npz"):
                candidates.insert(0, REPO_ROOT / str(meta["graph_npz"]))
        except Exception:
            pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_wall_geom_context(
    graph: dict[str, Any],
    step_path: str | Path,
    *,
    graph_npz: str | Path | None = None,
    opening_axis: Sequence[float] | None = None,
) -> _WallGeomContext:
    from feature_params import analyze_step, load_step_faces, require_occ
    from hole_detection import FaceGraph
    from pocket_detection import _part_axis_top
    from stock_cut_classification import _load_cad

    require_occ()
    step_path = Path(step_path)
    faces = analyze_step(step_path)
    n_faces = len(faces)
    if graph.get("n_faces") not in (None, n_faces):
        raise ValueError(
            f"graph n_faces={graph.get('n_faces')} != STEP face count {n_faces}",
        )

    occ_faces = load_step_faces(step_path)
    occ_map = {i: occ_faces[i] for i in range(n_faces)}
    by_index = {f.index: f for f in faces}

    npz_path = resolve_graph_npz(graph, step_path, graph_npz=graph_npz)
    if npz_path is None:
        raise FileNotFoundError(
            "edge graph NPZ not found; pass --graph-npz (needed for concave adjacency)",
        )
    data = np.load(npz_path)
    face_graph = FaceGraph.from_edge_tensors(
        data["edge_index"], data["edge_attr"], n_faces,
    )

    axis = resolve_opening_axis(graph, by_index, opening_axis=opening_axis)
    part_axis_top = _part_axis_top(faces, axis, occ_map)
    cad_ctx = _load_cad(step_path)

    return _WallGeomContext(
        step_path=step_path,
        faces=faces,
        by_index=by_index,
        occ_map=occ_map,
        face_graph=face_graph,
        cad_ctx=cad_ctx,
        opening_axis=axis,
        part_axis_top=part_axis_top,
    )


def compute_wall_geometry(
    face_ids: Sequence[int],
    ctx: _WallGeomContext,
    *,
    feature_id: int | str | None = None,
) -> WallGeometryResult:
    if not face_ids:
        return WallGeometryResult(warnings=["empty face_ids"])

    label = f"wall feature_id={feature_id}" if feature_id is not None else "wall"
    result = WallGeometryResult()
    warnings: list[str] = []

    depth, depth_warn = _wall_depth_mm(
        face_ids, ctx.opening_axis, ctx.part_axis_top, ctx.occ_map, ctx.by_index,
    )
    result.wall_depth_mm = _round_or_none(depth)
    if depth is None:
        msg = f"{label} face_ids={list(face_ids)}: could not compute wall_depth_mm"
        warnings.append(msg)
        logger.warning(msg)
    elif depth_warn:
        warnings.append(f"{label}: {depth_warn}")
        logger.warning("%s: %s", label, depth_warn)

    if len(face_ids) == 1:
        clearance, clearance_warn = _wall_min_clearance_mm(int(face_ids[0]), ctx)
    else:
        clearances = []
        for fid in face_ids:
            value, _warn = _wall_min_clearance_mm(int(fid), ctx)
            if value is not None:
                clearances.append(value)
        clearance = min(clearances) if clearances else None
        clearance_warn = None if clearance is not None else "no analytic blends at concave junctions"

    result.wall_min_clearance_mm = _round_or_none(clearance)
    if clearance is None:
        msg = f"{label} face_ids={list(face_ids)}: could not compute wall_min_clearance_mm"
        warnings.append(msg)
        logger.warning(msg)
    elif clearance_warn:
        warnings.append(f"{label}: {clearance_warn}")
        logger.warning("%s: %s", label, clearance_warn)

    lateral, lateral_warn = _wall_lateral_extent_mm(face_ids, ctx.opening_axis, ctx.occ_map)
    result.wall_lateral_extent_mm = _round_or_none(lateral)
    if lateral is None:
        msg = f"{label} face_ids={list(face_ids)}: could not compute wall_lateral_extent_mm"
        warnings.append(msg)
        logger.warning(msg)
    elif lateral_warn:
        warnings.append(f"{label}: {lateral_warn}")
        logger.warning("%s: %s", label, lateral_warn)

    result.warnings = warnings
    return result


def enrich_graph_wall_geometry(
    graph: dict[str, Any],
    step_path: str | Path,
    *,
    graph_npz: str | Path | None = None,
    opening_axis: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a copy of graph with wall sizing params attached to wall nodes."""
    ctx = build_wall_geom_context(
        graph, step_path, graph_npz=graph_npz, opening_axis=opening_axis,
    )
    enriched = copy.deepcopy(graph)
    for node in enriched.get("nodes", []):
        if node.get("class_name") != "wall":
            continue
        face_ids = [int(i) for i in node.get("face_ids", [])]
        geom = compute_wall_geometry(
            face_ids, ctx, feature_id=node.get("feature_id"),
        )
        params = node.setdefault("params", {})
        params.update(geom.to_params())
    return enriched


def _stats(values: Sequence[float | None]) -> dict[str, float | None]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return {"min": None, "max": None, "median": None}
    arr = np.asarray(nums, dtype=np.float64)
    return {
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "median": round(float(np.median(arr)), 4),
    }


def _effective_tool_cap_mm(
    depth: float | None,
    clearance: float | None,
    lateral: float | None,
) -> float | None:
    caps: list[float] = []
    if clearance is not None:
        caps.append(2.0 * clearance)
    if lateral is not None:
        caps.append(lateral)
    return min(caps) if caps else None


def verify_wall_geometry(
    graph: dict[str, Any],
    *,
    shop_small_tool_dia_mm: float = SHOP_SMALL_WALL_TOOL_DIA_MM,
    shop_large_tool_dia_mm: float = SHOP_LARGE_WALL_TOOL_DIA_MM,
) -> dict[str, Any]:
    """Summarize enriched wall params and sanity flags for CLI reporting."""
    walls = [n for n in graph.get("nodes", []) if n.get("class_name") == "wall"]
    rows: list[dict[str, Any]] = []
    flags: list[str] = []

    for node in sorted(walls, key=lambda item: int(item["feature_id"])):
        params = node.get("params") or {}
        depth = params.get("wall_depth_mm")
        clearance = params.get("wall_min_clearance_mm")
        lateral = params.get("wall_lateral_extent_mm")
        cap = _effective_tool_cap_mm(depth, clearance, lateral)

        row_flags: list[str] = []
        if depth is None:
            row_flags.append("depth=null")
        elif depth < WALL_DEPTH_MIN_MM or depth > WALL_DEPTH_MAX_MM:
            row_flags.append(f"depth={depth} OUT_OF_BAND")
        if clearance is None:
            row_flags.append("clearance=null")
        elif clearance < WALL_CLEARANCE_MIN_MM or clearance > WALL_CLEARANCE_MAX_MM:
            row_flags.append(f"clearance={clearance} OUT_OF_BAND")
        if lateral is None:
            row_flags.append("lateral=null")

        tool_class = None
        if cap is not None:
            if cap <= shop_small_tool_dia_mm + 1e-6:
                tool_class = "small_3_16"
            elif cap > shop_small_tool_dia_mm + 1e-6:
                tool_class = "large_3_8_candidate"

        rows.append({
            "feature_id": node["feature_id"],
            "face_ids": node.get("face_ids", []),
            "wall_depth_mm": depth,
            "wall_min_clearance_mm": clearance,
            "wall_lateral_extent_mm": lateral,
            "effective_tool_cap_mm": _round_or_none(cap),
            "predicted_tool_class": tool_class,
            "flags": row_flags,
        })
        flags.extend(row_flags)

    depths = [row["wall_depth_mm"] for row in rows]
    clearances = [row["wall_min_clearance_mm"] for row in rows]
    laterals = [row["wall_lateral_extent_mm"] for row in rows]

    small = [row for row in rows if row["predicted_tool_class"] == "small_3_16"]
    large = [row for row in rows if row["predicted_tool_class"] == "large_3_8_candidate"]

    lateral_vals = [r["wall_lateral_extent_mm"] for r in rows if r["wall_lateral_extent_mm"] is not None]
    tight_lateral = [v for v in lateral_vals if v < 2.0]
    open_lateral = [v for v in lateral_vals if v > 4.0]

    separation = {
        "n_walls": len(rows),
        "n_small_tool_candidates": len(small),
        "n_large_tool_candidates": len(large),
        "clearance_is_uniform": len({
            r["wall_min_clearance_mm"] for r in rows if r["wall_min_clearance_mm"] is not None
        }) <= 1,
        "n_tight_by_lateral": len(tight_lateral),
        "n_open_by_lateral": len(open_lateral),
        "lateral_separates_shop_split": len(tight_lateral) > 0 and len(open_lateral) > 0,
        "geometry_explains_shop_split": len(tight_lateral) > 0 and len(open_lateral) > 0,
    }

    return {
        "rows": rows,
        "stats": {
            "wall_depth_mm": _stats(depths),
            "wall_min_clearance_mm": _stats(clearances),
            "wall_lateral_extent_mm": _stats(laterals),
        },
        "sanity_flags": sorted(set(flags)),
        "shop_tool_reference_mm": {
            "small_3_16": shop_small_tool_dia_mm,
            "large_3_8": shop_large_tool_dia_mm,
        },
        "separation": separation,
    }


def print_verification_report(report: dict[str, Any]) -> None:
    rows = report["rows"]
    stats = report["stats"]
    sep = report["separation"]

    print(f"\n{'fid':>4} {'faces':>8} {'depth':>8} {'clearance':>10} {'lateral':>8} {'cap':>8}  flags")
    print("-" * 72)
    for row in rows:
        faces = ",".join(str(i) for i in row["face_ids"])
        flags = ",".join(row["flags"]) if row["flags"] else ""
        print(
            f"{row['feature_id']:4d} {faces:>8} "
            f"{_fmt(row['wall_depth_mm']):>8} "
            f"{_fmt(row['wall_min_clearance_mm']):>10} "
            f"{_fmt(row['wall_lateral_extent_mm']):>8} "
            f"{_fmt(row['effective_tool_cap_mm']):>8}  {flags}"
        )

    print("\nAggregate stats (14 walls):")
    for key, label in (
        ("wall_depth_mm", "wall_depth_mm"),
        ("wall_min_clearance_mm", "wall_min_clearance_mm"),
        ("wall_lateral_extent_mm", "wall_lateral_extent_mm"),
    ):
        s = stats[key]
        print(
            f"  {label}: min={_fmt(s['min'])} median={_fmt(s['median'])} max={_fmt(s['max'])}",
        )

    print("\nSanity gates:")
    out_of_band = [f for f in report["sanity_flags"] if "OUT_OF_BAND" in f]
    nulls = [f for f in report["sanity_flags"] if f.endswith("=null")]
    if out_of_band:
        print(f"  FLAG: {len(out_of_band)} value(s) outside sanity band")
        for item in out_of_band[:10]:
            print(f"    - {item}")
    else:
        print("  depth/clearance values within sanity bands (or null with log)")

    if nulls:
        print(f"  FLAG: {len(nulls)} null field(s) - see log lines above")

    print("\nShop tool-split cross-check:")
    print(
        f"  small-tool candidates (cap <= {report['shop_tool_reference_mm']['small_3_16']:.3f} mm): "
        f"{sep['n_small_tool_candidates']}",
    )
    print(
        f"  large-tool candidates (cap > {report['shop_tool_reference_mm']['small_3_16']:.3f} mm): "
        f"{sep['n_large_tool_candidates']}",
    )
    if sep["clearance_is_uniform"]:
        print(
            "  clearance is uniform across walls - lateral extent (not clearance) "
            "carries the shop tool-split signal",
        )
    print(
        f"  tight walls by lateral (< 2.0 mm): {sep['n_tight_by_lateral']}",
    )
    print(
        f"  open walls by lateral (> 4.0 mm): {sep['n_open_by_lateral']}",
    )
    if sep["geometry_explains_shop_split"]:
        print(
            "  PASS: lateral extent separates tight vs open walls "
            "(clearance alone does not)",
        )
    else:
        print(
            "  FLAG: lateral extent does not separate walls into tight and open groups",
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"


def write_enriched_graph(
    graph: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2)
        handle.write("\n")
    return output_path
