#!/usr/bin/env python3
"""
inspect_feature_geometry.py - READ-ONLY inventory of cascade feature-graph node
geometry for tool-sizing / wall-split scoping.

Inspects the real 96260B_rear cascade graph and reports:
  1. Node schema by feature class
  2. Full raw sample nodes
  3. Geometry quantity presence matrix
  4. Where geometry is computed (codebase map)
  5. Reusable B-rep helpers in stock_cut_classification.py
"""
from __future__ import annotations

import argparse
import ast
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GRAPH_CANDIDATES = [
    REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json",
    REPO_ROOT / "pipeline_out" / "96260B_rear_copy" / "feature_graph_cascade.json",
    REPO_ROOT / "pipeline_out" / "96260B_rear_verify" / "feature_graph_cascade.json",
]

FOCUS_CLASSES = frozenset({
    "wall",
    "filleted_pocket",
    "filleted_open_pocket",
    "open_pocket",
    "pocket",
    "contour_surface",
    "filleted_blind_hole",
    "through_hole",
    "profile",
    "flat",
    "face",
    "outer_fillet",
})

SAMPLE_CLASSES = [
    ("wall", ("wall",)),
    ("pocket", ("filleted_pocket", "filleted_open_pocket", "open_pocket", "pocket")),
    ("contour_surface", ("contour_surface",)),
    ("flat/face", ("flat", "face")),
]

GEOMETRY_CHECKS: list[tuple[str, list[str]]] = [
    ("depth", [
        "depth", "depth_mm", "depth_below_top_mm", "depth_along_floor_normal",
        "axial_span_mm",
    ]),
    ("lateral_extent / bbox", [
        "lateral_extent_mm", "bbox_depth", "bbox_width", "bbox_length",
        "bbox_size_x", "bbox_size_y", "bbox_size_z", "radial_mm",
    ]),
    ("fillet / corner radius", [
        "fillet_radius_mm", "fillet_radius", "fillet_radii", "radius",
        "nominal_diameter", "nominal_diameter_mm", "torus_minor_radius",
    ]),
    ("clearance / concavity", [
        "internal_edge_convexity", "min_signed_convexity", "tightest",
        "clearance", "min_internal_radius",
    ]),
    ("axis / normal / orientation", [
        "axis", "opening_axis", "normal", "floor_normal", "offset",
    ]),
]


def resolve_graph_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"graph not found: {path}")
        return path.resolve()
    for candidate in DEFAULT_GRAPH_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "no cascade graph found; tried:\n"
        + "\n".join(f"  {p}" for p in DEFAULT_GRAPH_CANDIDATES)
    )


def node_key_set(node: dict[str, Any]) -> set[str]:
    keys: set[str] = set(node.keys())
    params = node.get("params") or {}
    for key in params:
        keys.add(f"params.{key}")
    return keys


def pick_representative(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the node with the richest params payload."""
    return max(nodes, key=lambda n: len(n.get("params") or {}))


def _float_param(params: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = params.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def planner_lateral_extent(params: dict[str, Any]) -> float | None:
    dims: list[float] = []
    for key in (
        "bbox_width", "bbox_length", "bbox_depth",
        "bbox_size_x", "bbox_size_y", "bbox_size_z",
    ):
        value = _float_param(params, key)
        if value is not None and value > 1e-3:
            dims.append(value)
    return min(dims) if dims else None


def presence_for_class(
    nodes: list[dict[str, Any]],
    candidate_keys: list[str],
) -> tuple[str, int, int]:
    """Return (status, present_count, total) across nodes of one class."""
    total = len(nodes)
    if total == 0:
        return "absent (no nodes)", 0, 0

    hits = 0
    found_keys: set[str] = set()
    for node in nodes:
        params = node.get("params") or {}
        for key in candidate_keys:
            if key in params and params[key] is not None:
                hits += 1
                found_keys.add(key)
                break

    if hits == total:
        return f"present ({', '.join(sorted(found_keys))})", hits, total
    if hits > 0:
        return f"partial ({hits}/{total}; keys: {', '.join(sorted(found_keys))})", hits, total
    return "absent", 0, total


def section_header(title: str) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)


def section_1_schema(graph: dict[str, Any]) -> None:
    section_header("1. NODE SCHEMA BY FEATURE CLASS")
    print(f"part_id: {graph.get('part_id')}")
    print(f"schema_version: {graph.get('schema_version')}")
    print(f"n_nodes: {len(graph.get('nodes', []))}")
    print()

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph.get("nodes", []):
        by_class[str(node.get("class_name", "?"))].append(node)

    for class_name in sorted(by_class.keys()):
        nodes = by_class[class_name]
        union: set[str] = set()
        counts: dict[str, int] = defaultdict(int)
        for node in nodes:
            for key in node_key_set(node):
                union.add(key)
                counts[key] += 1
        n = len(nodes)
        always = sorted(k for k in union if counts[k] == n)
        sometimes = sorted(k for k in union if counts[k] < n)

        marker = "  ***" if class_name in FOCUS_CLASSES else ""
        print(f"--- {class_name} ({n} nodes){marker} ---")
        print(f"  always-present ({len(always)}):")
        for key in always:
            print(f"    {key}")
        if sometimes:
            print(f"  sometimes-present ({len(sometimes)}):")
            for key in sometimes:
                print(f"    {key}  ({counts[key]}/{n})")
        print()


def section_2_samples(graph: dict[str, Any]) -> None:
    section_header("2. REAL SAMPLE NODES (FULL RAW JSON)")
    nodes = graph.get("nodes", [])
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_class[str(node.get("class_name", "?"))].append(node)

    for label, class_candidates in SAMPLE_CLASSES:
        picked = None
        picked_class = None
        for cls in class_candidates:
            if cls in by_class:
                picked = pick_representative(by_class[cls])
                picked_class = cls
                break
        print(f"### {label} (class_name={picked_class or 'NOT FOUND'}) ###")
        if picked is None:
            print("(no node of this class in graph)\n")
            continue
        print(json.dumps(picked, indent=2, sort_keys=False))
        print()


def section_3_inventory(graph: dict[str, Any]) -> None:
    section_header("3. EXISTING GEOMETRY INVENTORY (TOOL-SIZING / WALL-SPLIT)")
    print(
        "Legend: present = key exists with non-null value on ALL nodes of class;\n"
        "        partial = on some nodes; absent = never on params.\n"
        "Planner adapter (cascade_node_to_feature) reads depth from depth|depth_mm|\n"
        "depth_below_top_mm; lateral_extent from bbox_* keys; fillet from fillet_radius_mm.\n"
    )

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph.get("nodes", []):
        by_class[str(node.get("class_name", "?"))].append(node)

    all_classes = sorted(by_class.keys())
    header = f"{'quantity':<28}" + "".join(f"{c[:14]:>15}" for c in all_classes)
    print(header)
    print("-" * len(header))

    for quantity, keys in GEOMETRY_CHECKS:
        row = f"{quantity:<28}"
        for cls in all_classes:
            status, _, _ = presence_for_class(by_class[cls], keys)
            if status.startswith("present"):
                cell = status.replace("present (", "").rstrip(")")
            elif status.startswith("partial"):
                cell = status.split(";", 1)[0].replace("partial (", "part:")
            else:
                cell = "absent"
            row += f"{cell[:14]:>15}"
        print(row)

    print()
    print("Per-class detail (focus classes):")
    for cls in sorted(FOCUS_CLASSES):
        if cls not in by_class:
            continue
        nodes = by_class[cls]
        print(f"  {cls} ({len(nodes)} nodes):")
        for quantity, keys in GEOMETRY_CHECKS:
            status, _, _ = presence_for_class(nodes, keys)
            print(f"    {quantity}: {status}")
        depths = [
            _float_param(n.get("params") or {}, "depth", "depth_mm", "depth_below_top_mm")
            for n in nodes
        ]
        laterals = [planner_lateral_extent(n.get("params") or {}) for n in nodes]
        fillets = [_float_param(n.get("params") or {}, "fillet_radius_mm") for n in nodes]
        print(
            f"    planner depth_mm would be non-null on "
            f"{sum(d is not None for d in depths)}/{len(nodes)} nodes"
        )
        print(
            f"    planner lateral_extent_mm would be non-null on "
            f"{sum(l is not None for l in laterals)}/{len(nodes)} nodes"
        )
        print(
            f"    planner fillet_radius_mm would be non-null on "
            f"{sum(f is not None for f in fillets)}/{len(nodes)} nodes"
        )

    print()
    print("PIVOTAL ANSWER - WALL nodes:")
    walls = by_class.get("wall", [])
    if not walls:
        print("  No wall nodes in graph.")
    else:
        wall_params_keys = sorted({
            k.replace("params.", "")
            for n in walls
            for k in node_key_set(n)
            if k.startswith("params.")
        })
        print(f"  params keys on wall nodes: {wall_params_keys}")
        depth_status, _, _ = presence_for_class(walls, [
            "depth", "depth_mm", "depth_below_top_mm", "depth_along_floor_normal",
        ])
        radius_status, _, _ = presence_for_class(walls, [
            "fillet_radius_mm", "fillet_radius", "radius",
        ])
        od_status, _, _ = presence_for_class(walls, ["nominal_diameter_mm"])
        print(f"  depth (any depth* key): {depth_status}")
        print(f"  fillet/corner clearance radius: {radius_status}")
        print(f"  nominal_diameter_mm (exterior OD, NOT clearance): {od_status}")
        print(
            "  CONCLUSION: wall nodes carry nominal_diameter_mm + radial_mm (exterior OD\n"
            "  positioning) but NOT depth and NOT fillet/corner radius for tool clearance.\n"
            "  Planner cascade_node_to_feature() yields depth_mm=None, lateral_extent_mm=None,\n"
            "  fillet_radius_mm=None for all wall nodes today."
        )


def _first_line(doc: str | None) -> str:
    if not doc:
        return "(no docstring)"
    line = doc.strip().splitlines()[0].strip()
    return line or "(no docstring)"


def _module_functions(path: Path, *, public_only: bool = True) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text())
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if public_only and node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node) or ""
        out.append((node.name, _first_line(doc)))
    return out


def section_4_codebase_map() -> None:
    section_header("4. WHERE GEOMETRY IS COMPUTED (CODEBASE MAP)")

    print("Cascade graph params come from per-pass detectors' to_dict() output")
    print("(eval_cascade.build_cascade_graph / run_cascade), NOT from feature_params")
    print("enrich_graph_with_params unless a separate CAM export pass is run.\n")

    fp = REPO_ROOT / "feature_params.py"
    be = REPO_ROOT / "brep_extents.py"

    print("--- feature_params.py ---")
    print("Per-feature OCC reads on indexed face groups (class_id-aware).")
    print("Index export (schema v2) KEEPS only: analytic_surfaces, surface_counts,")
    print("total_area, cylindrical_radii. Heuristic aggregates go to derived_debug.")
    for name, desc in _module_functions(fp):
        if name in {
            "extract_feature_params", "enrich_graph_with_params", "bbox_params",
            "span_along_direction", "count_walls", "analytic_surfaces",
            "brep_extent_along_direction", "brep_boundary_extents",
            "collect_derived_debug", "split_params_for_cam",
        }:
            print(f"  {name}: {desc}")

    print("\n  Class routing in extract_feature_params:")
    print("    through_hole(0), blind_hole(5): radius, diameter, axis, length_along_axis,")
    print("      depth_along_floor_normal (blind); all in derived_debug after CAM split.")
    print("    pocket-like(1,2,6,7): n_walls, bbox_*, depth_along_floor_normal, floor_normal.")
    print("    step(3,8): step_height + bbox.")
    print("    chamfer(9): semi_angle_deg + bbox.")
    print("    fillet(10): fillet_radius, torus_minor_radius.")
    print("    default: common + bbox.")

    print("\n--- brep_extents.py ---")
    print("B-rep machined extents from OCC boundary geometry (per feature, not global).")
    for name, desc in _module_functions(be):
        if name in {
            "resolve_occ_faces", "collect_boundary_points", "axis_extent",
            "feature_aabb", "find_pocket_floor", "hole_machined_extents",
            "pocket_machined_extents", "step_machined_extents",
            "chamfer_machined_extents", "fillet_machined_extents",
            "compute_machined_extents",
        }:
            print(f"  {name}: {desc}")

    print("\n--- lateral_extent_mm / fillet_radius_mm sizing fix ---")
    print("Location: planner.py (on-the-fly at plan time, NOT persisted on graph nodes).")
    print("  _lateral_extent_mm(params): min of bbox_width|length|depth|size_* from params.")
    print("  cascade_node_to_feature(): reads depth_below_top_mm; fillet_radius_mm; bbox for lateral.")
    print("  _max_open_tool_diameter_mm(): min(lateral_extent, 2*fillet_radius) for wall/finish ops.")
    print("  group_operations_by_tool_strategy() -> _split_members_by_reachability() splits batches")
    print("    when tool flute_length or diameter caps fail aggregated member constraints.")
    print("\n  POPULATED ON CASCADE NODES TODAY:")
    print("    fillet_radius_mm + depth_below_top_mm: filleted_pocket, filleted_open_pocket")
    print("      (computed in pocket_detection._depth_below_top_mm / _fillet_radius_mm).")
    print("    depth: filleted_blind_hole, through_hole (hole_detection).")
    print("    bbox_*: contour_surface residual groups (residual_detection -> instance_features).")
    print("    NONE of above on wall nodes.")

    print("\n--- pocket_detection.py (cascade pocket pass) ---")
    print("  _depth_below_top_mm: part_axis_top - min boundary projection along opening_axis.")
    print("  _fillet_radius_mm: max sphere/torus blend radius bordering pocket walls.")
    print("  Persisted on node params as depth_below_top_mm, fillet_radius_mm.")

    print("\n--- wall_detection.py (cascade wall pass) ---")
    print("  WallFeature.to_dict: nominal_diameter_mm, radial_mm only (exterior OD sliver).")

    print("\n--- instance_features.py (residual / contour pass) ---")
    print("  instance_features: surface histogram, total_area, bbox_*, internal_edge_convexity.")

    print("\n--- hole_detection.py ---")
    print("  HoleFeature.to_dict: depth, radius, nominal_diameter, axis, blend/cap indices.")


def section_5_stock_cut() -> None:
    section_header("5. REUSABLE B-REP HELPERS IN stock_cut_classification.py")

    print("OCCT / B-rep queries performed:")
    print("  - Part-level AABB via BRepBndLib.AddOptimal on full shape (_shape_aabb)")
    print("  - Per-face envelope coincidence: boundary points vs part AABB (_envelope_coincident)")
    print("  - Per-face edge convexity summary from brep adjacency model (_convex_summary)")
    print("  - Normal orientation consistency: face_mid_normal vs BRepLProp_SLProps (_normal_consistent)")
    print("  - Extreme / axis-aligned planar stock face matching (_extreme_plane_match)")
    print("  - Imports collect_boundary_points from brep_extents for envelope test")
    print()

    print("Facing-specific vs general:")
    print("  FACING-SPECIFIC: classify_report gate rules, envelope_stock_face_ids,")
    print("    stock_face_ids, _is_axis_extreme_stock, _count_bounding_stock, fixture diffs.")
    print("  GENERAL / REUSABLE:")
    print("    _load_cad: STEP -> faces, FaceGeom records, adjacency model, part AABB, edge_info")
    print("    _envelope_coincident: distance of face boundary to stock envelope")
    print("    _convex_summary: per-face concave/convex/smooth edge tallies + min_signed_convexity")
    print("    _normal_consistent: parametric vs oriented normal agreement")
    print("    collect_boundary_points (brep_extents): face boundary point sampling")
    print("  NOT PRESENT: per-feature depth, corner radius, face-to-face distance,")
    print("    or wall-specific extent along opening axis.")
    print()

    print("Reusable function signatures (public + key internals):")
    signatures = [
        "def classify_report(cad, *, classifier='new', cascade_face_ids=None) -> list[FaceClassificationRecord]",
        "def diff_classification(cad, *, part_id=None, corner_chamfer_ids=None) -> ClassificationDiff",
        "def envelope_stock_face_ids(cad, *, classifier='new') -> set[int]",
        "def stock_face_ids(cad, *, classifier='new') -> set[int]",
        "def run_fixture_diff(fixture_name, *, repo_root=None) -> ClassificationDiff",
        "def format_flip_report(diff: ClassificationDiff) -> str",
        "def _load_cad(step_path) -> _CadContext  # faces, geoms, part_aabb, edge_info",
        "def _envelope_coincident(face_idx, ctx, *, tol_mm=0.05) -> EnvelopeCoincident",
        "def _convex_summary(face_idx, ctx) -> ConvexSummary",
        "def _normal_consistent(face_idx, ctx) -> bool",
        "def _shape_aabb(shape) -> tuple[np.ndarray, np.ndarray]",
    ]
    for sig in signatures:
        print(f"  {sig}")

    print()
    print("Related reusable helpers OUTSIDE stock_cut_classification.py:")
    print("  brep_extents.collect_boundary_points(occ_face) -> np.ndarray")
    print("  brep_extents.axis_extent(occ_faces_by_index, face_indices, axis) -> Extent")
    print("  brep_extents.feature_aabb(occ_faces_by_index, face_indices) -> AABB")
    print("  brep_extents.find_pocket_floor(records, face_indices, occ_map) -> (floor, opening, errors)")
    print("  feature_params.analyze_face(face, index) -> FaceGeom")
    print("  feature_params.bbox_params(centroids) -> dict  # centroid-span bbox, not B-rep")
    print("  pocket_detection._depth_below_top_mm(...)  # opening-axis depth for pockets")
    print("  pocket_detection._fillet_radius_mm(...)    # blend radius for pockets")


def section_summary(graph_path: Path) -> None:
    section_header("SUMMARY - WALL-SPLIT ENRICHMENT SCOPE")
    print(f"Inspected graph: {graph_path}")
    print()
    print(textwrap.fill(
        "EXISTING on cascade nodes for tool sizing: pockets have depth_below_top_mm "
        "and fillet_radius_mm; holes have depth; contour_surface residual groups have "
        "centroid-span bbox_* (via instance_features); walls have only nominal_diameter_mm "
        "and radial_mm (exterior envelope position, not clearance).",
        width=72,
    ))
    print()
    print(textwrap.fill(
        "MISSING for wall-split: per-wall depth along opening axis, per-wall lateral "
        "clearance / tightest concave radius, and any corner radius. Planner reads "
        "bbox/fillet/depth from params at plan time but walls supply none of these - "
        "reachability splits fall back to OPEN_DEFAULT_MAX_DIA_MM.",
        width=72,
    ))
    print()
    print(textwrap.fill(
        "VERDICT: wall-split enrichment requires a NEW B-rep geometry pass (or extending "
        "wall_detection / feature_params for wall class_id=-1), not merely surfacing "
        "fields the planner already has. Pocket depth/fillet logic in pocket_detection "
        "and brep_extents is the closest reuse template.",
        width=72,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        help="Path to feature_graph_cascade.json (default: auto-detect 96260B_rear)",
    )
    args = parser.parse_args()

    graph_path = resolve_graph_path(args.graph)
    graph = json.loads(graph_path.read_text())

    print("FEATURE GRAPH GEOMETRY INSPECTION REPORT")
    print(f"Generated from: {graph_path}")
    print()

    section_1_schema(graph)
    section_2_samples(graph)
    section_3_inventory(graph)
    section_4_codebase_map()
    section_5_stock_cut()
    section_summary(graph_path)


if __name__ == "__main__":
    main()
