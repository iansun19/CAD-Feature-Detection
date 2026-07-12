"""Supabase-backed storage for ground-truth feature detections.

Persists the output of the feature-detection cascade (a ``feature_graph_cascade.json``
dict, as written by ``feature_graph.write_feature_graph``) into two tables:

    molds     - one row per detection run (name + detection_version baseline)
    features  - one row per detected graph node

The normalized columns (``feature_type``/``face_ids``/``dimensions``/``depth``) are
a queryable projection of each node; the full raw node is preserved in
``features.metadata`` and the graph-level fields in ``molds.metadata`` so a saved
mold can be reconstructed losslessly into a graph the planner accepts
(see :func:`reconstruct_feature_graph`).

Connection + env handling reuse ``tool_store`` (supabase-py client, repo-root .env,
``SUPABASE_URL`` + ``SUPABASE_SERVICE_ROLE_KEY``/``SUPABASE_KEY``).
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

# Reuse the existing connection/env pattern rather than duplicating it.
from tool_store import (  # noqa: F401  (re-exported for callers/tests)
    SupabaseConfigError,
    create_supabase_client,
    supabase_env,
)

MOLDS_TABLE = "molds"
FEATURES_TABLE = "features"
INSERT_RPC = "insert_mold_with_features"

# Graph-level keys (everything that is NOT the per-node list) that must survive a
# round-trip so the reconstructed graph still drives the planner correctly.
GRAPH_LEVEL_KEYS = (
    "schema_version",
    "part_id",
    "source",
    "n_faces",
    "n_features",
    "n_edges",
    "edges",
    "approach_frame",
    "reachability_summary",
    "slope_profile_summary",
    "chamfer_summary",
    "stock_face_ids",
    "stock_classifier",
)

# Candidate numeric param keys to surface as normalized `dimensions`. Names mirror
# what planner.cascade_node_to_feature reads out of `params`.
_DIMENSION_KEYS = (
    "nominal_diameter",
    "diameter_mm",
    "radius",
    "radius_mm",
    "fillet_radius_mm",
    "area",
    "area_mm2",
    "radial_mm",
    "nominal_diameter_mm",
    "axial_span_mm",
    "chamfer_size_mm",
    "chamfer_angle_deg",
    "template_deviation",
)

_DEPTH_KEYS = ("depth", "depth_mm", "depth_below_top_mm")


def _first_float(params: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = params.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return None


def _node_depth(params: Mapping[str, Any]) -> float | None:
    return _first_float(params, *_DEPTH_KEYS)


def _node_dimensions(params: Mapping[str, Any]) -> dict[str, float]:
    dims: dict[str, float] = {}
    for key in _DIMENSION_KEYS:
        val = _first_float(params, key)
        if val is not None:
            dims[key] = val
    return dims


def feature_node_to_row(node: Mapping[str, Any]) -> dict[str, Any]:
    """Project one feature-graph node into a `features` insert row.

    `metadata` keeps the entire raw node so reconstruction is lossless; the other
    columns are a normalized, queryable view of it.
    """
    params = node.get("params") or {}
    return {
        "feature_type": str(node.get("class_name", "")),
        "face_ids": list(node.get("face_ids") or []),
        "dimensions": _node_dimensions(params),
        "depth": _node_depth(params),
        "metadata": dict(node),
    }


def graph_to_mold_metadata(
    graph: Mapping[str, Any],
    *,
    setup_descriptor: Mapping[str, Any] | None = None,
    extents: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect graph-level fields (+ optional planning inputs) for `molds.metadata`.

    ``setup_descriptor`` and ``extents`` are optional but recommended: they let the
    plan generator rebuild a MachiningContext straight from the DB without needing
    the original STEP or setup YAML on disk.
    """
    meta = {key: graph[key] for key in GRAPH_LEVEL_KEYS if key in graph}
    if setup_descriptor is not None:
        meta["setup_descriptor"] = dict(setup_descriptor)
    if extents is not None:
        meta["stock_extents"] = dict(extents)
    return meta


def insert_mold_with_features(
    graph: Mapping[str, Any],
    *,
    name: str,
    detection_version: str,
    step_file_ref: str | None = None,
    setup_descriptor: Mapping[str, Any] | None = None,
    extents: Mapping[str, Any] | None = None,
    client: Any | None = None,
) -> str:
    """Persist a feature graph as a mold + its features in one DB transaction.

    ``graph`` is a loaded ``feature_graph_cascade.json`` dict. Returns the new
    mold id (uuid). Atomicity is provided by the ``insert_mold_with_features``
    Postgres function (see the migration) invoked via RPC.
    """
    nodes = graph.get("nodes") or []
    mold_payload = {
        "name": name,
        "detection_version": detection_version,
        "step_file_ref": step_file_ref,
        "metadata": graph_to_mold_metadata(
            graph, setup_descriptor=setup_descriptor, extents=extents
        ),
    }
    feature_payload = [feature_node_to_row(node) for node in nodes]

    sb = create_supabase_client(client)
    resp = sb.rpc(
        INSERT_RPC,
        {"p_mold": mold_payload, "p_features": feature_payload},
    ).execute()
    mold_id = resp.data
    if isinstance(mold_id, list):  # some client versions wrap scalar returns
        mold_id = mold_id[0] if mold_id else None
    if not mold_id:
        raise RuntimeError(f"{INSERT_RPC} returned no mold id: {resp!r}")
    return str(mold_id)


def load_mold(mold_id: str, *, client: Any | None = None) -> dict[str, Any]:
    """Fetch a single mold row by id."""
    sb = create_supabase_client(client)
    resp = sb.table(MOLDS_TABLE).select("*").eq("id", mold_id).limit(1).execute()
    rows = resp.data or []
    if not rows:
        raise KeyError(f"mold not found: {mold_id}")
    return rows[0]


def find_mold(
    name: str, detection_version: str, *, client: Any | None = None
) -> dict[str, Any] | None:
    """Look up a mold by its (name, detection_version) baseline key."""
    sb = create_supabase_client(client)
    resp = (
        sb.table(MOLDS_TABLE)
        .select("*")
        .eq("name", name)
        .eq("detection_version", detection_version)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def load_mold_features(
    mold_id: str, *, client: Any | None = None
) -> list[dict[str, Any]]:
    """Load a mold's feature rows, ordered by the node feature_id."""
    sb = create_supabase_client(client)
    resp = (
        sb.table(FEATURES_TABLE)
        .select("*")
        .eq("mold_id", mold_id)
        .execute()
    )
    rows = list(resp.data or [])
    # Order by the original node feature_id (kept in metadata), not row insert order.
    rows.sort(key=lambda r: _row_feature_id(r))
    return rows


def _row_feature_id(row: Mapping[str, Any]) -> int:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    fid = meta.get("feature_id")
    return int(fid) if isinstance(fid, (int, float)) else 1 << 30


def reconstruct_feature_graph(
    mold_id: str, *, client: Any | None = None
) -> dict[str, Any]:
    """Rebuild a feature_graph_cascade.json-shaped dict from stored rows.

    The reconstructed graph is byte-compatible with what the cascade wrote: nodes
    come verbatim from ``features.metadata`` and graph-level fields from
    ``molds.metadata``. Suitable input for machining_context.build_context_v0.
    """
    mold = load_mold(mold_id, client=client)
    feature_rows = load_mold_features(mold_id, client=client)

    meta = mold.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)

    nodes = []
    for row in feature_rows:
        node = row.get("metadata") or {}
        if isinstance(node, str):
            node = json.loads(node)
        nodes.append(node)

    graph: dict[str, Any] = {
        key: meta[key] for key in GRAPH_LEVEL_KEYS if key in meta
    }
    graph.setdefault("part_id", mold.get("name"))
    graph.setdefault("source", "cascade")
    graph["nodes"] = nodes
    graph["n_features"] = len(nodes)
    graph.setdefault("edges", meta.get("edges", []))
    return graph


def mold_planning_inputs(
    mold_id: str, *, client: Any | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Return (graph, setup_descriptor, extents) stored for a mold.

    setup_descriptor / extents are None if they were not captured at insert time.
    """
    mold = load_mold(mold_id, client=client)
    meta = mold.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    graph = reconstruct_feature_graph(mold_id, client=client)
    return graph, meta.get("setup_descriptor"), meta.get("stock_extents")
