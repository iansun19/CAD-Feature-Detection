"""Editable feature-type -> operation mapping config.

The authoritative op generation is done by ``planner.map_feature_to_operations``
(geometry-driven: it picks drill vs. bore by fit, rough vs. finish by access, tool
by catalog, feeds/speeds by handbook). This module is the human-readable,
hand-editable declaration of that intent — grouped by the same feature classes the
planner keys on (``planner.HOLE_CLASSES`` etc.) and using canonical
``operation_bank`` op names.

Use it to:
  * see/adjust which feature_types are in planning scope,
  * document the expected operations per feature_type,
  * validate that a generated CamPlan only emitted expected ops (report layer),
  * emit a lightweight placeholder op plan when a full MachiningContext (tools,
    stock) isn't available — see ``lightweight_operations_for``.

Edit ``FEATURE_OPERATION_MAP`` freely; nothing here changes cutter geometry.
"""
from __future__ import annotations

from typing import Any, Mapping

# Canonical op names (operation_bank.Operation values). Kept as plain strings so
# this config stays import-light and editable without pulling in the planner.
FEATURE_OPERATION_MAP: dict[str, list[str]] = {
    # holes -> drill (peck/tap/bore/ream cycles), escalate to helix bore if no drill fits
    "through_hole": ["drill"],
    "hole": ["drill"],
    "blind_hole": ["drill"],
    "filleted_blind_hole": ["drill", "helix_bore"],
    # counterbore = through bore + enlarged flat-bottomed recess: drill the bore,
    # helix-bore the enlargement (dedicated toolpath class, not overloaded).
    "counterbore": ["drill", "helix_bore"],
    # countersink = through bore + conical flush-seat entry: drill the bore, then
    # cut the cone with a chamfer/countersink tool (dedicated toolpath class).
    "countersink": ["drill", "chamfer"],
    # pockets -> rough clear + wall/floor finish
    "pocket": ["pocket", "contour_2d"],
    "blind_pocket": ["pocket", "contour_2d"],
    "open_pocket": ["dynamic_mill_2d", "contour_2d"],
    "filleted_pocket": ["pocket", "contour_2d"],
    "filleted_open_pocket": ["dynamic_mill_2d", "contour_2d"],
    # walls / profiles -> contour
    "wall": ["contour_2d"],
    "profile": ["contour_2d"],
    # freeform / axisymmetric surfaces -> 3D finish
    "contour_surface": ["waterline", "raster"],
    # fillets -> pencil / waterline blend cleanup (planner FILLET_CLASSES incl. inner_fillet)
    "outer_fillet": ["waterline"],
    "fillet": ["waterline"],
    "inner_fillet": ["pencil"],
    # flats / faces -> facing
    "flat": ["facing"],
    "face": ["facing"],
    # chamfers -> chamfer / edge break
    "chamfer": ["chamfer"],
}

# feature_types the planner will actually consume (everything mapped above).
IN_SCOPE_FEATURE_TYPES = frozenset(FEATURE_OPERATION_MAP)


def operations_for(feature_type: str) -> list[str]:
    """Expected operation names for a feature_type ([] if out of scope)."""
    return list(FEATURE_OPERATION_MAP.get(feature_type, ()))


def is_in_scope(feature_type: str) -> bool:
    return feature_type in IN_SCOPE_FEATURE_TYPES


def lightweight_operations_for(
    feature: Mapping[str, Any], setup_id: str = "default"
) -> list[dict[str, Any]]:
    """Emit placeholder operation records for one stored `features` row.

    A config-only fallback (no tool selection / feeds-speeds computed) for
    inspecting the mapping without a MachiningContext. Shape mirrors the fields of
    cam_plan_schema.Operation so it reads the same as the planner path.
    """
    ftype = str(feature.get("feature_type", ""))
    fid = str((feature.get("metadata") or {}).get("feature_id", feature.get("id", "")))
    dims = feature.get("dimensions") or {}
    ops: list[dict[str, Any]] = []
    for op_name in operations_for(ftype):
        ops.append(
            {
                "operation": op_name,
                "feature_type": ftype,
                "feature_refs": [fid],
                "setup_id": setup_id,
                "suggested_tool": None,          # placeholder — resolved by planner
                "parameters": {                  # placeholders — resolved by planner
                    "spindle_rpm": None,
                    "feed_mm_per_min": None,
                    "plunge_mm_per_min": None,
                    "stepdown_mm": None,
                    "stepover_mm": None,
                    "param_source": "placeholder",
                },
                "dimensions": dict(dims),
                "depth": feature.get("depth"),
            }
        )
    return ops
