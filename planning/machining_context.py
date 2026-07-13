# DESIGN NOTES
# -----------
# v0 assumptions:
#   - Single setup per context slice (setups is list-typed for future multi-setup).
#   - 3-axis only: opening_axis is a discrete label (e.g. "+Y"), not a free vector.
#   - No in-process stock tracking: remaining_material is always null.
#   - completed_operations is always empty in v0.
#   - machine_limits is optional and usually null.
#
# Layer boundary rule (repo convention):
#   Pydantic v2 at layer boundaries; dataclasses remain internal to perception.
#   Do not convert this module to dataclasses - it is a planner input contract.
#
# Access authority rule:
#   Setup descriptor YAML is the SOLE source of truth for pocket open/closed access.
#   Cascade params.access may be cross-checked but must NEVER override the descriptor.
#   Geometry cannot recover access; missing descriptor entries resolve to "unknown".
#
# brep_extents API note:
#   There is no part-level envelope helper; stock uses feature_aabb() over all STEP faces.
#
"""machining_context.py - planner input contract: machining state assembly (v0).

Assembles stock envelope, setup facts, and tool catalog from existing artifacts:
STEP / brep extents, setup descriptor YAML, and feature_graph_cascade.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cascade.setup_descriptor import (
    PartSetupDescriptor,
    ResolvedSetup,
    SetupDescriptorError,
    load_setup_descriptor,
    pocket_access_for_seed,
    resolve_setup_entry,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "0.1.0"
from env_bootstrap import REPO_ROOT
DEFAULT_TOOL_LIBRARIES_DIR = REPO_ROOT / "tool_libraries"

# Tool types requested when loading catalogs for the v0 planner slice.
V0_PLANNER_TOOL_TYPES: tuple[str, ...] = (
    "drill",
    "endmill",
    "bullnose_endmill",
    "tap",
    "ball_endmill",
    "face_mill",
)

# Pocket classes covered by setup descriptor pocket_access / pockets.by_seed_face.
DESCRIPTOR_POCKET_CLASSES = frozenset({
    "filleted_pocket",
    "filleted_open_pocket",
})

# All pocket-like classes we emit into SetupContext.pocket_access.
POCKET_ACCESS_CLASSES = frozenset({
    "filleted_pocket",
    "filleted_open_pocket",
    "pocket",
    "open_pocket",
    "blind_pocket",
})

_AXIS_LABELS = ("X", "Y", "Z")
_INCH_TO_MM = 25.4

# Fusion/Autodesk raw type string -> planner vocabulary (single normalization point).
_FUSION_TOOL_TYPE_MAP: dict[str, str] = {
    "flat end mill": "endmill",
    "face mill": "face_mill",
    "ball end mill": "ball_endmill",
    "bull nose end mill": "bullnose_endmill",
    "chamfer mill": "chamfer_mill",
    "counter sink": "countersink",
    "drill": "drill",
    "slot mill": "slot_mill",
    "spot drill": "spot_drill",
    "tap right hand": "tap",
    "tapping": "tap",
}


class Stock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_type: str = Field(default="bbox", description='v0: always "bbox".')
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    source: str = Field(
        default="brep_extents+offset",
        description="Envelope provenance, e.g. brep_extents+offset.",
    )


class SetupScopeSpec(BaseModel):
    """Declared machinable scope for one setup (from setup descriptor YAML)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["full", "filtered"] = "full"
    classes: list[str] = Field(default_factory=list)
    feature_ids: list[str] = Field(default_factory=list)

    @property
    def is_full(self) -> bool:
        return self.mode == "full"

    @property
    def stock_boundary_only(self) -> bool:
        return any(c in ("facing", "stock_face") for c in self.classes)


class OpeningAxisResolutionError(ValueError):
    """Raised when a setup descriptor lacks a resolvable opening axis for planning."""


class SetupContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    setup_id: str
    opening_axis: str
    opening_axis_vector: tuple[float, float, float] = Field(
        description="Unit opening-axis vector resolved from the setup descriptor.",
    )
    opening_axis_determined: bool = Field(
        default=True,
        description=(
            "True when the opening axis is authoritative: either explicitly set "
            "in the descriptor, or auto-detected from geometry that could resolve "
            "it. False when the cascade could not derive the axis from geometry "
            "(approach_frame.opening_axis_determined is False) -- the planner then "
            "fails loud and asks for an explicit axis. Defaults True for graphs "
            "emitted before this provenance existed."
        ),
    )
    machining_side: str | None = Field(
        default=None,
        description="front | back from descriptor; required for reachability scoping.",
    )
    orientation: str | None = Field(
        default=None,
        description=(
            "General cardinal setup orientation (e.g. '+X'), extending front/back "
            "to all six cardinals. None => legacy machining_side + opening_axis path. "
            "PROVISIONAL lateral ±X/±Y path when set (see lateral_axes.py)."
        ),
    )
    orientation_provisional: bool = Field(
        default=False,
        description="True when orientation is a provisional lateral-axis direction.",
    )
    pocket_access: dict[str, str] = Field(
        default_factory=dict,
        description='feature_id (str) -> open|closed|unknown; descriptor YAML only.',
    )
    scope: SetupScopeSpec = Field(
        default_factory=SetupScopeSpec,
        description="Machinable feature scope: full, class list, or explicit feature_ids.",
    )
    engrave: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Declared engraving specs for this setup (explicit process input): each "
            '{text, target:{feature_id|datum}, depth_mm}. Drives the engraving producer.'
        ),
    )
    fixture: str | None = None
    source_step_file: str | None = None


class ToolPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset_name: str
    preset_material: str | None = None
    spindle_rpm: float | None = Field(default=None, gt=0.0)
    feed_mm_per_min: float | None = Field(default=None, gt=0.0)
    plunge_mm_per_min: float | None = Field(default=None, gt=0.0)
    retract_mm_per_min: float | None = Field(default=None, gt=0.0)
    stepdown_mm: float | None = Field(default=None, gt=0.0)
    stepover_mm: float | None = Field(default=None, gt=0.0)
    coolant: str | None = None
    chip_load: float | None = Field(default=None, description="Fusion f_z passthrough.")
    surface_speed: float | None = Field(default=None, description="Fusion v_c passthrough.")
    feed_per_rev_mm: float | None = Field(
        default=None, gt=0.0, description="Fusion f_n converted to mm/rev."
    )
    feed_per_rev_retract_mm: float | None = Field(
        default=None, gt=0.0, description="Fusion f_n_retract converted to mm/rev."
    )
    use_feed_per_revolution: bool | None = Field(
        default=None, description="Fusion use-feed-per-revolution flag."
    )


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str
    tool_type: str
    diameter_mm: float = Field(..., gt=0.0)
    flute_length_mm: float | None = Field(default=None, gt=0.0)
    max_depth_mm: float | None = Field(default=None, gt=0.0)
    source: str = "hardcoded_v0"
    name: str | None = None
    corner_radius_mm: float | None = Field(default=None, ge=0.0)
    taper_angle_deg: float | None = Field(default=None, ge=0.0)
    point_angle_deg: float | None = Field(
        default=None, ge=0.0, description="Drill point angle from Fusion geometry.SIG."
    )
    flute_count: int | None = Field(default=None, gt=0)
    shank_diameter_mm: float | None = Field(default=None, gt=0.0)
    overall_length_mm: float | None = Field(default=None, gt=0.0)
    tool_material: str | None = None
    vendor: str | None = None
    product_id: str | None = None
    product_link: str | None = None
    tool_catalog_product_id: str | None = None
    source_unit: str | None = None
    guid: str | None = None
    presets: list[ToolPreset] = Field(default_factory=list)


class MachineLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rpm: float | None = Field(default=None, gt=0.0)
    max_feed_mm_per_min: float | None = Field(default=None, gt=0.0)
    travel_x_mm: float | None = Field(default=None, gt=0.0)
    travel_y_mm: float | None = Field(default=None, gt=0.0)
    travel_z_mm: float | None = Field(default=None, gt=0.0)


class MachiningContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    part_id: str
    feature_graph_ref: str
    stock: Stock
    setups: list[SetupContext]
    tools: list[Tool]
    material: str | None = Field(
        default=None,
        description=(
            "Workpiece material category (e.g. aluminum, steel, stainless). "
            "Selects toolpath presets by preset_material; None prefers 'all' presets "
            "or handbook fallbacks."
        ),
    )
    completed_operations: list[str] = Field(default_factory=list)
    remaining_material: None = None
    machine_limits: MachineLimits | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def resolve_pocket_access(
    feature_id: str,
    descriptor_access: str | None,
    cascade_access: str | None = None,
) -> str:
    """Return descriptor access; warn when cascade disagrees. Missing -> unknown."""
    if descriptor_access is None:
        return "unknown"

    label = str(descriptor_access).strip().lower()
    if label not in ("open", "closed", "unknown"):
        label = "unknown"

    if cascade_access is not None:
        cascade_label = str(cascade_access).strip().lower()
        if cascade_label != label:
            logger.warning(
                "pocket access disagreement for feature_id=%s: "
                "descriptor=%s cascade=%s (descriptor wins)",
                feature_id,
                label,
                cascade_label,
            )

    return label


def vector_to_opening_axis_label(vector: Sequence[float]) -> str:
    """Map a unit-ish 3-vector to a discrete 3-axis label, e.g. (0,1,0) -> '+Y'."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("opening axis vector must be non-zero")
    arr = arr / norm

    best_i = int(np.argmax(np.abs(arr)))
    if abs(arr[best_i]) < 0.5:
        raise ValueError(
            f"opening axis vector {list(vector)!r} is not aligned to a principal axis"
        )

    sign = "+" if arr[best_i] >= 0 else "-"
    return f"{sign}{_AXIS_LABELS[best_i]}"


def normalize_tool_type(raw_type: str) -> str:
    """Map a Fusion vendor type string to planner vocabulary.

    TODO: tap strings ("tap right hand", "tapping") are mapped here but not yet
    verified against a real tap library export.
    """
    key = str(raw_type).strip().lower()
    normalized = _FUSION_TOOL_TYPE_MAP.get(key)
    if normalized is None:
        logger.warning("unknown Fusion tool type %r; mapping to 'unknown'", raw_type)
        return "unknown"
    return normalized


def _is_fusion_holder_type(raw_type: str) -> bool:
    """Return True for Fusion toolholder entries (not cutting tools).

    TODO: holders may later get their own table for reach / gauge-length checks.
    """
    return str(raw_type).strip().lower() == "holder"


def _library_name_from_path(path: Path) -> str:
    """Stable library prefix for globally unique tool_id values."""
    stem = path.stem.strip()
    cleaned = "".join(c if c.isalnum() else "_" for c in stem)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "fusion_library"


def _fusion_unit(raw_unit: str | None) -> str:
    unit = str(raw_unit or "millimeters").strip().lower()
    if unit in ("inches", "inch", "in"):
        return "inches"
    if unit in ("millimeters", "millimeter", "mm"):
        return "millimeters"
    logger.warning("unknown Fusion tool unit %r; treating values as millimeters", raw_unit)
    return "millimeters"


def _length_to_mm(value: Any, unit: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if unit == "inches":
        return numeric * _INCH_TO_MM
    return numeric


def _positive_length_to_mm(value: Any, unit: str) -> float | None:
    converted = _length_to_mm(value, unit)
    if converted is None:
        return None
    if converted <= 0.0:
        return None
    return converted


def _preset_material(raw_material: Any) -> str | None:
    if isinstance(raw_material, Mapping):
        category = raw_material.get("category")
        return str(category).strip().lower() if category is not None else None
    if raw_material is None:
        return None
    return str(raw_material).strip().lower()


def _preset_matches_material(preset_material: str | None, material: str | None) -> bool:
    if material is None:
        return True
    if preset_material is None:
        return True
    if preset_material == "all":
        return True
    return preset_material == material.strip().lower()


def _parse_fusion_preset(raw: Mapping[str, Any], unit: str) -> ToolPreset:
    use_feed_per_rev = raw.get("use-feed-per-revolution")
    return ToolPreset(
        preset_name=str(raw.get("name", "")),
        preset_material=_preset_material(raw.get("material")),
        spindle_rpm=float(raw["n"]) if raw.get("n") is not None else None,
        feed_mm_per_min=_positive_length_to_mm(raw.get("v_f"), unit),
        plunge_mm_per_min=_positive_length_to_mm(raw.get("v_f_plunge"), unit),
        retract_mm_per_min=_positive_length_to_mm(raw.get("v_f_retract"), unit),
        stepdown_mm=_positive_length_to_mm(raw.get("stepdown"), unit),
        stepover_mm=_positive_length_to_mm(raw.get("stepover"), unit),
        coolant=str(raw["tool-coolant"]) if raw.get("tool-coolant") is not None else None,
        chip_load=float(raw["f_z"]) if raw.get("f_z") is not None else None,
        surface_speed=float(raw["v_c"]) if raw.get("v_c") is not None else None,
        feed_per_rev_mm=_positive_length_to_mm(raw.get("f_n"), unit),
        feed_per_rev_retract_mm=_positive_length_to_mm(raw.get("f_n_retract"), unit),
        use_feed_per_revolution=(
            bool(use_feed_per_rev) if use_feed_per_rev is not None else None
        ),
    )


def _parse_fusion_tool(
    raw: Mapping[str, Any],
    *,
    library_name: str,
    material: str | None = None,
) -> Tool | None:
    raw_type = str(raw.get("type", ""))
    if _is_fusion_holder_type(raw_type):
        return None

    guid = raw.get("guid")
    if not guid:
        logger.warning("skipping tool without guid in library %s", library_name)
        return None

    unit = _fusion_unit(raw.get("unit"))
    geometry = raw.get("geometry") or {}
    diameter_mm = _positive_length_to_mm(geometry.get("DC"), unit)
    if diameter_mm is None:
        logger.warning(
            "tool guid=%s in library %s missing geometry.DC; skipping",
            guid,
            library_name,
        )
        return None

    max_depth_mm = _positive_length_to_mm(geometry.get("LB"), unit)
    if geometry.get("LB") is None:
        logger.warning(
            "tool guid=%s in library %s missing geometry.LB (max_depth)",
            guid,
            library_name,
        )

    raw_presets = (raw.get("start-values") or {}).get("presets") or []
    presets = [
        preset
        for item in raw_presets
        if isinstance(item, Mapping)
        for preset in [_parse_fusion_preset(item, unit)]
        if _preset_matches_material(preset.preset_material, material)
    ]

    product_id = raw.get("product-id")
    tool_catalog_product_id = raw.get("tool-catalog-product-id")

    return Tool(
        tool_id=f"{library_name}::{guid}",
        tool_type=normalize_tool_type(str(raw.get("type", ""))),
        diameter_mm=diameter_mm,
        flute_length_mm=_positive_length_to_mm(geometry.get("LCF"), unit),
        max_depth_mm=max_depth_mm,
        source=f"fusion_library:{library_name}",
        name=str(raw["description"]) if raw.get("description") is not None else None,
        corner_radius_mm=_length_to_mm(geometry.get("RE"), unit),
        taper_angle_deg=float(geometry["TA"]) if geometry.get("TA") is not None else None,
        point_angle_deg=float(geometry["SIG"]) if geometry.get("SIG") is not None else None,
        flute_count=int(geometry["NOF"]) if geometry.get("NOF") is not None else None,
        shank_diameter_mm=_positive_length_to_mm(geometry.get("SFDM"), unit),
        overall_length_mm=_positive_length_to_mm(geometry.get("OAL"), unit),
        tool_material=str(raw["BMC"]) if raw.get("BMC") is not None else None,
        vendor=str(raw["vendor"]) if raw.get("vendor") is not None else None,
        product_id=str(product_id) if product_id not in (None, "") else None,
        product_link=str(raw["product-link"]) if raw.get("product-link") not in (None, "") else None,
        tool_catalog_product_id=(
            str(tool_catalog_product_id) if tool_catalog_product_id not in (None, "") else None
        ),
        source_unit=unit,
        guid=str(guid),
        presets=presets,
    )


def load_tool_library_payload(
    payload: Mapping[str, Any],
    *,
    library_name: str,
    source_label: str | None = None,
    material: str | None = None,
) -> list[Tool]:
    """Load one Fusion/Autodesk tool-library JSON payload into Tool models."""
    label = source_label or library_name

    if not isinstance(payload, Mapping):
        raise ValueError(f"tool library root must be a mapping: {label}")

    raw_tools = payload.get("data")
    if not isinstance(raw_tools, list):
        raise ValueError(f"tool library missing data[] list: {label}")

    unit_counts: dict[str, int] = {}
    unknown_types: set[str] = set()
    tools: list[Tool] = []
    holders_skipped = 0

    for raw in raw_tools:
        if not isinstance(raw, Mapping):
            continue

        raw_type = str(raw.get("type", ""))
        if _is_fusion_holder_type(raw_type):
            holders_skipped += 1
            continue

        unit = _fusion_unit(raw.get("unit"))
        unit_counts[unit] = unit_counts.get(unit, 0) + 1

        normalized = normalize_tool_type(raw_type)
        if normalized == "unknown":
            unknown_types.add(raw_type)

        tool = _parse_fusion_tool(raw, library_name=library_name, material=material)
        if tool is not None:
            tools.append(tool)

    if holders_skipped:
        logger.info(
            "skipped %d holder entries (not cutting tools) in %s",
            holders_skipped,
            label,
        )

    unit_summary = ", ".join(f"{k}={v}" for k, v in sorted(unit_counts.items()))
    logger.info(
        "loaded %d tools from %s (%s)",
        len(tools),
        label,
        unit_summary or "no tools",
    )
    if unknown_types:
        logger.warning(
            "unknown tool types in %s: %s",
            label,
            sorted(unknown_types),
        )

    return tools


def load_tool_library(
    path: str | Path,
    *,
    material: str | None = None,
) -> list[Tool]:
    """Load one Fusion/Autodesk tool-library JSON file into Tool models."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return load_tool_library_payload(
        payload,
        library_name=_library_name_from_path(path),
        source_label=path.name,
        material=material,
    )


def load_enabled_library_payloads(
    libraries: Sequence[tuple[str, Mapping[str, Any], str | None]],
    material: str | None = None,
) -> list[Tool]:
    """Load and merge multiple Fusion library payloads; deduplicate by guid (first wins)."""
    merged: dict[str, Tool] = {}
    for library_name, payload, source_label in libraries:
        for tool in load_tool_library_payload(
            payload,
            library_name=library_name,
            source_label=source_label,
            material=material,
        ):
            if tool.guid is None:
                continue
            if tool.guid in merged:
                logger.debug(
                    "duplicate guid %s from %s; keeping first occurrence",
                    tool.guid,
                    tool.source,
                )
                continue
            merged[tool.guid] = tool
    return list(merged.values())


def load_enabled_libraries(
    paths: Sequence[str | Path],
    material: str | None = None,
) -> list[Tool]:
    """Load and merge multiple Fusion libraries; deduplicate by guid (first wins)."""
    libraries: list[tuple[str, Mapping[str, Any], str | None]] = []
    for path in paths:
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, Mapping):
            raise ValueError(f"tool library root must be a mapping: {path}")
        libraries.append((_library_name_from_path(path), payload, path.name))
    return load_enabled_library_payloads(libraries, material=material)


def load_library_directory(
    dir_path: str | Path,
    *,
    material: str | None = None,
) -> list[Tool]:
    """Load all Fusion JSON tool libraries (*.json, *.tools) from a directory.

    Deduplicates by guid (first file wins). Skips unsupported *.hsmlib files
    with a warning (re-export as JSON from Fusion/Toolpath).
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"tool library directory not found: {dir_path}")

    for hsmlib in sorted(dir_path.glob("*.hsmlib")):
        logger.warning(
            "skipping unsupported .hsmlib library %s; "
            "re-export as JSON from Fusion/Toolpath",
            hsmlib.name,
        )

    library_paths = sorted(
        [*dir_path.glob("*.json"), *dir_path.glob("*.tools")],
        key=lambda p: p.name.lower(),
    )
    if not library_paths:
        logger.info("no tool libraries found in %s", dir_path)
        return []

    tools = load_enabled_libraries(library_paths, material=material)
    logger.info(
        "loaded %d tools from %d library file(s) in %s (after guid dedup)",
        len(tools),
        len(library_paths),
        dir_path.name,
    )
    return tools


def load_default_libraries(
    material: str | None = None,
    *,
    dir_path: str | Path | None = None,
) -> list[Tool]:
    """Load bundled repo tool libraries from ``tool_libraries/`` by default."""
    resolved = Path(dir_path) if dir_path is not None else DEFAULT_TOOL_LIBRARIES_DIR
    return load_library_directory(resolved, material=material)


def load_tools_from_supabase(
    material: str | None = None,
    tool_types: list[str] | None = None,
    source_libraries: list[str] | None = None,
    *,
    client: Any | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[Tool]:
    """Load normalized tools from Supabase ``tools`` table.

    Optional filters: ``material`` (preset filtering), ``tool_types``,
    ``source_libraries``. Pass ``rows`` to bypass the network (tests).
    """
    from tools.tool_store import (
        fetch_tool_rows,
        filter_tool_presets_by_material,
        row_to_tool,
    )

    db_rows = fetch_tool_rows(
        client,
        tool_types=tool_types,
        source_libraries=source_libraries,
        rows=rows,
    )
    if not db_rows:
        logger.info("no tools returned from Supabase")
        return []

    tools = [row_to_tool(row) for row in db_rows]
    if material is not None:
        tools = [filter_tool_presets_by_material(tool, material) for tool in tools]

    logger.info(
        "loaded %d tools from Supabase (material=%s, tool_types=%s, source_libraries=%s)",
        len(tools),
        material,
        tool_types,
        source_libraries,
    )
    return tools


def default_tool_library() -> list[Tool]:
    """Hardcoded v0 tool catalog (6-8 entries)."""
    return [
        Tool(
            tool_id="T01",
            tool_type="drill",
            diameter_mm=3.2,
            flute_length_mm=25.0,
            max_depth_mm=30.0,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T02",
            tool_type="drill",
            diameter_mm=6.5,
            flute_length_mm=35.0,
            max_depth_mm=40.0,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T03",
            tool_type="drill",
            diameter_mm=8.0,
            flute_length_mm=40.0,
            max_depth_mm=45.0,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T04",
            tool_type="endmill",
            diameter_mm=3.0,
            flute_length_mm=12.0,
            max_depth_mm=None,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T05",
            tool_type="endmill",
            diameter_mm=6.0,
            flute_length_mm=19.0,
            max_depth_mm=None,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T06",
            tool_type="endmill",
            diameter_mm=10.0,
            flute_length_mm=25.0,
            max_depth_mm=None,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T07",
            tool_type="endmill",
            diameter_mm=12.0,
            flute_length_mm=30.0,
            max_depth_mm=None,
            source="hardcoded_v0",
        ),
        Tool(
            tool_id="T08",
            tool_type="endmill",
            diameter_mm=16.0,
            flute_length_mm=32.0,
            max_depth_mm=None,
            source="hardcoded_v0",
        ),
    ]


def _round_tuple3(values: Sequence[float]) -> tuple[float, float, float]:
    return (round(float(values[0]), 6), round(float(values[1]), 6), round(float(values[2]), 6))


def _envelope_corners(
    step_or_extents: str | Path | Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Resolve part AABB min/max from a STEP path or precomputed envelope dict."""
    if isinstance(step_or_extents, Mapping):
        data = step_or_extents
        mn = data.get("min") or data.get("min_corner") or data.get("bbox_min")
        mx = data.get("max") or data.get("max_corner") or data.get("bbox_max")
        if mn is None or mx is None:
            raise ValueError(
                "envelope mapping must include min/max (or min_corner/max_corner)"
            )
        return _round_tuple3(mn), _round_tuple3(mx)

    step_path = Path(step_or_extents)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    from brep.feature_params import load_step_faces, require_occ
    from brep.brep_extents import feature_aabb

    require_occ()
    occ_faces = load_step_faces(step_path)
    occ_map = {i: face for i, face in enumerate(occ_faces)}
    aabb = feature_aabb(occ_map, list(range(len(occ_faces))))
    return _round_tuple3(aabb.min_corner), _round_tuple3(aabb.max_corner)


def build_stock(
    step_or_extents: str | Path | Mapping[str, Any],
    oversize_mm: float = 2.0,
) -> Stock:
    """BBox stock from part envelope with uniform oversize offset."""
    bbox_min, bbox_max = _envelope_corners(step_or_extents)
    offset = float(oversize_mm)
    return Stock(
        stock_type="bbox",
        bbox_min=(
            bbox_min[0] - offset,
            bbox_min[1] - offset,
            bbox_min[2] - offset,
        ),
        bbox_max=(
            bbox_max[0] + offset,
            bbox_max[1] + offset,
            bbox_max[2] + offset,
        ),
        source="brep_extents+offset",
    )


def load_feature_graph(path: str | Path) -> dict[str, Any]:
    """Load a feature_graph_cascade.json artifact."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"feature graph root must be a mapping: {path}")
    return data


def _relative_ref(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(p)


def _pocket_seed_face(params: Mapping[str, Any]) -> int | None:
    """Best-effort seed face for descriptor pockets.by_seed_face lookup."""
    for key in ("seed_face_index", "floor_face_index"):
        raw = params.get(key)
        if raw is not None:
            return int(raw)

    floors = params.get("floor_face_indices")
    if isinstance(floors, list) and floors:
        return int(floors[0])

    face_indices = params.get("face_indices")
    if isinstance(face_indices, list) and face_indices:
        return int(face_indices[0])

    return None


def descriptor_access_for_node(
    node: Mapping[str, Any],
    resolved: ResolvedSetup,
) -> str | None:
    """Descriptor-sourced access for one graph node, or None if not covered."""
    class_name = str(node.get("class_name", ""))
    if class_name not in DESCRIPTOR_POCKET_CLASSES:
        return None

    params = node.get("params") or {}
    seed = _pocket_seed_face(params)
    if seed is not None:
        return pocket_access_for_seed(resolved, seed)

    return pocket_access_for_seed(resolved, -1)


def _opening_axis_from_cascade(graph: Mapping[str, Any]) -> tuple[float, float, float] | None:
    counts: dict[tuple[float, float, float], int] = {}
    for node in graph.get("nodes", []):
        params = node.get("params") or {}
        raw = params.get("opening_axis")
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        key = tuple(round(float(v), 6) for v in raw)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def resolve_opening_axis_vector_for_planning(
    resolved: ResolvedSetup,
    graph: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Resolve a unit opening-axis vector from the descriptor (fail-loud).

    Descriptor explicit vectors are authoritative. Auto mode may read
    approach_frame.z or pocket opening axes from the cascade graph; never
    a hardcoded principal axis.
    """
    spec = resolved.opening_axis
    setup_id = resolved.setup_id

    def _unit_or_raise(raw: Sequence[float], *, source: str) -> tuple[float, float, float]:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape != (3,):
            raise OpeningAxisResolutionError(
                f"setup {setup_id!r}: opening_axis from {source} must be length-3, "
                f"got {list(raw)!r}"
            )
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            raise OpeningAxisResolutionError(
                f"setup {setup_id!r}: opening_axis from {source} is zero: {list(raw)!r}"
            )
        unit = arr / norm
        return (float(unit[0]), float(unit[1]), float(unit[2]))

    if spec.mode == "explicit":
        if spec.vector is None:
            raise OpeningAxisResolutionError(
                f"setup {setup_id!r}: opening_axis mode 'explicit' but vector is missing"
            )
        return _unit_or_raise(spec.vector, source="descriptor")

    frame = graph.get("approach_frame") or {}
    z = frame.get("z")
    if isinstance(z, (list, tuple)) and len(z) == 3:
        return _unit_or_raise(z, source="approach_frame")

    cascade_vec = _opening_axis_from_cascade(graph)
    if cascade_vec is not None:
        return _unit_or_raise(cascade_vec, source="cascade pocket opening_axis")

    if spec.vector is not None:
        return _unit_or_raise(spec.vector, source="descriptor")

    raise OpeningAxisResolutionError(
        f"setup {setup_id!r}: opening_axis could not be resolved "
        f"(mode={spec.mode!r}); descriptor and graph lack a usable vector"
    )


def opening_axis_label(
    resolved: ResolvedSetup,
    graph: Mapping[str, Any],
) -> str:
    """Discrete opening axis label from descriptor spec + cascade fallback for auto mode."""
    return vector_to_opening_axis_label(
        resolve_opening_axis_vector_for_planning(resolved, graph)
    )


def build_setup_context(
    resolved: ResolvedSetup,
    graph: Mapping[str, Any],
    *,
    source_step_file: str | None = None,
) -> SetupContext:
    """Assemble one SetupContext from resolved descriptor + cascade cross-check."""
    pocket_access: dict[str, str] = {}

    for node in graph.get("nodes", []):
        class_name = str(node.get("class_name", ""))
        if class_name not in POCKET_ACCESS_CLASSES:
            continue

        feature_id = str(node["feature_id"])
        params = node.get("params") or {}
        descriptor_access = descriptor_access_for_node(node, resolved)
        cascade_access = params.get("access")
        pocket_access[feature_id] = resolve_pocket_access(
            feature_id,
            descriptor_access,
            cascade_access if cascade_access is not None else None,
        )

    axis_vector = resolve_opening_axis_vector_for_planning(resolved, graph)

    # Provenance: an explicit descriptor vector is authoritative by definition.
    # An auto-mode axis is authoritative only if the cascade could resolve it
    # from geometry (approach_frame.opening_axis_determined). Absent => True, so
    # graphs predating this field keep planning unchanged.
    if resolved.opening_axis.mode == "explicit":
        axis_determined = True
    else:
        frame = graph.get("approach_frame") or {}
        axis_determined = bool(frame.get("opening_axis_determined", True))

    orientation = getattr(resolved, "orientation", None)
    return SetupContext(
        setup_id=resolved.setup_id,
        opening_axis=vector_to_opening_axis_label(axis_vector),
        opening_axis_vector=axis_vector,
        opening_axis_determined=axis_determined,
        machining_side=resolved.machining_side,
        orientation=orientation,
        orientation_provisional=orientation is not None,
        pocket_access=pocket_access,
        scope=SetupScopeSpec.model_validate(resolved.scope.to_dict()),
        engrave=[e.to_dict() for e in getattr(resolved, "engrave", ())],
        fixture=None,
        source_step_file=source_step_file or resolved.part_step,
    )


def load_setup_descriptor_for_planning(
    *,
    setups_source: Literal["authored", "generated"],
    setup_yaml_path: str | Path,
    part_id: str,
    generated_descriptor: PartSetupDescriptor | None = None,
    generated_descriptor_path: str | Path | None = None,
    feature_graph_path: str | Path | None = None,
) -> tuple[PartSetupDescriptor, str]:
    """Load the setup descriptor for planner context assembly.

    Returns ``(descriptor, ref_path)`` where ``ref_path`` is the resolved source.
    """
    if setups_source == "authored":
        path = Path(setup_yaml_path)
        return load_setup_descriptor(path), str(path)

    if generated_descriptor is not None:
        ref = str(generated_descriptor_path or "generated:in_memory")
        return generated_descriptor, ref

    candidates: list[Path] = []
    if generated_descriptor_path is not None:
        candidates.append(Path(generated_descriptor_path))

    from cascade.setup_generation import resolve_generated_descriptor_path

    family_ids = {part_id}
    if "_" in part_id:
        family_ids.add(part_id.rsplit("_", 1)[0])
    for family_id in family_ids:
        candidates.append(resolve_generated_descriptor_path(family_id))

    if feature_graph_path is not None:
        export_dir = Path(feature_graph_path).parent
        candidates.append(export_dir / "setup_descriptor.yaml")

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return load_setup_descriptor(resolved), str(resolved)

    tried = ", ".join(str(p) for p in candidates)
    raise SetupDescriptorError(
        f"setups_source='generated' but no generated descriptor found (tried {tried})"
    )


def build_context_v0(
    step_or_extents: str | Path | Mapping[str, Any],
    setup_yaml_path: str | Path,
    feature_graph_path: str | Path,
    oversize_mm: float = 2.0,
    *,
    setup_id: str | None = None,
    step_path: str | Path | None = None,
    material: str | None = None,
    tool_library_paths: list[str | Path] | None = None,
    tool_library_material: str | None = None,
    tool_source: Literal["hardcoded", "directory", "supabase"] = "hardcoded",
    tool_library_dir: str | Path | None = None,
    setups_source: Literal["authored", "generated"] = "generated",
    generated_descriptor: PartSetupDescriptor | None = None,
    generated_descriptor_path: str | Path | None = None,
) -> MachiningContext:
    """Assemble a v0 MachiningContext from existing repo artifacts.

    Tool catalog precedence (first match wins):

    1. ``tool_library_paths`` - explicit Fusion library file(s) on disk
    2. ``tool_source``:
       - ``"supabase"`` - normalized rows from Supabase ``tools`` table
       - ``"directory"`` - all ``*.json`` / ``*.tools`` in ``tool_library_dir``
         (default: bundled ``tool_libraries/``)
       - ``"hardcoded"`` (default) - built-in v0 catalog (8 tools)
    """
    setup_yaml_path = Path(setup_yaml_path)
    feature_graph_path = Path(feature_graph_path)
    preset_material = material if material is not None else tool_library_material

    graph = load_feature_graph(feature_graph_path)

    if isinstance(step_or_extents, Mapping):
        family_part_id = str(graph.get("part_id") or "unknown")
    else:
        family_part_id = str(graph.get("part_id") or Path(step_or_extents).stem)

    descriptor, descriptor_ref = load_setup_descriptor_for_planning(
        setups_source=setups_source,
        setup_yaml_path=setup_yaml_path,
        part_id=family_part_id,
        generated_descriptor=generated_descriptor,
        generated_descriptor_path=generated_descriptor_path,
        feature_graph_path=feature_graph_path,
    )

    resolved_step = step_path
    if resolved_step is None and not isinstance(step_or_extents, Mapping):
        candidate = Path(step_or_extents)
        if candidate.suffix.lower() in (".stp", ".step", ".stp copy"):
            resolved_step = candidate

    resolved = resolve_setup_entry(
        descriptor,
        setup_id=setup_id,
        step_path=resolved_step,
    )

    if isinstance(step_or_extents, Mapping):
        part_id = str(graph.get("part_id") or descriptor.part_id)
    else:
        part_id = Path(step_or_extents).name

    stock = build_stock(step_or_extents, oversize_mm=oversize_mm)
    setup = build_setup_context(
        resolved,
        graph,
        source_step_file=(
            str(resolved.part_step)
            if resolved.part_step is not None
            else (Path(resolved_step).name if resolved_step is not None else None)
        ),
    )

    if tool_library_paths:
        tools = load_enabled_libraries(tool_library_paths, material=preset_material)
    elif tool_source == "supabase":
        tools = load_tools_from_supabase(
            material=preset_material,
            tool_types=list(V0_PLANNER_TOOL_TYPES),
        )
    elif tool_source == "directory":
        resolved_dir = (
            Path(tool_library_dir)
            if tool_library_dir is not None
            else DEFAULT_TOOL_LIBRARIES_DIR
        )
        tools = load_library_directory(resolved_dir, material=preset_material)
    else:
        tools = default_tool_library()

    metadata: dict[str, Any] = {
        "generator": "machining_context.build_context_v0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "setup_descriptor_ref": _relative_ref(descriptor_ref),
        "setups_source": setups_source,
        "oversize_mm": oversize_mm,
    }
    if material is not None:
        metadata["material"] = material
    if tool_library_paths:
        metadata["tool_library_paths"] = [str(Path(p)) for p in tool_library_paths]
        if preset_material is not None:
            metadata["tool_library_material"] = preset_material
    elif tool_source == "supabase":
        metadata["tool_library_source"] = "supabase"
        metadata["tool_library_tool_types"] = list(V0_PLANNER_TOOL_TYPES)
        if preset_material is not None:
            metadata["tool_library_material"] = preset_material
    elif tool_source == "directory":
        resolved_dir = (
            Path(tool_library_dir)
            if tool_library_dir is not None
            else DEFAULT_TOOL_LIBRARIES_DIR
        )
        metadata["tool_library_dir"] = str(resolved_dir)
        if preset_material is not None:
            metadata["tool_library_material"] = preset_material

    return MachiningContext(
        part_id=part_id,
        feature_graph_ref=_relative_ref(feature_graph_path),
        stock=stock,
        setups=[setup],
        tools=tools,
        material=material,
        completed_operations=[],
        remaining_material=None,
        machine_limits=None,
        metadata=metadata,
    )


def load_machining_context(path: str | Path) -> MachiningContext:
    """Load and validate a machining context JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return MachiningContext.model_validate(data)


def write_machining_context(path: str | Path, context: MachiningContext) -> None:
    """Write a validated machining context to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(context.model_dump(mode="json"), fh, indent=2)
        fh.write("\n")


def export_json_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Export MachiningContext JSON Schema (Pydantic v2) to disk and return the dict."""
    schema = MachiningContext.model_json_schema()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2)
            fh.write("\n")
    return schema


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build machining context v0 example.")
    parser.add_argument(
        "--step",
        type=Path,
        default=REPO_ROOT / "fixtures/step/96260B_rear.stp",
    )
    parser.add_argument(
        "--setup-yaml",
        type=Path,
        default=REPO_ROOT / "eval" / "gt" / "96260B_setup.yaml",
    )
    parser.add_argument(
        "--feature-graph",
        type=Path,
        default=REPO_ROOT / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "examples" / "machining_context_96260B.json",
    )
    parser.add_argument("--oversize-mm", type=float, default=2.0)
    args = parser.parse_args()

    ctx = build_context_v0(
        args.step,
        args.setup_yaml,
        args.feature_graph,
        oversize_mm=args.oversize_mm,
        step_path=args.step,
    )
    write_machining_context(args.out, ctx)
    print(f"Wrote {args.out}")
