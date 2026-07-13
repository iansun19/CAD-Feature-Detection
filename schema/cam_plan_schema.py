# DESIGN NOTES
# -----------
# Native output = the "bracket" CAM plan format (see bracket_pockets_plan_native.json):
#   - Operations nest under their setup as setups[].operations[]; there is no flat
#     top-level operations[] in the serialized document.
#   - Each operation carries an inline tool blob {id, type, dia, flutes} resolved from
#     the (hidden) top-level tool catalog.
#   - Cutting parameters serialize under bracket names (rpm/feed/plunge/doc/stepover)
#     via serialization aliases; internal code keeps using our field names.
#
# NOTHING IS DELETED. Every field the planner populates today stays on the model and
# stays populated. The bracket format merely HIDES a set of internal fields from
# serialization -- collected in BRACKET_HIDDEN_FIELDS so the set is auditable in one
# place. The two serialization modes:
#   - Lossless (default model_dump / write_cam_plan / load_cam_plan): every field,
#     using our field names. This is what round-trips (persist -> reload) and what
#     internal validation and the example files rely on.
#   - Bracket (to_bracket_dict / write_bracket_plan, model_dump(context={"bracket": True})):
#     drops BRACKET_HIDDEN_FIELDS and applies the bracket serialization aliases. This
#     is the external artifact emitted by generate_operation_plan.
# Hiding is context-gated (a @model_serializer keyed on the serialization context),
# NOT Field(exclude=True): exclude=True would silently drop required fields on the
# default model_dump and break the write_cam_plan -> load_cam_plan round-trip and the
# planner's in-process CamPlan.model_validate(cam_plan.model_dump()) self-checks.
#
# Placeholders / deferred (fields present or reserved, not populated yet):
#   - CamPlan.stock            -- TODO: requires an ingest/stock stage.
#   - CamPlan.remaining_material -- stock after partial ops (shape TBD).
#   - Setup.wcs                -- TODO: requires a WCS/setup-frame stage.
#   - Setup.fixture            -- fixture id/name when a fixture library exists.
#   - Operation.rationale      -- TODO: requires a rationale stage.
#   - Operation.geometry_ref   -- TODO: requires toolpath generation.
#   - MachiningParameters.woc/retract/cycle/depth -- TODO: requires toolpath generation.
#   - MachiningParameters.coolant -- shop policy not wired.
#   - ToolRef.flute_length_mm, max_depth_mm -- optional shop-library enrichments.
#   - InlineTool.flutes        -- nothing populates it; null is correct.
#
# Added beyond the task brief (flagged here):
#   - CamPlan @model_validator -- cross-field FK / sequence checks for emitters,
#     inline-tool resolution, and provenance derivation.
#   - load_cam_plan / write_cam_plan / to_bracket_dict / write_bracket_plan /
#     export_json_schema helpers.
#
"""cam_plan_schema.py - structured CAM plan output contract (Section 3).

The planner emits a CamPlan document referencing a feature_graph_cascade.json,
ordered operations nested under their setups, a tool catalog, and machining
parameters with provenance.

Data lineage:
  part_id, feature_graph_ref     <- ingest / cascade artifacts
  Setup.opening_axis, Operation.access <- setup_descriptor YAML (not inferred)
  feature_refs                   <- feature graph node feature_id (as string)
  tools, parameters              <- shop library + handbook defaults (planner)
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from planning.operation_bank import is_valid_operation

SCHEMA_VERSION = "1.0"
REPO_ROOT = Path(__file__).resolve().parent


# --- Fields hidden from the bracket serialization, per model ----------------------
# Every listed field STAYS on the model and stays populated by the planner; it is
# merely omitted from the bracket output (to_bracket_dict / context={"bracket": True}).
# The default (lossless) model_dump keeps them, so persist -> reload round-trips.
BRACKET_HIDDEN_FIELDS: dict[str, frozenset[str]] = {
    "CamPlan": frozenset(
        {
            "feature_graph_ref",
            "tools",  # dedup catalog; inline Operation.tool carries what output needs
            "remaining_material",
            "metadata",  # incl. planner_stats, sequence_score, split_note -> via provenance
        }
    ),
    "Setup": frozenset(
        {
            "orientation",
            "orientation_provisional",
        }
    ),
    "Operation": frozenset(
        {
            "sequence_index",  # nesting + list order carry it
            "setup_id",  # nesting carries it
            "tool_id",  # inline Operation.tool carries it
            "depends_on",
            "access",
            "attributes",
        }
    ),
    "MachiningParameters": frozenset(
        {
            "param_source",
            "coolant",
        }
    ),
}


class _BracketModel(BaseModel):
    """Base model carrying the context-gated bracket serializer.

    Default serialization is lossless (all fields, our field names). When dumped with
    context={"bracket": True} (see to_bracket_dict), the fields in BRACKET_HIDDEN_FIELDS
    for this model are dropped. populate_by_name lets internal code keep constructing
    with our field names even where serialization aliases are declared.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_serializer(mode="wrap")
    def _bracket_serialize(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        data = handler(self)
        if isinstance(data, dict) and info.context and info.context.get("bracket"):
            for name in BRACKET_HIDDEN_FIELDS.get(type(self).__name__, frozenset()):
                data.pop(name, None)
        return data


class PocketAccess(StrEnum):
    """Filleted-pocket machining access label from setup descriptor; not B-rep inferred."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class MachiningParameters(_BracketModel):
    """Cutting parameters for one operation.

    Sourced from handbook defaults, shop templates, or conservative heuristics.
    param_source records provenance so values can later be swapped for learned ones.
    The bracket names (rpm/feed/plunge/doc/stepover) are serialization aliases; internal
    code keeps using the *_mm / *_per_min field names.
    """

    spindle_rpm: float | None = Field(default=None, serialization_alias="rpm")
    feed_mm_per_min: float | None = Field(default=None, serialization_alias="feed")
    plunge_mm_per_min: float | None = Field(default=None, serialization_alias="plunge")
    stepdown_mm: float | None = Field(default=None, serialization_alias="doc")
    stepover_mm: float | None = Field(default=None, serialization_alias="stepover")
    # New bracket params -- nothing populates them yet; null is correct.
    woc: float | None = None  # TODO: requires toolpath generation
    retract: float | None = None  # TODO: requires toolpath generation
    cycle: str | None = None  # TODO: requires toolpath generation
    depth: float | None = None  # TODO: requires toolpath generation
    coolant: str | None = None  # hidden in bracket; shop policy not wired
    param_source: str = Field(
        ...,
        description='Provenance tag, e.g. "handbook_default", "shop_default", '
        '"conservative_heuristic". Hidden in bracket output.',
    )


class ToolRef(_BracketModel):
    """A tool in the deduplicated top-level catalog, referenced by Operation.tool_id.

    The catalog itself is hidden from the bracket output (see BRACKET_HIDDEN_FIELDS);
    the inline Operation.tool blob carries what the bracket consumer needs. diameter_mm
    is required; the remaining enrichments (corner_radius_mm, flute_length_mm,
    max_depth_mm, source) are hidden by construction because the catalog is hidden.
    """

    tool_id: str
    tool_type: str = Field(
        ...,
        description='Tool category, e.g. "drill", "endmill", "ball_endmill".',
    )
    diameter_mm: float = Field(..., gt=0.0)
    flute_length_mm: float | None = Field(default=None, gt=0.0)
    max_depth_mm: float | None = Field(default=None, gt=0.0)
    corner_radius_mm: float | None = Field(
        default=None,
        ge=0.0,
        description="Corner radius (RE) for bullnose/chamfer tools; omitted for plain endmills.",
    )
    source: str = Field(
        ...,
        description='Catalog origin, e.g. "shop_library", "hardcoded_v0".',
    )


class InlineTool(_BracketModel):
    """Inline per-operation tool blob for the bracket output.

    Populated by CamPlan by resolving Operation.tool_id against the top-level catalog;
    kept in sync by the CamPlan model validator. flutes has no source yet -> null.
    """

    id: str
    type: str
    dia: float
    flutes: int | None = None


class Stock(_BracketModel):
    """Work stock the plan is cut from. TODO: requires an ingest/stock stage."""

    type: str | None = None
    dims: list[float] | None = None
    origin: list[float] | None = None
    step_file: str | None = None


class WorkCoordinateSystem(_BracketModel):
    """Setup work coordinate system. TODO: requires a WCS/setup-frame stage."""

    origin: list[float] | None = None
    z_axis: list[float] | None = None
    x_axis: list[float] | None = None


class Provenance(_BracketModel):
    """Plan provenance. planner keeps our real planner string (from metadata.generator)."""

    planner: str | None = None
    model: str | None = None


class Operation(_BracketModel):
    """One executable machining operation in program order.

    feature_refs point at feature graph node ids (feature_id serialized as string).
    depends_on lists op_ids that must complete before this op -- precedence edges
    preserved even though execution order is already topologically sorted.
    access is required for pocket-class ops (from setup descriptor); null otherwise.

    sequence_index and setup_id stay on the model (still used by internal logic) but
    are hidden from the bracket output -- nesting and list order carry them. tool_id
    stays too; the inline `tool` blob is derived from it against the catalog.
    """

    op_id: str
    sequence_index: int = Field(..., ge=0)
    feature_refs: list[str] = Field(
        ...,
        min_length=1,
        description="Feature graph feature_id values as strings.",
    )
    feature_type: str = Field(
        ...,
        description="Recognizer / Toolpath class, e.g. through_hole, filleted_pocket.",
    )
    setup_id: str
    operation: str = Field(
        ...,
        description=(
            "Canonical CAM operation from the flat operation bank "
            "(operation_bank.Operation), e.g. \"drill\", \"pocket\", \"contour_2d\"."
        ),
    )
    tool_id: str
    tool: InlineTool | None = Field(
        default=None,
        description="Inline tool blob resolved from tool_id against the catalog by CamPlan.",
    )
    parameters: MachiningParameters
    depends_on: list[str] = Field(default_factory=list)
    access: PocketAccess | None = Field(
        default=None,
        description="Pocket access label from setup descriptor; null for non-pockets.",
    )
    rationale: str | None = None  # TODO: requires a rationale stage
    geometry_ref: str | None = None  # TODO: requires toolpath generation
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Op-specific extras carried to the CAM consumer, e.g. engraving "
            '{"text", "depth_mm", "source"}. Empty for ops with no extras.'
        ),
    )

    @field_validator("operation")
    @classmethod
    def _operation_in_bank(cls, value: str) -> str:
        if not is_valid_operation(value):
            raise ValueError(
                f"operation {value!r} is not in the canonical operation bank "
                f"(operation_bank.Operation)"
            )
        return value


class Setup(_BracketModel):
    """One machining setup (workholding + approach direction) and its operations.

    opening_axis is a discrete 3-axis label (e.g. "+Z") derived from the setup
    descriptor YAML, not a free-form vector in the CAM plan. Operations that belong to
    this setup live in operations[] (the bracket output has no flat top-level list).
    """

    setup_id: str
    opening_axis: str = Field(
        ...,
        description='Discrete approach direction, e.g. "+Z", "-Y". From setup descriptor.',
    )
    orientation: str | None = Field(
        default=None,
        description=(
            "General cardinal setup orientation the setup was evaluated from, e.g. "
            '"+X". Extends the binary front/back convention to all six cardinals. '
            "PROVISIONAL (lateral ±X/±Y path, no real-part validation) when set to a "
            "non-opening-axis direction; None means the calibrated opening_axis path. "
            "Hidden in bracket output."
        ),
    )
    orientation_provisional: bool = Field(
        default=False,
        description="True when orientation is a provisional lateral-axis direction. "
        "Hidden in bracket output.",
    )
    fixture: str | None = Field(
        default=None,
        description="Fixture id or name when a fixture library exists; null for now.",
    )
    wcs: WorkCoordinateSystem | None = Field(
        default=None,
        description="Setup work coordinate system. TODO: requires a WCS/setup-frame stage.",
    )
    notes: str | None = None
    operations: list[Operation] = Field(
        default_factory=list,
        description="Operations belonging to this setup, in program order.",
    )


class CamPlan(_BracketModel):
    """Top-level structured CAM plan emitted by the planner.

    Execution order equals the setup order followed by each setup's operation list
    order; each Operation.sequence_index must match its position in that flattened
    order. tools[] is the deduplicated catalog referenced by Operation.tool_id (hidden
    from the bracket output; the inline Operation.tool blob carries what output needs).

    Construction accepts either the nested setups[].operations[] form or a flat
    operations=[...] keyword (distributed into setups by setup_id) so the planner's
    existing construction call keeps working unchanged.
    """

    schema_version: str = Field(default=SCHEMA_VERSION)
    part_id: str = Field(
        ...,
        validation_alias=AliasChoices("part_id", "source_part"),
        description="Part identifier or STEP filename the plan was generated from.",
    )
    feature_graph_ref: str = Field(
        ...,
        description="Path or id of the feature_graph_cascade.json used as input. "
        "Hidden in bracket output.",
    )
    stock: Stock | None = Field(
        default=None,
        description="Work stock. TODO: requires an ingest/stock stage.",
    )
    provenance: Provenance | None = Field(
        default=None,
        description="Plan provenance; derived from metadata.generator when not set.",
    )
    setups: list[Setup] = Field(..., min_length=1)
    tools: list[ToolRef] = Field(
        ...,
        min_length=1,
        description="Deduplicated tool catalog referenced by Operation.tool_id. "
        "Hidden in bracket output.",
    )
    remaining_material: dict[str, Any] | None = Field(
        default=None,
        description="Reserved for in-process stock state after partial machining. "
        "Hidden in bracket output.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form generator metadata (version, timestamp, planner_stats, "
        "sequence_score, split_note). Hidden in bracket output.",
    )

    @property
    def source_part(self) -> str:
        """Backward-compatible read alias for part_id (renamed field)."""
        return self.part_id

    @property
    def operations(self) -> list[Operation]:
        """Flattened operations across setups, in program order.

        This is the flat view internal logic reads; the serialized document nests
        operations under setups[].operations[] and has no top-level operations[].
        """
        return [op for setup in self.setups for op in setup.operations]

    @model_validator(mode="before")
    @classmethod
    def _nest_flat_operations(cls, data: Any) -> Any:
        """Distribute a flat operations=[...] input into its setups by setup_id.

        Keeps the planner's CamPlan(operations=..., setups=...) construction working
        against the nested model. No-op for the nested/loaded form (no top-level
        operations key). Raises if an operation names a setup_id absent from setups so
        nesting can never silently drop an operation.
        """
        if not isinstance(data, dict) or data.get("operations") is None:
            return data

        data = dict(data)
        flat = data.pop("operations")
        setups = list(data.get("setups") or [])

        def _sid(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                return obj.get("setup_id")
            return getattr(obj, "setup_id", None)

        grouped: dict[Any, list[Any]] = {}
        for op in flat:
            grouped.setdefault(_sid(op), []).append(op)

        assigned = 0
        new_setups: list[Any] = []
        for setup in setups:
            ops_for = grouped.get(_sid(setup), [])
            assigned += len(ops_for)
            if isinstance(setup, Setup):
                new_setups.append(setup.model_copy(update={"operations": ops_for}))
            elif isinstance(setup, Mapping):
                new_setups.append({**setup, "operations": ops_for})
            else:
                new_setups.append(setup)

        if assigned != len(flat):
            raise ValueError(
                "operations reference setup_id(s) not present in setups "
                "(nesting would drop them)"
            )
        data["setups"] = new_setups
        return data

    @model_validator(mode="after")
    def _validate_and_resolve(self) -> CamPlan:
        setup_ids = {s.setup_id for s in self.setups}
        tool_by_id = {t.tool_id: t for t in self.tools}
        ops = self.operations
        op_ids = {op.op_id for op in ops}

        for idx, op in enumerate(ops):
            if op.sequence_index != idx:
                raise ValueError(
                    f"operations[{idx}].sequence_index is {op.sequence_index}, "
                    f"expected {idx} (must match flattened setup/op order)"
                )
            if op.setup_id not in setup_ids:
                raise ValueError(
                    f"operation {op.op_id!r} references unknown setup_id {op.setup_id!r}"
                )
            if op.tool_id not in tool_by_id:
                raise ValueError(
                    f"operation {op.op_id!r} references unknown tool_id {op.tool_id!r}"
                )
            for dep in op.depends_on:
                if dep not in op_ids:
                    raise ValueError(
                        f"operation {op.op_id!r} depends_on unknown op_id {dep!r}"
                    )
                if dep == op.op_id:
                    raise ValueError(f"operation {op.op_id!r} must not depend on itself")

        seen_op_ids: set[str] = set()
        for op in ops:
            if op.op_id in seen_op_ids:
                raise ValueError(f"duplicate op_id {op.op_id!r}")
            seen_op_ids.add(op.op_id)

        seen_tool_ids: set[str] = set()
        for tool in self.tools:
            if tool.tool_id in seen_tool_ids:
                raise ValueError(f"duplicate tool_id {tool.tool_id!r}")
            seen_tool_ids.add(tool.tool_id)

        # Resolve the inline per-operation tool blob against the catalog (kept in sync
        # every validation pass).
        for op in ops:
            tool = tool_by_id[op.tool_id]
            op.tool = InlineTool(
                id=tool.tool_id,
                type=tool.tool_type,
                dia=tool.diameter_mm,
                flutes=None,
            )

        # Surface the planner string as provenance.planner without dropping metadata.
        if self.provenance is None:
            generator = (
                self.metadata.get("generator") if isinstance(self.metadata, dict) else None
            )
            self.provenance = Provenance(planner=generator)

        return self


def load_cam_plan(path: str | Path) -> CamPlan:
    """Load and validate a lossless CAM plan JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return CamPlan.model_validate(data)


def write_cam_plan(path: str | Path, plan: CamPlan) -> None:
    """Write a validated CAM plan to JSON in the lossless (round-trippable) form."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan.model_dump(mode="json"), fh, indent=2)
        fh.write("\n")


def to_bracket_dict(plan: CamPlan) -> dict[str, Any]:
    """Serialize a CAM plan to the external bracket format.

    Drops BRACKET_HIDDEN_FIELDS and applies the bracket serialization aliases. This is
    a lossy, terminal representation -- do NOT feed it back to load_cam_plan.
    """
    return plan.model_dump(mode="json", by_alias=True, context={"bracket": True})


def write_bracket_plan(path: str | Path, plan: CamPlan) -> None:
    """Write a CAM plan to JSON in the external bracket format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_bracket_dict(plan), fh, indent=2)
        fh.write("\n")


def export_json_schema(path: str | Path | None = None) -> dict[str, Any]:
    """Export CamPlan JSON Schema (Pydantic v2) to disk and return the dict."""
    schema = CamPlan.model_json_schema()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2)
            fh.write("\n")
    return schema


def example_cam_plan() -> CamPlan:
    """Minimal hand-authored example: one setup, drill + pocket rough."""
    return CamPlan(
        part_id="demo_bracket.step",
        feature_graph_ref="pipeline_out/demo_bracket/feature_graph_cascade.json",
        setups=[
            Setup(
                setup_id="primary",
                opening_axis="+Z",
                fixture=None,
                notes="Single 3-axis setup; example.",
            ),
        ],
        tools=[
            ToolRef(
                tool_id="T01",
                tool_type="drill",
                diameter_mm=3.2,
                flute_length_mm=25.0,
                max_depth_mm=30.0,
                source="hardcoded_v0",
            ),
            ToolRef(
                tool_id="T02",
                tool_type="endmill",
                diameter_mm=6.0,
                flute_length_mm=19.0,
                max_depth_mm=None,
                source="hardcoded_v0",
            ),
        ],
        operations=[
            Operation(
                op_id="OP010",
                sequence_index=0,
                feature_refs=["0"],
                feature_type="through_hole",
                setup_id="primary",
                operation="drill",
                tool_id="T01",
                parameters=MachiningParameters(
                    spindle_rpm=3000.0,
                    feed_mm_per_min=180.0,
                    plunge_mm_per_min=90.0,
                    stepdown_mm=None,
                    stepover_mm=None,
                    coolant=None,
                    param_source="handbook_default",
                ),
                depends_on=[],
                access=None,
            ),
            Operation(
                op_id="OP020",
                sequence_index=1,
                feature_refs=["1"],
                feature_type="filleted_pocket",
                setup_id="primary",
                operation="pocket",
                tool_id="T02",
                parameters=MachiningParameters(
                    spindle_rpm=8000.0,
                    feed_mm_per_min=1200.0,
                    plunge_mm_per_min=300.0,
                    stepdown_mm=2.0,
                    stepover_mm=3.0,
                    coolant=None,
                    param_source="conservative_heuristic",
                ),
                depends_on=["OP010"],
                access=PocketAccess.CLOSED,
            ),
        ],
        remaining_material=None,
        metadata={
            "generator": "cam_plan_schema.example",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "notes": "Hand-written contract example; not produced by a planner.",
        },
    )


if __name__ == "__main__":
    schema_path = REPO_ROOT / "cam_plan.schema.json"
    example_path = REPO_ROOT / "examples" / "cam_plan_example.json"

    export_json_schema(schema_path)
    print(f"Wrote JSON Schema -> {schema_path}")

    plan = example_cam_plan()
    write_cam_plan(example_path, plan)
    print(f"Wrote example plan -> {example_path}")

    reloaded = load_cam_plan(example_path)
    assert reloaded.model_dump() == plan.model_dump()
    print("Example validates on load (round-trip OK).")
