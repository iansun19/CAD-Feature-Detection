# DESIGN NOTES
# -----------
# v0 assumptions (planner not yet implemented):
#   - 3-axis only: approach directions are discrete (e.g. "+Z"); no 5-axis articulation.
#   - Single setup per plan slice; multi-setup parts emit one CamPlan per setup or a
#     future multi-setup envelope -- v0 examples use setups: [one entry].
#   - First planner slice targets holes and pockets only, but feature_type and
#     operation_type are plain strings so new recognizer classes need no schema bump.
#   - No in-process stock tracking: remaining_material is always null in v0.
#   - Tool/strategy/parameter selection is rule/template based; param_source records
#     provenance until shop-specific or learned values exist.
#
# Placeholders / deferred (fields present or reserved, not populated in v0):
#   - CamPlan.remaining_material -- stock after partial ops (shape TBD).
#   - Setup.fixture -- fixture id/name when fixture library exists.
#   - ToolRef.flute_length_mm, max_depth_mm -- optional shop-library enrichments.
#   - MachiningParameters.coolant -- shop policy not wired in v0.
#   - Multi-setup coordination, 5-axis tool orientation, learned parameters.
#
# Added beyond the task brief (flagged here):
#   - CamPlan @model_validator -- cross-field FK / sequence checks for emitters.
#   - load_cam_plan / write_cam_plan / export_json_schema helpers (mirror feature_graph.py).
#
"""cam_plan_schema.py - structured CAM plan output contract (Section 3).

The planner emits a CamPlan document referencing a feature_graph_cascade.json,
ordered operations, tool catalog entries, and machining parameters with provenance.

Data lineage (v0):
  source_part, feature_graph_ref  <- ingest / cascade artifacts
  Setup.opening_axis, Operation.access <- setup_descriptor YAML (not inferred)
  feature_refs                      <- feature graph node feature_id (as string)
  tools, parameters                 <- shop library + handbook defaults (future planner)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from operation_bank import is_valid_operation

SCHEMA_VERSION = "0.1.0"
REPO_ROOT = Path(__file__).resolve().parent


class PocketAccess(StrEnum):
    """Filleted-pocket machining access label from setup descriptor; not B-rep inferred."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class MachiningParameters(BaseModel):
    """Cutting parameters for one operation.

    Sourced from handbook defaults, shop templates, or conservative heuristics in v0.
    param_source records provenance so values can later be swapped for learned ones.
    """

    model_config = ConfigDict(extra="forbid")

    spindle_rpm: float | None = None
    feed_mm_per_min: float | None = None
    plunge_mm_per_min: float | None = None
    stepdown_mm: float | None = None
    stepover_mm: float | None = None
    coolant: str | None = None
    param_source: str = Field(
        ...,
        description='Provenance tag, e.g. "handbook_default", "shop_default", '
        '"conservative_heuristic".',
    )


class ToolRef(BaseModel):
    """A tool referenced by one or more operations.

    Sourced from a shop tool library or hardcoded v0 defaults. diameter_mm is required;
    flute_length_mm and max_depth_mm are optional enrichments from the library.
    """

    model_config = ConfigDict(extra="forbid")

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


class Setup(BaseModel):
    """One machining setup (workholding + approach direction).

    opening_axis is a discrete 3-axis label (e.g. "+Z") derived from the setup
    descriptor YAML, not a free-form vector in the CAM plan.
    """

    model_config = ConfigDict(extra="forbid")

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
            "non-opening-axis direction; None means the calibrated opening_axis path."
        ),
    )
    orientation_provisional: bool = Field(
        default=False,
        description="True when orientation is a provisional lateral-axis direction.",
    )
    fixture: str | None = Field(
        default=None,
        description="Fixture id or name when a fixture library exists; null in v0.",
    )
    notes: str | None = None


class Operation(BaseModel):
    """One executable machining operation in program order.

    feature_refs point at feature graph node ids (feature_id serialized as string).
    depends_on lists op_ids that must complete before this op -- precedence edges
    preserved even though operations[] is already topologically sorted.
    access is required for pocket-class ops (from setup descriptor); null otherwise.
    """

    model_config = ConfigDict(extra="forbid")

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
            "(operation_bank.Operation), e.g. \"drill\", \"pocket\", \"contour_2d\". "
            "The operation_type/strategy split is retired."
        ),
    )
    tool_id: str
    parameters: MachiningParameters
    depends_on: list[str] = Field(default_factory=list)
    access: PocketAccess | None = Field(
        default=None,
        description="Pocket access label from setup descriptor; null for non-pockets.",
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


class CamPlan(BaseModel):
    """Top-level structured CAM plan emitted by the planner.

    operations[] execution order equals list order; sequence_index on each Operation
    must match its index. tools[] is the deduplicated catalog referenced by tool_id.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    source_part: str = Field(
        ...,
        description="Part identifier or STEP filename the plan was generated from.",
    )
    feature_graph_ref: str = Field(
        ...,
        description="Path or id of the feature_graph_cascade.json used as input.",
    )
    setups: list[Setup] = Field(..., min_length=1)
    operations: list[Operation] = Field(..., min_length=1)
    tools: list[ToolRef] = Field(..., min_length=1)
    remaining_material: dict[str, Any] | None = Field(
        default=None,
        description="Reserved for in-process stock state after partial machining. "
        "Always null in v0.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form generator metadata (version, timestamp, notes).",
    )

    @model_validator(mode="after")
    def _validate_cross_references(self) -> CamPlan:
        setup_ids = {s.setup_id for s in self.setups}
        tool_ids = {t.tool_id for t in self.tools}
        op_ids = {op.op_id for op in self.operations}

        for idx, op in enumerate(self.operations):
            if op.sequence_index != idx:
                raise ValueError(
                    f"operations[{idx}].sequence_index is {op.sequence_index}, "
                    f"expected {idx} (must match list order)"
                )
            if op.setup_id not in setup_ids:
                raise ValueError(
                    f"operation {op.op_id!r} references unknown setup_id {op.setup_id!r}"
                )
            if op.tool_id not in tool_ids:
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
        for op in self.operations:
            if op.op_id in seen_op_ids:
                raise ValueError(f"duplicate op_id {op.op_id!r}")
            seen_op_ids.add(op.op_id)

        seen_tool_ids: set[str] = set()
        for tool in self.tools:
            if tool.tool_id in seen_tool_ids:
                raise ValueError(f"duplicate tool_id {tool.tool_id!r}")
            seen_tool_ids.add(tool.tool_id)

        return self


def load_cam_plan(path: str | Path) -> CamPlan:
    """Load and validate a CAM plan JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return CamPlan.model_validate(data)


def write_cam_plan(path: str | Path, plan: CamPlan) -> None:
    """Write a validated CAM plan to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan.model_dump(mode="json"), fh, indent=2)
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
    """Minimal hand-authored v0 example: one setup, drill + pocket rough."""
    return CamPlan(
        source_part="demo_bracket.step",
        feature_graph_ref="pipeline_out/demo_bracket/feature_graph_cascade.json",
        setups=[
            Setup(
                setup_id="primary",
                opening_axis="+Z",
                fixture=None,
                notes="Single 3-axis setup; v0 slice.",
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
            "notes": "Hand-written v0 contract example; not produced by a planner.",
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
