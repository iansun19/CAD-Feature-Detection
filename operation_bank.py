"""Canonical CAM operation bank — the single source of truth for op vocabulary.

This is the closed set of operations the planner is allowed to emit. It replaces
the previous split between free-form ``operation_type`` and ``strategy`` strings
(see planner.py) with ONE flat vocabulary. The names and semantics come directly
from the shop's operation sheet (Operations.pdf, 2026-07-08).

Status: DEFINITION ONLY. The planner has NOT yet been rewired to emit these
values — see eval/operation_bank_audit.md for the current-emission -> bank mapping
that must be reviewed before any rewire. Nothing imports this module yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OpCategory(StrEnum):
    """Coarse grouping used for phase ordering (rough before finish, etc.)."""

    ROUGHING = "roughing"
    FINISHING = "finishing"
    HOLE = "hole"
    TWO_D = "2d"
    AUXILIARY = "auxiliary"


class Operation(StrEnum):
    """The 21 canonical operations. These are the ONLY allowed op values."""

    # --- 3D roughing ---
    OPTIROUGH = "optirough"                 # OptiRough (3D Dynamic)
    AREA_ROUGHING = "area_roughing"         # Area Roughing (3D)
    REST_ROUGHING = "rest_roughing"         # Rest Roughing (3D)

    # --- 3D finishing ---
    WATERLINE = "waterline"                 # Waterline / 3D Contour
    RASTER = "raster"                       # Raster / Parallel
    RADIAL_SPIRAL = "radial_spiral"         # Radial / Spiral
    CONSTANT_SCALLOP = "constant_scallop"   # Constant Scallop
    STEEP_SHALLOW = "steep_shallow"         # Steep / Shallow
    PENCIL = "pencil"                       # Pencil
    REST_FINISH = "rest_finish"             # Rest Finish (Leftover)

    # --- 2.5D ---
    FACING = "facing"                       # Facing
    CONTOUR_2D = "contour_2d"               # 2D Contour
    POCKET = "pocket"                       # Pocket
    DYNAMIC_MILL_2D = "dynamic_mill_2d"     # 2D Dynamic Mill (2D HST)

    # --- holes ---
    DRILL = "drill"                         # Drill (peck, tap, bore, ream cycles)
    CHIP_BREAK_DRILL = "chip_break_drill"   # Chip Break Drilling
    HELIX_BORE = "helix_bore"               # Circle Mill / Helix Bore
    THREAD_MILL = "thread_mill"             # Thread Mill

    # --- auxiliary ---
    ENGRAVING = "engraving"                 # Engraving
    CHAMFER = "chamfer"                     # Chamfer / Edge Break
    DEBURR = "deburr"                       # Deburr


@dataclass(frozen=True)
class OpDef:
    """Metadata for one canonical operation, transcribed from the op sheet."""

    op: Operation
    display_name: str
    category: OpCategory
    what_it_does: str
    when_needed: str


OPERATION_BANK: dict[Operation, OpDef] = {
    d.op: d
    for d in (
        OpDef(Operation.OPTIROUGH, "OptiRough (3D Dynamic)", OpCategory.ROUGHING,
              "Adaptive/trochoidal roughing that clears bulk material fast at constant tool load",
              "Primary roughing of any 3D part"),
        OpDef(Operation.AREA_ROUGHING, "Area Roughing (3D)", OpCategory.ROUGHING,
              "Z-level roughing that steps down following the model",
              "Roughing prismatic/steep parts"),
        OpDef(Operation.REST_ROUGHING, "Rest Roughing (3D)", OpCategory.ROUGHING,
              "Roughs only what a larger tool couldn't reach",
              "After a big-tool rough, before finishing"),
        OpDef(Operation.WATERLINE, "Waterline / 3D Contour", OpCategory.FINISHING,
              "Constant-Z finish passes", "Finishing steep walls"),
        OpDef(Operation.RASTER, "Raster / Parallel", OpCategory.FINISHING,
              "Parallel finish passes across a surface",
              "Finishing shallow/flat-ish surfaces"),
        OpDef(Operation.RADIAL_SPIRAL, "Radial / Spiral", OpCategory.FINISHING,
              "Concentric finish passes from a center",
              "Finishing round bosses/pockets/domes"),
        OpDef(Operation.CONSTANT_SCALLOP, "Constant Scallop", OpCategory.FINISHING,
              "Finish with even cusp height over curvature",
              "Uniform finish on freeform surfaces"),
        OpDef(Operation.STEEP_SHALLOW, "Steep / Shallow", OpCategory.FINISHING,
              "Splits model by slope (waterline steep, raster shallow)",
              "One clean finish across mixed slopes"),
        OpDef(Operation.PENCIL, "Pencil", OpCategory.FINISHING,
              "Traces inner corners/fillets a bigger tool misses",
              "Cleaning corners after finishing"),
        OpDef(Operation.REST_FINISH, "Rest Finish (Leftover)", OpCategory.FINISHING,
              "Small tool cleans material left by the finish tool",
              "Detail/corner cleanup pass"),
        OpDef(Operation.FACING, "Facing", OpCategory.TWO_D,
              "Flattens the stock top", "First op, establish top datum"),
        OpDef(Operation.CONTOUR_2D, "2D Contour", OpCategory.TWO_D,
              "Cuts along an outline", "Profiles, outer walls, open edges"),
        OpDef(Operation.POCKET, "Pocket", OpCategory.TWO_D,
              "Clears a closed 2D region", "Pockets, bosses, islands"),
        OpDef(Operation.DYNAMIC_MILL_2D, "2D Dynamic Mill (2D HST)", OpCategory.TWO_D,
              "Efficient adaptive 2D clearing", "Fast 2.5D roughing/pocketing"),
        OpDef(Operation.DRILL, "Drill", OpCategory.HOLE,
              "Makes holes (peck, tap, bore, ream cycles)", "Any drilled/tapped/bored hole"),
        OpDef(Operation.CHIP_BREAK_DRILL, "Chip Break Drilling", OpCategory.HOLE,
              "Peck drill that retracts slightly to snap chips",
              "Deeper holes where chips pack/clog"),
        OpDef(Operation.HELIX_BORE, "Circle Mill / Helix Bore", OpCategory.HOLE,
              "Interpolates round holes/bores with an endmill",
              "Holes bigger than the drill, bores"),
        OpDef(Operation.THREAD_MILL, "Thread Mill", OpCategory.HOLE,
              "Mills internal/external threads", "Threaded holes/features"),
        OpDef(Operation.ENGRAVING, "Engraving", OpCategory.AUXILIARY,
              "Cuts text, logos, serial numbers into a face", "Marked/identified parts"),
        OpDef(Operation.CHAMFER, "Chamfer / Edge Break", OpCategory.AUXILIARY,
              "Cuts a chamfer along a specified edge/contour", "Breaking a specific sharp edge"),
        OpDef(Operation.DEBURR, "Deburr", OpCategory.AUXILIARY,
              "Auto-detects all model edges and breaks/rounds them",
              "Whole-part deburring in one op"),
    )
}

assert len(OPERATION_BANK) == len(Operation) == 21, "bank must define all 21 operations"


# --- Sequencing phase classification -------------------------------------------------
# Shared by the planner (precedence/phase ordering) and the sequence scorer so both
# agree on what counts as roughing vs finishing. NOTE this is a sequencing phase, not
# the same as OpCategory: `pocket`/`dynamic_mill_2d` are OpCategory.TWO_D but rough in
# phase terms; `facing` and `helix_bore` also run in the rough phase.
ROUGH_PHASE_OPS: frozenset[Operation] = frozenset({
    Operation.OPTIROUGH,
    Operation.AREA_ROUGHING,
    Operation.REST_ROUGHING,
    Operation.POCKET,
    Operation.DYNAMIC_MILL_2D,
    Operation.FACING,
    Operation.HELIX_BORE,
})
FINISH_PHASE_OPS: frozenset[Operation] = frozenset({
    Operation.CONTOUR_2D,
    Operation.WATERLINE,
    Operation.RASTER,
    Operation.RADIAL_SPIRAL,
    Operation.CONSTANT_SCALLOP,
    Operation.STEEP_SHALLOW,
    Operation.PENCIL,
    Operation.REST_FINISH,
})


def is_valid_operation(value: str) -> bool:
    """True when ``value`` is one of the 21 canonical operation slugs."""
    return value in Operation._value2member_map_
