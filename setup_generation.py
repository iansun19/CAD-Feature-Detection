"""Generate setup descriptors from cascade runs (Stage 3, step 3 — interpretation A).

Synthesizes the ``PartSetupDescriptor`` that was previously hand-authored (see
``eval/gt/96260B_setup.yaml``) from what the cascade already knows: the detected
opening axis and the per-feature 3-axis approach directions from step 2
([[approach_vectors]]). Two process facts remain explicit *inputs*, not inferred
from geometry (they are fixturing / work-partitioning decisions the B-rep does
not encode): ``machining_side`` and ``scope``. Everything geometric — the
opening axis, part_step, and the feature grouping by approach direction — is
generated.

This is the unblocked half of step 3. True multi-orientation setup *discovery
and minimization* needs a unified single part model plus lateral (±X/±Y)
approach axes, neither of which exists yet; see [[cadcam-roadmap]].
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from setup_descriptor import (
    OpeningAxisSpec,
    PartSetupDescriptor,
    SetupDefaults,
    SetupEntry,
    SetupScope,
    _parse_setup_scope,
)


def features_by_approach_dir(graph: Mapping[str, Any]) -> dict[str | None, list[int]]:
    """Group feature ids by their step-2 ``approach.setup_dir`` (+Z / -Z / None).

    ``None`` collects features with no reachable 3-axis setup (oblique bores,
    ambiguous fillet/contour surfaces). This is the signal a caller uses to see
    whether one orientation covers the part or a second setup is needed.
    """
    groups: dict[str | None, list[int]] = defaultdict(list)
    for node in graph.get("nodes", []):
        approach = node.get("approach") or {}
        groups[approach.get("setup_dir")].append(int(node["feature_id"]))
    return dict(groups)


def _opening_axis_spec(
    opening_axis: Sequence[float] | None,
    *,
    mode: str,
) -> OpeningAxisSpec | None:
    if mode == "auto":
        return OpeningAxisSpec(mode="auto")
    if opening_axis is None:
        raise ValueError("opening_axis_mode='explicit' requires an opening_axis vector")
    return OpeningAxisSpec(
        mode="explicit",
        vector=(float(opening_axis[0]), float(opening_axis[1]), float(opening_axis[2])),
    )


def generate_setup_entry(
    *,
    setup_id: str,
    part_step: str | Path | None,
    opening_axis: Sequence[float] | None,
    machining_side: str | None = None,
    scope: SetupScope | str | Sequence[str] | None = None,
    opening_axis_mode: str = "explicit",
) -> SetupEntry:
    """Synthesize one ``SetupEntry`` from a cascade run.

    Geometric facts (opening axis, part_step) are captured from the run;
    ``machining_side`` and ``scope`` are carried-through process inputs.
    ``pocket_access`` is inferred from ``machining_side`` (front→open, back→
    closed) so the emitted entry is self-contained and matches the golden style.
    """
    if isinstance(scope, (str, list, tuple)) and not isinstance(scope, SetupScope):
        scope = _parse_setup_scope(scope, where=f"setups[{setup_id}]")

    pocket_access = None
    if machining_side == "front":
        pocket_access = "open"
    elif machining_side == "back":
        pocket_access = "closed"

    return SetupEntry(
        setup_id=setup_id,
        part_step=str(part_step) if part_step is not None else None,
        machining_side=machining_side,
        opening_axis=_opening_axis_spec(opening_axis, mode=opening_axis_mode),
        pocket_access=pocket_access,
        scope=scope if isinstance(scope, SetupScope) else None,
    )


def generate_setup_entry_from_graph(
    graph: Mapping[str, Any],
    *,
    setup_id: str,
    part_step: str | Path | None,
    machining_side: str | None = None,
    scope: SetupScope | str | Sequence[str] | None = None,
    opening_axis_mode: str = "explicit",
) -> SetupEntry:
    """Same as :func:`generate_setup_entry`, sourcing the opening axis from a
    step-2-annotated feature graph (``approach_frame.z``)."""
    frame = graph.get("approach_frame")
    if not frame or "z" not in frame:
        raise ValueError(
            "feature graph lacks approach_frame; build it with faces+opening_axis "
            "so step-2 approach vectors are present"
        )
    return generate_setup_entry(
        setup_id=setup_id,
        part_step=part_step,
        opening_axis=frame["z"],
        machining_side=machining_side,
        scope=scope,
        opening_axis_mode=opening_axis_mode,
    )


def infer_machining_side(step_path: str | Path) -> str | None:
    """Best-effort front/back hint from a split-panel STEP basename."""
    name = Path(step_path).name.upper()
    if "FRONT" in name:
        return "front"
    if "REAR" in name or "BACK" in name:
        return "back"
    return None


def default_scope_for_side(machining_side: str | None) -> SetupScope | str:
    """Process scope carried through generation (fixturing, not geometry)."""
    if machining_side == "front":
        return ["facing"]
    return "full"


def generate_setup_entry_for_export(
    graph: Mapping[str, Any],
    *,
    setup_id: str,
    part_step: str | Path,
    machining_side: str | None = None,
) -> SetupEntry:
    """Build one setup entry from an exported cascade graph + process hints."""
    side = machining_side if machining_side is not None else infer_machining_side(part_step)
    return generate_setup_entry_from_graph(
        graph,
        setup_id=setup_id,
        part_step=part_step,
        machining_side=side,
        scope=default_scope_for_side(side),
    )


def merge_setup_entries(
    part_id: str,
    entries: Sequence[SetupEntry],
    *,
    defaults: SetupDefaults | None = None,
) -> PartSetupDescriptor:
    """Alias for :func:`generate_part_setup_descriptor` (family merge)."""
    return generate_part_setup_descriptor(part_id, entries, defaults=defaults)


def resolve_generated_descriptor_path(part_id: str, base_dir: Path | None = None) -> Path:
    """Default on-disk path for a merged generated family descriptor."""
    root = base_dir if base_dir is not None else Path(__file__).resolve().parent / "pipeline_out"
    return root / part_id / "setup_descriptor.yaml"


def generate_part_setup_descriptor(
    part_id: str,
    entries: Sequence[SetupEntry],
    *,
    defaults: SetupDefaults | None = None,
) -> PartSetupDescriptor:
    """Merge per-run setup entries into one family descriptor.

    Mirrors the hand-authored structure: one part_id, one entry per orientation
    (STEP file). ``defaults`` fall back to auto opening-axis detection so a
    consumer can re-derive the axis if it ignores the explicit per-setup vector.
    """
    if defaults is None:
        defaults = SetupDefaults(
            opening_axis=OpeningAxisSpec(mode="auto", min_confidence=0.85),
        )
    setups: dict[str, SetupEntry] = {}
    for entry in entries:
        if entry.setup_id in setups:
            raise ValueError(f"duplicate setup_id {entry.setup_id!r}")
        setups[entry.setup_id] = entry
    return PartSetupDescriptor(part_id=part_id, defaults=defaults, setups=setups)
