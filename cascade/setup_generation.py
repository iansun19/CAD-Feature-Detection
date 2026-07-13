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

Part identity is DECLARED, never inferred from filenames
--------------------------------------------------------
A lone cascade run / lone STEP is its own single-setup part. Separate STEPs are
separate parts *by default* -- a shared name prefix or a "FRONT"/"REAR" side-word
in the basename does NOT make two STEPs setups of one part. (96260B_FRONT and
96260B_REAR are two independent parts with separate stock and separate jobs; the
old ``infer_machining_side`` filename heuristic that glued them was a bug.)

To plan several STEPs as one multi-setup part -- a genuine flip-job, where one
piece of stock is refixtured to reach features on multiple faces -- hand-author a
multi-setup ``PartSetupDescriptor`` that lists them under one ``part_id`` with one
``SetupEntry`` per orientation. That descriptor is the explicit declaration; the
planner's ``--multi-setup`` path consumes it. Nothing derives multi-setup-ness
from geometry or names.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cascade.setup_descriptor import (
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


# NOTE: part identity and machining side are NEVER inferred from a STEP basename.
# A shared name prefix ("96260B_FRONT" / "96260B_REAR") does not make two STEPs
# setups of one part, and a side-word ("FRONT"/"REAR") in the filename is not the
# fixturing side. Both are process declarations the B-rep does not encode; they
# must be supplied explicitly (via --machining-side / a setup descriptor). Absent
# a declaration, a lone STEP is its own single-setup part (see the module
# docstring). To plan several STEPs as one multi-setup part, hand-author a
# multi-setup ``PartSetupDescriptor`` listing them under one part_id -- that
# descriptor IS the explicit "these are one part" declaration.


# Reachability frame: step-4a annotates each feature along +/- the opening axis,
# with "+Z" = along +opening_axis. A front setup approaches from +opening_axis, a
# back setup from -opening_axis (see planner._reachability_dir_for_setup). This is
# the only role the fixturing side plays in scope: it selects the approach whose
# reachable set is examined. The side string is never itself the answer.
_SIDE_TO_APPROACH_DIR = {"front": "+Z", "back": "-Z"}
_FACE_FEATURE_CLASSES = frozenset({"flat", "face"})


def _node_reachable_from(node: Mapping[str, Any], approach_dir: str) -> bool:
    """Mirror planner._feature_reachable_for_setup on a raw cascade node."""
    reach = (node.get("approach") or {}).get("reachability")
    if not isinstance(reach, Mapping):
        return False
    if reach.get("exempt"):
        # Walls/oblique features: reachable if step-4a gave them any direction.
        return bool(reach.get("reachable_dirs"))
    return approach_dir in (reach.get("reachable_dirs") or [])


def derive_setup_scope(
    graph: Mapping[str, Any], machining_side: str | None
) -> SetupScope | str:
    """Scope a setup by what it can reach -- never by the part's filename.

    The scope falls out of the geometry: a setup whose only reachable machining
    work is a flat (its stock face) is a facing-only flip and is scoped to
    ``facing`` so it cannot re-cut walls that merely happen to be reachable
    (the split-panel over-machining the homolog gate flags). A setup that reaches
    substantial real feature work -- pockets, walls, holes, contours, fillets,
    profiles -- gets ``full`` scope.

    This deliberately drops the old ``front -> ["facing"]`` special-case. On
    96260B the rear owns facing because it reaches the envelope STOCK flat, and
    the front gets full scope because 47 features are reachable from its +Z
    approach -- both are facts read off the graph, not the word "FRONT" in the
    STEP name. Facing itself is likewise geometry-driven downstream via
    ``planner.identify_facing_feature_ids`` (envelope-coincident STOCK flat).

    Falls back to ``full`` (plan everything reachable, narrow nothing) when the
    side is unknown or the graph carries no verified reachability to reason from.
    """
    approach_dir = _SIDE_TO_APPROACH_DIR.get(machining_side or "")
    has_verified = any(
        isinstance((n.get("approach") or {}).get("reachability"), Mapping)
        and (n["approach"]["reachability"]).get("verified")
        for n in graph.get("nodes", [])
    )
    if approach_dir is None or not has_verified:
        return "full"

    reachable_work = 0
    saw_stock_flat = False
    for node in graph.get("nodes", []):
        if not _node_reachable_from(node, approach_dir):
            continue
        if node.get("class_name") in _FACE_FEATURE_CLASSES:
            saw_stock_flat = True
        else:
            reachable_work += 1

    if reachable_work == 0 and saw_stock_flat:
        return ["facing"]
    return "full"


def generate_setup_entry_for_export(
    graph: Mapping[str, Any],
    *,
    setup_id: str,
    part_step: str | Path,
    machining_side: str | None = None,
) -> SetupEntry:
    """Build one setup entry from an exported cascade graph + process hints.

    The opening axis is emitted as ``mode: auto`` (not a fabricated explicit
    vector): the cascade *auto-detects* the axis from geometry, it is not a
    hand-authored shop decision. The planner re-derives the concrete axis from
    ``approach_frame.z`` for auto setups and, via ``approach_frame`` provenance,
    fails loud when geometry could not resolve it. To pin a specific axis, set
    ``opening_axis.mode: explicit`` in the descriptor (e.g. via the wrapper's
    ``--opening-axis``).

    ``machining_side`` is used verbatim and is never inferred from the STEP
    basename. When it is ``None`` the setup gets ``full`` scope (plan everything
    reachable, narrow nothing); a facing-only flip scope is only derived once a
    side is explicitly declared.
    """
    return generate_setup_entry_from_graph(
        graph,
        setup_id=setup_id,
        part_step=part_step,
        machining_side=machining_side,
        scope=derive_setup_scope(graph, machining_side),
        opening_axis_mode="auto",
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
