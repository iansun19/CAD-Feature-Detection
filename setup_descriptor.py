"""setup_descriptor.py - per-part machining setup facts for run_cascade.

Facts that cannot be recovered from a single split-panel STEP (setup identity,
opening axis when auto-detect is ambiguous, filleted-pocket open/closed access)
live here - not in DetectionConfig tuning knobs or eval GT counts.

Load:
    descriptor = load_setup_descriptor(REPO_ROOT / "eval/gt/96260B_setup.yaml")
    setup = resolve_setup_entry(descriptor, step_path=step_path)
    axis, confidence = resolve_opening_axis(setup.opening_axis, wall_axis_directions)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from pocket_detection import PocketSetupConfig

REPO_ROOT = Path(__file__).resolve().parent

OpeningAxisMode = Literal["auto", "explicit"]
PocketAccessLabel = Literal["open", "closed", "unknown"]
ConventionalMachiningSide = Literal["front", "back"]

DEFAULT_MIN_AXIS_CONFIDENCE = 0.85


class SetupDescriptorError(ValueError):
    """Invalid setup descriptor YAML or unresolved required field."""


class OpeningAxisLowConfidenceError(SetupDescriptorError):
    """Auto opening-axis detect did not meet min_confidence - supply explicit axis."""

    def __init__(
        self,
        confidence: float,
        min_confidence: float,
        *,
        n_wall_axes: int,
        setup_id: str,
    ) -> None:
        self.confidence = confidence
        self.min_confidence = min_confidence
        self.n_wall_axes = n_wall_axes
        self.setup_id = setup_id
        super().__init__(
            f"setup {setup_id!r}: opening_axis mode=auto confidence "
            f"{confidence:.3f} < {min_confidence:.3f} "
            f"(from {n_wall_axes} interior wall axes). "
            "Set opening_axis.mode: explicit and supply opening_axis.vector: [x, y, z]."
        )


@dataclass
class OpeningAxisSpec:
    """Opening-axis policy for a setup (or defaults block)."""

    mode: OpeningAxisMode = "auto"
    vector: tuple[float, float, float] | None = None
    min_confidence: float = DEFAULT_MIN_AXIS_CONFIDENCE

    def __post_init__(self) -> None:
        if self.mode not in ("auto", "explicit"):
            raise SetupDescriptorError(
                f"opening_axis.mode must be 'auto' or 'explicit', got {self.mode!r}"
            )
        if self.mode == "explicit":
            if self.vector is None:
                raise SetupDescriptorError(
                    "opening_axis.mode is 'explicit' but opening_axis.vector is missing"
                )
            arr = np.asarray(self.vector, dtype=np.float64)
            if arr.shape != (3,):
                raise SetupDescriptorError(
                    f"opening_axis.vector must be [x, y, z], got {self.vector!r}"
                )
            if float(np.linalg.norm(arr)) <= 1e-12:
                raise SetupDescriptorError(
                    "opening_axis.vector must be non-zero when mode is 'explicit'"
                )
        if not (0.0 < float(self.min_confidence) <= 1.0):
            raise SetupDescriptorError(
                f"opening_axis.min_confidence must be in (0, 1], got {self.min_confidence!r}"
            )


@dataclass
class SetupDefaults:
    """Fallback values when no descriptor file exists or a setup field is omitted."""

    setup_id: str = "default"
    opening_axis: OpeningAxisSpec = field(default_factory=OpeningAxisSpec)
    pocket_access: PocketAccessLabel = "unknown"


@dataclass
class SetupEntry:
    """One named machining setup for a part (may map to one split-panel STEP)."""

    setup_id: str
    part_step: str | None = None
    machining_side: str | None = None
    opening_axis: OpeningAxisSpec | None = None
    pocket_access: PocketAccessLabel | None = None
    pockets_by_seed_face: dict[int, PocketAccessLabel] = field(default_factory=dict)

    def effective_opening_axis(self, defaults: SetupDefaults) -> OpeningAxisSpec:
        spec = self.opening_axis or defaults.opening_axis
        if spec.mode == "auto" and self.opening_axis is None:
            return OpeningAxisSpec(
                mode="auto",
                vector=None,
                min_confidence=defaults.opening_axis.min_confidence,
            )
        return spec

    def effective_pocket_access(self, defaults: SetupDefaults) -> PocketAccessLabel:
        if self.pocket_access is not None:
            return self.pocket_access
        if self.machining_side in ("front", "back"):
            return "open" if self.machining_side == "front" else "closed"
        return defaults.pocket_access


@dataclass
class PartSetupDescriptor:
    """All setup facts for one part_id."""

    part_id: str
    defaults: SetupDefaults = field(default_factory=SetupDefaults)
    setups: dict[str, SetupEntry] = field(default_factory=dict)


@dataclass
class ResolvedSetup:
    """Fully merged setup entry ready for cascade consumption."""

    part_id: str
    setup_id: str
    part_step: str | None
    machining_side: str | None
    opening_axis: OpeningAxisSpec
    pocket_access: PocketAccessLabel
    pockets_by_seed_face: dict[int, PocketAccessLabel]


def default_setup_descriptor(part_id: str = "unknown") -> PartSetupDescriptor:
    """Descriptor used when no YAML is supplied - degraded but non-crashing."""
    return PartSetupDescriptor(
        part_id=part_id,
        defaults=SetupDefaults(),
        setups={},
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SetupDescriptorError(
            "PyYAML required to load setup descriptors (pip install pyyaml)."
        ) from exc
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SetupDescriptorError(f"setup descriptor root must be a mapping: {path}")
    return data


def _parse_opening_axis(raw: Any, *, where: str) -> OpeningAxisSpec:
    if raw is None:
        return OpeningAxisSpec()
    if not isinstance(raw, dict):
        raise SetupDescriptorError(f"{where}.opening_axis must be a mapping")
    mode = raw.get("mode", "auto")
    vector_raw = raw.get("vector")
    vector: tuple[float, float, float] | None = None
    if vector_raw is not None:
        if not isinstance(vector_raw, (list, tuple)) or len(vector_raw) != 3:
            raise SetupDescriptorError(
                f"{where}.opening_axis.vector must be [x, y, z], got {vector_raw!r}"
            )
        vector = (float(vector_raw[0]), float(vector_raw[1]), float(vector_raw[2]))
    min_conf = raw.get("min_confidence", DEFAULT_MIN_AXIS_CONFIDENCE)
    return OpeningAxisSpec(mode=mode, vector=vector, min_confidence=float(min_conf))


def _parse_machining_side(raw: Any, *, where: str) -> str | None:
    if raw is None:
        return None
    side = str(raw).strip().lower()
    if side not in ("front", "back"):
        raise SetupDescriptorError(
            f"{where}.machining_side must be 'front' or 'back' "
            "(convenience shortcut only); for other setups omit machining_side "
            "and set pocket_access (setup-wide) or pockets.by_seed_face "
            f"(per-pocket), got {raw!r}"
        )
    return side


def _parse_pocket_access(raw: Any, *, where: str) -> PocketAccessLabel | None:
    if raw is None:
        return None
    label = str(raw).strip().lower()
    if label not in ("open", "closed", "unknown"):
        raise SetupDescriptorError(
            f"{where}.pocket_access must be open, closed, or unknown, got {raw!r}"
        )
    return label  # type: ignore[return-value]


def _parse_pockets_by_seed_face(raw: Any, *, where: str) -> dict[int, PocketAccessLabel]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SetupDescriptorError(f"{where}.pockets must be a mapping")
    by_seed = raw.get("by_seed_face")
    if by_seed is None:
        return {}
    if not isinstance(by_seed, dict):
        raise SetupDescriptorError(f"{where}.pockets.by_seed_face must be a mapping")
    out: dict[int, PocketAccessLabel] = {}
    for key, val in by_seed.items():
        try:
            face_id = int(key)
        except (TypeError, ValueError) as exc:
            raise SetupDescriptorError(
                f"{where}.pockets.by_seed_face keys must be integer face ids, got {key!r}"
            ) from exc
        label = _parse_pocket_access(val, where=f"{where}.pockets.by_seed_face[{face_id}]")
        if label is None or label == "unknown":
            raise SetupDescriptorError(
                f"{where}.pockets.by_seed_face[{face_id}] must be open or closed"
            )
        out[face_id] = label
    return out


def parse_setup_descriptor(data: dict[str, Any] | None) -> PartSetupDescriptor:
    """Parse a setup descriptor mapping (already loaded from YAML)."""
    data = data or {}
    part_id = str(data.get("part_id") or "unknown")

    defaults_raw = data.get("defaults") or {}
    if not isinstance(defaults_raw, dict):
        raise SetupDescriptorError("defaults must be a mapping")
    defaults = SetupDefaults(
        setup_id=str(defaults_raw.get("setup_id") or "default"),
        opening_axis=_parse_opening_axis(defaults_raw.get("opening_axis"), where="defaults"),
        pocket_access=_parse_pocket_access(defaults_raw.get("pocket_access"), where="defaults")
        or "unknown",
    )

    setups_raw = data.get("setups") or {}
    if not isinstance(setups_raw, dict):
        raise SetupDescriptorError("setups must be a mapping")
    setups: dict[str, SetupEntry] = {}
    for key, entry_raw in setups_raw.items():
        if not isinstance(entry_raw, dict):
            raise SetupDescriptorError(f"setups[{key!r}] must be a mapping")
        setup_id = str(entry_raw.get("setup_id") or key)
        side = _parse_machining_side(
            entry_raw.get("machining_side"), where=f"setups[{key}]"
        )
        setups[setup_id] = SetupEntry(
            setup_id=setup_id,
            part_step=entry_raw.get("part_step"),
            machining_side=side,
            opening_axis=(
                _parse_opening_axis(entry_raw.get("opening_axis"), where=f"setups[{key}]")
                if entry_raw.get("opening_axis") is not None
                else None
            ),
            pocket_access=_parse_pocket_access(
                entry_raw.get("pocket_access"), where=f"setups[{key}]"
            ),
            pockets_by_seed_face=_parse_pockets_by_seed_face(
                entry_raw.get("pockets"), where=f"setups[{key}]"
            ),
        )
    return PartSetupDescriptor(part_id=part_id, defaults=defaults, setups=setups)


def load_setup_descriptor(path: Path | str) -> PartSetupDescriptor:
    """Load and validate a setup descriptor YAML file."""
    p = Path(path)
    if not p.is_file():
        raise SetupDescriptorError(f"setup descriptor not found: {p}")
    return parse_setup_descriptor(_load_yaml(p))


def _normalize_step_ref(step_path: str | Path) -> str:
    return Path(step_path).name.replace(" ", "_").lower()


def find_setup_for_step(
    descriptor: PartSetupDescriptor,
    step_path: str | Path,
) -> SetupEntry | None:
    """Match a setup entry by part_step basename (case/space insensitive)."""
    target = _normalize_step_ref(step_path)
    for entry in descriptor.setups.values():
        if entry.part_step is None:
            continue
        if _normalize_step_ref(entry.part_step) == target:
            return entry
    return None


def resolve_setup_entry(
    descriptor: PartSetupDescriptor,
    *,
    setup_id: str | None = None,
    step_path: str | Path | None = None,
) -> ResolvedSetup:
    """Merge defaults with a named setup, or fall back to defaults-only."""
    entry: SetupEntry | None = None
    if setup_id is not None:
        entry = descriptor.setups.get(setup_id)
        if entry is None:
            raise SetupDescriptorError(
                f"setup_id {setup_id!r} not found in descriptor for part {descriptor.part_id!r}"
            )
    elif step_path is not None:
        entry = find_setup_for_step(descriptor, step_path)
        if entry is None and descriptor.setups:
            known = ", ".join(sorted(descriptor.setups))
            raise SetupDescriptorError(
                f"no setup entry matches step {Path(step_path).name!r} "
                f"for part {descriptor.part_id!r} (known setups: {known})"
            )

    if entry is None:
        defaults = descriptor.defaults
        return ResolvedSetup(
            part_id=descriptor.part_id,
            setup_id=defaults.setup_id,
            part_step=str(step_path) if step_path is not None else None,
            machining_side=None,
            opening_axis=defaults.opening_axis,
            pocket_access=defaults.pocket_access,
            pockets_by_seed_face={},
        )

    return ResolvedSetup(
        part_id=descriptor.part_id,
        setup_id=entry.setup_id,
        part_step=entry.part_step,
        machining_side=entry.machining_side,
        opening_axis=entry.effective_opening_axis(descriptor.defaults),
        pocket_access=entry.effective_pocket_access(descriptor.defaults),
        pockets_by_seed_face=dict(entry.pockets_by_seed_face),
    )


def auto_detect_opening_axis(
    wall_axis_directions: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[tuple[float, float, float], float]:
    """Mirror pocket_detection auto-detect; return (unit_axis, confidence).

    Confidence is the magnitude of the sign-folded mean direction in [0, 1].
    Empty input yields confidence 0.0 (never silently falls back to +Y).
    """
    if len(wall_axis_directions) == 0:
        return (0.0, 1.0, 0.0), 0.0

    A = np.asarray(wall_axis_directions, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != 3:
        raise SetupDescriptorError(
            "wall_axis_directions must be an Nx3 array of unit-ish 3-vectors"
        )
    dom = int(np.argmax(np.abs(A).sum(axis=0)))
    A = A * np.sign(A[:, dom] + 1e-12)[:, None]
    mean = A.mean(axis=0)
    mag = float(np.linalg.norm(mean))
    if mag <= 1e-12:
        return (0.0, 1.0, 0.0), 0.0
    axis = mean / mag
    return (float(axis[0]), float(axis[1]), float(axis[2])), mag


def resolve_opening_axis(
    spec: OpeningAxisSpec,
    wall_axis_directions: Sequence[Sequence[float]] | np.ndarray,
    *,
    setup_id: str = "default",
) -> tuple[tuple[float, float, float], float]:
    """Resolve opening axis from spec + optional wall-axis samples."""
    if spec.mode == "explicit":
        assert spec.vector is not None
        arr = np.asarray(spec.vector, dtype=np.float64)
        unit = arr / float(np.linalg.norm(arr))
        return (float(unit[0]), float(unit[1]), float(unit[2])), 1.0

    axis, confidence = auto_detect_opening_axis(wall_axis_directions)
    if confidence < spec.min_confidence:
        raise OpeningAxisLowConfidenceError(
            confidence,
            spec.min_confidence,
            n_wall_axes=len(wall_axis_directions),
            setup_id=setup_id,
        )
    return axis, confidence


def to_pocket_setup_config(resolved: ResolvedSetup) -> PocketSetupConfig:
    """Map resolved setup facts into existing PocketSetupConfig."""
    side: ConventionalMachiningSide | None = None
    if resolved.machining_side in ("front", "back"):
        side = resolved.machining_side  # type: ignore[assignment]

    access: Literal["open", "closed"] | None = None
    if resolved.pocket_access in ("open", "closed"):
        access = resolved.pocket_access  # type: ignore[assignment]

    return PocketSetupConfig(
        machining_side=side,
        pocket_access=access,
    )


def pocket_access_for_seed(
    resolved: ResolvedSetup,
    seed_face_id: int,
) -> PocketAccessLabel:
    """Per-pocket access: seed-face override, then setup-wide, then unknown."""
    if seed_face_id in resolved.pockets_by_seed_face:
        return resolved.pockets_by_seed_face[seed_face_id]
    if resolved.pocket_access in ("open", "closed"):
        return resolved.pocket_access
    if resolved.machining_side == "front":
        return "open"
    if resolved.machining_side == "back":
        return "closed"
    return "unknown"
