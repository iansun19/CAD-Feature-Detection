"""
cascade_regression.py - golden partition compare/report for cascade labeling regressions.

Pure data structures and comparison logic only (no OCC, no cascade runner).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1
KIND = "cascade_face_partition"
FACE_INDEX_SCHEME = "analyze_step_v1"
AREA_ROUND_DP = 3
IOU_ANNOTATION_THRESHOLD = 0.5

# Share-of-origin thresholds for regrouping REPORT only (gate unchanged).
CHILD_CONTAINMENT = 0.5  # split child: |G cap F| / |F| -- fraction of F from G
SPLIT_COVERAGE = 0.5  # split: children union covers this fraction of G
PARENT_CONTAINMENT = 0.5  # merge parent: |G cap F| / |G| -- fraction of G in F
MERGE_COVERAGE = 0.5  # merge: parents union covers this fraction of F

T_FEATURE_FEATURE = "feature\u2194feature"
T_FEATURE_STOCK = "feature\u2194stock"
T_FEATURE_UNCLAIMED = "feature\u2194unclaimed"
T_STOCK_UNCLAIMED = "stock\u2194unclaimed"

Bucket = tuple[str, tuple[int, ...]] | Literal["stock"] | Literal["unclaimed"]
CoarseBucket = Literal["stock"] | Literal["unclaimed"] | str
CompareBucket = tuple[str, int] | Literal["stock"] | Literal["unclaimed"]
TransitionType = Literal[
    "feature\u2194feature",
    "feature\u2194stock",
    "feature\u2194unclaimed",
    "stock\u2194unclaimed",
]


class PartitionValidationError(ValueError):
    """Raised when a partition violates the full-partition invariant."""


class FaceFingerprintMismatchError(ValueError):
    """Raised when face index fingerprints do not align index-by-index."""


class PartitionExtractionError(ValueError):
    """Raised when cascade output cannot be normalized to a full partition."""


@dataclass(frozen=True)
class FaceFingerprint:
    surface_type: str
    area_mm2: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FaceFingerprint:
        return cls(
            surface_type=str(raw["surface_type"]),
            area_mm2=round_area_mm2(raw["area_mm2"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_type": self.surface_type,
            "area_mm2": round_area_mm2(self.area_mm2),
        }


@dataclass
class PartitionInstance:
    class_name: str
    face_ids: list[int]

    def normalized(self) -> PartitionInstance:
        return PartitionInstance(
            class_name=self.class_name,
            face_ids=sorted(self.face_ids),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PartitionInstance:
        return cls(
            class_name=str(raw["class_name"]),
            face_ids=sorted(int(f) for f in raw["face_ids"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "face_ids": sorted(self.face_ids),
        }


@dataclass
class CascadePartition:
    fixture_id: str
    n_faces: int
    instances: list[PartitionInstance]
    stock_face_ids: list[int]
    unclaimed_face_ids: list[int]
    face_fingerprints: list[FaceFingerprint] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    kind: str = KIND
    face_index_scheme: str = FACE_INDEX_SCHEME
    run_config: dict[str, Any] = field(default_factory=dict)
    bless_meta: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> CascadePartition:
        return CascadePartition(
            schema_version=self.schema_version,
            kind=self.kind,
            fixture_id=self.fixture_id,
            face_index_scheme=self.face_index_scheme,
            n_faces=self.n_faces,
            run_config=dict(self.run_config),
            bless_meta=dict(self.bless_meta),
            face_fingerprints=[
                FaceFingerprint.from_dict(fp.to_dict())
                for fp in self.face_fingerprints
            ],
            instances=[
                inst.normalized()
                for inst in sorted(
                    self.instances,
                    key=lambda i: (i.class_name, tuple(sorted(i.face_ids))),
                )
            ],
            stock_face_ids=sorted(self.stock_face_ids),
            unclaimed_face_ids=sorted(self.unclaimed_face_ids),
        )


@dataclass(frozen=True)
class _FeatureInstance:
    class_name: str
    face_ids: frozenset[int]
    key: Bucket
    label: str


@dataclass
class FaceBucketChange:
    face_id: int
    old_bucket: str
    new_bucket: str
    transition: TransitionType


@dataclass
class FeatureMove:
    face_id: int
    old_instance: str
    new_instance: str
    transition: TransitionType = T_FEATURE_FEATURE


@dataclass(frozen=True)
class RegroupingEvent:
    kind: Literal["split", "merge"]
    class_name: str
    source_label: str
    result_labels: tuple[str, ...]
    leftover_face_count: int = 0

    def format_line(self) -> str:
        if self.kind == "split":
            targets = " + ".join(self.result_labels)
            line = f"1 split \u2014 {self.source_label} -> {targets}"
            if self.leftover_face_count > 0:
                face_word = "face" if self.leftover_face_count == 1 else "faces"
                line += (
                    f"  ({self.leftover_face_count} {face_word} left the feature:"
                    " see boundary crossings)"
                )
            return line
        sources = " + ".join(self.result_labels)
        line = f"1 merge \u2014 {sources} -> {self.source_label}"
        if self.leftover_face_count > 0:
            face_word = "face" if self.leftover_face_count == 1 else "faces"
            line += (
                f"  ({self.leftover_face_count} {face_word} joined from outside:"
                " see boundary crossings)"
            )
        return line


@dataclass
class PartitionDiff:
    fixture_id: str
    n_faces: int
    n_bucket_changes: int
    gate_passed: bool
    n_boundary_crossings: int = 0
    regrouping_events: list[RegroupingEvent] = field(default_factory=list)
    face_changes: list[FaceBucketChange] = field(default_factory=list)
    feature_moves: list[FeatureMove] = field(default_factory=list)
    class_changes: list[FaceBucketChange] = field(default_factory=list)
    split_hints: int = 0
    merge_hints: int = 0
    vanished_instances: list[str] = field(default_factory=list)
    appeared_instances: list[str] = field(default_factory=list)
    stock_changes: list[FaceBucketChange] = field(default_factory=list)
    golden_stock_face_ids: list[int] = field(default_factory=list)
    fresh_stock_face_ids: list[int] = field(default_factory=list)
    fingerprints_ok: bool | None = None

    @property
    def n_changed(self) -> int:
        return self.n_bucket_changes

    def summary_line(self) -> str:
        if self.gate_passed:
            return "PASS - partition unchanged"
        return "REGRESSION \u2014 gate FAIL"


def round_area_mm2(area: float) -> float:
    return round(float(area), AREA_ROUND_DP)


def instance_key(class_name: str, face_ids: list[int] | tuple[int, ...]) -> Bucket:
    return (class_name, tuple(sorted(face_ids)))


def _format_face_ids_compact(face_ids: tuple[int, ...]) -> str:
    if not face_ids:
        return ""
    if len(face_ids) == 1:
        return str(face_ids[0])
    if face_ids == tuple(range(face_ids[0], face_ids[-1] + 1)):
        return f"{face_ids[0]}..{face_ids[-1]}"
    return ",".join(str(f) for f in face_ids)


def format_bucket(bucket: Bucket) -> str:
    if bucket == "stock":
        return "stock"
    if bucket == "unclaimed":
        return "unclaimed"
    class_name, face_ids = bucket
    return f"{class_name}{{{_format_face_ids_compact(face_ids)}}}"


def _bucket_kind(bucket: CompareBucket) -> str:
    if bucket == "stock":
        return "stock"
    if bucket == "unclaimed":
        return "unclaimed"
    return "feature"


def transition_type(old_bucket: CompareBucket, new_bucket: CompareBucket) -> TransitionType:
    old_kind = _bucket_kind(old_bucket)
    new_kind = _bucket_kind(new_bucket)
    if old_kind == "feature" and new_kind == "feature":
        return T_FEATURE_FEATURE
    if old_kind == "feature" and new_kind == "stock":
        return T_FEATURE_STOCK
    if old_kind == "stock" and new_kind == "feature":
        return T_FEATURE_STOCK
    if old_kind == "feature" and new_kind == "unclaimed":
        return T_FEATURE_UNCLAIMED
    if old_kind == "unclaimed" and new_kind == "feature":
        return T_FEATURE_UNCLAIMED
    return T_STOCK_UNCLAIMED


def _face_class_name(
    face_id: int,
    feature_instances: list[_FeatureInstance],
) -> str | None:
    for inst in feature_instances:
        if face_id in inst.face_ids:
            return inst.class_name
    return None


def _collect_changed_face_ids(
    golden: CascadePartition,
    fresh: CascadePartition,
    golden_feature_instances: list[_FeatureInstance],
    fresh_feature_instances: list[_FeatureInstance],
) -> set[int]:
    """Faces whose fine-grained partition bucket changed (strict gate)."""
    del golden_feature_instances, fresh_feature_instances
    golden_buckets = build_face_to_bucket(golden)
    fresh_buckets = build_face_to_bucket(fresh)
    changed: set[int] = set()
    for face_id in range(golden.n_faces):
        if golden_buckets[face_id] != fresh_buckets[face_id]:
            changed.add(face_id)
    return changed


def build_face_to_coarse_bucket(partition: CascadePartition) -> dict[int, CoarseBucket]:
    mapping: dict[int, CoarseBucket] = {}
    for face_id in partition.stock_face_ids:
        mapping[int(face_id)] = "stock"
    for face_id in partition.unclaimed_face_ids:
        mapping[int(face_id)] = "unclaimed"
    for inst in partition.instances:
        coarse = f"feature:{inst.class_name}"
        for face_id in inst.face_ids:
            mapping[int(face_id)] = coarse
    return mapping


def _collect_boundary_crossing_face_ids(
    golden: CascadePartition,
    fresh: CascadePartition,
) -> set[int]:
    """Faces whose coarse bucket (class/stock/unclaimed) changed."""
    golden_coarse = build_face_to_coarse_bucket(golden)
    fresh_coarse = build_face_to_coarse_bucket(fresh)
    return {
        face_id
        for face_id in range(golden.n_faces)
        if golden_coarse[face_id] != fresh_coarse[face_id]
    }


def _child_containment(g_faces: frozenset[int], f_faces: frozenset[int]) -> float:
    if not f_faces:
        return 0.0
    return len(g_faces & f_faces) / len(f_faces)


def _parent_containment(g_faces: frozenset[int], f_faces: frozenset[int]) -> float:
    if not g_faces:
        return 0.0
    return len(g_faces & f_faces) / len(g_faces)


def _collect_regrouping_events(
    golden_instances: list[_FeatureInstance],
    fresh_instances: list[_FeatureInstance],
) -> list[RegroupingEvent]:
    """Same-class instance splits/merges by share-of-origin (partial unions OK)."""
    events: list[RegroupingEvent] = []
    seen_splits: set[str] = set()
    seen_merges: set[str] = set()

    for g_inst in golden_instances:
        children = [
            f_inst for f_inst in fresh_instances
            if f_inst.class_name == g_inst.class_name
            and _child_containment(g_inst.face_ids, f_inst.face_ids) >= CHILD_CONTAINMENT
        ]
        if len(children) < 2:
            continue
        child_union = frozenset().union(*(child.face_ids for child in children))
        coverage = len(child_union & g_inst.face_ids) / len(g_inst.face_ids)
        if coverage < SPLIT_COVERAGE:
            continue
        if g_inst.label in seen_splits:
            continue
        seen_splits.add(g_inst.label)
        children_sorted = sorted(
            children,
            key=lambda item: tuple(sorted(item.face_ids)),
        )
        leftover = len(g_inst.face_ids - child_union)
        events.append(
            RegroupingEvent(
                kind="split",
                class_name=g_inst.class_name,
                source_label=g_inst.label,
                result_labels=tuple(child.label for child in children_sorted),
                leftover_face_count=leftover,
            )
        )

    for f_inst in fresh_instances:
        parents = [
            g_inst for g_inst in golden_instances
            if g_inst.class_name == f_inst.class_name
            and _parent_containment(g_inst.face_ids, f_inst.face_ids) >= PARENT_CONTAINMENT
        ]
        if len(parents) < 2:
            continue
        parent_union = frozenset().union(*(parent.face_ids for parent in parents))
        coverage = len(parent_union & f_inst.face_ids) / len(f_inst.face_ids)
        if coverage < MERGE_COVERAGE:
            continue
        if f_inst.label in seen_merges:
            continue
        seen_merges.add(f_inst.label)
        parents_sorted = sorted(
            parents,
            key=lambda item: tuple(sorted(item.face_ids)),
        )
        leftover = len(f_inst.face_ids - parent_union)
        events.append(
            RegroupingEvent(
                kind="merge",
                class_name=f_inst.class_name,
                source_label=f_inst.label,
                result_labels=tuple(parent.label for parent in parents_sorted),
                leftover_face_count=leftover,
            )
        )

    return events


def _compare_bucket_for_transition(
    face_id: int,
    partition: CascadePartition,
    feature_instances: list[_FeatureInstance],
) -> CompareBucket:
    # Transition-kind discriminator ONLY (via transition_type). Uses
    # (class_name, min(face_ids)), not instance_key's full sorted face_ids
    # tuple - must NOT be used for instance identity (distinct instances
    # sharing a minimum face id would collide).
    if face_id in {int(f) for f in partition.stock_face_ids}:
        return "stock"
    if face_id in {int(f) for f in partition.unclaimed_face_ids}:
        return "unclaimed"
    class_name = _face_class_name(face_id, feature_instances)
    if class_name is None:
        return "unclaimed"
    for inst in feature_instances:
        if face_id in inst.face_ids:
            return (class_name, min(inst.face_ids))
    return "unclaimed"


def build_face_to_bucket(partition: CascadePartition) -> dict[int, Bucket]:
    mapping: dict[int, Bucket] = {}
    for face_id in partition.stock_face_ids:
        mapping[int(face_id)] = "stock"
    for face_id in partition.unclaimed_face_ids:
        mapping[int(face_id)] = "unclaimed"
    for inst in partition.instances:
        key = instance_key(inst.class_name, inst.face_ids)
        for face_id in inst.face_ids:
            mapping[int(face_id)] = key
    return mapping


def _face_display_bucket(
    face_id: int,
    partition: CascadePartition,
    feature_instances: list[_FeatureInstance],
) -> str:
    for face_id_check in partition.stock_face_ids:
        if int(face_id_check) == face_id:
            return "stock"
    for face_id_check in partition.unclaimed_face_ids:
        if int(face_id_check) == face_id:
            return "unclaimed"
    label = _face_to_instance_label(face_id, feature_instances)
    return label or "unclaimed"


def validate_partition(partition: CascadePartition) -> None:
    if partition.n_faces < 0:
        raise PartitionValidationError("n_faces must be non-negative")

    seen: dict[int, int] = {}
    for face_id in partition.stock_face_ids:
        seen[int(face_id)] = seen.get(int(face_id), 0) + 1
    for face_id in partition.unclaimed_face_ids:
        seen[int(face_id)] = seen.get(int(face_id), 0) + 1
    for inst in partition.instances:
        for face_id in inst.face_ids:
            seen[int(face_id)] = seen.get(int(face_id), 0) + 1

    duplicates = sorted(fid for fid, count in seen.items() if count > 1)
    if duplicates:
        raise PartitionValidationError(
            "partition overlap: face_ids appear in more than one bucket: "
            + ", ".join(str(f) for f in duplicates)
        )

    expected = set(range(partition.n_faces))
    present = set(seen)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append("missing face_ids: " + ", ".join(str(f) for f in missing))
        if extra:
            parts.append("unexpected face_ids: " + ", ".join(str(f) for f in extra))
        raise PartitionValidationError("partition gap: " + "; ".join(parts))

    if partition.face_fingerprints and len(partition.face_fingerprints) != partition.n_faces:
        raise PartitionValidationError(
            f"face_fingerprints length {len(partition.face_fingerprints)} "
            f"!= n_faces {partition.n_faces}"
        )


def verify_face_fingerprints(
    golden: list[FaceFingerprint],
    fresh: list[FaceFingerprint],
) -> None:
    if len(golden) != len(fresh):
        raise FaceFingerprintMismatchError(
            f"face fingerprint count mismatch: golden has {len(golden)}, "
            f"fresh has {len(fresh)}; re-bless or fix STEP/index scheme"
        )
    for face_id, (g_fp, f_fp) in enumerate(zip(golden, fresh)):
        if g_fp.surface_type != f_fp.surface_type:
            raise FaceFingerprintMismatchError(
                f"face reordering or geometry drift detected at face_id={face_id}: "
                f"surface_type golden={g_fp.surface_type!r} fresh={f_fp.surface_type!r}; "
                "re-bless or fix STEP/index scheme"
            )
        if round_area_mm2(g_fp.area_mm2) != round_area_mm2(f_fp.area_mm2):
            raise FaceFingerprintMismatchError(
                f"face reordering or geometry drift detected at face_id={face_id}: "
                f"area_mm2 golden={round_area_mm2(g_fp.area_mm2)} "
                f"fresh={round_area_mm2(f_fp.area_mm2)}; "
                "re-bless or fix STEP/index scheme"
            )


def face_iou(faces_a: set[int] | frozenset[int], faces_b: set[int] | frozenset[int]) -> float:
    inter = len(faces_a & faces_b)
    if inter == 0:
        return 0.0
    union = len(faces_a | faces_b)
    return inter / union


def _feature_instances(partition: CascadePartition) -> list[_FeatureInstance]:
    out: list[_FeatureInstance] = []
    for inst in partition.instances:
        faces = frozenset(inst.face_ids)
        key = instance_key(inst.class_name, inst.face_ids)
        out.append(
            _FeatureInstance(
                class_name=inst.class_name,
                face_ids=faces,
                key=key,
                label=format_bucket(key),
            )
        )
    out.sort(key=lambda item: (item.class_name, tuple(sorted(item.face_ids))))
    return out


def _match_instances(
    golden_instances: list[_FeatureInstance],
    fresh_instances: list[_FeatureInstance],
    *,
    iou_threshold: float = IOU_ANNOTATION_THRESHOLD,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g_inst in enumerate(golden_instances):
        for fi, f_inst in enumerate(fresh_instances):
            iou = face_iou(g_inst.face_ids, f_inst.face_ids)
            if iou >= iou_threshold:
                candidates.append((iou, gi, fi))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_golden: set[int] = set()
    matched_fresh: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gi, fi in candidates:
        if gi in matched_golden or fi in matched_fresh:
            continue
        matches.append((gi, fi, iou))
        matched_golden.add(gi)
        matched_fresh.add(fi)
    return matches, matched_golden, matched_fresh


def _count_split_merge_hints(
    golden_instances: list[_FeatureInstance],
    fresh_instances: list[_FeatureInstance],
    *,
    iou_threshold: float = IOU_ANNOTATION_THRESHOLD,
) -> tuple[int, int]:
    splits = 0
    for g_inst in golden_instances:
        overlaps = sum(
            1 for f_inst in fresh_instances
            if face_iou(g_inst.face_ids, f_inst.face_ids) >= iou_threshold
        )
        if overlaps >= 2:
            splits += 1

    merges = 0
    for f_inst in fresh_instances:
        overlaps = sum(
            1 for g_inst in golden_instances
            if face_iou(g_inst.face_ids, f_inst.face_ids) >= iou_threshold
        )
        if overlaps >= 2:
            merges += 1
    return splits, merges


def _face_to_instance_label(
    face_id: int,
    instances: list[_FeatureInstance],
) -> str | None:
    for inst in instances:
        if face_id in inst.face_ids:
            return inst.label
    return None


def compare_partitions(
    golden: CascadePartition,
    fresh: CascadePartition,
    *,
    verify_fingerprints: bool = True,
) -> PartitionDiff:
    validate_partition(golden)
    validate_partition(fresh)
    if golden.n_faces != fresh.n_faces:
        raise ValueError(
            f"n_faces mismatch: golden={golden.n_faces}, fresh={fresh.n_faces}"
        )
    if verify_fingerprints and golden.face_fingerprints and fresh.face_fingerprints:
        verify_face_fingerprints(golden.face_fingerprints, fresh.face_fingerprints)

    golden_feature_instances = _feature_instances(golden)
    fresh_feature_instances = _feature_instances(fresh)
    changed_face_ids = _collect_changed_face_ids(
        golden,
        fresh,
        golden_feature_instances,
        fresh_feature_instances,
    )

    face_changes: list[FaceBucketChange] = []
    for face_id in sorted(changed_face_ids):
        old_compare = _compare_bucket_for_transition(
            face_id, golden, golden_feature_instances,
        )
        new_compare = _compare_bucket_for_transition(
            face_id, fresh, fresh_feature_instances,
        )
        face_changes.append(
            FaceBucketChange(
                face_id=face_id,
                old_bucket=_face_display_bucket(
                    face_id, golden, golden_feature_instances,
                ),
                new_bucket=_face_display_bucket(
                    face_id, fresh, fresh_feature_instances,
                ),
                transition=transition_type(old_compare, new_compare),
            )
        )
    _matches, matched_golden, matched_fresh = _match_instances(
        golden_feature_instances,
        fresh_feature_instances,
    )
    split_hints, merge_hints = _count_split_merge_hints(
        golden_feature_instances,
        fresh_feature_instances,
    )

    feature_moves: list[FeatureMove] = []
    class_changes: list[FaceBucketChange] = []
    stock_changes: list[FaceBucketChange] = []

    for change in face_changes:
        if change.transition in (T_FEATURE_STOCK, T_FEATURE_UNCLAIMED, T_STOCK_UNCLAIMED):
            stock_changes.append(change)
        if change.transition != T_FEATURE_FEATURE:
            continue
        old_label = _face_to_instance_label(change.face_id, golden_feature_instances)
        new_label = _face_to_instance_label(change.face_id, fresh_feature_instances)
        if not old_label or not new_label:
            continue
        feature_moves.append(
            FeatureMove(
                face_id=change.face_id,
                old_instance=old_label,
                new_instance=new_label,
                transition=T_FEATURE_FEATURE,
            )
        )
        if old_label.split("{", 1)[0] != new_label.split("{", 1)[0]:
            class_changes.append(change)

    vanished_instances = [
        golden_feature_instances[gi].label
        for gi in range(len(golden_feature_instances))
        if gi not in matched_golden
    ]
    appeared_instances = [
        fresh_feature_instances[fi].label
        for fi in range(len(fresh_feature_instances))
        if fi not in matched_fresh
    ]

    n_bucket_changes = len(changed_face_ids)
    boundary_crossing_face_ids = _collect_boundary_crossing_face_ids(golden, fresh)
    regrouping_events = _collect_regrouping_events(
        golden_feature_instances,
        fresh_feature_instances,
    )
    fingerprints_ok = None
    if golden.face_fingerprints and fresh.face_fingerprints:
        try:
            verify_face_fingerprints(golden.face_fingerprints, fresh.face_fingerprints)
            fingerprints_ok = True
        except FaceFingerprintMismatchError:
            fingerprints_ok = False

    return PartitionDiff(
        fixture_id=golden.fixture_id or fresh.fixture_id,
        n_faces=golden.n_faces,
        n_bucket_changes=n_bucket_changes,
        n_boundary_crossings=len(boundary_crossing_face_ids),
        regrouping_events=regrouping_events,
        gate_passed=(n_bucket_changes == 0),
        face_changes=face_changes,
        feature_moves=sorted(feature_moves, key=lambda move: move.face_id),
        class_changes=class_changes,
        split_hints=split_hints,
        merge_hints=merge_hints,
        vanished_instances=sorted(vanished_instances),
        appeared_instances=sorted(appeared_instances),
        stock_changes=sorted(stock_changes, key=lambda change: change.face_id),
        golden_stock_face_ids=sorted(golden.stock_face_ids),
        fresh_stock_face_ids=sorted(fresh.stock_face_ids),
        fingerprints_ok=fingerprints_ok,
    )


def partition_from_dict(raw: dict[str, Any]) -> CascadePartition:
    return CascadePartition(
        schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
        kind=str(raw.get("kind", KIND)),
        fixture_id=str(raw.get("fixture_id", "")),
        face_index_scheme=str(raw.get("face_index_scheme", FACE_INDEX_SCHEME)),
        n_faces=int(raw["n_faces"]),
        run_config=dict(raw.get("run_config") or {}),
        bless_meta=dict(raw.get("bless_meta") or {}),
        face_fingerprints=[
            FaceFingerprint.from_dict(item)
            for item in raw.get("face_fingerprints") or []
        ],
        instances=[PartitionInstance.from_dict(item) for item in raw.get("instances") or []],
        stock_face_ids=sorted(int(f) for f in raw.get("stock_face_ids") or []),
        unclaimed_face_ids=sorted(int(f) for f in raw.get("unclaimed_face_ids") or []),
    )


def load_partition(path: str | Path) -> CascadePartition:
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"partition JSON must be an object: {path}")
    return partition_from_dict(raw)


def dump_partition(partition: CascadePartition, path: str | Path) -> None:
    normalized = partition.normalized()
    validate_partition(normalized)
    payload = {
        "schema_version": normalized.schema_version,
        "kind": normalized.kind,
        "fixture_id": normalized.fixture_id,
        "face_index_scheme": normalized.face_index_scheme,
        "n_faces": normalized.n_faces,
        "run_config": normalized.run_config,
        "bless_meta": normalized.bless_meta,
        "face_fingerprints": [fp.to_dict() for fp in normalized.face_fingerprints],
        "instances": [inst.to_dict() for inst in normalized.instances],
        "stock_face_ids": normalized.stock_face_ids,
        "unclaimed_face_ids": normalized.unclaimed_face_ids,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def format_partition_diff_report(
    diff: PartitionDiff,
    *,
    mute_stock: bool = False,
    mute_features: bool = False,
) -> str:
    lines = [
        f"=== {diff.fixture_id}: {diff.summary_line()} ===",
    ]
    if diff.fingerprints_ok is True:
        lines.append(
            f"run_config: (not compared) ; n_faces: {diff.n_faces}/{diff.n_faces} ; "
            f"fingerprints: {diff.n_faces}/{diff.n_faces} OK"
        )
    elif diff.fingerprints_ok is False:
        lines.append(
            f"run_config: (not compared) ; n_faces: {diff.n_faces}/{diff.n_faces} ; "
            "fingerprints: MISMATCH"
        )

    gate_label = (
        "PASS (partition unchanged)" if diff.gate_passed else "FAIL (partition changed)"
    )
    lines.append(f"gate: {gate_label}")
    if diff.gate_passed:
        lines.append("fine bucket changes (gate): 0")
        lines.append("boundary crossings (faces changing class/stock/unclaimed): 0")
        lines.append("regrouping: none")
    else:
        regroup_faces = diff.n_bucket_changes - diff.n_boundary_crossings
        lines.append(
            f"fine bucket changes (gate): {diff.n_bucket_changes} "
            f"({diff.n_boundary_crossings} boundary crossing(s), "
            f"{regroup_faces} same-class regrouping face(s))"
        )
        lines.append(
            "boundary crossings (faces changing class/stock/unclaimed): "
            f"{diff.n_boundary_crossings}"
        )
        if diff.regrouping_events:
            for event in diff.regrouping_events:
                lines.append(f"regrouping: {event.format_line()}")
        else:
            lines.append("regrouping: none")

    if not mute_features:
        feature_related = [
            change for change in diff.face_changes
            if change.transition in (T_FEATURE_FEATURE, T_FEATURE_STOCK, T_FEATURE_UNCLAIMED)
        ]
        lines.append("")
        lines.append(f"--- feature-attributed changes ({len(feature_related)}) ---")
        boundary_crossing_changes = [
            change for change in diff.face_changes
            if change.transition != T_FEATURE_FEATURE or change in diff.class_changes
        ]
        if boundary_crossing_changes:
            lines.append(
                f"--- boundary-crossing feature faces ({len(boundary_crossing_changes)}) ---"
            )
            for change in boundary_crossing_changes:
                lines.append(
                    f"  face {change.face_id}: {change.old_bucket} -> {change.new_bucket}"
                )
                lines.append(f"    transition: {change.transition}")
        if diff.feature_moves:
            lines.append(f"--- feature instance relabels ({len(diff.feature_moves)}) ---")
            for move in diff.feature_moves:
                lines.append(
                    f"  face {move.face_id}: {move.old_instance} -> {move.new_instance}"
                )
                lines.append(f"    transition: {move.transition}")
                lines.append(
                    f"    (moved from instance {move.old_instance} -> {move.new_instance})"
                )
        elif feature_related:
            for change in feature_related:
                lines.append(
                    f"  face {change.face_id}: {change.old_bucket} -> {change.new_bucket}"
                )
                lines.append(f"    transition: {change.transition}")
        else:
            lines.append(f"  (no {T_FEATURE_FEATURE} or {T_FEATURE_UNCLAIMED} changes)")

        lines.append(f"--- class changes ({len(diff.class_changes)}) ---")
        lines.append(f"--- split/merge hints ({diff.split_hints + diff.merge_hints}) ---")
        matched_old = {move.old_instance for move in diff.feature_moves}
        matched_new = {move.new_instance for move in diff.feature_moves}
        true_vanished = [label for label in diff.vanished_instances if label not in matched_old]
        true_appeared = [label for label in diff.appeared_instances if label not in matched_new]
        lines.append(
            f"--- true vanished / appeared ({len(true_vanished) + len(true_appeared)}) ---"
        )
        for label in true_vanished:
            lines.append(f"  vanished: {label}")
        for label in true_appeared:
            lines.append(f"  appeared: {label}")
        lines.append("--- optional param drift (0) ---")

    if not mute_stock:
        lines.append("")
        lines.append("--- stock partition (attribution summary) ---")
        if not diff.stock_changes:
            stock_ids = diff.golden_stock_face_ids
            if stock_ids:
                ids = ", ".join(str(f) for f in stock_ids)
                lines.append(f"  stock faces: unchanged ({len(stock_ids)} faces: [{ids}])")
            else:
                lines.append("  stock faces: unchanged (0 faces)")
            lines.append("(stock logic also covered by test_stock_cut_classification.py)")
        else:
            for change in diff.stock_changes:
                lines.append(
                    f"  face {change.face_id}: {change.old_bucket} -> {change.new_bucket} "
                    f"         transition: {change.transition}"
                )
            lines.append("(stock logic also covered by test_stock_cut_classification.py)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixture loading, cascade extraction, and manual bless (requires pythonocc)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
REGRESSION_DIR = REPO_ROOT / "eval" / "regression"
REGRESSION_FIXTURES_DIR = REGRESSION_DIR / "fixtures"


@dataclass
class RegressionFixture:
    fixture_id: str
    step: str
    graph_npz: str | None
    face_index_scheme: str
    setup: dict[str, Any]
    run: dict[str, Any]
    golden: str
    notes: str
    step_path: Path
    graph_npz_path: Path | None
    golden_path: Path
    step_exists: bool


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML required for regression fixture YAML (pip install pyyaml)."
        ) from exc
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"fixture YAML root must be a mapping: {path}")
    return data


def _resolve_repo_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_regression_fixture(
    fixture_yaml: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> RegressionFixture:
    """Load a per-fixture YAML descriptor; report STEP absence without raising."""
    root = Path(repo_root or REPO_ROOT).resolve()
    yaml_path = _resolve_repo_path(str(fixture_yaml), root)
    raw = _load_yaml_mapping(yaml_path)

    step_rel = str(raw["step"])
    graph_npz_raw = raw.get("graph_npz")
    golden_rel = str(raw["golden"])
    step_path = _resolve_repo_path(step_rel, root)
    graph_npz_path = (
        _resolve_repo_path(str(graph_npz_raw), root)
        if graph_npz_raw is not None
        else None
    )
    golden_path = _resolve_repo_path(golden_rel, root)

    return RegressionFixture(
        fixture_id=str(raw["fixture_id"]),
        step=step_rel,
        graph_npz=str(graph_npz_raw) if graph_npz_raw is not None else None,
        face_index_scheme=str(raw.get("face_index_scheme", FACE_INDEX_SCHEME)),
        setup=dict(raw.get("setup") or {}),
        run=dict(raw.get("run") or {}),
        golden=golden_rel,
        notes=str(raw.get("notes") or "").strip(),
        step_path=step_path,
        graph_npz_path=graph_npz_path,
        golden_path=golden_path,
        step_exists=step_path.is_file(),
    )


def list_regression_fixture_ids(
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Return sorted fixture_id values from eval/regression/fixtures/*.yaml."""
    root = Path(repo_root or REPO_ROOT).resolve()
    fixtures_dir = root / "eval" / "regression" / "fixtures"
    if not fixtures_dir.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(fixtures_dir.glob("*.yaml")):
        raw = _load_yaml_mapping(path)
        ids.append(str(raw["fixture_id"]))
    return ids


def load_regression_fixture_by_id(
    fixture_id: str,
    *,
    repo_root: str | Path | None = None,
) -> RegressionFixture:
    root = Path(repo_root or REPO_ROOT).resolve()
    path = root / "eval" / "regression" / "fixtures" / f"{fixture_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown regression fixture: {fixture_id} ({path})")
    fixture = load_regression_fixture(path, repo_root=root)
    if fixture.fixture_id != fixture_id:
        raise ValueError(
            f"fixture YAML {path} fixture_id={fixture.fixture_id!r} "
            f"!= requested {fixture_id!r}"
        )
    return fixture


def verify_graph_npz_face_count(
    fixture: RegressionFixture,
    n_faces: int,
) -> None:
    if fixture.graph_npz_path is None or not fixture.graph_npz_path.is_file():
        return
    import numpy as np

    edge_index = np.load(fixture.graph_npz_path)["edge_index"]
    graph_faces = int(edge_index.max()) + 1 if edge_index.size else 0
    if graph_faces != n_faces:
        raise ValueError(
            f"{fixture.fixture_id}: graph.npz implies {graph_faces} faces but "
            f"analyze_step/golden expect {n_faces}; regenerate graph or re-bless"
        )


def regression_preflight(
    fixture: RegressionFixture,
    *,
    golden: CascadePartition | None = None,
) -> CascadePartition:
    """Design pre-flight checks before partition compare (raises on failure)."""
    if not fixture.step_exists:
        raise FileNotFoundError(f"missing fixture STEP: {fixture.step_path}")
    if golden is None:
        if not fixture.golden_path.is_file():
            raise FileNotFoundError(
                f"missing golden partition (misconfig): {fixture.golden_path}"
            )
        golden = load_partition(fixture.golden_path)
    from feature_params import analyze_step

    n_step = len(analyze_step(fixture.step_path))
    if n_step != golden.n_faces:
        raise ValueError(
            f"{fixture.fixture_id}: analyze_step has {n_step} faces but "
            f"golden.n_faces={golden.n_faces}; re-bless or fix STEP/index scheme"
        )
    verify_graph_npz_face_count(fixture, golden.n_faces)
    if golden.face_fingerprints:
        fresh_fps = extract_face_fingerprints(fixture.step_path)
        verify_face_fingerprints(golden.face_fingerprints, fresh_fps)
    return golden


def run_config_drift_lines(
    golden: CascadePartition,
    fixture: RegressionFixture,
) -> list[str]:
    expected = _fixture_run_config(fixture)
    stored = golden.run_config
    if stored == expected:
        return ["run_config: matches fixture"]
    return [
        "run_config: DRIFT (warning only; does not affect gate)",
        f"  golden:  {json.dumps(stored, sort_keys=True)}",
        f"  fixture: {json.dumps(expected, sort_keys=True)}",
    ]


def run_fixture_regression(fixture: RegressionFixture) -> PartitionDiff:
    """Run cascade extraction and compare to the blessed golden partition."""
    golden = regression_preflight(fixture)
    fresh = extract_partition_for_fixture(fixture)
    return compare_partitions(golden, fresh, verify_fingerprints=True)


def partition_diff_to_dict(diff: PartitionDiff) -> dict[str, Any]:
    return {
        "fixture_id": diff.fixture_id,
        "n_faces": diff.n_faces,
        "gate_passed": diff.gate_passed,
        "n_bucket_changes": diff.n_bucket_changes,
        "n_boundary_crossings": diff.n_boundary_crossings,
        "fingerprints_ok": diff.fingerprints_ok,
        "face_changes": [
            {
                "face_id": c.face_id,
                "old_bucket": c.old_bucket,
                "new_bucket": c.new_bucket,
                "transition": c.transition,
            }
            for c in diff.face_changes
        ],
        "regrouping_events": [e.format_line() for e in diff.regrouping_events],
    }


def extract_face_fingerprints(step_path: str | Path) -> list[FaceFingerprint]:
    """Compute analyze_step_v1 fingerprints aligned with run_cascade face indices.

    Uses feature_params.analyze_step which enumerates
    TopologyExplorer(shape).faces() in order (same as run_cascade's
    n_faces loop over analyze_step output).

    Fields:
    - surface_type: FaceGeom.surface_type from step_ingest._surface_type_code
      (BRepAdaptor_Surface.GetType() mapped to names like plane, cylinder, ...).
    - area_mm2: FaceGeom.area, face surface area in mm^2 from OCC BRepGProp
      surface mass via step_ingest.sprops(face).Mass().
    """
    from feature_params import analyze_step

    records = analyze_step(step_path)
    return [
        FaceFingerprint(
            surface_type=rec.surface_type,
            area_mm2=rec.area,
        )
        for rec in records
    ]


def _fixture_run_config(fixture: RegressionFixture) -> dict[str, Any]:
    return {
        "stock_classifier": fixture.run.get("stock_classifier", "new"),
        "max_diameter_mm": float(fixture.run.get("max_diameter_mm", 150.0)),
        "machining_side": fixture.setup.get("machining_side"),
        "pocket_access": fixture.setup.get("pocket_access"),
        "wall_attach_dist": fixture.run.get("wall_attach_dist"),
    }


def run_cascade_for_fixture(
    fixture: RegressionFixture,
) -> tuple[Any, ...]:
    """Run the cascade with fixture-pinned config (same wiring as run_cascade.py)."""
    from hole_detection import HoleDetectionConfig
    from pocket_detection import PocketDetectionConfig
    from run_cascade import _load_edges, _resolve_pocket_setup_for_cascade, run_cascade

    if not fixture.step_exists:
        raise FileNotFoundError(f"missing fixture STEP: {fixture.step_path}")

    step_path = fixture.step_path
    edge_index, edge_attr = _load_edges(fixture.graph_npz_path, step_path)

    machining_side = fixture.setup.get("machining_side")
    pocket_access = fixture.setup.get("pocket_access")
    setup = _resolve_pocket_setup_for_cascade(
        step_path,
        machining_side=machining_side,
        pocket_access=pocket_access,
    )
    pocket_config = PocketDetectionConfig(
        wall_attach_dist=fixture.run.get("wall_attach_dist"),
        setup=setup,
    )
    hole_config = HoleDetectionConfig(
        max_hole_diameter_mm=float(fixture.run.get("max_diameter_mm", 150.0)),
    )
    stock_classifier = fixture.run.get("stock_classifier", "new")

    cascade = run_cascade(
        step_path,
        edge_index,
        edge_attr,
        stock_classifier=stock_classifier,
        pocket_config=pocket_config,
        hole_config=hole_config,
    )
    return cascade + (edge_index,)


def _normalize_instances_for_stock(
    instances: list[PartitionInstance],
    stock_ids: list[int],
) -> tuple[list[PartitionInstance], list[int]]:
    """Remove stock faces from feature instances (stock bucket wins).

    ``apply_filleted_lobe_tiers_to_result`` can claim STOCK-gated faces into
    pocket features; the golden partition keeps buckets disjoint with stock
    authoritative.
    """
    stock = set(stock_ids)
    stripped: set[int] = set()
    out: list[PartitionInstance] = []
    for inst in instances:
        kept = sorted(set(inst.face_ids) - stock)
        stripped |= set(inst.face_ids) & stock
        if kept:
            out.append(PartitionInstance(class_name=inst.class_name, face_ids=kept))
    return out, sorted(stripped)


def extract_partition_from_cascade(
    fixture: RegressionFixture,
    cascade_results: tuple[Any, ...],
    *,
    edge_index: Any | None = None,
) -> CascadePartition:
    """Convert cascade pass results to a normalized CascadePartition.

    Feature ``class_name`` values follow ``eval_cascade.build_cascade_feature_graph``
    (pocket ``toolpath_class``, hole ``CASCADE_KIND_TO_TP``, etc.). Stock faces
    come from the stock gate; unclaimed = all other faces not in any feature.
    """
    from eval_cascade import build_cascade_feature_graph
    from stock_cut_classification import stock_face_ids

    (
        faces,
        pocket_result,
        hole_result,
        coaxial_result,
        flat_result,
        outer_fillet_result,
        wall_result,
        profile_result,
        residual_result,
    ) = cascade_results[:9]
    if edge_index is None:
        edge_index = cascade_results[9]

    n_faces = len(faces)
    stock_classifier = fixture.run.get("stock_classifier", "new")
    stock_ids: list[int] = []
    if stock_classifier and stock_classifier != "off":
        stock_ids = sorted(
            stock_face_ids(
                fixture.step_path,
                classifier=stock_classifier,  # type: ignore[arg-type]
            )
        )

    graph = build_cascade_feature_graph(
        fixture.fixture_id,
        n_faces,
        pocket_result,
        hole_result,
        coaxial_result,
        flat_result,
        outer_fillet_result,
        wall_result,
        profile_result,
        residual_result,
        edge_index,
    )
    instances = [
        PartitionInstance(
            class_name=str(node["class_name"]),
            face_ids=[int(f) for f in node["face_ids"]],
        )
        for node in graph["nodes"]
    ]
    instances, stripped_from_features = _normalize_instances_for_stock(instances, stock_ids)
    if stripped_from_features:
        import sys
        print(
            f"WARNING [{fixture.fixture_id}]: removed {len(stripped_from_features)} "
            "STOCK-gated face(s) from feature instances (lobe-tier overlap); "
            "stock bucket wins. Sample: "
            + ", ".join(str(f) for f in stripped_from_features[:12])
            + ("..." if len(stripped_from_features) > 12 else ""),
            file=sys.stderr,
        )

    claimed: set[int] = set()
    for inst in instances:
        claimed.update(inst.face_ids)

    overlap = sorted(set(stock_ids) & claimed)
    if overlap:
        raise PartitionExtractionError(
            f"{fixture.fixture_id}: {len(overlap)} face(s) still in both stock and "
            f"features after normalization: {overlap[:20]}"
        )

    unclaimed = sorted(set(range(n_faces)) - set(stock_ids) - claimed)

    partition = CascadePartition(
        fixture_id=fixture.fixture_id,
        n_faces=n_faces,
        instances=instances,
        stock_face_ids=stock_ids,
        unclaimed_face_ids=unclaimed,
        face_fingerprints=extract_face_fingerprints(fixture.step_path),
        face_index_scheme=fixture.face_index_scheme,
        run_config=_fixture_run_config(fixture),
    )
    validate_partition(partition)
    return partition


def extract_partition_for_fixture(fixture: RegressionFixture) -> CascadePartition:
    """Run cascade + extraction for one fixture."""
    cascade = run_cascade_for_fixture(fixture)
    edge_index = cascade[9]
    return extract_partition_from_cascade(fixture, cascade, edge_index=edge_index)


def _git_bless_meta(repo_root: Path) -> tuple[str, bool]:
    import subprocess

    def _run(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                args,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    sha = _run(["git", "rev-parse", "--short", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return sha, dirty


def bless_regression_fixture(
    fixture: RegressionFixture,
    *,
    reason: str,
    repo_root: str | Path | None = None,
) -> CascadePartition:
    """Manual bless only - run extraction, validate, write golden, log diff."""
    from datetime import datetime, timezone

    if len(reason.strip()) < 10:
        raise ValueError("bless --reason must be at least 10 characters")

    root = Path(repo_root or REPO_ROOT).resolve()
    fresh = extract_partition_for_fixture(fixture)

    if fixture.golden_path.is_file():
        previous = load_partition(fixture.golden_path)
        diff = compare_partitions(previous, fresh, verify_fingerprints=True)
        print(format_partition_diff_report(diff))
        print("")
        print(f"Overwriting golden: {fixture.golden_path}")

    sha, dirty = _git_bless_meta(root)
    fresh.bless_meta = {
        "blessed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason.strip(),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    dump_partition(fresh, fixture.golden_path)

    blessings_log = root / "eval" / "regression" / "BLESSINGS.md"
    blessings_log.parent.mkdir(parents=True, exist_ok=True)
    with blessings_log.open("a", encoding="utf-8") as fh:
        fh.write(
            f"- {fresh.bless_meta['blessed_at']}  **{fixture.fixture_id}**  "
            f"sha={sha} dirty={dirty}  - {reason.strip()}\n"
        )

    return fresh
