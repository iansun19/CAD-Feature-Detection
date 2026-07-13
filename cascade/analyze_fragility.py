#!/usr/bin/env python3
"""Cross-reference cascade instrumentation artifacts for fragility signals.

Reads pass_claims.json and contested_faces.json under pipeline_out/<part>/instrument/
and reports where tie-broken lobe-tier assignments overlap geometry-only contested faces.

Usage:
  python analyze_fragility.py --part 96260B_rear
  python analyze_fragility.py --instrument-dir pipeline_out/96260B_rear/instrument
  python analyze_fragility.py --part 96260B_rear --no-specific-only
  python analyze_fragility.py --part 96260B_rear --focus-pair holes,pockets
  python analyze_fragility.py --part 96260B_rear --focus-pair flats,pockets
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from env_bootstrap import REPO_ROOT

TIER_ASSIGNMENTS = (
    "region_grow",
    "annexed_fillet",
    "overlap_resolved",
    "orphan_reassigned",
)
TIE_BROKEN = frozenset({"overlap_resolved", "orphan_reassigned"})

# Generic catch-all passes over-claim on the full dry-run pool; exclude from primary
# contested signal unless --no-specific-only.
GENERIC_CATCHALL_PASSES = frozenset({
    "residual_candidates",
    "contour",
    "wall",
    "profile",
})

CONTESTED_SATURATION_FRACTION = 0.50

# Rear-plate sanity reference only (not used for pass/fail).
REAR_PLATE_HISTOGRAM_REF = {
    "region_grow": 120,
    "annexed_fillet": 24,
    "overlap_resolved": 33,
    "orphan_reassigned": 33,
}

# Pocket-claim fields copied when present; radius/length are not in current artifacts.
POCKET_CLAIM_FIELDS = (
    "class",
    "tier_label",
    "mouth_axial_mm",
    "deep_axial_mm",
)
UNAVAILABLE_POCKET_GEOMETRY = (
    "cylinder_radius",
    "face_axial_length",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _resolve_instrument_dir(args: argparse.Namespace) -> Path:
    if args.instrument_dir is not None:
        return Path(args.instrument_dir).expanduser().resolve()
    if args.part is None:
        raise SystemExit("Provide --part NAME or --instrument-dir PATH.")
    return (REPO_ROOT / "pipeline_out" / args.part / "instrument").resolve()


def _total_face_count(pass_claims: dict[str, Any]) -> int | None:
    passes = pass_claims.get("passes")
    if isinstance(passes, list) and passes:
        first = passes[0]
        if isinstance(first, dict) and first.get("input_pool_size") is not None:
            return int(first["input_pool_size"])
    face_owner = pass_claims.get("face_owner")
    if isinstance(face_owner, dict):
        return len(face_owner)
    return None


def _extract_tier_records(pass_claims: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map face index -> tier metadata from pocket pass claims."""
    records: dict[int, dict[str, Any]] = {}
    passes = pass_claims.get("passes")
    if not isinstance(passes, list):
        return records

    for pass_rec in passes:
        if not isinstance(pass_rec, dict) or pass_rec.get("pass") != "pockets":
            continue
        claimed = pass_rec.get("claimed")
        if not isinstance(claimed, dict):
            continue
        for face_str, entry in claimed.items():
            if not isinstance(entry, dict):
                continue
            assignment = entry.get("tier_assignment")
            if assignment is None:
                continue
            face_id = int(face_str)
            if face_id in records:
                prev = records[face_id].get("tier_assignment")
                if prev != assignment:
                    records[face_id]["tier_assignment_conflict"] = assignment
                continue
            records[face_id] = {
                "tier_assignment": assignment,
                "band": entry.get("band"),
                "lobe_id": entry.get("lobe_id"),
                "band_axial_mm": entry.get("band_axial_mm"),
            }
    return records


def _extract_pocket_claims(pass_claims: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map face index -> pocket pass claim entry from face_owner."""
    claims: dict[int, dict[str, Any]] = {}
    face_owner = pass_claims.get("face_owner")
    if not isinstance(face_owner, dict):
        return claims
    for face_str, entry in face_owner.items():
        if isinstance(entry, dict) and entry.get("pass") == "pockets":
            claims[int(face_str)] = entry
    return claims


def _pocket_geometry_fields(pocket_claim: dict[str, Any] | None) -> dict[str, Any]:
    """Copy artifact fields and derived axial offsets; never fabricate radius/length."""
    out: dict[str, Any] = {
        field: None for field in UNAVAILABLE_POCKET_GEOMETRY
    }
    out["geometry_note"] = (
        f"{', '.join(UNAVAILABLE_POCKET_GEOMETRY)} not recorded in pass_claims.json"
    )
    if not pocket_claim:
        return out

    for field in POCKET_CLAIM_FIELDS:
        if field in pocket_claim:
            out[field] = pocket_claim[field]

    mouth = pocket_claim.get("mouth_axial_mm")
    deep = pocket_claim.get("deep_axial_mm")
    band_axial = pocket_claim.get("band_axial_mm")
    if isinstance(mouth, (int, float)) and isinstance(deep, (int, float)):
        out["lobe_axial_span_mm"] = round(abs(float(deep) - float(mouth)), 4)
    if isinstance(mouth, (int, float)) and isinstance(band_axial, (int, float)):
        out["band_offset_from_mouth_mm"] = round(
            abs(float(band_axial) - float(mouth)), 4,
        )
    return out


def _parse_focus_pair(raw: str) -> tuple[str, str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(
            f"--focus-pair must be two comma-separated pass names, got {raw!r}"
        )
    return parts[0], parts[1]


def _focus_pair_key(pass_a: str, pass_b: str) -> str:
    a, b = sorted([pass_a, pass_b])
    return f"{a}-vs-{b}"


def _focus_subset_json_key(pass_a: str, pass_b: str) -> str:
    a, b = sorted([pass_a, pass_b])
    return f"focus_subset_{a}_vs_{b}_tiebroken"


def _focus_subset_faces(
    tie_broken: set[int],
    contested_specific: dict[int, list[str]],
    pass_a: str,
    pass_b: str,
) -> set[int]:
    required = {pass_a, pass_b}
    return {
        face_id
        for face_id in tie_broken
        if required <= set(contested_specific.get(face_id, []))
    }


def _count_lobes(tier_records: dict[int, dict[str, Any]]) -> int | None:
    lobe_ids = {
        rec["lobe_id"]
        for rec in tier_records.values()
        if rec.get("lobe_id") is not None
    }
    if not lobe_ids:
        return None
    return len(lobe_ids)


def _build_focus_subset_rows(
    face_ids: set[int],
    *,
    tier_records: dict[int, dict[str, Any]],
    contested_specific: dict[int, list[str]],
    pocket_claims: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for face_id in face_ids:
        tier = tier_records.get(face_id, {})
        passes = contested_specific.get(face_id, [])
        geometry = _pocket_geometry_fields(pocket_claims.get(face_id))
        rows.append(
            {
                "face": face_id,
                "tier_assignment": tier.get("tier_assignment"),
                "band": tier.get("band"),
                "lobe_id": tier.get("lobe_id"),
                "band_axial_mm": tier.get("band_axial_mm"),
                "contested_by": passes,
                **geometry,
            }
        )
    rows.sort(
        key=lambda r: (
            r["lobe_id"] if r["lobe_id"] is not None else 10**9,
            _band_sort_key(r["band"]),
            r["face"],
        )
    )
    return rows


def _interpret_focus_subset(
    rows: list[dict[str, Any]],
    *,
    pass_a: str,
    pass_b: str,
    total_lobes: int | None,
) -> str:
    size = len(rows)
    pair_label = _focus_pair_key(pass_a, pass_b)
    if size <= 2:
        return (
            f"FOCUS INTERPRETATION ({pair_label} & tie-broken, n={size}): "
            "the two uncertainty axes don't coincide; "
            "pocket-before-hole ordering has margin here (reassuring)."
        )

    by_lobe = Counter(r["lobe_id"] for r in rows)
    by_band = Counter(r["band"] for r in rows)
    n_lobes = len(by_lobe)
    dominant_lobe = by_lobe.most_common(1)[0][1] if by_lobe else 0
    most_lobes_threshold = (
        max(4, (total_lobes * 3 + 4) // 5) if total_lobes else 4
    )

    if n_lobes == 1 or len(by_band) == 1 or dominant_lobe >= max(3, size // 2):
        return (
            f"FOCUS INTERPRETATION ({pair_label} & tie-broken, n={size}): "
            "subset clusters in one lobe or one band - a localized ambiguity to inspect first."
        )
    if total_lobes and n_lobes >= most_lobes_threshold:
        return (
            f"FOCUS INTERPRETATION ({pair_label} & tie-broken, n={size}): "
            f"subset spread across {n_lobes}/{total_lobes} lobes - "
            f"the {pass_a}/{pass_b} boundary is systematically soft and "
            "arbitration (not just ordering) is the real fix."
        )
    if n_lobes >= 4:
        return (
            f"FOCUS INTERPRETATION ({pair_label} & tie-broken, n={size}): "
            f"subset spread across {n_lobes} lobes - "
            f"the {pass_a}/{pass_b} boundary is systematically soft and "
            "arbitration (not just ordering) is the real fix."
        )
    return (
        f"FOCUS INTERPRETATION ({pair_label} & tie-broken, n={size}): "
        "moderate overlap - review subset rows above."
    )


def _print_focus_subset_block(
    rows: list[dict[str, Any]],
    *,
    pass_a: str,
    pass_b: str,
    pair_key: str,
) -> None:
    print("=" * 72)
    print(f"STEP 5 - focus subset ({pair_key} & tie-broken)")
    print("=" * 72)
    print(f"|focus subset| = {len(rows)}")
    print(
        f"NOTE: {', '.join(UNAVAILABLE_POCKET_GEOMETRY)} not recorded in "
        "pass_claims.json; showing axial band fields and pocket class only."
    )
    print()

    if not rows:
        print("  (no focus subset faces)")
        print()
        return

    print(
        f"{'face':>5}  {'tier_assignment':<18}  {'band':<5}  {'lobe':>4}  "
        f"{'band_axial':>10}  {'mouth_ax':>9}  {'deep_ax':>9}  "
        f"{'span':>6}  {'offset':>6}  class  contested_by"
    )
    for row in rows:
        def _fmt(val: Any, width: int = 10) -> str:
            if isinstance(val, (int, float)):
                return f"{val:>{width}.4f}"
            return f"{'-':>{width}}"

        lobe_s = str(row["lobe_id"]) if row["lobe_id"] is not None else "?"
        class_s = str(row.get("class") or row.get("tier_label") or "-")
        passes_s = ", ".join(row["contested_by"])
        print(
            f"{row['face']:5d}  {str(row['tier_assignment']):<18}  "
            f"{str(row['band']):<5}  {lobe_s:>4}  "
            f"{_fmt(row.get('band_axial_mm'))}  "
            f"{_fmt(row.get('mouth_axial_mm'), 9)}  "
            f"{_fmt(row.get('deep_axial_mm'), 9)}  "
            f"{_fmt(row.get('lobe_axial_span_mm'), 6)}  "
            f"{_fmt(row.get('band_offset_from_mouth_mm'), 6)}  "
            f"{class_s}  {passes_s}"
        )
    print()

    by_lobe = Counter(r["lobe_id"] for r in rows)
    by_band = Counter(r["band"] for r in rows)
    print("Focus subset grouped by lobe_id:")
    for lobe_id, count in sorted(by_lobe.items(), key=lambda x: (x[0] is None, x[0])):
        label = "?" if lobe_id is None else str(lobe_id)
        print(f"  lobe {label}: {count}")
    print()
    print("Focus subset grouped by band:")
    for band, count in sorted(by_band.items(), key=lambda x: _band_sort_key(x[0])):
        print(f"  {band}: {count}")
    print()


def _contested_set(contested_faces: dict[str, Any], variant: str) -> dict[int, list[str]]:
    variants = contested_faces.get("variants")
    if not isinstance(variants, dict):
        return {}
    block = variants.get(variant)
    if not isinstance(block, dict):
        return {}
    raw = block.get("contested_faces")
    if not isinstance(raw, dict):
        return {}
    out: dict[int, list[str]] = {}
    for face_str, passes in raw.items():
        if isinstance(passes, list):
            out[int(face_str)] = [str(p) for p in passes]
    return out


def _filter_specific_contested(
    contested: dict[int, list[str]],
    *,
    exclude: frozenset[str] = GENERIC_CATCHALL_PASSES,
) -> dict[int, list[str]]:
    """Keep faces with 2+ specific recognizers contesting after dropping catch-alls."""
    out: dict[int, list[str]] = {}
    for face_id, passes in contested.items():
        specific = [p for p in passes if p not in exclude]
        if len(specific) >= 2:
            out[face_id] = specific
    return out


def _pass_pair_key(passes: list[str]) -> str:
    keys = _pass_pair_keys(passes)
    return keys[0] if keys else "unknown"


def _pass_pair_keys(passes: list[str]) -> list[str]:
    uniq = sorted(set(passes))
    if len(uniq) < 2:
        return ["+".join(uniq) if uniq else "unknown"]
    return [f"{a}-vs-{b}" for a, b in combinations(uniq, 2)]


def _band_sort_key(band: Any) -> tuple[int, str]:
    if band == "mouth":
        return (0, "mouth")
    if band == "deep":
        return (1, "deep")
    return (2, str(band))


def _count_by_pair(rows: list[dict[str, Any]]) -> Counter[str]:
    by_pair: Counter[str] = Counter()
    for row in rows:
        for pair in row.get("contest_pass_pairs") or [_pass_pair_key(row["contested_by"])]:
            by_pair[pair] += 1
    return by_pair


def _lobe_grouping(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_lobe = Counter(r["lobe_id"] for r in rows)
    return {
        str(k) if k is not None else "null": v
        for k, v in sorted(by_lobe.items(), key=lambda x: (x[0] is None, x[0]))
    }


def _contested_fraction(contested_size: int, total_faces: int | None) -> float | None:
    if total_faces is None or total_faces <= 0:
        return None
    return contested_size / total_faces


def _is_saturated(contested_size: int, total_faces: int | None) -> bool:
    frac = _contested_fraction(contested_size, total_faces)
    return frac is not None and frac > CONTESTED_SATURATION_FRACTION


def _interpret_intersection(
    size: int,
    *,
    by_lobe: Counter[int | None],
    by_pair: Counter[str],
    contested_saturated: bool,
) -> str:
    if contested_saturated:
        return (
            "INTERPRETATION: contested set is saturated - signal is non-discriminating, "
            "treat intersection as unreliable."
        )
    if size <= 2:
        return (
            f"INTERPRETATION: intersection size {size} is reassuring; "
            "tie-breaking and contested overlaps largely do not coincide."
        )

    dominant_lobe_count = by_lobe.most_common(1)[0][1] if by_lobe else 0
    dominant_pair_count = by_pair.most_common(1)[0][1] if by_pair else 0
    n_lobes = len(by_lobe)

    if n_lobes == 1 or dominant_lobe_count >= max(3, size // 2):
        return (
            f"INTERPRETATION: intersection size {size} indicates localized fragility; "
            "faces cluster at one lobe boundary (likely tunable)."
        )
    if len(by_pair) == 1 or dominant_pair_count >= max(3, size // 2):
        pair = by_pair.most_common(1)[0][0] if by_pair else "unknown"
        return (
            f"INTERPRETATION: intersection size {size} indicates localized fragility; "
            f"dominant contesting pass-pair ({pair}) suggests a specific ordering conflict."
        )
    if n_lobes >= 4:
        return (
            f"INTERPRETATION: intersection size {size} is spread across {n_lobes} lobes; "
            "suggests a systematic tolerance issue rather than one boundary."
        )
    return (
        f"INTERPRETATION: intersection size {size} shows moderate overlap; "
        "review intersection rows and groupings above."
    )


def _build_intersection_rows(
    face_ids: set[int],
    tier_records: dict[int, dict[str, Any]],
    contested: dict[int, list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for face_id in face_ids:
        tier = tier_records.get(face_id, {})
        passes = contested.get(face_id, [])
        rows.append(
            {
                "face": face_id,
                "tier_assignment": tier.get("tier_assignment"),
                "band": tier.get("band"),
                "lobe_id": tier.get("lobe_id"),
                "band_axial_mm": tier.get("band_axial_mm"),
                "contested_by": passes,
                "contest_pass_pair": _pass_pair_key(passes),
                "contest_pass_pairs": _pass_pair_keys(passes),
            }
        )
    rows.sort(
        key=lambda r: (
            r["lobe_id"] if r["lobe_id"] is not None else 10**9,
            _band_sort_key(r["band"]),
            r["face"],
        )
    )
    return rows


def _analysis_block(
    *,
    label: str,
    tie_broken: set[int],
    contested: dict[int, list[str]],
    tier_records: dict[int, dict[str, Any]],
    total_faces: int | None,
) -> dict[str, Any]:
    intersection_ids = tie_broken & set(contested)
    rows = _build_intersection_rows(intersection_ids, tier_records, contested)
    by_pair = _count_by_pair(rows)
    by_lobe = Counter(r["lobe_id"] for r in rows)
    contested_size = len(contested)
    frac_tb = len(rows) / len(tie_broken) if tie_broken else 0.0
    contested_frac = _contested_fraction(contested_size, total_faces)
    return {
        "label": label,
        "contested_size": contested_size,
        "contested_fraction_of_all_faces": (
            round(contested_frac, 6) if contested_frac is not None else None
        ),
        "contested_saturated": _is_saturated(contested_size, total_faces),
        "intersection_size": len(rows),
        "intersection_fraction_of_tie_broken": round(frac_tb, 6),
        "rows": rows,
        "by_pair": by_pair,
        "by_lobe": by_lobe,
        "grouping_by_lobe_id": _lobe_grouping(rows),
        "grouping_by_contest_pass_pair": dict(by_pair.most_common()),
    }


def _print_step1(tier_records: dict[int, dict[str, Any]]) -> tuple[Counter[str], bool]:
    print("=" * 72)
    print("STEP 1 - data integrity")
    print("=" * 72)

    if not tier_records:
        print(
            "ERROR: no tier_assignment fields found in pass_claims.json.\n"
            "Regenerate instrumentation with --instrument (tier trace requires pocket lobe tiers)."
        )
        return Counter(), False

    overlap_faces = {
        fid for fid, rec in tier_records.items()
        if rec.get("tier_assignment") == "overlap_resolved"
    }
    orphan_faces = {
        fid for fid, rec in tier_records.items()
        if rec.get("tier_assignment") == "orphan_reassigned"
    }
    both = sorted(overlap_faces & orphan_faces)
    conflicts = sorted(
        fid for fid, rec in tier_records.items()
        if "tier_assignment_conflict" in rec
    )

    histogram = Counter(
        rec["tier_assignment"] for rec in tier_records.values()
        if rec.get("tier_assignment") in TIER_ASSIGNMENTS
    )

    print(f"Faces with tier_assignment: {len(tier_records)}")
    print()
    print("tier_assignment histogram:")
    for key in TIER_ASSIGNMENTS:
        count = histogram.get(key, 0)
        ref = REAR_PLATE_HISTOGRAM_REF.get(key)
        ref_note = f"  (rear-plate ref: {ref})" if ref is not None else ""
        print(f"  {key:20s} {count:4d}{ref_note}")
    print()

    ok = True
    if both:
        ok = False
        print("*** TRACE BUG: faces in BOTH overlap_resolved AND orphan_reassigned ***")
        print(f"    count={len(both)}  faces={both}")
        print("    Each face must have exactly one final tier_assignment.")
        print()
    else:
        print(
            "overlap_resolved & orphan_reassigned: empty (OK - sets are disjoint)"
        )
        print(f"  |overlap_resolved|   = {len(overlap_faces)}")
        print(f"  |orphan_reassigned| = {len(orphan_faces)}")
        print()

    if conflicts:
        ok = False
        print("*** TRACE BUG: duplicate tier_assignment values for same face ***")
        print(f"    faces={conflicts}")
        print()

    return histogram, ok


def _print_intersection_block(
    title: str,
    block: dict[str, Any],
    *,
    tie_broken_size: int,
) -> None:
    rows = block["rows"]
    contested_size = block["contested_size"]
    size = block["intersection_size"]
    frac = block["intersection_fraction_of_tie_broken"]
    print(title)
    print("-" * len(title))
    print(f"|intersection| = {size}")
    if block["contested_saturated"]:
        print(
            "  Contested set exceeds 50% of all faces - signal is non-discriminating."
        )
    elif size <= 2:
        print(
            "  Empty/tiny intersection is reassuring: tie-breaking and contested "
            "overlaps largely do not coincide."
        )
    else:
        print(
            "  Non-trivial intersection: tie-broken tier faces also contested "
            "by multiple passes - candidate fragility hotspots."
        )
    print()
    print(
        f"  |tie-broken|={tie_broken_size}  "
        f"|contested|={contested_size}  "
        f"|intersection|={size}  "
        f"intersection/tie-broken={frac:.1%}"
    )
    if block["contested_fraction_of_all_faces"] is not None:
        print(
            f"  contested/all_faces={block['contested_fraction_of_all_faces']:.1%}"
        )
    print()

    if not rows:
        print("  (no intersection faces)")
        print()
        return

    print(
        f"{'face':>5}  {'tier_assignment':<18}  {'band':<5}  "
        f"{'lobe':>4}  {'band_axial_mm':>13}  contested_by"
    )
    for row in rows:
        axial = row["band_axial_mm"]
        axial_s = f"{axial:.4f}" if isinstance(axial, (int, float)) else str(axial)
        lobe = row["lobe_id"]
        lobe_s = str(lobe) if lobe is not None else "?"
        passes_s = ", ".join(row["contested_by"])
        print(
            f"{row['face']:5d}  {row['tier_assignment']:<18}  "
            f"{str(row['band']):<5}  {lobe_s:>4}  {axial_s:>13}  {passes_s}"
        )
    print()


def _print_groupings(block: dict[str, Any], *, header: str) -> None:
    print(header)
    by_pair = block["by_pair"]
    if by_pair:
        for pair, count in by_pair.most_common():
            print(f"  {pair}: {count}")
    else:
        print("  (none)")
    print()

    print("Intersection grouped by lobe_id:")
    grouping = block["grouping_by_lobe_id"]
    if grouping:
        for lobe_id, count in grouping.items():
            label = "?" if lobe_id == "null" else lobe_id
            print(f"  lobe {label}: {count}")
    else:
        print("  (none)")
    print()


def analyze(
    instrument_dir: Path,
    *,
    specific_only: bool = True,
    focus_pair: tuple[str, str] = ("holes", "pockets"),
) -> int:
    pass_claims_path = instrument_dir / "pass_claims.json"
    contested_path = instrument_dir / "contested_faces.json"

    missing = [p.name for p in (pass_claims_path, contested_path) if not p.is_file()]
    if missing:
        print(
            f"Missing instrumentation artifact(s) in {instrument_dir}:\n"
            f"  {', '.join(missing)}\n"
            "Run the cascade with --instrument to generate pass_claims.json and "
            "contested_faces.json, then re-run this script."
        )
        return 1

    pass_claims = _load_json(pass_claims_path)
    contested_faces = _load_json(contested_path)

    print(f"Instrument dir: {instrument_dir}")
    print(f"Specific-only contested filter: {'on' if specific_only else 'off'}")
    if specific_only:
        print(f"  Excluded catch-all passes: {', '.join(sorted(GENERIC_CATCHALL_PASSES))}")
    focus_a, focus_b = focus_pair
    print(f"Focus pair: {_focus_pair_key(focus_a, focus_b)}")
    print()

    tier_records = _extract_tier_records(pass_claims)
    pocket_claims = _extract_pocket_claims(pass_claims)
    histogram, integrity_ok = _print_step1(tier_records)
    if not histogram:
        return 1

    if not integrity_ok:
        print(
            "WARNING: integrity checks failed - results below may reflect a trace bug, "
            "not real geometric fragility."
        )
        print()

    total_faces = _total_face_count(pass_claims)
    tie_broken = {
        fid for fid, rec in tier_records.items()
        if rec.get("tier_assignment") in TIE_BROKEN
    }

    contested_all = _contested_set(contested_faces, "no_context")
    contested_ctx = _contested_set(contested_faces, "context_influenced")
    contested_specific = _filter_specific_contested(contested_all)

    if not contested_all and "variants" not in contested_faces:
        print(
            "ERROR: contested_faces.json lacks variants.no_context.\n"
            "Regenerate instrumentation with --instrument."
        )
        return 1

    all_block = _analysis_block(
        label="all_passes_no_context",
        tie_broken=tie_broken,
        contested=contested_all,
        tier_records=tier_records,
        total_faces=total_faces,
    )
    specific_block = _analysis_block(
        label="specific_contested_no_context",
        tie_broken=tie_broken,
        contested=contested_specific,
        tier_records=tier_records,
        total_faces=total_faces,
    )
    primary = specific_block if specific_only else all_block
    comparison = all_block if specific_only else specific_block

    rows_ctx = _build_intersection_rows(
        tie_broken & set(contested_ctx),
        tier_records,
        contested_ctx,
    )

    residual_noise_faces = sorted(
        set(r["face"] for r in all_block["rows"])
        - set(r["face"] for r in specific_block["rows"])
    )

    focus_ids = _focus_subset_faces(
        tie_broken, contested_specific, focus_a, focus_b,
    )
    focus_rows = _build_focus_subset_rows(
        focus_ids,
        tier_records=tier_records,
        contested_specific=contested_specific,
        pocket_claims=pocket_claims,
    )
    total_lobes = _count_lobes(tier_records)
    focus_json_key = _focus_subset_json_key(focus_a, focus_b)
    focus_pair_key = _focus_pair_key(focus_a, focus_b)

    print("=" * 72)
    print("STEP 2 - signal sets")
    print("=" * 72)
    print(f"|tie-broken| (overlap_resolved | orphan_reassigned) = {len(tie_broken)}")
    if total_faces is not None:
        print(f"|all faces| (from pass_claims pool)                  = {total_faces}")
    print()
    print("Contested sets (no_context):")
    print(f"  all passes (comparison)     = {all_block['contested_size']}", end="")
    if all_block["contested_fraction_of_all_faces"] is not None:
        print(f"  ({all_block['contested_fraction_of_all_faces']:.1%} of all faces)", end="")
    if all_block["contested_saturated"]:
        print("  [SATURATED]", end="")
    print()
    print(f"  specific-only (primary)       = {specific_block['contested_size']}", end="")
    if specific_block["contested_fraction_of_all_faces"] is not None:
        print(
            f"  ({specific_block['contested_fraction_of_all_faces']:.1%} of all faces)",
            end="",
        )
    if specific_block["contested_saturated"]:
        print("  [SATURATED]", end="")
    print()
    print(f"  context_influenced (compare)  = {len(contested_ctx)}")
    print()

    print("=" * 72)
    primary_title = (
        "STEP 3 - primary intersection (tie-broken & specific-contested no_context)"
        if specific_only
        else "STEP 3 - primary intersection (tie-broken & all-pass contested no_context)"
    )
    print(primary_title)
    print("=" * 72)
    _print_intersection_block(
        "Primary intersection",
        primary,
        tie_broken_size=len(tie_broken),
    )

    print("=" * 72)
    print("STEP 4 - supporting context")
    print("=" * 72)

    print("Comparison - all passes vs specific-only (no_context):")
    print(f"  |tie-broken|                         = {len(tie_broken)}")
    print(
        f"  |contested all passes|               = {all_block['contested_size']}"
        + (
            f"  ({all_block['contested_fraction_of_all_faces']:.1%} of all faces)"
            if all_block["contested_fraction_of_all_faces"] is not None
            else ""
        )
    )
    print(
        f"  |contested specific-only|            = {specific_block['contested_size']}"
        + (
            f"  ({specific_block['contested_fraction_of_all_faces']:.1%} of all faces)"
            if specific_block["contested_fraction_of_all_faces"] is not None
            else ""
        )
    )
    print(f"  |intersection all passes|            = {all_block['intersection_size']}")
    print(f"  |intersection specific-only|         = {specific_block['intersection_size']}")
    print(
        f"  residual-only noise in all-pass ix   = "
        f"{len(residual_noise_faces)} of {all_block['intersection_size']} "
        f"(tie-broken faces contested only via catch-all passes)"
    )
    if residual_noise_faces:
        print(f"    faces: {residual_noise_faces}")
    print()

    print("Context-influenced (noisier, for comparison only):")
    print(f"  |intersection context_influenced| = {len(rows_ctx)}")
    if tie_broken:
        print(
            f"  intersection_ctx / tie-broken     = "
            f"{len(rows_ctx) / len(tie_broken):.1%}"
        )
    print()

    _print_groupings(
        primary,
        header="Primary intersection grouped by contesting pass-pair:",
    )

    if specific_only:
        print("All-pass intersection (comparison, includes catch-all noise):")
        print(f"  |intersection| = {all_block['intersection_size']}")
        _print_groupings(
            all_block,
            header="All-pass intersection grouped by contesting pass-pair:",
        )

    _print_focus_subset_block(
        focus_rows,
        pass_a=focus_a,
        pass_b=focus_b,
        pair_key=focus_pair_key,
    )

    report = {
        "instrument_dir": str(instrument_dir),
        "integrity_ok": integrity_ok,
        "specific_only_primary": specific_only,
        "excluded_catchall_passes": sorted(GENERIC_CATCHALL_PASSES),
        "total_faces": total_faces,
        "tier_assignment_histogram": {k: histogram.get(k, 0) for k in TIER_ASSIGNMENTS},
        "comparison_all_passes_no_context": {
            "description": (
                "Unfiltered no_context contested set; saturated when catch-alls "
                "over-claim on the full dry-run pool."
            ),
            "set_sizes": {
                "tie_broken": len(tie_broken),
                "contested": all_block["contested_size"],
                "contested_fraction_of_all_faces": all_block["contested_fraction_of_all_faces"],
                "contested_saturated": all_block["contested_saturated"],
                "intersection": all_block["intersection_size"],
                "intersection_fraction_of_tie_broken": all_block["intersection_fraction_of_tie_broken"],
            },
            "intersection": all_block["rows"],
            "grouping": {
                "by_contest_pass_pair": all_block["grouping_by_contest_pass_pair"],
                "by_lobe_id": all_block["grouping_by_lobe_id"],
            },
        },
        "primary_specific_contested_no_context": {
            "description": (
                "Specific recognizer vs specific recognizer contests only; "
                "catch-all passes removed."
            ),
            "set_sizes": {
                "tie_broken": len(tie_broken),
                "contested": specific_block["contested_size"],
                "contested_fraction_of_all_faces": specific_block["contested_fraction_of_all_faces"],
                "contested_saturated": specific_block["contested_saturated"],
                "intersection": specific_block["intersection_size"],
                "intersection_fraction_of_tie_broken": specific_block["intersection_fraction_of_tie_broken"],
            },
            "intersection": specific_block["rows"],
            "grouping": {
                "by_contest_pass_pair": specific_block["grouping_by_contest_pass_pair"],
                "by_lobe_id": specific_block["grouping_by_lobe_id"],
            },
            "residual_noise_from_all_passes_intersection": {
                "count": len(residual_noise_faces),
                "all_passes_intersection_size": all_block["intersection_size"],
                "faces": residual_noise_faces,
            },
        },
        "comparison_context_influenced": {
            "description": "Noisier dry-run variant; for comparison only.",
            "set_sizes": {
                "contested": len(contested_ctx),
                "intersection": len(rows_ctx),
            },
            "intersection": rows_ctx,
        },
        focus_json_key: {
            "description": (
                f"Sharpest signal: specific-contested {focus_pair_key} intersect tie-broken."
            ),
            "focus_pair": [focus_a, focus_b],
            "focus_pair_key": focus_pair_key,
            "set_sizes": {
                "tie_broken": len(tie_broken),
                "specific_contested_with_pair": sum(
                    1
                    for passes in contested_specific.values()
                    if {focus_a, focus_b} <= set(passes)
                ),
                "focus_subset": len(focus_rows),
            },
            "geometry_fields_unavailable": list(UNAVAILABLE_POCKET_GEOMETRY),
            "subset": focus_rows,
            "grouping": {
                "by_lobe_id": _lobe_grouping(focus_rows),
                "by_band": {
                    str(k) if k is not None else "null": v
                    for k, v in sorted(
                        Counter(r["band"] for r in focus_rows).items(),
                        key=lambda x: _band_sort_key(x[0]),
                    )
                },
            },
            "viewer_faces": [r["face"] for r in focus_rows],
        },
    }

    out_path = instrument_dir / "fragility_report.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {out_path}")
    print()

    if all_block["contested_saturated"]:
        print(
            "NOTE: all-passes contested set is saturated "
            f"({all_block['contested_fraction_of_all_faces']:.0%} of all faces) - "
            f"the all-pass intersection ({all_block['intersection_size']}) is unreliable."
        )
        if specific_only and all_block["intersection_size"]:
            print(
                f"      {len(residual_noise_faces)} of {all_block['intersection_size']} "
                "all-pass intersection faces were residual-only noise."
            )
        print()

    interpret_saturated = primary["contested_saturated"]
    if specific_only and all_block["contested_saturated"] and not interpret_saturated:
        interpret_saturated = False

    print(
        _interpret_intersection(
            primary["intersection_size"],
            by_lobe=primary["by_lobe"],
            by_pair=primary["by_pair"],
            contested_saturated=interpret_saturated,
        )
    )
    print()
    print(
        _interpret_focus_subset(
            focus_rows,
            pass_a=focus_a,
            pass_b=focus_b,
            total_lobes=total_lobes,
        )
    )
    viewer_list = ", ".join(str(r["face"]) for r in focus_rows)
    print(f"VIEWER FACES: {viewer_list}")

    return 0 if integrity_ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-reference pass_claims and contested_faces instrumentation "
            "for tie-break vs contest fragility signals."
        ),
    )
    parser.add_argument(
        "--part",
        help="Part slug under pipeline_out/<part>/instrument/ (e.g. 96260B_rear).",
    )
    parser.add_argument(
        "--instrument-dir",
        help="Explicit path to instrument/ directory (overrides --part).",
    )
    parser.add_argument(
        "--specific-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Primary contested set drops catch-all passes and keeps only "
            "specific-vs-specific contests (default: on)."
        ),
    )
    parser.add_argument(
        "--focus-pair",
        default="holes,pockets",
        help=(
            "Pass pair for the sharpest focus subset "
            "(default: holes,pockets)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        instrument_dir = _resolve_instrument_dir(args)
        focus_pair = _parse_focus_pair(args.focus_pair)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1

    return analyze(
        instrument_dir,
        specific_only=args.specific_only,
        focus_pair=focus_pair,
    )


if __name__ == "__main__":
    raise SystemExit(main())
