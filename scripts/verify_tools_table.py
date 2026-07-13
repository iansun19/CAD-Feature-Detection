#!/usr/bin/env python3
"""Read-only verification of the Supabase ``tools`` table after ingest."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_bootstrap  # noqa: F401, E402 - loads .env

from tools.tool_store import TOOLS_TABLE, create_supabase_client  # noqa: E402

EXPECTED_UNIQUE_TOOLS = 14_931
ROW_COUNT_TOLERANCE = 50
METRIC_DRILL_MIN_MM = 0.5
METRIC_DRILL_MAX_MM = 50.0
METRIC_DRILL_LIBRARY_PATTERN = "%Sandvik%Drills%Metric%"
EXPECTED_TYPES = ("drill", "endmill", "tap", "slot_mill", "countersink")


@dataclass
class CheckResult:
    label: str
    status: str  # PASS | FAIL | WARN
    lines: list[str] = field(default_factory=list)


def _print_check(result: CheckResult) -> None:
    print(f"\n[{result.status}] {result.label}")
    for line in result.lines:
        print(f"  {line}")


def _fetch_all_rows(client: Any, columns: str) -> list[dict[str, Any]]:
    """Paginate through the tools table."""
    rows: list[dict[str, Any]] = []
    page_size = 1000
    offset = 0
    while True:
        response = (
            client.table(TOOLS_TABLE)
            .select(columns)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _fetch_count(client: Any) -> int:
    response = client.table(TOOLS_TABLE).select("*", count="exact").limit(1).execute()
    return int(response.count or 0)


def _sample_spanning(rows: list[dict[str, Any]], sample_size: int = 15) -> list[dict[str, Any]]:
    if len(rows) <= sample_size:
        return rows
    indices = [round(i * (len(rows) - 1) / (sample_size - 1)) for i in range(sample_size)]
    seen: set[int] = set()
    sample: list[dict[str, Any]] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            sample.append(rows[idx])
    return sample


def _looks_double_converted_mm(diameter_mm: float) -> bool:
    """True when diameter_mm is about 25.4x too large for a typical metric tool size."""
    return diameter_mm > METRIC_DRILL_MAX_MM * 20


def _looks_underconverted_mm(diameter_mm: float) -> bool:
    """True when diameter_mm is about 25.4x too small (inch value stored as mm)."""
    return 0 < diameter_mm < METRIC_DRILL_MIN_MM / 5


def _matches_sandvik_metric_library(source_library: str | None) -> bool:
    key = str(source_library or "").lower()
    return "sandvik" in key and "drills" in key and "metric" in key


def check_metric_conversion(all_rows: list[dict[str, Any]]) -> CheckResult:
    result = CheckResult(label="Metric conversion correctness", status="PASS")
    critical_failures: list[str] = []

    sandvik_rows = sorted(
        [row for row in all_rows if _matches_sandvik_metric_library(row.get("source_library"))],
        key=lambda row: float(row["diameter_mm"]),
    )
    if not sandvik_rows:
        result.status = "FAIL"
        critical_failures.append(
            f"no rows matched source_library ILIKE {METRIC_DRILL_LIBRARY_PATTERN!r}"
        )
    else:
        sample = _sample_spanning(sandvik_rows)
        result.lines.append(
            f"Sandvik metric drills sample ({len(sample)} of {len(sandvik_rows)} rows):"
        )
        for row in sample:
            result.lines.append(
                f"  {row.get('name')!r}: diameter_mm={row.get('diameter_mm')}, "
                f"source_unit={row.get('source_unit')!r}"
            )

        out_of_band = [
            row
            for row in sandvik_rows
            if not (
                METRIC_DRILL_MIN_MM
                <= float(row["diameter_mm"])
                <= METRIC_DRILL_MAX_MM
            )
        ]
        if out_of_band:
            critical_failures.append(
                f"{len(out_of_band)} Sandvik metric drill(s) outside "
                f"{METRIC_DRILL_MIN_MM}-{METRIC_DRILL_MAX_MM} mm band"
            )
            for row in out_of_band[:5]:
                critical_failures.append(
                    f"  out of band: {row.get('name')!r} diameter_mm={row.get('diameter_mm')}"
                )

    metric_diameters = [
        float(row["diameter_mm"])
        for row in all_rows
        if str(row.get("source_unit")) == "millimeters"
    ]
    inch_diameters = [
        float(row["diameter_mm"])
        for row in all_rows
        if str(row.get("source_unit")) == "inches"
    ]

    if metric_diameters:
        result.lines.append(
            f"All metric tools (source_unit=millimeters): "
            f"min diameter_mm={min(metric_diameters):.4f}, max diameter_mm={max(metric_diameters):.4f}"
        )
        double_convert = [d for d in metric_diameters if _looks_double_converted_mm(d)]
        under_convert = [d for d in metric_diameters if _looks_underconverted_mm(d)]
        if double_convert:
            critical_failures.append(
                f"{len(double_convert)} metric tool(s) look 25.4x too large "
                f"(max suspicious={max(double_convert):.4f} mm)"
            )
        if under_convert:
            critical_failures.append(
                f"{len(under_convert)} metric tool(s) look 25.4x too small "
                f"(min suspicious={min(under_convert):.4f} mm)"
            )
    else:
        critical_failures.append("no tools with source_unit=millimeters found")

    if inch_diameters:
        result.lines.append(
            f"All inch tools (source_unit=inches): "
            f"min diameter_mm={min(inch_diameters):.4f}, max diameter_mm={max(inch_diameters):.4f}"
        )
    else:
        result.lines.append("WARN: no tools with source_unit=inches found")

    if critical_failures:
        result.status = "FAIL"
        result.lines.extend(critical_failures)

    return result


def check_type_distribution(all_rows: list[dict[str, Any]]) -> CheckResult:
    result = CheckResult(label="Type distribution (normalization)", status="PASS")
    counts = Counter(str(row.get("tool_type")) for row in all_rows)
    result.lines.append(f"{'tool_type':<20} {'count':>8}")
    result.lines.append(f"{'-' * 20} {'-' * 8}")
    for tool_type, count in counts.most_common():
        result.lines.append(f"{tool_type:<20} {count:>8}")

    if counts.get("unknown", 0) > 0:
        result.status = "FAIL"
        result.lines.append(f"FAIL: unknown tool_type count={counts['unknown']}")
    if counts.get("holder", 0) > 0:
        result.status = "FAIL"
        result.lines.append(f"FAIL: holder tool_type count={counts['holder']}")

    missing = [name for name in EXPECTED_TYPES if counts.get(name, 0) == 0]
    if missing:
        result.lines.append(f"WARN: expected types missing or zero: {', '.join(missing)}")

    return result


def _scan_local_guid_collisions(library_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Find guids exported in more than one local Fusion library file."""
    from tools.tool_store import prepare_tool_rows_from_library

    by_guid: dict[str, list[dict[str, str]]] = defaultdict(list)
    if not library_dir.is_dir():
        return {}

    paths = sorted(
        [*library_dir.glob("*.json"), *library_dir.glob("*.tools")],
        key=lambda p: p.name.lower(),
    )
    for path in paths:
        prepared = prepare_tool_rows_from_library(path)
        if prepared.error:
            continue
        for row in prepared.rows:
            guid = str(row.get("guid") or "")
            if not guid:
                continue
            by_guid[guid].append(
                {
                    "source_library": str(row.get("source_library") or prepared.source_library),
                    "name": str(row.get("name") or ""),
                    "file": path.name,
                }
            )
    return {guid: entries for guid, entries in by_guid.items() if len(entries) > 1}


def check_row_count_and_guid_integrity(
    all_rows: list[dict[str, Any]],
    total: int,
    *,
    library_dir: Path | None = None,
) -> CheckResult:
    result = CheckResult(label="Row count + guid integrity", status="PASS")
    result.lines.append(f"Total rows: {total}")

    delta = abs(total - EXPECTED_UNIQUE_TOOLS)
    if delta > ROW_COUNT_TOLERANCE:
        result.lines.append(
            f"WARN: row count differs from expected ~{EXPECTED_UNIQUE_TOOLS} by {delta}"
        )
    else:
        result.lines.append(
            f"Within tolerance of expected ~{EXPECTED_UNIQUE_TOOLS} (+/- {ROW_COUNT_TOLERANCE})"
        )

    by_guid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        guid = row.get("guid")
        if guid:
            by_guid[str(guid)].append(row)

    collisions = {guid: entries for guid, entries in by_guid.items() if len(entries) > 1}
    if collisions:
        result.lines.append(
            f"WARN: unexpected duplicate guids in DB table itself: {len(collisions)}"
        )

    local_collisions = (
        _scan_local_guid_collisions(library_dir)
        if library_dir is not None
        else {}
    )
    result.lines.append(
        f"Cross-library guid duplicates in local exports: {len(local_collisions)} "
        f"(row_upserts gap vs DB unique guids: {14936 - len(by_guid)})"
    )

    if not local_collisions:
        if library_dir is None:
            result.lines.append(
                "WARN: local_tool_libraries not scanned; pass --library-dir to analyze guid collisions"
            )
        else:
            result.lines.append("No cross-library guid collisions found in local exports.")
        return result

    same_tool = 0
    distinct_tool = 0
    for guid, entries in sorted(local_collisions.items()):
        libraries = sorted({str(e.get("source_library")) for e in entries})
        names = [str(e.get("name") or "") for e in entries]
        names_match = len(set(names)) <= 1 or all(
            names[0].strip().lower() == n.strip().lower() for n in names
        )
        if names_match:
            same_tool += 1
            verdict = "same-tool duplicate (benign cross-library)"
        else:
            distinct_tool += 1
            verdict = "DISTINCT tools sharing guid (data loss risk)"

        result.lines.append(f"  guid={guid}")
        result.lines.append(f"    libraries: {libraries}")
        result.lines.append(f"    names: {names}")
        result.lines.append(f"    files: {[e.get('file') for e in entries]}")
        result.lines.append(f"    verdict: {verdict}")

    result.lines.append(
        f"Collision summary: {same_tool} benign duplicate(s), "
        f"{distinct_tool} distinct-tool collision(s)"
    )
    if distinct_tool > 0:
        result.lines.append(
            "WARN: distinct tools share guids - last upsert wins in DB (review collisions)"
        )

    return result


def run_checks(
    client: Any | None = None,
    *,
    library_dir: Path | None = None,
) -> tuple[list[CheckResult], bool]:
    sb = create_supabase_client(client)
    all_rows = _fetch_all_rows(
        sb,
        "guid,source_library,name,tool_type,diameter_mm,source_unit",
    )
    total = _fetch_count(sb)
    results = [
        check_metric_conversion(all_rows),
        check_type_distribution(all_rows),
        check_row_count_and_guid_integrity(all_rows, total, library_dir=library_dir),
    ]
    critical_ok = results[0].status != "FAIL" and results[1].status != "FAIL"
    return results, critical_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Supabase tools table (read-only).")
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=REPO_ROOT / "local_tool_libraries",
        help="Local Fusion exports for cross-library guid collision analysis",
    )
    args = parser.parse_args()

    print("Tools table verification (read-only)")
    print("=" * 60)

    try:
        library_dir = args.library_dir if args.library_dir.is_dir() else None
        results, critical_ok = run_checks(library_dir=library_dir)
    except Exception as exc:
        print(f"\n[FAIL] Could not run verification: {exc}")
        return 1

    for result in results:
        _print_check(result)

    print("\n" + "=" * 60)
    if critical_ok:
        print("OVERALL: PASS (all critical checks passed)")
        return 0

    print("OVERALL: FAIL (one or more critical checks failed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
