"""Supabase-backed storage for normalized Fusion Tool rows."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from machining_context import (
    Tool,
    ToolPreset,
    _fusion_unit,
    _is_fusion_holder_type,
    _library_name_from_path,
    _positive_length_to_mm,
    _preset_matches_material,
    load_tool_library,
    normalize_tool_type,
)

logger = logging.getLogger(__name__)

TOOLS_TABLE = "tools"
UPSERT_BATCH_SIZE = 500
FETCH_PAGE_SIZE = 1000


class SupabaseConfigError(RuntimeError):
    """Raised when Supabase environment variables are missing."""


@dataclass
class IngestFileResult:
    """Per-file stats from preparing tools for Supabase upsert."""

    path: Path
    source_library: str
    unit_counts: dict[str, int] = field(default_factory=dict)
    tools_ingested: int = 0
    tools_skipped: int = 0
    holders_skipped: int = 0
    unknown_types: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class UpsertResult:
    """Outcome of a batched Supabase upsert."""

    attempted: int = 0
    upserted: int = 0
    chunk_count: int = 0
    failed_chunks: list[str] = field(default_factory=list)


def supabase_env() -> tuple[str, str]:
    """Return (url, key) from environment or raise SupabaseConfigError."""
    from env_bootstrap import load_repo_dotenv

    load_repo_dotenv()
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_KEY", "").strip()
    )
    if not url or not key:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) must be set"
        )
    return url, key


def create_supabase_client(client: Any | None = None) -> Any:
    """Create a Supabase client, or return a caller-provided client (for tests)."""
    if client is not None:
        return client
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ImportError(
            "supabase package is required for Supabase tool storage; "
            "install with: pip install supabase"
        ) from exc
    url, key = supabase_env()
    return create_client(url, key)


def tool_to_row(
    tool: Tool,
    *,
    source_library: str,
    raw_type: str,
    raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a normalized Tool for Supabase ``tools`` upsert."""
    return {
        "guid": tool.guid,
        "tool_id": tool.tool_id,
        "name": tool.name,
        "tool_type": tool.tool_type,
        "raw_type": raw_type or None,
        "diameter_mm": tool.diameter_mm,
        "flute_length_mm": tool.flute_length_mm,
        "max_depth_mm": tool.max_depth_mm,
        "flute_count": tool.flute_count,
        "corner_radius_mm": tool.corner_radius_mm,
        "point_angle_deg": tool.point_angle_deg,
        "tool_material": tool.tool_material,
        "vendor": tool.vendor,
        "source_library": source_library,
        "source_unit": tool.source_unit,
        "presets": [preset.model_dump(mode="json") for preset in tool.presets],
        "raw": dict(raw) if raw is not None else None,
    }


def row_to_tool(row: Mapping[str, Any]) -> Tool:
    """Reconstruct a Tool from a Supabase ``tools`` row."""
    presets_raw = row.get("presets") or []
    if isinstance(presets_raw, str):
        presets_raw = json.loads(presets_raw)
    presets = [ToolPreset.model_validate(item) for item in presets_raw]
    source_library = str(row.get("source_library") or "unknown")

    return Tool(
        tool_id=str(row["tool_id"]),
        tool_type=str(row["tool_type"]),
        diameter_mm=float(row["diameter_mm"]),
        flute_length_mm=(
            float(row["flute_length_mm"]) if row.get("flute_length_mm") is not None else None
        ),
        max_depth_mm=(
            float(row["max_depth_mm"]) if row.get("max_depth_mm") is not None else None
        ),
        source=f"supabase:{source_library}",
        name=str(row["name"]) if row.get("name") is not None else None,
        corner_radius_mm=(
            float(row["corner_radius_mm"])
            if row.get("corner_radius_mm") is not None
            else None
        ),
        point_angle_deg=(
            float(row["point_angle_deg"])
            if row.get("point_angle_deg") is not None
            else None
        ),
        flute_count=int(row["flute_count"]) if row.get("flute_count") is not None else None,
        tool_material=(
            str(row["tool_material"]) if row.get("tool_material") is not None else None
        ),
        vendor=str(row["vendor"]) if row.get("vendor") is not None else None,
        source_unit=str(row["source_unit"]) if row.get("source_unit") is not None else None,
        guid=str(row["guid"]) if row.get("guid") is not None else None,
        presets=presets,
    )


def filter_tool_presets_by_material(tool: Tool, material: str | None) -> Tool:
    """Return a copy of ``tool`` with presets filtered by material (if set)."""
    if material is None:
        return tool
    filtered = [
        preset
        for preset in tool.presets
        if _preset_matches_material(preset.preset_material, material)
    ]
    if filtered == tool.presets:
        return tool
    return tool.model_copy(update={"presets": filtered})


def _raw_tools_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_tools = payload.get("data")
    if not isinstance(raw_tools, list):
        return []
    return [item for item in raw_tools if isinstance(item, Mapping)]


def _count_skipped_raw_tools(raw_tools: Sequence[Mapping[str, Any]], unit: str) -> int:
    skipped = 0
    for raw in raw_tools:
        raw_type = str(raw.get("type", ""))
        if _is_fusion_holder_type(raw_type):
            continue
        if not raw.get("guid"):
            skipped += 1
            continue
        geometry = raw.get("geometry") or {}
        if _positive_length_to_mm(geometry.get("DC"), unit) is None:
            skipped += 1
    return skipped


def prepare_tool_rows_from_library(
    path: str | Path,
    *,
    material: str | None = None,
) -> IngestFileResult:
    """Parse one Fusion library file into Supabase row dicts (no network I/O)."""
    path = Path(path)
    source_library = _library_name_from_path(path)
    result = IngestFileResult(path=path, source_library=source_library)

    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        result.error = str(exc)
        return result

    if not isinstance(payload, Mapping):
        result.error = f"tool library root must be a mapping: {path.name}"
        return result

    raw_tools = _raw_tools_from_payload(payload)
    raw_by_guid: dict[str, Mapping[str, Any]] = {}
    for raw in raw_tools:
        raw_type = str(raw.get("type", ""))
        if _is_fusion_holder_type(raw_type):
            result.holders_skipped += 1
            continue

        guid = raw.get("guid")
        if guid:
            raw_by_guid[str(guid)] = raw

        unit = _fusion_unit(raw.get("unit"))
        result.unit_counts[unit] = result.unit_counts.get(unit, 0) + 1

        if normalize_tool_type(raw_type) == "unknown" and raw_type.strip():
            result.unknown_types[raw_type] = result.unknown_types.get(raw_type, 0) + 1

    try:
        tools = load_tool_library(path, material=material)
    except ValueError as exc:
        result.error = str(exc)
        return result

    unit = _fusion_unit(None)
    if raw_tools:
        unit = _fusion_unit(raw_tools[0].get("unit"))

    result.tools_skipped = _count_skipped_raw_tools(raw_tools, unit)
    result.tools_ingested = len(tools)
    result.rows = [
        tool_to_row(
            tool,
            source_library=source_library,
            raw_type=str((raw_by_guid.get(tool.guid or "") or {}).get("type", "")),
            raw=raw_by_guid.get(tool.guid or ""),
        )
        for tool in tools
        if tool.guid is not None
    ]
    return result


def iter_row_chunks(
    rows: Sequence[Mapping[str, Any]],
    batch_size: int = UPSERT_BATCH_SIZE,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield ``(start_index, chunk_rows)`` slices of at most ``batch_size`` rows."""
    rows_list = [dict(row) for row in rows]
    for start in range(0, len(rows_list), batch_size):
        yield start, rows_list[start : start + batch_size]


def _is_statement_timeout(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "57014" in message or "statement timeout" in message or "canceling statement" in message:
        return True
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        return _is_statement_timeout(cause)
    return False


def _execute_upsert_chunk(client: Any, chunk: list[dict[str, Any]]) -> None:
    client.table(TOOLS_TABLE).upsert(chunk, on_conflict="guid").execute()


def _upsert_chunk_with_retry(client: Any, chunk: list[dict[str, Any]], range_label: str) -> None:
    try:
        _execute_upsert_chunk(client, chunk)
    except Exception as exc:
        if _is_statement_timeout(exc):
            logger.warning("chunk timeout for %s; retrying once", range_label)
            time.sleep(1.0)
            _execute_upsert_chunk(client, chunk)
            return
        raise


def _upsert_rows_adaptive(
    client: Any,
    rows: list[dict[str, Any]],
    *,
    source_label: str,
    start_offset: int,
    result: UpsertResult,
) -> None:
    """Upsert rows, recursively halving the chunk on repeated statement timeouts."""
    if not rows:
        return

    end = start_offset + len(rows) - 1
    range_label = f"{source_label} rows {start_offset}-{end}"
    try:
        _upsert_chunk_with_retry(client, rows, range_label)
        result.upserted += len(rows)
        return
    except Exception as exc:
        if len(rows) <= 1 or not _is_statement_timeout(exc):
            msg = f"{range_label}: upsert failed: {exc}"
            logger.error("FAILED %s", msg)
            result.failed_chunks.append(msg)
            return

        mid = len(rows) // 2
        logger.warning(
            "splitting timed-out chunk %s into %d + %d row sub-chunks",
            range_label,
            mid,
            len(rows) - mid,
        )
        _upsert_rows_adaptive(
            client,
            rows[:mid],
            source_label=source_label,
            start_offset=start_offset,
            result=result,
        )
        _upsert_rows_adaptive(
            client,
            rows[mid:],
            source_label=source_label,
            start_offset=start_offset + mid,
            result=result,
        )


def upsert_tool_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    client: Any | None = None,
    dry_run: bool = False,
    batch_size: int = UPSERT_BATCH_SIZE,
    source_label: str | None = None,
) -> UpsertResult:
    """Upsert normalized tool rows in batches keyed on ``guid`` (latest wins)."""
    if not rows:
        return UpsertResult()

    label = source_label or "tools"
    chunks = list(iter_row_chunks(rows, batch_size))
    result = UpsertResult(attempted=len(rows), chunk_count=len(chunks))

    if dry_run:
        logger.info("dry-run: would upsert %d tool row(s)", len(rows))
        result.upserted = len(rows)
        return result

    sb = create_supabase_client(client)
    for start, chunk in chunks:
        _upsert_rows_adaptive(
            sb,
            chunk,
            source_label=label,
            start_offset=start,
            result=result,
        )

    if len(chunks) > 1:
        logger.info(
            "upserted %d/%d rows for %s in %d chunk(s)",
            result.upserted,
            result.attempted,
            label,
            len(chunks),
        )

    return result


def _fetch_filter_label(
    tool_types: Sequence[str] | None,
    source_libraries: Sequence[str] | None,
) -> str:
    parts: list[str] = []
    if tool_types:
        parts.append(f"tool_type in {list(tool_types)}")
    if source_libraries:
        parts.append(f"source_library in {list(source_libraries)}")
    return ", ".join(parts) if parts else "none"


def _fetch_tool_rows_paginated(
    client: Any,
    *,
    tool_types: Sequence[str] | None = None,
    source_libraries: Sequence[str] | None = None,
    page_size: int = FETCH_PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Fetch all matching tool rows from Supabase, paginating past the REST row cap."""
    accumulated: list[dict[str, Any]] = []
    offset = 0
    page_index = 0
    filter_label = _fetch_filter_label(tool_types, source_libraries)

    while True:
        page_index += 1
        query = client.table(TOOLS_TABLE).select("*")
        if tool_types:
            query = query.in_("tool_type", list(tool_types))
        if source_libraries:
            query = query.in_("source_library", list(source_libraries))
        end = offset + page_size - 1
        response = query.order("guid").range(offset, end).execute()
        page = [dict(row) for row in (response.data or [])]
        accumulated.extend(page)

        if len(page) < page_size:
            break
        offset += page_size

    if page_index > 1:
        logger.warning(
            "Supabase tools fetch required pagination: %d page(s), %d total row(s) "
            "(page size %d; filters: %s)",
            page_index,
            len(accumulated),
            page_size,
            filter_label,
        )
    logger.info(
        "fetched %d tool row(s) from Supabase in %d page(s) (filters: %s)",
        len(accumulated),
        page_index,
        filter_label,
    )
    return accumulated


def fetch_tool_rows(
    client: Any | None = None,
    *,
    tool_types: Sequence[str] | None = None,
    source_libraries: Sequence[str] | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch tool rows from Supabase, or use injected rows in tests."""
    if rows is not None:
        data = [dict(row) for row in rows]
    else:
        sb = create_supabase_client(client)
        data = _fetch_tool_rows_paginated(
            sb,
            tool_types=tool_types,
            source_libraries=source_libraries,
        )

    if rows is not None:
        if tool_types:
            allowed = {t.strip() for t in tool_types}
            data = [row for row in data if str(row.get("tool_type")) in allowed]
        if source_libraries:
            allowed = {s.strip() for s in source_libraries}
            data = [row for row in data if str(row.get("source_library")) in allowed]

    return data
