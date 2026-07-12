"""Supabase-backed storage for Fusion tool libraries (jsonb)."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from machining_context import (
    DEFAULT_TOOL_LIBRARIES_DIR,
    Tool,
    _library_name_from_path,
    load_enabled_library_payloads,
)

logger = logging.getLogger(__name__)

TOOL_LIBRARIES_TABLE = "tool_libraries"


class SupabaseConfigError(RuntimeError):
    """Raised when Supabase environment variables are missing."""


def library_slug_from_stem(stem: str) -> str:
    """Stable slug for upsert keys, derived from a library filename stem."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", stem.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "fusion_library"


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
            "supabase package is required for Supabase tool libraries; "
            "install with: pip install supabase"
        ) from exc
    url, key = supabase_env()
    return create_client(url, key)


def fetch_enabled_library_rows(
    client: Any | None = None,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch enabled tool library rows from Supabase, or use injected rows in tests."""
    if rows is not None:
        return [dict(row) for row in rows]

    sb = create_supabase_client(client)
    response = (
        sb.table(TOOL_LIBRARIES_TABLE)
        .select("slug,library_name,display_name,fusion_version,content")
        .eq("enabled", True)
        .order("slug")
        .execute()
    )
    data = response.data or []
    logger.info("fetched %d enabled tool library row(s) from Supabase", len(data))
    return data


def load_libraries_from_supabase(
    material: str | None = None,
    *,
    client: Any | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[Tool]:
    """Load enabled Fusion libraries from Supabase jsonb rows into Tool models."""
    library_rows = fetch_enabled_library_rows(client, rows=rows)
    if not library_rows:
        logger.info("no enabled tool libraries in Supabase")
        return []

    libraries: list[tuple[str, Mapping[str, Any], str | None]] = []
    for row in library_rows:
        content = row.get("content")
        if isinstance(content, str):
            content = json.loads(content)
        if not isinstance(content, Mapping):
            raise ValueError(
                f"tool library row slug={row.get('slug')!r} has invalid content payload"
            )
        library_name = str(row.get("library_name") or row.get("slug") or "fusion_library")
        source_label = str(row.get("slug") or library_name)
        libraries.append((library_name, content, source_label))

    tools = load_enabled_library_payloads(libraries, material=material)
    logger.info(
        "loaded %d tools from %d Supabase library row(s) (after guid dedup)",
        len(tools),
        len(library_rows),
    )
    return tools


def seed_tool_libraries(
    dir_path: str | Path | None = None,
    *,
    client: Any | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Upsert Fusion JSON libraries from disk into Supabase ``tool_libraries`` table."""
    resolved = Path(dir_path) if dir_path is not None else DEFAULT_TOOL_LIBRARIES_DIR
    if not resolved.is_dir():
        raise NotADirectoryError(f"tool library directory not found: {resolved}")

    library_paths = sorted(resolved.glob("*.json"), key=lambda p: p.name.lower())
    if not library_paths:
        logger.info("no .json libraries to seed in %s", resolved)
        return []

    sb = None if dry_run else create_supabase_client(client)
    seeded: list[str] = []

    for path in library_paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, Mapping):
            raise ValueError(f"tool library root must be a mapping: {path}")

        slug = library_slug_from_stem(path.stem)
        row = {
            "slug": slug,
            "library_name": _library_name_from_path(path),
            "display_name": path.stem,
            "fusion_version": payload.get("version"),
            "enabled": True,
            "content": payload,
        }
        if dry_run:
            logger.info("dry-run: would upsert slug=%s from %s", slug, path.name)
        else:
            sb.table(TOOL_LIBRARIES_TABLE).upsert(row, on_conflict="slug").execute()
            logger.info("upserted slug=%s from %s", slug, path.name)
        seeded.append(slug)

    return seeded
