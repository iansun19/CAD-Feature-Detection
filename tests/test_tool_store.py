"""Unit tests for normalized Supabase tool storage (tool_store + load_tools_from_supabase)."""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from planning.machining_context import (  # noqa: E402
    build_context_v0,
    load_tool_library,
    load_tools_from_supabase,
    normalize_tool_type,
)
from tools.tool_store import (  # noqa: E402
    FETCH_PAGE_SIZE,
    UPSERT_BATCH_SIZE,
    IngestFileResult,
    fetch_tool_rows,
    iter_row_chunks,
    prepare_tool_rows_from_library,
    row_to_tool,
    tool_to_row,
    upsert_tool_rows,
)
from scripts.ingest_tool_libraries import ingest_directory  # noqa: E402

SAMPLE_LIBRARY = Path(ROOT) / "tests" / "fixtures" / "Aluminum_Sample_Library__Inch_.json"
SETUP_YAML = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
CASCADE_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
MOCK_ENVELOPE = {"min": [0.0, -100.0, -50.0], "max": [200.0, 0.0, 50.0]}


class InMemoryToolsTable:
    """Minimal Supabase table double keyed on guid (latest upsert wins)."""

    def __init__(self, fail_at_call: int | None = None) -> None:
        self.rows: dict[str, dict] = {}
        self.upsert_calls: list[int] = []
        self.fail_at_call = fail_at_call
        self.fail_message = "57014 canceling statement due to statement timeout"
        self.call_count = 0

    def upsert(self, rows: list[dict], on_conflict: str = "guid") -> "InMemoryToolsTable":
        del on_conflict
        self.call_count += 1
        if self.fail_at_call == self.call_count:
            raise RuntimeError(self.fail_message)
        self.upsert_calls.append(len(rows))
        for row in rows:
            self.rows[str(row["guid"])] = dict(row)
        return self

    def execute(self) -> None:
        return None


class InMemorySupabaseClient:
    def __init__(self) -> None:
        self.tools = InMemoryToolsTable()

    def table(self, name: str) -> InMemoryToolsTable:
        if name != "tools":
            raise ValueError(f"unexpected table: {name}")
        return self.tools


class _QueryResponse:
    def __init__(self, data: list[dict[str, object]]) -> None:
        self.data = data


class PaginatedToolsTable:
    """Supabase table double with select/in_/order/range pagination."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.tool_types: set[str] | None = None
        self.source_libraries: set[str] | None = None
        self.order_col = "guid"
        self.range_start: int | None = None
        self.range_end: int | None = None
        self.fetch_calls = 0

    def select(self, _cols: str) -> "PaginatedToolsTable":
        return self

    def in_(self, column: str, values: list[str]) -> "PaginatedToolsTable":
        if column == "tool_type":
            self.tool_types = set(values)
        elif column == "source_library":
            self.source_libraries = set(values)
        else:
            raise ValueError(f"unexpected in_ column: {column}")
        return self

    def order(self, column: str, *, desc: bool = False) -> "PaginatedToolsTable":
        del desc
        self.order_col = column
        return self

    def range(self, start: int, end: int) -> "PaginatedToolsTable":
        self.range_start = start
        self.range_end = end
        return self

    def _filtered_rows(self) -> list[dict[str, object]]:
        rows = self.rows
        if self.tool_types is not None:
            rows = [row for row in rows if row.get("tool_type") in self.tool_types]
        if self.source_libraries is not None:
            rows = [
                row for row in rows if row.get("source_library") in self.source_libraries
            ]
        return sorted(rows, key=lambda row: str(row.get(self.order_col, "")))

    def execute(self) -> _QueryResponse:
        self.fetch_calls += 1
        filtered = self._filtered_rows()
        if self.range_start is None or self.range_end is None:
            return _QueryResponse(filtered)
        return _QueryResponse(filtered[self.range_start : self.range_end + 1])


class PaginatedSupabaseClient:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.fetch_calls = 0

    def table(self, name: str) -> PaginatedToolsTable:
        if name != "tools":
            raise ValueError(f"unexpected table: {name}")
        table = PaginatedToolsTable(self._rows)
        original_execute = table.execute

        def execute() -> _QueryResponse:
            self.fetch_calls += 1
            return original_execute()

        table.execute = execute  # type: ignore[method-assign]
        return table


def _mock_tool_row(index: int, *, tool_type: str = "endmill") -> dict[str, object]:
    return {
        "guid": f"{index:08d}-0000-0000-0000-000000000000",
        "tool_id": f"lib::{index:08d}",
        "tool_type": tool_type,
        "diameter_mm": 1.0 + index,
        "source_library": "mock_library",
        "presets": [],
    }


class TestFetchToolRowsPagination(unittest.TestCase):
    def test_single_page_under_cap(self) -> None:
        rows = [_mock_tool_row(i) for i in range(250)]
        client = PaginatedSupabaseClient(rows)
        fetched = fetch_tool_rows(client, tool_types=["endmill"])
        self.assertEqual(len(fetched), 250)
        self.assertEqual(client.fetch_calls, 1)
        self.assertEqual(len({row["guid"] for row in fetched}), 250)

    def test_multiple_pages_assembled_without_duplicates(self) -> None:
        rows = [_mock_tool_row(i) for i in range(2500)]
        client = PaginatedSupabaseClient(rows)
        with self.assertLogs("tools.tool_store", level="WARNING") as captured:
            fetched = fetch_tool_rows(client)
        self.assertEqual(len(fetched), 2500)
        self.assertEqual(client.fetch_calls, 3)
        self.assertEqual(len({row["guid"] for row in fetched}), 2500)
        self.assertTrue(
            any("required pagination" in msg and "3 page(s)" in msg for msg in captured.output)
        )

    def test_filtered_pagination_preserves_type_filter(self) -> None:
        rows = [
            _mock_tool_row(i, tool_type="drill" if i % 2 == 0 else "endmill")
            for i in range(2500)
        ]
        client = PaginatedSupabaseClient(rows)
        fetched = fetch_tool_rows(client, tool_types=["drill"])
        self.assertEqual(len(fetched), 1250)
        self.assertTrue(all(row["tool_type"] == "drill" for row in fetched))
        self.assertEqual(client.fetch_calls, 2)

    def test_exact_page_boundary_fetches_next_page(self) -> None:
        rows = [_mock_tool_row(i) for i in range(FETCH_PAGE_SIZE + 1)]
        client = PaginatedSupabaseClient(rows)
        fetched = fetch_tool_rows(client)
        self.assertEqual(len(fetched), FETCH_PAGE_SIZE + 1)
        self.assertEqual(client.fetch_calls, 2)


@unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library fixture missing")
class TestToolRowRoundTrip(unittest.TestCase):
    def test_ingest_serialize_query_deserialize_presets_intact(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        self.assertGreater(len(tools), 0)
        sample = next(t for t in tools if t.presets)

        row = tool_to_row(
            sample,
            source_library="Aluminum_Sample_Library_Inch",
            raw_type="flat end mill",
            raw={"guid": sample.guid, "type": "flat end mill"},
        )
        restored = row_to_tool(row)

        self.assertEqual(restored.tool_id, sample.tool_id)
        self.assertEqual(restored.tool_type, sample.tool_type)
        self.assertAlmostEqual(restored.diameter_mm, sample.diameter_mm)
        self.assertEqual(len(restored.presets), len(sample.presets))
        self.assertEqual(
            restored.presets[0].model_dump(mode="json"),
            sample.presets[0].model_dump(mode="json"),
        )
        self.assertTrue(restored.source.startswith("supabase:"))

    def test_load_tools_from_supabase_with_injected_rows(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        rows = [
            tool_to_row(
                tool,
                source_library="Aluminum_Sample_Library_Inch",
                raw_type="flat end mill",
            )
            for tool in tools
        ]
        loaded = load_tools_from_supabase(rows=rows)
        self.assertEqual(len(loaded), len(tools))
        self.assertEqual(
            {t.tool_id for t in loaded},
            {t.tool_id for t in tools},
        )

    def test_load_tools_from_supabase_filters_tool_types(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        rows = [
            tool_to_row(
                tool,
                source_library="Aluminum_Sample_Library_Inch",
                raw_type="flat end mill",
            )
            for tool in tools
        ]
        loaded = load_tools_from_supabase(rows=rows, tool_types=["drill"])
        self.assertEqual(loaded, [])
        loaded_endmill = load_tools_from_supabase(rows=rows, tool_types=["endmill"])
        self.assertEqual(len(loaded_endmill), 6)


@unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library fixture missing")
class TestIngestHelpers(unittest.TestCase):
    def test_prepare_tool_rows_counts_and_units(self) -> None:
        result = prepare_tool_rows_from_library(SAMPLE_LIBRARY)
        self.assertIsNone(result.error)
        self.assertEqual(result.tools_ingested, 12)
        self.assertEqual(result.tools_skipped, 0)
        self.assertEqual(result.unit_counts.get("inches"), 12)
        self.assertEqual(len(result.rows), 12)

    def test_unknown_type_logging_summary(self) -> None:
        payload = json.loads(SAMPLE_LIBRARY.read_text(encoding="utf-8"))
        payload["data"][0]["type"] = "custom reamer"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weird.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            log_stream = io.StringIO()
            with self.assertLogs("planning.machining_context", level="WARNING"):
                with redirect_stderr(log_stream):
                    file_results, global_unknown, _, _ = ingest_directory(
                        path.parent,
                        dry_run=True,
                    )

            self.assertEqual(len(file_results), 1)
            self.assertEqual(global_unknown["custom reamer"], 1)
            self.assertEqual(normalize_tool_type("custom reamer"), "unknown")

    def test_guid_dedup_on_reingest(self) -> None:
        client = InMemorySupabaseClient()
        result = prepare_tool_rows_from_library(SAMPLE_LIBRARY)
        upsert_tool_rows(result.rows, client=client)
        self.assertEqual(len(client.tools.rows), 12)

        modified_rows = [dict(row) for row in result.rows]
        modified_rows[0]["name"] = "UPDATED NAME"
        upsert_tool_rows(modified_rows, client=client)
        self.assertEqual(len(client.tools.rows), 12)
        guid = modified_rows[0]["guid"]
        self.assertEqual(client.tools.rows[guid]["name"], "UPDATED NAME")


class TestBatchedUpsert(unittest.TestCase):
    def test_iter_row_chunks_splits_library(self) -> None:
        rows = [{"guid": f"g{i}", "tool_id": f"t{i}"} for i in range(1201)]
        chunks = list(iter_row_chunks(rows, batch_size=500))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(len(chunks[0][1]), 500)
        self.assertEqual(chunks[1][0], 500)
        self.assertEqual(len(chunks[1][1]), 500)
        self.assertEqual(chunks[2][0], 1000)
        self.assertEqual(len(chunks[2][1]), 201)

    def test_chunk_failure_logs_and_continues(self) -> None:
        rows = [{"guid": f"g{i}", "tool_id": f"t{i}"} for i in range(1200)]
        client = InMemorySupabaseClient()
        client.tools.fail_at_call = 2
        client.tools.fail_message = "chunk write failed"

        with self.assertLogs("tools.tool_store", level="ERROR") as captured:
            result = upsert_tool_rows(
                rows,
                client=client,
                batch_size=500,
                source_label="big.json",
            )

        self.assertEqual(result.chunk_count, 3)
        self.assertEqual(result.upserted, 700)
        self.assertEqual(len(result.failed_chunks), 1)
        self.assertIn("big.json rows 500-999", result.failed_chunks[0])
        self.assertEqual(len(client.tools.rows), 700)
        self.assertEqual(client.tools.upsert_calls, [500, 200])
        self.assertTrue(any("FAILED big.json rows 500-999" in msg for msg in captured.output))

    def test_default_batch_size_constant(self) -> None:
        rows = [{"guid": f"g{i}"} for i in range(UPSERT_BATCH_SIZE + 1)]
        self.assertEqual(len(list(iter_row_chunks(rows))), 2)

    def test_timeout_chunk_splits_and_completes(self) -> None:
        rows = [{"guid": f"g{i}", "tool_id": f"t{i}"} for i in range(250)]
        client = InMemorySupabaseClient()
        client.tools.fail_message = "57014 canceling statement due to statement timeout"

        original_upsert = client.tools.upsert

        def upsert(rows_arg: list[dict], on_conflict: str = "guid") -> InMemoryToolsTable:
            if len(rows_arg) > 100:
                raise RuntimeError(client.tools.fail_message)
            return original_upsert(rows_arg, on_conflict=on_conflict)

        client.tools.upsert = upsert  # type: ignore[method-assign]

        result = upsert_tool_rows(
            rows,
            client=client,
            batch_size=250,
            source_label="heavy.json",
        )

        self.assertEqual(result.upserted, 250)
        self.assertEqual(result.failed_chunks, [])
        self.assertEqual(len(client.tools.rows), 250)


@unittest.skipUnless(
    SAMPLE_LIBRARY.is_file() and CASCADE_PATH.is_file(),
    "fixtures missing for build_context test",
)
class TestBuildContextSupabase(unittest.TestCase):
    def test_build_context_v0_tool_source_supabase(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        rows = [
            tool_to_row(
                tool,
                source_library="Aluminum_Sample_Library_Inch",
                raw_type="flat end mill",
            )
            for tool in tools
        ]

        with patch(
            "tools.tool_store.fetch_tool_rows",
            return_value=rows,
        ) as fetch_mock:
            ctx = build_context_v0(
                MOCK_ENVELOPE,
                SETUP_YAML,
                CASCADE_PATH,
                setup_id="rear",
                tool_source="supabase",
                setups_source="authored",
            )

        fetch_mock.assert_called_once()
        call_kwargs = fetch_mock.call_args.kwargs
        self.assertEqual(
            call_kwargs.get("tool_types"),
            ["drill", "endmill", "bullnose_endmill", "tap", "ball_endmill", "face_mill"],
        )
        self.assertEqual(len(ctx.tools), 12)
        self.assertEqual(ctx.metadata.get("tool_library_source"), "supabase")


if __name__ == "__main__":
    unittest.main()
