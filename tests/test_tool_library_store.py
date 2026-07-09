"""Unit tests for Supabase jsonb tool-library storage."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from machining_context import (  # noqa: E402
    DEFAULT_TOOL_LIBRARIES_DIR,
    build_context_v0,
    load_enabled_library_payloads,
    load_tool_library,
    load_tool_library_payload,
)
from tool_library_store import (  # noqa: E402
    fetch_enabled_library_rows,
    library_slug_from_stem,
    load_libraries_from_supabase,
    seed_tool_libraries,
)
from tool_store import tool_to_row  # noqa: E402

ALUMINUM_LIBRARY = DEFAULT_TOOL_LIBRARIES_DIR / "Aluminum_Sample_Library__Inch_.json"
DRILL_LIBRARY = DEFAULT_TOOL_LIBRARIES_DIR / "Kennametal_Standard_Drills__Inch_.json"
SETUP_YAML = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
CASCADE_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
MOCK_ENVELOPE = {"min": [0.0, -100.0, -50.0], "max": [200.0, 0.0, 50.0]}
BUNDLED_TOOL_COUNT = 236


class TestLibrarySlug(unittest.TestCase):
    def test_slug_from_stem(self) -> None:
        self.assertEqual(
            library_slug_from_stem("Aluminum_Sample_Library__Inch_"),
            "aluminum_sample_library_inch",
        )


@unittest.skipUnless(ALUMINUM_LIBRARY.is_file(), "aluminum sample library missing")
class TestPayloadLoader(unittest.TestCase):
    def test_payload_matches_file_loader(self) -> None:
        payload = json.loads(ALUMINUM_LIBRARY.read_text(encoding="utf-8"))
        from_file = load_tool_library(ALUMINUM_LIBRARY)
        from_payload = load_tool_library_payload(
            payload,
            library_name="Aluminum_Sample_Library_Inch",
            source_label=ALUMINUM_LIBRARY.name,
        )
        self.assertEqual(
            [t.model_dump(mode="json") for t in from_payload],
            [t.model_dump(mode="json") for t in from_file],
        )


@unittest.skipUnless(
    ALUMINUM_LIBRARY.is_file() and DRILL_LIBRARY.is_file(),
    "bundled tool libraries missing",
)
class TestSupabaseLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aluminum_payload = json.loads(ALUMINUM_LIBRARY.read_text(encoding="utf-8"))
        cls.drill_payload = json.loads(DRILL_LIBRARY.read_text(encoding="utf-8"))
        cls.rows = [
            {
                "slug": "aluminum_sample_library_inch",
                "library_name": "Aluminum_Sample_Library_Inch",
                "display_name": "Aluminum_Sample_Library__Inch_",
                "fusion_version": cls.aluminum_payload.get("version"),
                "content": cls.aluminum_payload,
            },
            {
                "slug": "kennametal_standard_drills_inch",
                "library_name": "Kennametal_Standard_Drills_Inch",
                "display_name": "Kennametal_Standard_Drills__Inch_",
                "fusion_version": cls.drill_payload.get("version"),
                "content": cls.drill_payload,
            },
        ]

    def test_fetch_uses_injected_rows(self) -> None:
        rows = fetch_enabled_library_rows(rows=self.rows)
        self.assertEqual(len(rows), 2)

    def test_load_libraries_from_supabase_matches_bundled_count(self) -> None:
        tools = load_libraries_from_supabase(rows=self.rows)
        self.assertEqual(len(tools), BUNDLED_TOOL_COUNT)

    def test_tool_ids_match_local_loader_prefixes(self) -> None:
        local = load_enabled_library_payloads(
            [
                ("Aluminum_Sample_Library_Inch", self.aluminum_payload, ALUMINUM_LIBRARY.name),
                ("Kennametal_Standard_Drills_Inch", self.drill_payload, DRILL_LIBRARY.name),
            ]
        )
        remote = load_libraries_from_supabase(rows=self.rows)
        self.assertEqual(
            {t.tool_id for t in remote},
            {t.tool_id for t in local},
        )


@unittest.skipUnless(
    ALUMINUM_LIBRARY.is_file() and DRILL_LIBRARY.is_file(),
    "bundled tool libraries missing",
)
class TestSeedDryRun(unittest.TestCase):
    def test_seed_dry_run_lists_bundled_libraries(self) -> None:
        seeded = seed_tool_libraries(DEFAULT_TOOL_LIBRARIES_DIR, dry_run=True)
        self.assertEqual(
            seeded,
            ["aluminum_sample_library_inch", "kennametal_standard_drills_inch"],
        )


@unittest.skipUnless(
    ALUMINUM_LIBRARY.is_file() and CASCADE_PATH.is_file(),
    "fixtures missing for build_context test",
)
class TestBuildContextSupabase(unittest.TestCase):
    def test_build_context_v0_tool_source_supabase(self) -> None:
        aluminum_payload = json.loads(ALUMINUM_LIBRARY.read_text(encoding="utf-8"))
        drill_payload = json.loads(DRILL_LIBRARY.read_text(encoding="utf-8"))
        tools = load_enabled_library_payloads(
            [
                ("Aluminum_Sample_Library_Inch", aluminum_payload, ALUMINUM_LIBRARY.name),
                ("Kennametal_Standard_Drills_Inch", drill_payload, DRILL_LIBRARY.name),
            ]
        )
        rows = [
            tool_to_row(
                tool,
                source_library=tool.source.removeprefix("fusion_library:"),
                raw_type="drill" if tool.tool_type == "drill" else "flat end mill",
            )
            for tool in tools
        ]

        with patch(
            "tool_store.fetch_tool_rows",
            return_value=rows,
        ):
            ctx = build_context_v0(
                MOCK_ENVELOPE,
                SETUP_YAML,
                CASCADE_PATH,
                setup_id="rear",
                tool_source="supabase",
                setups_source="authored",
            )

        self.assertEqual(len(ctx.tools), BUNDLED_TOOL_COUNT)
        self.assertEqual(ctx.metadata.get("tool_library_source"), "supabase")


if __name__ == "__main__":
    unittest.main()
