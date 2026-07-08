"""Unit tests for bundled tool-library directory loading."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from machining_context import (  # noqa: E402
    DEFAULT_TOOL_LIBRARIES_DIR,
    build_context_v0,
    load_default_libraries,
    load_library_directory,
)

SAMPLE_LIBRARY = DEFAULT_TOOL_LIBRARIES_DIR / "Aluminum_Sample_Library__Inch_.json"
DRILL_LIBRARY = DEFAULT_TOOL_LIBRARIES_DIR / "Kennametal_Standard_Drills__Inch_.json"
BUNDLED_TOOL_COUNT = 236  # 12 aluminum endmills + 224 Kennametal drills
SETUP_YAML = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
CASCADE_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
MOCK_ENVELOPE = {"min": [0.0, -100.0, -50.0], "max": [200.0, 0.0, 50.0]}


@unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "bundled sample library missing")
class TestLoadLibraryDirectory(unittest.TestCase):
    def test_default_directory_finds_sample(self) -> None:
        tools = load_default_libraries()
        self.assertEqual(len(tools), BUNDLED_TOOL_COUNT)
        self.assertTrue(all(t.source.startswith("fusion_library:") for t in tools))

    def test_load_library_directory_finds_sample(self) -> None:
        tools = load_library_directory(DEFAULT_TOOL_LIBRARIES_DIR)
        self.assertEqual(len(tools), BUNDLED_TOOL_COUNT)

    def test_glob_picks_up_json_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_lib = tmp_path / "sample.json"
            json_lib.write_text(SAMPLE_LIBRARY.read_text(encoding="utf-8"), encoding="utf-8")
            tools_lib = tmp_path / "sample.tools"
            tools_lib.write_text(SAMPLE_LIBRARY.read_text(encoding="utf-8"), encoding="utf-8")

            tools = load_library_directory(tmp_path)
            self.assertEqual(len(tools), 12)

    def test_hsmlib_skipped_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "legacy.hsmlib").write_bytes(b"not-a-real-hsmlib")
            json_lib = tmp_path / "sample.json"
            json_lib.write_text(SAMPLE_LIBRARY.read_text(encoding="utf-8"), encoding="utf-8")

            with self.assertLogs("machining_context", level="WARNING") as captured:
                tools = load_library_directory(tmp_path)
            self.assertEqual(len(tools), 12)
            self.assertTrue(
                any("skipping unsupported .hsmlib" in msg for msg in captured.output)
            )

    def test_dedup_across_duplicate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = SAMPLE_LIBRARY.read_text(encoding="utf-8")
            (tmp_path / "copy_one.json").write_text(content, encoding="utf-8")
            (tmp_path / "copy_two.tools").write_text(content, encoding="utf-8")

            tools = load_library_directory(tmp_path)
            self.assertEqual(len(tools), 12)
            guids = [t.guid for t in tools]
            self.assertEqual(len(guids), len(set(guids)))


@unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "bundled sample library missing")
@unittest.skipUnless(CASCADE_PATH.is_file(), "96260B rear cascade not present")
class TestBuildContextWithDefaultLibraries(unittest.TestCase):
    def test_build_context_v0_use_default_libraries(self) -> None:
        ctx = build_context_v0(
            MOCK_ENVELOPE,
            SETUP_YAML,
            CASCADE_PATH,
            setup_id="rear",
            tool_source="directory",
        )
        self.assertEqual(len(ctx.tools), BUNDLED_TOOL_COUNT)
        self.assertTrue(all(t.source.startswith("fusion_library:") for t in ctx.tools))
        self.assertIn("tool_library_dir", ctx.metadata)


if __name__ == "__main__":
    unittest.main()
