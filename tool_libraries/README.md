# Tool libraries (Supabase-backed)

Fusion / Toolpath tool-library exports are **not** stored in git. Ingest them once
into Supabase, then load normalized tools at runtime.

## Local library files (out of git)

Keep your ~52 Fusion exports on disk outside the repo, or in a gitignored folder:

```bash
mkdir -p local_tool_libraries
# copy *.json / *.tools exports here � this path is gitignored
```

Supported extensions: `.json`, `.tools`. Binary `.hsmlib` files are skipped; re-export
as JSON from Fusion / Toolpath.

## One-time ingestion

1. Apply migrations in `supabase/migrations/` (includes `tools` table).
2. Copy `.env.example` to `.env` and set Supabase keys (Project Settings ? API).
3. Run ingestion (uses `machining_context.load_tool_library()` for parsing):

```bash
python scripts/ingest_tool_libraries.py /path/to/local_tool_libraries
```

Dry-run (parse + log only, no Supabase writes):

```bash
python scripts/ingest_tool_libraries.py /path/to/local_tool_libraries --dry-run
```

Re-running is **idempotent**: rows upsert on `guid`; **latest ingest wins** on conflict.

Ingestion logs per-file stats, flags metric (`millimeters`) libraries, and prints a
summary of Fusion `type` strings that normalize to `unknown` (taps, reamers, etc.).

Requires `SUPABASE_SERVICE_ROLE_KEY` (or `SUPABASE_KEY`) and `SUPABASE_URL`.

## Runtime: point the planner at Supabase

```python
from planning.machining_context import build_context_v0, load_tools_from_supabase

# Full context assembly
ctx = build_context_v0(
    step_path,
    setup_yaml,
    feature_graph,
    tool_source="supabase",
    tool_library_material="aluminum",  # optional preset filter
)

# Or load tools directly
tools = load_tools_from_supabase(
    material="aluminum",
    tool_types=["endmill", "drill"],
    source_libraries=["Aluminum_Sample_Library_Inch"],
)
```

`tool_source` precedence in `build_context_v0`:

1. `tool_library_paths` � explicit on-disk file(s) override everything
2. `tool_source="supabase"` � normalized rows from Supabase `tools` table
3. `tool_source="directory"` � glob `tool_library_dir` (default: bundled `tool_libraries/`)
4. `tool_source="hardcoded"` (default) � built-in v0 catalog

## Bundled samples

`tool_libraries/` keeps a small set of sample JSON files for unit tests only.
Production catalogs should live in Supabase after ingestion.

## Legacy note

`scripts/seed_tool_libraries.py` and the `tool_libraries` jsonb table store **raw**
Fusion JSON. The normalized `tools` table (this workflow) is the preferred path.
