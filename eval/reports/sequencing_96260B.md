# Operation sequencing � 96260B validation

Generated: 2026-07-08

## Configuration

- Part: **96260B** (rear + front setups)
- Context: `eval/gt/96260B_setup.yaml`, `tool_source=hardcoded`, `setups_source=authored`
- Graphs: `pipeline_out/96260B_{rear,front}/feature_graph_cascade.json`
- Scorer weights (defaults): setup_change **100**, tool_change **10**, approach_change **1**, rough_finish_grouping_bonus **2**

Reproduce:

```bash
python scripts/96260B/sequencing_eval_96260B.py
# or per mode:
python planner.py --multi-setup --setups authored --tool-source hardcoded \
  --no-scope-diff --seq-search beam --out /tmp/cam_plan_beam.json
```

## Score breakdowns

| Strategy | Total cost | Setup changes | Tool changes | Weighted terms |
|----------|------------|---------------|--------------|----------------|
| **none** (topo-sort) | **230.0** | 1 | **13** | setup 100 + tool 130 |
| **greedy** | **158.0** | 1 | **6** | setup 100 + tool 60 ? bonus 2 |
| **beam** (default) | **158.0** | 1 | **6** | setup 100 + tool 60 ? bonus 2 |

### Ordering check

- **beam ? greedy ? none** on total cost: **158 ? 158 ? 230** (PASS)
- Setup changes are identical (1 rear?front transition) because cross-setup precedence fixes the fixture boundary; the win is entirely from **tool grouping within setups**.

### Concrete win (96260B)

| Metric | none ? beam |
|--------|-------------|
| Setup changes | 1 ? 1 (unchanged) |
| Tool changes | **13 ? 6** (?7 swaps) |
| Total cost | **230 ? 158** (?72) |

Greedy and beam tie on this instance; beam matches greedy�s tool-minimized order.

## Precedence validity

Every emitted plan was checked with `validate_sequence_precedence()` against each operation�s `depends_on` edges (drill?tap, rough?finish, cross-setup fixture boundary). **No violations** across none / greedy / beam.

## Plan-neutral geometry

| Check | none | greedy | beam | `examples/cam_plan_96260B.json` |
|-------|------|--------|------|----------------------------------|
| Op count | 16 | 16 | 16 | 16 |
| Work signature `(feature_refs, setup, op_type, tool, strategy)` | baseline | **identical** | **identical** | **identical** |
| Machined B-rep face set | **257** | **257** | **257** | **257** |

Sequencing **reorders** operations only; it does not add, drop, or retarget work.

## Metadata

`CamPlan.metadata.sequence_score` records strategy, total, raw counts, and weighted breakdown, e.g.:

```json
{
  "strategy": "beam",
  "total": 158.0,
  "setup_changes": 1,
  "tool_changes": 6,
  "approach_changes": 0,
  "rough_finish_groupings": 1,
  "weighted": {
    "setup_change": 100.0,
    "tool_change": 60.0,
    "approach_change": 0.0,
    "rough_finish_grouping_bonus": -2.0
  }
}
```

## VERDICT: **PASS**

Beam/greedy search reduces tool-change cost by 7 swaps on 96260B while preserving precedence, op multiset, and 257-face machined geometry.
