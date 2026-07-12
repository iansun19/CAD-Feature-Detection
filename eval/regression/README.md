# Cascade labeling golden regression

This directory holds **behavior snapshots** for the rule-based CAM cascade
(`run_cascade.py` ? face/feature labeling). It answers: *did our cascade
labeling change vs. the last output we intentionally blessed?*

This is **not** a correctness oracle. Toolpath hand truth and attribute
matching live under `eval/gt/*.yaml` and `eval_cascade.py`.

| | Regression (here) | Correctness (`eval/gt/`) |
|---|---|---|
| Question | Did the cascade change? | Is the cascade right? |
| Truth source | Last blessed run | Toolpath transcription |
| Signal | Full face partition | Counts / attributes |
| Update | Intentional re-bless | GT edits |

## Layout

```
eval/regression/
  README.md
  BLESSINGS.md              # append-only re-bless log
  fixtures/                 # per-part YAML descriptors
  golden/                   # blessed partition JSON
  graphs/                   # pinned graph.npz per fixture
```

Fixture STEP files stay **gitignored** at the repo root (same pattern as
`test_stock_cut_classification.py`). YAML + golden + npz are committed.

## Re-bless (manual only)

```bash
python scripts/eval_cascade_regression.py --bless 96260B_rear \
  --reason "Describe why the new output is intentional (min 10 chars)"
```

Compare (exit 1 on regression):

```bash
python scripts/eval_cascade_regression.py
python scripts/eval_cascade_regression.py --fixture 96260B_rear
```

Or from Python:

```python
from cascade_regression import load_regression_fixture, bless_regression_fixture

fixture = load_regression_fixture("eval/regression/fixtures/96260B_rear.yaml")
bless_regression_fixture(
    fixture,
    reason="Describe why the new output is intentional (min 10 chars)",
)
```

Review the git diff on `golden/*.partition.json` and `BLESSINGS.md` before
merging. Tests never write golden files.
