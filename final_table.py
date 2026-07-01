"""
final_table.py — Step 5 combined comparison table.

Evaluates every LLM run condition that exists on disk (free-range vs bounded,
stripped vs unstripped), pulls the existing heuristic/GNN numbers verbatim from
baseline_results.txt, and prints one table:

  columns: Acc | MacroF1 | WeakF1 | Malf% | Cost$   (LLM-specific cols blank for
           the non-LLM baselines, which don't have them)

LLM rows use the (b) full-test-set numbers (missing/malformed counted wrong) — the
fair end-to-end metric, same as the per-run reports.

Run the LLM conditions first (any subset; only those present are shown):
  python llm_baseline.py --run --limit N --output free_stripped
  python llm_baseline.py --run --limit N --output free_unstripped   --keep-labels
  python llm_baseline.py --run --limit N --output bounded_stripped   --bounded
  python llm_baseline.py --run --limit N --output bounded_unstripped --bounded --keep-labels
  python final_table.py
"""

import os
import re

from llm_baseline import load_cfg, cmd_eval, out_paths

# label -> (output prefix, note). Order = display order.
LLM_RUNS = [
    ("LLM free-range, stripped (fair)",        "stripped_run"),
    ("LLM free-range, unstripped (MVP as-is)", "unstripped_run"),
    ("LLM bounded, stripped (fair ceiling)",   "bounded_stripped"),
    ("LLM bounded, unstripped (+leak)",        "bounded_unstripped"),
    ("LLM features (serialized GNN geom)",      "features_run"),
    ("LLM features + 12-class templates",       "templates_run"),
]


def parse_baseline_summary(path="baseline_results.txt"):
    """Pull (label, acc, macro_f1) rows out of the existing SUMMARY block."""
    rows = []
    if not os.path.isfile(path):
        return rows
    in_summary = False
    for line in open(path):
        if "SUMMARY" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        m = re.match(r"\s*(\S.*?\S)\s{2,}([\d.]+)\s+([\d.]+)\s*$", line)
        if m and "Test Acc" not in line:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return rows


def main():
    cfg = load_cfg()
    base_rows = parse_baseline_summary()

    llm_rows = []
    for label, prefix in LLM_RUNS:
        if not os.path.isfile(out_paths(prefix)["results"]):
            continue
        m = cmd_eval(cfg, prefix, quiet=True)
        llm_rows.append((label, m))

    w = 42
    print("=" * 96)
    print("FINAL COMPARISON — B-rep face classification, MFCAD++ test split")
    print("  LLM rows = (b) full-set metric (missing/malformed = wrong). Same 12-class metric.")
    print("=" * 96)
    print(f"{'Model':<{w}}{'Acc':>9}{'MacroF1':>9}{'WeakF1':>9}{'Malf%':>8}{'Cost$':>9}")
    print("-" * 96)
    for label, acc, f1 in base_rows:
        print(f"{label:<{w}}{acc:>9.4f}{f1:>9.4f}{'-':>9}{'-':>8}{'-':>9}")
    if base_rows and llm_rows:
        print("-" * 96)
    for label, m in llm_rows:
        print(f"{label:<{w}}{m['acc_b']:>9.4f}{m['macrof1_b']:>9.4f}"
              f"{m['weak_macro_b']:>9.4f}{m['malf_rate_parts']*100:>7.2f}%"
              f"{m['cost']:>9.4f}")
    print("=" * 96)
    if not llm_rows:
        print("(no LLM runs found — run llm_baseline.py --run with --output prefixes first)")
    else:
        n = llm_rows[0][1]["n_parts"]
        print(f"note: LLM rows scored on {n} parts; baseline/GNN rows from "
              f"baseline_results.txt (full 8,949-part test set).")
        print("      compare like-for-like only after scaling the chosen LLM "
              "condition to the full test set.")


if __name__ == "__main__":
    main()
