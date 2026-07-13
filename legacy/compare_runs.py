"""
compare_runs.py — side-by-side of the stripped vs unstripped LLM smoke runs.

Thin wrapper around llm_baseline.py (no core logic duplicated): it calls
cmd_eval() on each --output prefix, prints a single combined table
[stripped | unstripped | delta] for accuracy / macro-F1 / per-weak-class F1 /
malformed rate, then surfaces example faces where STRIPPED was wrong but
UNSTRIPPED was right (the cleanest evidence the model reads the leaked name field
rather than reasoning about geometry). Finally prints combined token/cost.

Assumes both runs already exist:
  python llm_baseline.py --run --limit 20 --concurrency 8 --output stripped_run
  python llm_baseline.py --run --limit 20 --concurrency 8 --keep-labels --output unstripped_run

Usage:
  python compare_runs.py [--stripped stripped_run] [--unstripped unstripped_run]
                         [--examples 5]
"""

import argparse
import json
import re

from legacy.llm_baseline import (
    load_cfg, cmd_eval, out_paths, build_name_to_id, extract_json_obj,
    coerce_class, parse_faces, strip_labels, step_path, WEAK_CLASS_IDS,
)
from legacy.evaluate import load_class_names


def load_face_preds(path, name_to_id, C):
    """Return {(part_id, entity_id): pred_id_or_None} and {(part_id, entity_id): true}."""
    preds, truth = {}, {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["part_id"]
            eids, tls = rec["entity_ids"], rec["true_labels"]
            for eid, t in zip(eids, tls):
                truth[(pid, eid)] = t
            try:
                obj = extract_json_obj(rec["raw_response"])
                if not isinstance(obj, dict):
                    raise ValueError
            except Exception:
                for eid in eids:
                    preds[(pid, eid)] = None        # malformed -> unscored / wrong
                continue
            pmap = {}
            for k, v in obj.items():
                ks = str(k).strip()
                if not ks.startswith("#"):
                    ks = "#" + ks.lstrip("#")
                pmap[ks] = v
            for eid in eids:
                preds[(pid, eid)] = (coerce_class(pmap[eid], name_to_id, C)
                                     if eid in pmap else None)
    return preds, truth


def entity_text(step_text, eid):
    """Raw substring '#N = ADVANCED_FACE(...);' for one face entity."""
    num = eid.lstrip("#")
    m = re.search(rf"#{num}\s*=\s*ADVANCED_FACE.*?;", step_text, re.DOTALL)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else "(entity text not found)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stripped", default="stripped_run")
    ap.add_argument("--unstripped", default="unstripped_run")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    C = cfg["num_classes"]
    names = load_class_names(cfg["data_root"], C)
    name_to_id = build_name_to_id(names)

    # ---- Step 2: evaluate each run independently ----
    print("\n##### EVAL: STRIPPED (fair) #####")
    m_s = cmd_eval(cfg, args.stripped)
    print("\n##### EVAL: UNSTRIPPED (cofounder's MVP: free-range raw STEP) #####")
    m_u = cmd_eval(cfg, args.unstripped)

    # ---- Step 3: combined side-by-side table ----
    def row(label, s, u, pct=False):
        d = u - s
        fmt = (lambda x: f"{x*100:6.2f}%") if pct else (lambda x: f"{x:7.4f}")
        sign = "+" if d >= 0 else ""
        dd = (f"{sign}{d*100:.2f}%") if pct else f"{sign}{d:.4f}"
        print(f"  {label:<34}{fmt(s):>9}{fmt(u):>11}{dd:>12}")

    print("\n" + "=" * 70)
    print(f"SIDE-BY-SIDE  (same {m_s['n_parts']} parts; stripped vs unstripped)")
    print("  unstripped == cofounder MVP condition (label name field LEFT IN the text)")
    print("=" * 70)
    print(f"  {'metric':<34}{'stripped':>9}{'unstripped':>11}{'delta':>12}")
    print("  " + "-" * 64)
    row("accuracy (full set, b)", m_s["acc_b"], m_u["acc_b"])
    row("macro-F1 (full set, b)", m_s["macrof1_b"], m_u["macrof1_b"])
    row("accuracy (parsed-only, a)", m_s["acc_a"], m_u["acc_a"])
    row("malformed-response rate", m_s["malf_rate_parts"], m_u["malf_rate_parts"], pct=True)
    print("  weak-class F1 (full set, b):")
    for c in WEAK_CLASS_IDS:
        row(f"  [{c}] {names[c]}", float(m_s["f1_b"][c]), float(m_u["f1_b"][c]))
    row("weak-class mean F1", m_s["weak_macro_b"], m_u["weak_macro_b"])

    # ---- Step 3: example faces (stripped wrong, unstripped right) ----
    print("\n" + "=" * 70)
    print("EXAMPLE FACES: stripped WRONG, unstripped RIGHT (raw text shown, no commentary)")
    print("=" * 70)
    ps, _ = load_face_preds(out_paths(args.stripped)["results"], name_to_id, C)
    pu, truth = load_face_preds(out_paths(args.unstripped)["results"], name_to_id, C)
    step_cache = {}
    shown = 0
    for (pid, eid), t in truth.items():
        if shown >= args.examples:
            break
        sp, up = ps.get((pid, eid)), pu.get((pid, eid))
        if sp != t and up == t:
            if pid not in step_cache:
                step_cache[pid] = open(step_path(cfg, pid)).read()
            raw = entity_text(step_cache[pid], eid)
            stripped_seen = entity_text(strip_labels(step_cache[pid]), eid)
            shown += 1
            print(f"\n[{shown}] part {pid}  face {eid}")
            print(f"  unstripped model saw : {raw[:200]}")
            print(f"  stripped   model saw : {stripped_seen[:200]}")
            print(f"  ground truth         : {t} ({names[t]})")
            print(f"  stripped   predicted : {sp} ({names[sp] if sp is not None else 'MISSING/MALFORMED'})")
            print(f"  unstripped predicted : {up} ({names[up]})")
    if shown == 0:
        print("\n  (no such cases in this sample)")

    # ---- Step 4: combined cost ----
    tot_calls = m_s["n_parts"] + m_u["n_parts"]
    tot_in = m_s["prompt_tok"] + m_u["prompt_tok"]
    tot_out = m_s["compl_tok"] + m_u["compl_tok"]
    tot_cost = m_s["cost"] + m_u["cost"]
    print("\n" + "=" * 70)
    print("COST / SCOPE (this smoke test, both runs combined)")
    print("=" * 70)
    print(f"  total LLM calls   : {tot_calls}")
    print(f"  prompt tokens     : {tot_in:,}")
    print(f"  completion tokens : {tot_out:,}")
    print(f"  approx cost       : ${tot_cost:.4f}")
    print(f"  per-part avg cost  : ${tot_cost/max(tot_calls,1):.5f}  "
          f"-> full 8,949-part run (one mode) ~= ${tot_cost/max(tot_calls,1)*8949:.2f}")


if __name__ == "__main__":
    main()
