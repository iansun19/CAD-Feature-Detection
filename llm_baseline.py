"""
llm_baseline.py — LLM-based B-rep face classifier, directly comparable to the GNN.

This is a STANDALONE, purely-additive script. It does NOT import or modify
train.py / model.py / dataset.py's logic beyond reusing the read-only class-name
mapping (evaluate.load_class_names) and the existing test split, so class indices
line up exactly with evaluate.py.

WHAT THIS EVALUATES
-------------------
A cofounder's MVP reportedly uses an LLM (GPT-4o-mini) to classify machining
features directly from RAW STEP file text, instead of from the preprocessed
14-dim graph features the GNN/heuristics use.

  *** The cofounder's actual MVP code/prompt was NOT found in this repo. ***

So this is a COMPARABLE RECONSTRUCTION, not the cofounder's exact MVP. It mirrors
the described approach (raw STEP text -> per-face class, zero-shot, low temp) so the
comparison is honest about its limitations. If you obtain the real MVP prompt, drop
it into build_messages() to evaluate the real thing.

CRITICAL FINDING — LABEL LEAK (see --audit)
-------------------------------------------
In MFCAD++ STEP files the ground-truth label is embedded in each face's name field:

    #17 = ADVANCED_FACE('24', ...)   <-- '24' == the H5 ground-truth label (Stock)

Verified: the per-face label list in the H5 equals the ordered ADVANCED_FACE name
fields, exactly, on every part checked. Feeding *truly raw* STEP text to an LLM
therefore hands it the answer key and yields a meaningless ~100%.

  => By default this script STRIPS the name field before sending text to the model
     (ADVANCED_FACE('24', -> ADVANCED_FACE('', ). Faces are referenced by their STEP
     entity id (#17), which is NOT the label and is a stable per-face handle.
  => --keep-labels disables stripping ONLY to demonstrate the leak; numbers produced
     that way are invalid for comparison and are labeled as such.

FACE-ID ALIGNMENT
-----------------
The i-th ADVANCED_FACE entity in file order (== CLOSED_SHELL order) maps to label
index i in the H5. We map an LLM's per-entity prediction back to label index i via
that ordering, then score against the same 25-class metric evaluate.py uses.

USAGE
-----
  python llm_baseline.py --audit                 # print leak/token/alignment audit
  python llm_baseline.py --run [--limit N] [--concurrency 8] [--keep-labels]
  python llm_baseline.py --eval                  # score whatever's in the results file
  python llm_baseline.py --run --eval            # do both

Requires OPENAI_API_KEY in the environment for --run.
"""

import argparse
import asyncio
import json
import os
import re
import time

import numpy as np
import yaml

from evaluate import load_class_names, per_class_metrics

# ---------------------------------------------------------------------------
# config / constants
# ---------------------------------------------------------------------------
WEAK_CLASS_IDS = (0, 5, 6, 7, 8, 9, 10, 17, 18, 19)  # mirror heuristic_baseline.py
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
# gpt-4o-mini context = 128k. Leave headroom for system prompt + output.
MAX_INPUT_TOKENS = 110_000

RESULTS_FILE = "llm_baseline_results.jsonl"   # one JSON line per completed part
ERRORS_FILE = "llm_baseline_errors.jsonl"     # malformed / failed parts
REPORT_FILE = "llm_baseline_results.txt"      # final comparison table + metrics
CM_CSV = "llm_baseline_confusion_matrix.csv"


def out_paths(output):
    """Resolve the 4 artifact paths. --output PREFIX keeps concurrent runs separate."""
    if output:
        return {"results": f"{output}.jsonl", "errors": f"{output}.errors.jsonl",
                "report": f"{output}.report.txt", "cm": f"{output}.cm.csv"}
    return {"results": RESULTS_FILE, "errors": ERRORS_FILE,
            "report": REPORT_FILE, "cm": CM_CSV}

# ADVANCED_FACE('<label>'  — the name field is the leaked label
_FACE_NAME_RE = re.compile(r"(ADVANCED_FACE\(\s*')(\d+)(')")


def load_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def step_path(cfg, part_id):
    return os.path.join(cfg["data_root"], "step", "test", f"{part_id}.step")


def read_test_ids(cfg):
    with open(os.path.join(cfg["data_root"], "test.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


# ---------------------------------------------------------------------------
# STEP parsing (NO pythonocc / STEP library — raw text only, by design)
# ---------------------------------------------------------------------------
def parse_faces(step_text):
    """Return ordered list of (entity_id:str, true_label:int) for each face.

    Order == file order == CLOSED_SHELL order == H5 label index order.
    Whitespace is collapsed first so wrapped entity definitions still match.
    """
    flat = re.sub(r"\s+", "", step_text)
    pairs = re.findall(r"#(\d+)=ADVANCED_FACE\('(\d+)'", flat)
    return [(f"#{eid}", int(lbl)) for eid, lbl in pairs]


def strip_labels(step_text):
    """Blank out the leaked label in every ADVANCED_FACE name field."""
    return _FACE_NAME_RE.sub(r"\1\3", step_text)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def build_system_prompt(class_names):
    listing = "\n".join(f"  {i} - {n}" for i, n in enumerate(class_names))
    return (
        "You are an expert in CAD B-rep geometry and CNC machining features. "
        "You are given the raw text of an ISO-10303-21 (STEP) file describing a "
        "single solid part. Every B-rep face appears as an entity of the form "
        "`#N = ADVANCED_FACE('', (...loops...), #surface, .T./.F.);` where `#N` is "
        "the face's entity id. The face's name field has been intentionally blanked.\n\n"
        "Classify EVERY face in the part into exactly one of these 25 machining-"
        "feature classes (use the integer id):\n"
        f"{listing}\n\n"
        "Reason from the geometry: surface type (PLANE/CYLINDRICAL_SURFACE/CONICAL_"
        "SURFACE/etc.), the edge loops, how faces bound pockets/slots/steps/holes, "
        "and which faces are the original stock surfaces (class 24).\n\n"
        "OUTPUT FORMAT (STRICT): a single JSON object mapping each face entity id "
        '(string, including the leading "#") to its integer class id. Example:\n'
        '{"#17": 24, "#619": 3, "#808": 15}\n'
        "Include EVERY face entity id present in the file, exactly once. Output ONLY "
        "the JSON object — no markdown, no commentary."
    )


def build_messages(class_names, step_text):
    return [
        {"role": "system", "content": build_system_prompt(class_names)},
        {"role": "user",
         "content": "STEP file:\n\n" + step_text},
    ]


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------
def extract_json_obj(text):
    """Pull the first balanced {...} JSON object out of a response string."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in response")


def build_name_to_id(class_names):
    """Normalized class-name -> id, to accept either id or name from the model."""
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    return {norm(n): i for i, n in enumerate(class_names)}


def coerce_class(value, name_to_id, num_classes):
    """Map a model-emitted value (int id, numeric string, or class name) to an id."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < num_classes else None
    if isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"-?\d+", v):
            iv = int(v)
            return iv if 0 <= iv < num_classes else None
        key = re.sub(r"[^a-z0-9]", "", v.lower())
        return name_to_id.get(key)
    return None


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def cmd_audit(cfg):
    import random
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = None
    class_names = load_class_names(cfg["data_root"], cfg["num_classes"])
    ids = read_test_ids(cfg)
    print(f"test parts: {len(ids)}   step dir: {os.path.dirname(step_path(cfg, ids[0]))}")
    print(f"classes ({len(class_names)}): {class_names}\n")

    rng = random.Random(0)
    sample = rng.sample(ids, min(10, len(ids)))
    raw_t, strip_t = [], []
    print(f"{'part':>8} {'faces':>6} {'rawtok':>8} {'striptok':>8}  leak_in_raw")
    for pid in sample:
        txt = open(step_path(cfg, pid)).read()
        faces = parse_faces(txt)
        stripped = strip_labels(txt)
        leak = _FACE_NAME_RE.search(txt) is not None
        rt = len(enc.encode(txt)) if enc else -1
        st = len(enc.encode(stripped)) if enc else -1
        raw_t.append(rt); strip_t.append(st)
        print(f"{pid:>8} {len(faces):>6} {rt:>8} {st:>8}  {leak}")
        # leak proof: stripped text must contain ZERO ADVANCED_FACE labels
        assert _FACE_NAME_RE.search(stripped) is None
    if enc:
        print(f"\nstripped tokens: min {min(strip_t)} median "
              f"{int(np.median(strip_t))} max {max(strip_t)}  "
              f"(gpt-4o-mini ctx=128k -> fits, no chunking)")
    print("\nLEAK CHECK: the ADVANCED_FACE name field equals the H5 label; raw text "
          "is the answer key. --run strips it by default.")
    print("ALIGNMENT: i-th ADVANCED_FACE entity (file order) == H5 label index i.")


# ---------------------------------------------------------------------------
# runner (async, bounded concurrency, resumable)
# ---------------------------------------------------------------------------
def load_completed(path):
    done = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["part_id"])
                except Exception:
                    continue
    return done


async def run_one(client, sem, pid, cfg, class_names, keep_labels, model, temperature):
    txt = open(step_path(cfg, pid)).read()
    faces = parse_faces(txt)  # ground-truth-derived ordering & true labels
    send_text = txt if keep_labels else strip_labels(txt)
    messages = build_messages(class_names, send_text)
    async with sem:
        t0 = time.monotonic()
        resp = await client.chat.completions.create(
            model=model, temperature=temperature, messages=messages,
            response_format={"type": "json_object"},
        )
        dt = time.monotonic() - t0
    choice = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "part_id": pid,
        "raw_response": choice,
        "n_faces": len(faces),
        "entity_ids": [eid for eid, _ in faces],
        "true_labels": [lbl for _, lbl in faces],
        "latency_s": round(dt, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


async def cmd_run(cfg, args):
    from openai import AsyncOpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    paths = out_paths(args.output)
    class_names = load_class_names(cfg["data_root"], cfg["num_classes"])
    ids = read_test_ids(cfg)
    if args.limit:
        ids = ids[:args.limit]
    done = load_completed(paths["results"])
    todo = [p for p in ids if p not in done]
    print(f"model={args.model} temp={args.temperature} concurrency={args.concurrency} "
          f"keep_labels={args.keep_labels}")
    print(f"total={len(ids)} already_done={len(ids) - len(todo)} todo={len(todo)}")
    if args.keep_labels:
        print("!! --keep-labels: label LEAK is active; numbers are NOT valid for comparison.")
    if not todo:
        print("nothing to do."); return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    res_f = open(paths["results"], "a")
    err_f = open(paths["errors"], "a")
    n_ok = n_err = 0
    t_start = time.monotonic()
    lock = asyncio.Lock()

    async def worker(pid):
        nonlocal n_ok, n_err
        try:
            rec = await run_one(client, sem, pid, cfg, class_names,
                                args.keep_labels, args.model, args.temperature)
            async with lock:
                res_f.write(json.dumps(rec) + "\n"); res_f.flush()
                n_ok += 1
                if (n_ok + n_err) % 50 == 0:
                    rate = (n_ok + n_err) / (time.monotonic() - t_start)
                    print(f"  {n_ok+n_err}/{len(todo)}  ok={n_ok} err={n_err} "
                          f"{rate:.1f}/s", flush=True)
        except Exception as e:
            async with lock:
                err_f.write(json.dumps({"part_id": pid, "error": repr(e)}) + "\n")
                err_f.flush()
                n_err += 1

    await asyncio.gather(*(worker(p) for p in todo))
    res_f.close(); err_f.close()
    dt = time.monotonic() - t_start
    print(f"\ndone: ok={n_ok} api_errors={n_err} in {dt:.0f}s "
          f"({(n_ok+n_err)/max(dt,1):.1f} calls/s)")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def cmd_eval(cfg, output=None):
    paths = out_paths(output)
    class_names = load_class_names(cfg["data_root"], cfg["num_classes"])
    C = cfg["num_classes"]
    name_to_id = build_name_to_id(class_names)
    test_ids = set(read_test_ids(cfg))

    if not os.path.isfile(paths["results"]):
        raise SystemExit(f"no {paths['results']}; run --run first")

    # Confusion matrices:
    #   parsed-only: [C, C]            (faces with a valid predicted class)
    #   full:        [C, C+1]          (col C == missing/malformed -> wrong)
    cm_parsed = np.zeros((C, C), dtype=np.int64)
    cm_full = np.zeros((C, C + 1), dtype=np.int64)

    n_parts = n_faces_total = n_faces_pred = 0
    malformed_json = extra_faces = missing_faces = bad_class = 0
    prompt_tok = compl_tok = 0
    latencies = []
    malformed_part_ids = []

    with open(paths["results"]) as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["part_id"]
            if pid not in test_ids:
                continue
            n_parts += 1
            prompt_tok += rec.get("prompt_tokens") or 0
            compl_tok += rec.get("completion_tokens") or 0
            if rec.get("latency_s") is not None:
                latencies.append(rec["latency_s"])

            entity_ids = rec["entity_ids"]
            true_labels = rec["true_labels"]
            n_faces_total += len(true_labels)

            try:
                obj = extract_json_obj(rec["raw_response"])
                if not isinstance(obj, dict):
                    raise ValueError("top-level JSON is not an object")
            except Exception:
                malformed_json += 1
                malformed_part_ids.append(pid)
                # whole part unscored in (a); all faces wrong in (b)
                for t in true_labels:
                    cm_full[t, C] += 1
                continue

            # normalize predicted keys: accept "#17", "17", 17
            pred_map = {}
            for k, v in obj.items():
                ks = str(k).strip()
                if not ks.startswith("#"):
                    ks = "#" + ks.lstrip("#")
                pred_map[ks] = v
            gt_set = set(entity_ids)
            extra_faces += sum(1 for k in pred_map if k not in gt_set)

            for eid, t in zip(entity_ids, true_labels):
                if eid not in pred_map:
                    missing_faces += 1
                    cm_full[t, C] += 1
                    continue
                p = coerce_class(pred_map[eid], name_to_id, C)
                if p is None:
                    bad_class += 1
                    cm_full[t, C] += 1
                    continue
                cm_parsed[t, p] += 1
                cm_full[t, p] += 1
                n_faces_pred += 1

    # ---- metrics (a): parsed-only ----
    prec_a, rec_a, f1_a, sup_a = per_class_metrics(cm_parsed)
    total_a = cm_parsed.sum()
    acc_a = np.diag(cm_parsed).sum() / max(total_a, 1)
    present_a = sup_a > 0
    macrof1_a = f1_a[present_a].mean() if present_a.any() else 0.0

    # ---- metrics (b): full set, missing/malformed -> wrong ----
    cm_b_real = cm_full[:, :C]              # C x C real predictions
    tp_b = np.diag(cm_b_real).astype(float)
    support_b = cm_full.sum(axis=1).astype(float)        # all true faces
    pred_tot_b = cm_b_real.sum(axis=0).astype(float)     # predicted per real class
    prec_b = np.divide(tp_b, pred_tot_b, out=np.zeros_like(tp_b), where=pred_tot_b > 0)
    rec_b = np.divide(tp_b, support_b, out=np.zeros_like(tp_b), where=support_b > 0)
    den_b = prec_b + rec_b
    f1_b = np.divide(2 * prec_b * rec_b, den_b, out=np.zeros_like(tp_b), where=den_b > 0)
    total_b = cm_full.sum()
    acc_b = tp_b.sum() / max(total_b, 1)
    present_b = support_b > 0
    macrof1_b = f1_b[present_b].mean() if present_b.any() else 0.0

    n_malformed_faces = missing_faces + bad_class + \
        (n_faces_total - n_faces_pred - missing_faces - bad_class)
    face_fail_rate = (n_faces_total - n_faces_pred) / max(n_faces_total, 1)

    # cost (gpt-4o-mini, public pricing as of 2024-2025): $0.15 / 1M in, $0.60 / 1M out
    cost = prompt_tok / 1e6 * 0.15 + compl_tok / 1e6 * 0.60

    def weak_block(f1, sup):
        return [(c, class_names[c], f1[c], int(sup[c])) for c in WEAK_CLASS_IDS]

    lines = []
    def out(s=""):
        lines.append(s); print(s)

    out("=" * 70)
    out("LLM BASELINE (GPT-4o-mini, raw STEP text) — RECONSTRUCTION")
    out("  NOTE: cofounder's exact MVP prompt was NOT found; this is a comparable")
    out("        reconstruction (zero-shot, label name-field stripped to prevent leak).")
    out("=" * 70)
    out(f"parts scored        : {n_parts}")
    out(f"faces (ground truth): {n_faces_total}")
    out(f"faces predicted ok  : {n_faces_pred}")
    out(f"malformed JSON parts: {malformed_json}  -> {malformed_part_ids[:10]}"
        f"{'...' if len(malformed_part_ids) > 10 else ''}")
    out(f"missing faces       : {missing_faces}")
    out(f"unrecognized class  : {bad_class}")
    out(f"extra (hallucinated) faces not in GT: {extra_faces}")
    out(f"face failure rate   : {face_fail_rate:.4f}  (unscored in (a), wrong in (b))")
    out(f"tokens: prompt={prompt_tok:,}  completion={compl_tok:,}  "
        f"approx cost=${cost:.2f}")
    if latencies:
        out(f"avg response time   : {np.mean(latencies):.2f}s")
    out("")
    out("(a) PARSED-ONLY (faces with a valid prediction):")
    out(f"    accuracy = {acc_a:.4f}   macro-F1 = {macrof1_a:.4f}   "
        f"(classes present = {int(present_a.sum())})")
    out("(b) FULL TEST SET (missing/malformed counted as WRONG) <- fair end-to-end:")
    out(f"    accuracy = {acc_b:.4f}   macro-F1 = {macrof1_b:.4f}   "
        f"(classes present = {int(present_b.sum())})")
    out("")
    out("weak-class F1 (b, full set):")
    out(f"  {'id':>3} {'class':<28} {'f1':>7} {'support':>8}")
    for c, nm, f1v, sv in weak_block(f1_b, support_b):
        out(f"  {c:>3} {nm:<28} {f1v:>7.3f} {sv:>8}")
    weak_macro_b = np.mean([f1_b[c] for c in WEAK_CLASS_IDS])
    out(f"  weak-class mean F1 (b) = {weak_macro_b:.4f}")
    out("")

    # ---- comparison table row ----
    out("=" * 70)
    out("COMPARISON TABLE ROW (paste into baseline_results.txt SUMMARY)")
    out("=" * 70)
    out(f"{'Model':<40}{'Acc':>9}{'MacroF1':>9}{'WeakF1':>9}{'Malf%':>8}{'Cost$':>8}")
    out("-" * 83)
    malf_rate_parts = malformed_json / max(n_parts, 1)
    out(f"{'LLM (GPT-4o-mini, raw STEP text)':<40}{acc_b:>9.4f}{macrof1_b:>9.4f}"
        f"{weak_macro_b:>9.4f}{malf_rate_parts*100:>7.2f}%{cost:>8.2f}")
    out("  (row uses (b) full-set numbers; malf% = malformed-JSON parts / parts scored)")

    # ---- write confusion matrix (full, with missing column) ----
    with open(paths["cm"], "w") as f:
        hdr = "true\\pred," + ",".join(str(i) for i in range(C)) + ",MISSING\n"
        f.write(hdr)
        for i in range(C):
            f.write(str(i) + "," + ",".join(str(int(v)) for v in cm_full[i]) + "\n")
    out(f"\nwrote {paths['cm']}")

    with open(paths["report"], "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {paths['report']}")

    return {
        "n_parts": n_parts, "n_faces_total": n_faces_total,
        "n_faces_pred": n_faces_pred, "acc_a": acc_a, "macrof1_a": macrof1_a,
        "acc_b": acc_b, "macrof1_b": macrof1_b, "f1_b": f1_b,
        "support_b": support_b, "weak_macro_b": weak_macro_b,
        "malformed_parts": malformed_json, "malf_rate_parts": malf_rate_parts,
        "face_fail_rate": face_fail_rate, "cost": cost,
        "prompt_tok": prompt_tok, "compl_tok": compl_tok,
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--audit", action="store_true", help="print leak/token/alignment audit")
    ap.add_argument("--run", action="store_true", help="run the LLM over the test set")
    ap.add_argument("--eval", action="store_true", help="score the results file")
    ap.add_argument("--limit", type=int, default=0, help="only first N test parts (smoke test)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--keep-labels", action="store_true",
                    help="DO NOT strip leaked labels (demonstrates leak; invalid metrics)")
    ap.add_argument("--output", default=None,
                    help="artifact path PREFIX (keeps concurrent runs separate); "
                         "writes <prefix>.jsonl/.errors.jsonl/.report.txt/.cm.csv")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.audit:
        cmd_audit(cfg)
    if args.run:
        asyncio.run(cmd_run(cfg, args))
    if args.eval:
        cmd_eval(cfg, args.output)
    if not (args.audit or args.run or args.eval):
        ap.print_help()


if __name__ == "__main__":
    main()
