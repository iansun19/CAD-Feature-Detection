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
that ordering, then score against the same 12-class metric evaluate.py uses.

USAGE
-----
  python llm_baseline.py --audit                 # print leak/token/alignment audit
  python llm_baseline.py --run [--limit N] [--concurrency 4] [--tpm 180000]
                               [--max-retries 5] [--keep-labels] [--output PREFIX]
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
from collections import deque

import numpy as np
import yaml

from evaluate import load_class_names, per_class_metrics
from dataset import _brep_bounds, _edge_set, _canonical_edge, SPLIT_H5
from taxonomy import NEW_DESCRIPTIONS, NUM_CLASSES

# ---------------------------------------------------------------------------
# config / constants
# ---------------------------------------------------------------------------
# Interim weak-class set (legacy ids remapped via old_to_new, deduped). Mirror heuristic_baseline.py.
# TODO: re-derive from a fresh 12-class confusion matrix after the first 12-class eval.
WEAK_CLASS_IDS = (9, 2, 3, 7)
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
# gpt-4o-mini context = 128k. Leave headroom for system prompt + output.
MAX_INPUT_TOKENS = 110_000
# org TPM ceiling for gpt-4o-mini is 200k; default budget leaves ~10% headroom so
# our own throttle keeps us under it instead of relying on 429s.
DEFAULT_TPM = 180_000
DEFAULT_MAX_RETRIES = 5

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
def build_system_prompt(class_names, bounded=False):
    listing = "\n".join(f"  {i} - {n}" for i, n in enumerate(class_names))
    if bounded:
        out_fmt = (
            "OUTPUT FORMAT (STRICT): you will be given the EXACT list of face entity "
            "ids to classify. Return a single JSON object whose keys are EXACTLY those "
            "ids (and no others) mapping to the integer class id. Example:\n"
            '{"#17": 24, "#619": 3, "#808": 15}\n'
            "Classify only the listed faces — do NOT invent or enumerate any other "
            "entity ids. Output ONLY the JSON object — no markdown, no commentary."
        )
    else:
        out_fmt = (
            "OUTPUT FORMAT (STRICT): a single JSON object mapping each face entity id "
            '(string, including the leading "#") to its integer class id. Example:\n'
            '{"#17": 24, "#619": 3, "#808": 15}\n'
            "Include EVERY face entity id present in the file, exactly once. Output "
            "ONLY the JSON object — no markdown, no commentary."
        )
    return (
        "You are an expert in CAD B-rep geometry and CNC machining features. "
        "You are given the raw text of an ISO-10303-21 (STEP) file describing a "
        "single solid part. Every B-rep face appears as an entity of the form "
        "`#N = ADVANCED_FACE('', (...loops...), #surface, .T./.F.);` where `#N` is "
        "the face's entity id. The face's name field has been intentionally blanked.\n\n"
        "Classify EVERY face in the part into exactly one of these 12 machining-"
        "feature classes (use the integer id):\n"
        f"{listing}\n\n"
        "Reason from the geometry: surface type (PLANE/CYLINDRICAL_SURFACE/CONICAL_"
        "SURFACE/etc.), the edge loops, how faces bound pockets/slots/steps/holes, "
        "and which faces are the original stock surfaces (class 11).\n\n"
        + out_fmt
    )


def build_messages(class_names, step_text, face_ids=None, bounded=False):
    user = "STEP file:\n\n" + step_text
    if bounded:
        ids = " ".join(face_ids)
        user += (f"\n\nClassify EXACTLY these {len(face_ids)} face entity ids "
                 f"(return a class for every one, and include no other ids):\n{ids}")
    return [
        {"role": "system", "content": build_system_prompt(class_names, bounded)},
        {"role": "user", "content": user},
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


def coerce_class(value, name_to_id, num_classes=NUM_CLASSES):
    """Map a model-emitted value (int id, numeric string, or class name) to an id."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < NUM_CLASSES else None
    if isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"-?\d+", v):
            iv = int(v)
            return iv if 0 <= iv < NUM_CLASSES else None
        key = re.sub(r"[^a-z0-9]", "", v.lower())
        return name_to_id.get(key)
    return None


# ---------------------------------------------------------------------------
# --features mode: serialize the SAME geometric features the GNN consumes
# ---------------------------------------------------------------------------
# This block is purely ADDITIVE. It does not touch parse_faces / strip_labels /
# build_messages / the runner's retry/backoff/heartbeat/resumable machinery — it
# only swaps the INPUT representation (raw STEP text -> serialized features) when
# --features is set. Features come from the regenerated MFCAD++ H5 (the exact
# source dataset.py's build_node_features_regen / build_edge_features_regen read),
# NOT from re-parsing STEP geometry. No pythonocc / STEP kernel is used.
#
# IMPORTANT — area units. The spec asked for area in cm² via exp(log_area). The
# GNN's actual source (regen V_1 col 0) stores area PER-PART MIN-MAX NORMALIZED to
# [0,1] (0 = smallest face in the part, 1 = largest); there is no log-area and no
# physical cm² recoverable from it. To stay faithful to "the same features the GNN
# uses", we serialize that relative area and label it as such rather than inventing
# a cm² value. Normal is the exact per-face unit vector; centroid is the per-part
# [0,1]-normalized centroid; plane offset d is the raw signed plane distance (real
# model units), so it is only emitted for planar faces.
SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "other"]
CONVEXITY_NAMES = ["concave", "convex", "smooth"]  # 0=concave,1=convex,2=smooth

# One-line geometric signature of each MFCAD++ class (ids match feature_labels.txt).
CLASS_DESCRIPTIONS = NEW_DESCRIPTIONS

# Few-shot exemplar classes, spanning simple (Stock) to hard orientation-dependent.
# Remapped from legacy [0, 6, 10, 17, 24] via old_to_new → [9, 2, 3, 7, 11].
FEWSHOT_CLASSES = [9, 2, 3, 7, 11]


def _regen_h5_path(cfg, split_file):
    return os.path.join(cfg["data_root"], cfg["h5_dir"], SPLIT_H5[split_file])


def open_regen_split(cfg, split_file):
    """Open a regenerated split H5 and build a pid -> (batch_key, model_idx) index."""
    import h5py
    h5 = h5py.File(_regen_h5_path(cfg, split_file), "r")
    index = {}
    for bk in h5.keys():
        for i, raw in enumerate(h5[bk]["CAD_model"][()]):
            pid = raw.decode() if isinstance(raw, bytes) else str(raw)
            index[pid] = (bk, i)
    return h5, index


def part_features(h5, index, cfg, pid):
    """Per-face geometric features for one part, in H5/file/label order.

    Mirrors MFCADPPRegenGraphDataset._read_sample's edge construction exactly
    (concave-first convexity, median dihedral) but keeps the raw V_1 values and
    per-face neighbour lists for human-readable serialization. Returns a list of
    dicts, one per face, index i aligning with label index i (== i-th ADVANCED_FACE).
    """
    nst = cfg["num_surface_types"]
    bk, mi = index[pid]
    batch = h5[bk]
    idx_arr = batch["idx"][()]
    v1_len = batch["V_1"].shape[0]
    s, e = _brep_bounds(idx_arr, mi, v1_len)
    v1 = np.asarray(batch["V_1"][s:e], dtype=np.float32)
    n = e - s

    def eset(name):
        return {_canonical_edge(u, v) for u, v in _edge_set(batch[name][()], s, e)
                if u != v}
    e1, e2, e3 = eset("E_1_idx"), eset("E_2_idx"), eset("E_3_idx")

    a1 = batch["A_1_idx"][()]
    av = batch["A_1_values"][()]
    mask = ((a1[:, 0] >= s) & (a1[:, 0] < e) &
            (a1[:, 1] >= s) & (a1[:, 1] < e))
    rows = a1[mask] - s
    vals = av[mask]
    ang_by_pair = {}
    for (u, v), ang in zip(rows.tolist(), vals.tolist()):
        if u == v:
            continue
        ang_by_pair.setdefault(_canonical_edge(u, v), []).append(float(ang))

    reduce_fn = np.mean if cfg.get("angle_reduce") == "mean" else np.median
    # build directed adjacency in the same column order make_undirected produces:
    # all forward pairs first, then all reversed pairs.
    pairs, conv, deg = [], [], []
    for key, angs in ang_by_pair.items():
        if key in e2:        # concave-first precedence (keep the concave edge)
            cid = 0
        elif key in e1:
            cid = 1
        elif key in e3:
            cid = 2
        else:
            cid = 1
        cosv = float(np.cos(float(reduce_fn(angs))))
        pairs.append(key)
        conv.append(cid)
        deg.append(round(float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))), 1))

    nbrs = [[] for _ in range(n)]
    for i, (u, v) in enumerate(pairs):
        nbrs[u].append((v, CONVEXITY_NAMES[conv[i]], deg[i]))
    for i, (u, v) in enumerate(pairs):
        nbrs[v].append((u, CONVEXITY_NAMES[conv[i]], deg[i]))

    faces = []
    for i in range(n):
        t = int(np.clip(round(float(v1[i, 4]) * 11) - 1, 0, nst - 1))
        faces.append({
            "type": SURFACE_TYPES[t],
            "area": float(v1[i, 0]),
            "normal": (float(v1[i, 5]), float(v1[i, 6]), float(v1[i, 7])),
            "centroid": (float(v1[i, 1]), float(v1[i, 2]), float(v1[i, 3])),
            "plane_d": float(v1[i, 8]),
            "neighbors": nbrs[i],
        })
    return faces


def serialize_face(face, entity_ids, idx):
    """One human-readable face block. entity_ids maps face index -> '#NNN'."""
    nx, ny, nz = face["normal"]
    cx, cy, cz = face["centroid"]
    lines = [f"Face {entity_ids[idx]} [{face['type']}]"]
    lines.append(f"  relative area: {face['area']:.3f}  (0 = smallest, 1 = largest face in part)")
    lines.append(f"  normal: ({nx:.2f}, {ny:.2f}, {nz:.2f})")
    lines.append(f"  centroid (normalized 0-1): ({cx:.2f}, {cy:.2f}, {cz:.2f})")
    if face["type"] == "plane":
        lines.append(f"  plane offset d: {face['plane_d']:.2f}")
    nb = face["neighbors"]
    lines.append(f"  neighbors ({len(nb)}):")
    for nidx, cstr, d in nb:
        lines.append(f"    → {entity_ids[nidx]:<6} {cstr:<8} {d:.1f}°")
    return "\n".join(lines)


def serialize_part(faces, entity_ids):
    """Serialize all faces of a part. Requires feature count == STEP face count."""
    if len(faces) != len(entity_ids):
        raise ValueError(
            f"feature/STEP face count mismatch: {len(faces)} features vs "
            f"{len(entity_ids)} ADVANCED_FACE entities")
    return "\n\n".join(serialize_face(f, entity_ids, i) for i, f in enumerate(faces))


def build_fewshot(cfg, class_names):
    """Build the fixed few-shot block from the TRAINING set ONLY (never val/test,
    so there is no leakage into the test-set evaluation). One representative face
    is drawn for each of FEWSHOT_CLASSES from the first training part that contains
    that class. Computed once and reused as fixed context for every API call."""
    h5, index = open_regen_split(cfg, "train.txt")  # TRAIN split — confirmed no leak
    try:
        train_ids = []
        with open(os.path.join(cfg["data_root"], "train.txt")) as f:
            train_ids = [ln.strip() for ln in f if ln.strip()]
        train_ids = [p for p in train_ids if p in index]

        blocks = []
        for cls in FEWSHOT_CLASSES:
            found = None
            for pid in train_ids:
                bk, mi = index[pid]
                batch = h5[bk]
                s, e = _brep_bounds(batch["idx"][()], mi, batch["V_1"].shape[0])
                labels = np.asarray(batch["labels"][s:e], dtype=np.int64)
                hits = np.where(labels == cls)[0]
                if len(hits):
                    faces = part_features(h5, index, cfg, pid)
                    # fabricate entity ids local to the example (only for display)
                    eids = [f"#{j}" for j in range(len(faces))]
                    fi = int(hits[0])
                    block = serialize_face(faces[fi], eids, fi)
                    found = f"{block}\n→ class: {cls} ({class_names[cls]})"
                    break
            if found is None:
                found = f"(no training example found for class {cls})"
            blocks.append(found)
        return "\n\n".join(blocks)
    finally:
        h5.close()


# ---------------------------------------------------------------------------
# --templates mode (additive on top of --features): a full 12-class canonical
# template bank replaces the 5-example few-shot block, so EVERY class has one
# representative face in the prompt and the model is constrained to pick exactly
# one of the 12 class ids per face.
# ---------------------------------------------------------------------------
# The 12 canonical (part_id, face_index, class_id) tuples below were computed ONCE
# from the TRAINING SPLIT ONLY (never val/test — no leakage into test evaluation)
# by _build_templates.py against hierarchical_graphs_regen_12: for each class, the mean
# 14-dim node-feature vector over all training faces of that class (centroid in the
# same feature space the GNN uses, build_node_features_regen), then the actual
# training face nearest that centroid in L2. Re-run _build_templates.py to regenerate.
TEMPLATE_FACES = [
    ('33715', 11, 0),
    ('28157', 21, 1),
    ('3767', 26, 2),
    ('47084', 11, 3),
    ('39886', 35, 4),
    ('30861', 13, 5),
    ('19406', 18, 6),
    ('35580', 8, 7),
    ('49309', 24, 8),
    ('9346', 2, 9),
    ('12108', 2, 10),
    ('37842', 0, 11),
]


def build_template_bank(cfg, class_names):
    """Serialize the 12 canonical class templates from the TRAINING split ONLY.

    Uses the SAME serialize_face / per-face block format as --features mode so the
    template representation is byte-for-byte comparable to the test faces. Returns
    the full template-bank text (12 labelled blocks). Confirmed TRAIN-only: opened
    from train.txt's H5; TEMPLATE_FACES are training part ids."""
    h5, index = open_regen_split(cfg, "train.txt")  # TRAIN split — confirmed no leak
    try:
        blocks = []
        for pid, fi, cls in TEMPLATE_FACES:
            header = f"=== CLASS TEMPLATE: {class_names[cls]} (class {cls}) ==="
            if pid not in index:
                blocks.append(f"{header}\n(template part {pid} not found in train H5)")
                continue
            faces = part_features(h5, index, cfg, pid)
            eids = [f"#{j}" for j in range(len(faces))]  # local display ids
            block = serialize_face(faces[fi], eids, fi)
            blocks.append(f"{header}\n{block}\n→ LABEL: {class_names[cls]} (class {cls})")
        return "\n\n".join(blocks)
    finally:
        h5.close()


def build_template_system_prompt(class_names, template_text):
    listing = "\n".join(f"  {i} - {CLASS_DESCRIPTIONS[i]}" for i in range(len(class_names)))
    out_fmt = (
        "OUTPUT FORMAT (STRICT): you will be given the EXACT list of face entity "
        "ids to classify. Return a single JSON object whose keys are EXACTLY those "
        "ids (and no others) mapping to the integer class id. Example:\n"
        '{"#17": 24, "#619": 3, "#808": 15}\n'
        "Classify only the listed faces — do NOT invent or enumerate any other "
        "entity ids. You MUST output a prediction for every listed id. Output ONLY "
        "the JSON object — no markdown, no commentary."
    )
    return (
        "You are an expert in CAD B-rep geometry and CNC machining features. "
        "Instead of raw STEP text, you are given a SERIALIZED, decoded description "
        "of every face of a single solid part: its surface type, relative area, "
        "exact unit normal, normalized centroid, plane offset, and its adjacency to "
        "other faces (each neighbour labelled concave / convex / smooth with the "
        "dihedral angle in degrees). These are the same geometric features a graph "
        "neural network uses.\n\n"
        "You must classify each face as EXACTLY ONE of the 12 classes listed below. "
        "Do not invent new classes, do not output fractional or combined labels, "
        "do not refuse to classify. If a face is ambiguous, pick the single closest "
        "match from the 12 templates above.\n\n"
        "The 12 machining-feature classes (use the integer id):\n"
        f"{listing}\n\n"
        "CANONICAL CLASS TEMPLATES — one representative face per class, in the exact "
        "same format as the faces you will classify. Match each test face to its "
        "closest template:\n\n"
        f"{template_text}\n\n"
        + out_fmt
    )


def build_template_messages(class_names, faces_text, face_ids, template_text):
    ids = " ".join(face_ids)
    user = (
        "PART FACES:\n\n" + faces_text +
        f"\n\nClassify EXACTLY these {len(face_ids)} face entity ids "
        f"(return a class for every one, and include no other ids):\n{ids}"
    )
    return [
        {"role": "system", "content": build_template_system_prompt(class_names, template_text)},
        {"role": "user", "content": user},
    ]


def cmd_templates_audit(cfg):
    """--features --templates --audit-only: print the 12 canonical templates (class,
    centroid distance, serialized block), confirm all 12 classes are covered, and
    estimate the full system-prompt token count. Makes NO API calls."""
    class_names = load_class_names(cfg["data_root"], cfg["num_classes"])
    enc = _enc()
    C = cfg["num_classes"]
    h5, index = open_regen_split(cfg, "train.txt")
    try:
        print("=" * 72)
        print("CANONICAL TEMPLATE BANK (TRAIN split only — one face per class)")
        print("=" * 72)
        covered = set()
        for pid, fi, cls in TEMPLATE_FACES:
            covered.add(cls)
            print("-" * 72)
            print(f"CLASS {cls} ({class_names[cls]})  <- train part {pid}, face #{fi}")
            if pid in index:
                faces = part_features(h5, index, cfg, pid)
                eids = [f"#{j}" for j in range(len(faces))]
                print(serialize_face(faces[fi], eids, fi))
            else:
                print(f"  *** train part {pid} not found ***")
        print("=" * 72)
        missing = [c for c in range(C) if c not in covered]
        if missing:
            print(f"*** MISSING TEMPLATES for classes: {missing} ***")
        else:
            print(f"COVERAGE OK: all {C} classes have a template.")
        # token estimate for the full system prompt (template bank included)
        template_text = build_template_bank(cfg, class_names)
        msgs = build_template_messages(class_names, "(faces here)", ["#1"], template_text)
        if enc:
            sys_tok = len(enc.encode(msgs[0]["content"]))
            flag = "  <-- EXCEEDS 8000 flag threshold" if sys_tok > 8000 else "  (under 8000 — OK)"
            print(f"system prompt (incl. 12-template bank) tokens: {sys_tok}{flag}")
            print("note: per-part face blocks (~1,500 tok) + this system prompt should "
                  "total ~3,250 tok/part; --run flags any part exceeding 6,000.")
        print("Confirm coverage + token count, then drop --audit-only to run.")
    finally:
        h5.close()


def build_feature_system_prompt(class_names, fewshot_text):
    listing = "\n".join(f"  {i} - {CLASS_DESCRIPTIONS[i]}" for i in range(len(class_names)))
    out_fmt = (
        "OUTPUT FORMAT (STRICT): you will be given the EXACT list of face entity "
        "ids to classify. Return a single JSON object whose keys are EXACTLY those "
        "ids (and no others) mapping to the integer class id. Example:\n"
        '{"#17": 24, "#619": 3, "#808": 15}\n'
        "Classify only the listed faces — do NOT invent or enumerate any other "
        "entity ids. You MUST output a prediction for every listed id. Output ONLY "
        "the JSON object — no markdown, no commentary."
    )
    return (
        "You are an expert in CAD B-rep geometry and CNC machining features. "
        "Instead of raw STEP text, you are given a SERIALIZED, decoded description "
        "of every face of a single solid part: its surface type, relative area, "
        "exact unit normal, normalized centroid, plane offset, and its adjacency to "
        "other faces (each neighbour labelled concave / convex / smooth with the "
        "dihedral angle in degrees). These are the same geometric features a graph "
        "neural network uses. Reason from this geometry — surface types, dihedral "
        "convexity, how faces bound pockets/slots/steps/holes, and which faces are "
        "the original stock surfaces (class 11).\n\n"
        "Classify EVERY listed face into exactly one of these 12 machining-feature "
        "classes (use the integer id):\n"
        f"{listing}\n\n"
        "WORKED EXAMPLES (input face block -> correct class):\n"
        f"{fewshot_text}\n\n"
        + out_fmt
    )


def build_feature_messages(class_names, faces_text, face_ids, fewshot_text):
    ids = " ".join(face_ids)
    user = (
        "PART FACES:\n\n" + faces_text +
        f"\n\nClassify EXACTLY these {len(face_ids)} face entity ids "
        f"(return a class for every one, and include no other ids):\n{ids}"
    )
    return [
        {"role": "system", "content": build_feature_system_prompt(class_names, fewshot_text)},
        {"role": "user", "content": user},
    ]


def _enc():
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def cmd_features_audit(cfg, n_parts=3):
    """--features --audit-only: serialize the first N test parts and print them so
    the representation can be inspected by eye BEFORE any API call is made."""
    class_names = load_class_names(cfg["data_root"], cfg["num_classes"])
    enc = _enc()
    ids = read_test_ids(cfg)
    h5, index = open_regen_split(cfg, "test.txt")
    try:
        print("=" * 72)
        print("FEW-SHOT BLOCK (fixed context, drawn from TRAINING set only):")
        print("=" * 72)
        fewshot = build_fewshot(cfg, class_names)
        print(fewshot)
        print()
        for pid in ids[:n_parts]:
            txt = open(step_path(cfg, pid)).read()
            faces_gt = parse_faces(txt)              # ground-truth ordering & labels
            entity_ids = [eid for eid, _ in faces_gt]
            feats = part_features(h5, index, cfg, pid)
            print("=" * 72)
            print(f"PART {pid}")
            ok = len(feats) == len(entity_ids)
            print(f"  face count: features={len(feats)}  STEP/GT={len(entity_ids)}  "
                  f"{'MATCH' if ok else '*** MISMATCH ***'}")
            if not ok:
                print("  (skipping serialization — counts must match)")
                continue
            text = serialize_part(feats, entity_ids)
            # full prompt token estimate
            face_ids = entity_ids
            messages = build_feature_messages(class_names, text, face_ids, fewshot)
            if enc:
                tot = sum(len(enc.encode(m["content"])) for m in messages)
                flag = "  <-- EXCEEDS 8000, consider truncating neighbors" if tot > 8000 else ""
                print(f"  full prompt tokens (system+fewshot+faces): {tot}{flag}")
            # plausibility summary
            all_deg = [d for f in feats for _, _, d in f["neighbors"]]
            nbr_counts = [len(f["neighbors"]) for f in feats]
            types = sorted({f["type"] for f in feats})
            areas = [f["area"] for f in feats]
            print(f"  surface types present: {types}")
            print(f"  relative area range: {min(areas):.3f}..{max(areas):.3f}")
            print(f"  neighbors/face: min {min(nbr_counts)} max {max(nbr_counts)}")
            if all_deg:
                print(f"  dihedral angle range: {min(all_deg):.1f}°..{max(all_deg):.1f}°")
            print("-" * 72)
            print(text)
            print()
        print("=" * 72)
        print("AUDIT CHECKLIST: confirm face counts MATCH, dihedral angles are in "
              "[0,180]°, surface types decode sensibly, and relative areas are in "
              "[0,1] (NOT log-scale). Only then proceed to --run.")
    finally:
        h5.close()


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


class TokenRateLimiter:
    """Sliding-60s-window token admission gate, sized to the org's TPM ceiling.

    Before a request is sent, acquire(est) reserves its estimated tokens and blocks
    until the tokens admitted in the trailing 60s + est stays under `tpm`. This keeps
    us under the limit by design instead of firing requests blindly and eating 429s.
    The lock is held only for the brief bookkeeping, never across a sleep, so a
    throttled request never blocks an unrelated one.
    """

    def __init__(self, tpm):
        self.tpm = tpm
        self.window = 60.0
        self.events = deque()        # (timestamp, tokens)
        self.lock = asyncio.Lock()

    def _purge(self, now):
        while self.events and now - self.events[0][0] > self.window:
            self.events.popleft()

    async def acquire(self, est):
        # a single request larger than the whole budget can only run alone
        est = min(est, self.tpm)
        while True:
            async with self.lock:
                now = time.monotonic()
                self._purge(now)
                cur = sum(t for _, t in self.events)
                if not self.events or cur + est <= self.tpm:
                    self.events.append((now, est))
                    return
                wait = self.window - (now - self.events[0][0])
            await asyncio.sleep(min(max(wait, 0.05), self.window))


def estimate_tokens(send_text, n_faces):
    """Rough TPM accounting: input chars/4 + system/task overhead + JSON output."""
    return len(send_text) // 4 + 700 + n_faces * 10


def parse_retry_after(exc):
    """Seconds the API asked us to wait, from headers or the error message."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        h = getattr(resp, "headers", {}) or {}
        if "retry-after-ms" in h:
            try:
                return float(h["retry-after-ms"]) / 1000.0
            except (TypeError, ValueError):
                pass
        if "retry-after" in h:
            try:
                return float(h["retry-after"])
            except (TypeError, ValueError):
                pass
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", str(exc))
    if m:
        v = float(m.group(1))
        return v / 1000.0 if m.group(2) == "ms" else v
    return None


async def run_one(client, sem, limiter, pid, cfg, class_names, keep_labels,
                  model, temperature, max_retries, bounded=False,
                  features=False, feature_prompts=None, fewshot_text=None,
                  templates=False, template_text=None):
    """Send one part, retrying 429s (server-suggested wait) and transient errors
    (exponential backoff) independently. Backoff sleeps happen OUTSIDE the
    concurrency semaphore so a throttled request frees its slot for others."""
    import openai
    txt = open(step_path(cfg, pid)).read()
    faces = parse_faces(txt)  # ground-truth-derived ordering & true labels
    face_ids = [eid for eid, _ in faces]
    if features:
        # input representation = serialized GNN features (NOT raw STEP text);
        # everything else (ordering, eval, retry, output) is identical.
        send_text = feature_prompts[pid]
        if templates:
            # 12-class canonical template bank replaces the few-shot examples;
            # identical serialization, eval, retry and output paths.
            messages = build_template_messages(class_names, send_text, face_ids, template_text)
        else:
            messages = build_feature_messages(class_names, send_text, face_ids, fewshot_text)
    else:
        send_text = txt if keep_labels else strip_labels(txt)
        messages = build_messages(class_names, send_text, face_ids, bounded)
    est = estimate_tokens(send_text, len(faces))

    attempt = 0
    while True:
        await limiter.acquire(est)          # TPM gate (waits without holding sem)
        wait = None
        try:
            async with sem:                 # bounds concurrent in-flight sockets
                t0 = time.monotonic()
                resp = await client.chat.completions.create(
                    model=model, temperature=temperature, messages=messages,
                    response_format={"type": "json_object"},
                )
                dt = time.monotonic() - t0
            choice = resp.choices[0].message.content or ""
            usage = resp.usage
            return {
                "part_id": pid, "raw_response": choice, "n_faces": len(faces),
                "entity_ids": [eid for eid, _ in faces],
                "true_labels": [lbl for _, lbl in faces],
                "latency_s": round(dt, 3), "attempts": attempt + 1,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        except openai.RateLimitError as e:
            attempt += 1
            if attempt > max_retries:
                raise
            ra = parse_retry_after(e)
            wait = (ra + 1.0) if ra is not None else min(2 ** attempt, 30)
        except (openai.APITimeoutError, openai.APIConnectionError,
                openai.InternalServerError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            ra = parse_retry_after(e)
            wait = (ra + 1.0) if ra is not None else min(2 ** attempt, 30)
        # sleep here, outside `async with sem`, so other requests keep flowing
        await asyncio.sleep(wait)


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
          f"tpm={args.tpm} max_retries={args.max_retries} keep_labels={args.keep_labels} "
          f"bounded={args.bounded} features={args.features} templates={args.templates}")
    print(f"total={len(ids)} already_done={len(ids) - len(todo)} todo={len(todo)}")
    if args.keep_labels:
        print("!! --keep-labels: label LEAK is active; numbers are NOT valid for comparison.")
    if not todo:
        print("nothing to do."); return

    # --features: precompute the serialized-feature prompt for every todo part up
    # front (single-threaded H5 reads, avoids h5py thread-safety issues) and build
    # the fixed few-shot block once. Parts whose feature count != STEP face count
    # are logged as errors and dropped before any API call.
    feature_prompts = None
    fewshot_text = None
    template_text = None
    if args.features:
        enc = _enc()
        if args.templates:
            template_text = build_template_bank(cfg, class_names)
        else:
            fewshot_text = build_fewshot(cfg, class_names)
        h5, index = open_regen_split(cfg, "test.txt")
        feature_prompts = {}
        skipped = []
        max_tok = 0
        err_f0 = open(paths["errors"], "a")
        try:
            for pid in todo:
                txt = open(step_path(cfg, pid)).read()
                entity_ids = [eid for eid, _ in parse_faces(txt)]
                try:
                    feats = part_features(h5, index, cfg, pid)
                    feature_prompts[pid] = serialize_part(feats, entity_ids)
                except Exception as e:
                    err_f0.write(json.dumps({"part_id": pid, "error": repr(e)}) + "\n")
                    skipped.append(pid)
                    continue
                if enc:
                    if args.templates:
                        msgs = build_template_messages(class_names, feature_prompts[pid],
                                                       entity_ids, template_text)
                    else:
                        msgs = build_feature_messages(class_names, feature_prompts[pid],
                                                      entity_ids, fewshot_text)
                    tk = sum(len(enc.encode(m["content"])) for m in msgs)
                    max_tok = max(max_tok, tk)
                    flag_thr = 6000 if args.templates else 8000
                    if tk > flag_thr:
                        print(f"!! {pid}: prompt is {tk} tokens (>{flag_thr}) — unusually "
                              f"large; consider truncating neighbors.")
        finally:
            err_f0.close()
            h5.close()
        todo = [p for p in todo if p in feature_prompts]
        if skipped:
            print(f"!! dropped {len(skipped)} parts on feature/STEP count mismatch: "
                  f"{skipped[:10]}")
        if enc:
            print(f"feature prompts ready: {len(todo)}  max prompt tokens={max_tok}")
        if not todo:
            print("nothing to do."); return

    client = AsyncOpenAI(max_retries=0)     # we own retry/backoff, not the SDK
    sem = asyncio.Semaphore(args.concurrency)
    limiter = TokenRateLimiter(args.tpm)
    res_f = open(paths["results"], "a")
    err_f = open(paths["errors"], "a")
    n_ok = n_err = in_flight = 0
    t_start = time.monotonic()
    lock = asyncio.Lock()

    async def worker(pid):
        nonlocal n_ok, n_err, in_flight
        in_flight += 1
        try:
            rec = await run_one(client, sem, limiter, pid, cfg, class_names,
                                args.keep_labels, args.model, args.temperature,
                                args.max_retries, args.bounded,
                                args.features, feature_prompts, fewshot_text,
                                args.templates, template_text)
            async with lock:
                res_f.write(json.dumps(rec) + "\n"); res_f.flush()
            n_ok += 1
        except Exception as e:
            async with lock:
                err_f.write(json.dumps({"part_id": pid, "error": repr(e)}) + "\n")
                err_f.flush()
            n_err += 1
        finally:
            in_flight -= 1

    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            elapsed = time.monotonic() - t_start
            rate = (n_ok + n_err) / max(elapsed, 1)
            print(f"[hb] {n_ok+n_err}/{len(todo)} done  ok={n_ok} failed={n_err} "
                  f"in_flight={in_flight}  {rate:.2f}/s  elapsed={elapsed:.0f}s",
                  flush=True)

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(worker(p) for p in todo))
    finally:
        hb.cancel()
    res_f.close(); err_f.close()
    dt = time.monotonic() - t_start
    print(f"\ndone: ok={n_ok} api_errors={n_err} in {dt:.0f}s "
          f"({(n_ok+n_err)/max(dt,1):.2f} calls/s)")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def cmd_eval(cfg, output=None, quiet=False):
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
        lines.append(s)
        if not quiet:
            print(s)

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
    ap.add_argument("--concurrency", type=int, default=4,
                    help="max in-flight requests (the TPM limiter is the real throttle)")
    ap.add_argument("--tpm", type=int, default=DEFAULT_TPM,
                    help="token-per-minute budget; throttle stays under the org ceiling")
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                    help="per-request retries for 429 / transient errors")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--keep-labels", action="store_true",
                    help="DO NOT strip leaked labels (demonstrates leak; invalid metrics)")
    ap.add_argument("--bounded", action="store_true",
                    help="give the model the explicit face-id list (bounds output, "
                         "prevents runaway entity enumeration); free-range if omitted")
    ap.add_argument("--features", action="store_true",
                    help="input = serialized GNN geometric features instead of raw "
                         "STEP text (bounded face-id list; same eval/output path)")
    ap.add_argument("--templates", action="store_true",
                    help="with --features: replace the 5-example few-shot block with "
                         "a full 12-class canonical template bank (one face per class) "
                         "and constrain output to exactly one of the 12 class ids")
    ap.add_argument("--audit-only", dest="audit_only", action="store_true",
                    help="with --features: serialize 3 test parts and print them for "
                         "inspection, make NO API calls")
    ap.add_argument("--output", default=None,
                    help="artifact path PREFIX (keeps concurrent runs separate); "
                         "writes <prefix>.jsonl/.errors.jsonl/.report.txt/.cm.csv")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.templates and not args.features:
        raise SystemExit("--templates requires --features (they share the "
                         "serialization pipeline). Pass both, e.g. --features --templates.")
    if args.features and args.templates and args.audit_only:
        cmd_templates_audit(cfg)
        return
    if args.features and args.audit_only:
        cmd_features_audit(cfg)
        return
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
