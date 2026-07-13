"""
Relabel MFCAD++ ground truth from 25 classes → 12 classes.

Writes NEW copies only (original H5 + STEP untouched):
  hierarchical_graphs_regen/     → hierarchical_graphs_regen_12/
  MFCAD++_dataset/step/<split>/  → MFCAD++_dataset/step_12class/<split>/

Mapping is imported from taxonomy.py (single source of truth).
Idempotent: outputs carry sentinel mfcadpp_12class_v1; re-run skips migrated files.

Usage:
  python relabel_12class.py [--jobs N] [--sample-parts ID,ID,...]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brep.taxonomy import NUM_CLASSES, new_name, old_to_new  # noqa: E402

SENTINEL = "mfcadpp_12class_v1"
STEP_SENTINEL_LINE = f"/* MFCAD++ 12-class relabel ({SENTINEL}) */"

H5_SRC_DIR = os.path.join("MFCAD++_dataset", "hierarchical_graphs_regen")
H5_DST_DIR = os.path.join("MFCAD++_dataset", "hierarchical_graphs_regen_12")
STEP_SRC_ROOT = os.path.join("MFCAD++_dataset", "step")
STEP_DST_ROOT = os.path.join("MFCAD++_dataset", "step_12class")

SPLIT_H5 = {
    "train": "training_MFCAD++.h5",
    "val": "val_MFCAD++.h5",
    "test": "test_MFCAD++.h5",
}
SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}

_FACE_NAME_RE = re.compile(r"(ADVANCED_FACE\(\s*')(\d+)(')")
_FACE_ENTITY_RE = re.compile(r"ADVANCED_FACE\s*\(")


def _assert_source_not_migrated_h5(path: str) -> None:
    with h5py.File(path, "r") as f:
        for batch_key in f.keys():
            grp = f[batch_key]
            if grp.attrs.get("taxonomy") == SENTINEL:
                raise RuntimeError(
                    f"H5 source {path} batch {batch_key!r} already has taxonomy "
                    f"sentinel — refusing to double-map"
                )
            labels = grp.get("labels")
            if labels is not None and labels.attrs.get("taxonomy") == SENTINEL:
                raise RuntimeError(
                    f"H5 source {path} labels in {batch_key!r} already migrated"
                )


def h5_output_migrated(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with h5py.File(path, "r") as f:
        if not f.keys():
            return False
        for batch_key in f.keys():
            grp = f[batch_key]
            if grp.attrs.get("taxonomy") != SENTINEL:
                return False
            if grp["labels"].attrs.get("taxonomy") != SENTINEL:
                return False
    return True


def _map_old_labels(raw: np.ndarray) -> tuple[np.ndarray, Counter, Counter]:
    old_int = np.round(raw).astype(np.int64)
    if old_int.min() < 0 or old_int.max() > 24:
        bad = old_int[(old_int < 0) | (old_int > 24)]
        raise ValueError(f"label outside 0–24: {bad[:5]}")
    old_hist = Counter(int(x) for x in old_int)
    mapped = np.array([old_to_new(int(x)) for x in old_int], dtype=np.int64)
    new_hist = Counter(int(x) for x in mapped)
    if mapped.min() < 0 or mapped.max() >= NUM_CLASSES:
        raise ValueError(f"mapped label outside 0–{NUM_CLASSES - 1}")
    return mapped.astype(np.float32), old_hist, new_hist


def relabel_h5_file(src_path: str, dst_path: str) -> str:
    """Return 'written', 'skip', or raise."""
    if h5_output_migrated(dst_path):
        return "skip"
    _assert_source_not_migrated_h5(src_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    file_old = Counter()
    file_new = Counter()
    n_faces = 0

    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        for batch_key in fin.keys():
            src_grp = fin[batch_key]
            dst_grp = fout.create_group(batch_key)
            n_src_faces = int(src_grp["labels"].shape[0])

            for ds_key in src_grp.keys():
                if ds_key == "labels":
                    mapped, oh, nh = _map_old_labels(np.asarray(src_grp["labels"][()]))
                    if mapped.shape[0] != n_src_faces:
                        raise AssertionError(f"{batch_key}: label count mismatch")
                    ds = dst_grp.create_dataset(
                        "labels", data=mapped, compression="lzf"
                    )
                    ds.attrs["taxonomy"] = SENTINEL
                    file_old.update(oh)
                    file_new.update(nh)
                    n_faces += mapped.shape[0]
                else:
                    fin.copy(src_grp[ds_key], dst_grp, ds_key)
                    src_ds = src_grp[ds_key]
                    dst_ds = dst_grp[ds_key]
                    if not np.array_equal(np.asarray(src_ds[()]), np.asarray(dst_ds[()])):
                        raise AssertionError(
                            f"{batch_key}/{ds_key}: non-label dataset changed during copy"
                        )

            dst_grp.attrs["taxonomy"] = SENTINEL

    assert sum(file_old.values()) == n_faces == sum(file_new.values())
    print(
        f"  H5 {os.path.basename(dst_path)}: faces={n_faces} "
        f"old_classes={len(file_old)} new_classes={len(file_new)}"
    )
    return "written"


def step_output_migrated(text: str) -> bool:
    return SENTINEL in text[:800]


def relabel_step_text(text: str) -> tuple[str, Counter, Counter, int]:
    n_faces = len(_FACE_ENTITY_RE.findall(text))
    old_hist: Counter = Counter()
    new_hist: Counter = Counter()

    def repl(m: re.Match) -> str:
        old = int(m.group(2))
        if not 0 <= old <= 24:
            raise ValueError(f"ADVANCED_FACE label {old} outside 0–24")
        new = old_to_new(old)
        old_hist[old] += 1
        new_hist[new] += 1
        return f"{m.group(1)}{new}{m.group(3)}"

    out = _FACE_NAME_RE.sub(repl, text)
    n_after = len(_FACE_ENTITY_RE.findall(out))
    if n_faces != n_after:
        raise AssertionError(f"ADVANCED_FACE count changed: {n_faces} → {n_after}")
    if sum(old_hist.values()) != n_faces:
        raise AssertionError(
            f"relabeled {sum(old_hist.values())} named faces, expected {n_faces}"
        )
    return out, old_hist, new_hist, n_faces


def _inject_step_sentinel(text: str) -> str:
    if step_output_migrated(text):
        return text
    marker = "ISO-10303-21;"
    if text.startswith(marker):
        rest = text[len(marker) :]
        if rest.startswith("\n"):
            rest = rest[1:]
        elif rest.startswith("\r\n"):
            rest = rest[2:]
        return f"{marker}\n{STEP_SENTINEL_LINE}\n{rest}"
    return f"{STEP_SENTINEL_LINE}\n{text}"


def _count_step_labels(text: str) -> tuple[Counter, int]:
    """Count ADVANCED_FACE name labels without remapping (for skip / verification)."""
    hist: Counter = Counter()
    for m in _FACE_NAME_RE.finditer(text):
        hist[int(m.group(2))] += 1
    return hist, sum(hist.values())


def relabel_step_file(src_path: str, dst_path: str) -> tuple[str, Counter, Counter, int]:
    """Return (status, old_hist, new_hist, n_faces). status ∈ {written, skip}."""
    with open(src_path, encoding="utf-8", errors="replace") as f:
        src_text = f.read()

    if SENTINEL in src_text[:800]:
        raise RuntimeError(f"STEP source {src_path} already has migration sentinel")

    if os.path.isfile(dst_path):
        with open(dst_path, encoding="utf-8", errors="replace") as f:
            dst_text = f.read()
        if step_output_migrated(dst_text):
            nh, nf = _count_step_labels(dst_text)
            return "skip", Counter(), nh, nf

    out, oh, nh, nf = relabel_step_text(src_text)
    out = _inject_step_sentinel(out)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(out)
    return "written", oh, nh, nf


def _relabel_step_worker(paths: tuple[str, str]) -> tuple[str, str, Counter, Counter, int]:
    src, dst = paths
    status, oh, nh, nf = relabel_step_file(src, dst)
    return os.path.basename(src), status, oh, nh, nf


def relabel_all_step(jobs: int) -> tuple[Counter, Counter, int, int, int]:
    tasks = []
    for split, sub in SPLIT_DIRS.items():
        src_dir = os.path.join(STEP_SRC_ROOT, sub)
        dst_dir = os.path.join(STEP_DST_ROOT, sub)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(src_dir)
        for name in sorted(os.listdir(src_dir)):
            if not name.endswith(".step"):
                continue
            tasks.append(
                (os.path.join(src_dir, name), os.path.join(dst_dir, name))
            )

    global_old: Counter = Counter()
    global_new: Counter = Counter()
    written = skipped = 0
    total_faces = 0

    print(f"STEP: {len(tasks)} files, jobs={jobs}", flush=True)
    if jobs <= 1:
        for i, paths in enumerate(tasks):
            base, status, oh, nh, nf = _relabel_step_worker(paths)
            global_old.update(oh)
            global_new.update(nh)
            total_faces += nf
            if status == "written":
                written += 1
            else:
                skipped += 1
            if (i + 1) % 5000 == 0:
                print(f"  … {i + 1}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_relabel_step_worker, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                base, status, oh, nh, nf = fut.result()
                global_old.update(oh)
                global_new.update(nh)
                total_faces += nf
                if status == "written":
                    written += 1
                else:
                    skipped += 1
                if i % 5000 == 0:
                    print(f"  … {i}/{len(tasks)}", flush=True)

    print(f"STEP done: written={written} skipped={skipped} total_faces={total_faces}")
    return global_old, global_new, written, skipped, total_faces


def h5_part_labels(h5_path: str, part_id: str) -> np.ndarray | None:
    with h5py.File(h5_path, "r") as f:
        for batch_key in f.keys():
            batch = f[batch_key]
            for i, raw_id in enumerate(batch["CAD_model"][()]):
                pid = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                if pid != part_id:
                    continue
                idx = np.asarray(batch["idx"])
                lab = np.asarray(batch["labels"]).reshape(-1)
                base = int(idx[0, 0])
                start = int(idx[i, 0]) - base
                end = (
                    int(idx[i + 1, 0]) - base if i + 1 < len(idx) else lab.shape[0]
                )
                return np.round(lab[start:end]).astype(np.int64)
    return None


def parse_step_face_labels(step_path: str) -> list[int]:
    """Same contract as llm_baseline.parse_faces (file order == H5 index order)."""
    with open(step_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    flat = re.sub(r"\s+", "", text)
    pairs = re.findall(r"#(\d+)=ADVANCED_FACE\('(\d+)'", flat)
    return [int(lbl) for _, lbl in pairs]


def cross_source_check(part_ids: list[str]) -> None:
    print("\n=== Cross-source consistency (H5 vs STEP, face-for-face) ===")
    for pid in part_ids:
        h5_path = os.path.join(H5_DST_DIR, "test_MFCAD++.h5")
        step_path = os.path.join(STEP_DST_ROOT, "test", f"{pid}.step")
        if not os.path.isfile(step_path):
            step_path = os.path.join(STEP_DST_ROOT, "train", f"{pid}.step")
        h5_lab = h5_part_labels(h5_path, pid)
        if h5_lab is None:
            for h5_name in SPLIT_H5.values():
                p = os.path.join(H5_DST_DIR, h5_name)
                if os.path.isfile(p):
                    h5_lab = h5_part_labels(p, pid)
                    if h5_lab is not None:
                        h5_path = p
                        break
        step_lab = np.asarray(parse_step_face_labels(step_path), dtype=np.int64)
        if h5_lab is None:
            print(f"  {pid}: H5 labels not found — skip")
            continue
        match = bool(np.array_equal(h5_lab, step_lab))
        print(
            f"  {pid}: faces={len(h5_lab)} match={match} "
            f"h5={h5_path} step={step_path}"
        )
        if not match:
            diff = np.where(h5_lab != step_lab)[0][:5]
            for j in diff:
                print(f"    face {j}: h5={h5_lab[j]} step={step_lab[j]}")


def spot_check_step_ingest(step_path: str) -> None:
    from brep.step_ingest import extract_brep_from_step

    model, _ = extract_brep_from_step(
        step_path, require_labels=True, require_12class=True
    )
    assert model is not None, f"step_ingest returned None for {step_path}"
    labels = model["labels"]
    assert labels.max() <= 11, labels.max()
    print(
        f"  step_ingest {os.path.basename(step_path)}: N={model['N']} "
        f"labels min={labels.min()} max={labels.max()}"
    )


def global_reconciliation(
    h5_old: Counter, h5_new: Counter, step_old: Counter, step_new: Counter
) -> None:
    folded = Counter()
    for old_id, cnt in h5_old.items():
        folded[old_to_new(old_id)] += cnt

    print("\n=== Global reconciliation ===")
    print(f"H5  old faces: {sum(h5_old.values())}  new faces: {sum(h5_new.values())}")
    print(f"STEP old faces: {sum(step_old.values())}  new faces: {sum(step_new.values())}")
    print(f"H5 old→new folded: {sum(folded.values())}  H5 new sum: {sum(h5_new.values())}")
    assert sum(folded.values()) == sum(h5_new.values()), "H5 fold mismatch"
    assert h5_new == step_new, f"H5 vs STEP new hist differ: {h5_new} vs {step_new}"

    print(f"\n{'new_id':>6}  {'name':<22}  {'faces':>10}")
    print("-" * 42)
    for new_id in range(NUM_CLASSES):
        print(f"{new_id:>6}  {new_name(new_id):<22}  {h5_new.get(new_id, 0):>10}")


def verify_outputs() -> None:
    print("\n=== Post-write verification ===")
    for h5_name in SPLIT_H5.values():
        path = os.path.join(H5_DST_DIR, h5_name)
        assert os.path.isfile(path), path
        assert h5_output_migrated(path), f"missing H5 sentinel: {path}"
        src = os.path.join(H5_SRC_DIR, h5_name)
        _assert_source_not_migrated_h5(src)
        with h5py.File(path, "r") as f:
            for bk in f.keys():
                lab = np.asarray(f[bk]["labels"][()])
                assert lab.dtype == np.float32, f"{path}/{bk} labels dtype {lab.dtype}"
                ints = np.round(lab).astype(np.int64)
                assert ints.min() >= 0 and ints.max() <= 11, f"{path}/{bk} label range"
                assert f[bk].attrs.get("taxonomy") == SENTINEL
                assert f[bk]["labels"].attrs.get("taxonomy") == SENTINEL
        print(f"  OK H5 {h5_name}: float32 labels, sentinel present, src untouched")

    sample = os.path.join(STEP_DST_ROOT, "test", "29000.step")
    assert os.path.isfile(sample)
    with open(sample) as f:
        head = f.read(400)
    assert SENTINEL in head
    src_sample = os.path.join(STEP_SRC_ROOT, "test", "29000.step")
    with open(src_sample) as f:
        assert SENTINEL not in f.read(400)
    print(f"  OK STEP sentinel on dst, absent on src ({sample})")


def show_step_advanced_faces(step_path: str, limit: int = 15) -> None:
    print(f"\n=== ADVANCED_FACE lines from {step_path} (first {limit}) ===")
    with open(step_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "ADVANCED_FACE" in line:
                print(line.rstrip())
                limit -= 1
                if limit <= 0:
                    break


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel MFCAD++ 25→12 (new copies only)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel STEP workers")
    ap.add_argument(
        "--sample-parts",
        default="29000,100,42",
        help="comma-separated part ids for cross-source check",
    )
    ap.add_argument(
        "--skip-step",
        action="store_true",
        help="H5 only (dev); default relabels both",
    )
    args = ap.parse_args()

    repo = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo)

    h5_global_old: Counter = Counter()
    h5_global_new: Counter = Counter()

    def _accumulate_h5_hist(path: str, counter: Counter) -> None:
        with h5py.File(path, "r") as f:
            for bk in f.keys():
                counter.update(
                    int(x)
                    for x in np.round(f[bk]["labels"][()]).astype(np.int64)
                )

    print("=== H5 relabel ===")
    os.makedirs(H5_DST_DIR, exist_ok=True)
    for h5_name in SPLIT_H5.values():
        src = os.path.join(H5_SRC_DIR, h5_name)
        dst = os.path.join(H5_DST_DIR, h5_name)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        status = relabel_h5_file(src, dst)
        print(f"  {h5_name}: {status}")
    for h5_name in SPLIT_H5.values():
        _accumulate_h5_hist(os.path.join(H5_SRC_DIR, h5_name), h5_global_old)
    for h5_name in SPLIT_H5.values():
        _accumulate_h5_hist(os.path.join(H5_DST_DIR, h5_name), h5_global_new)

    step_global_old: Counter = Counter()
    step_global_new: Counter = Counter()
    if not args.skip_step:
        so, sn, w, s, _ = relabel_all_step(args.jobs)
        step_global_old.update(so)
        step_global_new.update(sn)

    verify_outputs()
    global_reconciliation(h5_global_old, h5_global_new, step_global_old, step_global_new)

    sample_ids = [p.strip() for p in args.sample_parts.split(",") if p.strip()]
    cross_source_check(sample_ids)

    sample_step = os.path.join(STEP_DST_ROOT, "test", "29000.step")
    show_step_advanced_faces(sample_step)
    print("\n=== step_ingest spot-check (require_12class=True) ===")
    spot_check_step_ingest(sample_step)
    print("\n=== llm_baseline.parse_faces spot-check ===")
    pf = parse_step_face_labels(sample_step)
    print(f"  parse_faces: {len(pf)} faces, labels[:12]={pf[:12]}")

    print("\n=== Idempotency re-run ===")
    for h5_name in SPLIT_H5.values():
        dst = os.path.join(H5_DST_DIR, h5_name)
        print(f"  {h5_name}: {relabel_h5_file(os.path.join(H5_SRC_DIR, h5_name), dst)}")
    if not args.skip_step:
        _, _, w2, s2, _ = relabel_all_step(1)
        print(f"  STEP re-run: written={w2} skipped={s2} (expect written=0)")


if __name__ == "__main__":
    main()
