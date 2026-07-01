"""
Run this on the CUDA box from the repo root:
    python count_class_frequencies.py

Counts per-class face frequency across train/val/test splits of the
regenerated MFCAD++ H5 data, and also counts per-PART frequency
(how many distinct parts contain at least one face of that class) —
the second number matters more for "is this class even represented
across the dataset" vs. "is it just one giant part with 9000 faces
of one class".
"""
import glob
import os
from collections import Counter

import h5py

from taxonomy import NEW_NAMES, NUM_CLASSES

CLASS_NAMES = NEW_NAMES

# Adjust this glob if your directory layout differs
H5_ROOT = "MFCAD++_dataset/hierarchical_graphs_regen"


def find_h5_files(root):
    files = glob.glob(os.path.join(root, "**", "*.h5"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(root, "**", "*.hdf5"), recursive=True)
    return sorted(files)


def _brep_bounds(idx_arr, model_idx, v1_len):
    """Per-model B-rep row range within a batched MFCAD++ H5 group."""
    base = int(idx_arr[0, 0])
    start = int(idx_arr[model_idx, 0]) - base
    if model_idx + 1 < len(idx_arr):
        end = int(idx_arr[model_idx + 1, 0]) - base
    else:
        end = v1_len
    return start, end


def _count_file(f):
    """Return (face_counts, part_presence_counts, total_parts, total_faces)."""
    face_counts = Counter()
    part_presence_counts = Counter()
    total_parts = 0
    total_faces = 0

    keys = list(f.keys())
    if "labels" in f:
        labels = f["labels"][:]
        total_faces += len(labels)
        total_parts += 1
        c = Counter(int(x) for x in labels.tolist())
        face_counts.update(c)
        for cls in c:
            part_presence_counts[cls] += 1
        return face_counts, part_presence_counts, total_parts, total_faces

    for batch_key in keys:
        grp = f[batch_key]
        if "labels" not in grp:
            continue

        labels = grp["labels"][()]
        if "idx" in grp:
            idx_arr = grp["idx"][()]
            v1_len = grp["V_1"].shape[0] if "V_1" in grp else len(labels)
            for model_idx in range(len(idx_arr)):
                start, end = _brep_bounds(idx_arr, model_idx, v1_len)
                part_labels = labels[start:end]
                if len(part_labels) == 0:
                    continue
                total_faces += len(part_labels)
                total_parts += 1
                c = Counter(int(x) for x in part_labels.tolist())
                face_counts.update(c)
                for cls in c:
                    part_presence_counts[cls] += 1
        else:
            total_faces += len(labels)
            total_parts += 1
            c = Counter(int(x) for x in labels.tolist())
            face_counts.update(c)
            for cls in c:
                part_presence_counts[cls] += 1

    return face_counts, part_presence_counts, total_parts, total_faces


def main():
    files = find_h5_files(H5_ROOT)
    if not files:
        print(f"No H5 files found under {H5_ROOT} -- edit H5_ROOT at top of script.")
        return

    print(f"Found {len(files)} H5 file(s)\n")

    face_counts = Counter()
    part_presence_counts = Counter()
    total_parts = 0
    total_faces = 0

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            fc, pc, tp, tf = _count_file(f)
            face_counts.update(fc)
            part_presence_counts.update(pc)
            total_parts += tp
            total_faces += tf

    print(f"Total parts scanned: {total_parts}")
    print(f"Total faces scanned: {total_faces}\n")
    print(f"{'ID':<4}{'Name':<32}{'Face count':<14}{'Face %':<10}{'Parts w/ class':<16}{'Part %':<8}")
    print("-" * 84)
    for cls_id in sorted(CLASS_NAMES):
        fc = face_counts.get(cls_id, 0)
        pc = part_presence_counts.get(cls_id, 0)
        fpct = 100 * fc / total_faces if total_faces else 0
        ppct = 100 * pc / total_parts if total_parts else 0
        print(f"{cls_id:<4}{CLASS_NAMES[cls_id]:<32}{fc:<14}{fpct:<10.3f}{pc:<16}{ppct:<8.2f}")

    print("\nSorted by rarity (face count, ascending):")
    for cls_id, fc in sorted(face_counts.items(), key=lambda x: x[1]):
        print(f"  {cls_id:>2} {CLASS_NAMES[cls_id]:<32} {fc}")


if __name__ == "__main__":
    main()
