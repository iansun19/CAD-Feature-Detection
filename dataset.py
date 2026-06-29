"""
dataset.py — turn MFCAD++ parts into PyG graphs.

MFCAD++ provides BOTH:
  (a) a prebuilt hierarchical B-rep graph in HDF5  -> H5GraphDataset  (preferred)
  (b) raw STEP files + per-face labels             -> StepGraphDataset (fallback)

You almost certainly want (a): no pythonocc dependency, no STEP parsing, and the
graph + labels are already aligned. Use `--inspect-h5` first to confirm field names,
because the exact keys inside the H5 are the single most error-prone part of this.

A PyG `Data` object here carries:
  x          : [num_faces, node_feat_dim]  float
  edge_index : [2, num_edges]              long  (undirected -> add both directions)
  edge_attr  : [num_edges, edge_dim]       float
  y          : [num_faces]                 long  (per-face class label)
"""

import argparse
import os
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# ----------------------------------------------------------------------------
# Node / edge feature construction (shared by both loaders)
# ----------------------------------------------------------------------------
def build_node_features(surface_type_ids, areas, num_surface_types):
    """One-hot surface type + log-normalized area -> [N, num_surface_types+1]."""
    n = len(surface_type_ids)
    onehot = np.zeros((n, num_surface_types), dtype=np.float32)
    onehot[np.arange(n), np.clip(surface_type_ids, 0, num_surface_types - 1)] = 1.0
    area = np.log1p(np.asarray(areas, dtype=np.float32)).reshape(-1, 1)
    # robust per-part normalization so big parts don't dominate
    area = (area - area.mean()) / (area.std() + 1e-6)
    return np.concatenate([onehot, area], axis=1)


def build_edge_features(convexity_ids, angles, lengths):
    """convexity one-hot(3) + angle + length -> [E, 5]."""
    e = len(convexity_ids)
    onehot = np.zeros((e, 3), dtype=np.float32)   # 0=concave,1=convex,2=smooth
    onehot[np.arange(e), np.clip(convexity_ids, 0, 2)] = 1.0
    ang = np.asarray(angles, dtype=np.float32).reshape(-1, 1) / np.pi  # ~[-1,1]
    ln = np.asarray(lengths, dtype=np.float32).reshape(-1, 1)
    ln = (ln - ln.mean()) / (ln.std() + 1e-6)
    return np.concatenate([onehot, ang, ln], axis=1)


def make_undirected(edge_index, edge_attr):
    """Duplicate edges in both directions for message passing."""
    ei = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    ea = np.concatenate([edge_attr, edge_attr], axis=0)
    return ei, ea


# ----------------------------------------------------------------------------
# (a) Preferred: prebuilt H5 graphs
# ----------------------------------------------------------------------------
class H5GraphDataset(Dataset):
    """
    Reads samples listed in split file (train.txt/val.txt/test.txt) from a single H5.

    >>> EXPECTED H5 LAYOUT (verify with --inspect-h5; rename keys below to match) <<<
    Each sample is a group keyed by the part id, containing datasets:
        'face_types'   [N]      int   surface type id per face
        'face_areas'   [N]      float area per face
        'edges'        [2, E]   int   face-index pairs sharing an edge
        'edge_convex'  [E]      int   0/1/2 concave/convex/smooth
        'edge_angle'   [E]      float dihedral angle (radians)
        'edge_length'  [E]      float
        'labels'       [N]      int   per-face feature class  <-- ground truth
    If your H5 nests these differently, adjust _read_sample only.
    """
    def __init__(self, data_root, h5_path, split_file, num_surface_types):
        super().__init__()
        self.h5_path = os.path.join(data_root, h5_path)
        self.num_surface_types = num_surface_types
        split_path = os.path.join(data_root, split_file)
        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"data_root not found: {os.path.abspath(data_root)!r}\n"
                "Unzip MFCAD++ into MFCAD_dataset/ (see README) or edit "
                "config.yaml:data_root to your dataset folder."
            )
        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"split file not found: {os.path.abspath(split_path)!r}\n"
                f"Expected {split_file} inside data_root."
            )
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(
                f"H5 file not found: {os.path.abspath(self.h5_path)!r}\n"
                "Check config.yaml:h5_path matches your downloaded file name."
            )
        with open(split_path) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self._h5 = None  # opened lazily per worker

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def len(self):
        return len(self.ids)

    def _read_sample(self, key):
        self._ensure_open()
        g = self._h5[key]
        x = build_node_features(g["face_types"][()], g["face_areas"][()],
                                self.num_surface_types)
        ei = np.asarray(g["edges"][()])
        ea = build_edge_features(g["edge_convex"][()], g["edge_angle"][()],
                                 g["edge_length"][()])
        ei, ea = make_undirected(ei, ea)
        y = np.asarray(g["labels"][()], dtype=np.int64)
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(ei).long(),
            edge_attr=torch.from_numpy(ea),
            y=torch.from_numpy(y),
        )

    def get(self, idx):
        return self._read_sample(self.ids[idx])


# ----------------------------------------------------------------------------
# (b) Fallback: parse STEP with pythonocc (only if you have no usable H5)
# ----------------------------------------------------------------------------
class StepGraphDataset(Dataset):
    """
    Stub showing where pythonocc parsing goes. Filling this in means: load each STEP,
    enumerate faces (-> nodes, surface type, area), enumerate shared edges (-> graph
    edges, convexity via face-normal test, dihedral angle), then read the matching
    per-face label from feature_labels.txt. This is real work — prefer the H5 path.
    Left intentionally minimal; ask if you end up needing it.
    """
    def __init__(self, *a, **k):
        raise NotImplementedError(
            "Use H5GraphDataset. Fill StepGraphDataset only if the H5 is unusable."
        )


def get_dataset(cfg, split_file):
    if cfg["loader"] == "h5":
        return H5GraphDataset(cfg["data_root"], cfg["h5_path"], split_file,
                              cfg["num_surface_types"])
    return StepGraphDataset()


# ----------------------------------------------------------------------------
# H5 inspector — RUN THIS FIRST to learn the real field names
# ----------------------------------------------------------------------------
def inspect_h5(path, max_depth=3):
    def walk(name, obj, depth=0):
        pad = "  " * depth
        if isinstance(obj, h5py.Group):
            print(f"{pad}{name}/  (group, {len(obj)} items)")
            if depth < max_depth:
                for k in list(obj.keys())[:8]:
                    walk(k, obj[k], depth + 1)
        else:
            print(f"{pad}{name}  shape={obj.shape} dtype={obj.dtype}")
    with h5py.File(path, "r") as f:
        print(f"Top-level keys ({len(f)}): {list(f.keys())[:10]} ...")
        for k in list(f.keys())[:3]:
            walk(k, f[k])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-h5", type=str, help="path to an MFCAD++ .h5 to inspect")
    args = ap.parse_args()
    if args.inspect_h5:
        inspect_h5(args.inspect_h5)
