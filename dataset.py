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
import warnings
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from brep_features import (
    build_edge_features_regen,
    build_node_features_regen,
    make_undirected,
)


# ----------------------------------------------------------------------------
# Node / edge feature construction (shared by both loaders)
# ----------------------------------------------------------------------------
def pool_facet_features_to_faces(V_2, A_3, num_faces):
    """Aggregate per-facet mesh features up to their parent B-rep face -> [num_faces, 4].

    Inputs (all for a SINGLE part / consistent index space):
      V_2  : [M, 4]   per-facet mesh features. Cols 0-2 are the (min-max encoded)
                      facet normal, col 3 is the plane offset `d`.
      A_3  : [K, 2]   facet<->face incidence. col0 indexes rows of V_2 (a facet),
                      col1 is the parent face id in [0, num_faces).
      num_faces : int number of B-rep faces (rows in the output).

    Output columns: [mean_normal_x, mean_normal_y, mean_normal_z, mean_d].

    Audit notes (see diag/): the stored normals are NOT raw unit vectors — they are
    per-batch min-max encoded into [0,1]. We decode with `2v-1` and L2-normalize per
    facet so the mean operates on approximate directions, then re-normalize the mean
    (the average of unit vectors is not itself unit length). The originally-planned
    normal-variance/curvature channel was dropped: on this data it does not track
    curvature (flat faces show as much or more facet-normal spread than curved ones),
    so it would be a misleading feature rather than a flatness signal.

    Faces with zero incident facets get a zero vector (and a warning) instead of NaN.
    """
    V_2 = np.asarray(V_2, dtype=np.float32)
    if A_3 is None or len(A_3) == 0:
        warnings.warn(
            f"pool_facet_features_to_faces: no facet incidence for {num_faces} "
            "faces; returning zero mesh features")
        return np.zeros((num_faces, 4), dtype=np.float32)

    A_3 = np.asarray(A_3)
    facet_idx = A_3[:, 0].astype(np.int64)
    face_idx = A_3[:, 1].astype(np.int64)

    # decode [0,1] -> [-1,1] then L2-normalize each facet normal to a direction
    n = 2.0 * V_2[facet_idx, :3] - 1.0
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-6, None)
    d = V_2[facet_idx, 3].astype(np.float64)

    sum_n = np.zeros((num_faces, 3), dtype=np.float64)
    sum_d = np.zeros(num_faces, dtype=np.float64)
    counts = np.zeros(num_faces, dtype=np.int64)
    np.add.at(sum_n, face_idx, n)
    np.add.at(sum_d, face_idx, d)
    np.add.at(counts, face_idx, 1)

    empty = counts == 0
    if empty.any():
        warnings.warn(
            f"pool_facet_features_to_faces: {int(empty.sum())}/{num_faces} face(s) "
            "have zero facets; using a zero mesh-feature vector for them")

    safe = np.clip(counts, 1, None).astype(np.float64)
    mean_n = sum_n / safe[:, None]
    mean_d = sum_d / safe

    # re-normalize the mean normal: mean of unit vectors isn't unit length
    mean_n = mean_n / np.clip(np.linalg.norm(mean_n, axis=1, keepdims=True), 1e-6, None)

    out = np.zeros((num_faces, 4), dtype=np.float32)
    out[:, :3] = mean_n
    out[:, 3] = mean_d
    out[empty] = 0.0
    return out


def build_node_features(surface_type_ids, areas, num_surface_types, centroids=None,
                        mesh_feats=None):
    """One-hot surface type + log-norm area (+ optional centroid xyz, mesh feats) -> [N, D].

    D = num_surface_types + 1                (no extras)
    D = num_surface_types + 1 + 3            (with centroids)
    D = num_surface_types + 1 + 3 + 1        (with centroids + pooled plane-d)
    Centroids give the model a sense of *where* a face sits, which separates
    otherwise-identical faces (e.g. two cylinders of equal type/area). From the 4-dim
    pooled mesh block we keep only plane-d (mean plane offset along the normal): the
    mean-normal xyz dims were audited to be orientation noise (the V_2 normal columns
    are not decodable to real geometry), so feeding them would inject noise. plane-d
    adds position-along-normal that the B-rep level alone does not expose.
    """
    n = len(surface_type_ids)
    onehot = np.zeros((n, num_surface_types), dtype=np.float32)
    onehot[np.arange(n), np.clip(surface_type_ids, 0, num_surface_types - 1)] = 1.0
    area = np.log1p(np.asarray(areas, dtype=np.float32)).reshape(-1, 1)
    # robust per-part normalization so big parts don't dominate
    area = (area - area.mean()) / (area.std() + 1e-6)
    feats = [onehot, area]
    if centroids is not None:
        c = np.asarray(centroids, dtype=np.float32).reshape(n, -1)
        # center per part: absolute bbox position is arbitrary, relative layout isn't
        c = c - c.mean(axis=0, keepdims=True)
        feats.append(c)
    if mesh_feats is not None:
        mf = np.asarray(mesh_feats, dtype=np.float32).reshape(n, -1)
        # plane-d only (col 3 of the pooled block). The 3 mean-normal dims (cols 0-2)
        # are dropped: an audit (diag/) confirmed the V_2 normal columns are orientation
        # noise, not decodable geometry. plane-d is stored raw and is the geometrically
        # meaningful signed distance from origin to the face plane, so we keep just it.
        d = mf[:, 3:4]
        # plane-d is raw and huge-ranged (~+/-8000): signed-log then per-part standardize
        d = np.sign(d) * np.log1p(np.abs(d))
        d = (d - d.mean()) / (d.std() + 1e-6)
        feats.append(d)
    return np.concatenate(feats, axis=1)


def build_edge_features(convexity_ids, angles, lengths):
    """convexity one-hot(3) + angle + length -> [E, 5]."""
    e = len(convexity_ids)
    onehot = np.zeros((e, 3), dtype=np.float32)   # 0=concave,1=convex,2=smooth
    onehot[np.arange(e), np.clip(convexity_ids, 0, 2)] = 1.0
    ang = np.asarray(angles, dtype=np.float32).reshape(-1, 1) / np.pi  # ~[-1,1]
    ln = np.asarray(lengths, dtype=np.float32).reshape(-1, 1)
    ln = (ln - ln.mean()) / (ln.std() + 1e-6)
    return np.concatenate([onehot, ang, ln], axis=1)


SPLIT_H5 = {
    "train.txt": "training_MFCAD++.h5",
    "val.txt": "val_MFCAD++.h5",
    "test.txt": "test_MFCAD++.h5",
}


def _canonical_edge(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))


def _brep_bounds(idx_arr, model_idx, v1_len):
    """Per-model B-rep row range within a batched H5 group.

    idx[i, 0] is the global B-rep start for CAD_model[i]; idx[i, 1] is a mesh
    (V_2) bound — not the B-rep end. End for model i is idx[i+1, 0].
    """
    base = int(idx_arr[0, 0])
    start = int(idx_arr[model_idx, 0]) - base
    if model_idx + 1 < len(idx_arr):
        end = int(idx_arr[model_idx + 1, 0]) - base
    else:
        end = v1_len
    return start, end


def _edge_set(idx, start, end):
    mask = ((idx[:, 0] >= start) & (idx[:, 0] < end) &
            (idx[:, 1] >= start) & (idx[:, 1] < end))
    local = idx[mask] - start
    return {tuple(row) for row in local}


class _H5PickleMixin:
    """Drop open h5py handles before DataLoader worker pickling."""

    def _close_h5(self):
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            h5.close()
            self._h5 = None
        # drop any per-handle array cache so it isn't pickled to workers
        self._batch_cache = {}

    def __getstate__(self):
        self._close_h5()
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None


# ----------------------------------------------------------------------------
# (a) Official MFCAD++ batched hierarchical H5 (training/val/test *_MFCAD++.h5)
# ----------------------------------------------------------------------------
class MFCADPPGraphDataset(_H5PickleMixin, Dataset):
    """
    Reads the official MFCAD++ H5 splits (batched hierarchical B-Rep graphs).

    Each split file lists part ids; graphs live in hierarchical_graphs/<split>.h5
    under batch groups. We extract the B-Rep level (V_1, A_1, E_1/E_2/E_3, labels).
    """
    def __init__(self, data_root, h5_dir, split_file, num_surface_types):
        super().__init__()
        self.data_root = data_root
        self.num_surface_types = num_surface_types
        split_path = os.path.join(data_root, split_file)
        h5_name = SPLIT_H5.get(split_file)
        if h5_name is None:
            raise ValueError(f"unknown split file: {split_file}")
        self.h5_path = os.path.join(data_root, h5_dir, h5_name)
        _check_data_root(data_root, split_path, self.h5_path)

        with open(split_path) as f:
            requested = [line.strip() for line in f if line.strip()]
        self._index = self._build_index()
        self.ids = [pid for pid in requested if pid in self._index]
        missing = len(requested) - len(self.ids)
        if missing:
            print(f"warning: {missing} ids in {split_file} not found in {h5_name}")
        self._h5 = None

    def _build_index(self):
        index = {}
        with h5py.File(self.h5_path, "r") as f:
            for batch_key in f.keys():
                batch = f[batch_key]
                for i, raw_id in enumerate(batch["CAD_model"][()]):
                    pid = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    index[pid] = (batch_key, i)
        return index

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            # per-handle cache of full batch arrays needed for edge masking;
            # avoids re-reading entire batch datasets on every __getitem__.
            self._batch_cache = {}

    def _batch_arrays(self, batch_key):
        """Cache the small index arrays read in full (one entry per batch group)."""
        cached = self._batch_cache.get(batch_key)
        if cached is None:
            batch = self._h5[batch_key]
            cached = {
                "idx": batch["idx"][()],
                "A_1_idx": batch["A_1_idx"][()],
                "A_3_idx": batch["A_3_idx"][()],
                "E_1_idx": batch["E_1_idx"][()],
                "E_2_idx": batch["E_2_idx"][()],
                "E_3_idx": batch["E_3_idx"][()],
            }
            self._batch_cache[batch_key] = cached
        return cached

    def len(self):
        return len(self.ids)

    def _pool_mesh_feats(self, batch, a3_idx, start, end, num_faces):
        """Pool this model's facet (V_2) features up to its faces -> [num_faces, 4].

        A_3_idx col1 (parent face) uses the same 0-based, batch-local face indexing
        as `start:end`; col0 indexes V_2 rows directly. We select this model's
        incidence rows by face range, slice only the facets we need out of V_2, and
        rebase indices into the local [0, num_faces) / [0, num_facets) space.
        """
        mask = (a3_idx[:, 1] >= start) & (a3_idx[:, 1] < end)
        if not mask.any():
            return np.zeros((num_faces, 4), dtype=np.float32)
        rows = a3_idx[mask]
        facet_global = rows[:, 0]
        face_local = rows[:, 1] - start
        fmin = int(facet_global.min())
        fmax = int(facet_global.max())
        v2 = np.asarray(batch["V_2"][fmin:fmax + 1], dtype=np.float32)  # slice in HDF5
        a3_local = np.stack([facet_global - fmin, face_local], axis=1)
        return pool_facet_features_to_faces(v2, a3_local, num_faces)

    def _read_sample(self, part_id):
        self._ensure_open()
        batch_key, model_idx = self._index[part_id]
        batch = self._h5[batch_key]
        arrs = self._batch_arrays(batch_key)
        idx_arr = arrs["idx"]
        v1_len = batch["V_1"].shape[0]
        start, end = _brep_bounds(idx_arr, model_idx, v1_len)
        v1 = np.asarray(batch["V_1"][start:end], dtype=np.float32)  # slice in HDF5
        surface_type_ids = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1,
                                 0, self.num_surface_types - 1)
        areas = v1[:, 0]
        centroids = v1[:, 1:4]   # normalized centroid x/y/z (unused before)
        num_faces = end - start
        mesh_feats = self._pool_mesh_feats(batch, arrs["A_3_idx"], start, end, num_faces)
        x = build_node_features(surface_type_ids, areas, self.num_surface_types,
                                centroids=centroids, mesh_feats=mesh_feats)

        a1 = _edge_set(arrs["A_1_idx"], start, end)
        e1 = {_canonical_edge(u, v) for u, v in _edge_set(arrs["E_1_idx"], start, end)}
        e2 = {_canonical_edge(u, v) for u, v in _edge_set(arrs["E_2_idx"], start, end)}
        e3 = {_canonical_edge(u, v) for u, v in _edge_set(arrs["E_3_idx"], start, end)}

        edges = sorted(a1)
        if edges:
            edge_index = np.asarray(edges, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        convexity, angles, lengths = [], [], []
        for u, v in edges:
            key = _canonical_edge(u, v)
            if key in e1:
                convexity.append(1)
            elif key in e2:
                convexity.append(0)
            elif key in e3:
                convexity.append(2)
            else:
                convexity.append(1)
            # MFCAD++ B-rep level stores no dihedral angle / edge length: every
            # *_values array in the H5 is constant 1.0 (pure adjacency indicator),
            # so convexity bucket is the only real edge signal. These stay constant.
            angles.append(0.0)
            lengths.append(1.0)
        ea = build_edge_features(
            np.asarray(convexity, dtype=np.int64),
            np.asarray(angles, dtype=np.float32),
            np.asarray(lengths, dtype=np.float32),
        )
        y = np.asarray(batch["labels"][start:end], dtype=np.int64)
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(ea),
            y=torch.from_numpy(y),
        )

    def get(self, idx):
        return self._read_sample(self.ids[idx])


# ----------------------------------------------------------------------------
# (a') Regenerated MFCAD++ B-rep graphs (hierarchical_graphs_regen/*.h5)
# ----------------------------------------------------------------------------
class MFCADPPRegenGraphDataset(_H5PickleMixin, Dataset):
    """
    Reads the REGENERATED MFCAD++ splits (diag/regen_dataset.py output), which mirror
    the released batched-hierarchical B-rep schema so this is a drop-in sibling of
    MFCADPPGraphDataset, with two correctness upgrades baked into the files:

      * A_1_values  : true per-edge dihedral angle in radians (released = const 1.0)
      * V_1 cols 5-8: exact per-face unit normal (nx,ny,nz) + plane-d (replaces the
                      old, unreliable V_2 mesh-pooling path entirely)

    Edge construction + dedup (built before any edge feature construction):
      One B-rep face pair can meet along several topological edges, so A_1 can hold
      several rows for the same pair, occasionally spanning >1 convexity bucket. Each
      pair is collapsed to ONE undirected graph edge with:
        - convexity by CONCAVE-FIRST precedence (concave > convex > smooth): the
          concave dihedral is the signal that separates rectangular- from
          slanted-step faces, so we keep the concave edge whenever one exists,
          regardless of angle magnitude (NOT an angle tiebreak). Full-scale audit
          (diag/dup_edge_scan.py + dup_edge_attribute.py): 39 differing-convexity
          pairs in 37/59,665 models, and in every one the convex/concave edges share
          the same angle, so "smaller angle" and "concave" never disagree here; the
          rule is still the unambiguous concave-first one.
        - angle = median of that pair's per-edge dihedral angles. Median (not mean)
          is robust to the ~2,274 same-bucket multi-edge pairs whose per-edge angles
          genuinely vary (curved faces meeting along several arcs, spread up to
          ~148 deg); for every differing-convexity pair the angles tie so the median
          equals the concave edge's angle. Stored as cos(angle).
    """
    def __init__(self, data_root, h5_dir, split_file, num_surface_types,
                 angle_reduce="median"):
        super().__init__()
        self.data_root = data_root
        self.num_surface_types = num_surface_types
        self.angle_reduce = angle_reduce
        split_path = os.path.join(data_root, split_file)
        h5_name = SPLIT_H5.get(split_file)
        if h5_name is None:
            raise ValueError(f"unknown split file: {split_file}")
        self.h5_path = os.path.join(data_root, h5_dir, h5_name)
        _check_data_root(data_root, split_path, self.h5_path)

        with open(split_path) as f:
            requested = [line.strip() for line in f if line.strip()]
        self._index = self._build_index()
        self.ids = [pid for pid in requested if pid in self._index]
        missing = len(requested) - len(self.ids)
        if missing:
            print(f"warning: {missing} ids in {split_file} not found in {h5_name}")
        self._h5 = None

    def _build_index(self):
        index = {}
        with h5py.File(self.h5_path, "r") as f:
            for batch_key in f.keys():
                batch = f[batch_key]
                for i, raw_id in enumerate(batch["CAD_model"][()]):
                    pid = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    index[pid] = (batch_key, i)
        return index

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._batch_cache = {}

    def _batch_arrays(self, batch_key):
        cached = self._batch_cache.get(batch_key)
        if cached is None:
            batch = self._h5[batch_key]
            cached = {
                "idx": batch["idx"][()],
                "A_1_idx": batch["A_1_idx"][()],
                "A_1_values": batch["A_1_values"][()],
                "E_1_idx": batch["E_1_idx"][()],
                "E_2_idx": batch["E_2_idx"][()],
                "E_3_idx": batch["E_3_idx"][()],
            }
            self._batch_cache[batch_key] = cached
        return cached

    def len(self):
        return len(self.ids)

    def _read_sample(self, part_id):
        self._ensure_open()
        batch_key, model_idx = self._index[part_id]
        batch = self._h5[batch_key]
        arrs = self._batch_arrays(batch_key)
        idx_arr = arrs["idx"]
        v1_len = batch["V_1"].shape[0]
        start, end = _brep_bounds(idx_arr, model_idx, v1_len)
        v1 = np.asarray(batch["V_1"][start:end], dtype=np.float32)  # slice in HDF5
        x = build_node_features_regen(v1, self.num_surface_types)

        # convexity buckets as canonical, self-loop-free sets
        e1 = {_canonical_edge(u, v) for u, v in _edge_set(arrs["E_1_idx"], start, end)
              if u != v}
        e2 = {_canonical_edge(u, v) for u, v in _edge_set(arrs["E_2_idx"], start, end)
              if u != v}
        e3 = {_canonical_edge(u, v) for u, v in _edge_set(arrs["E_3_idx"], start, end)
              if u != v}

        # group A_1 rows (both directions) by canonical pair -> list of angles (rad)
        a1 = arrs["A_1_idx"]
        av = arrs["A_1_values"]
        mask = ((a1[:, 0] >= start) & (a1[:, 0] < end) &
                (a1[:, 1] >= start) & (a1[:, 1] < end))
        rows = a1[mask] - start
        vals = av[mask]
        ang_by_pair = {}
        for (u, v), ang in zip(rows.tolist(), vals.tolist()):
            if u == v:
                continue
            key = _canonical_edge(u, v)
            ang_by_pair.setdefault(key, []).append(float(ang))

        reduce_fn = np.mean if self.angle_reduce == "mean" else np.median
        pairs, convexity, cos_ang = [], [], []
        for key, angs in ang_by_pair.items():
            if key in e2:          # concave-first precedence (keep the concave edge)
                cid = 0
            elif key in e1:
                cid = 1
            elif key in e3:
                cid = 2
            else:
                cid = 1            # no bucket (unexpected): default convex
            pairs.append(key)
            convexity.append(cid)
            cos_ang.append(np.cos(float(reduce_fn(angs))))

        if pairs:
            edge_index = np.asarray(pairs, dtype=np.int64).T
            ea = build_edge_features_regen(
                np.asarray(convexity, dtype=np.int64),
                np.asarray(cos_ang, dtype=np.float32))
            edge_index, ea = make_undirected(edge_index, ea)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            ea = np.zeros((0, 4), dtype=np.float32)

        y = np.asarray(batch["labels"][start:end], dtype=np.int64)
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_attr=torch.from_numpy(ea),
            y=torch.from_numpy(y),
        )

    def get(self, idx):
        return self._read_sample(self.ids[idx])


def _check_data_root(data_root, split_path, h5_path):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"data_root not found: {os.path.abspath(data_root)!r}\n"
            "Unzip MFCAD++ into MFCAD++_dataset/ (see README) or edit "
            "config.yaml:data_root."
        )
    if not os.path.isfile(split_path):
        setup_hint = (
            "Run: python setup_data.py"
            if not os.listdir(data_root)
            else f"Expected {os.path.basename(split_path)} inside data_root."
        )
        raise FileNotFoundError(
            f"split file not found: {os.path.abspath(split_path)!r}\n"
            f"{setup_hint}"
        )
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"H5 file not found: {os.path.abspath(h5_path)!r}\n"
            "Check config.yaml:h5_dir and that hierarchical_graphs/ is present."
        )


# ----------------------------------------------------------------------------
# (b) Simple single-H5 layout (one group per part id)
# ----------------------------------------------------------------------------
class H5GraphDataset(_H5PickleMixin, Dataset):
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
        _check_data_root(data_root, split_path, self.h5_path)
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


class CachedGraphDataset(Dataset):
    """Holds pre-built Data objects in memory; getitem is a list lookup.

    The expensive per-graph Python processing in the H5 loaders runs ONCE at
    build time (see build_cache); every epoch after that reads finished tensors,
    so the loader is no longer the bottleneck. Pickles cleanly to workers (no
    open h5py handles), so num_workers>0 is fine.
    """

    def __init__(self, data_list):
        super().__init__()
        self._data = data_list

    def len(self):
        return len(self._data)

    def get(self, idx):
        return self._data[idx]

    def _close_h5(self):  # no-op: train.py calls this on every dataset
        pass


def _build_uncached(cfg, split_file):
    if cfg["loader"] != "h5":
        return StepGraphDataset()
    fmt = cfg.get("h5_format", "mfcadpp")
    if fmt == "mfcadpp_regen":
        return MFCADPPRegenGraphDataset(
            cfg["data_root"], cfg.get("h5_dir", "hierarchical_graphs_regen"),
            split_file, cfg["num_surface_types"],
            angle_reduce=cfg.get("angle_reduce", "median"))
    if fmt == "mfcadpp":
        return MFCADPPGraphDataset(
            cfg["data_root"], cfg.get("h5_dir", "hierarchical_graphs"),
            split_file, cfg["num_surface_types"])
    return H5GraphDataset(cfg["data_root"], cfg["h5_path"], split_file,
                          cfg["num_surface_types"])


def _cache_path(cfg, split_file):
    fmt = cfg.get("h5_format", "mfcadpp")
    reduce_tag = cfg.get("angle_reduce", "median")
    cache_dir = os.path.join(cfg["data_root"], "graph_cache")
    base = os.path.splitext(os.path.basename(split_file))[0]
    return os.path.join(cache_dir, f"{base}.{fmt}.{reduce_tag}.pt")


def build_cache(cfg, split_file, verbose=True):
    """Materialize one split's graphs to a .pt cache. Safe to call repeatedly."""
    path = _cache_path(cfg, split_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = _build_uncached(cfg, split_file)
    n = len(ds)
    data_list = []
    for i in range(n):
        data_list.append(ds[i])
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {split_file}: {i + 1}/{n}", flush=True)
    if hasattr(ds, "_close_h5"):
        ds._close_h5()
    torch.save(data_list, path)
    if verbose:
        print(f"  wrote {path} ({n} graphs)", flush=True)
    return path


def get_dataset(cfg, split_file):
    """Return a dataset for one split, using the on-disk graph cache if enabled.

    Set cfg['use_cache']=True (default) to read pre-built graphs; the cache is
    built on first use. Delete MFCAD++_dataset/graph_cache/ to force a rebuild
    after changing any feature/edge logic.
    """
    if cfg.get("use_cache", True) and cfg["loader"] == "h5":
        path = _cache_path(cfg, split_file)
        if not os.path.isfile(path):
            print(f"[cache] building {path} (one-time)...", flush=True)
            build_cache(cfg, split_file)
        data_list = torch.load(path, weights_only=False)
        return CachedGraphDataset(data_list)
    return _build_uncached(cfg, split_file)


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
