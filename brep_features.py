"""NumPy-only B-rep graph feature builders (shared by dataset.py and step_ingest.py)."""
import numpy as np


def build_node_features_regen(v1, num_surface_types):
    """14-dim node features from a regenerated V_1 block [N, 9]."""
    v1 = np.asarray(v1, dtype=np.float32)
    n = v1.shape[0]
    type_ids = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1,
                       0, num_surface_types - 1)
    onehot = np.zeros((n, num_surface_types), dtype=np.float32)
    onehot[np.arange(n), type_ids] = 1.0
    area = v1[:, 0:1].copy()
    area = (area - area.mean()) / (area.std() + 1e-6)
    cent = v1[:, 1:4].copy()
    cent = cent - cent.mean(axis=0, keepdims=True)
    normal = v1[:, 5:8].copy()
    d = v1[:, 8:9].copy()
    d = np.sign(d) * np.log1p(np.abs(d))
    d = (d - d.mean()) / (d.std() + 1e-6)
    return np.concatenate([onehot, area, cent, normal, d], axis=1)


def build_edge_features_regen(convexity_ids, cos_angles):
    """convexity one-hot(3) + cos(dihedral)(1) -> [E, 4]."""
    e = len(convexity_ids)
    onehot = np.zeros((e, 3), dtype=np.float32)
    onehot[np.arange(e), np.clip(convexity_ids, 0, 2)] = 1.0
    cosv = np.asarray(cos_angles, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([onehot, cosv], axis=1)


def make_undirected(edge_index, edge_attr):
    """Duplicate edges in both directions for message passing."""
    ei = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    ea = np.concatenate([edge_attr, edge_attr], axis=0)
    return ei, ea
