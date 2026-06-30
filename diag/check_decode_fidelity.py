"""
check_decode_fidelity.py — is V_2's normal encoding actually recoverable by 2v-1?

If V_2[:, :3] were a clean per-column min-max of unit normals, then 2v-1 would be
the exact original unit vector (norm == 1) and angles between faces would be exact.
We test that directly:

  * norm of (2v-1) per facet BEFORE renormalization -> should be ~1 if recoverable.
  * for facets of a single PLANAR face (constant true normal), do the decoded facet
    normals actually point the same way (pairwise cos ~1)?
  * raw value distribution: a min-max of axis-aligned normals would be mostly {0,.5,1};
    a smooth spread instead means the stored encoding mixes facets/coordinates.

Usage:
    python diag/check_decode_fidelity.py [h5_path] [n_batches]
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
np.set_printoptions(precision=3, suppress=True, linewidth=120)

norms_2vm1 = []
raw_vals = []
with h5py.File(path, "r") as f:
    for k in list(f.keys())[:N_BATCHES]:
        v2 = np.asarray(f[k]["V_2"])[:, :3].astype(np.float64)
        dec = 2.0 * v2 - 1.0
        norms_2vm1.append(np.linalg.norm(dec, axis=1))
        if len(raw_vals) < 200000:
            raw_vals.append(v2.reshape(-1))

norms = np.concatenate(norms_2vm1)
raw = np.concatenate(raw_vals)

print(f"facets tested: {norms.size}")
print("\n=== ||2v-1|| (should be ~1.0 if encoding is recoverable unit normals) ===")
print(f"  mean={norms.mean():.4f} median={np.median(norms):.4f} std={norms.std():.4f}")
print(f"  frac in [0.97,1.03] = {np.mean(np.abs(norms-1)<0.03):.3f}")
for lo, hi in [(0, .5), (.5, .9), (.9, 1.1), (1.1, 1.5), (1.5, 3)]:
    print(f"    ||.|| in [{lo},{hi}) : {np.mean((norms>=lo)&(norms<hi)):.3f}")

print("\n=== raw V_2 normal value distribution (share per bin) ===")
for lo, hi in [(0, .05), (.05, .45), (.45, .55), (.55, .95), (.95, 1.001)]:
    print(f"    raw in [{lo},{hi}) : {np.mean((raw>=lo)&(raw<hi)):.3f}")
print("  (clean min-max of axis-aligned normals would concentrate near 0/.5/1;")
print("   heavy mass in the in-between bins => not a simple recoverable encoding)")
