# MFCAD++ Path 1 — B-Rep GNN face classifier

A minimal, well-commented PyTorch Geometric pipeline for per-face machining-feature
classification on the MFCAD++ dataset. Designed to be readable and to run overnight
on a single GPU.

## What this is
- **Path 1**: a single face-adjacency graph GNN (UV-Net-style, simplified).
- Node = B-rep face, edge = shared edge between two faces.
- Output = one feature-class label per face.

## Files
- `requirements.txt`   — exact install commands (PyTorch + PyG).
- `dataset.py`         — loads MFCAD++ into PyG `Data` graphs. **Has TWO loaders:**
                         one for the prebuilt H5 graphs, one that parses STEP via
                         pythonocc. Read the comments — you must pick/verify which
                         matches your downloaded files.
- `model.py`           — the GNN (GINEConv stack with edge features + residuals).
- `train.py`           — training loop with early stopping, checkpointing, logging.
- `overfit_check.py`   — fast sanity run on ~20 samples. RUN THIS FIRST.
- `config.yaml`        — all hyperparameters in one place.

## Order of operations (do not skip)
1. Install env (`requirements.txt`).
2. Unzip MFCAD++ into `MFCAD_dataset/` at the repo root (or set `config.yaml:data_root`).
3. Run `python overfit_check.py` — must reach ~100% train acc on 20 parts in a couple
   minutes. If it can't memorize 20 parts, your data loader is wrong. Fix before step 4.
4. Run `python train.py` overnight.
5. Read `runs/<timestamp>/log.txt` and `best_model.pt` in the morning.

### Mac / Apple Silicon
1. Use the **minimal Mac install** in `requirements.txt` (`torch` + `torch_geometric` only —
   no `torch_scatter` / `torch_sparse` wheels; they often fail to build on Mac and aren't
   required for this pipeline).
2. Device auto-selects **MPS** when available (`config.yaml: device: auto`). The overfit
   check prints `device=mps` at startup if it's working.
3. Run `overfit_check.py` first — even on MPS it finishes in a few minutes on 20 parts.
4. If overfit passes, either let the full run grind overnight on your Mac, or spin up a
   cloud GPU for a faster clean run. Set `device: cpu` in config if you hit a rare MPS op gap.

## The one thing that will bite you
The data loader. MFCAD++ ships BOTH STEP files AND a prebuilt hierarchical H5.
Inspect `h5_structure.h5` first (`python dataset.py --inspect-h5 path/to/h5`) and
confirm the field names match what `H5GraphDataset` expects. Field naming is the #1
cause of silent label-misalignment. The overfit check is your tripwire for this.
