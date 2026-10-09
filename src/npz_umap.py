#!/usr/bin/env python3
"""Perform UMAP dimensionality reduction on one or more tensor representations (comma-separated) and export embeddings with residue ids."""
import argparse
import os
import resource
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
import umap
from utils.yamlmanager import ConfigManager

###################################### constants
SEED = 42
N_COMPONENTS = 2
N_NEIGHBORS = 15
MIN_DIST = 0.1
METRIC = "euclidean"
N_EPOCHS = None
LEARNING_RATE = 1.0
INIT = "spectral"
OUTPUT_DIR = "data/umap"

###################################### cli
parser = argparse.ArgumentParser(description="UMAP on tensor representations (comma-separated inputs supported)")
parser.add_argument("--tensor", type=str, required=True, help="Path(s) to input .pt or .npz representations, comma-separated")
parser.add_argument("--output-tsv", type=str, default=None, help="Output TSV path (default: derived from first input)")
parser.add_argument("--output-npz", type=str, default=None, help="Output NPZ path for embedding+ids (default: derived)")
parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
parser.add_argument("--min-dist", type=float, default=MIN_DIST)
parser.add_argument("--metric", type=str, default=METRIC)
parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
parser.add_argument("--init", type=str, default=INIT)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--n-jobs", type=int, default=-1, help="-1 uses all cores")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
args = parser.parse_args()

CONSTANTS = dict(SEED=SEED, N_COMPONENTS=N_COMPONENTS, N_NEIGHBORS=N_NEIGHBORS, MIN_DIST=MIN_DIST, METRIC=METRIC, N_EPOCHS=N_EPOCHS, LEARNING_RATE=LEARNING_RATE, INIT=INIT, OUTPUT_DIR=OUTPUT_DIR)
config = {**CONSTANTS, **vars(args)}

###################################### setup
t0 = time.perf_counter()
n_cores = cpu_count() if config["n_jobs"] == -1 else config["n_jobs"]
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
sklearn.set_config(working_memory=1024)
print(f"[PYTHON-INFO] Detected {n_cores} CPU cores | seed={config['seed']}")

###################################### load & merge
paths = [Path(x.strip()) for x in config["tensor"].split(",") if x.strip()]
print(f"[PYTHON-INFO] Loading {len(paths)} file(s)...")
tensors, ids_list = [], []
for p in paths:
    if p.suffix == ".pt":
        data = torch.load(p, map_location="cpu", weights_only=False)
        t = data["tensor"].numpy() if torch.is_tensor(data["tensor"]) else np.asarray(data["tensor"])
        r = data["residue_ids"]
    elif p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        t = data["embeddings"]
        r = data["ids"]
    else:
        raise ValueError(f"Unsupported input: {p.suffix}")
    tensors.append(np.asarray(t))
    ids_list.append(np.asarray(r))
    print(f"[PYTHON-INFO]   {p.name}: shape={t.shape} residues={len(r)}")
tensor = np.concatenate(tensors, axis=0)
residue_ids = np.concatenate(ids_list, axis=0)
print(f"[PYTHON-INFO] Merged tensor shape: {tensor.shape} | total residues: {len(residue_ids)}")

###################################### umap
print(f"[PYTHON-INFO] Fitting UMAP n_components={config['n_components']} n_neighbors={config['n_neighbors']} min_dist={config['min_dist']}...")
reducer = umap.UMAP(n_components=config["n_components"], n_neighbors=config["n_neighbors"], min_dist=config["min_dist"], metric=config["metric"], n_epochs=config["n_epochs"], learning_rate=config["learning_rate"], init=config["init"], random_state=config["seed"], n_jobs=n_cores)
embedding = reducer.fit_transform(tensor)
print(f"[PYTHON-INFO] Embedding shape: {embedding.shape}")

###################################### output paths
stem = paths[0].stem
if stem.endswith("_reps"):
    stem = stem[:-5]
out_dir = Path(config["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
tsv_path = Path(config["output_tsv"]) if config["output_tsv"] else out_dir / f"{stem}_umap.tsv"
npz_path = Path(config["output_npz"]) if config["output_npz"] else out_dir / f"{stem}_umap.npz"

###################################### save
cols = {f"UMAP{i+1}": embedding[:, i] for i in range(embedding.shape[1])}
cols["residue_ids"] = residue_ids
df = pd.DataFrame(cols)
df.to_csv(tsv_path, sep="\t", index=False)
print(f"[PYTHON-INFO] UMAP TSV saved: {tsv_path}")
np.savez_compressed(npz_path, embedding=embedding, residue_ids=residue_ids, n_neighbors=config["n_neighbors"], min_dist=config["min_dist"], metric=config["metric"], seed=config["seed"], n_inputs=len(paths))
print(f"[PYTHON-INFO] UMAP NPZ (embedding+ids) saved: {npz_path}")

###################################### finalize
elapsed = time.perf_counter() - t0
max_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"[PYTHON-INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed={elapsed:.1f}s | max_mem={max_mem_mb:.1f}MB")

