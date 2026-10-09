#!/usr/bin/env python3
"""Perform t-SNE dimensionality reduction on one or more tensor representations (comma-separated) and export embeddings with residue ids."""
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
from sklearn.manifold import TSNE
import torch
from utils.yamlmanager import ConfigManager

###################################### constants
SEED = 42
N_COMPONENTS = 2
PERPLEXITY = 30.0
MAX_ITER = 1000
LEARNING_RATE = "auto"
EARLY_EXAGGERATION = 12.0
METRIC = "euclidean"
INIT = "pca"
OUTPUT_DIR = "data/tsne"

###################################### cli
parser = argparse.ArgumentParser(description="t-SNE on tensor representations (comma-separated inputs supported)")
parser.add_argument("--tensor", type=str, required=True, help="Path(s) to input .pt or .npz representations, comma-separated")
parser.add_argument("--output-tsv", type=str, default=None, help="Output TSV path (default: derived from first input)")
parser.add_argument("--output-npz", type=str, default=None, help="Output NPZ path for embedding+ids (default: derived)")
parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
parser.add_argument("--perplexity", type=float, default=PERPLEXITY)
parser.add_argument("--max-iter", type=int, default=MAX_ITER)
parser.add_argument("--learning-rate", type=str, default=LEARNING_RATE)
parser.add_argument("--early-exaggeration", type=float, default=EARLY_EXAGGERATION)
parser.add_argument("--metric", type=str, default=METRIC)
parser.add_argument("--init", type=str, default=INIT)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--n-jobs", type=int, default=-1, help="-1 uses all cores")
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--verbose", type=int, default=1)
args = parser.parse_args()

CONSTANTS = dict(SEED=SEED, N_COMPONENTS=N_COMPONENTS, PERPLEXITY=PERPLEXITY, MAX_ITER=MAX_ITER, LEARNING_RATE=LEARNING_RATE, EARLY_EXAGGERATION=EARLY_EXAGGERATION, METRIC=METRIC, INIT=INIT, OUTPUT_DIR=OUTPUT_DIR)
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

###################################### tsne
print(f"[PYTHON-INFO] Fitting t-SNE perplexity={config['perplexity']} max_iter={config['max_iter']}...")
lr = config["learning_rate"] if config["learning_rate"] != "auto" else "auto"
tsne = TSNE(n_components=config["n_components"], perplexity=config["perplexity"], max_iter=config["max_iter"], learning_rate=lr, early_exaggeration=config["early_exaggeration"], metric=config["metric"], init=config["init"], random_state=config["seed"], n_jobs=n_cores, verbose=config["verbose"])
embedding = tsne.fit_transform(tensor)
print(f"[PYTHON-INFO] Embedding shape: {embedding.shape}")

###################################### output paths
stem = paths[0].stem
if stem.endswith("_reps"):
    stem = stem[:-5]
out_dir = Path(config["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
tsv_path = Path(config["output_tsv"]) if config["output_tsv"] else out_dir / f"{stem}_tsne.tsv"
npz_path = Path(config["output_npz"]) if config["output_npz"] else out_dir / f"{stem}_tsne.npz"

###################################### save
cols = {f"tSNE{i+1}": embedding[:, i] for i in range(embedding.shape[1])}
cols["residue_ids"] = residue_ids
df = pd.DataFrame(cols)
df.to_csv(tsv_path, sep="\t", index=False)
print(f"[PYTHON-INFO] t-SNE TSV saved: {tsv_path}")
np.savez_compressed(npz_path, embedding=embedding, residue_ids=residue_ids, perplexity=config["perplexity"], max_iter=config["max_iter"], metric=config["metric"], seed=config["seed"], n_inputs=len(paths))
print(f"[PYTHON-INFO] t-SNE NPZ (embedding+ids) saved: {npz_path}")

###################################### finalize
elapsed = time.perf_counter() - t0
max_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"[PYTHON-INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed={elapsed:.1f}s | max_mem={max_mem_mb:.1f}MB")

