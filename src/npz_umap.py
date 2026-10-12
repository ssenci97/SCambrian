#!/usr/bin/env python3
"""Perform UMAP dimensionality reduction on tensor representations."""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import os
import resource
import time
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
import umap
import yaml
INIT = "spectral"
LEARNING_RATE = 1.0
METRIC = "euclidean"
MIN_DIST = 0.1
N_COMPONENTS = 2
N_EPOCHS = None
N_NEIGHBORS = 15
OUTPUT_DIR = "data/dimred"
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="UMAP dimensionality reduction.")
parser.add_argument("--tensor", type=str, required=True, help="Comma-separated input paths")
parser.add_argument("--output-tsv", type=str, default=None)
parser.add_argument("--output-npz", type=str, default=None)
parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
parser.add_argument("--min-dist", type=float, default=MIN_DIST)
parser.add_argument("--metric", type=str, default=METRIC)
parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
parser.add_argument("--init", type=str, default=INIT)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--yaml-in", action="store_true")
parser.add_argument("--yaml-out", action="store_true")
args = parser.parse_args()
script_key = "npz_umap"
params = dict(seed=SEED, n_components=N_COMPONENTS, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric=METRIC, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, init=INIT, output_dir=OUTPUT_DIR)
if args.yaml_in and Path(YAML_PATH).exists():
    ydata = yaml.safe_load(Path(YAML_PATH).read_text()) or {}
    if script_key in ydata: params.update(ydata[script_key])
for k, v in params.items():
    if not getattr(args, k, None) and k in vars(args): setattr(args, k, v)
config = {**params, **{k: v for k, v in vars(args).items() if v is not None}}
if args.yaml_out:
    ydata = yaml.safe_load(Path(YAML_PATH).read_text()) if Path(YAML_PATH).exists() else {}
    ydata[script_key] = params
    Path(YAML_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(YAML_PATH).write_text(yaml.dump(ydata, sort_keys=False))
###################################### SETUP
t0 = time.perf_counter()
n_cores = cpu_count() if config["n_jobs"] == -1 else config["n_jobs"]
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
sklearn.set_config(working_memory=1024)
print(f"[PYTHON-INFO] Detected {n_cores} CPU cores | seed={config['seed']}")
###################################### LOAD
paths = [Path(x.strip()) for x in config["tensor"].split(",") if x.strip()]
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
    tensors.append(np.asarray(t))
    ids_list.append(np.asarray(r))
tensor = np.concatenate(tensors, axis=0)
residue_ids = np.concatenate(ids_list, axis=0)
print(f"[PYTHON-INFO] Merged tensor shape: {tensor.shape} | total residues: {len(residue_ids)}")
###################################### UMAP
reducer = umap.UMAP(n_components=config["n_components"], n_neighbors=config["n_neighbors"], min_dist=config["min_dist"], metric=config["metric"], n_epochs=config["n_epochs"], learning_rate=config["learning_rate"], init=config["init"], random_state=config["seed"], n_jobs=n_cores)
embedding = reducer.fit_transform(tensor)
###################################### SAVE
stems = [p.stem[:-5] if p.stem.endswith("_reps") else p.stem for p in paths]
stem = "_".join(stems) if len(stems) > 1 else stems[0]
out_dir = Path(config["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
meta_tag = f"umap_nc{config['n_components']}_nn{config['n_neighbors']}_mindist{config['min_dist']}"
tsv_path = Path(config["output_tsv"]) if config.get("output_tsv") else out_dir / f"{stem}_{meta_tag}_weights.tsv"
npz_path = Path(config["output_npz"]) if config.get("output_npz") else out_dir / f"{stem}_{meta_tag}_weights.npz"
cols = {f"UMAP{i+1}": embedding[:, i] for i in range(embedding.shape[1])}
cols["id"] = residue_ids
pd.DataFrame(cols).to_csv(tsv_path, sep="\t", index=False)
np.savez_compressed(npz_path, embedding=embedding, id=residue_ids, n_components=config["n_components"], n_neighbors=config["n_neighbors"], min_dist=config["min_dist"], metric=config["metric"], seed=config["seed"], n_inputs=len(paths))
print(f"[PYTHON-INFO] Saved UMAP weights to {tsv_path}")
###################################### FINALIZE
elapsed = time.perf_counter() - t0
max_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"[PYTHON-INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {elapsed:.2f}s {max_mem_mb:.2f}MB")
if __name__ == "__main__":
    pass