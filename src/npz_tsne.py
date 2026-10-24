#!/usr/bin/env python3
"""Perform t-SNE dimensionality reduction on tensor representations."""
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
from sklearn.manifold import TSNE
import torch
import yaml
EARLY_EXAGGERATION = 12.0
INIT = "pca"
LEARNING_RATE = "auto"
MAX_ITER = 1000
METRIC = "euclidean"
N_COMPONENTS = 2
OUTPUT_DIR = "data/dimred"
PERPLEXITY = 30.0
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="t-SNE dimensionality reduction.")
parser.add_argument("--tensor", type=str, required=True, help="Comma-separated input paths")
parser.add_argument("--output_tsv", "--output-tsv", type=str, default=None)
parser.add_argument("--output_npz", "--output-npz", type=str, default=None)
parser.add_argument("--n_components", "--n-components", type=int, default=N_COMPONENTS)
parser.add_argument("--perplexity", type=float, default=PERPLEXITY)
parser.add_argument("--max_iter", "--max-iter", type=int, default=MAX_ITER)
parser.add_argument("--learning_rate", "--learning-rate", type=str, default=LEARNING_RATE)
parser.add_argument("--early_exaggeration", "--early-exaggeration", type=float, default=EARLY_EXAGGERATION)
parser.add_argument("--metric", type=str, default=METRIC)
parser.add_argument("--init", type=str, default=INIT)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--n_jobs", "--n-jobs", type=int, default=-1)
parser.add_argument("--output_dir", "--output-dir", type=str, default=OUTPUT_DIR)
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
yargs, _ = parser.parse_known_args()
script_key = "npz_tsne"
params = dict(seed=SEED, n_components=N_COMPONENTS, perplexity=PERPLEXITY, max_iter=MAX_ITER, learning_rate=LEARNING_RATE, early_exaggeration=EARLY_EXAGGERATION, metric=METRIC, init=INIT, output_dir=OUTPUT_DIR)
if yargs.yaml_in and Path(yargs.yaml_in).exists():
    ydata = yaml.safe_load(Path(yargs.yaml_in).read_text()) or {}
    if script_key in ydata: params.update(ydata[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
config = {**params, **{k: v for k, v in vars(args).items() if v is not None}}
if args.yaml_out:
    ydata = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    ydata[script_key] = params
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(ydata, sort_keys=False))
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
###################################### TSNE
lr = float(config["learning_rate"]) if config["learning_rate"] != "auto" else "auto"
tsne = TSNE(n_components=config["n_components"], perplexity=config["perplexity"], max_iter=config["max_iter"], learning_rate=lr, early_exaggeration=config["early_exaggeration"], metric=config["metric"], init=config["init"], random_state=config["seed"], n_jobs=n_cores, verbose=1)
embedding = tsne.fit_transform(tensor)
###################################### SAVE
stems = [p.stem[:-5] if p.stem.endswith("_reps") else p.stem for p in paths]
stem = "_".join(stems) if len(stems) > 1 else stems[0]
out_dir = Path(config["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
meta_tag = f"tsne_nc{config['n_components']}_perp{config['perplexity']}_iter{config['max_iter']}"
tsv_path = Path(config["output_tsv"]) if config.get("output_tsv") else out_dir / f"{stem}_{meta_tag}_weights.tsv"
npz_path = Path(config["output_npz"]) if config.get("output_npz") else out_dir / f"{stem}_{meta_tag}_weights.npz"
cols = {f"tSNE{i+1}": embedding[:, i] for i in range(embedding.shape[1])}
cols["id"] = residue_ids
pd.DataFrame(cols).to_csv(tsv_path, sep="\t", index=False)
np.savez_compressed(npz_path, embedding=embedding, id=residue_ids, n_components=config["n_components"], perplexity=config["perplexity"], max_iter=config["max_iter"], metric=config["metric"], seed=config["seed"], n_inputs=len(paths))
print(f"[PYTHON-INFO] Saved t-SNE weights to {tsv_path}")
###################################### FINALIZE
elapsed = time.perf_counter() - t0
max_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"[PYTHON-INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {elapsed:.2f}s {max_mem_mb:.2f}MB")
if __name__ == "__main__":
    pass