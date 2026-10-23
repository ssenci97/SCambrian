#!/usr/bin/env python3
"""Run UMAP on one or more embedding files and save the coordinates as TSV and ``.npz``."""
############### <imports>
import argparse
import os
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import umap
from utils import add_yaml_args, fasta_label, load_npz_embeddings, log, log_resources, resolve_params

############### <constants>
SCRIPT_NAME = Path(__file__).stem
DEFAULTS = {"seed": 42, "n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "n_epochs": None, "learning_rate": 1.0, "init": "spectral", "n_jobs": -1, "output_dir": "data/dimred"}


############### <helpers>
def set_threads(n_jobs: int) -> int:
    """Resolve the core count and pin BLAS/OpenMP thread pools to it.

    Args:
        n_jobs: Requested cores, or -1 for all available.

    Returns:
        Number of cores in use.
    """
    n_cores = cpu_count() if n_jobs == -1 else n_jobs
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n_cores)
    sklearn.set_config(working_memory=1024)
    return n_cores


############### <main>
def main() -> None:
    """Parse arguments, resolve parameters, fit UMAP and save the coordinates."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tensor", help="Comma-separated .npz/.pt input paths")
    parser.add_argument("--output-tsv")
    parser.add_argument("--output-npz")
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--n-neighbors", type=int)
    parser.add_argument("--min-dist", type=float)
    parser.add_argument("--metric")
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--init")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--output-dir")
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    if not args.tensor:
        if args.yaml_out:
            sys.exit(0)
        parser.error("--tensor is required")
    n_cores = set_threads(params["n_jobs"])
    log(f"Detected {n_cores} CPU cores | seed={params['seed']}")
    paths = [Path(x.strip()) for x in args.tensor.split(",") if x.strip()]
    tensor, ids = load_npz_embeddings(paths)
    log(f"Merged tensor shape: {tensor.shape} | total entries: {len(ids)}")
    reducer = umap.UMAP(n_components=params["n_components"], n_neighbors=params["n_neighbors"], min_dist=params["min_dist"], metric=params["metric"], n_epochs=params["n_epochs"], learning_rate=params["learning_rate"], init=params["init"], random_state=params["seed"], n_jobs=n_cores)
    embedding = reducer.fit_transform(tensor)
    stem = "_".join(fasta_label(p) for p in paths)
    out_dir = Path(params["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_tag = f"umap_nc{params['n_components']}_nn{params['n_neighbors']}_mindist{params['min_dist']}"
    tsv_path = Path(args.output_tsv) if args.output_tsv else out_dir / f"{stem}_{meta_tag}_weights.tsv"
    npz_path = Path(args.output_npz) if args.output_npz else out_dir / f"{stem}_{meta_tag}_weights.npz"
    cols = {f"UMAP{i + 1}": embedding[:, i] for i in range(embedding.shape[1])}
    cols["id"] = ids
    pd.DataFrame(cols).to_csv(tsv_path, sep="\t", index=False)
    np.savez_compressed(npz_path, embedding=embedding, id=ids, n_components=params["n_components"], n_neighbors=params["n_neighbors"], min_dist=params["min_dist"], metric=params["metric"], seed=params["seed"], n_inputs=len(paths))
    log(f"Saved UMAP coordinates to {tsv_path} and {npz_path}")
    log_resources(start_time)


if __name__ == "__main__":
    main()
