#!/usr/bin/env python3
"""Projects the ESMC embeddings of one model into UMAP and t-SNE spaces, jointly (all samples) and internally (each target alone), and clusters every layout with Leiden.

The kept sequences of ``<analysis_dir>/sc_metadata.tsv`` (preprocess_sequences.py) are read from ``<embeddings_dir>/<model>/<label>_reps.npz``. Embeddings are L2-normalised (and optionally reduced to ``pca_components`` PCs). For the joint scope (every sample, probes included) and for each target alone (internal scope; probes get none), a cosine kNN graph of ``knn_neighbors`` neighbours feeds the UMAPs of every dimension in ``layout_dimensions``, and one cosine kNN per scope feeds the t-SNEs. Every layout is then clustered on its own coordinates with Leiden (``leiden_resolution``) on a Euclidean kNN graph of the layout. Raw integer cluster ids are stored per series key ``{u|t}_{j|i}_leiden{2|3}d`` in ``obs`` (NaN where the series does not apply), layouts as ``obsm["X_{umap|tsne}_{n}d_{j|i}"]``, and the joint embedding kNN graph in ``obsp`` for the re-projections of h5ad_scanpy.py. The output is ``<analysis_dir>/<model>/projections.h5ad``; compile_summary.py turns the raw ids into cluster names. The .h5ad is recomputed when it is older than an input .npz or the metadata, was built with different parameters, or with ``--overwrite``.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
from pynndescent import NNDescent
import resource
import scanpy as sc
from scipy.sparse import csr_array, csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
if not hasattr(csr_array, "getnnz"):
    csr_array.getnnz = csr_array._getnnz
EARLY_EXAGGERATION = 12.0
EMBEDDINGS_DIR = "data/esm-reps"
KNN_NEIGHBORS = 15
LAYOUT_DIMENSIONS = [2, 3]
LEIDEN_RESOLUTION = 1.0
METADATA_FILE = "sc_metadata.tsv"
METHODS = {"umap": "u", "tsne": "t"}
METRIC = "cosine"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
OUTPUT_FILE = "projections.h5ad"
OVERWRITE = False
PCA_COMPONENTS = None
SEED = 42
TSNE_ITERATIONS = 1000
TSNE_PERPLEXITY = 30.0
UMAP_MIN_DIST = 0.5
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Joint and internal UMAP/t-SNE projections of ESMC embeddings, clustered with Leiden")
parser.add_argument("--analysis_dir", type=str, required=True, help="Analysis folder analyses/<analysis_name>/ holding sc_metadata.tsv")
parser.add_argument("--model", type=str, required=True, help="ESMC model whose embeddings are projected (folder name under the embeddings_dir)")
parser.add_argument("--seed", type=int, default=SEED, help="Random seed of PCA, kNN, UMAP, t-SNE and Leiden")
parser.add_argument("--pca_components", type=int, default=PCA_COMPONENTS, help="PCs used for the kNN graphs (None or 0: full embedding space)")
parser.add_argument("--knn_neighbors", type=int, default=KNN_NEIGHBORS, help="Neighbours of the embedding kNN graph (UMAP) and of the layout kNN graphs (Leiden)")
parser.add_argument("--leiden_resolution", type=float, default=LEIDEN_RESOLUTION, help="Leiden resolution (higher: more, smaller clusters)")
parser.add_argument("--layout_dimensions", type=int, nargs="+", default=LAYOUT_DIMENSIONS, help="Dimensions of the UMAP and t-SNE layouts")
parser.add_argument("--umap_min_dist", type=float, default=UMAP_MIN_DIST, help="UMAP minimum distance between embedded points")
parser.add_argument("--tsne_perplexity", type=float, default=TSNE_PERPLEXITY, help="t-SNE perplexity")
parser.add_argument("--tsne_iterations", type=int, default=TSNE_ITERATIONS, help="t-SNE optimisation iterations")
parser.add_argument("--n_jobs", type=int, default=N_JOBS, help="CPU threads")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute an .h5ad that is fresh")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "npz_scanpy"
params = dict(seed=SEED, pca_components=PCA_COMPONENTS, knn_neighbors=KNN_NEIGHBORS, leiden_resolution=LEIDEN_RESOLUTION, layout_dimensions=LAYOUT_DIMENSIONS, umap_min_dist=UMAP_MIN_DIST, tsne_perplexity=TSNE_PERPLEXITY, tsne_iterations=TSNE_ITERATIONS, overwrite=OVERWRITE)
config_data = {}
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    if script_key in config_data: params.update(config_data[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    out_data = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    out_data[script_key] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(out_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def run_params() -> dict:
    """Return the parameters that define a run, as stored in ``uns["params"]``.

    Returns:
        dict: Every parameter except paths, threading and overwrite switches.
    """
    return {k: v for k, v in vars(args).items() if v is not None and k not in ("yaml_in", "yaml_out", "overwrite", "n_jobs", "analysis_dir")}
def is_fresh(h5ad_path: Path, inputs: list[Path], current: dict) -> bool:
    """Tell whether an existing .h5ad is newer than every input and was built with the current parameters.

    Args:
        h5ad_path (Path): Output .h5ad.
        inputs (list[Path]): Input files.
        current (dict): Current run parameters.

    Returns:
        bool: True when the run can be skipped.
    """
    if not h5ad_path.exists() or h5ad_path.stat().st_mtime < max(p.stat().st_mtime for p in inputs): return False
    stored = ad.read_h5ad(h5ad_path, backed="r")
    same = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in stored.uns["params"].items()} == {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in current.items()}
    stored.file.close()
    return same
def load_embeddings(metadata: pd.DataFrame, npz_files: dict[str, Path]) -> ad.AnnData:
    """Load the embeddings of the kept sequences of every sample into one AnnData indexed by record.

    Args:
        metadata (pd.DataFrame): Rows of sc_metadata.tsv with status kept.
        npz_files (dict[str, Path]): ``<label>_reps.npz`` per sample label.

    Returns:
        ad.AnnData: L2-normalised embeddings with ``record``, ``sc_label`` and ``sc_type`` observations.
    """
    parts = []
    for label, path in tqdm(npz_files.items(), desc="Loading embeddings"):
        data = np.load(path)
        ids, emb = data["ids"].astype(str), data["embeddings"].astype(np.float32)
        kept = metadata.loc[metadata["sc_label"] == label]
        keep = np.isin(ids, kept["record"].values)
        missing = len(set(kept["record"]) - set(ids))
        if missing or (~keep).any(): print(f"[PYTHON-WARNING] {label}: {missing:,} kept records without embedding, {(~keep).sum():,} embeddings not kept in the metadata (rerun step 04?)")
        types = dict(zip(kept["record"], kept["sc_type"]))
        parts.append(ad.AnnData(X=emb[keep], obs=pd.DataFrame({"record": ids[keep], "sc_label": label, "sc_type": [types[i] for i in ids[keep]]}, index=ids[keep])))
    adata = ad.concat(parts, join="outer")
    norms = np.linalg.norm(adata.X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    adata.X = adata.X / norms
    print(f"[PYTHON-INFO][DATA-SHAPE] {adata.shape} from {len(npz_files)} samples")
    return adata
def representation(adata: ad.AnnData) -> np.ndarray:
    """Return the representation the kNN graphs are built on: optional PCA of the normalised embeddings.

    Args:
        adata (ad.AnnData): Normalised embeddings.

    Returns:
        np.ndarray: Representation per protein.
    """
    if args.pca_components: return PCA(n_components=args.pca_components, random_state=args.seed).fit_transform(np.asarray(adata.X)).astype(np.float32)
    return np.asarray(adata.X)
def run_tsne(rep: np.ndarray) -> dict[int, np.ndarray]:
    """Run t-SNE in every layout dimension from a single cosine kNN graph and a shared PCA initialisation.

    Args:
        rep (np.ndarray): Representation to embed.

    Returns:
        dict[int, np.ndarray]: Layout per dimension.
    """
    n = rep.shape[0]
    k = min(int(3 * args.tsne_perplexity) + 2, n - 1)
    idx, dist = NNDescent(rep, n_neighbors=k + 1, metric=METRIC, random_state=args.seed, n_jobs=args.n_jobs).neighbor_graph
    graph = csr_matrix((np.clip(dist[:, 1:], 0, None).ravel(), idx[:, 1:].ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    pcs = PCA(n_components=max(args.layout_dimensions), svd_solver="randomized", random_state=args.seed).fit_transform(rep)
    pcs = (pcs / np.std(pcs[:, 0]) * 1e-4).astype(np.float32)
    tsne_kw = dict(perplexity=min(args.tsne_perplexity, (k - 2) / 3), max_iter=args.tsne_iterations, early_exaggeration=EARLY_EXAGGERATION, learning_rate="auto", metric="precomputed", random_state=args.seed, n_jobs=args.n_jobs)
    return {nc: TSNE(n_components=nc, init=pcs[:, :nc], **tsne_kw).fit_transform(graph).astype(np.float32) for nc in args.layout_dimensions}
def cluster_layout(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Cluster one layout on its own coordinates with Leiden on a Euclidean kNN graph of the layout.

    Args:
        coords (np.ndarray): Layout coordinates.

    Returns:
        dict[str, np.ndarray]: Integer labels per method (``leiden``).
    """
    tmp = ad.AnnData(obsm={"L": coords}, obs=pd.DataFrame(index=np.arange(len(coords)).astype(str)))
    sc.pp.neighbors(tmp, n_neighbors=min(args.knn_neighbors, len(coords) - 1), use_rep="L", metric="euclidean", random_state=args.seed)
    sc.tl.leiden(tmp, resolution=args.leiden_resolution, flavor="igraph", n_iterations=2, directed=False, random_state=args.seed)
    return {"leiden": tmp.obs["leiden"].astype(int).to_numpy()}
def project(adata: ad.AnnData, scope: str, desc: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the UMAP and t-SNE layouts of one scope and cluster every layout.

    Args:
        adata (ad.AnnData): Normalised embeddings of the scope, modified in place (kNN graph).
        scope (str): ``j`` (joint) or ``i`` (internal).
        desc (str): Progress bar label.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: Layouts per ``obsm`` key and raw cluster ids per series key.
    """
    rep = representation(adata)
    adata.obsm["X_rep"] = rep
    sc.pp.neighbors(adata, n_neighbors=min(args.knn_neighbors, adata.n_obs - 1), use_rep="X_rep", metric=METRIC, random_state=args.seed)
    layouts = {}
    for nc in tqdm(args.layout_dimensions, desc=f"{desc} UMAP", leave=False):
        sc.tl.umap(adata, n_components=nc, min_dist=args.umap_min_dist, random_state=args.seed, key_added="X_tmp")
        layouts[f"X_umap_{nc}d_{scope}"] = np.asarray(adata.obsm.pop("X_tmp"), dtype=np.float32)
    layouts |= {f"X_tsne_{nc}d_{scope}": coords for nc, coords in run_tsne(rep).items()}
    series = {}
    for key, coords in tqdm(layouts.items(), desc=f"{desc} clustering", leave=False):
        method, nc = key.split("_")[1], key.split("_")[2]
        series |= {f"{METHODS[method]}_{scope}_{stat}{nc}": labels for stat, labels in cluster_layout(coords).items()}
    return layouts, series
###################################### MAIN
def main() -> None:
    """Load the kept embeddings, project and cluster them jointly and per target, and save one .h5ad."""
    start_time = time.time()
    sc.settings.n_jobs, sc.settings.seed = args.n_jobs, args.seed
    analysis_dir = Path(args.analysis_dir)
    metadata_path = analysis_dir / METADATA_FILE
    if not metadata_path.exists(): raise FileNotFoundError(f"[PYTHON-ERROR] Missing {metadata_path}; run step 02 first.")
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    metadata = metadata[metadata["status"] == "kept"]
    labels = list(dict.fromkeys(metadata["sc_label"]))
    reps_dir = Path(config_data.get("extract_reps_biohub", {}).get("embeddings_dir", EMBEDDINGS_DIR)) / args.model
    npz_files = {lbl: reps_dir / f"{lbl}_reps.npz" for lbl in labels}
    missing = [str(p) for p in npz_files.values() if not p.exists()]
    if missing: raise FileNotFoundError(f"[PYTHON-ERROR] Missing embeddings {missing}; run step 04 first.")
    out_dir = analysis_dir / args.model
    h5ad_path, current = out_dir / OUTPUT_FILE, run_params()
    if not args.overwrite and is_fresh(h5ad_path, [metadata_path, *npz_files.values()], current):
        print(f"[PYTHON-INFO] '{h5ad_path}' is newer than its inputs and has the current parameters. Skipping.")
        return
    adata = load_embeddings(metadata, npz_files)
    layouts, series = project(adata, "j", "Joint")
    obs = adata.obs[["record", "sc_label", "sc_type"]].copy()
    for key, labels_j in series.items(): obs[key] = labels_j.astype(float)
    obsm = dict(layouts)
    targets = [lbl for lbl in labels if (metadata.loc[metadata["sc_label"] == lbl, "sc_type"] == "target").all()]
    for lbl in tqdm(targets, desc="Internal projections"):
        mask = (adata.obs["sc_label"] == lbl).values
        if mask.sum() <= max(args.knn_neighbors + 1, 3 * args.tsne_perplexity):
            print(f"[PYTHON-WARNING] {lbl}: {mask.sum()} proteins, too few for an internal projection. Skipping.")
            continue
        sub = ad.AnnData(X=adata.X[mask], obs=adata.obs.loc[mask, ["record"]].copy())
        sub_layouts, sub_series = project(sub, "i", lbl)
        for key, coords in sub_layouts.items():
            obsm.setdefault(key, np.full((adata.n_obs, coords.shape[1]), np.nan, dtype=np.float32))[mask] = coords
        for key, ids in sub_series.items():
            if key not in obs: obs[key] = np.nan
            obs.loc[mask, key] = ids.astype(float)
    print(f"[PYTHON-INFO][SERIES] {len([c for c in obs if c.startswith(('u_', 't_'))])} series: {', '.join(c for c in obs if c.startswith(('u_', 't_')))}")
    out = ad.AnnData(obs=obs, obsm=obsm, obsp={k: adata.obsp[k] for k in ("connectivities", "distances")}, uns={"neighbors": {k: v for k, v in adata.uns["neighbors"].items() if k != "params"} | {"params": {k: v for k, v in adata.uns["neighbors"]["params"].items() if v is not None}}, "params": current, "model": args.model, "npz_files": [str(p) for p in npz_files.values()]})
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(h5ad_path, compression="gzip")
    print(f"[PYTHON-INFO][SAVE] {h5ad_path} | obsm: {', '.join(out.obsm.keys())}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
