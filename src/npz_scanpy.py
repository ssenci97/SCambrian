#!/usr/bin/env python3
"""Computes joint and intra-species clusters, UMAP and t-SNE layouts of ESMC protein embeddings and saves them to one .h5ad per model.

Embeddings are L2-normalised before a cosine kNN graph that is reused by Leiden clustering and by the UMAPs of every requested dimension (default 2D and 3D); t-SNE affinities come from one cosine kNN per space shared by all its dimensions. The same is repeated per species (intra), whose layouts are written row-wise into full-size ``*_intra`` arrays. Wilcoxon channel markers, silhouette scores, the joint kNN graph and the run parameters are stored alongside, while X is dropped to keep the file small. Figures and tables are produced afterwards by h5ad_scanpy.py.
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
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import time
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log2.*")
DEFAULT_OUTDIR = "data/scanpy/"
EARLY_EXAGGERATION = 12.0
LEIDEN_RES = 1.0
MAX_ITER = 1000
METHODS = ("umap", "tsne")
METRIC = "cosine"
MIN_DIST = 0.5
N_COMPONENTS = [2, 3]
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
N_NEIGHBORS = 15
N_PCS = None
OVERWRITE = False
PERPLEXITY = 30.0
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="ESMC Proteome Embedding Clustering and Layouts")
parser.add_argument("--npzs", type=str, required=True, help="Comma-separated paths to .npz files")
parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Base output directory")
parser.add_argument("--n_pcs", type=int, default=N_PCS, help="Number of PCs for dimensionality reduction")
parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS, help="Number of neighbors for KNN")
parser.add_argument("--resolution", type=float, default=LEIDEN_RES, help="Leiden cluster resolution")
parser.add_argument("--n_components", type=int, nargs="+", default=N_COMPONENTS, help="UMAP/t-SNE dimensions")
parser.add_argument("--min_dist", type=float, default=MIN_DIST, help="UMAP minimum distance")
parser.add_argument("--perplexity", type=float, default=PERPLEXITY, help="t-SNE perplexity")
parser.add_argument("--max_iter", type=int, default=MAX_ITER, help="t-SNE iterations")
parser.add_argument("--n_jobs", type=int, default=N_JOBS, help="CPU threads")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute an .h5ad that already exists")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
yargs, _ = parser.parse_known_args()
script_key = "npz_scanpy"
params = dict(seed=SEED, resolution=LEIDEN_RES, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, n_components=N_COMPONENTS, min_dist=MIN_DIST, perplexity=PERPLEXITY, max_iter=MAX_ITER, outdir=DEFAULT_OUTDIR, overwrite=OVERWRITE)
if yargs.yaml_in and Path(yargs.yaml_in).exists():
    ydata = yaml.safe_load(Path(yargs.yaml_in).read_text()) or {}
    if script_key in ydata: params.update(ydata[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    ydata = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    ydata[script_key] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(ydata, sort_keys=False))
###################################### HELPER_FUNCTIONS
def run_tsne(rep: np.ndarray, n_components: list[int]) -> dict[int, np.ndarray]:
    """Run t-SNE in every requested dimension from a single cosine kNN graph and a shared PCA initialisation.

    Args:
        rep (np.ndarray): Representation to embed.
        n_components (list[int]): Output dimensions.

    Returns:
        dict[int, np.ndarray]: Layout per dimension.
    """
    n = rep.shape[0]
    k = min(int(3 * args.perplexity) + 2, n - 1)
    idx, dist = NNDescent(rep, n_neighbors=k + 1, metric=METRIC, random_state=SEED, n_jobs=args.n_jobs).neighbor_graph
    graph = csr_matrix((np.clip(dist[:, 1:], 0, None).ravel(), idx[:, 1:].ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    pcs = PCA(n_components=max(n_components), svd_solver="randomized", random_state=SEED).fit_transform(rep)
    pcs = (pcs / np.std(pcs[:, 0]) * 1e-4).astype(np.float32)
    tsne_kw = dict(perplexity=min(args.perplexity, (k - 2) / 3), max_iter=args.max_iter, early_exaggeration=EARLY_EXAGGERATION, learning_rate="auto", metric="precomputed", random_state=SEED, n_jobs=args.n_jobs)
    return {nc: TSNE(n_components=nc, init=pcs[:, :nc], **tsne_kw).fit_transform(graph).astype(np.float32) for nc in n_components}
def cluster_and_embed(adata: ad.AnnData, rep_key: str, n_pcs_use: int | None, key: str, desc: str) -> float:
    """Build the cosine kNN graph, cluster it with Leiden and store UMAP and t-SNE layouts of every requested dimension in ``obsm``.

    Args:
        adata (ad.AnnData): Normalised embeddings, modified in place.
        rep_key (str): Representation used for neighbors (``X`` or ``X_pca``).
        n_pcs_use (int | None): Number of PCs passed to neighbors.
        key (str): ``obs`` column receiving the Leiden labels.
        desc (str): Progress bar label.

    Returns:
        float: Silhouette score of the Leiden partition in the representation space.
    """
    sc.pp.neighbors(adata, n_neighbors=min(args.n_neighbors, adata.n_obs - 1), use_rep=rep_key, n_pcs=n_pcs_use, metric=METRIC, random_state=SEED)
    sc.tl.leiden(adata, resolution=args.resolution, key_added=key, flavor="igraph", random_state=SEED)
    rep = np.asarray(adata.obsm["X_pca"] if rep_key == "X_pca" else adata.X)
    for nc in tqdm(args.n_components, desc=f"{desc} UMAP", leave=False):
        sc.tl.umap(adata, n_components=nc, min_dist=args.min_dist, random_state=SEED, key_added=f"X_umap_{nc}d")
    for nc, coords in run_tsne(rep, args.n_components).items():
        adata.obsm[f"X_tsne_{nc}d"] = coords
    return float(silhouette_score(rep, adata.obs[key], metric=METRIC, random_state=SEED))
def markers_table(adata: ad.AnnData, groupby: str, analysis: str, species: str) -> pd.DataFrame:
    """Rank channels per cluster with the Wilcoxon test and flatten the result.

    Args:
        adata (ad.AnnData): Embeddings with full channels in X.
        groupby (str): ``obs`` column with the cluster labels.
        analysis (str): Analysis tag (``inter_group`` or ``intra_group``).
        species (str): Species tag (``all`` for the joint analysis).

    Returns:
        pd.DataFrame: Columns analysis, species, cluster, channel, score and pvals_adj.
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")
    rgg = adata.uns.pop("rank_genes_groups")
    df = pd.concat([pd.DataFrame({"cluster": g, "channel": rgg["names"][g], "score": rgg["scores"][g], "pvals_adj": rgg["pvals_adj"][g]}) for g in rgg["names"].dtype.names], ignore_index=True)
    df.insert(0, "analysis", analysis)
    df.insert(1, "species", species)
    return df
###################################### MAIN
def main() -> None:
    """Load embeddings, preprocess, cluster and embed jointly and per species, then save everything to .h5ad."""
    start_time = time.time()
    sc.settings.n_jobs = args.n_jobs
    npz_files = [p.strip() for p in args.npzs.split(",")]
    if len(npz_files) < 2:
        raise ValueError("[PYTHON-ERROR] This comparative script requires at least two .npz files.")
    use_pca = args.n_pcs is not None and args.n_pcs > 0
    param_tag = f"{Path(npz_files[0]).parent.name}__res{args.resolution}_nn{args.n_neighbors}" + (f"_npc{args.n_pcs}" if use_pca else "_full")
    target_dir = Path(args.outdir) / ("_".join([Path(f).stem.replace("_reps", "") for f in npz_files]) + f"__{Path(npz_files[0]).parent.name}")
    h5ad_path = target_dir / f"{target_dir.name}__{param_tag.removeprefix(Path(npz_files[0]).parent.name + '__')}.h5ad"
    if h5ad_path.exists() and not args.overwrite:
        print(f"[PYTHON-INFO] Target file '{h5ad_path}' already exists. Skipping.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    ###################################### LOAD_DATA
    adatas = []
    for f in tqdm(npz_files, desc="Loading embeddings"):
        data = np.load(f)
        k_emb = next(k for k in data.files if "embeddings" in k)
        k_ids = next(k for k in data.files if "ids" in k)
        species_name = Path(f).stem.replace("_reps", "")
        obs_index = [f"{species_name}_{pid}" for pid in data[k_ids]]
        adatas.append(ad.AnnData(X=data[k_emb].astype(np.float32), obs=pd.DataFrame({"protein_id": data[k_ids], "species": species_name}, index=obs_index)))
    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()
    print(f"[PYTHON-INFO][DATA-SHAPE] Initial data shape: {adata.shape}")
    ###################################### PREPROCESSING
    norms = np.linalg.norm(adata.X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    adata.X = adata.X / norms
    sc.settings.seed = SEED
    if use_pca:
        sc.tl.pca(adata, n_comps=args.n_pcs)
        print(f"[PYTHON-INFO][DATA-SHAPE] Shape after PCA (n_pcs={args.n_pcs}): {adata.obsm['X_pca'].shape}")
        rep_key, n_pcs_use = "X_pca", args.n_pcs
    else:
        print(f"[PYTHON-INFO][DATA-SHAPE] Operating in full-space dimensions: {adata.shape[1]}")
        rep_key, n_pcs_use = "X", None
    space = "pca" if use_pca else "full"
    ###################################### JOINT
    metrics = {f"joint_silhouette_{space}": cluster_and_embed(adata, rep_key, n_pcs_use, "cluster", "Joint")}
    print(f"[PYTHON-INFO][METRIC] Joint Silhouette Score ({space} space): {metrics[f'joint_silhouette_{space}']:.4f}")
    markers = [markers_table(adata, "cluster", "inter_group", "all")]
    ###################################### INTRA_GROUP
    intra = pd.Series(index=adata.obs.index, dtype="object")
    for m in METHODS:
        for nc in args.n_components:
            adata.obsm[f"X_{m}_{nc}d_intra"] = np.full((adata.n_obs, nc), np.nan, dtype=np.float32)
    for sp in tqdm(list(adata.obs["species"].unique()), desc="Intra-group clustering"):
        mask = (adata.obs["species"] == sp).values
        if mask.sum() < max(args.n_neighbors + 1, 10):
            print(f"[PYTHON-INFO][INTRA] Skipping {sp}: too few points ({mask.sum()})")
            continue
        sub = ad.AnnData(X=adata.X[mask], obs=adata.obs.loc[mask, ["protein_id", "species"]].copy(), obsm={k: adata.obsm[k][mask] for k in ("X_pca",) if k in adata.obsm})
        metrics[f"intra_silhouette_{sp}"] = cluster_and_embed(sub, rep_key, n_pcs_use, "intra_cluster", sp)
        print(f"[PYTHON-INFO][METRIC] Intra Silhouette ({sp}): {metrics[f'intra_silhouette_{sp}']:.4f}")
        intra[mask] = sub.obs["intra_cluster"].astype(str).values
        for m in METHODS:
            for nc in args.n_components:
                adata.obsm[f"X_{m}_{nc}d_intra"][mask] = sub.obsm[f"X_{m}_{nc}d"]
        markers.append(markers_table(sub, "intra_cluster", "intra_group", sp))
    adata.obs["intra_cluster"] = pd.Categorical(intra)
    ###################################### SAVE
    run_params = {k: v for k, v in vars(args).items() if v is not None and k not in ("yaml_in", "yaml_out", "overwrite")}
    out = ad.AnnData(obs=adata.obs[["protein_id", "species", "cluster", "intra_cluster"]].copy(), obsm={k: v for k, v in adata.obsm.items() if k.startswith(("X_umap", "X_tsne"))}, obsp=dict(adata.obsp), uns={"neighbors": adata.uns["neighbors"], "params": run_params, "npz_files": npz_files, "param_tag": param_tag, "metrics": metrics, "markers": pd.concat(markers, ignore_index=True)})
    out.write_h5ad(h5ad_path, compression="gzip")
    print(f"[PYTHON-INFO][SAVE] {h5ad_path} | obsm: {', '.join(out.obsm.keys())}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
