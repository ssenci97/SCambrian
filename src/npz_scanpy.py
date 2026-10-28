#!/usr/bin/env python3
"""Computes joint and intra-species clusters, UMAP and t-SNE layouts of ESMC protein embeddings and saves them to one .h5ad per model.

Embeddings are L2-normalised (and optionally reduced to ``pca_components`` PCs) before a cosine kNN graph of ``knn_neighbors`` neighbours that is reused by Leiden clustering (``leiden_resolution``) and by the UMAPs of every dimension in ``layout_dimensions`` (default 2D and 3D); t-SNE affinities come from one cosine kNN per space shared by all its dimensions. The same is repeated per target (intra), whose layouts are written row-wise into full-size ``*_intra`` arrays. Without any probe input the run is in comparative mode and is identical to a run without input roles (same parameters, no focus tiers, no gravity), so existing .h5ad files stay fresh; with at least one probe it is in focus mode. Every label takes the ``input_type`` of the query table: ``target`` (compared set, default) or ``probe`` (functional reference that takes part in the joint graph, clusters and layouts but not in the intra clustering). Target proteins listed in ``<features_dir>/<label>_focus_ids.txt`` (dw_uniprot_focus.py) are tagged ``annotated``, probe proteins ``probe``, the rest ``none`` (``obs["focus_tier"]``); probe accessions already present in a target are dropped from the probe, so that each protein is counted once. ``obs["focus_similarity"]`` is the cosine similarity of every protein to the normalised centroid of the focus set (probe and annotated proteins), the gravity centre summarised by h5ad_scanpy.py. Wilcoxon channel markers, silhouette scores, the joint kNN graph and the run parameters are stored alongside, while X is dropped to keep the file small. Figures and tables are produced afterwards by h5ad_scanpy.py. An existing .h5ad is recomputed when it is older than any input .npz or was built with different parameters, or with ``--overwrite``.
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
from sklearn.metrics import silhouette_score
import time
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log2.*")
if not hasattr(csr_array, "getnnz"):
    csr_array.getnnz = csr_array._getnnz
EARLY_EXAGGERATION = 12.0
FEATURES_DIR = "data/features"
INPUT_TYPES = ("target", "probe")
KNN_NEIGHBORS = 15
LAYOUT_DIMENSIONS = [2, 3]
LEIDEN_RESOLUTION = 1.0
METHODS = ("umap", "tsne")
METRIC = "cosine"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
OVERWRITE = False
PCA_COMPONENTS = None
QUERY_TABLE = "configs/query_configs.tsv"
SCANPY_DIR = "data/scanpy/"
SEED = 42
TSNE_ITERATIONS = 1000
TSNE_PERPLEXITY = 30.0
UMAP_MIN_DIST = 0.5
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="ESMC Proteome Embedding Clustering and Layouts")
parser.add_argument("--npzs", type=str, required=True, help="Comma-separated <label>_reps.npz files of one model (at least two species)")
parser.add_argument("--scanpy_dir", type=str, default=SCANPY_DIR, help="Base output folder; each run goes to <scanpy_dir>/<labels>__<model>/")
parser.add_argument("--pca_components", type=int, default=PCA_COMPONENTS, help="PCs used for the kNN graph (None or 0: full embedding space)")
parser.add_argument("--knn_neighbors", type=int, default=KNN_NEIGHBORS, help="Neighbours of the cosine kNN graph used by Leiden and UMAP")
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
params = dict(seed=SEED, leiden_resolution=LEIDEN_RESOLUTION, knn_neighbors=KNN_NEIGHBORS, pca_components=PCA_COMPONENTS, layout_dimensions=LAYOUT_DIMENSIONS, umap_min_dist=UMAP_MIN_DIST, tsne_perplexity=TSNE_PERPLEXITY, tsne_iterations=TSNE_ITERATIONS, scanpy_dir=SCANPY_DIR, overwrite=OVERWRITE)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    if script_key in config_data: params.update(config_data[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    config_data = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    config_data[script_key] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(config_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def load_roles(labels: list[str]) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    """Read the input type of every label from the query table and the focus accessions of every target.

    Args:
        labels (list[str]): Labels of the input .npz files.

    Returns:
        tuple[dict[str, str], dict[str, set[str]], dict[str, str]]: Input type per label, focus accessions per target and the focus definition.
    """
    config_data = (yaml.safe_load(Path(args.yaml_in).read_text()) or {}) if args.yaml_in and Path(args.yaml_in).exists() else {}
    tsv = Path(config_data.get("dw_uniprot_seqs", {}).get("query_table", QUERY_TABLE))
    focus_cfg = config_data.get("dw_uniprot_focus", {})
    features_dir = Path(focus_cfg.get("features_dir", FEATURES_DIR))
    table = pd.read_csv(tsv, sep="\t", dtype=str) if tsv.exists() else pd.DataFrame(columns=["label"])
    types = dict(zip(table["label"], table["input_type"])) if "input_type" in table else {}
    roles = {lbl: (str(types.get(lbl)).strip() if pd.notna(types.get(lbl)) else "target") for lbl in labels}
    bad = {lbl: r for lbl, r in roles.items() if r not in INPUT_TYPES}
    if bad: raise ValueError(f"[PYTHON-ERROR] input_type must be one of {INPUT_TYPES}: {bad}")
    focus = {lbl: set((features_dir / f"{lbl}_focus_ids.txt").read_text().split()) for lbl in labels if roles[lbl] == "target" and (features_dir / f"{lbl}_focus_ids.txt").exists()}
    print(f"[PYTHON-INFO][ROLES] {', '.join(f'{lbl}={r}' for lbl, r in roles.items())}; focus lists: {', '.join(f'{lbl} ({len(ids):,})' for lbl, ids in focus.items()) or 'none'}")
    return roles, focus, {"focus_name": str(focus_cfg.get("focus_name", "")), "focus_query": str(focus_cfg.get("focus_query", ""))}
def run_params(roles: dict[str, str] | None = None, focus: dict[str, set[str]] | None = None, focus_def: dict[str, str] | None = None) -> dict:
    """Return the parameters that define a run, as stored in ``uns["params"]`` of its .h5ad.

    Args:
        roles (dict[str, str] | None): Input type per label.
        focus (dict[str, set[str]] | None): Focus accessions per target.
        focus_def (dict[str, str] | None): Focus name and UniProtKB clause.

    Returns:
        dict: Every parameter except output, threading and overwrite switches, plus the input roles and the focus definition.
    """
    out = {k: v for k, v in vars(args).items() if v is not None and k not in ("yaml_in", "yaml_out", "overwrite", "n_jobs", "npzs", "scanpy_dir")}
    if roles is not None: out["input_types"] = [f"{lbl}:{r}" for lbl, r in roles.items()]
    if focus is not None: out["focus_counts"] = [f"{lbl}:{len(ids)}" for lbl, ids in focus.items()] or ["none"]
    if focus_def is not None: out.update(focus_def)
    return out
def is_fresh(h5ad_path: Path, npz_files: list[str], current: dict) -> bool:
    """Tell whether an existing .h5ad is newer than every input and was built with the current parameters.

    Args:
        h5ad_path (Path): Target .h5ad.
        npz_files (list[str]): Input .npz files.
        current (dict): Current run parameters (``run_params``).

    Returns:
        bool: True when the run can be skipped.
    """
    if not h5ad_path.exists() or h5ad_path.stat().st_mtime < max(Path(f).stat().st_mtime for f in npz_files): return False
    stored = ad.read_h5ad(h5ad_path, backed="r")
    same = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in stored.uns["params"].items()} == {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in current.items()}
    stored.file.close()
    return same
def run_tsne(rep: np.ndarray, n_components: list[int]) -> dict[int, np.ndarray]:
    """Run t-SNE in every requested dimension from a single cosine kNN graph and a shared PCA initialisation.

    Args:
        rep (np.ndarray): Representation to embed.
        n_components (list[int]): Output dimensions.

    Returns:
        dict[int, np.ndarray]: Layout per dimension.
    """
    n = rep.shape[0]
    k = min(int(3 * args.tsne_perplexity) + 2, n - 1)
    idx, dist = NNDescent(rep, n_neighbors=k + 1, metric=METRIC, random_state=SEED, n_jobs=args.n_jobs).neighbor_graph
    graph = csr_matrix((np.clip(dist[:, 1:], 0, None).ravel(), idx[:, 1:].ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    pcs = PCA(n_components=max(n_components), svd_solver="randomized", random_state=SEED).fit_transform(rep)
    pcs = (pcs / np.std(pcs[:, 0]) * 1e-4).astype(np.float32)
    tsne_kw = dict(perplexity=min(args.tsne_perplexity, (k - 2) / 3), max_iter=args.tsne_iterations, early_exaggeration=EARLY_EXAGGERATION, learning_rate="auto", metric="precomputed", random_state=SEED, n_jobs=args.n_jobs)
    return {nc: TSNE(n_components=nc, init=pcs[:, :nc], **tsne_kw).fit_transform(graph).astype(np.float32) for nc in n_components}
def cluster_and_embed(adata: ad.AnnData, rep_key: str, pcs_for_neighbors: int | None, key: str, desc: str) -> float:
    """Build the cosine kNN graph, cluster it with Leiden and store UMAP and t-SNE layouts of every requested dimension in ``obsm``.

    Args:
        adata (ad.AnnData): Normalised embeddings, modified in place.
        rep_key (str): Representation used for neighbors (``X`` or ``X_pca``).
        pcs_for_neighbors (int | None): Number of PCs passed to neighbors.
        key (str): ``obs`` column receiving the Leiden labels.
        desc (str): Progress bar label.

    Returns:
        float: Silhouette score of the Leiden partition in the representation space.
    """
    sc.pp.neighbors(adata, n_neighbors=min(args.knn_neighbors, adata.n_obs - 1), use_rep=rep_key, n_pcs=pcs_for_neighbors, metric=METRIC, random_state=SEED)
    sc.tl.leiden(adata, resolution=args.leiden_resolution, key_added=key, flavor="igraph", random_state=SEED)
    rep = np.asarray(adata.obsm["X_pca"] if rep_key == "X_pca" else adata.X)
    for nc in tqdm(args.layout_dimensions, desc=f"{desc} UMAP", leave=False):
        sc.tl.umap(adata, n_components=nc, min_dist=args.umap_min_dist, random_state=SEED, key_added=f"X_umap_{nc}d")
    for nc, coords in run_tsne(rep, args.layout_dimensions).items():
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
def load_embeddings(npz_files: list[str], roles: dict[str, str], focus: dict[str, set[str]]) -> ad.AnnData:
    """Load the embeddings of every .npz into one AnnData, the species being the file label, with input type and focus tier.

    Args:
        npz_files (list[str]): ``<label>_reps.npz`` files of one model.
        roles (dict[str, str]): Input type per label.
        focus (dict[str, set[str]]): Focus accessions per target.

    Returns:
        ad.AnnData: Concatenated embeddings with ``protein_id``, ``species``, ``input_type`` and ``focus_tier`` observations.
    """
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
    adata.obs["input_type"] = adata.obs["species"].map(roles).astype(str)
    is_target = (adata.obs["input_type"] == "target").values
    shared = (~is_target) & adata.obs["protein_id"].astype(str).isin(set(adata.obs.loc[is_target, "protein_id"].astype(str))).values
    if shared.any():
        print(f"[PYTHON-INFO][FOCUS] Dropping {shared.sum():,} probe proteins already present in a target (kept once, as target)")
        adata = adata[~shared].copy()
    tier = np.where(adata.obs["input_type"] == "probe", "probe", "none").astype(object)
    for lbl, ids in focus.items():
        mask = ((adata.obs["species"] == lbl) & adata.obs["protein_id"].astype(str).isin(ids)).values
        tier[mask] = "annotated"
        print(f"[PYTHON-INFO][FOCUS] {lbl}: {mask.sum():,}/{len(ids):,} focus accessions present after dedup, tagged annotated")
    adata.obs["focus_tier"] = pd.Categorical(tier, categories=["probe", "annotated", "none"])
    print(f"[PYTHON-INFO][DATA-SHAPE] Initial data shape: {adata.shape}")
    return adata
def preprocess(adata: ad.AnnData, use_pca: bool) -> tuple[str, int | None]:
    """L2-normalise the embeddings in place and optionally reduce them to ``pca_components`` PCs.

    Args:
        adata (ad.AnnData): Embeddings to preprocess.
        use_pca (bool): Whether to run the PCA.

    Returns:
        tuple[str, int | None]: Representation key and number of PCs for the kNN graph.
    """
    norms = np.linalg.norm(adata.X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    adata.X = adata.X / norms
    sc.settings.seed = SEED
    if use_pca:
        sc.tl.pca(adata, n_comps=args.pca_components)
        print(f"[PYTHON-INFO][DATA-SHAPE] Shape after PCA ({args.pca_components} PCs): {adata.obsm['X_pca'].shape}")
        return "X_pca", args.pca_components
    print(f"[PYTHON-INFO][DATA-SHAPE] Operating in full-space dimensions: {adata.shape[1]}")
    return "X", None
def focus_similarity(adata: ad.AnnData) -> np.ndarray:
    """Cosine similarity of every protein to the normalised centroid of the focus set (probe and annotated proteins).

    Args:
        adata (ad.AnnData): L2-normalised embeddings with ``focus_tier``.

    Returns:
        np.ndarray: Similarity per protein (NaN when the focus set is empty).
    """
    mask = adata.obs["focus_tier"].isin(["probe", "annotated"]).values
    if not mask.any():
        print("[PYTHON-INFO][FOCUS] No focus proteins: focus_similarity left empty")
        return np.full(adata.n_obs, np.nan, dtype=np.float32)
    centroid = np.asarray(adata.X[mask].mean(axis=0)).ravel()
    centroid /= np.linalg.norm(centroid) or 1.0
    print(f"[PYTHON-INFO][FOCUS] Gravity centre from {mask.sum():,} focus proteins")
    return (np.asarray(adata.X) @ centroid).astype(np.float32)
def intra_clustering(adata: ad.AnnData, rep_key: str, pcs_for_neighbors: int | None, metrics: dict[str, float], markers: list[pd.DataFrame]) -> None:
    """Cluster and embed every target alone, writing ``intra_cluster`` and the ``*_intra`` layouts into the joint AnnData.

    Args:
        adata (ad.AnnData): Joint embeddings, modified in place.
        rep_key (str): Representation key of the kNN graph.
        pcs_for_neighbors (int | None): Number of PCs of the kNN graph.
        metrics (dict[str, float]): Silhouette scores, extended in place.
        markers (list[pd.DataFrame]): Channel marker tables, extended in place.
    """
    intra = pd.Series(index=adata.obs.index, dtype="object")
    for m in METHODS:
        for nc in args.layout_dimensions:
            adata.obsm[f"X_{m}_{nc}d_intra"] = np.full((adata.n_obs, nc), np.nan, dtype=np.float32)
    for sp in tqdm(list(adata.obs.loc[adata.obs["input_type"] == "target", "species"].unique()), desc="Intra-group clustering"):
        mask = (adata.obs["species"] == sp).values
        if mask.sum() < max(args.knn_neighbors + 1, 10):
            print(f"[PYTHON-INFO][INTRA] Skipping {sp}: too few points ({mask.sum()})")
            continue
        sub = ad.AnnData(X=adata.X[mask], obs=adata.obs.loc[mask, ["protein_id", "species"]].copy(), obsm={k: adata.obsm[k][mask] for k in ("X_pca",) if k in adata.obsm})
        metrics[f"intra_silhouette_{sp}"] = cluster_and_embed(sub, rep_key, pcs_for_neighbors, "intra_cluster", sp)
        print(f"[PYTHON-INFO][METRIC] Intra Silhouette ({sp}): {metrics[f'intra_silhouette_{sp}']:.4f}")
        intra[mask] = sub.obs["intra_cluster"].astype(str).values
        for m in METHODS:
            for nc in args.layout_dimensions:
                adata.obsm[f"X_{m}_{nc}d_intra"][mask] = sub.obsm[f"X_{m}_{nc}d"]
        markers.append(markers_table(sub, "intra_cluster", "intra_group", sp))
    adata.obs["intra_cluster"] = pd.Categorical(intra)
###################################### MAIN
def main() -> None:
    """Load embeddings, preprocess, cluster and embed jointly and per species, then save everything to .h5ad."""
    start_time = time.time()
    sc.settings.n_jobs = args.n_jobs
    npz_files = [p.strip() for p in args.npzs.split(",")]
    if len(npz_files) < 2:
        raise ValueError("[PYTHON-ERROR] This comparative script requires at least two .npz files.")
    use_pca = args.pca_components is not None and args.pca_components > 0
    param_tag = f"{Path(npz_files[0]).parent.name}__res{args.leiden_resolution}_nn{args.knn_neighbors}" + (f"_npc{args.pca_components}" if use_pca else "_full")
    target_dir = Path(args.scanpy_dir) / ("_".join([Path(f).stem.replace("_reps", "") for f in npz_files]) + f"__{Path(npz_files[0]).parent.name}")
    h5ad_path = target_dir / f"{target_dir.name}__{param_tag.removeprefix(Path(npz_files[0]).parent.name + '__')}.h5ad"
    roles, focus, focus_def = load_roles([Path(f).stem.replace("_reps", "") for f in npz_files])
    if "target" not in roles.values():
        raise ValueError("[PYTHON-ERROR] At least one target input is required.")
    focus_mode = "probe" in roles.values()
    if not focus_mode: focus = {}
    print(f"[PYTHON-INFO][MODE] {'focus (probe inputs present)' if focus_mode else 'comparative (no probe input): no focus tiers, no gravity'}")
    current = run_params(roles, focus, focus_def) if focus_mode else run_params()
    if not args.overwrite and is_fresh(h5ad_path, npz_files, current):
        print(f"[PYTHON-INFO] '{h5ad_path}' is newer than its inputs and has the current parameters. Skipping.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    adata = load_embeddings(npz_files, roles, focus)
    rep_key, pcs_for_neighbors = preprocess(adata, use_pca)
    adata.obs["focus_similarity"] = focus_similarity(adata) if focus_mode else np.nan
    space = "pca" if use_pca else "full"
    metrics = {f"joint_silhouette_{space}": cluster_and_embed(adata, rep_key, pcs_for_neighbors, "cluster", "Joint")}
    print(f"[PYTHON-INFO][METRIC] Joint Silhouette Score ({space} space): {metrics[f'joint_silhouette_{space}']:.4f}")
    markers = [markers_table(adata, "cluster", "inter_group", "all")]
    intra_clustering(adata, rep_key, pcs_for_neighbors, metrics, markers)
    out = ad.AnnData(obs=adata.obs[["protein_id", "species", "input_type", "focus_tier", "focus_similarity", "cluster", "intra_cluster"]].copy(), obsm={k: v for k, v in adata.obsm.items() if k.startswith(("X_umap", "X_tsne"))}, obsp=dict(adata.obsp), uns={"neighbors": adata.uns["neighbors"], "params": current, "npz_files": npz_files, "param_tag": param_tag, "metrics": metrics, "markers": pd.concat(markers, ignore_index=True)})
    out.write_h5ad(h5ad_path, compression="gzip")
    print(f"[PYTHON-INFO][SAVE] {h5ad_path} | obsm: {', '.join(out.obsm.keys())}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
