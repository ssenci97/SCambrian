#!/usr/bin/env python3
"""Processes ESMC protein embeddings with joint and intra-group clustering, UMAP panels, and module extraction."""

###################################### DEPENDENCIES_AND_CONFIG
import anndata as ad
import argparse
import datetime
import json
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import resource
import scanpy as sc
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.metrics import silhouette_score
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log2.*")

PARAMS = json.loads(Path("configs/param_config.json").read_text())
SCANPY_CFG = PARAMS.get("scanpy", {})

ALPHA_BG = 0.25
ALPHA_MAX = 0.95
ALPHA_MIN = 0.55
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
CBAR_LABEL_INTRA = "Intra-Group Embedding Density (KDE)"
COLOR_BG = "#E5E7EB"
COLORMAP_INTRA = sns.color_palette("magma", as_cmap=True)
COLORMAP_SPEC = sns.color_palette("viridis_r", as_cmap=True)
DEFAULT_OUTDIR = SCANPY_CFG.get("outdir", "data/scanpy/tmp/")
LEIDEN_RES = SCANPY_CFG.get("resolution", 1.0)
MIN_MODULE_SIZE = SCANPY_CFG.get("min_module_size", 100)
MIN_PERCENT_ENRICHMENT = SCANPY_CFG.get("min_percent_enrichment", 80)
N_NEIGHBORS = SCANPY_CFG.get("n_neighbors", 15)
N_PCS = SCANPY_CFG.get("n_pcs", None)
PANEL_DPI = 300
PANEL_SIZE_INCHES = 2.5
PRIVATE_THRESHOLD = 0.95
PT_SIZE_BG = 0.8
PT_SIZE_FG = 2.2
SEED = PARAMS.get("seed", 42)

###################################### MAIN
def main():
    parser = argparse.ArgumentParser(description="ESMC Proteome Embedding Clustering")
    parser.add_argument("--npzs", type=str, required=True, help="Comma-separated paths to .npz files")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Base output directory")
    parser.add_argument("--n_pcs", type=int, default=N_PCS, help="Number of PCs for dimensionality reduction")
    parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS, help="Number of neighbors for KNN")
    parser.add_argument("--resolution", type=float, default=LEIDEN_RES, help="Leiden cluster resolution")
    args = parser.parse_args()

    npz_files = [p.strip() for p in args.npzs.split(",")]
    target_dir = Path(args.outdir) / "_".join([Path(f).stem for f in npz_files])
    target_dir.mkdir(parents=True, exist_ok=True)
    modules_dir = target_dir / "protein-modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    intra_dir = target_dir / "intra_group"
    intra_dir.mkdir(parents=True, exist_ok=True)

    ###################################### LOAD_DATA
    start_time = time.time()
    adatas = []
    for f in tqdm(npz_files, desc="Loading embeddings"):
        data = np.load(f)
        k_emb = next(k for k in data.files if "embeddings" in k)
        k_ids = next(k for k in data.files if "ids" in k)
        species_name = Path(f).stem.replace("_reps", "")
        obs_index = [f"{species_name}_{pid}" for pid in data[k_ids]] if len(npz_files) > 1 else data[k_ids]
        adata = ad.AnnData(X=data[k_emb].astype(np.float32), obs=pd.DataFrame({"protein_id": data[k_ids], "species": species_name}, index=obs_index))
        adatas.append(adata)
    adata = ad.concat(adatas, join="outer") if len(adatas) > 1 else adatas[0]
    adata.obs_names_make_unique()
    print(f"[PYTHON-INFO][DATA-SHAPE] Initial data shape: {adata.shape}")

    ###################################### PREPROCESSING
    norms = np.linalg.norm(adata.X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    adata.X = adata.X / norms
    sc.settings.seed = SEED
    use_pca = args.n_pcs is not None and args.n_pcs > 0
    if use_pca:
        sc.tl.pca(adata, n_comps=args.n_pcs)
        print(f"[PYTHON-INFO][DATA-SHAPE] Shape after PCA (n_pcs={args.n_pcs}): {adata.obsm['X_pca'].shape}")
        rep_key, n_pcs_use = "X_pca", args.n_pcs
    else:
        print(f"[PYTHON-INFO][DATA-SHAPE] Operating in full-space dimensions: {adata.shape[1]}")
        rep_key, n_pcs_use = "X", None

    ###################################### JOINT_CLUSTERING
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine")
    sc.tl.leiden(adata, resolution=args.resolution, key_added="cluster", flavor="igraph")
    sc.tl.umap(adata)
    sil_space = adata.obsm["X_pca"] if use_pca else adata.X
    sil_score = silhouette_score(sil_space, adata.obs["cluster"], metric="cosine", random_state=SEED)
    print(f"[PYTHON-INFO][METRIC] Joint Silhouette Score ({'PCA' if use_pca else 'Full'} space): {sil_score:.4f}")
    with open(target_dir / "clustering_metrics.txt", "w") as f:
        f.write(f"joint_silhouette_score_{'pca' if use_pca else 'full'}: {sil_score}\n")

    ###################################### JOINT_MARKERS
    sc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon")
    marker_dfs = []
    groups = adata.uns["rank_genes_groups"]["names"].dtype.names
    for group in groups:
        df_group = pd.DataFrame({"cluster": group, "channel": adata.uns["rank_genes_groups"]["names"][group], "score": adata.uns["rank_genes_groups"]["scores"][group], "pvals_adj": adata.uns["rank_genes_groups"]["pvals_adj"][group]})
        marker_dfs.append(df_group)
    pd.concat(marker_dfs, ignore_index=True).to_csv(target_dir / "channel_importance_rankings.csv", index=False)

    ###################################### JOINT_SPECIES_PANEL
    species_list = list(adata.obs["species"].unique())
    umap_coords = adata.obsm["X_umap"]
    cluster_sp_frac = pd.crosstab(adata.obs["cluster"], adata.obs["species"], normalize="index")
    n_sp = len(species_list)
    n_cols = min(n_sp, 4)
    n_rows = (n_sp + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(PANEL_SIZE_INCHES * n_cols, PANEL_SIZE_INCHES * n_rows + 0.5), dpi=PANEL_DPI)
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.06], hspace=0.30, wspace=0.25)
    axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]
    for idx, sp in enumerate(species_list):
        ax = axes[idx]
        is_sp = (adata.obs["species"] == sp).values
        spec_scores = adata.obs["cluster"].map(cluster_sp_frac[sp]).values
        ax.scatter(umap_coords[~is_sp, 0], umap_coords[~is_sp, 1], c=COLOR_BG, s=PT_SIZE_BG, alpha=ALPHA_BG, rasterized=True, linewidths=0)
        fg_idx = np.where(is_sp)[0]
        sort_order = fg_idx[np.argsort(spec_scores[is_sp])]
        fg_scores = spec_scores[sort_order]
        fg_colors = COLORMAP_SPEC(fg_scores)
        fg_colors[:, 3] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * fg_scores
        ax.scatter(umap_coords[sort_order, 0], umap_coords[sort_order, 1], c=fg_colors, s=PT_SIZE_FG, rasterized=True, linewidths=0)
        sp_priv = cluster_sp_frac.index[cluster_sp_frac[sp] >= PRIVATE_THRESHOLD]
        sp_counts = adata.obs[adata.obs["species"] == sp]["cluster"].value_counts().loc[sp_priv].sort_values(ascending=False)
        sp_ranks = {cl: r + 1 for r, cl in enumerate(sp_counts.index)}
        for cl in sp_counts.index:
            cl_pts = umap_coords[(adata.obs["cluster"] == cl).values]
            if len(cl_pts) > 0:
                cx, cy = cl_pts[:, 0].mean(), cl_pts[:, 1].mean()
                txt = ax.text(cx + 1.0, cy, str(sp_ranks[cl]), fontsize=3, ha="center", va="center", fontweight="bold", color="black")
                txt.set_path_effects([path_effects.withStroke(linewidth=0.8, foreground="white")])
        ax.set_title(f"{sp} (n={is_sp.sum():,})", fontsize=7, fontweight="bold", pad=2)
        ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
        ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
        ax.tick_params(labelsize=5)
        ax.set_aspect("equal", "datalim")
    for idx in range(n_sp, len(axes)):
        fig.delaxes(axes[idx])
    cax = fig.add_subplot(gs[-1, :])
    sm = plt.cm.ScalarMappable(cmap=COLORMAP_SPEC, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(CBAR_LABEL, fontsize=7)
    cbar.ax.tick_params(labelsize=5)
    plt.savefig(target_dir / "umap_species_panel.png", dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)

    ###################################### JOINT_SAVE
    adata.obs[["protein_id", "species", "cluster"]].to_csv(target_dir / "protein_clusters_metadata.csv", index=False)
    counts_df = pd.crosstab(adata.obs["cluster"], adata.obs["species"])
    counts_df.to_csv(target_dir / "species_cluster_enrichment_count.csv")
    enrichment = cluster_sp_frac * 100
    enrichment.to_csv(target_dir / "species_cluster_enrichment_fraction.csv")
    for sp in species_list:
        for cl in counts_df.index:
            if enrichment.loc[cl, sp] > MIN_PERCENT_ENRICHMENT and counts_df.loc[cl, sp] >= MIN_MODULE_SIZE:
                numgenes = counts_df.loc[cl, sp]
                adata.obs[(adata.obs["species"] == sp) & (adata.obs["cluster"] == cl)]["protein_id"].to_csv(modules_dir / f"{sp}_{numgenes:07d}_clust{cl}_{MIN_PERCENT_ENRICHMENT}.txt", index=False, header=False)

    ###################################### INTRA_GROUP_CLUSTERING
    adata.obs["intra_cluster"] = pd.Series(index=adata.obs.index, dtype="object")
    intra_metrics = []
    intra_panel_data = []
    for sp in tqdm(species_list, desc="Intra-group clustering"):
        mask = adata.obs["species"] == sp
        if mask.sum() < max(args.n_neighbors + 1, 10):
            print(f"[PYTHON-INFO][INTRA] Skipping {sp}: too few points ({mask.sum()})")
            continue
        adata_sp = adata[mask].copy()
        sc.pp.neighbors(adata_sp, n_neighbors=min(args.n_neighbors, mask.sum() - 1), use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine")
        sc.tl.leiden(adata_sp, resolution=args.resolution, key_added="intra_cluster", flavor="igraph")
        sc.tl.umap(adata_sp)
        sil_sp = silhouette_score(adata_sp.obsm["X_pca"] if use_pca else adata_sp.X, adata_sp.obs["intra_cluster"], metric="cosine", random_state=SEED)
        print(f"[PYTHON-INFO][METRIC] Intra Silhouette ({sp}): {sil_sp:.4f}")
        intra_metrics.append(f"intra_silhouette_{sp}: {sil_sp}")
        adata.obs.loc[mask, "intra_cluster"] = adata_sp.obs["intra_cluster"].astype(str).values
        n_cl = adata_sp.obs["intra_cluster"].nunique()
        coords = adata_sp.obsm["X_umap"]
        density = gaussian_kde(coords.T)(coords.T) if coords.shape[0] > 1 else np.ones(coords.shape[0])
        counts_intra = adata_sp.obs["intra_cluster"].value_counts()
        intra_ranks = {cl: r + 1 for r, cl in enumerate(counts_intra.index)}
        top30_cls = list(counts_intra.head(30).index)
        intra_panel_data.append((sp, mask.sum(), n_cl, coords, density, adata_sp.obs["intra_cluster"].values, top30_cls, intra_ranks))
        for cl, cnt in counts_intra.items():
            if cnt >= MIN_MODULE_SIZE:
                adata_sp.obs[adata_sp.obs["intra_cluster"] == cl]["protein_id"].to_csv(modules_dir / f"intra_{sp}_{cnt:07d}_clust{cl}.txt", index=False, header=False)
        adata_sp.obs[["protein_id", "intra_cluster"]].to_csv(intra_dir / f"protein_intra_clusters_{sp}.csv", index=False)
        sc.tl.rank_genes_groups(adata_sp, groupby="intra_cluster", method="wilcoxon")
        marker_dfs_sp = []
        groups_sp = adata_sp.uns["rank_genes_groups"]["names"].dtype.names
        for group in groups_sp:
            df_g = pd.DataFrame({"cluster": group, "channel": adata_sp.uns["rank_genes_groups"]["names"][group], "score": adata_sp.uns["rank_genes_groups"]["scores"][group], "pvals_adj": adata_sp.uns["rank_genes_groups"]["pvals_adj"][group]})
            marker_dfs_sp.append(df_g)
        pd.concat(marker_dfs_sp, ignore_index=True).to_csv(intra_dir / f"channel_importance_intra_{sp}.csv", index=False)

    n_intra = len(intra_panel_data)
    if n_intra > 0:
        n_cols_i = min(n_intra, 4)
        n_rows_i = (n_intra + n_cols_i - 1) // n_cols_i
        fig = plt.figure(figsize=(PANEL_SIZE_INCHES * n_cols_i, PANEL_SIZE_INCHES * n_rows_i + 0.5), dpi=PANEL_DPI)
        gs = fig.add_gridspec(n_rows_i + 1, n_cols_i, height_ratios=[1] * n_rows_i + [0.06], hspace=0.30, wspace=0.25)
        axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows_i) for c in range(n_cols_i)]
        scat = None
        for idx, (sp, n_pts, n_cl, coords, density, intra_cls, top30_cls, intra_ranks) in enumerate(intra_panel_data):
            ax = axes[idx]
            sort_idx = np.argsort(density)
            scat = ax.scatter(coords[sort_idx, 0], coords[sort_idx, 1], c=density[sort_idx], cmap=COLORMAP_INTRA, s=PT_SIZE_FG, rasterized=True, linewidths=0)
            for cl in top30_cls:
                cl_pts = coords[intra_cls == cl]
                if len(cl_pts) > 0:
                    cx, cy = cl_pts[:, 0].mean(), cl_pts[:, 1].mean()
                    txt = ax.text(cx + 1.0, cy, str(intra_ranks[cl]), fontsize=3, ha="center", va="center", fontweight="bold", color="black")
                    txt.set_path_effects([path_effects.withStroke(linewidth=0.8, foreground="white")])
            ax.set_title(f"{sp} intra (n={n_pts:,}, k={n_cl})", fontsize=7, fontweight="bold", pad=2)
            ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
            ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
            ax.tick_params(labelsize=5)
            ax.set_aspect("equal", "datalim")
        for idx in range(n_intra, len(axes)):
            fig.delaxes(axes[idx])
        cax = fig.add_subplot(gs[-1, :])
        cbar = fig.colorbar(scat, cax=cax, orientation="horizontal")
        cbar.set_label(CBAR_LABEL_INTRA, fontsize=7)
        cbar.ax.tick_params(labelsize=5)
        plt.savefig(intra_dir / "umap_intra_species_panel.png", dpi=PANEL_DPI, bbox_inches="tight")
        plt.close(fig)

    ###################################### LOGGING
    with open(target_dir / "clustering_metrics.txt", "a") as f:
        for line in intra_metrics:
            f.write(line + "\n")
    adata.obs[["protein_id", "species", "cluster", "intra_cluster"]].to_csv(target_dir / "protein_clusters_metadata.csv", index=False)

    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")

if __name__ == "__main__":
    main()