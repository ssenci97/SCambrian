#!/usr/bin/env python3
"""Jointly cluster ESMC protein embeddings of several groups with Scanpy, plot species panels and extract protein modules.

Outputs per run (tagged by resolution / neighbors / PCA setting) under ``<outdir>/<inputs>/``:
cluster metadata, species enrichment table, channel rankings, clustering metrics,
inter- and intra-group UMAP panels and ``protein-modules/`` ID lists.
"""
############### <imports>
import argparse
import sys
import time
import warnings
from pathlib import Path
import anndata as ad
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from utils import add_yaml_args, fasta_label, log, log_resources, resolve_params
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log2.*")

############### <constants>
SCRIPT_NAME = Path(__file__).stem
DEFAULTS = {"seed": 42, "resolution": 1.0, "n_neighbors": 15, "n_pcs": None, "min_module_size": 100, "min_percent_enrichment": 80, "outdir": "data/scanpy/"}
ALPHA_BG = 0.50
ALPHA_MAX = 0.95
ALPHA_MIN = 0.55
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
COLOR_BG = "#BFC5CE"
COLORMAP_SPEC = sns.color_palette("plasma", as_cmap=True)
COLORMAP_INTRA = sns.color_palette("magma", as_cmap=True)
MAX_INTRA_LABELS = 10
MAX_JOINT_LABELS = 10
PANEL_DPI = 300
PANEL_SIZE_INCHES = 3.0
PT_SIZE = 1.0


############### <helpers>
def load_adata(npz_files: list[str]) -> ad.AnnData:
    """Load embedding files into one AnnData with ``protein_id`` and ``species`` annotations.

    Args:
        npz_files: ``.npz`` files with ``embeddings`` and ``ids`` arrays; the label is the file stem minus ``_reps``.

    Returns:
        Concatenated AnnData.
    """
    adatas = []
    for f in tqdm(npz_files, desc="[PYTHON-INFO] Loading embeddings"):
        data = np.load(f)
        k_emb = next(k for k in data.files if "embeddings" in k)
        k_ids = next(k for k in data.files if "ids" in k)
        species = fasta_label(f)
        obs_index = [f"{species}_{pid}" for pid in data[k_ids]]
        adatas.append(ad.AnnData(X=data[k_emb].astype(np.float32), obs=pd.DataFrame({"protein_id": data[k_ids], "species": species}, index=obs_index)))
    adata = ad.concat(adatas, join="outer")
    adata.obs_names_make_unique()
    return adata


def rank_markers(adata: ad.AnnData, groupby: str, analysis: str, species: str) -> pd.DataFrame:
    """Rank embedding channels per cluster with a Wilcoxon test.

    Args:
        adata: Clustered AnnData.
        groupby: Cluster column in ``adata.obs``.
        analysis: Value of the ``analysis`` output column.
        species: Value of the ``species`` output column.

    Returns:
        Table with ``analysis``, ``species``, ``cluster``, ``channel``, ``score``, ``pvals_adj``.
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")
    rgg = adata.uns["rank_genes_groups"]
    df = pd.concat([pd.DataFrame({"cluster": g, "channel": rgg["names"][g], "score": rgg["scores"][g], "pvals_adj": rgg["pvals_adj"][g]}) for g in rgg["names"].dtype.names], ignore_index=True)
    df.insert(0, "analysis", analysis)
    df.insert(1, "species", species)
    return df


def style_axis(ax: plt.Axes, title: str) -> None:
    """Apply the shared UMAP panel axis style.

    Args:
        ax: Axis to style.
        title: Panel title.
    """
    ax.set_title(title, fontsize=7, fontweight="bold", pad=2)
    ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.set_aspect("equal", "datalim")


############### <joint-panel>
def plot_joint_panel(adata: ad.AnnData, species_list: list[str], cluster_sp_frac: pd.DataFrame, out_png: Path) -> None:
    """Plot one joint-UMAP panel per species colored by cluster specificity.

    Args:
        adata: AnnData with ``X_umap``, ``cluster`` and ``species``.
        species_list: Species to plot.
        cluster_sp_frac: Cluster-by-species fraction table (rows sum to 1).
        out_png: Output image.
    """
    umap_coords = adata.obsm["X_umap"]
    n_sp = len(species_list)
    n_cols = min(n_sp, 4)
    n_rows = (n_sp + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(PANEL_SIZE_INCHES * n_cols, PANEL_SIZE_INCHES * n_rows + 0.4), dpi=PANEL_DPI)
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.04], hspace=0.30, wspace=0.25)
    axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]
    for idx, sp in enumerate(species_list):
        ax = axes[idx]
        is_sp = (adata.obs["species"] == sp).values
        spec_scores = adata.obs["cluster"].map(cluster_sp_frac[sp]).values.astype(float)
        ax.scatter(umap_coords[~is_sp, 0], umap_coords[~is_sp, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0)
        fg_idx = np.where(is_sp)[0]
        sort_order = fg_idx[np.argsort(spec_scores[is_sp])]
        fg_scores = spec_scores[sort_order]
        fg_colors = COLORMAP_SPEC(fg_scores)
        fg_colors[:, 3] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * fg_scores
        ax.scatter(umap_coords[sort_order, 0], umap_coords[sort_order, 1], c=fg_colors, s=PT_SIZE, rasterized=True, linewidths=0)
        sp_counts = adata.obs.loc[is_sp, "cluster"].value_counts()
        sp_counts_top = sp_counts.sort_values(ascending=False).head(MAX_JOINT_LABELS)
        max_cnt_sp = sp_counts_top.max() if len(sp_counts_top) > 0 else 1
        for rank, cl in enumerate(sp_counts_top.index, start=1):
            cl_pts = umap_coords[(adata.obs["cluster"] == cl).values]
            if len(cl_pts) > 0:
                f_size = float(np.clip(3.5 + 5.5 * (sp_counts[cl] / max_cnt_sp), 3.5, 9.0))
                txt = ax.text(cl_pts[:, 0].mean() + 1.0, cl_pts[:, 1].mean(), str(rank), fontsize=f_size, ha="center", va="center", fontweight="bold", color="black")
                txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
        style_axis(ax, f"{sp} (n={is_sp.sum():,})")
    for idx in range(n_sp, len(axes)):
        fig.delaxes(axes[idx])
    cax = fig.add_subplot(gs[-1, :])
    sm = plt.cm.ScalarMappable(cmap=COLORMAP_SPEC, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(CBAR_LABEL, fontsize=6)
    cbar.ax.tick_params(labelsize=4)
    plt.savefig(out_png, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)


############### <intra-panel>
def plot_intra_panel(intra_panel_data: list[tuple], out_png: Path) -> None:
    """Plot per-species UMAPs highlighting the largest intra-group clusters.

    Args:
        intra_panel_data: Tuples ``(species, n_points, n_clusters, coords, labels, top_clusters, counts)``.
        out_png: Output image.
    """
    n_intra = len(intra_panel_data)
    n_cols = min(n_intra, 2)
    n_rows = (n_intra + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(PANEL_SIZE_INCHES * n_cols, PANEL_SIZE_INCHES * n_rows), dpi=PANEL_DPI, squeeze=False)
    axes = axes_grid.ravel()
    all_top_counts = [int(counts[cl]) for *_, top_cls, counts in intra_panel_data for cl in top_cls]
    count_min = float(min(all_top_counts)) if all_top_counts else 0.0
    count_max = float(max(all_top_counts)) if all_top_counts else 1.0
    count_norm = plt.Normalize(vmin=count_min, vmax=(count_max if count_max > count_min else count_min + 1.0))
    for idx, (sp, _, n_cl, coords, intra_cls, top_cls, counts) in enumerate(intra_panel_data):
        ax = axes[idx]
        is_top = np.isin(intra_cls, top_cls)
        ax.scatter(coords[~is_top, 0], coords[~is_top, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
        for rank, cl in enumerate(top_cls, start=1):
            cl_mask = intra_cls == cl
            if not cl_mask.any():
                continue
            cl_pts = coords[cl_mask]
            ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=[COLORMAP_INTRA(count_norm(int(counts[cl])))], s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
            txt = ax.text(np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1]), str(rank), fontsize=5.5, ha="center", va="center", fontweight="bold", color="black", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
        style_axis(ax, f"{sp} (Top {MAX_INTRA_LABELS} Clusters, k={n_cl})")
    for idx in range(n_intra, len(axes)):
        fig.delaxes(axes[idx])
    sm = plt.cm.ScalarMappable(cmap=COLORMAP_INTRA, norm=count_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes[:n_intra]), orientation="horizontal", fraction=0.035, pad=0.08, aspect=35)
    cbar.set_label("Proteins per highlighted cluster", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    fig.savefig(out_png, dpi=PANEL_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


############### <main>
def main() -> None:
    """Parse arguments, resolve parameters and run joint plus intra-group clustering."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npzs", help="Comma-separated paths to at least two .npz files")
    parser.add_argument("--outdir", help="Base output directory")
    parser.add_argument("--n_pcs", type=int, help="Number of PCs (unset or 0 = full space)")
    parser.add_argument("--n_neighbors", type=int, help="Number of neighbors for KNN")
    parser.add_argument("--resolution", type=float, help="Leiden cluster resolution")
    parser.add_argument("--min_module_size", type=int)
    parser.add_argument("--min_percent_enrichment", type=float)
    parser.add_argument("--seed", type=int)
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    if not args.npzs:
        if args.yaml_out:
            sys.exit(0)
        parser.error("--npzs is required")
    npz_files = [p.strip() for p in args.npzs.split(",") if p.strip()]
    if len(npz_files) < 2:
        log("Error: this comparative script requires at least two .npz files.")
        raise ValueError("This comparative script requires at least two .npz files.")
    seed, n_neighbors, resolution, n_pcs = params["seed"], params["n_neighbors"], params["resolution"], params["n_pcs"]
    min_module_size, min_percent_enrichment = params["min_module_size"], params["min_percent_enrichment"]
    use_pca = bool(n_pcs) and n_pcs > 0
    param_tag = f"res{resolution}_nn{n_neighbors}" + (f"_npc{n_pcs}" if use_pca else "_full")
    target_dir = Path(params["outdir"]) / "_".join(Path(f).stem for f in npz_files)
    modules_dir, intra_dir = target_dir / "protein-modules", target_dir / "intra_group"
    modules_dir.mkdir(parents=True, exist_ok=True)
    intra_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = target_dir / f"clustering_metrics__{param_tag}.txt"

    ############### <preprocessing>
    adata = load_adata(npz_files)
    log(f"Initial data shape: {adata.shape}")
    norms = np.linalg.norm(adata.X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    adata.X = adata.X / norms
    sc.settings.seed = seed
    if use_pca:
        sc.tl.pca(adata, n_comps=n_pcs)
        log(f"Shape after PCA (n_pcs={n_pcs}): {adata.obsm['X_pca'].shape}")
        rep_key, n_pcs_use = "X_pca", n_pcs
    else:
        log(f"Operating in full-space dimensions: {adata.shape[1]}")
        rep_key, n_pcs_use = "X", None

    ############### <joint-clustering>
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine", random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, key_added="cluster", flavor="igraph", random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    sil_score = silhouette_score(adata.obsm["X_pca"] if use_pca else adata.X, adata.obs["cluster"], metric="cosine", random_state=seed)
    log(f"Joint Silhouette Score ({'PCA' if use_pca else 'Full'} space): {sil_score:.4f}")
    metrics_lines = [f"joint_silhouette_score_{'pca' if use_pca else 'full'}: {sil_score}"]
    joint_marker_df = rank_markers(adata, "cluster", "inter_group", "all")
    species_list = list(adata.obs["species"].unique())
    cluster_sp_frac = pd.crosstab(adata.obs["cluster"], adata.obs["species"], normalize="index")
    plot_joint_panel(adata, species_list, cluster_sp_frac, target_dir / f"umap_species_panel__{param_tag}.png")

    ############### <joint-modules>
    counts_df = pd.crosstab(adata.obs["cluster"], adata.obs["species"])
    enrichment = cluster_sp_frac * 100
    count_table = counts_df.stack().rename("protein_count").reset_index()
    count_table.columns = ["cluster", "species", "protein_count"]
    frac_table = enrichment.stack().rename("species_fraction_percent").reset_index()
    frac_table.columns = ["cluster", "species", "species_fraction_percent"]
    count_table.merge(frac_table, on=["cluster", "species"], how="left").to_csv(target_dir / f"species_cluster_enrichment__{param_tag}.csv", index=False)
    for sp in species_list:
        for cl in counts_df.index:
            n_prot = counts_df.loc[cl, sp]
            if enrichment.loc[cl, sp] > min_percent_enrichment and n_prot >= min_module_size:
                adata.obs[(adata.obs["species"] == sp) & (adata.obs["cluster"] == cl)]["protein_id"].to_csv(modules_dir / f"{sp}_{n_prot:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)

    ############### <intra-group-clustering>
    adata.obs["intra_cluster"] = pd.Series(index=adata.obs.index, dtype="object")
    intra_panel_data, intra_marker_dfs = [], []
    for sp in tqdm(species_list, desc="[PYTHON-INFO] Intra-group clustering"):
        mask = adata.obs["species"] == sp
        if mask.sum() < max(n_neighbors + 1, 10):
            log(f"Skipping intra-group clustering of {sp}: too few points ({mask.sum()})")
            continue
        adata_sp = adata[mask].copy()
        sc.pp.neighbors(adata_sp, n_neighbors=min(n_neighbors, mask.sum() - 1), use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine", random_state=seed)
        sc.tl.leiden(adata_sp, resolution=resolution, key_added="intra_cluster", flavor="igraph", random_state=seed)
        sc.tl.umap(adata_sp, random_state=seed)
        sil_sp = silhouette_score(adata_sp.obsm["X_pca"] if use_pca else adata_sp.X, adata_sp.obs["intra_cluster"], metric="cosine", random_state=seed)
        log(f"Intra Silhouette ({sp}): {sil_sp:.4f}")
        metrics_lines.append(f"intra_silhouette_{sp}: {sil_sp}")
        adata.obs.loc[mask, "intra_cluster"] = adata_sp.obs["intra_cluster"].astype(str).values
        counts_intra = adata_sp.obs["intra_cluster"].value_counts()
        intra_panel_data.append((sp, mask.sum(), adata_sp.obs["intra_cluster"].nunique(), adata_sp.obsm["X_umap"], adata_sp.obs["intra_cluster"].values, list(counts_intra.head(MAX_INTRA_LABELS).index), counts_intra))
        for cl, cnt in counts_intra.items():
            if cnt >= min_module_size:
                adata_sp.obs[adata_sp.obs["intra_cluster"] == cl]["protein_id"].to_csv(modules_dir / f"intra_{sp}_{cnt:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
        intra_marker_dfs.append(rank_markers(adata_sp, "intra_cluster", "intra_group", sp))
    if intra_panel_data:
        plot_intra_panel(intra_panel_data, intra_dir / f"umap_intra_species_panel__{param_tag}.png")

    ############### <save>
    metrics_path.write_text("\n".join(metrics_lines) + "\n")
    adata.obs[["protein_id", "species", "cluster", "intra_cluster"]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    pd.concat([joint_marker_df] + intra_marker_dfs, ignore_index=True).to_csv(target_dir / f"channel_importance_rankings__{param_tag}.csv", index=False)
    log(f"Scanpy outputs written to {target_dir}")
    log_resources(start_time)


if __name__ == "__main__":
    main()
