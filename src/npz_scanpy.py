#!/usr/bin/env python3
"""Processes ESMC protein embeddings with joint clustering, globally lettered species-specificity UMAP panels, intra-species clustering and module extraction.

A first pass runs the full analysis and assigns unique letters to the most species-specific joint clusters; a second pass recomputes neighbors, Leiden and UMAP on the lettered clusters only and compares first- and second-pass memberships per species with Venn diagrams.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import datetime
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
from pathlib import Path
import resource
import scanpy as sc
from sklearn.metrics import silhouette_score
import time
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log2.*")
ALPHA_BG = 0.50
ALPHA_MAX = 0.95
ALPHA_MIN = 0.55
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
COLOR_BG = "#BFC5CE"
COLORMAP_SPEC = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
COLORMAP_INTRA = COLORMAP_SPEC
DEFAULT_OUTDIR = "data/scanpy/"
LEIDEN_RES = 1.0
MAX_INTRA_LABELS = 10
MAX_LETTERED = 8
MAX_SPECIFIC_LABELS = 4
MIN_MODULE_SIZE = 100
PALETTE = ["#ECDE90", "#93DEB6", "#F1B584", "#76BBE2", "#DA919B", "#899E64", "#8E81B9", "#A66C58", "#30767A", "#7A5271"]
MIN_PERCENT_ENRICHMENT = 80
N_NEIGHBORS = 15
N_PCS = None
PANEL_DPI = 300
PANEL_SIZE_INCHES = 3.0
PT_SIZE = 1.0
SEED = 42
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="ESMC Proteome Embedding Clustering")
parser.add_argument("--npzs", type=str, required=True, help="Comma-separated paths to .npz files")
parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Base output directory")
parser.add_argument("--n_pcs", type=int, default=N_PCS, help="Number of PCs for dimensionality reduction")
parser.add_argument("--n_neighbors", type=int, default=N_NEIGHBORS, help="Number of neighbors for KNN")
parser.add_argument("--resolution", type=float, default=LEIDEN_RES, help="Leiden cluster resolution")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
yargs, _ = parser.parse_known_args()
script_key = "npz_scanpy"
params = dict(seed=SEED, resolution=LEIDEN_RES, n_neighbors=N_NEIGHBORS, min_module_size=MIN_MODULE_SIZE, min_percent_enrichment=MIN_PERCENT_ENRICHMENT, outdir=DEFAULT_OUTDIR)
if yargs.yaml_in and Path(yargs.yaml_in).exists():
    ydata = yaml.safe_load(Path(yargs.yaml_in).read_text()) or {}
    if script_key in ydata: params.update(ydata[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    ydata = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    ydata[script_key] = params
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(ydata, sort_keys=False))
###################################### HELPER_FUNCTIONS
def letter_code(i: int) -> str:
    """Convert a zero-based index to a lowercase letter code (a, ..., z, aa, ab, ...).

    Args:
        i (int): Zero-based index.

    Returns:
        str: Letter code.
    """
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s
def assign_letters(counts_df: pd.DataFrame, cluster_sp_frac: pd.DataFrame, species_list: list[str]) -> dict[str, str]:
    """Assign unique global letters to the top species-specific joint clusters of every species.

    Args:
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).
        cluster_sp_frac (pd.DataFrame): Species fraction per cluster (rows) and species (columns).
        species_list (list[str]): Species to consider.

    Returns:
        dict[str, str]: Mapping cluster -> letter, ordered by decreasing specificity and size; no letter is shared by two clusters.
    """
    chosen = set()
    for sp in species_list:
        cand = pd.DataFrame({"spec": cluster_sp_frac[sp], "count": counts_df[sp]})
        chosen |= set(cand[cand["count"] >= MIN_MODULE_SIZE].sort_values(["spec", "count"], ascending=False).head(MAX_LETTERED).index)
    rank = pd.DataFrame({"spec": cluster_sp_frac.max(axis=1), "count": counts_df.sum(axis=1)}).loc[sorted(chosen)].sort_values(["spec", "count"], ascending=False)
    return {cl: letter_code(i) for i, cl in enumerate(rank.index)}
def facet_letters(letters: dict[str, str], counts_df: pd.DataFrame, cluster_sp_frac: pd.DataFrame, sp: str) -> dict[str, str]:
    """Select the lettered clusters shown in a species facet.

    Args:
        letters (dict[str, str]): Global mapping cluster -> letter.
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).
        cluster_sp_frac (pd.DataFrame): Species fraction per cluster (rows) and species (columns).
        sp (str): Species of the facet.

    Returns:
        dict[str, str]: Up to MAX_SPECIFIC_LABELS clusters -> letter, the most specific for ``sp``, sorted by letter.
    """
    cand = pd.DataFrame({"spec": cluster_sp_frac[sp], "count": counts_df[sp]}).loc[list(letters)]
    cand = cand[cand["count"] >= MIN_MODULE_SIZE].sort_values(["spec", "count"], ascending=False).head(MAX_SPECIFIC_LABELS)
    return dict(sorted(((cl, letters[cl]) for cl in cand.index), key=lambda kv: (len(kv[1]), kv[1])))
def draw_facet(ax: plt.Axes, coords: np.ndarray, obs: pd.DataFrame, cluster_key: str, sp: str, sp_frac: pd.Series, facet: dict[str, str], sp_counts: pd.Series) -> None:
    """Draw one species facet coloured by cluster specificity with letter labels at cluster centroids.

    Args:
        ax (plt.Axes): Target axes.
        coords (np.ndarray): UMAP coordinates of all points.
        obs (pd.DataFrame): Observation table with ``species`` and ``cluster_key`` columns.
        cluster_key (str): Column holding the lettered clusters.
        sp (str): Species of the facet.
        sp_frac (pd.Series): Fraction of ``sp`` per cluster.
        facet (dict[str, str]): Clusters -> letters to label.
        sp_counts (pd.Series): Proteins of ``sp`` per cluster.
    """
    is_sp = (obs["species"] == sp).values
    spec_scores = obs[cluster_key].map(sp_frac).astype(float).values
    ax.scatter(coords[~is_sp, 0], coords[~is_sp, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0)
    fg_idx = np.where(is_sp)[0]
    sort_order = fg_idx[np.argsort(spec_scores[is_sp])]
    fg_scores = spec_scores[sort_order]
    fg_colors = COLORMAP_SPEC(fg_scores)
    fg_colors[:, 3] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * fg_scores
    ax.scatter(coords[sort_order, 0], coords[sort_order, 1], c=fg_colors, s=PT_SIZE, rasterized=True, linewidths=0)
    max_cnt_sp = max((sp_counts[cl] for cl in facet), default=1)
    for cl, letter in facet.items():
        cl_pts = coords[(obs[cluster_key] == cl).values]
        if len(cl_pts) == 0: continue
        f_size = float(np.clip(3.5 + 5.5 * (sp_counts[cl] / max_cnt_sp), 3.5, 9.0))
        txt = ax.text(cl_pts[:, 0].mean() + 1.0, cl_pts[:, 1].mean(), letter, fontsize=f_size, ha="center", va="center", fontweight="bold", color="black")
        txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="white")])
    ax.set_title(f"{sp} (n={is_sp.sum():,})", fontsize=7, fontweight="bold", pad=2)
    ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.set_aspect("equal", "datalim")
def add_spec_cbar(fig: plt.Figure, cax: plt.Axes) -> None:
    """Add the horizontal cluster-specificity colorbar.

    Args:
        fig (plt.Figure): Target figure.
        cax (plt.Axes): Colorbar axes.
    """
    sm = plt.cm.ScalarMappable(cmap=COLORMAP_SPEC, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(CBAR_LABEL, fontsize=6)
    cbar.ax.tick_params(labelsize=4)
def run_pass(adata: ad.AnnData, target_dir: Path, param_tag: str, rep_key: str, n_pcs_use: int | None, use_pca: bool) -> tuple[dict[str, str], dict[str, dict[str, str]], pd.DataFrame, pd.DataFrame]:
    """Run joint clustering, lettered species panel, module extraction, intra-group clustering and logging.

    Args:
        adata (ad.AnnData): Normalised embeddings.
        target_dir (Path): Output directory for this pass.
        param_tag (str): Parameter tag appended to output file names.
        rep_key (str): Representation used for neighbors (``X`` or ``X_pca``).
        n_pcs_use (int | None): Number of PCs passed to neighbors.
        use_pca (bool): Whether the PCA space is used for silhouette scores.

    Returns:
        tuple[dict[str, str], dict[str, dict[str, str]], pd.DataFrame, pd.DataFrame]: Global letters, facet letters per species, species fractions and counts per cluster.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    modules_dir = target_dir / "protein-modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    intra_dir = target_dir / "intra_group"
    intra_dir.mkdir(parents=True, exist_ok=True)
    ###################################### JOINT_CLUSTERING
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine")
    sc.tl.leiden(adata, resolution=args.resolution, key_added="cluster", flavor="igraph")
    sc.tl.umap(adata)
    sil_space = adata.obsm["X_pca"] if use_pca else adata.X
    sil_score = silhouette_score(sil_space, adata.obs["cluster"], metric="cosine", random_state=SEED)
    print(f"[PYTHON-INFO][METRIC] Joint Silhouette Score ({'PCA' if use_pca else 'Full'} space): {sil_score:.4f}")
    with open(target_dir / f"clustering_metrics__{param_tag}.txt", "w") as f:
        f.write(f"joint_silhouette_score_{'pca' if use_pca else 'full'}: {sil_score}\n")
    ###################################### JOINT_MARKERS
    sc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon")
    marker_dfs = []
    groups = adata.uns["rank_genes_groups"]["names"].dtype.names
    for group in groups:
        df_group = pd.DataFrame({"cluster": group, "channel": adata.uns["rank_genes_groups"]["names"][group], "score": adata.uns["rank_genes_groups"]["scores"][group], "pvals_adj": adata.uns["rank_genes_groups"]["pvals_adj"][group]})
        marker_dfs.append(df_group)
    joint_marker_df = pd.concat(marker_dfs, ignore_index=True)
    joint_marker_df.insert(0, "analysis", "inter_group")
    joint_marker_df.insert(1, "species", "all")
    ###################################### JOINT_SPECIES_PANEL
    species_list = list(adata.obs["species"].unique())
    umap_coords = adata.obsm["X_umap"]
    cluster_sp_frac = pd.crosstab(adata.obs["cluster"], adata.obs["species"], normalize="index")
    counts_df = pd.crosstab(adata.obs["cluster"], adata.obs["species"])
    letters = assign_letters(counts_df, cluster_sp_frac, species_list)
    facets = {sp: facet_letters(letters, counts_df, cluster_sp_frac, sp) for sp in species_list}
    print(f"[PYTHON-INFO][LETTERS] {len(letters)} lettered clusters: {letters}")
    n_sp = len(species_list)
    n_cols = min(n_sp, 4)
    n_rows = (n_sp + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(PANEL_SIZE_INCHES * n_cols, PANEL_SIZE_INCHES * n_rows + 0.4), dpi=PANEL_DPI)
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.04], hspace=0.30, wspace=0.25)
    axes = [fig.add_subplot(gs[r, c]) for r in range(n_rows) for c in range(n_cols)]
    for idx, sp in enumerate(species_list):
        draw_facet(axes[idx], umap_coords, adata.obs, "cluster", sp, cluster_sp_frac[sp], facets[sp], counts_df[sp])
    for idx in range(n_sp, len(axes)):
        fig.delaxes(axes[idx])
    add_spec_cbar(fig, fig.add_subplot(gs[-1, :]))
    plt.savefig(target_dir / f"umap_species_panel__{param_tag}.png", dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)
    ###################################### JOINT_SAVE
    letter_rows = [(sp, cl, lt, counts_df.loc[cl, sp], cluster_sp_frac.loc[cl, sp]) for sp, sp_letters in facets.items() for cl, lt in sp_letters.items()]
    pd.DataFrame(letter_rows, columns=["species", "cluster", "letter", "protein_count", "specificity"]).to_csv(target_dir / f"cluster_letters__{param_tag}.csv", index=False)
    adata.obs[["protein_id", "species", "cluster"]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    enrichment = cluster_sp_frac * 100
    enrichment_table = counts_df.stack().rename("protein_count").reset_index()
    enrichment_table.columns = ["cluster", "species", "protein_count"]
    frac_table = enrichment.stack().rename("species_fraction_percent").reset_index()
    frac_table.columns = ["cluster", "species", "species_fraction_percent"]
    cluster_enrichment_table = enrichment_table.merge(frac_table, on=["cluster", "species"], how="left")
    cluster_enrichment_table.to_csv(target_dir / f"species_cluster_enrichment__{param_tag}.csv", index=False)
    for sp in species_list:
        for cl in counts_df.index:
            if enrichment.loc[cl, sp] > MIN_PERCENT_ENRICHMENT and counts_df.loc[cl, sp] >= MIN_MODULE_SIZE:
                numgenes = counts_df.loc[cl, sp]
                adata.obs[(adata.obs["species"] == sp) & (adata.obs["cluster"] == cl)]["protein_id"].to_csv(modules_dir / f"{sp}_{numgenes:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    ###################################### INTRA_GROUP_CLUSTERING
    adata.obs["intra_cluster"] = pd.Series(index=adata.obs.index, dtype="object")
    intra_metrics = []
    intra_panel_data = []
    intra_marker_dfs = []
    for sp in tqdm(species_list, desc="Intra-group clustering"):
        mask = adata.obs["species"] == sp
        if mask.sum() < max(args.n_neighbors + 1, 10):
            print(f"[PYTHON-INFO][INTRA] Skipping {sp}: too few points ({mask.sum()})")
            continue
        adata_sp = adata[mask].copy()
        sc.pp.neighbors(adata_sp, n_neighbors=min(args.n_neighbors, mask.sum() - 1), use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine")
        sc.tl.leiden(adata_sp, resolution=args.resolution, key_added="intra_cluster", flavor="igraph")
        sc.tl.umap(adata_sp)
        sil_space_sp = adata_sp.obsm["X_pca"] if use_pca else adata_sp.X
        sil_sp = silhouette_score(sil_space_sp, adata_sp.obs["intra_cluster"], metric="cosine", random_state=SEED)
        print(f"[PYTHON-INFO][METRIC] Intra Silhouette ({sp}): {sil_sp:.4f}")
        intra_metrics.append(f"intra_silhouette_{sp}: {sil_sp}")
        adata.obs.loc[mask, "intra_cluster"] = adata_sp.obs["intra_cluster"].astype(str).values
        n_cl = adata_sp.obs["intra_cluster"].nunique()
        coords = adata_sp.obsm["X_umap"]
        counts_intra = adata_sp.obs["intra_cluster"].value_counts()
        top_cls = list(counts_intra.head(MAX_INTRA_LABELS).index)
        intra_panel_data.append((sp, mask.sum(), n_cl, coords, adata_sp.obs["intra_cluster"].values, top_cls, counts_intra))
        for cl, cnt in counts_intra.items():
            if cnt >= MIN_MODULE_SIZE:
                adata_sp.obs[adata_sp.obs["intra_cluster"] == cl]["protein_id"].to_csv(modules_dir / f"intra_{sp}_{cnt:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
        sc.tl.rank_genes_groups(adata_sp, groupby="intra_cluster", method="wilcoxon")
        marker_dfs_sp = []
        groups_sp = adata_sp.uns["rank_genes_groups"]["names"].dtype.names
        for group in groups_sp:
            df_g = pd.DataFrame({"cluster": group, "channel": adata_sp.uns["rank_genes_groups"]["names"][group], "score": adata_sp.uns["rank_genes_groups"]["scores"][group], "pvals_adj": adata_sp.uns["rank_genes_groups"]["pvals_adj"][group]})
            marker_dfs_sp.append(df_g)
        marker_df_sp = pd.concat(marker_dfs_sp, ignore_index=True)
        marker_df_sp.insert(0, "analysis", "intra_group")
        marker_df_sp.insert(1, "species", sp)
        intra_marker_dfs.append(marker_df_sp)
    n_intra = len(intra_panel_data)
    if n_intra > 0:
        n_cols_i = min(n_intra, 2)
        n_rows_i = (n_intra + n_cols_i - 1) // n_cols_i
        fig, axes_grid = plt.subplots(n_rows_i, n_cols_i, figsize=(PANEL_SIZE_INCHES * n_cols_i, PANEL_SIZE_INCHES * n_rows_i), dpi=PANEL_DPI, squeeze=False)
        axes = axes_grid.ravel()
        all_top_counts = [int(counts_intra[cl]) for _, _, _, _, _, top_cls, counts_intra in intra_panel_data for cl in top_cls]
        count_min = float(min(all_top_counts)) if all_top_counts else 0.0
        count_max = float(max(all_top_counts)) if all_top_counts else 1.0
        count_norm = plt.Normalize(vmin=count_min, vmax=(count_max if count_max > count_min else count_min + 1.0))
        for idx, (sp, n_pts, n_cl, coords, intra_cls, top_cls, counts_intra) in enumerate(intra_panel_data):
            ax = axes[idx]
            is_top = np.isin(intra_cls, top_cls)
            ax.scatter(coords[~is_top, 0], coords[~is_top, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
            centroids = []
            for c_idx, cl in enumerate(top_cls):
                cl_mask = intra_cls == cl
                if not cl_mask.any():
                    continue
                cl_pts = coords[cl_mask]
                cnt = int(counts_intra[cl])
                color = COLORMAP_INTRA(count_norm(cnt))
                ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=[color], s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
                cx, cy = np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1])
                centroids.append((c_idx, cl, cx, cy, cnt))
            ax.set_title(f"{sp} (Top 10 Clusters, k={n_cl})", fontsize=7, fontweight="bold", pad=2)
            ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
            ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
            ax.tick_params(labelsize=5)
            ax.set_aspect("equal", "datalim")
            for c_idx, cl, cx, cy, cnt in centroids:
                txt = ax.text(cx, cy, str(c_idx + 1), fontsize=5.5, ha="center", va="center", fontweight="bold", color="black", zorder=6)
                txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
        for idx in range(n_intra, len(axes)):
            fig.delaxes(axes[idx])
        sm_intra = plt.cm.ScalarMappable(cmap=COLORMAP_INTRA, norm=count_norm)
        sm_intra.set_array([])
        cbar_intra = fig.colorbar(sm_intra, ax=list(axes[:n_intra]), orientation="horizontal", fraction=0.035, pad=0.08, aspect=35)
        cbar_intra.set_label("Proteins per highlighted cluster", fontsize=6)
        cbar_intra.ax.tick_params(labelsize=5)
        fig.savefig(intra_dir / f"umap_intra_species_panel__{param_tag}.png", dpi=PANEL_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    ###################################### LOGGING
    with open(target_dir / f"clustering_metrics__{param_tag}.txt", "a") as f:
        for line in intra_metrics:
            f.write(line + "\n")
    adata.obs[["protein_id", "species", "cluster", "intra_cluster"]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    marker_frames = [joint_marker_df] + intra_marker_dfs
    pd.concat(marker_frames, ignore_index=True).to_csv(target_dir / f"channel_importance_rankings__{param_tag}.csv", index=False)
    return letters, facets, cluster_sp_frac, counts_df
def run_second_pass(adata: ad.AnnData, letters: dict[str, str], facets: dict[str, dict[str, str]], cluster_sp_frac: pd.DataFrame, counts_df: pd.DataFrame, target_dir: Path, param_tag: str, rep_key: str, n_pcs_use: int | None, use_pca: bool) -> None:
    """Recompute neighbors, Leiden and UMAP on the lettered clusters and plot facets with first- vs second-pass Venn diagrams.

    Args:
        adata (ad.AnnData): First-pass embeddings carrying the ``cluster`` column.
        letters (dict[str, str]): Global mapping cluster -> letter.
        facets (dict[str, dict[str, str]]): Facet letters per species.
        cluster_sp_frac (pd.DataFrame): First-pass species fraction per cluster.
        counts_df (pd.DataFrame): First-pass protein counts per cluster and species.
        target_dir (Path): Output directory for this pass.
        param_tag (str): Parameter tag appended to output file names.
        rep_key (str): Representation used for neighbors (``X`` or ``X_pca``).
        n_pcs_use (int | None): Number of PCs passed to neighbors.
        use_pca (bool): Whether the PCA space is used for silhouette scores.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    adata_2 = adata[adata.obs["cluster"].isin(list(letters))].copy()
    adata_2.obs = adata_2.obs.drop(columns="intra_cluster").rename(columns={"cluster": "cluster_p1"})
    adata_2.obs["cluster_p1"] = adata_2.obs["cluster_p1"].cat.remove_unused_categories()
    print(f"[PYTHON-INFO][DATA-SHAPE] Second-pass data shape: {adata_2.shape}")
    ###################################### SECOND_PASS_CLUSTERING
    sc.pp.neighbors(adata_2, n_neighbors=min(args.n_neighbors, adata_2.n_obs - 1), use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine")
    sc.tl.leiden(adata_2, resolution=args.resolution, key_added="cluster_p2", flavor="igraph")
    sc.tl.umap(adata_2)
    if adata_2.obs["cluster_p2"].nunique() > 1:
        sil_score = silhouette_score(adata_2.obsm["X_pca"] if use_pca else adata_2.X, adata_2.obs["cluster_p2"], metric="cosine", random_state=SEED)
        print(f"[PYTHON-INFO][METRIC] Second-pass Silhouette Score: {sil_score:.4f}")
    ###################################### SECOND_PASS_PANEL
    obs, coords = adata_2.obs, adata_2.obsm["X_umap"]
    species_list = list(obs["species"].unique())
    n_sp = len(species_list)
    fig = plt.figure(figsize=(PANEL_SIZE_INCHES * n_sp, PANEL_SIZE_INCHES * 2 + 0.4), dpi=PANEL_DPI)
    gs = fig.add_gridspec(3, n_sp, height_ratios=[1, 1, 0.04], hspace=0.30, wspace=0.25)
    for idx, sp in enumerate(species_list):
        draw_facet(fig.add_subplot(gs[0, idx]), coords, obs, "cluster_p1", sp, cluster_sp_frac[sp], facets[sp], counts_df[sp])
        sub = gs[1, idx].subgridspec(2, 2, hspace=0.25, wspace=0.15)
        sp_obs = obs[obs["species"] == sp]
        facet_items = list(facets[sp].items())
        for v_idx in range(4):
            ax = fig.add_subplot(sub[v_idx // 2, v_idx % 2])
            ax.set_axis_off()
            if v_idx >= len(facet_items): continue
            cl, letter = facet_items[v_idx]
            p1_obs = sp_obs[sp_obs["cluster_p1"] == cl]
            if p1_obs.empty: continue
            best = p1_obs["cluster_p2"].value_counts().idxmax()
            v = venn2([set(p1_obs["protein_id"]), set(sp_obs[sp_obs["cluster_p2"] == best]["protein_id"])], set_labels=(f"{letter} p1", f"p2 {best}"), set_colors=(PALETTE[0], PALETTE[3]), alpha=ALPHA_MAX, ax=ax)
            for t in list(v.set_labels) + list(v.subset_labels):
                if t is not None: t.set_fontsize(4)
            ax.set_title(f"{letter} vs p2 cl{best}", fontsize=5, fontweight="bold", pad=1)
    add_spec_cbar(fig, fig.add_subplot(gs[-1, :]))
    fig.savefig(target_dir / f"umap_lettered_panel__{param_tag}.png", dpi=PANEL_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
###################################### MAIN
def main() -> None:
    """Load embeddings, preprocess, and run the first and second analysis passes."""
    start_time = time.time()
    npz_files = [p.strip() for p in args.npzs.split(",")]
    if len(npz_files) < 2:
        raise ValueError("[PYTHON-ERROR] This comparative script requires at least two .npz files.")
    use_pca = args.n_pcs is not None and args.n_pcs > 0
    param_tag = f"res{args.resolution}_nn{args.n_neighbors}" + (f"_npc{args.n_pcs}" if use_pca else "_full")
    target_dir = Path(args.outdir) / "_".join([Path(f).stem for f in npz_files])
    target_dir.mkdir(parents=True, exist_ok=True)
    ###################################### LOAD_DATA
    adatas = []
    for f in tqdm(npz_files, desc="Loading embeddings"):
        data = np.load(f)
        k_emb = next(k for k in data.files if "embeddings" in k)
        k_ids = next(k for k in data.files if "ids" in k)
        species_name = Path(f).stem.replace("_reps", "")
        obs_index = [f"{species_name}_{pid}" for pid in data[k_ids]]
        adata = ad.AnnData(X=data[k_emb].astype(np.float32), obs=pd.DataFrame({"protein_id": data[k_ids], "species": species_name}, index=obs_index))
        adatas.append(adata)
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
    ###################################### FIRST_PASS
    letters, facets, cluster_sp_frac, counts_df = run_pass(adata, target_dir / "first_pass", param_tag, rep_key, n_pcs_use, use_pca)
    ###################################### SECOND_PASS
    if len(letters) < 2:
        print(f"[PYTHON-INFO][SECOND-PASS] Skipping: fewer than two lettered clusters ({len(letters)})")
    else:
        run_second_pass(adata, letters, facets, cluster_sp_frac, counts_df, target_dir / "second_pass", param_tag, rep_key, n_pcs_use, use_pca)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
