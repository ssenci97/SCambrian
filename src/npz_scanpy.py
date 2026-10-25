#!/usr/bin/env python3
"""Processes ESMC protein embeddings with joint clustering, globally lettered species-specificity UMAP panels, intra-species clustering, intra-to-inter cluster flux diagrams onto all lettered clusters (UMAP-shown ones outlined in black) and module extraction.

A first pass runs the full analysis and assigns unique letters to the most species-specific joint clusters; a second pass recomputes neighbors, Leiden and UMAP on the lettered clusters only, plots their UMAP facets and a per-species protein-count barplot of the UMAP-shown clusters. All figures are saved as PNG and PDF, and the PDFs are merged into a final {dirname}_scanpy.pdf report.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pymupdf
import resource
import scanpy as sc
from sklearn.metrics import silhouette_score
import time
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in log2.*")
A4_WIDTH_INCHES = 11.69
ALPHA_BG = 0.50
ALPHA_MAX = 0.95
ALPHA_MIN = 0.55
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
COLOR_BG = "#BFC5CE"
COLORMAP_SPEC = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
DEFAULT_OUTDIR = "data/scanpy/"
FIG_EXTRA_HEIGHT = 1.0
FLUX_ALPHA = 0.6
FLUX_GAP = 0.02
FLUX_NODE_WIDTH = 0.04
FONT_FACET = 8
FONT_LABEL = 7
FONT_TICK = 6
FONT_TITLE = 10
LEIDEN_RES = 1.0
MAX_LETTERED = 8
MAX_SPECIFIC_LABELS = 5
MIN_MODULE_SIZE = 200
PALETTE = ["#ECDE90", "#93DEB6", "#F1B584", "#76BBE2", "#DA919B", "#899E64", "#8E81B9", "#A66C58", "#30767A", "#7A5271"]
MIN_PERCENT_ENRICHMENT = 80
N_NEIGHBORS = 15
N_PCS = None
PANEL_DPI = 300
PT_SIZE = 1.0
REPORT_ORDER = ["umap_joint_clusters_species_specificity", "umap_intra_species_top_clusters", "umap_reclustered_species_specific_clusters", "barplot_umap_labelled_clusters_protein_counts", "sankey_intra_to_inter_clusters"]
SANKEY_HEIGHT_SCALE = 2.0
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
def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a figure as both PNG and PDF.

    Args:
        fig (plt.Figure): Figure to save.
        path (Path): Output path; its suffix is replaced by ``.png`` and ``.pdf``.
    """
    for ext in (".png", ".pdf"):
        fig.savefig(path.with_suffix(ext), dpi=PANEL_DPI, facecolor="white")
def fig_size(n_cols: int) -> tuple[float, float]:
    """Return the shared figure size: landscape A4 width and a height fitting one row of square panels.

    Args:
        n_cols (int): Number of panels in the row.

    Returns:
        tuple[float, float]: Figure width and height in inches.
    """
    return A4_WIDTH_INCHES, A4_WIDTH_INCHES / n_cols + FIG_EXTRA_HEIGHT
def row_figure(n_cols: int, title: str, height_scale: float = 1.0) -> tuple[plt.Figure, GridSpec]:
    """Create a one-row panel figure with a bottom strip for a colorbar or legend.

    Args:
        n_cols (int): Number of panels in the row.
        title (str): Figure title.
        height_scale (float): Multiplier applied to the figure height.

    Returns:
        tuple[plt.Figure, GridSpec]: Figure and its grid (row 0 panels, row 1 strip).
    """
    width, height = fig_size(n_cols)
    fig = plt.figure(figsize=(width, height * height_scale), dpi=PANEL_DPI, layout="constrained")
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold")
    return fig, fig.add_gridspec(2, n_cols, height_ratios=[1, 0.04 / height_scale])
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
    sort_order = fg_idx[np.lexsort((spec_scores[is_sp], obs[cluster_key].isin(list(facet)).values[is_sp]))]
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
    ax.set_title(f"{sp} (n={is_sp.sum():,})", fontsize=FONT_FACET, fontweight="bold", pad=2)
    ax.set_xlabel("UMAP 1", fontsize=FONT_LABEL, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=FONT_LABEL, labelpad=1)
    ax.tick_params(labelsize=FONT_TICK)
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
    cbar.set_label(CBAR_LABEL, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)
def draw_flux(ax: plt.Axes, flows: pd.DataFrame, right_colors: dict[str, tuple], outlined: set[str], title: str) -> None:
    """Draw an alluvial diagram of protein fluxes from intra-species clusters (left) to inter-species clusters (right).

    Args:
        ax (plt.Axes): Target axes.
        flows (pd.DataFrame): Protein counts with intra-cluster ranks as rows and inter-cluster labels as columns, in plotting order.
        right_colors (dict[str, tuple]): Colour of each inter-cluster node.
        outlined (set[str]): Inter-cluster labels drawn with a thin black border.
        title (str): Axes title.
    """
    gap = FLUX_GAP * flows.values.sum()
    l_sz, r_sz = flows.sum(axis=1), flows.sum(axis=0)
    l_off = pd.Series(np.concatenate([[0], np.cumsum(l_sz.values + gap)[:-1]]), index=l_sz.index)
    r_off = pd.Series(np.concatenate([[0], np.cumsum(r_sz.values + gap)[:-1]]), index=r_sz.index)
    xs = np.linspace(FLUX_NODE_WIDTH, 1 - FLUX_NODE_WIDTH, 60)
    t = (1 - np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))) / 2
    for side, sizes, offs, x0 in (("l", l_sz, l_off, 0.0), ("r", r_sz, r_off, 1 - FLUX_NODE_WIDTH)):
        for lab, h in sizes.items():
            color = PALETTE[int(lab) - 1] if side == "l" else right_colors[lab]
            ax.fill_betweenx([offs[lab], offs[lab] + h], x0, x0 + FLUX_NODE_WIDTH, facecolor=color, edgecolor="black" if side == "r" and lab in outlined else "none", lw=0.6)
            ax.text(-0.02 if side == "l" else 1.02, offs[lab] + h / 2, lab, fontsize=FONT_LABEL, fontweight="bold", ha="right" if side == "l" else "left", va="center")
    l_cur, r_cur = l_off.copy(), r_off.copy()
    for i in flows.index:
        for j in flows.columns:
            v = flows.loc[i, j]
            if v == 0: continue
            bottom = l_cur[i] + (r_cur[j] - l_cur[i]) * t
            ax.fill_between(xs, bottom, bottom + v, color=right_colors[j], alpha=FLUX_ALPHA, lw=0)
            l_cur[i] += v
            r_cur[j] += v
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(max(l_off.iloc[-1] + l_sz.iloc[-1], r_off.iloc[-1] + r_sz.iloc[-1]), 0)
    ax.axis("off")
    ax.set_title(title, fontsize=FONT_FACET, fontweight="bold", pad=2)
def plot_counts_bar(lt_counts: pd.DataFrame, letters: dict[str, str], shown: set[str], species_list: list[str], title: str, path: Path) -> None:
    """Plot per-species protein counts of lettered clusters, outlining clusters shown in the UMAP facets.

    Args:
        lt_counts (pd.DataFrame): Protein counts per lettered cluster (rows) and species (columns), in plotting order.
        letters (dict[str, str]): Global mapping cluster -> letter.
        shown (set[str]): Clusters labelled in the UMAP facets, drawn with a thin black border.
        species_list (list[str]): Species to plot.
        title (str): Plot title.
        path (Path): Output path.
    """
    n_sp = len(species_list)
    x, width = np.arange(len(lt_counts)), 0.8 / n_sp
    edges = ["black" if cl in shown else "none" for cl in lt_counts.index]
    fig, ax = plt.subplots(figsize=fig_size(n_sp), dpi=PANEL_DPI, layout="constrained")
    for i, sp in enumerate(species_list):
        ax.bar(x - 0.4 + width * (i + 0.5), lt_counts[sp].values, width=width, color=PALETTE[i % len(PALETTE)], label=sp, edgecolor=edges, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([letters[cl] for cl in lt_counts.index], fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlim(-0.6, len(lt_counts) - 0.4)
    ax.set_xlabel("Species-specific cluster", fontsize=FONT_LABEL, labelpad=2)
    ax.set_ylabel("Proteins", fontsize=FONT_LABEL, labelpad=2)
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_TICK, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, path)
    plt.close(fig)
def merge_pdfs(pdfs: list[Path], out: Path) -> None:
    """Merge PDF files into a single report.

    Args:
        pdfs (list[Path]): PDF files in merge order.
        out (Path): Output report path.
    """
    with pymupdf.open() as doc:
        for pdf in pdfs:
            with pymupdf.open(pdf) as src:
                doc.insert_pdf(src)
        doc.save(out)
def run_pass(adata: ad.AnnData, target_dir: Path, param_tag: str, rep_key: str, n_pcs_use: int | None, use_pca: bool) -> tuple[dict[str, str], dict[str, dict[str, str]], pd.DataFrame, pd.DataFrame]:
    """Run joint clustering, lettered species panel, module extraction, intra-group clustering and logging.

    Args:
        adata (ad.AnnData): Normalised embeddings.
        target_dir (Path): Output directory for this pass.
        param_tag (str): Model and parameter tag appended to output file names.
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
    fig, gs = row_figure(n_sp, "Inter-Species")
    for idx, sp in enumerate(species_list):
        draw_facet(fig.add_subplot(gs[0, idx]), umap_coords, adata.obs, "cluster", sp, cluster_sp_frac[sp], facets[sp], counts_df[sp])
    add_spec_cbar(fig, fig.add_subplot(gs[-1, :]))
    save_fig(fig, target_dir / f"umap_joint_clusters_species_specificity__{param_tag}.png")
    plt.close(fig)
    ###################################### JOINT_SAVE
    letter_rows = [(sp, cl, lt, counts_df.loc[cl, sp], cluster_sp_frac.loc[cl, sp]) for sp, sp_letters in facets.items() for cl, lt in sp_letters.items()]
    pd.DataFrame(letter_rows, columns=["species", "cluster", "letter", "protein_count", "specificity"]).to_csv(target_dir / f"species_specific_cluster_letters__{param_tag}.csv", index=False)
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
        top_cls = list(counts_intra.head(len(PALETTE)).index)
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
        fig, gs = row_figure(n_sp, "Intra-species")
        for idx, (sp, n_pts, n_cl, coords, intra_cls, top_cls, counts_intra) in enumerate(intra_panel_data):
            ax = fig.add_subplot(gs[0, idx])
            is_top = np.isin(intra_cls, top_cls)
            ax.scatter(coords[~is_top, 0], coords[~is_top, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
            for c_idx, cl in enumerate(top_cls):
                cl_pts = coords[intra_cls == cl]
                if len(cl_pts) == 0: continue
                ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=PALETTE[c_idx], s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
                txt = ax.text(np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1]), str(c_idx + 1), fontsize=FONT_LABEL, ha="center", va="center", fontweight="bold", color="black", zorder=6)
                txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
            ax.set_title(f"{sp} (top {len(top_cls)} clusters, k={n_cl})", fontsize=FONT_FACET, fontweight="bold", pad=2)
            ax.set_xlabel("UMAP 1", fontsize=FONT_LABEL, labelpad=1)
            ax.set_ylabel("UMAP 2", fontsize=FONT_LABEL, labelpad=1)
            ax.tick_params(labelsize=FONT_TICK)
            ax.set_aspect("equal", "datalim")
        lax = fig.add_subplot(gs[-1, :])
        lax.axis("off")
        for c_idx in range(max(len(d[5]) for d in intra_panel_data)):
            lax.scatter([], [], c=PALETTE[c_idx], s=15, label=str(c_idx + 1))
        lax.legend(loc="center", ncol=len(PALETTE), frameon=False, fontsize=FONT_TICK, title="Intra-species cluster (ordered by size)", title_fontsize=FONT_LABEL)
        save_fig(fig, intra_dir / f"umap_intra_species_top_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### INTRA_INTER_FLUX
    if n_intra > 0:
        fig, gs = row_figure(n_sp, "Intra- to Inter-species cluster fluxes", SANKEY_HEIGHT_SCALE)
        for idx, (sp, n_pts, n_cl, coords, intra_cls, top_cls, counts_intra) in enumerate(intra_panel_data):
            obs_sp = adata.obs[(adata.obs["species"] == sp) & adata.obs["intra_cluster"].isin(top_cls)]
            rank = {cl: str(c_idx + 1) for c_idx, cl in enumerate(top_cls)}
            inter = obs_sp["cluster"].map(letters).fillna("other")
            cols = list(letters.values()) + ["other"]
            flows = pd.crosstab(obs_sp["intra_cluster"].map(rank), inter).reindex(index=list(rank.values()), columns=cols, fill_value=0)
            flows = flows.loc[flows.sum(axis=1) > 0, flows.sum(axis=0) > 0]
            right_colors = {lt: COLORMAP_SPEC(cluster_sp_frac.loc[cl, sp]) for cl, lt in letters.items()} | {"other": COLOR_BG}
            draw_flux(fig.add_subplot(gs[0, idx]), flows, right_colors, set(facets[sp].values()), f"{sp} (n={len(obs_sp):,})")
        add_spec_cbar(fig, fig.add_subplot(gs[-1, :]))
        save_fig(fig, target_dir / f"sankey_intra_to_inter_clusters__{param_tag}.png")
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
    """Recompute neighbors, Leiden and UMAP on the lettered clusters, plot their facets and a per-species count barplot of the UMAP-shown clusters.

    Args:
        adata (ad.AnnData): First-pass embeddings carrying the ``cluster`` column.
        letters (dict[str, str]): Global mapping cluster -> letter.
        facets (dict[str, dict[str, str]]): Facet letters per species.
        cluster_sp_frac (pd.DataFrame): First-pass species fraction per cluster.
        counts_df (pd.DataFrame): First-pass protein counts per cluster and species.
        target_dir (Path): Output directory for this pass.
        param_tag (str): Model and parameter tag appended to output file names.
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
    fig, gs = row_figure(n_sp, "Divergent Inter-species")
    for idx, sp in enumerate(species_list):
        draw_facet(fig.add_subplot(gs[0, idx]), coords, obs, "cluster_p1", sp, cluster_sp_frac[sp], facets[sp], counts_df[sp])
    add_spec_cbar(fig, fig.add_subplot(gs[-1, :]))
    save_fig(fig, target_dir / f"umap_reclustered_species_specific_clusters__{param_tag}.png")
    plt.close(fig)
    ###################################### LETTERED_COUNTS_BARPLOT
    lt_counts = counts_df.loc[list(letters), species_list]
    lt_counts = lt_counts.loc[lt_counts.sum(axis=1).sort_values(ascending=False).index]
    shown = {cl for sp_letters in facets.values() for cl in sp_letters}
    plot_counts_bar(lt_counts.loc[[cl for cl in lt_counts.index if cl in shown]], letters, shown, species_list, "Protein per cluster (number by species)", target_dir / f"barplot_umap_labelled_clusters_protein_counts__{param_tag}.png")
###################################### MAIN
def main() -> None:
    """Load embeddings, preprocess, and run the first and second analysis passes."""
    start_time = time.time()
    npz_files = [p.strip() for p in args.npzs.split(",")]
    if len(npz_files) < 2:
        raise ValueError("[PYTHON-ERROR] This comparative script requires at least two .npz files.")
    use_pca = args.n_pcs is not None and args.n_pcs > 0
    param_tag = f"{Path(npz_files[0]).parent.name}__res{args.resolution}_nn{args.n_neighbors}" + (f"_npc{args.n_pcs}" if use_pca else "_full")
    target_dir = Path(args.outdir) / "_".join([Path(f).stem.replace("_reps", "") for f in npz_files])
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
    letters, facets, cluster_sp_frac, counts_df = run_pass(adata, target_dir, param_tag, rep_key, n_pcs_use, use_pca)
    ###################################### SECOND_PASS
    if len(letters) < 2:
        print(f"[PYTHON-INFO][SECOND-PASS] Skipping: fewer than two lettered clusters ({len(letters)})")
    else:
        run_second_pass(adata, letters, facets, cluster_sp_frac, counts_df, target_dir, param_tag, rep_key, n_pcs_use, use_pca)
    ###################################### PDF_REPORT
    report = target_dir / f"{target_dir.name}_scanpy.pdf"
    pdfs = sorted((p for p in target_dir.rglob(f"*__{param_tag}.pdf") if p != report), key=lambda p: next((i for i, s in enumerate(REPORT_ORDER) if p.stem.startswith(s)), len(REPORT_ORDER)))
    merge_pdfs(pdfs, report)
    print(f"[PYTHON-INFO][REPORT] Merged {len(pdfs)} PDFs into {report}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
