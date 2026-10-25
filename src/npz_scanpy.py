#!/usr/bin/env python3
"""Processes ESMC protein embeddings with joint clustering, intra-species clustering, intra-to-inter cluster flux diagrams and module extraction.

Embeddings are L2-normalised before cosine kNN and Leiden clustering. The report panels, each A4/4 wide, are: (a) intra-species UMAPs with the MAX_INTRA_LABELS largest clusters coloured by protein count, (b) joint UMAPs coloured by cluster specificity index with the MAX_JOINT_LABELS largest joint clusters of each species ranked 1..N, (c) a re-projected (not reclustered) UMAP of all proteins of those top joint clusters coloured by cluster, (d) intra-to-top-joint cluster fluxes on a square-root (semi-quantitative) scale, where intra clusters inherit a cluster colour only when a strict majority of their proteins flows into it, and (e) a per-species protein-count barplot of the top joint clusters. All figures are saved as PNG and PDF, and the PDFs are merged into a final {dirname}_scanpy.pdf report.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import datetime
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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
COLOR_FG = "#8A929E"
COLORMAP_SPEC = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
DEFAULT_OUTDIR = "data/scanpy/"
FIG_EXTRA_HEIGHT = 1.0
FLUX_ALPHA = 0.6
FLUX_FONT_SCALE = 1.5
FLUX_GAP_TOTAL = 0.15
FLUX_MIN_NODE = 0.02
FLUX_NODE_WIDTH = 0.04
FLUX_SCALE_POWER = 0.5
FONT_FACET = 12
FONT_LABEL = 10
FONT_TICK = 9
FONT_TITLE = 14
LEGEND_CHAR_INCHES = 0.08
LEGEND_COL_INCHES = 0.4
LEGEND_MARKER = 7
LEIDEN_RES = 1.0
MAX_INTRA_LABELS = 10
MAX_JOINT_LABELS = 10
MIN_MODULE_SIZE = 100
MIN_PERCENT_ENRICHMENT = 80
N_NEIGHBORS = 15
N_PCS = None
PALETTE = ["#ECDE90", "#93DEB6", "#F1B584", "#76BBE2", "#DA919B", "#899E64", "#8E81B9", "#A66C58", "#30767A", "#7A5271"]
PANEL_DPI = 300
PANEL_WIDTH_INCHES = A4_WIDTH_INCHES / 4
PT_SIZE = 1.0
REPORT_ORDER = ["umap_intra_species_top_clusters", "umap_joint_clusters_species_specificity", "umap_reprojected_top_clusters", "sankey_intra_to_inter_clusters", "barplot_umap_labelled_clusters_protein_counts"]
SANKEY_HEIGHT_SCALE = 3.0
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
    ydata[script_key] = {k: getattr(args, k) for k in params}
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
    """Return the shared figure size: one A4/4-wide square panel per column plus a strip for titles and legends.

    Args:
        n_cols (int): Number of panels in the row.

    Returns:
        tuple[float, float]: Figure width and height in inches.
    """
    return PANEL_WIDTH_INCHES * n_cols, PANEL_WIDTH_INCHES + FIG_EXTRA_HEIGHT
def row_figure(n_cols: int, title: str, height_scale: float = 1.0, strip: bool = True, font_scale: float = 1.0) -> tuple[plt.Figure, GridSpec]:
    """Create a one-row panel figure, optionally with a bottom strip for a colorbar.

    Args:
        n_cols (int): Number of panels in the row.
        title (str): Figure title.
        height_scale (float): Multiplier applied to the figure height.
        strip (bool): Whether to add the bottom colorbar strip.
        font_scale (float): Multiplier applied to the title font size.

    Returns:
        tuple[plt.Figure, GridSpec]: Figure and its grid (row 0 panels, row 1 strip if requested).
    """
    width, height = fig_size(n_cols)
    fig = plt.figure(figsize=(width, height * height_scale), dpi=PANEL_DPI, layout="constrained")
    fig.suptitle(title, fontsize=FONT_TITLE * font_scale, fontweight="bold")
    return fig, fig.add_gridspec(2, n_cols, height_ratios=[1, 0.04 / height_scale]) if strip else fig.add_gridspec(1, n_cols)
def top_joint_clusters(counts_df: pd.DataFrame, sp: str) -> dict[str, str]:
    """Rank the MAX_JOINT_LABELS largest joint clusters of a species by its protein count.

    Args:
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).
        sp (str): Species of the facet.

    Returns:
        dict[str, str]: Mapping cluster -> rank label (1 = largest), in rank order.
    """
    top = counts_df[sp].sort_values(ascending=False).head(MAX_JOINT_LABELS)
    return {cl: str(r + 1) for r, cl in enumerate(top[top > 0].index)}
def cluster_colors(counts_df: pd.DataFrame, shown: set[str]) -> dict[str, str]:
    """Assign an alternating PALETTE colour to each shown joint cluster, in order of decreasing total size.

    Args:
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).
        shown (set[str]): Joint clusters ranked in at least one species facet.

    Returns:
        dict[str, str]: Mapping cluster -> colour.
    """
    order = counts_df.loc[sorted(shown)].sum(axis=1).sort_values(ascending=False, kind="stable").index
    return {cl: PALETTE[i % len(PALETTE)] for i, cl in enumerate(order)}
def gradient_hex(value: float, norm: Normalize) -> str:
    """Map a value to a hex colour on the quantitative gradient.

    Args:
        value (float): Value to map.
        norm (Normalize): Normalisation shared by the panel.

    Returns:
        str: Hex colour.
    """
    return to_hex(COLORMAP_SPEC(norm(value)))
def draw_facet(ax: plt.Axes, coords: np.ndarray, obs: pd.DataFrame, sp: str, colors: dict[str, str], facet: dict[str, str], sp_counts: pd.Series, spec: pd.Series | None = None) -> None:
    """Draw one species facet coloured by joint cluster, or by specificity index of all its proteins when ``spec`` is given, with rank labels at cluster centroids.

    Args:
        ax (plt.Axes): Target axes.
        coords (np.ndarray): UMAP coordinates of all points.
        obs (pd.DataFrame): Observation table with ``species`` and ``cluster`` columns.
        sp (str): Species of the facet.
        colors (dict[str, str]): Mapping cluster -> colour.
        facet (dict[str, str]): Clusters -> rank labels.
        sp_counts (pd.Series): Proteins of ``sp`` per cluster.
        spec (pd.Series | None): Specificity index of ``sp`` per cluster; when None clusters are coloured by ``colors``.
    """
    is_sp = (obs["species"] == sp).values
    cl_str = obs["cluster"].astype(str)
    ax.scatter(coords[~is_sp, 0], coords[~is_sp, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0)
    if spec is not None:
        scores = cl_str.map(spec).values.astype(float)
        order = np.where(is_sp)[0][np.argsort(scores[is_sp])]
        fg_colors = COLORMAP_SPEC(scores[order])
        fg_colors[:, 3] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * scores[order]
        ax.scatter(coords[order, 0], coords[order, 1], c=fg_colors, s=PT_SIZE, rasterized=True, linewidths=0)
    else:
        in_facet = cl_str.isin(list(facet)).values
        ax.scatter(coords[is_sp & ~in_facet, 0], coords[is_sp & ~in_facet, 1], c=COLOR_FG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0)
        fg = is_sp & in_facet
        ax.scatter(coords[fg, 0], coords[fg, 1], c=cl_str.map(colors).values[fg], s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0)
    max_cnt_sp = max((sp_counts[cl] for cl in facet), default=1)
    for cl, label in facet.items():
        cl_pts = coords[(cl_str == cl).values]
        if len(cl_pts) == 0: continue
        f_size = float(np.clip(7.0 + 7.0 * (sp_counts[cl] / max_cnt_sp), 7.0, 14.0))
        txt = ax.text(cl_pts[:, 0].mean() + 1.0, cl_pts[:, 1].mean(), label, fontsize=f_size, ha="center", va="center", fontweight="bold", color="black")
        txt.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
    ax.set_title(f"{sp} (n={is_sp.sum():,})", fontsize=FONT_FACET, fontweight="bold", pad=2)
    ax.set_xlabel("UMAP 1", fontsize=FONT_LABEL, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=FONT_LABEL, labelpad=1)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_aspect("equal", "datalim")
def add_bottom_legend(fig: plt.Figure, colors: dict[str, str], title: str, marker: str = "o", font_scale: float = 1.0) -> None:
    """Add a label -> colour legend below the panels, with as many columns as fit the figure width.

    Args:
        fig (plt.Figure): Constrained-layout figure.
        colors (dict[str, str]): Mapping label -> colour, in legend order.
        title (str): Legend title.
        marker (str): Marker of the legend handles.
        font_scale (float): Multiplier applied to the legend font sizes.
    """
    if not colors: return
    handles = [Line2D([], [], ls="", marker=marker, color=c, markersize=LEGEND_MARKER * font_scale, label=lab) for lab, c in colors.items()]
    ncol = max(1, min(len(handles), int(fig.get_figwidth() / ((LEGEND_COL_INCHES + LEGEND_CHAR_INCHES * max(len(str(lab)) for lab in colors)) * font_scale))))
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False, fontsize=FONT_TICK * font_scale, title=title, title_fontsize=FONT_LABEL * font_scale)
def add_colorbar(cax: plt.Axes, norm: Normalize, label: str) -> None:
    """Add a centred horizontal colorbar of the quantitative gradient.

    Args:
        cax (plt.Axes): Strip axes hosting the colorbar.
        norm (Normalize): Normalisation of the plotted values.
        label (str): Colorbar label, written on its left.
    """
    cax.axis("off")
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=COLORMAP_SPEC), cax=cax.inset_axes([0.35, 0.0, 0.3, 1.0]), orientation="horizontal")
    cb.ax.tick_params(labelsize=FONT_TICK, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.text(0.34, 0.5, label, fontsize=FONT_LABEL, ha="right", va="center", transform=cax.transAxes)
def flux_layout(sizes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Scale node sizes semi-quantitatively, with a floor of FLUX_MIN_NODE so small nodes are not crushed, and stack them to fill the unit height.

    Args:
        sizes (pd.Series): Protein count per node, in plotting order.

    Returns:
        tuple[pd.Series, pd.Series]: Node heights and top offsets.
    """
    h = sizes.astype(float) ** FLUX_SCALE_POWER
    gap = FLUX_GAP_TOTAL / (len(h) - 1) if len(h) > 1 else 0.0
    avail = 1 - gap * (len(h) - 1)
    floor = min(FLUX_MIN_NODE, avail / len(h) / 2)
    h = floor + h / h.sum() * (avail - floor * len(h))
    return h, pd.Series(np.concatenate([[0], np.cumsum(h.values + gap)[:-1]]), index=h.index)
def draw_flux(ax: plt.Axes, flows: pd.DataFrame, left_colors: dict[str, str], right_colors: dict[str, str], left_labels: dict[str, str], right_labels: dict[str, str], outlined: set[str], title: str) -> None:
    """Draw an alluvial diagram of protein fluxes from intra-species clusters (left) to top joint clusters (right) on a semi-quantitative scale.

    Args:
        ax (plt.Axes): Target axes.
        flows (pd.DataFrame): Protein counts with intra-clusters as rows and joint clusters as columns, in plotting order.
        left_colors (dict[str, str]): Colour of each intra-cluster node.
        right_colors (dict[str, str]): Colour of each joint cluster node.
        left_labels (dict[str, str]): Text of the labelled intra-cluster nodes.
        right_labels (dict[str, str]): Text of the labelled joint cluster nodes (rank and species protein count).
        outlined (set[str]): Joint clusters drawn with a thin black border.
        title (str): Axes title.
    """
    l_sz, r_sz = flows.sum(axis=1), flows.sum(axis=0)
    (l_h, l_off), (r_h, r_off) = flux_layout(l_sz), flux_layout(r_sz)
    xs = np.linspace(FLUX_NODE_WIDTH, 1 - FLUX_NODE_WIDTH, 60)
    t = (1 - np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))) / 2
    for side, heights, offs, x0, cols, labs in (("l", l_h, l_off, 0.0, left_colors, left_labels), ("r", r_h, r_off, 1 - FLUX_NODE_WIDTH, right_colors, right_labels)):
        for lab, h in heights.items():
            ax.fill_betweenx([offs[lab], offs[lab] + h], x0, x0 + FLUX_NODE_WIDTH, facecolor=cols[lab], edgecolor="black" if side == "r" and lab in outlined else "none", lw=0.6)
            if lab in labs: ax.text(-0.02 if side == "l" else 1.02, offs[lab] + h / 2, labs[lab], fontsize=FONT_LABEL * FLUX_FONT_SCALE, fontweight="bold", ha="right" if side == "l" else "left", va="center")
    l_cur, r_cur = l_off.copy(), r_off.copy()
    for (i, j), v in flows.stack().items():
        if v == 0: continue
        lh, rh = v / l_sz[i] * l_h[i], v / r_sz[j] * r_h[j]
        ax.fill_between(xs, l_cur[i] + (r_cur[j] - l_cur[i]) * t, l_cur[i] + lh + (r_cur[j] + rh - l_cur[i] - lh) * t, color=right_colors[j], alpha=FLUX_ALPHA, lw=0)
        l_cur[i] += lh
        r_cur[j] += rh
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(1, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=FONT_FACET * FLUX_FONT_SCALE, fontweight="bold", pad=2)
def plot_counts_bar(top_counts: pd.DataFrame, species_list: list[str], title: str, path: Path) -> None:
    """Plot per-species protein counts of the top joint clusters.

    Args:
        top_counts (pd.DataFrame): Protein counts per top joint cluster (rows) and species (columns), in plotting order.
        species_list (list[str]): Species to plot.
        title (str): Plot title.
        path (Path): Output path.
    """
    n_sp = len(species_list)
    x, width = np.arange(len(top_counts)), 0.8 / n_sp
    sp_colors = {sp: PALETTE[i % len(PALETTE)] for i, sp in enumerate(species_list)}
    fig, ax = plt.subplots(figsize=fig_size(n_sp), dpi=PANEL_DPI, layout="constrained")
    for i, sp in enumerate(species_list):
        ax.bar(x - 0.4 + width * (i + 0.5), top_counts[sp].values, width=width, color=sp_colors[sp], linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cl}" for cl in top_counts.index], fontsize=FONT_LABEL, rotation=90)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlim(-0.6, len(top_counts) - 0.4)
    ax.set_xlabel("Joint cluster", fontsize=FONT_LABEL, labelpad=2)
    ax.set_ylabel("Proteins", fontsize=FONT_LABEL, labelpad=2)
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold")
    add_bottom_legend(fig, sp_colors, "Species", "s")
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
def run_pass(adata: ad.AnnData, target_dir: Path, param_tag: str, rep_key: str, n_pcs_use: int | None, use_pca: bool) -> tuple[dict[str, dict[str, str]], dict[str, str], pd.DataFrame]:
    """Run joint clustering, species panel, module extraction, intra-group clustering, flux diagrams and logging.

    Args:
        adata (ad.AnnData): Normalised embeddings.
        target_dir (Path): Output directory for this pass.
        param_tag (str): Model and parameter tag appended to output file names.
        rep_key (str): Representation used for neighbors (``X`` or ``X_pca``).
        n_pcs_use (int | None): Number of PCs passed to neighbors.
        use_pca (bool): Whether the PCA space is used for silhouette scores.

    Returns:
        tuple[dict[str, dict[str, str]], dict[str, str], pd.DataFrame]: Top joint cluster ranks per species, cluster colours and counts per cluster.
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
    sil_score = silhouette_score(adata.obsm["X_pca"] if use_pca else adata.X, adata.obs["cluster"], metric="cosine", random_state=SEED)
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
    cluster_sp_frac.index, counts_df.index = cluster_sp_frac.index.astype(str), counts_df.index.astype(str)
    facets = {sp: top_joint_clusters(counts_df, sp) for sp in species_list}
    colors = cluster_colors(counts_df, {cl for f in facets.values() for cl in f})
    n_sp = len(species_list)
    fig, gs = row_figure(n_sp, "Inter-species")
    for idx, sp in enumerate(species_list):
        draw_facet(fig.add_subplot(gs[0, idx]), umap_coords, adata.obs, sp, colors, facets[sp], counts_df[sp], cluster_sp_frac[sp])
    add_colorbar(fig.add_subplot(gs[-1, :]), Normalize(0, 1), CBAR_LABEL)
    save_fig(fig, target_dir / f"umap_joint_clusters_species_specificity__{param_tag}.png")
    plt.close(fig)
    ###################################### JOINT_SAVE
    rank_rows = [(sp, cl, rk, counts_df.loc[cl, sp], cluster_sp_frac.loc[cl, sp]) for sp, sp_ranks in facets.items() for cl, rk in sp_ranks.items()]
    pd.DataFrame(rank_rows, columns=["species", "cluster", "rank", "protein_count", "species_fraction"]).to_csv(target_dir / f"species_top_clusters__{param_tag}.csv", index=False)
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
        sil_sp = silhouette_score(adata_sp.obsm["X_pca"] if use_pca else adata_sp.X, adata_sp.obs["intra_cluster"], metric="cosine", random_state=SEED)
        print(f"[PYTHON-INFO][METRIC] Intra Silhouette ({sp}): {sil_sp:.4f}")
        intra_metrics.append(f"intra_silhouette_{sp}: {sil_sp}")
        adata.obs.loc[mask, "intra_cluster"] = adata_sp.obs["intra_cluster"].astype(str).values
        n_cl = adata_sp.obs["intra_cluster"].nunique()
        coords = adata_sp.obsm["X_umap"]
        counts_intra = adata_sp.obs["intra_cluster"].value_counts()
        counts_intra.index = counts_intra.index.astype(str)
        intra_cls = adata_sp.obs["intra_cluster"].astype(str).values
        intra_panel_data.append((sp, mask.sum(), n_cl, coords, intra_cls, counts_intra.head(MAX_INTRA_LABELS)))
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
        top_vals = [cnt for *_, top in intra_panel_data for cnt in top.values]
        norm_size = Normalize(min(top_vals), max(top_vals) if max(top_vals) > min(top_vals) else min(top_vals) + 1)
        fig, gs = row_figure(n_sp, "Intra-species")
        for idx, (sp, n_pts, n_cl, coords, intra_cls, top) in enumerate(intra_panel_data):
            ax = fig.add_subplot(gs[0, idx])
            is_shown = np.isin(intra_cls, list(top.index))
            ax.scatter(coords[~is_shown, 0], coords[~is_shown, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
            for c_idx, (cl, cnt) in enumerate(top.items()):
                cl_pts = coords[intra_cls == cl]
                ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=gradient_hex(cnt, norm_size), s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
                txt = ax.text(np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1]), str(c_idx + 1), fontsize=FONT_LABEL, ha="center", va="center", fontweight="bold", color="black", zorder=6)
                txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
            ax.set_title(f"{sp} (Top {MAX_INTRA_LABELS} Clusters, k={n_cl})", fontsize=FONT_FACET, fontweight="bold", pad=2)
            ax.set_xlabel("UMAP 1", fontsize=FONT_LABEL, labelpad=1)
            ax.set_ylabel("UMAP 2", fontsize=FONT_LABEL, labelpad=1)
            ax.tick_params(labelsize=FONT_TICK)
            ax.set_aspect("equal", "datalim")
        add_colorbar(fig.add_subplot(gs[-1, :]), norm_size, "Proteins per highlighted cluster")
        save_fig(fig, intra_dir / f"umap_intra_species_top_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### INTRA_INTER_FLUX
    if n_intra > 0:
        fig, gs = row_figure(n_sp, "Intra- to Inter-species cluster fluxes", SANKEY_HEIGHT_SCALE, strip=False, font_scale=FLUX_FONT_SCALE)
        for idx, (sp, n_pts, n_cl, coords, intra_cls, top) in enumerate(intra_panel_data):
            ax = fig.add_subplot(gs[0, idx])
            obs_sp = adata.obs[adata.obs["species"] == sp]
            title = f"{sp}\n(n={len(obs_sp):,}, k={n_cl})"
            full = pd.crosstab(obs_sp["intra_cluster"].astype(str), obs_sp["cluster"].astype(str))
            ranked = [cl for cl in facets[sp] if cl in full.columns]
            if not ranked:
                ax.axis("off")
                ax.set_title(title, fontsize=FONT_FACET * FLUX_FONT_SCALE, fontweight="bold", pad=2)
                continue
            flows = full[ranked]
            flows = flows[flows.sum(axis=1) > 0]
            l_order = [cl for cl in top.index if cl in flows.index] + flows.sum(axis=1).drop(list(top.index), errors="ignore").sort_values(ascending=False).index.tolist()
            flows = flows.loc[l_order]
            best = flows.idxmax(axis=1)
            left_colors = {cl: colors[best[cl]] if flows.loc[cl, best[cl]] > full.loc[cl].sum() / 2 else COLOR_BG for cl in l_order}
            rank = {cl: str(c_idx + 1) for c_idx, cl in enumerate(top.index) if cl in flows.index}
            draw_flux(ax, flows, left_colors, {cl: colors[cl] for cl in ranked}, rank, {cl: f"{facets[sp][cl]} ({counts_df.loc[cl, sp]:,})" for cl in ranked}, set(), title)
        add_bottom_legend(fig, {f"C{cl}": c for cl, c in colors.items()}, "Joint cluster", font_scale=FLUX_FONT_SCALE)
        save_fig(fig, target_dir / f"sankey_intra_to_inter_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### LOGGING
    with open(target_dir / f"clustering_metrics__{param_tag}.txt", "a") as f:
        for line in intra_metrics:
            f.write(line + "\n")
    adata.obs[["protein_id", "species", "cluster", "intra_cluster"]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    marker_frames = [joint_marker_df] + intra_marker_dfs
    pd.concat(marker_frames, ignore_index=True).to_csv(target_dir / f"channel_importance_rankings__{param_tag}.csv", index=False)
    return facets, colors, counts_df
def run_second_pass(adata: ad.AnnData, facets: dict[str, dict[str, str]], colors: dict[str, str], counts_df: pd.DataFrame, target_dir: Path, param_tag: str, rep_key: str, n_pcs_use: int | None) -> None:
    """Re-project (without reclustering) all proteins of the top joint clusters, plot their facets and a per-species count barplot.

    Args:
        adata (ad.AnnData): First-pass embeddings carrying the ``cluster`` column.
        facets (dict[str, dict[str, str]]): Top joint cluster ranks per species.
        colors (dict[str, str]): Mapping cluster -> colour.
        counts_df (pd.DataFrame): First-pass protein counts per cluster and species.
        target_dir (Path): Output directory for this pass.
        param_tag (str): Model and parameter tag appended to output file names.
        rep_key (str): Representation used for neighbors (``X`` or ``X_pca``).
        n_pcs_use (int | None): Number of PCs passed to neighbors.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    adata_2 = adata[adata.obs["cluster"].astype(str).isin(list(colors))].copy()
    adata_2.obs = adata_2.obs.drop(columns="intra_cluster")
    adata_2.obs["cluster"] = adata_2.obs["cluster"].cat.remove_unused_categories()
    print(f"[PYTHON-INFO][DATA-SHAPE] Second-pass data shape: {adata_2.shape}")
    ###################################### SECOND_PASS_PROJECTION
    sc.pp.neighbors(adata_2, n_neighbors=min(args.n_neighbors, adata_2.n_obs - 1), use_rep=rep_key, n_pcs=n_pcs_use, metric="cosine")
    sc.tl.umap(adata_2)
    ###################################### SECOND_PASS_PANEL
    obs, coords = adata_2.obs, adata_2.obsm["X_umap"]
    species_list = list(obs["species"].unique())
    n_sp = len(species_list)
    fig, gs = row_figure(n_sp, "Top Inter-species clusters (re-projected)", strip=False)
    for idx, sp in enumerate(species_list):
        draw_facet(fig.add_subplot(gs[0, idx]), coords, obs, sp, colors, facets[sp], counts_df[sp])
    add_bottom_legend(fig, {f"C{cl}": c for cl, c in colors.items()}, "Joint cluster")
    save_fig(fig, target_dir / f"umap_reprojected_top_clusters__{param_tag}.png")
    plt.close(fig)
    ###################################### TOP_COUNTS_BARPLOT
    top_counts = counts_df.loc[list(colors), species_list]
    top_counts = top_counts.loc[top_counts.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    plot_counts_bar(top_counts, species_list, "Protein per cluster (number by species)", target_dir / f"barplot_umap_labelled_clusters_protein_counts__{param_tag}.png")
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
    facets, colors, counts_df = run_pass(adata, target_dir, param_tag, rep_key, n_pcs_use, use_pca)
    ###################################### SECOND_PASS
    run_second_pass(adata, facets, colors, counts_df, target_dir, param_tag, rep_key, n_pcs_use)
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
