#!/usr/bin/env python3
"""Reports the joint and intra-species clusters saved by npz_scanpy.py: tables, protein modules, cluster panels, intra-to-inter cluster flux diagrams and a merged PDF.

Only the .h5ad is read, so no clustering or embedding is recomputed except the re-projection of the selected joint clusters. Joint (inter-species) clusters are always named by lowercase letters of their Leiden id (0 -> a, 25 -> z, 26 -> aa) and intra-species clusters by their Leiden id as a number string, in every figure, table and module file. Species differences are called with one set of criteria (lowercase names are keys of the h5ad_scanpy section of configs/param_configs.yaml), where c_ks is the number of proteins of species s in joint cluster k and n_s the proteins of s:
    1. species-normalised share p_ks = c_ks / n_s, removing the proteome-size bias;
    2. specificity index S_ks = p_ks / sum_s' p_ks', uniform at 1/n_species;
    3. fold change FC_k = max_s p_ks / min_s p_ks (infinite when a species is absent), with min_s S_ks / max_s S_ks = 1 / FC_k;
    4. chi-square test of homogeneity of the 2 x n_species table [c_ks; n_s - c_ks], BH-corrected across all joint clusters (q_k), valid only when every expected count is at least MIN_EXPECTED;
    5. significant cluster: valid test, q_k < chi2_max_q and FC_k >= min_fold_change;
    6. selected for species s: significant, S_ks > 1/n_species and c_ks >= min_species_proteins, keeping the max_selected_per_species highest S_ks; the selection is the union over species;
    7. species module: significant, S_ks >= module_min_specificity_percent / 100 and c_ks >= min_module_proteins;
    8. balanced (conserved) cluster: FC_k <= 1 / balanced_min_share_ratio and min_s c_ks >= min_species_proteins, an effect-size call only.
The selection drives every report element. The panels, each A4/4 wide, are drawn for every method in ``--layout_methods`` (UMAP and t-SNE, 2D layouts): (a) intra-species layouts with all clusters of at least min_intra_cluster_proteins proteins coloured by protein count, (b) joint layouts coloured by specificity index, labelling per species its selected clusters on top of the others, then once: (c) a re-projected UMAP of all proteins of the selected clusters from every species where present, optimised on the stored joint kNN graph restricted to them and coloured by cluster in every species facet, (d) intra-to-selected joint cluster fluxes on a square-root (semi-quantitative) scale, with joint cluster nodes and flows coloured by specificity index and outlined in black from module_min_specificity_percent, where intra clusters inherit a cluster colour only when a strict majority of their proteins flows into it, and (e) a per-species protein-count barplot of the selected clusters with chi-square significance asterisks stacked vertically over the bars. A report whose outputs all exist and are newer than both this script and the .h5ad is skipped unless ``--overwrite`` is set. All figures are saved as PNG and PDF, and the PDFs are merged into a final {dirname}_scanpy.pdf report.
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
from scipy.stats import chi2_contingency, false_discovery_control
import time
from tqdm import tqdm
from umap.umap_ import find_ab_params, simplicial_set_embedding
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
A4_WIDTH_INCHES = 11.69
ALPHA_BG = 0.50
ALPHA_MAX = 0.95
ALPHA_MIN = 0.55
BALANCED_MIN_SHARE_RATIO = 0.6
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
CHI2_MAX_Q = 0.05
COLORMAP_SPEC = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
COLOR_BG = "#BFC5CE"
COLOR_FG = "#8A929E"
FIG_EXTRA_HEIGHT = 1.0
FLUX_ALPHA = 0.6
FLUX_BORDER_WIDTH = 1.2
FLUX_FONT_SCALE = 1.5
FLUX_GAP_TOTAL = 0.15
FLUX_MIN_NODE = 0.02
FLUX_NODE_WIDTH = 0.04
FLUX_SCALE_POWER = 0.5
FONT_FACET = 12
FONT_LABEL = 10
FONT_STAR = 14
FONT_TICK = 9
FONT_TITLE = 14
LAYOUT_METHODS = ["umap", "tsne"]
LEGEND_CHAR_INCHES = 0.08
LEGEND_COL_INCHES = 0.4
LEGEND_MARKER = 7
MAX_SELECTED_PER_SPECIES = 6
METHOD_NAMES = {"umap": "UMAP", "tsne": "t-SNE"}
MIN_EXPECTED = 5
MIN_FOLD_CHANGE = 2.0
MIN_INTRA_CLUSTER_PROTEINS = 150
MIN_MODULE_PROTEINS = 200
MIN_SPECIES_PROTEINS = 100
MODULE_MIN_SPECIFICITY_PERCENT = 80
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
PANEL_DPI = 300
PANEL_WIDTH_INCHES = A4_WIDTH_INCHES / 4
PT_SIZE = 1.0
REPORT_ORDER = ["umap_intra_species_top_clusters", "tsne_intra_species_top_clusters", "umap_joint_clusters_species_specificity", "tsne_joint_clusters_species_specificity", "umap_reprojected_top_clusters", "sankey_intra_to_inter_clusters", "barplot_umap_labelled_clusters_protein_counts"]
SANKEY_HEIGHT_SCALE = 3.0
SEED = 42
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
STAR_HEADROOM = 0.35
STAR_LINESPACING = 0.55
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="ESMC Proteome Cluster Report")
parser.add_argument("--h5ad", type=str, required=True, help="Comma-separated .h5ad files written by npz_scanpy.py")
parser.add_argument("--outdir", type=str, default=None, help="Base output directory; defaults to the folder of each .h5ad")
parser.add_argument("--layout_methods", type=str, nargs="+", default=LAYOUT_METHODS, help="Layouts to plot (umap, tsne)")
parser.add_argument("--max_selected_per_species", type=int, default=MAX_SELECTED_PER_SPECIES, help="Top joint clusters labelled per species")
parser.add_argument("--min_species_proteins", type=int, default=MIN_SPECIES_PROTEINS, help="Minimum species proteins of a selected or balanced joint cluster")
parser.add_argument("--min_intra_cluster_proteins", type=int, default=MIN_INTRA_CLUSTER_PROTEINS, help="Minimum proteins of a highlighted intra cluster")
parser.add_argument("--balanced_min_share_ratio", type=float, default=BALANCED_MIN_SHARE_RATIO, help="Minimum ratio of the smallest to the largest species-normalised share of a balanced joint cluster (FC <= 1/balanced_min_share_ratio)")
parser.add_argument("--min_module_proteins", type=int, default=MIN_MODULE_PROTEINS, help="Minimum proteins of a saved module")
parser.add_argument("--module_min_specificity_percent", type=float, default=MODULE_MIN_SPECIFICITY_PERCENT, help="Minimum specificity index (percent) of a species module and of an outlined flux node")
parser.add_argument("--min_fold_change", type=float, default=MIN_FOLD_CHANGE, help="Minimum ratio of the largest to the smallest species-normalised share of a significant joint cluster")
parser.add_argument("--chi2_max_q", type=float, default=CHI2_MAX_Q, help="BH-corrected chi-square q-value below which a joint cluster can be significant")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Redraw a report whose outputs already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "h5ad_scanpy"
params = dict(layout_methods=LAYOUT_METHODS, max_selected_per_species=MAX_SELECTED_PER_SPECIES, min_species_proteins=MIN_SPECIES_PROTEINS, min_intra_cluster_proteins=MIN_INTRA_CLUSTER_PROTEINS, balanced_min_share_ratio=BALANCED_MIN_SHARE_RATIO, min_module_proteins=MIN_MODULE_PROTEINS, module_min_specificity_percent=MODULE_MIN_SPECIFICITY_PERCENT, chi2_max_q=CHI2_MAX_Q, min_fold_change=MIN_FOLD_CHANGE, overwrite=OVERWRITE)
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
def letter(i: int) -> str:
    """Name a 0-based joint cluster id by lowercase letters, spreadsheet-style (0 -> a, 26 -> aa).

    Args:
        i (int): 0-based Leiden cluster id.

    Returns:
        str: Letter name.
    """
    return (letter(i // 26 - 1) if i >= 26 else "") + chr(97 + i % 26)
def sig_stars(q: float) -> str:
    """Convert a chi-square q-value into significance asterisks.

    Args:
        q (float): BH-corrected q-value.

    Returns:
        str: ``***``, ``**``, ``*`` or an empty string when not significant.
    """
    return next((stars for cut, stars in SIG_LEVELS if q < cut), "")
def homogeneity_test(counts_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Run per joint cluster the chi-square test of homogeneity of the 2 x n_species table of its proteins and the remaining ones.

    Args:
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).

    Returns:
        tuple[pd.Series, pd.Series]: Raw p-value per cluster and whether every expected count reaches MIN_EXPECTED.
    """
    totals = counts_df.sum().to_numpy()
    tests = [chi2_contingency(np.vstack([row, totals - row])) for row in counts_df.to_numpy()]
    return pd.Series([t[1] for t in tests], index=counts_df.index), pd.Series([bool((t[3] >= MIN_EXPECTED).all()) for t in tests], index=counts_df.index)
def top_joint_clusters(counts_df: pd.DataFrame, spec_df: pd.DataFrame, significant: pd.Series, sp: str) -> list[str]:
    """Select the max_selected_per_species most specific significant joint clusters of a species, enriched in it (specificity above the uniform 1/n_species) and with at least min_species_proteins of its proteins.

    Args:
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).
        spec_df (pd.DataFrame): Specificity index per cluster (rows) and species (columns).
        significant (pd.Series): Whether each cluster passes the chi-square and fold-change gates.
        sp (str): Species of the facet.

    Returns:
        list[str]: Clusters in decreasing specificity order.
    """
    return spec_df.loc[(counts_df[sp] >= args.min_species_proteins) & significant & (spec_df[sp] > 1 / spec_df.shape[1]), sp].sort_values(ascending=False, kind="stable").head(args.max_selected_per_species).index.tolist()
def gradient_hex(value: float, norm: Normalize) -> str:
    """Map a value to a hex colour on the quantitative gradient.

    Args:
        value (float): Value to map.
        norm (Normalize): Normalisation shared by the panel.

    Returns:
        str: Hex colour.
    """
    return to_hex(COLORMAP_SPEC(norm(value)))
def style_axes(ax: plt.Axes, title: str, axis_name: str) -> None:
    """Apply the shared title, axis labels, ticks and aspect of a layout panel.

    Args:
        ax (plt.Axes): Target axes.
        title (str): Panel title.
        axis_name (str): Layout name used in the axis labels (``UMAP`` or ``t-SNE``).
    """
    ax.set_title(title, fontsize=FONT_FACET, fontweight="bold", pad=2)
    ax.set_xlabel(f"{axis_name} 1", fontsize=FONT_LABEL, labelpad=1)
    ax.set_ylabel(f"{axis_name} 2", fontsize=FONT_LABEL, labelpad=1)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_aspect("equal", "datalim")
def draw_facet(ax: plt.Axes, coords: np.ndarray, obs: pd.DataFrame, sp: str, colors: dict[str, str], labelled: list[str], sp_counts: pd.Series, axis_name: str, spec: pd.Series | None = None) -> None:
    """Draw one species facet coloured by joint cluster, or by specificity index of all its proteins when ``spec`` is given, with labelled clusters on top and their letters at the centroids of the facet species' proteins.

    Args:
        ax (plt.Axes): Target axes.
        coords (np.ndarray): Layout coordinates of all points.
        obs (pd.DataFrame): Observation table with ``species`` and ``cluster`` columns.
        sp (str): Species of the facet.
        colors (dict[str, str]): Mapping cluster -> colour.
        labelled (list[str]): Joint clusters drawn on top and labelled.
        sp_counts (pd.Series): Proteins of ``sp`` per cluster.
        axis_name (str): Layout name used in the axis labels.
        spec (pd.Series | None): Specificity index of ``sp`` per cluster; when None clusters are coloured by ``colors``.
    """
    is_sp = (obs["species"] == sp).values
    cluster_labels = obs["cluster"].astype(str)
    in_facet = cluster_labels.isin(labelled).values
    if spec is not None:
        scores = cluster_labels.map(spec).values.astype(float)
        sp_idx = np.where(is_sp)[0]
        order = sp_idx[np.lexsort((scores[sp_idx], in_facet[sp_idx]))]
        fg_colors = COLORMAP_SPEC(scores[order])
        fg_colors[:, 3] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * scores[order]
        ax.scatter(coords[order, 0], coords[order, 1], c=fg_colors, s=PT_SIZE, rasterized=True, linewidths=0)
    else:
        ax.scatter(coords[is_sp & ~in_facet, 0], coords[is_sp & ~in_facet, 1], c=COLOR_FG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0)
        fg = is_sp & in_facet
        ax.scatter(coords[fg, 0], coords[fg, 1], c=cluster_labels.map(colors).values[fg], s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0)
    max_cnt_sp = max((sp_counts[cl] for cl in labelled), default=1)
    for cl in labelled:
        cl_pts = coords[(cluster_labels == cl).values & is_sp]
        if len(cl_pts) == 0: continue
        f_size = float(np.clip(7.0 + 7.0 * (sp_counts[cl] / max_cnt_sp), 7.0, 14.0))
        txt = ax.text(cl_pts[:, 0].mean() + 1.0, cl_pts[:, 1].mean(), cl, fontsize=f_size, ha="center", va="center", fontweight="bold", color="black")
        txt.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
    style_axes(ax, f"{sp} (n={is_sp.sum():,})", axis_name)
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
    """Draw an alluvial diagram of protein fluxes from intra-species clusters (left) to selected joint clusters (right) on a semi-quantitative scale.

    Args:
        ax (plt.Axes): Target axes.
        flows (pd.DataFrame): Protein counts with intra-clusters as rows and joint clusters as columns, in plotting order.
        left_colors (dict[str, str]): Colour of each intra-cluster node.
        right_colors (dict[str, str]): Colour of each joint cluster node.
        left_labels (dict[str, str]): Text of the labelled intra-cluster nodes.
        right_labels (dict[str, str]): Text of the labelled joint cluster nodes (letter and species protein count).
        outlined (set[str]): Joint clusters drawn with a thin black border.
        title (str): Axes title.
    """
    left_sizes, right_sizes = flows.sum(axis=1), flows.sum(axis=0)
    (left_heights, left_offsets), (right_heights, right_offsets) = flux_layout(left_sizes), flux_layout(right_sizes)
    xs = np.linspace(FLUX_NODE_WIDTH, 1 - FLUX_NODE_WIDTH, 60)
    t = (1 - np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))) / 2
    for side, heights, offs, x0, cols, labs in (("l", left_heights, left_offsets, 0.0, left_colors, left_labels), ("r", right_heights, right_offsets, 1 - FLUX_NODE_WIDTH, right_colors, right_labels)):
        for lab, h in heights.items():
            ax.fill_betweenx([offs[lab], offs[lab] + h], x0, x0 + FLUX_NODE_WIDTH, facecolor=cols[lab], edgecolor="black" if side == "r" and lab in outlined else "none", lw=FLUX_BORDER_WIDTH)
            if lab in labs: ax.text(-0.02 if side == "l" else 1.02, offs[lab] + h / 2, labs[lab], fontsize=FONT_LABEL * FLUX_FONT_SCALE, fontweight="bold", ha="right" if side == "l" else "left", va="center")
    left_cursor, right_cursor = left_offsets.copy(), right_offsets.copy()
    for (i, j), v in flows.stack().items():
        if v == 0: continue
        left_band, right_band = v / left_sizes[i] * left_heights[i], v / right_sizes[j] * right_heights[j]
        ax.fill_between(xs, left_cursor[i] + (right_cursor[j] - left_cursor[i]) * t, left_cursor[i] + left_band + (right_cursor[j] + right_band - left_cursor[i] - left_band) * t, color=right_colors[j], alpha=FLUX_ALPHA, lw=0)
        left_cursor[i] += left_band
        right_cursor[j] += right_band
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(1, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=FONT_FACET * FLUX_FONT_SCALE, fontweight="bold", pad=2)
def plot_counts_bar(top_counts: pd.DataFrame, species_list: list[str], stars: pd.Series, title: str, path: Path) -> None:
    """Plot per-species protein counts of the selected joint clusters with chi-square significance asterisks stacked vertically over their bars.

    Args:
        top_counts (pd.DataFrame): Protein counts per selected joint cluster (rows) and species (columns), in plotting order.
        species_list (list[str]): Species to plot.
        stars (pd.Series): Significance asterisks per selected joint cluster.
        title (str): Plot title.
        path (Path): Output path.
    """
    n_sp = len(species_list)
    x, width = np.arange(len(top_counts)), 0.8 / n_sp
    sp_colors = {sp: PALETTE[i % len(PALETTE)] for i, sp in enumerate(species_list)}
    fig, ax = plt.subplots(figsize=fig_size(n_sp), dpi=PANEL_DPI, layout="constrained")
    for i, sp in enumerate(species_list):
        ax.bar(x - 0.4 + width * (i + 0.5), top_counts[sp].values, width=width, color=sp_colors[sp], linewidth=0)
    group_max = top_counts[species_list].max(axis=1).values
    ax.set_ylim(0, group_max.max() * (1 + STAR_HEADROOM))
    for xi, ymax, st in zip(x, group_max, stars.reindex(top_counts.index).fillna("")):
        ax.annotate("\n".join(st), (xi, ymax), xytext=(0, 1), textcoords="offset points", fontsize=FONT_STAR, fontweight="bold", ha="center", va="bottom", linespacing=STAR_LINESPACING)
    ax.set_xticks(x)
    ax.set_xticklabels(top_counts.index, fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlim(-0.6, len(top_counts) - 0.4)
    ax.set_xlabel("Joint cluster\n$\\chi^2$ homogeneity across species, BH: * q<0.05, ** q<0.01, *** q<0.001", fontsize=FONT_LABEL, labelpad=2)
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
def reproject(adata: ad.AnnData, keep: np.ndarray) -> np.ndarray:
    """Optimise a 2D UMAP layout of a subset of proteins on the stored joint kNN graph restricted to them, without reclustering.

    Args:
        adata (ad.AnnData): Saved clusters carrying ``obsp["connectivities"]`` and ``obsm["X_umap_2d"]``.
        keep (np.ndarray): Boolean mask of the proteins to re-project.

    Returns:
        np.ndarray: 2D layout of the kept proteins.
    """
    sub = adata[keep]
    a, b = find_ab_params(1.0, float(adata.uns["params"]["umap_min_dist"]))
    coords, _ = simplicial_set_embedding(data=np.asarray(sub.obsm["X_umap_2d"]), graph=sub.obsp["connectivities"].tocoo(), n_components=2, initial_alpha=1.0, a=a, b=b, gamma=1.0, negative_sample_rate=5, n_epochs=None, init="spectral", random_state=np.random.RandomState(SEED), metric="euclidean", metric_kwds={}, densmap=False, densmap_kwds={}, output_dens=False)
    return coords
###################################### REPORT
def report(h5ad_path: Path) -> None:
    """Write tables, modules, panels, fluxes and the merged PDF of one saved clustering.

    Args:
        h5ad_path (Path): .h5ad written by npz_scanpy.py.
    """
    head = ad.read_h5ad(h5ad_path, backed="r")
    param_tag = head.uns["param_tag"]
    head.file.close()
    target_dir = Path(args.outdir) / h5ad_path.parent.name if args.outdir else h5ad_path.parent
    done = [target_dir / f"protein_clusters_metadata__{param_tag}.csv", target_dir / f"{target_dir.name}_scanpy.pdf"] + [target_dir / f"{m}_joint_clusters_species_specificity__{param_tag}.pdf" for m in args.layout_methods]
    src_mtime = max(Path(__file__).stat().st_mtime, h5ad_path.stat().st_mtime)
    if not args.overwrite and all(p.exists() and p.stat().st_mtime > src_mtime for p in done):
        print(f"[PYTHON-INFO] Report of {h5ad_path.name} already exists in {target_dir} and is newer than the script. Skipping.")
        return
    adata = ad.read_h5ad(h5ad_path)
    modules_dir, intra_dir = target_dir / "protein-modules", target_dir / "intra_group"
    for d in (modules_dir, intra_dir): d.mkdir(parents=True, exist_ok=True)
    obs = adata.obs
    species_list = list(obs["species"].unique())
    n_sp = len(species_list)
    print(f"[PYTHON-INFO][DATA-SHAPE] {h5ad_path.name}: {adata.n_obs:,} proteins, {n_sp} species, obsm: {', '.join(adata.obsm.keys())}")
    ###################################### JOINT_SELECTION
    ids = obs["cluster"].astype(int)
    obs["cluster"] = pd.Categorical(ids.map(letter), categories=[letter(i) for i in sorted(ids.unique())])
    cluster_labels, intra_labels, species_labels = obs["cluster"].astype(str), obs["intra_cluster"].astype(str), obs["species"].astype(str)
    counts_df = pd.crosstab(obs["cluster"], obs["species"])[species_list]
    counts_df.index = counts_df.index.astype(str)
    share_df = counts_df / counts_df.sum()
    spec_df = share_df.div(share_df.sum(axis=1), axis=0)
    fold_change = share_df.max(axis=1) / share_df.min(axis=1)
    p_vals, valid = homogeneity_test(counts_df)
    chi2_q = pd.Series(false_discovery_control(p_vals.to_numpy()), index=counts_df.index)
    significant = valid & (chi2_q < args.chi2_max_q) & (fold_change >= args.min_fold_change)
    balanced = (fold_change <= 1 / args.balanced_min_share_ratio) & (counts_df.min(axis=1) >= args.min_species_proteins)
    ranked = {sp: top_joint_clusters(counts_df, spec_df, significant, sp) for sp in species_list}
    selected = list(dict.fromkeys(cl for cls in ranked.values() for cl in cls))
    colors = {cl: PALETTE[i % len(PALETTE)] for i, cl in enumerate(selected)}
    print(f"[PYTHON-INFO][CHI2] {valid.sum()}/{len(valid)} joint clusters testable, {(valid & (chi2_q < args.chi2_max_q)).sum()} with q < {args.chi2_max_q}, {significant.sum()} also with fold change >= {args.min_fold_change}")
    print(f"[PYTHON-INFO][BALANCED] {balanced.sum()} balanced joint clusters: {', '.join(balanced.index[balanced])}")
    print(f"[PYTHON-INFO][SELECTION] {len(selected)} selected joint clusters: {', '.join(selected)}")
    ###################################### TABLES_AND_MODULES
    with open(target_dir / f"clustering_metrics__{param_tag}.txt", "w") as f:
        f.writelines(f"{k}: {v}\n" for k, v in adata.uns["metrics"].items())
    pd.DataFrame([(sp, cl, rk + 1, counts_df.loc[cl, sp], spec_df.loc[cl, sp], chi2_q[cl], fold_change[cl]) for sp, cls in ranked.items() for rk, cl in enumerate(cls)], columns=["species", "cluster", "rank", "protein_count", "specificity", "chi2_q", "fold_change"]).to_csv(target_dir / f"species_top_clusters__{param_tag}.csv", index=False)
    obs[["protein_id", "species", "cluster", "intra_cluster"]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    stats = pd.DataFrame({"chi2_q": chi2_q, "chi2_valid": valid, "fold_change": fold_change, "significant": significant, "balanced": balanced, "selected": counts_df.index.isin(selected)}, index=counts_df.index)
    pd.concat({"protein_count": counts_df.stack(), "specificity": spec_df.stack()}, axis=1).rename_axis(["cluster", "species"]).reset_index().join(stats, on="cluster").to_csv(target_dir / f"species_cluster_enrichment__{param_tag}.csv", index=False)
    markers = adata.uns["markers"].astype({"cluster": str})
    inter = markers["analysis"] == "inter_group"
    markers.loc[inter, "cluster"] = markers.loc[inter, "cluster"].astype(int).map(letter)
    markers.to_csv(target_dir / f"channel_importance_rankings__{param_tag}.csv", index=False)
    for sp in species_list:
        for cl in counts_df.index[significant & (spec_df[sp] >= args.module_min_specificity_percent / 100) & (counts_df[sp] >= args.min_module_proteins)]:
            obs.loc[(species_labels == sp).values & (cluster_labels == cl).values, "protein_id"].to_csv(modules_dir / f"{sp}_{counts_df.loc[cl, sp]:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    ###################################### INTRA_SUMMARY
    intra = {}
    for sp in species_list:
        is_sp = (obs["species"] == sp).values
        if obs.loc[is_sp, "intra_cluster"].isna().all(): continue
        counts_intra = intra_labels[is_sp].value_counts()
        intra[sp] = (is_sp, counts_intra.size, counts_intra[counts_intra >= args.min_intra_cluster_proteins])
        for cl, cnt in counts_intra.items():
            if cnt >= args.min_module_proteins:
                obs.loc[is_sp & (intra_labels == cl).values, "protein_id"].to_csv(modules_dir / f"intra_{sp}_{cnt:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    ###################################### LAYOUT_PANELS
    for m in tqdm(args.layout_methods, desc="Layout panels"):
        if f"X_{m}_2d" not in adata.obsm:
            print(f"[PYTHON-INFO][PANELS] Skipping {m}: X_{m}_2d not in {h5ad_path.name}")
            continue
        name = METHOD_NAMES.get(m, m)
        fig, gs = row_figure(n_sp, f"Inter-species ({name})")
        for idx, sp in enumerate(species_list):
            draw_facet(fig.add_subplot(gs[0, idx]), adata.obsm[f"X_{m}_2d"], obs, sp, colors, ranked[sp], counts_df[sp], name, spec_df[sp])
        add_colorbar(fig.add_subplot(gs[-1, :]), Normalize(0, 1), CBAR_LABEL)
        save_fig(fig, target_dir / f"{m}_joint_clusters_species_specificity__{param_tag}.png")
        plt.close(fig)
        if not intra: continue
        top_vals = [cnt for *_, top in intra.values() for cnt in top.values] or [0]
        norm_size = Normalize(min(top_vals), max(top_vals) if max(top_vals) > min(top_vals) else min(top_vals) + 1)
        fig, gs = row_figure(n_sp, f"Intra-species ({name})")
        for idx, (sp, (is_sp, _, top)) in enumerate(intra.items()):
            ax = fig.add_subplot(gs[0, idx])
            coords, intra_cls = adata.obsm[f"X_{m}_2d_intra"][is_sp], intra_labels.values[is_sp]
            is_shown = np.isin(intra_cls, list(top.index))
            ax.scatter(coords[~is_shown, 0], coords[~is_shown, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
            for cl, cnt in top.items():
                cl_pts = coords[intra_cls == cl]
                ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=gradient_hex(cnt, norm_size), s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
                txt = ax.text(np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1]), cl, fontsize=FONT_LABEL, ha="center", va="center", fontweight="bold", color="black", zorder=6)
                txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
            style_axes(ax, f"{sp} (n={is_sp.sum():,})", name)
        add_colorbar(fig.add_subplot(gs[-1, :]), norm_size, "Proteins per highlighted cluster")
        save_fig(fig, intra_dir / f"{m}_intra_species_top_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### INTRA_INTER_FLUX
    if intra:
        fig, gs = row_figure(n_sp, "Intra- to Inter-species cluster fluxes", SANKEY_HEIGHT_SCALE, font_scale=FLUX_FONT_SCALE)
        norm_spec, spec_cut = Normalize(0, 1), args.module_min_specificity_percent / 100
        for idx, (sp, (is_sp, n_cl, top)) in enumerate(intra.items()):
            ax = fig.add_subplot(gs[0, idx])
            title = f"{sp}\n(n={is_sp.sum():,}, k={n_cl})"
            full = pd.crosstab(intra_labels[is_sp], cluster_labels[is_sp])
            sp_ranked = [cl for cl in ranked[sp] if cl in full.columns]
            if not sp_ranked:
                ax.axis("off")
                ax.set_title(title, fontsize=FONT_FACET * FLUX_FONT_SCALE, fontweight="bold", pad=2)
                continue
            flows = full[sp_ranked]
            flows = flows[flows.sum(axis=1) > 0]
            l_order = [cl for cl in top.index if cl in flows.index] + flows.sum(axis=1).drop(list(top.index), errors="ignore").sort_values(ascending=False).index.tolist()
            flows = flows.loc[l_order]
            best = flows.idxmax(axis=1)
            spec = spec_df[sp]
            right_colors = {cl: gradient_hex(spec[cl], norm_spec) for cl in sp_ranked}
            left_colors = {cl: right_colors[best[cl]] if flows.loc[cl, best[cl]] > full.loc[cl].sum() / 2 else COLOR_BG for cl in l_order}
            draw_flux(ax, flows, left_colors, right_colors, {cl: cl for cl in top.index if cl in flows.index}, {cl: f"{cl} ({counts_df.loc[cl, sp]:,})" for cl in sp_ranked}, {cl for cl in sp_ranked if spec[cl] >= spec_cut}, title)
        add_colorbar(fig.add_subplot(gs[-1, :]), norm_spec, f"{CBAR_LABEL}; black border: $S_{{i}} \\geq$ {spec_cut:.2f}")
        save_fig(fig, target_dir / f"sankey_intra_to_inter_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### REPROJECTION
    keep = cluster_labels.isin(selected).values
    if "X_umap_2d" in adata.obsm and keep.sum() > 2:
        obs_selected = obs[keep]
        print(f"[PYTHON-INFO][SELECTION] Re-projecting {keep.sum():,} proteins of {len(selected)} selected joint clusters from all species: {', '.join(selected)}")
        coords_selected = reproject(adata, keep)
        species_selected = [sp for sp in species_list if (obs_selected["species"] == sp).any()]
        fig, gs = row_figure(len(species_selected), "Selected inter-species clusters (re-projected)", strip=False)
        for idx, sp in enumerate(species_selected):
            draw_facet(fig.add_subplot(gs[0, idx]), coords_selected, obs_selected, sp, colors, selected, counts_df[sp], "UMAP")
        add_bottom_legend(fig, colors, "Joint cluster")
        save_fig(fig, target_dir / f"umap_reprojected_top_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### TOP_COUNTS_BARPLOT
    if selected:
        top_counts = counts_df.loc[selected, species_list]
        top_counts = top_counts.loc[top_counts.sum(axis=1).sort_values(ascending=False, kind="stable").index]
        plot_counts_bar(top_counts, species_list, chi2_q.loc[selected].map(sig_stars), "Protein per cluster (number by species)", target_dir / f"barplot_umap_labelled_clusters_protein_counts__{param_tag}.png")
    ###################################### PDF_REPORT
    out = target_dir / f"{target_dir.name}_scanpy.pdf"
    pdfs = sorted((p for p in target_dir.rglob(f"*__{param_tag}.pdf") if p != out), key=lambda p: next((i for i, s in enumerate(REPORT_ORDER) if p.stem.startswith(s)), len(REPORT_ORDER)))
    merge_pdfs(pdfs, out)
    print(f"[PYTHON-INFO][REPORT] Merged {len(pdfs)} PDFs into {out}")
###################################### MAIN
def main() -> None:
    """Report every .h5ad passed on the command line."""
    start_time = time.time()
    for path in tqdm([Path(p.strip()) for p in args.h5ad.split(",") if p.strip()], desc="Reports"):
        report(path)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
