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
Inputs carry the ``input_type`` of the query table. Without probe inputs (comparative mode) the report is the species comparison alone; with at least one probe (focus mode) it adds the functional focus layer. the species criteria above use the ``target`` inputs only, while ``probe`` inputs (functional references) take part in the joint clusters and are summarised as a second layer, the functional focus. The focus set F is every probe protein plus every target protein whose own UniProt annotation matches the focus clause (``focus_tier`` annotated, dw_uniprot_focus.py), N is the number of proteins and n_k the size of joint cluster k:
    9. focus count f_k (probe and annotated proteins in k), focus share f_k / |F|, focus fraction f_k / n_k and fold enrichment (f_k / n_k) / (|F| / N);
    10. focus test: one-sided hypergeometric P(X >= f_k), BH-corrected across all joint clusters (focus_q);
    11. focus cluster: focus_q < focus_max_q and f_k >= min_focus_proteins, keeping the max_focus_clusters highest focus shares;
    12. capture: target proteins inside focus clusters, overall and per target, split into annotated (agreement with the annotation, reported but circular) and unannotated (candidate focus-like proteins); the report fails, after writing everything, when the overall capture is below min_capture_percent;
    13. focus selection (``focus_selection``, target proteins only, probe proteins excluded): I for every target protein with its own focus annotation, E (extended through probing) for every other target protein inside a focus cluster, none otherwise;
    14. gravity: median ``focus_similarity`` (cosine similarity to the focus centroid, computed by npz_scanpy.py) of the target proteins of every joint cluster, overall and per target, ranked; it is descriptive and does not take part in any selection.
A focus run is also compared with its baseline run, the run of the same targets, model and parameters without probes (supp_01 builds both): on the target proteins shared by the two runs, the agreement of their joint clusters (adjusted Rand index, normalised mutual information), the protein flow from every baseline cluster to every focus-run cluster with its Jaccard index, and the fate of every selected cluster (best-matching cluster of the other run, Jaccard, still selected), written as paired_* files in the focus run folder.
Focus outputs are separate files: focus_cluster_enrichment, focus_capture, focus_gravity and focus_selection tables, focus-modules/ protein lists (per focus cluster and per target selection), focus selection layouts with one facet per target coloured by ``focus_selection`` (I red, E dark red, none), focus cluster layouts with one facet per input, a re-projection of the focus clusters and a barplot with focus significance asterisks; in the focus cluster panels the UniProt-annotated target proteins (the special tier) are ringed in black. In focus mode the species panels (b), (c) and (e) also get one facet or bar group per probe, drawn for reference only: probes are coloured by selected cluster instead of specificity index and never enter the species criteria.
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
from scipy.stats import chi2_contingency, false_discovery_control, hypergeom
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
from tqdm import tqdm
from typing import NamedTuple
from umap.umap_ import find_ab_params, simplicial_set_embedding
import warnings
import yaml
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
A4_WIDTH_INCHES = 11.69
ALPHA_BG = 0.50
ALPHA_MAX = 0.95
ALPHA_MIN = 0.55
ANNOTATED_LINEWIDTH = 0.3
ANNOTATED_SIZE_SCALE = 4.0
BALANCED_MIN_SHARE_RATIO = 0.6
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
CHI2_MAX_Q = 0.05
COLORMAP_SPEC = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
COLOR_ANNOTATED = "#000000"
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
FOCUS_MAX_Q = 0.05
FONT_FACET = 12
FONT_LABEL = 10
FONT_STAR = 14
FONT_TICK = 9
FONT_TITLE = 14
LAYOUT_METHODS = ["umap", "tsne"]
LEGEND_CHAR_INCHES = 0.08
LEGEND_COL_INCHES = 0.4
LEGEND_MARKER = 7
MAX_FOCUS_CLUSTERS = 6
MAX_SELECTED_PER_SPECIES = 6
METHOD_NAMES = {"umap": "UMAP", "tsne": "t-SNE"}
MIN_CAPTURE_PERCENT = 1.0
MIN_EXPECTED = 5
MIN_FOCUS_PROTEINS = 50
MIN_FOLD_CHANGE = 2.0
MIN_INTRA_CLUSTER_PROTEINS = 150
MIN_MODULE_PROTEINS = 200
MIN_SPECIES_PROTEINS = 100
MODULE_MIN_SPECIFICITY_PERCENT = 80
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
PALETTE_ALT = PALETTE[::2] + PALETTE[1::2]
PANEL_DPI = 300
PANEL_WIDTH_INCHES = A4_WIDTH_INCHES / 4
PT_SIZE = 1.0
REPORT_ORDER = ["umap_intra_species_top_clusters", "tsne_intra_species_top_clusters", "umap_joint_clusters_species_specificity", "tsne_joint_clusters_species_specificity", "umap_reprojected_top_clusters", "sankey_intra_to_inter_clusters", "barplot_umap_labelled_clusters_protein_counts", "umap_focus_tiers", "tsne_focus_tiers", "umap_focus_clusters", "tsne_focus_clusters", "umap_reprojected_focus_clusters", "barplot_focus_clusters_protein_counts"]
RING_LABEL = "UniProt-annotated"
SANKEY_HEIGHT_SCALE = 3.0
SEED = 42
SELECTION_COLORS = {"none": COLOR_BG, "E": "#8B0000", "I": "#FF0000"}
SELECTION_LABELS = {"none": "none", "E": "E: extended by probing", "I": "I: immune annotated"}
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
parser.add_argument("--focus_max_q", type=float, default=FOCUS_MAX_Q, help="BH-corrected hypergeometric q-value below which a joint cluster can be a focus cluster")
parser.add_argument("--min_focus_proteins", type=int, default=MIN_FOCUS_PROTEINS, help="Minimum focus proteins (probe and annotated) of a focus cluster")
parser.add_argument("--max_focus_clusters", type=int, default=MAX_FOCUS_CLUSTERS, help="Focus clusters kept, by decreasing focus share")
parser.add_argument("--min_capture_percent", type=float, default=MIN_CAPTURE_PERCENT, help="Minimum percent of target proteins inside focus clusters; below it the report fails")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Redraw a report whose outputs already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "h5ad_scanpy"
params = dict(layout_methods=LAYOUT_METHODS, max_selected_per_species=MAX_SELECTED_PER_SPECIES, min_species_proteins=MIN_SPECIES_PROTEINS, min_intra_cluster_proteins=MIN_INTRA_CLUSTER_PROTEINS, balanced_min_share_ratio=BALANCED_MIN_SHARE_RATIO, min_module_proteins=MIN_MODULE_PROTEINS, module_min_specificity_percent=MODULE_MIN_SPECIFICITY_PERCENT, chi2_max_q=CHI2_MAX_Q, min_fold_change=MIN_FOLD_CHANGE, focus_max_q=FOCUS_MAX_Q, min_focus_proteins=MIN_FOCUS_PROTEINS, max_focus_clusters=MAX_FOCUS_CLUSTERS, min_capture_percent=MIN_CAPTURE_PERCENT, overwrite=OVERWRITE)
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
def facet_title(obs: pd.DataFrame, is_sp: np.ndarray, sp: str) -> str:
    """Title a facet by its input and protein count, suffixing probe inputs.

    Args:
        obs (pd.DataFrame): Observation table, optionally with ``input_type``.
        is_sp (np.ndarray): Boolean mask of the facet proteins.
        sp (str): Input of the facet.

    Returns:
        str: Facet title.
    """
    probe = "input_type" in obs and (obs["input_type"].astype(str).values[is_sp] == "probe").any()
    return f"{sp}{' (probe)' if probe else ''} (n={is_sp.sum():,})"
def draw_facet(ax: plt.Axes, coords: np.ndarray, obs: pd.DataFrame, sp: str, colors: dict[str, str], labelled: list[str], sp_counts: pd.Series, axis_name: str, spec: pd.Series | None = None, annotated: np.ndarray | None = None) -> None:
    """Draw one species facet coloured by joint cluster, or by specificity index of all its proteins when ``spec`` is given, with labelled clusters on top, the annotated proteins ringed when ``annotated`` is given and the cluster letters at the centroids of the facet species' proteins.

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
        annotated (np.ndarray | None): Boolean mask of the UniProt-annotated focus proteins, ringed in black.
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
    if annotated is not None:
        ring = is_sp & annotated
        ax.scatter(coords[ring, 0], coords[ring, 1], s=PT_SIZE * ANNOTATED_SIZE_SCALE, facecolors="none", edgecolors=COLOR_ANNOTATED, linewidths=ANNOTATED_LINEWIDTH, rasterized=True)
    max_cnt_sp = max((sp_counts[cl] for cl in labelled), default=1)
    for cl in labelled:
        cl_pts = coords[(cluster_labels == cl).values & is_sp]
        if len(cl_pts) == 0: continue
        f_size = float(np.clip(7.0 + 7.0 * (sp_counts[cl] / max_cnt_sp), 7.0, 14.0))
        txt = ax.text(cl_pts[:, 0].mean() + 1.0, cl_pts[:, 1].mean(), cl, fontsize=f_size, ha="center", va="center", fontweight="bold", color="black")
        txt.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
    style_axes(ax, facet_title(obs, is_sp, sp), axis_name)
def add_bottom_legend(fig: plt.Figure, colors: dict[str, str], title: str, marker: str = "o", font_scale: float = 1.0, ring_label: str | None = None) -> None:
    """Add a label -> colour legend below the panels, with as many columns as fit the figure width.

    Args:
        fig (plt.Figure): Constrained-layout figure.
        colors (dict[str, str]): Mapping label -> colour, in legend order.
        title (str): Legend title.
        marker (str): Marker of the legend handles.
        font_scale (float): Multiplier applied to the legend font sizes.
        ring_label (str | None): Label of an extra black-ringed handle marking the annotated proteins.
    """
    if not colors: return
    handles = [Line2D([], [], ls="", marker=marker, color=c, markersize=LEGEND_MARKER * font_scale, label=lab) for lab, c in colors.items()]
    if ring_label: handles.append(Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=COLOR_ANNOTATED, markersize=LEGEND_MARKER * font_scale, label=ring_label))
    ncol = max(1, min(len(handles), int(fig.get_figwidth() / ((LEGEND_COL_INCHES + LEGEND_CHAR_INCHES * max(len(h.get_label()) for h in handles)) * font_scale))))
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
def plot_counts_bar(top_counts: pd.DataFrame, species_list: list[str], stars: pd.Series, title: str, path: Path, xlabel: str = "Joint cluster\n$\\chi^2$ homogeneity across species, BH: * q<0.05, ** q<0.01, *** q<0.001", legend_title: str = "Species") -> None:
    """Plot per-species protein counts of the selected joint clusters with significance asterisks stacked vertically over their bars.

    Args:
        top_counts (pd.DataFrame): Protein counts per selected joint cluster (rows) and species (columns), in plotting order.
        species_list (list[str]): Species to plot.
        stars (pd.Series): Significance asterisks per selected joint cluster.
        title (str): Plot title.
        path (Path): Output path.
        xlabel (str): X-axis label naming the test behind the asterisks.
        legend_title (str): Title of the colour legend.
    """
    n_sp = len(species_list)
    x, width = np.arange(len(top_counts)), 0.8 / n_sp
    sp_colors = {sp: PALETTE_ALT[i % len(PALETTE_ALT)] for i, sp in enumerate(species_list)}
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
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, labelpad=2)
    ax.set_ylabel("Proteins", fontsize=FONT_LABEL, labelpad=2)
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold")
    add_bottom_legend(fig, sp_colors, legend_title, "s")
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
class JointStats(NamedTuple):
    """Per joint cluster statistics and selection of one saved clustering."""
    counts: pd.DataFrame
    spec: pd.DataFrame
    fold_change: pd.Series
    chi2_q: pd.Series
    valid: pd.Series
    significant: pd.Series
    balanced: pd.Series
    ranked: dict[str, list[str]]
    selected: list[str]
    colors: dict[str, str]
def joint_stats(obs: pd.DataFrame, species_list: list[str]) -> JointStats:
    """Rename joint clusters by letters in place and compute their counts, specificity, fold change, chi-square q-values and selection.

    Args:
        obs (pd.DataFrame): Observation table with ``species`` and integer ``cluster`` columns, modified in place.
        species_list (list[str]): Species in column order.

    Returns:
        JointStats: Statistics and selection of every joint cluster.
    """
    ids = obs["cluster"].astype(int)
    obs["cluster"] = pd.Categorical(ids.map(letter), categories=[letter(i) for i in sorted(ids.unique())])
    counts_df = pd.crosstab(obs["cluster"], obs["species"])[species_list]
    counts_df.index = counts_df.index.astype(str)
    counts_df = counts_df[counts_df.sum(axis=1) > 0]
    share_df = counts_df / counts_df.sum()
    spec_df = share_df.div(share_df.sum(axis=1), axis=0)
    fold_change = share_df.max(axis=1) / share_df.min(axis=1)
    p_vals, valid = homogeneity_test(counts_df)
    chi2_q = pd.Series(false_discovery_control(p_vals.to_numpy()), index=counts_df.index)
    significant = valid & (chi2_q < args.chi2_max_q) & (fold_change >= args.min_fold_change)
    balanced = (fold_change <= 1 / args.balanced_min_share_ratio) & (counts_df.min(axis=1) >= args.min_species_proteins)
    ranked = {sp: top_joint_clusters(counts_df, spec_df, significant, sp) for sp in species_list}
    selected = list(dict.fromkeys(cl for cls in ranked.values() for cl in cls))
    print(f"[PYTHON-INFO][CHI2] {valid.sum()}/{len(valid)} joint clusters testable, {(valid & (chi2_q < args.chi2_max_q)).sum()} with q < {args.chi2_max_q}, {significant.sum()} also with fold change >= {args.min_fold_change}")
    print(f"[PYTHON-INFO][BALANCED] {balanced.sum()} balanced joint clusters: {', '.join(balanced.index[balanced])}")
    print(f"[PYTHON-INFO][SELECTION] {len(selected)} selected joint clusters: {', '.join(selected)}")
    return JointStats(counts_df, spec_df, fold_change, chi2_q, valid, significant, balanced, ranked, selected, {cl: PALETTE_ALT[i % len(PALETTE_ALT)] for i, cl in enumerate(selected)})
def write_tables(adata: ad.AnnData, js: JointStats, target_dir: Path, modules_dir: Path, param_tag: str) -> None:
    """Write the metrics, top-cluster, metadata, enrichment and channel tables and the protein lists of every species module.

    Args:
        adata (ad.AnnData): Saved clusters with letter-named joint clusters.
        js (JointStats): Joint cluster statistics.
        target_dir (Path): Report folder.
        modules_dir (Path): Folder of the protein module lists.
        param_tag (str): Parameter tag of the clustering.
    """
    obs = adata.obs
    with open(target_dir / f"clustering_metrics__{param_tag}.txt", "w") as f:
        f.writelines(f"{k}: {v}\n" for k, v in adata.uns["metrics"].items())
    top_rows = [(sp, cl, rk + 1, js.counts.loc[cl, sp], js.spec.loc[cl, sp], js.chi2_q[cl], js.fold_change[cl]) for sp, cls in js.ranked.items() for rk, cl in enumerate(cls)]
    pd.DataFrame(top_rows, columns=["species", "cluster", "rank", "protein_count", "specificity", "chi2_q", "fold_change"]).to_csv(target_dir / f"species_top_clusters__{param_tag}.csv", index=False)
    obs[[c for c in ("protein_id", "species", "input_type", "focus_tier", "focus_selection", "focus_similarity", "cluster", "intra_cluster") if c in obs]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    stats = pd.DataFrame({"chi2_q": js.chi2_q, "chi2_valid": js.valid, "fold_change": js.fold_change, "significant": js.significant, "balanced": js.balanced, "selected": js.counts.index.isin(js.selected)}, index=js.counts.index)
    enrichment = pd.concat({"protein_count": js.counts.stack(), "specificity": js.spec.stack()}, axis=1).rename_axis(["cluster", "species"]).reset_index().join(stats, on="cluster")
    enrichment.to_csv(target_dir / f"species_cluster_enrichment__{param_tag}.csv", index=False)
    markers = adata.uns["markers"].astype({"cluster": str})
    inter = markers["analysis"] == "inter_group"
    markers.loc[inter, "cluster"] = markers.loc[inter, "cluster"].astype(int).map(letter)
    markers.to_csv(target_dir / f"channel_importance_rankings__{param_tag}.csv", index=False)
    cluster_labels, species_labels = obs["cluster"].astype(str), obs["species"].astype(str)
    for sp in js.counts.columns:
        for cl in js.counts.index[js.significant & (js.spec[sp] >= args.module_min_specificity_percent / 100) & (js.counts[sp] >= args.min_module_proteins)]:
            obs.loc[(species_labels == sp).values & (cluster_labels == cl).values, "protein_id"].to_csv(modules_dir / f"{sp}_{js.counts.loc[cl, sp]:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
def intra_summary(obs: pd.DataFrame, species_list: list[str], modules_dir: Path, param_tag: str) -> dict[str, tuple[np.ndarray, int, pd.Series]]:
    """Summarise the intra-species clusters of every species and write the protein lists of the large ones.

    Args:
        obs (pd.DataFrame): Observation table with ``species``, ``intra_cluster`` and ``protein_id`` columns.
        species_list (list[str]): Species to summarise.
        modules_dir (Path): Folder of the protein module lists.
        param_tag (str): Parameter tag of the clustering.

    Returns:
        dict[str, tuple[np.ndarray, int, pd.Series]]: Species -> (protein mask, number of intra clusters, counts of the clusters with at least min_intra_cluster_proteins proteins).
    """
    intra, intra_labels = {}, obs["intra_cluster"].astype(str)
    for sp in species_list:
        is_sp = (obs["species"] == sp).values
        if obs.loc[is_sp, "intra_cluster"].isna().all(): continue
        counts_intra = intra_labels[is_sp].value_counts()
        intra[sp] = (is_sp, counts_intra.size, counts_intra[counts_intra >= args.min_intra_cluster_proteins])
        for cl, cnt in counts_intra.items():
            if cnt >= args.min_module_proteins:
                obs.loc[is_sp & (intra_labels == cl).values, "protein_id"].to_csv(modules_dir / f"intra_{sp}_{cnt:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    return intra
def plot_intra_panel(adata: ad.AnnData, intra: dict[str, tuple[np.ndarray, int, pd.Series]], method: str, name: str, path: Path) -> None:
    """Draw the intra-species layouts of one method with the large intra clusters coloured by protein count.

    Args:
        adata (ad.AnnData): Saved clusters carrying ``obsm["X_<method>_2d_intra"]``.
        intra (dict[str, tuple[np.ndarray, int, pd.Series]]): Output of intra_summary.
        method (str): Layout method key.
        name (str): Display name of the method.
        path (Path): Output PNG path.
    """
    intra_labels = adata.obs["intra_cluster"].astype(str)
    top_vals = [cnt for *_, top in intra.values() for cnt in top.values] or [0]
    norm_size = Normalize(min(top_vals), max(top_vals) if max(top_vals) > min(top_vals) else min(top_vals) + 1)
    fig, gs = row_figure(len(intra), f"Intra-species ({name})")
    for idx, (sp, (is_sp, _, top)) in enumerate(intra.items()):
        ax = fig.add_subplot(gs[0, idx])
        coords, intra_cls = adata.obsm[f"X_{method}_2d_intra"][is_sp], intra_labels.values[is_sp]
        is_shown = np.isin(intra_cls, list(top.index))
        ax.scatter(coords[~is_shown, 0], coords[~is_shown, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
        for cl, cnt in top.items():
            cl_pts = coords[intra_cls == cl]
            ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=gradient_hex(cnt, norm_size), s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
            txt = ax.text(np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1]), cl, fontsize=FONT_LABEL, ha="center", va="center", fontweight="bold", color="black", zorder=6)
            txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
        style_axes(ax, f"{sp} (n={is_sp.sum():,})", name)
    add_colorbar(fig.add_subplot(gs[-1, :]), norm_size, "Proteins per highlighted cluster")
    save_fig(fig, path)
    plt.close(fig)
def plot_layout_panels(adata: ad.AnnData, js: JointStats, counts: pd.DataFrame, intra: dict[str, tuple[np.ndarray, int, pd.Series]], target_dir: Path, intra_dir: Path, param_tag: str) -> None:
    """Draw for every layout method the joint layouts coloured by specificity index, plus one facet per probe coloured by selected cluster, and the intra-species layouts.

    Args:
        adata (ad.AnnData): Saved clusters with letter-named joint clusters.
        js (JointStats): Joint cluster statistics.
        counts (pd.DataFrame): Protein counts per cluster (rows) and input (columns, targets then probes); its columns are the facets.
        intra (dict[str, tuple[np.ndarray, int, pd.Series]]): Output of intra_summary.
        target_dir (Path): Report folder.
        intra_dir (Path): Folder of the intra-species panels.
        param_tag (str): Parameter tag of the clustering.
    """
    facets = list(counts.columns)
    for m in tqdm(args.layout_methods, desc="Layout panels"):
        if f"X_{m}_2d" not in adata.obsm:
            print(f"[PYTHON-INFO][PANELS] Skipping {m}: X_{m}_2d not in the .h5ad")
            continue
        name = METHOD_NAMES.get(m, m)
        fig, gs = row_figure(len(facets), f"Inter-species ({name})")
        for idx, sp in enumerate(facets):
            if sp in js.spec: draw_facet(fig.add_subplot(gs[0, idx]), adata.obsm[f"X_{m}_2d"], adata.obs, sp, js.colors, js.ranked[sp], counts[sp], name, js.spec[sp])
            else: draw_facet(fig.add_subplot(gs[0, idx]), adata.obsm[f"X_{m}_2d"], adata.obs, sp, js.colors, js.selected, counts[sp], name)
        add_colorbar(fig.add_subplot(gs[-1, :]), Normalize(0, 1), CBAR_LABEL)
        save_fig(fig, target_dir / f"{m}_joint_clusters_species_specificity__{param_tag}.png")
        plt.close(fig)
        if intra: plot_intra_panel(adata, intra, m, name, intra_dir / f"{m}_intra_species_top_clusters__{param_tag}.png")
def plot_flux(obs: pd.DataFrame, js: JointStats, intra: dict[str, tuple[np.ndarray, int, pd.Series]], path: Path) -> None:
    """Draw per species the fluxes from its intra clusters to its selected joint clusters, coloured by specificity index.

    Args:
        obs (pd.DataFrame): Observation table with letter-named ``cluster`` and ``intra_cluster`` columns.
        js (JointStats): Joint cluster statistics.
        intra (dict[str, tuple[np.ndarray, int, pd.Series]]): Output of intra_summary.
        path (Path): Output PNG path.
    """
    cluster_labels, intra_labels = obs["cluster"].astype(str), obs["intra_cluster"].astype(str)
    fig, gs = row_figure(len(js.counts.columns), "Intra- to Inter-species cluster fluxes", SANKEY_HEIGHT_SCALE, font_scale=FLUX_FONT_SCALE)
    norm_spec, spec_cut = Normalize(0, 1), args.module_min_specificity_percent / 100
    for idx, (sp, (is_sp, n_cl, top)) in enumerate(intra.items()):
        ax = fig.add_subplot(gs[0, idx])
        title = f"{sp}\n(n={is_sp.sum():,}, k={n_cl})"
        full = pd.crosstab(intra_labels[is_sp], cluster_labels[is_sp])
        sp_ranked = [cl for cl in js.ranked[sp] if cl in full.columns]
        if not sp_ranked:
            ax.axis("off")
            ax.set_title(title, fontsize=FONT_FACET * FLUX_FONT_SCALE, fontweight="bold", pad=2)
            continue
        flows = full[sp_ranked]
        flows = flows[flows.sum(axis=1) > 0]
        l_order = [cl for cl in top.index if cl in flows.index] + flows.sum(axis=1).drop(list(top.index), errors="ignore").sort_values(ascending=False).index.tolist()
        flows = flows.loc[l_order]
        best, spec = flows.idxmax(axis=1), js.spec[sp]
        right_colors = {cl: gradient_hex(spec[cl], norm_spec) for cl in sp_ranked}
        left_colors = {cl: right_colors[best[cl]] if flows.loc[cl, best[cl]] > full.loc[cl].sum() / 2 else COLOR_BG for cl in l_order}
        draw_flux(ax, flows, left_colors, right_colors, {cl: cl for cl in top.index if cl in flows.index}, {cl: f"{cl} ({js.counts.loc[cl, sp]:,})" for cl in sp_ranked}, {cl for cl in sp_ranked if spec[cl] >= spec_cut}, title)
    add_colorbar(fig.add_subplot(gs[-1, :]), norm_spec, f"{CBAR_LABEL}; black border: $S_{{i}} \\geq$ {spec_cut:.2f}")
    save_fig(fig, path)
    plt.close(fig)
def plot_reprojection(adata: ad.AnnData, clusters: list[str], colors: dict[str, str], counts: pd.DataFrame, title: str, path: Path, ring_annotated: bool = False) -> None:
    """Re-project the proteins of a list of joint clusters and draw them per input, coloured by cluster.

    Args:
        adata (ad.AnnData): Saved clusters with letter-named joint clusters.
        clusters (list[str]): Joint clusters to re-project (selected or focus clusters).
        colors (dict[str, str]): Mapping cluster -> colour.
        counts (pd.DataFrame): Protein counts per cluster (rows) and input (columns); its columns are the facets.
        title (str): Figure title.
        path (Path): Output PNG path.
        ring_annotated (bool): Whether to ring the UniProt-annotated focus proteins in black.
    """
    keep = adata.obs["cluster"].astype(str).isin(clusters).values
    if "X_umap_2d" not in adata.obsm or keep.sum() <= 2: return
    obs_selected = adata.obs[keep]
    print(f"[PYTHON-INFO][SELECTION] Re-projecting {keep.sum():,} proteins of {len(clusters)} joint clusters from all inputs: {', '.join(clusters)}")
    coords_selected = reproject(adata, keep)
    species_selected = [sp for sp in counts.columns if (obs_selected["species"] == sp).any()]
    fig, gs = row_figure(len(species_selected), title, strip=False)
    annotated = (obs_selected["focus_tier"].astype(str) == "annotated").values if ring_annotated and "focus_tier" in obs_selected else None
    for idx, sp in enumerate(species_selected):
        draw_facet(fig.add_subplot(gs[0, idx]), coords_selected, obs_selected, sp, colors, clusters, counts[sp].reindex(clusters, fill_value=0), "UMAP", annotated=annotated)
    add_bottom_legend(fig, colors, "Joint cluster", ring_label=RING_LABEL if annotated is not None else None)
    save_fig(fig, path)
    plt.close(fig)
def report(h5ad_path: Path) -> str | None:
    """Write tables, modules, panels, fluxes and the merged PDF of one saved clustering.

    Args:
        h5ad_path (Path): .h5ad written by npz_scanpy.py.

    Returns:
        str | None: Failure message when the focus capture is below min_capture_percent, else None.
    """
    head = ad.read_h5ad(h5ad_path, backed="r")
    param_tag = head.uns["param_tag"]
    head.file.close()
    target_dir = Path(args.outdir) / h5ad_path.parent.name if args.outdir else h5ad_path.parent
    done = [target_dir / f"protein_clusters_metadata__{param_tag}.csv", target_dir / f"{target_dir.name}_scanpy.pdf"] + [target_dir / f"{m}_joint_clusters_species_specificity__{param_tag}.pdf" for m in args.layout_methods]
    src_mtime = max(Path(__file__).stat().st_mtime, h5ad_path.stat().st_mtime)
    if not args.overwrite and all(p.exists() and p.stat().st_mtime > src_mtime for p in done):
        print(f"[PYTHON-INFO] Report of {h5ad_path.name} already exists in {target_dir} and is newer than the script. Skipping.")
        return None
    adata = ad.read_h5ad(h5ad_path)
    modules_dir, intra_dir = target_dir / "protein-modules", target_dir / "intra_group"
    for d in (modules_dir, intra_dir): d.mkdir(parents=True, exist_ok=True)
    if "input_type" not in adata.obs: adata.obs["input_type"] = "target"
    inputs = list(adata.obs["species"].unique())
    species_list = [sp for sp in inputs if (adata.obs.loc[adata.obs["species"] == sp, "input_type"] == "target").all()]
    probes = [sp for sp in inputs if sp not in species_list]
    print(f"[PYTHON-INFO][DATA-SHAPE] {h5ad_path.name}: {adata.n_obs:,} proteins, {len(species_list)} targets ({', '.join(species_list)}), {len(probes)} probes ({', '.join(probes) or 'none'}), obsm: {', '.join(adata.obsm.keys())}")
    js = joint_stats(adata.obs, species_list)
    counts = pd.crosstab(adata.obs["cluster"].astype(str), adata.obs["species"]).reindex(columns=species_list + probes, fill_value=0)
    fs = focus_stats(adata.obs, species_list, species_list + probes)
    if fs is not None: adata.obs["focus_selection"] = focus_selection(adata.obs, fs.clusters, species_list)
    write_tables(adata, js, target_dir, modules_dir, param_tag)
    if fs is not None: write_focus_tables(adata.obs, fs, species_list, target_dir, param_tag)
    intra = intra_summary(adata.obs, species_list, modules_dir, param_tag)
    plot_layout_panels(adata, js, counts, intra, target_dir, intra_dir, param_tag)
    if intra: plot_flux(adata.obs, js, intra, target_dir / f"sankey_intra_to_inter_clusters__{param_tag}.png")
    plot_reprojection(adata, js.selected, js.colors, counts, "Selected inter-species clusters (re-projected)", target_dir / f"umap_reprojected_top_clusters__{param_tag}.png")
    if js.selected:
        top_counts = counts.loc[js.selected]
        top_counts = top_counts.loc[top_counts[species_list].sum(axis=1).sort_values(ascending=False, kind="stable").index]
        plot_counts_bar(top_counts, list(counts.columns), js.chi2_q.loc[js.selected].map(sig_stars), "Protein per cluster (number by species)", target_dir / f"barplot_umap_labelled_clusters_protein_counts__{param_tag}.png", legend_title="Input" if probes else "Species")
    if fs is not None: plot_focus_panels(adata, fs, species_list, target_dir, param_tag)
    out = target_dir / f"{target_dir.name}_scanpy.pdf"
    pdfs = sorted((p for p in target_dir.rglob(f"*__{param_tag}.pdf") if p != out), key=lambda p: next((i for i, s in enumerate(REPORT_ORDER) if p.stem.startswith(s)), len(REPORT_ORDER)))
    merge_pdfs(pdfs, out)
    print(f"[PYTHON-INFO][REPORT] Merged {len(pdfs)} PDFs into {out}")
    if fs is not None and fs.capture.iloc[0]["capture_percent"] < args.min_capture_percent:
        return f"{h5ad_path.name}: focus capture {fs.capture.iloc[0]['capture_percent']:.2f}% of target proteins is below min_capture_percent={args.min_capture_percent}% (see focus_capture__{param_tag}.csv)"
    return None
###################################### FOCUS
class FocusStats(NamedTuple):
    """Per joint cluster statistics of the functional focus and its capture of the target proteins."""
    table: pd.DataFrame
    clusters: list[str]
    colors: dict[str, str]
    counts: pd.DataFrame
    capture: pd.DataFrame
    gravity: pd.DataFrame
def focus_stats(obs: pd.DataFrame, targets: list[str], inputs: list[str]) -> FocusStats | None:
    """Test every joint cluster for enrichment in focus proteins, select the focus clusters and measure the capture of target proteins and the gravity.

    Args:
        obs (pd.DataFrame): Observation table with letter-named ``cluster``, ``species``, ``input_type``, ``focus_tier`` and ``focus_similarity``.
        targets (list[str]): Target inputs.
        inputs (list[str]): All inputs (targets then probes), the columns of the count table.

    Returns:
        FocusStats | None: Focus statistics, or None in comparative mode (no probe input) or without focus proteins.
    """
    if "focus_tier" not in obs or not (obs["input_type"] == "probe").any(): return None
    cluster_labels, tier = obs["cluster"].astype(str), obs["focus_tier"].astype(str)
    is_focus, is_target = tier.isin(["probe", "annotated"]).values, (obs["input_type"] == "target").values
    if not is_focus.any(): return None
    n_all, n_focus = len(obs), int(is_focus.sum())
    size = cluster_labels.value_counts()
    tiers = pd.crosstab(cluster_labels, tier).reindex(index=size.index, columns=["probe", "annotated", "none"], fill_value=0)
    table = pd.DataFrame({"cluster_size": size, "probe_count": tiers["probe"], "annotated_count": tiers["annotated"]})
    table["focus_count"] = table["probe_count"] + table["annotated_count"]
    table["focus_share"] = table["focus_count"] / n_focus
    table["focus_fraction"] = table["focus_count"] / table["cluster_size"]
    table["fold_enrichment"] = table["focus_fraction"] / (n_focus / n_all)
    table["focus_q"] = false_discovery_control(hypergeom.sf(table["focus_count"].to_numpy() - 1, n_all, n_focus, table["cluster_size"].to_numpy()))
    eligible = table[(table["focus_q"] < args.focus_max_q) & (table["focus_count"] >= args.min_focus_proteins)]
    clusters = eligible["focus_share"].sort_values(ascending=False, kind="stable").head(args.max_focus_clusters).index.tolist()
    table["focus_cluster"] = table.index.isin(clusters)
    table = table.sort_values(["focus_cluster", "focus_share"], ascending=False, kind="stable").rename_axis("cluster")
    counts = pd.crosstab(cluster_labels, obs["species"]).reindex(columns=inputs, fill_value=0)
    in_focus = cluster_labels.isin(clusters).values
    rows = []
    for name, mask in [("all_targets", is_target)] + [(sp, (obs["species"] == sp).values) for sp in targets]:
        annotated = mask & (tier == "annotated").values
        n_ann, cap, cap_ann = int(annotated.sum()), int((mask & in_focus).sum()), int((annotated & in_focus).sum())
        rows.append({"target": name, "proteins": int(mask.sum()), "captured": cap, "capture_percent": 100 * cap / max(int(mask.sum()), 1), "annotated": n_ann, "captured_annotated": cap_ann, "annotated_recall_percent": 100 * cap_ann / n_ann if n_ann else np.nan, "captured_unannotated": cap - cap_ann})
    capture = pd.DataFrame(rows)
    sim = obs["focus_similarity"].astype(float) if "focus_similarity" in obs else pd.Series(np.nan, index=obs.index)
    grav = [pd.DataFrame({"cluster": cluster_labels[m].values, "target": name, "sim": sim[m].values}) for name, m in [("all_targets", is_target)] + [(sp, (obs["species"] == sp).values) for sp in targets]]
    gravity = pd.concat(grav).groupby(["target", "cluster"])["sim"].agg(proteins="size", median_similarity="median").reset_index()
    gravity["rank"] = gravity.groupby("target")["median_similarity"].rank(ascending=False, method="min").astype(int)
    gravity["focus_cluster"] = gravity["cluster"].isin(clusters)
    gravity = gravity.sort_values(["target", "rank"], kind="stable")
    print(f"[PYTHON-INFO][FOCUS] {n_focus:,} focus proteins ({int(tiers['probe'].sum()):,} probe, {int(tiers['annotated'].sum()):,} annotated); {len(eligible)} enriched clusters, {len(clusters)} kept: {', '.join(clusters)}")
    print(f"[PYTHON-INFO][FOCUS] Capture: {capture.iloc[0]['captured']:,}/{capture.iloc[0]['proteins']:,} target proteins ({capture.iloc[0]['capture_percent']:.2f}%)")
    return FocusStats(table, clusters, {cl: PALETTE_ALT[i % len(PALETTE_ALT)] for i, cl in enumerate(clusters)}, counts, capture, gravity)
def focus_selection(obs: pd.DataFrame, clusters: list[str], targets: list[str]) -> pd.Series:
    """Label every target protein I (own focus annotation), E (extended through probing: unannotated, inside a focus cluster) or none; probe proteins get no label.

    Args:
        obs (pd.DataFrame): Observation table with letter-named ``cluster``, ``species``, ``input_type`` and ``focus_tier``.
        clusters (list[str]): Focus clusters.
        targets (list[str]): Target inputs.

    Returns:
        pd.Series: Focus selection per protein (NaN for probe proteins).
    """
    annotated, in_focus = (obs["focus_tier"].astype(str) == "annotated").values, obs["cluster"].astype(str).isin(clusters).values
    sel = np.where(annotated, "I", np.where(in_focus, "E", "none")).astype(object)
    sel[(obs["input_type"] != "target").values] = None
    sel = pd.Series(pd.Categorical(sel, categories=list(SELECTION_COLORS)[::-1]), index=obs.index)
    for sp in targets:
        is_sp = (obs["species"] == sp).values
        print(f"[PYTHON-INFO][FOCUS] Selection {sp}: {int((sel[is_sp] == 'I').sum()):,} I, {int((sel[is_sp] == 'E').sum()):,} E of {int(is_sp.sum()):,} proteins")
    return sel
def write_focus_tables(obs: pd.DataFrame, fs: FocusStats, targets: list[str], target_dir: Path, param_tag: str) -> None:
    """Write the focus enrichment, capture, gravity and selection tables, the target protein lists of every focus cluster and the selection list of every target.

    Args:
        obs (pd.DataFrame): Observation table with letter-named joint clusters.
        fs (FocusStats): Focus statistics.
        targets (list[str]): Target inputs.
        target_dir (Path): Report folder.
        param_tag (str): Parameter tag of the clustering.
    """
    fs.table.join(fs.counts.add_prefix("n_")).to_csv(target_dir / f"focus_cluster_enrichment__{param_tag}.csv")
    fs.capture.to_csv(target_dir / f"focus_capture__{param_tag}.csv", index=False)
    fs.gravity.to_csv(target_dir / f"focus_gravity__{param_tag}.csv", index=False)
    focus_dir = target_dir / "focus-modules"
    focus_dir.mkdir(parents=True, exist_ok=True)
    cluster_labels, species_labels = obs["cluster"].astype(str), obs["species"].astype(str)
    for cl in fs.clusters:
        for sp in targets:
            mask = (species_labels == sp).values & (cluster_labels == cl).values
            if mask.any(): obs.loc[mask, "protein_id"].to_csv(focus_dir / f"{sp}_{mask.sum():07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    sel = obs[obs["focus_selection"].isin(["I", "E"])]
    sel[[c for c in ("protein_id", "species", "cluster", "focus_selection", "focus_similarity") if c in sel]].sort_values(["species", "focus_selection", "focus_similarity"], ascending=[True, True, False], kind="stable").to_csv(target_dir / f"focus_selection__{param_tag}.csv", index=False)
    for sp in targets:
        ids = sel.loc[(sel["species"] == sp).values, "protein_id"]
        if len(ids): ids.to_csv(focus_dir / f"{sp}_{len(ids):07d}_focus_selection_{param_tag}.txt", index=False, header=False)
def plot_focus_tiers(adata: ad.AnnData, targets: list[str], target_dir: Path, param_tag: str) -> None:
    """Draw for every layout method one facet per target coloured by focus selection, I on top of E on top of the rest.

    Args:
        adata (ad.AnnData): Saved clusters with ``focus_selection``.
        targets (list[str]): Target inputs in facet order.
        target_dir (Path): Report folder.
        param_tag (str): Parameter tag of the clustering.
    """
    sel = adata.obs["focus_selection"].astype(str).values
    for m in args.layout_methods:
        if f"X_{m}_2d" not in adata.obsm: continue
        name, coords = METHOD_NAMES.get(m, m), adata.obsm[f"X_{m}_2d"]
        fig, gs = row_figure(len(targets), f"Functional focus selection ({name})", strip=False)
        for idx, sp in enumerate(targets):
            ax, is_sp = fig.add_subplot(gs[0, idx]), (adata.obs["species"] == sp).values
            for t, c in SELECTION_COLORS.items():
                pts = is_sp & (sel == t)
                ax.scatter(coords[pts, 0], coords[pts, 1], c=c, s=PT_SIZE if t == "none" else 2 * PT_SIZE, alpha=ALPHA_BG if t == "none" else ALPHA_MAX, rasterized=True, linewidths=0)
            style_axes(ax, facet_title(adata.obs, is_sp, sp), name)
        add_bottom_legend(fig, {f"{SELECTION_LABELS[t]} ({(sel == t).sum():,})": c for t, c in reversed(SELECTION_COLORS.items())}, "Focus selection")
        save_fig(fig, target_dir / f"{m}_focus_tiers__{param_tag}.png")
        plt.close(fig)
def plot_focus_panels(adata: ad.AnnData, fs: FocusStats, targets: list[str], target_dir: Path, param_tag: str) -> None:
    """Draw the focus selection layouts, then for every layout method one facet per input with the focus clusters coloured and the annotated proteins ringed, plus the re-projection and barplot of the focus clusters.

    Args:
        adata (ad.AnnData): Saved clusters with letter-named joint clusters.
        fs (FocusStats): Focus statistics.
        targets (list[str]): Target inputs.
        target_dir (Path): Report folder.
        param_tag (str): Parameter tag of the clustering.
    """
    inputs = list(fs.counts.columns)
    plot_focus_tiers(adata, targets, target_dir, param_tag)
    if not fs.clusters: return
    annotated = (adata.obs["focus_tier"].astype(str) == "annotated").values
    for m in args.layout_methods:
        if f"X_{m}_2d" not in adata.obsm: continue
        name = METHOD_NAMES.get(m, m)
        fig, gs = row_figure(len(inputs), f"Functional focus clusters ({name})", strip=False)
        for idx, sp in enumerate(inputs):
            draw_facet(fig.add_subplot(gs[0, idx]), adata.obsm[f"X_{m}_2d"], adata.obs, sp, fs.colors, fs.clusters, fs.counts[sp], name, annotated=annotated)
        add_bottom_legend(fig, fs.colors, "Focus cluster", ring_label=RING_LABEL)
        save_fig(fig, target_dir / f"{m}_focus_clusters__{param_tag}.png")
        plt.close(fig)
    plot_reprojection(adata, fs.clusters, fs.colors, fs.counts, "Functional focus clusters (re-projected)", target_dir / f"umap_reprojected_focus_clusters__{param_tag}.png", ring_annotated=True)
    top_counts = fs.counts.loc[fs.clusters, inputs]
    plot_counts_bar(top_counts, inputs, fs.table.loc[fs.clusters, "focus_q"].map(sig_stars), "Proteins per focus cluster (number by input)", target_dir / f"barplot_focus_clusters_protein_counts__{param_tag}.png", "Focus cluster\nhypergeometric focus enrichment, BH: * q<0.05, ** q<0.01, *** q<0.001", "Input")
###################################### PAIRED
def baseline_path(h5ad_path: Path, obs: pd.DataFrame) -> Path:
    """Locate the baseline .h5ad of a focus run: same targets, model and parameters, without probes.

    Args:
        h5ad_path (Path): .h5ad of the focus run.
        obs (pd.DataFrame): Its observations with ``species`` and ``input_type``.

    Returns:
        Path: Expected .h5ad of the baseline run.
    """
    targets = [sp for sp in obs["species"].astype(str).unique() if (obs.loc[obs["species"].astype(str) == sp, "input_type"].astype(str) == "target").all()]
    run, model = h5ad_path.parent.name, h5ad_path.parent.name.rsplit("__", 1)[1]
    base_run = "_".join(targets) + f"__{model}"
    return h5ad_path.parent.parent / base_run / (base_run + h5ad_path.name[len(run):])
def best_match(flow: pd.DataFrame, cluster: str, axis: int) -> tuple[str, int, float]:
    """Find the cluster of the other run that shares most proteins with a cluster, and their Jaccard index.

    Args:
        flow (pd.DataFrame): Shared proteins, baseline clusters as rows and focus-run clusters as columns.
        cluster (str): Cluster to match (a row when axis is 0, a column when axis is 1).
        axis (int): 0 for a baseline cluster, 1 for a focus-run cluster.

    Returns:
        tuple[str, int, float]: Best-matching cluster, shared proteins and Jaccard index.
    """
    line = flow.loc[cluster] if axis == 0 else flow[cluster]
    other = line.idxmax()
    shared = int(line[other])
    other_size = int(flow[other].sum() if axis == 0 else flow.loc[other].sum())
    return str(other), shared, shared / (int(line.sum()) + other_size - shared)
def compare_pair(h5ad_path: Path) -> None:
    """Compare a focus run with its baseline run on their shared target proteins and write the paired tables.

    Args:
        h5ad_path (Path): .h5ad of a run; runs without probes are skipped.
    """
    head = ad.read_h5ad(h5ad_path, backed="r")
    focus_obs, param_tag = head.obs.copy(), head.uns["param_tag"]
    head.file.close()
    if "input_type" not in focus_obs or not (focus_obs["input_type"].astype(str) == "probe").any(): return
    base = baseline_path(h5ad_path, focus_obs)
    if not base.exists():
        print(f"[PYTHON-INFO][PAIRED] No baseline run {base} for {h5ad_path.name}. Skipping the paired comparison.")
        return
    head = ad.read_h5ad(base, backed="r")
    base_obs = head.obs.copy()
    head.file.close()
    common = focus_obs.index[(focus_obs["input_type"].astype(str) == "target").values].intersection(base_obs.index)
    f_cl, b_cl = focus_obs.loc[common, "cluster"].astype(int).map(letter), base_obs.loc[common, "cluster"].astype(int).map(letter)
    flow = pd.crosstab(b_cl.rename("baseline_cluster"), f_cl.rename("focus_run_cluster"))
    focus_dir = Path(args.outdir) / h5ad_path.parent.name if args.outdir else h5ad_path.parent
    base_dir = Path(args.outdir) / base.parent.name if args.outdir else base.parent
    long = flow.stack().rename("proteins").reset_index().query("proteins > 0")
    long["jaccard"] = long["proteins"] / (long["baseline_cluster"].map(flow.sum(axis=1)) + long["focus_run_cluster"].map(flow.sum(axis=0)) - long["proteins"])
    focus_csv = focus_dir / f"focus_cluster_enrichment__{param_tag}.csv"
    focus_clusters = set(pd.read_csv(focus_csv, dtype={"cluster": str}).query("focus_cluster")["cluster"]) if focus_csv.exists() else set()
    long["focus_cluster"] = long["focus_run_cluster"].isin(focus_clusters)
    long.sort_values(["focus_run_cluster", "proteins"], ascending=[True, False], kind="stable").to_csv(focus_dir / f"paired_cluster_flow__{param_tag}.csv", index=False)
    selected = {run: (set(pd.read_csv(d / f"species_cluster_enrichment__{param_tag}.csv", dtype={"cluster": str}).query("selected")["cluster"]) if (d / f"species_cluster_enrichment__{param_tag}.csv").exists() else set()) for run, d in (("baseline", base_dir), ("focus_run", focus_dir))}
    rows = []
    for run, axis, other in (("baseline", 0, "focus_run"), ("focus_run", 1, "baseline")):
        for cl in sorted(selected[run], key=lambda c: (len(c), c)):
            if cl not in (flow.index if axis == 0 else flow.columns): continue
            match, shared, jac = best_match(flow, cl, axis)
            rows.append({"run": run, "cluster": cl, "best_match": match, "shared_proteins": shared, "jaccard": jac, "match_selected": match in selected[other]})
    pd.DataFrame(rows, columns=["run", "cluster", "best_match", "shared_proteins", "jaccard", "match_selected"]).to_csv(focus_dir / f"paired_selection__{param_tag}.csv", index=False)
    summary = {"baseline_h5ad": base.name, "focus_run_h5ad": h5ad_path.name, "shared_target_proteins": len(common), "baseline_clusters": int(flow.shape[0]), "focus_run_clusters": int(flow.shape[1]), "adjusted_rand_index": adjusted_rand_score(b_cl, f_cl), "normalized_mutual_info": normalized_mutual_info_score(b_cl, f_cl), "baseline_selected": len(selected["baseline"]), "focus_run_selected": len(selected["focus_run"])}
    (focus_dir / f"paired_summary__{param_tag}.txt").write_text("".join(f"{k}: {v}\n" for k, v in summary.items()))
    print(f"[PYTHON-INFO][PAIRED] {h5ad_path.parent.name} vs {base.parent.name}: {len(common):,} shared targets, ARI {summary['adjusted_rand_index']:.3f}, NMI {summary['normalized_mutual_info']:.3f}")
###################################### MAIN
def main() -> None:
    """Report every .h5ad passed on the command line."""
    start_time = time.time()
    paths = [Path(p.strip()) for p in args.h5ad.split(",") if p.strip()]
    failures = [msg for path in tqdm(paths, desc="Reports") if (msg := report(path))]
    for path in paths: compare_pair(path)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
    if failures: raise RuntimeError("[PYTHON-ERROR] " + "; ".join(failures))
if __name__ == "__main__":
    main()
