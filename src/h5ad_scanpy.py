#!/usr/bin/env python3
"""Reports the joint and intra-species clusters saved by npz_scanpy.py: tables, protein modules, cluster panels, intra-to-inter cluster flux diagrams and a merged PDF.

Only the .h5ad is read, so no clustering or embedding is recomputed except the re-projection of the top joint clusters. The report panels, each A4/4 wide, are drawn for every method in ``--methods`` (UMAP and t-SNE, 2D layouts): (a) intra-species layouts with all clusters with more than MIN_INTRA_SIZE proteins coloured by protein count, (b) joint layouts coloured by cluster specificity index, labelling per species the MAX_JOINT_LABELS most specific joint clusters with more than MIN_JOINT_SIZE proteins of that species, drawn on top of the others and named by lowercase letters assigned once and shared by all panels, where only joint clusters whose per-protein cluster membership differs across species by one-way ANOVA (BH q < ANOVA_ALPHA) can be labelled, then once: (c) a re-projected UMAP of all proteins of those top joint clusters, optimised on the stored joint kNN graph restricted to them and coloured by cluster, (d) intra-to-top-joint cluster fluxes on a square-root (semi-quantitative) scale, with joint cluster nodes and flows coloured by specificity index and outlined in black above MIN_PERCENT_ENRICHMENT, where intra clusters inherit a cluster colour only when a strict majority of their proteins flows into it, and (e) a per-species protein-count barplot of the top joint clusters with ANOVA significance asterisks over every cluster. A report whose outputs all exist and are newer than both this script and the .h5ad is skipped unless ``--overwrite`` is set. All figures are saved as PNG and PDF, and the PDFs are merged into a final {dirname}_scanpy.pdf report.
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
from scipy.stats import chisquare, f_oneway, false_discovery_control
from string import ascii_lowercase
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
ANOVA_ALPHA = 0.05
CBAR_LABEL = "Cluster Specificity Index ($S_{i}$)"
COLOR_BG = "#BFC5CE"
COLOR_FG = "#8A929E"
COLORMAP_SPEC = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
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
FONT_TICK = 9
FONT_TITLE = 14
LEGEND_CHAR_INCHES = 0.08
LEGEND_COL_INCHES = 0.4
LEGEND_MARKER = 7
MAX_JOINT_LABELS = 6
METHOD_NAMES = {"umap": "UMAP", "tsne": "t-SNE"}
METHODS = ["umap", "tsne"]
MIN_INTRA_SIZE = 150
MIN_BALANCE = 0.6
MIN_JOINT_SIZE = 100
MIN_MODULE_SIZE = 100
MIN_PERCENT_ENRICHMENT = 80
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
PANEL_DPI = 300
PANEL_WIDTH_INCHES = A4_WIDTH_INCHES / 4
PT_SIZE = 1.0
REPORT_ORDER = ["umap_intra_species_top_clusters", "tsne_intra_species_top_clusters", "umap_joint_clusters_species_specificity", "tsne_joint_clusters_species_specificity", "umap_reprojected_top_clusters", "sankey_intra_to_inter_clusters", "barplot_umap_labelled_clusters_protein_counts"]
SANKEY_HEIGHT_SCALE = 3.0
SEED = 42
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="ESMC Proteome Cluster Report")
parser.add_argument("--h5ad", type=str, required=True, help="Comma-separated .h5ad files written by npz_scanpy.py")
parser.add_argument("--outdir", type=str, default=None, help="Base output directory; defaults to the folder of each .h5ad")
parser.add_argument("--methods", type=str, nargs="+", default=METHODS, help="Layouts to plot (umap, tsne)")
parser.add_argument("--max_joint_labels", type=int, default=MAX_JOINT_LABELS, help="Top joint clusters labelled per species")
parser.add_argument("--min_joint_size", type=int, default=MIN_JOINT_SIZE, help="Minimum species proteins of a labelled joint cluster")
parser.add_argument("--min_intra_size", type=int, default=MIN_INTRA_SIZE, help="Minimum proteins of a highlighted intra cluster")
parser.add_argument("--min_balance", type=float, default=MIN_BALANCE, help="Minimum ratio of the smallest to the largest normalised species share of a balanced joint cluster")
parser.add_argument("--min_module_size", type=int, default=MIN_MODULE_SIZE, help="Minimum proteins of a saved module")
parser.add_argument("--min_percent_enrichment", type=float, default=MIN_PERCENT_ENRICHMENT, help="Species percentage of a joint module")
parser.add_argument("--anova_alpha", type=float, default=ANOVA_ALPHA, help="BH-corrected ANOVA q-value below which a joint cluster can be highlighted and re-projected")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Redraw a report whose outputs already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
yargs, _ = parser.parse_known_args()
script_key = "h5ad_scanpy"
params = dict(methods=METHODS, max_joint_labels=MAX_JOINT_LABELS, min_joint_size=MIN_JOINT_SIZE, min_intra_size=MIN_INTRA_SIZE, min_balance=MIN_BALANCE, min_module_size=MIN_MODULE_SIZE, min_percent_enrichment=MIN_PERCENT_ENRICHMENT, anova_alpha=ANOVA_ALPHA, overwrite=OVERWRITE)
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
def sig_stars(q: float) -> str:
    """Convert an ANOVA q-value into significance asterisks.

    Args:
        q (float): BH-corrected q-value.

    Returns:
        str: ``***``, ``**``, ``*`` or ``ns``.
    """
    return next((stars for cut, stars in SIG_LEVELS if q < cut), "ns")
def top_joint_clusters(counts_df: pd.DataFrame, spec_df: pd.DataFrame, anova_q: pd.Series, sp: str) -> list[str]:
    """Rank the max_joint_labels most specific joint clusters of a species among those with more than min_joint_size of its proteins and ANOVA q below anova_alpha.

    Args:
        counts_df (pd.DataFrame): Protein counts per cluster (rows) and species (columns).
        spec_df (pd.DataFrame): Specificity index per cluster (rows) and species (columns).
        anova_q (pd.Series): BH-corrected ANOVA q-value per cluster.
        sp (str): Species of the facet.

    Returns:
        list[str]: Clusters in decreasing specificity order.
    """
    return spec_df.loc[(counts_df[sp] > args.min_joint_size) & (anova_q < args.anova_alpha), sp].sort_values(ascending=False, kind="stable").head(args.max_joint_labels).index.tolist()
def cluster_letters(ranked: dict[str, list[str]]) -> dict[str, str]:
    """Assign once a lowercase letter to each ranked joint cluster, in order of first appearance across species.

    Args:
        ranked (dict[str, list[str]]): Ranked clusters per species.

    Returns:
        dict[str, str]: Mapping cluster -> letter, in letter order.
    """
    order = list(dict.fromkeys(cl for cls in ranked.values() for cl in cls))
    return {cl: ascii_lowercase[i % 26] * (i // 26 + 1) for i, cl in enumerate(order)}
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
def draw_facet(ax: plt.Axes, coords: np.ndarray, obs: pd.DataFrame, sp: str, colors: dict[str, str], facet: dict[str, str], sp_counts: pd.Series, axis_name: str, spec: pd.Series | None = None) -> None:
    """Draw one species facet coloured by joint cluster, or by specificity index of all its proteins when ``spec`` is given, with labelled clusters on top and their letters at cluster centroids.

    Args:
        ax (plt.Axes): Target axes.
        coords (np.ndarray): Layout coordinates of all points.
        obs (pd.DataFrame): Observation table with ``species`` and ``cluster`` columns.
        sp (str): Species of the facet.
        colors (dict[str, str]): Mapping cluster -> colour.
        facet (dict[str, str]): Clusters -> letter labels.
        sp_counts (pd.Series): Proteins of ``sp`` per cluster.
        axis_name (str): Layout name used in the axis labels.
        spec (pd.Series | None): Specificity index of ``sp`` per cluster; when None clusters are coloured by ``colors``.
    """
    is_sp = (obs["species"] == sp).values
    cl_str = obs["cluster"].astype(str)
    in_facet = cl_str.isin(list(facet)).values
    if spec is not None:
        scores = cl_str.map(spec).values.astype(float)
        sp_idx = np.where(is_sp)[0]
        order = sp_idx[np.lexsort((scores[sp_idx], in_facet[sp_idx]))]
        fg_colors = COLORMAP_SPEC(scores[order])
        fg_colors[:, 3] = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * scores[order]
        ax.scatter(coords[order, 0], coords[order, 1], c=fg_colors, s=PT_SIZE, rasterized=True, linewidths=0)
    else:
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
            ax.fill_betweenx([offs[lab], offs[lab] + h], x0, x0 + FLUX_NODE_WIDTH, facecolor=cols[lab], edgecolor="black" if side == "r" and lab in outlined else "none", lw=FLUX_BORDER_WIDTH)
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
def plot_counts_bar(top_counts: pd.DataFrame, species_list: list[str], stars: pd.Series, title: str, path: Path) -> None:
    """Plot per-species protein counts of the top joint clusters with ANOVA significance asterisks over every cluster.

    Args:
        top_counts (pd.DataFrame): Protein counts per top joint cluster letter (rows) and species (columns), in plotting order.
        species_list (list[str]): Species to plot.
        stars (pd.Series): Significance asterisks per top joint cluster letter.
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
    for xi, ymax, st in zip(x, group_max, stars.reindex(top_counts.index).fillna("ns")):
        ax.text(xi, ymax, st, fontsize=FONT_LABEL, fontweight="bold", ha="center", va="bottom")
    ax.set_ylim(0, group_max.max() * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(top_counts.index, fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlim(-0.6, len(top_counts) - 0.4)
    ax.set_xlabel("Joint cluster\nANOVA across species, BH: * q<0.05, ** q<0.01, *** q<0.001", fontsize=FONT_LABEL, labelpad=2)
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
    a, b = find_ab_params(1.0, float(adata.uns["params"]["min_dist"]))
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
    done = [target_dir / f"protein_clusters_metadata__{param_tag}.csv", target_dir / f"{target_dir.name}_scanpy.pdf"] + [target_dir / f"{m}_joint_clusters_species_specificity__{param_tag}.pdf" for m in args.methods]
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
    counts_df = pd.crosstab(obs["cluster"], obs["species"])
    counts_df.index = counts_df.index.astype(str)
    norm_df = counts_df / counts_df.sum()
    cluster_sp_frac = norm_df.div(norm_df.sum(axis=1), axis=0)
    cl_str, intra_str = obs["cluster"].astype(str), obs["intra_cluster"].astype(str)
    sp_str = obs["species"].astype(str)
    anova_p = f_oneway(*[pd.get_dummies(cl_str[sp_str == sp]).reindex(columns=counts_df.index, fill_value=0).to_numpy(np.float32) for sp in species_list], axis=0).pvalue
    anova_q = pd.Series(false_discovery_control(np.nan_to_num(anova_p, nan=1.0)), index=counts_df.index)
    print(f"[PYTHON-INFO][ANOVA] {(anova_q < args.anova_alpha).sum()}/{len(anova_q)} joint clusters with species-dependent membership (q < {args.anova_alpha})")
    chi2_q = pd.Series(false_discovery_control(chisquare(counts_df, np.outer(counts_df.sum(axis=1), counts_df.sum() / counts_df.values.sum()), axis=1).pvalue), index=counts_df.index)
    balanced = (cluster_sp_frac.min(axis=1) >= args.min_balance * cluster_sp_frac.max(axis=1)) & (counts_df.min(axis=1) >= args.min_joint_size)
    print(f"[PYTHON-INFO][BALANCED] {balanced.sum()} balanced joint clusters: {', '.join(balanced.index[balanced])}")
    ranked = {sp: top_joint_clusters(counts_df, cluster_sp_frac, anova_q, sp) for sp in species_list}
    letters = cluster_letters(ranked)
    facets = {sp: {cl: letters[cl] for cl in cls} for sp, cls in ranked.items()}
    colors = {cl: PALETTE[i % len(PALETTE)] for i, cl in enumerate(letters)}
    ###################################### TABLES_AND_MODULES
    with open(target_dir / f"clustering_metrics__{param_tag}.txt", "w") as f:
        f.writelines(f"{k}: {v}\n" for k, v in adata.uns["metrics"].items())
    rank_rows = [(sp, cl, rk, counts_df.loc[cl, sp], cluster_sp_frac.loc[cl, sp], anova_q[cl]) for sp, sp_ranks in facets.items() for cl, rk in sp_ranks.items()]
    pd.DataFrame(rank_rows, columns=["species", "cluster", "letter", "protein_count", "species_fraction", "anova_q"]).to_csv(target_dir / f"species_top_clusters__{param_tag}.csv", index=False)
    obs[["protein_id", "species", "cluster", "intra_cluster"]].to_csv(target_dir / f"protein_clusters_metadata__{param_tag}.csv", index=False)
    enrichment = cluster_sp_frac * 100
    enrichment_table = counts_df.stack().rename("protein_count").reset_index()
    enrichment_table.columns = ["cluster", "species", "protein_count"]
    frac_table = enrichment.stack().rename("species_fraction_percent").reset_index()
    frac_table.columns = ["cluster", "species", "species_fraction_percent"]
    enrichment_table.merge(frac_table, on=["cluster", "species"], how="left").assign(chi2_q=lambda d: d["cluster"].map(chi2_q), anova_q=lambda d: d["cluster"].map(anova_q), balanced=lambda d: d["cluster"].map(balanced)).to_csv(target_dir / f"species_cluster_enrichment__{param_tag}.csv", index=False)
    adata.uns["markers"].to_csv(target_dir / f"channel_importance_rankings__{param_tag}.csv", index=False)
    for sp in species_list:
        for cl in counts_df.index:
            if enrichment.loc[cl, sp] > args.min_percent_enrichment and counts_df.loc[cl, sp] >= args.min_module_size:
                obs.loc[(obs["species"] == sp) & (cl_str == cl), "protein_id"].to_csv(modules_dir / f"{sp}_{counts_df.loc[cl, sp]:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    ###################################### INTRA_SUMMARY
    intra = {}
    for sp in species_list:
        is_sp = (obs["species"] == sp).values
        if obs.loc[is_sp, "intra_cluster"].isna().all(): continue
        counts_intra = intra_str[is_sp].value_counts()
        intra[sp] = (is_sp, counts_intra.size, counts_intra[counts_intra > args.min_intra_size])
        for cl, cnt in counts_intra.items():
            if cnt >= args.min_module_size:
                obs.loc[is_sp & (intra_str == cl).values, "protein_id"].to_csv(modules_dir / f"intra_{sp}_{cnt:07d}_clust{cl}_{param_tag}.txt", index=False, header=False)
    ###################################### LAYOUT_PANELS
    for m in tqdm(args.methods, desc="Layout panels"):
        if f"X_{m}_2d" not in adata.obsm:
            print(f"[PYTHON-INFO][PANELS] Skipping {m}: X_{m}_2d not in {h5ad_path.name}")
            continue
        name = METHOD_NAMES.get(m, m)
        fig, gs = row_figure(n_sp, f"Inter-species ({name})")
        for idx, sp in enumerate(species_list):
            draw_facet(fig.add_subplot(gs[0, idx]), adata.obsm[f"X_{m}_2d"], obs, sp, colors, facets[sp], counts_df[sp], name, cluster_sp_frac[sp])
        add_colorbar(fig.add_subplot(gs[-1, :]), Normalize(0, 1), CBAR_LABEL)
        save_fig(fig, target_dir / f"{m}_joint_clusters_species_specificity__{param_tag}.png")
        plt.close(fig)
        if not intra: continue
        top_vals = [cnt for *_, top in intra.values() for cnt in top.values] or [0]
        norm_size = Normalize(min(top_vals), max(top_vals) if max(top_vals) > min(top_vals) else min(top_vals) + 1)
        fig, gs = row_figure(n_sp, f"Intra-species ({name})")
        for idx, (sp, (is_sp, _, top)) in enumerate(intra.items()):
            ax = fig.add_subplot(gs[0, idx])
            coords, intra_cls = adata.obsm[f"X_{m}_2d_intra"][is_sp], intra_str.values[is_sp]
            is_shown = np.isin(intra_cls, list(top.index))
            ax.scatter(coords[~is_shown, 0], coords[~is_shown, 1], c=COLOR_BG, s=PT_SIZE, alpha=ALPHA_BG, rasterized=True, linewidths=0, zorder=1)
            for c_idx, (cl, cnt) in enumerate(top.items()):
                cl_pts = coords[intra_cls == cl]
                ax.scatter(cl_pts[:, 0], cl_pts[:, 1], c=gradient_hex(cnt, norm_size), s=PT_SIZE, alpha=ALPHA_MAX, rasterized=True, linewidths=0, zorder=3)
                txt = ax.text(np.median(cl_pts[:, 0]), np.median(cl_pts[:, 1]), str(c_idx + 1), fontsize=FONT_LABEL, ha="center", va="center", fontweight="bold", color="black", zorder=6)
                txt.set_path_effects([path_effects.withStroke(linewidth=1.25, foreground="white")])
            style_axes(ax, f"{sp} (n={is_sp.sum():,})", name)
        add_colorbar(fig.add_subplot(gs[-1, :]), norm_size, "Proteins per highlighted cluster")
        save_fig(fig, intra_dir / f"{m}_intra_species_top_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### INTRA_INTER_FLUX
    if intra:
        fig, gs = row_figure(n_sp, "Intra- to Inter-species cluster fluxes", SANKEY_HEIGHT_SCALE, font_scale=FLUX_FONT_SCALE)
        norm_spec, spec_cut = Normalize(0, 1), args.min_percent_enrichment / 100
        for idx, (sp, (is_sp, n_cl, top)) in enumerate(intra.items()):
            ax = fig.add_subplot(gs[0, idx])
            title = f"{sp}\n(n={is_sp.sum():,}, k={n_cl})"
            full = pd.crosstab(intra_str[is_sp], cl_str[is_sp])
            sp_ranked = [cl for cl in facets[sp] if cl in full.columns]
            if not sp_ranked:
                ax.axis("off")
                ax.set_title(title, fontsize=FONT_FACET * FLUX_FONT_SCALE, fontweight="bold", pad=2)
                continue
            flows = full[sp_ranked]
            flows = flows[flows.sum(axis=1) > 0]
            l_order = [cl for cl in top.index if cl in flows.index] + flows.sum(axis=1).drop(list(top.index), errors="ignore").sort_values(ascending=False).index.tolist()
            flows = flows.loc[l_order]
            best = flows.idxmax(axis=1)
            spec = cluster_sp_frac[sp]
            right_colors = {cl: gradient_hex(spec[cl], norm_spec) for cl in sp_ranked}
            left_colors = {cl: right_colors[best[cl]] if flows.loc[cl, best[cl]] > full.loc[cl].sum() / 2 else COLOR_BG for cl in l_order}
            rank = {cl: str(c_idx + 1) for c_idx, cl in enumerate(top.index) if cl in flows.index}
            draw_flux(ax, flows, left_colors, right_colors, rank, {cl: f"{facets[sp][cl]} ({counts_df.loc[cl, sp]:,})" for cl in sp_ranked}, {cl for cl in sp_ranked if spec[cl] >= spec_cut}, title)
        add_colorbar(fig.add_subplot(gs[-1, :]), norm_spec, f"{CBAR_LABEL}; black border: $S_{{i}} \\geq$ {spec_cut:.2f}")
        save_fig(fig, target_dir / f"sankey_intra_to_inter_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### REPROJECTION
    keep = cl_str.isin(list(colors)).values
    if "X_umap_2d" in adata.obsm and keep.sum() > 2:
        obs_2 = obs[keep]
        print(f"[PYTHON-INFO][DATA-SHAPE] Re-projected top clusters: {keep.sum():,} proteins")
        coords_2 = reproject(adata, keep)
        species_2 = list(obs_2["species"].unique())
        fig, gs = row_figure(len(species_2), "Top Inter-species clusters (re-projected)", strip=False)
        for idx, sp in enumerate(species_2):
            draw_facet(fig.add_subplot(gs[0, idx]), coords_2, obs_2, sp, colors, facets[sp], counts_df[sp], "UMAP")
        add_bottom_legend(fig, {letters[cl]: c for cl, c in colors.items()}, "Joint cluster")
        save_fig(fig, target_dir / f"umap_reprojected_top_clusters__{param_tag}.png")
        plt.close(fig)
    ###################################### TOP_COUNTS_BARPLOT
    if colors:
        top_counts = counts_df.loc[list(colors), species_list]
        top_counts = top_counts.loc[top_counts.sum(axis=1).sort_values(ascending=False, kind="stable").index].rename(index=letters)
        plot_counts_bar(top_counts, species_list, anova_q.loc[list(colors)].map(sig_stars).rename(index=letters), "Protein per cluster (number by species)", target_dir / f"barplot_umap_labelled_clusters_protein_counts__{param_tag}.png")
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
