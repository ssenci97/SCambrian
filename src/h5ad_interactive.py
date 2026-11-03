#!/usr/bin/env python3
"""Writes one paged, self-contained HTML viewer of the UMAP and t-SNE layouts of npz_scanpy.py, coloured by the cluster series of compile_summary.py (Leiden and MMseqs2).

Per projections.h5ad a single ``<model>/interactive/views_3d.html`` is saved, with one page for the joint view and one per target (internal view), browsed with arrows, a page menu or the arrow keys. Every page has the views of the VIEWS registry of the template, switched in the header or with the V key: side-by-side 3D UMAP and t-SNE panels, a static 2D projection of the same 3D layouts, a full-size barplot (counts, log or per-cluster species shares) and a flux (Sankey) view between any two series, together with a species/cluster switch. Every species and cluster colour can be repicked from its checklist swatch (kept across views and pages by name), the point alpha is set in the header and the unticked points are either hidden or drawn as faint grey ghosts behind the shown ones, with their own alpha. Views needing data beyond the points get it from a builder of VIEW_BUILDERS, so a new view is one template entry and at most one builder. The cluster label sets are the UMAP and t-SNE series of ``sc_summary.tsv`` (``{analysis}_{u|t}_{i|j}_leiden{2|3}d``), so every name in the viewer is the SERIAL of the summary: letters for joint clusters (UPPER case when the cluster holds probe and target proteins) and ``<label>_001`` for internal ones. The MMseqs2 series (``{analysis}_s_{i|j}_id90``, ...) are added too, joint ones on every page and internal ones on the internal pages: their clusters smaller than ``min_series_cluster_size`` are pooled into one "other" category and only the ``max_colored_clusters`` largest among the points shown get a colour (recomputed on every filter, lasso or subset), the checklist showing the first ones matching its filter. The flux view counts the shown points between the clusters of two series. The barplot and the Sel/Probe buttons use the report series (``h5ad_scanpy.report_series``): Sel ticks the joint clusters selected by h5ad_scanpy.py (``selected`` column of its enrichment table, whose significant clusters also get chi-square asterisks over their bars), Probe ticks the probe clusters selected by h5ad_scanpy.py (``probe_selected`` column, probe enrichment), a subset of the UPPER-case clusters. A left sidebar holds a protein ID finder, an export of the IDs of the points shown, and species and cluster checklists. Probe samples are suffixed "(probe)" and get no internal page. Species colours follow the barplots of h5ad_scanpy.py (PALETTE_ALT by sample position, targets first). In the 2D view a lasso selects the shown points and cluster labels can be toggled. An output newer than this script, its template, the .h5ad, the summary and the enrichment table is skipped unless ``--overwrite`` is set.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import base64
import datetime
from html import escape
import json
import numpy as np
import pandas as pd
from pathlib import Path
from plotly.offline import get_plotlyjs
import re
import resource
import time
from tqdm import tqdm
import yaml
ALPHA = 0.8
GHOST_ALPHA = 0.08
COLOR_BG = "#BFC5CE"
COLOR_MINOR = "#D5D9DF"
COLOR_OTHER = "#E6E8EC"
DEFAULT_QUERY_TSV = "configs/query_configs.tsv"
DIM = 3
FONT_FAMILY = "DejaVu Sans"
FONT_LEGEND = 15
FONT_TICK = 12
FONT_TITLE = 20
MARKER_SIZE = 0.5
MAX_COLORED_CLUSTERS = 30
MAX_POINTS_PER_VIEW = None
METHODS = {"umap": "UMAP", "tsne": "t-SNE"}
MIN_SERIES_CLUSTER_SIZE = 5
MMSEQS_PATTERN = re.compile(r"_s_([ij])_id(\d+)$")
OUTPUT_FILE = "views_3d.html"
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
PALETTE_ALT = PALETTE[::2] + PALETTE[1::2]
REPORT_SERIES = "u_j_leiden2d"
SEED = 42
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
SPECIES_KEY = "sample"
SUBDIR = "interactive"
SUMMARY_FILE = "sc_summary.tsv"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "h5ad_interactive.html"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Interactive 3D UMAP/t-SNE HTML views of the summary cluster series")
parser.add_argument("--h5ad", type=str, required=True, help="Comma-separated projections.h5ad files of npz_scanpy.py, each with its sc_summary.tsv")
parser.add_argument("--outdir", type=str, default=None, help="Base output directory; defaults to the folder of each .h5ad")
parser.add_argument("--max_points_per_view", type=int, default=MAX_POINTS_PER_VIEW, help="Random subsample of plotted points per view (None plots all)")
parser.add_argument("--max_colored_clusters", type=int, default=MAX_COLORED_CLUSTERS, help="Largest clusters of an MMseqs2 series drawn in colour, the others in grey")
parser.add_argument("--min_series_cluster_size", type=int, default=MIN_SERIES_CLUSTER_SIZE, help="MMseqs2 clusters with fewer proteins on a page are pooled into one 'other' category")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute outputs that already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "h5ad_interactive"
params = dict(max_points_per_view=MAX_POINTS_PER_VIEW, max_colored_clusters=MAX_COLORED_CLUSTERS, min_series_cluster_size=MIN_SERIES_CLUSTER_SIZE, overwrite=OVERWRITE)
config_data = {}
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    if script_key in config_data: params.update(config_data[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    out_data = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    out_data[script_key] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(out_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def species_names() -> dict[str, str]:
    """Map each label of the query table to its taxonomic_ref display name.

    Returns:
        dict[str, str]: Label -> display name (empty if the table is missing).
    """
    tsv = Path(config_data.get("dw_uniprot_seqs", {}).get("query_table", DEFAULT_QUERY_TSV))
    df = pd.read_csv(tsv, sep="\t", dtype=str) if tsv.exists() else pd.DataFrame(columns=["label"])
    names = df["taxonomic_ref"] if "taxonomic_ref" in df else df["label"]
    return {lbl: (nm if pd.notna(nm) and str(nm).strip() else lbl) for lbl, nm in zip(df["label"], names) if pd.notna(lbl)}
def categories(values: pd.Series, colors: dict[str, str] | None = None, min_size: int = 1, max_colored: int | None = None) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    """Encode labels as integer codes ordered by decreasing size, coloured by alternating PALETTE colours or by a fixed mapping.

    Args:
        values (pd.Series): Labels per point (NaN allowed, drawn as unassigned).
        colors (dict[str, str] | None): Fixed label -> colour mapping; other labels take PALETTE_ALT by size.
        min_size (int): Labels with fewer points are pooled into one last "other" category.
        max_colored (int | None): Only this many largest labels take a PALETTE_ALT colour, the others COLOR_MINOR (None colours all).

    Returns:
        tuple[np.ndarray, list[tuple[str, str, int]]]: Codes per point and (name, colour, count) per code.
    """
    values = values.astype("object").where(values.notna(), None)
    counts = values.dropna().value_counts(sort=True)
    big, small = counts[counts >= min_size], counts[counts < min_size]
    palette = lambda i: PALETTE_ALT[i % len(PALETTE_ALT)] if max_colored is None or i < max_colored else COLOR_MINOR
    cats = [("unassigned", COLOR_BG, int(values.isna().sum()))] + [(str(n), (colors or {}).get(n, palette(i)), int(c)) for i, (n, c) in enumerate(zip(big.index, big.values))]
    lookup = {n: i + 1 for i, n in enumerate(big.index)}
    if len(small):
        cats.append((f"other (<{min_size}, {len(small):,} clusters)", COLOR_OTHER, int(small.sum())))
        lookup |= dict.fromkeys(small.index, len(cats) - 1)
    return np.asarray([lookup.get(v, 0) for v in values], dtype=np.int64), cats
def encode(values: np.ndarray, dtype: type) -> str:
    """Encode an array as base64 of its raw bytes in the given dtype.

    Args:
        values (np.ndarray): Values to encode.
        dtype (type): Numpy dtype of the bytes.

    Returns:
        str: Base64 string.
    """
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode()
def load_enrichment(enrichment_csv: Path) -> tuple[dict[str, str], set[str], set[str]]:
    """Read the significance asterisks, the selection and the probe clusters of the joint clusters from the enrichment table of h5ad_scanpy.py.

    Args:
        enrichment_csv (Path): species_cluster_enrichment table (may be missing).

    Returns:
        tuple[dict[str, str], set[str], set[str]]: Significant cluster -> asterisks, the selected clusters and the probe clusters.
    """
    columns = ["cluster", "chi2_q", "significant", "selected", "probe_selected"]
    table = pd.read_csv(enrichment_csv, dtype={"cluster": str}).drop_duplicates("cluster") if enrichment_csv.exists() else pd.DataFrame(columns=columns)
    if not set(columns) <= set(table.columns): table = pd.DataFrame(columns=columns)
    significant_rows = table[table["significant"].astype(bool)]
    stars = {c: next((st for cut, st in SIG_LEVELS if q < cut), "") for c, q in zip(significant_rows["cluster"], significant_rows["chi2_q"])}
    selected = set(table.loc[table["selected"].astype(bool), "cluster"])
    probe_selected = set(table.loc[table["probe_selected"].astype(bool), "cluster"])
    print(f"[PYTHON-INFO][SELECTION] {len(selected)} selected joint clusters from {enrichment_csv.name}: {', '.join(sorted(selected, key=lambda c: (len(c), c)))}; {len(probe_selected)} probe clusters: {', '.join(sorted(probe_selected, key=lambda c: (len(c), c)))}")
    return stars, selected, probe_selected
def species_colors(obs: pd.DataFrame, names: dict[str, str]) -> dict[str, str]:
    """Colour every sample as the barplots of h5ad_scanpy.py do: PALETTE_ALT by position, targets first then probes.

    Args:
        obs (pd.DataFrame): Observations with ``sc_label`` and ``sc_type``.
        names (dict[str, str]): Label -> display name.

    Returns:
        dict[str, str]: Display name -> colour.
    """
    inputs = list(dict.fromkeys(obs["sc_label"]))
    is_target = {sp: (obs.loc[obs["sc_label"] == sp, "sc_type"] == "target").all() for sp in inputs}
    ordered = [sp for sp in inputs if is_target[sp]] + [sp for sp in inputs if not is_target[sp]]
    return {names.get(sp, sp): PALETTE_ALT[i % len(PALETTE_ALT)] for i, sp in enumerate(ordered)}
###################################### VIEW_BUILDERS
def bars_payload(ctx: dict) -> dict:
    """Count the proteins of every species in every cluster of the barplot series.

    Args:
        ctx (dict): Page context of build_page.

    Returns:
        dict: Clusters, their asterisks, the species and the counts per species and cluster.
    """
    counts = pd.crosstab(ctx["series"][ctx["bar_key"]], ctx["species"])
    counts = counts.loc[sorted(counts.index, key=lambda c: (len(c), c.lower()))]
    species_order = [c[0] for c in ctx["sets"][SPECIES_KEY][1][1:] if c[0] in counts.columns]
    return dict(clusters=list(counts.index), stars=[ctx["stars"].get(c, "") for c in counts.index], species=species_order, counts=[counts[sp].astype(int).tolist() for sp in species_order])
VIEW_BUILDERS = {"bars": bars_payload}
###################################### VIEWER
def build_page(adata: ad.AnnData, series: pd.DataFrame, bar_key: str, mmseqs_keys: list[str], view: str, names: dict[str, str], stars: dict[str, str], selected: set[str], probe_selected: set[str], sp_colors: dict[str, str], rng: np.random.Generator, subtitle: str) -> dict:
    """Build the HTML page payload of one view: the points and their label sets, then one payload per VIEW_BUILDERS entry.

    Args:
        adata (ad.AnnData): Projections with 3D layouts.
        series (pd.DataFrame): Summary SERIAL per protein (rows aligned with adata) and series column.
        bar_key (str): Series column of the barplot and of the Sel/Probe buttons.
        mmseqs_keys (list[str]): MMseqs2 series columns, internal then joint, by decreasing identity.
        view (str): ``joint`` or a target label.
        names (dict[str, str]): Label -> display name.
        stars (dict[str, str]): Significant joint cluster -> asterisks.
        selected (set[str]): Selected joint clusters.
        probe_selected (set[str]): Probe clusters.
        sp_colors (dict[str, str]): Display name -> species colour shared with the report.
        rng (np.random.Generator): Generator of the point subsampling.
        subtitle (str): Run description shown under the title.

    Returns:
        dict: Page payload.
    """
    joint = view == "joint"
    mask = np.ones(adata.n_obs, dtype=bool) if joint else (adata.obs["sc_label"] == view).values
    coords = {m: np.asarray(adata.obsm[f"X_{m}_{DIM}d_{'j' if joint else 'i'}"][mask], dtype=np.float64) for m in METHODS}
    ids = adata.obs["record"].astype(str).values[mask]
    species = adata.obs["sc_label"].astype(str)[mask].map(lambda s: names.get(s, s)).reset_index(drop=True)
    page_mmseqs = [k for k in mmseqs_keys if not joint or MMSEQS_PATTERN.search(k).group(1) == "j"]
    keys = [k for k in series.columns if k not in mmseqs_keys] + page_mmseqs
    view_series = series.loc[mask, keys].reset_index(drop=True)
    n_max = args.max_points_per_view
    keep = np.sort(rng.choice(len(ids), n_max, replace=False)) if n_max and len(ids) > n_max else np.arange(len(ids))
    sets = {SPECIES_KEY: categories(species, sp_colors)} | {col: categories(view_series[col], min_size=args.min_series_cluster_size, max_colored=args.max_colored_clusters) if col in mmseqs_keys else categories(view_series[col]) for col in keys}
    ctx = dict(series=view_series, species=species, sets=sets, bar_key=bar_key, stars=stars)
    views = {name: builder(ctx) for name, builder in VIEW_BUILDERS.items()}
    clusters = views["bars"]["clusters"]
    points = f"{len(keep):,}/{len(ids):,} points shown" if len(keep) < len(ids) else f"{len(ids):,} points"
    return dict(title="All samples" if joint else names.get(view, view), subtitle=f"{subtitle} · {points}", slug=view, methods=[[m, t] for m, t in METHODS.items()], keys=keys,
                coords={m: encode(c[keep], np.float32) for m, c in coords.items()}, ids="\n".join(ids[keep]),
                sets={k: dict(codes=encode(codes[keep], np.uint32 if len(cats) > 65535 else np.uint16), wide=len(cats) > 65535, dyn=k in mmseqs_keys, cats=cats) for k, (codes, cats) in sets.items()}, views=views,
                selected=[c for c in clusters if c in selected], probe=[c for c in clusters if c in probe_selected])
def write_views(pages: list[dict], bar_key: str, title: str, out_path: Path) -> None:
    """Write all pages into one self-contained paged HTML viewer.

    Args:
        pages (list[dict]): Page payloads.
        bar_key (str): Label set of the barplot and of the Sel/Probe buttons.
        title (str): Document title.
        out_path (Path): Output HTML path.
    """
    config = dict(alpha=ALPHA, ghost_alpha=GHOST_ALPHA, color_ghost=COLOR_BG, color_minor=COLOR_MINOR, color_other=COLOR_OTHER, max_colored=args.max_colored_clusters, font_family=FONT_FAMILY, font_legend=FONT_LEGEND, font_tick=FONT_TICK, font_title=FONT_TITLE, marker_size=MARKER_SIZE, palette=PALETTE_ALT)
    payload = json.dumps(dict(title=title, config=config, hover_keys=[SPECIES_KEY, bar_key], leiden_key=bar_key, pages=pages), separators=(",", ":")).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.read_text().replace("__TITLE__", escape(title), 1).replace("__PLOTLYJS__", get_plotlyjs(), 1).replace("__DATA__", payload, 1), encoding="utf-8")
def interactive(h5ad_path: Path) -> None:
    """Write the paged HTML viewer of the joint view and of every target of one .h5ad.

    Args:
        h5ad_path (Path): projections.h5ad of npz_scanpy.py.
    """
    run_dir = h5ad_path.parent
    target_dir = (Path(args.outdir) / run_dir.name if args.outdir else run_dir) / SUBDIR
    summary_path, out_path = run_dir / SUMMARY_FILE, target_dir / OUTPUT_FILE
    report_series = config_data.get("h5ad_scanpy", {}).get("report_series", REPORT_SERIES)
    enrichment_csv = target_dir.parent / f"species_cluster_enrichment__{report_series}.csv"
    if not summary_path.exists(): raise FileNotFoundError(f"[PYTHON-ERROR] Missing {summary_path}; run compile_summary.py first.")
    src_mtime = max(fp.stat().st_mtime for fp in (Path(__file__), TEMPLATE, h5ad_path, summary_path, enrichment_csv) if fp.exists())
    if not args.overwrite and out_path.exists() and out_path.stat().st_mtime > src_mtime:
        print(f"[PYTHON-INFO] {out_path} is newer than its inputs. Skipping.")
        return
    adata = ad.read_h5ad(h5ad_path)
    if not all(f"X_{m}_{DIM}d_j" in adata.obsm for m in METHODS):
        print(f"[PYTHON-INFO] {h5ad_path}: no {DIM}D UMAP/t-SNE layouts. Skipping.")
        return
    summary = pd.read_csv(summary_path, sep="\t", dtype=str).set_index("record")
    layout_keys = [c for c in summary.columns if any(f"_{m}_{s}_" in c for m in ("u", "t") for s in ("i", "j"))]
    bar_key = next((c for c in layout_keys if c.endswith(f"_{report_series}")), next(c for c in layout_keys if "_u_j_" in c))
    mmseqs_keys = sorted((c for c in summary.columns if MMSEQS_PATTERN.search(c)), key=lambda c: (MMSEQS_PATTERN.search(c).group(1), -int(MMSEQS_PATTERN.search(c).group(2))))
    series_keys = [bar_key] + [k for k in layout_keys if k != bar_key] + mmseqs_keys
    series = summary.loc[adata.obs["record"].astype(str), series_keys].reset_index(drop=True)
    names = species_names()
    names |= {sp: f"{names.get(sp, sp)} (probe)" for sp in adata.obs.loc[adata.obs["sc_type"] == "probe", "sc_label"].unique()}
    stars, selected, probe_selected = load_enrichment(enrichment_csv)
    sp_colors = species_colors(adata.obs, names)
    has_intra = all(f"X_{m}_{DIM}d_i" in adata.obsm for m in METHODS)
    views = ["joint"] + [sp for sp in dict.fromkeys(adata.obs["sc_label"]) if has_intra and np.isfinite(adata.obsm[f"X_umap_{DIM}d_i"][(adata.obs["sc_label"] == sp).values]).all()]
    rng = np.random.default_rng(SEED)
    subtitle = f"{run_dir.parent.name} · {run_dir.name}"
    pages = [build_page(adata, series, bar_key, mmseqs_keys, view, names, stars, selected, probe_selected, sp_colors, rng, subtitle) for view in tqdm(views, desc=f"{run_dir.name} views")]
    target_dir.mkdir(parents=True, exist_ok=True)
    write_views(pages, bar_key, f"{run_dir.parent.name} · {run_dir.name} · views", out_path)
    print(f"[PYTHON-INFO][SAVE] {out_path} ({len(pages)} pages, {len(series_keys)} series, {out_path.stat().st_size / 1e6:.1f} MB)")
###################################### MAIN
def main() -> None:
    """Write the interactive views of every .h5ad passed on the command line."""
    start_time = time.time()
    for path in tqdm([Path(p.strip()) for p in args.h5ad.split(",") if p.strip()], desc="Interactive"):
        interactive(path)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
