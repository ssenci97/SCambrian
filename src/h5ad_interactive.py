#!/usr/bin/env python3
"""Writes one paged, self-contained 3D HTML viewer of the UMAP and t-SNE layouts of npz_scanpy.py, coloured by the cluster series of compile_summary.py.

Per projections.h5ad a single ``<model>/interactive/views_3d.html`` is saved, with one page for the joint view and one per target (internal view), browsed with arrows, a page menu or the arrow keys. Every page has three views, switched in the header or with the V key: side-by-side 3D UMAP and t-SNE panels, a static 2D projection of the same 3D layouts and a full-size barplot, together with a species/cluster switch. The cluster label sets are the UMAP and t-SNE series of ``sc_summary.tsv`` (``{analysis}_{u|t}_{i|j}_leiden{2|3}d``), so every name in the viewer is the SERIAL of the summary: letters for joint clusters (UPPER case when the cluster holds probe and target proteins) and ``<label>_001`` for internal ones; MMseqs2 series are left out because their tens of thousands of clusters do not fit a checklist. The barplot and the Sel/Probe buttons use the report series (``h5ad_scanpy.report_series``): Sel ticks the joint clusters selected by h5ad_scanpy.py (``selected`` column of its enrichment table, whose significant clusters also get chi-square asterisks over their bars), Probe ticks the probe clusters selected by h5ad_scanpy.py (``probe_selected`` column, probe enrichment), a subset of the UPPER-case clusters. A left sidebar holds a protein ID finder, an export of the IDs of the points shown, and species and cluster checklists. Probe samples are suffixed "(probe)" and get no internal page. Species colours follow the barplots of h5ad_scanpy.py (PALETTE_ALT by sample position, targets first). In the 2D view a lasso selects the shown points and cluster labels can be toggled. An output newer than this script, its template, the .h5ad, the summary and the enrichment table is skipped unless ``--overwrite`` is set.
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
import resource
import time
from tqdm import tqdm
import yaml
ALPHA = 0.8
COLOR_BG = "#BFC5CE"
DEFAULT_QUERY_TSV = "configs/query_configs.tsv"
DIM = 3
FONT_FAMILY = "DejaVu Sans"
FONT_LEGEND = 15
FONT_TICK = 12
FONT_TITLE = 20
MARKER_SIZE = 0.5
MAX_POINTS_PER_VIEW = None
METHODS = {"umap": "UMAP", "tsne": "t-SNE"}
OUTPUT_FILE = "views_3d.html"
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
PALETTE_ALT = PALETTE[::2] + PALETTE[1::2]
REPORT_SERIES = "u_j_leiden2d"
SEED = 42
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
SPECIES_KEY = "species"
SUBDIR = "interactive"
SUMMARY_FILE = "sc_summary.tsv"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "h5ad_interactive.html"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Interactive 3D UMAP/t-SNE HTML views of the summary cluster series")
parser.add_argument("--h5ad", type=str, required=True, help="Comma-separated projections.h5ad files of npz_scanpy.py, each with its sc_summary.tsv")
parser.add_argument("--outdir", type=str, default=None, help="Base output directory; defaults to the folder of each .h5ad")
parser.add_argument("--max_points_per_view", type=int, default=MAX_POINTS_PER_VIEW, help="Random subsample of plotted points per view (None plots all)")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute outputs that already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "h5ad_interactive"
params = dict(max_points_per_view=MAX_POINTS_PER_VIEW, overwrite=OVERWRITE)
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
def categories(values: pd.Series, colors: dict[str, str] | None = None) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    """Encode labels as integer codes ordered by decreasing size, coloured by alternating PALETTE colours or by a fixed mapping.

    Args:
        values (pd.Series): Labels per point (NaN allowed, drawn as unassigned).
        colors (dict[str, str] | None): Fixed label -> colour mapping; other labels take PALETTE_ALT by size.

    Returns:
        tuple[np.ndarray, list[tuple[str, str, int]]]: Codes per point and (name, colour, count) per code.
    """
    values = values.astype("object").where(values.notna(), None)
    counts = values.dropna().value_counts(sort=True)
    cats = [("unassigned", COLOR_BG, int(values.isna().sum()))] + [(str(n), (colors or {}).get(n, PALETTE_ALT[i % len(PALETTE_ALT)]), int(c)) for i, (n, c) in enumerate(zip(counts.index, counts.values))]
    lookup = {n: i + 1 for i, n in enumerate(counts.index)}
    return np.asarray([lookup.get(v, 0) for v in values], dtype=np.int32), cats
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
###################################### VIEWER
def build_page(adata: ad.AnnData, series: pd.DataFrame, bar_key: str, view: str, names: dict[str, str], stars: dict[str, str], selected: set[str], probe_selected: set[str], sp_colors: dict[str, str], rng: np.random.Generator, subtitle: str) -> dict:
    """Build the HTML page payload of one view.

    Args:
        adata (ad.AnnData): Projections with 3D layouts.
        series (pd.DataFrame): Summary SERIAL per protein (rows aligned with adata) and series column.
        bar_key (str): Series column of the barplot and of the Sel/Probe buttons.
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
    view_series = series[mask].reset_index(drop=True)
    n_max = args.max_points_per_view
    keep = np.sort(rng.choice(len(ids), n_max, replace=False)) if n_max and len(ids) > n_max else np.arange(len(ids))
    sets = {SPECIES_KEY: categories(species, sp_colors)} | {col: categories(view_series[col]) for col in series.columns}
    counts_by_cluster = pd.crosstab(view_series[bar_key], species)
    counts_by_cluster = counts_by_cluster.loc[sorted(counts_by_cluster.index, key=lambda c: (len(c), c.lower()))]
    species_order = [c[0] for c in sets[SPECIES_KEY][1][1:] if c[0] in counts_by_cluster.columns]
    bars = dict(clusters=list(counts_by_cluster.index), stars=[stars.get(c, "") for c in counts_by_cluster.index], species=species_order, counts=[counts_by_cluster[sp].astype(int).tolist() for sp in species_order])
    points = f"{len(keep):,}/{len(ids):,} points shown" if len(keep) < len(ids) else f"{len(ids):,} points"
    return dict(title="All samples" if joint else names.get(view, view), subtitle=f"{subtitle} · {points}", slug=view, methods=[[m, t] for m, t in METHODS.items()],
                coords={m: encode(c[keep], np.float32) for m, c in coords.items()}, ids="\n".join(ids[keep]),
                sets={k: dict(codes=encode(codes[keep], np.uint16), cats=cats) for k, (codes, cats) in sets.items()}, bars=bars,
                selected=[c for c in counts_by_cluster.index if c in selected], probe=[c for c in counts_by_cluster.index if c in probe_selected])
def write_views(pages: list[dict], series_keys: list[str], bar_key: str, title: str, out_path: Path) -> None:
    """Write all pages into one self-contained paged HTML viewer.

    Args:
        pages (list[dict]): Page payloads.
        series_keys (list[str]): Cluster label sets, in menu order.
        bar_key (str): Label set of the barplot and of the Sel/Probe buttons.
        title (str): Document title.
        out_path (Path): Output HTML path.
    """
    config = dict(alpha=ALPHA, font_family=FONT_FAMILY, font_legend=FONT_LEGEND, font_tick=FONT_TICK, font_title=FONT_TITLE, marker_size=MARKER_SIZE)
    payload = json.dumps(dict(title=title, config=config, hover_keys=[SPECIES_KEY, bar_key], cluster_keys=[bar_key] + [k for k in series_keys if k != bar_key], leiden_key=bar_key, pages=pages), separators=(",", ":")).replace("</", "<\\/")
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
    series_keys = [c for c in summary.columns if any(f"_{m}_{s}_" in c for m in ("u", "t") for s in ("i", "j"))]
    bar_key = next((c for c in series_keys if c.endswith(f"_{report_series}")), next(c for c in series_keys if "_u_j_" in c))
    series = summary.loc[adata.obs["record"].astype(str), series_keys].reset_index(drop=True)
    names = species_names()
    names |= {sp: f"{names.get(sp, sp)} (probe)" for sp in adata.obs.loc[adata.obs["sc_type"] == "probe", "sc_label"].unique()}
    stars, selected, probe_selected = load_enrichment(enrichment_csv)
    sp_colors = species_colors(adata.obs, names)
    has_intra = all(f"X_{m}_{DIM}d_i" in adata.obsm for m in METHODS)
    views = ["joint"] + [sp for sp in dict.fromkeys(adata.obs["sc_label"]) if has_intra and np.isfinite(adata.obsm[f"X_umap_{DIM}d_i"][(adata.obs["sc_label"] == sp).values]).all()]
    rng = np.random.default_rng(SEED)
    subtitle = f"{run_dir.parent.name} · {run_dir.name}"
    pages = [build_page(adata, series, bar_key, view, names, stars, selected, probe_selected, sp_colors, rng, subtitle) for view in tqdm(views, desc=f"{run_dir.name} views")]
    target_dir.mkdir(parents=True, exist_ok=True)
    write_views(pages, series_keys, bar_key, f"{run_dir.parent.name} · {run_dir.name} · 3D views", out_path)
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
