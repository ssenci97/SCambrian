#!/usr/bin/env python3
"""Writes interactive 3D HTML views of the UMAP and t-SNE layouts saved by npz_scanpy.py, plus an HDBSCAN explorer of the 3D UMAP.

For the joint view and for every species (intra view) two standalone HTML files are saved under ``<run>/interactive/``: side-by-side 3D UMAP and t-SNE panels recoloured by buttons (species, Leiden cluster, intra-species cluster), and the same panels coloured by HDBSCAN fitted on the 3D UMAP for every min_cluster_size of ``--mcs_grid``, switched with a slider. The HDBSCAN labels of all views are also written to one TSV per .h5ad. An .h5ad whose outputs all exist is skipped unless ``--overwrite`` is set.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import datetime
import hdbscan
import numpy as np
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import resource
import time
from tqdm import tqdm
import yaml
ALPHA = 0.8
COLOR_BG = "#BFC5CE"
COLOR_BY = ["species", "cluster", "intra_cluster"]
DIM = 3
FIGURE_HEIGHT = 900
FIGURE_WIDTH = 1700
FONT_FAMILY = "DejaVu Sans"
FONT_LEGEND = 15
FONT_TICK = 12
FONT_TITLE = 20
LEGEND_MAX = 40
LINE_COLOR = "black"
MARKER_SIZE = 2
MAX_CLUSTERS = 15
MAX_POINTS = None
MCS_GRID = [50, 100, 200, 400, 800, 1600]
METHODS = {"umap": "UMAP", "tsne": "t-SNE"}
MIN_SAMPLES = 10
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
OVERWRITE = False
PALETTE = ["#ECDE90", "#93DEB6", "#F1B584", "#76BBE2", "#DA919B", "#899E64", "#8E81B9", "#A66C58", "#30767A", "#7A5271"]
SEED = 42
SUBDIR = "interactive"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Interactive 3D UMAP/t-SNE and HDBSCAN HTML views")
parser.add_argument("--h5ad", type=str, required=True, help="Comma-separated .h5ad files written by npz_scanpy.py")
parser.add_argument("--outdir", type=str, default=None, help="Base output directory; defaults to the folder of each .h5ad")
parser.add_argument("--mcs_grid", type=int, nargs="+", default=MCS_GRID, help="HDBSCAN min_cluster_size values of the slider")
parser.add_argument("--min_samples", type=int, default=MIN_SAMPLES, help="HDBSCAN min_samples, capped at min_cluster_size")
parser.add_argument("--max_clusters", type=int, default=MAX_CLUSTERS, help="Largest HDBSCAN clusters kept, the rest unassigned")
parser.add_argument("--max_points", type=int, default=MAX_POINTS, help="Random subsample of plotted points per view (None plots all)")
parser.add_argument("--n_jobs", type=int, default=N_JOBS, help="CPU threads")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute outputs that already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
yargs, _ = parser.parse_known_args()
script_key = "h5ad_interactive"
params = dict(mcs_grid=MCS_GRID, min_samples=MIN_SAMPLES, max_clusters=MAX_CLUSTERS, max_points=MAX_POINTS, overwrite=OVERWRITE)
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
def list_views(obs: pd.DataFrame, keys: set[str]) -> list[str]:
    """List the joint view and every species with 3D intra layouts of all methods.

    Args:
        obs (pd.DataFrame): Observations of the .h5ad.
        keys (set[str]): ``obsm`` keys of the .h5ad.

    Returns:
        list[str]: ``joint`` followed by the species names.
    """
    views = ["joint"] if all(f"X_{m}_{DIM}d" in keys for m in METHODS) else []
    if all(f"X_{m}_{DIM}d_intra" in keys for m in METHODS):
        views += [sp for sp in obs["species"].astype(str).unique() if obs.loc[obs["species"].astype(str) == sp, "intra_cluster"].notna().any()]
    return views
def targets(target_dir: Path, views: list[str], param_tag: str) -> dict[str, Path]:
    """Return every output path of one .h5ad.

    Args:
        target_dir (Path): Output folder of the run.
        views (list[str]): Views of the .h5ad.
        param_tag (str): Parameter tag of the .h5ad.

    Returns:
        dict[str, Path]: Output key -> path.
    """
    out = {"tsv": target_dir / f"hdbscan_umap{DIM}d__{param_tag}.tsv"}
    for v in views:
        out[f"{v}|layouts"] = target_dir / f"{v}_{DIM}d_umap_tsne__{param_tag}.html"
        out[f"{v}|hdbscan"] = target_dir / f"{v}_hdbscan_umap{DIM}d__{param_tag}.html"
    return out
def fit_hdbscan_auto(coords: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Fit HDBSCAN and relabel the ``max_clusters`` largest clusters 1..n by decreasing size, the rest 0.

    Args:
        coords (np.ndarray): Layout coordinates.
        min_cluster_size (int): HDBSCAN minimum cluster size.

    Returns:
        np.ndarray: Labels, 0 being unassigned.
    """
    mcs = int(min_cluster_size)
    if len(coords) <= mcs: return np.zeros(len(coords), dtype=int)
    raw = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min(args.min_samples, mcs), metric="euclidean", core_dist_n_jobs=args.n_jobs).fit_predict(coords)
    unique, counts = np.unique(raw[raw >= 0], return_counts=True)
    mapping = np.zeros(raw.max() + 2, dtype=int)
    mapping[unique[np.argsort(-counts, kind="stable")][:args.max_clusters]] = np.arange(1, min(len(unique), args.max_clusters) + 1)
    return mapping[raw]
def categories(values: pd.Series, unassigned: str | None = None) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    """Encode labels as integer codes ordered by decreasing size, coloured by alternating PALETTE colours.

    Args:
        values (pd.Series): Labels per point (NaN allowed).
        unassigned (str | None): Name of the missing or 0 label, drawn in COLOR_BG.

    Returns:
        tuple[np.ndarray, list[tuple[str, str, int]]]: Codes per point and (name, colour, count) per code.
    """
    values = values.astype("object").where(values.notna(), None)
    counts = values.dropna().value_counts(sort=True)
    names = [str(n) for n in counts.index]
    cats = [(unassigned or "unassigned", COLOR_BG, int(values.isna().sum()))] + [(n, PALETTE[i % len(PALETTE)], int(c)) for i, (n, c) in enumerate(zip(names, counts.values))]
    lookup = {n: i + 1 for i, n in enumerate(counts.index)}
    return np.asarray([lookup.get(v, 0) for v in values], dtype=np.int32), cats
def colorscale(cats: list[tuple[str, str, int]]) -> list[list]:
    """Build a discrete colorscale mapping integer code i to the colour of category i.

    Args:
        cats (list[tuple[str, str, int]]): (name, colour, count) per code.

    Returns:
        list[list]: Plotly colorscale.
    """
    k = len(cats)
    return [s for i, (_, c, _) in enumerate(cats) for s in ([i / k, c], [(i + 1) / k, c])]
def build_figure(coords: dict[str, np.ndarray], hover: np.ndarray, sets: dict[str, tuple[np.ndarray, list[tuple[str, str, int]]]], title: str, control: str) -> go.Figure:
    """Build side-by-side 3D panels whose colouring switches between label sets with buttons or a slider.

    Args:
        coords (dict[str, np.ndarray]): 3D layout per method.
        hover (np.ndarray): Columns protein_id, species, cluster, intra_cluster per point.
        sets (dict[str, tuple[np.ndarray, list[tuple[str, str, int]]]]): Set name -> (codes per point, categories).
        title (str): Figure title.
        control (str): ``buttons`` or ``slider``.

    Returns:
        go.Figure: Figure with 2 data traces followed by the legend traces of every set.
    """
    fig = make_subplots(rows=1, cols=len(coords), specs=[[{"type": "scatter3d"}] * len(coords)], subplot_titles=[METHODS[m] for m in coords], horizontal_spacing=0.01)
    template = "<b>%{customdata[0]}</b><br>species: %{customdata[1]}<br>cluster: %{customdata[2]}<br>intra_cluster: %{customdata[3]}<extra></extra>"
    first_codes, first_cats = next(iter(sets.values()))
    for col, (m, c) in enumerate(coords.items(), 1):
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers", customdata=hover, hovertemplate=template, showlegend=False, marker=dict(size=MARKER_SIZE, opacity=ALPHA, color=first_codes, colorscale=colorscale(first_cats), cmin=-0.5, cmax=len(first_cats) - 0.5, showscale=False)), row=1, col=col)
    n_data, legend_idx = len(coords), {}
    for name, (_, cats) in sets.items():
        shown = sorted((i for i in range(len(cats)) if cats[i][2] > 0), key=lambda i: (i == 0, -cats[i][2]))[:LEGEND_MAX]
        legend_idx[name] = list(range(len(fig.data), len(fig.data) + len(shown)))
        for i in shown:
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=10, color=cats[i][1]), name=f"{cats[i][0]} (n={cats[i][2]:,})", visible=name == next(iter(sets))), row=1, col=1)
    steps, legend_colors = [], [t.marker.color for t in fig.data[n_data:]]
    for name, (codes, cats) in sets.items():
        visible = [True] * n_data + [i in legend_idx[name] for i in range(n_data, len(fig.data))]
        steps.append(dict(label=name, method="update", args=[{"visible": visible, "marker.color": [codes] * n_data + legend_colors, "marker.colorscale": [colorscale(cats)] * len(fig.data), "marker.cmax": [len(cats) - 0.5] * len(fig.data)}, {}]))
    if control == "buttons":
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", buttons=steps, x=0.5, xanchor="center", y=1.08, yanchor="bottom", font=dict(size=FONT_LEGEND), showactive=True)])
    else:
        fig.update_layout(sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="HDBSCAN min_cluster_size: ", font=dict(size=FONT_LEGEND)), pad=dict(t=10), x=0.15, len=0.7, y=1.02, yanchor="bottom", font=dict(size=FONT_TICK))])
    axis = dict(backgroundcolor="white", showbackground=True, showline=True, linecolor=LINE_COLOR, gridcolor="#DDDDDD", zerolinecolor="#BBBBBB", tickfont=dict(size=FONT_TICK, family=FONT_FAMILY))
    for col, m in enumerate(coords, 1):
        fig.update_layout({f"scene{'' if col == 1 else col}": {f"{ax}axis": dict(title=f"{METHODS[m]}-{i + 1}", **axis) for i, ax in enumerate("xyz")} | {"aspectmode": "cube"}})
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=FONT_TITLE)), width=FIGURE_WIDTH, height=FIGURE_HEIGHT, paper_bgcolor="white", hovermode="closest", font=dict(size=FONT_TICK, family=FONT_FAMILY, color="#222222"), margin=dict(l=10, r=10, t=140, b=10), legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top", font=dict(size=FONT_LEGEND), itemsizing="constant"))
    fig.update_annotations(font=dict(size=FONT_TITLE, family=FONT_FAMILY))
    return fig
###################################### REPORT
def interactive(h5ad_path: Path) -> None:
    """Write the layout and HDBSCAN HTML views of every view of one .h5ad, and the HDBSCAN TSV.

    Args:
        h5ad_path (Path): .h5ad written by npz_scanpy.py.
    """
    head = ad.read_h5ad(h5ad_path, backed="r")
    param_tag, views = head.uns["param_tag"], list_views(head.obs, set(head.obsm.keys()))
    head.file.close()
    target_dir = (Path(args.outdir) / h5ad_path.parent.name if args.outdir else h5ad_path.parent) / SUBDIR
    outs = targets(target_dir, views, param_tag)
    if not views:
        print(f"[PYTHON-INFO] {h5ad_path.name}: no {DIM}D UMAP/t-SNE layouts. Skipping.")
        return
    if not args.overwrite and all(p.exists() for p in outs.values()):
        print(f"[PYTHON-INFO] {h5ad_path.name}: all {len(outs)} outputs already exist in {target_dir}. Skipping.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path)
    obs, rng, rows = adata.obs.astype({"species": str}), np.random.default_rng(SEED), []
    for view in tqdm(views, desc=f"{h5ad_path.parent.name} views"):
        mask = np.ones(adata.n_obs, dtype=bool) if view == "joint" else (obs["species"] == view).values
        suffix = "" if view == "joint" else "_intra"
        coords = {m: np.asarray(adata.obsm[f"X_{m}_{DIM}d{suffix}"][mask], dtype=np.float64) for m in METHODS}
        vobs = obs[mask].reset_index(drop=True)
        intra = vobs["species"] + ":" + vobs["intra_cluster"].astype(str) if view == "joint" else vobs["intra_cluster"]
        intra = intra.where(vobs["intra_cluster"].notna())
        hover = np.column_stack([vobs["protein_id"].astype(str), vobs["species"], vobs["cluster"].astype(str), intra.astype(str)])
        hd = {mcs: fit_hdbscan_auto(coords["umap"], mcs) for mcs in tqdm(args.mcs_grid, desc=f"{view} HDBSCAN", leave=False)}
        rows.append(pd.DataFrame({"protein_id": vobs["protein_id"].astype(str), "species": vobs["species"], "view": view} | {f"mcs_{mcs}": lab for mcs, lab in hd.items()}))
        keep = np.sort(rng.choice(len(vobs), args.max_points, replace=False)) if args.max_points and len(vobs) > args.max_points else np.arange(len(vobs))
        sub = {m: c[keep] for m, c in coords.items()}
        series = {"species": vobs["species"], "cluster": vobs["cluster"], "intra_cluster": intra}
        layout_sets = {k: categories(series[k]) for k in COLOR_BY if not (view != "joint" and k == "species")}
        layout_sets = {k: (codes[keep], cats) for k, (codes, cats) in layout_sets.items()}
        hd_sets = {}
        for mcs, lab in hd.items():
            codes, cats = categories(pd.Series(np.where(lab > 0, [f"U-{x:02d}" for x in lab], None)))
            hd_sets[str(mcs)] = (codes[keep], cats)
        title = f"{h5ad_path.parent.name} | {view} | {param_tag}" + (f" | {len(keep):,}/{len(vobs):,} points shown" if len(keep) < len(vobs) else f" | {len(vobs):,} points")
        build_figure(sub, hover[keep], layout_sets, title, "buttons").write_html(outs[f"{view}|layouts"], include_plotlyjs="cdn")
        build_figure(sub, hover[keep], hd_sets, f"HDBSCAN on {DIM}D UMAP | {title}", "slider").write_html(outs[f"{view}|hdbscan"], include_plotlyjs="cdn")
        print(f"[PYTHON-INFO][SAVE] {outs[f'{view}|layouts']} and {outs[f'{view}|hdbscan'].name}")
    pd.concat(rows, ignore_index=True).to_csv(outs["tsv"], sep="\t", index=False)
    print(f"[PYTHON-INFO][SAVE] {outs['tsv']}")
###################################### MAIN
def main() -> None:
    """Write the interactive views of every .h5ad passed on the command line."""
    start_time = time.time()
    for path in tqdm([Path(p.strip()) for p in args.h5ad.split(",") if p.strip()], desc="Interactive"):
        interactive(path)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    exec_time = time.time() - start_time
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[PYTHON-INFO] {date_str} {exec_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
