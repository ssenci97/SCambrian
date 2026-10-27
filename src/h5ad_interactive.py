#!/usr/bin/env python3
"""Writes one paged, self-contained 3D HTML viewer of the UMAP and t-SNE layouts saved by npz_scanpy.py, plus the HDBSCAN labels of the 3D UMAP.

Per .h5ad a single HTML file is saved under ``<run>/interactive/`` with one page for the joint view and one per species (intra view), browsed with arrows, a page menu or the arrow keys. Every page has three views, switched in the header or with the V key: side-by-side 3D UMAP and t-SNE panels, a static 2D projection of the same 3D layouts (X and Y dimensions, azimuth and elevation, by default dimension 1 vs 2 with no rotation) and a full-size barplot, together with a species/cluster switch; the cluster labels are chosen among Leiden, intra-species and HDBSCAN fitted on the 3D UMAP for every min_cluster_size of ``--hdbscan_min_cluster_sizes``. Inter-species clusters (Leiden, joint HDBSCAN) are named by lowercase letters of their 0-based id (a, b, ..., aa) and intra-species clusters (intra Leiden, per-species HDBSCAN) by their 0-based id as a number string (0, 1, ...), HDBSCAN ids following decreasing size. The chi-square q-value of every significant inter-species cluster (chi-square and fold-change gates of h5ad_scanpy.py), read from its species_cluster_enrichment table, is shown as significance asterisks (* q<0.05, ** q<0.01, *** q<0.001, nothing otherwise) stacked vertically over each stacked bar of the joint view. A left sidebar holds a protein ID finder that rings the protein in both panels, an export of the IDs of the points shown, and species and cluster checklists, combined to select which points are drawn, with the significance asterisks and per-species counts of every inter-species cluster. Every page opens coloured by Leiden cluster with only the joint clusters selected by h5ad_scanpy.py (``selected`` column of the enrichment table, the clusters it re-projects) ticked, restored by the Sel button. The barplot view, with a linear/log switch, stacks the protein count of every inter-species cluster per species, and clicking a bar returns to the last scatter view with that cluster isolated. Points are drawn at alpha ALPHA and at the minimum MARKER_SIZE by default. The camera is kept across all changes until reset. The HDBSCAN labels of all views are also written to one TSV per .h5ad. An .h5ad whose outputs all exist and are newer than this script, its template, the .h5ad and the enrichment table is skipped unless ``--overwrite`` is set.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import base64
import datetime
import hdbscan
from html import escape
import json
import numpy as np
import os
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
HDBSCAN_MAX_CLUSTERS = 15
HDBSCAN_MIN_CLUSTER_SIZES = [500, 800, 1600]
HDBSCAN_MIN_SAMPLES = 10
LABEL_SETS = {"species": "species", "cluster": "Leiden cluster", "intra_cluster": "intra-species cluster"}
MARKER_SIZE = 0.5
MAX_POINTS_PER_VIEW = None
METHODS = {"umap": "UMAP", "tsne": "t-SNE"}
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
SEED = 42
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
SUBDIR = "interactive"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "h5ad_interactive.html"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Interactive 3D UMAP/t-SNE and HDBSCAN HTML views")
parser.add_argument("--h5ad", type=str, required=True, help="Comma-separated .h5ad files written by npz_scanpy.py")
parser.add_argument("--outdir", type=str, default=None, help="Base output directory; defaults to the folder of each .h5ad")
parser.add_argument("--hdbscan_min_cluster_sizes", type=int, nargs="+", default=HDBSCAN_MIN_CLUSTER_SIZES, help="HDBSCAN min_cluster_size values of the slider")
parser.add_argument("--hdbscan_min_samples", type=int, default=HDBSCAN_MIN_SAMPLES, help="HDBSCAN min_samples, capped at min_cluster_size")
parser.add_argument("--hdbscan_max_clusters", type=int, default=HDBSCAN_MAX_CLUSTERS, help="Largest HDBSCAN clusters kept, the rest unassigned")
parser.add_argument("--max_points_per_view", type=int, default=MAX_POINTS_PER_VIEW, help="Random subsample of plotted points per view (None plots all)")
parser.add_argument("--n_jobs", type=int, default=N_JOBS, help="CPU threads")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute outputs that already exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "h5ad_interactive"
params = dict(hdbscan_min_cluster_sizes=HDBSCAN_MIN_CLUSTER_SIZES, hdbscan_min_samples=HDBSCAN_MIN_SAMPLES, hdbscan_max_clusters=HDBSCAN_MAX_CLUSTERS, max_points_per_view=MAX_POINTS_PER_VIEW, overwrite=OVERWRITE)
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
def species_names() -> dict[str, str]:
    """Map each label of the query configuration to its exact_species display name.

    Returns:
        dict[str, str]: Label -> display name (empty if the configuration is missing).
    """
    config_data = (yaml.safe_load(Path(args.yaml_in).read_text()) or {}) if args.yaml_in and Path(args.yaml_in).exists() else {}
    tsv = Path(config_data.get("dw_uniprot_seqs", {}).get("query_table", DEFAULT_QUERY_TSV))
    df = pd.read_csv(tsv, sep="\t", dtype=str) if tsv.exists() else pd.DataFrame(columns=["label"])
    names = df["exact_species"] if "exact_species" in df else df["label"]
    return {lbl: (nm if pd.notna(nm) and str(nm).strip() else lbl) for lbl, nm in zip(df["label"], names) if pd.notna(lbl)}
def letter(i: int) -> str:
    """Name a 0-based inter-species cluster id by lowercase letters, spreadsheet-style (0 -> a, 26 -> aa).

    Args:
        i (int): 0-based cluster index.

    Returns:
        str: Letter name.
    """
    return (letter(i // 26 - 1) if i >= 26 else "") + chr(97 + i % 26)
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
def targets(target_dir: Path, param_tag: str) -> dict[str, Path]:
    """Return every output path of one .h5ad.

    Args:
        target_dir (Path): Output folder of the run.
        param_tag (str): Parameter tag of the .h5ad.

    Returns:
        dict[str, Path]: Output key -> path.
    """
    return {"tsv": target_dir / f"hdbscan_umap{DIM}d__{param_tag}.tsv", "html": target_dir / f"views_{DIM}d__{param_tag}.html"}
def fit_hdbscan_auto(coords: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Fit HDBSCAN and relabel the ``hdbscan_max_clusters`` largest clusters 1..n by decreasing size, the rest 0.

    Args:
        coords (np.ndarray): Layout coordinates.
        min_cluster_size (int): HDBSCAN minimum cluster size.

    Returns:
        np.ndarray: Labels, 0 being unassigned.
    """
    mcs = int(min_cluster_size)
    if len(coords) <= mcs: return np.zeros(len(coords), dtype=int)
    raw = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min(args.hdbscan_min_samples, mcs), metric="euclidean", core_dist_n_jobs=args.n_jobs).fit_predict(coords)
    unique, counts = np.unique(raw[raw >= 0], return_counts=True)
    mapping = np.zeros(raw.max() + 2, dtype=int)
    mapping[unique[np.argsort(-counts, kind="stable")][:args.hdbscan_max_clusters]] = np.arange(1, min(len(unique), args.hdbscan_max_clusters) + 1)
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
def encode(values: np.ndarray, dtype: type) -> str:
    """Encode an array as base64 of its raw bytes in the given dtype.

    Args:
        values (np.ndarray): Values to encode.
        dtype (type): Numpy dtype of the bytes.

    Returns:
        str: Base64 string.
    """
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode()
def write_views(pages: list[dict], title: str, out_path: Path) -> None:
    """Write all pages into one self-contained paged HTML viewer.

    Args:
        pages (list[dict]): Page payloads (title, subtitle, slug, methods, coords, ids, sets, bars, selected).
        title (str): Document title.
        out_path (Path): Output HTML path.
    """
    config = dict(alpha=ALPHA, font_family=FONT_FAMILY, font_legend=FONT_LEGEND, font_tick=FONT_TICK, font_title=FONT_TITLE, marker_size=MARKER_SIZE)
    cluster_keys = list(LABEL_SETS.values())[1:] + [f"HDBSCAN mcs={mcs}" for mcs in args.hdbscan_min_cluster_sizes]
    payload = json.dumps(dict(title=title, config=config, hover_keys=list(LABEL_SETS.values()), cluster_keys=cluster_keys, leiden_key=LABEL_SETS["cluster"], pages=pages), separators=(",", ":")).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.read_text().replace("__TITLE__", escape(title), 1).replace("__PLOTLYJS__", get_plotlyjs(), 1).replace("__DATA__", payload, 1), encoding="utf-8")
###################################### REPORT
def interactive(h5ad_path: Path) -> None:
    """Write the paged HTML viewer of every view of one .h5ad, and the HDBSCAN TSV.

    Args:
        h5ad_path (Path): .h5ad written by npz_scanpy.py.
    """
    head = ad.read_h5ad(h5ad_path, backed="r")
    param_tag, views = head.uns["param_tag"], list_views(head.obs, set(head.obsm.keys()))
    head.file.close()
    target_dir = (Path(args.outdir) / h5ad_path.parent.name if args.outdir else h5ad_path.parent) / SUBDIR
    out_paths = targets(target_dir, param_tag)
    if not views:
        print(f"[PYTHON-INFO] {h5ad_path.name}: no {DIM}D UMAP/t-SNE layouts. Skipping.")
        return
    enrichment_csv = target_dir.parent / f"species_cluster_enrichment__{param_tag}.csv"
    src_mtime = max(fp.stat().st_mtime for fp in (Path(__file__), TEMPLATE, h5ad_path, enrichment_csv) if fp.exists())
    if not args.overwrite and all(p.exists() and p.stat().st_mtime > src_mtime for p in out_paths.values()):
        print(f"[PYTHON-INFO] {h5ad_path.name}: all {len(out_paths)} outputs already exist in {target_dir} and are newer than their inputs. Skipping.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path)
    obs, rng, rows, pages, names = adata.obs.astype({"species": str}), np.random.default_rng(SEED), [], [], species_names()
    enrichment_table = pd.read_csv(enrichment_csv).drop_duplicates("cluster") if enrichment_csv.exists() else pd.DataFrame()
    enrichment_table = enrichment_table.astype({"cluster": str}) if {"cluster", "chi2_q", "significant", "selected"} <= set(enrichment_table.columns) else pd.DataFrame(columns=["cluster", "chi2_q", "significant", "selected"])
    significant_rows = enrichment_table[enrichment_table["significant"].astype(bool)]
    stars, selected = {c: next((st for cut, st in SIG_LEVELS if q < cut), "") for c, q in zip(significant_rows["cluster"], significant_rows["chi2_q"])}, set(enrichment_table.loc[enrichment_table["selected"].astype(bool), "cluster"])
    print(f"[PYTHON-INFO][SELECTION] {len(selected)} selected joint clusters from {enrichment_csv.name}: {', '.join(sorted(selected, key=lambda c: (len(c), c)))}")
    for view in tqdm(views, desc=f"{h5ad_path.parent.name} views"):
        mask = np.ones(adata.n_obs, dtype=bool) if view == "joint" else (obs["species"] == view).values
        suffix = "" if view == "joint" else "_intra"
        coords = {m: np.asarray(adata.obsm[f"X_{m}_{DIM}d{suffix}"][mask], dtype=np.float64) for m in METHODS}
        view_obs = obs[mask].reset_index(drop=True)
        view_obs["species"] = view_obs["species"].map(lambda s: names.get(s, s))
        intra = view_obs["intra_cluster"].astype("object").map(lambda c: str(c) if pd.notna(c) else None)
        intra = (view_obs["species"] + " · " + intra).where(intra.notna()) if view == "joint" else intra
        hdbscan_name = (lambda x: letter(x - 1)) if view == "joint" else (lambda x: str(x - 1))
        hdbscan_labels = {mcs: fit_hdbscan_auto(coords["umap"], mcs) for mcs in tqdm(args.hdbscan_min_cluster_sizes, desc=f"{view} HDBSCAN", leave=False)}
        rows.append(pd.DataFrame({"protein_id": view_obs["protein_id"].astype(str), "species": view_obs["species"], "view": view} | {f"mcs_{mcs}": lab for mcs, lab in hdbscan_labels.items()}))
        keep = np.sort(rng.choice(len(view_obs), args.max_points_per_view, replace=False)) if args.max_points_per_view and len(view_obs) > args.max_points_per_view else np.arange(len(view_obs))
        series = {"species": view_obs["species"], "cluster": view_obs["cluster"].map(lambda c: letter(int(c))), "intra_cluster": intra}
        sets = {LABEL_SETS[k]: categories(series[k]) for k in LABEL_SETS} | {f"HDBSCAN mcs={mcs}": categories(pd.Series([hdbscan_name(int(x)) if x > 0 else None for x in lab], dtype="object")) for mcs, lab in hdbscan_labels.items()}
        counts_by_cluster = pd.crosstab(series["cluster"], view_obs["species"])
        counts_by_cluster, species_order = counts_by_cluster.loc[sorted(counts_by_cluster.index, key=lambda c: (len(c), c))], [c[0] for c in sets["species"][1][1:] if c[0] in counts_by_cluster.columns]
        bars = dict(clusters=list(counts_by_cluster.index), stars=[stars.get(c, "") if view == "joint" else "" for c in counts_by_cluster.index], species=species_order, counts=[counts_by_cluster[sp].astype(int).tolist() for sp in species_order])
        selected_here = [c for c in counts_by_cluster.index if c in selected]
        if view == "joint" and len(selected_here) < len(selected): print(f"[PYTHON-WARNING][SELECTION] Selected clusters missing from the Leiden labels: {', '.join(sorted(selected - set(selected_here)))}")
        points = f"{len(keep):,}/{len(view_obs):,} points shown" if len(keep) < len(view_obs) else f"{len(view_obs):,} points"
        pages.append(dict(title="All species" if view == "joint" else names.get(view, view), subtitle=f"{h5ad_path.parent.name} · {param_tag} · {points}", slug=f"{view}_{param_tag}", methods=[[m, t] for m, t in METHODS.items()], coords={m: encode(c[keep], np.float32) for m, c in coords.items()}, ids="\n".join(view_obs["protein_id"].astype(str).values[keep]), sets={k: dict(codes=encode(codes[keep], np.uint16), cats=cats) for k, (codes, cats) in sets.items()}, bars=bars, selected=selected_here))
    write_views(pages, f"{h5ad_path.parent.name} · {param_tag} · 3D views", out_paths["html"])
    print(f"[PYTHON-INFO][SAVE] {out_paths['html']} ({len(pages)} pages, {out_paths['html'].stat().st_size / 1e6:.1f} MB)")
    pd.concat(rows, ignore_index=True).to_csv(out_paths["tsv"], sep="\t", index=False)
    print(f"[PYTHON-INFO][SAVE] {out_paths['tsv']}")
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
