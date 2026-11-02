#!/usr/bin/env python3
"""Names every cluster of an analysis and compiles the final summary: the kept rows of sc_metadata.tsv with one column per cluster series.

A series is one clustering, named ``{analysis}_{u|t|s}_{i|j}_{stat}``: ``u``/``t`` for UMAP/t-SNE layouts (npz_scanpy.py, stat ``leiden2d``, ``hdbscan3d``, ...), ``s`` for MMseqs2 sequence clustering (run_mmseqs2.py, stat ``id90`` for 90% identity, ...); ``i`` for internal clusters (each target alone) and ``j`` for joint clusters (all samples together). A cluster is ``{series}_{SERIAL}``. Within each series clusters are ranked by decreasing size (ties by first appearance); internal SERIALs are ``{label}_001``, ``{label}_002``, ... and joint SERIALs are spreadsheet letters ``a, b, ..., z, aa, ...``, written in UPPER case when the cluster holds at least one probe and at least one target protein. HDBSCAN noise and series that do not apply (internal series of a probe) are left blank. Outputs, in ``<analysis_dir>/<model>/``:
    - ``sc_summary.tsv``: one row per kept sequence (columns of sc_metadata.tsv), one column per series, the SERIAL in each cell;
    - ``sc_joint_clusters.tsv``: per joint series and cluster, its size, ``n_<sample>`` and ``pct_<sample>`` (percent of the cluster), ``pct_probe`` and ``probe_cluster``;
    - ``sc_probe_capture.tsv``: per target and joint series, the target proteins inside probe clusters (UPPER case) and their percent;
    - ``sc_joint_composition__<primary_series>.png``, ``sc_probe_capture.png`` and ``sc_series_cluster_counts.png``.
The summary is rebuilt when an output is older than this script, the metadata, the .h5ad or an MMseqs2 table, or with ``--overwrite``.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import anndata as ad
import argparse
import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import resource
import time
from tqdm import tqdm
import yaml
CLUSTERS_DIR = "data/clusters"
COLORMAP = LinearSegmentedColormap.from_list("scambrian", ["#76BBE2", "#7b2c6d"])
DPI = 300
FONT_LABEL = 14
FONT_TICK = 12
FONT_TITLE = 18
H5AD_FILE = "projections.h5ad"
JOINT_SUBDIR = "mmseqs_joint"
LEGEND_CHAR_INCHES = 0.1
LEGEND_COL_INCHES = 0.6
MAX_BARS = 30
METADATA_FILE = "sc_metadata.tsv"
OVERWRITE = False
PALETTE = ["#7A5271", "#30767A", "#A66C58", "#8E81B9", "#899E64", "#DA919B", "#76BBE2", "#F1B584", "#93DEB6", "#ECDE90"]
PALETTE_ALT = PALETTE[::2] + PALETTE[1::2]
PRIMARY_SERIES = "u_j_leiden2d"
SERIAL_MIN_DIGITS = 3
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Name every cluster and compile the analysis summary")
parser.add_argument("--analysis_dir", type=str, required=True, help="Analysis folder analyses/<analysis_name>/")
parser.add_argument("--model", type=str, required=True, help="Model folder of the analysis holding projections.h5ad")
parser.add_argument("--primary_series", type=str, default=PRIMARY_SERIES, help="Joint series drawn in the composition plot, without the analysis prefix")
parser.add_argument("--max_bars", type=int, default=MAX_BARS, help="Largest joint clusters drawn in the composition plot")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Rebuild a fresh summary")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "compile_summary"
params = dict(primary_series=PRIMARY_SERIES, max_bars=MAX_BARS, overwrite=OVERWRITE)
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
###################################### NAMING
def letters(i: int) -> str:
    """Name a 0-based rank by lowercase letters, spreadsheet-style (0 -> a, 26 -> aa).

    Args:
        i (int): 0-based rank.

    Returns:
        str: Letter name.
    """
    return (letters(i // 26 - 1) if i >= 26 else "") + chr(97 + i % 26)
def size_ranks(raw: pd.Series) -> pd.Series:
    """Rank the clusters of one labelling by decreasing size, ties by first appearance.

    Args:
        raw (pd.Series): Raw cluster id per protein (NaN: unassigned).

    Returns:
        pd.Series: 0-based rank per raw cluster id.
    """
    sizes = raw.dropna().value_counts(sort=False)
    first = pd.Series(range(len(raw)), index=raw.index).groupby(raw).min()
    order = sorted(sizes.index, key=lambda c: (-sizes[c], first[c]))
    return pd.Series(range(len(order)), index=order)
def internal_serials(raw: pd.Series, labels: pd.Series) -> pd.Series:
    """Name internal clusters ``{label}_001`` by decreasing size within each sample.

    Args:
        raw (pd.Series): Raw cluster id per protein (NaN: unassigned or not applicable).
        labels (pd.Series): Sample label per protein.

    Returns:
        pd.Series: SERIAL per protein (NaN when unassigned).
    """
    out = pd.Series(np.nan, index=raw.index, dtype=object)
    for lbl in labels.unique():
        mask = (labels == lbl) & raw.notna()
        if not mask.any(): continue
        ranks = size_ranks(raw[mask])
        width = max(SERIAL_MIN_DIGITS, len(str(len(ranks))))
        out[mask] = raw[mask].map(lambda c: f"{lbl}_{ranks[c] + 1:0{width}d}")
    return out
def joint_serials(raw: pd.Series, types: pd.Series) -> pd.Series:
    """Name joint clusters by letters of decreasing size, upper case when they hold a probe and a target protein.

    Args:
        raw (pd.Series): Raw cluster id per protein (NaN: unassigned).
        types (pd.Series): ``target`` or ``probe`` per protein.

    Returns:
        pd.Series: SERIAL per protein (NaN when unassigned).
    """
    ranks = size_ranks(raw)
    present = pd.crosstab(raw, types) > 0
    mixed = present.get("probe", False) & present.get("target", False)
    names = {c: letters(r).upper() if bool(mixed.get(c, False)) else letters(r) for c, r in ranks.items()}
    return raw.map(names)
###################################### LOADERS
def mmseqs_series(summary_tsv: Path, scope: str) -> dict[str, pd.Series]:
    """Read the per-threshold representative of every sequence from an MMseqs2 cascade summary.

    Args:
        summary_tsv (Path): ``<label>_all_clusters_summary.tsv`` of run_mmseqs2.py.
        scope (str): ``i`` or ``j``.

    Returns:
        dict[str, pd.Series]: Series key (``s_<scope>_id90``) -> representative id per record.
    """
    table = pd.read_csv(summary_tsv, sep="\t", dtype=str, index_col="sequence_id").replace("", np.nan)
    return {f"s_{scope}_id{round(float(col.removeprefix('cluster_id_').replace('_', '.')) * 100)}": table[col] for col in table.columns}
def collect_series(kept: pd.DataFrame, analysis_dir: Path, h5ad_path: Path) -> tuple[dict[str, pd.Series], list[Path]]:
    """Gather the raw cluster ids of every series, aligned on the kept records.

    Args:
        kept (pd.DataFrame): Kept rows of sc_metadata.tsv indexed by record.
        analysis_dir (Path): Analysis folder.
        h5ad_path (Path): projections.h5ad of the model.

    Returns:
        tuple[dict[str, pd.Series], list[Path]]: Series key -> raw id per kept record, and the input files read.
    """
    raw, inputs = {}, []
    clusters_dir = Path(config_data.get("run_mmseqs2", {}).get("clusters_dir", CLUSTERS_DIR))
    targets = list(dict.fromkeys(kept.loc[kept["sc_type"] == "target", "sc_label"]))
    internal = [clusters_dir / lbl / f"{lbl}_all_clusters_summary.tsv" for lbl in targets]
    for path in internal:
        if not path.exists():
            print(f"[PYTHON-WARNING] Missing {path}; its internal MMseqs2 series stay empty.")
            continue
        inputs.append(path)
        for key, s in mmseqs_series(path, "i").items(): raw[key] = s.reindex(kept.index).combine_first(raw.get(key, pd.Series(np.nan, index=kept.index, dtype=object)))
    joint = analysis_dir / JOINT_SUBDIR / "joint_all_clusters_summary.tsv"
    if joint.exists():
        inputs.append(joint)
        raw |= {key: s.reindex(kept.index) for key, s in mmseqs_series(joint, "j").items()}
    else: print(f"[PYTHON-WARNING] Missing {joint}; no joint MMseqs2 series.")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    adata.file.close()
    inputs.append(h5ad_path)
    raw |= {key: obs[key].reindex(kept.index).where(lambda s: s >= 0) for key in obs.columns if key[:2] in ("u_", "t_")}
    return raw, inputs
###################################### OUTPUTS
def joint_table(summary: pd.DataFrame, columns: dict[str, str], samples: list[str], probes: list[str]) -> pd.DataFrame:
    """Describe every joint cluster: size, proteins and percent per sample, probe percent and probe flag.

    Args:
        summary (pd.DataFrame): Summary table.
        columns (dict[str, str]): Series key -> summary column of the joint series.
        samples (list[str]): Sample labels in query-table order.
        probes (list[str]): Probe labels.

    Returns:
        pd.DataFrame: One row per joint series and cluster.
    """
    rows = []
    for key, col in columns.items():
        counts = pd.crosstab(summary[col], summary["sc_label"]).reindex(columns=samples, fill_value=0)
        size = counts.sum(axis=1)
        table = pd.DataFrame({"series": col, "serial": counts.index, "cluster": col + "_" + counts.index.astype(str), "size": size})
        table = table.join(counts.add_prefix("n_")).join((counts.div(size, axis=0) * 100).round(2).add_prefix("pct_"))
        table["pct_probe"] = (counts[probes].sum(axis=1) / size * 100).round(2) if probes else 0.0
        table["probe_cluster"] = counts.index.str.isupper()
        rows.append(table.sort_values("size", ascending=False, kind="stable"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
def capture_table(summary: pd.DataFrame, columns: dict[str, str], targets: list[str]) -> pd.DataFrame:
    """Count per target and joint series the proteins inside probe (UPPER case) clusters.

    Args:
        summary (pd.DataFrame): Summary table.
        columns (dict[str, str]): Series key -> summary column of the joint series.
        targets (list[str]): Target labels.

    Returns:
        pd.DataFrame: target, series, n_target, n_in_probe_clusters and pct_in_probe_clusters.
    """
    rows = []
    for col in columns.values():
        upper = summary[col].fillna("").astype(str).str.isupper()
        for lbl in targets:
            is_t = summary["sc_label"] == lbl
            rows.append(dict(target=lbl, series=col, n_target=int(is_t.sum()), n_in_probe_clusters=int((is_t & upper).sum()), pct_in_probe_clusters=round(100 * (is_t & upper).sum() / max(is_t.sum(), 1), 2)))
    return pd.DataFrame(rows)
def bottom_legend(fig: plt.Figure, handles: list) -> None:
    """Place a legend below the axes with as many columns as fit the figure width.

    Args:
        fig (plt.Figure): Constrained-layout figure.
        handles (list): Legend handles.
    """
    ncol = max(1, min(len(handles), int(fig.get_figwidth() / (LEGEND_COL_INCHES + LEGEND_CHAR_INCHES * max(len(h.get_label()) for h in handles)))))
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False, fontsize=FONT_TICK)
def plot_composition(clusters: pd.DataFrame, column: str, samples: list[str], probes: list[str], path: Path) -> None:
    """Draw the sample composition (percent) of the largest joint clusters of one series, probes hatched.

    Args:
        clusters (pd.DataFrame): Output of joint_table.
        column (str): Summary column of the series.
        samples (list[str]): Sample labels in query-table order.
        probes (list[str]): Probe labels.
        path (Path): Output PNG.
    """
    table = clusters[clusters["series"] == column].head(args.max_bars)
    if table.empty: return
    colors = {s: PALETTE_ALT[i % len(PALETTE_ALT)] for i, s in enumerate(samples)}
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(table) + 3), 7), dpi=DPI, layout="constrained")
    bottom = np.zeros(len(table))
    for s in samples:
        ax.bar(table["serial"], table[f"pct_{s}"], bottom=bottom, color=colors[s], hatch="//" if s in probes else None, edgecolor="white", linewidth=0.3, label=s)
        bottom += table[f"pct_{s}"].to_numpy()
    for x, n in enumerate(table["size"]): ax.text(x, 101, f"{n:,}", rotation=90, ha="center", va="bottom", fontsize=FONT_TICK - 3)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Proteins in the cluster (%)", fontsize=FONT_LABEL)
    ax.set_xlabel("Joint cluster (UPPER case: probe and target proteins together)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{args.primary_series}: the {len(table)} largest joint clusters", fontsize=FONT_TITLE, fontweight="bold")
    bottom_legend(fig, [Patch(facecolor=colors[s], hatch="//" if s in probes else None, label=f"{s} (probe)" if s in probes else s) for s in samples])
    fig.savefig(path, facecolor="white")
    plt.close(fig)
def plot_capture(capture: pd.DataFrame, analysis: str, path: Path) -> None:
    """Draw the heatmap of target proteins inside probe clusters, per target and joint series.

    Args:
        capture (pd.DataFrame): Output of capture_table.
        analysis (str): Analysis name, stripped from the series labels.
        path (Path): Output PNG.
    """
    if capture.empty: return
    grid = capture.pivot(index="target", columns="series", values="pct_in_probe_clusters")
    grid = grid.loc[list(dict.fromkeys(capture["target"])), list(dict.fromkeys(capture["series"]))]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * grid.shape[1] + 3), max(4, 0.6 * grid.shape[0] + 3)), dpi=DPI, layout="constrained")
    im = ax.imshow(grid.to_numpy(), cmap=COLORMAP, aspect="auto", vmin=0)
    for (i, j), v in np.ndenumerate(grid.to_numpy()): ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=FONT_TICK - 2, color="white" if v > np.nanmax(grid.to_numpy()) / 2 else "black")
    ax.set_xticks(range(grid.shape[1]), [c.removeprefix(f"{analysis}_") for c in grid.columns], rotation=60, ha="right", fontsize=FONT_TICK)
    ax.set_yticks(range(grid.shape[0]), grid.index, fontsize=FONT_TICK)
    ax.set_title("Target proteins in probe clusters (%)", fontsize=FONT_TITLE, fontweight="bold")
    fig.colorbar(im, ax=ax, location="bottom", shrink=0.5, label="% of the target's proteins")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
def plot_counts(summary: pd.DataFrame, columns: list[str], analysis: str, path: Path) -> None:
    """Draw the number of clusters of every series.

    Args:
        summary (pd.DataFrame): Summary table.
        columns (list[str]): Summary columns of every series.
        analysis (str): Analysis name, stripped from the series labels.
        path (Path): Output PNG.
    """
    names = [c.removeprefix(f"{analysis}_") for c in columns]
    kinds = {"s": "sequence (MMseqs2)", "u": "UMAP", "t": "t-SNE"}
    colors = {k: PALETTE_ALT[i] for i, k in enumerate(kinds)}
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(columns) + 2)), dpi=DPI, layout="constrained")
    ax.barh(names, [summary[c].nunique() for c in columns], color=[colors[n[0]] for n in names])
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Clusters (log scale)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{analysis}: clusters per series", fontsize=FONT_TITLE, fontweight="bold")
    bottom_legend(fig, [Patch(facecolor=colors[k], label=v) for k, v in kinds.items()])
    fig.savefig(path, facecolor="white")
    plt.close(fig)
###################################### MAIN
def main() -> None:
    """Name the clusters of every series and write the summary tables and figures."""
    start_time = time.time()
    analysis_dir, out_dir = Path(args.analysis_dir), Path(args.analysis_dir) / args.model
    analysis = analysis_dir.name
    metadata_path, h5ad_path = analysis_dir / METADATA_FILE, out_dir / H5AD_FILE
    for p in (metadata_path, h5ad_path):
        if not p.exists(): raise FileNotFoundError(f"[PYTHON-ERROR] Missing {p}; run steps 02 and 05 first.")
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    kept = metadata[metadata["status"] == "kept"].set_index("record", drop=False)
    samples = list(dict.fromkeys(metadata["sc_label"]))
    probes = [s for s in samples if (metadata.loc[metadata["sc_label"] == s, "sc_type"] == "probe").all()]
    targets = [s for s in samples if s not in probes]
    raw, inputs = collect_series(kept, analysis_dir, h5ad_path)
    outputs = {k: out_dir / f for k, f in (("summary", "sc_summary.tsv"), ("joint", "sc_joint_clusters.tsv"), ("capture", "sc_probe_capture.tsv"), ("counts", "sc_series_cluster_counts.png"))}
    if not args.overwrite and all(p.exists() for p in outputs.values()) and min(p.stat().st_mtime for p in outputs.values()) > max(p.stat().st_mtime for p in [Path(__file__), metadata_path, *inputs]):
        print(f"[PYTHON-INFO] Summary of {out_dir} is newer than its inputs. Skipping.")
        return
    order = sorted(raw, key=lambda k: ("sut".index(k[0]), k[2], k))
    summary, joint_columns = kept.reset_index(drop=True).copy(), {}
    for key in tqdm(order, desc="Naming series"):
        values = raw[key].reset_index(drop=True)
        column = f"{analysis}_{key}"
        if key[2] == "j":
            summary[column] = joint_serials(values, summary["sc_type"])
            joint_columns[key] = column
        else: summary[column] = internal_serials(values.where(summary["sc_type"] == "target"), summary["sc_label"])
    upper = {c: int(summary[c].dropna().drop_duplicates().astype(str).str.isupper().sum()) for c in joint_columns.values()}
    print(f"[PYTHON-INFO][SERIES] {len(order)} series; probe clusters per joint series: {', '.join(f'{c}={n}' for c, n in upper.items())}")
    summary.to_csv(outputs["summary"], sep="\t", index=False)
    clusters = joint_table(summary, joint_columns, samples, probes)
    clusters.to_csv(outputs["joint"], sep="\t", index=False)
    capture = capture_table(summary, joint_columns, targets)
    capture.to_csv(outputs["capture"], sep="\t", index=False)
    for k in ("summary", "joint", "capture"): print(f"[PYTHON-INFO][SAVE] {outputs[k]}")
    plot_composition(clusters, f"{analysis}_{args.primary_series}", samples, probes, out_dir / f"sc_joint_composition__{args.primary_series}.png")
    plot_capture(capture, analysis, out_dir / "sc_probe_capture.png")
    plot_counts(summary, [f"{analysis}_{k}" for k in order], analysis, outputs["counts"])
    print(f"[PYTHON-INFO][SAVE] Figures in {out_dir}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
