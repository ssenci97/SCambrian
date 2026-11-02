#!/usr/bin/env python3
"""Clusters the deduplicated sequences with MMseqs2 at successively lower identity thresholds, per target sample (internal) and for all samples together (joint).

Inputs are the ``<preprocessed_dir>/<label>_dedup.fasta`` files of preprocess_sequences.py. Internal mode (``--index``, 1-based row among the target samples of the query table, or every target when omitted) clusters one target; probes never get internal clusters. Joint mode (``--joint``) concatenates the dedup FASTA of every sample into ``<analysis_dir>/mmseqs_joint/joint_dedup.fasta`` and clusters it. Each threshold of ``cluster_identity_thresholds`` clusters the representatives of the previous threshold (cascade), and the member -> representative assignments are propagated back to every deduplicated sequence. MMseqs2 databases stay in a temporary run directory; only the assignment tables, representative FASTAs, metrics and summaries are copied to ``<clusters_dir>/<label>/`` (internal) or ``<analysis_dir>/mmseqs_joint/`` (joint). The clustering is redone when any of its outputs is missing or older than its input FASTA, or with ``--overwrite``. Cluster names are given later by compile_summary.py.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
from collections import Counter
import datetime
import os
import pandas as pd
from pathlib import Path
import resource
import shutil
import subprocess
import time
from tqdm import tqdm
import yaml
CLUSTER_COVERAGE_MODE = 0
CLUSTER_IDENTITY_THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
CLUSTER_MIN_COVERAGE = 0.8
CLUSTERS_DIR = "data/clusters"
JOINT_LABEL = "joint"
JOINT_SUBDIR = "mmseqs_joint"
MMSEQS_EXECUTABLE = "mmseqs"
OVERWRITE = False
PREPROCESSED_DIR = "data/preprocessed"
QUERY_TABLE = "configs/query_configs.tsv"
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="MMseqs2 cascaded clustering, internal and joint")
parser.add_argument("--index", type=int, default=None, help="1-based row among the target samples (SLURM array task); every target when omitted")
parser.add_argument("--joint", action="store_true", help="Cluster all samples together instead of one target")
parser.add_argument("--analysis_dir", type=str, default=None, help="Analysis folder analyses/<analysis_name>/, required with --joint")
parser.add_argument("--preprocessed_dir", type=str, default=PREPROCESSED_DIR, help="Folder of the <label>_dedup.fasta inputs (preprocess_sequences.py)")
parser.add_argument("--clusters_dir", type=str, default=CLUSTERS_DIR, help="Folder of the per-label cluster outputs")
parser.add_argument("--mmseqs_executable", type=str, default=MMSEQS_EXECUTABLE, help="MMseqs2 executable name or path")
parser.add_argument("--cluster_min_coverage", type=float, default=CLUSTER_MIN_COVERAGE, help="Minimum alignment coverage (0-1) of every clustering threshold")
parser.add_argument("--cluster_coverage_mode", type=int, default=CLUSTER_COVERAGE_MODE, help="MMseqs2 --cov-mode of the clustering")
parser.add_argument("--cluster_identity_thresholds", type=float, nargs="+", default=CLUSTER_IDENTITY_THRESHOLDS, help="Decreasing identity thresholds (0-1) of the cascaded clustering")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Redo the clustering even when its outputs are fresh")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "run_mmseqs2"
params = dict(preprocessed_dir=PREPROCESSED_DIR, clusters_dir=CLUSTERS_DIR, mmseqs_executable=MMSEQS_EXECUTABLE, cluster_min_coverage=CLUSTER_MIN_COVERAGE, cluster_coverage_mode=CLUSTER_COVERAGE_MODE, cluster_identity_thresholds=CLUSTER_IDENTITY_THRESHOLDS, overwrite=OVERWRITE)
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
def query_samples() -> list[tuple[str, str]]:
    """List the samples of the query table named in the dw_uniprot_seqs section of the YAML configuration.

    Returns:
        list[tuple[str, str]]: (label, input type) in row order.
    """
    config_data = (yaml.safe_load(Path(args.yaml_in).read_text()) or {}) if args.yaml_in and Path(args.yaml_in).exists() else {}
    df = pd.read_csv(config_data.get("dw_uniprot_seqs", {}).get("query_table", QUERY_TABLE), sep="\t", dtype=str)
    types = df["input_type"] if "input_type" in df else pd.Series("target", index=df.index)
    return [(str(lbl).strip(), str(t).strip() if pd.notna(t) and str(t).strip() else "target") for lbl, t in zip(df["label"], types) if pd.notna(lbl) and str(lbl).strip()]
def mmseqs(*command: str | Path | float | int) -> None:
    """Run one MMseqs2 subcommand and fail on a non-zero exit.

    Args:
        *command (str | Path | float | int): Subcommand and its arguments.
    """
    subprocess.run([args.mmseqs_executable, *map(str, command)], check=True)
def cluster_tag() -> str:
    """Return the parameter tag of the clustering outputs.

    Returns:
        str: Tag such as ``ccov0.8_cmode0``.
    """
    return f"ccov{args.cluster_min_coverage}_cmode{args.cluster_coverage_mode}"
def threshold_tag(identity: float) -> str:
    """Write an identity threshold as a file-name fragment.

    Args:
        identity (float): Identity threshold.

    Returns:
        str: Threshold with ``_`` as decimal separator (0.9 -> ``0_9``).
    """
    return str(identity).replace(".", "_")
def is_stale(outputs: list[Path], inputs: list[Path]) -> bool:
    """Tell whether outputs must be rebuilt: any is missing or older than the newest input.

    Args:
        outputs (list[Path]): Files produced by the step.
        inputs (list[Path]): Files the step reads.

    Returns:
        bool: True when the step must run.
    """
    if not all(p.exists() for p in outputs): return True
    return min(p.stat().st_mtime for p in outputs) < max(p.stat().st_mtime for p in inputs)
def cluster_outputs(label: str, folder: Path) -> list[Path]:
    """List every clustering output of a label kept in a folder.

    Args:
        label (str): Label of the input FASTA.
        folder (Path): Folder holding the outputs.

    Returns:
        list[Path]: Assignment tables and representative FASTAs per threshold, metrics, assignment summary and text summary.
    """
    per_threshold = [folder / f"{label}_{kind}_id{threshold_tag(t)}_{cluster_tag()}.{ext}" for t in args.cluster_identity_thresholds for kind, ext in (("ids", "txt"), ("representatives", "fasta"))]
    return per_threshold + [folder / f"{label}_cluster_metrics.tsv", folder / f"{label}_all_clusters_summary.tsv", folder / "clustering_summary.txt"]
def read_assignments(path: Path) -> dict[str, str]:
    """Read an MMseqs2 ``createtsv`` table of representative and member columns.

    Args:
        path (Path): Two-column TSV, representative then member.

    Returns:
        dict[str, str]: Member -> representative, in file order.
    """
    with open(path, encoding="utf-8") as f:
        return {fields[1]: fields[0] for fields in (line.rstrip("\n").split("\t") for line in f) if len(fields) >= 2}
###################################### PIPELINE_STEPS
def cascade_clusters(label: str, dedup_fasta: Path, run_dir: Path) -> None:
    """Cluster the deduplicated sequences at every threshold, each on the representatives of the previous one, and write all outputs into the run directory.

    Args:
        label (str): Label of the input FASTA.
        dedup_fasta (Path): Deduplicated FASTA.
        run_dir (Path): Temporary directory receiving databases and outputs.
    """
    work_dir, dedup_db = run_dir / "tmp", run_dir / "dedupDB"
    work_dir.mkdir(parents=True, exist_ok=True)
    mmseqs("createdb", dedup_fasta, dedup_db)
    with open(dedup_fasta, encoding="utf-8") as f:
        n_sequences = sum(line.startswith(">") for line in f)
    input_db, previous_assignments, assignments_by_threshold, metric_rows = dedup_db, None, {}, []
    for identity in tqdm(args.cluster_identity_thresholds, desc=f"{label} thresholds", leave=False):
        tag = threshold_tag(identity)
        cluster_db, ids_out, reps_out = run_dir / f"clusterDB_id{tag}", run_dir / f"{label}_ids_id{tag}_{cluster_tag()}.txt", run_dir / f"{label}_representatives_id{tag}_{cluster_tag()}.fasta"
        representatives_db = Path(f"{cluster_db}_rep")
        mmseqs("cluster", input_db, cluster_db, work_dir, "--min-seq-id", identity, "-c", args.cluster_min_coverage, "--cov-mode", args.cluster_coverage_mode, "--threads", THREADS)
        if previous_assignments is None:
            mmseqs("createtsv", dedup_db, dedup_db, cluster_db, ids_out)
        else:
            step_tsv = run_dir / f"{label}_intermediate_id{tag}.txt"
            mmseqs("createtsv", input_db, input_db, cluster_db, step_tsv)
            new_representative = read_assignments(step_tsv)
            with open(ids_out, "w", encoding="utf-8") as f:
                f.writelines(f"{new_representative[rep]}\t{member}\n" for member, rep in previous_assignments.items() if rep in new_representative)
            step_tsv.unlink()
        mmseqs("result2repseq", input_db, cluster_db, representatives_db)
        mmseqs("result2flat", representatives_db, representatives_db, representatives_db, reps_out, "--use-fasta-header", "1")
        assignments = read_assignments(ids_out)
        sizes = Counter(assignments.values())
        metric_rows.append((identity, len(sizes), sum(n == 1 for n in sizes.values()), max(sizes.values(), default=0), round(n_sequences / len(sizes), 2) if sizes else 0, args.cluster_min_coverage))
        assignments_by_threshold[identity] = assignments
        input_db, previous_assignments = representatives_db, assignments
    metrics = pd.DataFrame(metric_rows, columns=["threshold", "clusters", "singleton_clusters", "max_cluster_size", "avg_cluster_size", "min_coverage"])
    metrics.to_csv(run_dir / f"{label}_cluster_metrics.tsv", sep="\t", index=False)
    pd.DataFrame({f"cluster_id_{threshold_tag(t)}": pd.Series(a) for t, a in assignments_by_threshold.items()}).rename_axis("sequence_id").sort_index().fillna("").to_csv(run_dir / f"{label}_all_clusters_summary.tsv", sep="\t")
    summary = f"=== Clustering Summary for {label} ===\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nClustering: {cluster_tag()}\n\n--- Representative Sequences ---\nRepresentative sequences retained: {n_sequences}\n\n--- Clustering Metrics ---\n"
    (run_dir / "clustering_summary.txt").write_text(summary + metrics.to_csv(sep="\t", index=False), encoding="utf-8")
def run_label(label: str, in_fasta: Path, final_dir: Path) -> None:
    """Cluster one input FASTA into its output folder, unless its outputs are fresh.

    Args:
        label (str): Sample label, or ``joint``.
        in_fasta (Path): Deduplicated input FASTA.
        final_dir (Path): Output folder.
    """
    run_dir = final_dir.parent / f"{final_dir.name}_tmp"
    if not args.overwrite and not is_stale(cluster_outputs(label, final_dir), [in_fasta]):
        print(f"[PYTHON-INFO] Clusters of '{label}' are fresh. Skipping.")
        return
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    cascade_clusters(label, in_fasta, run_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for kept in cluster_outputs(label, run_dir): shutil.copy(kept, final_dir / kept.name)
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f"[PYTHON-INFO][SAVE] {final_dir} ({len(args.cluster_identity_thresholds)} thresholds, {cluster_tag()})")
def joint_fasta(samples: list[tuple[str, str]], out_dir: Path) -> Path:
    """Concatenate the dedup FASTA of every sample, rewriting the joint FASTA only when its content changes.

    Args:
        samples (list[tuple[str, str]]): (label, input type) of every sample.
        out_dir (Path): Joint output folder.

    Returns:
        Path: ``joint_dedup.fasta``.
    """
    inputs = [Path(args.preprocessed_dir) / f"{label}_dedup.fasta" for label, _ in samples]
    missing = [p for p in inputs if not p.exists()]
    if missing: raise FileNotFoundError(f"[PYTHON-ERROR] Missing {missing}; run step 02 first.")
    out = out_dir / f"{JOINT_LABEL}_dedup.fasta"
    text = "".join(p.read_text(encoding="utf-8") for p in inputs)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[PYTHON-INFO][SAVE] {out} ({text.count('>'):,} sequences from {len(inputs)} samples)")
    return out
###################################### MAIN
def main() -> None:
    """Cluster every target, only the ``--index`` one, or all samples together with ``--joint``."""
    start_time = time.time()
    if not shutil.which(args.mmseqs_executable) and not Path(args.mmseqs_executable).exists():
        raise FileNotFoundError(f"[PYTHON-ERROR] MMseqs2 executable '{args.mmseqs_executable}' not found.")
    samples = query_samples()
    if args.joint:
        if not args.analysis_dir: raise ValueError("[PYTHON-ERROR] --joint requires --analysis_dir.")
        out_dir = Path(args.analysis_dir) / JOINT_SUBDIR
        run_label(JOINT_LABEL, joint_fasta(samples, out_dir), out_dir)
    else:
        labels = [label for label, sc_type in samples if sc_type == "target"]
        if args.index is not None: labels = labels[args.index - 1:args.index] if 1 <= args.index <= len(labels) else []
        Path(args.clusters_dir).mkdir(parents=True, exist_ok=True)
        for label in tqdm(labels, desc="Targets"):
            in_fasta = Path(args.preprocessed_dir) / f"{label}_dedup.fasta"
            if not in_fasta.exists():
                print(f"[PYTHON-WARNING] Missing {in_fasta}. Skipping '{label}'.")
                continue
            run_label(label, in_fasta, Path(args.clusters_dir) / label)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
