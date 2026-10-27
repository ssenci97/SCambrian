#!/usr/bin/env python3
"""Deduplicates each label's sequences with MMseqs2 and clusters the deduplicated set at successively lower identity thresholds.

For every ``<sequences_dir>/<label>_seqs.fasta``, easy-linclust at ``dedup_min_identity`` writes ``<dedup_dir>/<label>_dedup.fasta`` and a ``<label>_dedup.params`` sidecar holding its parameter tag. Each threshold of ``cluster_identity_thresholds`` then clusters the representatives of the previous threshold (cascade), and the member -> representative assignments are propagated back to every deduplicated sequence. MMseqs2 databases stay in a temporary run directory; only the assignment tables, representative FASTAs, metrics and summaries are copied to ``<clusters_dir>/<label>/``. The dedup is redone when its FASTA or sidecar is missing, the sidecar tag differs from the current parameters or the input FASTA is newer; the clustering is redone when any of its outputs is missing or older than the dedup FASTA; ``--overwrite`` forces both.
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
DEDUP_COVERAGE_MODE = 0
DEDUP_DIR = "data/dedup"
DEDUP_MIN_COVERAGE = 0.8
DEDUP_MIN_IDENTITY = 0.95
MMSEQS_EXECUTABLE = "mmseqs"
OVERWRITE = False
SEQUENCES_DIR = "data/sequences"
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="MMseqs2 deduplication and cascaded clustering")
parser.add_argument("--index", type=int, default=None, help="1-based position of the label among the sorted *_seqs.fasta files (SLURM array task); all labels when omitted")
parser.add_argument("--sequences_dir", type=str, default=SEQUENCES_DIR, help="Folder of the <label>_seqs.fasta inputs")
parser.add_argument("--dedup_dir", type=str, default=DEDUP_DIR, help="Folder of the deduplicated <label>_dedup.fasta outputs")
parser.add_argument("--clusters_dir", type=str, default=CLUSTERS_DIR, help="Folder of the per-label cluster outputs")
parser.add_argument("--mmseqs_executable", type=str, default=MMSEQS_EXECUTABLE, help="MMseqs2 executable name or path")
parser.add_argument("--dedup_min_identity", type=float, default=DEDUP_MIN_IDENTITY, help="Minimum sequence identity (0-1) of two sequences merged by the dedup")
parser.add_argument("--dedup_min_coverage", type=float, default=DEDUP_MIN_COVERAGE, help="Minimum alignment coverage (0-1) of the dedup")
parser.add_argument("--dedup_coverage_mode", type=int, default=DEDUP_COVERAGE_MODE, help="MMseqs2 --cov-mode of the dedup (0: query and target)")
parser.add_argument("--cluster_min_coverage", type=float, default=CLUSTER_MIN_COVERAGE, help="Minimum alignment coverage (0-1) of every clustering threshold")
parser.add_argument("--cluster_coverage_mode", type=int, default=CLUSTER_COVERAGE_MODE, help="MMseqs2 --cov-mode of the clustering")
parser.add_argument("--cluster_identity_thresholds", type=float, nargs="+", default=CLUSTER_IDENTITY_THRESHOLDS, help="Decreasing identity thresholds (0-1) of the cascaded clustering")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Redo the dedup and clustering even when their outputs are fresh")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "run_mmseqs2"
params = dict(sequences_dir=SEQUENCES_DIR, dedup_dir=DEDUP_DIR, clusters_dir=CLUSTERS_DIR, mmseqs_executable=MMSEQS_EXECUTABLE, dedup_min_identity=DEDUP_MIN_IDENTITY, dedup_min_coverage=DEDUP_MIN_COVERAGE, dedup_coverage_mode=DEDUP_COVERAGE_MODE, cluster_min_coverage=CLUSTER_MIN_COVERAGE, cluster_coverage_mode=CLUSTER_COVERAGE_MODE, cluster_identity_thresholds=CLUSTER_IDENTITY_THRESHOLDS, overwrite=OVERWRITE)
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
def mmseqs(*command: str | Path | float | int) -> None:
    """Run one MMseqs2 subcommand and fail on a non-zero exit.

    Args:
        *command (str | Path | float | int): Subcommand and its arguments.
    """
    subprocess.run([args.mmseqs_executable, *map(str, command)], check=True)
def dedup_tag() -> str:
    """Return the parameter tag of the dedup.

    Returns:
        str: Tag such as ``dmin0.95_dcov0.8_dmode0``.
    """
    return f"dmin{args.dedup_min_identity}_dcov{args.dedup_min_coverage}_dmode{args.dedup_coverage_mode}"
def cluster_tag() -> str:
    """Return the parameter tag of the clustering outputs, which includes the dedup tag.

    Returns:
        str: Tag such as ``dmin0.95_dcov0.8_dmode0_ccov0.8_cmode0``.
    """
    return f"{dedup_tag()}_ccov{args.cluster_min_coverage}_cmode{args.cluster_coverage_mode}"
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
def dedup(label: str, in_fasta: Path) -> Path:
    """Deduplicate the sequences of a label with easy-linclust, unless its fresh output already exists.

    Args:
        label (str): Label of the input FASTA.
        in_fasta (Path): ``<label>_seqs.fasta`` input.

    Returns:
        Path: ``<dedup_dir>/<label>_dedup.fasta``.
    """
    dedup_dir = Path(args.dedup_dir)
    out_fasta, sidecar = dedup_dir / f"{label}_dedup.fasta", dedup_dir / f"{label}_dedup.params"
    if not args.overwrite and not is_stale([out_fasta, sidecar], [in_fasta]) and sidecar.read_text().strip() == dedup_tag():
        print(f"[PYTHON-INFO] Dedup of '{label}' is fresh ({dedup_tag()}). Skipping.")
        return out_fasta
    prefix, work_dir = dedup_dir / f"{label}_tmp", dedup_dir / f"tmp_{label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    mmseqs("easy-linclust", in_fasta, prefix, work_dir, "--min-seq-id", args.dedup_min_identity, "-c", args.dedup_min_coverage, "--cov-mode", args.dedup_coverage_mode, "--threads", THREADS)
    Path(f"{prefix}_rep_seq.fasta").rename(out_fasta)
    for p in dedup_dir.glob(f"{label}_tmp*"):
        if p.is_file(): p.unlink()
    shutil.rmtree(work_dir, ignore_errors=True)
    sidecar.write_text(dedup_tag() + "\n")
    print(f"[PYTHON-INFO][SAVE] {out_fasta} ({dedup_tag()})")
    return out_fasta
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
    summary = f"=== Clustering Summary for {label} ===\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nDedup: {dedup_tag()}\n\n--- Representative Sequences ---\nRepresentative sequences retained: {n_sequences}\n\n--- Clustering Metrics ---\n"
    (run_dir / "clustering_summary.txt").write_text(summary + metrics.to_csv(sep="\t", index=False), encoding="utf-8")
###################################### MAIN
def main() -> None:
    """Deduplicate and cluster every label, or only the ``--index`` one."""
    start_time = time.time()
    if not shutil.which(args.mmseqs_executable) and not Path(args.mmseqs_executable).exists():
        raise FileNotFoundError(f"[PYTHON-ERROR] MMseqs2 executable '{args.mmseqs_executable}' not found.")
    for folder in (args.dedup_dir, args.clusters_dir): Path(folder).mkdir(parents=True, exist_ok=True)
    in_fastas = sorted(Path(args.sequences_dir).glob("*_seqs.fasta"))
    if args.index is not None: in_fastas = in_fastas[args.index - 1:args.index] if 1 <= args.index <= len(in_fastas) else []
    for in_fasta in tqdm(in_fastas, desc="Labels"):
        label = in_fasta.name.removesuffix("_seqs.fasta")
        dedup_fasta = dedup(label, in_fasta)
        final_dir, run_dir = Path(args.clusters_dir) / label, Path(args.clusters_dir) / f"{label}_tmp"
        if not args.overwrite and not is_stale(cluster_outputs(label, final_dir), [dedup_fasta]):
            print(f"[PYTHON-INFO] Clusters of '{label}' are fresh. Skipping.")
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        cascade_clusters(label, dedup_fasta, run_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        for kept in cluster_outputs(label, run_dir): shutil.copy(kept, final_dir / kept.name)
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"[PYTHON-INFO][SAVE] {final_dir} ({len(args.cluster_identity_thresholds)} thresholds, {cluster_tag()})")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
