#!/usr/bin/env python3
"""Deduplicate FASTA inputs with MMseqs2 and cluster the representatives at successive identity thresholds.

This is the pipeline entry point: every ``*.fasta``/``*.fa``/``*.faa`` in ``seq_in_dir``
is processed, whether it was fetched from UniProt or provided by the user. Intermediate
MMseqs2 databases are kept in a temporary directory and only clean results are copied.
"""
############### <imports>
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from utils import add_yaml_args, fasta_label, list_fastas, log, log_resources, resolve_params

############### <constants>
SCRIPT_NAME = Path(__file__).stem
DEFAULTS = {
    "mmseqs_path": "mmseqs",
    "seq_in_dir": "data/sequences",
    "dedup_out_dir": "data/dedup",
    "clusters_out_dir": "data/clusters",
    "dedup_min_seq_id": 0.9,
    "dedup_cov": 0.8,
    "dedup_cov_mode": 0,
    "clust_cov": 0.8,
    "clust_cov_mode": 0,
    "thresholds": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
    "threads": None,
}


############### <helpers>
def run_cmd(cmd: list[str]) -> None:
    """Log and run an external command, failing on a non-zero exit code.

    Args:
        cmd: Command and arguments.
    """
    log(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def get_threads(requested: int | None) -> int:
    """Pick the number of MMseqs2 threads.

    Args:
        requested: Explicit thread count, or ``None`` for automatic detection.

    Returns:
        ``requested`` if set, else ``SLURM_CPUS_PER_TASK``, else ``os.cpu_count()``.
    """
    if requested:
        return int(requested)
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def param_tag(params: dict) -> str:
    """Build the parameter tag embedded in clustering output names.

    Args:
        params: Resolved parameters.

    Returns:
        Tag such as ``dmin0.9_dcov0.8_dmode0_ccov0.8_cmode0``.
    """
    return f"dmin{params['dedup_min_seq_id']}_dcov{params['dedup_cov']}_dmode{params['dedup_cov_mode']}_ccov{params['clust_cov']}_cmode{params['clust_cov_mode']}"


def outputs_complete(final_dir: Path, label: str, params: dict) -> bool:
    """Check whether all clustering outputs for a label already exist.

    Args:
        final_dir: Clean output directory of the label.
        label: Dataset label.
        params: Resolved parameters.

    Returns:
        True if every threshold output and the summary file are present.
    """
    tag = param_tag(params)
    for seq_id in params["thresholds"]:
        t_clean = str(seq_id).replace(".", "_")
        if not (final_dir / f"{label}_ids_id{t_clean}_{tag}.txt").exists() or not (final_dir / f"{label}_representatives_id{t_clean}_{tag}.fasta").exists():
            return False
    return (final_dir / "clustering_summary.txt").exists()


############### <dedup>
def dedup(in_fasta: Path, label: str, params: dict, threads: int) -> Path:
    """Remove redundant sequences with ``mmseqs easy-linclust``.

    Args:
        in_fasta: Input FASTA.
        label: Dataset label.
        params: Resolved parameters.
        threads: MMseqs2 thread count.

    Returns:
        Path to the deduplicated representative FASTA.
    """
    mmseqs, dedup_dir = params["mmseqs_path"], Path(params["dedup_out_dir"])
    out_dedup = dedup_dir / f"{label}_dedup.fasta"
    if out_dedup.exists():
        log(f"Dedup output exists, skipping: {out_dedup}")
        return out_dedup
    tmp_dir = dedup_dir / f"tmp_{label}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([mmseqs, "easy-linclust", str(in_fasta), str(dedup_dir / f"{label}_tmp"), str(tmp_dir), "--min-seq-id", str(params["dedup_min_seq_id"]), "-c", str(params["dedup_cov"]), "--cov-mode", str(params["dedup_cov_mode"]), "--threads", str(threads)])
    rep_file = dedup_dir / f"{label}_tmp_rep_seq.fasta"
    if rep_file.exists():
        rep_file.rename(out_dedup)
    for p in dedup_dir.glob(f"{label}_tmp*"):
        if p.is_file():
            p.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log(f"Deduplicated {in_fasta} -> {out_dedup}")
    return out_dedup


############### <clustering>
def read_pairs(path: Path) -> list[tuple[str, str]]:
    """Read ``representative<TAB>member`` pairs from an MMseqs2 cluster TSV.

    Args:
        path: Cluster TSV.

    Returns:
        List of ``(representative, member)`` tuples.
    """
    pairs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def cluster_thresholds(out_dedup: Path, label: str, params: dict, threads: int, tmp_run_dir: Path) -> tuple[dict, Path, int]:
    """Cluster hierarchically, feeding each threshold the previous representatives.

    Member-to-cluster maps are always expressed against the deduplicated sequences.

    Args:
        out_dedup: Deduplicated FASTA.
        label: Dataset label.
        params: Resolved parameters.
        threads: MMseqs2 thread count.
        tmp_run_dir: Scratch directory for MMseqs2 databases and outputs.

    Returns:
        Tuple of ``{threshold: {member: representative}}``, the metrics file and the number of input sequences.
    """
    mmseqs, tag = params["mmseqs_path"], param_tag(params)
    clust_tmp_dir = tmp_run_dir / "tmp"
    clust_tmp_dir.mkdir(parents=True, exist_ok=True)
    dedup_db = tmp_run_dir / "dedupDB"
    run_cmd([mmseqs, "createdb", str(out_dedup), str(dedup_db)])
    with open(out_dedup, encoding="utf-8") as handle:
        total_seqs = sum(1 for line in handle if line.startswith(">"))
    metrics_file = tmp_run_dir / f"{label}_cluster_metrics.tsv"
    metrics_file.write_text("threshold\tclusters\tsingleton_clusters\tmax_cluster_size\tavg_cluster_size\tcoverage\n")
    current_db, prev_txt, tsv_dict = str(dedup_db), None, {}
    for i, seq_id in enumerate(params["thresholds"]):
        t_clean = str(seq_id).replace(".", "_")
        cluster_db = str(tmp_run_dir / f"clusterDB_id{t_clean}")
        rep_db = f"{cluster_db}_rep"
        txt_out = tmp_run_dir / f"{label}_ids_id{t_clean}_{tag}.txt"
        fasta_out = tmp_run_dir / f"{label}_representatives_id{t_clean}_{tag}.fasta"
        run_cmd([mmseqs, "cluster", current_db, cluster_db, str(clust_tmp_dir), "--min-seq-id", str(seq_id), "-c", str(params["clust_cov"]), "--cov-mode", str(params["clust_cov_mode"]), "--threads", str(threads)])
        if i == 0:
            run_cmd([mmseqs, "createtsv", str(dedup_db), str(dedup_db), cluster_db, str(txt_out)])
        else:
            intermediate_txt = tmp_run_dir / f"{label}_intermediate_id{t_clean}.txt"
            run_cmd([mmseqs, "createtsv", current_db, current_db, cluster_db, str(intermediate_txt)])
            new_rep = {member: rep for rep, member in read_pairs(intermediate_txt)}
            with open(txt_out, "w", encoding="utf-8") as f_out:
                for prev_rep, member in read_pairs(prev_txt):
                    if prev_rep in new_rep:
                        f_out.write(f"{new_rep[prev_rep]}\t{member}\n")
            intermediate_txt.unlink(missing_ok=True)
        run_cmd([mmseqs, "result2repseq", current_db, cluster_db, rep_db])
        run_cmd([mmseqs, "result2flat", rep_db, rep_db, rep_db, str(fasta_out), "--use-fasta-header", "1"])
        seq_map, cluster_sizes = {}, {}
        for rep, member in read_pairs(txt_out):
            seq_map[member] = rep
            cluster_sizes[rep] = cluster_sizes.get(rep, 0) + 1
        tsv_dict[seq_id] = seq_map
        n_clust = len(cluster_sizes)
        singletons = sum(1 for size in cluster_sizes.values() if size == 1)
        max_size = max(cluster_sizes.values(), default=0)
        avg_size = total_seqs / n_clust if n_clust else 0
        with open(metrics_file, "a", encoding="utf-8") as mf:
            mf.write(f"{seq_id}\t{n_clust}\t{singletons}\t{max_size}\t{avg_size:.2f}\t{params['clust_cov']}\n")
        log(f"{label} id={seq_id}: {n_clust} clusters, {singletons} singletons, max size {max_size}")
        current_db, prev_txt = rep_db, txt_out
    return tsv_dict, metrics_file, total_seqs


############### <summary>
def write_summary(tsv_dict: dict, metrics_file: Path, total_seqs: int, label: str, thresholds: list, tmp_run_dir: Path) -> None:
    """Write the per-sequence cluster table and the human-readable summary.

    Args:
        tsv_dict: ``{threshold: {member: representative}}`` maps.
        metrics_file: Per-threshold metrics TSV.
        total_seqs: Number of deduplicated sequences.
        label: Dataset label.
        thresholds: Identity thresholds in clustering order.
        tmp_run_dir: Directory receiving the summary files.
    """
    all_seqs = sorted({seq for m in tsv_dict.values() for seq in m})
    header = ["sequence_id"] + [f"cluster_id_{str(s).replace('.', '_')}" for s in thresholds]
    lines = ["\t".join(header)] + ["\t".join([seq] + [tsv_dict.get(s, {}).get(seq, "") for s in thresholds]) for seq in all_seqs]
    (tmp_run_dir / f"{label}_all_clusters_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = (
        f"=== Clustering Summary for {label} ===\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"--- Representative Sequences ---\nRepresentative sequences retained: {total_seqs}\n\n--- Clustering Metrics ---\n"
        + metrics_file.read_text(encoding="utf-8")
    )
    (tmp_run_dir / "clustering_summary.txt").write_text(summary, encoding="utf-8")


def process_fasta(in_fasta: Path, params: dict, threads: int) -> None:
    """Deduplicate and cluster one FASTA file, copying clean outputs to the cluster directory.

    Args:
        in_fasta: Input FASTA.
        params: Resolved parameters.
        threads: MMseqs2 thread count.
    """
    label = fasta_label(in_fasta)
    out_base_dir = Path(params["clusters_out_dir"])
    final_dir, tmp_run_dir = out_base_dir / label, out_base_dir / f"{label}_tmp"
    if outputs_complete(final_dir, label, params):
        log(f"All clustering outputs exist for {label}, skipping")
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    out_dedup = dedup(in_fasta, label, params, threads)
    tsv_dict, metrics_file, total_seqs = cluster_thresholds(out_dedup, label, params, threads, tmp_run_dir)
    write_summary(tsv_dict, metrics_file, total_seqs, label, params["thresholds"], tmp_run_dir)
    for p in tmp_run_dir.glob("*"):
        if p.is_file() and not p.name.startswith(("clusterDB", "dedupDB")) and not p.name.endswith((".index", ".dbtype", ".lookup", ".source")):
            shutil.copy(p, final_dir / p.name)
    shutil.rmtree(tmp_run_dir, ignore_errors=True)
    log(f"Clustering outputs for {label} written to {final_dir}")


############### <main>
def main() -> None:
    """Parse arguments, resolve parameters and process the selected FASTA files."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fasta", type=Path, help="Process this FASTA only")
    parser.add_argument("--index", type=int, help="1-based index into the sorted FASTA files of seq_in_dir")
    parser.add_argument("--mmseqs_path")
    parser.add_argument("--seq_in_dir")
    parser.add_argument("--dedup_out_dir")
    parser.add_argument("--clusters_out_dir")
    parser.add_argument("--dedup_min_seq_id", type=float)
    parser.add_argument("--dedup_cov", type=float)
    parser.add_argument("--dedup_cov_mode", type=int)
    parser.add_argument("--clust_cov", type=float)
    parser.add_argument("--clust_cov_mode", type=int)
    parser.add_argument("--thresholds", type=float, nargs="+")
    parser.add_argument("--threads", type=int)
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    if args.yaml_out and args.fasta is None and args.index is None:
        sys.exit(0)
    mmseqs = params["mmseqs_path"]
    if not shutil.which(mmseqs) and not Path(mmseqs).exists():
        log(f"Error: MMseqs2 executable '{mmseqs}' not found.")
        raise FileNotFoundError(f"MMseqs2 executable '{mmseqs}' not found.")
    threads = get_threads(params["threads"])
    Path(params["dedup_out_dir"]).mkdir(parents=True, exist_ok=True)
    Path(params["clusters_out_dir"]).mkdir(parents=True, exist_ok=True)
    if args.fasta:
        fasta_files = [args.fasta]
    else:
        fasta_files = list_fastas(Path(params["seq_in_dir"]))
        if args.index is not None:
            if not 0 < args.index <= len(fasta_files):
                log(f"Index {args.index} exceeds the {len(fasta_files)} FASTA files in {params['seq_in_dir']}, nothing to do")
                fasta_files = []
            else:
                fasta_files = [fasta_files[args.index - 1]]
    log(f"Processing {len(fasta_files)} FASTA file(s) with {threads} threads")
    for in_fasta in tqdm(fasta_files, desc="[PYTHON-INFO] MMseqs2"):
        process_fasta(in_fasta, params, threads)
    log_resources(start_time)


if __name__ == "__main__":
    main()
