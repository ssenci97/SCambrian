#!/usr/bin/env python3
"""Run MMseqs2 clustering pipeline, keeping intermediate databases isolated in a temporary directory and copying only clean results."""

import argparse
import os
import random
import resource
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import yaml

SEED = 42
random.seed(SEED)
DEFAULT_PARAMS = {"overwrite": False, "mmseqs_path": "mmseqs", "seq_out_dir": "data/sequences", "dedup_out_dir": "data/dedup", "clusters_out_dir": "data/clusters", "mmseqs": {"dedup_min_seq_id": 0.9, "dedup_cov": 0.8, "dedup_cov_mode": 0, "clust_cov": 0.8, "clust_cov_mode": 0, "thresholds": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]}}

###################################### config
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-in", nargs="?", const=Path("configs/param_configs.yaml"), type=Path, default=Path("configs/param_configs.yaml"))
    parser.add_argument("--yaml-out", nargs="?", const=Path("configs/param_configs.yaml"), type=Path, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args, _ = parser.parse_known_args()
    script_name = Path(__file__).stem
    params = DEFAULT_PARAMS.copy()
    if args.yaml_in and args.yaml_in.exists():
        data = yaml.safe_load(args.yaml_in.read_text())
        if isinstance(data, dict) and script_name in data:
            params.update(data[script_name])
    params["overwrite"] = bool(params["overwrite"] or args.overwrite)
    if args.yaml_out:
        yaml_out_path = args.yaml_out if isinstance(args.yaml_out, Path) else Path("configs/param_configs.yaml")
        yaml_out_path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_load(yaml_out_path.read_text()) if yaml_out_path.exists() else {}
        if not isinstance(data, dict): data = {}
        data[script_name] = params
        yaml_out_path.write_text(yaml.dump(data, default_flow_style=False))
    return params, args.index

###################################### main
def main() -> None:
    start_time = time.time()
    params, idx = get_params()
    mmseqs = params["mmseqs_path"]
    if not shutil.which(mmseqs) and not Path(mmseqs).exists():
        print(f"[PYTHON-INFO] Error: MMseqs2 executable '{mmseqs}' not found.")
        raise FileNotFoundError(f"MMseqs2 executable '{mmseqs}' not found.")
    in_dir = Path(params["seq_out_dir"])
    dedup_dir = Path(params["dedup_out_dir"])
    out_base_dir = Path(params["clusters_out_dir"])
    min_seq_id = params["mmseqs"]["dedup_min_seq_id"]
    cov = params["mmseqs"]["dedup_cov"]
    cov_mode = params["mmseqs"]["dedup_cov_mode"]
    clust_cov = params["mmseqs"]["clust_cov"]
    clust_cov_mode = params["mmseqs"]["clust_cov_mode"]
    thresholds = params["mmseqs"]["thresholds"]
    threads = os.cpu_count() or 16
    Path("logs").mkdir(exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)
    out_base_dir.mkdir(parents=True, exist_ok=True)
    fasta_files = sorted(list(in_dir.glob("*_seqs.fasta")))
    if idx is not None:
        fasta_files = [fasta_files[idx - 1]] if 0 <= idx - 1 < len(fasta_files) else []
    for in_fasta in tqdm(fasta_files, desc="Pipeline execution"):
        base_name = in_fasta.name.replace("_seqs.fasta", "")
        final_dir = out_base_dir / base_name
        tmp_run_dir = out_base_dir / f"{base_name}_tmp"
        final_dir.mkdir(parents=True, exist_ok=True)
        tmp_run_dir.mkdir(parents=True, exist_ok=True)
        param_tag = f"dmin{min_seq_id}_dcov{cov}_dmode{cov_mode}_ccov{clust_cov}_cmode{clust_cov_mode}"
        all_exist = True
        for seq_id in thresholds:
            t_clean = str(seq_id).replace(".", "_")
            if not (final_dir / f"{base_name}_ids_id{t_clean}_{param_tag}.txt").exists() or not (final_dir / f"{base_name}_representatives_id{t_clean}_{param_tag}.fasta").exists():
                all_exist = False
                break
        if all_exist and (final_dir / "clustering_summary.txt").exists() and not params["overwrite"]:
            print(f"[PYTHON-INFO] Clusters of '{base_name}' already exist. Skipping.")
            shutil.rmtree(tmp_run_dir, ignore_errors=True)
            continue
        out_dedup = dedup_dir / f"{base_name}_dedup.fasta"
        tmp_dir = dedup_dir / f"tmp_{base_name}"
        if not out_dedup.exists() or params["overwrite"]:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([mmseqs, "easy-linclust", str(in_fasta), str(dedup_dir / f"{base_name}_tmp"), str(tmp_dir), "--min-seq-id", str(min_seq_id), "-c", str(cov), "--cov-mode", str(cov_mode), "--threads", str(threads)], check=True)
            rep_file = dedup_dir / f"{base_name}_tmp_rep_seq.fasta"
            if rep_file.exists(): rep_file.rename(out_dedup)
            for p in dedup_dir.glob(f"{base_name}_tmp*"):
                if p.is_file(): p.unlink()
            if tmp_dir.exists(): shutil.rmtree(tmp_dir, ignore_errors=True)
        clust_tmp_dir = tmp_run_dir / "tmp"
        clust_tmp_dir.mkdir(parents=True, exist_ok=True)
        dedup_db = tmp_run_dir / "dedupDB"
        subprocess.run([mmseqs, "createdb", str(out_dedup), str(dedup_db)], check=True)
        total_seqs = sum(1 for line in open(out_dedup, encoding="utf-8") if line.startswith(">"))
        metrics_file = tmp_run_dir / f"{base_name}_cluster_metrics.tsv"
        metrics_file.write_text("threshold\tclusters\tsingleton_clusters\tmax_cluster_size\tavg_cluster_size\tcoverage\n")
        current_db = str(dedup_db)
        prev_txt = ""
        tsv_dict = {}
        for i, seq_id in enumerate(thresholds):
            threshold_clean = str(seq_id).replace(".", "_")
            cluster_db = str(tmp_run_dir / f"clusterDB_id{threshold_clean}")
            txt_out = tmp_run_dir / f"{base_name}_ids_id{threshold_clean}_{param_tag}.txt"
            fasta_out = tmp_run_dir / f"{base_name}_representatives_id{threshold_clean}_{param_tag}.fasta"
            rep_db = f"{cluster_db}_rep"
            subprocess.run([mmseqs, "cluster", current_db, cluster_db, str(clust_tmp_dir), "--min-seq-id", str(seq_id), "-c", str(clust_cov), "--cov-mode", str(clust_cov_mode), "--threads", str(threads)], check=True)
            if i == 0:
                subprocess.run([mmseqs, "createtsv", str(dedup_db), str(dedup_db), cluster_db, str(txt_out)], check=True)
            else:
                intermediate_txt = tmp_run_dir / f"{base_name}_intermediate_id{threshold_clean}.txt"
                subprocess.run([mmseqs, "createtsv", current_db, current_db, cluster_db, str(intermediate_txt)], check=True)
                new_rep = {}
                with open(intermediate_txt, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2: new_rep[parts[1]] = parts[0]
                with open(prev_txt, encoding="utf-8") as f_prev, open(txt_out, "w", encoding="utf-8") as f_out:
                    for line in f_prev:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2 and parts[0] in new_rep:
                            f_out.write(f"{new_rep[parts[0]]}\t{parts[1]}\n")
                intermediate_txt.unlink(missing_ok=True)
            subprocess.run([mmseqs, "result2repseq", current_db, cluster_db, rep_db], check=True)
            subprocess.run([mmseqs, "result2flat", rep_db, rep_db, rep_db, str(fasta_out), "--use-fasta-header", "1"], check=True)
            cluster_sizes = {}
            num_clusters = set()
            seq_map = {}
            with open(txt_out, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        c_id, m_id = parts[0], parts[1]
                        num_clusters.add(c_id)
                        cluster_sizes[c_id] = cluster_sizes.get(c_id, 0) + 1
                        seq_map[m_id] = c_id
            tsv_dict[seq_id] = seq_map
            n_clust = len(num_clusters)
            singletons = sum(1 for sz in cluster_sizes.values() if sz == 1)
            max_size = max(cluster_sizes.values()) if cluster_sizes else 0
            avg_size = total_seqs / n_clust if n_clust > 0 else 0
            metrics_content = metrics_file.read_text(encoding="utf-8")
            if f"{seq_id}\t" not in metrics_content:
                with open(metrics_file, "a", encoding="utf-8") as mf:
                    mf.write(f"{seq_id}\t{n_clust}\t{singletons}\t{max_size}\t{avg_size:.2f}\t{clust_cov}\n")
            current_db = rep_db
            prev_txt = str(txt_out)
        summary_tsv = tmp_run_dir / f"{base_name}_all_clusters_summary.tsv"
        all_seqs = sorted(list(set(seq for m in tsv_dict.values() for seq in m.keys())))
        header = ["sequence_id"] + [f"cluster_id_{str(s).replace('.', '_')}" for s in thresholds]
        summary_lines = ["\t".join(header)]
        for seq in all_seqs:
            row = [seq] + [tsv_dict.get(s, {}).get(seq, "") for s in thresholds]
            summary_lines.append("\t".join(row))
        summary_tsv.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        summary_file = tmp_run_dir / "clustering_summary.txt"
        summary_text = f"=== Clustering Summary for {base_name} ===\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n--- Representative Sequences ---\nRepresentative sequences retained: {total_seqs}\n\n--- Clustering Metrics ---\n"
        if metrics_file.exists():
            summary_text += metrics_file.read_text(encoding="utf-8")
        else:
            summary_text += "No metrics file found.\n"
        summary_file.write_text(summary_text, encoding="utf-8")
        for p in tmp_run_dir.glob("*"):
            if p.is_file() and not p.name.startswith(("clusterDB", "dedupDB")) and not p.name.endswith((".index", ".dbtype", ".lookup", ".source")):
                shutil.copy(p, final_dir / p.name)
        shutil.rmtree(tmp_run_dir, ignore_errors=True)
    exec_time, max_mem, dt = f"{time.time() - start_time:.2f}s", f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB", datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[PYTHON-INFO] {dt} {exec_time} {max_mem}")

if __name__ == "__main__":
    main()