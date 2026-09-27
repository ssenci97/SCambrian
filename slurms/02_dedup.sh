#!/bin/bash
#SBATCH --job-name=mmseqs_dedup
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=1G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mmseqs2.log
#SBATCH --open-mode=append
#SBATCH --array=1-25
set -euo pipefail

# Read configs from JSON using jq
PARAM_CONFIG="configs/param_config.json"
MMSEQS=$(jq -r '.mmseqs_path' "$PARAM_CONFIG")
IN_DIR=$(jq -r '.seq_out_dir' "$PARAM_CONFIG")
DEDUP_DIR=$(jq -r '.dedup_out_dir' "$PARAM_CONFIG")

# MMseqs2 parameters
MIN_SEQ_ID=$(jq -r '.mmseqs.dedup_min_seq_id' "$PARAM_CONFIG")
COV=$(jq -r '.mmseqs.dedup_cov' "$PARAM_CONFIG")
COV_MODE=$(jq -r '.mmseqs.dedup_cov_mode' "$PARAM_CONFIG")

THREADS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p logs "${IN_DIR}" "${DEDUP_DIR}"
mapfile -t RAW_FASTA < <(find "${IN_DIR}" -maxdepth 1 -type f -name "*_seqs.fasta" | sort)
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
in_fasta="${RAW_FASTA[$ARRAY_INDEX]:-}"

if [ -n "${in_fasta}" ]; then
  base_name=$(basename "$in_fasta" _seqs.fasta)
  out_dedup="${DEDUP_DIR}/${base_name}_dedup.fasta"
  tmp_dir="${DEDUP_DIR}/tmp_${base_name}"
  
  if [ ! -f "$out_dedup" ]; then
    mkdir -p "$tmp_dir"
    "$MMSEQS" easy-linclust "$in_fasta" "${DEDUP_DIR}/${base_name}_tmp" "$tmp_dir" \
      --min-seq-id "$MIN_SEQ_ID" \
      -c "$COV" \
      --cov-mode "$COV_MODE" \
      --threads "$THREADS"
    mv "${DEDUP_DIR}/${base_name}_tmp_rep_seq.fasta" "$out_dedup"
    rm -rf "${DEDUP_DIR}/${base_name}_tmp"* "$tmp_dir"
  fi
fi

executiontime="${SECONDS}s"
maxmemory="$(grep VmPeak /proc/$$/status 2>/dev/null | awk '{print $2$3}' || echo 'N/A')"
echo -e "\n[SLURM-INFO] Task:${SLURM_ARRAY_TASK_ID} Time:${executiontime} Mem:${maxmemory} Date:$(date +'%Y-%m-%d %H:%M:%S')\n"