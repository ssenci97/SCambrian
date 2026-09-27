#!/bin/bash
#SBATCH --job-name=mmseqs_clust
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mmseqs2_clust.log
#SBATCH --array=1-25
#SBATCH --open-mode=append

set -euo pipefail

###################################### <CONSTANTS>
CONFIG_JSON="configs/param_config.json"

MMSEQS=$(jq -r '.mmseqs_path' "$CONFIG_JSON")
THREADS=${SLURM_CPUS_PER_TASK:-16}
SEED=$(jq -r '.seed' "$CONFIG_JSON")
DEDUP_DIR=$(jq -r '.dedup_out_dir' "$CONFIG_JSON")
OUT_BASE_DIR=$(jq -r '.clusters_out_dir' "$CONFIG_JSON")
cov=$(jq -r '.mmseqs.clust_cov' "$CONFIG_JSON")
cov_mode=$(jq -r '.mmseqs.clust_cov_mode' "$CONFIG_JSON")

mapfile -t THRESHOLDS < <(jq -r '.mmseqs.thresholds[]' "$CONFIG_JSON")

###################################### <SETUP>
mapfile -t FASTA_FILES < <(find "${DEDUP_DIR}" -maxdepth 1 -type f -name "*.fasta" | sort)
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
dedup_fasta="${FASTA_FILES[$ARRAY_INDEX]:-}"

if [ -z "${dedup_fasta}" ]; then
  exit 0
fi

base_name=$(basename "$dedup_fasta" .fasta)
base_name=${base_name%_dedup}
out_dir="${OUT_BASE_DIR}/${base_name}"
tmp_dir="${out_dir}/tmp"

mkdir -p "$out_dir" "$tmp_dir"

if [ ! -f "${out_dir}/dedupDB.index" ]; then
  "$MMSEQS" createdb "$dedup_fasta" "${out_dir}/dedupDB" || exit 1
fi

total_seqs=$(grep -c "^>" "$dedup_fasta" 2>/dev/null || echo "0")
metrics_file="${out_dir}/${base_name}_cluster_metrics.tsv"

if [ ! -f "$metrics_file" ]; then
  echo -e "threshold\tclusters\tsingleton_clusters\tmax_cluster_size\tavg_cluster_size\tcoverage" > "$metrics_file"
fi

###################################### <CLUSTERING>
current_db="${out_dir}/dedupDB"
prev_tsv=""

for i in "${!THRESHOLDS[@]}"; do
  seq_id="${THRESHOLDS[$i]}"
  threshold_clean="${seq_id//./_}"
  cluster_db="${out_dir}/clusterDB_id${threshold_clean}"
  tsv_out="${out_dir}/${base_name}_clusters_id${threshold_clean}.tsv"
  fasta_out="${out_dir}/${base_name}_representatives_id${threshold_clean}.fasta"

  if [ -f "$tsv_out" ] && [ -f "$fasta_out" ]; then
    rep_db="${cluster_db}_rep"
    if [ -f "${rep_db}.index" ]; then
      current_db="${rep_db}"
      prev_tsv="$tsv_out"
      continue
    fi
  fi

  "$MMSEQS" cluster "$current_db" "$cluster_db" "$tmp_dir" --min-seq-id "$seq_id" -c "$cov" --cov-mode "$cov_mode" --threads "$THREADS" || continue

  if [ "$i" -eq 0 ]; then
    "$MMSEQS" createtsv "${out_dir}/dedupDB" "${out_dir}/dedupDB" "$cluster_db" "$tsv_out" || continue
  else
    intermediate_tsv="${out_dir}/${base_name}_intermediate_id${threshold_clean}.tsv"
    "$MMSEQS" createtsv "$current_db" "$current_db" "$cluster_db" "$intermediate_tsv" || continue
    awk 'NR==FNR { new_rep[$2] = $1; next } { if ($1 in new_rep) { print new_rep[$1] "\t" $2; } }' "$intermediate_tsv" "$prev_tsv" > "$tsv_out" || continue
    rm -f "$intermediate_tsv"
  fi

  "$MMSEQS" result2repseq "$current_db" "$cluster_db" "${cluster_db}_rep" || continue
  "$MMSEQS" result2flat "${cluster_db}_rep" "${cluster_db}_rep" "${cluster_db}_rep" "$fasta_out" --use-fasta-header 1 || continue

  num_clusters=$(awk -F'\t' '{print $1}' "$tsv_out" | sort -u | wc -l)
  singleton_clusters=$(awk -F'\t' '{count[$1]++} END {singletons=0; for (c in count) if (count[c]==1) singletons++; print singletons+0}' "$tsv_out")
  max_size=$(awk -F'\t' '{count[$1]++} END {max=0; for (c in count) if (count[c]>max) max=count[c]; print max}' "$tsv_out")
  avg_size=$(echo "scale=2; $total_seqs / $num_clusters" | bc -l)

  if ! grep -q "^${seq_id}\t" "$metrics_file"; then
    echo -e "${seq_id}\t${num_clusters}\t${singleton_clusters}\t${max_size}\t${avg_size}\t${cov}" >> "$metrics_file"
  fi

  current_db="${cluster_db}_rep"
  prev_tsv="$tsv_out"
  rm -f "${cluster_db}" "${cluster_db}.index" "${cluster_db}_h" "${cluster_db}_h.index"
done

###################################### <CLEANUP>
summary_file="${out_dir}/clustering_summary.txt"
{
  echo -e "=== Clustering Summary for ${base_name} ===\nTimestamp: $(date +'%Y-%m-%d %H:%M:%S')\n\n--- Representative Sequences ---"
  echo -e "Representative sequences retained: ${total_seqs}\n\n--- Clustering Metrics ---"
  [ -f "$metrics_file" ] && cat "$metrics_file" || echo "No metrics file found."
} > "$summary_file"

rm -rf "$tmp_dir"
rm -f "${out_dir}/dedupDB"*

executiontime="${SECONDS}s"
maxmemory="$(grep VmPeak /proc/$$/status 2>/dev/null | awk '{print $2$3}' || echo 'N/A')"
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') ${executiontime} ${maxmemory}"