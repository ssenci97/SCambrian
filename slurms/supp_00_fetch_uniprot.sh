#!/bin/bash
#SBATCH --job-name=dw_UP
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dw_UP_%A_%a.log
#SBATCH --array=1-3
#SBATCH --open-mode=append
############### <init>
source slurms/common.sh
activate_env
[ ! -f "$CONFIG_YAML" ] && slurm_info "Missing config YAML $CONFIG_YAML" && exit 1

############### <inputs>
TSV_PATH=$(cfg dw_uniprot_seqs config_path configs/query_configs.tsv)
SEQ_DIR=$(cfg dw_uniprot_seqs seq_out_dir data/sequences)
FEAT_DIR=$(cfg dw_uniprot_features features_out_dir data/features)
[ ! -f "$TSV_PATH" ] && slurm_info "Missing query TSV $TSV_PATH" && exit 1
mapfile -t LABELS < <(awk -F'\t' '{sub(/\r$/,"")} NR==1 {for(i=1;i<=NF;i++) if($i=="label") col=i; next} col && $col!="" {print $col}' "$TSV_PATH")
[ "${#LABELS[@]}" -eq 0 ] && slurm_info "Error: no labels in $TSV_PATH" && exit 1

############### <run>
for IDX in $(task_indices "${#LABELS[@]}"); do
  if [ "$IDX" -gt "${#LABELS[@]}" ]; then
    slurm_info "Task $IDX exceeds TSV length (${#LABELS[@]}), exiting cleanly"
    break
  fi
  LABEL="${LABELS[$((IDX - 1))]}"
  if [ -f "$SEQ_DIR/${LABEL}_seqs.fasta" ] && [ -f "$FEAT_DIR/${LABEL}_feats.tsv" ]; then
    slurm_info "Files exist for $LABEL, skipping"
    continue
  fi
  python3 src/dw_uniprot_seqs.py --yaml-in "$CONFIG_YAML" --label "$LABEL" || exit 1
  python3 src/dw_uniprot_features.py --yaml-in "$CONFIG_YAML" --label "$LABEL" || exit 1
  slurm_info "Downloaded data for $LABEL"
done
finalize
