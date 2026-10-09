#!/bin/bash
#SBATCH --job-name=UPdownload
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=logs/log_downloads_UniProtKB.log
#SBATCH --array=1-5%3
#SBATCH --open-mode=append
###################################### constants-and-init
CONFIG_JSON="configs/param_config.json"
START_TIME=$(date +%s)
###################################### parse-config
[ ! -f "${CONFIG_JSON}" ] && echo "[SLURM-INFO] Missing config JSON" && exit 1
SEQ_DIR=$(jq -r '.seq_out_dir' "${CONFIG_JSON}")
FEAT_DIR=$(jq -r '.features_out_dir' "${CONFIG_JSON}")
TSV_PATH=$(jq -r '.config_path' "${CONFIG_JSON}")
[ ! -f "${TSV_PATH}" ] && echo "[SLURM-INFO] Missing TSV file" && exit 1
eval "$(awk -F'\t' -v idx="${SLURM_ARRAY_TASK_ID}" '{sub(/\r$/,"")} NR==1 {for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i); if($i=="query")p=i; if($i=="label")l=i}} NR==idx+1 {if(p && $p!=""){q=$p; gsub(/\\/,"\\\\",q); gsub(/"/,"\\\"",q); print "QUERY_STR=\""q"\""} if(l && $l!=""){lbl=$l; gsub(/\\/,"\\\\",lbl); gsub(/"/,"\\\"",lbl); print "LABEL=\""lbl"\""}} END {if(NR-1<idx)print "EXC=true";else print "EXC=false"}' "${TSV_PATH}")"
mkdir -p "${SEQ_DIR}" "${FEAT_DIR}"
[ "${EXC}" = "true" ] && echo "[SLURM-INFO] Task exceeds TSV length" && exit 0
[ -z "${QUERY_STR}" ] && echo "[SLURM-INFO] Missing query" && exit 0
###################################### run-downloads
FP_SEQ="${SEQ_DIR}/${LABEL}_seqs.tsv"
FP_FAST="${SEQ_DIR}/${LABEL}_seqs.fasta"
FP_FEAT="${FEAT_DIR}/${LABEL}_feats.tsv"
[ -f "${FP_SEQ}" ] && [ -f "${FP_FAST}" ] && [ -f "${FP_FEAT}" ] && echo "[SLURM-INFO] Files exist" && exit 0
python3 src/dw_uniprot_seqs.py --query "${QUERY_STR}" --label "${LABEL}" --fp_seqs "${FP_SEQ}" --fp_fasta "${FP_FAST}" --fp_ids "${SEQ_DIR}/${LABEL}_ids.txt"
python3 src/dw_uniprot_features.py --query "${QUERY_STR}" --label "${LABEL}" --fp_feats "${FP_FEAT}"
echo "[SLURM-INFO] Downloaded data for ${LABEL}"
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"

