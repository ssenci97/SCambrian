#!/bin/bash
#SBATCH --job-name=dw_UP
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uniprot/dw_UP_%A_%a.log
#SBATCH --array=1-3
#SBATCH --open-mode=append
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
CONFIG_YAML="configs/param_configs.yaml"
START_TIME=$(date +%s)
[ ! -f "${CONFIG_YAML}" ] && echo "[SLURM-INFO] Missing config YAML" && exit 1
if [ "${SLURM_ARRAY_TASK_ID:-1}" -eq 1 ]; then
  python3 src/dw_uniprot_seqs.py --yaml-out
  python3 src/dw_uniprot_features.py --yaml-out
fi
###################################### parse-config
eval "$(python3 -c "import yaml; p = yaml.safe_load(open('${CONFIG_YAML}')); s1 = p.get('dw_uniprot_seqs', {}); s2 = p.get('dw_uniprot_features', {}); print(f'SEQ_DIR={s1.get(\"sequences_dir\", \"data/sequences\")}'); print(f'FEAT_DIR={s2.get(\"features_dir\", \"data/features\")}'); print(f'TSV_PATH={s1.get(\"query_table\", \"configs/query_configs.tsv\")}'); print(f'OVERWRITE={str(bool(s1.get(\"overwrite\") or s2.get(\"overwrite\"))).lower()}')")"
[ ! -f "${TSV_PATH}" ] && echo "[SLURM-INFO] Missing TSV file" && exit 1
eval "$(awk -F'\t' -v idx="${SLURM_ARRAY_TASK_ID}" '{sub(/\r$/,"")} NR==1 {for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/,"",$i); if($i=="query")p=i; if($i=="label")l=i}} NR==idx+1 {if(p && $p!=""){q=$p; gsub(/\\/,"\\\\",q); gsub(/"/,"\\\"",q); print "QUERY_STR=\""q"\""} if(l && $l!=""){lbl=$l; gsub(/\\/,"\\\\",lbl); gsub(/"/,"\\\"",lbl); print "LABEL=\""lbl"\""}} END {if(NR-1<idx)print "EXC=true";else print "EXC=false"}' "${TSV_PATH}")"
mkdir -p "${SEQ_DIR}" "${FEAT_DIR}"
[ "${EXC}" = "true" ] && echo "[SLURM-INFO] Task exceeds TSV length" && exit 0
[ -z "${QUERY_STR}" ] && echo "[SLURM-INFO] Missing query" && exit 0
###################################### local-source
if [[ "${QUERY_STR}" =~ \.(fa|faa|fasta)$ ]]; then
  FP_FASTA="${SEQ_DIR}/${LABEL}_seqs.fasta"
  [ ! -f "${QUERY_STR}" ] && echo "[SLURM-INFO] Missing local source ${QUERY_STR}" && exit 1
  [ "${OVERWRITE}" != "true" ] && [ -f "${FP_FASTA}" ] && echo "[SLURM-INFO] Files exist. Skipping" && exit 0
  cp "${QUERY_STR}" "${FP_FASTA}" && echo "[SLURM-INFO] Copied ${QUERY_STR} -> ${FP_FASTA}"
  echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
  exit 0
fi
###################################### run-downloads
FP_TSV="${SEQ_DIR}/${LABEL}_seqs.tsv"
FP_FASTA="${SEQ_DIR}/${LABEL}_seqs.fasta"
FP_FEAT="${FEAT_DIR}/${LABEL}_feats.tsv"
[ "${OVERWRITE}" != "true" ] && [ -f "${FP_TSV}" ] && [ -f "${FP_FASTA}" ] && [ -f "${FP_FEAT}" ] && echo "[SLURM-INFO] Files exist. Skipping" && exit 0
python3 src/dw_uniprot_seqs.py --query "${QUERY_STR}" --label "${LABEL}" --out_tsv "${FP_TSV}" --out_fasta "${FP_FASTA}" --out_ids "${SEQ_DIR}/${LABEL}_ids.txt" --yaml-in "${CONFIG_YAML}"
python3 src/dw_uniprot_features.py --query "${QUERY_STR}" --label "${LABEL}" --out_tsv "${FP_FEAT}" --yaml-in "${CONFIG_YAML}"
echo "[SLURM-INFO] Downloaded data for ${LABEL}"
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
