#!/bin/bash
#SBATCH --job-name=dw_UP
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uniprot/dw_UP_%A_%a.log
#SBATCH --array=1-2
#SBATCH --open-mode=append
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
python3 src/dw_uniprot_seqs.py --yaml-out
python3 src/dw_uniprot_features.py --yaml-out

CONFIG_YAML="configs/param_configs.yaml"
START_TIME=$(date +%s)
###################################### parse-config
[ ! -f "${CONFIG_YAML}" ] && echo "[SLURM-INFO] Missing config YAML" && exit 1
eval "$(python3 -c "import yaml; p = yaml.safe_load(open('${CONFIG_YAML}')); s1 = p.get('dw_uniprot_seqs', {}); s2 = p.get('dw_uniprot_features', {}); print(f'SEQ_DIR={s1.get(\"seq_out_dir\", \"data/sequences\")}'); print(f'FEAT_DIR={s2.get(\"features_out_dir\", \"data/features\")}'); print(f'TSV_PATH={s1.get(\"config_path\", \"configs/query_configs.tsv\")}')")"
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
python3 src/dw_uniprot_seqs.py --query "${QUERY_STR}" --label "${LABEL}" --fp_seqs "${FP_SEQ}" --fp_fasta "${FP_FAST}" --fp_ids "${SEQ_DIR}/${LABEL}_ids.txt" --yaml-in "${CONFIG_YAML}"
python3 src/dw_uniprot_features.py --query "${QUERY_STR}" --label "${LABEL}" --fp_feats "${FP_FEAT}" --yaml-in "${CONFIG_YAML}"
echo "[SLURM-INFO] Downloaded data for ${LABEL}"
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
