#!/bin/bash
#SBATCH --job-name=mmseqs
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmseqs/mmseqs_%A_%a.log
#SBATCH --array=1-11%3
#SBATCH --open-mode=append
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
START_TIME=$(date +%s)
source slurms/analysis_env.sh
TSV_PATH=$(python3 -c "import yaml; print((yaml.safe_load(open('${CONFIG_YAML}')) or {}).get('dw_uniprot_seqs', {}).get('query_table', 'configs/query_configs.tsv'))")
[ -f "${TSV_PATH}" ] || { echo "[SLURM-INFO] Error: missing TSV ${TSV_PATH}." >&2; exit 1; }
N_TARGETS=$(awk -F'\t' '{sub(/\r$/,"")} NR==1{for(i=1;i<=NF;i++){if($i=="label")l=i; if($i=="input_type")t=i};next} l&&$l!=""&&(!t||$t!="probe"){n++} END{print n+0}' "${TSV_PATH}")
TASK=${SLURM_ARRAY_TASK_ID:-1}
###################################### run
YAML_OUT=""
[ "${TASK}" -eq 1 ] && YAML_OUT="--yaml-out"
if [ "${TASK}" -le "${N_TARGETS}" ]; then
  echo "[SLURM-INFO] Internal clustering of target ${TASK}/${N_TARGETS}"
  python3 src/run_mmseqs2.py --yaml-in ${YAML_OUT} --index "${TASK}" || exit 1
elif [ "${TASK}" -eq $((N_TARGETS + 1)) ]; then
  echo "[SLURM-INFO] Joint clustering of all samples"
  python3 src/run_mmseqs2.py --yaml-in ${YAML_OUT} --joint --analysis_dir "${ANALYSIS_DIR}" || exit 1
else
  echo "[SLURM-INFO] Task ${TASK} exceeds ${N_TARGETS} targets + 1 joint. Skipping."
fi
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
