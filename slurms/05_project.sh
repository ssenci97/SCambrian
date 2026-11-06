#!/bin/bash
#SBATCH --job-name=project
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/scanpy/project_%A_%a.log
#SBATCH --array=0-2
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
START_TIME=$(date +%s)
source slurms/analysis_env.sh
[ -f "${ANALYSIS_DIR}/sc_metadata.tsv" ] || { echo "[SLURM-INFO] Error: missing ${ANALYSIS_DIR}/sc_metadata.tsv; run step 02 first." >&2; exit 1; }
mapfile -t MODELS < <(python3 -c "import yaml; m = (yaml.safe_load(open('${CONFIG_YAML}')) or {}).get('extract_reps_biohub', {}).get('models', ['esmc_300m']); print('\n'.join([m] if isinstance(m, str) else m))")
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "${TASK}" -lt ${#MODELS[@]} ] || { echo "[SLURM-INFO] Task ${TASK} exceeds ${#MODELS[@]} models. Skipping."; exit 0; }
MODEL=${MODELS[$TASK]}
YAML_OUT=""
[ "${TASK}" -eq 0 ] && YAML_OUT="--yaml-out"
###################################### run
echo "[SLURM-INFO] Joint and internal UMAP/t-SNE projections, Leiden | model ${MODEL}"
python3 src/npz_scanpy.py --analysis_dir "${ANALYSIS_DIR}" --model "${MODEL}" --n_jobs "${SLURM_CPUS_PER_TASK:-1}" --yaml-in ${YAML_OUT} || exit 1
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
