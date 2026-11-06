#!/bin/bash
#SBATCH --job-name=SCambrian
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/snakemake/run_all_%j.log
#SBATCH --open-mode=append
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
CONFIG_YAML="configs/param_configs.yaml"
START_TIME=$(date +%s)
[ -f "${CONFIG_YAML}" ] || { echo "[SLURM-INFO] Error: ${CONFIG_YAML} not found in $(pwd). Submit from the repo root." >&2; exit 1; }
command -v snakemake >/dev/null || { echo "[SLURM-INFO] Error: snakemake missing from ./.env; rerun sbatch esmc_config.sh" >&2; exit 1; }
STEPS=("$@")
TARGETS=()
for step in "${STEPS[@]}"; do
  case "${step}" in
    01|02|03|04|05|06) TARGETS+=("step${step}") ;;
    *) echo "[SLURM-INFO] Unknown step: ${step} (use 01 02 03 04 05 06)" >&2; exit 1 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(all)
[[ " ${STEPS[*]} " == *" 02 "* ]] && NEW_ANALYSIS=1
[ ${#STEPS[@]} -eq 0 ] && [ -z "$(find analyses -mindepth 1 -maxdepth 1 -type d -name '*_SCambrian' 2>/dev/null | head -n 1)" ] && NEW_ANALYSIS=1
NEW_ANALYSIS="${NEW_ANALYSIS:-0}" source slurms/analysis_env.sh
mkdir -p logs/snakemake logs/uniprot logs/preprocess logs/mmseqs logs/ESMC logs/scanpy
echo "[SLURM-INFO] Analysis ${ANALYSIS_NAME} | targets ${TARGETS[*]}"
###################################### run
DRY=()
[ "${DRY_RUN:-0}" = "1" ] && DRY=(-n)
snakemake -s slurms/Snakefile "${TARGETS[@]}" "${DRY[@]}" \
  --jobs "${MAX_JOBS:-20}" --resources dgx=2 mmseqs=3 \
  --latency-wait 60 --rerun-incomplete --printshellcmds \
  --rerun-triggers mtime params code || { echo "[SLURM-INFO] Error: snakemake failed" >&2; exit 1; }
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
