#!/bin/bash
#SBATCH --job-name=summary
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scanpy/summary_%j.log
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
START_TIME=$(date +%s)
source slurms/analysis_env.sh
[ -d "${ANALYSIS_DIR}" ] || { echo "[SLURM-INFO] Error: missing ${ANALYSIS_DIR}." >&2; exit 1; }
mapfile -t H5ADS < <(find "${ANALYSIS_DIR}" -mindepth 2 -maxdepth 2 -name "projections.h5ad" | sort)
[ ${#H5ADS[@]} -gt 0 ] || { echo "[SLURM-INFO] Error: no projections.h5ad in ${ANALYSIS_DIR}/<model>/; run step 05 first." >&2; exit 1; }
###################################### run
YAML_OUT="--yaml-out"
for H5AD in "${H5ADS[@]}"; do
  MODEL=$(basename "$(dirname "${H5AD}")")
  echo "[SLURM-INFO] --- Cluster names and summary | model ${MODEL} ---"
  python3 src/compile_summary.py --analysis_dir "${ANALYSIS_DIR}" --model "${MODEL}" --yaml-in ${YAML_OUT} || exit 1
  YAML_OUT=""
done
H5AD_LIST=$(printf "%s\n" "${H5ADS[@]}" | paste -sd, -)
echo "[SLURM-INFO] --- Species report of the report series ---"
python3 src/h5ad_scanpy.py --h5ad "${H5AD_LIST}" --yaml-in --yaml-out || exit 1
echo "[SLURM-INFO] --- Interactive 3D views ---"
python3 src/h5ad_interactive.py --h5ad "${H5AD_LIST}" --yaml-in --yaml-out || exit 1
echo "[SLURM-INFO] --- RO-Crate of ${ANALYSIS_NAME} ---"
python3 src/ro_crate.py --analysis_dir "${ANALYSIS_DIR}" --yaml-in --yaml-out || exit 1
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
