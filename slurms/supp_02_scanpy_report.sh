#!/bin/bash
#SBATCH --job-name=scanpy_report
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scanpy/scanpy_report_%A.log

###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
[ -f configs/param_configs.yaml ] || { echo "[SLURM-INFO] Error: configs/param_configs.yaml not found in $(pwd). Submit from the repo root." >&2; exit 1; }

###################################### config
# Reports every <outdir>/<run>/*.h5ad written by supp_01_scanpy.sh
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml')).get('npz_scanpy', {}).get('outdir', 'data/scanpy/'))")
[ -n "$OUT_DIR" ] && [ "$OUT_DIR" != "/" ] && [ -d "$OUT_DIR" ] || { echo "[SLURM-INFO] Error: invalid OUT_DIR '$OUT_DIR'." >&2; exit 1; }
mapfile -t H5ADS < <(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -name "*.h5ad" | sort)
[ ${#H5ADS[@]} -gt 0 ] || { echo "[SLURM-INFO] Error: no .h5ad found in $OUT_DIR/<run>/." >&2; exit 1; }

###################################### run
echo "[SLURM-INFO] --- Reporting ${#H5ADS[@]} .h5ad files ---"
python3 src/h5ad_scanpy.py --h5ad "$(printf "%s\n" "${H5ADS[@]}" | paste -sd, -)" --yaml-in --yaml-out
