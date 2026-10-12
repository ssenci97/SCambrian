#!/bin/bash
#SBATCH --job-name=scanpy_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/scanpy_%A.log

source /orfeo/scratch/area/ssenci/venvs/ml_transformers/bin/activate

REPS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml'))['extract_reps_biohub']['reps_outdir'])")
mapfile -t NPZ_ARRAY < <(find "$REPS_DIR" -name "*.npz" | sort)

if [ ${#NPZ_ARRAY[@]} -lt 2 ]; then
    echo "[SLURM-INFO] Error: At least 2 .npz files are required in $REPS_DIR for comparative analysis." >&2
    exit 1
fi

echo "[SLURM-INFO] === MODE: COMBINED (All files jointly) ==="
du -sh "${NPZ_ARRAY[@]}"

NPZ_FILES_STR=$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)

echo "[SLURM-INFO] --- Running Scanpy Joint Analysis ---"
python3 src/npz_scanpy.py --npzs "$NPZ_FILES_STR" --yaml-in --yaml-out