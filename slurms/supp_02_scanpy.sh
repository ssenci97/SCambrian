#!/bin/bash
#SBATCH --job-name=scanpy_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/scanpy/scanpy_%A.log

###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env

# Fetch reps base directory (one subdirectory per model) and scanpy output directory from YAML config
REPS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml'))['extract_reps_biohub']['reps_outdir'])")
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml')).get('npz_scanpy', {}).get('outdir', 'data/scanpy/'))")
N_RUNS=0

for MODEL_DIR in "$REPS_DIR"/*/; do
    [ -d "$MODEL_DIR" ] || continue
    MODEL=$(basename "$MODEL_DIR")
    mapfile -t NPZ_ARRAY < <(find "$MODEL_DIR" -name "*_reps.npz" | sort)

    if [ ${#NPZ_ARRAY[@]} -lt 2 ]; then
        echo "[SLURM-INFO] Warning: At least 2 *_reps.npz files are required in $MODEL_DIR for comparative analysis. Skipping $MODEL." >&2
        continue
    fi

    echo "[SLURM-INFO] === MODE: COMBINED (All files jointly) | MODEL: $MODEL ==="
    du -sh "${NPZ_ARRAY[@]}"

    NPZ_FILES_STR=$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)

    echo "[SLURM-INFO] --- Running Scanpy Joint Analysis ---"
    python3 src/npz_scanpy.py --npzs "$NPZ_FILES_STR" --outdir "${OUT_DIR%/}/$MODEL" --yaml-in --yaml-out
    N_RUNS=$((N_RUNS + 1))
done

if [ "$N_RUNS" -eq 0 ]; then
    echo "[SLURM-INFO] Error: No model directory in $REPS_DIR has at least 2 *_reps.npz files." >&2
    exit 1
fi
