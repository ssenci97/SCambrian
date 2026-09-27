#!/bin/bash
#SBATCH --job-name=scanpy
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/scanpy.log

source /orfeo/scratch/area/ssenci/venvs/ml_transformers/bin/activate

# Fetch reps directory from JSON and find all .npz files dynamically
REPS_DIR=$(jq -r '.reps_out_dir' configs/param_config.json)
NPZ_FILES=$(find "$REPS_DIR" -name "*.npz" | paste -sd, -)

python3 src/npz_scanpy.py --npzs "$NPZ_FILES"
