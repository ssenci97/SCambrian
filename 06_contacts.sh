#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=vision
#SBATCH --output=logs/vision.log
###################################### env
module purge
unset LD_LIBRARY_PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'fedora41' | paste -sd:)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH="/usr/bin:$PATH"
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache"
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p logs tmp .cache "$TRITON_CACHE_DIR"
echo "[SLURM-INFO] Forced CC=$CC CXX=$CXX"
echo "[SLURM-INFO] which gcc: $(which gcc)"
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
echo "[SLURM-INFO] Active Python: $(which python)"
###################################### config

python3 src/vision_att.py