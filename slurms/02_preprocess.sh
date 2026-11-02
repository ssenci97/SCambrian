#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess/preprocess_%j.log
#SBATCH --open-mode=append
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
START_TIME=$(date +%s)
NEW_ANALYSIS=1 source slurms/analysis_env.sh
###################################### run
python3 src/preprocess_sequences.py --yaml-in --yaml-out || exit 1
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
