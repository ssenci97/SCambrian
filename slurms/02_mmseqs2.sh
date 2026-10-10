#!/bin/bash
#SBATCH --job-name=mmseqs_pipeline
#SBATCH --partition=GENOA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmseqs_pipeline.log
#SBATCH --array=1-5
#SBATCH --open-mode=append
###################################### constants-and-init
CONFIG_YAML="configs/param_configs.yaml"
START_TIME=$(date +%s)
###################################### run-pipeline
[ ! -f "${CONFIG_YAML}" ] && echo "[SLURM-INFO] Missing config YAML" && exit 1
python3 src/run_mmseqs2.py --yaml-in "${CONFIG_YAML}" --index "${SLURM_ARRAY_TASK_ID}"
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"

