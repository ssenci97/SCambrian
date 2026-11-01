#!/bin/bash
#SBATCH --job-name=tests
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tests/tests_%j.log
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
###################################### constants-and-init
MARKERS="${1:-static or cpu}"
START_TIME=$(date +%s)
[ -f pytest.ini ] || { echo "[SLURM-INFO] Error: pytest.ini not found in $(pwd). Submit from the repo root." >&2; exit 1; }
export MPLBACKEND=Agg
echo "[SLURM-INFO] pytest -m \"${MARKERS}\" on $(hostname)"
###################################### run
python -m pytest -m "${MARKERS}"
RC=$?
[ "${RC}" -eq 0 ] && echo "[SLURM-INFO] All selected tests passed" || echo "[SLURM-INFO] Tests FAILED (exit ${RC})"
###################################### finalize
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
exit "${RC}"
