#!/bin/bash
#SBATCH --job-name=scanpy_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/scanpy_%j.log
############### <init>
source slurms/common.sh
activate_env

############### <inputs>
REPS_DIR=$(cfg extract_reps_biohub reps_outdir data/esm-reps/esmc_6b)
mapfile -t NPZ_ARRAY < <(find "$REPS_DIR" -maxdepth 1 -type f -name "*_reps.npz" 2>/dev/null | LC_ALL=C sort)
[ "${#NPZ_ARRAY[@]}" -lt 2 ] && slurm_info "Error: at least 2 *_reps.npz files are required in $REPS_DIR for comparative analysis" >&2 && exit 1

############### <run>
slurm_info "=== MODE: COMBINED (all files jointly) ==="
du -sh "${NPZ_ARRAY[@]}"
python3 src/npz_scanpy.py --yaml-in "$CONFIG_YAML" --npzs "$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)" || exit 1
finalize
