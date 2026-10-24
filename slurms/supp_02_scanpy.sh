#!/bin/bash
#SBATCH --job-name=scanpy_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/scanpy_%A_%a.log
#SBATCH --array=0-2
############### <init>
source slurms/common.sh
activate_env

############### <inputs>
REPS_BASE=$(cfg extract_reps_biohub reps_outdir data/esm-reps)
SCANPY_BASE=$(cfg npz_scanpy outdir data/scanpy)
mapfile -t MODELS < <(cfg_list extract_reps_biohub model_names "esmc_300m esmc_600m esmc_6b")
[ "${#MODELS[@]}" -eq 0 ] && slurm_info "Error: no model_names in extract_reps_biohub" >&2 && exit 1
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASKS=("$SLURM_ARRAY_TASK_ID")
else
  TASKS=($(seq 0 $(( ${#MODELS[@]} - 1 ))))
fi

############### <run>
# task t -> model t; all *_reps.npz files of that model are clustered jointly
for TASK in "${TASKS[@]}"; do
  if [ "$TASK" -ge "${#MODELS[@]}" ]; then
    slurm_info "Task $TASK exceeds available models (${#MODELS[@]}), exiting cleanly"
    break
  fi
  MODEL="${MODELS[$TASK]}"
  mapfile -t NPZ_ARRAY < <(find "$REPS_BASE/$MODEL" -maxdepth 1 -type f -name "*_reps.npz" 2>/dev/null | LC_ALL=C sort)
  [ "${#NPZ_ARRAY[@]}" -lt 2 ] && slurm_info "Error: at least 2 *_reps.npz files are required in $REPS_BASE/$MODEL for comparative analysis" >&2 && exit 1
  slurm_info "=== MODEL: $MODEL | MODE: COMBINED (all files jointly) ==="
  du -sh "${NPZ_ARRAY[@]}"
  python3 src/npz_scanpy.py --yaml-in "$CONFIG_YAML" --npzs "$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)" --outdir "${SCANPY_BASE%/}/$MODEL" || exit 1
done
finalize
