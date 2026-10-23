#!/bin/bash
#SBATCH --job-name=tsne_umap_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tsne_umap_%A_%a.log
#SBATCH --array=0-2
############### <init>
source slurms/common.sh
activate_env
COMPONENTS=(${DIMRED_COMPONENTS:-2 3})
N_JOBS="${SLURM_CPUS_PER_TASK:-8}"

############### <inputs>
REPS_DIR=$(cfg extract_reps_biohub reps_outdir data/esm-reps/esmc_6b)
mapfile -t NPZ_ARRAY < <(find "$REPS_DIR" -maxdepth 1 -type f -name "*_reps.npz" 2>/dev/null | LC_ALL=C sort)
[ "${#NPZ_ARRAY[@]}" -eq 0 ] && slurm_info "Error: no *_reps.npz files found in $REPS_DIR" >&2 && exit 1
AVAIL_MEM_BYTES=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  MODES=("$SLURM_ARRAY_TASK_ID")
else
  MODES=($(seq 0 "${#NPZ_ARRAY[@]}"))
fi

############### <run>
for MODE in "${MODES[@]}"; do
  if [ "$MODE" -eq 0 ]; then
    slurm_info "=== MODE: COMBINED (all files) ==="
    INPUTS=("${NPZ_ARRAY[@]}")
  elif [ "$MODE" -gt "${#NPZ_ARRAY[@]}" ]; then
    slurm_info "Task $MODE exceeds available files (${#NPZ_ARRAY[@]}), exiting cleanly"
    break
  else
    INPUTS=("${NPZ_ARRAY[$((MODE - 1))]}")
    slurm_info "=== MODE: SEPARATE ($(basename "${INPUTS[0]}")) ==="
  fi
  du -sh "${INPUTS[@]}"
  TOTAL_BYTES=$(du -bc "${INPUTS[@]}" | tail -n1 | cut -f1)
  [ "$TOTAL_BYTES" -gt "$AVAIL_MEM_BYTES" ] && slurm_info "Error: input size exceeds available system memory" >&2 && exit 1
  TENSORS=$(printf "%s\n" "${INPUTS[@]}" | paste -sd, -)
  for N in "${COMPONENTS[@]}"; do
    slurm_info "Running t-SNE (n_components=$N)"
    python3 src/npz_tsne.py --yaml-in "$CONFIG_YAML" --tensor "$TENSORS" --n-components "$N" --n-jobs "$N_JOBS" || exit 1
    slurm_info "Running UMAP (n_components=$N)"
    python3 src/npz_umap.py --yaml-in "$CONFIG_YAML" --tensor "$TENSORS" --n-components "$N" --n-jobs "$N_JOBS" || exit 1
  done
done
finalize
