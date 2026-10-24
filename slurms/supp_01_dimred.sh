#!/bin/bash
#SBATCH --job-name=tsne_umap_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tsne_umap_%A_%a.log
#SBATCH --array=0-11
############### <init>
source slurms/common.sh
activate_env
COMPONENTS=(${DIMRED_COMPONENTS:-2 3})
N_JOBS="${SLURM_CPUS_PER_TASK:-8}"

############### <inputs>
REPS_BASE=$(cfg extract_reps_biohub reps_outdir data/esm-reps)
TSNE_BASE=$(cfg npz_tsne output_dir data/dimred)
UMAP_BASE=$(cfg npz_umap output_dir data/dimred)
mapfile -t MODELS < <(cfg_list extract_reps_biohub model_names "esmc_300m esmc_600m esmc_6b")
[ "${#MODELS[@]}" -eq 0 ] && slurm_info "Error: no model_names in extract_reps_biohub" >&2 && exit 1
list_npz() {
  find "$REPS_BASE/$1" -maxdepth 1 -type f -name "*_reps.npz" 2>/dev/null | LC_ALL=C sort
}
N_FILES=0
for M in "${MODELS[@]}"; do
  N=$(list_npz "$M" | wc -l)
  [ "$N" -gt "$N_FILES" ] && N_FILES=$N
done
[ "$N_FILES" -eq 0 ] && slurm_info "Error: no *_reps.npz files found under $REPS_BASE/{${MODELS[*]}}" >&2 && exit 1
N_MODES=$(( N_FILES + 1 ))
N_TASKS=$(( N_MODES * ${#MODELS[@]} ))
slurm_info "${#MODELS[@]} model(s) (${MODELS[*]}) x $N_MODES mode(s) (combined + $N_FILES file(s)) = $N_TASKS task(s)"
AVAIL_MEM_BYTES=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASKS=("$SLURM_ARRAY_TASK_ID")
else
  TASKS=($(seq 0 $(( N_TASKS - 1 ))))
fi

############### <run>
# task t -> model t/(N+1), mode t%(N+1); mode 0 = all files of that model combined, mode i = file i
for TASK in "${TASKS[@]}"; do
  if [ "$TASK" -ge "$N_TASKS" ]; then
    slurm_info "Task $TASK exceeds available model x mode pairs ($N_TASKS), exiting cleanly"
    break
  fi
  MODEL="${MODELS[$(( TASK / N_MODES ))]}"
  MODE=$(( TASK % N_MODES ))
  mapfile -t NPZ_ARRAY < <(list_npz "$MODEL")
  if [ "${#NPZ_ARRAY[@]}" -eq 0 ]; then
    slurm_info "Error: no *_reps.npz files found in $REPS_BASE/$MODEL" >&2
    exit 1
  elif [ "$MODE" -eq 0 ]; then
    slurm_info "=== MODEL: $MODEL | MODE: COMBINED (all files) ==="
    INPUTS=("${NPZ_ARRAY[@]}")
  elif [ "$MODE" -gt "${#NPZ_ARRAY[@]}" ]; then
    slurm_info "Task $TASK: mode $MODE exceeds available files for $MODEL (${#NPZ_ARRAY[@]}), skipping"
    continue
  else
    INPUTS=("${NPZ_ARRAY[$((MODE - 1))]}")
    slurm_info "=== MODEL: $MODEL | MODE: SEPARATE ($(basename "${INPUTS[0]}")) ==="
  fi
  du -sh "${INPUTS[@]}"
  TOTAL_BYTES=$(du -bc "${INPUTS[@]}" | tail -n1 | cut -f1)
  [ "$TOTAL_BYTES" -gt "$AVAIL_MEM_BYTES" ] && slurm_info "Error: input size exceeds available system memory" >&2 && exit 1
  TENSORS=$(printf "%s\n" "${INPUTS[@]}" | paste -sd, -)
  for N in "${COMPONENTS[@]}"; do
    slurm_info "Running t-SNE (n_components=$N)"
    python3 src/npz_tsne.py --yaml-in "$CONFIG_YAML" --tensor "$TENSORS" --n-components "$N" --n-jobs "$N_JOBS" --output-dir "${TSNE_BASE%/}/$MODEL" || exit 1
    slurm_info "Running UMAP (n_components=$N)"
    python3 src/npz_umap.py --yaml-in "$CONFIG_YAML" --tensor "$TENSORS" --n-components "$N" --n-jobs "$N_JOBS" --output-dir "${UMAP_BASE%/}/$MODEL" || exit 1
  done
done
finalize
