#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ESMC
#SBATCH --output=logs/ESMC_%A_%a.log
#SBATCH --array=1-9
#SBATCH --open-mode=append
############### <init>
source slurms/common.sh
sanitize_toolchain
activate_env
[ ! -f "$CONFIG_YAML" ] && slurm_info "Missing config YAML $CONFIG_YAML" && exit 1

############### <inputs>
DEDUP_DIR=$(cfg run_mmseqs2 dedup_out_dir data/dedup)
mapfile -t FILES < <(find "$DEDUP_DIR" -maxdepth 1 -type f -name '*_dedup.fasta' 2>/dev/null | LC_ALL=C sort)
[ "${#FILES[@]}" -eq 0 ] && slurm_info "Error: no *_dedup.fasta files in $DEDUP_DIR; run slurms/01_mmseqs2.sh first" && exit 1
mapfile -t MODELS < <(cfg_list extract_reps_biohub model_names "esmc_300m esmc_600m esmc_6b")
[ "${#MODELS[@]}" -eq 0 ] && slurm_info "Error: no model_names in extract_reps_biohub" && exit 1
N_TASKS=$(( ${#FILES[@]} * ${#MODELS[@]} ))
slurm_info "Found ${#FILES[@]} deduplicated FASTA file(s) in $DEDUP_DIR x ${#MODELS[@]} model(s) (${MODELS[*]}) = $N_TASKS task(s)"
TIMER=()
[ -x /usr/bin/time ] && TIMER=(/usr/bin/time -f "[SLURM-INFO] real %e s maxrss %M KB")

############### <run>
# task i -> file (i-1)/M, model (i-1)%M; contacts (always ESMC-6B) run once per file, on its first model task
for IDX in $(task_indices "$N_TASKS"); do
  if [ "$IDX" -gt "$N_TASKS" ]; then
    slurm_info "Task $IDX exceeds available file x model pairs ($N_TASKS), exiting cleanly"
    break
  fi
  F_IDX=$(( (IDX - 1) / ${#MODELS[@]} ))
  M_IDX=$(( (IDX - 1) % ${#MODELS[@]} ))
  INPUT="${FILES[$F_IDX]}"
  MODEL="${MODELS[$M_IDX]}"
  slurm_info "Task $IDX: $INPUT ($MODEL)"
  slurm_info "Running extract_reps_biohub.py"
  "${TIMER[@]}" python3 src/extract_reps_biohub.py --yaml-in "$CONFIG_YAML" --fasta "$INPUT" --model_names "$MODEL" || exit 1
  if [ "$M_IDX" -eq 0 ]; then
    slurm_info "Running extract_contacts_biohub.py"
    "${TIMER[@]}" python3 src/extract_contacts_biohub.py --yaml-in "$CONFIG_YAML" --fasta "$INPUT" || exit 1
  fi
done
finalize
