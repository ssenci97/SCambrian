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
#SBATCH --array=1-2
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
slurm_info "Found ${#FILES[@]} deduplicated FASTA file(s) in $DEDUP_DIR"
TIMER=()
[ -x /usr/bin/time ] && TIMER=(/usr/bin/time -f "[SLURM-INFO] real %e s maxrss %M KB")

############### <run>
for IDX in $(task_indices "${#FILES[@]}"); do
  if [ "$IDX" -gt "${#FILES[@]}" ]; then
    slurm_info "Task $IDX exceeds available files (${#FILES[@]}), exiting cleanly"
    break
  fi
  INPUT="${FILES[$((IDX - 1))]}"
  slurm_info "Task $IDX: $INPUT"
  slurm_info "Running extract_reps_biohub.py"
  "${TIMER[@]}" python3 src/extract_reps_biohub.py --yaml-in "$CONFIG_YAML" --fasta "$INPUT" || exit 1
  slurm_info "Running extract_contacts_biohub.py"
  "${TIMER[@]}" python3 src/extract_contacts_biohub.py --yaml-in "$CONFIG_YAML" --fasta "$INPUT" || exit 1
done
finalize
