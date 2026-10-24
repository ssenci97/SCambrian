#!/bin/bash
#SBATCH --job-name=mmseqs
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmseqs_%A_%a.log
#SBATCH --array=1-3
#SBATCH --open-mode=append
############### <init>
source slurms/common.sh
activate_env
[ ! -f "$CONFIG_YAML" ] && slurm_info "Missing config YAML $CONFIG_YAML" && exit 1

############### <inputs>
SEQ_DIR=$(cfg run_mmseqs2 seq_in_dir data/sequences)
mapfile -t FASTAS < <(find "$SEQ_DIR" -maxdepth 1 -type f \( -name '*.fasta' -o -name '*.fa' -o -name '*.faa' \) 2>/dev/null | LC_ALL=C sort)
[ "${#FASTAS[@]}" -eq 0 ] && slurm_info "Error: no FASTA files (*.fasta, *.fa, *.faa) in $SEQ_DIR" && exit 1
slurm_info "Found ${#FASTAS[@]} FASTA file(s) in $SEQ_DIR"

############### <run>
for IDX in $(task_indices "${#FASTAS[@]}"); do
  if [ "$IDX" -gt "${#FASTAS[@]}" ]; then
    slurm_info "Task $IDX exceeds available FASTA files (${#FASTAS[@]}), exiting cleanly"
    break
  fi
  slurm_info "Task $IDX: ${FASTAS[$((IDX - 1))]}"
  python3 src/run_mmseqs2.py --yaml-in "$CONFIG_YAML" --index "$IDX" || exit 1
done
finalize
