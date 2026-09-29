#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ESMC
#SBATCH --output=logs/CNT/ESMC_%A_%a.log
#SBATCH --array=0-2
#SBATCH --open-mode=append
###################################### env
module purge
unset LD_LIBRARY_PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'fedora41' | paste -sd:)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH="/usr/bin:$PATH"
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache"
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p logs tmp .cache "$TRITON_CACHE_DIR"
echo "[SLURM-INFO] Forced CC=$CC CXX=$CXX"
echo "[SLURM-INFO] which gcc: $(which gcc)"
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
echo "[SLURM-INFO] Active Python: $(which python)"

###################################### config
JSON_CONFIG="configs/param_config.json"
FASTA_DIR=$(jq -r '.dedup_out_dir' "$JSON_CONFIG")
CONFIG_FILE=$(jq -r '.config_path' "$JSON_CONFIG")
CNT_DIR=$(jq -r '.contacts_outdir' "$JSON_CONFIG")
mkdir -p "$CNT_DIR"
echo "[SLURM-INFO] Job started on $(date)"
###################################### files
if [ ! -f "$CONFIG_FILE" ]; then
  echo "[SLURM-INFO] Error: Config file $CONFIG_FILE not found."
  exit 1
fi
TARGET_HEADER="label"
files=()
while IFS= read -r label; do
  target_file="${FASTA_DIR}/${label}_dedup.fasta"
  if [ -f "$target_file" ]; then
    files+=("$target_file")
  else
    echo "[SLURM-INFO] Warning: $target_file (label: $label) not found."
  fi
done < <(awk -F'\t' -v header="$TARGET_HEADER" '
  NR==1{for(i=1;i<=NF;i++)if($i==header){col=i;break};next}
  col&&$col!=""{print $col}
' "$CONFIG_FILE")
if [ "${#files[@]}" -eq 0 ]; then
  echo "[SLURM-INFO] Error: No matching fasta files found."
  exit 1
elif [ "$SLURM_ARRAY_TASK_ID" -ge "${#files[@]}" ]; then
  echo "[SLURM-INFO] Task ID $SLURM_ARRAY_TASK_ID exceeds available files (${#files[@]}). Exiting cleanly."
  exit 0
fi
input_file="${files[$SLURM_ARRAY_TASK_ID]}"
filename=$(basename "$input_file")
base_name="${filename%_dedup.fasta}"
base_name="${base_name%_seqs}"
output_file="${CNT_DIR}/${base_name}_esmc6b_contacts.parquet"
echo "[SLURM-INFO] Target input: $input_file"
echo "[SLURM-INFO] Target output: $output_file"
if [ -f "$output_file" ]; then
  echo "[SLURM-INFO] SUCCESS: $output_file already exists. Skipping."
  exit 0
fi
###################################### run
START=$(date +%s)
/usr/bin/time -f "real %e s\nmaxrss %M KB" python src/extract_attentions_biohub.py --fasta "$input_file" --out "$output_file"
END=$(date +%s)
echo "[SLURM-INFO] Done $(date) elapsed $((END-START))s"
