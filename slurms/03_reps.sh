#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ESMC
#SBATCH --output=logs/ESMC/ESMC_%A_%a.log
#SBATCH --array=0-11%2
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
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
echo "[SLURM-INFO] Active Python: $(which python)"
###################################### config
YAML_CONFIG="configs/param_configs.yaml"
eval "$(python3 -c '
import yaml
with open("configs/param_configs.yaml") as f:
    d = yaml.safe_load(f)
dedup = d.get("run_mmseqs2", {}).get("dedup_dir", "data/dedup")
reps_dir = d.get("extract_reps_biohub", {}).get("embeddings_dir", "data/esm-reps")
models = d.get("extract_reps_biohub", {}).get("models", "esmc_6b")
models = [models] if isinstance(models, str) else models
print("MODELS=(" + " ".join(models) + ")")
print(f"FASTA_DIR={dedup}")
print(f"REPS_DIR={reps_dir}")
print("TSV_PATH=" + d.get("dw_uniprot_seqs", {}).get("query_table", "configs/query_configs.tsv"))
')"
[ ! -f "${TSV_PATH}" ] && echo "[SLURM-INFO] Missing TSV file" && exit 1
mkdir -p "$REPS_DIR"
echo "[SLURM-INFO] Job started on $(date)"
###################################### parse args
YAML_ONLY=false
for arg in "$@"; do
  if [ "$arg" == "--yaml-out" ]; then
    YAML_ONLY=true
  fi
done
###################################### files
mapfile -t LABELS < <(awk -F'\t' '{sub(/\r$/,"")} NR==1{for(i=1;i<=NF;i++)if($i=="label"){col=i;break};next} col&&$col!=""{print $col}' "${TSV_PATH}")
files=("${LABELS[@]/#/${FASTA_DIR}/}")
files=("${files[@]/%/_dedup.fasta}")
N_FOUND=0
for f in "${files[@]}"; do [ -f "$f" ] && N_FOUND=$((N_FOUND + 1)); done
echo "[SLURM-INFO] ${N_FOUND}/${#LABELS[@]} TSV labels with fasta files in $FASTA_DIR"
if [ "${N_FOUND}" -eq 0 ]; then
  echo "[SLURM-INFO] Warning: No matching fasta files found."
  if [ "$YAML_ONLY" = true ]; then
    echo "[SLURM-INFO] Running standalone --yaml-out updates..."
    python src/extract_reps_biohub.py --yaml-out
    echo "[SLURM-INFO] YAML-out update completed successfully."
    exit 0
  fi
  echo "[SLURM-INFO] Error: No matching fasta files found."
  exit 1
fi
N_MODELS=${#MODELS[@]}
FILE_IDX=$((SLURM_ARRAY_TASK_ID / N_MODELS))
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % N_MODELS))]}"
if [ "$FILE_IDX" -ge "${#files[@]}" ]; then
  echo "[SLURM-INFO] Task ID $SLURM_ARRAY_TASK_ID exceeds available files (${#files[@]}) x models ($N_MODELS). Exiting cleanly."
  exit 0
fi
input_file="${files[$FILE_IDX]}"
[ -f "$input_file" ] || { echo "[SLURM-INFO] Error: missing $input_file"; exit 1; }
filename=$(basename "$input_file")
base_name="${filename%_dedup.fasta}"
base_name="${base_name%_seqs}"
echo "[SLURM-INFO] Target input: $input_file"
echo "[SLURM-INFO] Target model: $MODEL"
echo "[SLURM-INFO] Target reps output: ${REPS_DIR}/${MODEL}/${base_name}_reps.npz"
###################################### run
START=$(date +%s)
echo "[SLURM-INFO] Running extract_reps_biohub.py ($MODEL)"
/usr/bin/time -f "real %e s\nmaxrss %M KB" python src/extract_reps_biohub.py --fasta "$input_file" --models "$MODEL" --yaml-in
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START ))s $(free -m | awk 'NR==2 {print $3"MB"}')"