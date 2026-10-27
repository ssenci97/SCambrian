#!/bin/bash
#SBATCH --job-name=scanpy_compute
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/scanpy/scanpy_compute_%A_%a.log
#SBATCH --array=0-2

###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
[ -f configs/param_configs.yaml ] || { echo "[SLURM-INFO] Error: configs/param_configs.yaml not found in $(pwd). Submit from the repo root." >&2; exit 1; }

###################################### config
START_TIME=$(date +%s)
REPS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml'))['extract_reps_biohub']['embeddings_dir'])")
[ -n "$REPS_DIR" ] && [ "$REPS_DIR" != "/" ] && [ -d "$REPS_DIR" ] || { echo "[SLURM-INFO] Error: invalid REPS_DIR '$REPS_DIR'." >&2; exit 1; }
TSV_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml')).get('dw_uniprot_seqs', {}).get('query_table', 'configs/query_configs.tsv'))")
[ -f "$TSV_PATH" ] || { echo "[SLURM-INFO] Error: missing TSV $TSV_PATH." >&2; exit 1; }
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml')).get('npz_scanpy', {}).get('scanpy_dir', 'data/scanpy/'))")
mapfile -t MODELS < <(python3 -c "import yaml; m = yaml.safe_load(open('configs/param_configs.yaml'))['extract_reps_biohub']['models']; print('\n'.join([m] if isinstance(m, str) else m))")
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "$TASK" -lt ${#MODELS[@]} ] || { echo "[SLURM-INFO] Task $TASK exceeds ${#MODELS[@]} models. Skipping."; exit 0; }
MODEL=${MODELS[$TASK]}
MODEL_DIR="$REPS_DIR/$MODEL"
[ -d "$MODEL_DIR" ] || { echo "[SLURM-INFO] Error: missing model directory $MODEL_DIR." >&2; exit 1; }
mapfile -t LABELS < <(awk -F'\t' '{sub(/\r$/,"")} NR==1{for(i=1;i<=NF;i++)if($i=="label"){col=i;break};next} col&&$col!=""{print $col}' "$TSV_PATH")
NPZ_ARRAY=()
for label in "${LABELS[@]}"; do [ -f "$MODEL_DIR/${label}_reps.npz" ] && NPZ_ARRAY+=("$MODEL_DIR/${label}_reps.npz") || echo "[SLURM-INFO] Warning: missing $MODEL_DIR/${label}_reps.npz"; done
[ ${#NPZ_ARRAY[@]} -ge 2 ] || { echo "[SLURM-INFO] Error: at least 2 *_reps.npz files are required in $MODEL_DIR for comparative analysis." >&2; exit 1; }

###################################### run
echo "[SLURM-INFO] === MODE: COMBINED (All files jointly) | MODEL: $MODEL ==="
du -sh "${NPZ_ARRAY[@]}"
NPZ_FILES_STR=$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)
YAML_OUT=""
[ "$TASK" -eq 0 ] && YAML_OUT="--yaml-out"
echo "[SLURM-INFO] --- Running Scanpy clustering + UMAP/t-SNE ---"
python3 src/npz_scanpy.py --npzs "$NPZ_FILES_STR" --scanpy_dir "$OUT_DIR" --n_jobs "${SLURM_CPUS_PER_TASK:-1}" --yaml-in $YAML_OUT
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
