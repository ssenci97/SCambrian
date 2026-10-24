#!/bin/bash
#SBATCH --job-name=tsne_umap_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dimred/tsne_umap_%A_%a.log
#SBATCH --array=0-3

###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env

# Fetch reps base directory (one subdirectory per model) and dimred output directory from YAML config
REPS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml'))['extract_reps_biohub']['reps_outdir'])")
OUT_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml')).get('npz_tsne', {}).get('output_dir', 'data/dimred'))")

# Read n_components dynamically from YAML (adjust the config key path if necessary)
IFS=' ' read -r -a COMPONENTS <<< "$(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/param_configs.yaml')); print(' '.join(map(str, cfg.get('tsne_umap', {}).get('n_components', [2, 3]))))")"

avail_mem_bytes=$(awk '/MemAvailable/ {print $2 * 1024} ' /proc/meminfo)
N_MODELS=0

for MODEL_DIR in "$REPS_DIR"/*/; do
    [ -d "$MODEL_DIR" ] || continue
    MODEL=$(basename "$MODEL_DIR")
    mapfile -t NPZ_ARRAY < <(find "$MODEL_DIR" -name "*_reps.npz" | sort)
    [ ${#NPZ_ARRAY[@]} -eq 0 ] && continue
    N_MODELS=$((N_MODELS + 1))
    echo "[SLURM-INFO] ##### MODEL: $MODEL (${#NPZ_ARRAY[@]} files) #####"

    if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
        echo "[SLURM-INFO] === MODE: COMBINED (All files) ==="
        du -sh "${NPZ_ARRAY[@]}"

        total_size_bytes=$(du -b "${NPZ_ARRAY[@]}" | awk '{sum+=$1} END {print sum}')
        if [ "$total_size_bytes" -gt "$avail_mem_bytes" ]; then
            echo "[SLURM-INFO] ERROR: Total input size exceeds available system memory!" >&2
            exit 1
        fi

        TENSORS=$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)

        # Loop over components dynamically (--yaml-out included only for the first component)
        for i in "${!COMPONENTS[@]}"; do
            n="${COMPONENTS[$i]}"
            if [ "$i" -eq 0 ]; then
                YAML_OUT="--yaml-out"
            else
                YAML_OUT=""
            fi

            echo "[SLURM-INFO] --- Running t-SNE (n_components=$n) ---"
            python3 src/npz_tsne.py --tensor "$TENSORS" --output_dir "$OUT_DIR/$MODEL" --n_components "$n" --n_jobs 8 --yaml-in $YAML_OUT

            echo "[SLURM-INFO] --- Running UMAP (n_components=$n) ---"
            python3 src/npz_umap.py --tensor "$TENSORS" --output_dir "$OUT_DIR/$MODEL" --n_components "$n" --n_jobs 8 --yaml-in $YAML_OUT
        done
    else
        FILE_IDX=$((SLURM_ARRAY_TASK_ID - 1))
        if [ "$FILE_IDX" -ge "${#NPZ_ARRAY[@]}" ]; then
            echo "[SLURM-INFO] Task ID exceeds available files for $MODEL. Skipping."
            continue
        fi
        CURRENT_FILE="${NPZ_ARRAY[$FILE_IDX]}"

        echo "[SLURM-INFO] === MODE: SEPARATE (File: $(basename "$CURRENT_FILE")) ==="
        du -sh "$CURRENT_FILE"

        file_size_bytes=$(du -b "$CURRENT_FILE" | awk '{print $1}')
        if [ "$file_size_bytes" -gt "$avail_mem_bytes" ]; then
            echo "[SLURM-INFO] ERROR: Input file size exceeds available system memory!" >&2
            exit 1
        fi

        # Loop over components dynamically (--yaml-out included only for the first component)
        for i in "${!COMPONENTS[@]}"; do
            n="${COMPONENTS[$i]}"
            if [ "$i" -eq 0 ]; then
                YAML_OUT="--yaml-out"
            else
                YAML_OUT=""
            fi

            echo "[SLURM-INFO] --- Running t-SNE (n_components=$n) ---"
            python3 src/npz_tsne.py --tensor "$CURRENT_FILE" --output_dir "$OUT_DIR/$MODEL" --n_components "$n" --n_jobs 8 --yaml-in $YAML_OUT

            echo "[SLURM-INFO] --- Running UMAP (n_components=$n) ---"
            python3 src/npz_umap.py --tensor "$CURRENT_FILE" --output_dir "$OUT_DIR/$MODEL" --n_components "$n" --n_jobs 8 --yaml-in $YAML_OUT
        done
    fi
done

if [ "$N_MODELS" -eq 0 ]; then
    echo "[SLURM-INFO] Error: No *_reps.npz files found in $REPS_DIR/<model>/" >&2
    exit 1
fi
