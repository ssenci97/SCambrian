#!/bin/bash
#SBATCH --job-name=tsne_umap_run
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tsne_umap_%A_%a.log
#SBATCH --array=0-3

source /orfeo/scratch/area/ssenci/venvs/ml_transformers/bin/activate

# Fetch reps directory and n_components list from YAML config using python
REPS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('configs/param_configs.yaml'))['extract_reps_biohub']['reps_outdir'])")

# Read n_components dynamically from YAML (adjust the config key path if necessary)
IFS=' ' read -r -a COMPONENTS <<< "$(python3 -c "import yaml; cfg = yaml.safe_load(open('configs/param_configs.yaml')); print(' '.join(map(str, cfg.get('tsne_umap', {}).get('n_components', [2, 3]))))")"

mapfile -t NPZ_ARRAY < <(find "$REPS_DIR" -name "*.npz" | sort)

if [ ${#NPZ_ARRAY[@]} -eq 0 ]; then
    echo "[SLURM-INFO] Error: No .npz files found in $REPS_DIR" >&2
    exit 1
fi

avail_mem_bytes=$(awk '/MemAvailable/ {print $2 * 1024} ' /proc/meminfo)

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
        python3 src/npz_tsne.py --tensor "$TENSORS" --output-dir data/dimred --n-components "$n" --n-jobs 8 --yaml-in $YAML_OUT
        
        echo "[SLURM-INFO] --- Running UMAP (n_components=$n) ---"
        python3 src/npz_umap.py --tensor "$TENSORS" --output-dir data/dimred --n-components "$n" --n-jobs 8 --yaml-in $YAML_OUT
    done
else
    FILE_IDX=$((SLURM_ARRAY_TASK_ID - 1))
    if [ "$FILE_IDX" -ge "${#NPZ_ARRAY[@]}" ]; then
        echo "[SLURM-INFO] Task ID exceeds available files. Exiting successfully."
        exit 0
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
        python3 src/npz_tsne.py --tensor "$CURRENT_FILE" --output-dir data/dimred --n-components "$n" --n-jobs 8 --yaml-in $YAML_OUT
        
        echo "[SLURM-INFO] --- Running UMAP (n_components=$n) ---"
        python3 src/npz_umap.py --tensor "$CURRENT_FILE" --output-dir data/dimred --n-components "$n" --n-jobs 8 --yaml-in $YAML_OUT
    done
fi