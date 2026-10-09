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

source /orfeo/scratch/area/ssenci/venvs/ml_transformers/bin/activate

# Discover input files from config and directory
REPS_DIR=$(jq -r '.reps_out_dir' configs/param_config.json)
mapfile -t NPZ_ARRAY < <(find "$REPS_DIR" -name "*.npz" | sort)

if [ ${#NPZ_ARRAY[@]} -eq 0 ]; then
    echo "Error: No .npz files found in $REPS_DIR" >&2
    exit 1
fi

avail_mem_bytes=$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo)

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "=== MODE: COMBINED (All files) ==="
    du -sh "${NPZ_ARRAY[@]}"
    
    total_size_bytes=$(du -b "${NPZ_ARRAY[@]}" | awk '{sum+=$1} END {print sum}')
    if [ "$total_size_bytes" -gt "$avail_mem_bytes" ]; then
        echo "ERROR: Total input size exceeds available system memory!" >&2
        exit 1
    fi
    
    TENSORS=$(printf "%s\n" "${NPZ_ARRAY[@]}" | paste -sd, -)
    
    echo "--- Running t-SNE ---"
    python3 src/npz_tsne.py --tensor "$TENSORS" --output-dir data/dimred --n-jobs 8
    
    echo "--- Running UMAP ---"
    python3 src/npz_umap.py --tensor "$TENSORS" --output-dir data/dimred --n-jobs 8
else
    FILE_IDX=$((SLURM_ARRAY_TASK_ID - 1))
    CURRENT_FILE="${NPZ_ARRAY[$FILE_IDX]}"
    
    echo "=== MODE: SEPARATE (File: $(basename "$CURRENT_FILE")) ==="
    du -sh "$CURRENT_FILE"
    
    file_size_bytes=$(du -b "$CURRENT_FILE" | awk '{print $1}')
    if [ "$file_size_bytes" -gt "$avail_mem_bytes" ]; then
        echo "ERROR: Input file size exceeds available system memory!" >&2
        exit 1
    fi
    
    echo "--- Running t-SNE ---"
    python3 src/npz_tsne.py --tensor "$CURRENT_FILE" --output-dir data/dimred --n-jobs 8
    
    echo "--- Running UMAP ---"
    python3 src/npz_umap.py --tensor "$CURRENT_FILE" --output-dir data/dimred --n-jobs 8
fi

