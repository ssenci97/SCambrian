#!/bin/bash
############### <guard>
if [ ! -f "slurms/common.sh" ]; then
  echo "[SLURM-INFO] Error: run or submit from the repository root (slurms/common.sh not found in $PWD)"
  exit 1
fi

############### <constants>
CONFIG_YAML="${CONFIG_YAML:-configs/param_configs.yaml}"
CONDA_SH="${CONDA_SH:-$HOME/scratch/miniconda/etc/profile.d/conda.sh}"
ENV_DIR="${ENV_DIR:-$PWD/.env}"
START_TIME=$(date +%s)

############### <paths>
mkdir -p logs tmp .cache .triton_cache
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache"
export TRITON_CACHE_DIR="$PWD/.triton_cache"

############### <functions>
slurm_info() {
  echo "[SLURM-INFO] $*"
}

sanitize_toolchain() {
  command -v module >/dev/null 2>&1 && module purge
  unset LD_LIBRARY_PATH
  export PATH="/usr/bin:$(echo "$PATH" | tr ':' '\n' | grep -v 'fedora41' | paste -sd:)"
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
  slurm_info "Using CC=$CC CXX=$CXX"
}

load_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    [ -f "$CONDA_SH" ] || { slurm_info "Error: conda not found; set CONDA_SH (currently $CONDA_SH)"; exit 1; }
    source "$CONDA_SH"
  fi
}

activate_env() {
  load_conda
  [ -f "$ENV_DIR/.setup_complete" ] || { slurm_info "Error: environment $ENV_DIR missing or incomplete; run slurms/00_setup_env.sh first"; exit 1; }
  conda activate "$ENV_DIR"
  slurm_info "Active Python: $(command -v python3)"
}

cfg() {
  python3 - "$CONFIG_YAML" "$1" "$2" "$3" <<'EOF'
import sys
from pathlib import Path
import yaml
path, section, key, default = sys.argv[1:5]
data = yaml.safe_load(Path(path).read_text()) if Path(path).exists() else {}
value = ((data or {}).get(section) or {}).get(key)
print(default if value is None else value)
EOF
}

task_indices() {
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "$SLURM_ARRAY_TASK_ID"
  else
    seq 1 "$1"
  fi
}

finalize() {
  slurm_info "$(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
}
