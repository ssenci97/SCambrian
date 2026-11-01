#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=setup_ESMC
#SBATCH --output=logs/setup_ESMC.log

###################################### env
module purge
unset LD_LIBRARY_PATH
module load gcc/11.4.0 2>/dev/null || true
export CC=$(command -v gcc)
export CXX=$(command -v g++)
if [[ "$CC" == *fedora41* ]]; then
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
fi
echo "[SLURM-INFO] Using CC=$CC CXX=$CXX"

###################################### paths
[ -f configs/param_configs.yaml ] || { echo "[SLURM-INFO] Error: submit from the repo root." >&2; exit 1; }
mkdir -p logs tmp .cache
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache"
ENV_PATH="$(realpath -m "${ENV_PREFIX:-.env}")"
CPU_ONLY="${CPU_ONLY:-0}"
LOCK_DIR="env"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
if [ "${CPU_ONLY}" = "1" ]; then
  LOCK_DIR="env/cpu"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
CONDA_LOCK="${LOCK_DIR}/conda-explicit-linux-64.txt"
PIP_LOCK="${LOCK_DIR}/requirements.lock.txt"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
START_TIME=$(date +%s)
echo "[SLURM-INFO] CPU_ONLY=${CPU_ONLY} (torch from ${TORCH_INDEX}, locks in ${LOCK_DIR}/)"

###################################### conda & conditional setup
source ~/scratch/miniconda/etc/profile.d/conda.sh

if [ -d "${ENV_PATH}" ]; then
  echo "[SLURM-INFO] Existing env ${ENV_PATH} found. Nothing is installed; verification only."
  conda activate "${ENV_PATH}"
elif [ -f "${CONDA_LOCK}" ] && [ -f "${PIP_LOCK}" ]; then
  echo "[SLURM-INFO] Creating ${ENV_PATH} from ${CONDA_LOCK} and ${PIP_LOCK}..."
  conda create -y -p "${ENV_PATH}" --file "${CONDA_LOCK}" || exit 1
  conda activate "${ENV_PATH}"
  pip install --no-deps -r "${PIP_LOCK}" || exit 1
else
  echo "[SLURM-INFO] No lock files in ${LOCK_DIR}/: creating ${ENV_PATH} by resolution, then freezing it into ${LOCK_DIR}/"
  conda create -y -p "${ENV_PATH}" -c conda-forge -c bioconda python=3.10 mmseqs2 || exit 1
  conda activate "${ENV_PATH}"

  pip install --upgrade pip setuptools wheel packaging pyarrow fastparquet

  ###################################### torch
  echo "[SLURM-INFO] Installing PyTorch 2.4.1 from ${TORCH_INDEX}..."
  pip install torch==2.4.1 torchvision torchaudio --index-url "${TORCH_INDEX}"

  ###################################### esmc
  echo "[SLURM-INFO] Installing ESMC + runtime deps..."
  pip install esm httpx accelerate ninja biopython scipy pandas matplotlib huggingface_hub

  ###################################### flash-attn
  if [ "${CPU_ONLY}" = "1" ]; then
    echo "[SLURM-INFO] CPU_ONLY=1: skipping flash-attn"
  else
    echo "[SLURM-INFO] Installing precompiled flash-attn 2.6.3..."
    pip install "${FLASH_ATTN_URL}"
  fi

  ###################################### analysis (uniprot / dimred / scanpy / tests)
  echo "[SLURM-INFO] Installing analysis and test deps..."
  pip install "numpy<2.3" pyyaml tqdm seaborn scikit-learn umap-learn pynndescent "scanpy>=1.10" anndata plotly ipywidgets anywidget hdbscan pyprojroot matplotlib-venn numba safetensors igraph pymupdf kaleido ipykernel pytest || exit 1

  ###################################### freeze
  OUT_DIR="${LOCK_DIR}" TORCH_INDEX="${TORCH_INDEX}" bash env/freeze_env.sh "${ENV_PATH}" || exit 1
  echo "[SLURM-INFO] Lock files written to ${LOCK_DIR}/. Commit them: git add env/ && git commit -m 'env: lock'"
fi

###################################### verify
echo "[SLURM-INFO] Comprehensive verification..."
CPU_ONLY="${CPU_ONLY}" python -c "
import os, torch, esm, httpx, accelerate, numpy, numba, scanpy, umap, sklearn, igraph, matplotlib_venn, pymupdf, pynndescent, kaleido, hdbscan, pytest
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'HTTPX Version   : {httpx.__version__}')
print(f'Accelerate      : {accelerate.__version__}')
print(f'Flash Attention : {\"skipped (CPU_ONLY=1)\" if os.environ[\"CPU_ONLY\"] == \"1\" else __import__(\"flash_attn\").__version__}')
print(f'NumPy / Numba   : {numpy.__version__} / {numba.__version__}')
print(f'Scanpy          : {scanpy.__version__}')
print(f'UMAP / sklearn  : {umap.__version__} / {sklearn.__version__}')
print(f'pytest          : {pytest.__version__}')
" || { echo "[SLURM-INFO] Error: import verification failed." >&2; exit 1; }
echo "[SLURM-INFO] MMseqs2         : $(mmseqs version 2>/dev/null || echo 'MISSING (step 02 will fail)')"
if [ -f "${PIP_LOCK}" ]; then
  if diff -q <(grep -v '^--' "${PIP_LOCK}") <(python -m pip freeze --all --exclude-editable | sed -E "s#^flash[-_]attn( @ .*|==.*)\$#flash_attn @ ${FLASH_ATTN_URL}#") > /dev/null; then
    echo "[SLURM-INFO] pip packages match ${PIP_LOCK}"
  else
    echo "[SLURM-INFO] WARNING: pip packages differ from ${PIP_LOCK}; see: diff <(grep -v '^--' ${PIP_LOCK}) <(pip freeze --all --exclude-editable)"
  fi
fi
echo "[SLURM-INFO] All pipeline dependencies successfully verified!"
echo "[SLURM-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
