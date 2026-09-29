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
# Prefer a GLIBC-compatible compiler. Adjust the module name to whatever
# your cluster provides (gcc/11, gcc/12, system-gcc, …).
# If no suitable module exists, force the system compiler.
module load gcc/11.4.0 2>/dev/null || true
export CC=$(command -v gcc)
export CXX=$(command -v g++)
# Sanity: never use the Fedora-41 binary
if [[ "$CC" == *fedora41* ]]; then
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
fi
echo "[SLURM-INFO] Using CC=$CC CXX=$CXX"
###################################### paths
mkdir -p logs tmp .cache
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache"
###################################### conda
source ~/scratch/miniconda/etc/profile.d/conda.sh
if [ -d ".env" ]; then
  echo "[SLURM-INFO] Removing previous .env..."
  rm -rf .env
fi
echo "[SLURM-INFO] Creating Python 3.10 env..."
conda create -y -p .env python=3.10
conda activate ./.env
pip install --upgrade pip setuptools wheel packaging pyarrow fastparquet
###################################### torch
echo "[SLURM-INFO] Installing PyTorch 2.4.1 (cu121)..."
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
###################################### esmc
echo "[SLURM-INFO] Installing ESMC + runtime deps..."
pip install esm httpx accelerate ninja biopython scipy pandas matplotlib huggingface_hub
###################################### flash-attn
echo "[SLURM-INFO] Installing precompiled flash-attn 2.6.3..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
###################################### verify
echo "[SLURM-INFO] Comprehensive verification..."
python -c "
import torch, esm, httpx, accelerate, flash_attn
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'HTTPX Version   : {httpx.__version__}')
print(f'Accelerate      : {accelerate.__version__}')
print(f'Flash Attention : {flash_attn.__version__}')
print('All ESMC core dependencies successfully verified!')
"
echo "[SLURM-INFO] Setup finished $(date)"