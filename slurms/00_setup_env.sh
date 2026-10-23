#!/bin/bash
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=setup_env
#SBATCH --output=logs/setup_env_%j.log
############### <init>
source slurms/common.sh
sanitize_toolchain
load_conda
TORCH_VERSION="2.4.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

############### <install>
if [ -f "$ENV_DIR/.setup_complete" ]; then
  slurm_info "Existing environment found at $ENV_DIR, skipping installation"
else
  set -e
  [ -d "$ENV_DIR" ] && slurm_info "Removing incomplete environment $ENV_DIR" && rm -rf "$ENV_DIR"
  slurm_info "Creating Python 3.10 environment at $ENV_DIR"
  conda create -y -p "$ENV_DIR" python=3.10
  conda activate "$ENV_DIR"
  slurm_info "Installing MMseqs2 (bioconda)"
  conda install -y -p "$ENV_DIR" -c conda-forge -c bioconda mmseqs2
  pip install --upgrade pip setuptools wheel packaging
  slurm_info "Installing PyTorch $TORCH_VERSION (cu121)"
  pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  slurm_info "Installing requirements.txt"
  pip install -r requirements.txt
  slurm_info "Installing precompiled flash-attn 2.6.3"
  pip install "$FLASH_ATTN_WHEEL"
  touch "$ENV_DIR/.setup_complete"
  set +e
fi
activate_env

############### <verify>
slurm_info "Verifying environment"
python3 -c "
import torch, esm, httpx, accelerate, flash_attn, scanpy, anndata, umap, sklearn, igraph, pyarrow, yaml
print(f'[PYTHON-INFO] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'[PYTHON-INFO] flash-attn {flash_attn.__version__} | scanpy {scanpy.__version__} | umap {umap.__version__} | sklearn {sklearn.__version__}')
" || { slurm_info "Error: Python dependency verification failed"; exit 1; }
mmseqs version >/dev/null 2>&1 || { slurm_info "Error: mmseqs not available in $ENV_DIR"; exit 1; }
slurm_info "MMseqs2 version: $(mmseqs version)"

############### <compile-yaml>
slurm_info "Compiling $CONFIG_YAML"
for script in run_mmseqs2 extract_reps_biohub extract_contacts_biohub npz_tsne npz_umap npz_scanpy dw_uniprot_seqs dw_uniprot_features; do
  python3 "src/${script}.py" --yaml-in "$CONFIG_YAML" --yaml-out "$CONFIG_YAML" || { slurm_info "Error: YAML compilation failed for $script"; exit 1; }
done
finalize
