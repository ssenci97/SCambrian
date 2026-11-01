#!/bin/bash
###################################### constants-and-init
ENV_PREFIX="${1:-.env}"
CONDA_SH="${CONDA_SH:-$HOME/scratch/miniconda/etc/profile.d/conda.sh}"
OUT_DIR="${OUT_DIR:-env}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
START_TIME=$(date +%s)
[ -d "${ENV_PREFIX}" ] || { echo "[BASH-INFO] Error: env ${ENV_PREFIX} not found. Run from the repo root." >&2; exit 1; }
[ -f "${CONDA_SH}" ] || { echo "[BASH-INFO] Error: missing ${CONDA_SH}" >&2; exit 1; }
source "${CONDA_SH}"
mkdir -p "${OUT_DIR}"
###################################### conda-explicit
conda list -p "${ENV_PREFIX}" --explicit --md5 > "${OUT_DIR}/conda-explicit-linux-64.txt"
echo "[BASH-INFO] Wrote ${OUT_DIR}/conda-explicit-linux-64.txt ($(grep -c '^http' "${OUT_DIR}/conda-explicit-linux-64.txt") packages)"
###################################### pip-freeze
{
  echo "--extra-index-url ${TORCH_INDEX}"
  "${ENV_PREFIX}/bin/python" -m pip freeze --all --exclude-editable | sed -E "s#^flash[-_]attn( @ .*|==.*)\$#flash_attn @ ${FLASH_ATTN_URL}#"
} > "${OUT_DIR}/requirements.lock.txt"
echo "[BASH-INFO] Wrote ${OUT_DIR}/requirements.lock.txt ($(grep -vc '^--' "${OUT_DIR}/requirements.lock.txt") packages)"
grep -E '^(torch|torchvision|torchaudio|flash_attn|esm|numpy|scanpy)( |==)' "${OUT_DIR}/requirements.lock.txt" | sed 's/^/[BASH-INFO] pinned: /'
###################################### external-tools
{
  echo "python: $("${ENV_PREFIX}/bin/python" -c 'import sys; print(sys.version.split()[0])')"
  echo "torch_cuda: $("${ENV_PREFIX}/bin/python" -c 'import torch; print(torch.version.cuda)')"
  echo "mmseqs: $(command -v mmseqs >/dev/null && mmseqs version || echo missing)"
  echo "mmseqs_path: $(command -v mmseqs || echo missing)"
} > "${OUT_DIR}/tools.txt"
cat "${OUT_DIR}/tools.txt" | sed 's/^/[BASH-INFO] /'
###################################### finalize
echo "[BASH-INFO] $(date +'%Y-%m-%d %H:%M:%S') $(( $(date +%s) - START_TIME ))s $(free -m | awk 'NR==2 {print $3"MB"}')"
