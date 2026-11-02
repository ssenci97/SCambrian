#!/bin/bash
###################################### resolve
CONFIG_YAML="${CONFIG_YAML:-configs/param_configs.yaml}"
[ -f "${CONFIG_YAML}" ] || { echo "[SLURM-INFO] Error: ${CONFIG_YAML} not found in $(pwd). Submit from the repo root." >&2; exit 1; }
eval "$(python3 -c "
import yaml
p = (yaml.safe_load(open('${CONFIG_YAML}')) or {}).get('preprocess_sequences', {})
print(f'ANALYSES_DIR={p.get(\"analyses_dir\") or \"analyses\"}')
print(f'YAML_ANALYSIS_NAME={p.get(\"analysis_name\") or \"\"}')
")"
ANALYSES_DIR="${ANALYSES_DIR%/}"
[ -n "${ANALYSES_DIR}" ] && [ "${ANALYSES_DIR}" != "/" ] || { echo "[SLURM-INFO] Error: invalid analyses_dir '${ANALYSES_DIR}'." >&2; exit 1; }
ANALYSIS_NAME="${ANALYSIS_NAME:-${YAML_ANALYSIS_NAME}}"
if [ -z "${ANALYSIS_NAME}" ] && [ "${NEW_ANALYSIS:-0}" = "1" ]; then
  ANALYSIS_NAME="$(date +%F)_SCambrian"
elif [ -z "${ANALYSIS_NAME}" ] && [ -d "${ANALYSES_DIR}" ]; then
  ANALYSIS_NAME=$(find "${ANALYSES_DIR}" -mindepth 1 -maxdepth 1 -type d -name '*_SCambrian' -printf '%f\n' | sort | tail -n 1)
fi
[ -n "${ANALYSIS_NAME}" ] || { echo "[SLURM-INFO] Error: no analysis name; run step 02 first or export ANALYSIS_NAME." >&2; exit 1; }
export ANALYSIS_NAME
ANALYSIS_DIR="${ANALYSES_DIR}/${ANALYSIS_NAME}"
echo "[SLURM-INFO] Analysis ${ANALYSIS_NAME} -> ${ANALYSIS_DIR}"
