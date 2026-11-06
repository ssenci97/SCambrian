#!/bin/bash
###################################### usage: bash slurms/list_relics.sh   (read-only; prints old-layout data, deletes nothing)
CONFIG_YAML="configs/param_configs.yaml"
[ -f "${CONFIG_YAML}" ] || { echo "[BASH-INFO] Error: ${CONFIG_YAML} not found in $(pwd). Run from the repo root." >&2; exit 1; }
eval "$(python3 -c "
import yaml
d = yaml.safe_load(open('${CONFIG_YAML}')) or {}
m = d.get('run_mmseqs2', {}); p = d.get('preprocess_sequences', {})
print(f'CLU_DIR={m.get(\"clusters_dir\", \"data/clusters\")}')
print(f'ANALYSES_DIR={p.get(\"analyses_dir\") or \"analyses\"}')
print(f'FEAT_DIR={d.get(\"dw_uniprot_features\", {}).get(\"features_dir\", \"data/features\")}')
print(f'TAG=ccov{m.get(\"cluster_min_coverage\", 0.8)}_cmode{m.get(\"cluster_coverage_mode\", 0)}')
print('THRESH=(' + ' '.join(str(t).replace('.', '_') for t in m.get('cluster_identity_thresholds', [])) + ')')
")"
###################################### old folders
for d in data/dedup data/scanpy; do [ -e "${d}" ] && echo "[RELIC] ${d}/ (old layout)"; done
[ -d "${FEAT_DIR}" ] && find "${FEAT_DIR}" -maxdepth 1 -name '*_focus_ids.*' -printf '[RELIC] %p (old focus list)\n'
[ -d "${ANALYSES_DIR}" ] && find "${ANALYSES_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name '*_SCambrian' -printf '[RELIC] %p/ (not an {analysis}_SCambrian folder)\n'
###################################### cluster files of other tags
[ -d "${CLU_DIR}" ] || exit 0
find "${CLU_DIR}" -mindepth 2 -maxdepth 2 -type f \( -name '*_ids_id*' -o -name '*_representatives_id*' -o -name '*dmin*' \) | while read -r f; do
  keep=0
  for t in "${THRESH[@]}"; do [[ "$(basename "${f}")" == *"_id${t}_${TAG}."* ]] && keep=1; done
  [ "${keep}" = "0" ] && echo "[RELIC] ${f} (tag or threshold not in the current config)"
done
