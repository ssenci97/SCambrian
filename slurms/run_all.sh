#!/bin/bash
###################################### constants-and-init
CONFIG_YAML="configs/param_configs.yaml"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(01 02 03 04 05 06)
###################################### parse-config
[ ! -f "${CONFIG_YAML}" ] && echo "[SLURM-INFO] Missing config YAML" && exit 1
eval "$(python3 -c "
import yaml
d = yaml.safe_load(open('${CONFIG_YAML}')) or {}
models = d.get('extract_reps_biohub', {}).get('models', ['esmc_300m'])
models = [models] if isinstance(models, str) else models
print(f'TSV_PATH={d.get(\"dw_uniprot_seqs\", {}).get(\"query_table\", \"configs/query_configs.tsv\")}')
print(f'N_MODELS={len(models)}')
")"
[ ! -f "${TSV_PATH}" ] && echo "[SLURM-INFO] Missing TSV file" && exit 1
read -r N N_TARGETS < <(awk -F'\t' '{sub(/\r$/,"")} NR==1{for(i=1;i<=NF;i++){if($i=="label")l=i; if($i=="input_type")t=i};next} l&&$l!=""{n++; if(!t||$t!="probe")nt++} END{print n+0, nt+0}' "${TSV_PATH}")
[ "${N}" -eq 0 ] && echo "[SLURM-INFO] No samples in ${TSV_PATH}" && exit 1
[ "${N_TARGETS}" -eq 0 ] && echo "[SLURM-INFO] No target sample in ${TSV_PATH}" && exit 1
[[ " ${STEPS[*]} " == *" 02 "* ]] && NEW_ANALYSIS=1
NEW_ANALYSIS="${NEW_ANALYSIS:-0}" source slurms/analysis_env.sh
echo "[SLURM-INFO] ${N} samples (${N_TARGETS} targets) x ${N_MODELS} models"
mkdir -p logs/uniprot logs/preprocess logs/mmseqs logs/ESMC logs/scanpy
###################################### submit
PREV=""
submit() {
  local script="$1"; shift
  local dep=()
  [ -n "${PREV}" ] && dep=(--dependency=afterok:"${PREV}")
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[SLURM-INFO] sbatch --parsable --export=ALL,ANALYSIS_NAME=${ANALYSIS_NAME} ${dep[*]} $* ${script}"
    PREV="DRY_$(basename "${script}" .sh)"
    return
  fi
  local jid
  jid=$(sbatch --parsable --export=ALL,ANALYSIS_NAME="${ANALYSIS_NAME}" "${dep[@]}" "$@" "${script}") || { echo "[SLURM-INFO] Submission of ${script} failed"; exit 1; }
  PREV="${jid%%;*}"
  echo "[SLURM-INFO] ${script} -> job ${PREV} $*"
}
for step in "${STEPS[@]}"; do
  case "${step}" in
    01) submit slurms/01_dw_uniprot.sh --array=1-${N} ;;
    02) submit slurms/02_preprocess.sh ;;
    03) submit slurms/03_mmseqs2.sh --array=1-$((N_TARGETS + 1))%3 ;;
    04) submit slurms/04_reps.sh --array=0-$((N * N_MODELS - 1))%2 ;;
    05) submit slurms/05_project.sh --array=0-$((N_MODELS - 1)) ;;
    06) submit slurms/06_summary.sh ;;
    *) echo "[SLURM-INFO] Unknown step: ${step} (use 01 02 03 04 05 06)" && exit 1 ;;
  esac
done
