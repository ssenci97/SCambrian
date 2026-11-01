#!/bin/bash
###################################### constants-and-init
CONFIG_YAML="configs/param_configs.yaml"
CPU_ONLY="${CPU_ONLY:-0}"
CPU_MODEL="esmc_300m"
CPU_PARTITION="EPYC"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(01 02 03 supp_01 supp_02)
###################################### parse-config
[ ! -f "${CONFIG_YAML}" ] && echo "[SLURM-INFO] Missing config YAML" && exit 1
eval "$(python3 -c "
import yaml
d = yaml.safe_load(open('${CONFIG_YAML}'))
models = d.get('extract_reps_biohub', {}).get('models', 'esmc_6b')
models = [models] if isinstance(models, str) else models
models = ['${CPU_MODEL}'] if '${CPU_ONLY}' == '1' else models
print(f'TSV_PATH={d.get(\"dw_uniprot_seqs\", {}).get(\"query_table\", \"configs/query_configs.tsv\")}')
print(f'N_MODELS={len(models)}')
")"
[ ! -f "${TSV_PATH}" ] && echo "[SLURM-INFO] Missing TSV file" && exit 1
N=$(awk -F'\t' '{sub(/\r$/,"")} NR==1{for(i=1;i<=NF;i++)if($i=="label"){col=i;break};next} col&&$col!=""{n++} END{print n+0}' "${TSV_PATH}")
[ "${N}" -eq 0 ] && echo "[SLURM-INFO] No labels in ${TSV_PATH}" && exit 1
echo "[SLURM-INFO] ${N} labels x ${N_MODELS} models (CPU_ONLY=${CPU_ONLY})"
export CPU_ONLY CPU_MODEL
REPS_RES=()
[ "${CPU_ONLY}" = "1" ] && REPS_RES=(--partition="${CPU_PARTITION}" --gres=none)
mkdir -p logs/uniprot logs/mmseqs logs/ESMC logs/scanpy logs/tests
###################################### submit
PREV=""
submit() {
  local script="$1"; shift
  local dep=()
  [ -n "${PREV}" ] && dep=(--dependency=afterok:"${PREV}")
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[SLURM-INFO] sbatch --parsable ${dep[*]} $* ${script}"
    PREV="DRY_$(basename "${script}" .sh)"
    return
  fi
  local jid
  jid=$(sbatch --parsable "${dep[@]}" "$@" "${script}") || { echo "[SLURM-INFO] Submission of ${script} failed"; exit 1; }
  PREV="${jid%%;*}"
  echo "[SLURM-INFO] ${script} -> job ${PREV} $*"
}
for step in "${STEPS[@]}"; do
  case "${step}" in
    01) submit slurms/01_dw_uniprot.sh --array=1-${N} ;;
    02) submit slurms/02_mmseqs2.sh --array=1-${N}%3 ;;
    03) submit slurms/03_reps.sh "${REPS_RES[@]}" --array=0-$((N * N_MODELS - 1))%2 ;;
    supp_01) submit slurms/supp_01_scanpy.sh --array=0-$((N_MODELS - 1)) ;;
    supp_02) submit slurms/supp_02_scanpy_report.sh ;;
    *) echo "[SLURM-INFO] Unknown step: ${step}" && exit 1 ;;
  esac
done
