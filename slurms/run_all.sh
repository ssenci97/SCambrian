#!/bin/bash
# Submit the pipeline with array sizes derived from the configs, chained with afterok dependencies.
# Run from the repo root on the login node:  bash slurms/run_all.sh [steps...]   (default: 01 02 03 supp_01 supp_02)
# DRY_RUN=1 prints the sbatch commands without submitting.
###################################### constants-and-init
CONFIG_YAML="configs/param_configs.yaml"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(01 02 03 supp_01 supp_02)
###################################### parse-config
[ ! -f "${CONFIG_YAML}" ] && echo "[SLURM-INFO] Missing config YAML" && exit 1
eval "$(python3 -c "
import yaml
d = yaml.safe_load(open('${CONFIG_YAML}'))
models = d.get('extract_reps_biohub', {}).get('model_name', 'esmc_6b')
models = [models] if isinstance(models, str) else models
print(f'TSV_PATH={d.get(\"dw_uniprot_seqs\", {}).get(\"config_path\", \"configs/query_configs.tsv\")}')
print(f'N_MODELS={len(models)}')
")"
[ ! -f "${TSV_PATH}" ] && echo "[SLURM-INFO] Missing TSV file" && exit 1
N=$(awk -F'\t' '{sub(/\r$/,"")} NR==1{for(i=1;i<=NF;i++)if($i=="label"){col=i;break};next} col&&$col!=""{n++} END{print n+0}' "${TSV_PATH}")
[ "${N}" -eq 0 ] && echo "[SLURM-INFO] No labels in ${TSV_PATH}" && exit 1
echo "[SLURM-INFO] ${N} labels x ${N_MODELS} models"
mkdir -p logs logs/ESMC
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
    03) submit slurms/03_reps.sh --array=0-$((N * N_MODELS - 1)) ;;
    supp_01) submit slurms/supp_01_dimred.sh --array=0-${N} ;;
    supp_02) submit slurms/supp_02_scanpy.sh ;;
    *) echo "[SLURM-INFO] Unknown step: ${step}" && exit 1 ;;
  esac
done
