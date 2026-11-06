#!/bin/bash
###################################### usage: submit_step.sh <slurm script> [array index]
SCRIPT="$1"
IDX="${2:-}"
[ -f "${SCRIPT}" ] || { echo "[SLURM-INFO] Error: missing step script '${SCRIPT}'" >&2; exit 1; }
[ -n "${ANALYSIS_NAME}" ] || { echo "[SLURM-INFO] Error: ANALYSIS_NAME is not set" >&2; exit 1; }
source slurms/best_node.sh
###################################### header
header() { sed -n "s/^#SBATCH --$1=//p" "${SCRIPT}" | head -n 1; }
to_minutes() {
  awk -v t="$1" 'BEGIN{
    if (t ~ /^(infinite|UNLIMITED)$/) { print 1e9; exit }
    d = 0; if (t ~ /-/) { split(t, a, "-"); d = a[1]; t = a[2]; n = split(t, b, ":"); h = b[1]; mi = (n > 1) ? b[2] : 0; print d*1440 + h*60 + mi; exit }
    n = split(t, b, ":"); if (n == 3) print b[1]*60 + b[2]; else print b[1]
  }'
}
PART=$(header partition)
CPUS=$(header cpus-per-task)
MEM=$(header mem)
TIME=$(header time)
MEM_GB=$(awk -v m="${MEM:-1G}" 'BEGIN{v = m + 0; if (m ~ /[Mm]$/) v = int((v + 1023) / 1024); else if (m ~ /[Tt]$/) v = v * 1024; print (v < 1) ? 1 : v}')
###################################### node
PLACE=()
if [ "${PART}" != "DGX" ] && [ "${NODE_PICK:-1}" = "1" ]; then
  NEED=$(to_minutes "${TIME:-01:00:00}")
  PARTS=$(sinfo -h -o "%P %l" | while read -r p l; do p="${p%\*}"; [[ "${p^^}" =~ ^(DGX|GPU) ]] && continue;[ "$(to_minutes "${l}")" -ge "${NEED}" ] && echo "${p}"; done | sort -u | paste -sd, -)
  read -r NODE BEST_PART < <(get_best_node "${CPUS:-1}" "${MEM_GB}" "${PARTS}")
  [ -n "${NODE}" ] && PLACE=(-p "${BEST_PART}" -w "${NODE}")
fi
###################################### submit
ARGS=(--parsable --wait --export=ALL,ANALYSIS_NAME="${ANALYSIS_NAME}" "${PLACE[@]}")
[ -n "${IDX}" ] && ARGS+=(--array="${IDX}")
echo "[SLURM-INFO] sbatch ${ARGS[*]} ${SCRIPT}"
[ "${PRINT_ONLY:-0}" = "1" ] && exit 0
mkdir -p .tmp
JID_FILE=$(mktemp -p .tmp sbatch_XXXXXX)
sbatch "${ARGS[@]}" "${SCRIPT}" > "${JID_FILE}" &
PID=$!
trap 'JID=$(cut -d";" -f1 "${JID_FILE}"); [ -n "${JID}" ] && scancel "${JID}"; kill "${PID}" 2>/dev/null; rm -f "${JID_FILE}"; exit 143' TERM INT
wait "${PID}"
RC=$?
echo "[SLURM-INFO] ${SCRIPT}${IDX:+[${IDX}]} job $(cut -d";" -f1 "${JID_FILE}") exited ${RC}"
rm -f "${JID_FILE}"
exit "${RC}"
