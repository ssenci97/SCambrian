#!/bin/bash
#SBATCH --job-name=prot
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/transdecoder_%A.log

# PROCESSING (inputs in raw/, from get_and_credit_transcriptomes.sh)
#   kor_it  TransDecoder raw/tsa.GKMD.fsa_nt -> proteomes/koreicus_italian_proteome{,_filtered}.fasta
#   jap_it  TransDecoder raw/tsa.GKMC.fsa_nt -> proteomes/japonicus_italian_proteome{,_filtered}.fasta
#   kor_hu  gffread raw/nuclear_assembly.*   -> proteomes/koreicus_hungarian_proteome.fasta
# _filtered = type=complete ORFs only; all FASTA are one line per sequence.
# usage: sbatch run_transdecoder.sh [kor_it jap_it kor_hu]   (default: all)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p proteomes logs
export PATH=/orfeo/cephfs/home/area/ssenci/.conda/envs/transdecoder/bin:$PATH

python3 -u - "$@" << 'PYEOF'
import re, subprocess, sys, time
from pathlib import Path

TSA = {"kor_it": ("GKMD", "Aedes_koreicus",  "IT_GKMD01", "data/italian",   "proteomes/koreicus_italian_proteome.fasta"),
       "jap_it": ("GKMC", "Aedes_japonicus", "IT_GKMC01", "data/japonicus", "proteomes/japonicus_italian_proteome.fasta")}
HU = ("Aedes_koreicus", "HU_GCA_024533555.2", "data/hungarian", "proteomes/koreicus_hungarian_proteome.fasta")

PEP_RE = re.compile(r"^(\S+).*type:(\S+) \(([+-])\),score=([^ ,]+) len:(\d+) \S+:(\d+-\d+)\(.\)")
HDR_RE = re.compile(r"^id=\S+ organism=\S+ dataset=\S+ type=\S+ len=\d+ strand=[+-] pos=\d+-\d+ score=\S+$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(*paths):
    missing = [str(p) for p in paths if not Path(p).exists() or not Path(p).stat().st_size]
    if missing:
        raise FileNotFoundError(f"missing input {missing}: run get_and_credit_transcriptomes.sh first")


def run(cmd):
    log("RUN " + " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def read_fasta(path):
    hdr, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(seq)
                hdr, seq = line[1:], []
            elif line:
                seq.append(line)
    if hdr is not None:
        yield hdr, "".join(seq)


def report(path, check_hdr):
    n = bad = 0
    first = None
    for h, _ in read_fasta(path):
        n += 1
        first = first or h
        bad += check_hdr and not HDR_RE.match(h)
    log(f"{path}: {n} sequences" + (f", malformed headers: {bad}" if check_hdr else ""))
    log(f"  >{first}")


def transdecoder(step):
    pfx, org, ds, wd, out = TSA[step]
    fna = Path(f"raw/tsa.{pfx}.fsa_nt")
    need(fna)
    od = Path(wd, "transdecoder_out")
    pep = od / f"{fna.name}.transdecoder.pep"
    if not pep.exists():
        run(["TransDecoder", "-t", fna, "--single_best_only", "-O", od])
    if not pep.exists():  # some TransDecoder versions write final files to the cwd
        found = list(od.glob("*.transdecoder.pep")) + list(Path(".").glob(f"{fna.name}.transdecoder.pep"))
        if not found:
            raise FileNotFoundError(f"no .transdecoder.pep for {fna}")
        pep = found[0]

    out = Path(out)
    filt = out.with_name(out.stem + "_filtered.fasta")
    unparsed = 0
    with out.open("w") as fa, filt.open("w") as ff:
        for h, s in read_fasta(pep):
            m = PEP_RE.match(h)
            if m:
                h = f"id={m[1]} organism={org} dataset={ds} type={m[2]} len={m[5]} strand={m[3]} pos={m[6]} score={m[4]}"
            else:
                unparsed += 1
            fa.write(f">{h}\n{s}\n")
            if m and m[2] == "complete":
                ff.write(f">{h}\n{s}\n")
    if unparsed:
        log(f"WARNING: {unparsed} TransDecoder headers could not be parsed (kept as-is in {out})")
    report(out, True)
    report(filt, True)


def gffread():
    org, ds, wd, out = HU
    fasta, gff = Path("raw/nuclear_assembly.fasta"), Path("raw/nuclear_assembly.gff3")
    need(fasta, gff)
    Path(wd).mkdir(parents=True, exist_ok=True)
    raw_pep = Path(wd, "koreicus_hungarian_proteome.fasta")
    if not raw_pep.exists():
        run(["gffread", gff, "-g", fasta, "-y", raw_pep])
    with open(out, "w") as fo:
        for h, s in read_fasta(raw_pep):
            fo.write(f">id={h.split()[0]} organism={org} dataset={ds}\n{s}\n")
    report(out, False)


STEPS = {"kor_it": lambda: transdecoder("kor_it"), "jap_it": lambda: transdecoder("jap_it"),
         "kor_hu": gffread}

steps = sys.argv[1:] or list(STEPS)
unknown = [s for s in steps if s not in STEPS]
if unknown:
    sys.exit(f"unknown step(s): {unknown}; choose from {list(STEPS)}")

failed = []
for s in steps:
    log(f"===== {s} =====")
    t0 = time.time()
    try:
        STEPS[s]()
        log(f"===== {s} done in {time.time() - t0:.0f} s =====")
    except Exception as e:
        log(f"===== {s} FAILED: {e!r} =====")
        failed.append(s)
if failed:
    sys.exit(f"failed steps: {' '.join(failed)}")
log("all processing done")
PYEOF

MAXRSS=$(sstat -j ${SLURM_JOB_ID:-0}.batch --format=MaxRSS --noheader 2>/dev/null | tr -d ' ' || true)
echo "[SLURM-INFO] job=${SLURM_JOB_ID:-NA} max_mem=${MAXRSS:-NA} elapsed=$(date -u -d @${SECONDS} +%H:%M:%S)"

