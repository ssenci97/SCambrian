#!/bin/bash
#SBATCH --job-name=get_kor_jap
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/get_kor_jap_%A.log

# DOWNLOAD + CREDITS (no processing)
#   kor_it  Ae. koreicus,  Italy   : NCBI TSA GKMD01                -> raw/tsa.GKMD.fsa_nt
#   jap_it  Ae. japonicus, Italy   : NCBI TSA GKMC01                -> raw/tsa.GKMC.fsa_nt
#   kor_hu  Ae. koreicus,  Hungary : Zenodo 10278300 assembly + gff -> raw/nuclear_assembly.{fasta,gff3}
#   + ./CITATION.bib and ./ACKNOWLEDGEMENTS.txt
# usage: sbatch get_and_credit_transcriptomes.sh [kor_it jap_it kor_hu]   (default: all)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p raw logs
export PATH=/orfeo/cephfs/home/area/ssenci/.conda/envs/transdecoder/bin:$PATH

python3 -u - "$@" << 'PYEOF'
import gzip, shutil, sys, time, urllib.error, urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TSA_FTP = "https://ftp.ncbi.nlm.nih.gov/genbank/tsa"
ZENODO = "https://zenodo.org/api/records/10278300/files"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url, dest, magic=None, retries=4, timeout=60):
    """Stream url -> dest (via .part) with timeout, retries, progress and content check; skip valid files."""
    dest = Path(dest)
    def ok(p):
        head = p.open("rb").read(64)
        return bool(head) and not head.lstrip().lower().startswith((b"<!doctype", b"<html")) \
            and (magic is None or head.startswith(magic))
    if dest.exists() and ok(dest):
        log(f"present: {dest}")
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            log(f"GET {url} (attempt {attempt}/{retries})")
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r, part.open("wb") as f:
                total, done, nxt = int(r.headers.get("Content-Length") or 0), 0, 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if done >= nxt:
                        log(f"  {done / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else ""))
                        nxt += 200 << 20
            if not ok(part):
                raise IOError("unexpected content (HTML page or wrong format)")
            part.replace(dest)
            log(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except (OSError, TimeoutError) as e:
            err = e
        log(f"  failed: {err}")
        time.sleep(15 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url}")


def tsa(pfx):
    """all chunks tsa.<pfx>.<i>.fsa_nt.gz (i = 1, 2, ... until 404) -> raw/tsa.<pfx>.fsa_nt"""
    fna = Path(f"raw/tsa.{pfx}.fsa_nt")
    if fna.exists() and fna.stat().st_size:
        log(f"present: {fna}")
        return
    gzs = []
    for i in range(1, 100):
        try:
            gzs.append(download(f"{TSA_FTP}/{pfx[0]}/tsa.{pfx}.{i}.fsa_nt.gz", f"raw/tsa.{pfx}.{i}.fsa_nt.gz", b"\x1f\x8b"))
        except urllib.error.HTTPError as e:
            if e.code == 404 and gzs:
                break
            raise
    part = fna.with_name(fna.name + ".part")
    with part.open("wb") as out:
        for g in gzs:
            with gzip.open(g, "rb") as src:
                shutil.copyfileobj(src, out, 1 << 20)
    part.replace(fna)
    log(f"{fna}: {len(gzs)} chunk(s)")


def hungarian():
    download(f"{ZENODO}/nuclear_assembly.fasta/content", "raw/nuclear_assembly.fasta", b">")
    download(f"{ZENODO}/nuclear_assembly.gff3/content", "raw/nuclear_assembly.gff3")


BIB = r"""@article{Catapano2023,
  author  = {Catapano, Paolo L. and Falcinelli, Monica and Damiani, Claudia and Cappelli, Alessia and Koukouli, Despoina and Rossi, Paolo and Ricci, Irene and Napolioni, Valerio and Favia, Guido},
  title   = {{De novo genome assembly of the invasive mosquito species \textit{Aedes japonicus} and \textit{Aedes koreicus}}},
  journal = {Parasites \& Vectors},
  year    = {2023},
  volume  = {16},
  number  = {1},
  pages   = {427},
  doi     = {10.1186/s13071-023-06048-w},
  pmid    = {37986088},
  pmcid   = {PMC10658958}
}

@article{Nagy2024,
  author  = {Nagy, Nikoletta Andrea and T{\'o}th, G{\'a}bor Endre and Kurucz, Korn{\'e}lia and Kemenesi, G{\'a}bor and Laczk{\'o}, Levente},
  title   = {{The updated genome of the Hungarian population of \textit{Aedes koreicus}}},
  journal = {Scientific Reports},
  year    = {2024},
  volume  = {14},
  pages   = {7545},
  doi     = {10.1038/s41598-024-58096-6}
}
"""

ACK = """DATA SOURCES
[1] Catapano PL, Falcinelli M, Damiani C, Cappelli A, Koukouli D, Rossi P, Ricci I, Napolioni V, Favia G.
    De novo genome assembly of the invasive mosquito species Aedes japonicus and Aedes koreicus.
    Parasites & Vectors. 2023;16(1):427. doi:10.1186/s13071-023-06048-w. PMID: 37986088; PMCID: PMC10658958.
[2] Nagy NA, Toth GE, Kurucz K, Kemenesi G, Laczko L.
    The updated genome of the Hungarian population of Aedes koreicus.
    Scientific Reports. 2024;14:7545. doi:10.1038/s41598-024-58096-6.
(BibTeX entries: CITATION.bib)

DATA AVAILABILITY
  Ae. koreicus  (Italy)   : NCBI TSA GKMD01 [1]
  Ae. japonicus (Italy)   : NCBI TSA GKMC01, isolate Unicam-2023b, BioProject PRJNA947978 [1]
  Ae. koreicus  (Hungary) : genome assembly and annotation, Zenodo record 10278300 (https://zenodo.org/records/10278300) [2]

ACKNOWLEDGEMENTS
We thank Paolo L. Catapano, Guido Favia and co-authors (School of Biosciences and Veterinary Medicine,
University of Camerino) and Nikoletta A. Nagy, Levente Laczko and co-authors for generating these
data and making them publicly available.
"""

STEPS = {"kor_it": lambda: tsa("GKMD"), "jap_it": lambda: tsa("GKMC"), "kor_hu": hungarian}

steps = sys.argv[1:] or list(STEPS)
unknown = [s for s in steps if s not in STEPS]
if unknown:
    sys.exit(f"unknown step(s): {unknown}; choose from {list(STEPS)}")

Path("CITATION.bib").write_text(BIB)
Path("ACKNOWLEDGEMENTS.txt").write_text(ACK)
log("wrote CITATION.bib, ACKNOWLEDGEMENTS.txt")

failed = []
for s in steps:
    log(f"===== {s} =====")
    try:
        STEPS[s]()
    except Exception as e:
        log(f"===== {s} FAILED: {e!r} =====")
        failed.append(s)
if failed:
    sys.exit(f"failed downloads: {' '.join(failed)}")
log("all downloads done")
PYEOF

MAXRSS=$(sstat -j ${SLURM_JOB_ID:-0}.batch --format=MaxRSS --noheader 2>/dev/null | tr -d ' ' || true)
echo "[SLURM-INFO] job=${SLURM_JOB_ID:-NA} max_mem=${MAXRSS:-NA} elapsed=$(date -u -d @${SECONDS} +%H:%M:%S)"

