# SCambrian usage guide

From a clean clone on ORFEO to a finished run. Run every command from the repository root.

## 1. Prerequisites

- Access to ORFEO with the SLURM partitions used by the scripts:
  - `DGX` (A100, for setup and step 03);
  - `THIN` (steps 01 and 02);
  - `EPYC` (supp_01, supp_02 and the tests).
- Miniconda at `~/scratch/miniconda`. Every slurm sources `~/scratch/miniconda/etc/profile.d/conda.sh`; if yours lives elsewhere, change that line in `esmc_config.sh` and `slurms/*.sh`.
- Network access from the compute nodes, for UniProt (step 01) and the Hugging Face model downloads (step 03 and the GPU test).

## 2. Install

```bash
git clone https://github.com/ssenci97/SCambrian.git
cd SCambrian
sbatch esmc_config.sh
tail -f logs/setup_ESMC.log
```

`esmc_config.sh` creates the conda env `./.env`: Python 3.10, PyTorch 2.4.1 (cu121), flash-attn 2.6.3, ESM, Scanpy, MMseqs2 and pytest. It works in one of three modes:

| situation | what it does |
|---|---|
| `./.env` exists | installs nothing; only verifies imports, MMseqs2, and that pip packages match `env/requirements.lock.txt` |
| no env, lock files in `env/` | creates the env exactly from `env/conda-explicit-linux-64.txt` + `env/requirements.lock.txt` |
| no env, no lock files | resolves and installs the packages, then writes the two lock files with `env/freeze_env.sh` |

The setup succeeded when the log ends with `All pipeline dependencies successfully verified!` and prints an MMseqs2 version.

**First install only.** In the third mode, commit the lock files so that every later install is identical:

```bash
git add env/conda-explicit-linux-64.txt env/requirements.lock.txt env/tools.txt
git commit -m "env: lock"
git push
```

To use the env in an interactive shell:

```bash
source ~/scratch/miniconda/etc/profile.d/conda.sh
conda activate ./.env
```

Never `pip install` into `./.env` by hand. To add a dependency:
1. Add it to `esmc_config.sh`.
2. Build a second env: `ENV_PREFIX=.env-new sbatch esmc_config.sh`. With the lock present, that would only rebuild the old env, so first move the lock files aside.
3. Compare the new lock files with the old ones (`git diff env/`).
4. Commit them.

## 3. Test

The tests use synthetic data in a temporary folder. They never read `data/` and never change `configs/`.

| command | where | checks | time |
|---|---|---|---|
| `python -m pytest -m static` | login node, env active | scripts compile, shell syntax, `--help` of every script, query table schema, config keys vs script options, `run_all.sh` array sizes | < 1 min |
| `sbatch slurms/tests.sh` | EPYC | `static` plus `cpu`: MMseqs2 dedup and clustering, Scanpy clustering and layouts, rerun skipping, seed determinism, PDF report, HTML viewer | a few min |
| `sbatch --partition=DGX --gres=gpu:A100:1 slurms/tests.sh gpu` | DGX | esmc_300m embeddings: shape (5, 960), ids, two runs agree | a few min (first run downloads esmc_300m) |

Results are in `logs/tests/tests_<jobid>.log`. A pass ends with `All selected tests passed`. pytest prints a short reason for every skipped test; for example, the GPU test is skipped on a node without CUDA.

Run the tests after installing, after changing the env, and before a long run.

## 4. Configure

### Inputs: `configs/query_configs.tsv`

One tab-separated row per input set. Comments and blank lines are not allowed.

| column | value |
|---|---|
| `label` | short unique name used in every file name, e.g. `aegypti` |
| `query` | a UniProtKB query, e.g. `(organism_id:7159)`, or a local FASTA path relative to the repo root, e.g. `raw/proteomes/koreicus_seqs.fasta` |
| `exact_species` | display name in figures, e.g. `Ae. aegypti` |
| `input_type` | `target` (compared set) or `probe` (functional reference, clustered but not counted in the species statistics) |

To add a species, add a row. `run_all.sh` sizes the SLURM arrays from the number of rows. At least two targets are needed.

### Parameters: `configs/param_configs.yaml`

One section per script under `src/`; each key is the script's command-line option of the same name. The main ones:
- `extract_reps_biohub.models`: which ESMC models to run (`esmc_300m`, `esmc_600m`, `esmc_6b`);
- `npz_scanpy`: `leiden_resolution`, `knn_neighbors`, `seed`;
- `h5ad_scanpy`: the cluster selection thresholds.

The README section "Configuration" describes every key. `python -m pytest -m static` checks that every key is a real option.

## 5. Run

From the repo root, on the login node (not with `sbatch`):

```bash
DRY_RUN=1 bash slurms/run_all.sh     # print the sbatch commands only
bash slurms/run_all.sh               # submit all steps, chained with afterok
bash slurms/run_all.sh 03 supp_01    # submit only some steps
```

| step | script | partition | output |
|---|---|---|---|
| 01 | `slurms/01_dw_uniprot.sh` | THIN | `data/sequences/`, `data/features/` |
| 02 | `slurms/02_mmseqs2.sh` | THIN | `data/dedup/`, `data/clusters/` |
| 03 | `slurms/03_reps.sh` | DGX | `data/esm-reps/<model>/<label>_reps.npz` |
| supp_01 | `slurms/supp_01_scanpy.sh` | EPYC | `data/scanpy/<labels>__<model>/*.h5ad` |
| supp_02 | `slurms/supp_02_scanpy_report.sh` | EPYC | `data/scanpy/<labels>__<model>/` PDF report, tables, `interactive/*.html` |

To follow the jobs:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
ls logs/uniprot logs/mmseqs logs/ESMC logs/scanpy
```

A step that fails cancels the dependent steps (`afterok`). Fix the cause and resubmit from that step, e.g. `bash slurms/run_all.sh 03 supp_01 supp_02`.

Each step can also be submitted alone with `sbatch slurms/<step>.sh`. In that case the `#SBATCH --array` line in the script header is used, sized for 5 labels × 3 models.

## 6. Rerun

Each step skips outputs that are already up to date and rebuilds missing or stale ones, so after a change you can simply rerun `bash slurms/run_all.sh`. To force a step, set `overwrite: true` in its config section.

Known limits, being fixed by the reorganization: the report (supp_02) is not rebuilt when only its own thresholds change, so set `h5ad_scanpy.overwrite: true` after changing them. Outputs built with older parameters are not deleted.

## 7. Explore the results

- PDF report: `data/scanpy/<labels>__<model>/<labels>__<model>_scanpy.pdf`.
- Interactive 3D viewer: `data/scanpy/<labels>__<model>/interactive/views_3d__*.html`. Copy it to your laptop and open it in a browser.
- Notebooks in `nbs/` load each stage into memory. Open them in Jupyter or VS Code with `./.env` as the kernel (`ipykernel` is installed).

## 8. Earlier results

Outputs produced before the reorganization were removed from the repository on 2026-11-01. They remain in git history under the tag `pre-reorg`:

```bash
git fetch --tags
git worktree add ../SCambrian-pre-reorg pre-reorg
```

`reorg/baseline.sha256` lists their checksums.

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| job ends at once with no log | submitted outside the repo root, or the log folder is missing: `cd` to the repo root (the `logs/` folders are part of the repo) |
| `configs/param_configs.yaml not found` | submitted outside the repo root |
| `MMseqs2 executable 'mmseqs' not found` | env not activated or created before MMseqs2 was added: rebuild it (`mv .env .env-old && sbatch esmc_config.sh`) |
| step 03 hangs or fails downloading a model | the node has no internet: download once on the login node with `python -c "from huggingface_hub import snapshot_download; snapshot_download('EvolutionaryScale/esmc-300m-2024-12')"` (same for `EvolutionaryScale/esmc-600m-2024-12` and `biohub/ESMC-6B`) |
| `WARNING: pip packages differ from env/requirements.lock.txt` | someone installed into `./.env` by hand: rebuild it from the lock |
| `focus capture below min_capture_percent` (supp_02 exit code 1) | too few probe or annotated proteins in the focus clusters; the viewers are still written. Check the focus lists or lower `h5ad_scanpy.min_capture_percent` |
| a test fails | the log shows the failing command and the tail of its output; rerun locally with `python -m pytest -x -k <name>` |

## 10. Startup guide PDF

`docs/STARTUP.md` is the short startup guide. Build the PDF (git-ignored) with:

```bash
mkdir -p tmp && pandoc docs/STARTUP.md -s -o tmp/STARTUP.tex -V geometry:margin=2cm -V fontsize=10pt -V colorlinks && tectonic tmp/STARTUP.tex --outdir docs
```
