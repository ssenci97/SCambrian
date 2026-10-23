# SCambrian — Protein Sequence Embedding & Clustering Pipeline

## Overview

The pipeline starts from **any set of protein FASTA files**. It removes redundancy with MMseqs2, builds multi-identity clusters, computes ESMC-6B embeddings and residue contacts, and then runs embedding-space analyses (t-SNE/UMAP, Scanpy clustering and enrichment).

```
(optional) supp_00  UniProt fetch ─┐
user-provided FASTA files ─────────┤
                                   ▼
                       data/sequences/<label>.fasta|.fa|.faa
                                   │
00  setup_env     one conda env (.env) + YAML compilation
                                   │
01  mmseqs2       easy-linclust dedup → successive clustering (0.9 → 0.2)
                  → data/dedup/<label>_dedup.fasta, data/clusters/<label>/
                                   │
02  reps          ESMC-6B mean-pooled embeddings + residue contacts (GPU)
                  → data/esm-reps/esmc_6b/<label>_reps.npz
                  → data/esm-contacts/esmc_6b/<label>_contacts.parquet
                                   │
supp_01  dimred   t-SNE / UMAP per file and combined → data/dimred/
supp_02  scanpy   joint + intra-group Leiden, enrichment, modules → data/scanpy/
```

`<label>` is the file stem with the `_seqs` / `_dedup` / `_reps` suffixes removed. For example, `Mammalia_seqs.fasta` → `Mammalia`.

## Installation

```bash
mkdir -p logs                      # SLURM needs the log directory to exist at submission
sbatch slurms/00_setup_env.sh      # or: bash slurms/00_setup_env.sh
```

The setup script:
- creates a single conda env in `./.env` (Python 3.10) with:
  - MMseqs2 (bioconda)
  - PyTorch 2.4.1 (cu121)
  - `requirements.txt`
  - the flash-attn 2.6.3 wheel
- is idempotent: an existing env, marked by `.env/.setup_complete`, is only verified
- verifies all imports, CUDA and `mmseqs`
- compiles `configs/param_configs.yaml` by running every script with `--yaml-out`; values already in the YAML are kept and missing keys are added

These environment variables can be overridden for every slurm script:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CONDA_SH` | `~/scratch/miniconda/etc/profile.d/conda.sh` | conda init script, used if `conda` is not on PATH |
| `ENV_DIR` | `./.env` | environment prefix |
| `CONFIG_YAML` | `configs/param_configs.yaml` | parameter file |
| `DIMRED_COMPONENTS` | `2 3` | `n_components` values looped by `supp_01_dimred.sh` |

## Input sequences

Put one FASTA per dataset into `data/sequences/`. The directory is set by `run_mmseqs2.seq_in_dir`. You can provide these files yourself, fetch them from UniProt, or both:

```bash
sbatch slurms/supp_00_fetch_uniprot.sh   # one array task per row of configs/query_configs.tsv
```

`configs/query_configs.tsv`:

```
label	query	exact_species
Mammalia	(xref:interpro-IPR001766) AND (taxonomy_id:40674)	Mammalia
```

## Running

Always run from the repository root. Every step can be run in two ways.

### 1. YAML + SLURM

Edit `configs/param_configs.yaml`, then submit each step. Size the arrays to the number of inputs; a task beyond the number of inputs exits cleanly.

```bash
N=$(ls data/sequences/*.fa* | wc -l)
sbatch --array=1-$N slurms/01_mmseqs2.sh
sbatch --array=1-$N slurms/02_reps.sh
sbatch --array=0-$N slurms/supp_01_dimred.sh   # 0 = all files combined, i = file i
sbatch slurms/supp_02_scanpy.sh
```

Without SLURM (`bash slurms/01_mmseqs2.sh`), each script processes every input in turn.

### 2. Command line

Every script in `src/` accepts its parameters as CLI flags:

```bash
python src/run_mmseqs2.py --fasta my_proteins.fasta --dedup_min_seq_id 0.95
python src/extract_reps_biohub.py --fasta data/dedup/Mammalia_dedup.fasta
python src/npz_umap.py --tensor data/esm-reps/esmc_6b/Mammalia_reps.npz --n-components 3
```

Parameter precedence is **built-in defaults < YAML section < explicit CLI flags**.

| Flag | Effect |
|------|--------|
| `--yaml-in [PATH]` | read parameters from the YAML (on by default) |
| `--no-yaml-in` | ignore the YAML |
| `--yaml-out [PATH]` | write the resolved parameters back to the script's YAML section (opt-in); run-specific paths such as `--fasta` and `--out` are not stored |

Running a script with only `--yaml-out`, i.e. no inputs, writes its section and exits. YAML writes are atomic and use a file lock, so concurrent array tasks are safe.

## Parameters (`configs/param_configs.yaml`)

| Section | Key parameters |
|---------|----------------|
| `run_mmseqs2` | `seq_in_dir`, `dedup_out_dir`, `clusters_out_dir`, `dedup_min_seq_id`, `dedup_cov(_mode)`, `clust_cov(_mode)`, `thresholds`, `threads` (null = `SLURM_CPUS_PER_TASK`) |
| `extract_reps_biohub` | `model_name`, `batch_size`, `chunk_size`, `reps_outdir` |
| `extract_contacts_biohub` | `model_name`, `batch_size`, `max_tokens`, `min_sep`, `contacts_outdir` |
| `npz_tsne` / `npz_umap` | reducer hyperparameters, `seed`, `n_jobs`, `output_dir` |
| `npz_scanpy` | `resolution`, `n_neighbors`, `n_pcs` (null = full space), `min_module_size`, `min_percent_enrichment`, `seed`, `outdir` |
| `dw_uniprot_seqs` / `dw_uniprot_features` | `config_path`, output dirs, `timeout` |

## Outputs

```
data/
  sequences/<label>_seqs.{fasta,tsv}, <label>_ids.txt   (UniProt fetch only)
  features/<label>_feats.tsv                            (UniProt fetch only)
  dedup/<label>_dedup.fasta
  clusters/<label>/
    <label>_ids_id<t>_<tag>.txt                 member → representative at threshold t
    <label>_representatives_id<t>_<tag>.fasta
    <label>_cluster_metrics.tsv
    <label>_all_clusters_summary.tsv
    clustering_summary.txt
  esm-reps/esmc_6b/<label>_reps.npz             embeddings, ids
  esm-contacts/esmc_6b/<label>_contacts.parquet accession, res_i, res_j, dist, value
  dimred/<labels>_{tsne,umap}_nc<k>_..._weights.{tsv,npz}
  scanpy/<inputs>/
    protein_clusters_metadata__<tag>.csv
    species_cluster_enrichment__<tag>.csv
    channel_importance_rankings__<tag>.csv
    clustering_metrics__<tag>.txt
    umap_species_panel__<tag>.png
    intra_group/umap_intra_species_panel__<tag>.png
    protein-modules/*.txt
logs/
```

## Logging

- Python messages are prefixed with `[PYTHON-INFO]`; errors are logged as `[PYTHON-INFO] Error: ...` before being raised.
- Shell messages are prefixed with `[SLURM-INFO]`.
- Every Python script and every slurm job ends with `<date> <elapsed>s <memory>MB`.

## Layout

```
configs/   param_configs.yaml, query_configs.tsv
slurms/    common.sh, 00_setup_env.sh, 01_mmseqs2.sh, 02_reps.sh, supp_0{0,1,2}_*.sh
src/       utils.py + one script per step
requirements.txt
```

The UniProt, MMseqs2, reps and contacts steps skip outputs that already exist; dimred and scanpy overwrite theirs. To start from scratch: `rm -rf data logs tmp`.
