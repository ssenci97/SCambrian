# Protein Sequence Embedding & Clustering Pipeline
Area Science Park -- Laboratory of Data Engineering

## Overview

Pipeline for downloading compartment-specific UniProt sets, reducing redundancy, building multi-identity sequence clusters, computing ESMC embeddings, and performing embedding-space clustering + enrichment analysis.

```
configs/param_configs.yaml
configs/query_config.tsv
        │
        ▼
01  UniProt download:
    sequences + features per query/label
    → data/raw_seqs/   data/features/
        │
        ▼
02  MMseqs2 deduplication (easy-linclust):
    min-seq-id / coverage from config
    → data/mmseqs_clusters/dedup/*_dedup.fasta
        │
        ▼
03  MMseqs2 multi-threshold clustering:
    successive clustering on previous representatives
    thresholds 0.95 → 0.15 (config)
    → data/mmseqs_clusters/<label>/
        │
        ▼
04  ESMC representation extraction (GPU):
    mean-pooled embeddings from deduplicated FASTA
    → data/reps/<model>/<label>_reps.npz
        │
        ▼
05  Scanpy analysis:
    L2-norm → (optional PCA) → neighbors → Leiden
    UMAP, channel ranking, species enrichment, modules
    → data/scanpy/<model>/
```

## Requirements

- SLURM cluster
- Python 3 + packages: pandas, tqdm, torch, esm, anndata, scanpy, seaborn, matplotlib, numpy
- MMseqs2 (path set in config)
- jq
- GPU for step 04 (A100)

## Configuration

`configs/param_config.json` — all paths and parameters.

`configs/query_configs.tsv` — input sources, one row per label. The `query` column takes one of two modes:

- **UniProtKB query**: any UniProt query string. Step 01 downloads the sequences (TSV + FASTA + IDs) and features.
- **Local FASTA**: a path ending in `.fa`/`.faa`/`.fasta`, relative to the repo root. Step 01 copies it to `<seq_out_dir>/<label>_seqs.fasta`, which MMseqs2 picks up automatically. The TSV/IDs/features downloads are skipped, and a missing file makes the task fail.

```
label	query	exact_species
nucleus	(taxonomy_id:9606) AND (cc_scl_term:SL-0191) AND (reviewed:true)	H.sapiens nuclear proteins
koreicus	../KOREICUS/proteomes/koreicus_seqs.fasta	Ae. koreicus
...
```

Keep the TSV comment-free, because the slurm awk and pandas parsers read every line.

Key parameters:

| Key | Meaning |
|-----|---------|
| seq_out_dir | Raw sequences |
| features_out_dir | UniProt features |
| dedup_out_dir | Deduplicated FASTA |
| clusters_out_dir | MMseqs clusters |
| reps_out_dir | ESMC embeddings |
| model_name | ESMC model, or a list of models (default: [esmc_300m, esmc_600m, esmc_6b]); outputs go to `<reps_outdir>/<model>/<label>_reps.npz` |
| mmseqs.* | Dedup / clustering thresholds |
| scanpy.* | PCA, neighbors, Leiden resolution, enrichment cutoffs |

## Steps

### Run everything

```bash
bash slurms/run_all.sh                 # all steps, chained with afterok dependencies
bash slurms/run_all.sh 03 supp_01      # only selected steps (01 02 03 supp_01 supp_02)
DRY_RUN=1 bash slurms/run_all.sh       # print the sbatch commands without submitting
```

Array sizes are computed from the configs (labels in `query_configs.tsv`, models in `extract_reps_biohub.model_name`), so no `--array` editing is needed. Run it from the repo root on the login node (not via `sbatch`). Each step can still be submitted on its own with `sbatch`; the header `--array` is then used.

### 01 — Download UniProt

```bash
sbatch slurms/01_dw_uniprot.sh
```

- Reads queries from `query_configs.tsv`
- Downloads sequences (TSV + FASTA + IDs) and features, or copies a local FASTA source to `<label>_seqs.fasta`
- Output: `data/raw_seqs/`, `data/features/`

### 02 — Deduplicate

```bash
sbatch slurms/02_dedup.sh
```

- MMseqs2 easy-linclust (default 99% identity)
- Output: `data/mmseqs_clusters/dedup/*_dedup.fasta`

### 03 — Cluster (multi-threshold)

```bash
sbatch slurms/03_clust.sh
```

- Hierarchical clustering at thresholds listed in config (0.95 → 0.15)
- Uses representatives of previous threshold as input for next
- Output: `data/mmseqs_clusters/<label>/`
  - cluster TSV
  - representative FASTA
  - metrics TSV
  - summary

### 04 — Extract ESMC embeddings

```bash
sbatch slurms/04_reps.sh
```

- Runs on GPU
- Processes each `*_dedup.fasta`
- Output: `data/reps/ESMC-6B/<label>_reps.npz`

### 05 — Scanpy analysis

```bash
sbatch slurms/05_scanpy.sh
```

- Loads all .npz files
- L2-normalizes embeddings
- Optional PCA → neighbors (cosine) → Leiden → UMAP
- Rank genes (channels) per cluster
- Species enrichment plots
- Extracts protein modules (highly enriched clusters)

Output: `data/scanpy/ESMC-6B/.../`

```
<species-set>_scanpy.pdf                                       # merged PDF report
umap_joint_clusters_species_specificity__<model>__<params>.png
umap_reclustered_species_specific_clusters__<model>__<params>.png
barplot_species_specific_clusters_protein_counts__<model>__<params>.png
barplot_umap_labelled_clusters_protein_counts__<model>__<params>.png
species_specific_cluster_letters__<model>__<params>.csv
protein_clusters_metadata__<model>__<params>.csv
species_cluster_enrichment__<model>__<params>.csv
channel_importance_rankings__<model>__<params>.csv
clustering_metrics__<model>__<params>.txt
intra_group/umap_intra_species_top_clusters__<model>__<params>.png
protein-modules/
```

## Directory layout

```
configs/
  param_config.json
  query_config.tsv
slurms/
  01_dw_uniprot.sh
  02_dedup.sh
  03_clust.sh
  04_reps.sh
  05_scanpy.sh
src/
  dw_uniprot_seqs.py
  dw_uniprot_features.py
  extract_reps_biohub.py
  npz_scanpy.py
data/
  raw_seqs/
  features/
  mmseqs_clusters/
  reps/
  scanpy/
logs/
```

## Notes

- All scripts are idempotent (skip existing outputs).
- `slurms/run_all.sh` sizes array jobs to the number of labels / models; adjust resources in the SBATCH headers if needed.
- Steps 02 and supp_01 index the files on disk, not the TSV: when removing labels from the TSV, delete their stale outputs too.

