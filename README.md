# Protein Sequence Embedding & Clustering Pipeline

## Overview

Pipeline for downloading compartment-specific UniProt sets, reducing redundancy, building multi-identity sequence clusters, computing ESMC embeddings, and performing embedding-space clustering + enrichment analysis.

```
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

`configs/query_config.tsv` — input queries:

```
label	query	exact_species
nucleus	(taxonomy_id:9606) AND (cc_scl_term:SL-0191) AND (reviewed:true)	H.sapiens nuclear proteins
...
```

Key parameters:

| Key | Meaning |
|-----|---------|
| seq_out_dir | Raw sequences |
| features_out_dir | UniProt features |
| dedup_out_dir | Deduplicated FASTA |
| clusters_out_dir | MMseqs clusters |
| reps_out_dir | ESMC embeddings |
| model_name | ESMC model (e.g. esmc_6b) |
| mmseqs.* | Dedup / clustering thresholds |
| scanpy.* | PCA, neighbors, Leiden resolution, enrichment cutoffs |

## Steps

### 01 — Download UniProt

```bash
sbatch slurms/01_dw_uniprot.sh
```

- Reads queries from `query_config.tsv`
- Downloads sequences (TSV + FASTA + IDs) and features
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
protein_clusters_metadata.csv
species_cluster_enrichment_*.csv
channel_importance_rankings.csv
umap_species_panel.png
protein-modules/
clustering_metrics.txt
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
- Array jobs are sized to the number of labels / files.
- Adjust array ranges and resources in the SBATCH headers if needed.

