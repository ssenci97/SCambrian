# Protein Sequence Embedding & Clustering Pipeline
Area Science Park -- Laboratory of Data Engineering

## Overview

The pipeline compares the proteomes of several species (currently *Ae. aegypti*, *Ae. albopictus* and *Ae. koreicus*) in ESMC embedding space. It finds groups of proteins whose share **differs across species**, and groups that are conserved. Every step is a SLURM script under `slurms/` that runs a Python script under `src/`. All parameters live in `configs/param_configs.yaml`, under one section per Python script.

```
configs/query_configs.tsv + configs/param_configs.yaml
        │
        ▼
01       UniProt download or local FASTA copy          src/dw_uniprot_seqs.py, src/dw_uniprot_features.py
         → data/sequences/<label>_seqs.{tsv,fasta}, <label>_ids.txt   data/features/<label>_feats.tsv
        │
        ▼
02       MMseqs2 dedup (easy-linclust, 95% identity)   src/run_mmseqs2.py
         + cascaded clustering at 0.9 → 0.2 identity
         → data/dedup/<label>_dedup.fasta (+ .params)   data/clusters/<label>/
        │
        ▼
03       ESMC embeddings (GPU), one task per label × model   src/extract_reps_biohub.py
         → data/esm-reps/<model>/<label>_reps.npz
        │
        ▼
supp_01  joint + intra-species Leiden, UMAP/t-SNE (2D, 3D)   src/npz_scanpy.py
         → data/scanpy/<labels>__<model>/<labels>__<model>__<params>.h5ad
        │
        ▼
supp_02  report (PDF, tables, modules) + 3D/2D HTML viewer   src/h5ad_scanpy.py, src/h5ad_interactive.py
         → data/scanpy/<labels>__<model>/
```

## Requirements

- SLURM cluster; one conda env `./.env` for every step, created by `sbatch esmc_config.sh`
- MMseqs2 on the PATH (or `run_mmseqs2.mmseqs_executable`)
- A100 GPU for step 03

## Running

```bash
bash slurms/run_all.sh                 # all steps, chained with afterok dependencies
bash slurms/run_all.sh 03 supp_01      # only selected steps (01 02 03 supp_01 supp_02)
DRY_RUN=1 bash slurms/run_all.sh       # print the sbatch commands without submitting
```

Run it from the repo root on the login node, not via `sbatch`. Array sizes are computed from the configs (labels in the query table, `extract_reps_biohub.models`). Each step can still be submitted alone with `sbatch slurms/<step>.sh`, in which case the header `--array` is used.

### Reruns and freshness

Every step decides on its own whether its outputs are fresh, so after any change you can simply run `bash slurms/run_all.sh` again. Only stale outputs are rebuilt. An output is stale when it is missing, older than one of its inputs, or was built with different parameters:

| step | rebuilt when |
|---|---|
| 01 | an output file is missing (downloads are not time-checked) |
| 02 dedup | `<label>_dedup.fasta` or its `<label>_dedup.params` sidecar is missing, the sidecar differs from the current `dedup_*` settings, or `<label>_seqs.fasta` is newer |
| 02 clustering | any output in `data/clusters/<label>/` is missing or older than the dedup FASTA (the file names carry the dedup and clustering parameters) |
| 03 | `<label>_reps.npz` is missing or older than the dedup FASTA |
| supp_01 | the `.h5ad` is older than an input `.npz`, or its stored `uns["params"]` differ from the current `npz_scanpy` settings |
| supp_02 | an output is older than the `.h5ad`, the script, the HTML template or the enrichment table |

`overwrite: true` (or `--overwrite`) forces a step. Files written with older parameters are left on disk and must be deleted by hand: for example `data/clusters/<label>/*dmin0.9*` after the switch to 95% dedup identity, the old modules in `data/scanpy/*/protein-modules/`, and `data/dimred/`, which no current script writes.

## Configuration

### `configs/query_configs.tsv`

One row per label. Keep it comment-free, because the slurm awk and pandas parsers read every line.

| column | meaning |
|---|---|
| `label` | short name used in every file name (`aegypti`) |
| `query` | a UniProtKB query (`(organism_id:7159)`), downloaded by step 01, or a local FASTA path ending in `.fa`/`.faa`/`.fasta` (relative to the repo root), copied to `<sequences_dir>/<label>_seqs.fasta`; a local FASTA has no UniProt tables |
| `exact_species` | display name in figures and the HTML (`Ae. aegypti`) |

### `configs/param_configs.yaml`

Every Python script reads its own section with `--yaml-in` (the default) and writes its current values back with `--yaml-out`. Command-line flags have the same names as the keys and override them. Every section also has `overwrite` (bool, default `false`), which forces the step to rebuild fresh outputs.

**`dw_uniprot_seqs`, `dw_uniprot_features`** (step 01)

| key | default | meaning |
|---|---|---|
| `query_table` | `configs/query_configs.tsv` | the label/query table above |
| `sequences_dir` | `data/sequences` | sequence outputs (`<label>_seqs.tsv`, `_seqs.fasta`, `_ids.txt`) |
| `features_dir` | `data/features` | UniProt annotation outputs (`<label>_feats.tsv`, every field except the sequence) |

**`run_mmseqs2`** (step 02)

| key | default | meaning |
|---|---|---|
| `sequences_dir` | `data/sequences` | input `<label>_seqs.fasta` files |
| `dedup_dir` | `data/dedup` | deduplicated `<label>_dedup.fasta` and `.params` sidecar |
| `clusters_dir` | `data/clusters` | per-label clustering outputs |
| `mmseqs_executable` | `mmseqs` | MMseqs2 binary name or path |
| `dedup_min_identity` | `0.95` | sequences at least this identical (0–1) are merged into one representative |
| `dedup_min_coverage` | `0.8` | minimum alignment coverage (0–1) for the dedup |
| `dedup_coverage_mode` | `0` | MMseqs2 `--cov-mode` of the dedup (0: coverage of query and target) |
| `cluster_identity_thresholds` | `[0.9 … 0.2]` | decreasing identities of the cascade; each threshold clusters the representatives of the previous one |
| `cluster_min_coverage` | `0.8` | minimum alignment coverage of every threshold |
| `cluster_coverage_mode` | `0` | MMseqs2 `--cov-mode` of the clustering |

**`extract_reps_biohub`** (step 03)

| key | default | meaning |
|---|---|---|
| `models` | `[esmc_300m, esmc_600m, esmc_6b]` | ESMC models; each writes `<embeddings_dir>/<model>/<label>_reps.npz` |
| `embeddings_dir` | `data/esm-reps` | base folder of the embeddings (git-ignored) |
| `sequences_per_chunk` | `500` | proteins per temporary checkpoint file (proteins are embedded one at a time) |

**`npz_scanpy`** (supp_01)

| key | default | meaning |
|---|---|---|
| `scanpy_dir` | `data/scanpy/` | base folder of the runs (`<labels>__<model>/`) |
| `seed` | `42` | random seed |
| `pca_components` | `null` | PCs used for the kNN graph; `null` uses the full L2-normalised embedding |
| `knn_neighbors` | `15` | neighbours of the cosine kNN graph shared by Leiden and UMAP |
| `leiden_resolution` | `1.0` | Leiden resolution (higher gives more, smaller clusters) |
| `layout_dimensions` | `[2, 3]` | dimensions of the UMAP and t-SNE layouts |
| `umap_min_dist` | `0.5` | UMAP minimum distance between embedded points |
| `tsne_perplexity` | `30.0` | t-SNE perplexity |
| `tsne_iterations` | `1000` | t-SNE optimisation iterations |

**`h5ad_scanpy`** (supp_02 report; see [Calling species differences](#calling-species-differences))

| key | default | meaning |
|---|---|---|
| `layout_methods` | `[umap, tsne]` | layouts drawn in the report |
| `chi2_max_q` | `0.05` | BH-corrected chi-square q-value below which a joint cluster can be significant |
| `min_fold_change` | `2.0` | minimum ratio of the largest to the smallest species-normalised share of a significant cluster |
| `min_species_proteins` | `100` | minimum proteins of a species in a cluster selected for it, and of every species in a balanced cluster |
| `max_selected_per_species` | `6` | most specific significant clusters selected per species |
| `balanced_min_share_ratio` | `0.6` | minimum smallest/largest species-normalised share of a balanced cluster |
| `module_min_specificity_percent` | `80` | minimum specificity index (%) of a saved species module; also the black outline of the flux diagram |
| `min_module_proteins` | `200` | minimum proteins of a saved module |
| `min_intra_cluster_proteins` | `150` | minimum proteins of an intra-species cluster highlighted in the intra panels |

**`h5ad_interactive`** (supp_02 viewer)

| key | default | meaning |
|---|---|---|
| `hdbscan_min_cluster_sizes` | `[500, 800, 1600]` | HDBSCAN `min_cluster_size` values, one label set each |
| `hdbscan_min_samples` | `10` | HDBSCAN `min_samples`, capped at `min_cluster_size` |
| `hdbscan_max_clusters` | `15` | largest HDBSCAN clusters kept; the rest are unassigned |
| `max_points_per_view` | `null` | random subsample of points per page; `null` keeps all |

## Outputs of supp_02

`data/scanpy/<labels>__<model>/`:

```
<labels>__<model>_scanpy.pdf                                    # merged PDF report
{umap,tsne}_joint_clusters_species_specificity__<params>.png
umap_reprojected_top_clusters__<params>.png                     # selected clusters, all species
sankey_intra_to_inter_clusters__<params>.png
barplot_umap_labelled_clusters_protein_counts__<params>.png
species_cluster_enrichment__<params>.csv                        # per cluster x species: counts, specificity, chi2_q, fold_change, significant, balanced, selected
species_top_clusters__<params>.csv                              # selected clusters per species, ranked
protein_clusters_metadata__<params>.csv
channel_importance_rankings__<params>.csv
clustering_metrics__<params>.txt
intra_group/{umap,tsne}_intra_species_top_clusters__<params>.png
protein-modules/
interactive/views_3d__<params>.html                             # 3D | 2D | Bars views, opens with the selected clusters ticked
interactive/hdbscan_umap3d__<params>.tsv
```

### Cluster labels

- **Lowercase letters = inter-species (joint) clusters**: Leiden id `k` -> `a, b, ..., z, aa, ab, ...` (0 -> a, 26 -> aa). Joint HDBSCAN clusters in the HTML follow the same rule.
- **Numbers (as strings) = intra-species clusters**: the intra Leiden id (`0, 1, ...`); per-species HDBSCAN ids are 0-based by decreasing size.

The same label is used in every figure, table, module file name (`<species>_<n>_clust<letter>_...`, `intra_<species>_<n>_clust<number>_...`), the HTML and the notebooks.

### Calling species differences

The criteria are defined once, in `src/h5ad_scanpy.py`. `h5ad_interactive.py` and the notebooks read its `significant`/`selected` columns and never recompute them. Notation: `c_ks` = proteins of species `s` in joint cluster `k`, `n_s` = proteins of `s`.

1. **Species-normalised share** `p_ks = c_ks / n_s` removes the proteome-size bias.
2. **Specificity index** `S_ks = p_ks / Σ_s' p_ks'` in [0, 1]; the uniform value is `1/n_species`.
3. **Fold change** `FC_k = max_s p_ks / min_s p_ks` (∞ if a species is absent); `min S / max S = 1/FC_k`.
4. **Test**: chi-square test of homogeneity on the 2 × n_species table `[c_ks ; n_s − c_ks]`, BH-corrected across all joint clusters (`chi2_q`). It is valid only if every expected count is ≥ 5 (`chi2_valid`, Cochran).
5. **Significant**: valid ∧ `chi2_q < chi2_max_q` ∧ `FC_k ≥ min_fold_change`.
6. **Selected for species s**: significant ∧ `S_ks > 1/n_species` ∧ `c_ks ≥ min_species_proteins`, keeping the `max_selected_per_species` highest `S_ks`. The **selection** is the union over species. It defines the letters on the joint panels, the re-projection (all proteins of these clusters from every species where present), the flux diagram, the barplot, the `selected` column and the clusters ticked by default in the HTML.
7. **Species module**: significant ∧ `S_ks ≥ module_min_specificity_percent/100` ∧ `c_ks ≥ min_module_proteins`.
8. **Balanced (conserved)**: `FC_k ≤ 1/balanced_min_share_ratio` ∧ `min_s c_ks ≥ min_species_proteins`. This is an effect-size call only, because non-significance is not evidence of equality.

All size and threshold gates are inclusive (`≥`). Clusters are defined on the same data that is tested, so q-values rank and filter clusters; they are not confirmatory p-values.

## Notebooks

`nbs/` holds loading-only notebooks, one per theme. Each reads its paths from `configs/param_configs.yaml` and ends with the data in memory, ready for analysis:

| notebook | loads |
|---|---|
| `01_sequences.ipynb` | query table, per-label sequence lengths, UniProt sequence and annotation tables |
| `02_mmseqs_clusters.ipynb` | dedup ids and parameters, cluster metrics per threshold, per-sequence cluster assignments |
| `03_embeddings.ipynb` | the ESMC embedding matrices of one `MODEL` |
| `04_scanpy_clusters.ipynb` | one `.h5ad` run (joint clusters as letters) and its report tables |

## Directory layout

```
configs/      param_configs.yaml, query_configs.tsv
slurms/       01_dw_uniprot.sh, 02_mmseqs2.sh, 03_reps.sh, supp_01_scanpy.sh, supp_02_scanpy_report.sh, run_all.sh
src/          dw_uniprot_seqs.py, dw_uniprot_features.py, run_mmseqs2.py, extract_reps_biohub.py,
              npz_scanpy.py, h5ad_scanpy.py, h5ad_interactive.py, templates/h5ad_interactive.html
nbs/          01_sequences.ipynb, 02_mmseqs_clusters.ipynb, 03_embeddings.ipynb, 04_scanpy_clusters.ipynb
data/         sequences/, features/, dedup/, clusters/, esm-reps/ (git-ignored), scanpy/
logs/         one folder per step (git-ignored)
esmc_config.sh   creates ./.env
```

## Notes

- `slurms/run_all.sh` sizes array jobs to the number of labels and models; adjust resources in the SBATCH headers if needed.
- Steps 02 and supp_01 index the files on disk, not the TSV: when removing labels from the TSV, delete their stale outputs too.
- Workflow manager: the steps are linear and already chained by `run_all.sh`, and the per-step freshness checks provide the rebuild-what-changed behaviour. Moving to Snakemake (with the SLURM executor plugin) becomes worthwhile when steps branch (parameter sweeps), the pipeline is shared, or per-output provenance is required.
