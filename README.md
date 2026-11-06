# SCambrian
#### SCambrian: a framework to dissect understudied proteomes
Area Science Park, Laboratory of Data Engineering

## What the pipeline does

SCambrian takes protein FASTA sequences of several **samples**. It clusters them by **sequence** (MMseqs2) and by **representation** (ESMC embeddings projected with UMAP and t-SNE). Each clustering is run twice: on every target alone (**internal**) and on all samples together (**joint**). Every cluster gets a unique name, and the result is **one summary table**: one row per sequence, one column per clustering.

Each sample is of one of two kinds:

- **Target**: a proteome you want to study, for example *Ae. aegypti*.
- **Probe**: a set of known proteins, for example Insecta immune proteins. A probe is clustered together with the targets, but it never counts as a species. Its only job is to show which target proteins co-cluster with it.

```
configs/query_configs.tsv + configs/param_configs.yaml
        │
01 acquire      UniProtKB download, or copy of a local FASTA                -> data/sequences/, data/features/
02 preprocess   validate, single-line FASTA, sc_label/sc_type tags,
                drop probe copies of targets, dedup per sample               -> data/preprocessed/<label>_dedup.fasta
                                                                              analyses/<A>/sc_metadata.tsv
03 mmseqs2      sequence clusters at several identities:
                internal (each target) + joint (all samples)                -> data/clusters/<label>/, analyses/<A>/mmseqs_joint/
04 reps         ESMC embeddings (default esmc_300m)                         -> data/esm-reps/<model>/<label>_reps.npz
05 project      UMAP + t-SNE (2D, 3D), internal and joint;
                Leiden on every layout                                      -> analyses/<A>/<model>/projections.h5ad
06 summary      cluster names + summary, species report, 3D viewer, RO-Crate -> analyses/<A>/<model>/
```

`<A>` is the analysis name, `{YYYY-MM-DD}_SCambrian` by default (the start date). Per-sample data that can be reused across analyses stays in `data/`. Everything joint or named stays in `analyses/<A>/`.

Each step is a SLURM script in `slurms/` that runs a Python script in `src/`. All parameters live in `configs/param_configs.yaml`, with one section per Python script.

## Vocabulary

| term | meaning |
|---|---|
| sample | one row of `configs/query_configs.tsv`. Its `label` names every file and must be unique; `taxonomic_ref` is its display name |
| `sc_type` | `target` or `probe` (the `input_type` column). Probes get no internal clusters and are excluded from the species statistics |
| record | the FASTA id (first word of the header). It must be unique across samples |
| analysis | one run of steps 02–06, in `analyses/<analysis_name>/` |
| series | one clustering: `{analysis}_{u\|t\|s}_{i\|j}_{stat}` |
| cluster | `{series}_{SERIAL}` |

### Cluster names

```
{analysis_name}_{u|t|s}_{i|j}_{stat}_{SERIAL}
```

| field | values |
|---|---|
| `u \| t \| s` | `u` = UMAP layout, `t` = t-SNE layout, `s` = sequence (MMseqs2) |
| `i \| j` | `i` = internal (one target alone), `j` = joint (all samples together) |
| `stat` | the method and the dimension (`leiden2d`, `leiden3d`), or the MMseqs2 identity (`id90`, `id50`, …) |
| `SERIAL` | internal: `{label}_001`, `{label}_002`, … Joint: `a`, `b`, …, `z`, `aa`, … In both, clusters are ranked by decreasing size. A joint cluster is written in **UPPER case** (`A`, `B`, …) when it holds at least one probe protein and at least one target protein |

Examples:

- `2026-11-02_SCambrian_u_j_leiden2d_C`: third-largest Leiden cluster of the joint 2D UMAP. It holds probe and target proteins.
- `2026-11-02_SCambrian_t_i_leiden3d_aegypti_004`: fourth-largest Leiden cluster of the 3D t-SNE of *aegypti* alone.
- `2026-11-02_SCambrian_s_j_id50_b`: second-largest MMseqs2 cluster at 50% identity, all samples together.

Leiden runs **on the layout coordinates**, on a kNN graph of the layout. A `u` or `t` series is therefore always "the clusters of that space". Names are given in one place only, `src/compile_summary.py`. UPPER case is a naming rule only: which probe clusters matter is decided by the probe criteria of the species report.

## The summary (the main result)

The files below are written to `analyses/<A>/<model>/` by step 06.

| file | content |
|---|---|
| `sc_summary.tsv` | The kept rows of `sc_metadata.tsv`, plus one column per series named `{analysis}_{u\|t\|s}_{i\|j}_{stat}`. Each cell holds the SERIAL. Filter or group by any column for post-hoc analysis |
| `sc_joint_clusters.tsv` | One row per joint series and cluster: `size`, `n_<sample>` and `pct_<sample>` (share of the cluster), `pct_probe`, `probe_cluster` |
| `sc_joint_composition__<primary_series>.png` | sample composition of the largest joint clusters; probes are hatched |
| `sc_series_cluster_counts.png` | number of clusters per series |

### `sc_metadata.tsv` keeps the score

Step 02 writes `analyses/<A>/sc_metadata.tsv`, with one row for **every** input record:

| column | meaning |
|---|---|
| `file` | source file: the local FASTA, or `data/sequences/<label>_seqs.fasta` |
| `record` | FASTA id |
| `sc_label`, `sc_type`, `sample`, `taxonomic_ref` | the sample the record belongs to |
| `length` | raw sequence length |
| `status` | `kept`, `invalid` (empty, bad letters, outside `[min_length, max_length]`, or a repeated id), `probe_overlap` (a probe record whose id or sequence is already in a target) or `duplicate` (merged by the dedup) |
| `representative` | the dedup representative of a kept or duplicate record |

The kept rows are the rows of the summary. The preprocessed FASTA headers carry the same tags: `>{header} sc_label={label} sc_type={type}`, each exactly once.

## Requirements

- SLURM cluster; one conda env `./.env` for every step and for Snakemake, created by `sbatch esmc_config.sh`
- MMseqs2 on the PATH (or `mmseqs_executable` in the YAML)
- A100 GPU for step 04

## Running

```bash
sbatch slurms/run_all.sh                        # every step whose outputs are missing or stale
sbatch slurms/run_all.sh 05 06                  # targets steps 05 and 06 (and whatever upstream they still need)
ANALYSIS_NAME=test_SCambrian sbatch slurms/run_all.sh 02
DRY_RUN=1 sbatch slurms/run_all.sh              # snakemake -n: list the jobs that would be submitted
bash slurms/list_relics.sh                      # read-only: print data of the old layout
```

Submit from the repo root. `run_all.sh` is a small THIN job (1 CPU, 2 GB) that runs **Snakemake** (`slurms/Snakefile`) inside `./.env`. Snakemake submits one job per unit of work and only when its outputs are missing, older than their inputs, or its YAML section changed. Each job is the unchanged `slurms/0X.sh` script, submitted by `slurms/submit_step.sh` with a single `--array` index (row of the query table for 01, target position for 03 with `N_TARGETS + 1` = joint, `row × N_MODELS + model` for 04, model for 05), so the scripts keep working with a manual `sbatch`. At most 2 DGX jobs (04) and 3 MMseqs2 jobs (03) run at once; `MAX_JOBS` (default 20) caps the total.

**Node choice.** For every non-DGX step, `submit_step.sh` reads `--cpus-per-task`, `--mem` and `--time` from the script's `#SBATCH` header, keeps the partitions whose time limit covers `--time` (never DGX or GPU ones), and asks `get_best_node` (`slurms/best_node.sh`) for the idle or mixed node with the most free CPUs and memory; the job is submitted there with `-p <partition> -w <node>`. When no node fits, the header partition is used. `NODE_PICK=0` disables it. Step 04 always goes to DGX.

The analysis name is resolved once and exported as `ANALYSIS_NAME`, in order:

1. the `ANALYSIS_NAME` environment variable;
2. `preprocess_sequences.analysis_name` in the YAML;
3. today's `{YYYY-MM-DD}_SCambrian` when step 02 is requested explicitly (or no analysis exists yet), otherwise the newest `analyses/*_SCambrian` folder, so a plain `run_all.sh` resumes the latest analysis.

`slurms/analysis_env.sh` holds this logic, and a single step submitted with `sbatch slurms/<step>.sh` resolves the name the same way. The default `#SBATCH --array` ranges cover up to 10 samples (01: `1-10`, 03: `1-11%3`, 04: `0-9%2` for one model, 05: `0-2`); out-of-range tasks exit cleanly.

### Reruns and freshness

Snakemake decides what to submit (`--rerun-triggers mtime params code`):

| step | job submitted when |
|---|---|
| 01 | a `<label>_seqs.fasta` (and, for UniProt rows, `_seqs.tsv`/`_feats.tsv`) is missing, or the local FASTA is newer |
| 02 | the query table or a `_seqs.fasta` is newer, the sample list or the `preprocess_sequences` section changed (stamp `analyses/<A>/.preprocess.<hash>.done`) |
| 03 | `clustering_summary.txt` is missing or older than the dedup FASTA(s), or `run_mmseqs2` changed |
| 04 | `<label>_reps.npz` is missing or older than its dedup FASTA |
| 05 | `projections.h5ad` is older than the metadata, the step-02 stamp or an `.npz`, or `npz_scanpy`/`preprocess_sequences` changed |
| 06 | an output is missing or older than a `.h5ad` or MMseqs2 summary, or `compile_summary`, `h5ad_scanpy`, `h5ad_interactive` or `ro_crate` changed |

Step 02 is a checkpoint: `preprocess_sequences.py` rewrites a dedup FASTA only when its content changes, and Snakemake reads the dedup times only after step 02 ran, so adding a sample re-clusters and re-embeds only that sample (plus the joint steps). Snakemake deletes the declared outputs of a job before submitting it, so the job's own freshness checks cannot skip a rebuild that Snakemake asked for. Inside each script the old checks stay as a second guard; `overwrite: true` (or `--overwrite`) still forces a rebuild of a manual run.

## Configuration

### `configs/query_configs.tsv`

One row per sample. Keep the file free of comments.

| column | meaning |
|---|---|
| `label` | short, unique name used in every file name (`aegypti`) |
| `query` | a UniProtKB query (`(organism_id:7159)`), downloaded by step 01 together with the annotation table, or a local FASTA path ending in `.fa`/`.faa`/`.fasta`, which is copied |
| `taxonomic_ref` | display name (`Ae. aegypti`) |
| `input_type` | `target` (the default) or `probe` |

### `configs/param_configs.yaml`

Each script reads its own section with `--yaml-in` (the default) and writes its current values back with `--yaml-out`. Command-line flags override the YAML.

| section | key parameters |
|---|---|
| `dw_uniprot_seqs`, `dw_uniprot_features` | `query_table`, `sequences_dir`, `features_dir` |
| `preprocess_sequences` | `analysis_name` (null: by date), `analyses_dir`, `preprocessed_dir`, `min_length` (30), `max_length` (null), `dedup_min_identity` (0.95), `dedup_min_coverage` (0.8), `dedup_coverage_mode` (0) |
| `run_mmseqs2` | `clusters_dir`, `cluster_identity_thresholds` (0.9 → 0.2, cascaded: each threshold clusters the representatives of the previous one), `cluster_min_coverage`, `cluster_coverage_mode` |
| `extract_reps_biohub` | `models` (`[esmc_300m]`; `esmc_600m` and `esmc_6b` are also supported), `embeddings_dir`, `sequences_per_chunk` |
| `npz_scanpy` | `pca_components` (null: full L2-normalised embedding), `knn_neighbors` (15), `leiden_resolution` (1.0), `layout_dimensions` ([2, 3]), `umap_min_dist`, `tsne_perplexity`, `tsne_iterations` |
| `compile_summary` | `primary_series` (`u_j_leiden2d`, the series drawn in the composition plot), `max_bars` |
| `h5ad_scanpy` | `report_series` (`u_j_leiden2d`), plus the species and probe thresholds below |
| `h5ad_interactive` | `max_points_per_view`, `max_colored_clusters` (30), `min_series_cluster_size` (5, smaller MMseqs2 clusters pooled) |
| `ro_crate` | `repo_url`, `metadata_dir`, `license` |

## Species report (step 06)

`h5ad_scanpy.py` reports the species differences of a single joint series, `report_series`, together with its internal counterpart (`u_j_leiden2d` is paired with `u_i_leiden2d`). Every label it draws is the SERIAL from the summary. The criteria use **targets only**. Notation: `c_ks` is the number of proteins of target `s` in joint cluster `k`, and `n_s` is the number of proteins of `s`.

1. **Species-normalised share**: `p_ks = c_ks / n_s`.
2. **Specificity index**: `S_ks = p_ks / Σ_s' p_ks'`.
3. **Fold change**: `FC_k = max_s p_ks / min_s p_ks`.
4. **Chi-square test of homogeneity** on `[c_ks ; n_s − c_ks]`, BH-corrected (`chi2_q`). The test is valid only if every expected count is at least 5.
5. **Significant**: the test is valid, `chi2_q < chi2_max_q` and `FC_k ≥ min_fold_change`.
6. **Selected for species s**: significant, `S_ks > 1/n_species` and `c_ks ≥ min_species_proteins`. At most `max_selected_per_species` clusters are kept, with the highest `S_ks`.
7. **Species module**: significant, `S_ks ≥ module_min_specificity_percent/100` and `c_ks ≥ min_module_proteins`.
8. **Balanced**: `FC_k ≤ 1/balanced_min_share_ratio` and `min_s c_ks ≥ min_species_proteins`.

Probe clusters are called per probe `P` on the same series. Every cluster that holds a probe protein is UPPER case, even with a single probe protein, so the probe readout uses only the clusters where the probe is enriched. Notation: `c_kP` is the number of proteins of `P` in joint cluster `k`, `n_k` the proteins of `k` (all samples), `n_P` the proteins of `P` and `N` all proteins.

9. **Probe share**: `p_kP = c_kP / n_P`. **Fold enrichment**: `E_kP = (c_kP / n_k) / (n_P / N)`.
10. **Probe test**: one-sided hypergeometric `P(X ≥ c_kP)`, BH-corrected across all joint clusters (`probe_q`).
11. **Probe cluster of P**: at least one target protein, `probe_q < probe_max_q`, `E_kP ≥ min_fold_change` and `c_kP ≥ min_probe_proteins`. At most `max_probe_clusters` clusters are kept, with the highest `p_kP`.

Outputs, all ending in `__<report_series>`:

- `species_cluster_enrichment` and `species_top_clusters` (CSV); the enrichment table also flags the probe clusters (`probe_selected`);
- `probe_cluster_enrichment` (one row per probe and joint cluster) and `probe_capture` (per target and probe, the target proteins inside the probe clusters, as a count and a percent; this is the probe readout);
- `protein-modules/`: species modules, and the target proteins of every probe cluster (`<probe>_<n>_clust<cluster>_…`);
- with `probe_mmseqs: true` (default), the probe test is repeated on every MMseqs2 joint series: `probe_cluster_enrichment`, `probe_capture` and probe-cluster modules ending in `__s_j_id90`, `__s_j_id80`, …;
- UMAP and t-SNE panels: internal and joint layouts (probe facets coloured by their probe clusters), the re-projected selected clusters, the internal-to-joint flux diagram and the count barplot;
- `<model>_scanpy.pdf`, which merges them.

`interactive/views_3d.html` is a self-contained protein **atlas** (built from `src/templates/atlas/`): opening on the joint page in 3D with nothing filtered, with an Overview of every page and, per page, the 3D/2D UMAP and t-SNE layouts coloured by any sample, `u`/`t`/`s` or UniProt series, a barplot (counts, log or per-cluster sample shares), a **Flux** (Sankey) view between any two series, a co-annotation **Network** (ForceAtlas2, force, community-grouped or circular layouts, Louvain communities) and an **Enrichment** view of the `uniprot_family`, GO or other feature terms over-represented in a cluster of any series, or in the shown proteins against the page (hypergeometric test, BH q, blue-to-red horizontal bars). The collapsible left pane holds every filter (samples, any cluster series, annotation terms; top 10 listed until Show all), shown as removable tags above the view; **Selected** and **Probe** tick the report's selected and probe clusters. A right drawer holds an Inspector (protein, cluster, term) and the Appearance settings, where every font size, palette colour and category colour is tunable with a slider or a number box; settings persist in the browser and export/import as JSON. `analyses/<A>/ro-crate-metadata.json` (RO-Crate 1.1) describes the analysis folder. A copy of it and of the configs is kept in the tracked `metadata/<A>/`.

## Notebooks

`nbs/` holds loading-only notebooks. Each reads its paths from the YAML, and `ANALYSIS_NAME`/`MODEL` choose the run.

| notebook | loads |
|---|---|
| `01_sequences.ipynb` | query table, sequences, UniProt tables, `sc_metadata.tsv` |
| `02_mmseqs_clusters.ipynb` | dedup ids, internal and joint MMseqs2 metrics and assignments |
| `03_embeddings.ipynb` | the ESMC embeddings of one model |
| `04_scanpy_clusters.ipynb` | `sc_summary.tsv`, `sc_joint_clusters.tsv`, `projections.h5ad`, species and probe report tables |

## Directory layout

```
configs/      param_configs.yaml, query_configs.tsv
slurms/       01_dw_uniprot.sh 02_preprocess.sh 03_mmseqs2.sh 04_reps.sh 05_project.sh 06_summary.sh
              run_all.sh Snakefile submit_step.sh best_node.sh analysis_env.sh list_relics.sh
src/          dw_uniprot_seqs.py dw_uniprot_features.py preprocess_sequences.py run_mmseqs2.py extract_reps_biohub.py
              npz_scanpy.py compile_summary.py h5ad_scanpy.py h5ad_interactive.py ro_crate.py templates/
data/         sequences/ features/ preprocessed/ clusters/ esm-reps/      (per sample, shared across analyses)
analyses/<A>/ sc_metadata.tsv mmseqs_joint/ <model>/ ro-crate-metadata.json configs/
```

## Leftovers from the previous layout

Outputs of the old pipeline are not deleted automatically. `bash slurms/list_relics.sh` prints them (read-only); remove them by hand if they are no longer needed:

- `data/dedup/` and `data/scanpy/`
- `data/features/*_focus_ids.*`
- `data/clusters/<label>/` files whose tag or identity threshold is not in the current config (old `*dmin*` names, older `ccov…_cmode…` tags); `run_mmseqs2.py` never removes them
- `analyses/` folders not named `*_SCambrian` (old `analyses/<labels joined by _>/`)
