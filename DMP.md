# Data Management Plan

**Protein embedding and clustering of *Aedes koreicus*, *Ae. japonicus*, *Ae. aegypti* and *Ae. albopictus* proteomes, probed with an Insecta immune-process reference set, with the SCambrian pipeline**

Data Management Plan (DMP), living document

| Item | Description |
|---|---|
| Version | 1.6 (2026-11-06) |
| Data steward | Sergio Senci ([sergio.senci@areasciencepark.it](mailto:sergio.senci@areasciencepark.it), ORCID [0009-0005-9543-0461](https://orcid.org/0009-0005-9543-0461)) |
| Principal investigator | Francesca Cuturello ([francesca.cuturello@areasciencepark.it](mailto:francesca.cuturello@areasciencepark.it), ORCID [0000-0001-7242-7298](https://orcid.org/0000-0001-7242-7298)) |
| Affiliation | Laboratory of Data Engineering (LADE), Research and Innovation Technology Institute (RIT), Area Science Park, Trieste, Italy |
| Infrastructure | ORFEO HPC cluster (Slurm; THIN and EPYC CPU partitions, DGX partition with NVIDIA A100), CephFS storage |
| Repositories | <https://github.com/ssenci97/SCambrian> (analysis). Curation of the *Ae. koreicus* and *Ae. japonicus* proteomes is a separate, self-contained workflow (`Aedes_japonicus_koreicus_curation`: download slurms pointing to stable sources, `CITATION.bib`, `ACKNOWLEDGEMENTS.txt`) |
| Out of scope | Differential gene expression. It will be added in a later version. |

## 1. Data summary

### 1.1 Purpose

The project compares the proteomes of four *Aedes* species in protein language model embedding space and in sequence space. An Insecta immune-process protein set is clustered together with them as a **probe**: it is not a species, and only shows which mosquito proteins co-cluster with known immune proteins. The project identifies protein groups whose share differs between species, groups that are conserved, and groups enriched in immune probe proteins. It reuses public data only and generates no experimental data.

### 1.2 Input data

| Label | Type | Species | Source | Identifiers | Content |
|---|---|---|---|---|---|
| `aegypti` | target | *Ae. aegypti* | UniProtKB | query `(organism_id:7159)`, release 2026_03 | Protein FASTA and annotation TSV |
| `albopictus` | target | *Ae. albopictus* | UniProtKB | query `(organism_id:7160)`, release 2026_03 | Protein FASTA and annotation TSV |
| `koreicus` | target | *Ae. koreicus* | Curated proteome from kor_it (complete ORFs) and kor_hu, concatenated | see below | Protein FASTA |
| `japonicus` | target | *Ae. japonicus* | Curated proteome from jap_it | see below | Protein FASTA |
| `insecta_immune` | probe | Insecta spp. | UniProtKB | query `(taxonomy_id:50557) AND ((go:0006955) OR (keyword:KW-0391) OR (keyword:KW-0399))`, release **[TBD: UniProt release at download]** | Protein FASTA and annotation TSV |

The UniProt queries return all entries of the taxon (Swiss-Prot and TrEMBL), not only the reference proteome. UniProtKB release 2026_03 (published 2026-09-02) was current for the whole project. The files were downloaded between 2026-09-28 (first commit) and 2026-10-26 (first commit of the files). The exact date is the file modification time on ORFEO.

Every sample has an `input_type` in `configs/query_configs.tsv`:

- **target**: a proteome under study. Targets are clustered alone (internal) and together with all samples (joint), and enter the species statistics.
- **probe**: a functional reference set. It is clustered only jointly and is excluded from the species statistics.

**The immune probe.** `insecta_immune` holds every UniProtKB entry of Insecta (NCBI Taxonomy 50557) that matches a documented, arbitrary functional clause:

- GO:0006955 *immune system process*, with its descendant terms;
- or keyword KW-0391 *Immunity*;
- or keyword KW-0399 *Innate immunity*.

KW-0393 *Immunoglobulin domain* is excluded because it marks a fold, not a function. On UniProtKB 2026_03 (checked 2026-10-28) the clause matches 23,310 Insecta entries, of which 255 are *Ae. aegypti* and 314 are *Ae. albopictus*.

**Probe overlap.** Probe records whose id or sequence is already in a target are dropped (status `probe_overlap` in `sc_metadata.tsv`). A probe therefore points only at target proteins, never at copies of them, and each protein is counted once. The probe release and download date are recorded when step 01 downloads it.

#### Sources of the curated proteomes

| ID | Data | Repository | Identifiers | Size |
|---|---|---|---|---|
| `kor_it` | *Ae. koreicus* transcriptome, Italy | NCBI TSA | GKMD01, BioProject PRJNA947548, BioSample SAMN34994639 | 310,687 contigs |
| `jap_it` | *Ae. japonicus* transcriptome, Italy | NCBI TSA | GKMC01, BioProject PRJNA947978, BioSample SAMN34997961, isolate Unicam-2023b | 394,152 contigs |
| `kor_hu` | *Ae. koreicus* genome and annotation, Hungary | Zenodo (v1, 2023-12-06) | [doi:10.5281/zenodo.10278300](https://doi.org/10.5281/zenodo.10278300), assembly GCA_024533555.2 | 1.2 GB (FASTA, GFF3) |

#### Models

| Model | Source | Version |
|---|---|---|
| ESMC 300M (default) | Hugging Face `biohub/esmc-300m-2024-12` (formerly `EvolutionaryScale/`) | via `esm` 3.2.1.post1; revision `7f10b20ae75017b2dbc884070e03434515709a8d` (to confirm from the ORFEO cache) |
| ESMC 600M | Hugging Face `biohub/esmc-600m-2024-12` (formerly `EvolutionaryScale/`) | via `esm` 3.2.1.post1; revision `e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c` (to confirm from the ORFEO cache) |
| ESMC 6B | Hugging Face `biohub/ESMC-6B` | snapshot `af1602ba7406f521b11bf8f81d52af378cde09e4` (25.4 GB) |

The models used in a run are listed in `extract_reps_biohub.models` of `configs/param_configs.yaml`.

### 1.3 Generated data

`<A>` is the analysis name, `{YYYY-MM-DD}_SCambrian` by default. Per-sample data reusable across analyses stays in `data/`; joint and named results stay in `analyses/<A>/`.

| Data | Location | Format |
|---|---|---|
| Curated proteomes | curation repository, `proteomes/` | FASTA |
| Sequences and UniProt annotations | `data/sequences/`, `data/features/` | FASTA, TSV |
| Validated, deduplicated sequences | `data/preprocessed/<label>_dedup.fasta` | FASTA, TSV, `.params` |
| Per-record metadata and status | `analyses/<A>/sc_metadata.tsv` | TSV |
| Internal sequence clusters | `data/clusters/<label>/` | TSV, FASTA, text |
| Joint sequence clusters | `analyses/<A>/mmseqs_joint/` | TSV, FASTA, text |
| Embeddings | `data/esm-reps/<model>/<label>_reps.npz` | NPZ (float32) |
| Projections and Leiden clusters | `analyses/<A>/<model>/projections.h5ad` | H5AD |
| Summary (one row per protein, one column per clustering) | `analyses/<A>/<model>/sc_summary.tsv`, `sc_joint_clusters.tsv` | TSV |
| Species and probe report | `analyses/<A>/<model>/` (enrichment, probe capture, `protein-modules/`) | CSV, PNG, PDF, text |
| Interactive atlas | `analyses/<A>/<model>/interactive/views_3d.html` | HTML |
| Provenance | `analyses/<A>/ro-crate-metadata.json`, `analyses/<A>/configs/`, tracked copy in `metadata/<A>/` | RO-Crate 1.1 (JSON-LD), YAML, TSV |
| Code, configuration, logs | `src/`, `slurms/`, `configs/`, `nbs/`, `logs/` | Python, Bash, YAML, TSV, text |

### 1.4 Volume

- Raw inputs: a few GB.
- Model weights: about 27 GB. They are cached, not redistributed.
- Embeddings: about 4 KB (300M), 4.5 KB (600M) and 10 KB (6B) per protein.
- Total derived data: below 50 GB.

### 1.5 Sensitive data

None. All data are public and come from non-human organisms.

## 2. FAIR data management

### 2.1 Findable

- Source data have persistent identifiers (section 1.2).
- Derived data will be deposited on Zenodo with a DOI: [doi:10.5281/zenodo.23741554](https://doi.org/10.5281/zenodo.23741554).
- The repository will be released through GitHub–Zenodo with a DOI: **[TBD: code DOI]**.
- Files are prefixed with the sample label. Every analysis lives in its own named folder.
- Every cluster has a unique, self-describing name: `{analysis}_{u|t|s}_{i|j}_{stat}_{SERIAL}` (UMAP, t-SNE or sequence; internal or joint; method or identity; serial). Joint clusters are named with letters, internal clusters with `{label}_{number}`. A joint cluster holding both probe and target proteins is written in UPPER case.
- Metadata:
  - curated FASTA headers carry provenance (section 2.3);
  - preprocessed FASTA headers carry `sc_label=` and `sc_type=` tags;
  - `sc_metadata.tsv` records the fate of every input record (`kept`, `invalid`, `probe_overlap`, `duplicate`) and its dedup representative;
  - `CITATION.bib` and `ACKNOWLEDGEMENTS.txt` of the curation repository list sources and licences;
  - `.h5ad` files store their parameters in `uns["params"]`;
  - dedup outputs have a `.params` sidecar;
  - each analysis has an RO-Crate 1.1 description (`ro-crate-metadata.json`).
- Each published result reports the repository commits, model versions, software versions, UniProt release and download date (2026-09-28 to 2026-10-26, see section 1.2).

### 2.2 Accessible

- Source data are open on NCBI, Zenodo and UniProt. No login is needed.
- ESMC weights are public. No token is needed.
  - 300M and 600M are downloaded by `ESMC.from_pretrained()`.
  - 6B is downloaded by `huggingface_hub.snapshot_download()` and mapped onto the `esm` model by `src/extract_reps_biohub.py`.
  - Inference runs locally. The hosted API (biohub.ai, formerly Forge) is not used.
- Derived data will be open on Zenodo at publication.
- Code is public on GitHub under Apache-2.0.
- Download scripts:
  - mosquito data: the curation repository download slurm (HTTPS, Zenodo API);
  - UniProtKB: `slurms/01_dw_uniprot.sh` (UniProt REST paginated `search` endpoint).

### 2.3 Interoperable

- Formats are open: FASTA, GFF3, TSV/CSV, NPZ, H5AD, YAML, JSON-LD, PNG/PDF/HTML.
- Curated FASTA headers:
  - Italy: `>id=<seqid> organism=<species> dataset=<IT_GKMD01|IT_GKMC01> type=<ORF type> len=<aa> strand=<+|-> pos=<start-end> score=<score>`
  - Hungary: `>id=<seqid> organism=Aedes_koreicus dataset=HU_GCA_024533555.2`
- UniProtKB accessions, NCBI Taxonomy IDs (7159, 7160, 140438, 50557), Gene Ontology terms (GO:0006955) and UniProt keywords (KW-0391, KW-0399) are kept as provided.
- Curation:
  - TransDecoder (`--single_best_only`) predicts ORFs from transcripts;
  - gffread extracts proteins from the genome annotation;
  - SCambrian uses the filtered (complete ORFs only) proteomes.
- The proteomes differ in origin (predicted ORFs, gene models, UniProtKB entries). Two steps limit the bias: deduplication at 95% identity within each sample, and normalisation of cluster counts by proteome size.
- Software: Python 3.10, PyTorch 2.4.1 (CUDA 12.1), flash-attn 2.6.3, `esm` 3.2.1.post1, Snakemake ≥ 8, Scanpy ≥ 1.10, igraph, umap-learn, scikit-learn, MMseqs2 (commit `8cc5ce367b5638c4306c2d7cfc652dd099a4643f`), TransDecoder 6.0.0, gffread **[TBD: gffread version]**. The environment `./.env` is built by `esmc_config.sh`.

### 2.4 Reusable

#### Licences

| Resource | Licence | Conditions |
|---|---|---|
| UniProtKB | CC BY 4.0 | Cite UniProt and the release |
| Zenodo 10278300 | CC BY 4.0 | Cite the dataset and article |
| NCBI TSA | NCBI policy: no restrictions; submitters may claim rights | Cite the accessions and article |
| ESMC weights and `esm` code | MIT (© 2026 Chan Zuckerberg Biohub) and Biohub Acceptable Use Policy (27 May 2026) | Keep the licence notice; follow the policy; cite ESMC |
| MMseqs2, gffread | MIT | Cite |
| TransDecoder | BSD-style (Broad Institute) | Cite |
| PyTorch, flash-attn, Scanpy, umap-learn, scikit-learn | BSD-3-Clause | Cite |
| Snakemake | MIT | Cite |
| python-igraph | GPL-2.0 | Used as a library, not redistributed |

#### Notes

- Earlier ESMC releases had other licences: Cambrian Open (300M), Cambrian Non-Commercial (600M), Cambrian Inference Clickthrough (6B, Forge/AWS only). The Hugging Face repositories were relicensed to MIT plus the Biohub Acceptable Use Policy on 2026-06-03, before this project started (2026-09-28), so all weights used here are under MIT. Snapshot hashes are recorded in section 1.2.
- Derived data, embeddings included, will be released under CC BY 4.0. Sources and changes will be stated. The RO-Crate `license` field is still unset: **[TBD: set `ro_crate.license`]**.

#### Quality assurance

- Downloads check HTTP codes and exit status, reject HTML error pages, are written atomically and retried.
- Every record is validated: empty sequences, invalid letters, lengths below 30 aa and repeated ids are marked `invalid`. Record ids shared by two samples stop the run. Counts are logged.
- Missing inputs or invalid paths stop a step with an error. Dependent steps are cancelled (`afterok`).
- The 6B checkpoint load fails on any missing or unexpected parameter.
- Cluster metrics are reviewed before interpretation.
- Probe clusters are called only when the probe is significantly enriched (section 3), not merely present; the capture of target proteins by probe clusters is recorded per target and probe, on the Leiden and on every MMseqs2 joint series.

#### Reproducibility

- Parameters are in `configs/param_configs.yaml`. Labels, queries and input types are in `configs/query_configs.tsv`. Both are copied into each analysis folder and into the tracked `metadata/<A>/`.
- Scripts read and write their parameters with `--yaml-in` and `--yaml-out`.
- Snakemake rebuilds an output only when it is missing, older than an input, or built with different parameters; its metadata (`.snakemake/`) and the driver log (`logs/snakemake/`) record every submitted job.
- The random seed is 42.
- Slurm logs record job ID, date, run time and memory.

**Availability:** Zenodo keeps deposits for the lifetime of the repository (at least 20 years).

### 2.5 Citations

#### Data

- Catapano PL et al. De novo genome assembly of the invasive mosquito species *Aedes japonicus* and *Aedes koreicus*. *Parasites & Vectors* 2023;16:427. [doi:10.1186/s13071-023-06048-w](https://doi.org/10.1186/s13071-023-06048-w).
- Nagy NA et al. The updated genome of the Hungarian population of *Aedes koreicus*. *Scientific Reports* 2024;14:7545. [doi:10.1038/s41598-024-58096-6](https://doi.org/10.1038/s41598-024-58096-6). Data: [doi:10.5281/zenodo.10278300](https://doi.org/10.5281/zenodo.10278300).
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research* 2025;53(D1):D609–D617. [doi:10.1093/nar/gkae1010](https://doi.org/10.1093/nar/gkae1010).
- The Gene Ontology Consortium. The Gene Ontology knowledgebase in 2023. *Genetics* 2023;224(1):iyad031. [doi:10.1093/genetics/iyad031](https://doi.org/10.1093/genetics/iyad031).

#### Models and software

- Candido S et al. Language Modeling Materializes a World Model of Protein Biology. bioRxiv 2026. [doi:10.64898/2026.06.03.729735](https://doi.org/10.64898/2026.06.03.729735).
- ESM Team. ESM Cambrian. EvolutionaryScale, 2024. <https://evolutionaryscale.ai/blog/esm-cambrian>
- Haas BJ et al. *Nature Protocols* 2013;8:1494–1512. [doi:10.1038/nprot.2013.084](https://doi.org/10.1038/nprot.2013.084) (TransDecoder).
- Pertea G, Pertea M. *F1000Research* 2020;9:304. [doi:10.12688/f1000research.23297.2](https://doi.org/10.12688/f1000research.23297.2) (gffread).
- Steinegger M, Söding J. *Nature Biotechnology* 2017;35:1026–1028. [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988) (MMseqs2).
- Steinegger M, Söding J. *Nature Communications* 2018;9:2542. [doi:10.1038/s41467-018-04964-5](https://doi.org/10.1038/s41467-018-04964-5) (Linclust).
- Wolf FA et al. *Genome Biology* 2018;19:15. [doi:10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0) (Scanpy).
- Traag VA et al. *Scientific Reports* 2019;9:5233. [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z) (Leiden).
- McInnes L et al. UMAP. arXiv:1802.03426, 2018.
- van der Maaten L, Hinton G. *Journal of Machine Learning Research* 2008;9:2579–2605 (t-SNE).
- Benjamini Y, Hochberg Y. *J R Stat Soc B* 1995;57:289–300.
- Mölder F et al. Sustainable data analysis with Snakemake. *F1000Research* 2021;10:33. [doi:10.12688/f1000research.29032.2](https://doi.org/10.12688/f1000research.29032.2).
- Soiland-Reyes S et al. Packaging research artefacts with RO-Crate. *Data Science* 2022;5(2):97–138. [doi:10.3233/DS-210053](https://doi.org/10.3233/DS-210053).
- SCambrian, <https://github.com/ssenci97/SCambrian>, release v1.0.0. doi: **[TBD: code DOI]**.

#### Acknowledgement

> We thank the UniProt Consortium for UniProtKB; P.L. Catapano, G. Favia and co-authors (University of Camerino) for the NCBI TSA transcriptomes GKMD01 and GKMC01; N.A. Nagy, L. Laczkó and co-authors for the Hungarian *Ae. koreicus* genome ([doi:10.5281/zenodo.10278300](https://doi.org/10.5281/zenodo.10278300)); NCBI/INSDC for data hosting; Biohub for the ESM Cambrian models; and Area Science Park for the ORFEO cluster.

## 3. Workflow

Curation (in the curation repository) runs once:

| Step | Tool and settings | Slurm resources | Output |
|---|---|---|---|
| Mosquito download | HTTPS, Zenodo API | 1 CPU, 4 GB, 2 h | raw downloads |
| Curation | TransDecoder, gffread | 2 CPU, 16 GB, 8 h | `proteomes/` |

SCambrian (this repository):

| Step | Script | Tool and settings | Slurm resources | Output |
|---|---|---|---|---|
| 00 Environment | `esmc_config.sh` | conda | DGX, 1 A100, 16 GB, 4 h | `./.env` |
| 01 Acquire | `slurms/01_dw_uniprot.sh` | UniProt REST, or copy of a local FASTA; UniProt annotations | THIN, 1 CPU, 2 GB, 2 h per sample | `data/sequences/`, `data/features/` |
| 02 Preprocess | `slurms/02_preprocess.sh` | Validation (min. 30 aa), `sc_label`/`sc_type` tags, probe overlap removal, MMseqs2 linclust dedup at 0.95 identity, coverage 0.8 | THIN, 16 CPU, 20 GB, 4 h | `data/preprocessed/`, `analyses/<A>/sc_metadata.tsv` |
| 03 Sequence clusters | `slurms/03_mmseqs2.sh` | MMseqs2 at 0.9 → 0.2 identity, coverage 0.8; internal (each target) and joint (all samples) | THIN, 16 CPU, 20 GB, 4 h | `data/clusters/`, `analyses/<A>/mmseqs_joint/` |
| 04 Embeddings | `slurms/04_reps.sh` | ESMC (300M by default; 600M, 6B); mean over residues | DGX, 1 A100, 100 GB, 4 h per sample × model | `data/esm-reps/` |
| 05 Projection | `slurms/05_project.sh` | cosine kNN (k = 15); UMAP (min. dist. 0.5) and t-SNE (perplexity 30), 2D and 3D, internal and joint; Leiden (res. 1.0) on every layout | EPYC, 16 CPU, 32 GB, 16 h per model | `projections.h5ad` |
| 06 Summary and report | `slurms/06_summary.sh` | Cluster names and summary; species report: chi-square with BH (q < 0.05), fold change ≥ 2; probe report: one-sided hypergeometric with BH (q < 0.05), fold enrichment ≥ 2, ≥ 10 probe proteins, ≥ 1 target protein, repeated on every MMseqs2 joint series; interactive atlas; RO-Crate | EPYC, 4 CPU, 16 GB, 2 h | TSV, CSV, PNG, PDF, HTML, JSON-LD |

`sbatch slurms/run_all.sh` runs Snakemake (`slurms/Snakefile`) as a small THIN job (1 CPU, 2 GB). Snakemake submits each step script unchanged, one array index at a time, and only when its outputs are missing, older than their inputs, or its parameters changed; adding a sample re-processes only that sample and the joint steps. Non-DGX jobs are placed on the idle or mixed CPU node with the most free CPUs and memory among the partitions whose time limit fits the job (`slurms/best_node.sh`, `slurms/submit_step.sh`); step 04 always runs on DGX. The array ranges cover up to 10 samples.

## 4. Resources

**Costs:** none for data, models or Zenodo. Compute and storage are covered by Area Science Park on ORFEO. Step 04 uses at most 48 A100 GPU-hours per full run with all three models.

### Roles

| Role | Name | Tasks |
|---|---|---|
| Data steward | Sergio Senci | Runs the workflow, maintains code and documentation, manages backups and deposits |
| Principal investigator | Francesca Cuturello | Approves sharing and checks licence compliance |

## 5. Storage and preservation

- **Storage:** ORFEO CephFS. Access is limited to project members.
- **Git:** embeddings, analysis outputs, logs, temporary files and the conda environment are not tracked. The RO-Crate and configuration copies in `metadata/<A>/` are tracked.
- **Raw data:** read-only after download. No script modifies it.
- **Preserved:** code, configuration, credit files, curated proteomes, downloaded UniProt files, preprocessed sequences and `sc_metadata.tsv`, sequence clusters, embeddings, `.h5ad` files, summaries, report tables and RO-Crates.
- **Not preserved:** the original mosquito downloads (re-fetched by the curation repository from NCBI TSA and Zenodo), intermediates and model weights. They can be regenerated.
- **Backup:** Git and GitHub for code. Data are copied to the ORFEO Long Term Storage (LTS) partition (see the ORFEO documentation) every **[TBD: backup frequency]**.
- **Integrity:** SHA-256 checksums are stored in `SHA256SUMS` and deposited with the data.
- **Cleaning:** temporary files are deleted after outputs are validated. Outputs of the previous layout and outputs made with outdated parameters are listed by `slurms/list_relics.sh` (read-only) and deleted by hand.
- **Long term:** Zenodo.

## 6. Ethics and legal compliance

- The data are public, non-human and non-personal. No ethics approval, consent or GDPR measure is needed.
- Licences are respected as described in section 2.4.
- ESMC is used to describe existing proteomes only. No sequences are designed. This complies with the Biohub Acceptable Use Policy.

## 7. Revision history

| Version | Date | Description |
|---|---|---|
| 1.0 | 2026-10-28 | Initial version |
| 1.1 | 2026-10-28 | Renamed to `DMP.md`; placeholders filled (PI, versions, UniProt release, model revisions, licence, backup); curation moved to `raw/` |
| 1.2 | 2026-10-28 | Converted to LaTeX (`DMP.tex`) for stable A4 pagination; built with Tectonic |
| 1.3 | 2026-10-28 | Added the `insecta_immune` reference set (UniProtKB, GO:0006955, Insecta); dataset DOI reserved |
| 1.4 | 2026-10-28 | Input roles (target, probe); documented immune focus clause (GO:0006955, KW-0391, KW-0399); annotated focus tier of UniProt targets; functional focus layer; paired baseline and focus runs |
| 1.5 | 2026-11-06 | Back to Markdown (`DMP.md`); rewritten for the six-step pipeline: four *Aedes* targets with the `insecta_immune` probe clustered internally and jointly, probe overlap removal, hypergeometric probe clusters on Leiden and MMseqs2 series, `analyses/<A>/` layout, RO-Crate provenance; curation in a separate repository; focus tier and paired runs removed |
| 1.6 | 2026-11-06 | Snakemake orchestration with output-based job submission and CPU node selection; array ranges for up to 10 samples; relic listing |
