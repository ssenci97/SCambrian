# SCambrian release audit

| | |
|---|---|
| Date | 2026-10-28 |
| Target | `ssenci97/SCambrian@8a28649` (`main`); a `dev` branch exists at `8a45c8e` |
| Release goal | v1.0.0, GitHub tag + Zenodo DOI (code and data) |
| References | ISO/IEC 25010 (product quality), ISO/IEC 5230 OpenChain (licence compliance), ISO/IEC 5962 SPDX, ISO 8601 (dates), ISO 15489 (record integrity), FAIR4RS, RO-Crate 1.1, CFF 1.2, Zenodo metadata |
| Status | Audit only. Updated 2026-10-28: A3 corrected, A5 and B12 resolved |

## Inventory

622 tracked files, 2.31 GB in the working tree, `.git` 855 MB.

| area | files | size | content |
|---|---|---|---|
| `raw/` (curation: NCBI TSA, TransDecoder, Hungarian assembly) | ~150 | ~1.3 GB | inputs and intermediates for koreicus and japonicus |
| `data/scanpy/*__esmc_{300m,600m,6b}` | 355 | 347 MB | h5ad, PDF/PNG reports, modules, HTML viewer |
| `data/clusters/<label>` | 76 | 341 MB | MMseqs2 cascade 0.9 → 0.2 |
| `data/features`, `data/sequences`, `data/dedup` | 18 | ~330 MB | UniProt tables, FASTA, dedup |
| code, configs, notebooks | 24 | < 0.3 MB | `src/`, `slurms/`, `nbs/`, `configs/` |
| `data/esm-reps` | – | git-ignored | ESMC embeddings, the costliest artefact |

## A. Blockers (P0)

1. **No code licence.** There is no `LICENSE`, so Zenodo would archive code that cannot legally be reused. MIT or BSD-3-Clause fits.
2. **No citation metadata.** `CITATION.cff` and `.zenodo.json` are missing: no creators, ORCID, affiliation, version, `date-released` or `related_identifiers`.
3. **Licences of models and sources are mixed and not declared** (ISO/IEC 5230). There is no `THIRD_PARTY_LICENSES` file.

   | component | licence | consequence |
   |---|---|---|
   | ESMC-300M, ESMC-600M (`biohub/esmc-{300m,600m}-2024-12`) | MIT + Biohub Acceptable Use Policy (relicensed 2026-06-03, before the project started; verified via the Hugging Face API) | keep the licence notice, follow the AUP, cite ESMC |
   | ESMC-6B (`biohub/ESMC-6B`, `af1602ba…`) | MIT + Biohub Acceptable Use Policy (verified via the Hugging Face API) | as above |
   | UniProtKB | CC BY 4.0 | attribution |
   | NCBI TSA GKMC01 / GKMD01 | no use restrictions, citation required | already cited in `raw/ACKNOWLEDGEMENTS.txt` |
   | Hungarian *Ae. koreicus* assembly (Zenodo 10278300) | to verify | `raw/raw/nuclear_assembly.gff3`, `.fai`, and the derived proteome are redistributed |
   | MMseqs2 (GPLv3), TransDecoder | invoked, not redistributed | no obligation, cite versions |

   *Correction:* an earlier version of this audit said the 600M outputs were non-commercial. That applied to the old Cambrian licences only. Every source is now permissive (MIT, CC BY 4.0, NCBI), so the derived data can go out under CC BY 4.0, as `DMP.tex` §2.4 states. Still missing: a `THIRD_PARTY_LICENSES` file, and confirming the Zenodo 10278300 licence (the DMP records CC BY 4.0).
4. **2.3 GB of data in git.**
   - GitHub recommends repositories under 1 GB.
   - The GitHub→Zenodo integration archives the whole tag, so large archives slow it down or fail, and they mix code and data licences in one record.
   - `data/features/albopictus_feats.tsv` is 96.7 MB, just under GitHub's 100 MB hard limit. The next UniProt release can break the push.
5. **Resolved 2026-10-28: the query table pointed outside the repo.** `configs/query_configs.tsv` reads `../Aedes_japonicus_koreicus_curation/proteomes/{koreicus,japonicus}_seqs.fasta`, but byte-identical copies are now in `raw/proteomes/`. A fresh clone could not run step 01 for those two labels. It now points to `raw/proteomes/{koreicus,japonicus}_seqs.fasta`.
6. **Personal absolute paths are published.** These 12 files under `raw/` contain `/orfeo/cephfs/home/area/ssenci/...` or `/orfeo/cephfs/scratch/...`:
   - 6 `pipeliner.*.cmds`
   - 4 `*.roc.auc.plot.Rscript`
   - `raw/slurm_dw_proc.sh`
   - `raw/slurm_transdecoder.sh`

   The 60 TransDecoder checkpoint files (`*.ok`, `*.cmds`) have no scientific value.

## B. Data integrity and redundancy (P1)

7. **Duplicates, confirmed by sha256:**
   - `raw/proteomes/japonicus_seqs.fasta` = `raw/proteomes/japonicus_italian_proteome_filtered.fasta` = `data/sequences/japonicus_seqs.fasta` (`04c089c7df5d…`)
   - `raw/proteomes/koreicus_seqs.fasta` = `data/sequences/koreicus_seqs.fasta` = the concatenation of `koreicus_italian_proteome_filtered.fasta` and `koreicus_hungarian_proteome.fasta` (`6ebf7ad9f1dd…`)
8. **Two Hungarian proteomes.**
   - `raw/data/hungarian/koreicus_hungarian_proteome.fasta` (12.2 MB): raw `>mRNA…` headers, wrapped at 70 columns.
   - `raw/proteomes/koreicus_hungarian_proteome.fasta` (13.2 MB): normalised headers `>id=… organism=… dataset=HU_GCA_024533555.2`.

   Label the first as provenance only, or drop it.
9. **No checksums.** Add `SHA256SUMS` and MD5s (Zenodo displays MD5) for every deposited file (ISO 15489).
10. **Provenance gaps.** The `tmp_DMP.txt` placeholders are still open:
    - UniProt release: 2026_03 (deduced, in `DMP.tex`); the exact download date is still to be taken from the ORFEO file times;
    - the model revisions are recorded in `DMP.tex` (6B `af1602ba…`; 300M and 600M taken from the current Hugging Face revision, still to be confirmed from the ORFEO cache), but they are not pinned in code: `snapshot_download` has no `revision=` (`src/extract_reps_biohub.py:182`). Versions recorded in `DMP.tex`: `esm` 3.2.1.post1, MMseqs2 `8cc5ce36…`, TransDecoder 6.0.0. gffread is still missing;
    - MMseqs2 and TransDecoder versions;
    - no environment lock file: `esmc_config.sh` installs unpinned `esm`, `scanpy>=1.10`, `numpy<2.3`, and so on;
    - the full `npz_scanpy` settings exist only inside each h5ad's `uns["params"]`.
11. **Embeddings not deposited.** `data/esm-reps/*.npz` are the most expensive outputs (GPU hours, 6B model) and the input of everything downstream. Deposit them in the data record.
12. **Resolved 2026-10-28: the DMP is now `DMP.tex`.** Remaining placeholders: `[dataset DOI]`, `[code DOI]`, `[gffread version]` and `[backup frequency]`. The chosen code licence is Apache-2.0.

## C. Naming conventions (ISO/IEC 25010 maintainability)

Recommendation: **document the conventions for v1.0.0 and rename in v1.1.** Renaming now would invalidate the freshness checks, the notebooks and the HTML viewer.

| issue | example |
|---|---|
| inconsistent decimal encoding | `id0_2` vs `dmin0.95`, `res1.0` |
| mixed separators | `__` between blocks, but single `_` before the model in modules: `albopictus_0000257_clustdn_esmc_300m__res1.0_nn15_full.txt` |
| labels joined with `_` | now concrete: with `insecta_immune` the run folder becomes `aegypti_albopictus_koreicus_japonicus_insecta_immune__esmc_*`, which cannot be split back into labels on `_` (no script does so; it affects people and metadata parsers) |
| mixed directory case | `protein-modules`, `esm-reps` (kebab) vs `intra_group` (snake) |
| mixed table formats | `data/clusters/*.tsv` vs `data/scanpy/*.csv` |
| partial parameter tag | `res1.0_nn15_full` encodes 3 of 9 `npz_scanpy` keys (no seed, `umap_min_dist`, perplexity, …) |
| species identity split across contexts | `label` (`aegypti`), `exact_species` (`Ae. aegypti`), FASTA `organism=Aedes_koreicus`, `dataset=IT_GKMD01`; no single species table with NCBI TaxIDs |
| modules without header | `protein-modules/*.txt` are bare accession lists whose meaning lives only in the file name |
| `raw/` layout | `raw/raw/` nesting; `raw/data/italian` vs `raw/data/japonicus` mixes country and species axes |
| curation scripts named differently in their own usage lines | they say `get_and_credit_transcriptomes.sh` and `run_transdecoder.sh`, but the files are `raw/slurm_dw_proc.sh` and `raw/slurm_transdecoder.sh` |

Conventions worth keeping and documenting:
- the FASTA header schema `id= organism= dataset= type= len= strand= pos= score=`;
- zero-padded module sizes;
- letters for joint clusters and numbers for intra clusters;
- the `.params` sidecars;
- one folder per `<labels>__<model>` run;
- the `__<params>` suffix on every report output.

## D. Scientific and reporting risks (P1)

13. **All silhouette scores are negative.** In `clustering_metrics__esmc_300m__res1.0_nn15_full.txt`, joint is −0.21 and intra ranges from −0.02 to −0.16. Reviewers will read that as "no cluster structure". Explain it (Leiden on a cosine kNN graph vs silhouette in the full embedding space; silhouette is not a selection criterion), or keep it out of the headline results.
14. **Post-selection inference is already acknowledged** in the README: q-values rank and filter clusters but are not confirmatory. Repeat that sentence in the Zenodo description and the paper.
15. **The proteome sources are heterogeneous.**
    - koreicus and japonicus are TransDecoder predictions from transcriptomes, complete ORFs only (`type=complete`, `--single_best_only`).
    - aegypti and albopictus are all UniProtKB entries (Swiss-Prot and TrEMBL, not the reference proteome).
    - `insecta_immune` (added 2026-10-28) is a functional reference set, not a species. *Resolved 2026-10-28:* it is now a `probe` (`input_type` column). The species statistics use the targets only, probe accessions already present in a target are dropped (each protein counted once), and the probe drives a separate functional focus layer (enrichment, capture, gravity) with a documented focus clause (GO:0006955, KW-0391, KW-0399). See README, "Input roles and functional focus".

    Completeness and redundancy differ by source and can confound species-share tests. State this explicitly, ideally with a BUSCO completeness per proteome.

## E. Hygiene (P2)

16. **`.gitignore`**: add `**/*.transdecoder_dir/`, `*.ok`, `*.cmds` and `*.part`. Decided: `raw/` (processed curation) is preserved; **`raw/raw/` (original downloads, re-fetched by `raw/slurm_dw_proc.sh`) is excluded** from the deposit. Add it to `.gitignore` and `git rm -r --cached raw/raw` (not done yet).
17. **`.mailmap`**: some commits carry HPC hostname emails (`ssenci@thin002…`, `ssenci@genoa001…`). Mapping them is cosmetic and doesn't rewrite history.
18. **Versioning**: add `CHANGELOG.md` with ISO 8601 dates and tag `v1.0.0` (annotated). Confirm that `dev` (`8a45c8e`) is merged or intentionally excluded.
19. **README**: add "How to cite", DOI and licence badges, porting notes (ORFEO partitions THIN/EPYC/DGX, `~/scratch/miniconda`), and expected runtime and resources.
20. **Smoke check**: `python -m py_compile src/*.py` and `DRY_RUN=1 bash slurms/run_all.sh`, documented or run as a minimal CI job.
21. **Local `src/__pycache__` is Python 3.13, while `./.env` is 3.10.** Notebooks or helpers were run outside the pinned env. Note it in the README, or run everything from `./.env`.
22. **Palette control (categorical colours).** The house palette `PALETTE` (10 colours, ordered so that it also reads as a grey ramp) is the only allowed categorical palette, and CLAUDE.md asks for alternating picks.
    - *Fixed 2026-10-28:* the report and viewer assigned it sequentially (`PALETTE[i]`), so with five or fewer categories (species bars, selected clusters, focus clusters, viewer categories) neighbouring colours had similar grey tones and contrasted poorly. Both scripts now use `PALETTE_ALT = PALETTE[::2] + PALETTE[1::2]`.
    - *Open:* `PALETTE` is defined twice (`src/h5ad_scanpy.py`, `src/h5ad_interactive.py`). A single source, such as a small shared constants module or the YAML, would prevent drift.
    - *Fixed 2026-10-28:* species colours followed the input order in the report but the protein-count order in the viewer. Both now use `PALETTE_ALT` by input position (targets first, then probes), so the PDF barplots and the HTML views match.

## F. Recommended release layout

- **Code record** (GitHub tag → Zenodo, Apache-2.0): `src/`, `slurms/`, `nbs/`, `configs/`, `esmc_config.sh`, `README.md`, `LICENSE`, `CITATION.cff`, `.zenodo.json`, `CHANGELOG.md`, `DMP.tex`, `Audit.md`.
- **Data record(s)** (direct Zenodo upload, own DOI, ~2–4 GB):
  - `raw/proteomes` and `raw/data`, without the duplicates and without `raw/raw/`;
  - `data/sequences`, `data/features`, `data/dedup`, `data/clusters`;
  - `data/esm-reps`;
  - `data/scanpy`;
  - the metadata bundle (section G).

  One record under CC BY 4.0, with per-file sources in the manifest. Link code and data through `related_identifiers` (`isSupplementedBy` / `isSupplementTo`), and reference the curation sources by accession (TSA GKMC01/GKMD01, Zenodo 10278300) instead of redistributing them where licences are unclear.

## G. Recommendations for the next session: metadata tooling

Build this after A–E are resolved. The fixes change the hashes, file names and licences that the metadata records, so generating it earlier would be wasted work.

**Script `src/release_metadata.py` + `slurms/supp_03_metadata.sh`**, registered in `run_all.sh` after `supp_02`, following CLAUDE.md conventions:
- a `release_metadata` section in `param_configs.yaml`;
- reads only the configured roots, writes only into `release_dir`.

Outputs:
1. `MANIFEST.tsv`, one row per file: `path, bytes, sha256, md5, mtime (ISO 8601), step, artefact, format (EDAM), labels, species, model, param_tag, n_records, licence, description`.
2. `SHA256SUMS`, `MD5SUMS`, verifiable with `sha256sum -c`.
3. `DATA_DICTIONARY.tsv`: one row per column of every table type. Columns and dtypes are read from real headers; descriptions come from a hand-written registry (`specificity`, `chi2_q`, `fold_change`, … as defined in the README). Columns missing from the registry are reported.
4. `PROVENANCE.json`:
   - git commit and dirty flag;
   - Python and package versions;
   - `mmseqs version`;
   - the Hugging Face cache revision and download date (snapshot directory mtime) of all three models: 300M, 600M and 6B. This confirms the revisions in `DMP.tex` §1.2;
   - every h5ad `uns["params"]` (read with `h5py`);
   - the `.params` sidecars;
   - the frozen YAML and TSV configs;
   - the UniProt release (flagged as manifest time, not download time).
5. `ro-crate-metadata.json` (RO-Crate 1.1): files, software (pinned to the commit), and one `CreateAction` per pipeline step linking inputs to outputs.
6. `NAMING_CONVENTIONS.md`: generated from the same pattern registry the parser uses, so the documentation cannot drift.

Core design:
- an ordered `PATTERNS` registry `(regex, step, artefact, format, licence_rule, description)` covering every file family in section C;
- unmatched files reported as `UNCLASSIFIED`, which doubles as a convention-coverage check;
- licences derived from the source;
- streamed record counts (FASTA `>`, table rows, h5ad `n_obs`);
- chunked hashing with `tqdm`;
- duplicate detection by sha256.

To decide before writing it: whether the v1.1 renames (section C) happen first, so that the registry targets the final names only.

## Priority summary

| priority | items |
|---|---|
| P0 before tagging | A1–A4, A6 (A5 resolved) |
| P1 before the data DOI | B7–B11, D13–D15 (B12 resolved) |
| P1 documentation | C (document now, rename in v1.1) |
| P2 | E16–E22 |
| next session | G (metadata tooling) |
