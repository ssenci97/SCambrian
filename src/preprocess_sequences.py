#!/usr/bin/env python3
"""Commits the input sequences of an analysis: validation, single-line FASTA with sc_label/sc_type tags, probe/target overlap removal, per-sample MMseqs2 deduplication and the sc_metadata.tsv mapping.

Every row of the query table (``dw_uniprot_seqs.query_table``) is a sample: its ``label`` names every file, ``taxonomic_ref`` is its display name and ``input_type`` (``sc_type``) is ``target`` or ``probe``. For each sample, in row order, ``<sequences_dir>/<label>_seqs.fasta`` is read (wrapped records allowed) and every record is checked: upper-cased, a trailing ``*`` stripped, rejected as ``invalid`` when empty, duplicated within the sample, outside ``[min_length, max_length]`` or containing letters outside AMINO_ACIDS. Probe records whose id or sequence is already in a target are dropped as ``probe_overlap``, so that a probe can only point at target proteins, never at copies of them. Record ids shared by two samples stop the run, because the joint steps key on them. The valid records go to ``<preprocessed_dir>/<label>_clean.fasta`` as ``>{header} sc_label={label} sc_type={type}`` on one line and one line of sequence (existing tags replaced, so each appears once), rewritten only when its content changes so that downstream freshness checks hold. MMseqs2 easy-linclust at ``dedup_min_identity`` then writes ``<label>_dedup.fasta`` (the committed input of every later step), its member table and a ``.params`` sidecar, and is redone when the sidecar differs from the current parameters or the clean FASTA is newer. Finally ``<analyses_dir>/<analysis_name>/sc_metadata.tsv`` lists every input record with columns file, record, sc_label, sc_type, sample, taxonomic_ref, length, status (kept, invalid, probe_overlap, duplicate) and representative (its dedup representative). The analysis name is ``--analysis_name``, else the ANALYSIS_NAME environment variable set by slurms/run_all.sh, else ``{YYYY-MM-DD}_SCambrian``.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import os
import pandas as pd
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time
from tqdm import tqdm
import yaml
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
ANALYSES_DIR = "analyses"
ANALYSIS_NAME = None
ANALYSIS_SUFFIX = "SCambrian"
DEDUP_COVERAGE_MODE = 0
DEDUP_MIN_COVERAGE = 0.8
DEDUP_MIN_IDENTITY = 0.95
INPUT_TYPES = ("target", "probe")
LOCAL_FASTA_SUFFIXES = (".fa", ".faa", ".fasta")
MAX_LENGTH = None
METADATA_FILE = "sc_metadata.tsv"
MIN_LENGTH = 30
MMSEQS_EXECUTABLE = "mmseqs"
OVERWRITE = False
PREPROCESSED_DIR = "data/preprocessed"
QUERY_TABLE = "configs/query_configs.tsv"
SEQUENCES_DIR = "data/sequences"
TAG_RE = re.compile(r"\s*\bsc_(label|type)=\S*")
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Validate, tag and deduplicate the input sequences of an analysis")
parser.add_argument("--analysis_name", type=str, default=ANALYSIS_NAME, help="Analysis name (default: $ANALYSIS_NAME, else {YYYY-MM-DD}_SCambrian)")
parser.add_argument("--analyses_dir", type=str, default=ANALYSES_DIR, help="Base folder of the analyses")
parser.add_argument("--sequences_dir", type=str, default=SEQUENCES_DIR, help="Folder of the <label>_seqs.fasta inputs (step 01)")
parser.add_argument("--preprocessed_dir", type=str, default=PREPROCESSED_DIR, help="Folder of the clean and deduplicated FASTA files")
parser.add_argument("--min_length", type=int, default=MIN_LENGTH, help="Shortest valid sequence")
parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Longest valid sequence (None: no limit)")
parser.add_argument("--mmseqs_executable", type=str, default=MMSEQS_EXECUTABLE, help="MMseqs2 executable name or path")
parser.add_argument("--dedup_min_identity", type=float, default=DEDUP_MIN_IDENTITY, help="Minimum sequence identity (0-1) of two sequences merged by the dedup")
parser.add_argument("--dedup_min_coverage", type=float, default=DEDUP_MIN_COVERAGE, help="Minimum alignment coverage (0-1) of the dedup")
parser.add_argument("--dedup_coverage_mode", type=int, default=DEDUP_COVERAGE_MODE, help="MMseqs2 --cov-mode of the dedup (0: query and target)")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Redo the dedup even when it is fresh")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "preprocess_sequences"
params = dict(analysis_name=ANALYSIS_NAME, analyses_dir=ANALYSES_DIR, sequences_dir=SEQUENCES_DIR, preprocessed_dir=PREPROCESSED_DIR, min_length=MIN_LENGTH, max_length=MAX_LENGTH, mmseqs_executable=MMSEQS_EXECUTABLE, dedup_min_identity=DEDUP_MIN_IDENTITY, dedup_min_coverage=DEDUP_MIN_COVERAGE, dedup_coverage_mode=DEDUP_COVERAGE_MODE, overwrite=OVERWRITE)
config_data = {}
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    if script_key in config_data: params.update(config_data[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    out_data = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    out_data[script_key] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(out_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def load_samples() -> pd.DataFrame:
    """Read the samples of the query table and check labels and input types.

    Returns:
        pd.DataFrame: label, query, taxonomic_ref and input_type per sample, in row order.
    """
    table = pd.read_csv(config_data.get("dw_uniprot_seqs", {}).get("query_table", QUERY_TABLE), sep="\t", dtype=str).replace(r"^\s*$", pd.NA, regex=True).dropna(subset=["label"])
    table["label"] = table["label"].str.strip()
    table["taxonomic_ref"] = table["taxonomic_ref"].fillna(table["label"]) if "taxonomic_ref" in table else table["label"]
    table["input_type"] = table["input_type"].fillna("target").str.strip() if "input_type" in table else "target"
    if table["label"].duplicated().any(): raise ValueError(f"[PYTHON-ERROR] Labels must be unique: {sorted(table.loc[table['label'].duplicated(), 'label'])}")
    bad = table.loc[~table["input_type"].isin(INPUT_TYPES), "label"].tolist()
    if bad: raise ValueError(f"[PYTHON-ERROR] input_type must be one of {INPUT_TYPES}: {bad}")
    if "target" not in set(table["input_type"]): raise ValueError("[PYTHON-ERROR] At least one target sample is required.")
    return table.reset_index(drop=True)
def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA file with wrapped or single-line records.

    Args:
        path (Path): FASTA file.

    Returns:
        list[tuple[str, str]]: (header without ``>``, sequence) per record.
    """
    records, header, chunks = [], None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None: records.append((header, "".join(chunks)))
                header, chunks = line[1:].strip(), []
            elif line and header is not None: chunks.append(line)
    if header is not None: records.append((header, "".join(chunks)))
    return records
def check_record(sequence: str) -> str | None:
    """Clean one sequence and tell whether it is valid.

    Args:
        sequence (str): Raw sequence.

    Returns:
        str | None: Upper-case sequence without terminal stop, or None when invalid.
    """
    seq = sequence.upper().rstrip("*")
    if not seq or set(seq) - AMINO_ACIDS or len(seq) < args.min_length or (args.max_length and len(seq) > args.max_length): return None
    return seq
def tagged_header(header: str, label: str, sc_type: str) -> str:
    """Append the sc_label and sc_type tags to a header, replacing any existing ones.

    Args:
        header (str): Original header without ``>``.
        label (str): Sample label.
        sc_type (str): ``target`` or ``probe``.

    Returns:
        str: Header carrying each tag exactly once.
    """
    return f"{TAG_RE.sub('', header).strip()} sc_label={label} sc_type={sc_type}"
def write_if_changed(path: Path, text: str) -> bool:
    """Write a text file only when its content differs, keeping the modification time of unchanged files.

    Args:
        path (Path): Output file.
        text (str): Content.

    Returns:
        bool: True when the file was written.
    """
    if path.exists() and path.read_text(encoding="utf-8") == text: return False
    path.write_text(text, encoding="utf-8")
    return True
def dedup_tag() -> str:
    """Return the parameter tag of the dedup.

    Returns:
        str: Tag such as ``dmin0.95_dcov0.8_dmode0``.
    """
    return f"dmin{args.dedup_min_identity}_dcov{args.dedup_min_coverage}_dmode{args.dedup_coverage_mode}"
###################################### PIPELINE_STEPS
def clean_sample(row: dict, target_ids: set[str], target_seqs: set[str]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Validate the records of one sample and drop probe records already present in a target.

    Args:
        row (dict): Sample row of the query table.
        target_ids (set[str]): Record ids of the valid target records seen so far.
        target_seqs (set[str]): Sequences of the valid target records seen so far.

    Returns:
        tuple[list[dict], list[tuple[str, str]]]: Metadata rows of every input record, and (tagged header, sequence) of the valid ones.
    """
    label, sc_type = row["label"], row["input_type"]
    in_fasta = Path(args.sequences_dir) / f"{label}_seqs.fasta"
    if not in_fasta.exists(): raise FileNotFoundError(f"[PYTHON-ERROR] Missing {in_fasta}; run step 01 first.")
    query = str(row.get("query", "")).strip()
    source = query if query.endswith(LOCAL_FASTA_SUFFIXES) else in_fasta.as_posix()
    rows, kept, seen = [], [], set()
    for header, raw_seq in read_fasta(in_fasta):
        record = header.split()[0] if header.split() else ""
        seq = check_record(raw_seq)
        status = "invalid" if seq is None or not record or record in seen else "kept"
        if status == "kept" and sc_type == "probe" and (record in target_ids or seq in target_seqs): status = "probe_overlap"
        seen.add(record)
        rows.append(dict(file=source, record=record, sc_label=label, sc_type=sc_type, sample=label, taxonomic_ref=row["taxonomic_ref"], length=len(raw_seq), status=status, representative=pd.NA))
        if status == "kept": kept.append((tagged_header(header, label, sc_type), seq))
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    print(f"[PYTHON-INFO][CLEAN] {label} ({sc_type}): {len(rows):,} records, {counts}")
    return rows, kept
def dedup(label: str, clean_fasta: Path) -> Path:
    """Deduplicate the clean records of a sample with easy-linclust, unless its fresh output exists.

    Args:
        label (str): Sample label.
        clean_fasta (Path): ``<label>_clean.fasta``.

    Returns:
        Path: Member table (representative, member) of the dedup.
    """
    out_dir = Path(args.preprocessed_dir)
    out_fasta, members, sidecar = out_dir / f"{label}_dedup.fasta", out_dir / f"{label}_dedup_members.tsv", out_dir / f"{label}_dedup.params"
    outputs = (out_fasta, members, sidecar)
    if not args.overwrite and all(p.exists() for p in outputs) and min(p.stat().st_mtime for p in outputs) >= clean_fasta.stat().st_mtime and sidecar.read_text().strip() == dedup_tag():
        print(f"[PYTHON-INFO] Dedup of '{label}' is fresh ({dedup_tag()}). Skipping.")
        return members
    prefix, work_dir = out_dir / f"{label}_tmp", out_dir / f"tmp_{label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.mmseqs_executable, "easy-linclust", str(clean_fasta), str(prefix), str(work_dir), "--min-seq-id", str(args.dedup_min_identity), "-c", str(args.dedup_min_coverage), "--cov-mode", str(args.dedup_coverage_mode), "--threads", str(THREADS)], check=True)
    Path(f"{prefix}_rep_seq.fasta").rename(out_fasta)
    Path(f"{prefix}_cluster.tsv").rename(members)
    for p in out_dir.glob(f"{label}_tmp*"):
        if p.is_file(): p.unlink()
    shutil.rmtree(work_dir, ignore_errors=True)
    sidecar.write_text(dedup_tag() + "\n")
    print(f"[PYTHON-INFO][SAVE] {out_fasta} ({dedup_tag()})")
    return members
###################################### MAIN
def main() -> None:
    """Clean, tag and deduplicate every sample and write the analysis metadata."""
    start_time = time.time()
    if not shutil.which(args.mmseqs_executable) and not Path(args.mmseqs_executable).exists(): raise FileNotFoundError(f"[PYTHON-ERROR] MMseqs2 executable '{args.mmseqs_executable}' not found.")
    analysis_name = args.analysis_name or os.environ.get("ANALYSIS_NAME") or f"{datetime.date.today().isoformat()}_{ANALYSIS_SUFFIX}"
    analysis_dir = Path(args.analyses_dir) / analysis_name
    print(f"[PYTHON-INFO] Analysis {analysis_name} -> {analysis_dir}")
    samples = load_samples()
    samples = pd.concat([samples[samples["input_type"] == "target"], samples[samples["input_type"] == "probe"]])
    Path(args.preprocessed_dir).mkdir(parents=True, exist_ok=True)
    target_ids, target_seqs, owner, tables = set(), set(), {}, []
    for row in tqdm(samples.to_dict("records"), desc="Clean samples"):
        rows, kept = clean_sample(row, target_ids, target_seqs)
        clashes = sorted({h.split()[0] for h, _ in kept} & owner.keys())
        if clashes: raise ValueError(f"[PYTHON-ERROR] {len(clashes)} record ids of '{row['label']}' already used by {sorted({owner[c] for c in clashes})} (e.g. {clashes[:3]}); record ids must be unique across samples.")
        owner |= {h.split()[0]: row["label"] for h, _ in kept}
        if row["input_type"] == "target":
            target_ids |= {h.split()[0] for h, _ in kept}
            target_seqs |= {s for _, s in kept}
        clean_fasta = Path(args.preprocessed_dir) / f"{row['label']}_clean.fasta"
        if write_if_changed(clean_fasta, "".join(f">{h}\n{s}\n" for h, s in kept)): print(f"[PYTHON-INFO][SAVE] {clean_fasta} ({len(kept):,} records)")
        members = pd.read_csv(dedup(row["label"], clean_fasta), sep="\t", header=None, names=["representative", "record"], dtype=str)
        table = pd.DataFrame(rows)
        representative = table["record"].map(dict(zip(members["record"], members["representative"])))
        table["representative"] = representative.where(table["status"] == "kept")
        table.loc[(table["status"] == "kept") & (table["representative"] != table["record"]), "status"] = "duplicate"
        print(f"[PYTHON-INFO][DEDUP] {row['label']}: {(table['status'] == 'kept').sum():,} kept, {(table['status'] == 'duplicate').sum():,} duplicates")
        tables.append(table)
    order = {lbl: i for i, lbl in enumerate(load_samples()["label"])}
    metadata = pd.concat(tables, ignore_index=True).sort_values("sc_label", key=lambda s: s.map(order), kind="stable")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(analysis_dir / METADATA_FILE, sep="\t", index=False)
    print(f"[PYTHON-INFO][SAVE] {analysis_dir / METADATA_FILE}: {len(metadata):,} records, {metadata['status'].value_counts().to_dict()}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
