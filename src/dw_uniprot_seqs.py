#!/usr/bin/env python3
"""Downloads UniProtKB sequences for every row of the query table, or for one ``--query``, as TSV, FASTA and accession list.

Outputs are ``<sequences_dir>/<label>_seqs.tsv``, ``<label>_seqs.fasta`` and ``<label>_ids.txt``. Rows whose query is a local FASTA path (.fa/.faa/.fasta) are skipped here, because slurms/01_dw_uniprot.sh copies them. A label whose three outputs exist is skipped unless ``--overwrite`` is set.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
from io import StringIO
import pandas as pd
from pathlib import Path
import resource
import time
from tqdm import tqdm
import urllib.parse
import urllib.request
import yaml
FIELDS = ["accession", "sequence", "length"]
LOCAL_FASTA_SUFFIXES = (".fa", ".faa", ".fasta")
OVERWRITE = False
PROBE_ACCESSION = "A0A0R4I9Y1"
QUERY_TABLE = "configs/query_configs.tsv"
SEQUENCES_DIR = "data/sequences"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Download UniProtKB sequences")
parser.add_argument("--query", default=None, help="Single UniProtKB query; when omitted every row of the query table is downloaded")
parser.add_argument("--label", default=None, help="Label of --query, or the only table row to download")
parser.add_argument("--out_tsv", type=Path, default=None, help="Output TSV path (default <sequences_dir>/<label>_seqs.tsv)")
parser.add_argument("--out_fasta", type=Path, default=None, help="Output FASTA path (default <sequences_dir>/<label>_seqs.fasta)")
parser.add_argument("--out_ids", type=Path, default=None, help="Output accession list path (default <sequences_dir>/<label>_ids.txt)")
parser.add_argument("--query_table", type=str, default=QUERY_TABLE, help="TSV with label, query and exact_species columns")
parser.add_argument("--sequences_dir", type=str, default=SEQUENCES_DIR, help="Output folder")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Download again labels whose outputs exist")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "dw_uniprot_seqs"
params = dict(query_table=QUERY_TABLE, sequences_dir=SEQUENCES_DIR, overwrite=OVERWRITE)
if cli_preview.yaml_in and Path(cli_preview.yaml_in).exists():
    config_data = yaml.safe_load(Path(cli_preview.yaml_in).read_text()) or {}
    if script_key in config_data: params.update(config_data[script_key])
parser.set_defaults(**params)
args = parser.parse_args()
if args.yaml_out:
    config_data = yaml.safe_load(Path(args.yaml_out).read_text()) if Path(args.yaml_out).exists() else {}
    config_data[script_key] = {k: getattr(args, k) for k in params}
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml.dump(config_data, sort_keys=False))
###################################### HELPER_FUNCTIONS
def fetch_uniprot(query: str, fields: list[str]) -> pd.DataFrame:
    """Stream the UniProtKB entries matching a query as a table.

    Args:
        query (str): UniProtKB query string.
        fields (list[str]): UniProtKB return fields.

    Returns:
        pd.DataFrame: One row per entry, with UniProt's display headers as columns.
    """
    url = f"{UNIPROT_STREAM_URL}?query={urllib.parse.quote(query, safe='')}&format=tsv&fields={','.join(fields)}"
    print(f"[PYTHON-INFO] Query URL: {url}")
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(StringIO(response.read().decode("utf-8")), sep="\t")
def uniprot_column_names(fields: list[str]) -> dict[str, str]:
    """Map UniProt's display headers to the requested field names, by fetching one probe entry.

    Args:
        fields (list[str]): UniProtKB return fields.

    Returns:
        dict[str, str]: Display header -> field name.
    """
    return dict(zip(fetch_uniprot(f"(accession:{PROBE_ACCESSION})", fields).columns, fields))
def query_jobs() -> list[tuple[str, str]]:
    """List the (label, query) pairs to download: ``--query`` alone, or the table rows filtered by ``--label``.

    Returns:
        list[tuple[str, str]]: Label and UniProtKB query per download.
    """
    if args.query: return [(args.label or "query", args.query)]
    table = pd.read_csv(args.query_table, sep="\t", dtype=str).replace(r"^\s*$", pd.NA, regex=True)
    jobs = []
    for row in table.to_dict("records"):
        label, query = row.get("label"), row.get("query")
        if args.label and label != args.label: continue
        if pd.isna(query): raise ValueError(f"[PYTHON-ERROR] Query missing for label '{label}' in {args.query_table}.")
        if str(query).strip().endswith(LOCAL_FASTA_SUFFIXES):
            print(f"[PYTHON-INFO] '{label}' is a local FASTA source, copied by slurms/01_dw_uniprot.sh. Skipping.")
            continue
        jobs.append((label, str(query).strip()))
    return jobs
def save_outputs(records: pd.DataFrame, out_tsv: Path, out_fasta: Path, out_ids: Path) -> None:
    """Write the downloaded entries as TSV, FASTA and accession list.

    Args:
        records (pd.DataFrame): Entries with accession and sequence columns.
        out_tsv (Path): TSV output.
        out_fasta (Path): FASTA output.
        out_ids (Path): Accession list output.
    """
    for path in (out_tsv, out_fasta, out_ids):
        if path.exists(): print(f"[PYTHON-INFO] Overwriting {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_tsv, sep="\t", index=False)
    complete = records.dropna(subset=["accession", "sequence"])
    accessions, sequences = complete["accession"].astype(str).str.strip().str.lstrip(">"), complete["sequence"].astype(str).str.strip()
    out_fasta.write_text("".join(f">{acc}\n{seq}\n" for acc, seq in zip(accessions, sequences)), encoding="utf-8")
    out_ids.write_text("".join(f"{acc}\n" for acc in records["accession"].dropna().astype(str).str.strip()), encoding="utf-8")
###################################### MAIN
def main() -> None:
    """Download the sequences of every requested label."""
    start_time = time.time()
    column_names = uniprot_column_names(FIELDS)
    for label, query in tqdm(query_jobs(), desc="Sequence downloads"):
        out_dir = Path(args.sequences_dir)
        out_paths = (args.out_tsv or out_dir / f"{label}_seqs.tsv", args.out_fasta or out_dir / f"{label}_seqs.fasta", args.out_ids or out_dir / f"{label}_ids.txt")
        if not args.overwrite and all(p.exists() for p in out_paths):
            print(f"[PYTHON-INFO] Sequences of '{label}' already exist. Skipping.")
            continue
        records = fetch_uniprot(query, FIELDS).rename(columns=column_names)
        save_outputs(records, *out_paths)
        print(f"[PYTHON-INFO][SAVE] {label}: {len(records):,} entries -> {out_paths[1]}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
