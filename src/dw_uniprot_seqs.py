#!/usr/bin/env python3
"""Download UniProtKB sequences for text queries as TSV, FASTA and accession lists.

Optional data source: any FASTA placed in ``data/sequences/`` is equally valid input
for the pipeline, which starts at ``run_mmseqs2.py``.
"""
############### <imports>
import argparse
import sys
import time
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from utils import add_yaml_args, log, log_resources, resolve_params

############### <constants>
SCRIPT_NAME = Path(__file__).stem
DEFAULTS = {"config_path": "configs/query_configs.tsv", "seq_out_dir": "data/sequences", "timeout": 600}
FIELDS = ["accession", "sequence", "length"]
MOCK_QUERY = "(accession:A0A0R4I9Y1)"


############### <config>
def load_config(path: Path) -> pd.DataFrame:
    """Load the query table.

    Args:
        path: TSV with at least ``label`` and ``query`` columns.

    Returns:
        Query table with blank cells set to NA.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        log(f"Error: Configuration file {path} not found.")
        raise FileNotFoundError(f"Configuration file {path} not found.")
    return pd.read_csv(path, sep="\t", dtype=str).replace(r"^\s*$", pd.NA, regex=True)


############### <uniprot>
def fetch_uniprot(query: str, fields: list[str], timeout: int) -> pd.DataFrame:
    """Stream a UniProtKB query result as a table.

    Args:
        query: UniProt query string.
        fields: UniProt return fields.
        timeout: Request timeout in seconds.

    Returns:
        Result table with UniProt's display column names.
    """
    url = f"https://rest.uniprot.org/uniprotkb/stream?query={urllib.parse.quote(query, safe='')}&format=tsv&fields={','.join(fields)}"
    log(f"Query sequence URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return pd.read_csv(StringIO(resp.read().decode("utf-8")), sep="\t")


def get_feature_map(fields: list[str], timeout: int) -> dict[str, str]:
    """Map UniProt display column names back to field names using a one-entry query.

    Args:
        fields: UniProt return fields.
        timeout: Request timeout in seconds.

    Returns:
        Mapping from display name to field name.
    """
    mock = fetch_uniprot(MOCK_QUERY, fields, timeout)
    return dict(zip(mock.columns, fields))


############### <io>
def save_outputs(df: pd.DataFrame, fp_seqs: Path, fp_fasta: Path, fp_ids: Path) -> None:
    """Write the sequence table, FASTA and accession list.

    Args:
        df: Table with ``accession`` and ``sequence`` columns.
        fp_seqs: Output TSV.
        fp_fasta: Output FASTA.
        fp_ids: Output accession list.
    """
    for path in (fp_seqs, fp_fasta, fp_ids):
        if path.exists():
            log(f"Overwriting existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp_seqs, sep="\t", index=False)
    with open(fp_fasta, "w", encoding="utf-8") as handle:
        for acc, seq in zip(df["accession"], df["sequence"]):
            if pd.notna(acc) and pd.notna(seq):
                acc, seq = str(acc).strip(), str(seq).strip()
                handle.write(f"{acc if acc.startswith('>') else f'>{acc}'}\n{seq}\n")
    ids = df["accession"].dropna().astype(str).str.strip()
    fp_ids.write_text("\n".join(ids) + ("\n" if not ids.empty else ""), encoding="utf-8")
    log(f"Saved {len(df)} entries to {fp_seqs}, {fp_fasta}, {fp_ids}")


############### <main>
def main() -> None:
    """Download sequences for one ``--query`` or for every row of the query table."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", help="UniProt query; if omitted, queries are read from config_path")
    parser.add_argument("--label", help="Output label for --query, or the only table row to process")
    parser.add_argument("--fp_seqs", type=Path)
    parser.add_argument("--fp_fasta", type=Path)
    parser.add_argument("--fp_ids", type=Path)
    parser.add_argument("--config_path")
    parser.add_argument("--seq_out_dir")
    parser.add_argument("--timeout", type=int)
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    if args.yaml_out and not args.query and not args.label:
        sys.exit(0)
    out_dir, timeout = Path(params["seq_out_dir"]), params["timeout"]
    if args.query:
        jobs = [(args.label or "example", args.query)]
    else:
        df_config = load_config(Path(params["config_path"]))
        if df_config.empty:
            log("Error: Configuration file is empty.")
            raise ValueError("Configuration file is empty.")
        jobs = []
        for item in df_config.to_dict("records"):
            label, query = item.get("label"), item.get("query")
            if args.label and label != args.label:
                continue
            if pd.isna(query) or not str(query).strip():
                log(f"Error: query missing for label '{label}'.")
                raise ValueError(f"query missing for label '{label}'.")
            jobs.append((label, str(query).strip()))
    feature_map = get_feature_map(FIELDS, timeout)
    for label, query in tqdm(jobs, desc="[PYTHON-INFO] Processing queries"):
        df = fetch_uniprot(query, FIELDS, timeout).rename(columns=feature_map)
        save_outputs(df, args.fp_seqs or out_dir / f"{label}_seqs.tsv", args.fp_fasta or out_dir / f"{label}_seqs.fasta", args.fp_ids or out_dir / f"{label}_ids.txt")
    log_resources(start_time)


if __name__ == "__main__":
    main()
