#!/usr/bin/env python3
"""Downloads, for every UniProtKB target of the query table, the accessions that match the functional focus clause.

The functional focus is a documented, arbitrary choice made once in ``focus_query`` (default: immune system process, GO:0006955 with its descendants as expanded by UniProt, or the keywords KW-0391 Immunity and KW-0399 Innate immunity; KW-0393 Immunoglobulin domain is left out on purpose because it marks a fold, not a function). For each row with ``input_type`` target (the default) and a UniProtKB query, the accessions of ``(<query>) AND (<focus_query>)`` are written to ``<features_dir>/<label>_focus_ids.txt``, with a ``<label>_focus_ids.params`` sidecar recording the focus, the UniProt release and the retrieval date. Targets from a local FASTA carry no annotation and get no focus tier. Nothing is fetched in comparative mode, when the query table has no probe row. Probe rows are not downloaded here; a warning is printed when their query does not contain the focus clause. A label is fetched again when its outputs are missing, when the sidecar records another focus clause or query, or with ``--overwrite``.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import pandas as pd
from pathlib import Path
import resource
import time
from tqdm import tqdm
import urllib.parse
import urllib.request
import yaml
FEATURES_DIR = "data/features"
FOCUS_NAME = "immune"
FOCUS_QUERY = "(go:0006955) OR (keyword:KW-0391) OR (keyword:KW-0399)"
INPUT_TYPES = ("target", "probe")
LOCAL_FASTA_SUFFIXES = (".fa", ".faa", ".fasta")
OVERWRITE = False
QUERY_TABLE = "configs/query_configs.tsv"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Download the focus accessions of every UniProtKB target")
parser.add_argument("--label", default=None, help="Only this table row")
parser.add_argument("--query_table", type=str, default=QUERY_TABLE, help="TSV with label, query, exact_species and input_type columns")
parser.add_argument("--features_dir", type=str, default=FEATURES_DIR, help="Output folder of <label>_focus_ids.txt")
parser.add_argument("--focus_name", type=str, default=FOCUS_NAME, help="Short name of the functional focus")
parser.add_argument("--focus_query", type=str, default=FOCUS_QUERY, help="UniProtKB clause defining the functional focus")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Download again labels whose outputs are fresh")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "dw_uniprot_focus"
params = dict(query_table=QUERY_TABLE, features_dir=FEATURES_DIR, focus_name=FOCUS_NAME, focus_query=FOCUS_QUERY, overwrite=OVERWRITE)
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
def fetch_accessions(query: str) -> tuple[list[str], str, str]:
    """Stream the accessions of the UniProtKB entries matching a query.

    Args:
        query (str): UniProtKB query string.

    Returns:
        tuple[list[str], str, str]: Accessions, UniProt release and release date.
    """
    url = f"{UNIPROT_STREAM_URL}?query={urllib.parse.quote(query, safe='')}&format=list"
    print(f"[PYTHON-INFO] Query URL: {url}")
    with urllib.request.urlopen(url) as response:
        release, release_date = response.headers.get("X-UniProt-Release", "unknown"), response.headers.get("X-UniProt-Release-Date", "unknown")
        return [a.strip() for a in response.read().decode("utf-8").splitlines() if a.strip()], release, release_date
def focus_jobs() -> list[tuple[str, str]]:
    """List the (label, query) pairs of the UniProtKB targets, logging the rows that get no focus tier.

    Returns:
        list[tuple[str, str]]: Label and UniProtKB query of every target to fetch.
    """
    table = pd.read_csv(args.query_table, sep="\t", dtype=str).replace(r"^\s*$", pd.NA, regex=True)
    if "input_type" not in table or not (table["input_type"].astype(str).str.strip() == "probe").any():
        print("[PYTHON-INFO][MODE] Comparative mode (no probe input in the query table): focus lists are not needed.")
        return []
    jobs = []
    for row in table.to_dict("records"):
        label, query = row.get("label"), str(row.get("query") or "").strip()
        input_type = str(row.get("input_type") if pd.notna(row.get("input_type")) else "target").strip()
        if args.label and label != args.label: continue
        if input_type not in INPUT_TYPES: raise ValueError(f"[PYTHON-ERROR] input_type '{input_type}' of '{label}' is not one of {INPUT_TYPES}.")
        if input_type == "probe":
            if args.focus_query not in query: print(f"[PYTHON-WARNING] Probe '{label}' query does not contain the focus clause '{args.focus_query}'.")
            continue
        if query.endswith(LOCAL_FASTA_SUFFIXES):
            print(f"[PYTHON-INFO] Target '{label}' is a local FASTA without annotation: no focus tier.")
            continue
        jobs.append((label, query))
    return jobs
def is_fresh(ids_path: Path, params_path: Path, query: str) -> bool:
    """Check that a focus list exists and was built with the current focus clause and query.

    Args:
        ids_path (Path): Accession list.
        params_path (Path): Sidecar of the accession list.
        query (str): UniProtKB query of the target.

    Returns:
        bool: True when both files exist and the sidecar matches.
    """
    if not (ids_path.exists() and params_path.exists()): return False
    stored = yaml.safe_load(params_path.read_text()) or {}
    return stored.get("focus_query") == args.focus_query and stored.get("target_query") == query
###################################### MAIN
def main() -> None:
    """Fetch the focus accessions of every UniProtKB target."""
    start_time = time.time()
    out_dir = Path(args.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, query in tqdm(focus_jobs(), desc="Focus downloads"):
        ids_path, params_path = out_dir / f"{label}_focus_ids.txt", out_dir / f"{label}_focus_ids.params"
        if not args.overwrite and is_fresh(ids_path, params_path, query):
            print(f"[PYTHON-INFO] Focus list of '{label}' is fresh. Skipping.")
            continue
        full_query = f"({query}) AND ({args.focus_query})"
        accessions, release, release_date = fetch_accessions(full_query)
        ids_path.write_text("".join(f"{a}\n" for a in accessions), encoding="utf-8")
        params_path.write_text(yaml.dump(dict(focus_name=args.focus_name, focus_query=args.focus_query, target_query=query, full_query=full_query, n_accessions=len(accessions), uniprot_release=release, uniprot_release_date=release_date, retrieved=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), sort_keys=False))
        print(f"[PYTHON-INFO][SAVE] {label}: {len(accessions):,} {args.focus_name} accessions (UniProt {release}) -> {ids_path}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
