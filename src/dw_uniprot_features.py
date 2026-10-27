#!/usr/bin/env python3
"""Downloads the UniProtKB annotation fields (everything except the sequence) for every row of the query table, or for one ``--query``.

The output is ``<features_dir>/<label>_feats.tsv`` with one column per field of FEATURE_FIELDS. Rows whose query is a local FASTA path (.fa/.faa/.fasta) have no UniProt annotation and are skipped. A label whose output exists is skipped unless ``--overwrite`` is set.
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
FEATURES_DIR = "data/features"
FEATURE_FIELDS = [f for f in ["accession", "id", "uniparc_id", "sequence_version", "version", "date_created", "date_modified", "date_sequence_modified", "gene_names", "gene_primary", "gene_synonym", "gene_oln", "gene_orf", "protein_name", "protein_existence", "reviewed", "fragment", "organism_name", "organism_id", "lineage", "lineage_ids", "virus_hosts", "sequence", "length", "mass", "encoded_in", "ft_act_site", "ft_binding", "ft_carbohyd", "ft_chain", "ft_coiled", "ft_compbias", "ft_conflict", "ft_crosslnk", "ft_disulfid", "ft_dna_bind", "ft_domain", "ft_helix", "ft_init_met", "ft_intramem", "ft_lipid", "ft_mod_res", "ft_motif", "ft_mutagen", "ft_non_cons", "ft_non_std", "ft_non_ter", "ft_peptide", "ft_propep", "ft_region", "ft_repeat", "ft_signal", "ft_site", "ft_strand", "ft_topo_dom", "ft_transit", "ft_transmem", "ft_turn", "ft_unsure", "ft_var_seq", "ft_variant", "ft_zn_fing", "ec", "rhea", "cc_catalytic_activity", "cc_cofactor", "cc_activity_regulation", "kinetics", "ph_dependence", "redox_potential", "temp_dependence", "absorption", "cc_function", "cc_pathway", "cc_miscellaneous", "cc_caution", "cc_sequence_caution", "cc_polymorphism", "cc_rna_editing", "cc_alternative_products", "cc_sc_epred", "cc_interaction", "cc_subunit", "cc_subcellular_location", "cc_developmental_stage", "cc_induction", "cc_tissue_specificity", "cc_allergen", "cc_biotechnology", "cc_disruption_phenotype", "cc_disease", "cc_pharmaceutical", "cc_toxic_dose", "cc_ptm", "cc_domain", "protein_families", "go", "go_id", "go_p", "go_c", "go_f", "xref_proteomes", "structure_3d", "lit_pubmed_id", "keywordid", "keyword", "annotation_score", "comment_count", "feature_count", "tools"] if f != "sequence"]
LOCAL_FASTA_SUFFIXES = (".fa", ".faa", ".fasta")
OVERWRITE = False
PROBE_ACCESSION = "A0A0R4I9Y1"
QUERY_TABLE = "configs/query_configs.tsv"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Download UniProtKB annotation fields")
parser.add_argument("--query", default=None, help="Single UniProtKB query; when omitted every row of the query table is downloaded")
parser.add_argument("--label", default=None, help="Label of --query, or the only table row to download")
parser.add_argument("--out_tsv", type=Path, default=None, help="Output TSV path (default <features_dir>/<label>_feats.tsv)")
parser.add_argument("--query_table", type=str, default=QUERY_TABLE, help="TSV with label, query and exact_species columns")
parser.add_argument("--features_dir", type=str, default=FEATURES_DIR, help="Output folder")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Download again labels whose output exists")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "dw_uniprot_features"
params = dict(query_table=QUERY_TABLE, features_dir=FEATURES_DIR, overwrite=OVERWRITE)
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
    """Stream the UniProtKB entries matching a query as a string table.

    Args:
        query (str): UniProtKB query string.
        fields (list[str]): UniProtKB return fields.

    Returns:
        pd.DataFrame: One row per entry, with UniProt's display headers as columns.
    """
    url = f"{UNIPROT_STREAM_URL}?query={urllib.parse.quote(query, safe='')}&format=tsv&fields={','.join(fields)}"
    print(f"[PYTHON-INFO] Query URL: {url}")
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(StringIO(response.read().decode("utf-8")), sep="\t", dtype=str)
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
            print(f"[PYTHON-INFO] '{label}' is a local FASTA source without UniProt annotation. Skipping.")
            continue
        jobs.append((label, str(query).strip()))
    return jobs
###################################### MAIN
def main() -> None:
    """Download the annotation fields of every requested label."""
    start_time = time.time()
    column_names = uniprot_column_names(FEATURE_FIELDS)
    for label, query in tqdm(query_jobs(), desc="Feature downloads"):
        out_tsv = args.out_tsv or Path(args.features_dir) / f"{label}_feats.tsv"
        if not args.overwrite and out_tsv.exists():
            print(f"[PYTHON-INFO] Features of '{label}' already exist. Skipping.")
            continue
        if out_tsv.exists(): print(f"[PYTHON-INFO] Overwriting {out_tsv}")
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        records = fetch_uniprot(query, FEATURE_FIELDS).rename(columns=column_names)
        records.to_csv(out_tsv, sep="\t", index=False)
        print(f"[PYTHON-INFO][SAVE] {label}: {len(records):,} entries x {records.shape[1]} fields -> {out_tsv}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
