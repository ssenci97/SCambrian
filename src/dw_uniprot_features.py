#!/usr/bin/env python3
"""Download UniProtKB annotation features (all fields except the sequence) for text queries."""
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
DEFAULTS = {"config_path": "configs/query_configs.tsv", "features_out_dir": "data/features", "timeout": 600}
MOCK_QUERY = "(accession:A0A0R4I9Y1)"
ALL_FIELDS = ["accession", "id", "uniparc_id", "sequence_version", "version", "date_created", "date_modified", "date_sequence_modified", "gene_names", "gene_primary", "gene_synonym", "gene_oln", "gene_orf", "protein_name", "protein_existence", "reviewed", "fragment", "organism_name", "organism_id", "lineage", "lineage_ids", "virus_hosts", "sequence", "length", "mass", "encoded_in", "ft_act_site", "ft_binding", "ft_carbohyd", "ft_chain", "ft_coiled", "ft_compbias", "ft_conflict", "ft_crosslnk", "ft_disulfid", "ft_dna_bind", "ft_domain", "ft_helix", "ft_init_met", "ft_intramem", "ft_lipid", "ft_mod_res", "ft_motif", "ft_mutagen", "ft_non_cons", "ft_non_std", "ft_non_ter", "ft_peptide", "ft_propep", "ft_region", "ft_repeat", "ft_signal", "ft_site", "ft_strand", "ft_topo_dom", "ft_transit", "ft_transmem", "ft_turn", "ft_unsure", "ft_var_seq", "ft_variant", "ft_zn_fing", "ec", "rhea", "cc_catalytic_activity", "cc_cofactor", "cc_activity_regulation", "kinetics", "ph_dependence", "redox_potential", "temp_dependence", "absorption", "cc_function", "cc_pathway", "cc_miscellaneous", "cc_caution", "cc_sequence_caution", "cc_polymorphism", "cc_rna_editing", "cc_alternative_products", "cc_sc_epred", "cc_interaction", "cc_subunit", "cc_subcellular_location", "cc_developmental_stage", "cc_induction", "cc_tissue_specificity", "cc_allergen", "cc_biotechnology", "cc_disruption_phenotype", "cc_disease", "cc_pharmaceutical", "cc_toxic_dose", "cc_ptm", "cc_domain", "protein_families", "go", "go_id", "go_p", "go_c", "go_f", "xref_proteomes", "structure_3d", "lit_pubmed_id", "keywordid", "keyword", "annotation_score", "comment_count", "feature_count", "tools"]
FIELDS_NOSEQ = [f for f in ALL_FIELDS if f != "sequence"]


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
    """Stream a UniProtKB query result as a string-typed table.

    Args:
        query: UniProt query string.
        fields: UniProt return fields.
        timeout: Request timeout in seconds.

    Returns:
        Result table with UniProt's display column names.
    """
    url = f"https://rest.uniprot.org/uniprotkb/stream?query={urllib.parse.quote(query, safe='')}&format=tsv&fields={','.join(fields)}"
    log(f"Query feature URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return pd.read_csv(StringIO(resp.read().decode("utf-8")), sep="\t", dtype=str)


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
def save_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a table as TSV, creating parent directories.

    Args:
        df: Table to write.
        path: Output TSV.
    """
    if path.exists():
        log(f"Overwriting existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    log(f"Saved {len(df)} entries to {path}")


############### <main>
def main() -> None:
    """Download features for one ``--query`` or for every row of the query table."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", help="UniProt query; if omitted, queries are read from config_path")
    parser.add_argument("--label", help="Output label for --query, or the only table row to process")
    parser.add_argument("--fp_feats", type=Path)
    parser.add_argument("--config_path")
    parser.add_argument("--features_out_dir")
    parser.add_argument("--timeout", type=int)
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    if args.yaml_out and not args.query and not args.label:
        sys.exit(0)
    out_dir, timeout = Path(params["features_out_dir"]), params["timeout"]
    if args.query:
        jobs = [(args.label or "example", args.query)]
    else:
        df_config = load_config(Path(params["config_path"]))
        if df_config.empty:
            log(f"Error: Configuration file {params['config_path']} is empty.")
            raise ValueError(f"Configuration file {params['config_path']} is empty.")
        jobs = []
        for item in df_config.to_dict("records"):
            label, query = item.get("label"), item.get("query")
            if args.label and label != args.label:
                continue
            if pd.isna(query) or not str(query).strip():
                log(f"Error: query missing for label '{label}'.")
                raise ValueError(f"query missing for label '{label}'.")
            jobs.append((label, str(query).strip()))
    feature_map = get_feature_map(ALL_FIELDS, timeout)
    for label, query in tqdm(jobs, desc="[PYTHON-INFO] Processing features"):
        df = fetch_uniprot(query, FIELDS_NOSEQ, timeout).rename(columns=feature_map)
        save_tsv(df, args.fp_feats or out_dir / f"{label}_feats.tsv")
    log_resources(start_time)


if __name__ == "__main__":
    main()
