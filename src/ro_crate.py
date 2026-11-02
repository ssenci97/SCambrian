#!/usr/bin/env python3
"""Writes an RO-Crate 1.1 JSON-LD description (``ro-crate-metadata.json``) of one analysis folder.

The analysis folder ``analyses/<analysis_name>/`` holds sc_metadata.tsv, the joint MMseqs2 clusters and one folder per model with the projections, the summary and the reports. The query table and the parameter YAML in use are copied to ``<analysis>/configs/`` so that the crate is self-contained. The root Dataset lists every file of the folder with its format, size and modification date, and every subfolder (``mmseqs_joint/``, ``<model>/``) is a Dataset produced by a CreateAction whose instrument is the pipeline at the current git commit and whose objects are the copied configs. The crate is rewritten at every call, and a copy of it and of ``configs/`` goes to ``<metadata_dir>/<analysis>/``, which is tracked by git while ``analyses/`` is not.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import json
import pandas as pd
from pathlib import Path
import resource
import shutil
import subprocess
import time
from tqdm import tqdm
import yaml
CONFIGS_SUBDIR = "configs"
CRATE_FILE = "ro-crate-metadata.json"
CRATE_PROFILE = "https://w3id.org/ro/crate/1.1"
ENCODING_FORMATS = {".csv": "text/csv", ".h5ad": "application/x-hdf5", ".html": "text/html", ".json": "application/json", ".pdf": "application/pdf", ".png": "image/png", ".svg": "image/svg+xml", ".tsv": "text/tab-separated-values", ".txt": "text/plain", ".yaml": "application/yaml"}
LICENSE = None
METADATA_DIR = "metadata"
QUERY_TABLE = "configs/query_configs.tsv"
REPO_URL = "https://github.com/ssenci97/SCambrian"
YAML_PATH = "configs/param_configs.yaml"
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="Write the RO-Crate metadata of one analysis folder")
parser.add_argument("--analysis_dir", type=str, required=True, help="Analysis folder, analyses/<analysis>/")
parser.add_argument("--repo_url", type=str, default=REPO_URL, help="Public URL of the pipeline repository")
parser.add_argument("--metadata_dir", type=str, default=METADATA_DIR, help="Tracked folder receiving a copy of the crate and its configs, as <metadata_dir>/<analysis>/")
parser.add_argument("--license", type=str, default=LICENSE, help="License URL of the analysis outputs (omitted when None)")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "ro_crate"
params = dict(repo_url=REPO_URL, metadata_dir=METADATA_DIR, license=LICENSE)
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
def git_commit() -> str:
    """Return the current git commit of the pipeline, or ``unknown`` outside a repository.

    Returns:
        str: Commit hash.
    """
    try: return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): return "unknown"
def file_entity(path: Path, root: Path) -> dict:
    """Describe one file of the crate.

    Args:
        path (Path): File inside the analysis folder.
        root (Path): Analysis folder.

    Returns:
        dict: RO-Crate File entity.
    """
    stat = path.stat()
    entity = {"@id": path.relative_to(root).as_posix(), "@type": "File", "name": path.name, "contentSize": str(stat.st_size), "dateModified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")}
    if path.suffix in ENCODING_FORMATS: entity["encodingFormat"] = ENCODING_FORMATS[path.suffix]
    return entity
def copy_configs(root: Path, query_table: Path) -> list[Path]:
    """Copy the query table and the parameter YAML in use into the crate.

    Args:
        root (Path): Analysis folder.
        query_table (Path): Query table of the analysis.

    Returns:
        list[Path]: Copied files.
    """
    out_dir = root / CONFIGS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    copies = []
    for src in (query_table, Path(args.yaml_in) if args.yaml_in else None):
        if src is not None and src.exists():
            shutil.copy2(src, out_dir / src.name)
            copies.append(out_dir / src.name)
    return copies
def describe_inputs(query_table: Path) -> str:
    """Summarise the inputs of the analysis from its query table.

    Args:
        query_table (Path): Query table of the analysis.

    Returns:
        str: One clause per label with species and input type.
    """
    if not query_table.exists(): return "query table not found"
    table = pd.read_csv(query_table, sep="\t", dtype=str).fillna("")
    return "; ".join(f"{r.get('label', '')} ({r.get('taxonomic_ref', '')}, {r.get('input_type', '') or 'target'})" for r in table.to_dict("records"))
###################################### MAIN
def main() -> None:
    """Write ``ro-crate-metadata.json`` in the analysis folder."""
    start_time = time.time()
    root = Path(args.analysis_dir)
    if not root.is_dir(): raise SystemExit(f"[PYTHON-ERROR] Missing analysis folder {root}")
    query_table = Path(config_data.get("dw_uniprot_seqs", {}).get("query_table", QUERY_TABLE))
    configs = copy_configs(root, query_table)
    commit = git_commit()
    software_id = f"{args.repo_url}/tree/{commit}"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != CRATE_FILE)
    file_entities = [file_entity(p, root) for p in tqdm(files, desc="Crate files")]
    runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name != CONFIGS_SUBDIR)
    dirs = sorted(p for p in root.rglob("*") if p.is_dir())
    children = {d: [e["@id"] for e in file_entities if Path(e["@id"]).parent == d.relative_to(root)] + [f"{s.relative_to(root).as_posix()}/" for s in dirs if s.parent == d] for d in dirs}
    dir_entities = [{"@id": f"{d.relative_to(root).as_posix()}/", "@type": "Dataset", "name": d.name, "hasPart": [{"@id": i} for i in children[d]]} for d in dirs]
    now = datetime.datetime.now().isoformat(timespec="seconds")
    root_entity = {"@id": "./", "@type": "Dataset", "name": root.name, "description": f"Protein embedding clustering analysis of {describe_inputs(query_table)}.", "datePublished": now, "hasPart": [{"@id": e["@id"]} for e in file_entities if "/" not in e["@id"]] + [{"@id": f"{d.relative_to(root).as_posix()}/"} for d in dirs if d.parent == root], "mentions": [{"@id": f"#run-{r.name}"} for r in runs]}
    if args.license: root_entity["license"] = {"@id": args.license}
    software = {"@id": software_id, "@type": "SoftwareSourceCode", "name": "SCambrian", "codeRepository": args.repo_url, "version": commit}
    actions = [{"@id": f"#run-{r.name}", "@type": "CreateAction", "name": f"Clustering and report of {r.name}", "instrument": {"@id": software_id}, "object": [{"@id": c.relative_to(root).as_posix()} for c in configs], "result": {"@id": f"{r.name}/"}, "endTime": datetime.datetime.fromtimestamp(max((p.stat().st_mtime for p in r.rglob("*") if p.is_file()), default=r.stat().st_mtime)).isoformat(timespec="seconds")} for r in runs]
    descriptor = {"@id": CRATE_FILE, "@type": "CreativeWork", "conformsTo": {"@id": CRATE_PROFILE}, "about": {"@id": "./"}}
    crate = {"@context": f"{CRATE_PROFILE}/context", "@graph": [descriptor, root_entity, software, *actions, *dir_entities, *file_entities]}
    (root / CRATE_FILE).write_text(json.dumps(crate, indent=1))
    print(f"[PYTHON-INFO][SAVE] RO-Crate of {root.name}: {len(runs)} runs, {len(file_entities):,} files, commit {commit[:12]} -> {root / CRATE_FILE}")
    meta_dir = Path(args.metadata_dir) / root.name
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / CRATE_FILE, meta_dir / CRATE_FILE)
    shutil.copytree(root / CONFIGS_SUBDIR, meta_dir / CONFIGS_SUBDIR, dirs_exist_ok=True)
    print(f"[PYTHON-INFO][SAVE] Tracked copy of the crate and configs -> {meta_dir}")
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
