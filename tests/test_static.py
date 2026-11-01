#!/usr/bin/env python3
"""Fast checks for the login node: syntax, CLI entry points, query table schema, config keys and run_all.sh array sizes."""
###################################### DEPENDENCIES_AND_CONSTANTS
import importlib.util
import os
from pathlib import Path
import py_compile
import re
import subprocess
import pandas as pd
import pytest
import yaml
from conftest import REPO, SRC, run_script
CONFIG_YAML = REPO / "configs" / "param_configs.yaml"
HEAVY_IMPORTS = {"extract_reps_biohub": "esm", "npz_scanpy": "scanpy", "h5ad_scanpy": "pymupdf", "h5ad_interactive": "hdbscan"}
INPUT_TYPES = {"target", "probe"}
LOCAL_FASTA_SUFFIXES = (".fa", ".faa", ".fasta")
QUERY_COLUMNS = {"label", "query", "exact_species", "input_type"}
SCRIPTS = sorted(p.stem for p in SRC.glob("*.py"))
SHELL_SCRIPTS = sorted([*REPO.glob("slurms/*.sh"), REPO / "esmc_config.sh", *REPO.glob("env/*.sh")])
pytestmark = pytest.mark.static
###################################### HELPERS
def script_options(name: str) -> set[str]:
    """Collect the long options a script declares with ``add_argument``.

    Args:
        name (str): Script name without ``.py``.

    Returns:
        set[str]: Option names with dashes turned into underscores.
    """
    return {m.replace("-", "_") for m in re.findall(r'add_argument\(\s*"--([A-Za-z0-9_-]+)"', (SRC / f"{name}.py").read_text())}
def load_config() -> dict:
    """Read the pipeline YAML configuration.

    Returns:
        dict: Parsed configuration.
    """
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}
###################################### TESTS
@pytest.mark.parametrize("name", SCRIPTS)
def test_python_compiles(name: str, tmp_path: Path) -> None:
    """Every pipeline script byte-compiles."""
    py_compile.compile(str(SRC / f"{name}.py"), cfile=str(tmp_path / f"{name}.pyc"), doraise=True)
@pytest.mark.parametrize("path", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_syntax(path: Path) -> None:
    """Every shell and SLURM script parses with ``bash -n``."""
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
@pytest.mark.parametrize("name", SCRIPTS)
def test_cli_help(name: str) -> None:
    """Every script imports its dependencies and prints its help."""
    if importlib.util.find_spec(HEAVY_IMPORTS.get(name, "pandas")) is None: pytest.skip(f"{HEAVY_IMPORTS.get(name, 'pandas')} not installed (run inside ./.env)")
    assert "--yaml-in" in run_script(name, "--help").stdout
def test_query_table_schema() -> None:
    """The query table has the required columns, unique labels, valid input types, existing local FASTAs and at least two targets."""
    tsv = REPO / load_config().get("dw_uniprot_seqs", {}).get("query_table", "configs/query_configs.tsv")
    table = pd.read_csv(tsv, sep="\t", dtype=str)
    assert QUERY_COLUMNS <= set(table.columns), f"missing columns: {QUERY_COLUMNS - set(table.columns)}"
    assert table["label"].notna().all() and table["label"].is_unique, "labels must be non-empty and unique"
    assert set(table["input_type"].fillna("target").str.strip()) <= INPUT_TYPES
    assert (table["input_type"].fillna("target").str.strip() == "target").sum() >= 2, "at least two targets are needed"
    missing = [q for q in table["query"].dropna().str.strip() if q.endswith(LOCAL_FASTA_SUFFIXES) and not (REPO / q).exists()]
    assert not missing, f"local FASTA sources not found: {missing}"
@pytest.mark.parametrize("section", sorted(load_config()))
def test_config_keys_are_script_options(section: str) -> None:
    """Every key of a config section is an option of the script with the same name."""
    assert (SRC / f"{section}.py").exists(), f"no script src/{section}.py for config section '{section}'"
    unknown = set(load_config()[section]) - script_options(section)
    assert not unknown, f"keys of '{section}' without a matching --option: {sorted(unknown)}"
@pytest.mark.parametrize("cpu_only", ["0", "1"])
def test_run_all_dry_run_array_sizes(cpu_only: str) -> None:
    """``DRY_RUN=1 run_all.sh`` submits every step, chained, with arrays sized from the labels and models; ``CPU_ONLY=1`` keeps only esmc_300m and moves step 03 to EPYC."""
    cfg = load_config()
    models = cfg.get("extract_reps_biohub", {}).get("models", ["esmc_6b"])
    n_models = 1 if isinstance(models, str) or cpu_only == "1" else len(models)
    table = pd.read_csv(REPO / cfg.get("dw_uniprot_seqs", {}).get("query_table", "configs/query_configs.tsv"), sep="\t", dtype=str)
    n = int(table["label"].notna().sum())
    result = subprocess.run(["bash", "slurms/run_all.sh"], cwd=REPO, capture_output=True, text=True, env={**os.environ, "DRY_RUN": "1", "CPU_ONLY": cpu_only})
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [l for l in result.stdout.splitlines() if "sbatch" in l]
    expected = [f"--array=1-{n} slurms/01_dw_uniprot.sh", f"--array=1-{n}%3 slurms/02_mmseqs2.sh", f"--array=0-{n * n_models - 1}%2 slurms/03_reps.sh", f"--array=0-{n_models - 1} slurms/supp_01_scanpy.sh", "slurms/supp_02_scanpy_report.sh"]
    assert len(lines) == len(expected)
    for line, tail in zip(lines, expected): assert line.rstrip().endswith(tail), f"{line!r} does not end with {tail!r}"
    assert all("--dependency=afterok:" in l for l in lines[1:])
    assert ("--partition=EPYC --gres=none" in lines[2]) == (cpu_only == "1")
