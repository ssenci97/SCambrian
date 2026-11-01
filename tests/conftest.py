#!/usr/bin/env python3
"""Shared fixtures of the SCambrian tests: a script runner and an isolated workspace with synthetic FASTA, embeddings, focus lists and a YAML config."""
###################################### DEPENDENCIES_AND_CONSTANTS
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
EMBED_DIM = 64
FAMILY_LENGTH = 120
LABELS = {"alpha": "target", "beta": "target", "gamma": "probe"}
MODEL_DIR = "esmc_test"
MUTATION_RATE = 0.08
N_FAMILIES = 6
PROTEINS_PER_LABEL = 150
REPO = Path(__file__).resolve().parents[1]
SEED = 42
SRC = REPO / "src"
###################################### HELPERS
def run_script(name: str, *args: str | Path, cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess:
    """Run one pipeline script as the slurms do, and fail with its output tail on a non-zero exit.

    Args:
        name (str): Script name without ``.py``.
        *args (str | Path): Command-line arguments.
        cwd (Path): Working directory.
        check (bool): Raise on a non-zero exit code.

    Returns:
        subprocess.CompletedProcess: Finished process with captured stdout and stderr.
    """
    cmd = [sys.executable, str(SRC / f"{name}.py"), *map(str, args)]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env={**os.environ, "SLURM_CPUS_PER_TASK": "2", "MPLBACKEND": "Agg"})
    if check and result.returncode != 0:
        raise AssertionError(f"[TEST-ERROR] {' '.join(cmd)} exited {result.returncode}\n--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}")
    return result
def family_members(rng: np.random.Generator, base: np.ndarray, n: int) -> list[str]:
    """Derive point-mutated variants of a family sequence.

    Args:
        rng (np.random.Generator): Seeded generator.
        base (np.ndarray): Family sequence as an array of residues.
        n (int): Number of variants.

    Returns:
        list[str]: Variant sequences.
    """
    out = []
    for _ in range(n):
        seq = base.copy()
        mask = rng.random(seq.size) < MUTATION_RATE
        seq[mask] = rng.choice(AMINO_ACIDS, mask.sum())
        out.append("".join(seq))
    return out
def build_workspace(root: Path) -> SimpleNamespace:
    """Write a self-contained workspace: query table, YAML config, FASTA per label, ``<label>_reps.npz`` per label and focus lists.

    Args:
        root (Path): Empty folder.

    Returns:
        SimpleNamespace: Paths of the workspace (root, config, query_table, sequences_dir, features_dir, reps_dir, npzs, scanpy_dir).
    """
    rng = np.random.default_rng(SEED)
    ws = SimpleNamespace(root=root, config=root / "param_configs.yaml", query_table=root / "query_configs.tsv", sequences_dir=root / "sequences", features_dir=root / "features", reps_dir=root / "reps" / MODEL_DIR, scanpy_dir=root / "scanpy", dedup_dir=root / "dedup", clusters_dir=root / "clusters")
    for d in (ws.sequences_dir, ws.features_dir, ws.reps_dir): d.mkdir(parents=True)
    bases = [rng.choice(AMINO_ACIDS, FAMILY_LENGTH) for _ in range(N_FAMILIES)]
    centroids = rng.normal(0, 4, (N_FAMILIES, EMBED_DIM))
    rows, ws.npzs = [], []
    for label, role in LABELS.items():
        families = np.repeat(np.arange(N_FAMILIES), PROTEINS_PER_LABEL // N_FAMILIES)
        ids = np.array([f"{label.upper()}{i:04d}" for i in range(families.size)])
        seqs = [s for f in range(N_FAMILIES) for s in family_members(rng, bases[f], int((families == f).sum()))]
        fasta = ws.sequences_dir / f"{label}_seqs.fasta"
        fasta.write_text("".join(f">{i}\n{s}\n" for i, s in zip(ids, seqs)))
        npz = ws.reps_dir / f"{label}_reps.npz"
        np.savez(npz, embeddings=(centroids[families] + rng.normal(0, 1, (families.size, EMBED_DIM))).astype(np.float32), ids=ids)
        ws.npzs.append(npz)
        if role == "target": (ws.features_dir / f"{label}_focus_ids.txt").write_text("".join(f"{i}\n" for i in ids[families == 0]))
        rows.append(f"{label}\t{fasta}\tSp. {label}\t{role}")
    ws.query_table.write_text("label\tquery\texact_species\tinput_type\n" + "\n".join(rows) + "\n")
    ws.config.write_text(yaml.dump({"dw_uniprot_seqs": {"query_table": str(ws.query_table), "sequences_dir": str(ws.sequences_dir)}, "dw_uniprot_focus": {"features_dir": str(ws.features_dir), "focus_name": "test", "focus_query": "(family:0)"}}, sort_keys=False))
    return ws
###################################### FIXTURES
@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Provide one synthetic workspace per test module.

    Args:
        tmp_path_factory (pytest.TempPathFactory): pytest temporary folder factory.

    Returns:
        SimpleNamespace: Workspace paths from ``build_workspace``.
    """
    return build_workspace(tmp_path_factory.mktemp("ws"))
