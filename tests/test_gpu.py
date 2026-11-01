#!/usr/bin/env python3
"""ESMC embedding check (step 03) on an A100: output shape, ids and run-to-run reproducibility of esmc_300m."""
###################################### DEPENDENCIES_AND_CONSTANTS
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import run_script
ATOL = 1e-3
D_MODEL = 960
MODEL = "esmc_300m"
N_SEQUENCES = 5
pytestmark = pytest.mark.gpu
###################################### TESTS
def test_esmc_embeddings(workspace: SimpleNamespace) -> None:
    """Two esmc_300m runs on five sequences give (5, 960) embeddings with matching ids that agree within ATOL."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("no CUDA device (submit with --partition=DGX --gres=gpu:A100:1)")
    lines = (workspace.sequences_dir / "alpha_seqs.fasta").read_text().splitlines()[:2 * N_SEQUENCES]
    fasta = workspace.root / "five.fasta"
    fasta.write_text("\n".join(lines) + "\n")
    outs = []
    for run in ("a", "b"):
        out_dir = workspace.root / f"reps_{run}"
        run_script("extract_reps_biohub", "--fasta", fasta, "--models", MODEL, "--embeddings_dir", out_dir, "--yaml-in", workspace.config, cwd=workspace.root)
        outs.append(np.load(out_dir / MODEL / "five_reps.npz"))
    assert outs[0]["embeddings"].shape == (N_SEQUENCES, D_MODEL)
    assert outs[0]["ids"].tolist() == [l[1:] for l in lines[::2]]
    assert np.isfinite(outs[0]["embeddings"]).all()
    assert np.allclose(outs[0]["embeddings"], outs[1]["embeddings"], atol=ATOL)
