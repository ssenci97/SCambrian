#!/usr/bin/env python3
"""Small end-to-end runs on synthetic data: MMseqs2 dedup and clustering (step 02), Scanpy clustering (supp_01), report and interactive views (supp_02)."""
###################################### DEPENDENCIES_AND_CONSTANTS
from pathlib import Path
import shutil
from types import SimpleNamespace
import pytest
from conftest import run_script
H5AD_COLUMNS = {"protein_id", "species", "input_type", "focus_tier", "cluster", "intra_cluster"}
H5AD_LAYOUTS = {"X_umap_2d", "X_tsne_2d", "X_umap_3d", "X_tsne_3d"}
NPZ_ARGS = ["--knn_neighbors", "10", "--tsne_iterations", "250", "--tsne_perplexity", "10", "--n_jobs", "2"]
REPORT_ARGS = ["--min_capture_percent", "0", "--min_species_proteins", "5", "--min_intra_cluster_proteins", "5", "--min_module_proteins", "5", "--min_focus_proteins", "2"]
pytestmark = pytest.mark.cpu
###################################### HELPERS
def scanpy_run(ws: SimpleNamespace, out_dir: Path) -> Path:
    """Run npz_scanpy.py on the synthetic embeddings and return the written .h5ad.

    Args:
        ws (SimpleNamespace): Workspace from conftest.
        out_dir (Path): Base output folder (``--scanpy_dir``).

    Returns:
        Path: The single .h5ad of the run.
    """
    run_script("npz_scanpy", "--npzs", ",".join(map(str, ws.npzs)), "--scanpy_dir", out_dir, *NPZ_ARGS, "--yaml-in", ws.config, cwd=ws.root)
    h5ads = sorted(out_dir.glob("*/*.h5ad"))
    assert len(h5ads) == 1, f"expected one .h5ad, found {h5ads}"
    return h5ads[0]
###################################### FIXTURES
@pytest.fixture(scope="module")
def h5ad(workspace: SimpleNamespace) -> Path:
    """Run the Scanpy step once per module.

    Args:
        workspace (SimpleNamespace): Workspace from conftest.

    Returns:
        Path: The written .h5ad.
    """
    return scanpy_run(workspace, workspace.scanpy_dir)
###################################### TESTS
def test_mmseqs_dedup_and_clusters(workspace: SimpleNamespace) -> None:
    """Step 02 writes a smaller dedup FASTA, a params sidecar and cluster metrics per label, then skips a fresh rerun."""
    if shutil.which("mmseqs") is None: pytest.skip("mmseqs not on PATH (installed in ./.env by esmc_config.sh)")
    args = ["--sequences_dir", workspace.sequences_dir, "--dedup_dir", workspace.dedup_dir, "--clusters_dir", workspace.clusters_dir, "--cluster_identity_thresholds", "0.9", "0.5", "--yaml-in", workspace.config]
    run_script("run_mmseqs2", *args, cwd=workspace.root)
    for fasta in workspace.sequences_dir.glob("*_seqs.fasta"):
        label = fasta.name.removesuffix("_seqs.fasta")
        dedup = workspace.dedup_dir / f"{label}_dedup.fasta"
        assert dedup.exists() and (workspace.dedup_dir / f"{label}_dedup.params").exists()
        assert 0 < dedup.read_text().count(">") <= fasta.read_text().count(">")
        assert (workspace.clusters_dir / label / f"{label}_cluster_metrics.tsv").exists()
    assert "fresh" in run_script("run_mmseqs2", *args, cwd=workspace.root).stdout
def test_scanpy_h5ad_content(h5ad: Path) -> None:
    """supp_01 stores clusters, roles, focus tiers and 2D/3D UMAP and t-SNE layouts."""
    ad = pytest.importorskip("anndata")
    adata = ad.read_h5ad(h5ad)
    assert H5AD_COLUMNS <= set(adata.obs.columns)
    assert H5AD_LAYOUTS <= set(adata.obsm.keys())
    assert set(adata.obs["input_type"].astype(str)) == {"target", "probe"}
    assert (adata.obs["focus_tier"].astype(str) == "annotated").any()
    assert adata.obs["cluster"].nunique() > 1
def test_scanpy_skips_fresh_output(workspace: SimpleNamespace, h5ad: Path) -> None:
    """A second supp_01 call with unchanged inputs and parameters does not recompute."""
    result = run_script("npz_scanpy", "--npzs", ",".join(map(str, workspace.npzs)), "--scanpy_dir", workspace.scanpy_dir, *NPZ_ARGS, "--yaml-in", workspace.config, cwd=workspace.root)
    assert "Skipping" in result.stdout
def test_scanpy_is_deterministic(workspace: SimpleNamespace, h5ad: Path) -> None:
    """Two supp_01 runs with the same seed give the same joint and intra clusters."""
    ad = pytest.importorskip("anndata")
    again = scanpy_run(workspace, workspace.root / "scanpy_again")
    first, second = ad.read_h5ad(h5ad).obs, ad.read_h5ad(again).obs
    assert first["cluster"].astype(str).tolist() == second["cluster"].astype(str).tolist()
    assert first["intra_cluster"].astype(str).tolist() == second["intra_cluster"].astype(str).tolist()
def test_report(workspace: SimpleNamespace, h5ad: Path) -> None:
    """supp_02 writes the merged PDF report and the enrichment and metadata tables."""
    run_script("h5ad_scanpy", "--h5ad", h5ad, *REPORT_ARGS, "--yaml-in", workspace.config, cwd=workspace.root)
    run_dir = h5ad.parent
    assert (run_dir / f"{run_dir.name}_scanpy.pdf").stat().st_size > 0
    assert list(run_dir.glob("species_cluster_enrichment__*.csv"))
    assert list(run_dir.glob("protein_clusters_metadata__*.csv"))
def test_interactive(workspace: SimpleNamespace, h5ad: Path) -> None:
    """supp_02 writes the 3D HTML viewer and the HDBSCAN table."""
    run_script("h5ad_interactive", "--h5ad", h5ad, "--hdbscan_min_cluster_sizes", "10", "20", "--hdbscan_min_samples", "5", "--n_jobs", "2", "--yaml-in", workspace.config, cwd=workspace.root)
    out = h5ad.parent / "interactive"
    assert list(out.glob("views_3d__*.html")) and list(out.glob("hdbscan_umap3d__*.tsv"))
