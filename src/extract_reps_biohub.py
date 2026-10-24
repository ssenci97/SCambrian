#!/usr/bin/env python3
"""Extract mean-pooled ESMC protein representations from a FASTA or sequence TSV into one ``.npz`` per model."""
############### <imports>
import argparse
import gc
import glob
import inspect
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import tqdm
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
warnings.filterwarnings("ignore", category=FutureWarning)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from utils import add_yaml_args, fasta_label, log, log_resources, parse_fasta, resolve_params

############### <constants>
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
SCRIPT_NAME = Path(__file__).stem
DEFAULTS = {"model_names": ["esmc_300m", "esmc_600m", "esmc_6b"], "batch_size": 300, "chunk_size": 500, "reps_outdir": "data/esm-reps"}
CFG = LogitsConfig(sequence=True, return_embeddings=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_6B_NAMES = ["esmc_6b", "esm_6b", "biohub/ESMC-6B", "biohub/esmc-6b-2024-12"]
HF_6B_REPO = "biohub/ESMC-6B"


############### <io>
def parse_tsv(path: Path) -> list[tuple[str, str]]:
    """Read ``(identifier, sequence)`` pairs from the first two columns of a TSV with header.

    Args:
        path: Sequence TSV, e.g. ``<label>_seqs.tsv``.

    Returns:
        Records in file order.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                records.append((parts[0], parts[1]))
    return records


############### <model>
def esmc_extract(seqs: list[str], model: ESMC) -> list[np.ndarray]:
    """Embed sequences and mean-pool over residues, excluding BOS/EOS tokens.

    Args:
        seqs: Amino-acid sequences.
        model: Loaded ESMC model in eval mode.

    Returns:
        One embedding vector per sequence.
    """
    tensors = [model.encode(ESMProtein(sequence=s)) for s in seqs]
    return [model.logits(t, CFG).embeddings.squeeze(0)[1:-1].mean(dim=0).float().cpu().numpy() for t in tensors]


def load_esmc_model(model_name: str, device: torch.device) -> ESMC:
    """Load an ESMC model from the ``esm`` registry or the ESMC-6B Hugging Face weights.

    Args:
        model_name: Registry name (e.g. ``esmc_600m``) or an alias of the 6B model.
        device: Target device.

    Returns:
        The model on ``device`` in eval mode.
    """
    if model_name not in HF_6B_NAMES:
        log(f"Loading standard ESMC model via registry: {model_name}")
        return ESMC.from_pretrained(model_name).to(device).eval()
    log(f"Loading 6B model from Hugging Face hub ({HF_6B_REPO})")
    model_dir = snapshot_download(repo_id=HF_6B_REPO)
    config_path = os.path.join(model_dir, "config.json")
    model_kwargs = {}
    if os.path.exists(config_path):
        with open(config_path) as handle:
            cfg = json.load(handle)
        valid_params = set(inspect.signature(ESMC.__init__).parameters) - {"self"}
        model_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
    if model_kwargs:
        log("Initializing ESMC architecture from config.json")
        model = ESMC(**model_kwargs)
    else:
        log("Initializing default ESMC architecture")
        model = ESMC.from_pretrained("esmc_600m")
    raw_state_dict = {}
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if st_files:
        for f in st_files:
            raw_state_dict.update(load_file(f))
    else:
        for f in sorted(glob.glob(os.path.join(model_dir, "*.bin")) + glob.glob(os.path.join(model_dir, "*.pt"))):
            raw_state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("model."):
            k = k[6:]
        if k.startswith("_orig_mod."):
            k = k[10:]
        clean_state_dict[k] = v
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    log(f"Loaded state dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    return model.to(device).eval()


def embed_records(records: list[tuple[str, str]], model: ESMC, chunk_size: int, batch_size: int, out: Path, desc: str) -> np.ndarray:
    """Embed records in chunks, spilling each chunk to a temporary ``.npz``, and write the merged ``.npz``.

    Args:
        records: ``(identifier, sequence)`` pairs.
        model: Loaded ESMC model in eval mode.
        chunk_size: Sequences per temporary chunk file.
        batch_size: Sequences per forward batch.
        out: Final ``.npz`` path.
        desc: Progress-bar label.

    Returns:
        The stacked embedding matrix.
    """
    chunk_files = []
    for c_idx, i in enumerate(tqdm.tqdm(range(0, len(records), chunk_size), desc=desc)):
        chunk_recs = records[i:i + chunk_size]
        c_ids, c_embs = [], []
        with torch.no_grad():
            for j in range(0, len(chunk_recs), batch_size):
                batch = chunk_recs[j:j + batch_size]
                c_ids.extend(r[0] for r in batch)
                c_embs.extend(esmc_extract([r[1] for r in batch], model))
        c_file = f"{out}.tmp_{c_idx}.npz"
        np.savez(c_file, embeddings=np.stack(c_embs), ids=np.array(c_ids))
        chunk_files.append(c_file)
        del c_embs, c_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_embs, all_ids = [], []
    for c_file in chunk_files:
        data = np.load(c_file)
        all_embs.append(data["embeddings"])
        all_ids.append(data["ids"])
        os.remove(c_file)
    embeddings = np.concatenate(all_embs, axis=0)
    np.savez(out, embeddings=embeddings, ids=np.concatenate(all_ids, axis=0))
    return embeddings


############### <main>
def main() -> None:
    """Parse arguments, resolve parameters and write one representation ``.npz`` per model."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seqs_tsv", type=Path)
    group.add_argument("--fasta", type=Path)
    parser.add_argument("--out", type=Path, help="Output .npz, single model only (default: <reps_outdir>/<model>/<label>_reps.npz)")
    parser.add_argument("--model_names", nargs="+")
    parser.add_argument("--reps_outdir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--chunk_size", type=int)
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    in_path = args.fasta or args.seqs_tsv
    if in_path is None:
        if args.yaml_out:
            sys.exit(0)
        parser.error("one of --seqs_tsv or --fasta is required")
    model_names = [params["model_names"]] if isinstance(params["model_names"], str) else list(params["model_names"])
    if args.out and len(model_names) > 1:
        parser.error("--out requires exactly one model in --model_names")
    outs = {m: args.out or Path(params["reps_outdir"]) / m / f"{fasta_label(in_path)}_reps.npz" for m in model_names}
    todo = [m for m in model_names if not outs[m].exists()]
    for m in model_names:
        if m not in todo:
            log(f"Target file '{outs[m]}' already exists. Skipping {m}.")
    if not todo:
        log_resources(start_time)
        sys.exit(0)
    records = parse_fasta(in_path) if args.fasta else parse_tsv(in_path)
    log(f"Loaded {len(records)} sequences from {in_path}; device={DEVICE}; models={todo}")
    for m in todo:
        out = outs[m]
        out.parent.mkdir(parents=True, exist_ok=True)
        model = load_esmc_model(m, DEVICE)
        embeddings = embed_records(records, model, params["chunk_size"], params["batch_size"], out, f"[PYTHON-INFO] Processing chunks for {in_path} ({m})")
        log(f"Saved {m} embeddings {embeddings.shape} to '{out}'")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log_resources(start_time)


if __name__ == "__main__":
    main()
