#!/usr/bin/env python3
"""Extracts mean-pooled per-protein ESMC embeddings from a FASTA (or id/sequence TSV) for every requested model.

Each model writes ``<embeddings_dir>/<model>/<label>_reps.npz`` with ``embeddings`` (n_proteins x d_model) and ``ids``, where the label is the input file name without ``_dedup``/``_seqs``. Embeddings average the last hidden states over residues, excluding BOS and EOS. Proteins are embedded one at a time and checkpointed every ``sequences_per_chunk`` proteins into temporary files that are merged at the end. The 300m and 600m models load from the esm registry; the 6B model loads from the ``biohub/ESMC-6B`` Hugging Face checkpoint, whose published layout is translated onto the esm ``ESMC`` module tree before a strict state-dict load. An output is recomputed when it is missing or older than the input file, or with ``--overwrite``.
"""
###################################### DEPENDENCIES_AND_CONSTANTS
import argparse
import datetime
import gc
import glob
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time
import warnings
from huggingface_hub import snapshot_download
import numpy as np
from safetensors.torch import load_file
import torch
from tqdm import tqdm
import yaml
warnings.filterwarnings("ignore", category=FutureWarning)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS_DIR = "data/esm-reps"
HF_6B_NAMES = ["esmc_6b", "esm_6b", "biohub/ESMC-6B", "biohub/esmc-6b-2024-12"]
HF_6B_REPO = "biohub/ESMC-6B"
HF_BLOCK_KEYS = {
    "input_layernorm.weight": "attn.layernorm_qkv.0.weight",
    "input_layernorm.bias": "attn.layernorm_qkv.0.bias",
    "self_attn.q_norm.weight": "attn.q_ln.weight",
    "self_attn.k_norm.weight": "attn.k_ln.weight",
    "self_attn.o_proj.weight": "attn.out_proj.weight",
    "post_attention_layernorm.weight": "ffn.0.weight",
    "post_attention_layernorm.bias": "ffn.0.bias",
    "mlp.down_proj.weight": "ffn.3.weight",
}
HF_BLOCK_RE = re.compile(r"^esmc\.layers\.(\d+)\.(.+)$")
HF_FUSED_KEYS = {
    "attn.layernorm_qkv.1.weight": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
    "ffn.1.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
}
HF_ROOT_KEYS = {
    "esmc.embed_tokens.weight": "embed.weight",
    "esmc.norm.weight": "transformer.norm.weight",
    "lm_head.dense.weight": "sequence_head.0.weight",
    "lm_head.dense.bias": "sequence_head.0.bias",
    "lm_head.layer_norm.weight": "sequence_head.2.weight",
    "lm_head.layer_norm.bias": "sequence_head.2.bias",
    "lm_head.decoder.weight": "sequence_head.3.weight",
    "lm_head.decoder.bias": "sequence_head.3.bias",
}
LOGITS_CONFIG = LogitsConfig(sequence=True, return_embeddings=True)
MODELS = ["esmc_300m", "esmc_600m", "esmc_6b"]
OVERWRITE = False
SEED = 42
SEQUENCES_PER_CHUNK = 500
YAML_PATH = "configs/param_configs.yaml"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
###################################### CONFIG_AND_PARSER
parser = argparse.ArgumentParser(description="ESMC protein embeddings")
input_group = parser.add_mutually_exclusive_group(required=False)
input_group.add_argument("--fasta", type=str, default=None, help="Input FASTA, typically <dedup_dir>/<label>_dedup.fasta")
input_group.add_argument("--seqs_tsv", type=str, default=None, help="Input TSV with a header and id, sequence columns")
parser.add_argument("--models", type=str, nargs="+", default=MODELS, help="ESMC models to run (esmc_300m, esmc_600m, esmc_6b)")
parser.add_argument("--embeddings_dir", type=str, default=EMBEDDINGS_DIR, help="Base output folder; outputs go to <embeddings_dir>/<model>/<label>_reps.npz")
parser.add_argument("--out", type=str, default=None, help="Explicit output path, only with a single model")
parser.add_argument("--sequences_per_chunk", type=int, default=SEQUENCES_PER_CHUNK, help="Proteins per temporary checkpoint file")
parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE, help="Recompute outputs that are fresh")
parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH)
parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None)
cli_preview, _ = parser.parse_known_args()
script_key = "extract_reps_biohub"
params = dict(models=MODELS, sequences_per_chunk=SEQUENCES_PER_CHUNK, embeddings_dir=EMBEDDINGS_DIR, overwrite=OVERWRITE)
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
def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Read a FASTA file.

    Args:
        path (str): FASTA file.

    Returns:
        list[tuple[str, str]]: (id, sequence) per record, ids cut at the first whitespace.
    """
    records, seq_id, seq_parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if seq_id: records.append((seq_id, "".join(seq_parts)))
                seq_id, seq_parts = line[1:].split()[0], []
            else: seq_parts.append(line)
    if seq_id: records.append((seq_id, "".join(seq_parts)))
    return records
def parse_tsv(path: str) -> list[tuple[str, str]]:
    """Read a two-column id/sequence TSV with a header row.

    Args:
        path (str): TSV file.

    Returns:
        list[tuple[str, str]]: (id, sequence) per row.
    """
    with open(path) as f:
        next(f, None)
        return [(fields[0], fields[1]) for fields in (line.strip().split("\t") for line in f) if len(fields) >= 2]
def mean_pooled_embeddings(sequences: list[str], model: ESMC) -> list[np.ndarray]:
    """Embed proteins one at a time and average their last hidden states over residues, excluding BOS and EOS.

    Args:
        sequences (list[str]): Protein sequences.
        model (ESMC): Loaded model.

    Returns:
        list[np.ndarray]: One vector of length d_model per protein.
    """
    return [model.logits(model.encode(ESMProtein(sequence=seq)), LOGITS_CONFIG).embeddings.squeeze(0)[1:-1].float().mean(dim=0).cpu().numpy() for seq in sequences]
def hf_to_esmc_state_dict(published: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate the published ESMC checkpoint layout onto the esm ``ESMC`` module tree.

    Mirrors ``esm.models.esmc.checkpoint_layout.published_to_native`` from esm 3.4.1, targeting the ``nn.Sequential`` indices of the esm 3.2.x modules. Separate q/k/v and gate/up projections are concatenated row-wise in that order.

    Args:
        published (dict[str, torch.Tensor]): Published state dict (``esmc.layers.N.*``, ``lm_head.*``).

    Returns:
        dict[str, torch.Tensor]: State dict keyed by ``ESMC`` parameter names; unrecognised keys pass through.
    """
    native, unfused = {}, {}
    for key, tensor in published.items():
        if key in HF_ROOT_KEYS:
            native[HF_ROOT_KEYS[key]] = tensor
            continue
        match = HF_BLOCK_RE.match(key)
        if match is None:
            native[key] = tensor
            continue
        block, leaf = match.groups()
        if leaf in HF_BLOCK_KEYS: native[f"transformer.blocks.{block}.{HF_BLOCK_KEYS[leaf]}"] = tensor
        else: unfused.setdefault(block, {})[leaf] = tensor
    for block, leaves in unfused.items():
        for fused_key, parts in HF_FUSED_KEYS.items():
            native[f"transformer.blocks.{block}.{fused_key}"] = torch.cat([leaves.pop(part) for part in parts], dim=0)
        native.update({f"esmc.layers.{block}.{leaf}": tensor for leaf, tensor in leaves.items()})
    return native
def load_esmc_model(model_name: str, device: torch.device) -> ESMC:
    """Load an ESMC model ready for inference.

    Args:
        model_name (str): Registry name (``esmc_300m``, ``esmc_600m``) or one of ``HF_6B_NAMES``.
        device (torch.device): Target device.

    Returns:
        ESMC: Model in eval mode, bfloat16 on GPU.

    Raises:
        RuntimeError: If the 6B checkpoint does not match the ``ESMC`` architecture exactly.
    """
    if model_name not in HF_6B_NAMES:
        print(f"[PYTHON-INFO] Loading {model_name} from the esm registry")
        return ESMC.from_pretrained(model_name, device=device).eval()
    print(f"[PYTHON-INFO] Loading the 6B model from the Hugging Face hub ({HF_6B_REPO})")
    model_dir = snapshot_download(repo_id=HF_6B_REPO)
    with open(os.path.join(model_dir, "config.json")) as f:
        hf_config = json.load(f)
    model = ESMC(d_model=hf_config["hidden_size"], n_heads=hf_config["num_attention_heads"], n_layers=hf_config["num_hidden_layers"], tokenizer=get_esmc_model_tokenizers(), use_flash_attn=device.type == "cuda")
    published = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))): published.update(load_file(shard))
    state_dict = hf_to_esmc_state_dict(published)
    del published
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[PYTHON-INFO] Loaded state dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    if missing or unexpected: raise RuntimeError(f"[PYTHON-ERROR] ESMC-6B checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    del state_dict
    gc.collect()
    model = model.to(device)
    if device.type != "cpu": model = model.to(torch.bfloat16)
    return model.eval()
def embed_file(records: list[tuple[str, str]], model_name: str, out_path: str) -> None:
    """Embed every record with one model, checkpointing chunks to temporary files, and save the merged ``.npz``.

    Args:
        records (list[tuple[str, str]]): (id, sequence) per protein.
        model_name (str): ESMC model name.
        out_path (str): Output ``.npz`` path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model, chunk_paths = load_esmc_model(model_name, DEVICE), []
    for chunk_index, start in enumerate(tqdm(range(0, len(records), args.sequences_per_chunk), desc=f"{model_name} chunks")):
        chunk = records[start:start + args.sequences_per_chunk]
        with torch.no_grad():
            chunk_embeddings = mean_pooled_embeddings([seq for _, seq in chunk], model)
        chunk_path = f"{out_path}.tmp_{chunk_index}.npz"
        np.savez(chunk_path, embeddings=np.stack(chunk_embeddings), ids=np.array([seq_id for seq_id, _ in chunk]))
        chunk_paths.append(chunk_path)
        del chunk_embeddings
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    chunks = [np.load(path) for path in chunk_paths]
    np.savez(out_path, embeddings=np.concatenate([c["embeddings"] for c in chunks]), ids=np.concatenate([c["ids"] for c in chunks]))
    for path in chunk_paths: os.remove(path)
    print(f"[PYTHON-INFO][SAVE] {model_name}: {len(records):,} proteins -> {out_path}")
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
###################################### MAIN
def main() -> None:
    """Embed the input file with every requested model whose output is missing or stale."""
    start_time = time.time()
    in_path = args.fasta or args.seqs_tsv
    if not in_path:
        if args.yaml_out: sys.exit(0)
        parser.error("one of --fasta or --seqs_tsv is required")
    models = [args.models] if isinstance(args.models, str) else list(args.models)
    if args.out and len(models) > 1: parser.error("--out takes a single --models value; omit --out to write <embeddings_dir>/<model>/<label>_reps.npz")
    label, records = Path(in_path).stem.removesuffix("_dedup").removesuffix("_seqs"), None
    for model_name in models:
        out_path = args.out or os.path.join(args.embeddings_dir, model_name, f"{label}_reps.npz")
        if not args.overwrite and Path(out_path).exists() and Path(out_path).stat().st_mtime > Path(in_path).stat().st_mtime:
            print(f"[PYTHON-INFO] '{out_path}' is newer than {in_path}. Skipping {model_name}.")
            continue
        if records is None: records = parse_fasta(in_path) if args.fasta else parse_tsv(in_path)
        embed_file(records, model_name, out_path)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[PYTHON-INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")
if __name__ == "__main__":
    main()
