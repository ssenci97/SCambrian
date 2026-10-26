#!/usr/bin/env python3
"""Extract mean-pooled protein sequence representations with ESMC models.

The 300m and 600m models load from the esm registry. The 6B model loads from the
``biohub/ESMC-6B`` Hugging Face checkpoint, whose published layout is translated
onto the esm ``ESMC`` module tree before a strict state dict load.
"""
###################################### imports-and-config
import argparse
import datetime
import gc
import glob
import json
import os
import random
import re
import resource
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import tqdm
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
warnings.filterwarnings("ignore", category=FutureWarning)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
YAML_PATH = Path("configs/param_configs.yaml")
SCRIPT_NAME = Path(__file__).stem
BATCH_SIZE = 300
CFG = LogitsConfig(sequence=True, return_embeddings=True)
CHUNK_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["esmc_300m", "esmc_600m", "esmc_6b"]
REPS_OUTDIR = "data/esm-reps"
HF_6B_REPO = "biohub/ESMC-6B"
HF_6B_NAMES = ["esmc_6b", "esm_6b", "biohub/ESMC-6B", "biohub/esmc-6b-2024-12"]
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
HF_FUSED_KEYS = {
    "attn.layernorm_qkv.1.weight": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
    "ffn.1.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
}
HF_BLOCK_RE = re.compile(r"^esmc\.layers\.(\d+)\.(.+)$")
START_TIME = time.time()
###################################### helper-functions
def parse_fasta(fp):
    """Parse a FASTA file into indexed records.

    Args:
        fp: Path to the FASTA file.

    Returns:
        List of ``(index, id, sequence)`` tuples, ids taken up to the first whitespace.
    """
    r, idx, cid, cseq = [], 0, None, []
    with open(fp) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            if l.startswith(">"):
                if cid: r.append((idx, cid, "".join(cseq))); idx += 1
                cid, cseq = l[1:].split()[0], []
            else: cseq.append(l)
    if cid: r.append((idx, cid, "".join(cseq)))
    return r
def parse_tsv(fp):
    """Parse a two-column ``id<TAB>sequence`` TSV with a header row.

    Args:
        fp: Path to the TSV file.

    Returns:
        List of ``(index, id, sequence)`` tuples.
    """
    r = []
    with open(fp) as f:
        next(f, None)
        for idx, l in enumerate(f):
            p = l.strip().split("\t")
            if len(p) >= 2: r.append((idx, p[0], p[1]))
    return r
def esmC_extract(seqs, model, device):
    """Compute mean-pooled per-sequence embeddings, excluding BOS and EOS tokens.

    Args:
        seqs: One sequence or a list of sequences.
        model: Loaded ``ESMC`` model.
        device: Torch device the model lives on.

    Returns:
        List of 1-D numpy arrays of length ``d_model``.
    """
    if isinstance(seqs, str): seqs = [seqs]
    tensors = [model.encode(ESMProtein(sequence=s)) for s in seqs]
    return [model.logits(t, CFG).embeddings.squeeze(0)[1:-1].float().mean(dim=0).cpu().numpy() for t in tensors]
def hf_to_esmc_state_dict(raw):
    """Translate the published ESMC checkpoint layout onto the esm ``ESMC`` module tree.

    Mirrors ``esm.models.esmc.checkpoint_layout.published_to_native`` from esm 3.4.1,
    targeting the ``nn.Sequential`` indices of the esm 3.2.x modules. Separate q/k/v
    and gate/up projections are concatenated row-wise in that order.

    Args:
        raw: Published state dict (``esmc.layers.N.*``, ``lm_head.*``).

    Returns:
        State dict keyed by ``ESMC`` parameter names. Unrecognised keys pass through.
    """
    out, parts = {}, {}
    for k, v in raw.items():
        if k in HF_ROOT_KEYS:
            out[HF_ROOT_KEYS[k]] = v
            continue
        m = HF_BLOCK_RE.match(k)
        if m is None:
            out[k] = v
            continue
        idx, leaf = m.groups()
        if leaf in HF_BLOCK_KEYS: out[f"transformer.blocks.{idx}.{HF_BLOCK_KEYS[leaf]}"] = v
        else: parts.setdefault(idx, {})[leaf] = v
    for idx, found in parts.items():
        for fused, leaves in HF_FUSED_KEYS.items():
            out[f"transformer.blocks.{idx}.{fused}"] = torch.cat([found.pop(l) for l in leaves], dim=0)
        out.update({f"esmc.layers.{idx}.{l}": v for l, v in found.items()})
    return out
def load_esmc_model(model_name, device):
    """Load an ESMC model ready for inference.

    Args:
        model_name: Registry name (``esmc_300m``, ``esmc_600m``) or one of ``HF_6B_NAMES``.
        device: Target torch device.

    Returns:
        ``ESMC`` model in eval mode, bfloat16 on GPU.

    Raises:
        RuntimeError: If the 6B checkpoint does not match the ``ESMC`` architecture exactly.
    """
    if model_name not in HF_6B_NAMES:
        print(f"[PYTHON-INFO] Loading standard ESMC model via registry: {model_name}")
        return ESMC.from_pretrained(model_name, device=device).eval()
    print(f"[PYTHON-INFO] Loading 6B model from Hugging Face hub ({HF_6B_REPO})...")
    model_dir = snapshot_download(repo_id=HF_6B_REPO)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    model = ESMC(d_model=cfg["hidden_size"], n_heads=cfg["num_attention_heads"], n_layers=cfg["num_hidden_layers"], tokenizer=get_esmc_model_tokenizers(), use_flash_attn=device.type == "cuda")
    raw = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))): raw.update(load_file(f))
    state_dict = hf_to_esmc_state_dict(raw)
    del raw
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[PYTHON-INFO] Loaded state dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    if missing or unexpected:
        raise RuntimeError(f"ESMC-6B checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    del state_dict
    gc.collect()
    model = model.to(device)
    if device.type != "cpu": model = model.to(torch.bfloat16)
    return model.eval()
###################################### main-execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--seqs_tsv")
    g.add_argument("--fasta")
    parser.add_argument("--model_name", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--reps_outdir", default=REPS_OUTDIR)
    parser.add_argument("--out")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, type=Path, default=YAML_PATH)
    parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, type=Path, default=None)
    yargs, _ = parser.parse_known_args()
    params = {"model_name": MODEL_NAMES, "batch_size": BATCH_SIZE, "reps_outdir": REPS_OUTDIR, "overwrite": False}
    if yargs.yaml_in and yargs.yaml_in.exists():
        data = yaml.safe_load(yargs.yaml_in.read_text()) or {}
        params.update(data.get(SCRIPT_NAME, {}))
    parser.set_defaults(**params)
    args = parser.parse_args()
    if args.yaml_out:
        data = yaml.safe_load(args.yaml_out.read_text()) if args.yaml_out.exists() else {}
        data[SCRIPT_NAME] = params
        args.yaml_out.parent.mkdir(parents=True, exist_ok=True)
        args.yaml_out.write_text(yaml.safe_dump(data))
    if not args.fasta and not args.seqs_tsv:
        if args.yaml_out: sys.exit(0)
        parser.error("one of --seqs_tsv or --fasta is required")
    model_names = [args.model_name] if isinstance(args.model_name, str) else list(args.model_name)
    if args.out and len(model_names) > 1:
        parser.error("--out takes a single --model_name; omit --out to write <reps_outdir>/<model>/<label>_reps.npz")
    fp = args.fasta if args.fasta else args.seqs_tsv
    label = Path(fp).stem.removesuffix("_dedup").removesuffix("_seqs")
    records = None
    for model_name in model_names:
        out = args.out or os.path.join(args.reps_outdir, model_name, f"{label}_reps.npz")
        if Path(out).exists() and not args.overwrite:
            print(f"[PYTHON-INFO] Target file '{out}' already exists. Skipping {model_name}.")
            continue
        if records is None: records = parse_fasta(args.fasta) if args.fasta else parse_tsv(args.seqs_tsv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        model = load_esmc_model(model_name, DEVICE)
        chunk_files = []
        for c_idx, i in enumerate(tqdm.tqdm(range(0, len(records), CHUNK_SIZE), desc=f"[PYTHON-INFO] Processing chunks for {fp} ({model_name})")):
            chunk_recs = records[i:i + CHUNK_SIZE]
            c_ids, c_embs = [], []
            with torch.no_grad():
                for j in range(0, len(chunk_recs), args.batch_size):
                    b = chunk_recs[j:j + args.batch_size]
                    c_ids.extend([r[1] for r in b])
                    c_embs.extend(esmC_extract([r[2] for r in b], model, DEVICE))
            c_file = f"{out}.tmp_{c_idx}.npz"
            np.savez(c_file, embeddings=np.stack(c_embs), ids=np.array(c_ids))
            chunk_files.append(c_file)
            del c_embs, c_ids
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        all_embs, all_ids = [], []
        for c_file in chunk_files:
            data = np.load(c_file)
            all_embs.append(data["embeddings"])
            all_ids.append(data["ids"])
            os.remove(c_file)
        np.savez(out, embeddings=np.concatenate(all_embs, axis=0), ids=np.concatenate(all_ids, axis=0))
        print(f"[PYTHON-INFO] Saved {model_name} representations to '{out}'")
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[PYTHON-INFO] date={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} execution_time_sec={time.time()-START_TIME:.2f} max_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f}")
