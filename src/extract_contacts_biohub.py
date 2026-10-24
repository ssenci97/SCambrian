#!/usr/bin/env python3
"""Predict ESMC-6B residue-pair similarity contacts for FASTA sequences with length-sorted batching."""
############### <imports>
import argparse
import gc
import glob
import os
import random
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from esm.tokenization import EsmSequenceTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from utils import add_yaml_args, fasta_label, log, log_resources, parse_fasta, resolve_params
warnings.filterwarnings("ignore", category=FutureWarning)

############### <constants>
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
SCRIPT_NAME = Path(__file__).stem
DEFAULTS = {"batch_size": 300, "max_tokens": 10000, "min_sep": 6, "contacts_outdir": "data/esm-contacts/esmc_6b"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
HF_REPO = "biohub/ESMC-6B"
N_LAYERS, N_HEADS, HEAD_DIM, D_MODEL, ROPE_THETA, EPS = 80, 40, 64, 2560, 10000.0, 1e-5
MIN_RECOVERY = 0.5


############### <model>
def load_weights() -> dict[str, torch.Tensor]:
    """Load the ESMC-6B safetensors weights onto the target device.

    Returns:
        Mapping from parameter name to tensor.
    """
    weights = {}
    for f in sorted(glob.glob(os.path.join(snapshot_download(repo_id=HF_REPO), "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                weights[k] = sf.get_tensor(k).to(device=DEVICE, dtype=DTYPE)
    log(f"Loaded {len(weights)} tensors, {sum(v.numel() for v in weights.values()) / 1e9:.2f}B parameters")
    return weights


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by halves for rotary position embeddings.

    Args:
        x: Input tensor.

    Returns:
        Tensor with halves ``(a, b)`` mapped to ``(-b, a)``.
    """
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


@torch.no_grad()
def forward_batch(ids: torch.Tensor, mask: torch.Tensor | None, W: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the ESMC-6B transformer trunk explicitly.

    Args:
        ids: Token ids of shape ``(B, L)``.
        mask: Boolean key mask of shape ``(B, L)``, True for real tokens, or None for no padding.
        W: Model weights from :func:`load_weights`.

    Returns:
        Final normalized hidden states of shape ``(B, L, D_MODEL)``.
    """
    B, L = ids.shape
    x = F.embedding(ids, W["esmc.embed_tokens.weight"])
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE).float() / HEAD_DIM))
    fr = torch.outer(torch.arange(L, device=DEVICE).float(), inv)
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos()[None, None].to(x.dtype), emb.sin()[None, None].to(x.dtype)
    attn_mask = mask[:, None, None, :] if mask is not None else None
    for i in range(N_LAYERS):
        p = f"esmc.layers.{i}."
        h = F.layer_norm(x, (D_MODEL,), W[p + "input_layernorm.weight"], W[p + "input_layernorm.bias"], EPS)
        q = F.layer_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]), (D_MODEL,), W[p + "self_attn.q_norm.weight"], None, EPS)
        k = F.layer_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]), (D_MODEL,), W[p + "self_attn.k_norm.weight"], None, EPS)
        v = F.linear(h, W[p + "self_attn.v_proj.weight"])
        q, k, v = (t.view(B, L, N_HEADS, HEAD_DIM).transpose(1, 2) for t in (q, k, v))
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2).reshape(B, L, D_MODEL)
        x = x + F.linear(a, W[p + "self_attn.o_proj.weight"])
        h = F.layer_norm(x, (D_MODEL,), W[p + "post_attention_layernorm.weight"], W[p + "post_attention_layernorm.bias"], EPS)
        x = x + F.linear(F.silu(F.linear(h, W[p + "mlp.gate_proj.weight"])) * F.linear(h, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])
    return F.layer_norm(x, (D_MODEL,), W["esmc.norm.weight"], None, EPS)


@torch.no_grad()
def check_model(seq: str, tok: EsmSequenceTokenizer, W: dict[str, torch.Tensor]) -> None:
    """Verify the explicit forward pass by masked-LM token recovery on one sequence.

    Args:
        seq: Amino-acid sequence.
        tok: ESM sequence tokenizer.
        W: Model weights.

    Raises:
        RuntimeError: If token recovery falls below ``MIN_RECOVERY``.
    """
    ids = torch.tensor([tok.encode(seq)], device=DEVICE)
    h = forward_batch(ids, None, W)
    h = F.gelu(F.linear(h, W["lm_head.dense.weight"], W["lm_head.dense.bias"]))
    h = F.layer_norm(h, (D_MODEL,), W["lm_head.layer_norm.weight"], W["lm_head.layer_norm.bias"], EPS)
    pred = F.linear(h, W["lm_head.decoder.weight"], W["lm_head.decoder.bias"]).argmax(-1)
    acc = (pred[0, 1:-1] == ids[0, 1:-1]).float().mean().item()
    log(f"Sanity check: LM token recovery on first sequence = {acc:.3f}")
    if acc < MIN_RECOVERY:
        log(f"Error: forward pass does not reproduce the model (recovery {acc:.3f} < {MIN_RECOVERY})")
        raise RuntimeError(f"Forward pass does not reproduce the model (recovery {acc:.3f} < {MIN_RECOVERY})")


############### <contacts>
def predict_batch_contacts(batch_records: list[tuple[str, str]], tok: EsmSequenceTokenizer, W: dict[str, torch.Tensor], min_sep: int) -> pd.DataFrame:
    """Compute cosine similarities between residue embeddings for a padded batch.

    Args:
        batch_records: ``(accession, sequence)`` pairs.
        tok: ESM sequence tokenizer.
        W: Model weights.
        min_sep: Minimum sequence separation ``j - i`` of reported pairs.

    Returns:
        Long table with ``accession``, ``res_i``, ``res_j`` (1-based), ``dist`` and ``value``.
    """
    encoded = [tok.encode(seq) for _, seq in batch_records]
    max_len = max(len(e) for e in encoded)
    pad_id = tok.pad_token_id
    ids = torch.tensor([e + [pad_id] * (max_len - len(e)) for e in encoded], device=DEVICE)
    out_h = forward_batch(ids, ids != pad_id, W)
    dfs = []
    for idx, (acc, _) in enumerate(batch_records):
        slen = len(encoded[idx])
        emb = F.normalize(out_h[idx, 1:slen - 1].float(), p=2, dim=-1)
        res = (emb @ emb.T).cpu().numpy()
        i, j = np.triu_indices(res.shape[0], k=min_sep)
        dfs.append(pd.DataFrame({"accession": pd.Categorical([acc] * len(i)), "res_i": (i + 1).astype(np.int32), "res_j": (j + 1).astype(np.int32), "dist": (j - i).astype(np.int32), "value": res[i, j].astype(np.float32)}))
    return pd.concat(dfs, ignore_index=True)


def make_batches(records: list[tuple[str, str]], batch_size: int, max_tokens: int) -> list[list[tuple[str, str]]]:
    """Group length-sorted records into batches bounded by count and total residues.

    Args:
        records: ``(accession, sequence)`` pairs, ideally sorted by length.
        batch_size: Maximum sequences per batch.
        max_tokens: Total residues above which a batch is closed.

    Returns:
        List of batches.
    """
    batches, cur, cur_tokens = [], [], 0
    for rec in records:
        cur.append(rec)
        cur_tokens += len(rec[1])
        if len(cur) >= batch_size or cur_tokens > max_tokens:
            batches.append(cur)
            cur, cur_tokens = [], 0
    if cur:
        batches.append(cur)
    return batches


############### <main>
def main() -> None:
    """Parse arguments, resolve parameters and write the contacts parquet."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--out", type=Path, help="Output parquet (default: <contacts_outdir>/<label>_contacts.parquet)")
    parser.add_argument("--contacts_outdir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--min_sep", type=int)
    add_yaml_args(parser)
    args = parser.parse_args()
    params = resolve_params(SCRIPT_NAME, DEFAULTS, args)
    if args.fasta is None:
        if args.yaml_out:
            sys.exit(0)
        parser.error("--fasta is required")
    out = args.out or Path(params["contacts_outdir"]) / f"{fasta_label(args.fasta)}_contacts.parquet"
    if out.exists():
        log(f"Target file '{out}' already exists. Skipping execution.")
        log_resources(start_time)
        sys.exit(0)
    records = sorted(parse_fasta(args.fasta), key=lambda r: len(r[1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Loaded and sorted {len(records)} sequences by length from {args.fasta}; min_sep={params['min_sep']}; device={DEVICE}")
    tok, W = EsmSequenceTokenizer(), load_weights()
    check_model(records[0][1], tok, W)
    dfs = []
    for batch in tqdm.tqdm(make_batches(records, params["batch_size"], params["max_tokens"]), desc=f"[PYTHON-INFO] Predicting batched contacts for {args.fasta}"):
        dfs.append(predict_batch_contacts(batch, tok, W, params["min_sep"]))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    df = pd.concat(dfs, ignore_index=True)
    df["accession"] = df["accession"].astype("category")
    df.to_parquet(out, index=False, compression="zstd")
    log(f"Saved {len(df)} residue pairs from {len(records)} accessions to '{out}'")
    log_resources(start_time)


if __name__ == "__main__":
    main()
