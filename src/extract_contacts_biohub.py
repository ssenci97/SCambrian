#!/usr/bin/env python3
"""Predict ESMC residue-pair contact values for FASTA sequences with length-sorted batching."""
###################################### imports-and-config
import argparse
import datetime
import gc
import glob
import os
import random
import resource
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import yaml
from esm.tokenization import EsmSequenceTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
YAML_PATH = Path("configs/param_configs.yaml")
SCRIPT_NAME = Path(__file__).stem
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
MIN_SEP = 6
BATCH_SIZE = 300
MODEL_NAME = "esmc_6b"
CONTACTS_OUTDIR = "data/esm-contacts"
HF_REPO = "biohub/ESMC-6B"
N_LAYERS, N_HEADS, HEAD_DIM, D_MODEL, ROPE_THETA, EPS = 80, 40, 64, 2560, 10000.0, 1e-5
MIN_RECOVERY = 0.5
MODEL_TAG = "ESMC-6b"
START_TIME = time.time()
###################################### helper-functions
def parse_fasta(fp):
    r, cid, cseq = [], None, []
    with open(fp) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            if l.startswith(">"):
                if cid: r.append((cid, "".join(cseq)))
                cid, cseq = l[1:].split()[0], []
            else: cseq.append(l)
    if cid: r.append((cid, "".join(cseq)))
    return r
def load_weights():
    W = {}
    for f in sorted(glob.glob(os.path.join(snapshot_download(repo_id=HF_REPO), "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys(): W[k] = sf.get_tensor(k).to(device=DEVICE, dtype=DTYPE)
    print(f"[PYTHON-INFO] Loaded {len(W)} tensors, {sum(v.numel() for v in W.values())/1e9:.2f}B parameters")
    return W
def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)
@torch.no_grad()
def forward_batch(ids, mask, W):
    B, L = ids.shape
    x = F.embedding(ids, W["esmc.embed_tokens.weight"])
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE).float() / HEAD_DIM))
    fr = torch.outer(torch.arange(L, device=DEVICE).float(), inv)
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos()[None, None].to(x.dtype), emb.sin()[None, None].to(x.dtype)
    for i in range(N_LAYERS):
        p = f"esmc.layers.{i}."
        h = F.layer_norm(x, (D_MODEL,), W[p + "input_layernorm.weight"], W[p + "input_layernorm.bias"], EPS)
        q = F.layer_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]), (D_MODEL,), W[p + "self_attn.q_norm.weight"], None, EPS)
        k = F.layer_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]), (D_MODEL,), W[p + "self_attn.k_norm.weight"], None, EPS)
        v = F.linear(h, W[p + "self_attn.v_proj.weight"])
        q, k, v = (t.view(B, L, N_HEADS, HEAD_DIM).transpose(1, 2) for t in (q, k, v))
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None).transpose(1, 2).reshape(B, L, D_MODEL)
        x = x + F.linear(a, W[p + "self_attn.o_proj.weight"])
        h = F.layer_norm(x, (D_MODEL,), W[p + "post_attention_layernorm.weight"], W[p + "post_attention_layernorm.bias"], EPS)
        x = x + F.linear(F.silu(F.linear(h, W[p + "mlp.gate_proj.weight"])) * F.linear(h, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])
    return F.layer_norm(x, (D_MODEL,), W["esmc.norm.weight"], None, EPS)
@torch.no_grad()
def check_model(seq, tok, W):
    ids = torch.tensor([tok.encode(seq)], device=DEVICE)
    h = forward_batch(ids, None, W)
    h = F.gelu(F.linear(h, W["lm_head.dense.weight"], W["lm_head.dense.bias"]))
    h = F.layer_norm(h, (D_MODEL,), W["lm_head.layer_norm.weight"], W["lm_head.layer_norm.bias"], EPS)
    pred = F.linear(h, W["lm_head.decoder.weight"], W["lm_head.decoder.bias"]).argmax(-1)
    acc = (pred[0, 1:-1] == ids[0, 1:-1]).float().mean().item()
    print(f"[PYTHON-INFO] Sanity check: LM token recovery on first sequence = {acc:.3f}")
    if acc < MIN_RECOVERY: raise RuntimeError(f"[PYTHON-INFO] Forward pass does not reproduce the model (recovery {acc:.3f} < {MIN_RECOVERY})")
def predict_batch_contacts(batch_records, tok, W, min_sep):
    encoded = [tok.encode(seq) for _, seq in batch_records]
    max_len = max(len(e) for e in encoded)
    pad_id = tok.pad_token_id if hasattr(tok, "pad_token_id") else 0
    ids_padded = [e + [pad_id] * (max_len - len(e)) for e in encoded]
    ids_tensor = torch.tensor(ids_padded, device=DEVICE)
    out_h = forward_batch(ids_tensor, None, W)
    dfs = []
    for idx, (acc, seq) in enumerate(batch_records):
        slen = len(encoded[idx])
        emb = F.normalize(out_h[idx, 1:slen-1].float(), p=2, dim=-1)
        res = (emb @ emb.T).cpu().numpy()
        i, j = np.triu_indices(res.shape[0], k=min_sep)
        dfs.append(pd.DataFrame({"accession": pd.Categorical([acc] * len(i)), "res_i": (i + 1).astype(np.int32), "res_j": (j + 1).astype(np.int32), "dist": (j - i).astype(np.int32), "value": res[i, j].astype(np.float32)}))
    return pd.concat(dfs, ignore_index=True)
###################################### main-execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--contacts_outdir", default=CONTACTS_OUTDIR)
    parser.add_argument("--out")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min_sep", type=int, default=MIN_SEP)
    parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, type=Path, default=YAML_PATH)
    parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, type=Path, default=None)
    yargs, _ = parser.parse_known_args()
    params = {"model_name": MODEL_NAME, "batch_size": BATCH_SIZE, "min_sep": MIN_SEP, "contacts_outdir": CONTACTS_OUTDIR}
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
    if not args.fasta:
        if args.yaml_out: sys.exit(0)
        parser.error("--fasta is required")
    if not args.out:
        args.out = os.path.join(args.contacts_outdir, args.model_name, f"{Path(args.fasta).stem.removesuffix('_dedup').removesuffix('_seqs')}_contacts.parquet")
    if Path(args.out).exists():
        print(f"[PYTHON-INFO] Target file '{args.out}' already exists. Skipping execution.")
        print(f"[PYTHON-INFO] date={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} execution_time_sec={time.time()-START_TIME:.2f} max_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f}")
        sys.exit(0)
    records = parse_fasta(args.fasta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda x: len(x[1]))
    print(f"[PYTHON-INFO] Loaded and sorted {len(records)} sequences by length from {args.fasta}; min_sep={args.min_sep}")
    tok, W = EsmSequenceTokenizer(), load_weights()
    check_model(records[0][1], tok, W)
    batches, cur_batch = [], []
    for r in records:
        cur_batch.append(r)
        if len(cur_batch) >= args.batch_size or sum(len(x[1]) for x in cur_batch) > 10000:
            batches.append(cur_batch)
            cur_batch = []
    if cur_batch: batches.append(cur_batch)
    dfs = []
    for b in tqdm.tqdm(batches, desc=f"[PYTHON-INFO] Predicting batched contacts for {args.fasta}"):
        dfs.append(predict_batch_contacts(b, tok, W, args.min_sep))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    df = pd.concat(dfs, ignore_index=True)
    df["accession"] = df["accession"].astype("category")
    df.to_parquet(args.out, index=False, compression="zstd")
    print(f"[PYTHON-INFO] Saved {len(df)} filtered residue pairs from {len(records)} accessions to '{args.out}'")
    print(f"[PYTHON-INFO] date={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} execution_time_sec={time.time()-START_TIME:.2f} max_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f}")


