#!/usr/bin/env python3
"""Predict ESMC-6B residue-pair contact values for FASTA sequences and save them as a filtered long-format dataframe."""
###################################### imports-and-config
import argparse, datetime, gc, glob, json, os, re, resource, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, tqdm
from esm.tokenization import EsmSequenceTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
warnings.filterwarnings("ignore", category=FutureWarning)
PARAMS = json.loads(Path("configs/param_config.json").read_text()) if Path("configs/param_config.json").exists() else {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
MIN_SEP = 6
CONTACTS_OUTDIR = PARAMS.get("contacts_outdir", "data/contacts_esmc-6b/")
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
def forward(ids, W):
    L = ids.shape[1]
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
        q, k, v = (t.view(1, L, N_HEADS, HEAD_DIM).transpose(1, 2) for t in (q, k, v))
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, L, D_MODEL)
        x = x + F.linear(a, W[p + "self_attn.o_proj.weight"])
        h = F.layer_norm(x, (D_MODEL,), W[p + "post_attention_layernorm.weight"], W[p + "post_attention_layernorm.bias"], EPS)
        x = x + F.linear(F.silu(F.linear(h, W[p + "mlp.gate_proj.weight"])) * F.linear(h, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])
    return F.layer_norm(x, (D_MODEL,), W["esmc.norm.weight"], None, EPS)
@torch.no_grad()
def check_model(seq, tok, W):
    ids = torch.tensor([tok.encode(seq)], device=DEVICE)
    h = forward(ids, W)
    h = F.gelu(F.linear(h, W["lm_head.dense.weight"], W["lm_head.dense.bias"]))
    h = F.layer_norm(h, (D_MODEL,), W["lm_head.layer_norm.weight"], W["lm_head.layer_norm.bias"], EPS)
    pred = F.linear(h, W["lm_head.decoder.weight"], W["lm_head.decoder.bias"]).argmax(-1)
    acc = (pred[0, 1:-1] == ids[0, 1:-1]).float().mean().item()
    print(f"[PYTHON-INFO] Sanity check: LM token recovery on first sequence = {acc:.3f}")
    if acc < MIN_RECOVERY: raise RuntimeError(f"[PYTHON-INFO] Forward pass does not reproduce the model (recovery {acc:.3f} < {MIN_RECOVERY})")
def predict_contacts(acc, seq, tok, W, min_sep):
    ids = torch.tensor([tok.encode(seq)], device=DEVICE)
    emb = F.normalize(forward(ids, W)[0, 1:-1].float(), p=2, dim=-1)
    res = (emb @ emb.T).cpu().numpy()
    i, j = np.triu_indices(res.shape[0], k=min_sep)
    return pd.DataFrame({"accession": pd.Categorical([acc] * len(i)), "res_i": (i + 1).astype(np.int32), "res_j": (j + 1).astype(np.int32), "dist": (j - i).astype(np.int32), "value": res[i, j].astype(np.float32)})
###################################### main-execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_sep", type=int, default=MIN_SEP)
    args = parser.parse_args()
    records = parse_fasta(args.fasta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[PYTHON-INFO] Loaded {len(records)} sequences from {args.fasta}; filtering out pairs with distance <= 5 (min_sep={args.min_sep}) and saving upper triangle only")
    tok, W = EsmSequenceTokenizer(), load_weights()
    check_model(records[0][1], tok, W)
    dfs = []
    for acc, seq in tqdm.tqdm(records, desc=f"[PYTHON-INFO] Predicting contacts for {args.fasta}"):
        dfs.append(predict_contacts(acc, seq, tok, W, args.min_sep))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    df = pd.concat(dfs, ignore_index=True)
    df["accession"] = df["accession"].astype("category")
    df.to_parquet(args.out, index=False, compression="zstd")
    print(f"[PYTHON-INFO] Saved {len(df)} filtered residue pairs from {len(records)} accessions to '{args.out}'")
    print(f"[PYTHON-INFO] date={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} execution_time_sec={time.time()-START_TIME:.2f} max_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f}")