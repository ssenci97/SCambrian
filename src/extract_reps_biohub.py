#!/usr/bin/env python3
"""Extract protein sequence representations using ESMC models."""
###################################### imports-and-config
import argparse
import datetime
import gc
import os
import random
import resource
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import tqdm
import yaml
from transformers import AutoModel, AutoTokenizer
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
YAML_PATH = Path("configs/param_configs.yaml")
SCRIPT_NAME = Path(__file__).stem
BATCH_SIZE = 20
CHUNK_SIZE = 500
MODEL_NAMES = ["esmc_300m", "esmc_600m", "esmc_6b"]
HF_REPOS = {"esmc_300m": "biohub/ESMC-300M", "esmc_600m": "biohub/ESMC-600M", "esmc_6b": "biohub/ESMC-6B"}
REPS_OUTDIR = "data/esm-reps"
START_TIME = time.time()
###################################### helper-functions
def parse_fasta(fp):
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
    r = []
    with open(fp) as f:
        next(f, None)
        for idx, l in enumerate(f):
            p = l.strip().split("\t")
            if len(p) >= 2: r.append((idx, p[0], p[1]))
    return r
def esmC_extract(seqs, model, tok):
    if isinstance(seqs, str): seqs = [seqs]
    return [model(**tok(s, return_tensors="pt").to(model.device)).last_hidden_state[0, 1:-1].float().mean(0).cpu().numpy() for s in seqs]
def load_esmc_model(model_name):
    repo = HF_REPOS.get(model_name, model_name)
    print(f"[PYTHON-INFO] Loading {model_name} from Hugging Face hub ({repo})...")
    model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    return model, AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
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
    parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, type=Path, default=YAML_PATH)
    parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, type=Path, default=None)
    yargs, _ = parser.parse_known_args()
    params = {"model_name": MODEL_NAMES, "batch_size": BATCH_SIZE, "reps_outdir": REPS_OUTDIR}
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
        if Path(out).exists():
            print(f"[PYTHON-INFO] Target file '{out}' already exists. Skipping {model_name}.")
            continue
        if records is None: records = parse_fasta(args.fasta) if args.fasta else parse_tsv(args.seqs_tsv)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        model, tok = load_esmc_model(model_name)
        chunk_files = []
        for c_idx, i in enumerate(tqdm.tqdm(range(0, len(records), CHUNK_SIZE), desc=f"[PYTHON-INFO] Processing chunks for {fp} ({model_name})")):
            chunk_recs = records[i:i + CHUNK_SIZE]
            c_ids, c_embs = [], []
            with torch.no_grad():
                for j in range(0, len(chunk_recs), args.batch_size):
                    b = chunk_recs[j:j + args.batch_size]
                    c_ids.extend([r[1] for r in b])
                    seqs = [r[2] for r in b]
                    c_embs.extend(esmC_extract(seqs, model, tok))
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
        del model, tok
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[PYTHON-INFO] date={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} execution_time_sec={time.time()-START_TIME:.2f} max_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f}")
