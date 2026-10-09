#!/usr/bin/env python3
"""Extract protein sequence representations using ESMC models with intermediate chunking."""

###################################### imports-and-config
import argparse, datetime, gc, glob, inspect, json, os, resource, time, warnings, numpy as np, torch, tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Suppress PyTorch and third-party FutureWarnings (e.g. from esm.pretrained)
warnings.filterwarnings("ignore", category=FutureWarning)

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

PARAMS = json.loads(Path("configs/param_config.json").read_text())
BATCH_SIZE = PARAMS.get("batch_size", 500)
CFG = LogitsConfig(sequence=True, return_embeddings=True)
CHUNK_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = PARAMS.get("model_name", "esmc_300m")
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

def esmC_extract(seqs, model, device):
    if isinstance(seqs, str): seqs = [seqs]
    tensors = [model.encode(ESMProtein(sequence=s)) for s in seqs]
    return [model.logits(t, CFG).embeddings.squeeze(0)[1:-1].mean(dim=0).cpu().numpy() for t in tensors]

def load_esmc_model(model_name, device):
    """Unified loader: handles esmc_300m, esmc_600m, and biohub/ESMC-6B HF checkpoint."""
    hf_6b_names = ["esmc_6b", "esm_6b", "biohub/ESMC-6B", "biohub/esmc-6b-2024-12"]

    if model_name in hf_6b_names:
        print(f"[PYTHON-INFO] Loading 6B model from Hugging Face hub (biohub/ESMC-6B)...")
        model_dir = snapshot_download(repo_id="biohub/ESMC-6B")
        
        config_path = os.path.join(model_dir, "config.json")
        model_kwargs = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            sig = inspect.signature(ESMC.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            model_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
        
        if model_kwargs:
            print("[PYTHON-INFO] Initializing ESMC architecture from config.json...")
            model = ESMC(**model_kwargs)
        else:
            print("[PYTHON-INFO] Initializing default ESMC architecture...")
            model = ESMC.from_pretrained("esmc_600m")

        raw_state_dict = {}
        st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if st_files:
            for f in st_files:
                raw_state_dict.update(load_file(f))
        else:
            pt_files = sorted(glob.glob(os.path.join(model_dir, "*.bin")) + glob.glob(os.path.join(model_dir, "*.pt")))
            for f in pt_files:
                raw_state_dict.update(torch.load(f, map_location="cpu", weights_only=True))

        clean_state_dict = {}
        for k, v in raw_state_dict.items():
            new_key = k
            if new_key.startswith("model."):
                new_key = new_key[6:]
            if new_key.startswith("_orig_mod."):
                new_key = new_key[10:]
            clean_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        if missing:
            print(f"[PYTHON-INFO] Missing keys count: {len(missing)}")
        if unexpected:
            print(f"[PYTHON-INFO] Unexpected keys count: {len(unexpected)}")
            
        return model.to(device).eval()
    else:
        print(f"[PYTHON-INFO] Loading standard ESMC model via registry: {model_name}")
        return ESMC.from_pretrained(model_name).to(device).eval()

###################################### main-execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--seqs_tsv")
    g.add_argument("--fasta")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    fp = args.fasta if args.fasta else args.seqs_tsv
    records = parse_fasta(args.fasta) if args.fasta else parse_tsv(args.seqs_tsv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    # Unified model loading
    model = load_esmc_model(args.model_name, DEVICE)
    
    # Unified chunked processing loop for all models
    chunk_files = []
    for c_idx, i in enumerate(tqdm.tqdm(range(0, len(records), CHUNK_SIZE), desc=f"[PYTHON-INFO] Processing chunks for {fp}")):
        chunk_recs = records[i:i + CHUNK_SIZE]
        c_ids, c_embs = [], []
        with torch.no_grad():
            for j in range(0, len(chunk_recs), args.batch_size):
                b = chunk_recs[j:j + args.batch_size]
                c_ids.extend([r[1] for r in b])
                c_embs.extend(esmC_extract([r[2] for r in b], model, DEVICE))
        c_file = f"{args.out}.tmp_{c_idx}.npz"
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
    np.savez(args.out, embeddings=np.concatenate(all_embs, axis=0), ids=np.concatenate(all_ids, axis=0))

    print(f"[PYTHON-INFO] date={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} execution_time_sec={time.time()-START_TIME:.2f} max_memory_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f}")