#!/usr/bin/env python3
"""Download UniProt sequences."""
import argparse, gzip, json, resource, time, warnings, urllib.parse, urllib.request
from datetime import datetime
from io import StringIO
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PARAMS = json.loads(Path("configs/param_config.json").read_text())
CONFIG_PATH = Path(PARAMS["config_path"])
OUT_DIR = Path(PARAMS["seq_out_dir"])
FIELDS = ["accession", "sequence", "length"]

###################################### config
def load_config_proteomes() -> pd.DataFrame:
    if CONFIG_PATH.exists(): return pd.read_csv(CONFIG_PATH, sep="\t", dtype=str).replace(r"^\s*$", pd.NA, regex=True)
    print(f"[PYTHON-INFO] Error: Configuration file {CONFIG_PATH} not found.")
    raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found.")

###################################### uniprot
def fetch_uniprot(query: str, fields: list[str]) -> pd.DataFrame:
    url = f"https://rest.uniprot.org/uniprotkb/stream?query={urllib.parse.quote(query, safe='')}&format=tsv&fields={','.join(fields)}"
    print(f"[PYTHON-INFO] Query sequence URL: {url}")
    with urllib.request.urlopen(url) as resp: return pd.read_csv(StringIO(resp.read().decode("utf-8")), sep="\t")

def get_feature_map() -> dict[str, str]:
    mock = fetch_uniprot("(accession:A0A0R4I9Y1)", FIELDS)
    return dict(zip(mock.columns, FIELDS))

###################################### io
def save_tsv(df: pd.DataFrame, path: Path) -> None:
    if path.exists(): warnings.warn(f"Overwriting existing file: {path}", UserWarning)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def save_fasta(df: pd.DataFrame, path: Path) -> None:
    if path.exists(): warnings.warn(f"Overwriting existing file: {path}", UserWarning)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if pd.notna(row.get("accession")) and pd.notna(row.get("sequence")):
                acc, seq = str(row["accession"]).strip(), str(row["sequence"]).strip()
                f.write(f"{acc if acc.startswith('>') else f'>{acc}'}\n{seq}\n")

def save_ids(df: pd.DataFrame, path: Path) -> None:
    if path.exists(): warnings.warn(f"Overwriting existing file: {path}", UserWarning)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "accession" in df.columns:
        ids = df["accession"].dropna().astype(str).str.strip()
        path.write_text("\n".join(ids) + ("\n" if not ids.empty else ""), encoding="utf-8")

###################################### main
def main() -> None:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Fetch UniProt sequences.")
    parser.add_argument("--query", default=None)
    parser.add_argument("--label", default="example")
    parser.add_argument("--fp_seqs", type=Path)
    parser.add_argument("--fp_fasta", type=Path)
    parser.add_argument("--fp_ids", type=Path)
    args = parser.parse_args()
    feature_map = get_feature_map()
    processed_count, total_fetched = 0, 0
    if args.query:
        df_sub = fetch_uniprot(args.query, FIELDS).rename(columns=feature_map)
        save_tsv(df_sub, args.fp_seqs or (OUT_DIR / f"{args.label}_seqs.tsv"))
        save_fasta(df_sub, args.fp_fasta or (OUT_DIR / f"{args.label}_seqs.fasta"))
        save_ids(df_sub, args.fp_ids or (OUT_DIR / f"{args.label}_ids.txt"))
        processed_count, total_fetched = 1, len(df_sub)
    else:
        df_config = load_config_proteomes()
        if df_config.empty:
            print("[PYTHON-INFO] Error: Configuration file is empty.")
            raise ValueError("Configuration file is empty.")
        items = df_config.to_dict("records")
        for item in tqdm(items, desc="Processing proteomes"):
            pid, lbl = item.get("proteome_id"), item.get("label", "example")
            if args.label and lbl != args.label: continue
            if pd.isna(pid) or not str(pid).strip():
                print(f"[PYTHON-INFO] Error: proteome_id missing for label '{lbl}'.")
                raise ValueError(f"proteome_id missing for label '{lbl}'.")
            pid_str = str(pid).strip()
            if not pid_str.startswith("UP"):
                print(f"[PYTHON-INFO] Error: Malformed Proteome ID detected ('{pid_str}').")
                raise ValueError(f"Malformed Proteome ID detected ('{pid_str}').")
            q = f"proteome:{pid_str}"
            df_sub = fetch_uniprot(q, FIELDS).rename(columns=feature_map)
            save_tsv(df_sub, args.fp_seqs or (OUT_DIR / f"{lbl}_seqs.tsv"))
            save_fasta(df_sub, args.fp_fasta or (OUT_DIR / f"{lbl}_seqs.fasta"))
            save_ids(df_sub, args.fp_ids or (OUT_DIR / f"{lbl}_ids.txt"))
            processed_count += 1
            total_fetched += len(df_sub)
    exec_time, max_mem, dt = f"{time.time() - start_time:.2f}s", f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB", datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[PYTHON-INFO] {dt} {exec_time} {max_mem}")

if __name__ == "__main__": main()