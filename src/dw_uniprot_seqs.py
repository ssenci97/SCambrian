#!/usr/bin/env python3
"""Download UniProt sequences based on generic text queries."""
import argparse, random, resource, time, urllib.parse, urllib.request, warnings
from datetime import datetime
from io import StringIO
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm
SEED = 42
random.seed(SEED)
DEFAULT_CONFIG_PATH = "configs/query_configs.tsv"
DEFAULT_OUT_DIR = "data/sequences"
###################################### config
def load_config(path: Path) -> pd.DataFrame:
    if path.exists(): return pd.read_csv(path, sep="\t", dtype=str).replace(r"^\s*$", pd.NA, regex=True)
    print(f"[PYTHON-INFO] Error: Configuration file {path} not found.")
    raise FileNotFoundError(f"Configuration file {path} not found.")
###################################### uniprot
def fetch_uniprot(query: str, fields: list[str]) -> pd.DataFrame:
    url = f"https://rest.uniprot.org/uniprotkb/stream?query={urllib.parse.quote(query, safe='')}&format=tsv&fields={','.join(fields)}"
    print(f"[PYTHON-INFO] Query sequence URL: {url}")
    with urllib.request.urlopen(url) as resp: return pd.read_csv(StringIO(resp.read().decode("utf-8")), sep="\t")
def get_feature_map(fields: list[str]) -> dict[str, str]:
    mock = fetch_uniprot("(accession:A0A0R4I9Y1)", fields)
    return dict(zip(mock.columns, fields))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None)
    parser.add_argument("--label", default="example")
    parser.add_argument("--fp_seqs", type=Path)
    parser.add_argument("--fp_fasta", type=Path)
    parser.add_argument("--fp_ids", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yaml-in", nargs="?", const=Path("configs/param_configs.yaml"), type=Path, default=Path("configs/param_configs.yaml"))
    parser.add_argument("--yaml-out", nargs="?", const=Path("configs/param_configs.yaml"), type=Path)
    args = parser.parse_args()
    script_name = Path(__file__).stem
    params = {"config_path": DEFAULT_CONFIG_PATH, "seq_out_dir": DEFAULT_OUT_DIR, "overwrite": False}
    if args.yaml_in and args.yaml_in.exists():
        data = yaml.safe_load(args.yaml_in.read_text())
        if isinstance(data, dict) and script_name in data:
            params.update(data[script_name])
    params["overwrite"] = bool(params["overwrite"] or args.overwrite)
    if args.yaml_out:
        yaml_out_path = args.yaml_out if isinstance(args.yaml_out, Path) else Path("configs/param_configs.yaml")
        yaml_out_path.parent.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_load(yaml_out_path.read_text()) if yaml_out_path.exists() else {}
        if not isinstance(data, dict): data = {}
        data[script_name] = params
        yaml_out_path.write_text(yaml.dump(data, default_flow_style=False))
    config_path = Path(params["config_path"])
    out_dir = Path(params["seq_out_dir"])
    fields = ["accession", "sequence", "length"]
    feature_map = get_feature_map(fields)
    if args.query:
        fps = [args.fp_seqs or (out_dir / f"{args.label}_seqs.tsv"), args.fp_fasta or (out_dir / f"{args.label}_seqs.fasta"), args.fp_ids or (out_dir / f"{args.label}_ids.txt")]
        if not params["overwrite"] and all(p.exists() for p in fps): print(f"[PYTHON-INFO] Sequences of '{args.label}' already exist. Skipping.")
        else:
            df_sub = fetch_uniprot(args.query, fields).rename(columns=feature_map)
            save_tsv(df_sub, fps[0])
            save_fasta(df_sub, fps[1])
            save_ids(df_sub, fps[2])
    else:
        df_config = load_config(config_path)
        if df_config.empty:
            print("[PYTHON-INFO] Error: Configuration file is empty.")
            raise ValueError("Configuration file is empty.")
        for item in tqdm(df_config.to_dict("records"), desc="Processing queries"):
            q_val, lbl = item.get("query"), item.get("label", "example")
            if args.label and lbl != args.label: continue
            if pd.isna(q_val) or not str(q_val).strip():
                print(f"[PYTHON-INFO] Error: query missing for label '{lbl}'.")
                raise ValueError(f"query missing for label '{lbl}'.")
            fps = [args.fp_seqs or (out_dir / f"{lbl}_seqs.tsv"), args.fp_fasta or (out_dir / f"{lbl}_seqs.fasta"), args.fp_ids or (out_dir / f"{lbl}_ids.txt")]
            if not params["overwrite"] and all(p.exists() for p in fps):
                print(f"[PYTHON-INFO] Sequences of '{lbl}' already exist. Skipping.")
                continue
            df_sub = fetch_uniprot(str(q_val).strip(), fields).rename(columns=feature_map)
            save_tsv(df_sub, fps[0])
            save_fasta(df_sub, fps[1])
            save_ids(df_sub, fps[2])
    exec_time, max_mem, dt = f"{time.time() - start_time:.2f}s", f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.2f}MB", datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[PYTHON-INFO] {dt} {exec_time} {max_mem}")
if __name__ == "__main__": main()