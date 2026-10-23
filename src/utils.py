#!/usr/bin/env python3
"""Shared helpers for logging, YAML/CLI parameter resolution and sequence/embedding I/O."""
############### <imports>
import argparse
import copy
import fcntl
import os
import resource
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

############### <constants>
YAML_PATH = Path("configs/param_configs.yaml")
FASTA_SUFFIXES = (".fasta", ".fa", ".faa")
LABEL_SUFFIXES = ("_dedup", "_seqs", "_reps")


############### <logging>
def log(msg: str) -> None:
    """Print a message using the pipeline's Python logging convention.

    Args:
        msg: Message to print after the ``[PYTHON-INFO]`` tag.
    """
    print(f"[PYTHON-INFO] {msg}", flush=True)


def log_resources(start_time: float) -> None:
    """Print the closing timestamp, elapsed wall time and peak resident memory.

    Args:
        start_time: Value of ``time.time()`` captured when the script started.
    """
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    log(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {time.time() - start_time:.2f}s {max_mem:.2f}MB")


############### <config>
def add_yaml_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--yaml-in`` and ``--yaml-out`` options.

    ``--yaml-in`` is active by default and reads ``configs/param_configs.yaml``.
    ``--yaml-out`` is opt-in and writes the resolved parameters back to the YAML.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument("--yaml-in", nargs="?", const=YAML_PATH, default=YAML_PATH, type=Path, help="YAML to read parameters from (default: %(default)s)")
    parser.add_argument("--no-yaml-in", action="store_true", help="Ignore the YAML and use built-in defaults plus CLI values")
    parser.add_argument("--yaml-out", nargs="?", const=YAML_PATH, default=None, type=Path, help="Write resolved parameters to this YAML (default when flag given: %(const)s)")


def write_yaml_section(path: Path, section: str, params: dict) -> None:
    """Atomically replace one top-level section of a YAML file under a file lock.

    Args:
        path: YAML file to update; created if missing.
        section: Top-level key to replace, usually the script stem.
        params: Parameters stored under ``section``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.stem}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        if not isinstance(data, dict):
            data = {}
        data[section] = params
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
        os.replace(tmp_path, path)
    log(f"Wrote parameters for '{section}' to {path}")


def resolve_params(script_name: str, defaults: dict, args: argparse.Namespace) -> dict:
    """Merge built-in defaults, the script's YAML section and explicit CLI values.

    Precedence is defaults < YAML < CLI. A CLI value counts as explicit when it is
    not ``None``, so every parameter argument must default to ``None``. When
    ``--yaml-out`` is given the resolved parameters are written back to the YAML.

    Args:
        script_name: YAML section name, normally ``Path(__file__).stem``.
        defaults: Built-in parameter defaults; only these keys are resolved.
        args: Parsed arguments containing ``yaml_in``, ``no_yaml_in`` and ``yaml_out``.

    Returns:
        The resolved parameter dictionary.
    """
    params = copy.deepcopy(defaults)
    if not args.no_yaml_in and args.yaml_in and Path(args.yaml_in).exists():
        data = yaml.safe_load(Path(args.yaml_in).read_text()) or {}
        section = data.get(script_name) if isinstance(data, dict) else None
        if isinstance(section, dict):
            params.update({k: v for k, v in section.items() if k in defaults})
    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            params[key] = value
    if args.yaml_out:
        write_yaml_section(args.yaml_out, script_name, params)
    return params


############### <sequences>
def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA file into ``(identifier, sequence)`` tuples.

    The identifier is the first whitespace-delimited token of the header.

    Args:
        path: FASTA file to read.

    Returns:
        Records in file order.
    """
    records, cur_id, cur_seq = [], None, []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    records.append((cur_id, "".join(cur_seq)))
                cur_id, cur_seq = line[1:].split()[0], []
            else:
                cur_seq.append(line)
    if cur_id:
        records.append((cur_id, "".join(cur_seq)))
    return records


def list_fastas(directory: Path, pattern: str = "*") -> list[Path]:
    """List FASTA files in a directory, sorted by name.

    Args:
        directory: Directory to scan (not recursive).
        pattern: Glob stem applied before the FASTA suffix.

    Returns:
        Sorted paths ending in ``.fasta``, ``.fa`` or ``.faa``.
    """
    directory = Path(directory)
    return sorted(p for suffix in FASTA_SUFFIXES for p in directory.glob(f"{pattern}{suffix}") if p.is_file())


def fasta_label(path: Path) -> str:
    """Derive a dataset label from a sequence or embedding file name.

    Args:
        path: File such as ``Mammalia_seqs.fasta`` or ``Mammalia_dedup.fasta``.

    Returns:
        The stem without known pipeline suffixes, e.g. ``Mammalia``.
    """
    stem = Path(path).stem
    for suffix in LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


############### <embeddings>
def load_npz_embeddings(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate embeddings and identifiers from ``.npz`` or ``.pt`` files.

    Args:
        paths: Files holding ``embeddings``/``ids`` (npz) or ``tensor``/``residue_ids`` (pt).

    Returns:
        Tuple of the stacked embedding matrix and the matching identifier array.

    Raises:
        ValueError: If a file has an unsupported suffix.
    """
    tensors, ids = [], []
    for path in paths:
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            tensors.append(np.asarray(data["embeddings"]))
            ids.append(np.asarray(data["ids"]))
        elif path.suffix == ".pt":
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)
            tensor = data["tensor"]
            tensors.append(tensor.numpy() if torch.is_tensor(tensor) else np.asarray(tensor))
            ids.append(np.asarray(data["residue_ids"]))
        else:
            log(f"Error: unsupported embedding file {path}")
            raise ValueError(f"Unsupported embedding file {path}")
    return np.concatenate(tensors, axis=0), np.concatenate(ids, axis=0)
