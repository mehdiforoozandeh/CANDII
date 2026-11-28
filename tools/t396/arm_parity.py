"""t396 iteration 2, gate (e): parity of the factorial arms (parameter count and batch stream).

    python tools/t396/arm_parity.py --configs <arm json>... --tables <tables dir> --seeds 0 1 2 \\
        [--reference-run <run dir>] --out parity.tsv

One row per arm config (the arm name is the file stem). Columns:

  arm, config                the arm and its config path
  param_count                `registry.build_model(train.full_config(cfg)).param_count()`
  rel_diff_993615            (param_count - 993615) / 993615
  hash_seed<k>               sha256 over the first 200 training batches of seed k, drawn as
                             `train.py --workers 1` draws them (`train.worker_batches`, batch j from
                             `default_rng([k, 4, j])`), after the held-out exclusion of
                             `heldout_loss`, from the dataset class train.py routes the config to
                             (`res_dataset.make_dataset` under `read_start`, else `OnionDataset`).
                             Per batch it covers the whole `meta` (cell, chromosome, start bin,
                             input and target measurement ids, input and target file, kinds, halved
                             tracks) and the tensors `x_avail`, `x_bin_mask`, `y_avail`, `y_kind`
                             (the hidden-track mask and the bin mask). Counts are not hashed: they
                             differ between a start arm and a count arm by design.
  keys_differ_from_base      config keys whose value differs from the base arm (`;`-joined, `-`
                             if none). The base arm is `count_in_count_out` when one of the configs
                             has that stem, else the first config.
  reference_diff             base arm only, with --reference-run: the keys where
                             `full_config(base)` differs from `<run>/run.json["config"]`
  status                     `pass` when |rel_diff_993615| <= 0.001 and every seed hash equals the
                             base arm's hash of that seed, else `fail` (last column)

Exit 1 when any row fails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

TARGET_PARAMS = 993615
PARAM_TOL = 1e-3
N_BATCHES = 200
WORKERS = 1
BASE_ARM = "count_in_count_out"
MASK_KEYS = ("x_avail", "x_bin_mask", "y_avail", "y_kind")


def param_count(config: dict) -> int:
    from minicandi.registry import build_model
    from minicandi.train import full_config
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return int(build_model(full_config(config)).param_count())


def batch_digest(batch: dict) -> bytes:
    """The bytes one batch adds to the stream hash (module docstring, `hash_seed<k>`)."""
    parts = [json.dumps(batch["meta"], sort_keys=True, default=str).encode()]
    for k in MASK_KEYS:
        a = np.ascontiguousarray(torch.as_tensor(batch[k]).cpu().numpy())
        parts.append(f"{k}:{a.dtype.str}:{a.shape}".encode())
        parts.append(a.tobytes())
    return b"\0".join(parts)


def stream_hash(config: dict, tables, seed: int, n_batches: int = N_BATCHES) -> str:
    """sha256 of the first `n_batches` training batches of `seed`, as train.py draws them with
    `--workers 1` (module docstring)."""
    from minicandi import train as T
    from minicandi.res_dataset import make_dataset
    cfg = T.full_config(config)
    factory = make_dataset if cfg.get("read_start") else T.OnionDataset
    saved = T.OnionDataset
    T.OnionDataset = factory            # heldout_batches builds its dataset through this name
    try:
        ds = factory(tables, "train", cfg, int(seed))
        if cfg.get("heldout_loss"):
            held = T.heldout_batches(tables, cfg)
            ds.exclude_windows([ds.window_span(m) for b in held for m in b["meta"]])
        h = hashlib.sha256()
        it = T.worker_batches(ds, int(cfg["batch_size"]), int(seed), 0, int(n_batches), WORKERS)
        for b in it:
            h.update(batch_digest(b))
        del it
    finally:
        T.OnionDataset = saved
    return h.hexdigest()


def differing_keys(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def parity_rows(configs: list, tables, seeds: list, reference_run=None,
                n_batches: int = N_BATCHES) -> list[dict]:
    paths = [Path(p) for p in configs]
    cfgs = {p: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    base = next((p for p in paths if p.stem == BASE_ARM), paths[0])
    ref_diff = "-"
    if reference_run:
        from minicandi.train import full_config
        run_cfg = json.loads((Path(reference_run) / "run.json").read_text(encoding="utf-8"))
        ref_diff = ";".join(differing_keys(full_config(cfgs[base]), run_cfg["config"])) or "-"
    hashes = {p: {s: stream_hash(cfgs[p], tables, s, n_batches) for s in seeds} for p in paths}
    rows = []
    for p in paths:
        n = param_count(cfgs[p])
        rel = (n - TARGET_PARAMS) / TARGET_PARAMS
        ok = abs(rel) <= PARAM_TOL and all(hashes[p][s] == hashes[base][s] for s in seeds)
        row = {"arm": p.stem, "config": str(p), "param_count": n,
               f"rel_diff_{TARGET_PARAMS}": f"{rel:.6g}"}
        row.update({f"hash_seed{s}": hashes[p][s] for s in seeds})
        row["keys_differ_from_base"] = ";".join(differing_keys(cfgs[p], cfgs[base])) or "-"
        row["reference_diff"] = ref_diff if p == base else "-"
        row["status"] = "pass" if ok else "fail"
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--reference-run", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = parity_rows(a.configs, a.tables, a.seeds, a.reference_run)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['arm']}\tparams={r['param_count']}\t{r['status']}")
    return 1 if any(r["status"] == "fail" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
