"""Synthetic controls for the t125 part-2 smokes and tests, on `ladder.synth.make_products` output.

`add_controls(products_dir, manifest_tsv, seed=0)`: for every product of track T1 (the synthetic
track with a matched control; T2 is DNase and stays without one) write
`<pid>/control_counts25.npz` = Poisson(lambda) per bin on every chromosome of the product's counts,
lambda = 0.6 x the mean count of the track base (`T1__base__base`, over all its bins), written with
`tools/t112/bin25.py::write_npz` (uint32), and rewrite the manifest's `control_counts25_md5`
column (columns and row order kept). Every T1 product draws its own control from
`default_rng([seed, 7300, i])`, i = its index among the T1 pids in sorted order, so a rerun with the
same seed writes the same arrays. Returns the pids written, sorted.

`rl_tsv` (optional): also write the measured-control-read-length table that `hmap.products.load`
needs (`products.read_control_read_length` format), one row per control, the control read length
= the product's treatment read length from the manifest (synthetic: nothing was measured).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[2]

CONTROL_TRACK = "T1"
CONTROL_NPZ = "control_counts25.npz"
MD5_COLUMN = "control_counts25_md5"
LAMBDA_FRACTION = 0.6
RL_COLUMNS = ("pid", "control_tagalign", "control_read_length_mean", "n_lines_read")


def _load_bin25():
    spec = importlib.util.spec_from_file_location("t112_bin25_for_hmap_synth",
                                                  _TOOLS / "t112" / "bin25.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _read_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def add_controls(products_dir, manifest_tsv, seed: int = 0, rl_tsv=None) -> list[str]:
    products_dir, manifest_tsv = Path(products_dir), Path(manifest_tsv)
    bin25 = _load_bin25()
    with manifest_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    if MD5_COLUMN not in cols:
        raise ValueError(f"{manifest_tsv}: no column {MD5_COLUMN}")
    pids = sorted(r["pid"] for r in rows if r["track"] == CONTROL_TRACK)
    if not pids:
        return []
    base_pid = f"{CONTROL_TRACK}__base__base"
    base = _read_npz(products_dir / base_pid / "counts25.npz")
    lam = LAMBDA_FRACTION * float(np.concatenate([v.astype(np.float64)
                                                  for v in base.values()]).mean())
    md5 = {}
    for i, pid in enumerate(pids):
        counts = _read_npz(products_dir / pid / "counts25.npz")
        rng = np.random.default_rng([int(seed), 7300, i])
        ctl = {c: rng.poisson(lam, v.shape[0]).astype(np.uint32) for c, v in counts.items()}
        path = products_dir / pid / CONTROL_NPZ
        bin25.write_npz(path, ctl)
        md5[pid] = _md5(path)
    for r in rows:
        if r["pid"] in md5:
            r[MD5_COLUMN] = md5[r["pid"]]
    tmp = manifest_tsv.with_name(manifest_tsv.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(manifest_tsv)
    if rl_tsv is not None:
        rl = {r["pid"]: r["read_length"] for r in rows}
        with Path(rl_tsv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(RL_COLUMNS)
            for pid in pids:
                w.writerow([pid, "synthetic", float(rl[pid]), 0])
    return pids
