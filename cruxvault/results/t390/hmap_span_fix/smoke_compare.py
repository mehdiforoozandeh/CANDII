"""Compare two `tools/t125/smoke.py <dir> h` work dirs number by number.

    python smoke_compare.py <dir_a> <dir_b> <out.tsv>

Numeric leaves of every scores.json / results_h.json / summary_data_h.json / pair_*.json (keys
naming times, dates or paths are skipped), the loss columns of every train_log.tsv, and every
tensor of every ckpt.pt. Writes one row per differing value (path, key, a, b) and prints the
counts: values compared, values equal.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

SKIP = ("seconds", "timing", "time", "utc", "wall", "created", "threads", "host", "path", "dir", "blacklist")


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if any(s in str(k).lower() for s in SKIP):
                continue
            yield from leaves(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    elif isinstance(x, bool) or x is None or isinstance(x, str):
        return
    elif isinstance(x, (int, float)):
        yield pre, float(x)


def tensors(x, pre=""):
    if isinstance(x, torch.Tensor):
        yield pre, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from tensors(v, f"{pre}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from tensors(v, f"{pre}[{i}]")


def main() -> None:
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n = same = 0
    diffs = []
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    for rel in files:
        pa, pb = a / rel, b / rel
        if rel.parts[0] == "products" or rel.parts[0] == "ctl_cache":
            continue
        if not pb.exists():
            diffs.append((str(rel), "<missing in b>", "", ""))
            continue
        if pa.suffix == ".json" and "timing" not in pa.name:
            la, lb = dict(leaves(json.loads(pa.read_text()))), dict(leaves(json.loads(pb.read_text())))
            for k in sorted(set(la) | set(lb)):
                n += 1
                va, vb = la.get(k), lb.get(k)
                if va == vb or (va is not None and vb is not None and va != va and vb != vb):
                    same += 1
                else:
                    diffs.append((str(rel), k, repr(va), repr(vb)))
        elif pa.name == "train_log.tsv":
            ra = list(csv.DictReader(open(pa), delimiter="\t"))
            rb = list(csv.DictReader(open(pb), delimiter="\t"))
            for i, (x, y) in enumerate(zip(ra, rb)):
                for col in ("loss", "val_loss"):
                    n += 1
                    if x.get(col) == y.get(col):
                        same += 1
                    else:
                        diffs.append((str(rel), f"row{i}/{col}", x.get(col), y.get(col)))
        elif pa.suffix == ".pt":
            ta = dict(tensors(torch.load(pa, map_location="cpu", weights_only=False)))
            tb = dict(tensors(torch.load(pb, map_location="cpu", weights_only=False)))
            for k in sorted(set(ta) | set(tb)):
                n += 1
                if k in ta and k in tb and ta[k].shape == tb[k].shape and torch.equal(ta[k], tb[k]):
                    same += 1
                else:
                    diffs.append((str(rel), k, "tensor", "tensor differs"))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["file", "key", "a", "b"])
        w.writerows(diffs)
    print(f"compared {n} values, {same} equal, {len(diffs)} differ -> {out}")


if __name__ == "__main__":
    main()
