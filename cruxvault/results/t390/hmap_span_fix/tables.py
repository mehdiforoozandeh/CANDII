"""Before/after tables of the hmap span fix.

    python tables.py synth <old.npz> <new.npz> <out.tsv>
    python tables.py smoke <old_smoke_dir> <new_smoke_dir> <out.tsv>

synth: per synthetic product and array of `ref_formula.py`: elements, elements that differ, mean
old, mean new, max |new - old|. smoke: every row of h/agg/h_alone_summary.tsv and
h_chain_summary.tsv whose `mean` differs, with old and new means and the relative change.
"""
from __future__ import annotations

import csv
import sys

import numpy as np


def synth(old, new, out):
    o, n = np.load(old), np.load(new)
    rows = []
    for k in o.files:
        if "/" not in k:
            continue
        a, b = o[k].astype(np.float64), n[k].astype(np.float64)
        fin = np.isfinite(a) & np.isfinite(b)
        prod, arr = k.split("/")
        rows.append(dict(product=prod, array=arr, n=a.size, n_differ=int(np.count_nonzero(a != b)),
                         mean_old=f"{a[fin].mean():.6g}", mean_new=f"{b[fin].mean():.6g}",
                         max_abs_diff=f"{np.abs(b[fin] - a[fin]).max():.6g}"))
    write(out, rows)


def smoke(old, new, out):
    rows = []
    for name, keys in (("h_alone_summary.tsv", ("h_name", "mark_class", "metric")),
                       ("h_chain_summary.tsv", ("f_rung", "h_name", "mark_class", "metric"))):
        def read(d):
            with open(f"{d}/h/agg/{name}") as fh:
                return {tuple(r[k] for k in keys): r for r in csv.DictReader(fh, delimiter="\t")}
        a, b = read(old), read(new)
        assert set(a) == set(b), name
        for key in sorted(a):
            va, vb = float(a[key]["mean"]), float(b[key]["mean"])
            if va != vb and not (va != va and vb != vb):
                rows.append(dict(table=name.replace("_summary.tsv", ""), h_name=a[key]["h_name"],
                                 mark_class=a[key]["mark_class"], metric=a[key]["metric"],
                                 mean_old=f"{va:.6g}", mean_new=f"{vb:.6g}",
                                 rel_change=f"{(vb - va) / va:+.4f}" if va else "inf"))
    write(out, rows)


def write(out, rows):
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows")


if __name__ == "__main__":
    {"synth": synth, "smoke": smoke}[sys.argv[1]](*sys.argv[2:])
