"""Deduplication at 1 bp, read for read (t396 C12a; serves R43, R24, R28).

Question: is the dedup-on read set of one dedup unit equal to the distinct (chromosome, 5' position,
strand) set of the dedup-off reads of the same unit, and which read set is the dedup unit (one
replicate, or the pooled union of the replicates)?

A read's position is its 5' end in tagAlign coordinates: `start` on the + strand, `end - 1` on the
- strand (SE MarkDuplicates groups reads by 5' end and strand). A strand-bin is one strand of one
25 bp bin, the bin of the 5' end (`pos // 25`).

Subcommands
  unit   --off <tagAligns> --on <tagAligns> --experiment E --label L --n-units U --out <json>
         (i) share of distinct dedup-on positions absent from the distinct dedup-off positions, the
         reverse share, and the share of dedup-on lines that repeat a position. Run it once per
         replicate (U = 1) and once on the pooled files of all replicates (U = number of replicates).
  shape  --off <tagAligns> --on <tagAligns> --experiment E --label L --n-units U --out <json>
         [--chrom-sizes <json|tsv>]
         (ii) histogram of dedup-off reads per position; (iii) the symmetric Dirichlet shape `a`
         over the 25 positions of a strand-bin fitted at 1 bp (Dirichlet-multinomial likelihood),
         next to `caps.fit_shape` at 25 bp (dedup-off count -> dedup-on count, K = 25 U);
         (iv) `caps.dm_distinct` at 25 bp against the exact 1 bp number of distinct dedup-off
         positions, in the top 1% of strand-bins by dedup-off count.
  rule   --units <unit jsons> --out UNIT_RULE.json [--tol 1e-3]
         {"dedup_unit": "replicate"|"pooled"|"other", "on_equals_distinct_off": bool,
          "max_share_absent": float, "n_experiments": int, ...}

numpy, pandas, scipy only. `caps` (C4) is needed by `shape` (iii, iv) only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = 25                  # bp per bin
POS_BITS = 32             # 5' position field of a key
CHUNK = 5_000_000         # tagAlign lines per pandas chunk
TOL = 1e-3                # largest share that still counts as "equal"
TOP_FRACTION = 0.01       # top 1% of strand-bins by dedup-off count


# ------------------------------------------------------------------ reading
class ChromIndex(dict):
    """chromosome name -> small int, shared by every file read in one process."""

    def code(self, names: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(names, return_inverse=True)
        codes = np.array([self.setdefault(str(u), len(self)) for u in uniq], np.int64)
        return codes[inv]


def keys_from_columns(chrom_codes, start, end, strand_minus) -> np.ndarray:
    """int64 key = chrom << 33 | pos5 << 1 | minus."""
    start = np.asarray(start, np.int64)
    end = np.asarray(end, np.int64)
    minus = np.asarray(strand_minus, bool)
    pos5 = np.where(minus, end - 1, start)
    if pos5.size and (pos5.min() < 0 or pos5.max() >= (1 << POS_BITS)):
        raise ValueError("5' position out of range")
    return (np.asarray(chrom_codes, np.int64) << (POS_BITS + 1)) | (pos5 << 1) | minus.astype(np.int64)


def read_keys(paths, index: ChromIndex) -> np.ndarray:
    """One key per tagAlign line over all `paths` (columns chrom, start, end, name, score, strand)."""
    import pandas as pd
    out = []
    for p in paths:
        for df in pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2, 5],
                              names=["c", "s", "e", "t"], dtype={"c": str, "s": np.int64,
                                                                 "e": np.int64, "t": str},
                              chunksize=CHUNK, compression="infer"):
            out.append(keys_from_columns(index.code(df["c"].to_numpy()), df["s"].to_numpy(),
                                         df["e"].to_numpy(), df["t"].to_numpy() == "-"))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def strand_bin(keys: np.ndarray) -> np.ndarray:
    """Key of the strand-bin of each position key: chrom, pos5 // 25, strand."""
    chrom = keys >> (POS_BITS + 1)
    pos = (keys >> 1) & ((1 << POS_BITS) - 1)
    return (chrom << (POS_BITS + 1)) | ((pos // RES) << 1) | (keys & 1)


# ------------------------------------------------------------------ (i) unit comparison
def compare_unit(off: np.ndarray, on: np.ndarray) -> dict:
    d_off = np.unique(off)
    d_on = np.unique(on)
    on_absent = int(np.setdiff1d(d_on, d_off, assume_unique=True).size)
    off_absent = int(np.setdiff1d(d_off, d_on, assume_unique=True).size)
    return {
        "n_off_lines": int(off.size), "n_off_distinct": int(d_off.size),
        "n_on_lines": int(on.size), "n_on_distinct": int(d_on.size),
        "n_on_absent_from_off": on_absent, "n_off_absent_from_on": off_absent,
        "share_on_absent_from_off": on_absent / max(d_on.size, 1),
        "share_off_absent_from_on": off_absent / max(d_off.size, 1),
        "share_on_repeated_lines": (on.size - d_on.size) / max(on.size, 1),
        "share_off_duplicate_lines": (off.size - d_off.size) / max(off.size, 1),
    }


def max_share(u: dict) -> float:
    return max(u["share_on_absent_from_off"], u["share_off_absent_from_on"],
               u["share_on_repeated_lines"])


# ------------------------------------------------------------------ (ii)-(iv) shape
def hist_dict(values: np.ndarray) -> dict:
    v, n = np.unique(values, return_counts=True)
    return {int(a): int(b) for a, b in zip(v, n)}


def dm_loglik(a: float, hist_c: dict, hist_x: dict, K: int = RES) -> float:
    """Dirichlet-multinomial log likelihood (up to a constant in a) of the per-position counts of
    every strand-bin, from the histogram of bin totals `hist_c` and of nonzero position counts
    `hist_x`."""
    from scipy.special import gammaln
    c = np.array(list(hist_c.keys()), np.float64)
    nc = np.array(list(hist_c.values()), np.float64)
    x = np.array(list(hist_x.keys()), np.float64)
    nx = np.array(list(hist_x.values()), np.float64)
    return float((nc * (gammaln(K * a) - gammaln(K * a + c))).sum()
                 + (nx * (gammaln(a + x) - gammaln(a))).sum())


def fit_dm_shape(hist_c: dict, hist_x: dict, K: int = RES, bounds=(1e-3, 1e3)) -> float:
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda la: -dm_loglik(math.exp(la), hist_c, hist_x, K),
                        bounds=(math.log(bounds[0]), math.log(bounds[1])), method="bounded",
                        options={"xatol": 1e-5})
    return float(math.exp(r.x))


def load_sizes(path) -> dict:
    if path.endswith(".json"):
        d = json.load(open(path))
        return {c: int(v) for c, v in d.get("chrom_sizes", d).items()}
    out = {}
    for line in open(path):
        f = line.split()
        if len(f) >= 2:
            out[f[0]] = int(f[1])
    return out


def shape_stats(off: np.ndarray, on: np.ndarray, n_units: int, n_strand_bins=None) -> dict:
    """(ii)-(iv) for one read set. `n_strand_bins` = 2 x genome bins (sets the top-1% size);
    None uses the number of strand-bins with dedup-off reads."""
    pos, x = np.unique(off, return_counts=True)            # per position dedup-off reads
    sb_pos = strand_bin(pos)
    sb, inv = np.unique(sb_pos, return_inverse=True)
    c = np.bincount(inv, weights=x).astype(np.int64)        # dedup-off reads per strand-bin
    d = np.bincount(inv).astype(np.int64)                   # distinct dedup-off positions per strand-bin
    on_sb = strand_bin(on)
    j = np.searchsorted(sb, on_sb)
    inside = (j < sb.size) & (sb[np.minimum(j, sb.size - 1)] == on_sb)
    t = np.bincount(j[inside], minlength=sb.size).astype(np.int64)  # dedup-on reads per strand-bin
    hist_x, hist_c = hist_dict(x), hist_dict(c)
    a_1bp = fit_dm_shape(hist_c, hist_x, RES)

    denom = n_strand_bins if n_strand_bins else sb.size
    n_top = min(int(math.ceil(TOP_FRACTION * denom)), sb.size)
    top = np.zeros(sb.size, bool)
    top[np.argsort(-c, kind="stable")[:n_top]] = True

    out = {
        "n_units": n_units,
        "n_strand_bins_with_off_reads": int(sb.size),
        "top_denominator": "genome_strand_bins" if n_strand_bins else "nonzero_strand_bins",
        "n_top_strand_bins": int(n_top),
        "n_on_reads_outside_off_bins": int((~inside).sum()),
        "hist_off_reads_per_position": hist_x,
        "dirichlet_a_1bp": a_1bp,
        "top1pct": {"sum_off": int(c[top].sum()), "sum_distinct_off": int(d[top].sum()),
                    "sum_on": int(t[top].sum())},
    }
    try:
        from t396 import caps
    except ImportError as e:                                 # C4 not merged yet
        out["caps"] = f"unavailable: {e}"
        return out
    K = caps.K_of(n_units)
    jh_on = caps.joint_hist(c, t, top_mask=top)
    jh_d = caps.joint_hist(c, d, top_mask=top)
    a_25 = caps.fit_shape(jh_on["genome"], K)
    out["dirichlet_a_25bp_caps_fit_shape"] = a_25
    out["K_25bp"] = float(K)
    laws = {"dm_distinct_a_25bp": lambda s: caps.dm_distinct(s, a_25, K),
            "dm_distinct_a_1bp": lambda s: caps.dm_distinct(s, a_1bp, K),
            "occupancy": lambda s: caps.occupancy(s, K)}
    for name, law in laws.items():
        pred = float(np.asarray(law(c[top].astype(np.float64))).sum())
        out["top1pct"][name] = {
            "sum_pred": pred,
            "rel_err_vs_distinct_off": pred / max(out["top1pct"]["sum_distinct_off"], 1) - 1.0,
            "rel_err_vs_on": pred / max(out["top1pct"]["sum_on"], 1) - 1.0,
            "deviance_distinct_off": caps.deviance_of(jh_d["top1pct"], law),
            "deviance_on": caps.deviance_of(jh_on["top1pct"], law)}
    return out


# ------------------------------------------------------------------ rule
def decide(units: list, tol: float = TOL) -> dict:
    """Unit rule from `unit` jsons. Replicate units are the jsons with n_units = 1; a pooled set is
    a json with n_units >= 2. `replicate`: every replicate unit has on = distinct(off) and some
    pooled set does not (positions repeat across replicates). `pooled`: every pooled set has
    on = distinct(off) and some replicate unit does not. Single-replicate experiments only, all
    equal: `replicate` (each replicate is its own unit); `n_multi_unit_experiments` = 0 says the
    two rules were not told apart."""
    reps = [u for u in units if u["n_units"] == 1]
    pools = [u for u in units if u["n_units"] >= 2]
    rep_ok = bool(reps) and all(max_share(u) <= tol for u in reps)
    pool_ok = bool(pools) and all(max_share(u) <= tol for u in pools)
    if rep_ok:          # with pool_ok too, no replicate pair shares a position: both fit
        unit, chosen = "replicate", reps
    elif pool_ok and not rep_ok:
        unit, chosen = "pooled", pools
    else:
        unit, chosen = "other", units
    m = max((max_share(u) for u in chosen), default=float("nan"))
    return {"dedup_unit": unit, "on_equals_distinct_off": bool(unit != "other" and m <= tol),
            "max_share_absent": float(m),
            "n_experiments": len({u["experiment"] for u in units}),
            "n_multi_unit_experiments": len({u["experiment"] for u in pools}),
            "tol": tol}


# ------------------------------------------------------------------ CLI
def _write(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("unit", "shape"):
        p = sub.add_parser(name)
        p.add_argument("--off", nargs="+", required=True)
        p.add_argument("--on", nargs="+", required=True)
        p.add_argument("--experiment", required=True)
        p.add_argument("--label", required=True)
        p.add_argument("--n-units", type=int, default=1)
        p.add_argument("--out", required=True)
        if name == "shape":
            p.add_argument("--chrom-sizes")
    p = sub.add_parser("rule")
    p.add_argument("--units", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args(argv)

    if a.cmd == "rule":
        res = decide([json.load(open(f)) for f in a.units], a.tol)
        _write(res, a.out)
        print(json.dumps(res))
        return res
    index = ChromIndex()
    off, on = read_keys(a.off, index), read_keys(a.on, index)
    head = {"experiment": a.experiment, "label": a.label, "n_units": a.n_units,
            "off": list(a.off), "on": list(a.on)}
    if a.cmd == "unit":
        res = {**head, **compare_unit(off, on)}
    else:
        nsb = None
        if a.chrom_sizes:
            nsb = 2 * sum(math.ceil(v / RES) for v in load_sizes(a.chrom_sizes).values())
        res = {**head, **shape_stats(off, on, a.n_units, nsb)}
    _write(res, a.out)
    return res


if __name__ == "__main__":
    main()
