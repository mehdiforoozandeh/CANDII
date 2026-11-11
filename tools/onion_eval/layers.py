"""Onion layers along the replicate ladder (D12, primary decomposition; plan chunk E2).

For every replicate pair (a, b) of one rung, the pair variance is the variance of the depth-matched
difference d = log1p(a') - log1p(b') of the two count tracks, where the deeper track (or both, when a
common target depth is given) is binomially thinned to the shallower depth. The variance is split
exactly over Haar scale bands (orthonormal transform from `tools/eic_decomp/errdecomp.py`, J = 14) and
computed inside each region stratum.

A rung's variance is the mean pair variance over its pairs. Its layer increment is the rung variance
minus the variance of the nearest populated lower rung (the row names that rung in `increment_over`);
the innermost populated rung's increment is its own variance. Intervals are 95% intervals on the layer
increment from a cluster bootstrap over cells (both rungs resampled with the same cells): increment
+- t(0.975, n_cells - 1) x bootstrap SE x sqrt(n_cells / (n_cells - 1)), n_cells counted over both
rungs. The plain percentile interval under-covered with 8 cells (70-90% over 20 seeds).

Rung order: pseudoreplicate, sequencing, technical, isogenic, anisogenic, other_experiment,
other_cell_type. A rung with no pair gets a row with `n_pairs = 0` and flag `not populated by EIC`;
it is never dropped and never filled. A rung with fewer than 5 pairs or fewer than 3 cells is flagged
`few pairs`, and a pooled row by assay class is added for it.

CLI: python tools/onion_eval/layers.py run --tables <dir> --evalset <json> --out <layers.tsv>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from eic_decomp import errdecomp  # noqa: E402

J = errdecomp.J
RUNGS = ["pseudoreplicate", "sequencing", "technical", "isogenic", "anisogenic",
         "other_experiment", "other_cell_type"]
# Haar scale indices 0..J (level j is index j - 1; J is the coarse remainder), from errdecomp.BANDS:
# fine = levels 1-3, peak = 4-7, domain = 8-11, broad = 12-14 plus the coarse remainder.
BANDS = {name.split(" ")[0]: list(idx) for name, idx in errdecomp.BANDS.items()}
BAND_ORDER = ["fine", "peak", "domain", "broad", "all"]
MIN_PAIRS, MIN_CELLS = 5, 3
FLAG_OK, FLAG_FEW, FLAG_EMPTY = "ok", "few pairs", "not populated by EIC"
COLUMNS = ["assay_or_class", "pooled_by_class", "stratum", "band", "rung", "n_pairs", "n_cells",
           "variance", "layer_increment", "ci_low", "ci_high", "ci_width", "flag", "increment_over"]


# ---------------------------------------------------------------------------------------------------
# One pair

def thin(x, p, rng):
    """Binomial thinning of integer counts with keep probability p (p >= 1 returns x unchanged)."""
    x = np.asarray(x)
    if p >= 1.0:
        return x.astype(np.int64)
    return rng.binomial(x.astype(np.int64), p)


def pair_variance(a, b, depth_a, depth_b, strata=None, bands=None, *, depth=None, seed=0):
    """Variance of the depth-matched difference in log1p counts, per stratum and per Haar band.

    a, b: integer counts on the same bins, shape (L,) or (R, L), with L a multiple of 2^J (each row a
    region; the Haar transform runs within each row). depth_a, depth_b: the tracks' depths (reads).
    The deeper track is thinned to the shallower depth; with `depth` given, both are thinned to it
    (it must not exceed either depth). strata: dict name -> boolean array shaped like a (bins in the
    stratum); "all" (every bin) is always included. bands: dict name -> list of Haar scale indices
    (0..J); default fine/peak/domain/broad; "all" is always included and equals the sum of the bands
    when the bands partition the scales.

    Returns {"variance": {stratum: {band: v}}, "n_bins": {stratum: n}, "depth": matched depth}.
    The band variances are Haar energy of the stratum-centred difference (zero outside the stratum)
    divided by the number of stratum bins, so they add up exactly to the stratum variance. Centering
    removes one degree of freedom; on the "all" stratum it comes out of the coarse remainder (broad
    band), a relative bias of 1 / (8 R) there for white noise over R regions."""
    a = np.atleast_2d(np.asarray(a))
    b = np.atleast_2d(np.asarray(b))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.shape[-1] % (2 ** J):
        raise ValueError(f"region length {a.shape[-1]} is not a multiple of 2^{J}")
    depth_a, depth_b = float(depth_a), float(depth_b)
    target = min(depth_a, depth_b) if depth is None else float(depth)
    if target > min(depth_a, depth_b) * (1 + 1e-9):
        raise ValueError(f"target depth {target} exceeds a track depth ({depth_a}, {depth_b})")
    rng = np.random.default_rng(seed)
    a2 = thin(a, target / depth_a, rng)
    b2 = thin(b, target / depth_b, rng)
    d = np.log1p(a2) - np.log1p(b2)

    bands = dict(BANDS if bands is None else bands)
    strata = {} if strata is None else dict(strata)
    masks = {"all": np.ones(d.shape, bool)}
    for name, m in strata.items():
        m = np.atleast_2d(np.asarray(m, bool))
        if m.shape != d.shape:
            raise ValueError(f"stratum {name!r} shape {m.shape} != {d.shape}")
        masks[name] = m

    out, nb = {}, {}
    for name, m in masks.items():
        n = int(m.sum())
        nb[name] = n
        if n == 0:
            out[name] = {k: np.nan for k in list(bands) + ["all"]}
            continue
        v = np.where(m, d - d[m].mean(), 0.0)
        energy = np.array([float(np.sum(c.astype(np.float64) ** 2)) for c in errdecomp.haar(v, J)])
        row = {k: float(energy[idx].sum()) / n for k, idx in bands.items()}
        row["all"] = float(energy.sum()) / n
        out[name] = row
    return {"variance": out, "n_bins": nb, "depth": target}


# ---------------------------------------------------------------------------------------------------
# Rungs and layers

def _interval(est, boot, n_clusters):
    """95% interval from the cluster-bootstrap standard error, with the small-sample correction
    sqrt(n / (n - 1)) and a t quantile on n - 1 degrees of freedom (n = cells). With one cell there is
    no interval: (-inf, inf)."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if n_clusters < 2 or len(boot) < 2:
        return -np.inf, np.inf
    from scipy.stats import t as student_t
    se = float(boot.std()) * np.sqrt(n_clusters / (n_clusters - 1))
    h = float(student_t.ppf(0.975, n_clusters - 1)) * se
    return est - h, est + h


def layer_table(records, n_boot=1000, seed=0):
    """Rung variances, layer increments and intervals for one group (an assay, or a pooled class).

    records: list of dicts {"rung", "cell", "v": {(stratum, band): variance}} (one per pair).
    Returns one dict per (stratum, band, rung), every rung of RUNGS included, with n_pairs, n_cells,
    variance, layer_increment, ci_low, ci_high, ci_width, flag, increment_over."""
    keys = sorted({k for r in records for k in r["v"]})
    by_rung = {r: [x for x in records if x["rung"] == r] for r in RUNGS}
    populated = [r for r in RUNGS if by_rung[r]]
    lower = {}
    for r in RUNGS:
        prev = [q for q in RUNGS[:RUNGS.index(r)] if q in populated]
        lower[r] = prev[-1] if prev else ""
    cells = sorted({x["cell"] for x in records})
    cidx = {c: i for i, c in enumerate(cells)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cells), size=(n_boot, len(cells))) if cells else np.zeros((n_boot, 0), int)
    weights = np.stack([np.bincount(dr, minlength=len(cells)) for dr in draws]) if cells else draws

    def stats(rung, key):
        recs = [x for x in by_rung[rung] if np.isfinite(x["v"].get(key, np.nan))]
        if not recs:
            return np.nan, np.full(n_boot, np.nan)
        vals = np.array([x["v"][key] for x in recs])
        ci = np.array([cidx[x["cell"]] for x in recs])
        w = weights[:, ci].astype(float)                   # (n_boot, n_pairs): copies of each pair
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (w * vals).sum(1) / w.sum(1)
        return float(vals.mean()), boot

    rows = []
    for key in keys:
        cache = {r: stats(r, key) for r in populated}
        for r in RUNGS:
            recs = by_rung[r]
            n_pairs, n_cells = len(recs), len({x["cell"] for x in recs})
            row = dict(stratum=key[0], band=key[1], rung=r, n_pairs=n_pairs, n_cells=n_cells,
                       increment_over=lower[r] if recs else "")
            if not recs:
                row.update(variance=np.nan, layer_increment=np.nan, ci_low=np.nan, ci_high=np.nan,
                           ci_width=np.nan, flag=FLAG_EMPTY)
                rows.append(row)
                continue
            var, boot = cache[r]
            if lower[r]:
                lvar, lboot = cache[lower[r]]
                inc, iboot = var - lvar, boot - lboot
                ncl = len({x["cell"] for x in recs + by_rung[lower[r]]})
            else:
                inc, iboot, ncl = var, boot, n_cells
            lo, hi = _interval(inc, iboot, ncl)
            row.update(variance=var, layer_increment=inc, ci_low=lo, ci_high=hi, ci_width=hi - lo,
                       flag=FLAG_OK if (n_pairs >= MIN_PAIRS and n_cells >= MIN_CELLS) else FLAG_FEW)
            rows.append(row)
    return rows


def assemble(records, n_boot=1000, seed=0):
    """All layers.tsv rows. records: list of {"assay", "assay_class", "rung", "cell", "v"}.

    One row block per assay (every rung, every stratum and band). For each (assay class, rung) where
    some assay of the class is flagged few pairs or is empty while the class has pairs at that rung, a
    pooled block over all the class's assays is added (pooled_by_class = 1), keeping only those rungs."""
    rows = []
    assays = sorted({r["assay"] for r in records})
    cls_of = {r["assay"]: r["assay_class"] for r in records}
    weak = {}                                           # class -> set of rungs needing a pooled row
    for a in assays:
        ar = [r for r in records if r["assay"] == a]
        t = layer_table(ar, n_boot=n_boot, seed=seed)
        for row in t:
            rows.append(dict(assay_or_class=a, pooled_by_class=0, **row))
            if row["flag"] != FLAG_OK:
                weak.setdefault(cls_of[a], set()).add(row["rung"])
    for c in sorted(weak):
        cr = [r for r in records if r["assay_class"] == c]
        t = layer_table(cr, n_boot=n_boot, seed=seed)
        for row in t:
            if row["rung"] in weak[c] and row["n_pairs"] > 0:
                rows.append(dict(assay_or_class=c, pooled_by_class=1, **row))
    return rows


def write_tsv(rows, path):
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else repr(v)
        return str(v)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([fmt(r[c]) for c in COLUMNS])


# ---------------------------------------------------------------------------------------------------
# Reading the tables

def _read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


class CountsReader:
    """Reads one region of a measurement from its `counts_path` (h5:<path> or store:<root>:<bs>:<assay>)."""

    def __init__(self):
        self._h5, self._stores = {}, {}

    def read(self, counts_path, chrom, start, end):
        kind, _, rest = counts_path.partition(":")
        if kind == "h5":
            import h5py
            f = self._h5.get(rest) or self._h5.setdefault(rest, h5py.File(rest, "r"))
            return np.asarray(f[chrom][start:end])
        if kind == "store":
            root, bs, assay = rest.rsplit(":", 2)
            if root not in self._stores:
                from candi.store.reader import CorpusStore
                self._stores[root] = CorpusStore(root)
            return self._stores[root][bs].counts(chrom, start, end, assays=[assay])[:, 0]
        raise ValueError(f"unknown counts_path {counts_path!r}")


def load_strata_regions(masks_path, names, regions, L):
    """Stratum masks over the evaluation regions: {name: bool (R, L)}; names absent from the file are
    reported on stderr and skipped."""
    if not names or masks_path is None or not Path(masks_path).exists():
        if names:
            print(f"layers: no strata file {masks_path}; strata {names} not computed", file=sys.stderr)
        return {}
    import h5py
    out = {}
    with h5py.File(masks_path, "r") as f:
        for n in names:
            if n not in f:
                print(f"layers: stratum {n!r} not in {masks_path}; skipped", file=sys.stderr)
                continue
            out[n] = np.stack([np.asarray(f[n][c][s:s + L], bool) for c, s in regions])
    return out


def run(tables, evalset, out, strata_path=None, n_boot=1000, seed=0):
    tables = Path(tables)
    ev = json.loads(Path(evalset).read_text())
    meas = {r["measurement_id"]: r for r in _read_tsv(tables / "measurements.tsv")}
    pairs = {r["pair_id"]: r for r in _read_tsv(tables / "pairs.tsv")}
    want = ev.get("replicate_pairs")
    sel = [pairs[p] for p in want] if want is not None else list(pairs.values())
    sel = [p for p in sel if p["rung"] in RUNGS]
    regions = [(c, int(s)) for c, s in ev["regions"]]
    L = 2 ** J
    if strata_path is None:
        strata_path = tables.parent.parent / "eval" / "strata" / "masks.h5"
    strata = load_strata_regions(strata_path, ev.get("strata", []), regions, L)

    def depth(m):
        return float(m["depth_reads"] or m["depth_fragments"])

    # common depth per assay: the shallowest side of any of its pairs (every rung at one depth)
    common = {}
    for p in sel:
        d = min(depth(meas[p["a"]]), depth(meas[p["b"]]))
        common[p["assay"]] = min(common.get(p["assay"], np.inf), d)

    reader, records = CountsReader(), []
    for p in sel:
        ma, mb = meas[p["a"]], meas[p["b"]]
        a = np.stack([reader.read(ma["counts_path"], c, s, s + L) for c, s in regions])
        b = np.stack([reader.read(mb["counts_path"], c, s, s + L) for c, s in regions])
        pv = pair_variance(a, b, depth(ma), depth(mb), strata, depth=common[p["assay"]],
                           seed=zlib.crc32(p["pair_id"].encode()))
        v = {(s, bd): x for s, row in pv["variance"].items() for bd, x in row.items()}
        records.append(dict(assay=p["assay"], assay_class=ma["assay_class"], rung=p["rung"],
                            cell=p["cell"], v=v))
    # assays in the evalset's layers scope with no pair at all still get their empty rows
    scope = set()
    for assays in (ev.get("scorable", {}).get("layers", {}) or {}).values():
        scope.update(assays)
    rows = assemble(records, n_boot=n_boot, seed=seed)
    for a in sorted(scope - {r["assay"] for r in records}):
        for s in ["all"] + list(strata):
            for bd in BAND_ORDER:
                for r in RUNGS:
                    rows.append(dict(assay_or_class=a, pooled_by_class=0, stratum=s, band=bd, rung=r,
                                     n_pairs=0, n_cells=0, variance=np.nan, layer_increment=np.nan,
                                     ci_low=np.nan, ci_high=np.nan, ci_width=np.nan, flag=FLAG_EMPTY,
                                     increment_over=""))
    write_tsv(rows, out)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tables", required=True)
    r.add_argument("--evalset", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--strata", default=None, help="masks.h5 (default <tables>/../../eval/strata/masks.h5)")
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = run(args.tables, args.evalset, args.out, args.strata, args.n_boot, args.seed)
    empty = sorted({(x["assay_or_class"], x["rung"]) for x in rows if x["flag"] == FLAG_EMPTY})
    print(f"layers: {len(rows)} rows -> {args.out}; {len(empty)} (assay, rung) not populated by EIC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
