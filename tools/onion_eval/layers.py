"""Onion layers along the replicate ladder (D12, primary decomposition; plan chunk E2).

For every replicate pair (a, b) of one rung, `pair_variance` gives Var(d), the variance of the
depth-matched difference d = log1p(a') - log1p(b') of the two count tracks, where the deeper track (or
both, when a common target depth is given) is binomially thinned to the shallower depth. The variance is
split exactly over Haar scale bands (orthonormal transform from `tools/eic_decomp/errdecomp.py`, J = 14)
and computed inside each region stratum. A pseudoreplicate row of pairs.tsv (a = b, the base view) is
read as its two halves `pr1.h5` and `pr2.h5` (siblings of an `h5:` counts path), each at its own depth
(the file's `depth_reads` attribute, else half the base depth).

Column definitions of layers.tsv. `variance` is the pair value of the rung, 0.5 E[(Y - Y')^2] =
0.5 Var(d) (the mean term of d is 0 for exchangeable sides), i.e. E[Var(Y | Z(k-1))] of the identity in
`onion total variance.md`; it is the per-track value, half the variance of the pair difference. A
rung's variance is the mean over its cells of the cell's mean pair value. Its `layer_increment` is its
variance minus the pair value named in `increment_over`, taken per cell over the cells that populate
every rung involved and averaged over those cells; the innermost populated rung's increment is its own
variance. `layer` names the layer(s) of the identity that the increment holds.

Which pair value an increment subtracts (PI ruling 30, `lower_terms`). Donor and experiment are crossed
in ENCODE, so they are sibling layers, not nested ones:
  pseudoreplicate -> sequencing -> technical -> isogenic   nested: each over the nearest populated rung
                                                           before it (layers sampling, sequencing,
                                                           library, biological_replicate)
  anisogenic        over isogenic: the donor layer (the pair shares cell type and experiment)
  other_experiment  over isogenic: the experiment (lab) layer (the pair shares cell type and donor)
  other_cell_type   over the pair value of a pair that shares only the cell type, built as
                    anisogenic + other_experiment - isogenic (`increment_over` =
                    "anisogenic+other_experiment-isogenic"; exact when donor and experiment add):
                    the cell-type layer
With isogenic missing a sibling is taken over the nearest populated nested rung, and its layer name
lists the nested layers it also holds (e.g. `biological_replicate+donor`). With a sibling or isogenic
missing, other_cell_type is taken over other_experiment, else anisogenic, else the nearest nested rung,
and its layer name lists what it also holds (e.g. `cell_type+experiment`). When every rung has the same
cells, the increments of the chain innermost -> isogenic, donor, experiment and cell type add up exactly
to the other_cell_type variance. A rung whose cells differ from those of a rung it subtracts is flagged
`cells differ from lower rung` (its increment uses the common cells only, and is empty when there is
none). An other_cell_type pair (two cells) is clustered on its `a` side's cell. Intervals are 95%
intervals on the layer increment from a cluster bootstrap over the cells it uses: increment +-
t(0.975, n - 1) x bootstrap SE x sqrt(n / (n - 1)), n = those cells. The plain percentile interval
under-covered with 8 cells (70-90% over 20 seeds). `depth` is the common depth (reads) at which every
pair of the assay was taken. On log1p counts a layer holds at that depth only (assumption 4 of the
note): never set it against headline 1 of a target at another depth (headline 1 writes each target's
`depth_reads` beside it); a pooled row whose assays have different depths has depth `mixed` and the flag
`assays at different depths`.

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
FLAG_CELLS, FLAG_DEPTHS = "cells differ from lower rung", "assays at different depths"
NESTED = RUNGS[:4]                                   # pseudoreplicate .. isogenic: nested rungs
SIBLINGS = ("anisogenic", "other_experiment")        # crossed: each over isogenic (PI ruling 30)
LAYER = {"pseudoreplicate": "sampling", "sequencing": "sequencing", "technical": "library",
         "isogenic": "biological_replicate", "anisogenic": "donor", "other_experiment": "experiment",
         "other_cell_type": "cell_type"}
COLUMNS = ["assay_or_class", "pooled_by_class", "stratum", "band", "rung", "n_pairs", "n_cells",
           "variance", "layer_increment", "ci_low", "ci_high", "ci_width", "flag", "increment_over",
           "depth", "layer"]


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

def lower_terms(populated):
    """Which pair value each populated rung's increment subtracts (module doc, PI ruling 30).

    populated: rung names with pairs. Returns {rung: (terms, layer)}: terms a list of (rung, coefficient)
    whose weighted sum of pair values is subtracted ([] for the innermost rung), layer the layer name(s)
    the increment holds. `increment_over` is `format_terms(terms)`."""
    pop = [r for r in RUNGS if r in set(populated)]

    def nested_before(r):
        """The nearest populated nested rung before r ("" if none) and the nested layers in between
        (from just after it up to r's own nested position, or up to isogenic for a later rung)."""
        stop = NESTED.index(r) if r in NESTED else len(NESTED)
        prev = [q for q in NESTED[:stop] if q in pop]
        q = prev[-1] if prev else ""
        start = NESTED.index(q) + 1 if q else 0
        return q, [LAYER[x] for x in NESTED[start:stop]]

    out = {}
    for r in pop:
        q, held = nested_before(r)
        if r in NESTED or r in SIBLINGS:
            out[r] = ([(q, 1)] if q else [], "+".join(held + [LAYER[r]]))
            continue
        sib = [x for x in SIBLINGS if x in pop]                  # other_cell_type
        if "isogenic" in pop and len(sib) == 2:
            out[r] = ([("anisogenic", 1), ("other_experiment", 1), ("isogenic", -1)], LAYER[r])
        elif sib:
            s_ = sib[-1]                                         # other_experiment, else anisogenic
            out[r] = ([(s_, 1)], "+".join([LAYER[r]] + [LAYER[x] for x in SIBLINGS if x != s_]))
        else:
            out[r] = ([(q, 1)] if q else [], "+".join(held + [LAYER[x] for x in SIBLINGS] + [LAYER[r]]))
    return out


def format_terms(terms):
    """`increment_over` text of a term list: "isogenic", or "anisogenic+other_experiment-isogenic"."""
    return "".join(("" if i == 0 and c > 0 else ("+" if c > 0 else "-")) + q for i, (q, c) in enumerate(terms))


def parse_terms(text):
    """Inverse of `format_terms`: [(rung, coefficient)]."""
    import re
    return [(q, -1 if sign == "-" else 1) for sign, q in re.findall(r"([+-]?)([a-z_]+)", text or "")]


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
    """Rung pair values, layer increments and intervals for one group (an assay, or a pooled class).

    records: list of dicts {"rung", "cell", "v": {(stratum, band): Var(d)}} (one per pair, the
    `pair_variance` value; halved here to the pair value). Returns one dict per (stratum, band, rung),
    every rung of RUNGS included, with n_pairs, n_cells, variance, layer_increment, ci_low, ci_high,
    ci_width, flag, increment_over, layer (see the module doc for the definitions)."""
    keys = sorted({k for r in records for k in r["v"]})
    by_rung = {r: [x for x in records if x["rung"] == r] for r in RUNGS}
    populated = [r for r in RUNGS if by_rung[r]]
    lower = lower_terms(populated)
    rung_cells = {r: {x["cell"] for x in by_rung[r]} for r in RUNGS}
    cells = sorted({x["cell"] for x in records})
    cidx = {c: i for i, c in enumerate(cells)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cells), size=(n_boot, len(cells))) if cells else np.zeros((n_boot, 0), int)
    weights = (np.stack([np.bincount(dr, minlength=len(cells)) for dr in draws]) if cells else draws).astype(float)

    def cell_means(rung, key):
        """{cell: mean pair value of the cell} over the rung's finite records."""
        acc = {}
        for x in by_rung[rung]:
            v = x["v"].get(key, np.nan)
            if np.isfinite(v):
                acc.setdefault(x["cell"], []).append(0.5 * v)
        return {c: float(np.mean(v)) for c, v in acc.items()}

    def mean_boot(per_cell):
        """Mean over cells and its bootstrap draws (cells resampled with the shared weights)."""
        if not per_cell:
            return np.nan, np.full(n_boot, np.nan), 0
        cs = sorted(per_cell)
        vals = np.array([per_cell[c] for c in cs])
        w = weights[:, [cidx[c] for c in cs]]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (w * vals).sum(1) / w.sum(1)
        return float(vals.mean()), boot, len(cs)

    rows = []
    for key in keys:
        cm = {r: cell_means(r, key) for r in populated}
        for r in RUNGS:
            recs = by_rung[r]
            n_pairs, n_cells = len(recs), len(rung_cells[r])
            terms, layer = lower.get(r, ([], LAYER[r]))
            row = dict(stratum=key[0], band=key[1], rung=r, n_pairs=n_pairs, n_cells=n_cells,
                       increment_over=format_terms(terms) if recs else "", layer=layer)
            if not recs:
                row.update(variance=np.nan, layer_increment=np.nan, ci_low=np.nan, ci_high=np.nan,
                           ci_width=np.nan, flag=FLAG_EMPTY)
                rows.append(row)
                continue
            var, _, _ = mean_boot(cm[r])
            common = set(cm[r]).intersection(*(set(cm[q]) for q, _ in terms))
            inc, iboot, ncl = mean_boot({c: cm[r][c] - sum(k * cm[q][c] for q, k in terms) for c in common})
            lo, hi = _interval(inc, iboot, ncl) if np.isfinite(inc) else (np.nan, np.nan)
            flag = FLAG_OK if (n_pairs >= MIN_PAIRS and n_cells >= MIN_CELLS) else FLAG_FEW
            if any(rung_cells[r] != rung_cells[q] for q, _ in terms):
                flag = FLAG_CELLS if flag == FLAG_OK else f"{flag}; {FLAG_CELLS}"
            row.update(variance=var, layer_increment=inc, ci_low=lo, ci_high=hi, ci_width=hi - lo, flag=flag)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------
# Explained fraction per layer (PI ruling 32; `onion total variance.md`, explained-per-layer)

EXPLAINED_COLUMNS = ["rung", "layer", "increment_over", "explained_fraction", "ci_low", "ci_high",
                     "captured", "layer_term", "n_pairs", "n_cells", "flag"]


def predicted_layer_mean(n, mu, keep=1.0, *, exact_below=30.0):
    """The model's predicted mean on the scale of the layer table: E[log1p Y'] under its predictive law
    NB(n, mu) thinned to the common depth (keep = depth / the prediction's depth; a thinned NB(n, mu) is
    NB(n, keep mu)). Exact (pmf sum) where keep mu <= exact_below, else the second-order expansion
    log1p(m) - (m + m^2 / n) / (2 (1 + m)^2). log1p(mu) itself is not used: at a few reads per bin it
    is not E[log1p Y] (the audit's cell-type model scored 1.07 with it, 0.99 with this)."""
    from scipy.stats import nbinom
    n = np.broadcast_to(np.asarray(n, np.float64), np.shape(mu)).astype(np.float64)
    m = np.asarray(mu, np.float64) * float(keep)
    out = np.log1p(m) - (m + m * m / n) / (2.0 * (1.0 + m) ** 2)
    sm = m <= exact_below
    if sm.any():
        k = np.arange(0, int(exact_below * 6 + 80), dtype=np.float64)
        lk = np.log1p(k)
        ms, ns = m[sm], n[sm]
        res = np.empty(ms.size)
        for i in range(0, ms.size, 8192):
            pm = nbinom.pmf(k[None, :], ns[i:i + 8192, None], (ns / (ns + ms))[i:i + 8192, None])
            res[i:i + 8192] = pm @ lk
        out[sm] = res
    return out


def pair_explained(a, b, ea, eb, depth_a, depth_b, *, depth=None, seed=0):
    """Captured part and pair value of one pair on the layer-table scale.

    a, b: integer counts (bins of the pair); ea, eb: the model's predicted layer-scale means of the two
    members (`predicted_layer_mean` at the common depth), each from its own inputs and covariates. The
    counts are thinned to the common depth exactly as `pair_variance` does (same seed, same draws).
    With d = log1p(a') - log1p(b') and d_hat = ea - eb, both centred: returns {"captured":
    0.5 mean(d_hat d), "pair_value": 0.5 mean(d^2) (= half of pair_variance's Var(d)), "depth"}."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    depth_a, depth_b = float(depth_a), float(depth_b)
    target = min(depth_a, depth_b) if depth is None else float(depth)
    if target > min(depth_a, depth_b) * (1 + 1e-9):
        raise ValueError(f"target depth {target} exceeds a track depth ({depth_a}, {depth_b})")
    rng = np.random.default_rng(seed)
    a2 = thin(a, target / depth_a, rng)
    b2 = thin(b, target / depth_b, rng)
    d = np.log1p(a2) - np.log1p(b2)
    dh = np.asarray(ea, np.float64).reshape(-1) - np.asarray(eb, np.float64).reshape(-1)
    d, dh = d - d.mean(), dh - dh.mean()
    return {"captured": 0.5 * float(np.mean(d * dh)), "pair_value": 0.5 * float(np.mean(d * d)),
            "depth": target}


def explained_table(records, n_boot=1000, seed=0):
    """Explained fraction per layer from per-pair records {"rung", "cell", "captured", "pair_value"}
    (`pair_explained`; "captured" may be None for a rung whose members got no prediction, which leaves
    every layer that subtracts it undefined).

    For the layer of rung r, with the terms of `lower_terms` (the same pair value the layer subtracts):
      explained_fraction = mean_c [C_r(c) - sum_q k_q C_q(c)] / mean_c [PV_r(c) - sum_q k_q PV_q(c)]
    over the cells c that populate r and every q (C, PV: the cell's mean over its pairs). The numerator
    is the captured part of the layer term, the denominator the layer term itself. No clipping: with few
    pairs it can be negative or above 1. Interval: 95% cluster bootstrap over those cells (`_interval`)."""
    by_rung = {r: [x for x in records if x["rung"] == r] for r in RUNGS}
    populated = [r for r in RUNGS if by_rung[r]]
    terms_of = lower_terms(populated)
    cells = sorted({x["cell"] for x in records})
    cidx = {c: i for i, c in enumerate(cells)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cells), size=(n_boot, len(cells))) if cells else np.zeros((n_boot, 0), int)
    weights = (np.stack([np.bincount(dr, minlength=len(cells)) for dr in draws]) if cells else draws).astype(float)

    def cell_means(rung, key):
        acc = {}
        for x in by_rung[rung]:
            v = x.get(key)
            acc.setdefault(x["cell"], []).append(np.nan if v is None else float(v))
        return {c: float(np.mean(v)) for c, v in acc.items()}

    cm = {r: (cell_means(r, "captured"), cell_means(r, "pair_value")) for r in populated}
    rows = []
    for r in populated:
        terms, layer = terms_of[r]
        if not terms:
            continue
        rungs = [r] + [q for q, _ in terms]
        common = sorted(set(cm[r][1]).intersection(*(set(cm[q][1]) for q in rungs[1:])))
        num = np.array([cm[r][0][c] - sum(k * cm[q][0][c] for q, k in terms) for c in common])
        den = np.array([cm[r][1][c] - sum(k * cm[q][1][c] for q, k in terms) for c in common])
        flag = "ok"
        if not common:
            ef, lo, hi, flag = np.nan, np.nan, np.nan, "no cell populates every rung of the layer"
        elif not np.isfinite(num).all():
            ef, lo, hi, flag = np.nan, np.nan, np.nan, "a rung of the layer has no prediction for its pairs"
        else:
            ef = float(num.mean() / den.mean()) if den.mean() != 0 else np.nan
            w = weights[:, [cidx[c] for c in common]]
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = (w @ num) / (w @ den)
            lo, hi = _interval(ef, boot, len(common)) if np.isfinite(ef) else (np.nan, np.nan)
        rows.append(dict(rung=r, layer=layer, increment_over=format_terms(terms), explained_fraction=ef,
                         ci_low=lo, ci_high=hi, captured=float(num.mean()) if common else np.nan,
                         layer_term=float(den.mean()) if common else np.nan,
                         n_pairs=len(by_rung[r]), n_cells=len(common), flag=flag))
    return rows


def assemble(records, n_boot=1000, seed=0):
    """All layers.tsv rows. records: list of {"assay", "assay_class", "rung", "cell", "v"}.

    One row block per assay (every rung, every stratum and band). For each (assay class, rung) where
    some assay of the class is flagged few pairs or is empty while the class has pairs at that rung, a
    pooled block over all the class's assays is added (pooled_by_class = 1), keeping only those rungs.
    A record may carry "depth" (the common depth of its assay); a row's `depth` is the one depth of its
    records, `mixed` (flagged) when they differ, NaN when none is given."""
    rows = []
    assays = sorted({r["assay"] for r in records})
    cls_of = {r["assay"]: r["assay_class"] for r in records}

    def depth_of(recs):
        ds = sorted({float(r["depth"]) for r in recs if r.get("depth") is not None})
        return ds[0] if len(ds) == 1 else ("mixed" if ds else np.nan)

    weak = {}                                           # class -> set of rungs needing a pooled row
    for a in assays:
        ar = [r for r in records if r["assay"] == a]
        t = layer_table(ar, n_boot=n_boot, seed=seed)
        dep = depth_of(ar)
        for row in t:
            rows.append(dict(assay_or_class=a, pooled_by_class=0, depth=dep, **row))
            if row["flag"] == FLAG_EMPTY or FLAG_FEW in row["flag"]:
                weak.setdefault(cls_of[a], set()).add(row["rung"])
    for c in sorted(weak):
        cr = [r for r in records if r["assay_class"] == c]
        t = layer_table(cr, n_boot=n_boot, seed=seed)
        dep = depth_of(cr)
        for row in t:
            if row["rung"] in weak[c] and row["n_pairs"] > 0:
                if dep == "mixed":
                    row["flag"] = FLAG_DEPTHS if row["flag"] == FLAG_OK else f"{row['flag']}; {FLAG_DEPTHS}"
                rows.append(dict(assay_or_class=c, pooled_by_class=1, depth=dep, **row))
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


def half_depths(m):
    """The pseudoreplicate halves of a measurement row with an `h5:` counts path:
    ((pr1 path, depth), (pr2 path, depth)), or None when either file is missing. A half's depth is its
    file's `depth_reads` attribute; without both attributes each half gets half the row's depth."""
    kind, _, rest = m["counts_path"].partition(":")
    if kind != "h5":
        return None
    paths = [Path(rest).with_name(f"{h}.h5") for h in ("pr1", "pr2")]
    if not all(q.is_file() for q in paths):
        return None
    import h5py
    deps = []
    for q in paths:
        with h5py.File(q, "r") as f:
            deps.append(float(f.attrs.get("depth_reads", 0) or 0))
    if not all(d > 0 for d in deps):
        d = float(m.get("depth_reads") or m.get("depth_fragments")) / 2.0
        deps = [d, d]
    return tuple((f"h5:{q}", d) for q, d in zip(paths, deps))


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

    # the two sides (counts path, depth) of each pair; a == b is a pseudoreplicate row: its halves
    sides = {}
    for p in sel:
        ma, mb = meas[p["a"]], meas[p["b"]]
        if p["a"] == p["b"]:
            h = half_depths(ma)
            if h is None:
                print(f"layers: {p['pair_id']}: a == b and no pr1.h5 / pr2.h5 halves; skipped", file=sys.stderr)
                continue
            sides[p["pair_id"]] = h
        else:
            sides[p["pair_id"]] = ((ma["counts_path"], depth(ma)), (mb["counts_path"], depth(mb)))
    sel = [p for p in sel if p["pair_id"] in sides]

    # common depth per assay: the shallowest side of any of its pairs (every rung at one depth)
    common = {}
    for p in sel:
        (_, da), (_, db) = sides[p["pair_id"]]
        common[p["assay"]] = min(common.get(p["assay"], np.inf), da, db)

    reader, records = CountsReader(), []
    for p in sel:
        (pa, da), (pb, db) = sides[p["pair_id"]]
        a = np.stack([reader.read(pa, c, s, s + L) for c, s in regions])
        b = np.stack([reader.read(pb, c, s, s + L) for c, s in regions])
        pv = pair_variance(a, b, da, db, strata, depth=common[p["assay"]],
                           seed=zlib.crc32(p["pair_id"].encode()))
        v = {(s, bd): x for s, row in pv["variance"].items() for bd, x in row.items()}
        records.append(dict(assay=p["assay"], assay_class=meas[p["a"]]["assay_class"], rung=p["rung"],
                            cell=p["cell"] or meas[p["a"]]["cell"], v=v, depth=common[p["assay"]]))
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
                                     increment_over="", depth=np.nan, layer=LAYER[r]))
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
