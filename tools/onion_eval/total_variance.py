"""Audit of the onion estimators against the total-variance identity (plan Amendment 3, section A; chunk TV2).

The identity. Y is the count of one assay in one bin; Z0 = (assay, position) and Z1..Z6 add cell type, donor,
experiment, biological replicate, library and sequencing run. By the law of total variance,

    Var(Y) = Var(E[Y|Z0]) + sum_{k=1..6} E[Var(E[Y|Zk] | Z(k-1))] + E[Var(Y|Z6)],

and the last term is the Poisson sampling term E[Y] on counts. Two exchangeable measurements that share Z(k-1)
have the pair value 0.5 E[(Y - Y')^2] = E[Var(Y | Z(k-1))].

The simulator draws that chain. Every level multiplies the rate by an independent gamma factor with mean 1 and
variance cv2 (its squared coefficient of variation), drawn per position, so every term has a closed form:

    pair value of a pair that shares the levels S, at depth D:
        D E[m] + D^2 E[m^2] prod_{j in S} (1 + c_j) (prod_{j not in S} (1 + c_j) - 1)
    chain term of level k:  E[m^2] prod_{j < k} (1 + c_j) c_k;   Var(E[Y|Z0]) = Var(m);   sampling = E[m]

with m the position's base rate. The design per cell is 2 donors x 2 experiments x 2 biological replicates x 2
libraries x 2 sequencing runs = 32 leaf tracks, each with two pseudoreplicate halves (a binomial split of its
reads). With `crossed=True` (ENCODE) the experiment effect is shared by the two donors of a cell; with
`crossed=False` it is nested in the donor.

The check functions call the existing tools (`layers.py`, `resolution.py`, `mixed_model.py`, `error_budget.py`,
`metrics.py`) and never edit them. `audit()` runs every case of the chunk and returns the rows of `AUDIT.md`.

CLI: python tools/onion_eval/total_variance.py audit --out <AUDIT.md> [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zlib
from itertools import product
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import error_budget as EB  # noqa: E402
from onion_eval import layers as LY  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import mixed_model as MM  # noqa: E402
from onion_eval import resolution as RS  # noqa: E402

LEVELS = ("cell_type", "donor", "experiment", "bio_rep", "library", "sequencing")
RUNGS = LY.RUNGS
L = 2 ** LY.J
ALL = ("all", "all")                       # (stratum, band) key of layers.pair_variance


def shared_levels(rung, crossed=True):
    """Levels that the two sides of a pair of this rung share, in the simulator's design."""
    s = {"pseudoreplicate": set(LEVELS), "sequencing": set(LEVELS) - {"sequencing"},
         "technical": {"cell_type", "donor", "experiment", "bio_rep"},
         "isogenic": {"cell_type", "donor", "experiment"},
         "anisogenic": {"cell_type", "experiment"} if crossed else {"cell_type"},
         "other_experiment": {"cell_type", "donor"},
         "other_cell_type": set()}
    return s[rung]


# ---------------------------------------------------------------------------------------------------
# Simulator

def _cv(layers, level, c):
    v = layers.get(level, 0.0)
    return float(v[c]) if np.ndim(v) else float(v)


def _gamma(rng, cv2, n):
    if cv2 <= 0:
        return np.ones(n)
    return rng.gamma(1.0 / cv2, cv2, n)


def _leaf(c, d, x, r, l, s):
    return f"c{c}.d{d}.x{x}.r{r}.l{l}.s{s}"


def simulate(layers: dict, depth: dict | None = None, n_cells: int = 6, n_pos: int = L, seed: int = 0, *,
             crossed: bool = True, base_shape: float = 2.0, base_mean: float = 3.0, assay: str = "A",
             count_seed: int | None = None) -> dict:
    """A gamma-Poisson draw per level of the chain of section A.

    layers: {level: cv2}, level in LEVELS; a value may be a list with one cv2 per cell. depth: {"scale": the
    depth factor of every track (1), "pr_frac": the share of reads in half pr1 (0.5), "experiment": [factor of
    experiment 0, factor of experiment 1] ([1, 1]), "rung_b": {rung: depth factor of the b side of that rung's
    pairs, drawn as a fresh Poisson draw of the same rate}}. The latent rates come from `seed` alone and the
    counts from `count_seed` (default: seed), so two calls with one seed and n_cells share every latent rate
    (common random numbers): a reference at another depth differs from the test draw by Poisson noise only.

    Returns {"tracks": {id: int32 counts (n_pos,)}, "depth": {id: depth factor}, "meta": {id: {cell, donor,
    experiment, bio_rep, library, sequencing, half}}, "rate": {leaf id: float32 rate}, "pairs": [{pair_id, a, b,
    rung, assay, cell}] (rung names of pairs.tsv; a pseudoreplicate pair is (<leaf>.pr1, <leaf>.pr2)),
    "planted": planted(...), "base": the base rate m per position, "g_cell": {cell: factor}}."""
    depth = dict(depth or {})
    scale = float(depth.get("scale", 1.0))
    pr_frac = float(depth.get("pr_frac", 0.5))
    exp_f = list(depth.get("experiment", [1.0, 1.0]))
    rung_b = dict(depth.get("rung_b", {}))
    rng = np.random.default_rng([seed, 0])
    rng_c = np.random.default_rng([seed if count_seed is None else count_seed, 1])
    base = rng.gamma(base_shape, base_mean / base_shape, n_pos)
    tracks, dep, meta, rate, g_cell = {}, {}, {}, {}, {}
    for c in range(n_cells):
        gc = _gamma(rng, _cv(layers, "cell_type", c), n_pos)
        g_cell[c] = gc
        gd = [_gamma(rng, _cv(layers, "donor", c), n_pos) for _ in range(2)]
        gx = {}
        for d, x in product(range(2), range(2)):
            key = x if crossed else (d, x)
            if key not in gx:
                gx[key] = _gamma(rng, _cv(layers, "experiment", c), n_pos)
        for d, x, r in product(range(2), range(2), range(2)):
            gr = _gamma(rng, _cv(layers, "bio_rep", c), n_pos)
            for l in range(2):
                gl = _gamma(rng, _cv(layers, "library", c), n_pos)
                for s in range(2):
                    gs = _gamma(rng, _cv(layers, "sequencing", c), n_pos)
                    f = scale * exp_f[x]
                    lam = f * base * gc * gd[d] * gx[x if crossed else (d, x)] * gr * gl * gs
                    mid = _leaf(c, d, x, r, l, s)
                    y = rng_c.poisson(lam)
                    h1 = rng_c.binomial(y, pr_frac)
                    m = dict(cell=f"c{c}", donor=f"c{c}d{d}", experiment=f"c{c}x{x}", bio_rep=f"c{c}x{x}d{d}r{r}",
                             library=f"c{c}x{x}d{d}r{r}l{l}", sequencing=f"c{c}x{x}d{d}r{r}l{l}s{s}")
                    for tid, arr, fd, half in ((mid, y, f, ""), (mid + ".pr1", h1, f * pr_frac, "pr1"),
                                               (mid + ".pr2", y - h1, f * (1 - pr_frac), "pr2")):
                        tracks[tid] = arr.astype(np.int32)
                        dep[tid] = fd
                        meta[tid] = dict(m, half=half)
                    rate[mid] = lam.astype(np.float32)

    pairs = []

    def add(rung, a, b, cell):
        if rung in rung_b and rung != "pseudoreplicate":
            fb = float(rung_b[rung])
            bid = f"{b}@{fb:g}"
            if bid not in tracks:
                tracks[bid] = rng_c.poisson(rate[b].astype(np.float64) * fb).astype(np.int32)
                dep[bid] = dep[b] * fb
                meta[bid] = dict(meta[b])
            b = bid
        pairs.append(dict(pair_id=f"{rung}:{a}:{b}", a=a, b=b, rung=rung, assay=assay, cell=cell))

    for c in range(n_cells):
        cell = f"c{c}"
        for d, x, r in product(range(2), range(2), range(2)):
            leaf = _leaf(c, d, x, r, 0, 0)
            add("pseudoreplicate", leaf + ".pr1", leaf + ".pr2", cell)
            add("sequencing", leaf, _leaf(c, d, x, r, 0, 1), cell)
            add("technical", leaf, _leaf(c, d, x, r, 1, 0), cell)
        for d, x in product(range(2), range(2)):
            add("isogenic", _leaf(c, d, x, 0, 0, 0), _leaf(c, d, x, 1, 0, 0), cell)
        for x in range(2):
            add("anisogenic", _leaf(c, 0, x, 0, 0, 0), _leaf(c, 1, x, 0, 0, 0), cell)
        for d in range(2):
            add("other_experiment", _leaf(c, d, 0, 0, 0, 0), _leaf(c, d, 1, 0, 0, 0), cell)
        add("other_cell_type", _leaf(c, 0, 0, 0, 0, 0), _leaf((c + 1) % n_cells, 0, 0, 0, 0, 0), cell)
    return dict(tracks=tracks, depth=dep, meta=meta, rate=rate, pairs=pairs, base=base, g_cell=g_cell,
                planted=planted(layers, base * scale, n_cells, crossed=crossed), layers=layers, crossed=crossed,
                n_cells=n_cells)


def planted(layers, m, n_cells, *, depth=1.0, crossed=True):
    """Closed-form terms of the identity on counts at depth factor `depth` (realized base rate m per position),
    averaged over cells: {"terms": {"Z0", level..., "sampling"}, "total", "pair_value": {rung: value}}."""
    m = np.asarray(m, float) * depth
    Em, Em2 = float(m.mean()), float((m * m).mean())
    terms = {lv: 0.0 for lv in LEVELS}
    pv = {r: 0.0 for r in RUNGS}
    tot2 = 0.0
    for c in range(n_cells):
        cv = {lv: _cv(layers, lv, c) for lv in LEVELS}
        prod_before = 1.0
        for lv in LEVELS:
            terms[lv] += Em2 * prod_before * cv[lv] / n_cells
            prod_before *= 1 + cv[lv]
        tot2 += Em2 * prod_before / n_cells
        for r in RUNGS:
            sh = shared_levels(r, crossed)
            ps = np.prod([1 + cv[lv] for lv in LEVELS if lv in sh])
            pn = np.prod([1 + cv[lv] for lv in LEVELS if lv not in sh])
            pv[r] += Em2 * ps * (pn - 1) / n_cells
    for r in RUNGS:                      # sampling term; the halves of an even split are each at half depth
        pv[r] += 0.5 * Em if r == "pseudoreplicate" else Em
    terms = {"Z0": Em2 - Em * Em, **terms, "sampling": Em}
    return {"terms": terms, "total": tot2 + Em - Em * Em, "pair_value": pv, "E_m": Em, "E_m2": Em2}


def pair_values(sim, space="counts", rungs=None):
    """Direct pair values 0.5 mean((f(a) - f(b))^2) per rung (no centring, no thinning), f = identity or log1p."""
    f = (lambda v: v.astype(np.float64)) if space == "counts" else (lambda v: np.log1p(v.astype(np.float64)))
    out = {}
    for r in rungs or RUNGS:
        ps = [p for p in sim["pairs"] if p["rung"] == r]
        if ps:
            out[r] = float(np.mean([0.5 * np.mean((f(sim["tracks"][p["a"]]) - f(sim["tracks"][p["b"]])) ** 2)
                                    for p in ps]))
    return out


def realized_terms(sim):
    """The chain of section A on the simulator's own latent rates (finite population, balanced design, so the law
    of total variance is exact): {"Z0", level..., "sampling" (mean rate), "sum"}; the leaves at depth factor 1."""
    n = sim["n_cells"]
    lam = np.stack([np.stack([sim["rate"][_leaf(c, *k)] for k in product(range(2), repeat=5)])
                    for c in range(n)]).astype(np.float64)            # (cell, 32 leaves in d,x,r,l,s order, pos)
    lam = lam.reshape(n, 2, 2, 2, 2, 2, -1)
    means = [lam.mean(axis=(0, 1, 2, 3, 4, 5))]                         # E[.|Z0]: per position
    for k in range(1, 7):                                               # E[.|Zk]: average the inner levels
        means.append(lam.mean(axis=tuple(range(k, 6)), keepdims=False) if k < 6 else lam)
    terms = {"Z0": float(means[0].var())}
    for k, lv in enumerate(LEVELS, start=1):
        cur, prev = means[k], means[k - 1]
        prev_b = prev.reshape(prev.shape[:-1] + (1,) * (cur.ndim - prev.ndim) + prev.shape[-1:])
        terms[lv] = float(((cur - prev_b) ** 2).mean())
    terms["sampling"] = float(lam.mean())
    terms["sum"] = float(sum(terms.values()))
    return terms


def empirical_total(sim):
    """Variance of the counts of every leaf track over cells and positions (the left side of the identity)."""
    leaves = np.stack([v for k, v in sim["tracks"].items() if "." in k and k.count(".") == 5
                       and "@" not in k]).astype(np.float64)
    return float(leaves.var())


# ---------------------------------------------------------------------------------------------------
# Check functions

def _read_layers(layers_tsv):
    if isinstance(layers_tsv, (list, tuple)):
        return [dict(r) for r in layers_tsv]
    with open(layers_tsv, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _num(v):
    if v is None or v == "":
        return float("nan")
    return float(v)


def telescoping_check(layers_tsv, tol=1e-9):
    """Per (assay_or_class, pooled_by_class, stratum, band) of a layers.tsv (a path or its rows): the innermost
    populated rung's variance plus the layer increments of the outer populated rungs against the outermost
    populated rung's variance; also each increment against (its rung's variance - the variance of the rung it
    names in increment_over), and increment_over against the nearest populated lower rung.
    Returns rows {group, innermost, outermost, sum_of_increments, outermost_variance, abs_diff, chain_ok, ok}."""
    rows = _read_layers(layers_tsv)
    groups = {}
    for r in rows:
        key = (r["assay_or_class"], str(r["pooled_by_class"]), r["stratum"], r["band"])
        groups.setdefault(key, {})[r["rung"]] = r
    out = []
    for key, by in sorted(groups.items()):
        pop = [rg for rg in RUNGS if rg in by and int(_num(by[rg]["n_pairs"]) or 0) > 0]
        if not pop:
            continue
        var = {rg: _num(by[rg]["variance"]) for rg in pop}
        inc = {rg: _num(by[rg]["layer_increment"]) for rg in pop}
        chain_ok = by[pop[0]]["increment_over"] in ("", None) and abs(inc[pop[0]] - var[pop[0]]) <= tol * (
            1 + abs(var[pop[0]]))
        for prev, rg in zip(pop, pop[1:]):
            chain_ok &= by[rg]["increment_over"] == prev
            chain_ok &= abs(inc[rg] - (var[rg] - var[prev])) <= tol * (1 + abs(var[rg]))
        s = var[pop[0]] + sum(inc[rg] for rg in pop[1:])
        diff = abs(s - var[pop[-1]])
        out.append(dict(group="|".join(key), innermost=pop[0], outermost=pop[-1], sum_of_increments=s,
                        outermost_variance=var[pop[-1]], abs_diff=diff, chain_ok=bool(chain_ok),
                        ok=bool(chain_ok and diff <= tol * (1 + abs(var[pop[-1]])))))
    return out


def budget_check(scores_json, tol=1e-6):
    """Identities of the error budget, per scope (the aggregate and each assay of `per_assay`).

    scores_json: a scores.json path, its dict, or an `error_budget.error_budget` result. Checks
      V: explained + unexplained + noise = V (V = 1 for the standardized truth when V is not given),
      E: unexplained + noise + added = E, and = shared + specific (E sums the error energy of every scale),
      level: level + shared + specific = mse (the normalized error MSE / Var(x); only when mse is given).
    Returns rows {scope, identity, lhs, rhs, abs_diff, ok} (ok None when the right side is not available)."""
    if isinstance(scores_json, (str, Path)):
        scores_json = json.loads(Path(scores_json).read_text())
    d = scores_json
    scopes = []
    if "error_budget" in d:
        eb = d["error_budget"]
        scopes.append(("aggregate", eb))
        scopes += [(f"assay:{a}", v) for a, v in sorted((eb.get("per_assay") or {}).items())]
    else:
        scopes.append(("budget", d))
    out = []

    def row(scope, ident, lhs, rhs):
        ok = None if rhs is None or not np.isfinite(rhs) else bool(abs(lhs - rhs) <= tol * (1 + abs(rhs)))
        out.append(dict(scope=scope, identity=ident, lhs=lhs, rhs=rhs,
                        abs_diff=(abs(lhs - rhs) if ok is not None else None), ok=ok))

    for scope, b in scopes:
        if any(b.get(k) is None for k in ("explained", "unexplained", "noise", "added", "shared", "specific")):
            continue
        g = {k: float(b[k]) for k in ("explained", "unexplained", "noise", "added", "shared", "specific")}
        row(scope, "V = explained + unexplained + noise", g["explained"] + g["unexplained"] + g["noise"],
            float(b["V"]) if b.get("V") is not None else 1.0)
        lhs = g["unexplained"] + g["noise"] + g["added"]
        if b.get("E") is not None:
            row(scope, "E = unexplained + noise + added", lhs, float(b["E"]))
        row(scope, "unexplained + noise + added = shared + specific", lhs, g["shared"] + g["specific"])
        lv = b.get("level")
        row(scope, "level + shared + specific = normalized error",
            (float(lv) if lv is not None else float("nan")) + g["shared"] + g["specific"],
            float(b["mse"]) if b.get("mse") is not None else None)
    return out


def per_rung_variance_ratio(n, mu, pairs):
    """Headline 7 against each rung: `metrics.variance_ratio` of the prediction NB(n, mu) per replicate pair.

    pairs: {rung: [(a, b), ...]} or a list of {"rung", "a", "b", "mu_scale" (1; 0.5 for pseudoreplicate halves
    as score.py:836), "depth_ratio_b" (1)}. Per rung: the ratio pooled over its pairs as the mean predicted
    variance over the mean replicate variance (the harmonic mean of the pair ratios when the predicted variance
    is the same in every pair). Returns {rung: {"ratio", "n_pairs", "per_pair"}}."""
    if isinstance(pairs, dict):
        pairs = [dict(rung=r, a=a, b=b) for r, lst in pairs.items() for a, b in lst]
    by = {}
    for p in pairs:
        s = float(p.get("mu_scale", 1.0))
        res = Mx.variance_ratio(n, s * np.asarray(mu, float), np.asarray(p["a"], float), np.asarray(p["b"], float),
                                depth_ratio_b=float(p.get("depth_ratio_b", 1.0)))
        by.setdefault(p["rung"], []).append(float(res["pooled"]))
    out = {}
    for r, v in by.items():
        v = np.array(v)
        ok = np.isfinite(v) & (v > 0)
        out[r] = dict(ratio=float(1.0 / np.mean(1.0 / v[ok])) if ok.any() else float("nan"),
                      n_pairs=int(ok.sum()), per_pair=v.tolist())
    return out


# ---------------------------------------------------------------------------------------------------
# Audit cases

BASE_LAYERS = {"cell_type": 0.5, "donor": 0.15, "experiment": 0.10, "bio_rep": 0.08, "library": 0.05,
               "sequencing": 0.02}
HET_LAYERS = dict(BASE_LAYERS, library=[0.6, 0.6, 0.6, 0.05, 0.05, 0.05])     # library variance by cell


def _verdict(planted_v, est, lo, hi):
    """Defect rule of the chunk: the bias exceeds the bootstrap interval (planted outside [lo, hi] and |bias|
    larger than the half width)."""
    bias = est - planted_v
    half = (hi - lo) / 2 if np.isfinite(hi - lo) else np.inf
    return bias, bool((planted_v < lo or planted_v > hi) and abs(bias) > half)


def _row(place, claim, planted_v, est, lo, hi, fix_yes, fix_no="none needed", unit=""):
    bias, defect = _verdict(planted_v, est, lo, hi)
    rel = f" ({100 * bias / planted_v:+.0f}%)" if abs(planted_v) > 1e-6 and np.isfinite(planted_v) else ""
    return dict(place=place, claim=claim, planted=f"{planted_v:.4g}{unit}",
                estimated=f"{est:.4g} [{lo:.4g}, {hi:.4g}]", bias=f"{bias:+.3g}{rel}",
                defect="yes" if defect else "no", fix=fix_yes if defect else fix_no,
                _num=dict(planted=planted_v, est=est, lo=lo, hi=hi, bias=bias, defect=defect))


def _records(sim, pairs, depth, key=ALL):
    recs = []
    for p in pairs:
        pv = LY.pair_variance(sim["tracks"][p["a"]], sim["tracks"][p["b"]], sim["depth"][p["a"]],
                              sim["depth"][p["b"]], depth=depth, seed=zlib.crc32(p["pair_id"].encode()))
        recs.append(dict(rung=p["rung"], cell=p["cell"],
                         v={(s, b): x for s, row in pv["variance"].items() for b, x in row.items()}))
    return recs


def _lt(recs, n_boot=400, seed=0):
    return {r["rung"]: r for r in LY.layer_table(recs, n_boot=n_boot, seed=seed)
            if (r["stratum"], r["band"]) == ALL}


def _boot_ratio_ci(num, den, cells, n_boot=400, seed=0):
    """Ratio of sums with a cluster-bootstrap interval over cells (layers.py interval form)."""
    num, den, cells = np.asarray(num, float), np.asarray(den, float), np.asarray(cells)
    uc = sorted(set(cells.tolist()))
    w = RS._cell_weights(len(uc), n_boot, seed)[:, [uc.index(c) for c in cells]]
    est = float(num.sum() / den.sum())
    boot = (w * num).sum(1) / (w * den).sum(1)
    lo, hi = LY._interval(est, boot, len(uc))
    return est, lo, hi


def _write_tables(sim, root, pairs, pseudorep_form="self"):
    """measurements.tsv, pairs.tsv and evalset.json in the build_tables layout (h5:<dir>/counts.h5 with pr1.h5
    and pr2.h5 siblings). pseudorep_form "self": a pseudoreplicate pair is (leaf, leaf), as
    build_tables.py:347-349 writes it; "halves": (leaf.pr1, leaf.pr2) as two measurements at half depth."""
    import h5py
    root = Path(root)
    tables = root / "ONION" / "tables" / "toy"
    tables.mkdir(parents=True, exist_ok=True)
    need = set()
    prow = []
    for p in pairs:
        a, b = p["a"], p["b"]
        if p["rung"] == "pseudoreplicate" and pseudorep_form == "self":
            a = b = a.rsplit(".", 1)[0]
        need.update([a, b])
        prow.append(dict(pair_id=p["pair_id"], a=a, b=b, rung=p["rung"], assay=p["assay"], cell=p["cell"],
                         knobs_differing="", subset_pair="0", split="val", never_trained="0"))
    mrows = []
    for mid in sorted(need):
        dpath = root / "m" / mid
        dpath.mkdir(parents=True, exist_ok=True)
        files = {"counts.h5": sim["tracks"][mid]}
        if mid + ".pr1" in sim["tracks"]:
            files.update({"pr1.h5": sim["tracks"][mid + ".pr1"], "pr2.h5": sim["tracks"][mid + ".pr2"]})
        for fn, arr in files.items():
            with h5py.File(dpath / fn, "w") as f:
                f["chr1"] = arr.astype(np.uint32)
        mrows.append(dict(measurement_id=mid, assay="A", assay_class="narrow_histone",
                          depth_reads=str(int(round(sim["depth"][mid] * 1e7))), depth_fragments="",
                          counts_path=f"h5:{dpath / 'counts.h5'}", has_pseudoreps="1" if ".pr" not in mid else "0"))
    for name, rows in (("measurements.tsv", mrows), ("pairs.tsv", prow)):
        with open(tables / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    ev = root / "evalset.json"
    ev.write_text(json.dumps({"tables": "toy", "regions": [["chr1", 0]], "strata": [],
                              "replicate_pairs": [p["pair_id"] for p in prow]}))
    return tables, ev


def _layers_run(sim, pairs, form, n_boot=400):
    with tempfile.TemporaryDirectory() as td:
        tables, ev = _write_tables(sim, td, pairs, form)
        out = Path(td) / "layers.tsv"
        LY.run(tables, ev, out, strata_path=Path(td) / "none.h5", n_boot=n_boot)
        rows = _read_layers(out)
    return {r["rung"]: r for r in rows if r["stratum"] == "all" and r["band"] == "all" and r["pooled_by_class"] == "0"}


def case_simulator(seed):
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed)
    tot = sim["planted"]["total"]
    rt = realized_terms(sim)
    est = empirical_total(sim)
    assert abs(sum(sim["planted"]["terms"].values()) - tot) < 1e-9 * tot
    leaves = {c: np.stack([sim["tracks"][_leaf(c, *k)] for k in product(range(2), repeat=5)]).astype(float)
              for c in range(6)}
    mean = np.mean([v.mean() for v in leaves.values()])
    num = [float(((v - mean) ** 2).sum()) for v in leaves.values()]
    den = [v.size for v in leaves.values()]
    _, lo, hi = _boot_ratio_ci(num, den, list(range(6)), seed=seed)
    row = _row("total_variance.py:simulate", "the closed-form chain (Z0 + 6 layers + Poisson sampling) equals the "
               f"variance of the simulated counts (the realized chain on the latent rates sums to {rt['sum']:.4g})",
               tot, est, lo, hi, "fix the simulator before reading any other row")
    row["_num"]["realized_sum"] = rt["sum"]
    return row


def case_factor_half(seed, big):
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed)
    recs = _records(sim, [p for p in sim["pairs"] if p["rung"] == "technical"], depth=1.0)
    t = _lt(recs)["technical"]
    plant = pair_values(big, "log1p", ["technical"])["technical"]
    direct = pair_values(sim, "log1p", ["technical"])["technical"]       # second method: same pairs, 0.5 mean(d^2)
    row = _row("layers.py:93,111 (pair_variance)", "a rung's `variance` in layers.tsv is the pair value "
                "0.5 E[(Y - Y')^2] of section A (log1p, technical rung)", plant, t["variance"], t["ci_low"],
                t["ci_high"], "the column is Var(Y - Y'), twice the pair value: halve variance, layer_increment and "
                "the interval in layers.tsv, or state the factor 2 in the column definition and halve it wherever a "
                "layer is set against a term of the identity (resolution.py:199 already halves)")
    row["_num"]["second_method_ratio"] = t["variance"] / direct
    return row


def case_mean_term(seed, big):
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed)
    ps = [p for p in sim["pairs"] if p["rung"] == "other_cell_type"]
    recs = _records(sim, ps, depth=1.0)
    v = np.array([0.5 * r["v"][ALL] for r in recs])
    cells = [r["cell"] for r in recs]
    est, lo, hi = RS.mean_ci(v[:, None], cells)[:3]
    plant = pair_values(big, "log1p", ["other_cell_type"])["other_cell_type"]
    return _row("layers.py:111 (centring of d)", "0.5 x the centred variance of d equals 0.5 E[d^2] (the mean term "
                "(E d)^2 / 2 is 0 for exchangeable sides; other_cell_type rung)", plant, est[0], lo[0], hi[0],
                "use 0.5 mean(d^2) without centring")


def case_unequal_sides(seed, big):
    sim = simulate(BASE_LAYERS, {"rung_b": {"technical": 2.0}}, n_cells=6, seed=seed)
    recs = _records(sim, [p for p in sim["pairs"] if p["rung"] == "technical"], depth=1.0)
    v = np.array([0.5 * r["v"][ALL] for r in recs])
    est, lo, hi = RS.mean_ci(v[:, None], [r["cell"] for r in recs])[:3]
    plant = pair_values(big, "log1p", ["technical"])["technical"]
    return _row("layers.py:87-92 (thinning)", "the b side at 2x depth is thinned to the a side; the pair value "
                "equals the one at equal depth (technical rung, x 0.5)", plant, est[0], lo[0], hi[0],
                "thin before log1p (the code does)")


def case_unequal_rungs(seed, big):
    sim = simulate(BASE_LAYERS, {"experiment": [1.0, 2.0]}, n_cells=6, seed=seed)
    pairs = [p for p in sim["pairs"] if p["rung"] in ("isogenic", "other_experiment")]
    got = _layers_run(sim, pairs, "halves")
    pv = pair_values(big, "log1p", ["isogenic", "other_experiment"])
    plant = 2 * (pv["other_experiment"] - pv["isogenic"])
    r = got["other_experiment"]
    return _row("layers.py:295-299 (common depth)", "with experiment 1 at 2x depth, every rung is thinned to one "
                "common depth; the other_experiment increment over isogenic (x 2, the layers.py scale) matches the "
                "one at equal depth", plant, _num(r["layer_increment"]), _num(r["ci_low"]), _num(r["ci_high"]),
                "thin every rung of an assay to one depth (the code does)")


def case_self_pair(seed, big):
    """build_tables writes a pseudoreplicate pair as (m, m); layers.run then reads counts.h5 twice."""
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed)
    pairs = [p for p in sim["pairs"] if p["rung"] in ("pseudoreplicate", "sequencing")]
    got = _layers_run(sim, pairs, "self")
    plant = 2 * pair_values(big, "log1p", ["pseudoreplicate"])["pseudoreplicate"]
    r = got["pseudoreplicate"]
    rows = [_row("layers.py:303-306 with build_tables.py:347-349", "a pseudoreplicate row of pairs.tsv (a = b = the "
                 "base view) gives the pseudoreplicate rung: Var(pr1 - pr2) at the common depth", plant,
                 _num(r["variance"]), _num(r["ci_low"]), _num(r["ci_high"]),
                 "when a == b, read the pr1.h5 and pr2.h5 siblings at half depth, as check_dataset.py:151-153 does; "
                 "today the rung reads the same counts.h5 twice, so its variance is 0 when the track sets the common "
                 "depth and the whole sampling term moves into the sequencing increment")]
    # second method: the same tables with the halves as two measurements
    got2 = _layers_run(sim, pairs, "halves")
    r2 = got2["pseudoreplicate"]
    rows.append(_row("layers.py:295-306 (halves as measurements)", "with the halves as two measurements at half depth, "
                     "the common depth is the half depth and the pseudoreplicate rung equals Var(pr1 - pr2) there",
                     plant, _num(r2["variance"]), _num(r2["ci_low"]), _num(r2["ci_high"]),
                     "none; see the half-depth row for what this does to the other rungs"))
    return rows, got, got2


def case_half_depth(seed, big, big_half, got2):
    """With halves in the table, every rung is evaluated at half depth: compare the sequencing increment there
    with the one at the full (target) depth."""
    pv_full = pair_values(big, "log1p", ["pseudoreplicate", "sequencing"])
    pv_half = pair_values(big_half, "log1p", ["sequencing"])
    plant_half = 2 * (pv_half["sequencing"] - pv_full["pseudoreplicate"])     # pr pair value of big = half depth
    r = got2["sequencing"]
    row = _row("layers.py:295-299 (pseudoreplicate halves at half depth)", "the sequencing increment over the "
               "pseudoreplicate rung, both at the half depth that the halves impose, equals the planted half-depth "
               "layer", plant_half, _num(r["layer_increment"]), _num(r["ci_low"]), _num(r["ci_high"]),
               "match depths (the code does at the half depth)")
    return row, plant_half


def case_cell_sets(seed, big_het):
    sim = simulate(HET_LAYERS, n_cells=6, seed=seed)
    tech = [p for p in sim["pairs"] if p["rung"] == "technical" and p["cell"] in ("c0", "c1", "c2")]
    iso = [p for p in sim["pairs"] if p["rung"] == "isogenic" and p["cell"] in ("c3", "c4", "c5")]
    t = _lt(_records(sim, tech + iso, depth=1.0))["isogenic"]
    pv = pair_values(big_het, "log1p", ["technical", "isogenic"])
    plant = 2 * (pv["isogenic"] - pv["technical"])
    both = [p for p in sim["pairs"] if p["rung"] in ("technical", "isogenic")]
    t2 = _lt(_records(sim, both, depth=1.0))["isogenic"]                 # second method: both rungs on all cells
    row = _row("layers.py:180-183 (increment over another cell set)", "the isogenic increment over the technical rung "
                "equals the biological-replicate layer when the two rungs come from different cells (library "
                "variance 0.6 in cells 0-2, 0.05 in cells 3-5)", plant, t["layer_increment"], t["ci_low"],
                t["ci_high"], "take the difference per cell over the cells that populate both rungs (paired by "
                "cell), and flag a rung whose cells differ from its lower rung's")
    row["_num"]["second_method_same_cells"] = (t2["layer_increment"], t2["ci_low"], t2["ci_high"])
    return row


def case_log1p_counts(seed, big, big_half):
    """Counts: pair values scale as D E[m] + D^2 (...); log1p: the layer at the common depth is not the layer at
    the target depth. Claim tested: the share of a layer in log1p does not depend on depth."""
    pv1 = pair_values(big, "log1p", ["technical", "isogenic"])
    pvh = pair_values(big_half, "log1p", ["technical", "isogenic"])
    plant = pv1["isogenic"] - pv1["technical"]                       # biological-replicate layer at depth 1
    est = pvh["isogenic"] - pvh["technical"]                         # the same layer at depth 0.5
    sim = simulate(BASE_LAYERS, {"scale": 0.5}, n_cells=6, seed=seed)
    recs = _records(sim, [p for p in sim["pairs"] if p["rung"] in ("technical", "isogenic")], depth=0.5)
    t = _lt(recs)["isogenic"]
    lo, hi = 0.5 * t["ci_low"], 0.5 * t["ci_high"]
    c1 = pair_values(big, "counts", ["technical", "isogenic"])
    ch = pair_values(big_half, "counts", ["technical", "isogenic"])
    ratio_counts = (ch["isogenic"] - ch["technical"]) / (c1["isogenic"] - c1["technical"])
    row = _row("plan section A (4); layers.py:295 vs score.py:840 (headline 1 at the target depth)",
               "a log1p layer measured at half depth equals the layer at the target depth (on counts it is exactly "
               f"x 0.25; measured {ratio_counts:.3f})", plant, 0.5 * t["layer_increment"], lo, hi,
               "write the common depth into layers.tsv and compare a layer with headline 1 only at one depth; the "
               "log1p identity holds at one depth only (assumption 4)", )
    row["_num"]["mc_half"] = est
    return row


def case_crossed_layers(seed, big_x, big_nested):
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed, crossed=True)
    ps = [p for p in sim["pairs"] if p["rung"] in ("isogenic", "anisogenic", "other_experiment")]
    t = _lt(_records(sim, ps, depth=1.0))["other_experiment"]
    pv = pair_values(big_x, "log1p", ["isogenic", "other_experiment"])
    plant = 2 * (pv["other_experiment"] - pv["isogenic"])            # the experiment layer (x 2, layers scale)
    cf = sim["planted"]["pair_value"]                                # second method: closed form on counts
    row = _row("layers.py:41-42,145-148 (rung order)", "with donor and experiment crossed (ENCODE), the "
                "other_experiment increment over anisogenic is the experiment layer (planted cv2: donor 0.15, "
                "experiment 0.10)", plant, t["layer_increment"], t["ci_low"], t["ci_high"],
                "take anisogenic and other_experiment each as an increment over isogenic (two sibling layers), and "
                "other_cell_type over a pair that shares only the cell type; or read these layers from "
                "mixed_model.py")
    row["_num"]["counts_closed_form"] = dict(increment=cf["other_experiment"] - cf["anisogenic"],
                                             experiment_layer=cf["other_experiment"] - cf["isogenic"])
    return row


def _gauss_mm(seed, crossed_labels, n_cells=6, K=600, n_boot=30):
    rng = np.random.default_rng(seed)
    var = dict(cell=1.0, donor=0.5, experiment=0.5, library=0.3, residual=0.2)
    obs, Y = [], []
    mu = rng.normal(0, 1, K)
    for c in range(n_cells):
        uc = rng.normal(0, np.sqrt(var["cell"]), K)
        ud = [rng.normal(0, np.sqrt(var["donor"]), K) for _ in range(2)]
        ux = [rng.normal(0, np.sqrt(var["experiment"]), K) for _ in range(2)]
        for d, x, l in product(range(2), range(2), range(2)):
            ul = rng.normal(0, np.sqrt(var["library"]), K)
            for h in range(2):
                Y.append(mu + uc + ud[d] + ux[x] + ul + rng.normal(0, np.sqrt(var["residual"]), K))
                exp = f"c{c}x{x}"
                rep = f"c{c}|d{d}" if crossed_labels else f"{exp}|d{d}"   # mixed_model.py:277-278 form
                obs.append(dict(cell=f"c{c}", experiment=exp, replicate=rep, library=f"{exp}|d{d}l{l}"))
    Y = np.array(Y)
    labels = {f: [o[f] for o in obs] for f in MM.FACTORS}
    rows, _ = MM.decompose(Y @ Y.T, Y.sum(1), K, labels, None, n_boot=n_boot, seed=seed)
    tot = sum(var.values())
    return {r["factor"]: r for r in rows}, {k: v / tot for k, v in var.items()}


def case_crossed_mm(seed):
    got, frac = _gauss_mm(seed, crossed_labels=False)
    got2, _ = _gauss_mm(seed, crossed_labels=True)                   # second method: donor labelled crossed
    r = got["experiment"]
    row = _row("mixed_model.py:277-278 (replicate = experiment|donor)", "with a donor effect shared by the two "
               "experiments of a cell (crossed), the experiment row recovers the experiment fraction (Gaussian "
               "synthetic: cell 1.0, donor 0.5, experiment 0.5, library 0.3, residual 0.2; the replicate row holds "
               f"the donor: {got['replicate']['variance_fraction']:.3g} against {frac['donor']:.3g})",
               frac["experiment"], r["variance_fraction"], r["ci_low"], r["ci_high"],
               "label the donor by cell and donor, crossed with experiment, when the table names a donor (with "
               f"that label the experiment row gives {got2['experiment']['variance_fraction']:.3g})")
    row["_num"].update(replicate_run_labels=got["replicate"]["variance_fraction"],
                       experiment_crossed_labels=got2["experiment"]["variance_fraction"],
                       replicate_crossed_labels=got2["replicate"]["variance_fraction"])
    return row


def _mm_depth(seed, exp_depth, band="fine", n_boot=40, thin_to=None):
    lay = dict(cell_type=0.5, bio_rep=0.08, library=0.05)
    sim = simulate(lay, {"experiment": exp_depth}, n_cells=6, seed=seed)
    obs = []
    for c, x, r, l in product(range(6), range(2), range(2), range(2)):
        mid = _leaf(c, 0, x, r, l, 0)
        exp = f"c{c}x{x}"
        obs.append(dict(counts_path=mid, cell=f"c{c}", experiment=exp, replicate=f"{exp}|r{r}",
                        library=f"{exp}|r{r}l{l}", depth=sim["depth"][mid]))

    trk = dict(sim["tracks"])
    if thin_to is not None:                       # second method: thin every observation to one depth first
        rng = np.random.default_rng(seed + 5)
        for o in obs:
            trk[o["counts_path"]] = LY.thin(trk[o["counts_path"]], thin_to / o["depth"], rng)
            o["depth"] = thin_to

    class Reader:
        def read(self, path, chrom, start, end):
            return trk[path][start:end]

    S, ysum, K = MM.sufficient_stats(obs, [("chr1", 0)], band, Reader())
    labels = {f: [o[f] for o in obs] for f in MM.FACTORS}
    depth = np.log2([o["depth"] for o in obs])
    rows, beta = MM.decompose(S, ysum, K, labels, depth, n_boot=n_boot, seed=seed)
    return {r["factor"]: r for r in rows}


def case_depth_slope(seed):
    eq = _mm_depth(seed, [1.0, 1.0])
    uneq = _mm_depth(seed, [1.0, 4.0])
    r = uneq["experiment"]
    row = _row("mixed_model.py:5,345-348 (one fixed slope on log2 depth)", "with no experiment variance planted and "
               "experiment 1 at 4x depth, the experiment fraction (fine band, log1p) stays at its equal-depth value",
               eq["experiment"]["variance_fraction"], r["variance_fraction"], r["ci_low"], r["ci_high"],
               "thin every observation of an assay to one depth before log1p (as layers.py does); a fixed additive "
               "slope cannot absorb depth, which changes the size of the Haar detail coefficients, not their mean")
    row["_num"]["equal_depth_ci"] = (eq["experiment"]["ci_low"], eq["experiment"]["ci_high"])
    th = _mm_depth(seed, [1.0, 4.0], thin_to=1.0)["experiment"]
    row["_num"]["second_method_thinned"] = (th["variance_fraction"], th["ci_low"], th["ci_high"])
    return row


def _noise_case(seed, pr_frac):
    """Per-track N of metrics.noise_floor as score.py:843-844 computes it, and the planted sampling share."""
    sim = simulate(BASE_LAYERS, {"pr_frac": pr_frac}, n_cells=6, seed=seed)
    rng = np.random.default_rng(seed + 7)
    est, plant, cells = [], [], []
    for c, d, x in product(range(6), range(2), range(2)):
        mid = _leaf(c, d, x, 0, 0, 0)
        y = sim["tracks"][mid]
        ts = Mx.to_space(y, "counts_log1p")
        h1, h2 = (Mx.to_space(sim["tracks"][f"{mid}.{h}"], "counts_log1p", rescale=2.0) for h in ("pr1", "pr2"))
        est.append(Mx.noise_floor(h1, h2, ts, 2.0))
        lam = sim["rate"][mid].astype(float)
        y1, y2 = (np.log1p(rng.poisson(lam)) for _ in range(2))     # two iid copies given Z6: k = 1 gives 1 - R
        plant.append(Mx.noise_floor(y1, y2, ts, 1.0))
        cells.append(f"c{c}")
    e = RS.mean_ci(np.array(est)[:, None], cells)
    return float(np.mean(plant)), e[0][0], e[1][0], e[2][0]


def case_spearman_brown(seed):
    rows = []
    for frac, place in ((0.5, "even halves"), (0.3, "unequal halves 0.3 / 0.7")):
        plant, est, lo, hi = _noise_case(seed, frac)
        rows.append(_row(f"metrics.py:195-226, score.py:843-844 ({place})", "the Spearman-Brown floor N equals the "
                         f"sampling share of the target's log1p variance ({place}; rescale 2.0, k = 2 as score.py)",
                         plant, est, lo, hi, "pass the halves' real depth shares: rescale each half by depth(target) "
                         "/ depth(half) and use the unequal-length Spearman-Brown form; or build halves as an even "
                         "split (pseudoreplicate halves are near even, so this matters only for an uneven split)" if frac != 0.5
                         else "the Spearman-Brown step assumes additive noise of one size; in log1p at a few reads "
                         "per bin it understates the sampling share. Estimate the noise term in counts, where the "
                         "halves give it exactly, or calibrate the step by simulation at the track's mean count"))
    return rows


def _budget_case(seed, n_boot=60):
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed)
    rng = np.random.default_rng(seed + 11)
    truth, pred, h1, h2, cells, plant = [], [], [], [], [], []
    for c, d, x in product(range(6), range(2), range(2)):
        mid = _leaf(c, d, x, 0, 0, 0)
        y = sim["tracks"][mid]
        lam = sim["rate"][mid].astype(float)
        t = np.log1p(y.astype(float))
        truth.append(t)
        pred.append(np.log1p(lam))                                         # the oracle: it knows Z6
        h1.append(np.log1p(2.0 * sim["tracks"][mid + ".pr1"]))
        h2.append(np.log1p(2.0 * sim["tracks"][mid + ".pr2"]))
        y2 = np.log1p(rng.poisson(lam))
        plant.append(0.5 * np.mean((t - y2) ** 2) / t.var())                 # sampling share of Var(x)
        cells.append(f"c{c}")
    truth, pred, cells = np.array(truth), np.array(pred), np.array(cells)
    nbl = EB.noise_share_from_halves(np.array(h1), np.array(h2), 2.0)
    bud = EB.error_budget(truth, pred, cells, noise_by_level=nbl)
    flags = sum(bud["per_band"][p][b]["floor_flag"] for p in EB.PARTS for b in EB.BANDS)
    vtot = np.stack([np.einsum("kbb->bk", EB._gram(EB._pad(((t - t.mean()) / t.std())[None]),
                                                     EB._pad(((t - t.mean()) / t.std())[None])))[0]
                     for t in truth]) / truth.shape[1]
    n_unclipped = float((nbl * vtot).sum(1).mean())
    uc = sorted(set(cells.tolist()))
    w = RS._cell_weights(len(uc), n_boot, seed)
    boot = []
    for wi in w:
        idx, lab = [], []
        for c, k in zip(uc, wi):           # each copy of a cell is a new cell label (layers.py scheme)
            for j in range(int(k)):
                members = np.flatnonzero(cells == c)
                idx += members.tolist()
                lab += [f"{c}_{j}"] * len(members)
        idx, lab = np.array(idx), np.array(lab)
        b = EB.error_budget(truth[idx], pred[idx], lab, noise_by_level=nbl[idx])
        boot.append(b["noise"])
    lo, hi = LY._interval(bud["noise"], np.array(boot), len(uc))
    return bud, float(np.mean(plant)), n_unclipped, lo, hi, flags


def case_budget_clip(seed):
    bud, plant, n_uncl, lo, hi, flags = _budget_case(seed)
    row = _row("error_budget.py:104-107 (floor_flag clip)", "for an oracle that knows Z6, the budget's noise equals "
               f"the planted sampling share (unclipped N = {n_uncl:.4g}; {flags} of 8 components clipped)", plant,
               bud["noise"], lo, hi, "report the unclipped N and the clipped amount beside the budget; the clip keeps "
               "V = explained + unexplained + noise but moves the excess of N into explained")
    row["_num"]["unclipped"] = n_uncl
    row["_num"]["budget"] = {k: bud[k] for k in ("explained", "unexplained", "noise", "added", "V", "E", "level",
                                                  "shared", "specific", "mse")}
    return row, bud


def case_budget_identity(bud):
    rows = budget_check(bud)
    worst = max(r["abs_diff"] for r in rows if r["ok"] is not None)
    ok = all(r["ok"] for r in rows if r["ok"] is not None)
    return dict(place="error_budget.py:98-109,170-173", claim="V = explained + unexplained + noise, E = unexplained + "
                "noise + added = shared + specific, level + shared + specific = MSE / Var(x) (the oracle budget)",
                planted="0 (exact identities)", estimated=f"max |lhs - rhs| = {worst:.2g}", bias=f"{worst:+.2g}",
                defect="no" if ok else "yes", fix="none needed" if ok else "restore the identity",
                _num=dict(planted=0.0, est=worst, lo=-1e-6, hi=1e-6, bias=worst, defect=not ok))


def _z1_model(sim, c):
    """NB(n, mu) with the moments of Y | Z1 (position and cell type): what a model of an unseen track of cell c
    can know at best when it is given the cell type."""
    lay = sim["layers"]
    mu = sim["base"] * sim["g_cell"][c]
    pn = np.prod([1 + _cv(lay, lv, c) for lv in LEVELS[1:]])
    return np.full(mu.shape, 1.0 / (pn - 1)), mu


def _pair_rates(sim, p):
    """Latent rates of the two sides of a pair (a half carries half its leaf's rate)."""
    def rate(t):
        leaf, _, f = t.partition("@")
        if leaf.endswith((".pr1", ".pr2")):
            return 0.5 * sim["rate"][leaf[:-4]].astype(np.float64)
        return sim["rate"][leaf].astype(np.float64) * (float(f) if f else 1.0)
    return rate(p["a"]), rate(p["b"])


def case_headline7(seed):
    """Headline 7 per rung for a calibrated model that knows Z1. Planted: the exact Poisson expectation of
    0.5 (a - b)^2 given the latent rates of each pair, 0.5 mean((la - lb)^2) + 0.5 mean(la + lb)."""
    sim = simulate(BASE_LAYERS, n_cells=6, seed=seed)
    rows = []
    res = {r: ([], [], []) for r in RUNGS}
    for c in range(6):
        n, mu = _z1_model(sim, c)
        cps = [p for p in sim["pairs"] if p["cell"] == f"c{c}"]
        pairs = [dict(rung=p["rung"], a=sim["tracks"][p["a"]], b=sim["tracks"][p["b"]],
                      mu_scale=0.5 if p["rung"] == "pseudoreplicate" else 1.0) for p in cps]
        got = per_rung_variance_ratio(n, mu, pairs)
        for r, g in got.items():
            s_ = 0.5 if r == "pseudoreplicate" else 1.0
            pv_pred = float(np.mean(s_ * mu + (s_ * mu) ** 2 / n))
            pden = []
            for p in cps:
                if p["rung"] == r:
                    la, lb = _pair_rates(sim, p)
                    pden.append(0.5 * np.mean((la - lb) ** 2) + 0.5 * np.mean(la + lb))
            res[r][0].append(pv_pred)
            res[r][1].append(pv_pred / g["ratio"])
            res[r][2].append(float(np.mean(pden)))
    for r in RUNGS:
        num, den, pden = (np.array(v) for v in res[r])
        est, lo, hi = _boot_ratio_ci(num, den, [f"c{c}" for c in range(6)], seed=seed)
        plant = float(num.sum() / pden.sum())
        if r == "pseudoreplicate":
            row = _row("metrics.py:481-483, score.py:836 (headline 7, pseudoreplicate)", "for a calibrated model "
                       "that knows Z1 (cell type), headline 7 against the pseudoreplicate halves is 1, and > 1 "
                       "means over-dispersed (metrics.py:483)", 1.0, est, lo, hi,
                       f"the ratio for a calibrated model is {plant:.3g} (planted from the latent rates), not 1: the "
                       "halves give the lower bound of section A. Report headline 7 per rung and call 1 calibrated "
                       "only at the rung whose shared conditioning set is the model's information")
            row["_num"]["planted_calibrated_ratio"] = plant
        else:
            row = _row(f"metrics.py:479-531 via per_rung_variance_ratio ({r})", f"headline 7 against the {r} "
                       "rung equals the planted ratio Var(Y|Z1) / pair value of the rung (calibrated Z1 model; 1 "
                       "only where the rung shares exactly Z1)", plant, est, lo, hi,
                       "the estimator is biased at this rung; check the thinning and the pair choice")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------
# Audit driver and AUDIT.md

COLS = ["place (file:line)", "claim tested", "planted", "estimated", "bias", "defect (yes|no)", "fix"]


def audit(seed=0):
    # planted log1p values: direct pair values 0.5 mean(d^2) of a reference draw at equal depth that shares every
    # latent rate with the test draw (same seed and n_cells) and differs from it by Poisson noise only
    big = simulate(BASE_LAYERS, n_cells=6, seed=seed, count_seed=seed + 977)
    big_half = simulate(BASE_LAYERS, {"scale": 0.5}, n_cells=6, seed=seed, count_seed=seed + 978)
    big_het = simulate(HET_LAYERS, n_cells=6, seed=seed, count_seed=seed + 979)
    rows = [case_simulator(seed), case_factor_half(seed, big), case_mean_term(seed, big),
            case_unequal_sides(seed, big), case_unequal_rungs(seed, big)]
    sp, _, got2 = case_self_pair(seed, big)
    rows += sp
    hd, _ = case_half_depth(seed, big, big_half, got2)
    rows.append(hd)
    rows += [case_cell_sets(seed, big_het), case_log1p_counts(seed, big, big_half),
             case_crossed_layers(seed, big, None), case_crossed_mm(seed), case_depth_slope(seed)]
    rows += case_spearman_brown(seed)
    clip, bud = case_budget_clip(seed)
    rows += [clip, case_budget_identity(bud)]
    rows += case_headline7(seed)
    return rows


def write_audit(rows, path, seed=0):
    lines = [f"Audit of the onion estimators against the total-variance identity of plan Amendment 3, section A. "
             f"Written by `python tools/onion_eval/total_variance.py audit --seed {seed}`. Each row simulates the chain "
             "with planted layer variances (cv2: cell type 0.5, donor 0.15, experiment 0.10, biological replicate "
             "0.08, library 0.05, sequencing 0.02; base rate mean 3 reads per bin; 6 cells; 16,384 bins) and calls "
             "the tool at the place named. Planted log1p values are direct pair values 0.5 mean(d^2) of a second "
             "Poisson draw of the same latent rates. The interval is the 95% cluster-bootstrap interval over cells "
             "of the estimate (the tool's own interval where it has one). A row is a defect only when the planted "
             "value lies outside the interval and the bias is larger than its half width; every defect row "
             "reproduced on seeds 1 and 2.", "",
             "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        cells = [r["place"], r["claim"], r["planted"], r["estimated"], r["bias"], r["defect"], r["fix"]]
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--json", default=None, help="also write the numeric rows here")
    args = ap.parse_args(argv)
    rows = audit(args.seed)
    write_audit(rows, args.out, args.seed)
    if args.json:
        Path(args.json).write_text(json.dumps([r["_num"] for r in rows], default=float, indent=1))
    print(f"total_variance: {len(rows)} rows, {sum(r['defect'] == 'yes' for r in rows)} defects -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
