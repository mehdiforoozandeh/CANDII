"""Cap test of deduplication at 25 bp on the dedup-off / dedup-on view pairs (t396 C10a; R43, R29, R30).

A pair is a `*.nodedup` measurement (dedup off, the source) and its `parent_measurement_id` (the
dedup-on base, the target). Start space: per strand, the read starts of `starts.h5` of the
read-start store. Count space: the overlap counts of `counts.h5` of the read-count store. Laws,
fits, deviances and floors come from `caps` (C4); this tool reads the stores, builds the joint
tables and writes the tables.

Subcommands
  rows        --tables <dir> --rs-store <dir> [--count-store <dir>] [--layer testbed
              [--totals totals.tsv]] --out rows.tsv
  stats       --pair <off id> <on id> --unit-rule UNIT_RULE.json [--rows rows.tsv] --out <npz>
  fit         --stats <npz...> --rows rows.tsv [--iteration1-fit per_assay.tsv] --out <tsv>
  floor-count --pair <off id> <on id> --unit-rule UNIT_RULE.json --fit per_assay.tsv
              --rows rows.tsv --out <npz>
  score       --stats <npz...> --fit per_assay.tsv --rows rows.tsv [--floor-count <npz...>]
              --out <dir>

Testbed pairs (iteration 2; R43, R44, R39). `rows --layer testbed` picks the controlled pairs by
`layer` and `arm_knob`, never by `in_eic`: covariate `dedup` = (`.nodedup` -> its parent `.base`),
covariate `run_type` = (`.pe` -> its parent `.base`, `sepe_cap.py`). Each arm of a testbed track is
its own 30M-line (DNase 50M) subsample, so a row carries the subsampling rates
`q = stored lines (depth_reads) / lines before the subsample` of its source and target. `--totals`
is a tsv with columns `measurement_id`, `lines_before_subsample` (`NA` when unknown); a pair with a
missing total gets `q_origin = fitted` (flag `q_fitted`) and `fit` fits the missing rate on the
pair's own start table at the iteration-1 shape (both missing: one rate shared by both). The share
of a source read kept in the target's library is `lines_per_fragment` of the target over that of
the source (0.5 for a PE source and an SE target; `depth_reads / depth_fragments`). Start law:
`caps.subsampled_dm_distinct(c, q_source, q_target, a, K, share)`; count law:
`caps.subsampled_count_saturation`; baselines identity and depth scaling. The unseen reads of a
source bin come from a Gamma prior on the bin's full-library intensity, fitted per pair to the
source track (`caps.fit_completion_prior`, start and count space apart): with the scale-free prior
an empty source bin would predict an empty target, which is wrong genome-wide at q_source < 1. Two shapes, scored side by
side: `iteration1` (the per-assay `a` and `kappa` of `cruxvault/results/t396/dedup_cap/
per_assay.tsv`, fitted on EIC train pairs, out of sample here) and `loto` (fitted leave-one-track-
out on the testbed pairs of the covariate). `stats` also stores the target's half-vs-half tables
(`starts_pr1/2.h5`, `pr1/2.h5`); `score` reports `halves_floor`, the deviance of the target `pr2`
under `pr1` scaled by the genome-wide pr2 / pr1 ratio (half depth; reported beside
`attainable_floor`, never used in a gap). The testbed rows of both tools share one column set, so
the two rows files concatenate into one; every testbed subcommand keeps the rows of its own
covariate. The count-space floor (`floor-count`) uses the `loto` shape for both shape rows.

K. `K = caps.K_of(units)` when `UNIT_RULE.json` says `"dedup_unit": "replicate"`, else `K = 25`.
`units` is the number of read files of one half in the dedup-on view's read-start provenance.

Shared positional weights (secondary law `dm_distinct_shared`). `caps.dm_distinct(c, a, 25 U)`
treats the 25 U slots of a pooled view as independent Dirichlet weights, so the split of the reads
between units follows the weights. In the data the units share their 25 positional weights and the
split follows the units' read shares w_u. Expectation is linear, so the exact mean is
sum_u thinned_dm_distinct(c, w_u, a, 25) (each unit sees Binomial(c, w_u) reads on 25 positions).
With 2 equal units the independent law is low by 3.8-8.1% at a = 0.05, 0.5-2.2% at a = 0.2 and
under 0.5% at a >= 1 (c from 2 to 200; tests/test_t396_dedup_cap.py measures it).

numpy, scipy, h5py only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from t396 import caps  # noqa: E402
from onion_eval.deviance import unit_deviance  # noqa: E402

RS_STORE = "/scratch/mforooz/rs392/store"
COUNT_STORE = "/scratch/mforooz/onion/measurements"
TABLES = "/project/def-maxwl/mforooz/ONION/tables/eic_v1_2"
RES = 25
REACH = 8
SPLITS = ("train", "val")          # rows with in_eic = blind are not read
GATE_GAP = 0.8                     # gate (d): explains if gap closed vs depth scaling >= 0.8
N_BOOT_FIT = 50
N_BOOT_SCORE = 1000
SIM_CHUNK = 200_000                # strand-bins per simulation chunk
PRIMARY_START_LAW = "dm_distinct"
COUNT_LAW = "count_saturation"

ROW_COLUMNS = ["off_id", "on_id", "experiment", "assay", "layer", "run_type", "in_eic",
               "read_length", "units", "unit_shares", "reads_off", "reads_on", "in_store"]
TABLE_COLUMNS = ["law", "space", "layer", "run_type", "in_eic", "scope",
                 "gap_closed_vs_depth_scaling", "gap_closed_vs_identity", "attainable_floor",
                 "bootstrap_spread", "bootstrap_lo", "bootstrap_hi", "n_pairs", "n_experiments",
                 "deviance_law", "deviance_depth_scaling", "deviance_identity", "explains",
                 "beats_other_law"]


# ------------------------------------------------------------------ tables and stores
def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, columns):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, columns, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.6g}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return v


def read_length_of(s) -> int:
    """The shortest read length in a `read_length` cell ("36", "36,50", ...)."""
    nums = [int(float(x)) for x in str(s).replace(";", ",").split(",") if x.strip()]
    if not nums:
        raise ValueError(f"no read length in {s!r}")
    return min(nums)


def in_store(store, mid) -> bool:
    return os.path.isfile(os.path.join(store, mid, "provenance.json"))


def units_of(mid, rs_store, count_store=None) -> int:
    """Dedup units of a view: read files of one half in the read-start provenance, else the
    `pooled_from` list of the read-count provenance, else 1."""
    p = os.path.join(rs_store, mid, "provenance.json")
    if os.path.isfile(p):
        rf = json.load(open(p)).get("read_files") or {}
        v = rf.get("pr1")
        if v:
            return len(v) if isinstance(v, list) else 1
    if count_store:
        p = os.path.join(count_store, mid, "provenance.json")
        if os.path.isfile(p):
            pf = json.load(open(p)).get("pooled_from")
            if pf:
                return len(pf)
    return 1


def unit_shares(mid, units, by_id, arm_off="nodedup") -> str:
    """Read shares of the dedup units of a pooled view, from `depth_reads` of its `br<i>` rows of
    the dedup-off arm (else of the view's own arm); equal shares when a row is missing."""
    if units <= 1:
        return "1"
    exp, unit, layer, arm = mid.split(".", 3)
    for a in (arm_off, arm):
        d = [by_id.get(f"{exp}.br{i + 1}.{layer}.{a}", {}).get("depth_reads", "") for i in range(units)]
        if all(x not in ("", None) for x in d):
            v = np.array([float(x) for x in d])
            if v.sum() > 0:
                return ",".join(f"{x:.6g}" for x in v / v.sum())
    return ",".join(["%.6g" % (1.0 / units)] * units)


def shares_of(s) -> np.ndarray:
    v = np.array([float(x) for x in str(s).split(",") if x.strip()] or [1.0])
    return v / v.sum()


def dedup_rows(tables, rs_store, count_store=None):
    """One row per `*.nodedup` view with `in_eic` in train/val and its dedup-on parent."""
    ms = read_tsv(os.path.join(tables, "measurements.tsv"))
    by_id = {r["measurement_id"]: r for r in ms}
    out = []
    for r in ms:
        if r.get("arm_knob") != "nodedup" or r.get("in_eic") not in SPLITS:
            continue
        on_id = r["parent_measurement_id"]
        on = by_id.get(on_id, {})
        u = units_of(on_id, rs_store, count_store)
        out.append({
            "off_id": r["measurement_id"], "on_id": on_id, "experiment": r["experiment"],
            "assay": r["assay"], "layer": r["layer"],
            "run_type": on.get("run_type") or r["run_type"], "in_eic": r["in_eic"],
            "read_length": read_length_of(on.get("read_length") or r["read_length"]),
            "units": u, "unit_shares": unit_shares(on_id, u, by_id),
            "reads_off": r["depth_reads"], "reads_on": on.get("depth_reads", ""),
            "in_store": int(in_store(rs_store, r["measurement_id"]) and in_store(rs_store, on_id)),
        })
    out.sort(key=lambda x: (x["layer"], x["in_eic"], x["off_id"]))
    return out


def load_unit_rule(path) -> dict:
    return json.load(open(path)) if path else {"dedup_unit": "pooled"}


def K_for(units, rule) -> int:
    return int(caps.K_of(units)) if rule.get("dedup_unit") == "replicate" else RES


# ------------------------------------------------------------------ joint tables
def _chroms(*paths):
    import h5py
    keys = None
    for p in paths:
        with h5py.File(p, "r") as f:
            k = list(f.keys())
        keys = k if keys is None else [c for c in keys if c in set(k)]
    return keys


def _read(path, chrom, col=None):
    import h5py
    with h5py.File(path, "r") as f:
        d = f[chrom]
        v = d[:, col] if col is not None else d[...]
    return np.asarray(v, np.int64)


def top_masks(values_by_chrom, fraction=caps.TOP_FRACTION):
    """Per-chromosome masks of the genome-wide top `fraction` of bins by value: ceil(fraction x
    bins) bins with the largest value, ties broken by genome order (chromosomes in the given
    order), as `caps.joint_hist` breaks them on one concatenated track."""
    chroms = list(values_by_chrom)
    n = sum(values_by_chrom[c].size for c in chroms)
    n_top = int(math.ceil(fraction * n)) if n else 0
    vmax = max((int(values_by_chrom[c].max()) for c in chroms if values_by_chrom[c].size), default=0)
    hist = np.zeros(vmax + 1, np.int64)
    for c in chroms:
        hist += np.bincount(values_by_chrom[c], minlength=vmax + 1)
    above = np.cumsum(hist[::-1])[::-1]               # above[v] = bins with value >= v
    gt = np.append(above[1:], 0)                      # bins with value > v
    v_star = int(np.nonzero(above >= n_top)[0].max()) if n_top else vmax + 1
    take_eq = n_top - (int(gt[v_star]) if v_star <= vmax else 0)
    masks = {}
    for c in chroms:
        v = values_by_chrom[c]
        m = v > v_star
        eq = np.nonzero(v == v_star)[0]
        k = min(take_eq, eq.size) if n_top else 0
        m[eq[:k]] = True
        take_eq -= k
        masks[c] = m
    return masks


def fast_table(s, t):
    """`caps._table` for non-negative integer arrays, by int64 keys."""
    s = np.asarray(s, np.int64)
    t = np.asarray(t, np.int64)
    if s.size == 0:
        return {}
    base = int(t.max()) + 1
    keys, n = np.unique(s * base + t, return_counts=True)
    return {(int(k // base), int(k % base)): int(c) for k, c in zip(keys, n)}


def hist_to_arr(h):
    if not h:
        return np.zeros((0, 3))
    return np.array([[k[0], k[1], v] for k, v in h.items()], np.float64)


def arr_to_hist(a):
    return {(caps._key(r[0]), caps._key(r[1])): int(r[2]) for r in np.asarray(a)}


def joint_tables(src_path, tgt_path, col=None):
    """{"genome", "top1pct"} joint tables of one strand (col 0/1 of starts.h5) or of the overlap
    counts (col None), whole genome; the top 1% chosen by the source."""
    chroms = _chroms(src_path, tgt_path)
    src = {c: _read(src_path, c, col) for c in chroms}
    masks = top_masks(src)
    g, t1 = [], []
    for c in chroms:
        t = _read(tgt_path, c, col)
        if t.shape != src[c].shape:
            raise ValueError(f"{c}: source {src[c].shape} and target {t.shape} differ")
        g.append(fast_table(src[c], t))
        t1.append(fast_table(src[c][masks[c]], t[masks[c]]))
    return {"genome": caps.merge_hists(g), "top1pct": caps.merge_hists(t1)}


def pair_stats(src_id, tgt_id, rs_store, count_store):
    """Joint tables of a (source, target) pair: start space per strand, count space."""
    out = {}
    s_rs = os.path.join(rs_store, src_id, "starts.h5")
    t_rs = os.path.join(rs_store, tgt_id, "starts.h5")
    for col, strand in ((0, "plus"), (1, "minus")):
        jh = joint_tables(s_rs, t_rs, col)
        for scope, h in jh.items():
            out[f"start_{strand}_{scope}"] = hist_to_arr(h)
    jh = joint_tables(os.path.join(count_store, src_id, "counts.h5"),
                      os.path.join(count_store, tgt_id, "counts.h5"))
    for scope, h in jh.items():
        out[f"count_{scope}"] = hist_to_arr(h)
    return out


def save_stats(path, tables: dict, meta: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **tables)


def load_stats(path) -> dict:
    """{"meta": dict, "start": {scope: hist (strands merged)}, "count": {scope: hist}, ...}."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = {"meta": meta, "start": {}, "count": {}}
    for scope in ("genome", "top1pct"):
        out["start"][scope] = caps.merge_hists(
            [arr_to_hist(z[f"start_{s}_{scope}"]) for s in ("plus", "minus")])
        out["count"][scope] = arr_to_hist(z[f"count_{scope}"])
    out["extra"] = {k: z[k] for k in z.files if k.startswith("extra_")}
    if "halves_count_genome" in z.files:
        out["halves"] = {"start": {}, "count": {}}
        for scope in ("genome", "top1pct"):
            out["halves"]["start"][scope] = caps.merge_hists(
                [arr_to_hist(z[f"halves_start_{s}_{scope}"]) for s in ("plus", "minus")])
            out["halves"]["count"][scope] = arr_to_hist(z[f"halves_count_{scope}"])
    return out


# ------------------------------------------------------------------ shared-weight law
def dm_distinct_shared(c, a, shares, P=RES):
    """Exact mean of the distinct kept reads when the units share P Dirichlet(a) positional weights
    and the c reads split between units by `shares`: sum_u thinned_dm_distinct(c, w_u, a, P)."""
    shares = np.asarray(shares, np.float64)
    return sum(caps.thinned_dm_distinct(c, float(w), a, P) for w in shares)


def draw_shared(m, a, shares, rng, P=RES):
    """Monte Carlo of `dm_distinct_shared`: one Dirichlet(a) over P positions per draw, shared by
    the units; reads split between units by `shares`; distinct kept per unit, summed."""
    m = np.asarray(m, np.int64)
    if np.isinf(a):
        p = np.full((m.size, P), 1.0 / P)
    else:
        g = rng.gamma(a, 1.0, size=(m.size, P))
        z = g.sum(1, keepdims=True)
        bad = z[:, 0] <= 0
        if bad.any():
            g[bad] = 0.0
            g[bad, rng.integers(0, P, bad.sum())] = 1.0
            z = g.sum(1, keepdims=True)
        p = g / z
    split = rng.multinomial(m, np.asarray(shares, np.float64))       # [n, U]
    out = np.zeros(m.size, np.int64)
    for u in range(split.shape[1]):
        out += (rng.multinomial(split[:, u], p) > 0).sum(1)
    return out


def _fit_bounded(dev):
    return caps._fit(dev)


def fit_shape_shared(hists, shares_list):
    def dev(la):
        a = float(np.exp(la))
        return sum(caps.deviance_of(h, lambda s, w=w: dm_distinct_shared(s, a, w))
                   for h, w in zip(hists, shares_list))
    return _fit_bounded(dev)


def simulate_floor_shared(hist, a, shares, rng, n_rep=20, max_bins_per_value=500):
    """`caps.simulate_floor` for the shared-weight law."""
    s, _, w = caps._arrays(hist)
    if s.size == 0:
        return 0.0
    us = np.unique(s)
    ws = np.array([w[s == u].sum() for u in us])
    mu = dm_distinct_shared(us, a, shares)
    total = 0.0
    for u, wu, mu_u in zip(us, ws, mu):
        n = int(n_rep * min(wu, max_bins_per_value))
        y = draw_shared(np.full(n, int(u)), a, shares, rng)
        total += wu * float(unit_deviance(y, np.full(n, mu_u)).mean())
    return total


# ------------------------------------------------------------------ fit
def _boot_ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fit_groups(items, n_boot=N_BOOT_FIT, seed=0, shared=True):
    """Per (assay, run_type) and per (assay, "all") and ("all", "all"): `a` (caps.fit_shape),
    `a_shared` (shared-weight law) and `kappa` (caps.fit_kappa) on the given items, each with a
    2.5-97.5% bootstrap range over experiments. `items`: dicts with assay, run_type, experiment,
    K, shares, read_length, start (genome hist), count (genome hist)."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for it in items:
        groups[(it["assay"], it["run_type"])].append(it)
        groups[(it["assay"], "all")].append(it)
        groups[("all", "all")].append(it)
    rows = []
    for (assay, rt), its in sorted(groups.items()):
        def fits(sel, want_shared):
            a = caps.fit_shape([x["start"] for x in sel], [x["K"] for x in sel])
            k = caps.fit_kappa([x["count"] for x in sel], [x["read_length"] for x in sel])
            sh = (fit_shape_shared([x["start"] for x in sel], [x["shares"] for x in sel])
                  if want_shared else float("nan"))
            return a, sh, k
        a, a_sh, kap = fits(its, shared)
        exps = sorted({x["experiment"] for x in its})
        boots = []
        for _ in range(n_boot if len(exps) > 1 else 0):
            pick = rng.choice(exps, len(exps), replace=True)
            sel = [x for e in pick for x in its if x["experiment"] == e]
            boots.append(fits(sel, shared))
        b = np.array(boots) if boots else np.full((0, 3), np.nan)
        lo_hi = [_boot_ci(b[:, j]) if b.size else (float("nan"),) * 2 for j in range(3)]
        rows.append({"assay": assay, "run_type": rt, "n_pairs": len(its), "n_experiments": len(exps),
                     "a": a, "a_lo": lo_hi[0][0], "a_hi": lo_hi[0][1],
                     "a_shared": a_sh, "a_shared_lo": lo_hi[1][0], "a_shared_hi": lo_hi[1][1],
                     "kappa": kap, "kappa_lo": lo_hi[2][0], "kappa_hi": lo_hi[2][1]})
    return rows


FIT_COLUMNS = ["assay", "run_type", "n_pairs", "n_experiments", "a", "a_lo", "a_hi", "a_shared",
               "a_shared_lo", "a_shared_hi", "kappa", "kappa_lo", "kappa_hi"]


def lookup_fit(fit_rows, assay, run_type):
    idx = {(r["assay"], r["run_type"]): r for r in fit_rows}
    for key in ((assay, run_type), (assay, "all"), ("all", "all")):
        if key in idx:
            r = idx[key]
            return {k: float(r[k]) for k in ("a", "a_shared", "kappa")}, f"{key[0]}/{key[1]}"
    raise KeyError(f"no fit for {assay}/{run_type}")


def _items(stats_paths, rows_by_on, key="on_id"):
    items = []
    for p in stats_paths:
        st = load_stats(p)
        m = st["meta"]
        r = rows_by_on[m["target"]]
        items.append({"assay": r["assay"], "run_type": r["run_type"], "experiment": r["experiment"],
                      "in_eic": r["in_eic"], "layer": r.get("layer", ""), "id": m["target"],
                      "K": int(m["K"]), "shares": shares_of(m.get("unit_shares", "1"))
                      if m.get("dedup_unit") == "replicate" else np.array([1.0]),
                      "q": float(m.get("q", 1.0)), "read_length": int(r["read_length"]),
                      "start": st["start"]["genome"], "count": st["count"]["genome"], "stats": st})
    return items


# ------------------------------------------------------------------ count-space floor
def simulate_count_floor(src_starts_h5, tgt_starts_h5, src_counts_h5, a, K, q=1.0, n_rep=1,
                         rng=None, sub=None):
    """Attainable floor of the count-space law: dedup-on starts drawn from the start-space law
    (m ~ Binomial(source starts, q) reads on K positions, Dirichlet(a)), pushed through the target's
    span kernel (attribute `kernel` [2, 17] of its starts.h5), against the same push of the law's
    mean. Poisson deviance summed over the genome and over the top 1% of bins by source overlap
    count. Returns {"genome", "top1pct"}. ``sub = (q_source, q_target, share, prior)`` replaces the
    thinning by the subsampled law (``caps.subsampled_dm_distinct``, ``caps.draw_subsampled_distinct``;
    testbed pairs)."""
    import h5py
    from t396 import deconv
    rng = np.random.default_rng(0) if rng is None else rng
    with h5py.File(tgt_starts_h5, "r") as f:
        kern = np.asarray(f.attrs["kernel"], np.float64)
    chroms = _chroms(src_starts_h5, src_counts_h5)
    cnt = {c: _read(src_counts_h5, c) for c in chroms}
    masks = top_masks(cnt)
    tot = {"genome": 0.0, "top1pct": 0.0}
    for c in chroms:
        s = _read(src_starts_h5, c)                         # [n, 2]
        n = min(s.shape[0], cnt[c].size)
        s = s[:n]
        if sub is None:
            ey = np.stack([caps.thinned_dm_distinct(s[:, j], q, a, K) for j in (0, 1)], 1)
        else:
            ey = np.stack([caps.subsampled_dm_distinct(s[:, j], sub[0], sub[1], a, K, sub[2],
                                                       sub[3]) for j in (0, 1)], 1)
        mean = deconv.forward(ey[:, 0], kern[0]) + deconv.forward(ey[:, 1], kern[1])
        for _ in range(n_rep):
            y = np.zeros_like(s)
            for j in (0, 1):
                # with a completion prior an empty source bin has a positive mean: draw it too
                nz = np.nonzero(s[:, j] if sub is None else ey[:, j] > 0)[0]
                for i0 in range(0, nz.size, SIM_CHUNK):
                    idx = nz[i0:i0 + SIM_CHUNK]
                    m = s[idx, j]
                    if sub is not None:
                        y[idx, j] = caps.draw_subsampled_distinct(m, sub[0], sub[1], a, K, rng,
                                                                  sub[2], sub[3])
                        continue
                    if q != 1.0:
                        m = rng.binomial(m, q)
                    y[idx, j] = caps._draw_distinct(m, a, K, rng)
            sim = deconv.forward(y[:, 0], kern[0]) + deconv.forward(y[:, 1], kern[1])
            d = unit_deviance(sim, mean)
            tot["genome"] += float(d.sum()) / n_rep
            tot["top1pct"] += float(d[masks[c][:n]].sum()) / n_rep
    return tot


# ------------------------------------------------------------------ score
def _law_fns(K, a, a_sh, shares, kappa, L, q, ratio_start, ratio_count):
    start = {
        "identity": lambda s: np.asarray(s, np.float64),
        "depth_scaling": lambda s: ratio_start * np.asarray(s, np.float64),
        "dm_distinct": lambda s: caps.thinned_dm_distinct(s, q, a, K),
        "occupancy": lambda s: caps.thinned_dm_distinct(s, q, np.inf, K),
        "start_cap": lambda s: caps.start_cap(q * np.asarray(s, np.float64), a, K),
        "dm_distinct_shared": lambda s: dm_distinct_shared(s, a_sh, shares)
        if q == 1.0 else caps.thinned_dm_distinct(s, q, a_sh, K),
    }
    count = {
        "identity": lambda s: np.asarray(s, np.float64),
        "depth_scaling": lambda s: ratio_count * np.asarray(s, np.float64),
        "count_saturation": lambda s: caps.count_saturation(q * np.asarray(s, np.float64), kappa, L),
    }
    return start, count


def score_pair(it, params, floor_count=None, rng=None, n_rep=20, start_laws=None):
    """Long rows (law, space, scope, D_law, D_identity, D_depth, D_floor) of one pair.
    `params`: {"a", "a_shared", "kappa"} (and optional extra start laws {name: a})."""
    rng = np.random.default_rng(0) if rng is None else rng
    st = it["stats"]
    K, q, L, shares = it["K"], it["q"], it["read_length"], it["shares"]
    a, a_sh, kap = params["a"], params.get("a_shared", float("nan")), params["kappa"]
    r_s = caps.depth_ratio(st["start"]["genome"])
    r_c = caps.depth_ratio(st["count"]["genome"])
    start, count = _law_fns(K, a, a_sh, shares, kap, L, q, r_s, r_c)
    for name, a_x in (params.get("extra") or {}).items():
        start[name] = (lambda s, ax=a_x: caps.thinned_dm_distinct(s, q, ax, K))
    want = start_laws or [k for k in start if k not in ("identity", "depth_scaling")]
    if not np.isfinite(a_sh):
        want = [w for w in want if w != "dm_distinct_shared"]
    rows = []
    for scope in ("genome", "top1pct"):
        h = st["start"][scope]
        d_id = caps.deviance_of(h, start["identity"])
        d_dp = caps.deviance_of(h, start["depth_scaling"])
        fl_main = caps.simulate_floor(h, a, K, q=q, rng=rng, n_rep=n_rep)
        for law in want:
            if law == "occupancy":
                fl = caps.simulate_floor(h, np.inf, K, q=q, rng=rng, n_rep=n_rep)
            elif law == "dm_distinct_shared":
                fl = (simulate_floor_shared(h, a_sh, shares, rng, n_rep=n_rep) if q == 1.0
                      else caps.simulate_floor(h, a_sh, K, q=q, rng=rng, n_rep=n_rep))
            elif law in (params.get("extra") or {}):
                fl = caps.simulate_floor(h, params["extra"][law], K, q=q, rng=rng, n_rep=n_rep)
            else:                                     # dm_distinct, start_cap
                fl = fl_main
            rows.append({"law": law, "space": "start", "scope": scope,
                         "D_law": caps.deviance_of(h, start[law]), "D_identity": d_id,
                         "D_depth": d_dp, "D_floor": fl})
        h = st["count"][scope]
        rows.append({"law": COUNT_LAW, "space": "count", "scope": scope,
                     "D_law": caps.deviance_of(h, count[COUNT_LAW]),
                     "D_identity": caps.deviance_of(h, count["identity"]),
                     "D_depth": caps.deviance_of(h, count["depth_scaling"]),
                     "D_floor": float(floor_count[scope]) if floor_count else float("nan")})
    return rows


def _gap(D):
    with np.errstate(divide="ignore", invalid="ignore"):
        g_dp = (D[..., 2] - D[..., 0]) / (D[..., 2] - D[..., 3])
        g_id = (D[..., 1] - D[..., 0]) / (D[..., 1] - D[..., 3])
    return g_dp, g_id


def aggregate(pair_rows, group_cols, n_boot=N_BOOT_SCORE, seed=0,
              primary=PRIMARY_START_LAW, other=COUNT_LAW, all_cols=("in_eic",)):
    """table.tsv rows: deviances summed over the pairs of a group, gaps closed from the sums, the
    95% bootstrap interval over experiments of the gap vs depth scaling, gate (d)."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in pair_rows:
        for g in _group_keys(r, group_cols, all_cols):
            groups[g + (r["scope"],)].append(r)
    out = []
    boots_by = {}
    for key, rs in sorted(groups.items()):
        by_law = defaultdict(list)
        for r in rs:
            by_law[(r["law"], r["space"])].append(r)
        exps = sorted({r["experiment"] for r in rs})
        picks = [rng.integers(0, len(exps), len(exps)) for _ in range(n_boot)]
        for (law, space), lr in sorted(by_law.items()):
            per_exp = defaultdict(lambda: np.zeros(4))
            for r in lr:
                per_exp[r["experiment"]] += [r["D_law"], r["D_identity"], r["D_depth"], r["D_floor"]]
            E = np.array([per_exp.get(e, np.zeros(4)) for e in exps])
            tot = E.sum(0)
            g_dp, g_id = _gap(tot)
            bs = np.array([_gap(E[p].sum(0))[0] for p in picks])
            boots_by[key + (law,)] = bs
            lo, hi = _boot_ci(bs)
            row = dict(zip(group_cols, key[:-1]))
            row.update({"law": law, "space": space, "scope": key[-1],
                        "gap_closed_vs_depth_scaling": float(g_dp),
                        "gap_closed_vs_identity": float(g_id), "attainable_floor": float(tot[3]),
                        "bootstrap_lo": lo, "bootstrap_hi": hi,
                        "bootstrap_spread": hi - lo if np.isfinite(lo) else float("nan"),
                        "n_pairs": len(lr), "n_experiments": len({r["experiment"] for r in lr}),
                        "deviance_law": float(tot[0]), "deviance_depth_scaling": float(tot[2]),
                        "deviance_identity": float(tot[1]),
                        "explains": ("" if not np.isfinite(g_dp) else bool(g_dp >= GATE_GAP)),
                        "beats_other_law": ""})
            out.append(row)
    for row in out:                                   # gate (d): primary start law vs count law
        if row["law"] not in (primary, other):
            continue
        key = tuple(row[c] for c in group_cols) + (row["scope"],)
        b1, b2 = boots_by.get(key + (primary,)), boots_by.get(key + (other,))
        if b1 is None or b2 is None:
            continue
        diff = b1 - b2
        lo, hi = _boot_ci(diff)
        if not np.isfinite(lo):
            continue
        win = primary if lo > 0 else other if hi < 0 else None
        row["beats_other_law"] = (row["law"] == win)
    return out


def _group_keys(r, cols, all_cols=("in_eic",)):
    """The pair's own key plus, per column of `all_cols` in `cols`, the key with that column
    pooled as `all` (default: `in_eic`, pooled train + val)."""
    k = tuple(r[c] for c in cols)
    keys = [k]
    for c in all_cols:
        if c in cols:
            i = cols.index(c)
            keys.append(k[:i] + ("all",) + k[i + 1:])
    return keys


# ------------------------------------------------------------------ testbed pairs (iteration 2)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITER1_FIT = os.path.join(REPO, "cruxvault", "results", "t396", "dedup_cap", "per_assay.tsv")
COVARIATES = {"dedup": ("nodedup", None), "run_type": ("se", "pe")}   # arm_knob, arm_value
SHAPES = ("iteration1", "loto")
TB_PRIMARY = "subsampled_dm_distinct"
TB_UNSUB = "dm_distinct_unsubsampled"             # the same law with q_source = q_target = 1
Q_BOUNDS = (1e-3, 1.0)
TB_ROW_COLUMNS = ["source_id", "target_id", "covariate", "experiment", "assay", "layer",
                  "run_type", "in_eic", "read_length", "units", "lines_source", "lines_target",
                  "lines_per_fragment_source", "lines_per_fragment_target", "share",
                  "total_source", "total_target", "q_source", "q_target", "q_origin", "in_store"]
TB_FIT_COLUMNS = ["experiment", "assay", "covariate", "n_pairs_fit", "a_loto", "kappa_loto",
                  "a_iteration1", "kappa_iteration1", "iteration1_group", "q_source", "q_target",
                  "q_origin", "share", "K", "prior_alpha_start", "prior_beta_start",
                  "prior_alpha_count", "prior_beta_count"]
TB_GROUP = ["covariate", "shape", "assay_group"]
TB_TABLE_COLUMNS = (TB_GROUP + ["law", "space", "scope", "gap_closed_vs_depth_scaling",
                                "gap_closed_vs_identity", "attainable_floor", "halves_floor",
                                "bootstrap_spread", "bootstrap_lo", "bootstrap_hi", "n_pairs",
                                "n_experiments", "n_q_fitted", "deviance_law",
                                "deviance_depth_scaling", "deviance_identity", "explains",
                                "beats_other_law"])
TB_PAIR_COLUMNS = ["pair", "source", "experiment", "assay", "assay_group", "covariate", "shape",
                   "q_source", "q_target", "q_origin", "share", "a", "kappa", "law", "space",
                   "scope", "D_law", "D_identity", "D_depth", "D_floor", "D_halves"]


def is_testbed_rows(path) -> bool:
    """True when a rows file has the testbed columns (header `source_id`)."""
    if not path or not os.path.isfile(path):
        return False
    with open(path) as f:
        return "source_id" in f.readline().rstrip("\n").split("\t")


def read_totals(path) -> dict:
    """{measurement_id: lines before the subsample, or None} from a totals tsv."""
    out = {}
    for r in read_tsv(path) if path else []:
        v = (r.get("lines_before_subsample") or "").strip()
        out[r["measurement_id"]] = float(v) if v not in ("", "NA", "nan") else None
    return out


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _lpf(r) -> float:
    """Lines per fragment of a measurements row: depth_reads / depth_fragments rounded to an
    integer (1 single-end, 2 paired-end), else 1."""
    d, f = _num(r.get("depth_reads")), _num(r.get("depth_fragments"))
    return float(max(round(d / f), 1)) if d and f else 1.0


def _rate(lines, total, mid):
    if total is None or not lines:
        return None
    if total < lines:
        raise ValueError(f"{mid}: total {total:g} below the stored lines {lines:g}")
    return lines / total


def testbed_rows(tables, rs_store, count_store=None, covariate="dedup", totals=None):
    """One row per controlled pair of the covariate: the testbed row with the covariate's
    `arm_knob` (and `arm_value`) is the source, its `parent_measurement_id` the target."""
    knob, value = COVARIATES[covariate]
    ms = read_tsv(os.path.join(tables, "measurements.tsv"))
    by_id = {r["measurement_id"]: r for r in ms}
    totals = totals or {}
    out = []
    for r in ms:
        if r.get("layer") != "testbed" or r.get("arm_knob") != knob:
            continue
        if value is not None and r.get("arm_value") != value:
            continue
        sid, tid = r["measurement_id"], r["parent_measurement_id"]
        t = by_id.get(tid, {})
        ls, lt = _num(r.get("depth_reads")), _num(t.get("depth_reads"))
        tot_s, tot_t = totals.get(sid), totals.get(tid)
        qs, qt = _rate(ls, tot_s, sid), _rate(lt, tot_t, tid)
        lpf_s, lpf_t = _lpf(r), _lpf(t)
        out.append({
            "source_id": sid, "target_id": tid, "covariate": covariate,
            "experiment": r["experiment"], "assay": r["assay"], "layer": r["layer"],
            "run_type": t.get("run_type") or r["run_type"], "in_eic": r.get("in_eic", ""),
            "read_length": read_length_of(t.get("read_length") or r["read_length"]),
            "units": units_of(tid, rs_store, count_store),
            "lines_source": "" if ls is None else f"{ls:.0f}",
            "lines_target": "" if lt is None else f"{lt:.0f}",
            "lines_per_fragment_source": f"{lpf_s:g}", "lines_per_fragment_target": f"{lpf_t:g}",
            "share": f"{min(lpf_t / lpf_s, 1.0):.6g}",
            "total_source": "NA" if tot_s is None else f"{tot_s:.0f}",
            "total_target": "NA" if tot_t is None else f"{tot_t:.0f}",
            "q_source": "" if qs is None else f"{qs:.8g}",
            "q_target": "" if qt is None else f"{qt:.8g}",
            "q_origin": "measured" if qs is not None and qt is not None else "fitted",
            "in_store": int(all(in_store(st, x) for st in (rs_store, count_store) if st
                                for x in (sid, tid))),
        })
    out.sort(key=lambda x: x["source_id"])
    return out


def testbed_rows_of(path, covariate) -> dict:
    """{target id: row} of the covariate's rows in a testbed rows file."""
    return {r["target_id"]: r for r in read_tsv(path) if r["covariate"] == covariate}


def halves_tables(src_path, pr1_path, pr2_path, col=None):
    """{"genome", "top1pct"} joint tables of (target pr1, target pr2); the top 1% is chosen by the
    pair's source, as in `joint_tables`."""
    chroms = _chroms(src_path, pr1_path, pr2_path)
    src = {c: _read(src_path, c, col) for c in chroms}
    masks = top_masks(src)
    g, t1 = [], []
    for c in chroms:
        a, b = _read(pr1_path, c, col), _read(pr2_path, c, col)
        g.append(fast_table(a, b))
        t1.append(fast_table(a[masks[c]], b[masks[c]]))
    return {"genome": caps.merge_hists(g), "top1pct": caps.merge_hists(t1)}


def testbed_stats(row, rs_store, count_store, rule):
    """(tables, meta) of one testbed pair: the joint tables of `pair_stats` plus the target's
    half-vs-half tables when its halves are in both stores."""
    sid, tid = row["source_id"], row["target_id"]
    tables = pair_stats(sid, tid, rs_store, count_store)
    h = {"rs": [os.path.join(rs_store, tid, f"starts_pr{i}.h5") for i in (1, 2)],
         "ct": [os.path.join(count_store, tid, f"pr{i}.h5") for i in (1, 2)]}
    has_halves = all(os.path.isfile(p) for p in h["rs"] + h["ct"])
    if has_halves:
        for col, strand in ((0, "plus"), (1, "minus")):
            jh = halves_tables(os.path.join(rs_store, sid, "starts.h5"), *h["rs"], col=col)
            for scope, t in jh.items():
                tables[f"halves_start_{strand}_{scope}"] = hist_to_arr(t)
        jh = halves_tables(os.path.join(count_store, sid, "counts.h5"), *h["ct"])
        for scope, t in jh.items():
            tables[f"halves_count_{scope}"] = hist_to_arr(t)
    units = units_of(tid, rs_store, count_store)
    meta = {"source": sid, "target": tid, "covariate": row["covariate"], "layer": "testbed",
            "units": units, "K": K_for(units, rule), "q": 1.0, "dedup_unit": rule.get("dedup_unit"),
            "q_source": _num(row["q_source"]), "q_target": _num(row["q_target"]),
            "q_origin": row["q_origin"], "share": float(row["share"]), "halves": has_halves,
            "rs_store": rs_store, "count_store": count_store}
    return tables, meta


def testbed_items(stats_paths, rows_by_target):
    items = []
    for p in stats_paths:
        st = load_stats(p)
        m = st["meta"]
        r = rows_by_target.get(m["target"])
        if r is None or r["source_id"] != m["source"]:
            continue                                  # a stats file of the other covariate
        items.append({"id": m["target"], "source": m["source"], "experiment": r["experiment"],
                      "assay": r["assay"], "run_type": r["run_type"],
                      "assay_group": "DNase-seq" if r["assay"] == "DNase-seq" else "histone",
                      "covariate": r["covariate"], "K": int(m["K"]),
                      "read_length": int(r["read_length"]), "share": float(r["share"]),
                      "q_source": _num(r["q_source"]), "q_target": _num(r["q_target"]),
                      "q_origin": r["q_origin"], "start": st["start"]["genome"],
                      "count": st["count"]["genome"], "stats": st,
                      "nb_start": _nb_of(st["start"]["genome"]),
                      "nb_count": _nb_of(st["count"]["genome"])})
    return items


def _nb_of(hist):
    """(alpha, beta at q_source = 1) of the completion prior from the source marginal of a joint
    table; at rate q the prior is (alpha, q beta)."""
    s, _, w = caps._arrays(hist)
    if s.size == 0:
        return 0.0, 0.0
    v = np.unique(s)
    wv = np.array([w[s == x].sum() for x in v])
    return caps.fit_completion_prior(v, wv, 1.0)


def _prior(nb, q_source):
    return (float(nb[0]), float(q_source) * float(nb[1]))


def fit_rates(hist, K, share, q_source, q_target, a, nb=(0.0, 0.0)):
    """(q_source, q_target) with each missing rate fitted on the pair's start table at shape `a`
    (bounded search on log q in [1e-3, 1]); both missing: one rate shared by both. `nb`: the
    completion prior at q_source = 1 (`_nb_of`)."""
    if q_source is not None and q_target is not None:
        return q_source, q_target

    def law(q):
        qs = q if q_source is None else q_source
        qt = q if q_target is None else q_target
        return qs, qt

    def dev(lq):
        qs, qt = law(float(np.exp(lq)))
        return caps.deviance_of(hist, lambda s: caps.subsampled_dm_distinct(
            s, qs, qt, a, K, share, _prior(nb, qs)))
    r = optimize.minimize_scalar(dev, bounds=tuple(np.log(Q_BOUNDS)), method="bounded",
                                 options={"xatol": 1e-4})
    return law(float(np.exp(r.x)))


def testbed_fit(items, iter1_fit):
    """Per track: the iteration-1 shape (lookup by assay and run type), the rates (measured, or
    fitted at the iteration-1 shape) and the leave-one-track-out `a` and `kappa` of the subsampled
    laws, fitted on the other tracks' genome tables."""
    for it in items:
        p, grp = lookup_fit(iter1_fit, it["assay"], it["run_type"])
        it["iteration1"], it["iteration1_group"] = p, grp
        it["q_source"], it["q_target"] = fit_rates(it["start"], it["K"], it["share"],
                                                   it["q_source"], it["q_target"], p["a"],
                                                   it["nb_start"])
        it["prior_start"] = _prior(it["nb_start"], it["q_source"])
        it["prior_count"] = _prior(it["nb_count"], it["q_source"])
    rows = []
    for it in sorted(items, key=lambda x: x["experiment"]):
        rest = [x for x in items if x["experiment"] != it["experiment"]]
        if rest:
            args = ([x["q_source"] for x in rest], [x["q_target"] for x in rest],
                    [x["share"] for x in rest])
            a = caps.fit_shape_subsampled([x["start"] for x in rest], [x["K"] for x in rest], *args,
                                          prior=[x["prior_start"] for x in rest])
            k = caps.fit_kappa_subsampled([x["count"] for x in rest],
                                          [x["read_length"] for x in rest], *args,
                                          prior=[x["prior_count"] for x in rest])
        else:
            a = k = float("nan")
        rows.append({"experiment": it["experiment"], "assay": it["assay"],
                     "covariate": it["covariate"], "n_pairs_fit": len(rest), "a_loto": a,
                     "kappa_loto": k, "a_iteration1": it["iteration1"]["a"],
                     "kappa_iteration1": it["iteration1"]["kappa"],
                     "iteration1_group": it["iteration1_group"], "q_source": it["q_source"],
                     "q_target": it["q_target"], "q_origin": it["q_origin"], "share": it["share"],
                     "K": it["K"], "prior_alpha_start": it["prior_start"][0],
                     "prior_beta_start": it["prior_start"][1],
                     "prior_alpha_count": it["prior_count"][0],
                     "prior_beta_count": it["prior_count"][1]})
    return rows


def _halves_dev(halves, scope):
    """Deviance of pr2 under pr1 scaled by the genome-wide pr2 / pr1 ratio."""
    if not halves:
        return float("nan")
    r = caps.depth_ratio(halves["genome"])
    return caps.deviance_of(halves[scope], lambda s: r * np.asarray(s, np.float64))


def testbed_score_pair(it, fit_row, shape, floor_count=None, rng=None, n_rep=20):
    """Long rows of one pair and one shape: the subsampled start law, the same law at q = 1, the
    subsampled count law, each with the identity and depth-scaling deviances, the attainable floor
    and the half-vs-half floor."""
    rng = np.random.default_rng(0) if rng is None else rng
    st = it["stats"]
    K, L, sh = it["K"], it["read_length"], float(fit_row["share"])
    qs, qt = float(fit_row["q_source"]), float(fit_row["q_target"])
    a, kap = float(fit_row[f"a_{shape}"]), float(fit_row[f"kappa_{shape}"])
    pr_s = (float(fit_row["prior_alpha_start"]), float(fit_row["prior_beta_start"]))
    pr_c = (float(fit_row["prior_alpha_count"]), float(fit_row["prior_beta_count"]))
    r_s = caps.depth_ratio(st["start"]["genome"])
    r_c = caps.depth_ratio(st["count"]["genome"])
    start = {TB_PRIMARY: lambda s: caps.subsampled_dm_distinct(s, qs, qt, a, K, sh, pr_s),
             TB_UNSUB: lambda s: caps.subsampled_dm_distinct(s, 1.0, 1.0, a, K, sh)}
    halves = st.get("halves") or {}
    common = {"pair": it["id"], "source": it["source"], "experiment": it["experiment"],
              "assay": it["assay"], "assay_group": it["assay_group"], "covariate": it["covariate"],
              "shape": shape, "q_source": qs, "q_target": qt, "q_origin": it["q_origin"],
              "share": sh, "a": a, "kappa": kap}
    rows = []
    for scope in ("genome", "top1pct"):
        h = st["start"][scope]
        d_id = caps.deviance_of(h, lambda s: np.asarray(s, np.float64))
        d_dp = caps.deviance_of(h, lambda s: r_s * np.asarray(s, np.float64))
        d_h = _halves_dev(halves.get("start"), scope)
        for law, fn in start.items():
            q2 = (qs, qt, pr_s) if law == TB_PRIMARY else (1.0, 1.0, (0.0, 0.0))
            fl = caps.simulate_subsampled_floor(h, a, K, q2[0], q2[1], sh, rng=rng, n_rep=n_rep,
                                                prior=q2[2])
            rows.append(dict(common, law=law, space="start", scope=scope,
                             D_law=caps.deviance_of(h, fn), D_identity=d_id, D_depth=d_dp,
                             D_floor=fl, D_halves=d_h))
        h = st["count"][scope]
        rows.append(dict(common, law=COUNT_LAW, space="count", scope=scope,
                         D_law=caps.deviance_of(h, lambda s: caps.subsampled_count_saturation(
                             s, qs, qt, kap, L, sh, pr_c)),
                         D_identity=caps.deviance_of(h, lambda s: np.asarray(s, np.float64)),
                         D_depth=caps.deviance_of(h, lambda s: r_c * np.asarray(s, np.float64)),
                         D_floor=float(floor_count[scope]) if floor_count else float("nan"),
                         D_halves=_halves_dev(halves.get("count"), scope)))
    return rows


def testbed_table(pair_rows, n_boot=N_BOOT_SCORE, seed=0):
    """`aggregate` over (covariate, shape, assay group; assay group also pooled as `all`), plus
    the summed half-vs-half floor and the number of pairs with fitted rates."""
    table = aggregate(pair_rows, TB_GROUP, n_boot=n_boot, seed=seed, primary=TB_PRIMARY,
                      other=COUNT_LAW, all_cols=("assay_group",))
    halves, fitted = defaultdict(float), defaultdict(set)
    for r in pair_rows:
        for g in _group_keys(r, TB_GROUP, ("assay_group",)):
            key = g + (r["law"], r["space"], r["scope"])
            halves[key] += r["D_halves"]
            if r["q_origin"] != "measured":
                fitted[key].add(r["pair"])
    for row in table:
        key = tuple(row[c] for c in TB_GROUP) + (row["law"], row["space"], row["scope"])
        row["halves_floor"] = float(halves[key])
        row["n_q_fitted"] = len(fitted[key])
    return table


def testbed_score(stats_paths, rows_path, fit_path, covariate, out, floor_paths=None,
                  n_boot=N_BOOT_SCORE, n_rep=20, seed=0, tool="dedup_cap"):
    items = testbed_items(stats_paths, testbed_rows_of(rows_path, covariate))
    fit = {r["experiment"]: r for r in read_tsv(fit_path) if r["covariate"] == covariate}
    floors = load_floor_counts(floor_paths)
    rng = np.random.default_rng(seed)
    pair_rows = []
    for it in items:
        for shape in SHAPES:
            pair_rows += testbed_score_pair(it, fit[it["experiment"]], shape, floors.get(it["id"]),
                                            rng=rng, n_rep=n_rep)
    os.makedirs(out, exist_ok=True)
    write_tsv(os.path.join(out, "pairs.tsv"), pair_rows, TB_PAIR_COLUMNS)
    table = testbed_table(pair_rows, n_boot=n_boot, seed=seed)
    write_tsv(os.path.join(out, "table.tsv"), table, TB_TABLE_COLUMNS)
    print(f"[{tool}] score testbed: {len(items)} pairs, {len(pair_rows)} pair rows, "
          f"{len(table)} table rows -> {out}")
    return table


def testbed_floor_count(pair, rows_path, fit_path, covariate, rule, rs_store, count_store,
                        n_rep, seed, out, tool="dedup_cap"):
    src, tgt = pair
    r = testbed_rows_of(rows_path, covariate)[tgt]
    f = {x["experiment"]: x for x in read_tsv(fit_path) if x["covariate"] == covariate}[
        r["experiment"]]
    K = K_for(units_of(tgt, rs_store, count_store), rule)
    sub = (float(f["q_source"]), float(f["q_target"]), float(f["share"]),
           (float(f["prior_alpha_start"]), float(f["prior_beta_start"])))
    a = float(f["a_loto"])
    fl = simulate_count_floor(os.path.join(rs_store, src, "starts.h5"),
                              os.path.join(rs_store, tgt, "starts.h5"),
                              os.path.join(count_store, src, "counts.h5"), a, K, n_rep=n_rep,
                              rng=np.random.default_rng(seed), sub=sub)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, genome=fl["genome"], top1pct=fl["top1pct"],
             meta=np.array(json.dumps({"source": src, "target": tgt, "a": a, "K": K,
                                       "q_source": sub[0], "q_target": sub[1], "share": sub[2],
                                       "prior": list(sub[3])})))
    print(f"[{tool}] floor-count testbed {tgt}: {fl}")


# ------------------------------------------------------------------ CLI
def cmd_rows(a, covariate="dedup", tool="dedup_cap"):
    if a.layer == "testbed":
        rows = testbed_rows(a.tables, a.rs_store, a.count_store, covariate, read_totals(a.totals))
        write_tsv(a.out, rows, TB_ROW_COLUMNS)
        n_fit = sum(r["q_origin"] != "measured" for r in rows)
        print(f"[{tool}] rows testbed: {len(rows)} pairs ({n_fit} q_fitted) -> {a.out}")
        return rows
    if a.totals:
        raise SystemExit("--totals needs --layer testbed")
    rows = dedup_rows(a.tables, a.rs_store, a.count_store)
    write_tsv(a.out, rows, ROW_COLUMNS)
    print(f"[dedup_cap] rows: {len(rows)} pairs -> {a.out}")


def cmd_stats_testbed(a, covariate, tool):
    rule = load_unit_rule(a.unit_rule)
    src, tgt = a.pair
    row = testbed_rows_of(a.rows, covariate).get(tgt)
    if row is None or row["source_id"] != src:
        raise SystemExit(f"no {covariate} testbed row {src} -> {tgt} in {a.rows}")
    tables, meta = testbed_stats(row, a.rs_store, a.count_store, rule)
    save_stats(a.out, tables, meta)
    print(f"[{tool}] stats testbed {src} -> {tgt}: K={meta['K']} q=({meta['q_source']}, "
          f"{meta['q_target']}) halves={meta['halves']} -> {a.out}")


def cmd_stats(a):
    if is_testbed_rows(a.rows):
        return cmd_stats_testbed(a, "dedup", "dedup_cap")
    rule = load_unit_rule(a.unit_rule)
    off, on = a.pair
    units = units_of(on, a.rs_store, a.count_store)
    shares = "1"
    if a.rows:
        r = {x["on_id"]: x for x in read_tsv(a.rows)}.get(on)
        shares = r["unit_shares"] if r else shares
    tables = pair_stats(off, on, a.rs_store, a.count_store)
    meta = {"source": off, "target": on, "units": units, "K": K_for(units, rule), "q": 1.0,
            "dedup_unit": rule.get("dedup_unit"), "unit_shares": shares,
            "rs_store": a.rs_store, "count_store": a.count_store}
    save_stats(a.out, tables, meta)
    print(f"[dedup_cap] stats {off} -> {on}: K={meta['K']} -> {a.out}")


def cmd_fit_testbed(a, covariate, tool):
    items = testbed_items(a.stats, testbed_rows_of(a.rows, covariate))
    rows = testbed_fit(items, read_tsv(a.iteration1_fit))
    write_tsv(a.out, rows, TB_FIT_COLUMNS)
    print(f"[{tool}] fit testbed: {len(items)} pairs, leave-one-track-out -> {a.out}")
    return rows


def cmd_fit(a):
    if is_testbed_rows(a.rows):
        return cmd_fit_testbed(a, "dedup", "dedup_cap")
    rows = {r["on_id"]: r for r in read_tsv(a.rows)}
    items = [it for it in _items(a.stats, rows) if it["in_eic"] == "train"]
    fit = fit_groups(items, n_boot=a.n_boot, seed=a.seed)
    write_tsv(a.out, fit, FIT_COLUMNS)
    print(f"[dedup_cap] fit: {len(items)} train pairs, {len(fit)} groups -> {a.out}")


def cmd_floor_count(a):
    if is_testbed_rows(a.rows):
        return testbed_floor_count(a.pair, a.rows, a.fit, "dedup", load_unit_rule(a.unit_rule),
                                   a.rs_store, a.count_store, a.n_rep, a.seed, a.out)
    rule = load_unit_rule(a.unit_rule)
    off, on = a.pair
    r = {x["on_id"]: x for x in read_tsv(a.rows)}[on]
    p, _ = lookup_fit(read_tsv(a.fit), r["assay"], r["run_type"])
    K = K_for(units_of(on, a.rs_store, a.count_store), rule)
    fl = simulate_count_floor(os.path.join(a.rs_store, off, "starts.h5"),
                              os.path.join(a.rs_store, on, "starts.h5"),
                              os.path.join(a.count_store, off, "counts.h5"), p["a"], K,
                              n_rep=a.n_rep, rng=np.random.default_rng(a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, genome=fl["genome"], top1pct=fl["top1pct"],
             meta=np.array(json.dumps({"source": off, "target": on, "a": p["a"], "K": K})))
    print(f"[dedup_cap] floor-count {on}: {fl}")


def load_floor_counts(paths):
    out = {}
    for p in paths or []:
        z = np.load(p, allow_pickle=False)
        out[json.loads(str(z["meta"]))["target"]] = {"genome": float(z["genome"]),
                                                     "top1pct": float(z["top1pct"])}
    return out


def cmd_score(a):
    if is_testbed_rows(a.rows):
        return testbed_score(a.stats, a.rows, a.fit, "dedup", a.out, a.floor_count,
                             n_boot=a.n_boot, n_rep=a.n_rep, seed=a.seed)
    rows = {r["on_id"]: r for r in read_tsv(a.rows)}
    fit = read_tsv(a.fit)
    floors = load_floor_counts(a.floor_count)
    rng = np.random.default_rng(a.seed)
    pair_rows = []
    for it in _items(a.stats, rows):
        p, src = lookup_fit(fit, it["assay"], it["run_type"])
        for r in score_pair(it, p, floors.get(it["id"]), rng=rng, n_rep=a.n_rep):
            r.update({"pair": it["id"], "experiment": it["experiment"], "assay": it["assay"],
                      "layer": it["layer"], "run_type": it["run_type"], "in_eic": it["in_eic"],
                      "fit_group": src})
            pair_rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    write_tsv(os.path.join(a.out, "pairs.tsv"), pair_rows,
              ["pair", "experiment", "assay", "layer", "run_type", "in_eic", "fit_group", "law",
               "space", "scope", "D_law", "D_identity", "D_depth", "D_floor"])
    table = aggregate(pair_rows, ["layer", "run_type", "in_eic"], n_boot=a.n_boot, seed=a.seed)
    write_tsv(os.path.join(a.out, "table.tsv"), table, TABLE_COLUMNS)
    print(f"[dedup_cap] score: {len(pair_rows)} pair rows, {len(table)} table rows -> {a.out}")
    return table


def add_store_args(p):
    p.add_argument("--rs-store", default=RS_STORE)
    p.add_argument("--count-store", default=COUNT_STORE)


def add_testbed_args(sub):
    """--layer/--totals on `rows`, --iteration1-fit on `fit` (shared with sepe_cap)."""
    sub.choices["rows"].add_argument("--layer", choices=["testbed"],
                                     help="the controlled pairs of the testbed layer")
    sub.choices["rows"].add_argument("--totals", help="lines before the subsample per row")
    sub.choices["fit"].add_argument("--iteration1-fit", default=ITER1_FIT,
                                    help="per-assay shapes of iteration 1 (testbed rows only)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rows")
    p.add_argument("--tables", default=TABLES)
    add_store_args(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stats")
    p.add_argument("--pair", nargs=2, required=True, metavar=("OFF", "ON"))
    p.add_argument("--unit-rule", required=True)
    p.add_argument("--rows", help="rows.tsv, for the unit read shares")
    add_store_args(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--stats", nargs="+", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT_FIT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("floor-count")
    p.add_argument("--pair", nargs=2, required=True, metavar=("OFF", "ON"))
    p.add_argument("--unit-rule", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--n-rep", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    add_store_args(p)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--stats", nargs="+", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--floor-count", nargs="*")
    p.add_argument("--n-boot", type=int, default=N_BOOT_SCORE)
    p.add_argument("--n-rep", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    add_testbed_args(sub)
    a = ap.parse_args(argv)
    return {"rows": cmd_rows, "stats": cmd_stats, "fit": cmd_fit, "floor-count": cmd_floor_count,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
