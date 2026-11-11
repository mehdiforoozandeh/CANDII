"""Crossed mixed-model cross-check of the onion layers (D12, supplementary order-free decomposition; chunk MM).

Per assay, a linear mixed model of the band-level Haar coefficients of log1p counts:

    y[o, k] = mu[k] + beta * log2_depth[o] + sum_f u_f[level_f(o), k] + e[o, k]

o is one observation (one track), k one Haar coefficient of the band over the eval-set regions. mu[k]
is a fixed effect per coefficient (the genomic signal shared by every track), beta one fixed slope on
log2 depth, and u_f an independent random effect per factor level and coefficient, with variance s_f.
The factors are cell, experiment, replicate (biological replicate, or the donor when the table names
one) and library; the residual e is the variance between the two pseudoreplicate halves of one library
when the halves exist, else between observations of one library. The factors may be crossed or nested:
the fit takes any set of incidence matrices. The coefficients k are treated as independent replicates
of the model (working independence), so the fit needs only the sufficient statistics Y Y' and Y 1.

Estimation: restricted maximum likelihood (REML), maximized with `scipy.optimize.minimize` (L-BFGS-B
over the log variance components; mu and beta are profiled out in closed form). numpy and scipy only.
variance_fraction = s_f / (sum of every s + s_residual). The interval is a 95% cluster bootstrap over
cells (cells drawn with replacement, each draw a new set of levels): estimate +- t(0.975, G - 1) x
bootstrap SE x sqrt(G / (G - 1)), G = number of cells, clipped to [0, 1] (the form of layers.py).

Estimability: a factor is `estimable = 0` when it has fewer than 2 levels, when its levels partition
the observations exactly as an earlier factor does (e.g. one experiment per cell, as EIC has), or when
each level holds one observation (it is then the residual). A non-estimable factor is left out of the
fit; its variance sits in the row its note names. Lab is never fitted: EIC has one lab per cell and
assay, so lab is not separable from cell; its row says so (estimable = 0).

Observations: the measurements of the assay with `in_eic` train or val (never blind), `arm_knob` empty
or `base`, of one layer (`--layer`, default: uniform when the assay has it, else native), at the finest
unit that each experiment has (libraries `br<i>tr<j>`, else replicates `br<i>`, else `pooled`), so that
no two observations share reads. A measurement with `has_pseudoreps = 1` and an `h5:` counts path enters
as its two halves `pr1.h5` and `pr2.h5` (same folder as counts.h5), each at half its depth.

CLI: python tools/onion_eval/mixed_model.py run --tables <dir> --evalset <json> --band fine|peak|domain|broad
     --out <mixed_model.tsv> [--layer uniform|native|testbed] [--n-boot 200] [--seed 0]
Columns: assay, band, factor, variance_fraction, ci_low, ci_high, n_levels, estimable (0|1), note.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from eic_decomp import errdecomp  # noqa: E402

J = errdecomp.J
BANDS = {name.split(" ")[0]: list(idx) for name, idx in errdecomp.BANDS.items()}
FACTORS = ["cell", "experiment", "replicate", "library"]
COLUMNS = ["assay", "band", "factor", "variance_fraction", "ci_low", "ci_high", "n_levels",
           "estimable", "note"]
METHOD = "REML (scipy L-BFGS-B, log variance components; mu and beta profiled)"


# ---------------------------------------------------------------------------------------------------
# REML on sufficient statistics

def _partition(labels):
    """Canonical partition of observations: tuple of first-seen group index per observation."""
    seen = {}
    return tuple(seen.setdefault(x, len(seen)) for x in labels)


def estimability(labels, order=FACTORS):
    """labels: {factor: list of level labels per observation}. Returns {factor: (estimable, note)}."""
    n = len(next(iter(labels.values())))
    obs = tuple(range(n))
    out, kept = {}, []
    for f in order:
        p = _partition(labels[f])
        n_lev = len(set(labels[f]))
        if n_lev < 2:
            out[f] = (0, "one level; not estimable")
            continue
        same = [g for g, q in kept if q == p]
        if same:
            out[f] = (0, f"one {f} per {same[0]}; confounded with {same[0]} (variance in its row)")
            continue
        if p == obs:
            out[f] = (0, f"one observation per {f}; confounded with residual (variance in its row)")
            continue
        out[f] = (1, "")
        kept.append((f, p))
    return out


def _incidence(labels):
    lev = {x: i for i, x in enumerate(dict.fromkeys(labels))}
    Z = np.zeros((len(labels), len(lev)))
    Z[np.arange(len(labels)), [lev[x] for x in labels]] = 1.0
    return Z


def _neg2_reml(theta, ZZ, S, ysum, d, K):
    s = np.exp(theta)
    n = S.shape[0]
    V = s[-1] * np.eye(n)
    for sf, G in zip(s[:-1], ZZ):
        V = V + sf * G
    try:
        C = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e300, np.zeros_like(theta)
    logdet = 2.0 * np.sum(np.log(np.diag(C)))
    Ci = np.linalg.inv(C)
    P = Ci.T @ Ci
    one = np.ones(n)
    P1 = P @ one
    a = one @ P1
    Q = P - np.outer(P1, P1) / a
    val = K * logdet + K * math.log(a) + float(np.sum(Q * S))
    # gradient: d(-2 l)/d s_f = tr(Pi (I_K x G_f)) - sum_k e_k' G_f e_k, e_k = Q (y_k - beta d)
    Sr = S
    if d is not None:
        Qd = Q @ d
        c = d @ Qd
        qy = float(Qd @ ysum)
        val += math.log(c) - qy ** 2 / (K * c)
        beta = qy / (K * c)
        Sr = S - beta * (np.outer(ysum, d) + np.outer(d, ysum)) + K * beta ** 2 * np.outer(d, d)
    QSQ = Q @ Sr @ Q
    grad = np.empty(len(s))
    for i, G in enumerate(list(ZZ) + [None]):
        if G is None:
            tr = K * np.trace(Q) - (float(Qd @ Qd) / c if d is not None else 0.0)
            quad = np.trace(QSQ)
        else:
            tr = K * float(np.sum(Q * G)) - (float(Qd @ G @ Qd) / c if d is not None else 0.0)
            quad = float(np.sum(QSQ * G))
        grad[i] = s[i] * (tr - quad)
    return val, grad


def reml_fit(S, ysum, K, Zs, depth=None, theta0=None):
    """Fit the variance components by REML.

    S: (n, n) sum over coefficients of y y'; ysum: (n,) sum over coefficients; K: number of
    coefficients; Zs: list of (n, q_f) incidence matrices of the fitted factors; depth: (n,) log2
    depth or None (no slope; also None when depth does not vary). Returns (sigma [s_f..., s_resid],
    beta, theta)."""
    S = np.asarray(S, float)
    ysum = np.asarray(ysum, float)
    n = S.shape[0]
    d = None
    if depth is not None:
        d = np.asarray(depth, float)
        if np.ptp(d) < 1e-9:
            d = None
    ZZ = [Z @ Z.T for Z in Zs]
    if theta0 is None:
        Qc = np.eye(n) - 1.0 / n
        tot = max(float(np.sum(Qc * S)) / (K * max(n - 1, 1)), 1e-12)
        theta0 = np.full(len(Zs) + 1, math.log(tot / (len(Zs) + 1)))
    def fun(th):  # scaled by 1 / (n K) so that L-BFGS-B's tolerances fit any data size
        v, g = _neg2_reml(th, ZZ, S, ysum, d, K)
        return v / (n * K), g / (n * K)

    res = optimize.minimize(fun, theta0, jac=True, method="L-BFGS-B",
                            bounds=[(-30.0, 30.0)] * len(theta0),
                            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    theta = res.x
    beta = float("nan")
    if d is not None:
        s = np.exp(theta)
        V = s[-1] * np.eye(n) + sum(sf * G for sf, G in zip(s[:-1], ZZ))
        P = np.linalg.inv(V)
        P1 = P @ np.ones(n)
        Q = P - np.outer(P1, P1) / P1.sum()
        Qd = Q @ d
        beta = float(Qd @ ysum) / (K * float(d @ Qd))
    return np.exp(theta), beta, theta


def decompose(S, ysum, K, labels, depth=None, n_boot=200, seed=0):
    """Variance fractions with a cluster-bootstrap interval over cells.

    labels: {factor: list of level labels per observation} for every factor of FACTORS (cell first).
    Returns (rows, beta): rows is a list of dicts with factor, variance_fraction, ci_low, ci_high,
    n_levels, estimable, note (factors of FACTORS, then residual)."""
    est = estimability(labels)
    fit_f = [f for f in FACTORS if est[f][0]]
    n = S.shape[0]
    Zs = [_incidence(labels[f]) for f in fit_f]
    sig, beta, theta = reml_fit(S, ysum, K, Zs, depth)
    frac = sig / sig.sum()

    cells = list(dict.fromkeys(labels["cell"]))
    G = len(cells)
    by_cell = {c: [i for i, x in enumerate(labels["cell"]) if x == c] for c in cells}
    rng = np.random.default_rng(seed)
    boots = []
    if G >= 2 and n_boot > 0:
        for _ in range(n_boot):
            draw = rng.integers(0, G, size=G)
            idx, lab = [], {f: [] for f in fit_f}
            for j, ci in enumerate(draw):
                for i in by_cell[cells[ci]]:
                    idx.append(i)
                    for f in fit_f:
                        lab[f].append((j, labels[f][i]))
            idx = np.array(idx)
            Zb = [_incidence(lab[f]) for f in fit_f]
            db = None if depth is None else np.asarray(depth, float)[idx]
            sb, _, _ = reml_fit(S[np.ix_(idx, idx)], ysum[idx], K, Zb, db, theta0=theta)
            boots.append(sb / sb.sum())
    boots = np.array(boots) if boots else None

    def interval(k):
        if boots is None:
            return float("nan"), float("nan")
        se = float(np.std(boots[:, k], ddof=1)) * math.sqrt(G / (G - 1))
        t = float(stats.t.ppf(0.975, G - 1))
        return max(0.0, frac[k] - t * se), min(1.0, frac[k] + t * se)

    rows, k = [], 0
    for f in FACTORS:
        n_lev = len(set(labels[f]))
        if est[f][0]:
            lo, hi = interval(k)
            rows.append(dict(factor=f, variance_fraction=float(frac[k]), ci_low=lo, ci_high=hi,
                             n_levels=n_lev, estimable=1, note=""))
            k += 1
        else:
            rows.append(dict(factor=f, variance_fraction=float("nan"), ci_low=float("nan"),
                             ci_high=float("nan"), n_levels=n_lev, estimable=0, note=est[f][1]))
    lo, hi = interval(k)
    rows.append(dict(factor="residual", variance_fraction=float(frac[k]), ci_low=lo, ci_high=hi,
                     n_levels=n, estimable=1, note=f"{METHOD}; {len(boots) if boots is not None else 0}"
                     f" cluster-bootstrap draws over {G} cells"))
    return rows, beta


# ---------------------------------------------------------------------------------------------------
# Reading the tables

def _read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _unit_rank(unit):
    if unit == "pooled":
        return 0
    if "tr" in unit and "ln" not in unit:
        return 2
    if unit.startswith("br") and "tr" not in unit:
        return 1
    return -1  # lanes and anything else: not used


def select_observations(meas_rows, assay, layer=None):
    """Observations of one assay: list of dicts with counts_path, depth, cell, experiment, replicate,
    library, lab (see the module docstring for the rule)."""
    rows = [m for m in meas_rows if m["assay"] == assay and m.get("in_eic") in ("train", "val")
            and m.get("arm_knob", "") in ("", "base") and _unit_rank(m.get("unit", "")) >= 0]
    if layer is None:
        layers = {m.get("layer") for m in rows}
        layer = "uniform" if "uniform" in layers else "native"
    rows = [m for m in rows if m.get("layer") == layer]
    best = {}
    for m in rows:
        best[m["experiment"]] = max(best.get(m["experiment"], -1), _unit_rank(m["unit"]))
    obs = []
    for m in rows:
        if _unit_rank(m["unit"]) != best[m["experiment"]]:
            continue
        exp, unit = m["experiment"], m["unit"]
        rep_key = m.get("donor") or (m.get("bio_rep") or unit.split("tr")[0])
        rep = f"{exp}|{rep_key}"
        lib = f"{exp}|{unit}"
        depth = float(m.get("depth_reads") or m.get("depth_fragments") or "nan")
        base = dict(cell=m["cell"], experiment=exp, replicate=rep, library=lib, lab=m.get("lab", ""),
                    measurement_id=m["measurement_id"])
        kind, _, path = m["counts_path"].partition(":")
        if m.get("has_pseudoreps") == "1" and kind == "h5":
            for h in ("pr1", "pr2"):
                obs.append(dict(base, counts_path="h5:" + str(Path(path).with_name(f"{h}.h5")),
                                depth=depth / 2.0, half=h))
        else:
            obs.append(dict(base, counts_path=m["counts_path"], depth=depth, half=""))
    return obs, layer


def sufficient_stats(obs, regions, band, reader=None):
    """Accumulates S = Y Y' and ysum = Y 1 over the regions for one Haar band. Returns (S, ysum, K)."""
    if reader is None:
        from onion_eval.layers import CountsReader  # noqa: E402
        reader = CountsReader()
    idx = BANDS[band]
    L = 2 ** J
    n = len(obs)
    S, ysum, K = np.zeros((n, n)), np.zeros(n), 0
    for chrom, start in regions:
        X = np.stack([np.log1p(np.asarray(reader.read(o["counts_path"], chrom, start, start + L),
                                          np.float64)) for o in obs])
        c = errdecomp.haar(X, J)
        Y = np.concatenate([c[i] for i in idx], axis=1)
        S += Y @ Y.T
        ysum += Y.sum(axis=1)
        K += Y.shape[1]
    return S, ysum, K


def lab_row(obs):
    labs = [o["lab"] for o in obs]
    n_lev = len(set(labs))
    cell_lab = {}
    for o in obs:
        cell_lab.setdefault(o["cell"], set()).add(o["lab"])
    if all(len(v) == 1 for v in cell_lab.values()):
        note = "EIC has one lab per cell and assay: lab is not separable from cell (its variance is in cell)"
    else:
        note = "some cell has 2 or more labs, but lab is not fitted in this version"
    return dict(factor="lab", variance_fraction=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                n_levels=n_lev, estimable=0, note=note)


def run(tables, evalset, band, out, layer=None, n_boot=200, seed=0, reader=None):
    if band not in BANDS:
        raise ValueError(f"band {band!r} not in {sorted(BANDS)}")
    tables = Path(tables)
    ev = json.loads(Path(evalset).read_text())
    regions = [(c, int(s)) for c, s in ev["regions"]]
    meas = _read_tsv(tables / "measurements.tsv")
    assays = sorted({m["assay"] for m in meas if m.get("assay_class") != "control"})
    rows = []
    for assay in assays:
        obs, used_layer = select_observations(meas, assay, layer)
        if len(obs) < 3 or len({o["cell"] for o in obs}) < 2:
            rows.append(dict(assay=assay, band=band, factor="all", variance_fraction=float("nan"),
                             ci_low=float("nan"), ci_high=float("nan"), n_levels=len(obs), estimable=0,
                             note=f"too few observations ({len(obs)}) or cells to fit"))
            continue
        S, ysum, K = sufficient_stats(obs, regions, band, reader)
        labels = {f: [o[f] for o in obs] for f in FACTORS}
        depth = np.log2(np.array([o["depth"] for o in obs]))
        if not np.all(np.isfinite(depth)):
            depth = None
        fr, beta = decompose(S, ysum, K, labels, depth, n_boot=n_boot, seed=seed)
        halves = sum(1 for o in obs if o["half"])
        for r in fr:
            if r["factor"] == "residual":
                r["note"] += (f"; layer {used_layer}; {halves} of {len(obs)} observations are"
                              " pseudoreplicate halves")
            rows.append(dict(assay=assay, band=band, **r))
        rows.append(dict(assay=assay, band=band, **lab_row(obs)))
        rows.append(dict(assay=assay, band=band, factor="log2_depth", variance_fraction=float("nan"),
                         ci_low=float("nan"), ci_high=float("nan"), n_levels=len(obs), estimable=int(
                             depth is not None and np.ptp(depth) > 1e-9),
                         note=f"fixed slope beta = {beta:.6g} per log2 depth (not a variance component)"))
    write_tsv(rows, out)
    return rows


def write_tsv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([("" if isinstance(r[c], float) and math.isnan(r[c]) else
                         (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c])) for c in COLUMNS])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tables", required=True)
    r.add_argument("--evalset", required=True)
    r.add_argument("--band", required=True, choices=list(BANDS))
    r.add_argument("--out", required=True)
    r.add_argument("--layer", default=None, choices=["uniform", "native", "testbed"])
    r.add_argument("--n-boot", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = run(args.tables, args.evalset, args.band, args.out, args.layer, args.n_boot, args.seed)
    n_ne = sum(1 for x in rows if x["estimable"] == 0 and x["factor"] in FACTORS)
    print(f"mixed_model: {len(rows)} rows -> {args.out}; {n_ne} non-estimable factor rows; {METHOD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
