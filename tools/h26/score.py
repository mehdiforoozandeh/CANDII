"""h26 steps 2 and 4 (Fir, CPU): the pre-registered checks of the covariate-swap probe.

Two stages, run in this order, so eligibility is frozen before any model output exists:

  eligibility   truth only. `effectfloor` on the 19 knob pairs, `thinningblank` on the 21 depth
                pairs (the same statistic), depthcontrol part (3), and the real full-product ratios.
                Writes eligibility.json. Run BEFORE infer.py is submitted.
  score         reads eligibility.json and the infer.py outputs. `swapwired`, `depthcontrol`
                parts (1)-(2), `gapleft`, `swapcorr`, and the reported-only statistics.

RERUN 1 applies the PI's post-hoc amendments of 2026-10-26 (h26 node, "Amendments after seeing
data"): (ii)(a) `effectfloor` and `thinningblank` drop the decile stratification; (ii)(b)
`depthcontrol` is part (2) AND part (3), part (1) reported only; 99 block shifts everywhere,
including the 25 bp reported-only `swapcorr` null. Everything else is as first pre-registered.

Every definition is the h26 node's `## Verifiables`, verbatim where it is a number:
  ratios        log2((arm + 1) / (base + 1)), median-centred per pair
  gate res.     1 kb = 40 bins of 25 bp; 25 bp, 200 bp and 5 kb reported, never gating
  effectfloor   r_split = Spearman(pr1 ratio, pr2 ratio) across 1 kb bins, each median-centred, NO
                deciles (amendment ii-a; the decile-averaged first-run statistic is reported only);
                eligible when r_split > 99th pct of 99 circular >= 1 Mb shifts of the pr2 ratio
                AND r_split >= 0.2
  gapleft       arm pr2 scored at 25 bp under P_base, P_arm, P_oracle = P_base x 2^R_hat (R_hat from
                pr1 only, 1 kb, median-centred within base-count deciles); 100 kb block bootstrap;
                met when CI_lo(G_oracle) > 0 and CI_hi(G_model / G_oracle) <= 0.25
  swapcorr      Spearman(model delta, real R) at 1 kb, both median-centred; met when rho <= 95th pct
                of 99 circular >= 1 Mb shifts of delta; max|delta| < 1e-4 -> rho = 0, "covariate unused"
  depthcontrol  met = (2) NLL(arm | told arm depth) < NLL(arm | told base depth) AND (3) |real
                log2(sum arm / sum base) - log2 frac| <= 0.05 (amendment ii-b); (1) |pred global
                delta - log2(arm/base depth)| <= 0.01 is reported only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import rankdata

TRACKS = ["C12M02", "C19M16", "C40M17", "C40M18", "C07M20", "C19M22", "C07M29"]
KNOBS = {"crop__36": ("crop36", "read_length"), "crop__50": ("crop50", "read_length"),
         "pe__pe": ("pe", "run_type")}
DEPTHS = {"depth__15M": "d15M", "depth__7.5M": "d7.5M", "depth__3.75M": "d3.75M"}
RES = {"25bp": 1, "200bp": 8, "1kb": 40, "5kb": 200}
GATE = "1kb"
BIN_BP = 25
SHIFT_MIN_BP = 1_000_000
N_SHIFT = 99
N_BOOT = 1000
BOOT_BLOCK_BINS = 100_000 // BIN_BP
R_FLOOR = 0.2
GAP_RATIO_MAX = 0.25
PRODUCTS = Path("/project/def-maxwl/mforooz/t112_cf/products")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pool(x: np.ndarray, k: int) -> np.ndarray:
    nb = len(x) // k
    return x[: nb * k].astype(np.float64).reshape(nb, k).sum(1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = rankdata(x)
    return r - r.mean()


def spearman_shift_null(a: np.ndarray, b: np.ndarray, sh: np.ndarray) -> list:
    """Spearman(roll(a, s), b) for every shift s. Ranks commute with a circular shift, so ranking
    once and rolling the ranks is the same number as `spearman(np.roll(a, s), b)`, without
    re-ranking 1.9 M bins 99 times at 25 bp."""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return [float("nan")] * len(sh)
    ra, rb = centred_ranks(a), centred_ranks(b)
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return [float((np.roll(ra, int(s)) * rb).sum() / den) for s in sh]


def deciles(x: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (r * 10) // len(x)


def lratio(arm: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.log2((arm + 1.0) / (base + 1.0))


def centre_in_groups(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    out = x.copy()
    for d in np.unique(g):
        m = g == d
        out[m] -= np.median(x[m])
    return out


def shifts(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    lo = math.ceil(SHIFT_MIN_BP / (BIN_BP * k))
    if n - 2 * lo < 1:
        return np.array([], int)
    return rng.integers(lo, n - lo, size=N_SHIFT)


def r_split_stat(r1: np.ndarray, r2: np.ndarray, g: np.ndarray) -> float:
    vals = [spearman(r1[g == d], r2[g == d]) for d in range(10)]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def effectfloor_one(job):
    """r_split and its block-shift null at every resolution, for one (arm, base) pair.

    Amendment (ii)(a), 2026-10-26: r_split is the Spearman across bins of the two median-centred
    half ratios, with NO decile stratification; the null is 99 circular >= 1 Mb shifts of the pr2
    ratio. The first run's decile-averaged statistic is kept as `r_split_decile_reported` (no null).
    """
    key, arm1, base1, arm2, base2, seed = job
    rng = np.random.default_rng(seed)
    out = {}
    for name, k in RES.items():
        A1, B1, A2, B2 = pool(arm1, k), pool(base1, k), pool(arm2, k), pool(base2, k)
        r1 = lratio(A1, B1)
        r1 -= np.median(r1)
        r2 = lratio(A2, B2)
        r2 -= np.median(r2)
        obs = spearman(r1, r2)
        sh = shifts(len(r1), k, rng)
        null = spearman_shift_null(r2, r1, sh)
        p99 = float(np.nanpercentile(null, 99)) if len(null) else float("nan")
        g = deciles((B1 + B2) / 2.0)
        obs_dec = r_split_stat(centre_in_groups(lratio(A1, B1), g), centre_in_groups(lratio(A2, B2), g), g)
        out[name] = {"r_split": obs, "null_p99": p99, "null_mean": float(np.nanmean(null)) if len(null) else None,
                     "n_shifts": int(len(sh)), "n_bins": int(len(r1)),
                     "eligible": bool(np.isfinite(obs) and obs > p99 and obs >= R_FLOOR),
                     "r_split_decile_reported": obs_dec}
    return key, out


def load_rows(path: str) -> dict:
    return {r["pid"]: r for r in csv.DictReader(open(path), delimiter="\t")}


def load_full(pid: str, rows: dict, cache: dict) -> np.ndarray:
    if pid not in cache:
        f = PRODUCTS / pid / "counts25.npz"
        got = md5(f)
        want = rows[pid]["counts25_md5"]
        if got != want:
            raise SystemExit(f"{f}: md5 {got} != MANIFEST {want}")
        with np.load(f) as z:
            cache[pid] = z["chr21"].astype(np.float64)
    return cache[pid]


def pairs_of(rows: dict):
    knob, depth = [], []
    for t in TRACKS:
        for suf, (prompt, field) in KNOBS.items():
            if f"{t}__{suf}" in rows:
                knob.append((t, suf, prompt, field))
        for suf, prompt in DEPTHS.items():
            if f"{t}__{suf}" in rows:
                depth.append((t, suf, prompt))
    return knob, depth


# ---------------------------------------------------------------------------
# stage 1 — eligibility, truth only
# ---------------------------------------------------------------------------

def stage_eligibility(a) -> None:
    rows = load_rows(a.cf_manifest)
    knob, depth = pairs_of(rows)
    halves = np.load(a.halves)
    hmeta = json.loads(Path(a.halves).with_suffix(".json").read_text())
    hmeta = {k: v for k, v in hmeta.items() if "missing" not in v}   # C12M02 has no crop arms
    bad = {k: v for k, v in hmeta.items() if not v.get("md5_ok")}
    if bad:
        raise SystemExit(f"half md5 mismatch or missing: {sorted(bad)}")
    native = {}
    for t in TRACKS:
        b = rows[f"{t}__base__base"]
        native[t] = {"read_length": int(b["read_length"]), "run_type": b["run_type"],
                     "depth": int(b["depth"])}
    dropped = []
    keep_knob = []
    for t, suf, prompt, field in knob:
        armv = rows[f"{t}__{suf}"][field]
        if str(armv) == str(native[t][field]):
            dropped.append({"track": t, "arm": suf, "reason": f"arm {field} {armv} equals native"})
        else:
            keep_knob.append((t, suf, prompt, field))
    jobs = []
    for i, (t, suf, *_rest) in enumerate(keep_knob + depth):
        g = lambda pid, h: halves[f"{pid}|{h}"].astype(np.float64)
        arm, base = f"{t}__{suf}", f"{t}__base__base"
        jobs.append((f"{t}|{suf}", g(arm, "pr1"), g(base, "pr1"), g(arm, "pr2"), g(base, "pr2"), 1000 + i))
    with Pool(a.workers) as p:
        res = dict(p.map(effectfloor_one, jobs))
    half_frac = {k: (v["n_lines"] / v["product_lines"]) for k, v in hmeta.items()}

    cache = {}
    real = {}
    for t, suf, *_r in keep_knob + depth:
        arm = load_full(f"{t}__{suf}", rows, cache)
        base = load_full(f"{t}__base__base", rows, cache)
        real[f"{t}|{suf}"] = {"sum_arm": float(arm.sum()), "sum_base": float(base.sum()),
                              "log2_total_ratio": float(math.log2(arm.sum() / base.sum()))}
    dc3 = {}
    for t, suf, prompt in depth:
        frac = int(rows[f"{t}__{suf}"]["depth"]) / int(rows[f"{t}__base__base"]["depth"])
        r = real[f"{t}|{suf}"]["log2_total_ratio"]
        dc3[f"{t}|{suf}"] = {"frac": frac, "log2_frac": math.log2(frac), "real": r,
                             "abs_err": abs(r - math.log2(frac)), "met": abs(r - math.log2(frac)) <= 0.05}

    knob_keys = [f"{t}|{suf}" for t, suf, *_ in keep_knob]
    depth_keys = [f"{t}|{suf}" for t, suf, _ in depth]
    elig = {k: res[k][GATE]["eligible"] for k in knob_keys}
    rl_ok = any(elig[k] for k in knob_keys if "crop" in k)
    rt_ok = any(elig[k] for k in knob_keys if "pe__pe" in k)
    out = {
        "stage": "eligibility", "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "amendments": "PI 2026-10-26 (ii)(a): no decile stratification in effectfloor/thinningblank",
        "halves": a.halves, "halves_md5": md5(Path(a.halves)), "gate_resolution": GATE,
        "native": native, "dropped_pairs": dropped,
        "half_read_fraction": {"min": min(half_frac.values()), "max": max(half_frac.values())},
        "effectfloor": {k: res[k] for k in knob_keys},
        "eligible_pairs": [k for k in knob_keys if elig[k]],
        "effectfloor_met": bool(rl_ok and rt_ok),
        "effectfloor_by_knob": {"read_length": rl_ok, "run_type": rt_ok},
        "thinningblank": {k: res[k] for k in depth_keys},
        "thinningblank_met": not any(res[k][GATE]["eligible"] for k in depth_keys),
        "depthcontrol_part3": dc3, "depthcontrol_part3_met": all(v["met"] for v in dc3.values()),
        "real_totals": real,
    }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "eligibility.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("eligible_pairs", "effectfloor_met", "effectfloor_by_knob",
                                          "thinningblank_met", "depthcontrol_part3_met",
                                          "dropped_pairs", "half_read_fraction")}, indent=1))


# ---------------------------------------------------------------------------
# stage 2 — score, needs infer.py outputs
# ---------------------------------------------------------------------------

def nb_nll(y: np.ndarray, mu: np.ndarray, n: np.ndarray) -> np.ndarray:
    mu = np.maximum(mu, 1e-12)
    return -(gammaln(y + n) - gammaln(n) - gammaln(y + 1.0)
             + n * np.log(n / (n + mu)) + y * np.log(mu / (n + mu)))


def boot_ci(block_base, block_model, block_oracle, nbins, rng):
    """G_oracle, G_model and their ratio, with 95% CIs over 100 kb blocks."""
    nblk = len(block_base)
    go = (block_base - block_oracle)
    gm = (block_base - block_model)
    pt_o, pt_m = go.sum() / nbins, gm.sum() / nbins
    idx = rng.integers(0, nblk, size=(N_BOOT, nblk))
    bo, bm = go[idx].sum(1) / nbins, gm[idx].sum(1) / nbins
    ratio = bm / bo
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"G_oracle": float(pt_o), "G_oracle_ci": q(bo), "G_model": float(pt_m), "G_model_ci": q(bm),
            "ratio": float(pt_m / pt_o) if pt_o != 0 else float("nan"), "ratio_ci": q(ratio)}


def block_sums(x: np.ndarray) -> np.ndarray:
    nb = len(x) // BOOT_BLOCK_BINS
    head = x[: nb * BOOT_BLOCK_BINS].reshape(nb, BOOT_BLOCK_BINS).sum(1)
    tail = x[nb * BOOT_BLOCK_BINS:].sum()
    return np.append(head, tail) if len(x) % BOOT_BLOCK_BINS else head


def stage_score(a) -> None:
    rows = load_rows(a.cf_manifest)
    elig = json.loads((Path(a.out) / "eligibility.json").read_text())
    halves = np.load(a.halves)
    hmeta = json.loads(Path(a.halves).with_suffix(".json").read_text())
    knob, depth = pairs_of(rows)
    dropped = {(d["track"], d["arm"]) for d in elig["dropped_pairs"]}
    knob = [k for k in knob if (k[0], k[1]) not in dropped]
    seeds = sorted(int(p.stem.split("_s")[1]) for p in Path(a.preds).glob("C12M02_s*.npz"))
    cache = {}
    res = {"stage": "score", "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": seeds,
           "amendments": "PI 2026-10-26 (ii)(b): depthcontrol = part2 AND part3; 99 shifts at every resolution",
           "eligibility_written": elig["written"], "pairs": {}, "depthcontrol": {}, "swapwired": {}}
    preds_written = {}
    for seed in seeds:
        for t in TRACKS:
            f = Path(a.preds) / f"{t}_s{seed}.npz"
            preds_written[f.name] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime))
            P = np.load(f)
            M = json.loads(f.with_suffix(".json").read_text())
            mu = lambda p: P[f"{p}|mu"].astype(np.float64)
            nn = lambda p: P[f"{p}|n"].astype(np.float64)
            eta = lambda p: P[f"{p}|eta"].astype(np.float64)
            base_full = load_full(f"{t}__base__base", rows, cache)

            # swapwired — read off the tensors infer.py recorded
            for p, w in M["swapwired"].items():
                if p == "base":
                    continue
                want = {"crop36": "read_length", "crop50": "read_length", "pe": "run_type"}.get(p, "depth")
                rl = w["y_meta_target_col"][2]
                rt = w["y_meta_target_col"][3]
                ok = (w["diff_fields"] == [want] and w["diff_cols"] == [M["assay_col"]]
                      and w["x_meta_untouched"]
                      and M["train_read_lengths"]["min"] <= rl <= M["train_read_lengths"]["max"]
                      and rt in (0.0, 1.0))
                d_eta = eta(p) - eta("base")
                d_l2 = np.log2(mu(p)) - np.log2(mu("base"))
                res["swapwired"][f"{t}|{p}|s{seed}"] = {
                    "met": bool(ok), "diff_fields": w["diff_fields"], "diff_cols": w["diff_cols"],
                    "y_meta": w["y_meta_target_col"], "x_meta_untouched": w["x_meta_untouched"],
                    "read_length_seen_in_training": int(rl) in {int(k) for k in M["train_read_lengths"]["counts"]},
                    "eta_delta_mean": float(d_eta.mean()), "eta_delta_maxabs": float(np.abs(d_eta).max()),
                    "log2mu_delta_mean": float(d_l2.mean()), "log2mu_delta_maxabs": float(np.abs(d_l2).max()),
                }

            # depthcontrol (1)-(2)
            for t2, suf, prompt in depth:
                if t2 != t:
                    continue
                arm_full = load_full(f"{t}__{suf}", rows, cache)
                frac = int(rows[f"{t}__{suf}"]["depth"]) / int(rows[f"{t}__base__base"]["depth"])
                pred = math.log2(mu(prompt).sum()) - math.log2(mu("base").sum())
                nll_d = nb_nll(arm_full, mu(prompt), nn(prompt)).mean()
                nll_b = nb_nll(arm_full, mu("base"), nn("base")).mean()
                p3 = elig["depthcontrol_part3"][f"{t}|{suf}"]
                res["depthcontrol"][f"{t}|{suf}|s{seed}"] = {
                    "pred_global_delta": pred, "log2_frac": math.log2(frac),
                    "part1_err": abs(pred - math.log2(frac)), "part1_met": abs(pred - math.log2(frac)) <= 0.01,
                    "nll_told_arm": float(nll_d), "nll_told_base": float(nll_b),
                    "part2_met": bool(nll_d < nll_b), "part3_met": p3["met"], "part3_err": p3["abs_err"],
                    # amendment (ii)(b): part (1) reported only
                    "met": bool(nll_d < nll_b and p3["met"]),
                }

            # knob pairs: swapcorr, gapleft, global level
            for t2, suf, prompt, field in knob:
                if t2 != t:
                    continue
                key = f"{t}|{suf}"
                arm_full = load_full(f"{t}__{suf}", rows, cache)
                rng = np.random.default_rng(7 + seed)
                rec = {"eligible": key in elig["eligible_pairs"], "knob": field, "prompt": prompt}
                rec["global_pred"] = math.log2(mu(prompt).sum()) - math.log2(mu("base").sum())
                rec["global_real"] = elig["real_totals"][key]["log2_total_ratio"]
                sc = {}
                for name, k in RES.items():
                    D = np.log2(pool(mu(prompt), k)) - np.log2(pool(mu("base"), k))
                    D -= np.median(D)
                    R = lratio(pool(arm_full, k), pool(base_full, k))
                    R -= np.median(R)
                    unused = float(np.abs(D).max()) < 1e-4
                    if unused:
                        rho, p95 = 0.0, float("nan")
                    else:
                        rho = spearman(D, R)
                        sh = shifts(len(D), k, rng)    # 99 everywhere, 25 bp included (PI 2026-10-26)
                        null = spearman_shift_null(D, R, sh)
                        p95 = float(np.nanpercentile(null, 95))
                    sc[name] = {"rho": rho, "null_p95": p95, "max_abs_delta": float(np.abs(D).max()),
                                "covariate_unused": unused,
                                "met": bool(unused or (np.isfinite(p95) and rho <= p95))}
                rec["swapcorr"] = sc

                # gapleft: arm pr2 at 25 bp; R_hat from pr1 only
                y = halves[f"{t}__{suf}|pr2"].astype(np.float64)
                hm = hmeta[f"{t}__{suf}|pr2"]
                f2 = hm["n_lines"] / hm["product_lines"]
                a1 = halves[f"{t}__{suf}|pr1"].astype(np.float64)
                b1 = halves[f"{t}__base__base|pr1"].astype(np.float64)
                gl = {}
                for name, k in RES.items():
                    nb = len(y) // k
                    A1, B1 = pool(a1, k), pool(b1, k)
                    Rh = centre_in_groups(lratio(A1, B1), deciles(B1))
                    nbin = nb * k
                    yy = y[:nbin]
                    ma, na = mu("base")[:nbin] * f2, nn("base")[:nbin]
                    mb, nbb = mu(prompt)[:nbin] * f2, nn(prompt)[:nbin]
                    mo = ma * np.repeat(2.0 ** Rh, k)
                    l_base = block_sums(nb_nll(yy, ma, na))
                    l_arm = block_sums(nb_nll(yy, mb, nbb))
                    l_or = block_sums(nb_nll(yy, mo, na))
                    ci = boot_ci(l_base, l_arm, l_or, nbin, np.random.default_rng(11 + seed))
                    ci["met"] = bool(ci["G_oracle_ci"][0] > 0 and ci["ratio_ci"][1] <= GAP_RATIO_MAX)
                    ci["half_fraction"] = f2
                    gl[name] = ci
                rec["gapleft"] = gl
                res["pairs"].setdefault(key, {})[f"s{seed}"] = rec
                print(f"[h26] s{seed} {key} elig={rec['eligible']} rho={sc[GATE]['rho']:.4f} "
                      f"p95={sc[GATE]['null_p95']:.4f} G_or={gl[GATE]['G_oracle']:.5f} "
                      f"G_m={gl[GATE]['G_model']:.6f} ratio_hi={gl[GATE]['ratio_ci'][1]:.4f}", flush=True)

    # roll-ups, as pre-registered
    res["preds_written"] = preds_written
    res["swapwired_met"] = all(v["met"] for v in res["swapwired"].values())
    res["depthcontrol_met_by_seed"] = {f"s{s}": all(v["met"] for k, v in res["depthcontrol"].items()
                                                    if k.endswith(f"|s{s}")) for s in seeds}
    res["depthcontrol_met"] = all(res["depthcontrol_met_by_seed"].values())
    el = elig["eligible_pairs"]
    for chk in ("gapleft", "swapcorr"):
        per = {}
        for key in el:
            per[key] = all(res["pairs"][key][f"s{s}"][chk][GATE]["met"] for s in seeds)
        res[f"{chk}_by_pair"] = per
        res[f"{chk}_met"] = (all(per.values()) if per else None)
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1, default=float))

    # flat table for the report
    with open(Path(a.out) / "pairs.tsv", "w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pair", "seed", "eligible", "r_split_1kb", "null_p99_1kb", "global_pred", "global_real",
                    "rho_1kb", "rho_null_p95_1kb", "max_abs_delta_1kb", "swapcorr_met",
                    "G_oracle", "G_oracle_lo", "G_model", "G_model_lo", "G_model_hi", "ratio", "ratio_hi",
                    "gapleft_met"])
        for key, by in res["pairs"].items():
            ef = elig["effectfloor"][key][GATE]
            for s, r in by.items():
                sc, gl = r["swapcorr"][GATE], r["gapleft"][GATE]
                w.writerow([key, s, r["eligible"], f"{ef['r_split']:.4f}", f"{ef['null_p99']:.4f}",
                            f"{r['global_pred']:.5f}", f"{r['global_real']:.5f}", f"{sc['rho']:.4f}",
                            f"{sc['null_p95']:.4f}", f"{sc['max_abs_delta']:.3g}", sc["met"],
                            f"{gl['G_oracle']:.6f}", f"{gl['G_oracle_ci'][0]:.6f}", f"{gl['G_model']:.3g}",
                            f"{gl['G_model_ci'][0]:.3g}", f"{gl['G_model_ci'][1]:.3g}", f"{gl['ratio']:.4f}",
                            f"{gl['ratio_ci'][1]:.4f}", gl["met"]])
    print(json.dumps({k: res[k] for k in ("swapwired_met", "depthcontrol_met", "depthcontrol_met_by_seed",
                                          "gapleft_met", "gapleft_by_pair", "swapcorr_met",
                                          "swapcorr_by_pair")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eligibility", "score"])
    ap.add_argument("--halves", required=True)
    ap.add_argument("--cf-manifest", default=str(PRODUCTS / "MANIFEST.tsv"))
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (stage_eligibility if a.stage == "eligibility" else stage_score)(a)


if __name__ == "__main__":
    main()
