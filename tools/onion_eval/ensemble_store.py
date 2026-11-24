"""Seed ensemble and NB dispersion recalibration over prediction stores (task t370, chunk ens).

PI request: seed ensembling and post-training dispersion recalibration on held-out data, with no
training. The scorer is not changed: this module writes a new prediction store (the format of
`predict_store.py`) whose (n, mu) are the combined model's, and the metric phase
(`score.py --phase metrics`, job script slurm/onion_eval/score.sh) scores it as it scores any store.

Moment matching (`moment_match`). K seeds predict NB(n_k, mu_k) per bin; the combined NB has
    mean     m = mean_k mu_k
    variance V = mean_k (mu_k + mu_k^2 / n_k) + var_k(mu_k)        (var with ddof 0)
and n = m^2 / (V - m), i.e. V - m = mean_k(mu_k^2 / n_k) + var_k(mu_k); p = n / (n + m)
(`p_from_mu`, the scorer's own conversion). The predicted control mean `ctl_mu` is the mean of the
members' `ctl_mu`.

Recalibration (`recalibrate`). One factor c per assay multiplies the NB dispersion alpha = 1 / n:
n' = n / c, so the variance becomes mu + c mu^2 / n (c > 1 widens). The factor is applied to the
imputation items of the regimes it was fitted on (`FIT_REGIMES`), never to a pair, a control item
or another regime.

Fit (`fit`, command `fit`). The factor of an assay minimizes the pooled PIT deviation of that
assay (`metrics.pit_deviation(...)["pooled"]`, the quantity score.py's uncertainty term averages
over assays) on calibration regions: candidate blocks of the tables' genome (evalset.py's own
`candidate_blocks`) that are in no `regions` or `fit_regions` of the given eval sets, drawn with a
seeded generator. The fit items are the eval set's imputation targets of `FIT_REGIMES` (the same
cells and inputs as the scored items, on other genomic regions). The scored regions are never read.

    python tools/onion_eval/ensemble_store.py fit --runs <seed0 dir> <seed1 dir> <seed2 dir>
        --evalset <json> [--exclude-evalset <json> ...] --out <dir> [--n-regions 32]
        [--region-seed 370] [--infer-batch 8] [--max-bins 300000]
    -> <out>/calibration_evalset.json, <out>/factors.json

    python tools/onion_eval/ensemble_store.py combine --members <store> [<store> ...]
        --out <new store> [--factors <factors.json> --factor-set ensemble|seed0] [--record <json>]
    -> <new store>/{STORE.json, <sha1>.npz, PREDICTED_0of1.json, guardrail.json, COMBINED.json}

The combined STORE.json keeps the first member's identity (run, ckpt_sha256), so the metric phase
is called with `--run <first member's run dir>`; its `ensemble` key names every member. The
guardrail (copy-leak) record is the first member's.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

FIT_REGIMES = ("leave_one_assay_out",)
#: log2 of the dispersion factor: a coarse grid, then a fine grid around the coarse optimum
COARSE = np.arange(-4.0, 4.0 + 1e-9, 0.5)
FINE_HALF, FINE_STEP = 0.5, 1.0 / 32
#: the coarse grid grows by COARSE step while its optimum sits on an end, up to this |log2|
COARSE_CAP = 16.0
N_MAX = 1e30                  # n of a bin whose members agree exactly (V - m = 0)
COMBINED = "COMBINED.json"
#: STORE.json keys that every member must share (identity apart from the run and its checkpoint)
SHARED_KEYS = ("format", "run_tables_sha256", "evalset_sha256", "tables_sha256", "device",
               "infer_batch", "tf32", "amp", "region_bins", "n_requests_total", "requests_sha256")
#: switches a combined store does not carry (score.py refuses a store with them in normal scoring)
REFUSED_KEYS = ("foreign_covariates", "input_depth_at_scoring")


# ------------------------------------------------------------------------------------ formulas

def moment_match(ns, mus):
    """(n, mu) of the NB with the mixture's mean and variance; ns, mus: [K, ...]."""
    ns = np.asarray(ns, np.float64)
    mus = np.asarray(mus, np.float64)
    if ns.shape != mus.shape or ns.ndim < 1:
        raise ValueError("ns and mus must have one shape [K, ...]")
    m = mus.mean(0)
    excess = (mus ** 2 / ns).mean(0) + mus.var(0)            # V - m
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(excess > 0, m ** 2 / excess, N_MAX)
    return np.minimum(n, N_MAX), m


def recalibrate(n, c):
    """n of the NB whose dispersion 1 / n is c times the given one."""
    if not c > 0:
        raise ValueError(f"dispersion factor {c!r} must be positive")
    return np.asarray(n, np.float64) / float(c)


def pit_dev(n, mu, y) -> float:
    """Pooled PIT deviation, exactly as score.py's uncertainty term reads it per assay."""
    from onion_eval import metrics as Mx
    return float(Mx.pit_deviation(n, mu, y)["pooled"])


def fit_factor(n, mu, y) -> dict:
    """The dispersion factor c (n' = n / c) with the least pooled PIT deviation: a coarse log2 grid
    (COARSE, extended one step at a time while the optimum sits on an end, up to |log2| =
    COARSE_CAP), then a step of FINE_STEP within FINE_HALF of the coarse optimum. A tie goes to the
    factor nearest 1. Returns {factor, log2_factor, pit_at_1, pit_at_factor, at_bound}."""
    seen = {}

    def dev(l2):
        k = round(float(l2), 9)
        if k not in seen:
            seen[k] = pit_dev(recalibrate(n, 2.0 ** k), mu, y)
        return seen[k]

    def best(grid):
        return min(grid, key=lambda l2: (dev(l2), abs(l2)))
    step = float(COARSE[1] - COARSE[0])
    lo, hi = float(COARSE[0]), float(COARSE[-1])
    b0 = best(COARSE)
    while b0 <= lo + 1e-9 and lo > -COARSE_CAP + 1e-9:     # optimum on the low end: extend down
        lo -= step
        b0 = best([lo, b0])
    while b0 >= hi - 1e-9 and hi < COARSE_CAP - 1e-9:      # optimum on the high end: extend up
        hi += step
        b0 = best([b0, hi])
    fine = np.arange(b0 - FINE_HALF, b0 + FINE_HALF + 1e-9, FINE_STEP)
    b = best(np.concatenate([fine, [b0, 0.0]]))
    return {"factor": float(2.0 ** b), "log2_factor": float(b), "pit_at_1": dev(0.0),
            "pit_at_factor": dev(b), "at_bound": bool(abs(b0) >= COARSE_CAP - 1e-9),
            "coarse_range": [lo, hi], "n_evaluations": len(seen)}


# ------------------------------------------------------------------------------------ combine

def _read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _member_head(d: Path) -> dict:
    head = _read_json(d / "STORE.json")
    N = int(head["n_shards"])
    for i in range(N):
        if not (d / f"PREDICTED_{i}of{N}.json").is_file():
            raise ValueError(f"{d}: shard {i} of {N} is not predicted")
    bad = [k for k in REFUSED_KEYS if head.get(k) is not None]
    if bad:
        raise ValueError(f"{d}: a store with {', '.join(bad)}; combine takes normal stores only")
    return head


def target_factors(factors: dict, factor_set: str) -> dict:
    """{target: c} of one factor set of a factors.json (fit output)."""
    fs = factors["factor_sets"].get(factor_set)
    if fs is None:
        raise ValueError(f"no factor set {factor_set!r} in the factors file "
                         f"({', '.join(sorted(factors['factor_sets']))})")
    return {t: float(fs[a]["factor"]) for t, a in factors["target_assay"].items() if a in fs}


def combine(members, out, *, factors=None, factor_set=None, record=None, log=print) -> dict:
    """Write the combined store of `members` (moment matching when there are two or more) with the
    dispersion factors of `factor_set` applied (module docstring). Skips a store that holds
    COMBINED.json; refuses an out dir that holds anything else."""
    from onion_eval import predict_store as P
    members = [Path(m) for m in members]
    out = Path(out)
    if not members:
        raise ValueError("combine needs at least one member store")
    if (factors is None) != (factor_set is None):
        raise ValueError("--factors and --factor-set go together")
    if (out / COMBINED).is_file():
        log(f"combine: skip, {out / COMBINED} exists")
        return _read_json(out / COMBINED)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out}: exists and is not a finished combined store; give a new dir")
    heads = [_member_head(m) for m in members]
    diff = sorted(k for k in SHARED_KEYS for h in heads[1:] if h.get(k) != heads[0].get(k))
    if diff:
        raise ValueError(f"the member stores differ in {', '.join(sorted(set(diff)))}")
    tf = {}
    fac_info = None
    if factors is not None:
        fdoc = _read_json(factors)
        tf = target_factors(fdoc, factor_set)
        fac_info = {"file": str(Path(factors).resolve()), "factor_set": factor_set,
                    "regimes": list(fdoc["regimes"]),
                    "per_assay": {a: v["factor"] for a, v in fdoc["factor_sets"][factor_set].items()}}
    files = sorted(p.name for p in members[0].glob("*.npz"))
    if len(files) != int(heads[0]["n_requests_total"]):
        raise ValueError(f"{members[0]}: {len(files)} request files, the store lists "
                         f"{heads[0]['n_requests_total']}")
    for m in members[1:]:
        if sorted(p.name for p in m.glob("*.npz")) != files:
            raise ValueError(f"{m}: holds other request files than {members[0]}")
    out.mkdir(parents=True, exist_ok=True)
    head = dict(heads[0])
    head["n_shards"] = 1
    head["ensemble"] = {"members": [str(m.resolve()) for m in members],
                        "member_runs": [h.get("run") for h in heads],
                        "member_ckpt_sha256": [h.get(P.CKPT_KEY) for h in heads],
                        "method": ("moment matching: mean of means; mean of variances plus "
                                   "variance of means" if len(members) > 1 else "single member"),
                        "dispersion": fac_info}
    (out / "STORE.json").write_text(json.dumps(head, indent=1, sort_keys=True) + "\n",
                                    encoding="utf-8")
    n_fac = 0
    t0 = time.time()
    for k, f in enumerate(files):
        arrs = []
        for m in members:
            with np.load(m / f) as z:
                arrs.append({x: z[x] for x in z.files})
        key = str(arrs[0]["key"])
        if any(str(a["key"]) != key for a in arrs):
            raise ValueError(f"{f}: the members hold different keys")
        if len(arrs) > 1:
            n, mu = moment_match([a["n"] for a in arrs], [a["mu"] for a in arrs])
        else:
            n, mu = arrs[0]["n"].astype(np.float64), arrs[0]["mu"].astype(np.float64)
        parts = key.split("/")
        if factors is not None and parts[0] == "item" and len(parts) == 3 and parts[2] in FIT_REGIMES:
            if parts[1] not in tf:
                raise ValueError(f"{key}: no dispersion factor for this target in {factors}")
            n = recalibrate(n, tf[parts[1]])
            n_fac += 1
        res = {"n": np.minimum(n, np.finfo(np.float32).max).astype(np.float32),
               "mu": mu.astype(np.float32)}
        if "ctl_mu" in arrs[0]:
            res["ctl_mu"] = np.mean([a["ctl_mu"].astype(np.float64) for a in arrs], 0).astype(np.float32)
        P._atomic_write(out / f, lambda fh: np.savez(fh, key=np.array(key), **res))
        if (k + 1) % 500 == 0:
            log(f"combine: {k + 1}/{len(files)} requests ({time.time() - t0:.0f} s)")
    if factors is not None and n_fac == 0:
        raise ValueError("no request took a dispersion factor: the factors match no item")
    g = members[0] / P.GUARDRAIL_FILE
    if g.is_file():
        shutil.copyfile(g, out / P.GUARDRAIL_FILE)
    ident = {k: head.get(k) for k in P.IDENTITY_KEYS + P.SETTING_KEYS}
    P._write_json(out / "PREDICTED_0of1.json",
                  {**ident, "n_requests": len(files),
                   **({P.CKPT_KEY: head[P.CKPT_KEY]} if head.get(P.CKPT_KEY) else {})})
    info = {"store": str(out.resolve()), **head["ensemble"], "n_requests": len(files),
            "n_requests_recalibrated": n_fac, "seconds": round(time.time() - t0, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    P._write_json(out / COMBINED, info)
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        P._write_json(Path(record), info)
    log(f"combine: {len(files)} requests from {len(members)} member(s), {n_fac} recalibrated -> {out}")
    return info


# ------------------------------------------------------------------------------------ fit

def calibration_regions(ev: dict, tables: Path, exclude_evs, n: int, seed: int) -> list:
    """`n` candidate blocks in no `regions` / `fit_regions` of `ev` or of `exclude_evs`, drawn
    with numpy's default_rng(seed) and sorted."""
    from onion_eval import evalset as E
    rb = int(ev["region_bins"])
    cands = E.candidate_blocks(E.genome_dir(tables), rb)
    if ev.get("n_candidate_regions") is not None and len(cands) != int(ev["n_candidate_regions"]):
        raise ValueError(f"{len(cands)} candidate blocks now, {ev['n_candidate_regions']} at build")
    used = set()
    for e in [ev, *exclude_evs]:
        if int(e["region_bins"]) != rb:
            raise ValueError("the eval sets have different region_bins")
        used |= {(c, int(s)) for c, s in e["regions"] + e.get("fit_regions", [])}
    free = [(c, int(s)) for c, s in cands if (c, int(s)) not in used]
    if len(free) < n:
        raise ValueError(f"{len(free)} free candidate blocks, need {n}")
    pick = np.random.default_rng(seed).choice(len(free), size=n, replace=False)
    return sorted(free[i] for i in pick)


def fit(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ensemble_store.py fit", description="per-assay NB "
                                 "dispersion factors on calibration regions")
    ap.add_argument("--runs", nargs="+", required=True, help="seed run dirs; the first is seed0")
    ap.add_argument("--evalset", required=True, help="eval set whose imputation items are fitted")
    ap.add_argument("--exclude-evalset", nargs="*", default=[],
                    help="further eval sets whose regions the calibration regions avoid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-regions", type=int, default=32)
    ap.add_argument("--region-seed", type=int, default=370)
    ap.add_argument("--infer-batch", type=int, default=8)
    ap.add_argument("--max-bins", type=int, default=300000,
                    help="seeded subsample of an assay's bins for the PIT search")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import torch
    from onion_eval import score as S
    from minicandi.registry import load_run
    S.set_threads()
    S.tf32_off()

    def log(*m):
        print(*m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "factors.json").is_file():
        log(f"fit: skip, {out / 'factors.json'} exists")
        return 0
    t0 = time.time()
    ev = _read_json(a.evalset)
    exc = [_read_json(p) for p in a.exclude_evalset]
    shas = {S.run_tables_sha256(r) for r in a.runs}
    if len(shas) != 1:
        raise SystemExit("refused: the runs were trained on different tables")
    models = [load_run(r) for r in a.runs]
    cfg0 = {k: v for k, v in models[0][1].items() if k != "seed"}
    for r, (_, cfg) in zip(a.runs[1:], models[1:]):
        d = sorted(k for k in set(cfg0) | set(cfg) if k != "seed" and cfg0.get(k) != cfg.get(k))
        if d:
            log(f"fit: WARNING: {r} config differs from the first run's in {d}")
    regions = calibration_regions(ev, Path(ev["tables_dir"]), exc, a.n_regions, a.region_seed)
    cal = dict(ev)
    cal.update({"regions": [list(x) for x in regions], "pair_regions": [0], "strata": [],
                "calibration_of": {"evalset": str(Path(a.evalset).resolve()),
                                   "evalset_sha256": S.sha256_file(a.evalset),
                                   "excluded": [str(Path(p).resolve()) for p in a.exclude_evalset],
                                   "n_regions": a.n_regions, "region_seed": a.region_seed}})
    cal.pop("pair_regions_p", None)
    cal_path = out / "calibration_evalset.json"
    cal_path.write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
    sc = S.Scorer(str(cal_path), main="model", model=models[0][0], config=models[0][1],
                  device=a.device, run_tables_sha256=shas.pop(), log=log, n_boot=0)
    preds = [S.ModelPredictor(m, a.device) for m, _ in models]
    imp, _, _, _ = sc.jobs()
    items = [j for j in imp if j["regime"] in FIT_REGIMES]
    log(f"fit: {len(items)} items, {len(regions)} calibration regions, {len(preds)} models")
    by_assay = {}
    tm = time.time()
    for ji, job in enumerate(items):
        t = sc.C.TRACK_INDEX[sc._row(job["target"])["assay"]]
        ys, ns, mus = [], [[] for _ in preds], [[] for _ in preds]
        for ri, tiles in sc._tiles(sc.regions):
            sc._set_region(*sc.regions[ri])
            for b0 in range(0, len(tiles), a.infer_batch):
                tl = tiles[b0:b0 + a.infer_batch]
                batch = sc.C.collate([sc.ds.window(c, s, job["inputs"], [job["target"]])
                                      for c, s in tl])
                ys.append(batch["y_counts"][:, :, t].numpy().reshape(-1).astype(np.float64))
                for k, pr in enumerate(preds):
                    with torch.no_grad():
                        p = pr(batch)
                    ns[k].append(p["n"][:, :, t].reshape(-1))
                    mus[k].append(p["mu"][:, :, t].reshape(-1))
        d = by_assay.setdefault(job["assay"], {"y": [], "n": [[] for _ in preds],
                                               "mu": [[] for _ in preds], "targets": []})
        d["y"].append(np.concatenate(ys).astype(np.float32))
        d["targets"].append(job["target"])
        for k in range(len(preds)):
            d["n"][k].append(np.concatenate(ns[k]).astype(np.float32))
            d["mu"][k].append(np.concatenate(mus[k]).astype(np.float32))
        log(f"fit: item {ji + 1}/{len(items)} {job['target']} ({time.time() - tm:.0f} s)")
    t_fwd = time.time() - tm
    sets = {"seed0": {}, **({"ensemble": {}} if len(preds) > 1 else {})}
    n_bins = {}
    for assay in sorted(by_assay):
        d = by_assay[assay]
        y = np.concatenate(d["y"]).astype(np.float64)
        nk = np.stack([np.concatenate(x) for x in d["n"]]).astype(np.float64)
        mk = np.stack([np.concatenate(x) for x in d["mu"]]).astype(np.float64)
        ok = np.isfinite(y) & np.isfinite(nk).all(0) & np.isfinite(mk).all(0) & (nk > 0).all(0)
        idx = np.flatnonzero(ok)
        if idx.size > a.max_bins:
            idx = np.sort(np.random.default_rng(0).choice(idx, size=a.max_bins, replace=False))
        n_bins[assay] = {"bins": int(y.size), "finite": int(ok.sum()), "used": int(idx.size)}
        variants = {"seed0": (nk[0, idx], mk[0, idx])}
        if len(preds) > 1:
            variants["ensemble"] = moment_match(nk[:, idx], mk[:, idx])
        for v, (n, mu) in variants.items():
            r = fit_factor(n, mu, y[idx])
            sets[v][assay] = r
            log(f"fit: {assay} {v}: factor {r['factor']:.4g}; PIT deviation {r['pit_at_1']:.5f} "
                f"at 1, {r['pit_at_factor']:.5f} at the factor")
    target_assay = {j["target"]: j["assay"] for j in items}
    doc = {"criterion": "least pooled PIT deviation per assay (metrics.pit_deviation pooled, the "
                        "per-assay quantity of score.py's uncertainty term); n' = n / factor",
           "fit_data": {"source": "calibration regions: candidate blocks in no regions or "
                                  "fit_regions of the eval sets, items of the eval set",
                        "calibration_evalset": str(cal_path.resolve()),
                        "regions": [list(x) for x in regions], "region_seed": a.region_seed,
                        "bins_per_assay": n_bins, "max_bins": a.max_bins, "subsample_seed": 0},
           "regimes": list(FIT_REGIMES), "runs": [str(Path(r).resolve()) for r in a.runs],
           "evalset_items": str(Path(a.evalset).resolve()),
           "evalset_sha256_items": S.sha256_file(a.evalset),
           "excluded_evalsets": [str(Path(p).resolve()) for p in a.exclude_evalset],
           "grid": {"coarse_log2": [float(x) for x in COARSE], "coarse_cap": COARSE_CAP,
                    "fine_half": FINE_HALF,
                    "fine_step": FINE_STEP},
           "factor_sets": sets, "target_assay": target_assay,
           "seconds": {"forward": round(t_fwd, 1), "total": round(time.time() - t0, 1)}}
    tmp = out / ".factors.json.part"
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out / "factors.json")
    log(f"fit: wrote {out / 'factors.json'} ({time.time() - t0:.0f} s)")
    return 0


def combine_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ensemble_store.py combine", description="combined "
                                 "prediction store of seed stores, with dispersion factors")
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factors", default=None)
    ap.add_argument("--factor-set", default=None)
    ap.add_argument("--record", default=None, help="also write the COMBINED.json record here")
    a = ap.parse_args(argv)
    try:
        combine(a.members, a.out, factors=a.factors, factor_set=a.factor_set, record=a.record)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fit":
        sys.exit(fit(sys.argv[2:]))
    if cmd == "combine":
        sys.exit(combine_cli(sys.argv[2:]))
    print(__doc__)
    sys.exit(2)
