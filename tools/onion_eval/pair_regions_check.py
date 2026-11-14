"""PCk -- the check that chooses the number of pair regions p (Amendment 8, section B.5).

    python tools/onion_eval/pair_regions_check.py run --arm <arm dir> --arm <arm dir> \
        --baselines <dir> --evalset <evalset.json> [--sizes 8,16,32] [--n-boot 2000] --out <dir>
    python tools/onion_eval/pair_regions_check.py thresholds

Each `--arm` is a directory of seed score directories (`seed0`, `seed1`, `seed2`), each holding
the `scores.json` and `pairs_per_region.npz` of one scoring with all pair regions. The first arm
is the incumbent of the D13 verdict (rung 1 `base`), the second the challenger
(`scrambled_covariates`). `--baselines` holds `depth_only_scaling/seed0` and
`pseudoreplicate_floor/seed0`; their per-region pair sums are compared with the reference sums
that each run carries (a consistency row; not a condition).

For every p in `--sizes` the pair regions are the positions 0, R/p, 2R/p, ... of the region order
(the nested subsets of chunk EVd). The tool rebuilds the pair records from the per-region sums
(a pair's CRPS = its CRPS sum over the chosen regions / its bins there; the totals are sums), and
calls the scorer's own aggregation `score.pair_metrics_from_records`. The region bootstrap draws
clusters of regions with replacement (adjacent regions, i.e. same chromosome and touching bins, are
one cluster), 2,000 draws, with one fixed seed and the same draws for every run.

Conditions (written down before the scoring they read; the text is in
`cruxvault/results/<task>/PAIR_CHECK_PREREGISTRATION.md`; the constants below equal it):
  (1) per run: the macro of headline 4, of headline 5, and of every populated rung of the
      replicate-pair gap closed, on the subset, lies inside the 95% region-bootstrap interval of
      its all-regions value;
  (2) per (run, correction-scorable assay) cell of headline 4, the same; at least 95% of the cells
      pass;
  (3) the D13 verdict (`select_rule.decide`, on copies of the scores whose pair metrics are the
      subset's) equals the verdict on the all-regions scores;
  (4) per arm, the seed spread of the correction term on the subset is at most 1.5 times its
      all-regions value;
  (5) metric 7 per rung (`predicted_over_replicate_variance`) is equal in the subset copies, and
      the aggregation returns no key outside the pair keys of section B.4.
p is the smallest size that passes all five. Outputs: `<out>/check.tsv`, `<out>/PAIR_REGIONS.md`
(first line `pair regions: <p>` or `pair regions: 256 (none of 8, 16, 32 passes)`), and the
subset copies of the scores under `<out>/subset_scores/`.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from onion_eval import select_rule as SR  # noqa: E402

# ------------------------------------------------------------------ thresholds (pre-registered)
SIZES = (8, 16, 32)            # candidate pair-region counts, smallest first
N_BOOT = 2000                  # region-bootstrap draws
BOOT_SEED = 0                  # one fixed seed; the same draws for every run
INTERVAL = (2.5, 97.5)         # the 95% percentile interval
PER_ASSAY_SHARE_MIN = 0.95     # condition (2): share of (run, assay) cells inside
SPREAD_RATIO_MAX = 1.5         # condition (4): subset spread / all-regions spread
INSIDE_RTOL = 1e-9             # numerical tolerance of "inside" (float rounding of sums only)
MIN_DEFINED_SHARE = 0.5        # an interval needs at least this share of defined draws
CONDITIONS = ("1", "2", "3", "4", "5")

# the keys of section B.4 that a pair-region subset may move (and nothing else)
B4_KEYS = ("correction_gap_closed_vs_depth_only_never_trained", "correction_crps_model",
           "correction_crps_model_oracle_scaled", "correction_crps_depth_only",
           "correction_crps_pseudoreplicate_floor", "subset_pairs_gap_closed",
           "depth_law_abs_log2_error_never_trained", "replicate_pair_gap_closed_vs_depth_only")
H4, H5, REP = B4_KEYS[0], B4_KEYS[6], B4_KEYS[7]
METRIC7 = "predicted_over_replicate_variance"
SUM_KEYS = ("crps_sum_model", "crps_sum_depth_only", "crps_sum_floor", "bins", "pred_total",
            "source_total")
META_STR = ("assay", "cell", "rung", "knob", "source", "target")
META_INT = ("never_trained", "subset_pair")
NOT_ADDITIVE = "not additive per region (oracle split)"


def thresholds() -> dict:
    return {"sizes": list(SIZES), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "interval_percentiles": list(INTERVAL), "per_assay_share_min": PER_ASSAY_SHARE_MIN,
            "spread_ratio_max": SPREAD_RATIO_MAX, "inside_rtol": INSIDE_RTOL,
            "min_defined_share": MIN_DEFINED_SHARE, "verdict": "equal on subset and all regions",
            "metric7": "keys equal"}


# ------------------------------------------------------------------ regions
def pair_region_positions(n_regions: int, p: int) -> list[int]:
    """Section B.4: the positions 0, R/p, 2R/p, ... of the region order (p must divide R). SF0: the
    one implementation is `evalset.pair_positions`, which `evalset.py derive` uses; the scorer
    reads the derived eval set's `pair_regions`."""
    from onion_eval.evalset import pair_positions
    return pair_positions(n_regions, p)


def clusters(regions, region_bins: int) -> np.ndarray:
    """Cluster id per region (regions in region order): a region that starts where the previous one
    ends on the same chromosome joins its cluster."""
    cid, out = -1, []
    for i, (c, s) in enumerate(regions):
        pc, ps = regions[i - 1] if i else (None, None)
        if not (i and c == pc and int(s) - int(ps) == region_bins):
            cid += 1
        out.append(cid)
    return np.asarray(out, np.int64)


def boot_weights(cl: np.ndarray, n_boot: int, seed: int = BOOT_SEED) -> np.ndarray:
    """[n_boot, R] multiplicity of each region in each draw of the clusters with replacement."""
    k = int(cl.max()) + 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(n_boot, k))
    counts = np.zeros((n_boot, k), np.float64)
    for b in range(n_boot):
        np.add.at(counts[b], pick[b], 1.0)
    return counts[:, cl]


# ------------------------------------------------------------------ inputs
def _bad(x) -> bool:
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def load_evalset(path) -> dict:
    ev = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("regions", "region_bins", "correction_pairs", "scorable"):
        if k not in ev:
            raise KeyError(f"{path}: no {k!r}")
    return ev


class Run:
    """One scoring: `scores.json` and `pairs_per_region.npz` of a seed directory, with the pair
    metadata (from the npz when it carries it; else the eval set for correction pairs and
    `<tables>/pairs.tsv` for replicate pairs, as the scorer's `jobs` reads them)."""

    def __init__(self, d, ev: dict):
        self.dir = Path(d)
        self.scores = json.loads((self.dir / "scores.json").read_text(encoding="utf-8"))
        z = np.load(self.dir / "pairs_per_region.npz", allow_pickle=False)
        self.pair_ids = [str(x) for x in z["pair_ids"]]
        self.region_indices = np.asarray(z["region_indices"], np.int64)
        P, Rp = len(self.pair_ids), len(self.region_indices)
        self.arr = {}
        for k in SUM_KEYS:
            if k not in z:
                raise KeyError(f"{self.dir}/pairs_per_region.npz: no {k!r}")
            a = np.asarray(z[k], np.float64)
            if a.shape != (P, Rp):
                raise ValueError(f"{self.dir}: {k} has shape {a.shape}, need {(P, Rp)}")
            self.arr[k] = a
        if "log2_depth_ratio" not in z:
            raise KeyError(f"{self.dir}/pairs_per_region.npz: no 'log2_depth_ratio' (one float per "
                           f"pair; headline 5 cannot be rebuilt without it)")
        self.log2r = np.asarray(z["log2_depth_ratio"], np.float64)
        self.meta = self._meta(z, ev)

    def _meta(self, z, ev) -> list[dict]:
        cor = {p["pair_id"]: p for p in ev["correction_pairs"]}
        rep_ids = set(ev.get("replicate_pairs") or [])
        npz_meta = all(k in z for k in META_STR + META_INT)
        tsv = None
        out = []
        for i, pid in enumerate(self.pair_ids):
            if pid in cor:
                kind = "correction"
            elif pid in rep_ids:
                kind = "replicate"
            else:
                raise KeyError(f"{self.dir}: pair {pid!r} is in no pair list of the eval set")
            if npz_meta:
                m = {k: str(z[k][i]) for k in META_STR}
                m.update({k: int(z[k][i]) for k in META_INT})
                if kind == "correction":
                    m["rung"] = None
            elif kind == "correction":
                p = cor[pid]
                m = {"assay": p["assay"], "cell": p["cell"], "knob": p.get("knob", ""),
                     "never_trained": int(p["never_trained"]), "subset_pair": int(p["subset_pair"]),
                     "rung": None, "source": p["source"], "target": p["target"]}
            else:
                if tsv is None:
                    from onion_eval.evalset import read_tsv
                    tsv = {r["pair_id"]: r for r in read_tsv(Path(self.scores["tables"]) / "pairs.tsv")}
                p = tsv[pid]
                m = {"assay": p["assay"], "cell": p["cell"], "knob": p.get("knobs_differing", ""),
                     "never_trained": int(p["never_trained"] or 0),
                     "subset_pair": int(p["subset_pair"] or 0), "rung": p["rung"],
                     "source": p["a"], "target": p["b"]}
            m.update(pair_id=pid, kind=kind, log2_depth_ratio=float(self.log2r[i]))
            out.append(m)
        return out

    def records(self, sums: dict) -> tuple[list, list]:
        """Pair records of the scorer's `_pair_record` form from summed arrays [P]."""
        from onion_eval import metrics as Mx
        cor, rep = [], []
        b = sums["bins"]
        for i, m in enumerate(self.meta):
            r = {k: m[k] for k in ("pair_id", "assay", "cell", "knob", "never_trained",
                                   "subset_pair", "rung", "source", "target")}
            for key, sk in (("crps_model", "crps_sum_model"), ("crps_depth_only", "crps_sum_depth_only"),
                            ("crps_floor", "crps_sum_floor")):
                v = sums[sk][i] / b[i] if b[i] > 0 else float("nan")
                r[key] = float(v) if math.isfinite(v) else Mx.Undefined(f"{key} undefined on this pair")
                r[key + "_oracle_scaled"] = Mx.Undefined(NOT_ADDITIVE)
            r["pred_total"] = float(sums["pred_total"][i])
            r["source_total"] = float(sums["source_total"][i])
            r["log2_depth_ratio"] = m["log2_depth_ratio"]
            (cor if m["kind"] == "correction" else rep).append(r)
        return cor, rep

    def columns(self, positions) -> np.ndarray:
        """Weights [Rp]: 1 on the npz columns of the given region positions, else 0."""
        pos = set(int(x) for x in positions)
        miss = pos - set(self.region_indices.tolist())
        if miss:
            raise ValueError(f"{self.dir}: pair regions {sorted(miss)[:5]} not in pairs_per_region.npz")
        return np.array([1.0 if int(r) in pos else 0.0 for r in self.region_indices])


def values(out: dict) -> dict:
    """The check's quantities from the aggregation's output."""
    h4 = out.get(H4) or {}
    return {"h4": h4.get("macro"), "h4_assay": dict(h4.get("per_assay") or {}),
            "h5": (out.get(H5) or {}).get("macro"),
            "rep": {g: v.get("macro") for g, v in ((out.get(REP) or {}).get("per_rung") or {}).items()}}


def _agg(run: Run, w: np.ndarray, pmfr, scorable) -> dict:
    sums = {k: run.arr[k] @ w for k in SUM_KEYS}
    cor, rep = run.records(sums)
    return pmfr(cor, rep, scorable, 0, 0)


def _inside(v, lo, hi) -> bool:
    if _bad(v) or _bad(lo) or _bad(hi):
        return False
    tol = INSIDE_RTOL * max(1.0, abs(v), abs(lo), abs(hi))
    return lo - tol <= v <= hi + tol


def _interval(vals) -> tuple:
    ok = [float(v) for v in vals if not _bad(v)]
    if len(ok) < max(1, MIN_DEFINED_SHARE * len(vals)):
        return None, None
    return float(np.percentile(ok, INTERVAL[0])), float(np.percentile(ok, INTERVAL[1]))


# ------------------------------------------------------------------ subset copies of the scores
def subset_scores(orig: dict, out: dict, p: int) -> dict:
    """`orig` with the pair keys of section B.4 replaced by `out` and the selection's correction
    term and its denominators recomputed as the scorer does."""
    s = copy.deepcopy(orig)
    for k in B4_KEYS:
        if k in out:
            s[k] = out[k]
    cor = out[H4]
    sel = s.setdefault("selection", {})
    sel["correction_gap_closed"] = cor.get("macro")
    sel["correction_reason"] = cor.get("reason", "") if cor.get("macro") is None else ""
    g = [sel.get(SR.GOAL_KEYS[k]) for k in SR.GOALS]
    sel["composite"] = (None if any(_bad(x) for x in g)
                        else sum(SR.WEIGHTS[k] * float(x) for k, x in zip(SR.GOALS, g)))
    den_by = cor.get("denominator_relative_by_assay") or {}
    den = [v for v in den_by.values() if v is not None]
    d = sel.setdefault("denominators", {})
    d["correction_baseline_minus_floor_relative_min"] = (
        min(den) if den and len(den) == len(den_by) else None)
    d["correction_baseline_minus_floor_relative_by_assay"] = den_by
    s["pair_regions_check_subset"] = p
    return s


def _metric7_paths(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == METRIC7:
                yield path + (k,), v
            else:
                yield from _metric7_paths(v, path + (k,))


def _verdict(inc_dir: Path, ch_dir: Path, simpler: str) -> tuple[str, str]:
    try:
        r = SR.decide(SR.load_arm(inc_dir), SR.load_arm(ch_dir), simpler)
        return r["verdict"], r.get("reason", "")
    except SR.Refused as e:
        return "REFUSED", str(e)


def _spread(arm_dir: Path):
    seeds = SR.load_arm(arm_dir)
    v = [s["goals"]["correction"] for s in seeds]
    return None if any(x is None for x in v) else SR.spread(v)


# ------------------------------------------------------------------ the check
def _row(p, cond, scope, quantity, value, ref, lo, hi, threshold, ok, note=""):
    f = lambda x: "" if x is None else (f"{x:.10g}" if isinstance(x, float) else str(x))  # noqa: E731
    return {"p": p, "condition": cond, "scope": scope, "quantity": quantity, "value": f(value),
            "value_all_regions": f(ref), "lo": f(lo), "hi": f(hi), "threshold": f(threshold),
            "pass": ("pass" if ok else "fail") if ok is not None else "", "note": note}


def run_check(arms, baselines, evalset, out, *, sizes=SIZES, n_boot=N_BOOT, pmfr=None,
              log=print) -> dict:
    if pmfr is None:
        from onion_eval import score as S
        if not hasattr(S, "pair_metrics_from_records"):
            raise SystemExit("score.py has no pair_metrics_from_records (chunk SF0 not merged)")
        pmfr = S.pair_metrics_from_records
    if len(arms) != 2:
        raise SystemExit("give exactly two --arm (incumbent first, then challenger)")
    sizes = sorted(int(x) for x in sizes)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log("thresholds: " + json.dumps(thresholds(), sort_keys=True))
    if n_boot != N_BOOT:
        log(f"WARNING: --n-boot {n_boot} differs from the pre-registered {N_BOOT}")
    ev = load_evalset(evalset)
    regions, rb = [tuple(r[:2]) for r in ev["regions"]], int(ev["region_bins"])
    R = len(regions)
    scorable = ev["scorable"]
    arm_dirs = [Path(a) for a in arms]
    runs = {a.name: [(d.name, Run(d, ev)) for d in SR.seed_dirs(a)
                     if (d / "pairs_per_region.npz").is_file()] for a in arm_dirs}
    for a, rs in runs.items():
        if len(rs) < SR.MIN_SEEDS:
            raise SystemExit(f"arm {a}: {len(rs)} seeds with pairs_per_region.npz; need {SR.MIN_SEEDS}")
    rows = []
    all_runs = [(a, s, r) for a, rs in runs.items() for s, r in rs]
    ri0 = all_runs[0][2].region_indices
    for a, s, r in all_runs:
        if not np.array_equal(r.region_indices, ri0):
            raise SystemExit(f"{r.dir}: region_indices differ from {all_runs[0][2].dir}")
    if len(ri0) != R or not np.array_equal(np.sort(ri0), np.arange(R)):
        raise SystemExit(f"the runs carry {len(ri0)} pair regions; the check needs all {R}")
    cl = clusters([regions[i] for i in ri0], rb)
    W = boot_weights(cl, n_boot)
    log(f"regions {R}, clusters {int(cl.max()) + 1}, draws {n_boot}, runs {len(all_runs)}")

    # consistency rows: references against the baselines, and the recompute against scores.json
    for nm, key in (("depth_only_scaling", "crps_sum_depth_only"), ("pseudoreplicate_floor", "crps_sum_floor")):
        bd = Path(baselines) / nm / "seed0"
        if not (bd / "pairs_per_region.npz").is_file():
            rows.append(_row("", "consistency", nm, key, None, None, None, None, None, None,
                             f"no {bd}/pairs_per_region.npz"))
            continue
        z = np.load(bd / "pairs_per_region.npz", allow_pickle=False)
        bidx = {str(p): i for i, p in enumerate(z["pair_ids"])}
        bcol = {int(x): j for j, x in enumerate(z["region_indices"])}
        worst = 0.0
        for a, s, r in all_runs:
            for i, pid in enumerate(r.pair_ids):
                if pid not in bidx:
                    continue
                cols = [bcol[int(x)] for x in r.region_indices]
                bv = np.asarray(z["crps_sum_model"][bidx[pid]], np.float64)[cols]
                rv = r.arr[key][i]
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = np.nanmax(np.abs(rv - bv) / np.maximum(np.abs(bv), 1e-300)) if rv.size else 0.0
                worst = max(worst, float(d) if np.isfinite(d) else float("inf"))
        rows.append(_row("", "consistency", nm, f"max relative difference of {key} against the "
                         f"baseline's crps_sum_model", worst, None, None, None, None, None))

    full, boot = {}, {}
    ones = np.ones(R)
    for a, s, r in all_runs:
        o = pmfr(*r.records({k: r.arr[k] @ ones for k in SUM_KEYS}), scorable, 0, 0)
        extra = sorted(set(o) - set(B4_KEYS))
        full[(a, s)] = (o, values(o), extra)
        v = values(o)
        st = r.scores
        for q, sv in (("h4", (st.get(H4) or {}).get("macro")), ("h5", (st.get(H5) or {}).get("macro"))):
            d = None if _bad(sv) or _bad(v[q]) else abs(float(sv) - float(v[q]))
            rows.append(_row("", "consistency", f"{a}/{s}", f"{q} recomputed - scores.json", d,
                             sv, None, None, None, None))
        sums = {k: r.arr[k] @ W.T for k in SUM_KEYS}        # [P, n_boot]
        bv = []
        for b in range(n_boot):
            cor, rep = r.records({k: sums[k][:, b] for k in SUM_KEYS})
            bv.append(values(pmfr(cor, rep, scorable, 0, 0)))
        boot[(a, s)] = bv
        log(f"bootstrap done: {a}/{s}")

    inc_name, ch_name = arm_dirs[0].name, arm_dirs[1].name
    try:
        simpler = SR.simpler_from_config(arm_dirs[1])
    except SR.Refused:
        simpler = "none"
    v256, why256 = _verdict(arm_dirs[0], arm_dirs[1], simpler)
    spread256 = {a.name: _spread(a) for a in arm_dirs}
    summary = {}
    for p in sizes:
        pos = pair_region_positions(R, p)
        cond = {c: [] for c in CONDITIONS}
        sub_root = out / "subset_scores" / f"p{p}"
        for a, s, r in all_runs:
            o = _agg(r, r.columns(pos), pmfr, scorable)
            vs, (o256, v2, extra) = values(o), full[(a, s)]
            bv = boot[(a, s)]
            items = [("h4", v2["h4"], vs["h4"], [x["h4"] for x in bv]),
                     ("h5", v2["h5"], vs["h5"], [x["h5"] for x in bv])]
            items += [(f"replicate gap closed, rung {g}", v2["rep"][g], vs["rep"].get(g),
                       [x["rep"].get(g) for x in bv]) for g in sorted(v2["rep"]) if not _bad(v2["rep"][g])]
            for q, ref, val, bl in items:
                if _bad(ref):
                    rows.append(_row(p, "1", f"{a}/{s}", q, val, ref, None, None, None, None,
                                     "undefined on all regions: not checked"))
                    continue
                lo, hi = _interval(bl)
                ok = _inside(val, lo, hi)
                cond["1"].append(ok)
                rows.append(_row(p, "1", f"{a}/{s}", q, val, ref, lo, hi, "inside 95% interval", ok))
            for asy in sorted(scorable.get("correction", [])):
                ref = v2["h4_assay"].get(asy)
                if _bad(ref):
                    rows.append(_row(p, "2", f"{a}/{s}/{asy}", "h4 per assay", vs["h4_assay"].get(asy),
                                     ref, None, None, None, None, "undefined on all regions: not counted"))
                    continue
                lo, hi = _interval([x["h4_assay"].get(asy) for x in bv])
                ok = _inside(vs["h4_assay"].get(asy), lo, hi)
                cond["2"].append(ok)
                rows.append(_row(p, "2", f"{a}/{s}/{asy}", "h4 per assay", vs["h4_assay"].get(asy),
                                 ref, lo, hi, "inside 95% interval", ok))
            sc = subset_scores(r.scores, o, p)
            d = sub_root / a / s
            d.mkdir(parents=True, exist_ok=True)
            (d / "scores.json").write_text(json.dumps(sc, indent=1, sort_keys=True), encoding="utf-8")
            m7a, m7b = dict(_metric7_paths(r.scores)), dict(_metric7_paths(sc))
            ok5 = m7a == m7b and not (set(o) - set(B4_KEYS))
            cond["5"].append(ok5)
            rows.append(_row(p, "5", f"{a}/{s}", "metric 7 keys equal; no key outside B.4",
                             len(m7a), None, None, None, "equal", ok5,
                             "" if ok5 else f"extra keys {sorted(set(o) - set(B4_KEYS))[:4]}"))
        vp, whyp = _verdict(sub_root / inc_name, sub_root / ch_name, simpler)
        ok3 = v256 != "REFUSED" and vp == v256
        cond["3"].append(ok3)
        rows.append(_row(p, "3", f"{inc_name} vs {ch_name}", "D13 verdict", vp, v256, None, None,
                         "equal", ok3, f"simpler={simpler}; subset: {whyp}; all regions: {why256}"))
        for arm in (inc_name, ch_name):
            sp, s0 = _spread(sub_root / arm), spread256[arm]
            if sp is None or s0 is None:
                ok4 = False
            else:
                ok4 = sp <= SPREAD_RATIO_MAX * s0 * (1 + INSIDE_RTOL) + 1e-300 if s0 > 0 else sp == 0
            cond["4"].append(ok4)
            rows.append(_row(p, "4", arm, "seed spread of the correction term", sp, s0, None,
                             None if s0 is None else SPREAD_RATIO_MAX * s0,
                             f"<= {SPREAD_RATIO_MAX} x all regions", ok4))
        res = {}
        for c in CONDITIONS:
            if c == "2":
                share = (sum(cond[c]) / len(cond[c])) if cond[c] else None
                ok = share is not None and share >= PER_ASSAY_SHARE_MIN
                rows.append(_row(p, "2", "ALL", "share of cells inside", share, None, None, None,
                                 f">= {PER_ASSAY_SHARE_MIN}", ok))
            else:
                ok = bool(cond[c]) and all(cond[c])
                rows.append(_row(p, c, "ALL", f"{sum(cond[c])} of {len(cond[c])} pass", None, None,
                                 None, None, "all", ok))
            res[c] = ok
        summary[p] = res
        log(f"p = {p}: " + ", ".join(f"({c}) {'pass' if res[c] else 'fail'}" for c in CONDITIONS))

    passing = [p for p in sizes if all(summary[p].values())]
    chosen = passing[0] if passing else None
    cols = ("p", "condition", "scope", "quantity", "value", "value_all_regions", "lo", "hi",
            "threshold", "pass", "note")
    with open(out / "check.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in cols) + "\n")
    first = (f"pair regions: {chosen}" if chosen else
             f"pair regions: {R} (none of {', '.join(str(p) for p in sizes)} passes)")
    md = [first, "",
          f"The check of Amendment 8, section B.5, ran on {len(all_runs)} runs ({inc_name} and "
          f"{ch_name}) with {n_boot} region-bootstrap draws over {int(cl.max()) + 1} clusters of "
          f"{R} regions.", "",
          "| p | " + " | ".join(f"({c})" for c in CONDITIONS) + " | all |",
          "|---|" + "---|" * (len(CONDITIONS) + 1)]
    for p in sizes:
        md.append(f"| {p} | " + " | ".join("pass" if summary[p][c] else "fail" for c in CONDITIONS)
                  + f" | {'pass' if all(summary[p].values()) else 'fail'} |")
    md += ["", f"Thresholds: `{json.dumps(thresholds(), sort_keys=True)}`.",
           f"D13 verdict on all regions: {v256}.",
           "Limit: the check rests on one rung and one pair of arms.",
           "The per-row values are in `check.tsv`."]
    (out / "PAIR_REGIONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(first)
    return {"chosen": chosen, "summary": summary, "first_line": first, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", action="append", required=True)
    r.add_argument("--baselines", required=True)
    r.add_argument("--evalset", required=True)
    r.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    r.add_argument("--out", required=True)
    sub.add_parser("thresholds")
    a = ap.parse_args(argv)
    if a.cmd == "thresholds":
        print(json.dumps(thresholds(), indent=1, sort_keys=True))
        return 0
    run_check(a.arm, a.baselines, a.evalset, a.out, sizes=[int(x) for x in a.sizes.split(",")],
              n_boot=a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
