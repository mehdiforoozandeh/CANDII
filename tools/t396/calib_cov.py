"""t396 C9 (leg A): start-count correlation per assay, and the calibration of the converted
read-start model with that correlation, next to an equally recalibrated read-count reference.

    python tools/t396/calib_cov.py fit   --truth <dir> [--arrays <dir>] [--ref-pred-store <dir>]
        --seed <k> --out <fit.npz> [--fit-regions odd|even] [--window-bins 1024] [--infer-batch 8]
    python tools/t396/calib_cov.py score --truth <dir> --arrays <dir> --ref-pred-store <dir>
        --seed <k> --fit <fit.npz> --out <records.jsonl> [--window-bins 1024] [--infer-batch 8]
    python tools/t396/calib_cov.py table --records <files...> [--legB-table <table.tsv>]
        [--scores-json <scores.json>] --out <dir>

Inputs (the C6 arrays contract): `--truth` holds `<target>.npz` (y [R, rb], s [R, rb, 2], kernel
[2, 17], n_reads_strand [2], stranded, assay; R = all regions of the evaluation set, in its order);
`--arrays` holds the read-start model files `<target>.npz` (mu_sub, n_sub [R', rb, 2]; the stored
converted mu, n [R', rb]; region_indices [R']). The read-count reference is read in place from its
prediction store, key `item/<target>/leave_one_assay_out` (file name sha1(key).npz, the rule of
`predict_store.key_file`; read here with numpy so that the tool needs no torch). Targets: the
files present in both `--truth` and `--arrays`.

Region split. `fit --fit-regions odd` (the default) fits on the odd regions (evaluation-set index
mod 2 = 1), `--fit-regions even` on the even ones (mod 2 = 0; they hold the 16 pair regions); the
fit file records its half as `fit_regions`. `score` scores the other half. The two sets are
disjoint, so a fit and the regions it is applied to never share a region. Iteration 2 makes one fit
per half and run; `score_spaces.py records --rho-odd --rho-even` applies each to the other half.

fit (the `--fit-regions` half). It takes `--arrays` (a start-output arm), `--ref-pred-store` (a
count-output prediction store: the read-count reference of iteration 1, or the arm
`count_in_count_out` of iteration 2), or both; at least one. With `--arrays` absent the rho parts are
0 and the `rs` keys NaN (`has_arrays` false); with `--ref-pred-store` absent the `ref` keys are NaN
(`has_ref` false); the targets are then all truth files. Per target: the `covconv.estimate_rho` parts (num, den) of the read-start model
(truth `s` against the native mu_sub, n_sub; an unstranded track folded to channel 0); the odd
calibration ratio of the stored read-start conversion and of the reference; and the scalar
`covconv.fit_scale` of the read-start model (stored conversion) and of the reference, fitted on the
OTHER targets of the target's assay (leave-one-target-out). An assay with one target takes the
geometric mean of the in-sample scalars of the other assays (flag `fallback`).

score (the half the fit did not use; needs both --arrays and --ref-pred-store), per target, `deviance.target_terms(y, mu, n)` of six variants:
  rs_independent    (i)   read-start, the stored independent conversion
  rs_cov_loo        (ii)  read-start, `convert_nb_cov` with the rho of the target's assay summed
                          over the assay's other targets (leave-one-target-out); an assay with one
                          target uses the mean rho of the other assays of the same strandedness
                          (all other assays when there is none), flag `fallback`
  rs_cov_insample   (iii) read-start, rho of all targets of the assay, the scored one included
                          (in-sample; an upper bound)
  rs_scaled_loo     (iv)  read-start, stored conversion with n / c, c the leave-one-target-out
                          `fit_scale` of the odd regions
  ref_stored        (v)   reference as stored
  ref_scaled_loo    (vi)  reference with n / c, c the leave-one-target-out `fit_scale` of the odd
                          regions
(ii) and (vi) are the like-for-like pair: both fit on the other half and exclude the scored target.

Conversion runs. The adapter converted each forward batch on its own (`rs_adapter.forward`:
`read_start.stitch` runs of one batch; `predict_store._batches`: a region's windows in chunks of
`--infer-batch` when that is <= rb / `--window-bins`, else the flat window list of all regions in
chunks). Every fresh conversion here runs on those same segments, so variant (i) recomputed with
`identity_rho` reproduces the stored n; each (i) record carries that check as
`n_rel_diff_identity` (max |fresh - stored| / stored over the even regions).

Caveat. The residuals y - mu of `estimate_rho` include the model's mean error (finding f156), not
only the NB excess around a right mean, so rho may over-correct. `rho.tsv` reports the unclipped
num / den beside rho, and `lag0_raw` (the same-strand lag-0 num / den, forced to 1 in rho): a
`lag0_raw` above 1 says the residuals exceed the model's own excess variance at lag 0, and the
off-lag rho then carries part of that size error.

table: `table.tsv` (variant, macro, calibration_ratio = mean over seeds, seed_range = max - min over
seeds, per_seed, n_targets, flag); macro `all_assays` = mean over assays of the per-assay
`deviance.calibration_ratio` (the scorer's aggregation), and `scorable` over the 9 scorable assays
of `scores.json["scorable"]["imputation"]` when `--scores-json` is given. `rho.tsv`: per seed,
assay, strand pair and lag, from the in-sample rho parts that `score` appends to its records (lines
with "kind": "rho"). `native_start.tsv` (only with `--legB-table`): the leg B rows of the
read-start model in start space. numpy and scipy only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import deviance as DV  # noqa: E402
from t396 import covconv as CC  # noqa: E402

REGIME = "leave_one_assay_out"
VARIANTS = ("rs_independent", "rs_cov_loo", "rs_cov_insample", "rs_scaled_loo",
            "ref_stored", "ref_scaled_loo")
TERM_KEYS = ("bins", "D0", "D_model", "D_scaled", "D_iso", "floor", "E_nb", "nll_model")
FIT_ITERS = 40          # bisection steps of fit_scale on log c over [1e-3, 1e3]: 13.8 / 2^40


# ------------------------------------------------------------------ inputs
def ref_key_file(target: str) -> str:
    key = f"item/{target}/{REGIME}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npz"


def read_ref(store_dir, target: str) -> tuple:
    p = Path(store_dir) / ref_key_file(target)
    with np.load(p) as z:
        key = str(z["key"])
        if key != f"item/{target}/{REGIME}":
            raise SystemExit(f"{p}: holds {key}, not item/{target}/{REGIME}")
        return np.asarray(z["mu"]), np.asarray(z["n"])


def targets_of(truth_dir, arrays_dir=None) -> list:
    t = {p.stem for p in Path(truth_dir).glob("*.npz")}
    if arrays_dir is None:
        return sorted(t)
    a = {p.stem for p in Path(arrays_dir).glob("*.npz")}
    return sorted(t & a)


def kernel_row(kernel, n_reads_strand, stranded: bool) -> np.ndarray:
    """The kernel `convert_nb` gets from the adapter: [2, 17] stranded, else the read-weighted mix
    of the two rows (`read_start.unstranded_kernel`)."""
    k = np.asarray(kernel, np.float64)
    if stranded:
        return k
    nr = np.asarray(n_reads_strand, np.float64)
    return (nr[:, None] * k).sum(0) / nr.sum()


def segments(n_regions: int, rb: int, L: int, B: int) -> list:
    """Per region, the (lo, hi) bin ranges that one forward batch converted as one run."""
    if rb % L:
        raise SystemExit(f"--window-bins {L} does not divide region bins {rb}")
    tiles = rb // L
    if B <= tiles:
        segs = [(b0 * L, min(b0 + B, tiles) * L) for b0 in range(0, tiles, B)]
        return [list(segs) for _ in range(n_regions)]
    out = [[] for _ in range(n_regions)]
    flat = [(r, k) for r in range(n_regions) for k in range(tiles)]
    for b0 in range(0, len(flat), B):
        by_r = defaultdict(list)
        for r, k in flat[b0:b0 + B]:
            by_r[r].append(k)
        for r, ks in by_r.items():
            out[r].append((min(ks) * L, (max(ks) + 1) * L))
    return out


class Target:
    """The arrays of one target on one region subset (`which`: 'odd' or 'even'). Without
    `a.arrays` only the truth and the reference are read, on every region of that half."""

    def __init__(self, a, tid: str, which: str, need_ref: bool = True):
        t = np.load(Path(a.truth) / f"{tid}.npz")
        m = np.load(Path(a.arrays) / f"{tid}.npz") if a.arrays else None
        self.tid = tid
        self.assay = str(t["assay"])
        self.stranded = bool(t["stranded"])
        n_all, rb = t["y"].shape
        ridx = (np.asarray(m["region_indices"], np.int64) if m is not None
                else np.arange(n_all, dtype=np.int64))
        keep = np.flatnonzero(ridx % 2 == (1 if which == "odd" else 0))
        self.ridx = ridx[keep]
        segs = segments(n_all, rb, int(a.window_bins), int(a.infer_batch))
        self.segs = [segs[r] for r in self.ridx]
        self.y = np.asarray(t["y"])[self.ridx].astype(np.float64)
        s = np.asarray(t["s"])[self.ridx].astype(np.float64)
        if not self.stranded:
            s = np.stack([s.sum(-1), np.zeros(s.shape[:2])], -1)
        self.s = s
        self.w = kernel_row(t["kernel"], t["n_reads_strand"], self.stranded)
        if m is None:
            self.mu_sub = self.n_sub = self.mu_rs = self.n_rs = None
        if need_ref:
            mu_ref, n_ref = read_ref(a.ref_pred_store, tid)
            self.mu_ref = np.asarray(mu_ref)[self.ridx].astype(np.float64)
            self.n_ref = np.asarray(n_ref)[self.ridx].astype(np.float64)
        if m is None:
            return
        self.mu_sub = np.asarray(m["mu_sub"])[keep].astype(np.float64)
        self.n_sub = np.asarray(m["n_sub"])[keep].astype(np.float64)
        self.mu_rs = np.asarray(m["mu"])[keep].astype(np.float64)
        self.n_rs = np.asarray(m["n"])[keep].astype(np.float64)

    def convert(self, rho) -> tuple:
        """(mu, n) [R, rb] of `convert_nb_cov` run on each forward-batch segment."""
        mu = np.empty_like(self.y)
        n = np.empty_like(self.y)
        for r, segs in enumerate(self.segs):
            for lo, hi in segs:
                mu[r, lo:hi], n[r, lo:hi] = CC.convert_nb_cov(
                    self.mu_sub[r, lo:hi], self.n_sub[r, lo:hi], self.w, self.stranded, rho)
        return mu, n


def rel_diff(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))) if a.size else 0.0


def _terms(y, mu, n) -> dict:
    """`deviance.target_terms` of one track: all its regions' bins as one vector."""
    return DV.target_terms(np.ravel(y), np.ravel(mu), np.ravel(n))


def _ratio(y, mu, n) -> float:
    return DV.calibration_ratio([_terms(y, mu, n)])


# ------------------------------------------------------------------ fit
def _geo_mean_other(c_by_assay: dict, assay: str) -> float:
    v = [c for a, c in c_by_assay.items() if a != assay and np.isfinite(c)]
    return float(np.exp(np.mean(np.log(v)))) if v else float("nan")


def cmd_fit(a) -> int:
    has_rs, has_ref = bool(a.arrays), bool(a.ref_pred_store)
    if not (has_rs or has_ref):
        print("fit: give --arrays, --ref-pred-store or both", file=sys.stderr)
        return 2
    tids = targets_of(a.truth, a.arrays)
    if not tids:
        print("fit: no target in both --truth and --arrays", file=sys.stderr)
        return 2
    by_assay = defaultdict(list)
    assay_of, stranded = {}, {}
    for tid in tids:
        with np.load(Path(a.truth) / f"{tid}.npz") as t:
            assay_of[tid], stranded[tid] = str(t["assay"]), bool(t["stranded"])
        by_assay[assay_of[tid]].append(tid)
    T = len(tids)
    pos = {t: i for i, t in enumerate(tids)}
    num = np.zeros((T, 2, 2, CC.NLAG))
    den = np.zeros((T, 2, 2, CC.NLAG))
    ratio_rs, ratio_ref = np.full(T, np.nan), np.full(T, np.nan)
    c_rs_loo, c_ref_loo = np.full(T, np.nan), np.full(T, np.nan)
    fallback = np.zeros(T, bool)
    c_rs_assay, c_ref_assay = {}, {}
    for assay in sorted(by_assay):
        recs_rs, recs_ref = {}, {}
        for tid in by_assay[assay]:
            g = Target(a, tid, a.fit_regions, need_ref=has_ref)
            i = pos[tid]
            if has_rs:
                num[i], den[i] = CC.estimate_rho(g.s, g.mu_sub, g.n_sub)
                ratio_rs[i] = _ratio(g.y, g.mu_rs, g.n_rs)
                recs_rs[tid] = {"y": g.y, "mu": g.mu_rs, "n": g.n_rs}
            if has_ref:
                ratio_ref[i] = _ratio(g.y, g.mu_ref, g.n_ref)
                recs_ref[tid] = {"y": g.y, "mu": g.mu_ref, "n": g.n_ref}
            print(f"fit: {tid} ({assay}) {a.fit_regions} ratio rs {ratio_rs[i]:.4f} "
                  f"ref {ratio_ref[i]:.4f}", flush=True)
        c_rs_assay[assay] = (CC.fit_scale(list(recs_rs.values()), iters=FIT_ITERS)
                             if has_rs else float("nan"))
        c_ref_assay[assay] = (CC.fit_scale(list(recs_ref.values()), iters=FIT_ITERS)
                              if has_ref else float("nan"))
        if len(by_assay[assay]) > 1:
            for tid in by_assay[assay]:
                if has_rs:
                    c_rs_loo[pos[tid]] = CC.fit_scale(
                        [r for k, r in recs_rs.items() if k != tid], iters=FIT_ITERS)
                if has_ref:
                    c_ref_loo[pos[tid]] = CC.fit_scale(
                        [r for k, r in recs_ref.items() if k != tid], iters=FIT_ITERS)
        del recs_rs, recs_ref
    for tid in tids:
        if len(by_assay[assay_of[tid]]) == 1:
            i = pos[tid]
            fallback[i] = True
            c_rs_loo[i] = _geo_mean_other(c_rs_assay, assay_of[tid])
            c_ref_loo[i] = _geo_mean_other(c_ref_assay, assay_of[tid])
    assays = sorted(by_assay)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, seed=np.array(int(a.seed)), targets=np.array(tids), assays_t=np.array(
            [assay_of[t] for t in tids]), stranded=np.array([stranded[t] for t in tids]),
        num=num, den=den, odd_ratio_rs=ratio_rs, odd_ratio_ref=ratio_ref,
        c_rs_loo=c_rs_loo, c_ref_loo=c_ref_loo, fallback=fallback, assays=np.array(assays),
        c_rs_assay=np.array([c_rs_assay[x] for x in assays]),
        c_ref_assay=np.array([c_ref_assay[x] for x in assays]),
        window_bins=np.array(int(a.window_bins)), infer_batch=np.array(int(a.infer_batch)),
        fit_regions=np.array(a.fit_regions), has_arrays=np.array(has_rs), has_ref=np.array(has_ref))
    print(f"fit: wrote {out} ({T} targets, {len(assays)} assays)", flush=True)
    return 0


# ------------------------------------------------------------------ rho choice
class Fit:
    def __init__(self, path):
        with np.load(path) as z:
            self.seed = int(z["seed"])
            self.targets = [str(x) for x in z["targets"]]
            self.assay = dict(zip(self.targets, (str(x) for x in z["assays_t"])))
            self.stranded = dict(zip(self.targets, (bool(x) for x in z["stranded"])))
            self.num, self.den = z["num"], z["den"]
            self.c_rs_loo = dict(zip(self.targets, z["c_rs_loo"].tolist()))
            self.c_ref_loo = dict(zip(self.targets, z["c_ref_loo"].tolist()))
            self.c_fallback = dict(zip(self.targets, z["fallback"].tolist()))
            self.window_bins, self.infer_batch = int(z["window_bins"]), int(z["infer_batch"])
            self.fit_regions = str(z["fit_regions"]) if "fit_regions" in z.files else "odd"
            self.has_arrays = bool(z["has_arrays"]) if "has_arrays" in z.files else True
            self.has_ref = bool(z["has_ref"]) if "has_ref" in z.files else True
        self.pos = {t: i for i, t in enumerate(self.targets)}

    def members(self, assay: str) -> list:
        return [t for t in self.targets if self.assay[t] == assay]

    def rho_sum(self, tids) -> np.ndarray:
        idx = [self.pos[t] for t in tids]
        return CC.finish_rho(self.num[idx].sum(0), self.den[idx].sum(0))

    def rho_for(self, target: str, mode: str) -> tuple:
        """(rho, ids it was fitted on, fallback flag); mode 'loo' or 'insample'."""
        assay = self.assay[target]
        mem = self.members(assay)
        if mode == "insample":
            return self.rho_sum(mem), mem, False
        others = [t for t in mem if t != target]
        if others:
            return self.rho_sum(others), others, False
        rest = sorted({self.assay[t] for t in self.targets if self.assay[t] != assay})
        same = [x for x in rest if self.stranded[self.members(x)[0]] == self.stranded[target]]
        use = same or rest
        if not use:
            raise SystemExit(f"{target}: the only target, no other assay to borrow rho from")
        rho = np.mean([self.rho_sum(self.members(x)) for x in use], 0)
        ids = [t for x in use for t in self.members(x)]
        return rho, ids, True


# ------------------------------------------------------------------ score
def other_half(which: str) -> str:
    return {"odd": "even", "even": "odd"}[which]


def _row(seed, g, regions, variant, terms, **extra) -> dict:
    r = {"seed": seed, "target": g.tid, "assay": g.assay, "stranded": g.stranded,
         "variant": variant, "regions": regions, "n_regions": int(len(g.ridx))}
    r.update({k: terms[k] for k in TERM_KEYS if k in terms})
    r["calibration_ratio"] = DV.calibration_ratio([terms])
    r.update(extra)
    return r


def cmd_score(a) -> int:
    fit = Fit(a.fit)
    if fit.seed != int(a.seed):
        print(f"score: --fit is seed {fit.seed}, not {a.seed}", file=sys.stderr)
        return 2
    if (fit.window_bins, fit.infer_batch) != (int(a.window_bins), int(a.infer_batch)):
        print("score: --window-bins / --infer-batch differ from the fit's", file=sys.stderr)
        return 2
    if not (fit.has_arrays and fit.has_ref):
        print("score: the fit needs both --arrays and --ref-pred-store", file=sys.stderr)
        return 2
    half = other_half(fit.fit_regions)
    tids = targets_of(a.truth, a.arrays)
    missing = sorted(set(tids) ^ set(fit.targets))
    if missing:
        print(f"score: targets differ from the fit's: {missing}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed = int(a.seed)
    tmp = out.with_name(f".{out.name}.part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for tid in tids:
            g = Target(a, tid, half)
            rows = []
            _, n_id = g.convert(CC.identity_rho())
            rows.append(_row(seed, g, half, "rs_independent", _terms(g.y, g.mu_rs, g.n_rs),
                             n_rel_diff_identity=rel_diff(n_id, g.n_rs)))
            for variant, mode in (("rs_cov_loo", "loo"), ("rs_cov_insample", "insample")):
                rho, ids, fb = fit.rho_for(tid, mode)
                mu, n = g.convert(rho)
                rows.append(_row(seed, g, half, variant, _terms(g.y, mu, n),
                                 rho_from=ids, fallback=fb,
                                 mu_rel_diff_stored=rel_diff(mu, g.mu_rs)))
            c = fit.c_rs_loo[tid]
            rows.append(_row(seed, g, half, "rs_scaled_loo",
                             _terms(g.y, g.mu_rs, CC.scale_size(g.n_rs, c)),
                             c=c, fallback=bool(fit.c_fallback[tid])))
            rows.append(_row(seed, g, half, "ref_stored", _terms(g.y, g.mu_ref, g.n_ref)))
            c = fit.c_ref_loo[tid]
            rows.append(_row(seed, g, half, "ref_scaled_loo",
                             _terms(g.y, g.mu_ref, CC.scale_size(g.n_ref, c)),
                             c=c, fallback=bool(fit.c_fallback[tid])))
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            print(f"score: {tid} " + " ".join(f"{r['variant']}={r['calibration_ratio']:.4f}"
                                              for r in rows), flush=True)
        for assay in sorted(set(fit.assay.values())):     # the fit's rho parts, for rho.tsv
            mem = fit.members(assay)
            idx = [fit.pos[t] for t in mem]
            fh.write(json.dumps({"kind": "rho", "seed": seed, "assay": assay,
                                 "n_targets": len(mem), "stranded": fit.stranded[mem[0]],
                                 "num": fit.num[idx].sum(0).tolist(),
                                 "den": fit.den[idx].sum(0).tolist()}) + "\n")
    tmp.replace(out)
    print(f"score: wrote {out}", flush=True)
    return 0


# ------------------------------------------------------------------ table
def macro_ratio(recs, assays=None) -> tuple:
    """(mean over assays of the per-assay calibration_ratio, number of assays); None when an
    assay of `assays` has no record."""
    per = defaultdict(list)
    for r in recs:
        per[r["assay"]].append(r)
    keep = sorted(per) if assays is None else sorted(assays)
    if any(x not in per for x in keep) or not keep:
        return None, len(keep)
    return float(np.mean([DV.calibration_ratio(per[x]) for x in keep])), len(keep)


def _flag(variant, recs) -> str:
    f = []
    if variant in ("rs_cov_loo", "rs_cov_insample"):
        f.append("residuals_include_mean_error")
    if variant == "rs_cov_insample":
        f.append("in_sample_upper_bound")
    nfb = len({r["assay"] for r in recs if r.get("fallback")})
    if nfb:
        f.append(f"fallback_assays={nfb}")
    return ",".join(f)


def _write_tsv(path: Path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _fmt(x) -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.6g}"


def cmd_table(a) -> int:
    recs = []
    for p in a.records:
        with open(p, encoding="utf-8") as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    rho_recs = [r for r in recs if r.get("kind") == "rho"]
    recs = [r for r in recs if r.get("kind") != "rho"]
    if not recs:
        print("table: no records", file=sys.stderr)
        return 2
    macros = {"all_assays": None}
    if a.scores_json:
        sj = json.loads(Path(a.scores_json).read_text(encoding="utf-8"))
        macros["scorable"] = list(sj["scorable"]["imputation"])
    seeds = sorted({int(r["seed"]) for r in recs})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        vr = [r for r in recs if r["variant"] == variant]
        for macro, assays in macros.items():
            per_seed = {}
            for s in seeds:
                val, _ = macro_ratio([r for r in vr if int(r["seed"]) == s], assays)
                per_seed[s] = val
            vals = [v for v in per_seed.values() if v is not None]
            ok = len(vals) == len(seeds) and vals
            mean = float(np.mean(vals)) if ok else None
            rng = float(max(vals) - min(vals)) if ok else None
            n_t = len({r["target"] for r in vr})
            rows.append([variant, macro, _fmt(mean), _fmt(rng),
                         ";".join(f"seed{s}:{_fmt(v)}" for s, v in per_seed.items()),
                         str(n_t), _flag(variant, vr)])
    _write_tsv(out / "table.tsv", ["variant", "macro", "calibration_ratio", "seed_range",
                                   "per_seed", "n_targets", "flag"], rows)
    rho_rows = []
    for rr in rho_recs:
        nm, dn = np.asarray(rr["num"], np.float64), np.asarray(rr["den"], np.float64)
        rho = CC.finish_rho(nm, dn)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(dn > 0, nm / np.where(dn > 0, dn, 1.0), np.nan)
        st = bool(rr["stranded"])
        ss = (0, 1) if st else (0,)
        lag0 = float(np.nanmean([raw[s, s, CC.MAXLAG] for s in ss]))
        for s in ss:
            for s2 in ss:
                for li in range(CC.NLAG):
                    rho_rows.append([rr["seed"], rr["assay"], rr["n_targets"], st, f"{s}{s2}",
                                     li - CC.MAXLAG, _fmt(float(rho[s, s2, li])),
                                     _fmt(float(raw[s, s2, li])), _fmt(lag0)])
    _write_tsv(out / "rho.tsv", ["seed", "assay", "n_targets", "stranded", "strand_pair", "lag",
                                 "rho", "raw_num_over_den", "lag0_raw"], rho_rows)
    if a.legB_table:
        with open(a.legB_table, encoding="utf-8") as fh:
            rd = list(csv.DictReader(fh, delimiter="\t"))
        keep = [r for r in rd if r.get("space") == "start_sum"
                and not str(r.get("model", "")).startswith(("rcp", "ref", "observed"))]
        if keep:
            _write_tsv(out / "native_start.tsv", list(keep[0].keys()),
                       [list(r.values()) for r in keep])
    print(f"table: wrote {out}/table.tsv ({len(rows)} rows)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "score"):
        p = sub.add_parser(name)
        p.add_argument("--truth", required=True)
        p.add_argument("--arrays", required=(name == "score"))
        p.add_argument("--ref-pred-store", required=(name == "score"),
                       help="prediction store of a count-output model")
        p.add_argument("--seed", required=True, type=int)
        p.add_argument("--out", required=True)
        p.add_argument("--window-bins", type=int, default=1024,
                       help="L of the read-start run (config context_bins; 1024)")
        p.add_argument("--infer-batch", type=int, default=8,
                       help="infer_batch of the read-start prediction store (STORE.json; 8)")
        if name == "score":
            p.add_argument("--fit", required=True)
        else:
            p.add_argument("--fit-regions", choices=("odd", "even"), default="odd",
                           help="the half of the regions the fit uses (index mod 2)")
    t = sub.add_parser("table")
    t.add_argument("--records", nargs="+", required=True)
    t.add_argument("--legB-table", dest="legB_table")
    t.add_argument("--scores-json")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "score": cmd_score, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
