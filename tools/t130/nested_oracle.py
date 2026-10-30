"""t130 — the nested-read oracle for the t112 depth pairs (PI ruling 2026-10-30, plan_gaps.md A9).

WHY. A t112 depth arm (15M, 7.5M, 3.75M) is a read subsample of its track's base tagAlign (30M;
the DNase base C12M02 50M), so an arm and its base SHARE reads. Two independent pseudoreplicate
halves (the t137 floor) share none, so the floor is no upper bound on how well the base predicts a
nested arm. The top of the gap-closed interval for a depth pair is therefore this oracle: the exact
predictive of the target given the source under the way the arm was made.

PROVENANCE (verified 2026-10-30; the slip of chunk G21 records it):
  * `tools/t112/bam_arm.py::plan_commands` (origin/main), arm `depth`: one job per arm runs the
    pipeline's own `encode_task_subsample_ctl.py <base 30M/50M tagAlign> --subsample <N>` (`shuf`
    seeded by the input's uncompressed byte count). Nibi provenance.json of C19M16__depth__{15M,
    7.5M,3.75M}: all three read `/scratch/mforooz/t112_cf/ta/ENCFF254LWX.merged.srt.nodup.30M.
    tagAlign.gz` (987786676 bytes) in three separate jobs (22177545-7).
  * Each arm is a subset of the base: on chr21 of C19M16, C40M17, C12M02 no bin has arm > base.
  * The arms are NOT chained: 3-6 % of chr21 bins have 3.75M > 7.5M, 7.5M > 15M, 3.75M > 15M.
    They are independent subsamples of the same base.
  * The subsample is drawn from the tagAlign MACS2 consumed, i.e. AFTER filtering and dedup (the
    `nodup` BAM -> bam2ta), and `counts25.npz` bins that tagAlign (`bin25.py`: each read adds 1 to
    every 25 bp bin it overlaps). So per bin the arm count given the base count b is
    Hypergeometric(N_base reads, b, N_arm), which is Binomial(b, r), r = N_arm / N_base, up to the
    without-replacement factor (N - b) / (N - 1) ~ 1. The dispersion check reads how exact it is.

THE PREDICTIVES (counts space only; -log10 p cannot be thinned). rho = a product's depth relative
to its base (base 1; an arm depth / base depth). Source s, target t, per bin source count x:
  binomial          base -> arm. T = Binomial(x, rho_t). Exact (up to the hypergeometric factor).
  poisson_remainder arm -> base. T = x + Poisson(x (1 - rho_s) / rho_s): the base = the arm plus the
                    reads the arm did not keep, approximated as Poisson with the rate the arm
                    implies. APPROXIMATE: at x = 0 it is a point mass at 0 although the base can
                    hold reads there. Reported; not the oracle analyze.py reads.
  eb_binomial       arm -> base. The exact posterior P(b | x) ~ pi(b) Binomial(x; b, rho_s) with
                    pi = the base's histogram on the TRAINING chromosomes (every chromosome but
                    chr19, chr21, chr22). Exact under binomial thinning given that prior. This is
                    the arm -> base oracle analyze.py reads: on true thinning at rho 0.125 the
                    poisson_remainder sits ABOVE the one-warp (the interval inverts again), while
                    eb_binomial does not (tests/test_t130_nested_oracle.py).
  base_mediated     arm -> arm (independent subsamples of one base). T = Binomial(x, rho_t) +
                    Poisson(x rho_t (1 - rho_s) / rho_s): the poisson_remainder base, thinned.
                    APPROXIMATE for the same reason.
`binomial`, `poisson_remainder` and `base_mediated` are one family, T = Binomial(x, q) + Poisson(x mu),
q = rho_t, mu = rho_t (1 - rho_s) / rho_s. Every rho is computed two ways (`r_source`): `manifest`
(the MANIFEST `depth` column) and `empirical` (the arm's total count over the base's, every main
chromosome of the npz).

CRPS of a discrete predictive on 0, 1, 2, ..., exactly: CRPS(F, y) = sum_k (F(k) - 1{k >= y})^2,
summed over the support 0..K, K = max(the largest y of the group, mean + 12 sd + 10) (the mass past K
is < 1e-30). Bins are grouped by their source count x (the predictive depends on x only), so each
distinct x builds one CDF. No PRED_FLOOR: x = 0 gives a point mass at 0, exact for base -> arm.

THE SPLIT. `crps_oracle_scaled` / `scale_error` / `c_star` by the same search as
`baseline_rungs.scale_split` (c in [-6, 6] step 0.25, then +-0.25 step 0.01, on the same 20 000-bin
subsample, seed 0; both CRPS on every bin of the subset). Scaling the predictive mean by 2^c scales
the depth ratio rho = rho_t / rho_s: q' = min(q 2^c, 1), mu' = max(rho 2^c - q', 0), so the mean
x rho 2^c is kept exactly; eb_binomial scales its thinning ratio, rho_s' = min(rho_s / 2^c, 1).

SCORED BINS: chr19 + chr21, blacklist bins removed (`baseline_rungs.load_eval`, the rungs' rule);
subsets `all` and `top1` (`baseline_rungs.subset_index`, top 1 % ranked by the TARGET).

DISPERSION CHECK per (track, depth arm): on the scored bins, x = base, y = arm, grouped by base
count (0, 1, 2, 3-4, 5-8, ..., >= 129): Var(y - rho x) over its binomial r(1 - r) x mean, for both
rho sources (1 = binomial; < 1 = under-dispersed, as the hypergeometric factor makes it); the mean
ratio y / (rho x); and `n_arm_gt_base` (0 if the arm is nested in the base).

    python tools/t130/nested_oracle.py tracks <manifest>
    python tools/t130/nested_oracle.py run <manifest> <products> <blacklist> <out> <index>
    python tools/t130/nested_oracle.py aggregate <manifest> <out>
`run` writes <out>/parts/<track>.oracle.tsv, <track>.dispersion.tsv, <track>.DONE (atomic);
`aggregate` (stdlib) concatenates them into <out>/nested_oracle.tsv, nested_dispersion.tsv and
NESTED_DONE, and fails if a track is missing.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "t118"))   # baseline_rungs, imported lazily (needs candi)
SCORE_CHROMS = ("chr19", "chr21")
HELD_OUT = frozenset({"chr19", "chr21", "chr22"})
SUBSETS = ("all", "top1")
SCALE_FIT_BUDGET = 20_000            # baseline_rungs.SCALE_FIT_BUDGET
SCALE_SEED = 0                       # baseline_rungs.SCALE_SEED
DISP_EDGES = (0, 1, 2, 3, 5, 9, 17, 33, 65, 129)   # base-count bin lower edges
MARK_CLASS = {"DNase-seq": "DNase", "H3K27ac": "narrow", "H3K4me3": "narrow", "H3K4me1": "narrow",
              "H3K27me3": "broad", "H3K36me3": "broad", "H3K9me3": "broad"}
PREDICTIVES = {"base_to_arm": ("binomial",), "arm_to_base": ("poisson_remainder", "eb_binomial"),
               "arm_to_arm": ("base_mediated",)}
OCOLS = ["space", "eval", "track", "assay", "mark_class", "pid", "direction", "source_pid",
         "target_pid", "predictive", "exact", "r_source", "rho_src", "rho_tgt", "ratio", "n_all",
         "n_top1"] + [f"{k}_{s}" for s in SUBSETS
                      for k in ("crps", "crps_oracle_scaled", "scale_error", "c_star")]
DCOLS = ["track", "pid", "rho_manifest", "rho_empirical", "base_lo", "base_hi", "n_bins",
         "mean_base", "mean_arm", "mean_ratio_manifest", "var_obs_manifest", "var_binom_manifest",
         "dispersion_manifest", "var_obs_empirical", "var_binom_empirical",
         "dispersion_empirical", "n_arm_gt_base"]


# ---------------------------------------------------------------------------------------------
# the exact CRPS of a discrete predictive
# ---------------------------------------------------------------------------------------------


def crps_from_cdf(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sum_k (F(k) - 1{k >= y})^2 over k = 0..len(F)-1, for integer 0 <= y <= len(F)-1.

    = sum_{k < y} F(k)^2 + sum_{k >= y} (1 - F(k))^2, read from two prefix sums.
    """
    F = np.clip(np.asarray(F, np.float64), 0.0, 1.0)
    a = np.concatenate([[0.0], np.cumsum(F * F)])
    b = np.concatenate([np.cumsum(((1.0 - F) ** 2)[::-1])[::-1], [0.0]])
    y = np.asarray(y, np.int64)
    return a[y] + b[y]


def _support(mean: float, var: float, ymax: int) -> int:
    return int(max(ymax, math.ceil(mean + 12.0 * math.sqrt(max(var, 0.0)) + 10.0)))


def family_pmf(x: int, q: float, mu: float, K: int) -> np.ndarray:
    """PMF on 0..K of Binomial(x, q) + Poisson(x mu)."""
    from scipy.signal import fftconvolve
    from scipy.stats import binom, poisson
    k = np.arange(K + 1)
    if x == 0:
        out = np.zeros(K + 1)
        out[0] = 1.0
        return out
    if q >= 1.0:
        pb = np.zeros(x + 1)
        pb[x] = 1.0
    elif q <= 0.0:
        pb = np.zeros(1)
        pb[0] = 1.0
    else:
        pb = binom.pmf(np.arange(x + 1), x, q)
    lam = x * mu
    if lam <= 0.0:
        out = np.zeros(K + 1)
        n = min(pb.size, K + 1)
        out[:n] = pb[:n]
        return out
    pp = poisson.pmf(k, lam)
    nz = np.flatnonzero(pb > 0)
    if nz.size == 1:                                  # a shifted Poisson
        out = np.zeros(K + 1)
        s = int(nz[0])
        out[s:] = pp[:K + 1 - s] * pb[s]
        return out
    return np.clip(fftconvolve(pb, pp)[:K + 1], 0.0, None)


def family_cdf(x: int, q: float, mu: float, ymax: int) -> np.ndarray:
    mean = x * (q + mu)
    var = x * q * (1.0 - q) + x * mu
    K = _support(mean, var, ymax)
    return np.cumsum(family_pmf(x, q, mu, K))


def _groups(x: np.ndarray):
    """(distinct values, order, starts): bins sorted by x, one slice per distinct value."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    u, starts = np.unique(xs, return_index=True)
    return u, order, np.append(starts, xs.size)


def grouped_crps(x: np.ndarray, y: np.ndarray, cdf_of) -> np.ndarray:
    """Per-bin CRPS where the predictive depends on the source count only: cdf_of(x, ymax)."""
    x = np.asarray(x, np.int64)
    y = np.asarray(y, np.int64)
    out = np.empty(x.size, np.float64)
    u, order, bounds = _groups(x)
    for i, v in enumerate(u):
        idx = order[bounds[i]:bounds[i + 1]]
        yy = y[idx]
        F = cdf_of(int(v), int(yy.max()))
        if F.size <= yy.max():                         # never: the support covers ymax
            F = np.concatenate([F, np.ones(int(yy.max()) + 1 - F.size)])
        out[idx] = crps_from_cdf(F, yy)
    return out


def family_crps(x, y, rho_src: float, rho_tgt: float, c: float = 0.0) -> np.ndarray:
    """CRPS of Binomial(x, q') + Poisson(x mu') against y, the depth ratio scaled by 2^c."""
    rho = rho_tgt / rho_src * 2.0 ** c
    q = min(rho_tgt * 2.0 ** c, 1.0)
    mu = max(rho - q, 0.0)
    return grouped_crps(x, y, lambda v, ymax: family_cdf(v, q, mu, ymax))


def eb_crps(x, y, rho_src: float, prior: np.ndarray, c: float = 0.0) -> np.ndarray:
    """CRPS of the posterior P(b | x) ~ prior(b) Binomial(x; b, rho') (arm -> base), rho' scaled
    by 2^-c. A source with no prior support (x above every base count) falls back to x."""
    from scipy.stats import binom
    r = min(max(rho_src / 2.0 ** c, 1e-9), 1.0)
    b = np.flatnonzero(prior > 0)
    lp = np.log(prior[b])

    def cdf_of(v, ymax):
        K = max(int(b[-1]), ymax)
        keep = b >= v
        out = np.zeros(K + 1)
        if not keep.any():
            out[v if v <= K else K] = 1.0
            return np.cumsum(out)
        w = lp[keep] + binom.logpmf(v, b[keep], r)
        w = np.exp(w - w.max())
        out[b[keep]] = w / w.sum()
        return np.cumsum(out)
    return grouped_crps(x, y, cdf_of)


def split(x, y, crps_at) -> dict:
    """crps, crps_oracle_scaled, scale_error, c_star: baseline_rungs.scale_split's search.

    crps_at(x, y, c) returns the per-bin CRPS of the predictive with its mean scaled by 2^c.
    The subsample is drawn exactly as scale_split draws it (same budget, same seed, same call).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    crps = float(np.mean(crps_at(x, y, 0.0)))
    sel = (np.arange(x.size) if x.size <= SCALE_FIT_BUDGET
           else np.random.default_rng(SCALE_SEED).choice(x.size, SCALE_FIT_BUDGET, replace=False))
    xf, yf = x[sel], y[sel]

    def fit(c: float) -> float:
        return float(np.mean(crps_at(xf, yf, c)))

    c = float(min(np.arange(-6.0, 6.001, 0.25), key=fit))
    c = float(min(np.arange(c - 0.25, c + 0.2501, 0.01), key=fit))
    cs = float(np.mean(crps_at(x, y, c)))
    return {"crps": crps, "crps_oracle_scaled": cs, "scale_error": crps - cs, "c_star": c}


# ---------------------------------------------------------------------------------------------
# the dispersion check
# ---------------------------------------------------------------------------------------------


def dispersion(base: np.ndarray, arm: np.ndarray, rho_m: float, rho_e: float) -> list[dict]:
    x = np.asarray(base, np.float64)
    y = np.asarray(arm, np.float64)
    rows = []
    edges = list(DISP_EDGES) + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        r = {"base_lo": lo, "base_hi": ("inf" if hi == np.inf else int(hi - 1)),
             "n_bins": int(m.sum()), "n_arm_gt_base": int((y[m] > x[m]).sum())}
        if m.sum() < 2:
            rows.append(r)
            continue
        xm, ym = x[m], y[m]
        r.update(mean_base=float(xm.mean()), mean_arm=float(ym.mean()),
                 mean_ratio_manifest=(float(ym.mean() / (rho_m * xm.mean()))
                                      if xm.mean() > 0 else float("nan")))
        for tag, rho in (("manifest", rho_m), ("empirical", rho_e)):
            e = ym - rho * xm
            vo = float(e.var())
            vb = float(np.mean(rho * (1.0 - rho) * xm))
            r[f"var_obs_{tag}"] = vo
            r[f"var_binom_{tag}"] = vb
            r[f"dispersion_{tag}"] = vo / vb if vb > 0 else float("nan")
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------
# the manifest: tracks and depth pairs
# ---------------------------------------------------------------------------------------------


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def depth_tracks(rows) -> list[str]:
    """The tracks with at least one depth arm, sorted; the list index is the array index."""
    return sorted({r["track"] for r in rows if r["arm"] == "depth"})


def depth_pairs(rows, track: str) -> list[dict]:
    """base -> arm, arm -> base for every depth arm, and arm -> arm for every ordered arm pair."""
    tr = [r for r in rows if r["track"] == track]
    base = [r for r in tr if r["arm"] == "base"]
    if len(base) != 1:
        raise ValueError(f"track {track}: {len(base)} base rows")
    base = base[0]
    arms = sorted((r for r in tr if r["arm"] == "depth"), key=lambda r: r["pid"])
    rho = {base["pid"]: 1.0}
    for a in arms:
        rho[a["pid"]] = float(a["depth"]) / float(base["depth"])
    out = []
    for a in arms:
        out.append({"pid": a["pid"], "direction": "base_to_arm", "source_pid": base["pid"],
                    "target_pid": a["pid"]})
        out.append({"pid": a["pid"], "direction": "arm_to_base", "source_pid": a["pid"],
                    "target_pid": base["pid"]})
    for a in arms:
        for b in arms:
            if a["pid"] != b["pid"]:
                out.append({"pid": f"{a['pid']}->{b['pid']}", "direction": "arm_to_arm",
                            "source_pid": a["pid"], "target_pid": b["pid"]})
    for p in out:
        p.update(track=track, assay=base["assay"], mark_class=MARK_CLASS[base["assay"]],
                 base_pid=base["pid"], rho_manifest_src=rho[p["source_pid"]],
                 rho_manifest_tgt=rho[p["target_pid"]])
    return out


# ---------------------------------------------------------------------------------------------
# run one track
# ---------------------------------------------------------------------------------------------


def _write_tsv(path: Path, cols, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", restval="NA",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if v is None or (isinstance(v, float) and not math.isfinite(v))
                            else (f"{v:.10g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    os.replace(tmp, path)


def score_predictive(x, y, crps_at) -> dict:
    import baseline_rungs as br
    sub = br.subset_index(np.asarray(y))
    out = {}
    for s in SUBSETS:
        idx = sub[s]
        out[f"n_{s}"] = int(idx.size)
        sp = split(x[idx], y[idx], crps_at)
        for k, v in sp.items():
            out[f"{k}_{s}"] = v
    return out


def run_track(manifest, products, blacklist_path, out, index: int) -> Path:
    import baseline_rungs as br                      # imports candi: the kit venv
    rows = read_manifest(manifest)
    tracks = depth_tracks(rows)
    if not 0 <= index < len(tracks):
        raise SystemExit(f"index {index} outside 0..{len(tracks) - 1}")
    track = tracks[index]
    pairs = depth_pairs(rows, track)
    blacklist = br.read_blacklist(blacklist_path)
    products = Path(products)
    pids = sorted({p["source_pid"] for p in pairs} | {p["target_pid"] for p in pairs})
    base_pid = pairs[0]["base_pid"]
    totals, prior = {}, None
    for pid in pids:                                  # empirical totals; the base's prior
        with br.Track(products / pid, "counts") as t:
            tot = 0
            for c in t.chroms:
                a = t.get(c)
                tot += int(a.sum(dtype=np.int64))
                if pid == base_pid and c not in HELD_OUT:
                    h = np.bincount(a)
                    prior = h if prior is None else (
                        np.pad(prior, (0, max(0, h.size - prior.size)))
                        + np.pad(h, (0, max(0, prior.size - h.size))))
            totals[pid] = tot
    prior = prior.astype(np.float64) / prior.sum()
    rho_e = {pid: totals[pid] / totals[base_pid] for pid in pids}
    orows, drows = [], []
    for p in pairs:
        with br.Track(products / p["source_pid"], "counts") as src, \
                br.Track(products / p["target_pid"], "counts") as tgt:
            d = br.load_eval(src, tgt, list(SCORE_CHROMS), blacklist)
        x, y = d["x"].astype(np.int64), d["y"].astype(np.int64)
        for r_source in ("manifest", "empirical"):
            rs = p["rho_manifest_src"] if r_source == "manifest" else rho_e[p["source_pid"]]
            rt = p["rho_manifest_tgt"] if r_source == "manifest" else rho_e[p["target_pid"]]
            for pred in PREDICTIVES[p["direction"]]:
                if pred == "eb_binomial":
                    def crps_at(xx, yy, c, rs=rs):
                        return eb_crps(xx, yy, rs, prior, c)
                else:
                    def crps_at(xx, yy, c, rs=rs, rt=rt):
                        return family_crps(xx, yy, rs, rt, c)
                orows.append({"space": "counts", "eval": "score",
                              **{k: p[k] for k in ("track", "assay", "mark_class", "pid",
                                                   "direction", "source_pid", "target_pid")},
                              "predictive": pred,
                              "exact": int(pred in ("binomial", "eb_binomial")),
                              "r_source": r_source, "rho_src": rs, "rho_tgt": rt,
                              "ratio": rt / rs, **score_predictive(x, y, crps_at)})
        if p["direction"] == "base_to_arm":
            for r in dispersion(x, y, p["rho_manifest_tgt"], rho_e[p["target_pid"]]):
                drows.append({"track": track, "pid": p["pid"],
                              "rho_manifest": p["rho_manifest_tgt"],
                              "rho_empirical": rho_e[p["target_pid"]], **r})
        print(f"{track} {p['pid']} {p['direction']} done", flush=True)
    out = Path(out) / "parts"
    _write_tsv(out / f"{track}.oracle.tsv", OCOLS, orows)
    _write_tsv(out / f"{track}.dispersion.tsv", DCOLS, drows)
    (out / f"{track}.DONE").write_text(f"{len(orows)} oracle rows, {len(drows)} dispersion rows\n")
    return out / f"{track}.oracle.tsv"


def aggregate(manifest, out) -> int:
    out = Path(out)
    tracks = depth_tracks(read_manifest(manifest))
    missing = [t for t in tracks if not (out / "parts" / f"{t}.DONE").is_file()]
    if missing:
        print(f"missing tracks: {missing}", file=sys.stderr)
        return 1
    n = {}
    for kind, dest in (("oracle", "nested_oracle.tsv"), ("dispersion", "nested_dispersion.tsv")):
        rows = []
        for t in tracks:
            with (out / "parts" / f"{t}.{kind}.tsv").open(newline="", encoding="utf-8") as fh:
                rows += list(csv.DictReader(fh, delimiter="\t"))
        _write_tsv(out / dest, OCOLS if kind == "oracle" else DCOLS, rows)
        n[kind] = len(rows)
    (out / "NESTED_DONE").write_text(f"{len(tracks)} tracks, {n['oracle']} oracle rows, "
                                     f"{n['dispersion']} dispersion rows\n")
    print(f"{len(tracks)} tracks -> {out}/nested_oracle.tsv ({n['oracle']} rows), "
          f"nested_dispersion.tsv ({n['dispersion']} rows)")
    return 0


def main(argv) -> int:
    usage = ["usage: nested_oracle.py tracks <manifest>",
             "       nested_oracle.py run <manifest> <products> <blacklist> <out> <index>",
             "       nested_oracle.py aggregate <manifest> <out>"]
    if not argv:
        print("\n".join(usage), file=sys.stderr)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "tracks" and len(args) == 1:
        for i, t in enumerate(depth_tracks(read_manifest(args[0]))):
            print(f"{i}\t{t}")
        return 0
    if cmd == "run" and len(args) == 5:
        print(run_track(*args[:4], int(args[4])))
        return 0
    if cmd == "aggregate" and len(args) == 2:
        return aggregate(*args)
    print("\n".join(usage), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
