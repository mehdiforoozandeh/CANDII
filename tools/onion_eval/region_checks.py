"""Region-sampling checks of the miniCANDI ablations (close-out item CO-REG, PI concern 2026-11-17).

    python tools/onion_eval/region_checks.py signal --evalset <evalset_p16.json> --out <dir> [--workers 4]
    -> <out>/region_signal_candidates.tsv (one row per candidate block), region_signal_eval256.tsv,
       region_signal_summary.tsv
    python tools/onion_eval/region_checks.py bootstrap --runs <runs_fir> --out <dir> [--n-boot 5000]
    -> <out>/reproduction.tsv, bootstrap_effects.tsv, bootstrap_base_floor.tsv, loro_regions.tsv

Check 1 (`signal`). Per candidate block of the eval set's genome mask (evalset.candidate_blocks;
the eval set's 256 regions were drawn from these), the share of bins in each stratum of the eval
set's masks.h5, the mean Umap k36 decile, and a direct signal measure read from the 43 distinct
imputation target tracks of the eval set (validation tracks, raw counts). Per track: the
"count enrichment" of a block is its mean count over the mean count of all candidate bins; a "high
bin" is a bin above the track's 90th percentile over all candidate bins, and the "high-bin
enrichment" of a block is its share of high bins over the share of high bins in all candidate
blocks. Per block, each measure is summarized by the median over the tracks (all tracks, and per
assay class of measurements.tsv).

Check 2 (`bootstrap`). Reads, per scored seed, `<run>/seed<s>/p16_sv/{pairs_per_region.npz,
same_view_per_region.npz, scores.json}` and the oracle reference that oracle_floor.find_oracle
picks. From the per-region sums it recomputes, for any non-negative weight vector over the 16 pair
regions, the two correction terms exactly as the scorer and select_rule compute them:
  (a) gap closed with the oracle reference on the headline correction pairs with no
      duplicate-keeping side (select_rule arm-summary `correction_v3_raw`; oracle_floor
      pair_records + macro(require_all=False)): per pair, CRPS = weighted sum of region sums over
      weighted sum of bins; per assay, (mean d - mean m) / (mean d - mean O); macro = mean over the
      correction-scorable assays present; undefined when one assay is undefined.
  (b) the correction-only term (score.correction_only_block): per pair type, 1 - (mean m - mean s)
      / (mean d - mean f), pair types whose relative denominator is below
      select_rule.CORRECTION_DENOM_REL_TOL left out, mean over the rest; undefined when a headline
      pair is undefined.
Weights of 1 on every region reproduce the stored terms (checked against oracle_floor.seed_terms
and scores.json). The paired bootstrap draws the 16 regions with replacement (fixed seed); the SAME
draw weights every seed of the arm and of the base. Effect = mean over arm seeds - mean over base
seeds. Leave-one-region-out sets one weight to 0. Imputation and calibration have no per-region
data (suffstats.npz pools them over the 256 regions), so they are not resampled.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

SEED = 20261117
STRATA_MAIN = ("promoters", "enhancers", "gene_bodies", "heterochromatin")
MODELS = {"convae": "rung2_base_heldout", "convattn": "rung3_base_heldout", "unet": "rung4_base_heldout"}
MODEL_NAMES = {"convae": "conv autoencoder", "convattn": "conv + attention", "unet": "U-Net"}
#: effect name -> (description, arm run name pattern per model; {m} = model suffix)
EFFECTS = {
    "scrambled_covariates": ("covariates scrambled across tracks",
                             "abl_scrambled_covariates_scrambled_covariates_{m}"),
    "views_base_only": ("training on base views only", "VIEWS:base_only"),
    "views_base_alignment": ("training on base + alignment views", "VIEWS:base_alignment"),
    "views_all_same_view_only": ("all views, same-view pairs only", "VIEWS:all_same_view_only"),
    "two_step_native": ("two-step correction, native base depth", "abl_two_step_two_step_native_{m}"),
    "two_step_fixed": ("two-step correction, fixed base depth", "abl_two_step_two_step_fixed_{m}"),
    "input_depth_fixed": ("fixed input depth", "abl_depth_input_input_depth_fixed_{m}"),
    "average_input": ("training-cell average as an extra input", "abl_average_input_average_input_{m}"),
    "loss_impute_x4": ("imputation loss weight 4", "abl_losses_loss_impute_x4_{m}"),
    "cov_numeric_only": ("numeric-only covariate embedding", "abl_cov_embedding_cov_numeric_only_{m}"),
    "tf_local_ratio": ("count transform local ratio", "abl_count_transforms_tf_local_ratio_{m}"),
    "tf_quantile": ("count transform quantile", "abl_count_transforms_tf_quantile_{m}"),
}
TERMS = ("gap_closed_crop_oracle", "correction_only")


def arm_run(effect: str, m: str) -> str:
    pat = EFFECTS[effect][1]
    if pat.startswith("VIEWS:"):
        v = pat.split(":", 1)[1]
        return f"rung4_views_{v}" if m == "unet" else f"abl_views_{v}_{m}"
    return pat.format(m=m)


def write_tsv(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def _local(p: str) -> str:
    return p.replace("/project/6014832/", "/project/def-maxwl/")


# ------------------------------------------------------------------------------------- check 1

def _track_block_stats(args):
    """(mid, per-block mean count [N], per-block high-bin share [N], candidate mean, high share, thr)."""
    mid, path, blocks, rb = args
    import h5py
    by = defaultdict(list)
    for i, (c, s) in enumerate(blocks):
        by[c].append((i, s))
    vals = np.empty((len(blocks), rb), np.float32)
    nvalid = np.full(len(blocks), rb, np.int64)     # bins that exist in the track (no zero fill)
    with h5py.File(path, "r") as f:
        for c, lst in by.items():
            a = f[c][:]
            for i, s in lst:
                seg = a[s:s + rb]
                vals[i, :len(seg)] = seg
                vals[i, len(seg):] = np.nan
                nvalid[i] = len(seg)
            del a
    flat = vals.reshape(-1)
    flat = flat[np.isfinite(flat)] if (nvalid < rb).any() else flat
    thr = float(np.partition(flat, int(0.9 * flat.size))[int(0.9 * flat.size)])
    high = vals > thr                                # NaN bins compare False
    gm = float(flat.mean(dtype=np.float64))
    gh = float(high.sum() / flat.size)
    bm = np.nansum(vals, axis=1, dtype=np.float64) / np.maximum(nvalid, 1)
    bh = high.sum(axis=1) / np.maximum(nvalid, 1)
    bm[nvalid == 0] = bh[nvalid == 0] = np.nan
    return (mid, bm, bh, gm, gh, thr, int((nvalid < rb).sum()))


def _quart(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {k: float("nan") for k in ("n", "min", "q1", "median", "q3", "max", "mean")}
    q = np.percentile(x, [0, 25, 50, 75, 100])
    return {"n": int(x.size), "min": q[0], "q1": q[1], "median": q[2], "q3": q[3], "max": q[4],
            "mean": float(x.mean())}


def signal(evalset, out, workers=4, log=print) -> None:
    import h5py
    from onion_eval import evalset as E
    ev = json.loads(Path(evalset).read_text(encoding="utf-8"))
    rb = int(ev["region_bins"])
    tables = Path(_local(ev["tables_dir"]))
    gdir = E.genome_dir(tables)
    cands = E.candidate_blocks(gdir, rb)
    if len(cands) != int(ev["n_candidate_regions"]):
        raise SystemExit(f"{len(cands)} candidate blocks, eval set says {ev['n_candidate_regions']}")
    regions = [[c, int(s)] for c, s in ev["regions"]]
    pos = {(c, s): i for i, (c, s) in enumerate(map(tuple, cands))}
    eval_idx = [pos[(c, s)] for c, s in regions]          # every eval region is a candidate block
    pair_set = {int(i) for i in ev["pair_regions"]}
    log(f"{len(cands)} candidate blocks; {len(regions)} eval regions; {len(pair_set)} pair regions")
    N = len(cands)
    rows = [{"block": i, "chrom": c, "start_bp": s * 25, "end_bp": (s + rb) * 25} for i, (c, s) in enumerate(cands)]
    eval_pos = {ci: k for k, ci in enumerate(eval_idx)}
    for i, r in enumerate(rows):
        k = eval_pos.get(i)
        r["eval_index"] = -1 if k is None else k
        r["in_eval256"] = int(k is not None)
        r["in_pair16"] = int(k is not None and k in pair_set)
    # strata
    with h5py.File(_local(ev["strata_path"]), "r") as mf:
        names = list(ev["strata"])
        genome = {}
        for st in names:
            share = np.full(N, np.nan)
            tot = n = 0
            for c in sorted({c for c, _ in cands}):
                a = np.asarray(mf[st][c][:], np.uint8)
                tot += int(a.sum())
                n += a.size
                for i, (cc, s) in enumerate(cands):
                    if cc == c:
                        share[i] = a[s:s + rb].mean()
            genome[st] = tot / n
            for i in range(N):
                rows[i][f"share_{st}"] = float(share[i])
        for i in range(N):
            d = [rows[i][f"share_mappability_d{k}"] for k in range(10)]
            rows[i]["mappability_mean_decile"] = (float(np.dot(range(10), d) / sum(d)) if sum(d) > 0
                                                  else float("nan"))
    # signal
    meas = {r["measurement_id"]: r for r in csv.DictReader(open(tables / "measurements.tsv"), delimiter="\t")}
    mids = sorted({t["target"] for t in ev["imputation_targets"]})
    jobs = [(m, meas[m]["counts_path"][3:], cands, rb) for m in mids]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as p:
            res = p.map(_track_block_stats, jobs)
    else:
        res = [_track_block_stats(j) for j in jobs]
    tracks = []
    ce = np.empty((len(res), N))
    he = np.empty((len(res), N))
    for k, (mid, bm, bh, gm, gh, thr, nshort) in enumerate(res):
        ce[k] = bm / gm if gm > 0 else np.nan
        he[k] = bh / gh if gh > 0 else np.nan
        r = meas[mid]
        tracks.append({"measurement_id": mid, "assay": r["assay"], "assay_class": r["assay_class"],
                       "cell": r["cell"], "candidate_mean_count": gm, "high_bin_threshold": thr,
                       "candidate_high_bin_share": gh, "n_blocks_past_track_end": nshort,
                       "pair16_count_enrichment_mean": float(np.mean(ce[k][[eval_idx[j] for j in sorted(pair_set)]])),
                       "eval256_count_enrichment_mean": float(np.mean(ce[k][eval_idx]))})
    write_tsv(Path(out) / "signal_tracks.tsv", tracks)
    classes = sorted({t["assay_class"] for t in tracks})
    for i in range(N):
        rows[i]["count_enrichment_median"] = float(np.median(ce[:, i]))
        rows[i]["high_bin_enrichment_median"] = float(np.median(he[:, i]))
        rows[i]["high_bin_enrichment_max"] = float(np.max(he[:, i]))
        for cl in classes:
            ks = [k for k, t in enumerate(tracks) if t["assay_class"] == cl]
            rows[i][f"high_bin_enrichment_{cl}"] = float(np.median(he[ks, i]))
    act = np.array([r["count_enrichment_median"] for r in rows])
    for i in range(N):
        rows[i]["count_enrichment_percentile_among_candidates"] = float((act < act[i]).mean() * 100)
    write_tsv(Path(out) / "region_signal_candidates.tsv", rows)
    write_tsv(Path(out) / "region_signal_eval256.tsv", sorted([r for r in rows if r["in_eval256"]],
                                                               key=lambda r: r["eval_index"]))
    sets = {"pair16": [r for r in rows if r["in_pair16"]], "eval256": [r for r in rows if r["in_eval256"]],
            "candidates": rows}
    meas_cols = ([f"share_{s}" for s in STRATA_MAIN] + ["mappability_mean_decile", "count_enrichment_median",
                 "high_bin_enrichment_median", "high_bin_enrichment_max"]
                 + [f"high_bin_enrichment_{cl}" for cl in classes]
                 + ["count_enrichment_percentile_among_candidates"])
    summ = []
    for col in meas_cols:
        for name, rs in sets.items():
            summ.append(dict({"measure": col, "set": name}, **_quart([r[col] for r in rs])))
        if col.startswith("share_"):
            summ.append({"measure": col, "set": "genome_chr_all_bins", "n": "", "min": "", "q1": "",
                         "median": "", "q3": "", "max": "", "mean": genome[col[len("share_"):]]})
    for thr in (0.1, 0.25, 0.5):
        for name, rs in sets.items():
            summ.append({"measure": f"n_blocks_high_bin_enrichment_max_below_{thr}", "set": name,
                         "n": len(rs), "mean": float(sum(r["high_bin_enrichment_max"] < thr for r in rs))})
            summ.append({"measure": f"n_blocks_count_enrichment_median_below_{thr}", "set": name,
                         "n": len(rs), "mean": float(sum(r["count_enrichment_median"] < thr for r in rs))})
    write_tsv(Path(out) / "region_signal_summary.tsv", summ,
              ["measure", "set", "n", "min", "q1", "median", "q3", "max", "mean"])
    log(f"wrote {out}")


# ------------------------------------------------------------------------------------- check 2

class Seed:
    """The per-region arrays of one scored seed, ready for weighted recomputation of both terms."""

    def __init__(self, d: Path, oracle_cache: dict):
        from onion_eval import oracle_floor as OF
        from onion_eval.select_rule import CORRECTION_DENOM_REL_TOL
        self.dir, self.tol = Path(d), CORRECTION_DENOM_REL_TOL
        s = json.loads((self.dir / "scores.json").read_text(encoding="utf-8"))
        self.scores = s
        od = OF.find_oracle(self.dir / "scores.json", s)
        if od is None:
            raise SystemExit(f"{d}: no oracle reference")
        key = str(od)
        if key not in oracle_cache:
            oracle_cache[key] = OF.load_oracle(od)
        o = oracle_cache[key]
        z = dict(np.load(self.dir / "pairs_per_region.npz", allow_pickle=False))
        self.region_indices = [int(i) for i in z["region_indices"]]
        if self.region_indices != [int(i) for i in o["region_indices"]]:
            raise SystemExit(f"{d}: pair regions differ from the oracle reference")
        scorable = set((s.get("scorable") or {}).get("correction") or [])
        # (a) oracle gap closed, headline pairs with no duplicate-keeping side
        rows = []
        for i, k in enumerate(z["pair_ids"]):
            if str(z["rung"][i]) != "":
                continue
            j = o["row"][str(k)]
            if not (int(z["never_trained"][i]) == 1 and int(z["subset_pair"][i]) == 0):
                continue
            if int(o["nodedup_side"][j]):
                continue
            why = str(o["oracle_reason"][j]) if "oracle_reason" in o else ""
            if str(z["assay"][i]) not in scorable:
                continue
            rows.append((i, j, str(z["assay"][i]), bool(why)))
        self.a_assays = sorted({r[2] for r in rows})
        self.a_group = np.array([self.a_assays.index(r[2]) for r in rows])
        I = [r[0] for r in rows]
        J = [r[1] for r in rows]
        self.a_m, self.a_d = z["crps_sum_model"][I], z["crps_sum_depth_only"][I]
        self.a_o = o["crps_sum_oracle"][J].astype(np.float64)
        self.a_o[[k for k, r in enumerate(rows) if r[3]]] = np.nan
        self.a_bins = z["bins"][I]
        if not np.array_equal(self.a_bins, o["bins"][J]):
            raise SystemExit(f"{d}: bins differ from the oracle reference")
        # (b) correction-only: headline correction pairs, all pair types
        sv = dict(np.load(self.dir / "same_view_per_region.npz", allow_pickle=False)) \
            if (self.dir / "same_view_per_region.npz").is_file() else None
        head = [i for i in range(len(z["pair_ids"])) if str(z["rung"][i]) == ""
                and int(z["never_trained"][i]) == 1 and int(z["subset_pair"][i]) == 0]
        self.sv_missing = sv is None             # reported in reproduction.tsv, never a silent NaN
        self.b_ok = sv is not None
        if self.b_ok:
            tpos = {str(t): k for k, t in enumerate(sv["target"])}
            miss = [i for i in head if str(z["target"][i]) not in tpos]
            self.b_ok = not miss
        if self.b_ok:
            self.b_knobs = sorted({str(z["knob"][i]) for i in head})
            self.b_group = np.array([self.b_knobs.index(str(z["knob"][i])) for i in head])
            self.b_m, self.b_d = z["crps_sum_model"][head], z["crps_sum_depth_only"][head]
            self.b_f, self.b_bins = z["crps_sum_floor"][head], z["bins"][head]
            T = [tpos[str(z["target"][i])] for i in head]
            self.b_s, self.b_sbins = sv["crps_sum_model"][T], sv["bins"][T]
            self.b_n = len(head)

    def term_a(self, W):
        """[B] gap closed (oracle, no duplicate-keeping side) for weight rows W [B, R]."""
        bw = W @ self.a_bins.T
        m, d, o = (W @ x.T / bw for x in (self.a_m, self.a_d, self.a_o))   # [B, P]
        G = len(self.a_assays)
        vals = np.empty((W.shape[0], G))
        for g in range(G):
            k = self.a_group == g
            mm, dd, oo = m[:, k].mean(1), d[:, k].mean(1), o[:, k].mean(1)
            den = dd - oo
            vals[:, g] = np.where(den > 0, (dd - mm) / np.where(den > 0, den, 1), np.nan)
        return vals.mean(1)                  # NaN when any assay is undefined (as the macro)

    def term_b(self, W):
        """[B] the correction-only term for weight rows W [B, R]."""
        if not self.b_ok:
            return np.full(W.shape[0], np.nan)
        bw = W @ self.b_bins.T
        m, d, f = (W @ x.T / bw for x in (self.b_m, self.b_d, self.b_f))
        s = W @ self.b_s.T / (W @ self.b_sbins.T)
        bad = ~(np.isfinite(m) & np.isfinite(d) & np.isfinite(f) & np.isfinite(s)).all(1)
        K = len(self.b_knobs)
        vals = np.full((W.shape[0], K), np.nan)
        used = np.zeros((W.shape[0], K), bool)
        for g in range(K):
            k = self.b_group == g
            mm, dd, ss, ff = m[:, k].mean(1), d[:, k].mean(1), s[:, k].mean(1), f[:, k].mean(1)
            den = dd - ff
            rel = np.where(dd > 0, den / np.where(dd > 0, dd, 1), np.nan)
            used[:, g] = np.isfinite(rel) & (rel >= self.tol)
            vals[:, g] = np.where(den > 0, 1 - (mm - ss) / np.where(den > 0, den, 1), np.nan)
        n_used = used.sum(1)
        out = np.where(n_used > 0, np.where(used, vals, 0).sum(1) / np.maximum(n_used, 1), np.nan)
        out[np.isnan(np.where(used, vals, 0)).any(1)] = np.nan
        out[bad] = np.nan
        return out

    def terms(self, W):
        return {"gap_closed_crop_oracle": self.term_a(W), "correction_only": self.term_b(W)}


def load_run(runs: Path, name: str, cache: dict) -> list:
    out = []
    for sd in sorted((runs / name).glob("seed*")):
        d = sd / "p16_sv"
        if (d / "scores.json").is_file() and (d / "pairs_per_region.npz").is_file():
            out.append(Seed(d, cache))
    return out


def stored_terms(seed: Seed) -> dict:
    """The stored terms: oracle_floor.seed_terms (what select_rule's table reads) and scores.json."""
    from onion_eval import oracle_floor as OF
    t = OF.seed_terms(seed.dir / "scores.json", seed.scores, "auto")
    blk = (t or {}).get("block") or {}
    a = (blk.get("no_nodedup_side") or {}).get("gap_closed_oracle")
    b = (seed.scores.get("correction_only_covariate_cost_closed") or {}).get("value")
    return {"gap_closed_crop_oracle": a, "correction_only": b}


def _draws(R: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, R, size=(n_boot, R))
    return np.stack([np.bincount(r, minlength=R) for r in idx]).astype(np.float64)


def _range(xs) -> float:
    """max - min over seeds; NaN (undefined) when any seed is undefined."""
    a = np.asarray(xs, float)
    return float(np.max(a) - np.min(a)) if a.size and np.isfinite(a).all() else float("nan")


def _mean_over_seeds(seeds, W, term):
    return np.mean([s.terms(W)[term] for s in seeds], axis=0)


def bootstrap(runs, out, n_boot=5000, log=print) -> None:
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    base = {m: load_run(runs, r, cache) for m, r in MODELS.items()}
    R = len(base["convae"][0].region_indices)
    regions = base["convae"][0].region_indices
    ones = np.ones((1, R))
    arms = {}
    for e in EFFECTS:
        for m in MODELS:
            arm = arm_run(e, m)
            arms[arm] = load_run(runs, arm, cache)
            if not arms[arm]:
                log(f"missing arm {arm}")
    # reproduction: every seed used (base and arms); undefined matches undefined
    rep = []
    worst, n_mismatch, n_checked = 0.0, 0, 0
    used = [(MODELS[m], s) for m, seeds in base.items() for s in seeds] + \
           [(a, s) for a, seeds in arms.items() for s in seeds]
    for name, s in used:
        st = stored_terms(s)
        mine = s.terms(ones)
        for t in TERMS:
            a, b = st[t], float(mine[t][0])
            a_def, b_def = a is not None and math.isfinite(float(a)), math.isfinite(b)
            diff = abs(float(a) - b) if a_def and b_def else (0.0 if a_def == b_def else float("inf"))
            n_checked += 1
            if not diff <= 1e-6:
                n_mismatch += 1
            if math.isfinite(diff):
                worst = max(worst, diff)
            note = ""
            if t == "correction_only" and s.sv_missing:
                note = "same_view_per_region.npz missing"
            elif not b_def:
                note = "undefined in both" if not a_def else "undefined in the recomputation only"
            rep.append({"run": name, "seed": s.dir.parent.name, "term": t, "stored": a,
                        "recomputed": b, "abs_diff": diff, "note": note})
            if t == "correction_only" and s.sv_missing:
                log(f"{name} {s.dir.parent.name}: same_view_per_region.npz missing")
    write_tsv(out / "reproduction.tsv", rep, ["run", "seed", "term", "stored", "recomputed", "abs_diff", "note"])
    log(f"reproduction: {n_checked} values over {len(used)} seeds; worst abs diff {worst:.3g}; "
        f"{n_mismatch} mismatches")
    if n_mismatch:
        raise SystemExit("stored terms not reproduced to 1e-6; stopping (reproduction.tsv)")
    W = _draws(R, n_boot, SEED)
    L = np.ones((R, R)) - np.eye(R)          # leave-one-region-out rows
    # base floor
    floor = []
    for m, seeds in base.items():
        for t in TERMS:
            full = float(_mean_over_seeds(seeds, ones, t)[0])
            bt = _mean_over_seeds(seeds, W, t)
            per = [float(s.terms(ones)[t][0]) for s in seeds]
            floor.append({"model": MODEL_NAMES[m], "run": MODELS[m], "term": t, "n_seeds": len(seeds),
                          "value_all_regions": full, "seed_range": _range(per),
                          "boot_lo": float(np.nanpercentile(bt, 2.5)), "boot_hi": float(np.nanpercentile(bt, 97.5)),
                          "boot_width": float(np.nanpercentile(bt, 97.5) - np.nanpercentile(bt, 2.5)),
                          "n_boot_undefined": int(np.isnan(bt).sum())})
    write_tsv(out / "bootstrap_base_floor.tsv", floor)
    # effects
    eff, loro = [], []
    for e in EFFECTS:
        for m in MODELS:
            arm = arm_run(e, m)
            aseeds = arms[arm]
            if not aseeds:
                continue
            if any(s.region_indices != regions for s in aseeds):
                raise SystemExit(f"{arm}: other pair regions")
            for t in TERMS:
                a_full = _mean_over_seeds(aseeds, ones, t)[0]
                b_full = _mean_over_seeds(base[m], ones, t)[0]
                obs = float(a_full - b_full)
                d = _mean_over_seeds(aseeds, W, t) - _mean_over_seeds(base[m], W, t)
                dl = _mean_over_seeds(aseeds, L, t) - _mean_over_seeds(base[m], L, t)
                ok = np.isfinite(d)
                aper = [float(s.terms(ones)[t][0]) for s in aseeds]
                bper = [float(s.terms(ones)[t][0]) for s in base[m]]
                row = {"effect": e, "description": EFFECTS[e][0], "model": MODEL_NAMES[m], "term": t,
                       "arm": arm, "n_seeds_arm": len(aseeds), "n_seeds_base": len(base[m]),
                       "arm_value": float(a_full), "base_value": float(b_full), "effect_all_regions": obs,
                       "seed_range_arm": _range(aper), "seed_range_base": _range(bper),
                       "boot_lo": float(np.percentile(d[ok], 2.5)) if ok.any() else float("nan"),
                       "boot_hi": float(np.percentile(d[ok], 97.5)) if ok.any() else float("nan"),
                       "sign_flip_share": (float(np.mean(np.sign(d[ok]) != np.sign(obs))) if ok.any()
                                           and math.isfinite(obs) else float("nan")),
                       "n_boot_undefined": int((~ok).sum()),
                       "loro_min": float(np.nanmin(dl)) if np.isfinite(dl).any() else float("nan"),
                       "loro_max": float(np.nanmax(dl)) if np.isfinite(dl).any() else float("nan")}
                sr = float(np.max([row["seed_range_arm"], row["seed_range_base"]]))   # NaN if a seed is undefined
                row["beyond_seed_range"] = int(math.isfinite(obs) and math.isfinite(sr) and abs(obs) > sr)
                row["region_robust"] = int(ok.any() and math.isfinite(obs) and (row["boot_lo"] > 0 or row["boot_hi"] < 0)
                                           and row["sign_flip_share"] <= 0.05)
                if np.isfinite(dl).any() and math.isfinite(obs):
                    k = int(np.nanargmax(np.abs(dl - obs)))
                    row["loro_most_influential_region"] = regions[k]
                    row["loro_effect_without_it"] = float(dl[k])
                    row["loro_sign_flips"] = int(np.sum(np.sign(dl[np.isfinite(dl)]) != np.sign(obs)))
                eff.append(row)
                for k in range(R):
                    loro.append({"effect": e, "model": MODEL_NAMES[m], "term": t, "dropped_region": regions[k],
                                 "effect_without_region": float(dl[k])})
            log(f"{arm} done")
    cols = list(dict.fromkeys(k for r in eff for k in r))
    write_tsv(out / "bootstrap_effects.tsv", eff, cols)
    write_tsv(out / "loro_regions.tsv", loro)
    log(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("signal")
    a.add_argument("--evalset", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=4)
    b = sub.add_parser("bootstrap")
    b.add_argument("--runs", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n-boot", type=int, default=5000)
    x = ap.parse_args(argv)
    Path(x.out).mkdir(parents=True, exist_ok=True)
    if x.cmd == "signal":
        signal(x.evalset, x.out, x.workers)
    else:
        bootstrap(x.runs, x.out, x.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
