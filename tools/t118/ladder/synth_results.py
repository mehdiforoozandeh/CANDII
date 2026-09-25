"""t118 ladder — a complete FAKE aggregation directory, to develop and test the figures and report.

    python tools/t118/ladder/synth_results.py <out_dir> [--seed 0]

Writes, under `<out_dir>` (an `agg_dir` as `aggregate.py` would leave it):
  results.json            the pinned results schema (plan t118-C3), all four rungs, both versions
                          of g (per_track, across), both spaces, the three models, seeds 0 1 2
  checks_<rung>.json      {"rung", "checks": [...]} per rung;  checks_main.json
  qm_curves.json          {space: {track: {arm_pid: {"knots_x", "knots_y"}}}} (figure 5B)
  runs/<run_name>/        timing.json for every present run; for the real model at seed 0 of the
                          per-track g's also figdata.npz, scores.json (snippet loci, timing only)
                          and config.json

Every number is invented. The generator follows the t112 structure (7 tracks, 10 arms, 19 arm
products per histone track, 7 usable for DNase) and the aggregation rules of plan t118-C3 (mean
over pairs per track, macro over the class's tracks, seed wobble = max pairwise |delta| over the
seeds), so the checks and the rung choice are consistent with the per-class table. To keep the
file small, `per_pair` holds every trained/score record but val, shuffle and swap records only for
the real model, and law records only for the DNase track.

numpy + stdlib only: runs under the candii env and under a plain python with numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

RUNGS = ("A", "B", "C", "D")
MODELS = ("real", "nocov", "ids")
SPACES = ("counts", "pval")
SEEDS = (0, 1, 2)
TRACKS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18")
G_IDS = TRACKS + ("all",)
METRICS = ("crps_all", "crps_nonzero", "crps_top1", "spearman_all", "spearman_nonzero",
           "spearman_top1")
TRACK_ASSAY = {"C07M20": "H3K4me1", "C07M29": "H3K9me3", "C12M02": "DNase-seq",
               "C19M16": "H3K27ac", "C19M22": "H3K4me3", "C40M17": "H3K27me3",
               "C40M18": "H3K36me3"}
MARK_CLASS = {"DNase-seq": "DNase", "H3K27ac": "narrow", "H3K4me3": "narrow",
              "H3K4me1": "narrow", "H3K27me3": "broad", "H3K36me3": "broad",
              "H3K9me3": "broad"}
HISTONE_ARMS = {"abproxy": ("f0.5", "f0.9"), "crop": ("36", "50"), "ctldepth": ("q0.25", "q0.5"),
                "ctlid": ("none", "other"), "dedup": ("off",), "depth": ("15M", "3.75M", "7.5M"),
                "extsize": ("k0.5", "k2"), "mapq": ("0", "10"), "pe": ("pe",),
                "ratio": ("k0.5", "k2")}
#: the DNase MAPQ arms are excluded (byte-identical to the base, a t112 defect)
DNASE_ARMS = {"dedup": ("off",), "depth": ("15M", "3.75M", "7.5M"), "extsize": ("k0.5", "k2"),
              "pe": ("pe",)}
P_ONLY_ARMS = {"ratio", "ctlid", "ctldepth", "extsize"}
BASE_DEPTH = 30e6
#: noSolution CRPS (all, top1) and Spearman (all) per (mark class, space) — the plan's v2 table
D0 = {("DNase", "counts"): (2.63, 107.3, 0.67), ("narrow", "counts"): (0.81, 9.45, 0.40),
      ("broad", "counts"): (0.81, 2.99, 0.40), ("DNase", "pval"): (1.35, 74.7, 0.62),
      ("narrow", "pval"): (0.40, 13.6, 0.38), ("broad", "pval"): (0.29, 1.76, 0.36)}
#: CRPS multiplier over noSolution, per (kind, model) and rung for the real model
FACTOR_REAL = {"trained": {"A": 0.80, "B": 0.76, "C": 0.745, "D": 0.74},
               "law": {"A": 0.84, "B": 0.80, "C": 0.79, "D": 0.788},
               "shuffle": {"A": 0.875, "B": 0.87, "C": 0.87, "D": 0.87}}
FACTOR_TWIN = {("trained", "nocov"): 0.88, ("law", "nocov"): 0.90,
               ("law", "ids"): 0.895, ("shuffle", "nocov"): 0.88, ("shuffle", "ids"): 0.88}
MISSING_RUNS = ("D_C07M29_pval_ids_s2", "D_all_pval_ids_s2")
KNOTS_X = {"counts": [0.0, 0.35, 0.69, 1.10, 1.39, 1.79, 2.08, 2.40, 2.77, 3.04, 3.30, 3.90],
           "pval": [-6.91, -3.0, -2.0, -1.2, -0.7, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.5]}


# ---------------------------------------------------------------------------------------------
# products and pairs (the t112 structure, restated so this file needs no manifest)
# ---------------------------------------------------------------------------------------------


def _arm_products(track):
    arms = DNASE_ARMS if TRACK_ASSAY[track].startswith("DNase") else HISTONE_ARMS
    prods = [{"pid": f"{track}__{a}__{lv}", "track": track, "arm": a, "level": lv}
             for a, lvs in arms.items() for lv in lvs]
    return sorted(prods, key=lambda p: p["pid"])


def _base(track):
    return {"pid": f"{track}__base__base", "track": track, "arm": "base", "level": "base"}


def _counts_same_as_base(p):
    return p["arm"] == "base" or p["arm"] in P_ONLY_ARMS


def _pair(s, t, direction):
    track = s["track"]
    return {"track": track, "cell": track[:3], "assay": TRACK_ASSAY[track],
            "mark_class": MARK_CLASS[TRACK_ASSAY[track]],
            "arm_src": s["arm"], "level_src": s["level"], "arm_tgt": t["arm"],
            "level_tgt": t["level"], "source_pid": s["pid"], "target_pid": t["pid"],
            "direction": direction,
            "knob_combo": f"{s['arm']}:{s['level']}->{t['arm']}:{t['level']}",
            "counts_identical": _counts_same_as_base(s) and _counts_same_as_base(t),
            "pval_identical": False}


def train_pairs(track):
    out = []
    for p in _arm_products(track):
        out.append(_pair(_base(track), p, "base_to_arm"))
        out.append(_pair(p, _base(track), "arm_to_base"))
    return out


def law_pairs(track):
    prods = _arm_products(track)
    return [_pair(a, b, "arm_to_arm") for a in prods for b in prods if a["pid"] != b["pid"]]


def depth_of(p):
    """Reads of a product: the depth level ('15M') or the base depth."""
    if p["arm"] == "depth":
        return float(p["level"].rstrip("M")) * 1e6
    return BASE_DEPTH


def run_name(rung, g, space, model, seed):
    return f"{rung}_{g}_{space}_{model}_s{seed}"


def _ln_ratio(pair):
    s = {"arm": pair["arm_src"], "level": pair["level_src"]}
    t = {"arm": pair["arm_tgt"], "level": pair["level_tgt"]}
    return math.log(depth_of(t) / depth_of(s))


def _is_depth_pair(pair):
    """The two products differ only in depth: base<->depth or depth<->depth."""
    arms = {pair["arm_src"], pair["arm_tgt"]}
    return arms <= {"base", "depth"} and "depth" in arms


# ---------------------------------------------------------------------------------------------
# metric generator
# ---------------------------------------------------------------------------------------------


class Gen:
    def __init__(self, seed):
        self.rng = np.random.default_rng([seed, 7140])
        # per-pair difficulty, fixed across runs: {(track, kind): array[n_pairs]}
        self.difficulty = {}
        for t in TRACKS:
            for kind, pl in (("trained", train_pairs(t)), ("law", law_pairs(t))):
                self.difficulty[(t, kind)] = np.exp(self.rng.normal(0.0, 0.25, len(pl)))

    def values(self, pairs, track, kind, rung, g_version, space, model, seed_mult):
        """dict metric -> float64[n_pairs] for one run on one track's pairs."""
        cls = MARK_CLASS[TRACK_ASSAY[track]]
        d_all, d_top, s_all = D0[(cls, space)]
        diff_key = "law" if kind == "law" else "trained"
        dif = self.difficulty[(track, diff_key)]
        if model == "real":
            f = FACTOR_REAL[kind][rung]
        elif (kind, model) in FACTOR_TWIN:
            f = FACTOR_TWIN[(kind, model)]
        else:                                   # ids on trained pairs: about the real g
            f = FACTOR_REAL["trained"][rung] * 1.01
        if g_version == "across":
            f *= 1.01
        n = len(pairs)
        noise = np.exp(self.rng.normal(0.0, 0.01, n)) * seed_mult
        crps_all = d_all * dif * f * noise
        crps_top1 = d_top * dif * f ** 1.3 * noise
        gain = 1.0 - f
        sp = np.clip(s_all + 0.3 * gain + self.rng.normal(0, 0.004, n), -1, 1)
        return {"crps_all": crps_all, "crps_nonzero": 1.3 * crps_all, "crps_top1": crps_top1,
                "spearman_all": sp, "spearman_nonzero": sp - 0.05, "spearman_top1": sp - 0.15}

    def pit_hist(self, model):
        u = (np.arange(20) + 0.5) / 20
        ushape = {"real": 0.15, "nocov": 0.6, "ids": 0.25}[model]
        w = 1.0 + ushape * (np.abs(u - 0.5) * 4 - 1) + self.rng.normal(0, 0.03, 20)
        return [int(v) for v in np.maximum(w, 0.05) * 5000]

    def describe(self, rung, space, pair, n_theta_seed):
        rng = self.rng
        lr = _ln_ratio(pair) if space == "counts" else 0.8 * _ln_ratio(pair)
        arm = pair["arm_tgt"] if pair["arm_src"] == "base" else pair["arm_src"]
        mag = 0.0 if arm in P_ONLY_ARMS and space == "counts" else 0.1
        if rung == "A":
            return {"a": round(lr + rng.normal(0, 0.03 + mag / 3), 4),
                    "b": round(1.0 + rng.normal(0, mag / 2 + 0.01), 4),
                    "disp": round(math.log(5.0) + rng.normal(0, 0.1), 4)}
        kx = np.asarray(KNOTS_X[space])
        bend = rng.normal(0, mag + 0.02) * (kx - kx.mean()) / np.ptp(kx)
        curve = kx + lr + np.cumsum(np.abs(rng.normal(0, 0.01, 12))) + bend
        curve = np.maximum.accumulate(curve)
        out_b = {"knots_x": [round(v, 4) for v in kx], "curve_y": [round(v, 4) for v in curve],
                 "disp": [round(math.log(5.0) + v, 4) for v in rng.normal(0, 0.1, 12)]}
        if rung == "B":
            return out_b
        if rung == "C":
            width = {"extsize": 2.5, "pe": 1.5}.get(arm, 0.4) + abs(rng.normal(0, 0.1))
            off = np.arange(-16, 17)
            k = np.exp(-0.5 * (off / width) ** 2)
            k = k / k.sum() + rng.normal(0, 0.003, 33)
            return {"kernel": [round(v, 5) for v in k], **out_b}
        amp = {"extsize": 0.3, "pe": 0.25, "depth": 0.15}.get(arm, 0.08)
        gam = 1.0 + rng.normal(0, amp, (4, 32))
        bet = rng.normal(0, amp, (4, 32))
        return {"film_gamma": np.round(gam, 4).tolist(), "film_beta": np.round(bet, 4).tolist()}


def _wobble(vals):
    v = [x for x in vals if x is not None]
    return float(max(v) - min(v)) if len(v) >= 2 else 0.0


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


# ---------------------------------------------------------------------------------------------
# the results set
# ---------------------------------------------------------------------------------------------


def make_results(out_dir: Path, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    gen = Gen(seed)
    all_runs = [run_name(r, g, s, m, sd) for r in RUNGS for g in G_IDS for s in SPACES
                for m in MODELS for sd in SEEDS]
    present = [r for r in all_runs if r not in MISSING_RUNS]
    per_pair, depth_law = [], []
    # (rung, g_version, space, model, kind, variant, metric, track) -> {seed: value}
    track_vals: dict = {}
    # (rung, g_version, space, model, kind, metric, track, pair_index) -> [values over seeds]
    pair_vals: dict = {}
    val_crps: dict = {}           # (rung, g_version, space, track) -> {seed: val crps_all}
    swap_vals: dict = {}          # (rung, g_version, space, track) -> {seed: mean swap ratio}

    for rung in RUNGS:
        for g in G_IDS:
            g_version = "across" if g == "all" else "per_track"
            tracks = TRACKS if g == "all" else (g,)
            for space in SPACES:
                for model in MODELS:
                    for sd in SEEDS:
                        name = run_name(rung, g, space, model, sd)
                        if name in MISSING_RUNS:
                            continue
                        seed_mult = math.exp(gen.rng.normal(0.0, 0.004))
                        run_fields = {"rung": rung, "g": g, "g_version": g_version,
                                      "space": space, "model": model, "seed": sd}
                        kinds = ["trained", "law"] + (["shuffle"] if model == "real" else [])
                        for track in tracks:
                            for kind in kinds:
                                pl = law_pairs(track) if kind == "law" else train_pairs(track)
                                vals = gen.values(pl, track, kind, rung, g_version, space,
                                                  model, seed_mult)
                                ident = np.array([p[f"{space}_identical"] for p in pl])
                                for metric, arr in vals.items():
                                    for variant, keep in (("all", np.ones(len(pl), bool)),
                                                          ("nonidentical", ~ident)):
                                        key = (rung, g_version, space, model, kind, variant,
                                               metric, track)
                                        track_vals.setdefault(key, {})[sd] = \
                                            float(arr[keep].mean()) if keep.any() else None
                                    if metric in ("crps_all", "crps_top1"):
                                        for i, v in enumerate(arr):
                                            pair_vals.setdefault(
                                                (rung, g_version, space, model, kind, metric,
                                                 track, i), []).append(float(v))
                                _emit_records(per_pair, gen, pl, vals, kind, run_fields, track)
                            if model == "real":
                                vv = gen.values(train_pairs(track), track, "trained", rung,
                                                g_version, space, model, seed_mult)
                                vv = {k: v * 1.02 for k, v in vv.items()}
                                val_crps.setdefault((rung, g_version, space, track), {})[sd] = \
                                    float(vv["crps_all"].mean())
                                _emit_records(per_pair, gen, train_pairs(track), vv, "val",
                                              run_fields, track)
                                sw = _emit_swap(per_pair, gen, rung, track, run_fields)
                                swap_vals.setdefault((rung, g_version, space, track), {})[sd] = sw
                            if space == "counts":
                                _emit_depth(per_pair, depth_law, gen, track, rung, model,
                                            run_fields)

    per_track = []
    for (rung, gv, space, model, kind, variant, metric, track), by_seed in track_vals.items():
        for sd, v in sorted(by_seed.items()):
            n = len(law_pairs(track) if kind == "law" else train_pairs(track))
            per_track.append({"rung": rung, "g_version": gv, "space": space, "model": model,
                              "seed": sd, "kind": kind, "track": track,
                              "mark_class": MARK_CLASS[TRACK_ASSAY[track]], "metric": metric,
                              "value": v, "n_pairs": n, "variant": variant})
    per_class = _per_class(track_vals)
    checks = _checks(per_class, depth_law, swap_vals)
    results = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_dir": str(runs_dir),
        "runs_present": present,
        "runs_missing": list(MISSING_RUNS),
        "refs": _refs(gen),
        "per_pair": per_pair,
        "per_track": per_track,
        "per_class": per_class,
        "checks": checks,
        "knob_gain": _knob_gain(pair_vals),
        "law_grid": _law_grid(pair_vals),
        "depth_law": depth_law,
        "rung_choice": _rung_choice(val_crps),
        "figdata": {},
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name in present:
        _write_run_files(runs_dir, name, gen, results)
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh)
    for rung in RUNGS:
        with open(out_dir / f"checks_{rung}.json", "w") as fh:
            json.dump({"rung": rung, "checks": [c for c in checks if c["rung"] == rung]}, fh,
                      indent=1)
    main = [c for c in checks
            if results["rung_choice"][f"{c['g_version']}|{c['space']}"]["chosen"] == c["rung"]]
    with open(out_dir / "checks_main.json", "w") as fh:
        json.dump({"rung": "main", "checks": main}, fh, indent=1)
    with open(out_dir / "qm_curves.json", "w") as fh:
        json.dump(_qm_curves(gen), fh)
    return results


def _emit_records(per_pair, gen, pl, vals, kind, run_fields, track):
    """per_pair records: every trained/score record; val and shuffle only for the real model;
    law records only for the DNase track."""
    model = run_fields["model"]
    if kind == "law" and track != "C12M02":
        return
    if kind in ("val", "shuffle") and model != "real":
        return
    rec_kind = "trained" if kind == "val" else kind
    ev = "val" if kind == "val" else "score"
    for i, p in enumerate(pl):
        r = {"kind": rec_kind, "eval": ev, **p, **run_fields,
             "cov_src_pid": p["source_pid"], "cov_tgt_pid": p["target_pid"],
             "n_all": 60000, "n_nonzero": 30000, "n_top1": 600, "n_blacklisted": 120}
        for m in METRICS:
            r[m] = round(float(vals[m][i]), 6)
        if rec_kind == "shuffle":
            r["cov_tgt_pid"] = pl[(i + 2) % len(pl)]["target_pid"]
        if rec_kind == "trained" and ev == "score":
            c = r["crps_all"]
            r["crps_oracle_scaled_all"] = round(0.93 * c, 6)
            r["scale_error_all"] = round(0.07 * c, 6)
            r["c_star_all"] = round(float(gen.rng.normal(0, 0.2)), 3)
            r["pit_hist"] = gen.pit_hist(model)
        if rec_kind == "trained" and model == "real" and ev == "score":
            r["describe"] = gen.describe(run_fields["rung"], run_fields["space"], p, 0)
        per_pair.append(r)


def _emit_swap(per_pair, gen, rung, track, run_fields):
    prods = [_base(track)] + _arm_products(track)
    base_level = {"A": 0.03, "B": 0.04, "C": 0.07, "D": 0.09}[rung]
    vals = []
    for p in prods:
        v = float(abs(gen.rng.normal(base_level, 0.015)))
        vals.append(v)
        per_pair.append({"kind": "swap", "eval": "score", **_pair(p, p, "swap"), **run_fields,
                         "cov_src_pid": p["pid"], "cov_tgt_pid": p["pid"],
                         "swap_median_abs_log_ratio": round(v, 6)})
    return float(np.mean(vals))


def _emit_depth(per_pair, depth_law, gen, track, rung, model, run_fields):
    base = _base(track)
    deps = [p for p in _arm_products(track) if p["arm"] == "depth"]
    pairs = [(_pair(base, d, "base_to_arm"), "trained") for d in deps]
    pairs += [(_pair(d, base, "arm_to_base"), "trained") for d in deps]
    pairs += [(_pair(a, b, "arm_to_arm"), "law") for a, b in combinations(deps, 2)]
    pairs += [(_pair(b, a, "arm_to_arm"), "law") for a, b in combinations(deps, 2)]
    for p, kind in pairs:
        true = _ln_ratio(p) / math.log(2)
        if model == "real":
            pred = true * (1 + gen.rng.normal(0, 0.03)) + gen.rng.normal(0, 0.02)
        elif model == "ids" and kind == "trained":
            pred = true * (1 + gen.rng.normal(0, 0.05))
        else:
            pred = 0.4 * true + gen.rng.normal(0, 0.05)
        depth_law.append({"rung": rung, "g_version": run_fields["g_version"], "space": "counts",
                          "track": track, "source_pid": p["source_pid"],
                          "target_pid": p["target_pid"], "model": model,
                          "seed": run_fields["seed"], "kind": kind, "log2_true": true,
                          "log2_pred": round(float(pred), 6)})
        if model == "real" and kind == "trained":
            per_pair.append({"kind": "depthlaw", "eval": "score", **p, **run_fields,
                             "cov_src_pid": p["source_pid"], "cov_tgt_pid": p["target_pid"],
                             "depth_log2_ratio_true": true,
                             "depth_log2_scale_pred": round(float(pred), 6)})


def _classes():
    return ("DNase", "narrow", "broad")


def _class_tracks(cls):
    return [t for t in TRACKS if MARK_CLASS[TRACK_ASSAY[t]] == cls]


def _per_class(track_vals):
    out = []
    keys = {k[:7] for k in track_vals}
    for (rung, gv, space, model, kind, variant, metric) in sorted(keys):
        for cls in _classes():
            per_seed = []
            for sd in SEEDS:
                vs = [track_vals.get((rung, gv, space, model, kind, variant, metric, t), {})
                      .get(sd) for t in _class_tracks(cls)]
                per_seed.append(float(np.mean(vs)) if vs and all(v is not None for v in vs)
                                else None)
            if all(v is None for v in per_seed):
                continue
            out.append({"rung": rung, "g_version": gv, "space": space, "model": model,
                        "kind": kind, "mark_class": cls, "metric": metric, "variant": variant,
                        "per_seed": per_seed, "mean": _mean(per_seed),
                        "seed_wobble": _wobble(per_seed)})
    return out


def _pc_index(per_class):
    return {(r["rung"], r["g_version"], r["space"], r["model"], r["kind"], r["mark_class"],
             r["metric"], r["variant"]): r for r in per_class}


def _checks(per_class, depth_law, swap_vals):
    idx = _pc_index(per_class)
    out = []

    def add(rung, gv, space, cls, metric, check, value, bar, met, wob, comp):
        out.append({"rung": rung, "g_version": gv, "space": space, "mark_class": cls,
                    "metric": metric, "check": check, "value": value, "bar": bar,
                    "met": bool(met), "seed_wobble": wob, "components": comp})

    for rung in RUNGS:
        for gv in ("per_track", "across"):
            for space in SPACES:
                for cls in _classes():
                    for metric in ("crps_all", "crps_top1"):
                        def pc(model, kind, r=rung):
                            return idx[(r, gv, space, model, kind, cls, metric, "all")]
                        real, nocov = pc("real", "trained"), pc("nocov", "trained")
                        v = nocov["mean"] - real["mean"]
                        b = 2 * real["seed_wobble"]
                        add(rung, gv, space, cls, metric, "beatstwin", v, b, v > b,
                            real["seed_wobble"], {"d_nocov": nocov["mean"],
                                                  "d_real": real["mean"],
                                                  "wobble_real": real["seed_wobble"]})
                        lreal = pc("real", "law")
                        for twin in ("nocov", "ids"):
                            t = pc(twin, "law")
                            v = t["mean"] - lreal["mean"]
                            b = 2 * lreal["seed_wobble"]
                            add(rung, gv, space, cls, metric, f"lawtest_{twin}", v, b, v > b,
                                lreal["seed_wobble"], {f"d_{twin}": t["mean"],
                                                       "d_real": lreal["mean"],
                                                       "wobble_real": lreal["seed_wobble"]})
                        if rung != "A":
                            below = pc("real", "trained", RUNGS[RUNGS.index(rung) - 1])
                            v = below["mean"] - real["mean"]
                            w = max(below["seed_wobble"], real["seed_wobble"])
                            add(rung, gv, space, cls, metric, "beatsbelow", v, 2 * w, v > 2 * w,
                                w, {"rung_below": below["rung"], "d_below": below["mean"],
                                    "d_this": real["mean"], "wobble_below": below["seed_wobble"],
                                    "wobble_this": real["seed_wobble"]})
                        sh = pc("real", "shuffle")
                        v = nocov["mean"] - sh["mean"]
                        b = 2 * real["seed_wobble"]
                        add(rung, gv, space, cls, metric, "shuffle", v, b, v < b,
                            real["seed_wobble"], {"d_nocov": nocov["mean"],
                                                  "d_real_shuffled": sh["mean"],
                                                  "wobble_real": real["seed_wobble"]})
                    sw = [swap_vals.get((rung, gv, space, t), {}) for t in _class_tracks(cls)]
                    per_seed = [float(np.mean([s[sd] for s in sw])) for sd in SEEDS]
                    v = _mean(per_seed)
                    add(rung, gv, space, cls, "swap_median_abs_log_ratio", "swap", v, 0.1,
                        v < 0.1, _wobble(per_seed), {"per_seed": per_seed})
                for cls in (_classes() if space == "counts" else ()):     # counts only
                    rows = [d for d in depth_law if d["rung"] == rung and d["g_version"] == gv
                            and d["model"] == "real"
                            and MARK_CLASS[TRACK_ASSAY[d["track"]]] == cls]
                    by_pair: dict = {}
                    for d in rows:
                        by_pair.setdefault((d["track"], d["source_pid"], d["target_pid"]),
                                           []).append(d)
                    errs, wobs = [], []
                    for ds in by_pair.values():
                        pred = float(np.mean([d["log2_pred"] for d in ds]))
                        errs.append(abs(2 ** (pred - ds[0]["log2_true"]) - 1))
                        e_s = [abs(2 ** (d["log2_pred"] - d["log2_true"]) - 1) for d in ds]
                        wobs.append(_wobble(e_s))
                    v = float(max(errs))
                    add(rung, gv, "counts", cls, "depth_scale_rel_error", "depthlaw", v, 0.10,
                        v <= 0.10, float(max(wobs)), {"n_pairs": len(errs)})
    return out


def _refs(gen):
    refs = {}
    for t in TRACKS:
        cls = MARK_CLASS[TRACK_ASSAY[t]]
        refs[t] = {}
        for space in SPACES:
            d_all, d_top, s_all = D0[(cls, space)]
            j = float(np.exp(gen.rng.normal(0, 0.05)))
            ns = {"spread_crps_all": d_all * j, "spread_crps_top1": d_top * j,
                  "spearman_all": s_all, "crps_oracle_scaled_all": 0.9 * d_all * j,
                  "scale_error_all": 0.1 * d_all * j}
            qm = {"spread_crps_all": 0.83 * d_all * j, "spread_crps_top1": 0.55 * d_top * j,
                  "spearman_all": s_all - 0.003, "crps_oracle_scaled_all": 0.8 * d_all * j,
                  "scale_error_all": 0.03 * d_all * j}
            refs[t][space] = {"noSolution": ns, "QuantileMatching": qm}
    return refs


def _knob_gain(pair_vals):
    acc: dict = {}
    for (rung, gv, space, model, kind, metric, track, i), vs in pair_vals.items():
        if kind != "trained" or model == "ids":
            continue
        p = train_pairs(track)[i]
        arm = p["arm_tgt"] if p["direction"] == "base_to_arm" else p["arm_src"]
        acc.setdefault((rung, gv, space, arm, metric), {"real": [], "nocov": []})[model] \
            .extend(vs)
    out = []
    for (rung, gv, space, arm, metric), d in sorted(acc.items()):
        dr, dn = float(np.mean(d["real"])), float(np.mean(d["nocov"]))
        out.append({"rung": rung, "g_version": gv, "space": space, "arm": arm,
                    "metric": metric, "gain_rel": (dn - dr) / dn})
    return out


def _law_grid(pair_vals):
    out = []
    for rung in RUNGS:
        for gv in ("per_track", "across"):
            for space in SPACES:
                for track in TRACKS:
                    pl = law_pairs(track)
                    for metric in ("crps_all", "crps_top1"):
                        for i, p in enumerate(pl):
                            d = {m: pair_vals.get((rung, gv, space, m, "law", metric, track, i))
                                 for m in MODELS}
                            if any(v is None for v in d.values()):
                                continue
                            dr, dn, di = (float(np.mean(d[m])) for m in MODELS)
                            out.append({"rung": rung, "g_version": gv, "space": space,
                                        "track": track, "arm_src": p["arm_src"],
                                        "level_src": p["level_src"], "arm_tgt": p["arm_tgt"],
                                        "level_tgt": p["level_tgt"], "metric": metric,
                                        "d_real": dr, "d_nocov": dn, "d_ids": di,
                                        "gain_nocov": dn - dr, "gain_ids": di - dr,
                                        "identical_target": bool(p[f"{space}_identical"])})
    return out


def _rung_choice(val_crps):
    out = {}
    for gv in ("per_track", "across"):
        for space in SPACES:
            val, wob = {}, {}
            for rung in RUNGS:
                tracks = [(k, v) for k, v in val_crps.items() if k[:3] == (rung, gv, space)]
                per_seed = [float(np.mean([v[sd] for _, v in tracks])) for sd in SEEDS]
                val[rung], wob[rung] = _mean(per_seed), _wobble(per_seed)
            best = min(val, key=val.get)
            chosen = next(r for r in RUNGS if val[r] <= val[best] + wob[best])
            out[f"{gv}|{space}"] = {"chosen": chosen, "val_by_rung": val,
                                    "wobble_by_rung": wob}
    return out


def _qm_curves(gen):
    out = {}
    for space in SPACES:
        out[space] = {}
        for t in TRACKS:
            out[space][t] = {}
            for p in _arm_products(t):
                if space == "counts":
                    kx = np.arange(0, 60, dtype=float)
                    r = depth_of(p) / BASE_DEPTH
                    ky = r * kx * (1 + 0.02 * gen.rng.normal()) + 0.05 * np.sqrt(kx)
                else:
                    kx = np.concatenate([[0.0], np.geomspace(0.01, 60, 40)])
                    r = (depth_of(p) / BASE_DEPTH) ** 0.8
                    ky = r * kx
                out[space][t][p["pid"]] = {"knots_x": np.round(kx, 4).tolist(),
                                           "knots_y": np.round(ky, 4).tolist()}
    return out


# ---------------------------------------------------------------------------------------------
# per-run files
# ---------------------------------------------------------------------------------------------


def _profile(rng, n, peak, width, bg):
    d = np.arange(n) - n // 2
    return bg + peak * np.exp(-0.5 * (d / width) ** 2)


def _arm_effect(arm, level, x, space):
    """X' from X for one arm (a rough, invented effect)."""
    if arm == "depth":
        r = float(level.rstrip("M")) * 1e6 / BASE_DEPTH
        return x * (r if space == "counts" else r ** 0.8)
    if arm in ("extsize", "pe"):
        w = 9 if (arm == "pe" or level == "k2") else 3
        k = np.ones(w) / w
        y = np.convolve(x, k, mode="same")
        return y if arm == "pe" or space == "pval" else x
    if arm == "dedup":
        return x * 1.15
    if arm in ("ratio", "ctlid", "ctldepth", "abproxy"):
        return x if space == "counts" else x * {"ratio": 0.7}.get(arm, 0.85)
    return x * 0.95


def _write_run_files(runs_dir, name, gen, results):
    rung, g, space, model, s = name.split("_")
    sd = int(s[1:])
    rd = runs_dir / name
    rd.mkdir(parents=True, exist_ok=True)
    rng = gen.rng
    setup = float(rng.uniform(20, 40))
    train = float(rng.uniform(300, 900)) * (1 + 0.3 * RUNGS.index(rung))
    with open(rd / "timing.json", "w") as fh:
        json.dump({"setup": setup, "train": train, "total": setup + train,
                   "peak_rss_mb": float(rng.uniform(900, 2500))}, fh)
    if not (model == "real" and sd == 0 and g != "all"):
        return
    scale = 1.0 if space == "counts" else 0.6
    arrays, snippets = {}, []
    q = {"A": 0.0, "B": 0.1, "C": 0.6, "D": 0.8}[rung]
    for p in train_pairs(g):
        x = _profile(rng, 161, 12 * scale, 12, 1.0 * scale) + rng.normal(0, 0.05, 161)
        src_is_base = p["direction"] == "base_to_arm"
        arm, lv = (p["arm_tgt"], p["level_tgt"]) if src_is_base else (p["arm_src"], p["level_src"])
        xt = _arm_effect(arm, lv, x, space) if src_is_base else x
        xs = x if src_is_base else _arm_effect(arm, lv, x, space)
        mag = xs * xt.sum() / xs.sum()
        mu = (1 - q) * mag + q * xt
        arrays[f"meta__{p['source_pid']}__{p['target_pid']}"] = \
            np.stack([xs, xt, mu]).astype(np.float32)
    arms = sorted({p["arm_tgt"] for p in train_pairs(g) if p["direction"] == "base_to_arm"})
    rules = ("top1_max_abs_diff", "top1_random", "background_random")
    for ai, arm in enumerate(arms):
        lv = sorted(lv for a, lvs in (HISTONE_ARMS if g != "C12M02" else DNASE_ARMS).items()
                    if a == arm for lv in lvs)[0]
        for i, rule in enumerate(rules):
            bg = rng.gamma(1.5, 0.6, 400) * scale
            peaks = 0.0 if i == 2 else 1.0
            x = bg + peaks * _profile(rng, 400, 25 * scale, 10, 0) \
                + peaks * _profile(rng, 400, 8 * scale, 30, 0)[::-1]
            xt = np.maximum(_arm_effect(arm, lv, x, space) + rng.normal(0, 0.3 * scale, 400), 0)
            mu = np.maximum((1 - q) * x * xt.sum() / x.sum() + q * xt, 0)
            arrays[f"snip__{arm}__{i}"] = np.stack(
                [x, xt, mu, 0.35 * mu, 1.9 * mu + 1.0 * scale]).astype(np.float32)
            snippets.append({"arm": arm, "i": i, "rule": rule,
                             "chrom": ("chr19", "chr21")[i % 2],
                             "start_bin": int(rng.integers(0, 2_000_000)),
                             "source_pid": f"{g}__base__base",
                             "target_pid": f"{g}__{arm}__{lv}"})
    np.savez_compressed(rd / "figdata.npz", **arrays)
    results["figdata"][name] = str(rd / "figdata.npz")
    with open(rd / "scores.json", "w") as fh:
        json.dump({"run": {"run_name": name, "rung": rung, "g": g, "space": space,
                           "model": model, "seed": sd},
                   "timing": {"load": float(rng.uniform(5, 15)),
                              "score_trained": float(rng.uniform(60, 200)),
                              "shuffle_swap_depth": float(rng.uniform(30, 90))},
                   "snippets": snippets, "records": []}, fh)
    with open(rd / "config.json", "w") as fh:
        json.dump({"run_name": name, "rung": rung, "g": g, "space": space, "model": model,
                   "seed": sd, "max_steps": 4000, "lr": 1e-3, "window": 2048, "batch": 32,
                   "warmup": 100, "grad_clip": 1.0, "eval_every": 250, "patience": 4,
                   "optimizer": "Adam", "git_sha": "synthetic"}, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    res = make_results(Path(a.out_dir), a.seed)
    print(f"synthetic results: {len(res['runs_present'])} runs present, "
          f"{len(res['runs_missing'])} missing, {len(res['per_pair'])} per_pair records, "
          f"{len(res['checks'])} checks -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
