"""t118 ladder — the nine figures of one rung, drawn from the pinned `results.json` + `figdata.npz`.

    python tools/t118/ladder/figures.py <agg_dir> <rung> [--refs-qm <qm_curves.json>]
                                                         [--snippet-track <track>]
    python tools/t118/ladder/figures.py --check-schema <agg_dir>

Writes `<agg_dir>/<rung>/figures/`:
  fig1_ladder         per mark class x space: noSolution, both twins, A-D, QuantileMatching;
                      CRPS on all bins and on the top 1 %; bars = seed range (every rung present,
                      `<rung>` highlighted)
  fig2_knob_heatmap   10 arms x rungs, cell = relative CRPS gain over the no-covariates twin
                      (every rung present, `<rung>` boxed)
  fig3_law_grid       per track, product x product matrix of never-trained arm->arm pairs,
                      colour = CRPS gain over each twin, identical-target cells hatched
  fig4_depth_law      true log2 depth ratio vs predicted log2 count scale, identity line and the
                      +-10 % band (every rung present, `<rung>` highlighted)
  fig5_learned_f      A: a and b per arm level; B: g's curve per arm over QuantileMatching's curve;
                      C: the kernel per arm; D: the size of the FiLM modulation per layer
  fig6_meta_profiles  mean signal +-2 kb around the top bins of X', for X, X' and the prediction
  fig7_snippets       10 kb loci picked by the fixed rule, X, X', predicted mean, 90 % interval
  fig8_calibration    PIT histograms per mark class (secondary)
  fig9_checks         the checks card: each check's value against its bar, per version of g

`--check-schema` validates results.json against the pinned keys (plan t118-C3) and draws
nothing; it needs numpy only. Plotting imports matplotlib lazily (Agg backend), so this module
imports under the candii env too (report.py reuses its labels and loaders). No torch, no candi,
no scipy, no ladder modules.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

RUNGS = ("A", "B", "C", "D")
MODELS = ("real", "nocov", "ids")
SPACES = ("counts", "pval")
SEEDS = (0, 1, 2)
G_VERSIONS = ("per_track", "across")
CLASSES = ("DNase", "narrow", "broad")
METRICS = ("crps_all", "crps_nonzero", "crps_top1", "spearman_all", "spearman_nonzero",
           "spearman_top1")
VARIANTS = ("all", "nonidentical")
KINDS = ("trained", "shuffle", "swap", "depthlaw", "law")
ARMS = ("abproxy", "crop", "ctldepth", "ctlid", "dedup", "depth", "extsize", "mapq", "pe",
        "ratio")
REF_RUNGS = ("noSolution", "QuantileMatching")
FIG_NAMES = ("fig1_ladder", "fig2_knob_heatmap", "fig3_law_grid", "fig4_depth_law",
             "fig5_learned_f", "fig6_meta_profiles", "fig7_snippets", "fig8_calibration",
             "fig9_checks")
DEFAULT_SNIPPET_TRACK = "C19M16"    # the pilot track: H3K27ac, all ten arms

RUNG_NAME = {"A": "design A, the per-bin affine map",
             "B": "design B, the per-bin monotone curve",
             "C": "design C, a 33-bin kernel then the monotone curve",
             "D": "design D, a dilated CNN modulated by g (FiLM)"}
MODEL_NAME = {"real": "real g", "nocov": "no-covariates twin", "ids": "labels-as-ids twin"}
GV_NAME = {"per_track": "one g per track", "across": "one g across tracks"}
SPACE_NAME = {"counts": "counts", "pval": "-log10 p"}
METRIC_NAME = {"crps_all": "CRPS, all bins", "crps_nonzero": "CRPS, non-zero bins",
               "crps_top1": "CRPS, top 1 % bins", "spearman_all": "Spearman, all bins",
               "spearman_nonzero": "Spearman, non-zero bins",
               "spearman_top1": "Spearman, top 1 % bins",
               "swap_median_abs_log_ratio": "median |log(mean / X)|",
               "depth_scale_rel_error": "|scale / ratio - 1|"}
CHECK_ORDER = ("beatstwin", "lawtest_nocov", "lawtest_ids", "depthlaw", "beatsbelow", "shuffle",
               "swap")
CHECK_NAME = {"beatstwin": "beats the no-covariates twin (held-out chromosomes)",
              "lawtest_nocov": "law test: beats the no-covariates twin",
              "lawtest_ids": "law test: beats the labels-as-ids twin",
              "depthlaw": "depth law: count scale within 10 % of the depth ratio",
              "beatsbelow": "beats the rung below",
              "shuffle": "shuffle: a wrong C' removes the advantage",
              "swap": "swap: C' = C leaves X unchanged"}
CHECK_SHORT = {"beatstwin": "beats no-cov twin", "lawtest_nocov": "law test vs no-cov twin",
               "lawtest_ids": "law test vs ids twin", "depthlaw": "depth law",
               "beatsbelow": "beats rung below", "shuffle": "shuffle (wrong C')",
               "swap": "swap (C' = C)"}
#: the comparison `met` encodes, per check (plan t118-C3)
CHECK_RULE = {"beatstwin": ">", "lawtest_nocov": ">", "lawtest_ids": ">", "beatsbelow": ">",
              "depthlaw": "<=", "shuffle": "<", "swap": "<"}
RUNG_COLOUR = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
MODEL_COLOUR = {"nocov": "#7f4f9f", "ids": "#8c564b", "real": "#1f77b4"}
TRACK_COLOUR = {"C07M20": "#1f77b4", "C07M29": "#ff7f0e", "C12M02": "#2ca02c",
                "C19M16": "#d62728", "C19M22": "#9467bd", "C40M17": "#8c564b",
                "C40M18": "#e377c2"}
SNIP_RULE_NAME = {0: "top-1 % bin, largest |X' - X|", 1: "random top-1 % bin",
                  2: "random background bin"}

# ---------------------------------------------------------------------------------------------
# the pinned schema (plan t118-C3) and its validator
# ---------------------------------------------------------------------------------------------

TOP_KEYS = ("created_utc", "runs_dir", "runs_present", "runs_missing", "refs", "per_pair",
            "per_track", "per_class", "checks", "knob_gain", "law_grid", "depth_law",
            "rung_choice", "figdata")
ROW_KEYS = {
    "per_pair": ("kind", "eval", "track", "rung", "g", "g_version", "space", "model", "seed"),
    "per_track": ("rung", "g_version", "space", "model", "seed", "kind", "track", "mark_class",
                  "metric", "value", "n_pairs", "variant"),
    "per_class": ("rung", "g_version", "space", "model", "kind", "mark_class", "metric",
                  "variant", "per_seed", "mean", "seed_wobble"),
    "checks": ("rung", "g_version", "space", "mark_class", "metric", "check", "value", "bar",
               "met", "seed_wobble", "components"),
    "knob_gain": ("rung", "g_version", "space", "arm", "metric", "gain_rel"),
    "law_grid": ("rung", "g_version", "space", "track", "arm_src", "level_src", "arm_tgt",
                 "level_tgt", "metric", "d_real", "d_nocov", "d_ids", "gain_nocov", "gain_ids",
                 "identical_target"),
    "depth_law": ("rung", "g_version", "space", "track", "source_pid", "target_pid", "model",
                  "seed", "kind", "log2_true", "log2_pred"),
}
ENUMS = {"rung": RUNGS, "g_version": G_VERSIONS, "space": SPACES, "model": MODELS,
         "variant": VARIANTS, "mark_class": CLASSES}
#: per_pair fields a trained/score record must carry (the CRPS split and the PIT histogram)
TRAINED_SCORE_KEYS = ("crps_all", "crps_top1", "spearman_all", "crps_oracle_scaled_all",
                      "scale_error_all", "c_star_all", "pit_hist")


def check_schema(res: dict, max_per_section: int = 10) -> list[str]:
    """Errors (empty = valid) of a results dict against the pinned keys and enums."""
    errs: list[str] = []
    for k in TOP_KEYS:
        if k not in res:
            errs.append(f"top level: missing key {k!r}")
    for sec, keys in ROW_KEYS.items():
        rows = res.get(sec)
        if rows is None:
            continue
        if not isinstance(rows, list):
            errs.append(f"{sec}: not a list")
            continue
        n_err = 0
        for i, r in enumerate(rows):
            bad = [k for k in keys if k not in r]
            bad += [f"{k}={r[k]!r}" for k, allowed in ENUMS.items()
                    if k in r and k in keys and r[k] not in allowed]
            if sec in ("per_track", "per_class") and r.get("metric") not in METRICS:
                bad.append(f"metric={r.get('metric')!r}")
            if sec in ("per_track", "per_class") and r.get("kind") not in KINDS:
                bad.append(f"kind={r.get('kind')!r}")
            if sec == "per_pair":
                if r.get("kind") not in KINDS:
                    bad.append(f"kind={r.get('kind')!r}")
                if r.get("eval") not in ("score", "val"):
                    bad.append(f"eval={r.get('eval')!r}")
                if r.get("kind") == "trained" and r.get("eval") == "score":
                    bad += [k for k in TRAINED_SCORE_KEYS if k not in r]
                    if "pit_hist" in r and len(r["pit_hist"]) != 20:
                        bad.append("pit_hist length != 20")
            if sec == "per_class" and isinstance(r.get("per_seed"), list) \
                    and len(r["per_seed"]) != len(SEEDS):
                bad.append(f"per_seed length {len(r['per_seed'])} != {len(SEEDS)}")
            if sec == "checks" and r.get("check") not in CHECK_ORDER:
                bad.append(f"check={r.get('check')!r}")
            if bad:
                n_err += 1
                if n_err <= max_per_section:
                    errs.append(f"{sec}[{i}]: missing or invalid {', '.join(map(str, bad))}")
        if n_err > max_per_section:
            errs.append(f"{sec}: {n_err - max_per_section} more rows with errors")
    rc = res.get("rung_choice", {})
    if isinstance(rc, dict):
        for key, v in rc.items():
            gv, _, space = key.partition("|")
            if gv not in G_VERSIONS or space not in SPACES:
                errs.append(f"rung_choice: key {key!r} is not '<g_version>|<space>'")
            for k in ("chosen", "val_by_rung", "wobble_by_rung"):
                if k not in v:
                    errs.append(f"rung_choice[{key!r}]: missing {k!r}")
    else:
        errs.append("rung_choice: not a dict")
    refs = res.get("refs", {})
    if not isinstance(refs, dict):
        errs.append("refs: not a dict")
    else:
        for track, by_space in refs.items():
            for space, by_rung in by_space.items():
                if space not in SPACES:
                    errs.append(f"refs[{track}]: space {space!r}")
                for rung in by_rung:
                    if rung not in REF_RUNGS:
                        errs.append(f"refs[{track}][{space}]: rung {rung!r} (not a reference)")
    if not isinstance(res.get("figdata", {}), dict):
        errs.append("figdata: not a dict")
    for k in ("runs_present", "runs_missing"):
        if k in res and not isinstance(res[k], list):
            errs.append(f"{k}: not a list")
    return errs


# ---------------------------------------------------------------------------------------------
# loaders and lookups (numpy + stdlib; report.py reuses these)
# ---------------------------------------------------------------------------------------------


def load_results(agg_dir) -> dict:
    with open(Path(agg_dir) / "results.json") as fh:
        return json.load(fh)


def load_checks(agg_dir, rung, res=None) -> list[dict]:
    """`checks_<rung>.json` (a list, or a dict with "checks"); else results["checks"]."""
    p = Path(agg_dir) / f"checks_{rung}.json"
    if p.is_file():
        with open(p) as fh:
            obj = json.load(fh)
        rows = obj.get("checks", []) if isinstance(obj, dict) else obj
    else:
        rows = (res or load_results(agg_dir)).get("checks", [])
    return [c for c in rows if c.get("rung") == rung]


def sort_checks(rows):
    def key(c):
        return (CHECK_ORDER.index(c["check"]) if c["check"] in CHECK_ORDER else 99,
                G_VERSIONS.index(c["g_version"]) if c["g_version"] in G_VERSIONS else 9,
                c["space"], CLASSES.index(c["mark_class"]) if c["mark_class"] in CLASSES else 9,
                c["metric"])
    return sorted(rows, key=key)


def class_index(res) -> dict:
    """(rung, g_version, space, model, kind, mark_class, metric, variant) -> per_class row."""
    return {(r["rung"], r["g_version"], r["space"], r["model"], r["kind"], r["mark_class"],
             r["metric"], r["variant"]): r for r in res.get("per_class", [])}


def rungs_present(res) -> list[str]:
    have = {r["rung"] for r in res.get("per_class", [])}
    return [r for r in RUNGS if r in have]


def class_tracks(res) -> dict:
    """mark class -> sorted tracks, read from per_track (or refs when per_track is empty)."""
    out: dict = {}
    for r in res.get("per_track", []):
        out.setdefault(r["mark_class"], set()).add(r["track"])
    return {c: sorted(v) for c, v in out.items()}


def ref_value(res, track, space, ref_rung, metric):
    """A reference value for one track, by metric name or its `spread_` column name."""
    d = res.get("refs", {}).get(track, {}).get(space, {}).get(ref_rung, {})
    for k in (metric, f"spread_{metric}"):
        if k in d and d[k] is not None:
            return float(d[k])
    return None


def ref_class(res, cls, space, ref_rung, metric, tracks_by_class=None):
    """Macro mean over the class's tracks of a reference value (None if any track lacks it)."""
    tbc = tracks_by_class or class_tracks(res)
    vals = [ref_value(res, t, space, ref_rung, metric) for t in tbc.get(cls, [])]
    if not vals or any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def seed_range(row):
    v = [x for x in row.get("per_seed", []) if x is not None]
    return (min(v), max(v)) if v else (None, None)


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    v = float(v)
    if not math.isfinite(v):
        return str(v)
    if v != 0 and (abs(v) >= 1e5 or abs(v) < 1e-4):
        return f"{v:.3e}"
    return f"{v:.{nd}g}" if abs(v) >= 1 else f"{v:.{nd}f}"


def figdata_path(res, agg_dir, run_name):
    cands = []
    if run_name in res.get("figdata", {}):
        cands.append(Path(res["figdata"][run_name]))
    cands += [Path(res.get("runs_dir", "")) / run_name / "figdata.npz",
              Path(agg_dir) / "runs" / run_name / "figdata.npz",
              Path(agg_dir).parent / "runs" / run_name / "figdata.npz"]
    for p in cands:
        if p.is_file():
            return p
    return None


def run_dir(res, agg_dir, run_name):
    for d in (Path(res.get("runs_dir", "")) / run_name, Path(agg_dir) / "runs" / run_name,
              Path(agg_dir).parent / "runs" / run_name):
        if d.is_dir():
            return d
    return None


def split_pid(pid):
    """`<track>__<arm>__<level>` -> (track, arm, level)."""
    parts = pid.split("__")
    return parts[0], parts[1], "__".join(parts[2:])


def parse_meta_key(key):
    """`meta__<src pid>__<tgt pid>` (pids hold `__` themselves: 3 parts each)."""
    parts = key.split("__")[1:]
    return "__".join(parts[:3]), "__".join(parts[3:])


def arm_of_pair(r):
    """The non-base arm of a base<->arm pair."""
    return r["arm_tgt"] if r.get("arm_src") == "base" else r.get("arm_src")


def level_of_pair(r):
    return r["level_tgt"] if r.get("arm_src") == "base" else r.get("level_src")


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "figure.dpi": 100, "savefig.dpi": 110})
    return plt


def _save(fig, out):
    fig.savefig(out, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _no_data(ax, text="no data"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.5",
            fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _footer(fig, text):
    fig.text(0.01, -0.005, text, ha="left", va="top", fontsize=7, color="0.25", wrap=True)


def fig1_ladder(ctx, out):
    plt, res, rung = ctx["plt"], ctx["res"], ctx["rung"]
    idx, tbc = ctx["idx"], ctx["tbc"]
    xs = ["noSolution", "nocov twin", "ids twin", "A", "B", "C", "D", "QuantileMatching"]
    rows = [(s, m) for s in SPACES for m in ("crps_all", "crps_top1")]
    fig, axes = plt.subplots(len(rows), len(CLASSES), figsize=(13, 12), squeeze=False)
    offs = {"per_track": -0.13, "across": 0.13}
    marks = {"per_track": "o", "across": "s"}
    for i, (space, metric) in enumerate(rows):
        for j, cls in enumerate(CLASSES):
            ax = axes[i][j]
            ax.axvspan(xs.index(rung) - 0.45, xs.index(rung) + 0.45, color="#fff2cc", zorder=0)
            for k, ref in ((0, "noSolution"), (7, "QuantileMatching")):
                v = ref_class(res, cls, space, ref, metric, tbc)
                if v is not None:
                    ax.plot([k], [v], marker="D", color="0.2", ms=6, zorder=3)
            for gv in G_VERSIONS:
                for k, (r_, model) in enumerate([(rung, "nocov"), (rung, "ids")] +
                                                [(r_, "real") for r_ in RUNGS], start=1):
                    row = idx.get((r_, gv, space, model, "trained", cls, metric, "all"))
                    if row is None or row["mean"] is None:
                        continue
                    lo, hi = seed_range(row)
                    col = RUNG_COLOUR[r_] if model == "real" else MODEL_COLOUR[model]
                    big = model != "real" or r_ == rung
                    ax.errorbar([k + offs[gv]], [row["mean"]],
                                yerr=[[row["mean"] - lo], [hi - row["mean"]]],
                                fmt=marks[gv], color=col, ms=6 if big else 4,
                                mec="k" if (model == "real" and r_ == rung) else col,
                                capsize=3, lw=1.2, alpha=1.0 if big else 0.6, zorder=4)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs, rotation=35, ha="right")
            for t in ax.get_xticklabels():
                if t.get_text() == rung:
                    t.set_fontweight("bold")
            ax.set_title(f"{cls} · {SPACE_NAME[space]} · {METRIC_NAME[metric]}")
            ax.grid(axis="y", color="0.9")
            if j == 0:
                ax.set_ylabel(METRIC_NAME[metric])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="0.3", label=GV_NAME["per_track"]),
               Line2D([], [], marker="s", ls="", color="0.3", label=GV_NAME["across"]),
               Line2D([], [], marker="D", ls="", color="0.2", label="reference (no seed)")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"Ladder — rung {rung} highlighted ({RUNG_NAME[rung]}); twins are rung {rung}'s",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _footer(fig, "Held-out chromosomes chr19 + chr21, trained pairs, all pairs (identical-target "
                 "pairs kept). Point = mean over 3 seeds; bar = seed range, its height is the seed "
                 "wobble. References: macro mean over the class's tracks, deterministic.")
    _save(fig, out)


def fig2_knob_heatmap(ctx, out):
    plt, res, rung = ctx["plt"], ctx["res"], ctx["rung"]
    present = ctx["rungs"]
    kg = {(r["rung"], r["g_version"], r["space"], r["arm"], r["metric"]): r["gain_rel"]
          for r in res.get("knob_gain", [])}
    arms = sorted({r["arm"] for r in res.get("knob_gain", [])}) or list(ARMS)
    panels = [(gv, m) for gv in G_VERSIONS for m in ("crps_all", "crps_top1")]
    vals = [abs(v) for v in kg.values() if v is not None and math.isfinite(v)]
    vmax = max(np.percentile(vals, 98), 1e-6) if vals else 1.0
    fig, axes = plt.subplots(len(panels), len(SPACES), figsize=(9, 15), squeeze=False)
    im = None
    for i, (gv, metric) in enumerate(panels):
        for j, space in enumerate(SPACES):
            ax = axes[i][j]
            m = np.full((len(arms), len(present)), np.nan)
            for a, arm in enumerate(arms):
                for b, r_ in enumerate(present):
                    v = kg.get((r_, gv, space, arm, metric))
                    if v is not None:
                        m[a, b] = v
            im = ax.imshow(m, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
            for a in range(len(arms)):
                for b in range(len(present)):
                    if np.isfinite(m[a, b]):
                        ax.text(b, a, f"{100 * m[a, b]:+.1f}%", ha="center", va="center",
                                fontsize=6.5, color="k" if abs(m[a, b]) < 0.6 * vmax else "w")
            if rung in present:
                b = present.index(rung)
                ax.add_patch(plt.Rectangle((b - 0.5, -0.5), 1, len(arms), fill=False, lw=2,
                                           ec="k"))
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present)
            ax.set_yticks(range(len(arms)))
            ax.set_yticklabels(arms)
            ax.set_title(f"{GV_NAME[gv]} · {SPACE_NAME[space]} · {METRIC_NAME[metric]}")
    cb = fig.colorbar(im, ax=axes, shrink=0.4, location="right")
    cb.set_label("(D_nocov - D_real) / D_nocov")
    fig.suptitle(f"Relative CRPS gain of the real g over the no-covariates twin, per arm — "
                 f"rung {rung} boxed", x=0.01, ha="left", fontsize=10)
    _footer(fig, "Trained pairs of each arm (base->arm and arm->base, all levels, all tracks), "
                 "held-out chromosomes, mean over seeds. Positive (blue) = the covariates help. "
                 "Per-arm seed wobble is not in results.json; see the checks card for the "
                 "class-level wobble.")
    _save(fig, out)


def fig3_law_grid(ctx, out):
    plt, res, rung = ctx["plt"], ctx["res"], ctx["rung"]
    metric = "crps_all"
    rows = [r for r in res.get("law_grid", []) if r["rung"] == rung and r["metric"] == metric]
    tracks = sorted({r["track"] for r in rows})
    cols = [(gv, s, tw) for gv in G_VERSIONS for s in SPACES for tw in ("nocov", "ids")]
    if not tracks:
        fig, ax = plt.subplots(figsize=(6, 3))
        _no_data(ax, f"no law-test rows for rung {rung}")
        _save(fig, out)
        return
    by = {}
    for r in rows:
        by.setdefault((r["track"], r["g_version"], r["space"]), []).append(r)
    fig, axes = plt.subplots(len(tracks), len(cols), figsize=(3.1 * len(cols), 3.0 * len(tracks)),
                             squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.04, hspace=0.55, wspace=0.25)
    for i, track in enumerate(tracks):
        prods = sorted({(r["arm_src"], r["level_src"]) for k, rs in by.items() if k[0] == track
                        for r in rs} | {(r["arm_tgt"], r["level_tgt"]) for k, rs in by.items()
                                        if k[0] == track for r in rs})
        pos = {p: n for n, p in enumerate(prods)}
        labels = [f"{a}:{lv}" for a, lv in prods]
        mats = []
        for gv, space, tw in cols:
            m = np.full((len(prods), len(prods)), np.nan)
            ident = np.zeros_like(m, dtype=bool)
            for r in by.get((track, gv, space), []):
                a, b = pos[(r["arm_src"], r["level_src"])], pos[(r["arm_tgt"], r["level_tgt"])]
                m[a, b] = r[f"d_{tw}"] - r["d_real"]
                ident[a, b] = bool(r.get("identical_target"))
            mats.append((m, ident))
        vmax = {}
        for sp in SPACES:           # one colour scale per (track, space): CRPS units differ
            fin = [np.abs(m[np.isfinite(m)]) for (_, s_, _), (m, _) in zip(cols, mats)
                   if s_ == sp and np.isfinite(m).any()]
            vmax[sp] = max(float(np.percentile(np.concatenate(fin), 98)), 1e-9) if fin else 1.0
        for j, ((gv, space, tw), (m, ident)) in enumerate(zip(cols, mats)):
            ax = axes[i][j]
            if not np.isfinite(m).any():
                _no_data(ax)
                continue
            diag = np.where(np.eye(len(prods), dtype=bool), 1.0, np.nan)
            ax.imshow(diag, cmap="Greys", vmin=0, vmax=3, aspect="equal")
            im = ax.imshow(m, cmap="RdBu", vmin=-vmax[space], vmax=vmax[space], aspect="equal")
            if gv == G_VERSIONS[-1] and tw == "ids":
                cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
                cb.set_label(f"D_twin - D_real ({SPACE_NAME[space]})", fontsize=6)
                cb.ax.tick_params(labelsize=6)
            for a, b in zip(*np.nonzero(ident)):
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, hatch="////",
                                           ec="0.35", lw=0.3))
            ax.set_xticks(range(len(prods)))
            ax.set_xticklabels(labels, rotation=90, fontsize=4.5)
            ax.set_yticks(range(len(prods)))
            ax.set_yticklabels(labels if j == 0 else [], fontsize=4.5)
            ax.set_title(f"{track} · {GV_NAME[gv]}\n{SPACE_NAME[space]} · gain over "
                         f"{MODEL_NAME[tw]}", fontsize=7)
            if j == 0:
                ax.set_ylabel("source product")
    fig.suptitle(f"Law test, rung {rung} ({RUNG_NAME[rung]}): never-trained arm->arm pairs, "
                 f"{METRIC_NAME[metric]}, rows = source, columns = target", x=0.01, ha="left",
                 fontsize=11, y=0.985)
    _footer(fig, "Colour = CRPS of the twin minus CRPS of the real g, mean over seeds; blue = the "
                 "real g is better. Hatched = the target equals the source in that space (kept, "
                 "flagged). Grey diagonal = no pair. Colour scale shared within a track and space (the colour bars at the right of each space block). "
                 "Per-cell seed wobble is not in results.json; the class-level law-test wobble "
                 "is on the checks card.")
    _save(fig, out)


def fig4_depth_law(ctx, out):
    plt, res, rung, checks = ctx["plt"], ctx["res"], ctx["rung"], ctx["checks"]
    rows = res.get("depth_law", [])
    fig, axes = plt.subplots(1, len(G_VERSIONS), figsize=(13, 6.2), squeeze=False)
    for j, gv in enumerate(G_VERSIONS):
        ax = axes[0][j]
        sub = [r for r in rows if r["g_version"] == gv]
        if not sub:
            _no_data(ax)
            continue
        lim = max(abs(r["log2_true"]) for r in sub) * 1.15 + 0.2
        t = np.linspace(-lim, lim, 50)
        ax.fill_between(t, t + math.log2(0.9), t + math.log2(1.1), color="0.85", zorder=0,
                        label="±10 % of the depth ratio")
        ax.plot(t, t, color="0.3", lw=0.8, zorder=1, label="identity")
        true_of = {(x["source_pid"], x["target_pid"]): x["log2_true"] for x in sub}
        order = [r_ for r_ in ctx["rungs"] if r_ != rung] + [rung]
        for r_ in order:
            for model in ("real", "nocov") if r_ == rung else ("real",):
                by: dict = {}
                for r in sub:
                    if r["rung"] == r_ and r["model"] == model:
                        by.setdefault((r["track"], r["source_pid"], r["target_pid"], r["kind"]),
                                      []).append(r["log2_pred"])
                for kind, mk in (("trained", "o"), ("law", "^")):
                    pts = [(true_of[(k[1], k[2])], v) for k, v in by.items() if k[3] == kind]
                    if not pts:
                        continue
                    xt = np.array([p[0] for p in pts])
                    ys = [np.array(p[1]) for p in pts]
                    ym = np.array([y.mean() for y in ys])
                    if model == "nocov":
                        ax.scatter(xt, ym, marker="x", color=MODEL_COLOUR["nocov"], s=18,
                                   alpha=0.7, zorder=2,
                                   label=f"{rung}: no-covariates twin, {kind}")
                    elif r_ == rung:
                        err = np.array([[y.mean() - y.min() for y in ys],
                                        [y.max() - y.mean() for y in ys]])
                        ax.errorbar(xt, ym, yerr=err, fmt=mk, color=RUNG_COLOUR[r_], ms=5,
                                    mec="k", mew=0.5, capsize=2, lw=0.8, zorder=4,
                                    label=f"{r_} real g, {kind} pairs")
                    else:
                        ax.scatter(xt, ym, marker=mk, color=RUNG_COLOUR[r_], s=12, alpha=0.4,
                                   zorder=3, label=f"{r_} real g, {kind} pairs")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("true log2 depth ratio  log2(depth_tgt / depth_src)")
        ax.set_ylabel("predicted log2 count scale  log2(Σ mean / Σ X)")
        ax.set_title(GV_NAME[gv])
        ax.grid(color="0.93")
        ax.legend(loc="upper left", fontsize=6, frameon=False)
        lines = []
        for c in sort_checks([c for c in checks if c["check"] == "depthlaw"
                              and c["g_version"] == gv]):
            lines.append(f"{c['mark_class']}: max |scale/ratio - 1| = {fmt(c['value'])} "
                         f"(seed wobble {fmt(c['seed_wobble'])}) vs bar {fmt(c['bar'])} "
                         f"-> {'met' if c['met'] else 'unmet'}")
        if lines:
            ax.text(0.98, 0.02, f"rung {rung}, drafted, not ticked:\n" + "\n".join(lines),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                    bbox={"fc": "white", "ec": "0.6", "lw": 0.5})
    fig.suptitle(f"Depth law (counts): pairs that differ only in depth — rung {rung} "
                 f"highlighted", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _footer(fig, "Circles = trained base<->depth pairs, triangles = never-trained depth<->depth "
                 "pairs, both on the held-out chromosomes. Point = mean over seeds; bar = seed "
                 "range (rung highlighted only).")
    _save(fig, out)


def _describe_rows(ctx, gv="per_track"):
    return [r for r in ctx["res"].get("per_pair", [])
            if r.get("rung") == ctx["rung"] and r.get("kind") == "trained"
            and r.get("eval") == "score" and r.get("model") == "real"
            and r.get("g_version") == gv and r.get("direction", "base_to_arm") == "base_to_arm"
            and isinstance(r.get("describe"), dict)]


def _load_qm(path):
    if not path:
        return None
    with open(path) as fh:
        qm = json.load(fh)
    if qm and set(qm) <= set(SPACES):
        return qm
    return {s: qm for s in SPACES}      # track-keyed: the same curves for both spaces


def fig5_learned_f(ctx, out):
    plt, res, rung = ctx["plt"], ctx["res"], ctx["rung"]
    rows = _describe_rows(ctx)
    arms = sorted({arm_of_pair(r) for r in rows}) or list(ARMS)
    if rung == "A":
        _fig5_a(ctx, rows, arms, out)
        return
    track = ctx["snippet_track"]
    rows_t = [r for r in rows if r["track"] == track]
    fig, axes = plt.subplots(len(SPACES), len(arms), figsize=(2.3 * len(arms), 5.2),
                             squeeze=False)
    qm = _load_qm(ctx.get("refs_qm"))
    for i, space in enumerate(SPACES):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            sub = [r for r in rows_t if r["space"] == space and arm_of_pair(r) == arm]
            if not sub:
                _no_data(ax)
                ax.set_title(f"{arm} · {SPACE_NAME[space]}")
                continue
            levels = sorted({level_of_pair(r) for r in sub})
            cmap = plt.get_cmap("viridis")
            for li, lv in enumerate(levels):
                col = cmap(0.1 + 0.8 * li / max(len(levels) - 1, 1))
                for r in sorted((r for r in sub if level_of_pair(r) == lv),
                                key=lambda r: r["seed"]):
                    d = r["describe"]
                    lab = f"{lv}" if r["seed"] == min(x["seed"] for x in sub) else None
                    if rung == "C" and "kernel" in d:
                        k = np.asarray(d["kernel"], float)
                        off = np.arange(len(k)) - len(k) // 2
                        ax.plot(off, k, color=col, lw=1.0 if r["seed"] == 0 else 0.5,
                                alpha=1.0 if r["seed"] == 0 else 0.5, label=lab)
                    elif rung == "B" and "curve_y" in d:
                        ax.plot(d["knots_x"], d["curve_y"], color=col, marker=".", ms=3,
                                lw=1.0 if r["seed"] == 0 else 0.5,
                                alpha=1.0 if r["seed"] == 0 else 0.5, label=lab)
                    elif rung == "D" and "film_gamma" in d:
                        gm = np.abs(np.asarray(d["film_gamma"], float) - 1).mean(axis=1)
                        bt = np.abs(np.asarray(d["film_beta"], float)).mean(axis=1)
                        L = np.arange(1, len(gm) + 1)
                        ax.plot(L, gm, color=col, marker="o", ms=3, lw=0.8, label=lab)
                        ax.plot(L, bt, color=col, marker="s", ms=3, lw=0.8, ls="--")
                if rung == "B" and qm is not None:
                    tgt = f"{track}__{arm}__{lv}"
                    q = qm.get(space, {}).get(track, {}).get(tgt)
                    if q:
                        kx = np.asarray(q["knots_x"], float)
                        ky = np.asarray(q["knots_y"], float)
                        if space == "counts":
                            qx, qy = np.log1p(np.maximum(kx, 0)), np.log1p(np.maximum(ky, 0))
                        else:
                            qx, qy = np.log(np.maximum(kx, 1e-3)), np.log(np.maximum(ky, 1e-3))
                        kxs = np.asarray(sub[0]["describe"]["knots_x"], float)
                        keep = (qx >= kxs.min()) & (qx <= kxs.max())
                        ax.plot(qx[keep], qy[keep], color=col, ls=":", lw=1.3)
            if rung == "B":
                kxs = np.asarray(sub[0]["describe"]["knots_x"], float)
                ax.plot(kxs, kxs, color="0.6", lw=0.6, ls="-", zorder=0)
                ax.set_xlabel("x = log(1 + X)" if space == "counts" else "x = log(max(p, 1e-3))",
                              fontsize=6)
                if j == 0:
                    ax.set_ylabel("loc (log mean)" if space == "counts" else "loc (log median)")
            elif rung == "C":
                ax.axvline(0, color="0.6", lw=0.6, ls=":")
                ax.set_xlabel("kernel offset (bins of 25 bp)", fontsize=6)
                if j == 0:
                    ax.set_ylabel("kernel weight")
            else:
                ax.set_xlabel("layer (dilation 1, 2, 4, 8)", fontsize=6)
                if j == 0:
                    ax.set_ylabel("mean |γ - 1| (solid), mean |β| (dashed)", fontsize=6)
            ax.set_title(f"{arm} · {SPACE_NAME[space]}")
            ax.legend(fontsize=5.5, frameon=False, loc="best")
            ax.grid(color="0.93")
    what = {"B": "g's monotone curve per arm level (base->arm), solid = seed 0, thin = seeds 1-2",
            "C": "the 33-bin kernel g outputs per arm level (base->arm), solid = seed 0, "
                 "thin = seeds 1-2; identity = one-hot at offset 0",
            "D": "size of the FiLM modulation g outputs per layer and arm level (base->arm); "
                 "0 = identity"}[rung]
    qm_note = ""
    if rung == "B":
        qm_note = (" Dotted = QuantileMatching's per-pair curve (chr1), same colour as its level."
                   if qm is not None else
                   " QuantileMatching curves were not supplied (--refs-qm absent): B's curves "
                   "alone.")
    fig.suptitle(f"The learned f, rung {rung} ({RUNG_NAME[rung]}) — track {track}, "
                 f"{GV_NAME['per_track']}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _footer(fig, what + "." + qm_note + " Grey line = identity (noSolution).")
    _save(fig, out)


def _fig5_a(ctx, rows, arms, out):
    plt, res = ctx["plt"], ctx["res"]
    true_l2 = {(r["source_pid"], r["target_pid"]): r["log2_true"]
               for r in res.get("depth_law", [])}
    panels = [(s, p) for s in SPACES for p in ("a", "b")]
    fig, axes = plt.subplots(len(panels), len(arms), figsize=(2.3 * len(arms), 9.5),
                             squeeze=False)
    tracks = sorted({r["track"] for r in rows})
    for i, (space, par) in enumerate(panels):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            sub = [r for r in rows if r["space"] == space and arm_of_pair(r) == arm]
            if not sub:
                _no_data(ax)
                ax.set_title(f"{arm} · {SPACE_NAME[space]} · {par}")
                continue
            levels = sorted({level_of_pair(r) for r in sub})
            depth_x = arm == "depth" and all((r["source_pid"], r["target_pid"]) in true_l2
                                             for r in sub)
            for ti, track in enumerate(tracks):
                for li, lv in enumerate(levels):
                    rs = [r for r in sub if r["track"] == track and level_of_pair(r) == lv]
                    if not rs:
                        continue
                    v = np.array([r["describe"][par] for r in rs], float)
                    if depth_x:
                        x = true_l2[(rs[0]["source_pid"], rs[0]["target_pid"])] * math.log(2)
                        x += (ti - len(tracks) / 2) * 0.012
                    else:
                        x = li + (ti - len(tracks) / 2) * 0.06
                    ax.errorbar([x], [v.mean()], yerr=[[v.mean() - v.min()], [v.max() - v.mean()]],
                                fmt="o", ms=3.5, color=TRACK_COLOUR.get(track, "0.4"), capsize=1.5,
                                lw=0.7, label=track if li == 0 else None)
            if depth_x:
                xx = np.array(ax.get_xlim())
                if par == "a":
                    ax.plot(xx, xx, color="0.4", ls="--", lw=0.8, label="slope 1")
                ax.set_xlabel("ln(depth_tgt / depth_src)", fontsize=6)
            else:
                ax.set_xticks(range(len(levels)))
                ax.set_xticklabels(levels)
                ax.set_xlim(-0.6, len(levels) - 0.4)
            ax.axhline(1.0 if par == "b" else 0.0, color="0.6", lw=0.6, ls=":")
            ax.set_title(f"{arm} · {SPACE_NAME[space]} · {par}")
            ax.grid(color="0.93")
            if j == 0:
                ax.set_ylabel(f"{par}  (log mean = a + b·x)" if par == "a" else par)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes.flat:
        h, lb = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lb):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper right", ncol=8, frameon=False, fontsize=7)
    fig.suptitle(f"The learned f, rung A ({RUNG_NAME['A']}): a and b per arm level, "
                 f"{GV_NAME['per_track']}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _footer(fig, "base->arm pairs, real g; point = mean over seeds, bar = seed range. Depth: x is "
                 "the natural-log depth ratio and a should follow the dashed slope-1 line if b = 1. "
                 "Dotted = identity (a = 0, b = 1).")
    _save(fig, out)


def _load_npz(ctx, name):
    cache = ctx.setdefault("_npz", {})
    if name not in cache:
        p = figdata_path(ctx["res"], ctx["agg_dir"], name)
        cache[name] = dict(np.load(p)) if p is not None else None
    return cache[name]


def _first_level(ctx, track, arm):
    """The arm's first level in sorted order, read from the figdata / per_pair of that track."""
    lv = set()
    for r in ctx["res"].get("per_pair", []):
        if r.get("track") == track and r.get("kind") == "trained" and \
                r.get("direction") == "base_to_arm" and r.get("arm_tgt") == arm:
            lv.add(r["level_tgt"])
    return sorted(lv)[0] if lv else None


def fig6_meta_profiles(ctx, out):
    plt, res, rung, tbc = ctx["plt"], ctx["res"], ctx["rung"], ctx["tbc"]
    cols = [(s, c) for s in SPACES for c in CLASSES]
    fig, axes = plt.subplots(len(ARMS), len(cols), figsize=(15, 2.0 * len(ARMS)), squeeze=False)
    off_kb = (np.arange(161) - 80) * 25 / 1000
    levels = {}
    for i, arm in enumerate(ARMS):
        for j, (space, cls) in enumerate(cols):
            ax = axes[i][j]
            acc: dict = {}
            for r_ in ctx["rungs"]:
                for track in tbc.get(cls, []):
                    d = _load_npz(ctx, f"{r_}_{track}_{space}_real_s0")
                    if d is None:
                        continue
                    lv = levels.setdefault((track, arm), _first_level(ctx, track, arm))
                    key = f"meta__{track}__base__base__{track}__{arm}__{lv}"
                    if lv is None or key not in d:
                        continue
                    acc.setdefault(r_, []).append(d[key])
            if not acc:
                _no_data(ax, "no such arm" if not any(
                    _first_level(ctx, t, arm) for t in tbc.get(cls, [])) else "no data")
                ax.set_title(f"{arm} · {cls} · {SPACE_NAME[space]}", fontsize=7)
                continue
            ref_r = rung if rung in acc else next(iter(acc))
            m = np.mean(acc[ref_r], axis=0)
            ax.plot(off_kb, m[0], color="0.6", lw=1.0, label="X (source)")
            ax.plot(off_kb, m[1], color="k", lw=1.2, label="X' (target)")
            for r_, arrs in acc.items():
                mu = np.mean(arrs, axis=0)[2]
                if r_ == rung:
                    ax.plot(off_kb, mu, color=RUNG_COLOUR[r_], lw=1.4, label=f"{r_} mean")
                else:
                    ax.plot(off_kb, mu, color=RUNG_COLOUR[r_], lw=0.6, alpha=0.6,
                            label=f"{r_} mean")
            ax.set_title(f"{arm} · {cls} · {SPACE_NAME[space]} (n = {len(acc[ref_r])} tracks)",
                         fontsize=7)
            ax.grid(color="0.93")
            if i == len(ARMS) - 1:
                ax.set_xlabel("offset from the top bin (kb)")
    handles, labels = [], []
    for ax in axes.flat:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="upper right", ncol=6, frameon=False)
    fig.suptitle(f"Peak meta-profiles, rung {rung} ({RUNG_NAME[rung]}) in bold; other rungs thin",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _footer(fig, "Mean over the 500 highest X' bins of chr19 + chr21 (blacklist removed), ±80 bins "
                 "= ±2 kb, for the base->arm pair of the arm's first level; averaged over the "
                 "class's tracks. One g per track, seed 0, real g. Counts: predicted NB mean; "
                 "-log10 p: exp(loc).")
    _save(fig, out)


def fig7_snippets(ctx, out):
    plt, res, rung = ctx["plt"], ctx["res"], ctx["rung"]
    track = ctx["snippet_track"]
    cols = [(s, i) for s in SPACES for i in (0, 1, 2)]
    fig, axes = plt.subplots(len(ARMS), len(cols), figsize=(18, 1.9 * len(ARMS)), squeeze=False)
    kb = np.arange(400) * 25 / 1000
    loci = {}
    for space in SPACES:
        name = f"{rung}_{track}_{space}_real_s0"
        rd = run_dir(res, ctx["agg_dir"], name)
        if rd is not None and (rd / "scores.json").is_file():
            try:
                with open(rd / "scores.json") as fh:
                    sj = json.load(fh)
                for s in (sj.get("snippets", []) if isinstance(sj, dict) else []):
                    loci[(space, s["arm"], int(s["i"]))] = s
            except (OSError, ValueError, KeyError):
                pass
    for i, arm in enumerate(ARMS):
        for j, (space, k) in enumerate(cols):
            ax = axes[i][j]
            d = _load_npz(ctx, f"{rung}_{track}_{space}_real_s0")
            key = f"snip__{arm}__{k}"
            title = f"{arm} · {SPACE_NAME[space]} · {SNIP_RULE_NAME[k]}"
            if d is None or key not in d:
                _no_data(ax, "no such arm" if d is not None else "no data")
                ax.set_title(title, fontsize=6.5)
                continue
            a = d[key]
            ax.fill_between(kb, a[3], a[4], color=RUNG_COLOUR[rung], alpha=0.2, lw=0,
                            label="90 % interval")
            ax.plot(kb, a[0], color="0.6", lw=0.7, label="X (source)")
            ax.plot(kb, a[1], color="k", lw=0.8, label="X' (target)")
            ax.plot(kb, a[2], color=RUNG_COLOUR[rung], lw=0.9, label="predicted mean")
            s = loci.get((space, arm, k))
            if s:
                title += f"\n{s.get('chrom')}:{int(s.get('start_bin', 0)) * 25:,} " \
                         f"({s.get('target_pid', '')})"
            ax.set_title(title, fontsize=6.5)
            ax.grid(color="0.93")
            if i == len(ARMS) - 1:
                ax.set_xlabel("position in the 10 kb locus (kb)")
    handles, labels = [], []
    for ax in axes.flat:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False)
    fig.suptitle(f"Track snippets, rung {rung} ({RUNG_NAME[rung]}) — track {track}, "
                 f"{GV_NAME['per_track']}, seed 0", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _footer(fig, "Loci of 400 bins (10 kb) on chr19/chr21 picked by a fixed seeded rule, never by "
                 "eye: per arm, around the top-1 % bin with the largest |X' - X|, a random top-1 % "
                 "bin and a random background bin; pair = base->arm, the arm's first level. Band = "
                 "5 % to 95 % quantiles of the predicted distribution.")
    _save(fig, out)


def fig8_calibration(ctx, out):
    plt, res, rung = ctx["plt"], ctx["res"], ctx["rung"]
    rows = [(gv, s) for gv in G_VERSIONS for s in SPACES]
    acc: dict = {}
    for r in res.get("per_pair", []):
        if r.get("rung") == rung and r.get("kind") == "trained" and r.get("eval") == "score" \
                and r.get("pit_hist") is not None:
            k = (r["g_version"], r["space"], r["mark_class"], r["model"])
            h = np.asarray(r["pit_hist"], float)
            acc[k] = acc.get(k, 0) + h
    fig, axes = plt.subplots(len(rows), len(CLASSES), figsize=(12, 11), squeeze=False)
    edges = np.linspace(0, 1, 21)
    for i, (gv, space) in enumerate(rows):
        for j, cls in enumerate(CLASSES):
            ax = axes[i][j]
            drawn = False
            for model in MODELS:
                h = acc.get((gv, space, cls, model))
                if h is None or np.sum(h) == 0:
                    continue
                dens = h / h.sum() * len(h)
                if model == "real":
                    ax.bar(edges[:-1], dens, width=1 / len(h), align="edge",
                           color=RUNG_COLOUR[rung], alpha=0.7, label=MODEL_NAME[model])
                else:
                    ax.step(edges, np.r_[dens, dens[-1]], where="post", color=MODEL_COLOUR[model],
                            lw=1.2, label=MODEL_NAME[model])
                drawn = True
            if not drawn:
                _no_data(ax)
            ax.axhline(1.0, color="k", lw=0.7, ls="--")
            ax.set_title(f"{cls} · {SPACE_NAME[space]} · {GV_NAME[gv]}")
            ax.set_xlim(0, 1)
            if i == len(rows) - 1:
                ax.set_xlabel("PIT value")
            if j == 0:
                ax.set_ylabel("density (1 = calibrated)")
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.suptitle(f"Calibration (secondary), rung {rung} ({RUNG_NAME[rung]}): PIT histograms",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _footer(fig, "Randomised PIT over the held-out bins of every trained pair, summed over pairs, "
                 "tracks and the 3 seeds. Flat at 1 = calibrated; a U shape = too narrow; a hump = "
                 "too wide.")
    _save(fig, out)


def fig9_checks(ctx, out):
    plt, rung, checks = ctx["plt"], ctx["rung"], sort_checks(ctx["checks"])
    by_gv = {gv: [c for c in checks if c["g_version"] == gv] for gv in G_VERSIONS}
    n = max([len(v) for v in by_gv.values()] + [1])
    fig, axes = plt.subplots(1, len(G_VERSIONS), figsize=(22, 0.21 * n + 1.6), squeeze=False)
    head = ["check", "metric", "space", "class", "value", "rule", "bar", "seed wobble",
            "reading"]
    for j, gv in enumerate(G_VERSIONS):
        ax = axes[0][j]
        ax.axis("off")
        rows = by_gv[gv]
        if not rows:
            _no_data(ax, f"no checks for {GV_NAME[gv]}")
            continue
        cells, colours = [], []
        for c in rows:
            label = CHECK_SHORT.get(c["check"], c["check"])
            if c["check"] == "beatsbelow":
                below = (c.get("components") or {}).get("rung_below")
                if below:
                    label += f" ({below})"
            reading = "met" if c["met"] else "unmet"
            cells.append([label, METRIC_NAME.get(c["metric"], c["metric"]),
                          SPACE_NAME.get(c["space"], c["space"]), c["mark_class"],
                          fmt(c["value"]), CHECK_RULE.get(c["check"], "?"), fmt(c["bar"]),
                          fmt(c["seed_wobble"]), reading])
            bg = "#d9f2d9" if c["met"] else "#f8d7d7"
            colours.append(["white"] * 8 + [bg])
        tb = ax.table(cellText=cells, colLabels=head, cellColours=colours, loc="upper center",
                      cellLoc="left", colLoc="left",
                      colWidths=[0.2, 0.17, 0.08, 0.08, 0.1, 0.05, 0.08, 0.1, 0.07])
        tb.auto_set_font_size(False)
        tb.set_fontsize(7)
        tb.scale(1, 1.15)
        ax.set_title(f"{GV_NAME[gv]} ({sum(c['met'] for c in rows)} of {len(rows)} met)",
                     fontsize=9)
    fig.suptitle(f"Checks card, rung {rung} ({RUNG_NAME[rung]}) — drafted reading for the PI, "
                 f"NOT ticked: value against bar as computed by the aggregator",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _footer(fig, "Gain checks: value = D_other - D_real (CRPS; positive favours the real g), bar = "
                 "2 x seed wobble (the larger wobble for 'beats rung below'). Depth law: bar 0.10. "
                 "Swap: bar 0.1. Shuffle and swap gate only the main claim. Seed wobble = the "
                 "largest pairwise |difference| over the 3 seeds.")
    _save(fig, out)


FIG_FUNCS = (fig1_ladder, fig2_knob_heatmap, fig3_law_grid, fig4_depth_law, fig5_learned_f,
             fig6_meta_profiles, fig7_snippets, fig8_calibration, fig9_checks)


def draw_all(agg_dir, rung, refs_qm=None, snippet_track=None) -> list[Path]:
    agg_dir = Path(agg_dir)
    res = load_results(agg_dir)
    out_dir = agg_dir / rung / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_with_fig = sorted({n.split("_")[1] for n in res.get("figdata", {})
                              if n.startswith(f"{rung}_")})
    if snippet_track is None:
        snippet_track = DEFAULT_SNIPPET_TRACK if DEFAULT_SNIPPET_TRACK in tracks_with_fig or \
            not tracks_with_fig else next((t for t in tracks_with_fig if t != "all"),
                                          DEFAULT_SNIPPET_TRACK)
    ctx = {"plt": _mpl(), "res": res, "rung": rung, "agg_dir": agg_dir,
           "idx": class_index(res), "tbc": class_tracks(res), "rungs": rungs_present(res),
           "checks": load_checks(agg_dir, rung, res), "refs_qm": refs_qm,
           "snippet_track": snippet_track}
    written = []
    for name, fn in zip(FIG_NAMES, FIG_FUNCS):
        out = out_dir / f"{name}.png"
        fn(ctx, out)
        written.append(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agg_dir")
    ap.add_argument("rung", nargs="?", choices=RUNGS)
    ap.add_argument("--refs-qm", default=None, help="qm_curves.json for figure 5B")
    ap.add_argument("--snippet-track", default=None,
                    help=f"track for figures 5 (B-D) and 7 (default {DEFAULT_SNIPPET_TRACK})")
    ap.add_argument("--check-schema", action="store_true",
                    help="validate results.json against the pinned keys; draw nothing")
    a = ap.parse_args(argv)
    if a.check_schema:
        errs = check_schema(load_results(a.agg_dir))
        for e in errs:
            print(f"SCHEMA ERROR {e}")
        if errs:
            return 1
        print("SCHEMA OK")
        return 0
    if a.rung is None:
        ap.error("rung is required unless --check-schema")
    for p in draw_all(a.agg_dir, a.rung, a.refs_qm, a.snippet_track):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
