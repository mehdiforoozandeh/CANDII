"""Calibration (b) — what a uniformly-sampled eval window actually CONTAINS, and what that
costs in precision.

The mid-training monitor scores a random subset of the eval chromosome's windows and reports the
number as if it were the whole chromosome's. That is only honest if the subset is a *uniform*
sample -- which is the ruling -- but uniform does not mean cheap: epigenomic signal is clustered,
so most windows are background and the few that are not carry all the spread. This script prices
that, using only the data. No model, no loader, no gradients.

It answers two things:

1. **What is in a window.** Per track: what fraction of bins clear the foreground threshold, what
   fraction of windows contain no foreground bin at all, and how concentrated the signal is.

2. **What a k% sample costs you.** For a metric that is a mean over bins -- MSE, NLL, CRPS all are
   -- the standard error of a k%-window estimate follows in closed form from the between-window
   variance, so the whole coverage-vs-precision curve comes out without simulating anything:

       SE(k) = sqrt( (1 - f) / n_sampled * var_between_windows(per-window score) )

   with `f = n_sampled / n_windows` the finite-population correction. The per-window score used
   here is a SURROGATE -- squared deviation from the track's own mean, i.e. the error a
   predict-the-mean model would make. It is not CANDI's error, and this script does not pretend it
   is. What it measures is how CLUSTERED per-window scores are on this chromosome, and that
   clustering is the whole reason a small uniform sample is imprecise. Treat the resulting curve as
   the shape of the answer; the constant is nailed down by calibration (a), which needs the model.

Runs where the store is (Fir):

    python tools/window_content.py --store /…/CANDI_STORE/eic --chrom chr21 \
        --out /scratch/$USER/candi_kit/window_content --prefix T_
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Window length in BINS. Not a flag: it is `CandiModel.context_length`, and a calibration run at a
#: different window length would be pricing a model we do not train.
CONTEXT_BINS = 768

#: Foreground is the top `FG_FRAC` of a track's own positions, matching `quick_eval(fg_frac=0.02)`.
#: Per track, never pooled -- tracks differ in depth by more than an order of magnitude, so a shared
#: absolute threshold would call whole assays background.
FG_FRAC = 0.02

#: Coverage levels priced, as a fraction of the chromosome's windows. The first is what the shipped
#: monitor does today (8 windows of 2,433 on chr21).
LEVELS = (0.0033, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


def window_scores(x: np.ndarray, context_bins: int = CONTEXT_BINS) -> np.ndarray:
    """Per-window surrogate score: mean squared deviation from the track mean.

    The ragged tail is DROPPED rather than padded. Padding with zeros would invent background and
    pull the last window's score toward zero; a partial window is also not something the model is
    ever handed, because the loader tiles in whole contexts.
    """
    x = np.asarray(x, dtype=np.float64)
    n_win = x.size // context_bins
    if n_win == 0:
        return np.zeros(0, dtype=np.float64)
    w = x[: n_win * context_bins].reshape(n_win, context_bins)
    return ((w - x.mean()) ** 2).mean(axis=1)


def se_curve(scores: np.ndarray, levels=LEVELS) -> list[dict]:
    """Standard error of a k%-window estimate of the whole-chromosome mean, in closed form.

    Includes the finite-population correction, which is not a nicety here: at k=100% the sample IS
    the population and the error must be exactly zero. Without the `(1 - f)` term the table would
    report a non-zero error for scoring the entire chromosome, which is the kind of wrong number a
    reader trusts.
    """
    n = int(scores.size)
    if n == 0:
        return []
    mean = float(scores.mean())
    var = float(scores.var(ddof=1)) if n > 1 else 0.0
    out = []
    for k in levels:
        m = max(1, min(n, int(round(k * n))))
        f = m / n
        se = float(np.sqrt(max(0.0, (1.0 - f) / m * var)))
        out.append({"level": k, "n_windows": m, "se": se,
                    "rel_se": (se / mean if mean > 0 else float("nan"))})
    return out


def foreground_profile(x: np.ndarray, context_bins: int = CONTEXT_BINS,
                       fg_frac: float = FG_FRAC) -> dict:
    """What the windows hold: how much foreground, and how much of it a window sample can miss."""
    x = np.asarray(x, dtype=np.float64)
    n_win = x.size // context_bins
    w = x[: n_win * context_bins].reshape(n_win, context_bins)
    thr = float(np.quantile(x, 1.0 - fg_frac))
    fg = w > thr
    per_win_fg = fg.sum(axis=1)
    total_fg = int(per_win_fg.sum())
    # Signal MASS, not bin count: a window can hold one enormous bin and a thousand empty ones, and
    # a metric averaged over bins feels the mass.
    per_win_mass = w.sum(axis=1)
    order = np.sort(per_win_mass)[::-1]
    mass = float(per_win_mass.sum())
    top10 = float(order[: max(1, n_win // 10)].sum() / mass) if mass > 0 else float("nan")
    return {
        "n_windows": int(n_win),
        "fg_threshold": thr,
        "frac_bins_foreground": float(fg.mean()),
        "frac_windows_with_no_foreground": float((per_win_fg == 0).mean()),
        "frac_bins_zero": float((x == 0).mean()),
        "mass_in_top_10pct_windows": top10,
        "total_foreground_bins": total_fg,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chrom", default="chr21")
    p.add_argument("--prefix", default="T_")
    p.add_argument("--space", default="pval", choices=["pval", "counts"])
    p.add_argument("--max-tracks", type=int, default=0, help="0 = every track")
    args = p.parse_args(argv)

    from candi.store.reader import CorpusStore

    store = CorpusStore(args.store)
    train = [b for b in store.biosamples if b.startswith(args.prefix)]
    tracks = [(b, a) for b in train for a in store[b].assays(args.space)]
    if args.max_tracks:
        tracks = tracks[: args.max_tracks]
    if not tracks:
        raise SystemExit(f"no {args.space} track under prefix {args.prefix!r} in {args.store}")
    print(f"[wc] {len(train)} biosamples, {len(tracks)} {args.space} tracks, {args.chrom}",
          flush=True)

    t0 = time.time()
    rows, curves = [], []
    for i, (b, a) in enumerate(tracks):
        view = store[b][a]
        n_bins = int(view.n_bins(args.chrom)) if hasattr(view, "n_bins") else None
        # `start` is REQUIRED on every reader accessor (t18 found this the hard way).
        x = np.asarray(getattr(view, args.space)(args.chrom, 0, n_bins), dtype=np.float64)
        prof = foreground_profile(x)
        prof.update(biosample=b, assay=a)
        rows.append(prof)
        curves.append(se_curve(window_scores(x)))
        if (i + 1) % 25 == 0:
            print(f"[wc]   {i + 1}/{len(tracks)} ({time.time() - t0:.0f}s)", flush=True)

    # Average the RELATIVE standard errors across tracks, not the absolute ones. Tracks differ in
    # scale by orders of magnitude, so a mean of absolute SEs is the deepest track's SE and nothing
    # else; the relative figure is the one that is comparable and the one a coverage choice needs.
    agg = []
    for j, k in enumerate(LEVELS):
        rel = np.array([c[j]["rel_se"] for c in curves if c], dtype=np.float64)
        rel = rel[np.isfinite(rel)]
        agg.append({"level": k, "n_windows": curves[0][j]["n_windows"],
                    "rel_se_median": float(np.median(rel)) if rel.size else float("nan"),
                    "rel_se_p90": float(np.quantile(rel, 0.90)) if rel.size else float("nan")})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "window_content.json").write_text(json.dumps(
        {"store": args.store, "chrom": args.chrom, "space": args.space,
         "context_bins": CONTEXT_BINS, "fg_frac": FG_FRAC,
         "n_tracks": len(tracks), "tracks": rows, "coverage_curve": agg}, indent=2))

    def med(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    lines = [
        f"# Calibration (b) — what a uniform eval window holds on `{args.chrom}`",
        "",
        f"- store `{args.store}`, space **{args.space}**, prefix `{args.prefix}`",
        f"- **{len(tracks)}** tracks over **{len(train)}** biosamples",
        f"- window = **{CONTEXT_BINS} bins = {CONTEXT_BINS * 25:,} bp**; "
        f"**{rows[0]['n_windows']:,}** whole windows tile {args.chrom}",
        f"- foreground = each track's own top **{FG_FRAC:.0%}** of positions",
        f"- built in {(time.time() - t0) / 60:.1f} min",
        "",
        "## What is in a window (median over tracks)",
        "",
        "| | median |",
        "|---|---:|",
        f"| bins that are exactly zero | {med('frac_bins_zero'):.1%} |",
        f"| bins above the foreground threshold | {med('frac_bins_foreground'):.2%} |",
        f"| **windows holding no foreground bin at all** | "
        f"**{med('frac_windows_with_no_foreground'):.1%}** |",
        f"| signal mass sitting in the top 10% of windows | "
        f"{med('mass_in_top_10pct_windows'):.1%} |",
        "",
        "## What a uniform k% sample costs in precision",
        "",
        "Relative standard error of the whole-chromosome mean, estimated from k% of windows.",
        "Closed form, finite-population corrected, so 100% is exactly 0 by construction.",
        "The per-window score is a **surrogate** — a predict-the-mean model's squared error — so",
        "this is the SHAPE of the curve, not CANDI's own error. Calibration (a) fixes the constant.",
        "",
        "| coverage | windows/track | median rel. SE | p90 rel. SE |",
        "|---|---:|---:|---:|",
    ]
    for a_ in agg:
        note = "  ← shipped default" if abs(a_["level"] - 0.0033) < 1e-9 else ""
        lines.append(f"| {a_['level']:.2%}{note} | {a_['n_windows']:,} | "
                     f"{a_['rel_se_median']:.3f} | {a_['rel_se_p90']:.3f} |")
    lines += ["", "## Per track", "",
              "| biosample | assay | zero bins | fg bins | windows w/o fg | mass in top 10% win |",
              "|---|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["assay"], r["biosample"])):
        lines.append(f"| {r['biosample']} | {r['assay']} | {r['frac_bins_zero']:.1%} | "
                     f"{r['frac_bins_foreground']:.2%} | "
                     f"{r['frac_windows_with_no_foreground']:.1%} | "
                     f"{r['mass_in_top_10pct_windows']:.1%} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[wc] wrote {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
