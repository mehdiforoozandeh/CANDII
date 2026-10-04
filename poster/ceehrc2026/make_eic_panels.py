"""The poster's ENCODE Imputation Challenge (EIC) panels, drawn at print size.

Every number is read from data/, which holds unmodified copies of the tables in
github.com/mlibbrecht/2026-07-26_epi_imputation (main, 2026-09-28):

  leaderboard_candi.tcfloor.csv  025  the published 25-entry field + CANDI with its
                                      output correction; the challenge's own
                                      statistic, nine measures, 23 chromosomes
  leaderboard_candi.none.csv     025  the same field + CANDI's output as it comes
                                      (quoted in the text: 14th, 0.351)
  tcfloor_coefficients.csv       025  the correction's 24 numbers (fitted in 017)
  measure_ratios.csv             025  each measure over the 51 experiments, with the
                                      average-activity baseline at 1
  scores_official.csv            025  every arm's nine measures per blind-test
                                      experiment (commit 5218f31a); the Pearson r
                                      panel reads gwcorr, common grid, all 23
                                      chromosomes
  field_none.csv                 026  the whole field re-scored on 22 chromosomes
                                      (chr22 held out), every entry as submitted and
                                      CANDI uncorrected; the challenge's statistic
  field_pilot.csv                026  the same field after the same correction for
                                      every entry: max(b0 + b1 * prediction + b2 *
                                      Average, 0), fitted on chr22 of each blind
                                      experiment (commit 0ed50ce, 2026-09-07)

"CANDI" means CANDI + the training-side output correction (025's `candi.tcfloor`)
everywhere.

Writes panels/eic_{pearson,ranks,measures}_landscape.{pdf,svg} for
poster_landscape.tex, drawn at the size they are printed at, so a point here is a
point there.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "panels"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # the landing-page panels use it too
    "pdf.fonttype": 42,
    "axes.linewidth": 2.0,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 2.0, "xtick.major.size": 9,
    "ytick.major.width": 2.0, "ytick.major.size": 9,
})

# The poster's pastel palette (common.tex): CANDI is coral, as in the schematic.
INK, MUTED, RULE = "#1B2A32", "#5E6E78", "#B9C2C7"
TEAL = "#D46A5A"          # CANDI (name kept: the colour was teal)
BASE = "#E2B35C"          # the average-activity baseline
FIELD = "#A3AFB6"         # the 24 other submitted entries

FS = 30                   # body text in the panels, pt
CANDI, AVG = "CANDI", "Average"
SHORT = {"Hongyang Li and Yuanfang Guan": "HLYG",
         "Hongyang Li and Yuanfang Guan v1": "HLYG v1",
         "Hongyang Li and Yuanfang Guan v2": "HLYG v2",
         "UIOWA Michaelson Lab": "UIOWA Michaelson",
         "KKT-ENCODE-Impute": "KKT-ENCODE",
         "Aug2019Imputation": "Aug2019Imp.",
         "Average": "average-activity baseline",
         "CANDI, uncorrected": "CANDI"}


def colour(team):
    return TEAL if team == CANDI else BASE if team == AVG else FIELD


def name(team):
    return SHORT.get(team, team)


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white")
    w, h = fig.get_size_inches()
    print(f"wrote panels/{stem}.{{pdf,svg}}  ({w:.1f} x {h:.1f} in)")
    plt.close(fig)


def width_in(fig, text):
    """Width of a drawn text artist, in inches."""
    return text.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


# Not plotted; these are the numbers the section text quotes, printed so a rebuild
# shows them beside the figures.
raw = pd.read_csv(DATA / "leaderboard_candi.none.csv")
r0 = raw[raw.team == "CANDI, uncorrected"].iloc[0]
coef = pd.read_csv(DATA / "tcfloor_coefficients.csv")
print(f"[text] CANDI as it comes: {r0.position} of {len(raw)}, score {r0.score_mean:.3f}; "
      f"correction: {coef.shape[0]} assays x 3 = {3 * coef.shape[0]} numbers, "
      f"fitted on {sorted(set(coef.n_train))} training experiments per assay")

# ====================================================================== ranks ==
def ranks(stem, w, h, fs):
    """Each entry's place on the challenge's statistic before the correction (left)
    and after the same correction for every entry (right), one line per entry.
    Both sides: the same 26 entries, the same scorer, the same 22 chromosomes."""
    a = pd.read_csv(DATA / "field_none.csv").set_index("team").position
    b = pd.read_csv(DATA / "field_pilot.csv").set_index("team").position
    assert len(a) == len(b) == 26 and set(a.index) == set(b.index)
    print(f"[B] before: CANDI {a[CANDI]}, baseline {a[AVG]}; after the same "
          f"correction: CANDI {b[CANDI]}, baseline {b[AVG]} (of 26)")
    label = lambda t: "baseline" if t == AVG else name(t)   # the long name fits neither side
    fig = plt.figure(figsize=(w, h))
    names = {}
    for side, pos, ha in (("l", a, "right"), ("r", b, "left")):
        names[side] = [fig.text(0, 0, f"{label(t)}  {p}", fontsize=fs - 4, ha=ha,
                                fontweight="bold" if t in (CANDI, AVG) else "normal")
                       for t, p in pos.items()]
    fig.canvas.draw()
    lw_ = {k: max(width_in(fig, t) for t in v) for k, v in names.items()}
    for v in names.values():
        for t in v:
            t.remove()
    top, bottom = 1.0, 0.25                       # inches: column titles, then rows
    xl, xr = 0.1 + lw_["l"] + 0.3, w - 0.1 - lw_["r"] - 0.3     # 0.3: the text's offset from the dots
    ax = fig.add_axes([xl / w, bottom / h, (xr - xl) / w, 1 - (top + bottom) / h])
    ax.set_xlim(0, 1)
    ax.set_ylim(26.6, 0.4)
    ax.axis("off")
    for t in a.index:                              # the field first, CANDI and the baseline on top
        big = t in (CANDI, AVG)
        ax.plot([0, 1], [a[t], b[t]], color=colour(t), lw=5 if big else 2,
                alpha=1 if big else .7, zorder=3 if big else 2, solid_capstyle="round")
        ax.plot([0, 1], [a[t], b[t]], "o", ms=.55 * fs if big else .3 * fs,
                color=colour(t), zorder=4 if big else 2)
        kw = dict(fontsize=fs - 4, va="center", color=colour(t) if big else MUTED,
                  fontweight="bold" if big else "normal", transform=ax.transData)
        ax.text(-0.04, a[t], f"{label(t)}  {a[t]}", ha="right", **kw)
        ax.text(1.04, b[t], f"{b[t]}  {label(t)}", ha="left", **kw)
    for x, ha, lab in ((xl, "right", "before"), (xr, "left", "after the same\ncorrection")):
        fig.text(x / w, 1 - 0.12 / h, lab, fontsize=fs - 2, color=INK, ha=ha, va="top",
                 fontweight="bold", linespacing=1.1)
    save(fig, stem)


# =================================================================== measures ==
# Each measure on its own scale, with the baseline at 1. Seven are errors and two
# are correlations; the two correlation rows are drawn reversed, so that left is
# better on every row.
ratios = pd.read_csv(DATA / "measure_ratios.csv")
ratios = ratios[ratios.method != "Avocado, retrained"].set_index("method")
assert len(ratios) == 26
MEAS = [("mse", "MSE, genome-wide", False),
        ("mseprom", "MSE, promoters", False),
        ("msegene", "MSE, gene bodies", False),
        ("mseenh", "MSE, enhancers", False),
        ("msevar", "MSE, variance-weighted", False),
        ("mse1obs", "MSE, top 1% observed", False),
        ("gwcorr", "Pearson r (reversed)", True),
        ("gwspear", "Spearman ρ (reversed)", True)]


def measures(stem, w, h, fs):
    rng = np.random.default_rng(4)
    fig = plt.figure(figsize=(w, h))
    labs = [fig.text(0, 0, lab, fontsize=fs - 2) for _, lab, _ in MEAS]
    fig.canvas.draw()
    lab_w = max(width_in(fig, t) for t in labs)
    for t in labs:
        t.remove()
    # Inches: the label column, then the strips; below them the arrow row and
    # the key row.
    Lin = 0.1 + lab_w + 0.25
    L, Wd = Lin / w, 1 - (Lin + 0.15) / w
    row = 1.5 * fs / 72
    B, H = (2 * row + 0.1) / h, 1 - (2 * row + 0.25) / h
    rh = H / len(MEAS)
    for i, (key, lab, rev) in enumerate(MEAS):
        ax = fig.add_axes([L, B + H - (i + 1) * rh, Wd, rh * 0.92])
        v = ratios[key].astype(float)
        v = -v if rev else v          # left is better on every row
        # A few entries are orders of magnitude off on the MSE family; the axis
        # stops at the Tukey fence and anything past it is a caret on the right.
        q1, q3 = v.quantile([0.25, 0.75])
        lo, hi = v.min(), min(v.max(), q3 + 1.5 * (q3 - q1))
        pad = 0.10 * (hi - lo if hi > lo else 1.0)
        for team, val in v.items():
            off = val > hi
            x = hi + pad if off else val
            if team == CANDI:
                ax.plot([x], [0], ">" if off else "o", ms=0.87 * fs, color=TEAL, zorder=6)
            elif team == AVG:
                ax.plot([x], [0], ">" if off else "o", ms=0.67 * fs, color=BASE, zorder=5)
            else:
                ax.plot([x], [rng.uniform(-0.33, 0.33)], ">" if off else "o",
                        ms=0.43 * fs, mfc=FIELD if off else "none", mec=FIELD,
                        mew=0.08 * fs, zorder=3)
        ax.set_xlim(lo - pad, hi + 2 * pad)
        ax.set_ylim(-0.62, 0.62)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "left", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#DDE3E9")
        fig.text(L - 0.25 / w, B + H - (i + 0.5) * rh, lab, ha="right", va="center",
                 fontsize=fs - 2, color=INK)
    ya, yk = (row + 0.1 + row / 2) / h, (0.1 + row / 2) / h
    aw = min(4.0, 0.3 * Wd * w)                 # arrow length, in
    fig.patches.append(FancyArrowPatch((L + aw / w, ya), (L, ya),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=1.3 * fs, lw=0.1 * fs, color=INK))
    fig.text(L + (aw + 0.15) / w, ya, "better", fontsize=fs, color=INK, va="center")
    x = 0.15                                    # the key, left to right, in
    for kw, lab in ((dict(ms=0.87 * fs, color=TEAL), "CANDI"),
                    (dict(ms=0.67 * fs, color=BASE), "average-activity baseline"),
                    (dict(ms=0.43 * fs, mfc="none", mec=FIELD, mew=0.08 * fs),
                     "the 24 other entries")):
        x += 0.45 * fs / 72
        fig.lines.append(plt.Line2D([x / w], [yk], marker="o", transform=fig.transFigure,
                                    ls="none", **kw))
        t = fig.text((x + 0.5 * fs / 72) / w, yk, lab, fontsize=fs - 4, color=MUTED,
                     va="center")
        fig.canvas.draw()
        x += 0.5 * fs / 72 + width_in(fig, t) + 0.35
    save(fig, stem)


# ============================================================= Pearson r box ==
def pearson(stem, w, h, fs):
    """Genome-wide Pearson r of CANDI against each of the 51 blind-test
    experiments, one box per assay, assays from best to worst median. Each grey
    point is one experiment (one cell type)."""
    sc = pd.read_csv(DATA / "scores_official.csv")
    sc = sc[(sc.grid == "common") & (sc.chrom_set == "all23")]
    cd = sc[sc.arm == "candi.tcfloor"]
    assert len(cd) == 51
    order = cd.groupby("assay_name").gwcorr.median().sort_values(ascending=False).index
    print("[A] CANDI median Pearson r by assay:",
          ", ".join(f"{a} {cd[cd.assay_name == a].gwcorr.median():.2f}" for a in order))
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([1.45 / w, 2.75 / h, 1 - 1.65 / w, 1 - 3.0 / h])
    data = [cd[cd.assay_name == a].gwcorr.values for a in order]
    bp = ax.boxplot(data, widths=.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#A3302A", lw=3.2),
                    boxprops=dict(facecolor="#F6C9BE", edgecolor="#8A969C", lw=2.0),
                    whiskerprops=dict(color="#333333", lw=2.0),
                    capprops=dict(color="#333333", lw=2.0), zorder=2)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-.16, .16, len(v)), v, s=fs * 5.0, color="#9AA4AA",
                   alpha=.75, edgecolor="none", zorder=3)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=90, fontsize=fs, color=INK)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.tick_params(axis="y", labelsize=fs, colors=INK)
    ax.set_ylabel("Pearson r (genome-wide)", fontsize=fs, color=INK)
    ax.grid(axis="both", color="#E3E6E8", lw=1.2, zorder=0)
    ax.set_axisbelow(True)
    save(fig, stem)


# (Pearson w, h), (ranks w, h), (measures w, h), panel text size in pt.
SIZES = {"_landscape": ((7.0, 10.15), (8.8, 10.15), (11.8, 10.15), 22)}
for suffix, (pr, rk, me, fs) in SIZES.items():
    pearson(f"eic_pearson{suffix}", *pr, fs)
    ranks(f"eic_ranks{suffix}", *rk, fs)
    measures(f"eic_measures{suffix}", *me, fs)
