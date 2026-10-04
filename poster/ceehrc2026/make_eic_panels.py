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

"CANDI" means CANDI + the training-side output correction (025's `candi.tcfloor`)
everywhere.

Writes panels/eic_{leaderboard,measures}_landscape.{pdf,svg} for
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


tc = pd.read_csv(DATA / "leaderboard_candi.tcfloor.csv")
assert len(tc) == 26
# Not plotted; these are the numbers the section text quotes, printed so a rebuild
# shows them beside the figures.
raw = pd.read_csv(DATA / "leaderboard_candi.none.csv")
r0 = raw[raw.team == "CANDI, uncorrected"].iloc[0]
coef = pd.read_csv(DATA / "tcfloor_coefficients.csv")
print(f"[text] CANDI as it comes: {r0.position} of {len(raw)}, score {r0.score_mean:.3f}; "
      f"correction: {coef.shape[0]} assays x 3 = {3 * coef.shape[0]} numbers, "
      f"fitted on {sorted(set(coef.n_train))} training experiments per assay")

# ================================================================ leaderboard ==
def leaderboard(stem, w, h, fs):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    lab_tr = blended_transform_factory(fig.transFigure, ax.transData)
    n = len(tc)
    ys = np.arange(n)[::-1]
    rank_x = 0.05 + 0.9 * fs / 72            # right edge of the rank column, in
    names = []
    for y, r in zip(ys, tc.itertuples()):
        c, big = colour(r.team), r.team in (CANDI, AVG)
        ax.plot([r.score_lb, r.score_ub], [y, y], color=c, lw=5 if big else 3,
                solid_capstyle="round", zorder=3)
        ax.plot([r.score_mean], [y], "o", ms=0.75 * fs if big else 0.45 * fs,
                color=c, zorder=4)
        kw = dict(fontsize=fs - 4, va="center", color=INK if big else MUTED,
                  fontweight="bold" if big else "normal", transform=lab_tr)
        ax.text(rank_x / w, y, str(r.position), ha="right", **kw)
        names.append(ax.text((rank_x + 0.2 * fs / 72) / w, y, name(r.team),
                             ha="left", **{**kw, "color": c if big else MUTED}))
    fig.canvas.draw()
    left = rank_x + 0.2 * fs / 72 + max(width_in(fig, t) for t in names) + 0.25
    bottom, top, right = 2.9 * fs / 72, 0.12, 0.25
    ax.set_position([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h])
    r1 = tc.iloc[0]
    ax.text(r1.score_mean + 0.012, n - 1, f"{r1.score_mean:.3f}", fontsize=fs - 4,
            color=TEAL, fontweight="bold", va="center")
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(0.15, 0.52)
    ax.set_yticks([])
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=fs - 4, colors=INK)
    ax.grid(axis="x", ls="--", lw=1.4, color="#C9CFD3", zorder=0)
    # Right-aligned to the axis, so on a narrow panel it runs on under the names.
    fig.text(1 - right / w, 0.12 / h, "challenge ranking statistic (lower = better)",
             fontsize=fs - 2, color=INK, ha="right", va="bottom")
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


# (leaderboard w, h), (measures w, h), panel text size in pt.
SIZES = {"_landscape": ((11.4, 11.1), (16.4, 11.1), 22)}
for suffix, (lb, me, fs) in SIZES.items():
    leaderboard(f"eic_leaderboard{suffix}", *lb, fs)
    measures(f"eic_measures{suffix}", *me, fs)
