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

Writes panels/eic_{leaderboard,measures}.pdf. Sizes are in
inches as printed on the 44-inch-wide poster, so a point here is a point there.
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

# The landing page's palette: CANDI is its teal, observed data its grey.
INK, MUTED, RULE = "#1B2A32", "#5E6E78", "#B9C2C7"
TEAL = "#12868C"
BASE = "#C98A1B"          # the average-activity baseline
FIELD = "#8A98A0"         # the 24 other submitted entries

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
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    w, h = fig.get_size_inches()
    print(f"wrote panels/{stem}.pdf  ({w:.1f} x {h:.1f} in)")


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
fig = plt.figure(figsize=(15.0, 17.0))
ax = fig.add_axes([0.46, 0.085, 0.51, 0.885])
lab_tr = blended_transform_factory(fig.transFigure, ax.transData)
n = len(tc)
ys = np.arange(n)[::-1]
for y, r in zip(ys, tc.itertuples()):
    c, big = colour(r.team), r.team in (CANDI, AVG)
    ax.plot([r.score_lb, r.score_ub], [y, y], color=c, lw=5 if big else 3,
            solid_capstyle="round", zorder=3)
    ax.plot([r.score_mean], [y], "o", ms=22 if big else 13, color=c, zorder=4)
    kw = dict(fontsize=FS - 4, va="center", color=INK if big else MUTED,
              fontweight="bold" if big else "normal", transform=lab_tr)
    ax.text(0.055, y, str(r.position), ha="right", **kw)
    ax.text(0.070, y, name(r.team), ha="left", **{**kw, "color": c if big else MUTED})
r1 = tc.iloc[0]
ax.text(r1.score_mean + 0.012, n - 1, f"{r1.score_mean:.3f}", fontsize=FS - 4,
        color=TEAL, fontweight="bold", va="center")
ax.set_ylim(-0.8, n - 0.2)
ax.set_xlim(0.15, 0.52)
ax.set_yticks([])
for s in ("left", "top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=FS - 4, colors=INK)
ax.grid(axis="x", ls="--", lw=1.4, color="#C9CFD3", zorder=0)
ax.set_xlabel("challenge ranking statistic (lower = better)", fontsize=FS - 2,
              color=INK, labelpad=12)
save(fig, "eic_leaderboard")

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
        ("mse1imp", "MSE, top 1% imputed", False),
        ("gwcorr", "Pearson r (reversed)", True),
        ("gwspear", "Spearman ρ (reversed)", True)]
rng = np.random.default_rng(4)
fig = plt.figure(figsize=(27.5, 16.0))
L, Wd, B, H = 0.25, 0.72, 0.13, 0.85
rh = H / len(MEAS)
for i, (key, lab, rev) in enumerate(MEAS):
    ax = fig.add_axes([L, B + H - (i + 1) * rh, Wd, rh * 0.92])
    v = ratios[key].astype(float)
    v = -v if rev else v          # left is better on every row
    # A few entries are orders of magnitude off on the MSE family; the axis stops
    # at the Tukey fence and anything past it is a caret on the right edge.
    q1, q3 = v.quantile([0.25, 0.75])
    lo, hi = v.min(), min(v.max(), q3 + 1.5 * (q3 - q1))
    pad = 0.10 * (hi - lo if hi > lo else 1.0)
    for team, val in v.items():
        off = val > hi
        x = hi + pad if off else val
        if team == CANDI:
            ax.plot([x], [0], ">" if off else "o", ms=26, color=TEAL, zorder=6)
        elif team == AVG:
            ax.plot([x], [0], ">" if off else "o", ms=20, color=BASE, zorder=5)
        else:
            ax.plot([x], [rng.uniform(-0.33, 0.33)], ">" if off else "o",
                    ms=13, mfc=FIELD if off else "none", mec=FIELD, mew=2.4, zorder=3)
    ax.set_xlim(lo - pad, hi + 2 * pad)
    ax.set_ylim(-0.62, 0.62)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#DDE3E9")
    fig.text(L - 0.012, B + H - (i + 0.5) * rh, lab, ha="right", va="center",
             fontsize=FS - 2, color=INK)
fig.patches.append(FancyArrowPatch((L + 0.145, 0.07), (L, 0.07),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=40, lw=3, color=INK))
fig.text(L + 0.156, 0.07, "better", fontsize=FS, color=INK, va="center")
lx = 0.02
for mk, kw, lab in (("o", dict(ms=26, color=TEAL), "CANDI"),
                    ("o", dict(ms=20, color=BASE), "average-activity baseline"),
                    ("o", dict(ms=13, mfc="none", mec=FIELD, mew=2.4),
                     "the 24 other entries")):
    fig.lines.append(plt.Line2D([lx], [0.02], marker=mk, transform=fig.transFigure,
                                ls="none", **kw))
    t = fig.text(lx + 0.011, 0.02, lab, fontsize=FS - 4, color=MUTED, va="center")
    lx += 0.033 + 0.0076 * len(lab)
save(fig, "eic_measures")
