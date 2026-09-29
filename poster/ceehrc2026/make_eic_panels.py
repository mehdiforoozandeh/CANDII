"""The poster's ENCODE Imputation Challenge (EIC) panels, drawn at print size.

Every number is read from data/, which holds unmodified copies of the tables in
github.com/mlibbrecht/2026-07-26_epi_imputation (main, 2026-09-28):

  leaderboard_candi.tcfloor.csv  025  the published 25-entry field + CANDI with its
                                      output correction; the challenge's own
                                      statistic, nine measures, 23 chromosomes
  leaderboard_candi.none.csv     025  the same field + CANDI's output as it comes
  tcfloor_coefficients.csv       025  the correction's 24 numbers (fitted in 017)
  measure_ratios.csv             025  each measure over the 51 experiments, with the
                                      average-activity baseline at 1
  field_pilot.csv                026  the whole field given a chr22 pilot correction
  skill_shifted.csv              029  the absolute axis, before and after that

"CANDI" means CANDI + the training-side output correction (025's `candi.tcfloor`)
everywhere except where a panel says "as it comes".

Writes panels/eic_{leaderboard,correction,measures,pilot}.pdf. Sizes are in
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
raw = pd.read_csv(DATA / "leaderboard_candi.none.csv")
raw["team"] = raw.team.replace({"CANDI, uncorrected": CANDI})
assert len(tc) == len(raw) == 26

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

# ================================================================= correction ==
# Left: the rank of every entry in the two fields -- with CANDI's output as it
# comes, and with the correction. The other 25 entries are the same tracks both
# times; they move a place or two only because CANDI moved past them.
coef = pd.read_csv(DATA / "tcfloor_coefficients.csv")
fig = plt.figure(figsize=(26.0, 13.0))
axS = fig.add_axes([0.035, 0.12, 0.40, 0.75])
pos0 = raw.set_index("team").position
pos1 = tc.set_index("team").position
for team in pos1.index:
    c, big = colour(team), team in (CANDI, AVG)
    y0, y1 = pos0[team], pos1[team]
    axS.plot([0, 1], [y0, y1], color=c, lw=6 if big else 1.6,
             alpha=1 if big else 0.55, zorder=4 if big else 2)
    for x, y in ((0, y0), (1, y1)):
        axS.plot([x], [y], "o", ms=24 if big else 10, color=c, zorder=5 if big else 3)
    if big:
        axS.text(-0.06, y0, f"{name(team) if team == CANDI else 'baseline'}  {y0}", ha="right", va="center",
                 fontsize=FS - 2, color=c, fontweight="bold")
        axS.text(1.06, y1, f"{y1}  {name(team) if team == CANDI else 'baseline'}", ha="left", va="center",
                 fontsize=FS - 2, color=c, fontweight="bold")
axS.set_ylim(26.8, 0.2)
axS.set_xlim(-1.1, 2.1)
axS.axis("off")
for x, lab in ((0, "as it\ncomes"), (1, "with the\ncorrection")):
    axS.text(x, -0.35, lab, ha="center", va="bottom", fontsize=FS - 2, color=INK,
             fontweight="bold", linespacing=1.15)
axS.text(0.5, 27.3, "rank of each of the 26 entries", ha="center", va="top",
         fontsize=FS - 4, color=MUTED)
r0 = raw.set_index("team").loc[CANDI]
r1 = tc.set_index("team").loc[CANDI]
axS.text(0.5, 28.6, f"CANDI's ranking statistic {r0.score_mean:.3f} → {r1.score_mean:.3f}",
         ha="center", va="top", fontsize=FS - 2, color=TEAL, fontweight="bold")

# Right: the 24 numbers. Per assay, how much of the corrected track comes from
# CANDI and how much from the average-activity baseline.
axC = fig.add_axes([0.62, 0.11, 0.35, 0.74])
coef = coef.sort_values("b1_candi")
yy = np.arange(len(coef))
h = 0.38
axC.barh(yy + h / 2, coef.b1_candi, height=h, color=TEAL, label="β₁  weight on CANDI",
         zorder=3)
axC.barh(yy - h / 2, coef.b2_average, height=h, color=BASE,
         label="β₂  weight on the baseline", zorder=3)
axC.axvline(0, color="#333333", lw=2)
axC.set_yticks(yy)
axC.set_yticklabels(coef.assay_name, fontsize=FS - 2, color=INK)
axC.tick_params(axis="x", labelsize=FS - 4, colors=INK)
axC.tick_params(axis="y", length=0)
axC.set_xlim(-0.15, 1.45)
axC.grid(axis="x", ls="--", lw=1.4, color="#C9CFD3", zorder=0)
for s in ("top", "right", "left"):
    axC.spines[s].set_visible(False)
axC.set_xlabel("fitted coefficient", fontsize=FS - 2, color=INK, labelpad=10)
axC.legend(fontsize=FS - 6, frameon=False, loc="lower center", ncol=1,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.2, borderaxespad=0.2)
save(fig, "eic_correction")

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
fig = plt.figure(figsize=(20.0, 13.0))
L, Wd, B, H = 0.33, 0.64, 0.14, 0.83
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
fig.patches.append(FancyArrowPatch((L + 0.20, 0.075), (L, 0.075),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=40, lw=3, color=INK))
fig.text(L + 0.215, 0.075, "better", fontsize=FS, color=INK, va="center")
lx = 0.02
for mk, kw, lab in (("o", dict(ms=26, color=TEAL), "CANDI"),
                    ("o", dict(ms=20, color=BASE), "average-activity baseline"),
                    ("o", dict(ms=13, mfc="none", mec=FIELD, mew=2.4),
                     "the 24 other entries")):
    fig.lines.append(plt.Line2D([lx], [0.018], marker=mk, transform=fig.transFigure,
                                ls="none", **kw))
    t = fig.text(lx + 0.015, 0.018, lab, fontsize=FS - 4, color=MUTED, va="center")
    lx += 0.045 + 0.0105 * len(lab)
save(fig, "eic_measures")

# ====================================================================== pilot ==
# The absolute axis of 029, flipped so that lower is better (0 = the baseline,
# 1 = the baseline with its positions scrambled). Each arrow is one entry before
# and after the pilot correction. CANDI's "before" is CANDI + its training-side
# correction, as on every other panel.
board = tc[["team", "position"]].rename(columns={"position": "r_none"})
pilot = pd.read_csv(DATA / "field_pilot.csv")[["team", "position"]].rename(
    columns={"position": "r_pilot"})
sk = pd.read_csv(DATA / "skill_shifted.csv").pivot(index="entry", columns="rung",
                                                  values="skill")
sk.loc[CANDI, "none"] = sk.loc["CANDI + tcfloor", "tcfloor"]
sk = 1.0 - sk.drop(index=["CANDI + tcfloor"])
D = (board.merge(pilot, on="team").join(sk[["none", "pilot"]], on="team")
     .dropna(subset=["none", "pilot"]).sort_values("r_none"))
assert len(D) == 26
fig = plt.figure(figsize=(22.0, 14.0))
ax = fig.add_axes([0.37, 0.14, 0.60, 0.78])
XLO, XHI = -0.33, 0.70
ys = np.arange(len(D))[::-1]
for y, r in zip(ys, D.itertuples()):
    c, big = colour(r.team), r.team in (CANDI, AVG)
    x0, off = min(r.none, XHI), r.none > XHI
    ax.add_patch(FancyArrowPatch((x0, y), (max(r.pilot, XLO), y), arrowstyle="-|>",
                                 mutation_scale=26 if big else 18, color=c,
                                 alpha=1 if big else 0.6, lw=4 if big else 2.2,
                                 shrinkA=6, shrinkB=0, zorder=5 if big else 3))
    ax.plot([x0], [y], ">" if off else "o", ms=13 if off else (20 if big else 11),
            color=c, zorder=6 if big else 4)
    kw = dict(fontsize=FS - 6, va="center", transform=ax.get_yaxis_transform(),
              color=INK if big else MUTED, fontweight="bold" if big else "normal")
    ax.text(-0.515, y, str(int(r.r_none)), ha="right", **kw)
    ax.text(-0.44, y, str(int(r.r_pilot)), ha="right", **kw)
    ax.text(-0.42, y, name(r.team), ha="left", **{**kw, "color": c if big else MUTED})
top = len(D) - 0.1
for x, lab in ((-0.515, "as\nsubm."), (-0.44, "pilot"), (-0.42, "")):
    ax.text(x, top, lab, ha="right", va="bottom", fontsize=FS - 8, color=MUTED,
            transform=ax.get_yaxis_transform(), linespacing=1.05)
ax.text(-0.42, top, "rank", ha="left", va="bottom", fontsize=FS - 8, color=MUTED,
        transform=ax.get_yaxis_transform())
ax.axvline(0, color=BASE, lw=2.4, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(XLO, XHI)
ax.set_ylim(-0.8, len(D) + 0.9)
ax.set_yticks([])
for s in ("left", "top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=FS - 6, colors=INK)
ax.set_xlabel("error on an absolute scale  (0 = baseline, 1 = baseline scrambled;"
              " lower is better)", fontsize=FS - 6, color=INK, labelpad=10)
save(fig, "eic_pilot")
