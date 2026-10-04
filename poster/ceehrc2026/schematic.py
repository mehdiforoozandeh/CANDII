"""The poster's CANDI schematic workflow, and the A and B result panels.

Started as landing/build_ga.py (the graphical abstract on the landing page, kept
there verbatim) and changed for the poster:
- the poster's colours: dark red for measured and denoised signal, salmon for
  imputed signal, neutral grey for missing experiments;
- the tensor is the real ENCODE Imputation Challenge data (data/
  eic_experiments.csv): 35 assays x 51 cell types, 363 of them measured;
- no signal heatmaps: one cell type goes straight to its tracks, and the 30 kb
  window is marked on the tracks;
- the signal head's output passes through the per-assay output layer, with the
  average-activity track as its second input.

Run by split_ga.py, which crops the strip and panels A and B out of `fig`.
"""
import csv
from collections import Counter
import json
from math import atan2, degrees
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    # Text as vector paths, not <text> elements. With "none" the SVG only names
    # the font, so any browser without DejaVu Sans silently substitutes a serif
    # and the web page stops matching the PNG and PDF. Paths cost file size and
    # make the text unselectable; matching output everywhere is worth it.
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
})

# Paths resolve from this file, not the working directory, so the script runs
# from anywhere.
HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "landing" / "digitized.json").read_text())

INK, MUTED, RULE = "#1B2A32", "#5E6E78", "#B9C2C7"
TEAL, TEAL_IMP = "#A3302A", "#F4A08F"
MISS_C = "#E3E3E3"          # experiment never done
MASK_C = "#F2DFA8"          # held out during self-supervised training
COV_C = "#CFE7D4"           # covariates, green as in Figure 1
# Observed data is grey in BOTH rows: the noisy input track up top is the same
# thing the results panels call "Observed".
SRC = {"Observed": "#9AA7AF", "Denoised": "#7FAED6",
       "Denoised+Imputed": "#86C295", "Latent": "#D46A5A"}
OBS = SRC["Observed"]

# The workflow strip carries far more information than the result row, so it
# gets the lion's share of the page. Its own aspect is locked (below), which
# makes it width-limited — the ratios here are tuned so the result row is just
# small enough that no vertical slack opens up between them.
fig = plt.figure(figsize=(13.2, 9.4))
outer = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0],
                         left=0.055, right=0.980, top=0.972, bottom=0.105,
                         hspace=0.08)

# ================================================== the workflow strip =======
# Drawn in one axes with an isotropic coordinate system, so the cabinet
# projection stays square and every arrow can be placed by hand. Heatmaps and
# track panels are inset axes positioned in those same data coordinates.
AX = fig.add_subplot(outer[0])
AX.set_xlim(0, 100)
AX.set_ylim(0, 46)
AX.set_aspect("equal")
AX.axis("off")

# ------------------------------------------------------- the data, schematic -
# Six assays, not the full panel: the window tracks have to be tall enough that
# the confidence band around each prediction is legible, which is the whole
# point of the output side.
ASSAYS = ["DNase-seq", "H3K4me3", "H3K4me1", "H3K27ac", "H3K27me3", "ChIP control"]
N_ASSAY = len(ASSAYS)
MISSING = [2, 4]                        # never measured in the cell type we follow
MASKED = [3]                            # measured, but held out in training

# ------------------------------------------------------------- the data, real -
# The tensor is the ENCODE Imputation Challenge as released: 35 assays x 51 cell
# types, 363 experiments. Training, validation and blind experiments are all
# simply "measured" here. Assays and cell types both run from most to least
# measured, so the dense corner reads at a glance.
_eic = list(csv.DictReader(open(HERE / "data" / "eic_experiments.csv")))
EIC_ASSAYS = [a for a, _ in Counter(r["mark/assay"] for r in _eic).most_common()]
EIC_CTS = [c for c, _ in Counter(r["cell_type_id"] for r in _eic).most_common()]
avail = np.zeros((len(EIC_ASSAYS), len(EIC_CTS)), bool)
for r in _eic:
    avail[EIC_ASSAYS.index(r["mark/assay"]), EIC_CTS.index(r["cell_type_id"])] = True
N_TA, N_CT = avail.shape
# upper lobe of left lung: DNase-seq, H3K4me3 and H3K27ac measured, H3K4me1 and
# H3K27me3 not, as in the tracks
K = EIC_CTS.index("C47")

# The window itself is drawn at finer resolution, as raw counts.
NB = 64
_wx = np.arange(NB)
_wr = np.random.default_rng(3)
_wpk = _wr.choice(NB, 9, replace=False)


def window_truth(strength, seed):
    r = np.random.default_rng(seed)
    v = np.full(NB, .05)
    for c in _wpk:
        if r.random() < strength:
            v += r.uniform(.5, 1.) * np.exp(-.5 * ((_wx - c) / r.uniform(1.4, 3.2)) ** 2)
    return v


truth = np.vstack([window_truth(.55 - .03 * i, 70 + i) for i in range(N_ASSAY)])
RATE = truth * 26.0                                        # expected read counts


def spikiness(seed):
    """Real tracks carry sharp features a smooth prediction rides straight over.

    Those are exactly what the confidence interval has to cover, so the observed
    counts get them and the predicted mean does not.
    """
    r = np.random.default_rng(seed)
    s = np.ones(NB)
    for c in r.choice(_wpk, 4, replace=False):
        s += r.uniform(.45, 1.05) * np.exp(
            -.5 * ((_wx - (c + r.integers(-2, 3))) / r.uniform(.5, 1.2)) ** 2)
    return s


counts = np.random.default_rng(8).poisson(
    RATE * np.vstack([spikiness(120 + i) for i in range(N_ASSAY)])).astype(float)


def smooth_bias(seed):
    """CANDI's mean is close to the truth but not identical.

    A slowly varying multiplicative error, so the predicted mean visibly misses
    some of the real signal and the confidence interval has to do the work of
    covering it — which is the whole point of the calibration result.
    """
    r = np.random.default_rng(seed)
    k = r.uniform(.72, 1.28, 8)
    b = np.interp(np.linspace(0, 1, NB), np.linspace(0, 1, 8), k)
    return np.convolve(np.pad(b, 4, mode="edge"), np.ones(9) / 9, "same")[4:-4]


PRED = RATE * np.vstack([smooth_bias(90 + i) for i in range(N_ASSAY)])
SD = 1.96 * np.sqrt(PRED) + .30 * PRED       # Poisson term plus overdispersion
# per-track scaling, the way a genome browser autoscales each row
VAL = np.maximum(counts.max(1), (PRED + SD).max(1))[:, None]
OBS_N, MU_N, SD_N = counts / VAL, PRED / VAL, SD / VAL


# ------------------------------------------------------------------ the cube -
def cube(x0, y0, cw, ch, dx, dy, filled, labels=False):
    """Front face = assays x cell types; depth = the genome.

    One square on the front is one experiment of the real challenge, and the
    lanes ruled across the top and right faces show those same experiments end
    to end: a missing one is a thread absent along the entire genome.
    """
    fx, fy = cw / N_CT, ch / N_TA
    ty, rx = y0 + ch, x0 + cw
    SHADE = {"front": 1.00, "right": .60, "top": .40}   # 3D shading, not data
    EDGE = dict(edgecolor="white", lw=.15)

    def thread(present, face):
        if present:
            return TEAL, SHADE[face]
        return (TEAL_IMP, 0.8 * SHADE[face]) if filled else ("#BDBDBD", SHADE[face])

    # top face: one lane per cell type (the top assay row), along the genome
    for c in range(N_CT):
        fc, al = thread(avail[0, c], "top")
        AX.add_patch(Polygon([(x0 + c * fx, ty), (x0 + (c + 1) * fx, ty),
                              (x0 + (c + 1) * fx + dx, ty + dy),
                              (x0 + c * fx + dx, ty + dy)],
                             facecolor=fc, alpha=al, zorder=2, **EDGE))
    # right face: one lane per assay (the last cell type column), same idea
    for a in range(N_TA):
        ry = y0 + (N_TA - 1 - a) * fy
        fc, al = thread(avail[a, N_CT - 1], "right")
        AX.add_patch(Polygon([(rx, ry), (rx, ry + fy), (rx + dx, ry + fy + dy),
                              (rx + dx, ry + dy)],
                             facecolor=fc, alpha=al, zorder=2, **EDGE))
    # front face: the most-measured assay on top
    for a in range(N_TA):
        ry = y0 + (N_TA - 1 - a) * fy
        for c in range(N_CT):
            fc, al = thread(avail[a, c], "front")
            AX.add_patch(Rectangle((x0 + c * fx, ry), fx, fy, facecolor=fc,
                                   alpha=al, zorder=4, **EDGE))

    sil = dict(fill=False, edgecolor="#6C7A80", lw=.9, zorder=5)
    AX.add_patch(Rectangle((x0, y0), cw, ch, **sil))
    AX.add_patch(Polygon([(x0, ty), (rx, ty), (rx + dx, ty + dy), (x0 + dx, ty + dy)],
                         closed=True, **sil))
    AX.add_patch(Polygon([(rx, y0), (rx + dx, y0 + dy), (rx + dx, ty + dy), (rx, ty)],
                         closed=True, **sil))
    # the cell type we follow — outlined on the front face and carried back
    # along the genome on the top face, so the slice reads as a whole sheet
    AX.add_patch(Rectangle((x0 + K * fx, y0), fx, ch, fill=False, edgecolor=INK,
                           lw=1.3, zorder=6))
    AX.add_patch(Polygon([(x0 + K * fx, ty), (x0 + (K + 1) * fx, ty),
                          (x0 + (K + 1) * fx + dx, ty + dy),
                          (x0 + K * fx + dx, ty + dy)],
                         closed=True, fill=False, edgecolor=INK, lw=1.0, zorder=6))

    # "genome" rides just outside the top-left edge, parallel to the depth axis
    ang = degrees(atan2(dy, dx))
    L = np.hypot(dx, dy)
    px, py = -dy / L, dx / L
    AX.text(x0 + .55 * dx + px * 1.0, ty + .55 * dy + py * 1.0, "genome",
            fontsize=7.4, color=MUTED, ha="center", va="center", rotation=ang,
            rotation_mode="anchor")
    if labels:
        # which assay is which row; the cell types are counted, not named
        for a, name in enumerate(EIC_ASSAYS):
            AX.text(x0 - .25, y0 + (N_TA - .5 - a) * fy, name, fontsize=3.8,
                    color=INK, ha="right", va="center")
        AX.text(x0 - 3.1, y0 + ch / 2, f"{N_TA} assays", fontsize=7.4, color=MUTED,
                rotation=90, va="center", ha="center")
    else:
        AX.text(x0 - .7, y0 + ch / 2, f"{N_TA} assays", fontsize=7.4, color=MUTED,
                rotation=90, va="center", ha="center")
    AX.text(x0 + (cw + dx) / 2, y0 - .5,
            f"{N_CT} cell types" + ("  (ENCODE Imputation Challenge)" if labels else ""),
            fontsize=7.4, color=MUTED, ha="center", va="top")


# ---------------------------------------------------------------- the window -
TOP, STEP, BH = .980, .1180, .1000


def track_panel(x0, y0, w, h, predicted, ylo=0.0):
    a = AX.inset_axes([x0, y0, w, h], transform=AX.transData)
    a.set_xlim(-.315, 1.075)
    a.set_ylim(ylo, 1)
    a.patch.set_visible(False)
    a.axis("off")
    gx = np.linspace(0, 1, NB)
    for i, name in enumerate(ASSAYS):
        y = TOP - i * STEP - BH
        obs, mu, sd = OBS_N[i], MU_N[i], SD_N[i]
        gone, held = i in MISSING, i in MASKED
        # on the output side the assay name itself says whether it was imputed,
        # which keeps the right-hand edge clear for the arrow out of the model
        tag = predicted and (gone or held)
        a.text(-.030, y + BH / 2, name, fontsize=6.8, ha="right", va="center",
               color=TEAL if tag else INK, style="italic" if tag else "normal")
        if not predicted and (gone or held):
            a.add_patch(Rectangle((0, y), 1, BH, facecolor=MASK_C if held else MISS_C,
                                  edgecolor="none"))
            a.text(.5, y + BH / 2, "masked in training" if held else "never measured",
                   fontsize=6.4, style="italic", ha="center", va="center",
                   color="#7A6A3A" if held else MUTED)
            a.plot([0, 1], [y, y], color=RULE, lw=.5)
            continue
        if predicted:
            # band first, real measurement on top of it: the interval is what
            # catches the signal where the mean does not
            a.fill_between(gx, y + np.clip(mu - sd, 0, None) * BH, y + (mu + sd) * BH,
                           color=(TEAL if not tag else TEAL_IMP), alpha=(.28 if not tag else .30), lw=0, zorder=2)
            if not gone:
                a.fill_between(gx, y, y + obs * BH, step="mid", color=OBS,
                               alpha=.30, lw=0, zorder=3)
                a.step(gx, y + obs * BH, where="mid", color="#54666F", lw=.7,
                       zorder=4)
            a.plot(gx, y + mu * BH, color=(TEAL if not tag else TEAL_IMP), alpha=(1 if not tag else 0.8), lw=1.4, zorder=5)
        else:
            a.fill_between(gx, y, y + obs * BH, step="mid", color=OBS, alpha=.38, lw=0)
            a.step(gx, y + obs * BH, where="mid", color=OBS, lw=.65)
        a.plot([0, 1], [y, y], color=RULE, lw=.5)
    return a


# ============================================================ row A : inputs ==
# The tensor: 35 x 51 cells on the front face, each one experiment.
CX0, CW, CH, DX, DY = 7.4, 16.0, 15.6, 3.5, 1.8   # short genome tail, tall assay axis
cube(CX0, 28.2, CW, CH, DX, DY, filled=False, labels=True)

# short arrows between tensor and tracks: the label sits under the arrow
AX.add_patch(FancyArrowPatch((27.7, 36.0), (33.8, 36.0), arrowstyle="-|>",
                             mutation_scale=13, lw=1.4, color=MUTED))
AX.text(30.75, 35.2, "slice one\ncell type\n(upper lobe\nof left lung)", fontsize=6.6,
        color=MUTED, ha="center", va="top", linespacing=1.35)

# The two rows are the two sides of CANDI: a tinted band behind each, with a tab
# naming it, both running into the model box on the right.
# The colour legend sits in the gap between the bands.
for _y0, _y1, _bg, _tab, _name in ((26.3, 46.0, "#F2F5F7", "#6C7A80", "INPUT"),
                                   (1.9, 24.4, "#EDF2F8", "#5B7FA6", "OUTPUT")):
    AX.add_patch(FancyBboxPatch((0.2, _y0), 61.8, _y1 - _y0,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=_bg, edgecolor="none", zorder=0))
    AX.add_patch(FancyBboxPatch((0.2, _y0), 1.6, _y1 - _y0,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=_tab, edgecolor="none", zorder=1))
    AX.text(1.0, (_y0 + _y1) / 2, _name, fontsize=8.0, fontweight="bold",
            color="white", rotation=90, ha="center", va="center", zorder=2)

WX0, WY0, WW, WH = 34.5, 27.0, 27.0, 17.2


def window_bracket(y, label):
    """The 30 kb window, marked on the tracks themselves: a bracket over their
    full width (track x 0..1), with what the tracks show."""
    x0, x1 = (WX0 + WW * (u + .315) / 1.39 for u in (0, 1))
    AX.plot([x0, x0, x1, x1], [y - .45, y, y, y - .45], color=INK, lw=.8)
    AX.text((x0 + x1) / 2, y + .25, label, fontsize=7.0, color=INK, ha="center",
            va="bottom")


track_panel(WX0, WY0, WW, WH, predicted=False)
window_bracket(WY0 + WH + .45, "one 30 kb window:  raw read counts")

# one-hot DNA and the covariate chip live under the input tracks only
_dna_ax = AX.inset_axes([WX0, WY0 + 2.9, WW, 2.2], transform=AX.transData)
_dna_ax.set_xlim(-.315, 1.075)
_dna_ax.set_ylim(0, 1)
_dna_ax.patch.set_visible(False)
_dna_ax.axis("off")
DNA_COL = {"A": "#95CBA2", "C": "#8DB8DE", "G": "#EFCB7E", "T": "#EC9C94"}
_seq = np.random.default_rng(5).choice(list("ACGT"), 72)
_w = 1 / len(_seq)
for j, b in enumerate(_seq):
    _dna_ax.add_patch(Rectangle((j * _w, .34), _w * .9, .46, facecolor=DNA_COL[b],
                                edgecolor="none"))
_dna_ax.text(-.030, .57, "DNA sequence", fontsize=6.8, color=INK, ha="right",
             va="center")
_dna_ax.text(1.012, .57, "one-hot", fontsize=6.1, color=MUTED, style="italic",
             ha="left", va="center")

_cov = AX.inset_axes([WX0, WY0 - 1.0, WW, 3.4], transform=AX.transData)
_cov.set_xlim(-.315, 1.075)
_cov.set_ylim(0, 1)
_cov.patch.set_visible(False)
_cov.axis("off")
# data coordinates, so the chip lines up with the tracks rather than with the
# axes box (which extends left under the assay names)
_cov.add_patch(FancyBboxPatch((0, .22), 1, .60,
                              boxstyle="round,pad=0.004,rounding_size=0.02",
                              facecolor=COV_C, edgecolor="none"))
_cov.text(.5, .52, "covariates\ndepth · read length · run type · platform",
          fontsize=6.3, color="#2C4A33", ha="center", va="center", linespacing=1.45)

# ============================================================= the model ======
# A legible redraw of Figure 1's architecture: same stages, same colour coding
# (conv towers salmon, transformer grey, latent purple, deconv blue), sized so
# it can be read at graphical-abstract scale.
CONV, TRANS, LAT, DECONV = "#F4D3B5", "#A9AFB2", "#C6B4E2", "#9FC8E9"
MX0, MY0, MW, MH = 66.2, 3.0, 33.4, 41.0
MS = 1.3                                # text and arrowheads, a little larger than the rest
axm = AX.inset_axes([MX0, MY0, MW, MH], transform=AX.transData)
axm.patch.set_visible(False)
axm.axis("off")
axm.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=axm.transAxes,
                             boxstyle="round,pad=0,rounding_size=0.024",
                             facecolor="#FAFCFC", edgecolor=RULE, lw=1.0, zorder=0))


def mbox(x0, x1, y0, y1, head, sub=None, fc=TRANS, fs=7.8 * MS, ec="none", tc=INK,
         sub_fs=6.3 * MS):
    axm.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, transform=axm.transAxes,
                                 boxstyle="round,pad=0.004,rounding_size=0.014",
                                 facecolor=fc, edgecolor=ec, lw=1.0, zorder=3))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub is None:
        axm.text(cx, cy, head, fontsize=fs, color=tc, ha="center", va="center",
                 transform=axm.transAxes, zorder=4)
    else:
        axm.text(cx, cy + .014, head, fontsize=fs, color=tc, ha="center",
                 va="center", transform=axm.transAxes, zorder=4)
        axm.text(cx, cy - .015, sub, fontsize=sub_fs, color=tc, alpha=.85,
                 ha="center", va="center", transform=axm.transAxes, zorder=4)


def down(x, y_from, y_to):
    axm.add_patch(FancyArrowPatch((x, y_from), (x, y_to), transform=axm.transAxes,
                                  arrowstyle="-|>", mutation_scale=9 * MS, lw=1.0,
                                  color=MUTED, zorder=3))


axm.add_patch(FancyBboxPatch((.25, .936), .50, .046, transform=axm.transAxes,
                             boxstyle="round,pad=0.008,rounding_size=0.011",
                             facecolor="#D46A5A", edgecolor="none", zorder=3))
axm.text(.5, .959, "CANDI", fontsize=13.5 * MS, fontweight="bold", color="white",
         ha="center", va="center", transform=axm.transAxes, zorder=4)

mbox(.07, .47, .832, .904, "Conv1D", "DNA", CONV)
mbox(.53, .93, .832, .904, "Conv1D", "counts", CONV)
axm.text(.5, .824, "covariates modulate every layer", fontsize=6.3 * MS, color=MUTED,
         ha="center", va="top", transform=axm.transAxes)
down(.27, .788, .760)
down(.73, .788, .760)
mbox(.07, .93, .692, .756, "Transformer encoder", fc=TRANS)
down(.5, .686, .658)
mbox(.33, .67, .592, .652, "latent  Z", fc=LAT)

# Deconvolution towers mirror the convolution towers on the way in — one per
# output head, as in Figure 1's decoder.
COLX = (.19, .50, .81)
HALF = .13
axm.plot([.5, .5], [.592, .564], color=MUTED, lw=1.0, transform=axm.transAxes,
         zorder=3)
axm.plot([COLX[0], COLX[2]], [.564, .564], color=MUTED, lw=1.0,
         transform=axm.transAxes, zorder=3)
for cx in COLX:
    down(cx, .564, .534)
    mbox(cx - HALF, cx + HALF, .462, .528, "Deconv1D", fc=DECONV, fs=6.7 * MS)
    down(cx, .456, .428)
for cx, (head, sub) in zip(COLX, (("counts", "neg. binomial"), ("signal", "Gaussian"),
                                  ("peaks", "Bernoulli"))):
    mbox(cx - HALF, cx + HALF, .348, .422, head, sub, "#FFFFFF", fs=6.9 * MS, ec=TEAL,
         tc=TEAL, sub_fs=5.4 * MS)
axm.text(.44, .336, "a distribution\nat every position", fontsize=6.3 * MS,
         color=MUTED, ha="right", va="top", transform=axm.transAxes, linespacing=1.35)
down(.5, .348, .266)
axm.text(.56, .336, "per assay", fontsize=6.3 * MS, color=MUTED, ha="left", va="top",
         transform=axm.transAxes)
mbox(.31, .97, .176, .262, "max(β₀ + β₁·CANDI", "+ β₂·average-activity, 0)", "#FFFFFF",
     fs=6.9 * MS, ec=TEAL, tc=TEAL, sub_fs=6.9 * MS)
mbox(.03, .21, .184, .254, "average-", "activity", "#FFFFFF", fs=6.3 * MS, ec="#E2B35C",
     tc="#7A5A14", sub_fs=6.3 * MS)
axm.add_patch(FancyArrowPatch((.218, .219), (.305, .219), transform=axm.transAxes,
                              arrowstyle="-|>", mutation_scale=9 * MS, lw=1.0, color=MUTED,
                              zorder=3))
axm.text(.64, .164, "final signal", fontsize=6.3 * MS, color=MUTED, ha="center", va="top",
         transform=axm.transAxes)
axm.text(.5, .112, "Self-supervised.  No cell-type embedding,\n"
                   "so it runs on cell types it has never seen.",
         fontsize=7.0 * MS, color=INK, ha="center", va="top", transform=axm.transAxes,
         linespacing=1.55)

# in near the top, out near the bottom — the model is the turn in the loop
AX.add_patch(FancyArrowPatch((62.4, 36.0), (65.9, 36.0), arrowstyle="-|>",
                             mutation_scale=13, lw=1.4, color=MUTED))
AX.add_patch(FancyArrowPatch((65.9, 14.0), (62.4, 14.0), arrowstyle="-|>",
                             mutation_scale=13, lw=1.4, color=MUTED))

# ========================================================== row B : outputs ===
OY0, OWH = 6.8, 14.4
track_panel(WX0, OY0, WW, OWH, predicted=True, ylo=.255)
window_bracket(OY0 + OWH + .45, "one 30 kb window:  predicted mean and 95% interval")
AX.text(WX0 + WW * .58, OY0 - 1.1,
        "grey = the measurement.  Red names = imputed.\n"
        "The mean misses spikes; the interval covers them.",
        fontsize=7.0, color=MUTED, ha="center", va="top", linespacing=1.5)

AX.add_patch(FancyArrowPatch((33.8, 13.4), (27.7, 13.4), arrowstyle="-|>",
                             mutation_scale=13, lw=1.4, color=MUTED))
AX.text(30.75, 12.6, "every window,\nevery cell type", fontsize=6.6, color=MUTED,
        ha="center", va="top", linespacing=1.35)

cube(CX0, 5.6, CW, CH, DX, DY, filled=True)
# colour legend between the two bands: dark red = available, salmon = imputed,
# grey = missing
for _x, _fc, _al, _lab in ((CX0 - 2.0, TEAL, 1.0, "available assays"),
                           (CX0 + 8.6, TEAL_IMP, 0.8, "imputed assays"),
                           (CX0 + 18.2, "#BDBDBD", 1.0, "missing assays")):
    AX.add_patch(Rectangle((_x, 24.85), 1.3, 1.0, facecolor=_fc, alpha=_al,
                           edgecolor="none"))
    AX.text(_x + 1.8, 25.35, _lab, fontsize=7.0, color=INK, ha="left", va="center")

# ============================================================== results =======
bot = outer[1].subgridspec(1, 3, width_ratios=[1, 1.06, 1.5], wspace=0.30)


def panel_tag(ax, letter, finding):
    ax.set_title(finding, fontsize=10, color=INK, pad=7, fontweight="bold")
    ax.grid(axis="y", linestyle="--", lw=.7, color="#C9CFD3", alpha=.8, zorder=0)
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ax.spines.values():
        s.set_color("#333333")


# --- A: calibration ---------------------------------------------------------
axA = fig.add_subplot(bot[0])
c = D["calib"]["B_DND-41"]
x, y = np.array(c["x"]), np.array(c["y"])
o = np.argsort(x); x, y = x[o], y[o]
# Where the curve crosses the dashed diagonal the colour match drops a pixel or
# two, leaving single-column notches. A width-5 rolling median removes those
# without moving the curve itself.
pad = np.pad(y, 2, mode="edge")
y = np.array([np.median(pad[i:i + 5]) for i in range(len(y))])
axA.plot([0, 1], [0, 1], ls="--", lw=1.1, color="#9AA4AA", zorder=2,
         label="perfect calibration")
axA.plot(x, y, lw=2.0, color="#D46A5A", zorder=3, label="CANDI (DNase-seq)")
# Both axes are fractions and both run the full 0-1: a calibration curve is
# only interpretable against the whole diagonal.
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
axA.xaxis.set_major_locator(MultipleLocator(.2))
axA.yaxis.set_major_locator(MultipleLocator(.2))
axA.set_xlabel("Stated confidence", fontsize=9, color=INK)
axA.set_ylabel("Observed coverage", fontsize=9, color=INK)
axA.legend(fontsize=7.6, frameon=False, loc="upper left", handlelength=1.6,
           borderpad=0.1, labelspacing=.35)
panel_tag(axA, "A", "Confidence intervals are calibrated")
axA.grid(axis="x", linestyle="--", lw=.7, color="#C9CFD3", alpha=.8, zorder=0)

# --- B: gene expression vs number of available assays -----------------------
axB = fig.add_subplot(bot[1])
for key, lab in (("Observed", "Observed"), ("Den", "Denoised"),
                 ("DenImp", "Denoised+Imputed"), ("Latent", "Latent")):
    d = D["rna"][key]
    xs, ys = np.array(d["x"]), np.array(d["y"])
    o = np.argsort(xs)
    # Observed and Denoised nearly coincide at the sparse end; dash the former
    # so it stays readable underneath.
    dashed = lab == "Observed"
    axB.plot(xs[o], ys[o], lw=2.0, ls="--" if dashed else "-", color=SRC[lab],
             label=lab, zorder=5 if dashed else 3)
axB.set_xlim(4, 13); axB.set_ylim(.55, .83)
axB.xaxis.set_major_locator(MultipleLocator(2))
axB.yaxis.set_major_locator(MultipleLocator(.1))
axB.set_xlabel("Number of available assays", fontsize=9, color=INK)
axB.set_ylabel("Expression prediction (r)", fontsize=9, color=INK)
panel_tag(axB, "B", "Latent features resist data sparsity")

# --- C: chromatin-state reproducibility -------------------------------------
axC = fig.add_subplot(bot[2])
saga = {
    "GM23248": [0.660, 0.645, 0.720, 0.755],
    "GM23338": [0.430, 0.545, 0.550, 0.745],
    "Cardiac muscle": [0.425, 0.895, 0.570, 0.490],
    "Keratinocyte": [0.440, 0.355, 0.800, 0.840],
    "Hepatocyte": [0.550, 0.585, 0.690, 0.640],
    "Neural prog.": [0.600, 0.420, 0.350, 0.870],
}
labels = list(SRC)
xb = np.arange(len(saga)); w = 0.2
for i, lab in enumerate(labels):
    axC.bar(xb + (i - 1.5) * w, [v[i] for v in saga.values()], width=w,
            color=SRC[lab], label=lab, zorder=3)
axC.set_xticks(xb)
axC.set_xticklabels(saga.keys(), fontsize=7.8, color=INK, rotation=18, ha="right")
axC.set_ylim(0, 0.95); axC.yaxis.set_major_locator(MultipleLocator(.2))
axC.set_ylabel("Reproducible fraction\nof the genome", fontsize=9, color=INK,
               linespacing=1.4)
panel_tag(axC, "C", "Chromatin state (SAGA) annotations\nbecome more reproducible")
h, l = axC.get_legend_handles_labels()
fig.legend(h, l, fontsize=9, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 0.010), handlelength=1.4, columnspacing=2.2)

