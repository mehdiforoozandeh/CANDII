"""The four downstream-application icons in the poster's introduction.

Each icon is a small schematic, not data: the tracks are drawn from a made-up
state sequence so that the picture reads at a glance. Three applications are
ones Ernst & Kellis (2015, ChromImpute) demonstrate with imputed data:
chromatin-state annotation, disease-variant (GWAS) enrichment and relation to
gene expression. The fourth is CANDI's own: with a predicted interval at every
position, uncertain predictions can be set aside and confident ones used.

Colours follow the schematic workflow (schematic.py): dark red is a measured
track, salmon an imputed one.

Writes panels/use_{states,gwas,expr,conf}.{pdf,svg} at the size they are printed
at (W x H in), so a point here is a point there.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "panels"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "path"})

W, H = 5.7, 1.75                      # inches; poster_landscape.tex \UseW is the same width
HD = 2.3                              # the drawing's own height units, squeezed into H
INK, MUTED, GRID = "#1B2A32", "#5E6E78", "#B9C2C7"
SIG, IMP = "#A3302A", "#F4A08F"       # measured, imputed (schematic.py)
KEEP, DROP = "#7FA87A", "#BDBDBD"     # sage: confident, kept; grey: uncertain, set aside
STATE = {"promoter": "#5B7FA6", "enhancer": "#E2B35C", "transcribed": "#7FA87A",
         "repressed": "#9A8FBF", "quiescent": "#E3E3E3"}

RNG = np.random.default_rng(7)
X = np.linspace(0, 1, 400)


def canvas():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, HD)
    ax.axis("off")
    return fig, ax


def smooth(v, k=9):
    return np.convolve(v, np.ones(k) / k, mode="same")


def track(ax, x0, x1, y0, h, y, color, alpha=1.0, base=True):
    """A filled signal track: y in [0, 1] scaled into the band y0 .. y0 + h."""
    xs = x0 + X * (x1 - x0)
    ax.fill_between(xs, y0, y0 + h * np.clip(y, 0, 1), color=color, alpha=alpha, lw=0)
    if base:
        ax.plot([x0, x1], [y0, y0], color=GRID, lw=1.2)


def peaks(centres, widths, heights, noise=0.03):
    y = sum(hh * np.exp(-0.5 * ((X - c) / w) ** 2) for c, w, hh in zip(centres, widths, heights))
    return np.clip(y + smooth(RNG.normal(0, noise, X.size)), 0, None)


def arrow(ax, a, b, lw=3, scale=26, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=scale, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0))


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", transparent=True)
    plt.close(fig)


# ---- 1 chromatin states: three tracks, segmented into states underneath ------
def states():
    fig, ax = canvas()
    x0, x1 = 0.1, W - 0.1
    segs = [(0.00, 0.10, "quiescent"), (0.10, 0.20, "promoter"), (0.20, 0.52, "transcribed"),
            (0.52, 0.62, "quiescent"), (0.62, 0.72, "enhancer"), (0.72, 0.92, "repressed"),
            (0.92, 1.00, "quiescent")]
    level = {  # state -> level on the three marks (promoter, enhancer, gene-body-like)
        "promoter": (.95, .8, .1), "enhancer": (.1, .85, .05), "transcribed": (.05, .1, .7),
        "repressed": (.02, .02, .05), "quiescent": (.02, .02, .03)}
    for i, (colour, alpha) in enumerate(((SIG, 1), (IMP, .8), (IMP, .8))):
        y = np.zeros_like(X)
        for a, b, s in segs:
            y[(X >= a) & (X < b)] = level[s][i]
        y = np.clip(smooth(y, 15) + smooth(RNG.normal(0, .05, X.size), 5), 0, 1)
        track(ax, x0, x1, 1.72 - 0.5 * i, 0.46, y, colour, alpha)
    for a, b, s in segs:
        ax.add_patch(Rectangle((x0 + a * (x1 - x0), 0.1), (b - a) * (x1 - x0), 0.42,
                               color=STATE[s], lw=0))
    save(fig, "use_states")


# ---- 2 disease variants: a GWAS hit falls in one cell type's peak -------------
def gwas():
    fig, ax = canvas()
    x0, x1 = 0.1, W - 0.1
    xs = np.sort(RNG.uniform(0, 1, 70))
    ys = RNG.gamma(1.4, 0.09, xs.size).clip(0, .4)
    lead = 0.63
    ys[np.abs(xs - lead) < .06] *= 1.6
    ax.scatter(x0 + xs * (x1 - x0), 1.35 + ys, s=26, color=GRID, lw=0)
    ax.scatter([x0 + lead * (x1 - x0)], [1.35 + .86], s=120, color=INK, lw=0, zorder=3)
    ax.plot([x0, x1], [1.35 + .62] * 2, color=MUTED, lw=1.6, ls=(0, (4, 3)))
    rows = ((0.82, [.2, .85], [.03, .03], [.5, .35], IMP, .8),
            (0.44, [.3, lead], [.03, .025], [.3, .95], SIG, 1),
            (0.06, [.12, .45], [.03, .03], [.45, .5], IMP, .8))
    for y0, c, w, hh, colour, alpha in rows:
        track(ax, x0, x1, y0, 0.34, peaks(c, w, hh), colour, alpha)
    xl = x0 + lead * (x1 - x0)
    ax.plot([xl, xl], [0.44, 2.18], color=INK, lw=1.8, ls=(0, (3, 3)), zorder=2)
    save(fig, "use_gwas")


# ---- 3 gene expression: marks over a gene -> predicted vs measured expression --
def expr():
    fig, ax = canvas()
    x0, x1 = 0.1, 3.3
    track(ax, x0, x1, 1.55, 0.6, peaks([.12, .5], [.03, .25], [.95, .35]), SIG)
    track(ax, x0, x1, 0.95, 0.5, peaks([.12, .16], [.025, .04], [.8, .5]), IMP, .8)
    g = 0.42                                                  # the gene model
    ax.plot([x0 + .12 * (x1 - x0), x1 - .15], [g, g], color=INK, lw=2.4)
    for a, b in ((.12, .2), (.38, .45), (.6, .66), (.82, .92)):
        ax.add_patch(Rectangle((x0 + a * (x1 - x0), g - .13), (b - a) * (x1 - x0), .26,
                               color=INK, lw=0))
    xt = x0 + .12 * (x1 - x0)                                 # the TSS's bent arrow
    ax.plot([xt, xt, xt + .32], [g + .13, g + .36, g + .36], color=INK, lw=2.4)
    ax.add_patch(Polygon([(xt + .32, g + .27), (xt + .32, g + .45), (xt + .47, g + .36)],
                         color=INK, lw=0))
    arrow(ax, (3.45, 1.2), (3.95, 1.2))
    sx0, sy0, s = 4.15, 0.25, 1.85                            # scatter: predicted vs measured
    ax.plot([sx0, sx0, sx0 + s], [sy0 + s, sy0, sy0], color=INK, lw=2)
    t = RNG.uniform(.08, .92, 22)
    ax.scatter(sx0 + s * t, sy0 + s * (t + RNG.normal(0, .07, t.size)).clip(.04, .96),
               s=30, color=SIG, lw=0)
    ax.plot([sx0 + .1, sx0 + s - .1], [sy0 + .1, sy0 + s - .1], color=MUTED, lw=1.6,
            ls=(0, (4, 3)))
    save(fig, "use_expr")


# ---- 4 confidence: a mean and its interval; wide intervals are set aside ------
def conf():
    fig, ax = canvas()
    x0, x1 = 0.1, W - 0.1
    xs = x0 + X * (x1 - x0)
    mu = 0.95 + 0.85 * peaks([.12, .38, .62, .88], [.03, .04, .035, .03],
                             [.9, .6, .8, .7], noise=0)
    bumps = [(.30, .47), (.70, .80)]                          # the uncertain stretches
    sd = 0.05 + sum(0.32 * np.exp(-0.5 * ((X - .5 * (a + b)) / (.28 * (b - a))) ** 2)
                    for a, b in bumps)
    for a, b in bumps:                                         # grey out, behind the band
        ax.add_patch(Rectangle((x0 + a * (x1 - x0), 0.55), (b - a) * (x1 - x0), 1.7,
                               color=DROP, alpha=.35, lw=0))
    ax.fill_between(xs, np.maximum(mu - 1.96 * sd, 0.6), mu + 1.96 * sd, color=IMP,
                    alpha=.55, lw=0)
    ax.plot(xs, mu, color=SIG, lw=2.4)
    edges = sorted({0.0, 1.0, *[e for ab in bumps for e in ab]})
    for a, b in zip(edges[:-1], edges[1:]):                    # the keep / set-aside bar
        bad = any(abs(a - p) < 1e-9 for p, _ in bumps)
        ax.add_patch(Rectangle((x0 + a * (x1 - x0), 0.08), (b - a) * (x1 - x0), 0.32,
                               color=DROP if bad else KEEP, lw=0))
        ax.text(x0 + .5 * (a + b) * (x1 - x0), 0.24, "\u2715" if bad else "\u2713",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="#5E6E78" if bad else "white")
    save(fig, "use_conf")


for draw in (states, gwas, expr, conf):
    draw()
print("wrote panels/use_{states,gwas,expr,conf}.{pdf,svg}")
