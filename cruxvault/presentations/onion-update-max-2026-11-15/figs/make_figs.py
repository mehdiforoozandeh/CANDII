"""Figures for the progress report "Peeling the onion: progress to 2026-11-15".

Run: /Users/mforooz/miniforge3/bin/python figs/make_figs.py   (from the report folder)
Every data figure reads its numbers from a vault result file (paths below) or from
figs/data/rung4_table_nibi_2026-11-15.tsv (select_rule.py table output, Nibi, 2026-11-15).
"""
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
R = os.path.join(VAULT, "results")

# Okabe-Ito palette (colorblind-safe)
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", red="#D55E00", purple="#CC79A7", grey="#999999", black="#000000")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def out(name):
    return os.path.join(HERE, name)


def read_tsv(path):
    with open(path) as fh:
        return list(csv.reader(fh, delimiter="\t"))


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=7.5, bold=False, tc="black", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#333333"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, shrinkA=0, shrinkB=0))



def canvas(w, h):
    """Figure whose single axes fills the page; 1 data unit = 1 inch."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

# ---------------------------------------------------------------- Fig 1: the onion
def fig_onion():
    fig, ax = canvas(6.6, 2.9)
    cx, cy = 1.45, 1.45
    rings = [(1.40, "population", OI["grey"]), (1.14, "cell type", OI["green"]),
             (0.88, "replicates", OI["yellow"]), (0.62, "views", OI["green"]),
             (0.36, "", OI["green"])]
    for r, lab, c in rings:
        ax.add_patch(Circle((cx, cy), r, fc=c, ec="white", lw=1.5, alpha=0.85))
    for (r, lab, c), (r2, _, _) in zip(rings[:-1], rings[1:]):
        ax.text(cx, cy + (r + r2) / 2, lab, ha="center", va="center", fontsize=6.3)
    ax.text(cx, cy, "read\nsampling", ha="center", va="center", fontsize=6.3)
    rows = [
        ("read sampling (Poisson)", "measured", "pseudoreplicate halves pr1 and pr2\nof every view", OI["green"]),
        ("data processing", "measured", "8 designed views per experiment:\nsingle-end, read crops, depth thinning,\nno duplicate removal, no MAPQ filter", OI["green"]),
        ("data generation", "partial", "isogenic replicates only; lab,\nplatform and antigen not separated", OI["yellow"]),
        ("biological: cell type,\nposition, assay", "measured", "pairs of different cell types;\ncell type is mixed with donor, lab", OI["green"]),
        ("biological: population", "not addressed", "no data view", OI["grey"]),
    ]
    x0, x1, x2 = 3.0, 3.22, 4.55
    ax.text(x1, 2.75, "source of variation", fontsize=7.5, fontweight="bold", va="center")
    ax.text(x2, 2.75, "data view that exposes it", fontsize=7.5, fontweight="bold", va="center")
    y = 2.38
    for name, status, view, c in rows:
        ax.add_patch(plt.Rectangle((x0, y - 0.11), 0.16, 0.22, fc=c, ec="none", alpha=0.85))
        ax.text(x1, y, f"{name}\n[{status}]", fontsize=6.6, va="center")
        ax.text(x2, y, view, fontsize=6.6, va="center")
        y -= 0.53
    fig.savefig(out("fig1_onion.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: measurement dimension
def fig_measurement():
    fig, ax = canvas(6.6, 2.75)
    box(ax, 0.05, 1.1, 0.95, 0.65, "one EIC\nexperiment\n(cell, assay)", "#eeeeee", fs=6.8, bold=True)
    box(ax, 1.25, 1.1, 0.95, 0.65, "its reads\n(FASTQ or\nredacted BAM)", "#eeeeee", fs=6.8)
    arrow(ax, 1.0, 1.425, 1.25, 1.425)
    ax.text(2.6, 2.62, "base view: all reads, standard ENCODE pipeline; 8 designed views:", fontsize=6.8, fontweight="bold", va="center")
    ax.text(2.6, 2.38, "FASTQ-level", fontsize=6.8, color=OI["blue"], fontweight="bold", va="center")
    ax.text(2.6, 1.36, "alignment-level", fontsize=6.8, color=OI["red"], fontweight="bold", va="center")
    fq = ["se: mate 1 only", "crop36: reads cut to 36 bp", "crop50: reads cut to 50 bp"]
    al = ["depth0.50, depth0.25, depth0.125", "nodedup: keep duplicates", "mapq0: no mapping-quality filter"]
    for i, t in enumerate(fq):
        y = 2.14 - i * 0.25
        box(ax, 2.6, y - 0.1, 1.9, 0.2, t, "#d9ecf7", fs=6.4)
        arrow(ax, 2.2, 1.425, 2.6, y)
    for i, t in enumerate(al):
        y = 1.1 - i * 0.25
        box(ax, 2.6, y - 0.1, 1.9, 0.2, t, "#f7e0d4", fs=6.4)
        arrow(ax, 2.2, 1.425, 2.6, y)
    box(ax, 4.85, 1.65, 1.7, 0.75, "every view, base included,\nis split at random into two\nread halves pr1 and pr2\n(pseudoreplicates)", "#dff2ea", fs=6.4)
    box(ax, 4.85, 0.35, 1.7, 0.95, "one measurement = one view\nof one experiment in 25 bp\nbins: counts.h5, pr1.h5,\npr2.h5; 2,740 in the\nfrozen dataset", "#eeeeee", fs=6.4)
    arrow(ax, 4.5, 1.85, 4.85, 2.0)
    arrow(ax, 4.5, 0.85, 4.85, 0.85)
    fig.savefig(out("fig2_measurement.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: objectives and metrics
def fig_objectives():
    fig, ax = canvas(6.6, 2.1)
    rows = [
        ("1. biological variation\nimpute across Cell, Assay, Position",
         "specific variance recovered above noise\n1 - (E_sp - N) / (V_sp - N)", "x 0.50", OI["green"]),
        ("2. artifact variation\ncovariate set A to covariate set B",
         "gap closed on never-trained view pairs\n0 = depth-only scaling, 1 = oracle floor", "x 0.25", OI["blue"]),
        ("3. uncertainty\ncalibrated predicted NB",
         "uncertainty term: 1 - PIT deviation of the\nmodel / PIT deviation of the marginal NB", "x 0.25", OI["orange"]),
    ]
    y = 1.75
    for i, (goal, metric, wt, c) in enumerate(rows):
        box(ax, 0.05, y - 0.25, 2.15, 0.5, goal, c + "33", ec=c, fs=6.6)
        box(ax, 2.45, y - 0.25, 2.45, 0.5, metric, "#f4f4f4", fs=6.6)
        arrow(ax, 2.2, y, 2.45, y)
        arrow(ax, 4.9, y, 5.55, 1.05)
        ax.text(5.0, [y + 0.0, y + 0.04, y - 0.02][i], wt, fontsize=6.6, va=["bottom", "bottom", "top"][i])
        y -= 0.7
    box(ax, 5.55, 0.7, 1.0, 0.7, "composite\n(each term\nclipped to\n[0, 1])", "#e8e8e8", fs=6.6)
    fig.savefig(out("fig3_objectives.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: variance layers
ASSAYS = ["DNase-seq", "H3K27ac", "H3K4me3", "H3K4me2", "H3K9ac", "H3K4me1", "H3K79me2",
          "H3K27me3", "H3K36me3", "H3K9me3", "chipseq-control"]


def fig_layers():
    rows = read_tsv(os.path.join(R, "t305", "layers_by_assay.tsv"))
    h = rows[0]
    ix = {k: h.index(k) for k in ["assay_or_class", "pooled_by_class", "rung", "share_of_layers", "pair_value"]}
    share = defaultdict(dict)
    for r in rows[1:]:
        if len(r) <= ix["share_of_layers"] or r[ix["pooled_by_class"]] != "0" or r[ix["pair_value"]] == "" or r[ix["share_of_layers"]] == "":
            continue
        share[r[ix["assay_or_class"]]][r[ix["rung"]]] = float(r[ix["share_of_layers"]])
    eta = read_tsv(os.path.join(R, "eta_space", "eta_layers_by_assay.tsv"))
    eh = eta[0]
    # columns: replicate_share_of_V at 21 (0-based) lo 22 hi 23; celltype_share_of_V 24,25,26
    i_rep = eh.index("replicate_share_of_V"); i_ct = eh.index("celltype_share_of_V")
    etad = {r[0]: r for r in eta[1:]}
    order = sorted(ASSAYS, key=lambda a: share[a]["other_cell_type"])
    labels = [a.replace("chipseq-control", "ChIP control") for a in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True,
                                 gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.08))
    y = range(len(order))
    s = [share[a]["pseudoreplicate"] for a in order]
    rp = [max(share[a]["isogenic"], 0) for a in order]
    ct = [share[a]["other_cell_type"] for a in order]
    a1.barh(y, s, color=OI["grey"], label="sampling")
    a1.barh(y, rp, left=s, color=OI["orange"], label="replicate")
    a1.barh(y, [max(c, 0) for c in ct], left=[a + b for a, b in zip(s, rp)], color=OI["green"], label="cell type")
    a1.set_yticks(list(y)); a1.set_yticklabels(labels)
    a1.set_xlim(0, 1); a1.set_xlabel("share of the three layers")
    a1.set_title("(a) offset off: log1p counts, half depth")
    a1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=3, frameon=False, handlelength=1)
    for k, (col, c, dy) in enumerate([(i_rep, OI["orange"], 0.17), (i_ct, OI["green"], -0.17)]):
        m = [float(etad[a][col]) for a in order]
        lo = [float(etad[a][col + 1]) for a in order]
        hi = [float(etad[a][col + 2]) for a in order]
        yy = [v + dy for v in y]
        a2.errorbar(m, yy, xerr=[[mm - l for mm, l in zip(m, lo)], [hh - mm for mm, hh in zip(m, hi)]],
                    fmt="o", ms=3.5, color=c, elinewidth=0.9, capsize=1.5,
                    label=["replicate", "cell type"][k])
    a2.axvline(0, color="#777777", lw=0.6)
    a2.set_xlim(-0.6, 1.25)
    a2.set_xlabel("share of the sampling-free variance (95% CI)")
    a2.set_title("(b) offset on: rate space, sampling removed")
    a2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2, frameon=False, handlelength=1)
    fig.savefig(out("fig4_layers.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: baselines and AlphaGenome 2
def fig_baselines():
    # (a) fair set, recalibrated r_spec, 11 tracks in 9 cells: results/t308/AG2_VS_BASELINES.md lines 41-47
    meth = ["AlphaGenome 2", "position average", "marginal NB", "closest-five oracle", "training-track mean"]
    val = [0.438, 0.222, 0.121, 0.066, 0.0]
    floor, flo, fhi = 0.333, 0.276, 0.382
    # (b) gap closed (oracle floor), full set, 102 pairs: results/oracle_floor/ORACLE_FLOOR.md subset (a)
    bm = ["position average", "half-against-half floor", "depth-only scaling", "marginal NB"]
    bv = [0.258, 0.167, 0.0, -0.286]
    blo = [0.192, 0.094, 0.0, -0.986]
    bhi = [0.373, 0.219, 0.0, -0.123]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw=dict(wspace=0.85))
    yy = list(range(len(meth)))[::-1]
    a1.axvspan(flo, fhi, color=OI["grey"], alpha=0.25, lw=0)
    a1.axvline(floor, color=OI["grey"], lw=1)
    a1.text(floor + 0.005, 4.55, "noise floor\n0.333", fontsize=6.3, color="#444444", va="center")
    cols = [OI["blue"]] + [OI["black"]] * 4
    for y, v, c in zip(yy, val, cols):
        a1.plot([0, v], [y, y], color=c, lw=0.8); a1.plot(v, y, "o", color=c, ms=4)
    a1.set_yticks(yy); a1.set_yticklabels(meth)
    a1.set_xlim(-0.02, 0.5); a1.set_ylim(-0.6, 5.0)
    a1.set_xlabel("recalibrated r_spec (signal space)")
    a1.set_title("(a) imputation, 11 tracks unseen by all")
    yb = list(range(len(bm)))[::-1]
    for y, v, l, h in zip(yb, bv, blo, bhi):
        a2.errorbar(v, y, xerr=[[v - l], [h - v]], fmt="o", color=OI["black"], ms=4, capsize=1.5, elinewidth=0.9)
    a2.axvline(0, color="#777777", lw=0.6); a2.axvline(1, color="#777777", lw=0.6, ls=":")
    a2.text(0.98, 3.3, "oracle\nfloor = 1", fontsize=6.3, ha="right", color="#444444")
    a2.set_yticks(yb); a2.set_yticklabels(bm)
    a2.set_xlim(-1.05, 1.05); a2.set_ylim(-0.6, 3.8)
    a2.set_xlabel("gap closed, oracle floor (95% CI)")
    a2.set_title("(b) correction, 102 never-trained pairs")
    fig.savefig(out("fig5_baselines.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6: miniCANDI schematic
def fig_minicandi():
    fig, ax = canvas(6.6, 2.5)
    ins = [("X: counts of all assays of one cell\n(W/25 bins x assays; some masked)", 2.1, "#d9ecf7"),
           ("S: DNA sequence (one-hot, W bp)", 1.5, "#d9ecf7"),
           ("C: covariates of the input views\n(log2 depth, run type, read length, ...)", 0.9, "#fbe7c6"),
           ("C': covariates of the wanted output", 0.3, "#fbe7c6")]
    for t, y, c in ins:
        box(ax, 0.05, y - 0.22, 2.2, 0.44, t, c, fs=6.3)
    box(ax, 2.6, 0.65, 1.55, 1.7, "trunk: one of six\narchitectures, about\n1.0 M parameters\n\ncovariates enter by\nFiLM on the input side\nand before the head", "#eeeeee", fs=6.3)
    for t, y, c in ins[:3]:
        arrow(ax, 2.25, y, 2.6, y)
    arrow(ax, 2.25, 0.3, 4.5, 0.3)
    box(ax, 4.5, 1.05, 2.05, 1.35, "NB head per assay and 25 bp bin\nmean = depth' x 2^eta (offset on)\nsize n: a per-bin output\n\nloss: NB NLL on masked assays\n(imputation) and on visible\nassays under C' (correction)", "#dff2ea", fs=6.3)
    box(ax, 4.5, 0.08, 2.05, 0.52, "depth' (from C') enters the\nmean as a fixed offset", "#fbe7c6", fs=6.3)
    arrow(ax, 4.15, 1.7, 4.5, 1.7)
    arrow(ax, 5.52, 0.6, 5.52, 1.05)
    fig.savefig(out("fig6_minicandi.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7: architecture ladder
NAMES = {"rung1_base": "pointwise MLP", "rung2_base": "conv autoencoder", "rung3_base": "conv plus attention",
         "rung4_base": "U-Net", "rung5_base": "axial transformer", "rung6_base": "Perceiver IO"}


def ladder_seeds():
    txt = open(os.path.join(R, "metrics_screening", "scores_ladder.md")).read()
    sec = txt.split("### Per seed")[1].split("## Adjacent")[0]
    d = defaultdict(lambda: defaultdict(list))
    for line in sec.splitlines():
        m = re.match(r"\| (rung\d_base) \| (\d) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            k = m.group(1)
            d[k]["spec"].append(float(m.group(3)))
            d[k]["gap"].append(float(m.group(5)))
            d[k]["comp"].append(float(m.group(8)))
    return d


def fig_ladder():
    d = ladder_seeds()
    order = ["rung2_base", "rung3_base", "rung5_base", "rung4_base", "rung1_base", "rung6_base"]
    tiers = {"rung2_base": 0, "rung3_base": 0, "rung5_base": 1, "rung4_base": 1, "rung1_base": 1, "rung6_base": 2}
    tcol = [OI["green"], OI["sky"], OI["grey"]]
    ref = {"comp": 0.1601, "spec": 0.1190, "gap": 0.2349}  # position average, scores_ladder.md
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    titles = {"comp": "composite", "spec": "specific variance recovered", "gap": "gap closed, oracle floor"}
    for ax, key in zip(axs, ["comp", "spec", "gap"]):
        for i, k in enumerate(order):
            y = len(order) - 1 - i
            v = d[k][key]
            mu = sum(v) / len(v)
            ax.plot([min(v), max(v)], [y, y], color=tcol[tiers[k]], lw=2.5, alpha=0.6, solid_capstyle="butt")
            ax.plot(v, [y] * len(v), "|", color="#333333", ms=5)
            ax.plot(mu, y, "o", color=tcol[tiers[k]], mec="#333333", mew=0.5, ms=5)
        ax.axvline(ref[key], color=OI["black"], ls=":", lw=0.8)
        ax.set_title(titles[key])
        ax.set_xlim(0.0, 0.65)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([NAMES[k] for k in order][::-1])
    axs[1].set_xlabel("seed mean (dot), seeds (ticks), seed range (bar); 3 seeds each")
    from matplotlib.patches import Patch
    axs[2].legend(handles=[Patch(color=tcol[0], alpha=0.6, label="tier 1"), Patch(color=tcol[1], alpha=0.6, label="tier 2"),
                           Patch(color=tcol[2], alpha=0.6, label="tier 3"),
                           plt.Line2D([], [], color="black", ls=":", lw=0.8, label="position average")], loc="upper center", bbox_to_anchor=(-0.65, -0.24), ncol=4, frameon=False, fontsize=6.8)
    fig.savefig(out("fig7_ladder.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 8: covariate and depth arms
def fig_arms():
    rows = read_tsv(os.path.join(HERE, "data", "rung4_table_nibi_2026-11-15.tsv"))
    h = rows[0]
    keys = ["composite", "imputation_raw", "correction_raw", "correction_gap_closed_oracle_no_nodedup_side", "uncertainty_raw"]
    d = defaultdict(lambda: defaultdict(list))
    for r in rows[1:]:
        if len(r) < len(h) or r[2] != "done" or r[3] == "":
            continue
        for k in keys:
            d[r[0]][k].append(float(r[h.index(k)]))
    arms = [("rung4_base", "U-Net base"), ("rung4_scrambled_covariates", "covariates scrambled"),
            ("rung4_hide_depth_input", "depth hidden, input side"), ("rung4_hide_depth_output", "depth hidden, output side*"),
            ("rung4_hard_offset", "depth hidden, both sides")]
    cols = [OI["black"], OI["purple"], OI["blue"], OI["sky"], OI["red"]]
    titles = ["composite", "specific var.\nrecovered", "gap closed,\nall 102 pairs", "gap closed,\n18 clean pairs", "uncertainty\nterm"]
    fig, axs = plt.subplots(1, 5, figsize=(6.9, 2.3), sharey=True, gridspec_kw=dict(wspace=0.15))
    for ax, k, t in zip(axs, keys, titles):
        base = sum(d["rung4_base"][k]) / 3
        ax.axvline(base, color="#bbbbbb", lw=0.8)
        for i, ((a, lab), c) in enumerate(zip(arms, cols)):
            v = d[a][k]; y = len(arms) - 1 - i
            mu = sum(v) / len(v)
            if len(v) > 1:
                ax.plot([min(v), max(v)], [y, y], color=c, lw=2.2, alpha=0.5, solid_capstyle="butt")
            ax.plot(mu, y, "o", color=c, ms=4.5, mfc=c if len(v) > 1 else "white")
        ax.set_title(t, fontsize=7.5)
        ax.tick_params(axis="x", labelsize=6.5)
    axs[4].set_xlim(-0.25, 0.8)
    axs[0].set_yticks(range(len(arms))); axs[0].set_yticklabels([l for _, l in arms][::-1])
    axs[2].set_xlabel("seed mean (dot) and seed range (bar), 3 seeds; * 1 seed (open dot)")
    fig.savefig(out("fig8_arms.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 9: coverage
COVER = [  # (group, short label, status) from results/ladder_audit/COVERAGE.md rows 1-39 without 12
    ("Architecture", "local context (MLP to conv AE)", "F"), ("Architecture", "global attention", "F"),
    ("Architecture", "skips (U-Net)", "F"), ("Architecture", "attention across tracks", "F"),
    ("Architecture", "latent bottleneck (Perceiver)", "F"), ("Architecture", "U-Net skip source", "U"),
    ("Architecture", "axial axis arms", "U"), ("Architecture", "Perceiver latent count", "U"),
    ("Architecture", "matched compute", "P"),
    ("Every architecture", "DNA off", "P"), ("Every architecture", "scrambled covariates", "P"),
    ("Every architecture", "covariates by concatenation", "P"), ("Every architecture", "covariates as tokens", "P"),
    ("Every architecture", "FiLM as an arm", "U"), ("Every architecture", "mask token or drop", "P"),
    ("Every architecture", "copy-leak guardrail", "F"),
    ("Inputs, outputs, loss", "ChIP control as input", "U"), ("Inputs, outputs, loss", "predict ChIP control", "U"),
    ("Inputs, outputs, loss", "average baseline as input", "U"), ("Inputs, outputs, loss", "two-step CAC, native depth", "U"),
    ("Inputs, outputs, loss", "two-step CAC, fixed depth", "U"), ("Inputs, outputs, loss", "QC on input", "U"),
    ("Inputs, outputs, loss", "DNA embedding source", "U"), ("Inputs, outputs, loss", "covariate embedding variants", "U"),
    ("Inputs, outputs, loss", "depth-controlled input", "U"), ("Inputs, outputs, loss", "depth hidden, both sides", "F"),
    ("Inputs, outputs, loss", "depth hidden, input side", "F"), ("Inputs, outputs, loss", "depth hidden, output side", "P"),
    ("Inputs, outputs, loss", "count transforms", "U"), ("Inputs, outputs, loss", "loss and head variants", "U"),
    ("Inputs, outputs, loss", "RNA or cell embedding input", "U"), ("Inputs, outputs, loss", "resolution", "U"),
    ("Inputs, outputs, loss", "scaling (deferred)", "U"),
    ("Training regions", "region budget curve", "U"), ("Training regions", "region selection rule", "F"),
    ("Training regions", "annotation strata (deferred)", "U"), ("Training regions", "train windows in eval regions", "F"),
    ("Training regions", "eval and pair region count", "U"),
]


def fig_coverage():
    assert len(COVER) == 38
    n = {s: sum(1 for c in COVER if c[2] == s) for s in "FPU"}
    assert (n["F"], n["P"], n["U"]) == (10, 7, 21), n
    col = {"F": OI["green"], "P": OI["yellow"], "U": "#dddddd"}
    lines, last = [], None
    for g, lab, s in COVER:
        if g != last:
            lines.append(("H", g, None)); last = g
        lines.append(("I", lab, s))
    per = 14
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(-2.0, per)
    for i, (kind, lab, s) in enumerate(lines):
        c, r = divmod(i, per)
        y = per - 1 - r
        if kind == "H":
            ax.text(c + 0.02, y, lab, fontsize=7.5, fontweight="bold", va="center")
        else:
            ax.add_patch(plt.Rectangle((c + 0.02, y - 0.36), 0.09, 0.72, fc=col[s], ec="#888888", lw=0.4))
            ax.text(c + 0.15, y, lab, fontsize=6.9, va="center")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=col["F"], ec="#888888", label=f"fully tested, 3 seeds ({n['F']})"),
                       Patch(fc=col["P"], ec="#888888", label=f"partial ({n['P']})"),
                       Patch(fc=col["U"], ec="#888888", label=f"untested ({n['U']})")],
              loc="lower left", bbox_to_anchor=(0.0, -0.04), ncol=3, frameon=False, fontsize=7)
    fig.savefig(out("fig9_coverage.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 10: counterfactual views
def fig_views():
    # (a) headroom per knob, all 1,480 correction pairs: results/t307/correction_headroom.tsv
    rows = read_tsv(os.path.join(R, "t307", "correction_headroom.tsv"))
    h = rows[0]
    ix = {k: h.index(k) for k in ["pair_set", "knob", "n_pairs", "n_cells", "gap_depth_only_minus_floor",
                                   "gap_depth_only_minus_floor_lo", "gap_depth_only_minus_floor_hi"]}
    keep = []
    for r in rows[1:]:
        ps, kn = r[ix["pair_set"]], r[ix["knob"]]
        if int(r[ix["n_cells"]]) < 3 or kn == "__all_subset__":
            continue
        if not (ps == "all correction pairs" or (ps == "by depth direction" and kn.endswith(" up"))):
            continue
        kn = {"depth": "depth only", "__all_non_subset__": "all non-subset pairs"}.get(kn, kn)
        r = list(r); r[ix["knob"]] = kn
        keep.append((r[ix["knob"]], int(r[ix["n_pairs"]]), float(r[ix["gap_depth_only_minus_floor"]]),
                     float(r[ix["gap_depth_only_minus_floor_lo"]]), float(r[ix["gap_depth_only_minus_floor_hi"]])))
    keep.sort(key=lambda k: k[2])
    # (b) gap closed (oracle) per headline knob, screening set: results/oracle_floor/per_knob_p16.tsv
    pk = read_tsv(os.path.join(R, "oracle_floor", "per_knob_p16.tsv"))
    ph = pk[0]
    knobs = ["crop", "crop+se", "mapq+nodedup"]
    npairs = {"crop": 14, "crop+se": 4, "mapq+nodedup": 84}  # ORACLE_FLOOR.md, subsets (a) and (b)
    meth = [("rung1_base", "pointwise MLP (3 seeds)", OI["blue"]), ("position_average", "position average", OI["black"]),
            ("pseudoreplicate_floor", "half-against-half floor", OI["grey"]), ("marginal_nb", "marginal NB", OI["orange"])]
    val = {}
    for r in pk[1:]:
        if len(r) < 9 or r[2] != "never_trained_non_subset_oracle" or r[3] not in knobs:
            continue
        val[(r[0], r[3])] = (float(r[6]), float(r[7]) if r[7] else 0.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.1), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.6))
    for i, (k, n, g, lo, hi) in enumerate(keep):
        a1.errorbar(g, i, xerr=[[g - lo], [hi - g]], fmt="o", color=OI["black"], ms=3.5, capsize=1.5, elinewidth=0.9)
    a1.set_yticks(range(len(keep)))
    a1.set_yticklabels([f"{kk} ({nn})" for kk, nn, _, _, _ in keep])
    for i, (kk, _, _, _, _) in enumerate(keep):
        if kk.endswith(" up"):
            a1.get_yticklabels()[i].set_color(OI["red"])
    a1.axvline(0, color="#777777", lw=0.6)
    a1.set_xlabel("CRPS gap, depth-only minus half floor (95% CI)")
    a1.set_title("(a) room left by depth-only scaling\n(all 1,480 pairs; red: target deeper)")
    for j, (m, lab, c) in enumerate(meth):
        for i, k in enumerate(knobs):
            v, sp = val[(m, k)]
            y = len(knobs) - 1 - i + (1.5 - j) * 0.17
            a2.plot(v, y, "o", color=c, ms=4, label=lab if i == 0 else None)
            if sp > 0:
                a2.plot([v - sp / 2, v + sp / 2], [y, y], color=c, lw=2, alpha=0.5)
    a2.axvline(0, color="#777777", lw=0.6)
    a2.set_yticks(range(len(knobs)))
    a2.set_yticklabels([f"{k} ({npairs[k]})" for k in knobs][::-1])
    a2.set_xlabel("gap closed, oracle floor")
    a2.set_title("(b) gap closed on the 102 headline\npairs, per view change (screening)")
    a2.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.22), ncol=2, frameon=False, fontsize=6.6)
    fig.savefig(out("fig10_views.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig: onion plus measurement views (merged)
def fig_onion_views():
    fig, ax = canvas(6.8, 2.45)
    cx, cy = 1.2, 1.22
    rings = [(1.18, OI["grey"]), (0.95, OI["green"]), (0.72, OI["yellow"]), (0.49, OI["green"]), (0.26, OI["green"])]
    for r, c in rings:
        ax.add_patch(Circle((cx, cy), r, fc=c, ec="white", lw=1.2, alpha=0.85))
    labs = [
        ("population: not addressed", 1.07),
        ("cell type (+ donor, lab): pairs of different cell types", 0.84),
        ("data generation: isogenic replicates only [partial]", 0.61),
        ("data processing: 8 designed views", 0.38),
        ("read sampling: pseudoreplicate halves pr1, pr2", 0.0),
    ]
    import math
    ys = [2.25, 1.85, 1.45, 1.05, 0.65]
    for (t, rad), y in zip(labs, ys):
        ang = math.radians(35 - (2.25 - y) * 30)
        px, py = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
        if rad == 0.0:
            px, py = cx + 0.1, cy
        ax.plot([px, 2.45], [py, y], color="#555555", lw=0.5)
        ax.text(2.5, y, t, fontsize=6.4, va="center")
    # views tree
    x0 = 4.85
    box(ax, x0, 1.95, 1.9, 0.38, "one EIC experiment (cell, assay)\nbase view: all reads", "#eeeeee", fs=6.2)
    box(ax, x0, 1.35, 1.9, 0.48, "FASTQ-level views:\nse (mate 1), crop36, crop50", "#d9ecf7", fs=6.2)
    box(ax, x0, 0.65, 1.9, 0.58, "alignment-level views: depth\n0.50, 0.25, 0.125; nodedup;\nmapq0 (no MAPQ filter)", "#f7e0d4", fs=6.2)
    box(ax, x0, 0.05, 1.9, 0.48, "every view split into two random\nread halves pr1, pr2", "#dff2ea", fs=6.2)
    for y0, y1 in [(1.95, 1.83), (1.35, 1.23), (0.65, 0.53)]:
        arrow(ax, x0 + 0.95, y0, x0 + 0.95, y1)
    fig.savefig(out("fig1_onion_views.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_onion(); fig_measurement(); fig_objectives(); fig_layers(); fig_baselines()
    fig_minicandi(); fig_ladder(); fig_arms(); fig_coverage(); fig_views(); fig_onion_views()
    print("ok")
