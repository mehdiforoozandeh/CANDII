"""miniCANDI architecture evolution on eic_v1_2: imputation, correction, calibration per step.

Reads only the score tables in this folder (copied from Fir runs_rcp/_sel/). No number is typed.
  all_p16_sv.tsv          arm summaries, 16 pair regions (screen), every arm
  base_p256_sv.tsv        base models, 256 pair regions (confirm)
  arms_p256_sv_1200.tsv   recipe arms, 256 pair regions, read 2026-11-22 12:00 UTC
  scores_ladder.md        the six-rung ladder on eic_v1 (onion run, results/metrics_screening/)
Run: python plot_evolution.py  ->  evolution.png, evolution.svg, evolution_points.tsv
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
IMP, COR, CAL = "imputation_raw", "correction_v3_raw", "uncertainty_raw"


def load(name):
    with open(HERE / name) as fh:
        return {r["arm"]: r for r in csv.DictReader((l for l in fh if not l.startswith("#")),
                                                    delimiter="\t")}


p16 = load("all_p16_sv.tsv")
p256 = {**load("base_p256_sv.tsv"), **load("arms_p256_sv_1200.tsv")}

# the path: (arm, short label, what changed)
STEPS = [
    ("eic_v1_2_rung2_base_heldout", "Conv AE", "conv encoder-decoder"),
    ("eic_v1_2_rung3_base_heldout", "Conv + attention", "+ transformer at bottleneck"),
    ("eic_v1_2_rung4_base_heldout", "U-Net", "+ skip connections"),
    ("rcp_headskip_off_unet", "U-Net, head skip off", "diagnosis: cut the copy path"),
    ("rcp_halves_unet", "U-Net + read-disjoint pairs", "RECIPE: pr1 in, pr2 target"),
    ("rcp_halves_mix_pearson_unet", "+ Pearson-residual input", "candidate add-on"),
    ("rcp_halves_mix_avg_unet", "+ average-signal input", "candidate add-on"),
]
# U-Net single fixes without read-disjoint pairs (the grey cloud)
FIXES = ["rcp_cac_cnn_unet", "rcp_cac_cnn_corefilm_off_unet", "rcp_cac_curve_unet",
         "rcp_cac_curve_corefilm_off_unet", "rcp_film_all_unet", "rcp_film_bottleneck_unet",
         "rcp_film_decoder_unet", "rcp_film_encoder_unet", "rcp_headskip_film_unet",
         "rcp_skipfilm_ccp_unet", "rcp_skipfilm_ccp_headskip_film_unet", "rcp_mix_avg_unet",
         "rcp_mix_pearson_unet", "rcp_mix_fixdepth_unet", "rcp_nocov_unet"]
BASES = STEPS[:3]


def val(tab, arm, m):
    r = tab[arm]
    return float(r[f"{m}.mean"]), float(r[f"{m}.range"]), int(r["n_seeds"])


rows = []
for arm, lab, why in STEPS:
    i, ir, n = val(p16, arm, IMP)
    c, cr, _ = val(p16, arm, COR)
    k, kr, _ = val(p16, arm, CAL)
    c256 = val(p256, arm, COR) if arm in p256 else (None, None, 0)
    rows.append(dict(arm=arm, label=lab, why=why, n16=n, imp=i, imp_r=ir, cor16=c, cor16_r=cr,
                     cal=k, cal_r=kr, cor256=c256[0], cor256_r=c256[1], n256=c256[2]))
with open(HERE / "evolution_points.tsv", "w") as fh:
    w = csv.DictWriter(fh, rows[0].keys(), delimiter="\t")
    w.writeheader()
    w.writerows(rows)

best = {m: max(val(p16, a, m)[0] for a, *_ in BASES) for m in (IMP, COR, CAL)}
# D1 floor (tools/onion_eval/pareto_select.py): best base mean minus that base's own seed range
floor = {}
for m in (IMP, COR, CAL):
    a = max(BASES, key=lambda t: val(p16, t[0], m)[0])[0]
    floor[m] = val(p16, a, m)[0] - val(p16, a, m)[1]

C_BASE, C_FIX, C_WIN, C_ADD, C_DIAG = "#6b7280", "#c7cbd1", "#d1495b", "#edae49", "#00798c"
col = [C_BASE, C_BASE, C_BASE, C_DIAG, C_WIN, C_ADD, C_ADD]

# ladder era (eic_v1): imputation and correction (102 pairs, oracle floor) from scores_ladder.md
LADDER = {"rung1_base": "MLP, no context", "rung2_base": "Conv AE", "rung3_base": "Conv + attention",
          "rung4_base": "U-Net", "rung5_base": "Axial attention", "rung6_base": "Perceiver IO",
          "baseline position_average": "position average"}
lad = {}
for line in (HERE / "scores_ladder.md").read_text().splitlines():
    cells = [c.strip() for c in line.strip("|").split("|")]
    if cells and cells[0] in LADDER and cells[0] not in lad:
        num = lambda c: (float(c.split()[0]), float(c.split("(")[1].rstrip(")")) if "(" in c else 0.0)
        lad[cells[0]] = (num(cells[2]), num(cells[4]))

fig = plt.figure(figsize=(20, 6.4))
gs = fig.add_gridspec(3, 3, width_ratios=[0.85, 1.25, 1], hspace=0.12, wspace=0.22)
c0 = fig.add_subplot(gs[:, 0])
for k, ((i, ir), (c, cr)) in lad.items():
    u = k == "rung4_base"
    kept = k in ("rung2_base", "rung3_base", "rung4_base")
    colr = C_BASE if kept else C_FIX
    c0.errorbar(i, c, xerr=ir / 2, yerr=cr / 2, fmt="o", ms=8, color=colr, mec="white", lw=1)
    left = k == "rung2_base"
    c0.annotate(LADDER[k] + ("" if kept else " (dropped)"), (i, c),
                xytext=(-8, 4) if left else (8, 6 if k == "baseline position_average" else -10 if k == "rung3_base" else 4),
                ha="right" if left else "left", textcoords="offset points", fontsize=8.5, color="#374151" if kept else "#9ca3af")
c0.annotate("", xy=(lad["rung4_base"][0][0], lad["rung4_base"][1][0]),
            xytext=(lad["rung3_base"][0][0], lad["rung3_base"][1][0]),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#374151"))
c0.set_xlabel("imputation")
c0.set_ylabel("correction, older scale (102 pairs)")
c0.set_title("A   2026-11-14: six-model ladder (eic_v1)", loc="left", fontsize=11,
             fontweight="bold")
c0.text(0.02, 0.47, "skips already traded correction for imputation;\n"
        "axial and Perceiver dropped. Not comparable\nwith B and C (other data, scale).",
        transform=c0.transAxes, fontsize=8, color="#6b7280", va="top")
c0.set_xlim(0.08, 0.56)
for s_ in ("top", "right"):
    c0.spines[s_].set_visible(False)
ax = fig.add_subplot(gs[:, 1])

# A: imputation x correction, the trade-off and how it was broken
fx = [val(p16, a, IMP)[0] for a in FIXES]
fy = [val(p16, a, COR)[0] for a in FIXES]
ax.scatter(fx, fy, s=40, c=C_FIX, edgecolor="none", zorder=1,
           label=f"{len(FIXES)} single U-Net fixes, no pairs\n(FiLM placement, CAC, conditioned skips,\nextra inputs)")
ax.axvline(best[IMP], ls="--", lw=1, c="#9ca3af")
ax.axhline(best[COR], ls="--", lw=1, c="#9ca3af")
ax.axvline(floor[IMP], ls=":", lw=1, c="#9ca3af")
ax.axhline(floor[COR], ls=":", lw=1, c="#9ca3af")
ax.text(best[IMP], 0.435, " best base", fontsize=8, color="#6b7280", va="bottom")
ax.text(floor[IMP], 0.435, "floor ", fontsize=8, color="#6b7280", va="bottom", ha="right")
ax.text(0.499, best[COR], "best base", fontsize=8, color="#6b7280", va="bottom", ha="right")
ax.text(0.499, floor[COR], "floor", fontsize=8, color="#6b7280", va="top", ha="right")
path = [0, 1, 2, 4]
for a, b in zip(path, path[1:]):
    ax.annotate("", xy=(rows[b]["imp"], rows[b]["cor16"]), xytext=(rows[a]["imp"], rows[a]["cor16"]),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#374151"), zorder=2)
for j in (5, 6):
    ax.annotate("", xy=(rows[j]["imp"], rows[j]["cor16"]), xytext=(rows[4]["imp"], rows[4]["cor16"]),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, ls=":", color=C_ADD), zorder=2)
ax.annotate("", xy=(rows[3]["imp"], rows[3]["cor16"]), xytext=(rows[2]["imp"], rows[2]["cor16"]),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, ls=":", color=C_DIAG), zorder=2)
offs = [(-8, 10), (8, 8), (8, -14), (-8, -14), (10, -4), (-8, 8), (-2, -16)]
ha = ["right", "left", "left", "right", "left", "right", "right"]
for r, c, (dx, dy), h, num in zip(rows, col, offs, ha, range(1, 8)):
    ax.errorbar(r["imp"], r["cor16"], xerr=r["imp_r"] / 2, yerr=r["cor16_r"] / 2, fmt="o",
                ms=11 if c == C_WIN else 8, color=c, mec="white", mew=1, capsize=0, lw=1, zorder=3)
    ax.annotate(f"{num}. {r['label']}" + (f"  (n={r['n16']})" if r["n16"] < 3 else ""),
                (r["imp"], r["cor16"]), xytext=(dx, dy), textcoords="offset points", ha=h,
                fontsize=9, fontweight="bold" if c == C_WIN else "normal", color=c, zorder=4)
ax.fill_between([floor[IMP], 0.5], floor[COR], 0.82, color=C_WIN, alpha=0.06, zorder=0)
ax.text(0.499, 0.818, "at or above every base model on both (D1 floor)", ha="right", va="top",
        fontsize=8, color=C_WIN)
ax.set_xlim(0.34, 0.50)
ax.set_ylim(0.43, 0.82)
ax.set_xlim(0.34, 0.50)
ax.set_xlabel("imputation (higher is better)")
ax.set_ylabel("correction, 16 pair regions (higher is better)")
ax.set_title("B   2026-11-21/22 (eic_v1_2): the U-Net trade-off and the fix", loc="left",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower left", fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# B: each objective per step
xs = range(1, len(rows) + 1)
for k, (m, key, name) in enumerate([(IMP, "imp", "imputation"), (COR, "cor16", "correction"),
                                    (CAL, "cal", "calibration")]):
    b = fig.add_subplot(gs[k, 2])
    rk = "cor16_r" if key == "cor16" else f"{key}_r"
    for x, r, c in zip(xs, rows, col):
        b.errorbar(x, r[key], yerr=r[rk] / 2, fmt="o", ms=7, color=c, mec="white", lw=1)
        if key == "cor16" and r["cor256"] is not None:
            b.plot(x + 0.22, r["cor256"], marker="D", ms=5, mfc="white", mec=c, ls="none")
    b.axhline(best[m], ls="--", lw=1, c="#9ca3af")
    b.axhline(floor[m], ls=":", lw=1, c="#9ca3af")
    b.set_ylabel(name, fontsize=9)
    b.set_xticks(list(xs))
    b.set_xticklabels([str(x) for x in xs] if k == 2 else [])
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    if k == 0:
        b.set_title("C   Each objective per step (dashed = best base, dotted = floor)", loc="left",
                    fontsize=11, fontweight="bold")
    if key == "cor16":
        b.text(len(rows) + 0.4, best[m], "◇ = 256 regions", fontsize=7, color="#6b7280",
               ha="right", va="bottom")
b.set_xlabel("step (numbers as in B)")

fig.text(0.01, 0.005, "B and C: eic_v1_2, same scorer for every point; dot = seed mean, bar = seed range "
         "(3 seeds unless n shown).\nCorrection at 16 regions for all points (screen); 256-region "
         "confirm where run (step 7: 1 seed at 256). Floor = best base mean minus its seed range (Pareto rule). Sources: runs_rcp/_sel/all_p16_sv.tsv, base_p256_sv.tsv, "
         "arms_p256_sv_1200.tsv (Fir); A: onion results/metrics_screening/scores_ladder.md.", fontsize=7.5, color="#6b7280")
fig.savefig(HERE / "evolution.png", dpi=160, bbox_inches="tight")
fig.savefig(HERE / "evolution.svg", bbox_inches="tight")
print("ok", {k: round(v, 4) for k, v in best.items()})
