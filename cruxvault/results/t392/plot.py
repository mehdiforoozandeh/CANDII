"""t392 side-by-side: the read-start miniCANDI against the read-count reference on 16 regions.

Runs on Fir (runall.sh copies it there): python plot.py <kit> <runs_both> <out dir>
Reads, never writes, the score folders of both models:
  <runs_both>/<arm>/seed{0,1,2}/p16_sv/scores.json   headline terms (select_rule.table, as arm-summary)
  <runs_both>/<arm>/seed{0,1,2}/p16_dev/scores.json  deviance metrics (["deviance"][key]["macro"])
  <runs_both>/rs_.../seed{0,1,2}/sanity_p16/sanity.json and p16_sv/open_edges_<i>of4.json (read-start only)
Writes in <out dir>: per_seed.tsv, table.tsv, sanity.tsv, open_edges.tsv, side_by_side.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KIT, RUNS, OUT = (Path(a) for a in sys.argv[1:4])
sys.path[:0] = [str(KIT / "tools" / "onion_eval"), str(KIT / "tools"), str(KIT / "src"), str(KIT)]
import select_rule  # noqa: E402

REF, RS = "rcp_halves_mix_avg_pearson_unet", "rs_halves_mix_avg_pearson_unet"
LABEL = {REF: "read-count reference", RS: "read-start model"}
SEEDS = (0, 1, 2)
HEADLINE = {"imputation": "imputation_raw",                                   # arm-summary columns
            "correction": "correction_gap_closed_oracle_no_nodedup_side",    # = correction_v3_raw
            "calibration": "uncertainty_raw"}
DEVIANCE = ("dev_imputation_skill", "dev_imputation_r2", "dev_correction_gap_closed",
            "dev_calibration_ratio", "nb_logscore_gain_vs_marginal")
COLS = list(HEADLINE) + list(DEVIANCE)

OUT.mkdir(parents=True, exist_ok=True)
rows = {(r["arm"], r["seed"]): r for r in select_rule.table(RUNS, "", "p16_sv", "auto", 3)}
per_seed = []
for arm in (REF, RS):
    for s in SEEDS:
        r = rows[(arm, s)]
        assert r["state"] == "done", (arm, s, r["state"])
        dv = json.loads((RUNS / arm / f"seed{s}" / "p16_dev" / "scores.json").read_text())["deviance"]
        v = {k: r[c] for k, c in HEADLINE.items()}
        v.update({k: dv[k]["macro"] for k in DEVIANCE})
        per_seed.append({"model": arm, "seed": s, **v})


def tsv(name, header, lines):
    (OUT / name).write_text("\t".join(header) + "\n"
                            + "".join("\t".join(str(x) for x in ln) + "\n" for ln in lines))


tsv("per_seed.tsv", ["model", "seed"] + COLS,
    [[p["model"], p["seed"]] + [f"{p[c]:.6f}" for c in COLS] for p in per_seed])
summ = {}
for arm in (REF, RS):
    vals = {c: [p[c] for p in per_seed if p["model"] == arm] for c in COLS}
    summ[arm] = {c: (sum(v) / len(v), max(v) - min(v)) for c, v in vals.items()}
tsv("table.tsv", ["model", "n_seeds"] + [f"{c}.{q}" for c in COLS for q in ("mean", "range")],
    [[arm, len(SEEDS)] + [f"{x:.4f}" for c in COLS for x in summ[arm][c]] for arm in (REF, RS)])

san = [json.loads((RUNS / RS / f"seed{s}" / "sanity_p16" / "sanity.json").read_text()) for s in SEEDS]
tsv("sanity.tsv", ["seed", "nb_deviance_mean", "pearson_log1p", "n_bins", "n_targets", "n_regions"],
    [[s, f"{d['nb_deviance_mean']:.6f}", f"{d['pearson_log1p']:.6f}", d["n_bins"], d["n_targets"],
      d["n_regions"]] for s, d in zip(SEEDS, san)])
edges = []
for s in SEEDS:
    for i in range(4):
        e = json.loads((RUNS / RS / f"seed{s}" / "p16_sv" / f"open_edges_{i}of4.json").read_text())
        edges.append([s, i, e["n_open_edges"], e["n_windows"]])
tsv("open_edges.tsv", ["seed", "shard", "n_open_edges", "n_windows"], edges)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [("imputation", "A  imputation (headline)", None),
          ("correction", "B  correction (headline)", None),
          ("calibration", "C  calibration (headline)", None),
          ("dev_imputation_skill", "D  imputation skill (deviance)", None),
          ("dev_correction_gap_closed", "E  correction gap closed (deviance)", None),
          ("dev_calibration_ratio", "F  calibration ratio (deviance; 1 = calibrated)", 1.0)]
COLOR = {REF: "#4C72B0", RS: "#DD8452"}
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
for ax, (c, title, line) in zip(axes.flat, PANELS):
    for j, arm in enumerate((REF, RS)):
        ys = [p[c] for p in per_seed if p["model"] == arm]
        m = summ[arm][c][0]
        ax.bar(j, m, width=0.6, color=COLOR[arm], alpha=0.35)
        ax.scatter([j] * len(ys), ys, color=COLOR[arm], s=22, zorder=3)
        ax.text(j, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    if line is not None:
        ax.axhline(line, color="k", lw=0.8, ls="--")
    lo = min(p[c] for p in per_seed)
    hi = max(p[c] for p in per_seed)
    pad = (hi - lo) * 0.6 + 0.02
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([0, 1], [LABEL[REF], LABEL[RS]], fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
fig.suptitle("Read-start vs read-count miniCANDI, eic_v1_2, 16 regions; bar = mean, dots = seeds 0-2",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "side_by_side.png", dpi=110)
print("wrote", sorted(p.name for p in OUT.iterdir()))
