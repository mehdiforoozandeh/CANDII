"""h74 report: figures + markdown for the residual-vs-raw contrast.

HISTORICAL FORMAT. Every `M1` key below is an h5-era `candi.eval` key, and `candi.eval` is deleted
(D15). Nothing writes it any more, so this reads the ARCHIVED h74 run jsons and nothing this repo
produces. A bench json is a different file with different keys — `EVAL.md` maps the two.

    python -m candi.report_h74 \
      --case  runs_reference/reference_residual_seed0.json \
      --control runs_reference/reference_raw_seed0.json \
      --out-dir cruxvault/results/h74

Reads two `candi.train` run jsons and emits one figure per verifiable, plus `report.md` with the
`compare_arms` tables inline. Everything it needs is in the jsons — `train_terms` carries the obs/imp
split and `eval_curve` the mid-training scores, so no W&B round-trip is required (h73's forensics
needed one, which made the report un-reproducible from artifacts alone).

Sign convention throughout matches `compare_arms`: CRPS is a loss, `delta = control - case`, so
POSITIVE means the residual arm won.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

CASE_C, CTRL_C = "#c1272d", "#3b6ea5"      # residual, raw
V_C, B_C = "#2a7f62", "#b07d2b"

# One plain-language line per figure, emitted under it in the report. A figure a reader has to
# reverse-engineer from the axis labels is a figure that will be skipped.
CAPTIONS = {
    "eval_curve":
        "**Held-out score as training runs.** Lower is better; a ring marks the checkpoint each arm "
        "was scored on. Both curves fall, bottom out around epoch 8-11, then climb again for the "
        "rest of the 25 — the arms are overfitting held-out imputation well before they stop. h73 "
        "never looked mid-training and scored its last epoch, on the wrong side of this bend.",
    "train_terms":
        "**What the model was actually optimising.** Left is denoising (positions it can see), right "
        "is imputation (positions hidden from it). Both fall further than h73's 8.2%, which is what "
        "V4 demands: the recipe did change something, so a flat headline is a real negative rather "
        "than a run that never trained.",
    "paired_targets_crps":
        "**Arm against arm, one dot per target.** Below the dashed line the residual arm won. The "
        "cloud sits ON the line — the two arms are the same model with a different bookkeeping. Note "
        "the axes are shared: the `V_` dots crowd near zero while `B_` spreads to 6, which is the "
        "whole reason the `B_` delta prints as a bigger number.",
    "marginal_margin":
        "**Skill over the honest floor, per assay.** The floor is one fixed distribution per assay — "
        "no positional information at all. Both arms clear it everywhere, and the dotted line shows "
        "h73 cleared it by the same amount. This bar is easy; clearing it was never evidence of the "
        "cell-type deviation h27 names.",
    "vs_reference_bar":
        "**The bar that matters, log-log.** The x axis is what you get by averaging the OTHER cell "
        "types at the same position, with no model at all. On the diagonal means the trained model "
        "added nothing. On `V_` both arms sit on it; on `B_` most dots sit ABOVE it, so the model is "
        "worse than the average on the split it never got to select on.",
    "delta_vs_reference_depth":
        "**Does the gain arrive where the reference is a real average?** x is how many cell types "
        "went into that assay's reference; y is the delta as a PERCENT of the arm's own level, so "
        "the two splits can share an axis. The black diamonds are bin means with 95% CIs, and they "
        "do climb: -2.6% -> +2.8% -> +6.4% on `V_`. Two cautions before anyone spends a GPU on it. "
        "The top bin's CI still grazes zero, and depth is nearly the same variable as assay "
        "identity — the deep end IS DNase-seq, H3K27ac, H3K36me3, H3K4me3. So this is a lead for a "
        "better reference, not a finding.",
    "scale_vs_shape":
        "**Is any gain about magnitude or about placement?** `crps` is the score as measured; "
        "`crps_oracle_scaled` hands each target its best possible overall level for free, so what "
        "survives is placement alone; `scale_error` is the level miss on its own, which a "
        "recalibration could fix. All three barely move between the arms.",
}


def _load(p) -> dict:
    return json.loads(Path(p).read_text())


def _label(d: dict, p) -> str:
    return d.get("config", {}).get("tag") or Path(p).stem


def _split(key: str) -> str:
    parts = key.split("|")
    return parts[1][:1] if len(parts) >= 2 and parts[1][:1] in ("V", "B") else "?"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_eval_curve(case, ctrl, cl, kl, out: Path) -> Optional[Path]:
    """V4 + the over/under-fitting question h73 could not answer.

    A curve that falls and then rises is overfitting; one that is flat means the recipe changed
    nothing and V1 is uninterpretable rather than negative. h73 had a single point here.
    """
    cc, kc = case.get("eval_curve") or [], ctrl.get("eval_curve") or []
    if not cc and not kc:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, split, col in zip(axes, ("V", "B"), (V_C, B_C)):
        for curve, lab, c, m in ((kc, kl, CTRL_C, "o"), (cc, cl, CASE_C, "s")):
            if not curve:
                continue
            x = [p["epoch"] for p in curve]
            y = [p[f"{split}_imp_crps"] for p in curve]
            ax.plot(x, y, marker=m, color=c, label=lab, lw=1.8, ms=5)
            best = int(np.nanargmin(y))
            ax.scatter([x[best]], [y[best]], s=140, facecolors="none", edgecolors=c, lw=2, zorder=5)
        ax.set_title(f"{split}_ held-out imputation ({'gating' if split == 'V' else 'clean'})",
                     color=col)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    # All positions, not the foreground: this is the SELECTION metric and it must match V1's.
    axes[0].set_ylabel("imputation CRPS, all positions\n(lower is better)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Mid-training imputation, every 3 epochs — rings mark the selected checkpoint",
                 fontsize=11)
    fig.tight_layout()
    p = out / "eval_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_train_terms(case, ctrl, cl, kl, out: Path) -> Optional[Path]:
    """V4 training side: does `imp` actually descend, and by more than h73's 8.2%?"""
    def per_epoch(d, key):
        t = d.get("train_terms") or []
        by: Dict[int, List[float]] = {}
        for r in t:
            v = r.get(key)
            if v is not None and np.isfinite(v):
                by.setdefault(r["epoch"], []).append(float(v))
        eps = sorted(by)
        return eps, [float(np.median(by[e])) for e in eps]

    if not (case.get("train_terms") or ctrl.get("train_terms")):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, ("obs", "imp"),
                              ("obs — denoising (unmasked)", "imp — imputation (cloze)")):
        for d, lab, c in ((ctrl, kl, CTRL_C), (case, cl, CASE_C)):
            e, y = per_epoch(d, key)
            if not e:
                continue
            ax.plot(e, y, color=c, lw=2, label=f"{lab}  ({(y[0] - y[-1]) / y[0]:+.1%})")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("median NB NLL per epoch")
    fig.suptitle("Training loss, split. h73's imp fell 8.2% over 25 epochs — V4 needs more than that",
                 fontsize=11)
    fig.tight_layout()
    p = out / "train_terms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_paired_targets(case, ctrl, cl, kl, out: Path, metric: str = "crps") -> Optional[Path]:
    """V1 made visible: one point per target, both arms scoring the same positions."""
    ct = case["M1"].get("imp_per_target") or {}
    kt = ctrl["M1"].get("imp_per_target") or {}
    shared = sorted(set(ct) & set(kt))
    if not shared:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lo = hi = None
    for ax, split, col in zip(axes, ("V", "B"), (V_C, B_C)):
        ks = [k for k in shared if _split(k) == split]
        x = np.array([kt[k].get(metric, np.nan) for k in ks], float)   # control (raw)
        y = np.array([ct[k].get(metric, np.nan) for k in ks], float)   # case (residual)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if not x.size:
            continue
        lo = min(x.min(), y.min()) if lo is None else min(lo, x.min(), y.min())
        hi = max(x.max(), y.max()) if hi is None else max(hi, x.max(), y.max())
        wins = int((y < x).sum())
        ax.scatter(x, y, s=26, alpha=0.75, color=col, edgecolors="none")
        ax.set_title(f"{split}_  n={x.size}  residual better on {wins}/{x.size}", color=col)
        ax.set_xlabel(f"{kl}  ({metric})")
        ax.set_ylabel(f"{cl}  ({metric})")
        ax.grid(alpha=0.25)
    for ax in axes:
        if lo is not None:
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    fig.suptitle("Paired per-target imputation CRPS — below the diagonal = the residual arm won",
                 fontsize=11)
    fig.tight_layout()
    p = out / f"paired_targets_{metric}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


H73_MARGINAL_MARGIN = 0.196   # measured on h73's `off` arm: 0.691 vs 0.887 over 23/23 imputed assays


def fig_marginal_margin(case, ctrl, cl, kl, out: Path) -> Optional[Path]:
    """V2: the margin over the per-assay marginal baseline, against h73's MEASURED +0.196.

    The '~+0.02' this figure first drew was the WEAKEST assay (H2AK5ac), not the mean. It is the
    same wrong number V2 was pre-registered against and corrected for before the run.
    """
    cpa = case["M1"].get("imp_per_assay") or {}
    kpa = ctrl["M1"].get("imp_per_assay") or {}
    assays = [a for a in sorted(set(cpa) & set(kpa))
              if np.isfinite(cpa[a].get("marg_crps", np.nan))
              and np.isfinite(kpa[a].get("marg_crps", np.nan))]
    if not assays:
        return None
    cm = np.array([kpa[a]["marg_crps"] - cpa[a]["crps"] for a in assays])   # case margin
    km = np.array([kpa[a]["marg_crps"] - kpa[a]["crps"] for a in assays])   # control margin
    order = np.argsort(-cm)
    assays = [assays[i] for i in order]
    cm, km = cm[order], km[order]
    fig, ax = plt.subplots(figsize=(max(8, 0.34 * len(assays)), 4.6))
    xs = np.arange(len(assays))
    ax.bar(xs - 0.2, km, 0.4, color=CTRL_C, label=f"{kl}  (mean {km.mean():+.4f})")
    ax.bar(xs + 0.2, cm, 0.4, color=CASE_C, label=f"{cl}  (mean {cm.mean():+.4f})")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(H73_MARGINAL_MARGIN, color="grey", ls=":", lw=1.2)
    ax.text(len(assays) - 0.5, H73_MARGINAL_MARGIN + 0.005,
            f"h73 `off` sat here (mean {H73_MARGINAL_MARGIN:+.3f})", ha="right", va="bottom",
            fontsize=8, color="grey")
    ax.set_xticks(xs)
    ax.set_xticklabels(assays, rotation=90, fontsize=7)
    ax.set_ylabel("marginal CRPS − arm CRPS\n(higher = more skill over the honest bar)")
    ax.set_title("V2 — margin over the per-assay marginal baseline, imputation targets")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = out / "marginal_margin.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _assay_contributors(case: dict) -> Optional[Dict[str, int]]:
    """Per-assay `T_` contributor counts, read from the reference the run actually used."""
    p = (case.get("config") or {}).get("reference_path")
    if not p or not Path(p).exists():
        return None
    import h5py
    with h5py.File(p, "r") as f:
        assays = json.loads(f.attrs["assays"])
        return {a: int(c) for a, c in zip(assays, f["count"][:])}


DEPTH_BINS = ((1, 9), (10, 19), (20, 40))


def depth_strata(case, ctrl) -> List[dict]:
    """Per-target RELATIVE delta against reference depth, binned, `n_fg`-weighted, with a CI.

    Relative, not absolute: `B_` sits ~4x higher in CRPS units than `V_`, so an absolute y-axis
    plots the count level and buries the effect. Dividing by the arm's own level is what lets the
    two splits share an axis at all.
    """
    counts = _assay_contributors(case)
    if not counts:
        return []
    ct = case["M1"].get("imp_per_target") or {}
    kt = ctrl["M1"].get("imp_per_target") or {}
    rows = []
    for k in sorted(set(ct) & set(kt)):
        a = k.split("|")[-1]
        va, vb = ct[k].get("crps"), kt[k].get("crps")
        if a not in counts or va is None or vb is None:
            continue
        if not (np.isfinite(va) and np.isfinite(vb)) or vb <= 0:
            continue
        rows.append(dict(depth=counts[a], assay=a, split=_split(k), lvl=float(vb),
                         rel=(float(vb) - float(va)) / float(vb),      # raw - residual, over raw
                         n_fg=max(int(kt[k].get("n_points", 1) or 1), 1)))
    return rows


def _wboot(v, w, n_boot: int = 4000, seed: int = 0):
    v, w = np.asarray(v, float), np.asarray(w, float)
    i = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    b = (v[i] * w[i]).sum(1) / w[i].sum(1)
    return float((v * w).sum() / w.sum()), *(float(x) for x in np.quantile(b, [0.025, 0.975]))


def fig_delta_vs_reference_depth(case, ctrl, cl, kl, out: Path) -> Optional[Path]:
    """Does the residual arm win where the reference is built from more cell types?

    Reported as a stratification, not a pre-registered verifiable — and read with care. Depth is
    nearly collinear with assay identity here (most depth values carry 1-2 assays, and the deep end
    IS the four core marks), so a gradient along this axis cannot be separated from "core marks are
    easier". What it is NOT is the count level: the partial correlation of depth with the relative
    delta, holding the CRPS level fixed, keeps its sign and size.
    """
    rows = depth_strata(case, ctrl)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    jit = np.random.default_rng(0)
    for split, col, m in (("V", V_C, "o"), ("B", B_C, "^")):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        xs = np.array([r["depth"] for r in sub], float)
        ax.scatter(xs * jit.uniform(0.94, 1.06, len(xs)), [r["rel"] * 100 for r in sub],
                   s=30, alpha=0.6, color=col, marker=m, edgecolors="none", label=f"{split}_")
    for lo, hi in DEPTH_BINS:
        sub = [r for r in rows if lo <= r["depth"] <= hi]
        if len(sub) < 3:
            continue
        mu, cl_, ch = _wboot([r["rel"] for r in sub], [r["n_fg"] for r in sub])
        xc = float(np.exp(np.mean(np.log([r["depth"] for r in sub]))))
        ax.errorbar(xc, mu * 100, yerr=[[(mu - cl_) * 100], [(ch - mu) * 100]], fmt="D",
                    color="k", ms=6, capsize=4, lw=1.6, zorder=5)
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("T_ cell types contributing to that assay's reference (log scale, jittered)")
    ax.set_ylabel("per-target delta as % of the raw arm's own CRPS\n(positive = residual won)")
    ax.set_title("Does the residual arm win where the reference is built from more cells?")
    ax.legend(frameon=False, fontsize=9, title="black = bin mean, 95% CI", title_fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out / "delta_vs_reference_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _reference_by_target(case: dict, key: str = "crps") -> Dict[str, float]:
    """Collapse the reference baseline to ONE value per target, `n_points`-weighted.

    `reference_only_baseline` emits one record per EVAL UNIT — ~11 window batches per target on the
    full panel — whereas `imp_per_target` pools all of a target's units into a single record. Keying
    the reference by target with a plain dict comprehension silently keeps only the LAST unit, so the
    arm's ~46-window score gets compared against the reference's 4-window score and the contrast comes
    out enormous and meaningless. (Measured: it turned a true +0.005 tie into a spurious -0.355 'the
    model crushes the reference'.) Collapse first, the same way `_cluster_bootstrap_ci` does.
    """
    rb = case.get("reference_only_baseline") or {}
    agg: Dict[str, List[float]] = {}
    for r in rb.get("per_target", []):
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        k = "|".join(r["target"])
        w = max(float(r.get("n_points", 1) or 1), 1.0)
        s = agg.setdefault(k, [0.0, 0.0])
        s[0] += w * float(v)
        s[1] += w
    return {k: a / b for k, (a, b) in agg.items() if b > 0}


def _vs_reference_ci(arm: dict, case: dict, split: str, n_boot: int = 2000) -> Optional[dict]:
    """Paired arm-vs-reference delta. Positive = the REFERENCE is better."""
    from candi.stats import cluster_bootstrap_ci as _cluster_bootstrap_ci
    ref = _reference_by_target(case)
    if not ref:
        return None
    pt = arm["M1"].get("imp_per_target") or {}
    recs = []
    for k in pt:
        if _split(k) != split or k not in ref:
            continue
        a = pt[k].get("crps")
        if a is None or not np.isfinite(a):
            continue
        recs.append(dict(target=tuple(k.split("|")),
                         n_fg=max(int(pt[k].get("n_points", 1) or 1), 1), d=float(a) - ref[k]))
    return _cluster_bootstrap_ci(recs, value_key="d", n_boot=n_boot, seed=0) if recs else None


def fig_vs_reference_bar(case, ctrl, cl, kl, out: Path) -> Optional[Path]:
    """Both arms against the h27 bar, per target. The question behind q23, drawn.

    A trained model whose points sit ON the diagonal is reproducing the cross-cell average and
    nothing more — which is what the pre-run measurement found for h73, and the reason this figure
    exists rather than a summary line.
    """
    if not (case.get("reference_only_baseline") or {}).get("available"):
        return None
    ref = _reference_by_target(case)
    if not ref:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, split, col in zip(axes, ("V", "B"), (V_C, B_C)):
        # Log scale FIRST, then limits derived from the DATA. Reading `get_xlim()` off a
        # linear-autoscaled axis and then switching to log collapses the panel to an empty box —
        # which is exactly what this figure did until it was looked at.
        ax.set_xscale("log")
        ax.set_yscale("log")
        vals = []
        for d, lab, c, m in ((ctrl, kl, CTRL_C, "o"), (case, cl, CASE_C, "s")):
            pt = d["M1"].get("imp_per_target") or {}
            ks = [k for k in pt if _split(k) == split and k in ref
                  and np.isfinite(pt[k].get("crps", np.nan))]
            x = np.array([ref[k] for k in ks], float)
            y = np.array([pt[k]["crps"] for k in ks], float)
            keep = (x > 0) & (y > 0)                       # a log axis cannot show a zero CRPS
            x, y = x[keep], y[keep]
            if not x.size:
                continue
            ax.scatter(x, y, s=28, alpha=0.7, color=c, marker=m, edgecolors="none",
                       label=f"{lab} — beats the bar on {int((y < x).sum())}/{x.size}")
            vals += [x, y]
        if not vals:
            continue
        allv = np.concatenate(vals)
        lo, hi = float(allv.min()) * 0.7, float(allv.max()) * 1.4
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{split}_ imputation targets", color=col)
        ax.set_xlabel("reference alone — no model (CRPS)")
        ax.set_ylabel("trained arm (CRPS)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, which="both")
    fig.suptitle("Both arms against the h27 bar. On the diagonal = the model matched the average "
                 "of the other cell types and added nothing.", fontsize=10)
    fig.tight_layout()
    p = out / "vs_reference_bar.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_scale_vs_shape(case, ctrl, cl, kl, out: Path) -> Optional[Path]:
    """V5: a gain that vanishes under crps_oracle_scaled is a LEVEL claim, not a PLACEMENT claim."""
    rows = []
    for d, lab in ((ctrl, kl), (case, cl)):
        pt = d["M1"].get("imp_per_target") or {}
        for split in ("V", "B"):
            ks = [k for k in pt if _split(k) == split]
            if not ks:
                continue
            def w(metric):
                v = np.array([pt[k].get(metric, np.nan) for k in ks], float)
                n = np.array([max(pt[k].get("n_points", 1) or 1, 1) for k in ks], float)
                m = np.isfinite(v)
                return float((v[m] * n[m]).sum() / n[m].sum()) if m.any() else np.nan
            rows.append((lab, split, w("crps"), w("crps_oracle_scaled"), w("scale_error")))
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, split, col in zip(axes, ("V", "B"), (V_C, B_C)):
        sub = [r for r in rows if r[1] == split]
        if not sub:
            continue
        xs = np.arange(3)
        for i, (lab, _s, a, b, c) in enumerate(sub):
            ax.bar(xs + (i - 0.5) * 0.38, [a, b, c], 0.38,
                   color=(CTRL_C if i == 0 else CASE_C), label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels(["crps", "crps_oracle_scaled\n(capability)", "scale_error\n(fixable)"],
                           fontsize=8)
        ax.set_title(f"{split}_ imputation", color=col)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("V5 — scale versus shape. A win in `crps` that is absent in `crps_oracle_scaled` "
                 "is a level claim.", fontsize=10)
    fig.tight_layout()
    p = out / "scale_vs_shape.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _contrast_table(case, ctrl) -> Tuple[str, dict]:
    from candi.compare_arms import compare
    lines = ["| metric | split | n | delta (raw − residual) | 95% CI | +/−/= | sign p | verdict |",
             "|---|---|---:|---:|---|---|---:|---|"]
    res = {}
    for metric in ("crps", "crps_oracle_scaled", "scale_error"):
        for stratum in ("V", "B"):
            r = compare(case, ctrl, block="imp", metric=metric, stratum=stratum)
            res[f"imp|{metric}|{stratum}"] = r
            if not r["n_clusters"]:
                continue
            verdict = ("**residual better**" if r["supports_direction"]
                       else "**raw better**" if r["hi"] < 0 else "no difference")
            lines.append(f"| `{metric}` | `{stratum}_` | {r['n_clusters']} | {r['mean']:+.5f} | "
                         f"[{r['lo']:+.5f}, {r['hi']:+.5f}] | "
                         f"{r['n_pos']}/{r['n_neg']}/{r['n_tied']} | {r['sign_test_p']:.4f} | "
                         f"{verdict} |")
    return "\n".join(lines), res


def _relative_delta_section(case, ctrl) -> List[str]:
    """Why the `B_` delta looks ~3x the `V_` one when the two arms differ by the same amount.

    CRPS carries the units of the observable, so a split whose tracks hold ~4x the reads per bin
    shows ~4x the CRPS at IDENTICAL relative accuracy. Reporting delta/raw alongside the raw delta
    is what keeps a units artifact from being read as a `B_`-specific effect.
    """
    from candi.stats import cluster_bootstrap_ci as _cluster_bootstrap_ci
    cpt = case["M1"].get("imp_per_target") or {}
    kpt = ctrl["M1"].get("imp_per_target") or {}
    out = ["## Absolute versus relative — why the `B_` delta looks bigger", "",
           "CRPS is in the units of the observable. The `B_` tracks are simply denser than the `V_` "
           "ones, so the same relative gain prints as a bigger number there. Both the delta and the "
           "CI scale with the level; the relative column is the one that compares across splits.", "",
           "| split | n | median counts/bin | raw CRPS | delta (abs) | 95% CI (abs) | delta (rel) | "
           "95% CI (rel) |",
           "|---|---:|---:|---:|---:|---|---:|---|"]
    for split in ("V", "B"):
        rows, lvl, cnt = [], [], []
        for k, v in kpt.items():
            if _split(k) != split or k not in cpt:
                continue
            a, b = v.get("crps"), cpt[k].get("crps")
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a <= 0:
                continue
            rows.append(dict(target=tuple(k.split("|")),
                             n_fg=max(int(v.get("n_points", 1) or 1), 1),
                             d=float(a) - float(b), rel=(float(a) - float(b)) / float(a)))
            lvl.append(float(a))
            cnt.append(float(v.get("median_target", np.nan)))
        if not rows:
            continue
        w = np.array([r["n_fg"] for r in rows], float)
        ab = _cluster_bootstrap_ci(rows, value_key="d", n_boot=2000, seed=0)
        re = _cluster_bootstrap_ci(rows, value_key="rel", n_boot=2000, seed=0)
        out.append(f"| `{split}_` | {ab['n_clusters']} | {np.nanmedian(cnt):.1f} | "
                   f"{np.average(lvl, weights=w):.5f} | {ab['mean']:+.5f} | "
                   f"[{ab['lo']:+.5f}, {ab['hi']:+.5f}] | {re['mean'] * 100:+.2f}% | "
                   f"[{re['lo'] * 100:+.2f}%, {re['hi'] * 100:+.2f}%] |")
    out += ["", "Read the last two columns first: the relative effect and its interval are "
                "near-identical on the two splits, so there is no `B_`-specific effect to explain — "
                "the absolute gap is the count level, not the objective.", ""]
    return out


def _fall(d, key) -> Optional[float]:
    t = d.get("train_terms") or []
    by: Dict[int, List[float]] = {}
    for r in t:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            by.setdefault(r["epoch"], []).append(float(v))
    if len(by) < 2:
        return None
    eps = sorted(by)
    a, b = float(np.median(by[eps[0]])), float(np.median(by[eps[-1]]))
    return (a - b) / a if a else None


def _prior_section(case, ctrl, cl, kl, prior_p) -> List[str]:
    """h74's arms against a PRIOR run (h73) — what the harness fixes bought, separately from the idea.

    Not a controlled contrast: `raw` differs from h73's `off` in all four harness fixes AND
    best-checkpoint selection at once. It is still worth reporting, because eval is deterministic and
    both were scored on the same targets and positions, so the delta is paired — and because it turns
    out to be larger and better-supported than the contrast this node was built to test.
    """
    from candi.compare_arms import compare
    try:
        prior = _load(prior_p)
    except Exception as e:                                        # noqa: BLE001
        return [f"_Prior-run comparison unavailable: {e}_", ""]
    pl = _label(prior, prior_p)
    out = ["## What the harness fixes bought (not a pre-registered verifiable)", "",
           f"Both h74 arms against `{pl}`. Positive delta = the h74 arm is better. This is NOT a "
           "controlled contrast — the four harness fixes and best-checkpoint selection all changed at "
           "once — but evaluation is deterministic, so the same targets and positions were scored and "
           "the per-target delta is paired.", "",
           "| h74 arm | split | n | delta (prior − h74) | 95% CI | verdict |",
           "|---|---|---:|---:|---|---|"]
    for arm, lab in ((ctrl, kl), (case, cl)):
        for s in ("V", "B"):
            r = compare(arm, prior, block="imp", metric="crps", stratum=s)
            if not r["n_clusters"]:
                continue
            v = ("**h74 better**" if r["supports_direction"]
                 else "**prior better**" if r["hi"] < 0 else "no difference")
            out.append(f"| `{lab}` | `{s}_` | {r['n_clusters']} | {r['mean']:+.5f} | "
                       f"[{r['lo']:+.5f}, {r['hi']:+.5f}] | {v} |")
    out += ["", ""]
    return out


def build(case_p, ctrl_p, out_dir, prior_p=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    case, ctrl = _load(case_p), _load(ctrl_p)
    cl, kl = _label(case, case_p), _label(ctrl, ctrl_p)

    figs = [f for f in (
        fig_eval_curve(case, ctrl, cl, kl, out),
        fig_train_terms(case, ctrl, cl, kl, out),
        fig_paired_targets(case, ctrl, cl, kl, out),
        fig_marginal_margin(case, ctrl, cl, kl, out),
        fig_vs_reference_bar(case, ctrl, cl, kl, out),
        fig_delta_vs_reference_depth(case, ctrl, cl, kl, out),
        fig_scale_vs_shape(case, ctrl, cl, kl, out),
    ) if f is not None]

    table, res = _contrast_table(case, ctrl)
    v1 = res.get("imp|crps|V")
    ccfg, kcfg = case.get("config", {}), ctrl.get("config", {})
    cbest, kbest = case.get("best_checkpoint", {}), ctrl.get("best_checkpoint", {})

    def margin(d):
        pa = d["M1"].get("imp_per_assay") or {}
        v = [pa[a]["marg_crps"] - pa[a]["crps"] for a in pa
             if np.isfinite(pa[a].get("marg_crps", np.nan))]
        return float(np.mean(v)) if v else float("nan")

    md = [
        "# h74 — predicting the residual over a per-position average reference",
        "",
        f"`{cl}` (case) versus `{kl}` (control). CRPS is a loss, so **delta = raw − residual** and a "
        "**positive** delta means the residual arm won. Both arms carry the four harness fixes, so "
        "none of them can explain the contrast; they differ in `--reference` alone.",
        "",
        "## Headline (V1, gating)",
        "",
    ]
    if v1 and v1["n_clusters"]:
        md += [f"On the `V_` split, paired per-target imputation CRPS: "
               f"**{v1['mean']:+.5f} [{v1['lo']:+.5f}, {v1['hi']:+.5f}]** over {v1['n_clusters']} "
               f"targets — "
               f"{'the CI excludes zero in the residual arms favour, V1 MET' if v1['supports_direction'] else ('the CI excludes zero AGAINST the residual arm' if v1['hi'] < 0 else 'the CI spans zero, V1 UNMET')}.",
               ""]
    md += [
        "## All contrasts",
        "",
        table,
        "",
    ]
    md += _relative_delta_section(case, ctrl)
    md += [
        "## Run configuration",
        "",
        "| | residual | raw |",
        "|---|---|---|",
        f"| `--reference` | `{ccfg.get('reference')}` | `{kcfg.get('reference')}` |",
        f"| epochs x steps | {ccfg.get('epochs')} x {ccfg.get('steps_per_epoch')} | "
        f"{kcfg.get('epochs')} x {kcfg.get('steps_per_epoch')} |",
        f"| `--imp-weight` | {ccfg.get('imp_weight')} | {kcfg.get('imp_weight')} |",
        f"| `--unmask-frac` | {ccfg.get('unmask_frac')} | {kcfg.get('unmask_frac')} |",
        f"| seed | {ccfg.get('seed')} | {kcfg.get('seed')} |",
        f"| selected checkpoint | epoch {cbest.get('epoch')} ({cbest.get('scored')}) | "
        f"epoch {kbest.get('epoch')} ({kbest.get('scored')}) |",
        f"| reference contributors | {ccfg.get('reference_n_contributors')} | "
        f"{kcfg.get('reference_n_contributors')} |",
        f"| wall clock | {case.get('wall_s')}s | {ctrl.get('wall_s')}s |",
        "",
        "## V2 — margin over the marginal baseline",
        "",
        f"Mean over imputed assays: residual **{margin(case):+.5f}**, raw **{margin(ctrl):+.5f}**. "
        "For scale, h73's `off` arm measured +0.196 here (0.691 vs 0.887, 22% relative). V2 is the "
        "PAIRED comparison between these two, not a fixed constant.",
        "",
        "### The h27 bar — the average reference scored on its own",
        "",
    ]
    rb = case.get("reference_only_baseline") or {}
    if rb.get("available"):
        cpt = case["M1"].get("imp_per_target") or {}
        kpt = ctrl["M1"].get("imp_per_target") or {}

        def arm_mean(pt, split):
            ks = [k for k in pt if _split(k) == split]
            v = np.array([pt[k].get("crps", np.nan) for k in ks], float)
            n = np.array([max(pt[k].get("n_points", 1) or 1, 1) for k in ks], float)
            m = np.isfinite(v)
            return float((v[m] * n[m]).sum() / n[m].sum()) if m.any() else float("nan")

        md += ["The marginal baseline above is one CONSTANT distribution per assay. h27's claim is "
               "about the per-position average ACROSS cell types, which is far stronger and which no "
               "candi run had ever been scored against. Now it can be: the forecast is "
               "`mu = R * 2^(d - depth_center)` with the model's own depth prompt and a per-target "
               "CRPS-optimal dispersion — an oracle nuisance parameter, granted deliberately so the "
               "bar is not beaten on a technicality. It is a property of the DATA, so one number "
               "serves both arms.",
               "",
               "| split | reference alone | raw | residual |",
               "|---|---:|---:|---:|"]
        for split in ("V", "B"):
            md.append(f"| `{split}_` | {rb.get(f'{split}_crps', float('nan')):.5f} "
                      f"(n={rb.get(f'{split}_n_targets', 0)}) | {arm_mean(kpt, split):.5f} | "
                      f"{arm_mean(cpt, split):.5f} |")
        md += ["",
               "Paired and target-clustered, 2,000 bootstrap samples. **Positive delta = the "
               "REFERENCE is better.**", "",
               "| arm | split | n | delta (arm − reference) | 95% CI | arm wins | verdict |",
               "|---|---|---:|---:|---|---|---|"]
        for arm, lab in ((ctrl, kl), (case, cl)):
            for split in ("V", "B"):
                c = _vs_reference_ci(arm, case, split)
                if not c:
                    continue
                v = ("**reference better**" if c["lo"] > 0
                     else "**arm better**" if c["hi"] < 0 else "no difference")
                md.append(f"| `{lab}` | `{split}_` | {c['n_clusters']} | {c['mean']:+.5f} | "
                          f"[{c['lo']:+.5f}, {c['hi']:+.5f}] | "
                          f"{c['n_neg']}/{c['n_clusters']} | {v} |")
        md += ["", "**An arm that does not beat this bar has learned no deviation at all**, whatever "
                   "it does against the constant baseline.", ""]
    else:
        md += ["_Not available: only a run carrying the reference can compute it, and the raw arm "
               "does not._", ""]
    md += [
        "## Reference coverage (not a verifiable — a stratification)",
        "",
    ]
    counts = _assay_contributors(case)
    strata = depth_strata(case, ctrl)
    if counts:
        thin = {a: c for a, c in counts.items() if c <= 2}
        scored = sorted({r["assay"] for r in strata})
        thin_scored = {a: c for a, c in thin.items() if a in scored}
        md += [f"The reference is not equally strong everywhere: contributors per assay run from "
               f"{min(counts.values())} to {max(counts.values())}, and {len(thin)} assays have <=2 "
               f"({', '.join(f'`{a}`={c}' for a, c in sorted(thin.items(), key=lambda x: x[1]))}).",
               "",
               f"**None of those thin assays is ever scored.** {len(thin_scored)} of them appear "
               f"among the {len(scored)} assays that carry an imputation target, so the shallowest "
               f"reference any measured number rests on is "
               f"{min(r['depth'] for r in strata)} cell types, not 1. An earlier draft of this "
               "section implied h27's mechanism was being tested without an average; it was not.",
               ""]
    if strata:
        md += ["Binned, `n_fg`-weighted, delta as a percent of the arm's own level so the splits are "
               "comparable:", "",
               "| split | reference depth | n | relative delta | 95% CI |",
               "|---|---|---:|---:|---|"]
        for split in ("V", "B"):
            for lo, hi in DEPTH_BINS:
                sub = [r for r in strata if r["split"] == split and lo <= r["depth"] <= hi]
                if len(sub) < 3:
                    continue
                mu, c_lo, c_hi = _wboot([r["rel"] for r in sub], [r["n_fg"] for r in sub])
                md.append(f"| `{split}_` | {lo}-{hi} cells | {len(sub)} | {mu * 100:+.2f}% | "
                          f"[{c_lo * 100:+.2f}%, {c_hi * 100:+.2f}%] |")
        d = np.array([r["depth"] for r in strata], float)
        lv = np.array([r["lvl"] for r in strata], float)
        rl = np.array([r["rel"] for r in strata], float)

        def _resid(y, x):
            return y - np.polyval(np.polyfit(x, y, 1), x)

        md += ["",
               f"The gradient is real in sign and it is NOT the count level: depth correlates with "
               f"level at {np.corrcoef(d, lv)[0, 1]:+.3f}, but the partial correlation of depth with "
               f"the relative delta HOLDING level fixed is "
               f"{np.corrcoef(_resid(d, lv), _resid(rl, lv))[0, 1]:+.3f}, while level holding depth "
               f"fixed is {np.corrcoef(_resid(lv, d), _resid(rl, d))[0, 1]:+.3f}. "
               "What it cannot separate is depth from assay identity: most depth values carry one or "
               "two assays, and the deep bin is exactly the four core marks. Treat it as the "
               "motivation for a better reference (ontology-group mean, median, learned pooling), "
               "not as evidence that depth per se is the lever.",
               ""]
    md += [
        "## V4 — training-side sanity",
        "",
    ]
    for d, lab in ((case, cl), (ctrl, kl)):
        fo, fi = _fall(d, "obs"), _fall(d, "imp")
        md.append(f"- `{lab}`: obs fell {fo:+.1%}, imp fell {fi:+.1%} across training."
                  if fo is not None and fi is not None else f"- `{lab}`: training terms unavailable.")
    md += ["", "h73's recipe managed 8.2% on `imp`. V4 requires more than that AND a non-flat "
                "mid-training eval curve; if neither moves, V1 is uninterpretable rather than negative.",
           ""]
    if prior_p:
        md += _prior_section(case, ctrl, cl, kl, prior_p)
    md += ["## Figures", "",
           "Every figure carries the plain-language reading under it; the axis labels alone are not "
           "the finding.", ""]
    for f in figs:
        md.append(f"### {f.stem}")
        md.append("")
        md.append(f"![{f.stem}]({f.name})")
        md.append("")
        cap = CAPTIONS.get(f.stem)
        if cap:
            md.append(cap)
            md.append("")
    md += [
        "## Scope limits, pre-registered",
        "",
        "- **V6 selection bias.** The best checkpoint is chosen on `V_`, so absolute `V_` numbers are "
        "optimistic for BOTH arms. The bias is common-mode, so the paired delta survives it; `B_` is "
        "the clean split because selection never touches it.",
        "- **V7 scope.** One seed per arm. The target-clustered CI speaks to variability across "
        "targets within this run, not to run-to-run variability.",
        "- **`B_` is under-characterised by design.** 9 of the 12 `B_` cell types carry <=2 `T_` "
        "tracks, so a `B_` null partly reports ENCODE's data collection rather than the objective.",
        "- **The train/eval mismatch is only half-fixed.** 15% unmasked batches correct the encoder's "
        "input distribution, not the decoder prompt structure.",
        "- **Counts only.** The Gaussian signal head and the Bernoulli peak head receive no gradient "
        "in this harness.",
        "",
    ]
    p = out / "report.md"
    p.write_text("\n".join(md) + "\n")
    print(f"[report] wrote {p} and {len(figs)} figures", flush=True)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True, help="the residual arm run json")
    ap.add_argument("--control", required=True, help="the raw arm run json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prior", default=None,
                    help="optional earlier run json (e.g. h73's off arm) to report the harness-fix "
                         "delta against; scored on the same targets because eval is deterministic")
    a = ap.parse_args()
    build(a.case, a.control, a.out_dir, prior_p=a.prior)


if __name__ == "__main__":
    main()
