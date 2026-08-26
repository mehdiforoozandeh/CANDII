"""Roll the per-chromosome anchor files into one verdict table. `RIVALS_PLAN.md` §7.4.

The anchor gate is not "is our MSE small" — it is **"does the port, retrained on the challenge's
own training data, score like their submission scored"**. So every row carries both columns and
their ratio, and the verdict is read off the ratio.

Two aggregations, because they answer different questions and disagree when tracks differ in
difficulty:

- **pooled** — concatenate every bin of every chromosome, then one MSE. This is what a
  genome-wide challenge measure does.
- **macro** — mean over tracks of the per-track value. This is what `RIVALS_PLAN.md` §6.4 reports,
  and it refuses to let one enormous chromosome carry the table.

`ratio = ours / theirs`, so **1.0 is parity, below 1 is better than their submission, above 1 is
worse**. The gate wording in §7.4 is "should approach their published rows"; the operational
reading used here is stated in `VERDICT_BANDS` and is a proposal to the PI, not a ruling.

**What this is not.** Not the 001 vendored EIC scorer (§7.5). Not comparable to a published
leaderboard row, and not to any Dataset-2 number — 005's translation result is why. Every record
carries `anchor.CAVEAT` saying so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .anchor import CAVEAT

__all__ = ["VERDICT_BANDS", "verdict_for", "build_report", "main"]

#: Proposal to the PI, not a ruling. The bands are on the macro MSE ratio, ours / theirs.
VERDICT_BANDS = (
    (1.10, "APPROACHES — within 10% of their submitted tracks on the same instrument"),
    (1.25, "NEAR — within 25%; the port reproduces the method but not their exact run"),
    (1.60, "SHORT — materially worse; look for a training or feature difference before quoting"),
    (float("inf"), "FAILS — not the same method's performance; do not proceed to P1/P2 on this"),
)


def verdict_for(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "UNDEFINED — no comparable tracks"
    for edge, text in VERDICT_BANDS:
        if ratio <= edge:
            return text
    return VERDICT_BANDS[-1][1]


def build_report(files: List[Path]) -> Dict:
    per_chrom, all_rows = [], []
    for f in sorted(files):
        obj = json.loads(Path(f).read_text(encoding="utf-8"))
        rows = obj.get("tracks", [])
        for r in rows:
            r = dict(r)
            r["chrom"] = obj["chrom"]
            all_rows.append(r)
        per_chrom.append({
            "chrom": obj["chrom"], "n_tracks": obj["n_tracks"],
            "ours": obj["macro_ours_vs_truth_mse"], "theirs": obj["macro_theirs_vs_truth_mse"],
            "ratio": (obj["macro_ours_vs_truth_mse"] / obj["macro_theirs_vs_truth_mse"]
                      if obj["macro_theirs_vs_truth_mse"] > 0 else float("nan")),
            "r_ours": obj["macro_ours_vs_truth_pearson"],
            "r_theirs": obj["macro_theirs_vs_truth_pearson"],
            "r_ours_theirs": obj["macro_ours_vs_theirs_pearson"],
        })

    def m(key):
        vals = [r[key] for r in all_rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    # Pooled: weight each track-chromosome by its bin count, which is what a genome-wide MSE does.
    w = np.array([r["n_bins"] for r in all_rows], dtype=np.float64)
    def pooled(key):
        v = np.array([r[key] for r in all_rows], dtype=np.float64)
        ok = np.isfinite(v)
        return float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")

    macro_o, macro_t = m("ours_vs_truth_mse"), m("theirs_vs_truth_mse")
    pool_o, pool_t = pooled("ours_vs_truth_mse"), pooled("theirs_vs_truth_mse")
    macro_ratio = macro_o / macro_t if macro_t > 0 else float("nan")

    # Per-track ratio distribution — one bad assay hiding inside a good mean is the thing to catch.
    ratios = np.array([r["ours_vs_truth_mse"] / r["theirs_vs_truth_mse"]
                       for r in all_rows
                       if r.get("theirs_vs_truth_mse", 0) > 0
                       and np.isfinite(r.get("ours_vs_truth_mse", np.nan))])
    by_mark: Dict[str, List[float]] = {}
    for r in all_rows:
        if r.get("theirs_vs_truth_mse", 0) > 0:
            by_mark.setdefault(r["mark"], []).append(r["ours_vs_truth_mse"] / r["theirs_vs_truth_mse"])

    return {
        "caveat": CAVEAT,
        "verdict": verdict_for(macro_ratio),
        "macro_ratio_ours_over_theirs": macro_ratio,
        "pooled_ratio_ours_over_theirs": pool_o / pool_t if pool_t > 0 else float("nan"),
        "macro_ours_mse": macro_o, "macro_theirs_mse": macro_t,
        "pooled_ours_mse": pool_o, "pooled_theirs_mse": pool_t,
        "macro_ours_pearson": m("ours_vs_truth_pearson"),
        "macro_theirs_pearson": m("theirs_vs_truth_pearson"),
        "macro_ours_theirs_pearson": m("ours_vs_theirs_pearson"),
        "n_track_chromosomes": len(all_rows),
        "n_chromosomes": len(per_chrom),
        "ratio_quantiles": {q: float(np.quantile(ratios, v)) for q, v in
                            (("p05", 0.05), ("p25", 0.25), ("p50", 0.5),
                             ("p75", 0.75), ("p95", 0.95))} if ratios.size else {},
        "worse_than_theirs_fraction": float((ratios > 1.0).mean()) if ratios.size else float("nan"),
        "by_mark_median_ratio": {k: float(np.median(v)) for k, v in sorted(by_mark.items())},
        "per_chromosome": per_chrom,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--anchor-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    ns = p.parse_args(argv)

    files = sorted(ns.anchor_dir.glob("anchor_chr*.json"))
    if not files:
        raise SystemExit(f"no anchor_chr*.json under {ns.anchor_dir}")
    rec = build_report(files)
    ns.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")

    print(f"{rec['n_chromosomes']} chromosomes, {rec['n_track_chromosomes']} track-chromosomes\n")
    print(f"{'chrom':8s} {'n':>4s} {'ours':>9s} {'theirs':>9s} {'ratio':>7s} "
          f"{'r_ours':>7s} {'r_thrs':>7s} {'r_o~t':>7s}")
    for c in rec["per_chromosome"]:
        print(f"{c['chrom']:8s} {c['n_tracks']:4d} {c['ours']:9.5f} {c['theirs']:9.5f} "
              f"{c['ratio']:7.3f} {c['r_ours']:7.4f} {c['r_theirs']:7.4f} {c['r_ours_theirs']:7.4f}")
    print(f"\nmacro   ours {rec['macro_ours_mse']:.5f}  theirs {rec['macro_theirs_mse']:.5f}  "
          f"ratio {rec['macro_ratio_ours_over_theirs']:.4f}")
    print(f"pooled  ours {rec['pooled_ours_mse']:.5f}  theirs {rec['pooled_theirs_mse']:.5f}  "
          f"ratio {rec['pooled_ratio_ours_over_theirs']:.4f}")
    print(f"pearson ours {rec['macro_ours_pearson']:.4f}  theirs {rec['macro_theirs_pearson']:.4f}"
          f"  ours~theirs {rec['macro_ours_theirs_pearson']:.4f}")
    if rec["ratio_quantiles"]:
        q = rec["ratio_quantiles"]
        print(f"per-track ratio  p05 {q['p05']:.3f}  p25 {q['p25']:.3f}  p50 {q['p50']:.3f}  "
              f"p75 {q['p75']:.3f}  p95 {q['p95']:.3f}")
        print(f"worse than theirs on {rec['worse_than_theirs_fraction']*100:.1f}% of tracks")
    print(f"\nVERDICT: {rec['verdict']}")
    print(f"-> {ns.out}")
    return 0


if __name__ == "__main__":              # run as `python -m lavawizard.anchor_report`
    raise SystemExit(main())
