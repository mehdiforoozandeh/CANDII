"""t133: rank the t112 processing knobs by how much each one moves the signal.

Exploratory (PI, 2026-10-26). For every base -> arm pair of the t112 counterfactual products, on
chr19 + chr21 with ENCODE blacklist bins removed (the t118 scored set):

  magnitude  |log2(mean arm / mean base)| over the base's top-1% 25 bp bins (its peaks).
             The signed value is kept as `log2fc_top1`.
  shape      1 - Spearman(base, arm) on 1 kb bins (40 x 25 bp: counts summed, -log10 p averaged;
             a 1 kb bin is dropped if any of its 25 bp bins is blacklisted). Rank-based, so a pure
             change of scale scores 0; the 1 kb sum shrinks the counting noise that thinning adds.
  FRiP       counts only: the fraction of a product's reads in its OWN top-1% 25 bp bins.
             Reported for base and arm, and their difference.

Counts and -log10 p are scored separately. The four p-only knobs (ratio, ctlid, ctldepth, extsize)
leave the counts bit-identical to the base, so they get p scores only. The two DNase MAPQ arms are
byte-identical to the DNase base (a t112 defect) and are dropped.

Summary: per (arm, level, mark class) the median over tracks; each knob is ranked by its
strongest level.

    python knob_effects.py --products DIR --blacklist BED --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RES = 25
COARSE = 40  # 40 x 25 bp = 1 kb
CHROMS = ("chr19", "chr21")
TOP_FRACTION = 0.01
P_ONLY = {"ratio", "ctlid", "ctldepth", "extsize"}
SPACES = {"counts": "counts25.npz", "pval": "pval25.npz"}
MARK_CLASS = {"DNase-seq": "DNase",
              "H3K27ac": "narrow", "H3K4me3": "narrow", "H3K4me1": "narrow",
              "H3K27me3": "broad", "H3K36me3": "broad", "H3K9me3": "broad"}
METRICS = ("abs_log2fc_top1", "shape_1k", "dfrip")


def read_blacklist(path) -> dict:
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and not line.startswith(("#", "track", "browser")):
            out.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    return out


def bad_bins(intervals, n: int) -> np.ndarray:
    """True where a 25 bp bin overlaps any blacklist interval by >= 1 bp."""
    d = np.zeros(n + 1, dtype=np.int64)
    for s, e in intervals or []:
        a, b = s // RES, min((e - 1) // RES, n - 1)
        if e > s and a < n:
            d[a] += 1
            d[b + 1] -= 1
    return np.cumsum(d[:-1]) > 0


def load(product_dir: Path, space: str, blacklist: dict) -> dict:
    """{'fine': kept 25 bp bins, 'coarse': kept 1 kb bins}, chr19 then chr21."""
    z = np.load(product_dir / SPACES[space], allow_pickle=False)
    fine, coarse = [], []
    for c in CHROMS:
        x = z[c].astype(np.float64)
        bad = bad_bins(blacklist.get(c), x.size)
        fine.append(x[~bad])
        m = (x.size // COARSE) * COARSE
        xc = x[:m].reshape(-1, COARSE)
        xc = xc.sum(1) if space == "counts" else xc.mean(1)
        coarse.append(xc[~bad[:m].reshape(-1, COARSE).any(1)])
    z.close()
    return {"fine": np.concatenate(fine), "coarse": np.concatenate(coarse)}


def top_idx(x: np.ndarray) -> np.ndarray:
    k = max(1, int(round(TOP_FRACTION * x.size)))
    return np.argpartition(x, -k)[-k:]


def frip(x: np.ndarray) -> float:
    return float(x[top_idx(x)].sum() / x.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True, type=Path)
    ap.add_argument("--blacklist", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    blacklist = read_blacklist(a.blacklist)
    with open(a.products / "MANIFEST.tsv", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    records = []
    for track in sorted({r["track"] for r in rows}):
        tr = [r for r in rows if r["track"] == track]
        base = next(r for r in tr if r["arm"] == "base")
        mclass = MARK_CLASS[base["assay"]]
        for space in SPACES:
            b = load(a.products / base["pid"], space, blacklist)
            top = top_idx(b["fine"])
            frip_b = frip(b["fine"]) if space == "counts" else None
            for r in sorted((r for r in tr if r["arm"] != "base"), key=lambda r: r["pid"]):
                if mclass == "DNase" and r["arm"] == "mapq":
                    continue
                if space == "counts" and r["arm"] in P_ONLY:
                    continue
                x = load(a.products / r["pid"], space, blacklist)
                lfc = float(np.log2(x["fine"][top].mean() / b["fine"][top].mean()))
                rec = {"track": track, "assay": base["assay"], "mark_class": mclass,
                       "arm": r["arm"], "level": r["level"], "knob": r["knob"], "space": space,
                       "log2fc_top1": lfc, "abs_log2fc_top1": abs(lfc),
                       "shape_1k": float(1 - spearmanr(b["coarse"], x["coarse"]).statistic),
                       "frip_base": frip_b, "frip_arm": None, "dfrip": None,
                       "n_fine": int(b["fine"].size), "n_coarse": int(b["coarse"].size)}
                if space == "counts":
                    rec["frip_arm"] = frip(x["fine"])
                    rec["dfrip"] = rec["frip_arm"] - frip_b
                records.append(rec)
                print(f"{r['pid']:32s} {space:6s} lfc={lfc:+.3f} shape={rec['shape_1k']:.4f}"
                      f" dfrip={rec['dfrip']}", flush=True)

    cols = list(records[0])
    with open(a.out / "pairs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(records)

    # median over tracks per (space, mark class, arm, level)
    groups: dict = {}
    for rec in records:
        groups.setdefault((rec["space"], rec["mark_class"], rec["arm"], rec["level"]), []).append(rec)
    levels = []
    for (space, mclass, arm, level), recs in sorted(groups.items()):
        row = {"space": space, "mark_class": mclass, "arm": arm, "level": level, "n_tracks": len(recs),
               "median_log2fc_top1": float(np.median([r["log2fc_top1"] for r in recs]))}
        for m in METRICS:
            v = [r[m] for r in recs if r[m] is not None]
            row[f"median_{m}"] = float(np.median(v)) if v else None
        levels.append(row)
    with open(a.out / "levels.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(levels[0]), delimiter="\t")
        w.writeheader()
        w.writerows(levels)

    # each knob's strongest level, ranked, per (space, mark class, metric)
    ranks = []
    for space in SPACES:
        for mclass in ("DNase", "narrow", "broad"):
            for m in METRICS:
                if m == "dfrip" and space != "counts":
                    continue
                best: dict = {}
                for row in levels:
                    v = row[f"median_{m}"]
                    if row["space"] != space or row["mark_class"] != mclass or v is None:
                        continue
                    if row["arm"] not in best or abs(v) > abs(best[row["arm"]][1]):
                        best[row["arm"]] = (row["level"], v)
                order = sorted(best.items(), key=lambda kv: -abs(kv[1][1]))
                for i, (arm, (level, v)) in enumerate(order, 1):
                    ranks.append({"space": space, "mark_class": mclass, "metric": m, "rank": i,
                                  "arm": arm, "strongest_level": level, "value": v})
    with open(a.out / "ranking.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ranks[0]), delimiter="\t")
        w.writeheader()
        w.writerows(ranks)
    (a.out / "config.json").write_text(json.dumps(
        {"chroms": CHROMS, "res": RES, "coarse_bins": COARSE, "top_fraction": TOP_FRACTION,
         "blacklist": a.blacklist, "products": str(a.products), "p_only": sorted(P_ONLY)}, indent=1))
    print("done", len(records), "pair records")


if __name__ == "__main__":
    main()
