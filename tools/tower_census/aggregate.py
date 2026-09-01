"""Aggregate the per-biosample tower-census JSONs into the corpus table and the answers."""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

IN = sys.argv[1]
OUT = sys.argv[2]
DIAG = sys.argv[3] if len(sys.argv) > 3 else None

rows = []
for p in sorted(glob.glob(f"{IN}/*.json")):
    d = json.load(open(p))
    for name, r in d["tracks"].items():
        r = dict(r)
        r["biosample"] = d["biosample"]
        r["split"] = d["split"]
        r["n_bins_genome"] = d["n_bins_genome"]
        rows.append(r)

print(f"tracks: {len(rows)}  biosamples: {len(set(r['biosample'] for r in rows))}")
print("by class:", Counter(r["assay_class"] for r in rows))
print("by class/split:", Counter((r["assay_class"], r["split"]) for r in rows))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(rows, open(OUT, "w"), indent=1)

# ---------- reproduction check against results/t78/diag_keepdup ----------
if DIAG:
    keys = ["depth", "total_counts", "bins_per_read", "L_eff_bp", "keepdup1_ceiling",
            "n_bins_over_ceiling", "frac_bins_over_ceiling", "excess_mass_over_ceiling",
            "frac_mass_over_ceiling", "global_max_count", "max_over_ceiling_ratio"]
    idx = {(r["biosample"], r["assay"]): r for r in rows}
    bad, n = [], 0
    for p in sorted(glob.glob(f"{DIAG}/*.json")):
        d = json.load(open(p))
        r = idx[(d["biosample"], "DNase-seq")]
        n += 1
        for k in keys:
            a, b = d[k], r[k]
            if abs(a - b) > 1e-9 * max(1.0, abs(a)):
                bad.append((d["biosample"], k, a, b))
    print(f"\nreproduction vs diag_keepdup: {n} tracks x {len(keys)} fields, "
          f"{len(bad)} mismatches")
    for x in bad[:20]:
        print("   MISMATCH", x)

# ---------- summaries ----------
def q(v, p):
    return float(np.percentile(np.asarray(v, dtype=float), p)) if len(v) else float("nan")

print("\n%-10s %-6s %4s | %-28s | %-24s | %s"
      % ("class", "split", "n", "max/ceiling ratio  med/max", "frac mass>ceil med/max",
         ">100x  >10x  >1x"))
summary = {}
for cls in ["DNase-seq", "ChIP-seq", "ATAC-seq"]:
    for sp in ["ALL", "T", "V", "B"]:
        sub = [r for r in rows if r["assay_class"] == cls and (sp == "ALL" or r["split"] == sp)]
        if not sub:
            continue
        ratio = [r["max_over_ceiling_ratio"] for r in sub]
        fm = [r["frac_mass_over_ceiling"] for r in sub]
        s = dict(n=len(sub),
                 ratio_median=q(ratio, 50), ratio_max=max(ratio), ratio_min=min(ratio),
                 fracmass_median=q(fm, 50), fracmass_max=max(fm),
                 n_ratio_gt100=sum(1 for x in ratio if x > 100),
                 n_ratio_gt10=sum(1 for x in ratio if x > 10),
                 n_ratio_gt1=sum(1 for x in ratio if x > 1))
        summary[f"{cls}/{sp}"] = s
        print("%-10s %-6s %4d | %12.2f %14.1f | %10.5f %12.5f | %5d %5d %5d"
              % (cls, sp, s["n"], s["ratio_median"], s["ratio_max"],
                 s["fracmass_median"], s["fracmass_max"],
                 s["n_ratio_gt100"], s["n_ratio_gt10"], s["n_ratio_gt1"]))

# ---------- shared tower coordinates ----------
# Cluster the top-N bins of every track by genomic position; a cluster seen in many
# BIOSAMPLES is a locus-level (mappability) artifact, one seen in a single biosample
# is a per-library blow-up.
WIN = 200  # bins == 5 kb
by_chrom = defaultdict(list)
for r in rows:
    for tb in r["top_bins"]:
        if tb["count"] > r["keepdup1_ceiling"]:
            by_chrom[tb["chrom"]].append((tb["bin"], r, tb))

clusters = []
for c, items in by_chrom.items():
    items.sort(key=lambda t: t[0])
    cur = []
    for it in items:
        if cur and it[0] - cur[-1][0] > WIN:
            clusters.append((c, cur))
            cur = []
        cur.append(it)
    if cur:
        clusters.append((c, cur))

cl = []
for c, items in clusters:
    bios = {r["biosample"] for _, r, _ in items}
    cl.append(dict(
        chrom=c,
        start_bp=min(b for b, _, _ in items) * 25,
        end_bp=(max(b for b, _, _ in items) + 1) * 25,
        n_bins=len({b for b, _, _ in items}),
        n_tracks=len({(r["biosample"], r["assay"]) for _, r, _ in items}),
        n_biosamples=len(bios),
        biosamples=sorted(bios),
        classes=dict(Counter(r["assay_class"] for _, r, _ in items)),
        splits=dict(Counter(r["split"] for _, r, _ in items)),
        assays=dict(Counter(r["assay"] for _, r, _ in items)),
        max_count=max(tb["count"] for _, _, tb in items),
        max_ratio=max(tb["over_ceiling_ratio"] for _, _, tb in items),
        any_blacklist=any(tb["in_blacklist"] for _, _, tb in items),
        any_masked_out=any(tb["mask"] == 0 for _, _, tb in items),
    ))
cl.sort(key=lambda d: (-d["n_biosamples"], -d["n_tracks"]))
print(f"\nclusters of over-ceiling top bins: {len(cl)}   "
      f"multi-biosample (>=2): {sum(1 for d in cl if d['n_biosamples'] >= 2)}")
print("%-6s %-24s %5s %5s %-22s %12s %6s %5s"
      % ("chrom", "span", "#bio", "#trk", "classes", "max", "black", "mask0"))
for d in cl[:25]:
    print("%-6s %-24s %5d %5d %-22s %12d %6s %5s"
          % (d["chrom"], f"{d['start_bp']:,}-{d['end_bp']:,}", d["n_biosamples"],
             d["n_tracks"], ",".join(f"{k}:{v}" for k, v in d["classes"].items()),
             d["max_count"], d["any_blacklist"], d["any_masked_out"]))

json.dump(dict(summary=summary, clusters=cl),
          open(OUT.replace(".json", ".clusters.json"), "w"), indent=1)
print("\nwrote", OUT, "and", OUT.replace(".json", ".clusters.json"))
