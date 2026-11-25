#!/usr/bin/env python3
"""Join the four validation boards on the tracks all four scored -> SIDE_BY_SIDE.tsv, stdout summary."""
import csv, statistics as st
B = [("raw", "rcp_combo_w2x_unet_s0_raw"), ("insample", "rcp_s0_slope_insample"),
     ("loeo", "rcp_s0_slope_loeo"), ("depthcap", "rcp_s0_depthcap")]
d = {}
for k, f in B:
    for r in csv.DictReader(open(f"{f}/board_validation.tsv"), delimiter="\t"):
        if r["measure"] in ("combined_rank", "nmse", "gwcorr"):
            d.setdefault(k, {}).setdefault(r["experiment"], {})[r["measure"]] = float(r["value"])
common = sorted(set.intersection(*(set(v) for v in d.values())))
with open("SIDE_BY_SIDE.tsv", "w") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["experiment"] + [f"{k}_{m}" for k, _ in B for m in ("combined_rank", "nmse")])
    for e in common:
        w.writerow([e] + [f"{d[k][e][m]:.4g}" for k, _ in B for m in ("combined_rank", "nmse")])
print(len(common), "common tracks")
for k, _ in B:
    cr = [d[k][e]["combined_rank"] for e in common]; nm = [d[k][e]["nmse"] for e in common]
    print(f"{k:9s} mean combined {st.mean(cr):.2f}  median nmse {st.median(nm):.3g}  nmse>1 {sum(x>1 for x in nm)}/{len(nm)}  rank<=3 {sum(x<=3 for x in cr)}  rank>=10 {sum(x>=10 for x in cr)}")
for k, _ in B[1:]:
    dc = [d[k][e]["combined_rank"] - d["raw"][e]["combined_rank"] for e in common]
    print(f"{k:9s} vs raw: better {sum(x<-0.05 for x in dc)}, worse {sum(x>0.05 for x in dc)}, mean change {st.mean(dc):+.2f}")
