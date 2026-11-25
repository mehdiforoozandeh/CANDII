"""Fit a read-depth rule on TRAINING tracks only; apply it to the validation tracks' observed floor (check)."""
import json, glob, math, statistics as st, sys
O = "/scratch/mforooz/eicx/val/depth_rule"
def load(s): return [json.load(open(p)) for p in sorted(glob.glob(f"{O}/{s}/depth/*.json"))]
tr, va = load("train"), load("val")
def nm(d, f):  # nMSE at the grid fraction nearest to f (log scale)
    r = min(d["by_frac"], key=lambda r: abs(math.log(r["frac"]) - math.log(f))); return r["nmse"], r["frac"]
out = []
for route in ("chip", "dnase"):
    T = [d for d in tr if d["route"] == route]
    if not T: continue
    out.append(f"## {route}: {len(T)} training tracks")
    bd = [d["best_depth"] for d in T]
    out.append(f"best effective depth: median {st.median(bd)/1e6:.1f}M, quartiles {st.quantiles(bd, n=4)[0]/1e6:.1f}-{st.quantiles(bd, n=4)[2]/1e6:.1f}M")
    out.append(f"tracks whose best fraction is 1 (no thinning): {sum(d['best_frac']==1 for d in T)} of {len(T)}")
    x = [math.log(d['depth_reads']) for d in T]; y = [math.log(d['best_depth']) for d in T]
    mx, my = sum(x)/len(x), sum(y)/len(y); b = sum((a-mx)*(c-my) for a, c in zip(x, y))/sum((a-mx)**2 for a in x)
    out.append(f"log best depth = {my - b*mx:.3f} + {b:.3f} log depth")
    # candidate rules, scored on training tracks (median nMSE at the rule's fraction)
    caps = [10e6, 15e6, 20e6, 25e6, 30e6, 40e6, 60e6, 1e12]
    best = None
    for D in caps:
        v = [nm(d, min(1, D/d["depth_reads"]))[0] for d in T]
        m = st.mean(v)
        out.append(f"  cap {D/1e6:>7.0f}M: training mean nMSE {m:.4f}, median {st.median(v):.4f}")
        if best is None or m < best[1]: best = (D, m)
    out.append(f"best cap on training: {best[0]/1e6:.0f}M")
    V = [d for d in va if d["route"] == route]
    for d in V:
        n1, _ = nm(d, 1); nc, f = nm(d, min(1, best[0]/d["depth_reads"]))
        out.append(f"  val {d['track']} {d['depth_reads']/1e6:.1f}M: floor nMSE {n1:.3f} -> {nc:.3f} (frac {f}); its own best frac {d['best_frac']}")
    if V:
        a = [nm(d, 1)[0] for d in V]; c = [nm(d, min(1, best[0]/d["depth_reads"]))[0] for d in V]
        out.append(f"validation floor (chr9+chr21): mean nMSE {st.mean(a):.3f} -> {st.mean(c):.3f}; above 1 on {sum(v>1 for v in a)} -> {sum(v>1 for v in c)} of {len(V)}; better on {sum(cc<aa for aa,cc in zip(a,c))}")
open(f"{O}/DEPTH_RULE.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
