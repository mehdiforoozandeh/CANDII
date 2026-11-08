#!/usr/bin/env python3
"""Aggregate decompose.py's per-(truth, assay, entry) outputs into the two tables the EIC board reads (t149).

Definitions are Max's sprint 06 (report.html, report2.html; aggregate2.py), in units of each target track's
variance at 25 bp, asinh units:

  normalized MSE = level + residual,  level = offset + scale + nonlinear   (errdecomp.calibrate)
  residual       = shared + specific, each split over four Haar bands      (errdecomp.shared_specific;
                   groups = cell type, so a track never estimates its own shared part)

and, per component (shared or specific part of one band) of an assay, with V, P, C, E the track means of the
truth's energy, the calibrated prediction's energy, the cross term and the error energy:

  explained   = C^2 / P   (0 if C <= 0; at most V)
  added       = E - (V - explained)   (variation the prediction adds; what shrinking it would remove)
  unexplained = V - explained          (no sampling-noise term: the EIC truth has no pseudoreplicates)

"all" rows are track-weighted means over assays, as in Max's eue.tsv.

Usage: aggregate.py OUT DEST
Writes DEST/summary.tsv (one row per truth x entry x assay-or-all) and DEST/components.tsv
(one row per truth x entry x assay-or-all x part x band).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import errdecomp as E

K = E.J + 1
BANDS = {"fine": [0, 1, 2], "peak": [3, 4, 5, 6], "domain": [7, 8, 9, 10], "broad": [11, 12, 13, 14]}
assert sorted(sum(BANDS.values(), [])) == list(range(K))

out, dest = sys.argv[1:3]
rows = []
for fn in sorted(glob.glob(os.path.join(out, "*", "*", "*.npz"))):
    T = pd.read_csv(fn[:-4] + ".tracks.tsv", sep="\t").set_index("experiment")
    z = np.load(fn)
    exps, cells = list(z["experiments"]), z["cells"]
    Gx, Gy = z["Gx"], z["Gy"]
    Gc = 0.5 * (z["Gxy"] + np.transpose(z["Gxy"], (0, 2, 1)))
    Ge = Gx - 2 * Gc + Gy
    parts = {nm: E.shared_specific(G, cells) for nm, G in (("V", Gx), ("P", Gy), ("C", Gc), ("E", Ge))}
    for j, e in enumerate(exps):
        t = T.loc[e]
        r = {"truth": t.truth, "entry": t.entry, "assay": t.assay, "experiment": e, "cell": t.cell,
             "mse": t.mse, "level": t.offset + t.scale + t.nonlinear, "offset": t.offset, "scale": t.scale,
             "nonlinear": t.nonlinear, "residual": t.residual, "r": t.r, "slope_ratio": t.slope_ratio,
             "e_sum_check": t.e_sum_check}
        for nm, (sh, sp) in parts.items():
            for bn, idx in BANDS.items():
                r[f"{nm}sh_{bn}"] = float(np.sum(sh[idx, j]))
                r[f"{nm}sp_{bn}"] = float(np.sum(sp[idx, j]))
        rows.append(r)
R = pd.DataFrame(rows)
bad = R[np.abs(R.e_sum_check) > 1e-4]
if len(bad):
    sys.exit(f"identity check failed on {len(bad)} tracks: Haar energies of the error do not add to the residual")


def eue(V, C, P, Ev):
    expl = (C * C / P) if (C > 0 and P > 0) else 0.0
    expl = min(expl, max(V, 0.0))
    return dict(V=V, explained=expl, unexplained=V - expl, added=Ev - (V - expl), E=Ev)


summ, comp = [], []
for (truth, entry), g in R.groupby(["truth", "entry"]):
    per = []
    for a, h in g.groupby("assay"):
        n = len(h)
        s = {"truth": truth, "entry": entry, "assay": a, "n_tracks": n,
             **{c: h[c].mean() for c in ("mse", "level", "offset", "scale", "nonlinear", "residual", "r", "slope_ratio")}}
        for p_ in ("sh", "sp"):
            for bn in BANDS:
                s[f"{p_}_{bn}"] = h[f"E{p_}_{bn}"].mean()
                s[f"truth_{p_}_{bn}"] = h[f"V{p_}_{bn}"].mean()
        s["shared"] = sum(s[f"sh_{bn}"] for bn in BANDS)
        s["specific"] = sum(s[f"sp_{bn}"] for bn in BANDS)
        summ.append(s); per.append(s)
        for p_, pn in (("sh", "shared"), ("sp", "specific")):
            for bn in BANDS:
                d = eue(*(h[f"{nm}{p_}_{bn}"].mean() for nm in ("V", "C", "P", "E")))
                comp.append({"truth": truth, "entry": entry, "assay": a, "part": pn, "band": bn, "n_tracks": n, **d})
        comp.append({"truth": truth, "entry": entry, "assay": a, "part": "level", "band": "all", "n_tracks": n,
                     "V": 0.0, "explained": 0.0, "unexplained": 0.0, "added": s["level"], "E": s["level"]})
    P_ = pd.DataFrame(per)
    w = P_.n_tracks / P_.n_tracks.sum()
    s = {"truth": truth, "entry": entry, "assay": "all", "n_tracks": int(P_.n_tracks.sum())}
    for c in P_.columns:
        if c not in s and c not in ("truth", "entry", "assay"):
            s[c] = float((P_[c] * w).sum())
    summ.append(s)
    C_ = pd.DataFrame([d for d in comp if d["truth"] == truth and d["entry"] == entry])
    for (pn, bn), h in C_.groupby(["part", "band"], sort=False):
        w = h.n_tracks / h.n_tracks.sum()
        comp.append({"truth": truth, "entry": entry, "assay": "all", "part": pn, "band": bn,
                     "n_tracks": int(h.n_tracks.sum()),
                     **{c: float((h[c] * w).sum()) for c in ("V", "explained", "unexplained", "added", "E")}})

os.makedirs(dest, exist_ok=True)
R.to_csv(os.path.join(dest, "tracks.tsv.gz"), sep="\t", index=False, float_format="%.6g")
pd.DataFrame(summ).to_csv(os.path.join(dest, "summary.tsv"), sep="\t", index=False, float_format="%.6g")
pd.DataFrame(comp).to_csv(os.path.join(dest, "components.tsv"), sep="\t", index=False, float_format="%.6g")
print(f"aggregated {len(R)} tracks -> {dest}")
