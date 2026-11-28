#!/usr/bin/env python3
"""Per-assay count-slope calibration of miniCANDI's predicted mean (crux task t390, fix a).

The predicted mean `mu` is over-dispersed against the observed counts (val_scale_diag): its contrasts
are too large for how well it correlates. The calibration shrinks them toward the track's own mean,

    mu' = m + b (mu - m),   m = mean of mu over the track (so the total is kept),

with b the pooled least-squares slope of observed counts Y on mu, centred per item:
b = sum_i cov_i(Y, mu) / sum_i var_i(mu), over the recalibration study's imputation items of the
same assay (`<study>/fit/items/*.npz`: Y, mu[3], meta; 32 genome regions of 16384 bins, target masked).

Two fits per scored track (no p-value truth is read by either):
  insample  every item of the assay, the track's own experiment included (its reads in other regions):
            a preview of the blind-entry gain, NOT a validation number.
  loeo      leave one experiment out: only the OTHER experiments of the assay; an assay with no other
            experiment gets b = 1 (no correction).
The mu row of the items that equals the scored seed-0 prediction is found by comparing the items with
<pred>/counts/<track>.h5 at the study's regions (`--check`), and used for the fit.

  fit    -> <out>/slopes.tsv (per track: assay, b_insample, b_loeo, n items each), <out>/row_check.tsv
  apply  -> <out>/<variant>/counts/<track>.h5 (mu', n copied)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def items(study):
    out = []
    for f in sorted(glob.glob(f"{study}/fit/items/*.npz")):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        if m["kind"] != "imputation":
            continue
        yk = "Y" if "Y" in z.files else "y"         # imputation items store the target as `y`
        out.append(dict(target=m["target"], assay=m["assay"], cell=m["cell"], regime=m["regime"],
                        Y=z[yk].astype(np.float64), mu=z["mu"].astype(np.float64), file=f))
    return out


def moments(Y, mu):
    y, x = Y - Y.mean(), mu - mu.mean()
    return float((x * y).sum()), float((x * x).sum())


def cmd_fit(a):
    import h5py
    its = items(a.study)
    ev = json.load(open(f"{a.study}/fit/calibration_evalset.json"))
    regions, rb = ev["regions"], int(ev["region_bins"])
    cov = {r["track"]: r for r in csv.DictReader(open(a.cov), delimiter="\t")}
    byid = {r["measurement_id"]: t for t, r in cov.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # which mu row is the scored seed-0 prediction: compare with the scored counts at the study's regions
    chk = []
    for it in its:
        t = byid.get(it["target"])
        h5 = Path(a.pred) / "counts" / f"{t}.h5" if t else None
        if not (h5 and h5.exists() and h5.stat().st_size == a.full_size):
            continue
        with h5py.File(h5, "r") as f:
            ref = np.concatenate([f["mu"][c][s: s + rb].astype(np.float64) for c, s in regions])
        for k in range(it["mu"].shape[0]):
            d = it["mu"][k][: ref.size] - ref[: it["mu"].shape[1]]
            chk.append([t, it["regime"], k, float(np.max(np.abs(d))), float(np.corrcoef(it["mu"][k], ref)[0, 1])])
    with open(out / "row_check.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["track", "regime", "row", "max_abs_diff", "pearson"])
        w.writerows(chk)
    rows = sorted({c[2] for c in chk if c[3] < a.tol})
    if a.row is None:
        if len(rows) != 1:
            raise SystemExit(f"mu row matching the scored prediction: {rows} (see row_check.tsv)")
        row = rows[0]
    else:
        row = a.row
    mom = [(it, *moments(it["Y"], it["mu"][row])) for it in its if it["regime"] == a.regime]
    res = []
    for t in a.tracks.split(","):
        r = cov[t]
        assay = next((it["assay"] for it in its if it["target"] == r["measurement_id"]), None)
        same = [m for m in mom if m[0]["assay"] == assay] if assay else []
        other = [m for m in same if m[0]["target"] != r["measurement_id"]]
        b_in = sum(m[1] for m in same) / sum(m[2] for m in same) if same else 1.0
        b_lo = sum(m[1] for m in other) / sum(m[2] for m in other) if other else 1.0
        res.append([t, assay or "", f"{b_in:.6g}", len(same), f"{b_lo:.6g}", len(other)])
    with open(out / "slopes.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["track", "assay", "b_insample", "n_insample", "b_loeo", "n_loeo"])
        w.writerows(res)
    json.dump(dict(study=a.study, row=row, regime=a.regime, n_items=len(its), n_used=len(mom)),
              open(out / "fit.json", "w"), indent=1)
    print(f"row {row}; {len(mom)} items ({a.regime}); {sum(1 for x in res if x[5] == 0)} of {len(res)} tracks "
          f"have no other experiment of their assay (loeo b = 1)")
    for x in res:
        print("\t".join(map(str, x)))
    return 0


def cmd_apply(a):
    import h5py
    out = Path(a.out)
    sl = {r["track"]: r for r in csv.DictReader(open(out / "slopes.tsv"), delimiter="\t")}
    for t in a.tracks.split(","):
        b = float(sl[t][f"b_{a.variant}"])
        src = Path(a.pred) / "counts" / f"{t}.h5"
        dst = out / a.variant / "counts" / f"{t}.h5"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        with h5py.File(src, "r") as f, h5py.File(tmp, "w") as g:
            chroms = list(f["mu"])
            tot = sum(float(f["mu"][c][:].astype(np.float64).sum()) for c in chroms)
            m = tot / sum(f["mu"][c].shape[0] for c in chroms)
            for c in chroms:
                mu = f["mu"][c][:].astype(np.float64)
                g.create_dataset(f"mu/{c}", data=np.maximum(m + b * (mu - m), 0.0).astype(f["mu"][c].dtype))   # b > 1 can go below 0: clip (the conversion adds eps)
                g.create_dataset(f"n/{c}", data=f["n"][c][:])
            g.attrs["calibration"] = json.dumps(dict(variant=a.variant, b=b, center=m, source=str(src)))
        tmp.rename(dst)
        print(f"{t}: b {b:.4g}, centre {m:.4g} -> {dst}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "apply"):
        q = sp.add_parser(name)
        q.add_argument("--pred", required=True, help="scored predict.py folder (counts/<track>.h5)")
        q.add_argument("--out", required=True)
        q.add_argument("--tracks", required=True, help="comma-separated EIC tracks")
        if name == "fit":
            q.add_argument("--study", required=True)
            q.add_argument("--cov", required=True)
            q.add_argument("--regime", default="leave_one_assay_out")
            q.add_argument("--row", type=int, default=None)
            q.add_argument("--tol", type=float, default=1e-2)
            q.add_argument("--full-size", type=int, default=485110096)
        else:
            q.add_argument("--variant", choices=("insample", "loeo"), required=True)
    a = p.parse_args(argv)
    return cmd_fit(a) if a.cmd == "fit" else cmd_apply(a)


if __name__ == "__main__":
    raise SystemExit(main())
