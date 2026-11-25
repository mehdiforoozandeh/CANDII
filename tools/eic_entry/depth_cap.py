#!/usr/bin/env python3
"""Read-depth cap for the count-to-p-value conversion (crux task t390, fix b).

The challenge truth of a deep ChIP-seq experiment looks like fewer reads than the store holds: the
observed reads, thinned to an effective depth near 20 M, reproduce the truth far better (depth_rule,
DEPTH_RULE.md). The cap was chosen on the 196 TRAINING ChIP-seq tracks only (mean nMSE over the cap grid
10-60 M and none); DNase-seq gets no cap (16 training tracks, best with none).

For each ChIP-seq track with depth_reads > CAP: f = CAP / depth_reads, the predicted mean mu * f, and the
covariate depth_reads * f (lambda_bg and the treatment-to-control ratio follow). Other tracks are listed
as unchanged.
  -> <out>/counts/<track>.h5 (mu * f, n copied), <out>/cov.tsv (depth_reads capped), <out>/cap.tsv

--pred2: first average the predicted mean of two models, mu = (mu_pred + mu_pred2) / 2 (n from --pred), and
write every requested track, capped or not.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pred", required=True)
    p.add_argument("--cov", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--cap", type=float, default=20e6)
    p.add_argument("--pred2", default=None, help="second model's predict.py folder: average the two means")
    a = p.parse_args(argv)
    import h5py

    out = Path(a.out)
    (out / "counts").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.cov), delimiter="\t"))
    want = set(a.tracks.split(","))
    log = []
    for r in rows:
        t = r["track"]
        d = float(r["depth_reads"])
        f = a.cap / d if (r["route"] == "chip" and d > a.cap) else 1.0
        if t in want:
            log.append([t, r["route"], f"{d:.0f}", f"{f:.6g}", "capped" if f < 1 else "unchanged"])
        if f < 1:
            r["depth_reads"] = f"{d * f:.0f}"
        if t in want and (f < 1 or a.pred2):
            src, dst = Path(a.pred) / "counts" / f"{t}.h5", out / "counts" / f"{t}.h5"
            tmp = dst.with_name(dst.name + ".tmp")
            g2 = h5py.File(Path(a.pred2) / "counts" / f"{t}.h5", "r") if a.pred2 else None
            with h5py.File(src, "r") as g, h5py.File(tmp, "w") as h:
                for c in g["mu"]:
                    mu = g["mu"][c][:].astype(np.float64)
                    if g2 is not None:
                        mu = 0.5 * (mu + g2["mu"][c][:].astype(np.float64))
                    h.create_dataset(f"mu/{c}", data=(mu * f).astype(g["mu"][c].dtype))
                    h.create_dataset(f"n/{c}", data=g["n"][c][:])
            if g2 is not None:
                g2.close()
            tmp.rename(dst)
    with open(out / "cov.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(out / "cap.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["track", "route", "depth_reads", "factor", "status"])
        w.writerows(log)
    print("\n".join("\t".join(x) for x in log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
