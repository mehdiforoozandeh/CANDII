"""The formula h on real t112 products, one chromosome, with SPAN_EXTRA = 24 (old) and 25 (store rule).

    python real_chr.py <products_dir> <out.tsv> [--chrom chr21] [--pids P1,P2,...]

Per product and variant ("bin", "frag"): formula.P with `hmap.formula.SPAN_EXTRA` set to 24 and
to 25, against the product's true P (`pval25.npz`): mean P, MSE, Pearson r; and the 24 -> 25 change:
bins that differ, max |dP|, mean dP. ProductInfo comes from `hmap.products.load` on the repo's
t112 manifest, `tools/t118/covariates.tsv` and `hmap/control_read_length.tsv`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
for p in (REPO / "tools" / "t118", REPO / "tools" / "t125"):
    sys.path.insert(0, str(p))

from hmap import formula, products  # noqa: E402

PIDS = ("C07M20__base__base", "C07M20__crop__36", "C07M20__ratio__k2", "C07M20__ctldepth__q0.25",
        "C07M20__ctlid__none", "C19M22__base__base", "C12M02__base__base", "C40M17__base__base")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("products_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--chrom", default="chr21")
    ap.add_argument("--pids", default=",".join(PIDS))
    a = ap.parse_args()
    infos = products.load(REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv",
                          REPO / "tools" / "t118" / "covariates.tsv")
    rows = []
    for pid in a.pids.split(","):
        inf = infos[pid]
        d = a.products_dir / pid
        t = np.load(d / "counts25.npz")[a.chrom].astype(np.float64)
        y = np.load(d / "pval25.npz")[a.chrom].astype(np.float64)
        c = np.load(d / "control_counts25.npz")[a.chrom].astype(np.float64) if inf.has_control else None
        for v in formula.VARIANTS:
            got = {}
            for span in (24, 25):
                formula.SPAN_EXTRA = span
                got[span] = formula.P(v, inf, t, c)
            dp = got[25] - got[24]
            r = dict(pid=pid, assay=inf.assay, read_length=inf.read_length,
                     control_read_length=inf.control_read_length, fraglen=inf.fraglen,
                     has_control=inf.has_control, variant=v, chrom=a.chrom, n_bins=t.size,
                     mean_true=f"{y.mean():.6f}")
            for span in (24, 25):
                p = got[span]
                r[f"mean_P_{span}"] = f"{p.mean():.6f}"
                r[f"mse_{span}"] = f"{np.mean((p - y) ** 2):.6f}"
                r[f"pearson_{span}"] = f"{np.corrcoef(p, y)[0, 1]:.6f}"
            r["bins_changed"] = int(np.count_nonzero(dp))
            r["frac_changed"] = f"{np.count_nonzero(dp) / dp.size:.6f}"
            r["max_abs_dP"] = f"{np.abs(dp).max():.6f}"
            r["mean_dP"] = f"{dp.mean():.6g}"
            rows.append(r)
            print(pid, v, r["mean_P_24"], r["mean_P_25"], r["mse_24"], r["mse_25"], flush=True)
    formula.SPAN_EXTRA = 25
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
