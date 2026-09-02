"""Store-wide artifact-tower census over the CANDI_STORE eic `counts` layer.

One task per BIOSAMPLE.  `--tracks signal` (the default) measures every non-control
track; `--tracks control` measures only the `chipseq-control` column; `--tracks all`
measures both.  The default reproduces the 2026-08-30 signal census bit-for-bit.

The keep-dup-1 ceiling is reproduced verbatim from
`cruxvault/results/t78/diag_keepdup/` (fir:/scratch/mforooz/CANDII_t78_diag/code/diag_keepdup.py):

    bins_per_read = total_counts / depth
    L_eff         = RES * bins_per_read - (RES - 1)
    ceiling       = int(round(2 * (L_eff + RES - 1)))

`depth` is the manifest's read depth for that track; `total_counts` is summed off the
store itself, so no read-length metadata is trusted.

READ-ONLY.  Opens counts.h5 / mask.h5 with mode "r" and writes only to --outdir.
No model, no agreement metric, no p-value: this is a data-provenance census, and the
only thing it does with a V_/B_ track is count its own bins.
"""
import argparse
import json
import os

import h5py
import numpy as np

RES = 25
CONTROL = "chipseq-control"
ROOT = "/project/def-maxwl/mforooz/CANDI_STORE/eic"
GENOME = "/project/def-maxwl/mforooz/CANDI_STORE/genome"
TOPK = 1000          # kept per (chrom, track), merged to a global top-1000
TOPN_REPORT = 50     # emitted with coordinates (the memo quotes the first 10)
XMULT = (1, 10, 100, 1000)   # bins above ceiling x this


def load_blacklist():
    iv = {}
    with open(f"{GENOME}/hg38-blacklist.v2.bed") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or not p[0].startswith("chr"):
                continue
            iv.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3] if len(p) > 3 else ""))
    for c in iv:
        iv[c].sort()
    return iv


def blacklist_hit(iv, chrom, b):
    """Does bin b -- covering [b*25, (b+1)*25) bp -- overlap any blacklist interval?"""
    lo, hi = b * RES, (b + 1) * RES
    for s, e, reason in iv.get(chrom, []):
        if s < hi and e > lo:
            return reason or "blacklist"
        if s >= hi:
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tracks", choices=("signal", "control", "all"), default="signal",
                    help="which columns of counts.h5 to measure (default: signal, as run "
                         "for the 363-track census on 2026-08-30)")
    a = ap.parse_args()

    with open(f"{ROOT}/manifest.json") as fh:
        man = json.load(fh)
    entry = man["biosamples"][a.biosample]
    meta = {t["assay"]: t for t in entry["tracks"]}

    path = f"{ROOT}/biosamples/{a.biosample}/counts.h5"
    with h5py.File(path, "r") as fc:
        tracks = json.loads(fc.attrs["tracks"])
        chroms = [c for c in entry["chroms"] if c in fc]
        want = {"signal": lambda nm: nm != CONTROL,
                "control": lambda nm: nm == CONTROL,
                "all": lambda nm: True}[a.tracks]
        cols = [(i, nm) for i, nm in enumerate(tracks) if want(nm)]
        if not cols:
            print(f"{a.biosample}: no {a.tracks} tracks, nothing to do")
            return

        # ---- pass 1: genome-wide totals, so the ceiling comes off the store ----
        totals = np.zeros(len(tracks), dtype=np.int64)
        n_bins_total = 0
        for c in chroms:
            blk = fc[c][:]
            totals += blk.sum(axis=0, dtype=np.int64)
            n_bins_total += blk.shape[0]
            del blk

        ceil_by_col, base = {}, {}
        for col, name in cols:
            depth = float(meta[name]["depth"])
            total = float(totals[col])
            bpr = total / depth
            L_eff = RES * bpr - (RES - 1)
            ceiling = max(1, int(round(2 * (L_eff + RES - 1))))
            ceil_by_col[col] = ceiling
            base[col] = dict(bins_per_read=bpr, L_eff_bp=L_eff,
                             keepdup1_ceiling=ceiling, total_counts=total, depth=depth)

        # ---- pass 2: ceiling-relative statistics + top-K with coordinates ----
        acc = {col: dict(n_over=0, excess=0, mass_in_over=0, top=[],
                         nx={x: 0 for x in XMULT}, nonzero=0) for col, _ in cols}
        for c in chroms:
            blk = fc[c][:]
            for col, _ in cols:
                v = blk[:, col].astype(np.int64)
                ceiling = ceil_by_col[col]
                m = v > ceiling
                s = acc[col]
                nov = int(m.sum())
                s["n_over"] += nov
                s["nonzero"] += int((v > 0).sum())
                for x in XMULT:
                    s["nx"][x] += int((v > ceiling * x).sum()) if x > 1 else nov
                if nov:
                    over = v[m]
                    s["mass_in_over"] += int(over.sum())
                    s["excess"] += int(over.sum()) - nov * ceiling
                k = min(TOPK, v.shape[0])
                idx = np.argpartition(v, -k)[-k:]
                idx = idx[np.argsort(v[idx])[::-1]]
                s["top"].extend((int(v[i]), c, int(i)) for i in idx if v[i] > 0)
                del v, m
            del blk

    iv = load_blacklist()
    mask_f = h5py.File(f"{GENOME}/mask.h5", "r")

    out = {"biosample": a.biosample, "split": a.biosample.split("_")[0],
           "n_bins_genome": n_bins_total, "chroms": chroms,
           "counts_dtype": entry["dtype"], "tracks": {}}

    for col, name in cols:
        s = acc[col]
        s["top"].sort(key=lambda t: -t[0])
        top = s["top"][:TOPK]
        m = meta[name]
        b = base[col]
        ceiling = b["keepdup1_ceiling"]
        total = b["total_counts"]
        gmax = top[0][0] if top else 0
        vals = np.array([t[0] for t in top], dtype=np.int64)

        rec = dict(
            assay=name,
            assay_class=("DNase-seq" if name == "DNase-seq"
                         else "ATAC-seq" if name == "ATAC-seq"
                         else "chipseq-control" if name == CONTROL else "ChIP-seq"),
            col=col,
            depth=b["depth"],
            read_length=m.get("read_length"),
            run_type=m.get("run_type"),
            file_accession=m.get("file_accession"),
            exp_accession=m.get("exp_accession"),
            total_counts=total,
            bins_per_read=b["bins_per_read"],
            L_eff_bp=b["L_eff_bp"],
            keepdup1_ceiling=ceiling,
            global_max_count=gmax,
            max_over_ceiling_ratio=gmax / ceiling,
            n_bins_over_ceiling=s["n_over"],
            frac_bins_over_ceiling=s["n_over"] / n_bins_total,
            excess_mass_over_ceiling=float(s["excess"]),
            frac_mass_over_ceiling=s["excess"] / total if total else 0.0,
            mass_in_bins_over_ceiling=float(s["mass_in_over"]),
            frac_mass_in_bins_over_ceiling=s["mass_in_over"] / total if total else 0.0,
            frac_nonzero_bins=s["nonzero"] / n_bins_total,
        )
        for x in XMULT:
            rec[f"n_bins_over_{x}x_ceiling"] = s["nx"][x]
        for k in (1, 2, 10, 100, 1000):
            rec[f"frac_mass_top{k}"] = float(vals[:k].sum()) / total if total else 0.0

        rec["top_bins"] = []
        for value, chrom, b_idx in top[:TOPN_REPORT]:
            reason = blacklist_hit(iv, chrom, b_idx)
            rec["top_bins"].append(dict(
                chrom=chrom, bin=b_idx, start_bp=b_idx * RES, end_bp=(b_idx + 1) * RES,
                count=value, over_ceiling_ratio=value / ceiling,
                mask=int(mask_f[chrom][b_idx]),
                in_blacklist=reason is not None,
                blacklist_reason=reason,
            ))
        out["tracks"][name] = rec

    mask_f.close()
    os.makedirs(a.outdir, exist_ok=True)
    with open(f"{a.outdir}/{a.biosample}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{a.biosample}: {len(cols)} tracks, {n_bins_total} bins")
    for col, name in cols:
        r = out["tracks"][name]
        print("  %-18s ceil=%-5d max=%-12d x%-12.1f over=%d (%.4f%%) mass>%.3f%%"
              % (name, r["keepdup1_ceiling"], r["global_max_count"],
                 r["max_over_ceiling_ratio"], r["n_bins_over_ceiling"],
                 100 * r["frac_bins_over_ceiling"], 100 * r["frac_mass_over_ceiling"]))


if __name__ == "__main__":
    main()
