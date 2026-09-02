"""Annotate and aggregate the ingestion census (t78) — the numbers the memo quotes.

Two jobs:

1. **Annotate.** For every reported max-ingested bin and every reported top window, read
   `genome/mask.h5` at that bin and test hg38 blacklist v2 membership.  That separates the
   two ways a tower can be ingested: a bin the mask already calls invalid but which rides
   into the batch inside a 0.9-valid window, and a bin the mask calls valid outright.
2. **Aggregate.** Per assay class, per split (`T_` vs `V_`/`B_`) and per regime.

Read-only; writes `per_track.tsv` / `per_track.json` back with the extra columns, and
`summary.json` beside them.

    python ingestion_report.py <mirror_dir> [--genome …]
"""
import argparse
import json
import os

import h5py
import numpy as np

RES = 25
PFX = {("eic.19", "train"): "e19_train", ("eic.19", "eval"): "e19_eval",
       ("eic.pilot", "train"): "pilot_train", ("eic.pilot", "eval"): "pilot_eval"}


def load_blacklist(path):
    iv = {}
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or not p[0].startswith("chr"):
                continue
            iv.setdefault(p[0], []).append((int(p[1]), int(p[2]), p[3] if len(p) > 3 else ""))
    for c in iv:
        iv[c].sort()
    return iv


def bl_hit(iv, chrom, b):
    lo, hi = b * RES, (b + 1) * RES
    for s, e, reason in iv.get(chrom, []):
        if s < hi and e > lo:
            return (reason or "blacklist", e - s)
        if s >= hi:
            break
    return None


def q(vals, p):
    return float(np.percentile(np.asarray(vals, dtype=float), p)) if len(vals) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mirrordir")
    ap.add_argument("--genome", default="/project/def-maxwl/mforooz/CANDI_STORE/genome")
    a = ap.parse_args()

    rows = json.load(open(f"{a.mirrordir}/per_track.json"))
    iv = load_blacklist(f"{a.genome}/hg38-blacklist.v2.bed")
    mask = h5py.File(f"{a.genome}/mask.h5", "r")
    cache = {}

    def maskbit(chrom, b):
        if chrom not in cache:
            cache[chrom] = mask[chrom][:]
        return int(cache[chrom][b])

    for r in rows:
        for (reg, sp), p in PFX.items():
            loc = r[f"{p}_max_locus"]
            if loc == "-" or r[f"{p}_max"] == 0:
                r[f"{p}_max_mask"] = -1
                r[f"{p}_max_bl"] = ""
                continue
            c, bp = loc.split(":")
            b = int(bp) // RES
            r[f"{p}_max_mask"] = maskbit(c, b)
            hit = bl_hit(iv, c, b)
            r[f"{p}_max_bl"] = hit[0] if hit else ""
            r[f"{p}_max_bl_width_bp"] = hit[1] if hit else 0
        # the track's own top-1 bin: is its 768-bin tile eligible, and by how much does the
        # tile miss the 0.9 budget?  This is what decides whether the biggest tower is ingested.
        gl = r.get("global_max_locus", "-")
        if gl != "-":
            c, bp = gl.split(":")
            b = int(bp) // RES
            t0 = (b // 768) * 768
            if c not in cache:
                cache[c] = mask[c][:]
            w = cache[c][t0:t0 + 768]
            inv = int(768 - int(np.asarray(w, dtype=np.int64).sum())) if w.shape[0] == 768 else -1
            r["top1_tile_start_bp"] = t0 * RES
            r["top1_tile_invalid_bins"] = inv
            r["top1_tile_eligible"] = int(0 <= inv <= 76)
            r["top1_mask"] = maskbit(c, b)
            hit = bl_hit(iv, c, b)
            r["top1_in_blacklist"] = int(hit is not None)
            r["top1_bl_width_bp"] = hit[1] if hit else 0
        for tws in r["_top_windows"].values():
            for t in tws:
                t["mask"] = maskbit(t["chrom"], t["max_bin"])
                hit = bl_hit(iv, t["chrom"], t["max_bin"])
                t["in_blacklist"] = hit is not None
                t["blacklist_reason"] = hit[0] if hit else None
                t["blacklist_width_bp"] = hit[1] if hit else 0
    mask.close()

    cols = [c for c in rows[0] if c != "_top_windows"]
    for r in rows:
        for c in cols:
            r.setdefault(c, "")
    with open(f"{a.mirrordir}/per_track.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(f"{a.mirrordir}/per_track.json", "w") as fh:
        json.dump(rows, fh, indent=1)

    # ---- aggregate ------------------------------------------------------------------
    S = {"n_tracks": len(rows), "by_regime": {}}
    for reg, tag in (("eic.19", "e19"), ("eic.pilot", "pilot")):
        out = {}
        for cls in sorted({r["assay_class"] for r in rows}):
            sub = [r for r in rows if r["assay_class"] == cls]
            hit = [r for r in sub if r[f"applies_{tag}_n_over"] > 0]
            out[cls] = dict(
                n=len(sub), n_tracks_ingesting=len(hit),
                max_ratio=max((r[f"applies_{tag}_max_x_ceil"] for r in sub), default=0.0),
                median_ratio=q([r[f"applies_{tag}_max_x_ceil"] for r in sub], 50),
                worst=max(sub, key=lambda r: r[f"applies_{tag}_max"])["biosample"] if sub else "",
            )
        by_split = {}
        for sp in ("T", "V", "B"):
            sub = [r for r in rows if r["split"] == sp]
            hit = [r for r in sub if r[f"applies_{tag}_n_over"] > 0]
            by_split[sp] = dict(n=len(sub), n_tracks_ingesting=len(hit),
                                assays=sorted({r["assay_class"] for r in hit}))
        worst = max(rows, key=lambda r: r[f"applies_{tag}_max"])
        out["_by_split"] = by_split
        out["_worst_track"] = dict(
            biosample=worst["biosample"], assay=worst["assay"], split=worst["split"],
            value=worst[f"applies_{tag}_max"], locus=worst[f"applies_{tag}_max_locus"],
            ceiling=worst["keepdup1_ceiling"],
            x_ceiling=worst[f"applies_{tag}_max_x_ceil"],
            x_median_nonzero=worst[f"applies_{tag}_max_x_mednz"],
            mask=worst.get(f"{PFX[(reg, worst['applies_split'])]}_max_mask"),
            blacklist=worst.get(f"{PFX[(reg, worst['applies_split'])]}_max_bl"),
            window_set=f"{reg}/{worst['applies_split']}",
        )
        out["_n_tracks_ingesting_total"] = sum(
            1 for r in rows if r[f"applies_{tag}_n_over"] > 0)
        out["_n_tracks_with_top1_ingested"] = sum(
            1 for r in rows if r[f"applies_{tag}_has_top1"])
        S["by_regime"][reg] = out

    S["train_side_regime_diff"] = [
        dict(biosample=r["biosample"], assay=r["assay"],
             e19=dict(n_over=r["e19_train_n_over"], mx=r["e19_train_max"],
                      x=r["e19_train_max_x_ceil"], locus=r["e19_train_max_locus"]),
             pilot=dict(n_over=r["pilot_train_n_over"], mx=r["pilot_train_max"],
                        x=r["pilot_train_max_x_ceil"], locus=r["pilot_train_max_locus"]))
        for r in rows if r["split"] == "T"
        and (r["e19_train_n_over"] > 0) != (r["pilot_train_n_over"] > 0)
    ]
    inv = [r["top1_tile_invalid_bins"] for r in rows if r.get("top1_tile_invalid_bins", -1) >= 0]
    S["top1_tile"] = dict(
        n=len(inv),
        n_eligible=sum(1 for r in rows if r.get("top1_tile_eligible")),
        n_top1_blacklisted=sum(1 for r in rows if r.get("top1_in_blacklist")),
        invalid_bins_min=min(inv) if inv else -1,
        invalid_bins_median=q(inv, 50), invalid_bins_max=max(inv) if inv else -1,
        budget=76,
        bl_width_bp_min=min((r["top1_bl_width_bp"] for r in rows
                             if r.get("top1_in_blacklist")), default=0),
        bl_width_bp_median=q([r["top1_bl_width_bp"] for r in rows
                              if r.get("top1_in_blacklist")], 50),
        bl_width_bp_max=max((r["top1_bl_width_bp"] for r in rows
                             if r.get("top1_in_blacklist")), default=0),
    )
    S["eval_identical_across_regimes"] = all(
        r["e19_eval_max"] == r["pilot_eval_max"] and
        r["e19_eval_n_over"] == r["pilot_eval_n_over"] and
        r["e19_eval_n_elig"] == r["pilot_eval_n_elig"] for r in rows)

    with open(f"{a.mirrordir}/summary.json", "w") as fh:
        json.dump(S, fh, indent=1)
    print(json.dumps(S, indent=1)[:6000])


if __name__ == "__main__":
    main()
