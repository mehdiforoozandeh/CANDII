"""The mechanism the OPEN block names: what do the INVALID bins inside an eligible
window actually carry? (t78, second pass)

`ingestion_census.py` reports the largest bin any eligible window ingests, wherever it
sits.  This pass asks the narrower question the §10 OPEN block asked: D12 lets a 768-bin
window carry up to 76 bins at `mask == 0`, and nothing zeroes them — so how many masked
bins actually enter a batch, and what is the largest value on one?

Only the regime's own window slices are read, so this is a small fraction of the
genome-wide first pass.  Ceilings come from the first pass's `per_track.json`.

READ-ONLY.  Writes only to --outdir.

    python ingestion_masked.py --biosample … --outdir … --per-track … --src … --pilot-bed …
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

RES = 25
CB = 768
MVF = 0.9
PILOT_TRAIN_CHROMS = [
    "chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
    "chr12", "chr13", "chr14", "chr15", "chr16", "chr18", "chr19", "chrX",
]
EVAL_CHROMS = ["chr20", "chr21", "chr22"]
SHA = "13e11a198fdee08edb7797d1e402b5d985846b5a7d973ade91e8511462acb7a3"


def spans(starts):
    """Merge sorted window starts into contiguous `[a, b)` bin spans."""
    out = []
    for s in starts:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s + CB)
        else:
            out.append([s, s + CB])
    return [(int(a), int(b)) for a, b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per-track", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--pilot-bed", required=True)
    ap.add_argument("--store", default="/project/def-maxwl/mforooz/CANDI_STORE/eic")
    ap.add_argument("--genome", default="/project/def-maxwl/mforooz/CANDI_STORE/genome")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    from candi.store.genome import eligible_window_mask
    from candi.store.regime import RegionSet

    pilot = RegionSet.from_obj({"bed": os.path.abspath(a.pilot_bed), "sha256": SHA})
    ceil = {r["assay"]: r["keepdup1_ceiling"]
            for r in json.load(open(a.per_track)) if r["biosample"] == a.biosample}

    with open(f"{a.store}/manifest.json") as fh:
        man = json.load(fh)
    entry = man["biosamples"][a.biosample]

    with h5py.File(f"{a.genome}/mask.h5", "r") as fm, \
            h5py.File(f"{a.store}/biosamples/{a.biosample}/counts.h5", "r") as fc:
        tracks = json.loads(fc.attrs["tracks"])
        chroms = [c for c in entry["chroms"] if c in fc]
        maskc = {}
        elig = {}
        for c in chroms:
            n = fc[c].shape[0]
            m = np.asarray(fm[c][:n], dtype=np.uint8)
            maskc[c] = m
            ok = eligible_window_mask(m, CB, MVF)
            e = ok[::CB] if ok.size else np.zeros(0, dtype=bool)
            elig[c] = np.flatnonzero(e).astype(np.int64) * CB

        def wins(chrom_list, regions):
            out = {}
            for c in chrom_list:
                if c not in elig:
                    continue
                s = elig[c]
                if regions is not None:
                    s = regions.contained_starts(c, s, CB, RES)
                if s.size:
                    out[c] = np.sort(s)
            return out

        sets = {
            "eic.19/train": wins(["chr19"], None),
            "eic.19/eval": wins(EVAL_CHROMS, None),
            "eic.pilot/train": wins(PILOT_TRAIN_CHROMS, pilot),
            "eic.pilot/eval": wins(EVAL_CHROMS, None),
        }

        out = {"biosample": a.biosample, "split": a.biosample.split("_")[0], "sets": {}}
        for sname, ws in sets.items():
            acc = {t: dict(n_windows=0, n_windows_with_invalid=0, n_invalid_bins=0,
                           max_invalid_bin_count=0, max_invalid_locus="-",
                           n_invalid_over_ceiling=0, max_invalid_over_ceiling_ratio=0.0)
                   for t in tracks}
            for c, starts in ws.items():
                m = maskc[c]
                for a0, b0 in spans(starts):
                    blk = fc[c][a0:b0]
                    msl = m[a0:b0]
                    for s in starts[(starts >= a0) & (starts + CB <= b0)]:
                        o = int(s) - a0
                        inv = np.flatnonzero(msl[o:o + CB] == 0)
                        for ti, t in enumerate(tracks):
                            r = acc[t]
                            r["n_windows"] += 1
                            if inv.size == 0:
                                continue
                            r["n_windows_with_invalid"] += 1
                            r["n_invalid_bins"] += int(inv.size)
                            v = blk[o:o + CB, ti][inv].astype(np.int64)
                            k = int(v.max())
                            if k > r["max_invalid_bin_count"]:
                                r["max_invalid_bin_count"] = k
                                b = int(s) + int(inv[int(np.argmax(v))])
                                r["max_invalid_locus"] = f"{c}:{b * RES}"
                            cl = ceil.get(t, 1)
                            r["n_invalid_over_ceiling"] += int((v > cl).sum())
                    del blk
            for t in tracks:
                r = acc[t]
                cl = ceil.get(t, 1)
                r["keepdup1_ceiling"] = cl
                r["max_invalid_over_ceiling_ratio"] = r["max_invalid_bin_count"] / cl
            out["sets"][sname] = acc

    os.makedirs(a.outdir, exist_ok=True)
    with open(f"{a.outdir}/{a.biosample}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for sname, acc in out["sets"].items():
        t0 = next(iter(acc))
        print(f"{a.biosample} {sname}: {acc[t0]['n_windows']} windows, "
              f"{acc[t0]['n_windows_with_invalid']} with an invalid bin, "
              f"{acc[t0]['n_invalid_bins']} invalid bins")


if __name__ == "__main__":
    main()
