"""Does a keep-dup-1 artifact tower actually reach the model's input? (t78)

The two 2026-08-30 censuses (`tower_census.py`) measured towers against the MACS2
`--keep-dup 1` ceiling and noted that the large SHARED towers sit where
`genome/mask.h5` is 0.  That does not close the question, because `mask.h5` is
consumed by exactly one thing — D12 window eligibility (`genome.py::eligible_starts`).
Nothing in the batch path (`dataset.py::_make_batch`) zeroes, drops or excludes bin
VALUES at masked positions, and D12 is
`mask[s : s+L].mean() >= min_valid_frac` with `min_valid_frac = 0.9`, so a 768-bin
window may carry up to 76 invalid bins and still be sampled.

This measures what actually gets ingested: over the window geometry the benchmark
trains on, for every track already censused, how many D12-eligible windows contain a
bin over that track's ceiling, and the largest such value with its coordinate.

Geometry, from `configs/regime.eic_pilot.json` and `plan/BENCHMARK_DESIGN.md` §3:

    context_bins = 768,  window_plan = tile,  stride_bins = 768,  min_valid_frac = 0.9

    eic.19      train chr19                              (no regions BED)
                eval  chr20 chr21 chr22
    eic.pilot   train the 18 chroms of regime.eic_pilot, D32-contained in the
                      44 hg38 Pilot Regions
                eval  chr20 chr21 chr22

D12 and D32 are NOT reimplemented here: `eligible_window_mask` and
`RegionSet.contained_starts` are imported from the repo's own `candi.store`, so this
measures the rule the loader applies rather than a copy of it.

The keep-dup-1 ceiling is recomputed by the same arithmetic `tower_census.py` uses
(`2 * (L_eff + RES - 1)`, `L_eff = RES * total_counts/depth - (RES-1)`), off this
script's own genome-wide totals, and the caller cross-checks it against the censuses.

READ-ONLY.  Opens counts.h5 / mask.h5 with mode "r" and writes only to --outdir.
Nothing is de-duplicated, clipped or thresholded: this is a measurement.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

RES = 25
CONTROL = "chipseq-control"
CONTEXT_BINS = 768
STRIDE = 768
MIN_VALID_FRAC = 0.9
HIST_CAP = 200_000          # median-of-nonzero histogram; values above are counted only
TOP_WINDOWS = 10            # ingested windows reported with coordinates
XMULT = (1, 10, 100, 1000)

PILOT_TRAIN_CHROMS = [
    "chr1", "chr2", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11",
    "chr12", "chr13", "chr14", "chr15", "chr16", "chr18", "chr19", "chrX",
]
EVAL_CHROMS = ["chr20", "chr21", "chr22"]
PILOT_BED_SHA = "13e11a198fdee08edb7797d1e402b5d985846b5a7d973ade91e8511462acb7a3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--store", default="/project/def-maxwl/mforooz/CANDI_STORE/eic")
    ap.add_argument("--genome", default="/project/def-maxwl/mforooz/CANDI_STORE/genome")
    ap.add_argument("--pilot-bed", required=True,
                    help="configs/regions/encode_pilot_hg38.bed, hash-checked against D32")
    ap.add_argument("--src", required=True, help="the CANDII src/ to import candi.store from")
    ap.add_argument("--tracks", choices=("signal", "control", "all"), default="all")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    from candi.store.genome import eligible_window_mask
    from candi.store.regime import RegionSet

    pilot = RegionSet.from_obj({"bed": os.path.abspath(a.pilot_bed), "sha256": PILOT_BED_SHA})

    with open(f"{a.store}/manifest.json") as fh:
        man = json.load(fh)
    entry = man["biosamples"][a.biosample]
    meta = {t["assay"]: t for t in entry["tracks"]}

    want = {"signal": lambda nm: nm != CONTROL,
            "control": lambda nm: nm == CONTROL,
            "all": lambda nm: True}[a.tracks]

    # -- pass 1: genome-wide totals (the ceiling) + the nonzero histogram -----------------
    path = f"{a.store}/biosamples/{a.biosample}/counts.h5"
    with h5py.File(path, "r") as fc:
        tracks = json.loads(fc.attrs["tracks"])
        chroms = [c for c in entry["chroms"] if c in fc]
        cols = [(i, nm) for i, nm in enumerate(tracks) if want(nm)]
        if not cols:
            print(f"{a.biosample}: no {a.tracks} tracks, nothing to do")
            return

        totals = np.zeros(len(tracks), dtype=np.int64)
        hist = {col: np.zeros(HIST_CAP + 1, dtype=np.int64) for col, _ in cols}
        n_over_cap = {col: 0 for col, _ in cols}
        gmax = {col: (0, "", -1) for col, _ in cols}          # (value, chrom, bin)
        tile_max = {col: {} for col, _ in cols}               # chrom -> (n_tiles,) int64
        tile_arg = {col: {} for col, _ in cols}               # chrom -> (n_tiles,) offset in tile
        n_bins_total = 0
        n_bins_chrom = {}

        for c in chroms:
            blk = fc[c][:]
            n = blk.shape[0]
            n_bins_chrom[c] = n
            n_bins_total += n
            n_tiles = n // CONTEXT_BINS
            for col, _ in cols:
                v = blk[:, col].astype(np.int64)
                totals[col] += int(v.sum())
                nz = v[v > 0]
                if nz.size:
                    over = nz > HIST_CAP
                    n_over_cap[col] += int(over.sum())
                    if over.any():
                        nz = nz[~over]
                    if nz.size:
                        hist[col] += np.bincount(nz, minlength=HIST_CAP + 1)[: HIST_CAP + 1]
                    mi = int(np.argmax(v))
                    if int(v[mi]) > gmax[col][0]:
                        gmax[col] = (int(v[mi]), c, mi)
                if n_tiles:
                    w = v[: n_tiles * CONTEXT_BINS].reshape(n_tiles, CONTEXT_BINS)
                    tile_max[col][c] = w.max(axis=1)
                    tile_arg[col][c] = w.argmax(axis=1)
                else:
                    tile_max[col][c] = np.zeros(0, dtype=np.int64)
                    tile_arg[col][c] = np.zeros(0, dtype=np.int64)
                del v
            del blk

    # -- the D12 eligible tiles, per chromosome (mask.h5, the rule imported) --------------
    eligible = {}
    with h5py.File(f"{a.genome}/mask.h5", "r") as fm:
        for c in chroms:
            n = n_bins_chrom[c]
            m = fm[c][:n]
            if m.shape[0] != n:
                raise SystemExit(f"{c}: mask has {m.shape[0]} bins but counts has {n}")
            ok = eligible_window_mask(m, CONTEXT_BINS, MIN_VALID_FRAC)
            e = ok[::STRIDE] if ok.size else np.zeros(0, dtype=bool)
            n_tiles = n // CONTEXT_BINS
            if e.shape[0] != n_tiles:
                raise SystemExit(f"{c}: {e.shape[0]} tile flags but {n_tiles} tiles")
            eligible[c] = e

    # -- the two regimes' window sets ----------------------------------------------------
    def window_set(chrom_list, regions):
        """`{chrom: bool over tiles}` — D12, then D32 containment when `regions` is given."""
        out = {}
        for c in chrom_list:
            if c not in eligible:
                continue
            e = eligible[c].copy()
            if regions is not None:
                starts = np.flatnonzero(e).astype(np.int64) * STRIDE
                kept = regions.contained_starts(c, starts, CONTEXT_BINS, RES)
                e2 = np.zeros_like(e)
                if kept.size:
                    e2[(kept // STRIDE).astype(np.int64)] = True
                e = e2
            out[c] = e
        return out

    win = {
        "eic.19": {"train": window_set(["chr19"], None),
                   "eval": window_set(EVAL_CHROMS, None)},
        "eic.pilot": {"train": window_set(PILOT_TRAIN_CHROMS, pilot),
                      "eval": window_set(EVAL_CHROMS, None)},
    }

    out = {"biosample": a.biosample, "split": a.biosample.split("_")[0],
           "context_bins": CONTEXT_BINS, "stride_bins": STRIDE,
           "min_valid_frac": MIN_VALID_FRAC, "n_bins_genome": n_bins_total,
           "chroms": chroms, "counts_dtype": entry["dtype"],
           "pilot_bed": pilot.resolved, "pilot_bed_sha256": pilot.sha256,
           "tracks": {}}

    for col, name in cols:
        m = meta[name]
        depth = float(m["depth"])
        total = float(totals[col])
        bpr = total / depth
        L_eff = RES * bpr - (RES - 1)
        ceiling = max(1, int(round(2 * (L_eff + RES - 1))))

        h = hist[col]
        n_nonzero = int(h[1:].sum()) + n_over_cap[col]
        med_nz = 0.0
        if n_nonzero:
            cum = np.cumsum(h[1:])
            half = (n_nonzero + 1) // 2
            idx = int(np.searchsorted(cum, half)) + 1
            med_nz = float(idx) if idx <= HIST_CAP else float("nan")

        gv, gc, gb = gmax[col]
        rec = dict(
            assay=name,
            assay_class=("DNase-seq" if name == "DNase-seq"
                         else "ATAC-seq" if name == "ATAC-seq"
                         else "chipseq-control" if name == CONTROL else "ChIP-seq"),
            col=col, depth=depth, run_type=m.get("run_type"),
            read_length=m.get("read_length"),
            file_accession=m.get("file_accession"), exp_accession=m.get("exp_accession"),
            total_counts=total, bins_per_read=bpr, L_eff_bp=L_eff,
            keepdup1_ceiling=ceiling,
            median_nonzero_bin=med_nz, n_nonzero_bins=n_nonzero,
            global_max_count=gv,
            global_max_locus=(f"{gc}:{gb * RES}" if gb >= 0 else "-"),
            global_max_chrom=gc, global_max_bin=gb,
            regimes={},
        )

        for reg in ("eic.19", "eic.pilot"):
            rec["regimes"][reg] = {}
            for split in ("train", "eval"):
                ws = win[reg][split]
                n_elig = 0
                n_hit = {x: 0 for x in XMULT}
                best = (0, "", -1)
                tops = []
                has_top1 = 0
                for c, e in ws.items():
                    tm = tile_max[col].get(c)
                    if tm is None or tm.size == 0:
                        continue
                    n_elig += int(e.sum())
                    vals = tm[e]
                    if vals.size == 0:
                        continue
                    tidx = np.flatnonzero(e)
                    for x in XMULT:
                        n_hit[x] += int((vals > ceiling * x).sum())
                    k = int(np.argmax(vals))
                    if int(vals[k]) > best[0]:
                        t = int(tidx[k])
                        b = t * STRIDE + int(tile_arg[col][c][t])
                        best = (int(vals[k]), c, b)
                    sel = np.flatnonzero(vals > ceiling)
                    if sel.size:
                        order = sel[np.argsort(vals[sel])[::-1]][:TOP_WINDOWS]
                        for j in order:
                            t = int(tidx[j])
                            tops.append((int(vals[j]), c, t * STRIDE,
                                         t * STRIDE + int(tile_arg[col][c][t])))
                    if gc == c and gb >= 0:
                        t = gb // STRIDE
                        if t < e.shape[0] and bool(e[t]):
                            has_top1 = 1
                tops.sort(key=lambda r: -r[0])
                bv, bc, bb = best
                rec["regimes"][reg][split] = dict(
                    n_eligible_windows=n_elig,
                    n_windows_over_ceiling=n_hit[1],
                    frac_windows_over_ceiling=(n_hit[1] / n_elig) if n_elig else 0.0,
                    n_windows_over_10x=n_hit[10],
                    n_windows_over_100x=n_hit[100],
                    n_windows_over_1000x=n_hit[1000],
                    max_ingested_count=bv,
                    max_ingested_locus=(f"{bc}:{bb * RES}" if bb >= 0 else "-"),
                    max_ingested_chrom=bc, max_ingested_bin=bb,
                    max_ingested_window_start=(bb // STRIDE) * STRIDE if bb >= 0 else -1,
                    max_over_ceiling_ratio=(bv / ceiling) if ceiling else 0.0,
                    max_over_median_nonzero=(bv / med_nz) if med_nz else float("nan"),
                    n_windows_with_track_top1=has_top1,
                    top_windows=[dict(chrom=c, window_start_bin=s, window_start_bp=s * RES,
                                      max_bin=b, max_bin_bp=b * RES, count=v,
                                      over_ceiling_ratio=v / ceiling)
                                 for v, c, s, b in tops[:TOP_WINDOWS]],
                )
        out["tracks"][name] = rec

    os.makedirs(a.outdir, exist_ok=True)
    with open(f"{a.outdir}/{a.biosample}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{a.biosample}: {len(cols)} tracks, {n_bins_total} bins")
    for col, name in cols:
        r = out["tracks"][name]
        for reg in ("eic.19", "eic.pilot"):
            for split in ("train", "eval"):
                s = r["regimes"][reg][split]
                print("  %-18s %-10s %-6s elig=%-7d over=%-6d max=%-12d x%.1f"
                      % (name, reg, split, s["n_eligible_windows"],
                         s["n_windows_over_ceiling"], s["max_ingested_count"],
                         s["max_over_ceiling_ratio"]))


if __name__ == "__main__":
    main()
