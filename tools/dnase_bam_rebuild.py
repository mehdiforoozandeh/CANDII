"""t78 G1 fallback — rebuild one experiment's 25 bp `counts` from a BAM, with duplicates removed.

`plan/BENCHMARK_DESIGN.md` §10 ruling (2026-08-30): ENCODE's DNase pipeline *marks* duplicates and
keeps them, ENCODE's ATAC pipeline *removes* them, and §10 chose ATAC as the template. The store's
`counts` were built straight off the released DNase BAM, so they carry the duplicate mass —
`T_K562` holds 9.73 M untrimmed adapter read-throughs in two adjacent 25 bp bins. Deduplication
needs read start positions, which binned counts do not carry, so the repair starts from the BAM.

**The counting rule is reproduced verbatim from the code that built the store**, EpiDenoise
`data_utils.py::BAM_TO_SIGNAL.calculate_coverage_pysam` at `downsampling_factor=1`:

    bins[chrom] = [0] * (chrom_len // 25 + 1)
    for read in bam.fetch(chrom):            # coordinate-sorted, mapped reads only
        if read.is_unmapped: continue
        for i in range(read.reference_start // 25, read.reference_end // 25 + 1):
            bins[chrom][i] += 1
    depth = number of reads counted

Two things about that rule that are easy to get wrong and are kept on purpose: the grid is
`len // 25 + 1` bins (the store later truncates to `floor(len / 25)` — `layout.py` D13), and the
end bin is `reference_end // 25` **inclusive**, so a read is credited to one more bin than it
strictly overlaps when its end falls on a bin boundary. Both are the store's behaviour today.

The only change is `--drop-duplicates`, which skips a read whose `0x400` flag Picard already set.
That is the single named difference between the two ENCODE pipelines; nothing else moves. Run it
without the flag and this file reproduces the store's existing `counts` bit for bit, which is how
the rebuild is checked (`--compare-store`).

The inner loop is a difference array plus one `cumsum`, not a Python `for i in range(...)`. That is
an implementation detail with no effect on the result, and `--slow-reference` runs the literal loop
on one chromosome so the two can be compared.

Deliberately standalone: pysam + numpy + h5py, no `candi` import, so it runs beside a store on a
cluster whose checkout of this repo is stale.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pysam

#: EpiDenoise's `read_chr_sizes` keeps exactly these and nothing else.
MAIN_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)

#: ceiling statistics — reproduced from `tower_census.py`, itself verbatim from `diag_keepdup.py`.
XMULT = (1, 10, 100, 1000)


def load_chrom_sizes(path: Path) -> dict:
    """A UCSC `.chrom.sizes` TSV, or the store's `genome/chrom_sizes.json`. Main chromosomes only."""
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text())["chrom_sizes"]
        return {k: int(v) for k, v in raw.items() if k in MAIN_CHROMS}
    sizes = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        name, size = line.split("\t")[:2]
        if name in MAIN_CHROMS:
            sizes[name] = int(size)
    return sizes


def count_chrom(bam: pysam.AlignmentFile, chrom: str, n_grid: int, resolution: int,
                drop_duplicates: bool) -> tuple:
    """One chromosome's pileup, as a difference array. Returns `(counts, n_reads, n_dropped)`."""
    diff = np.zeros(n_grid + 2, dtype=np.int64)
    n_reads = 0
    n_dropped = 0
    add = np.add.at
    starts = np.empty(1 << 20, dtype=np.int64)
    ends = np.empty(1 << 20, dtype=np.int64)
    k = 0
    for read in bam.fetch(chrom):
        if read.is_unmapped:
            continue
        if drop_duplicates and read.is_duplicate:
            n_dropped += 1
            continue
        end = read.reference_end
        if end is None:            # no cigar / no alignment length; the store's loop would raise
            continue
        starts[k] = read.reference_start // resolution
        ends[k] = end // resolution
        k += 1
        n_reads += 1
        if k == starts.shape[0]:
            add(diff, starts, 1)
            add(diff, ends + 1, -1)
            k = 0
    if k:
        add(diff, starts[:k], 1)
        add(diff, ends[:k] + 1, -1)
    counts = np.cumsum(diff[:n_grid])
    return counts, n_reads, n_dropped


def slow_reference(bam_path: Path, chrom: str, n_grid: int, resolution: int,
                   drop_duplicates: bool) -> np.ndarray:
    """EpiDenoise's literal Python loop, for one chromosome. The second method for `count_chrom`."""
    bins = [0] * n_grid
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(chrom):
            if read.is_unmapped:
                continue
            if drop_duplicates and read.is_duplicate:
                continue
            if read.reference_end is None:
                continue
            for i in range(read.reference_start // resolution,
                           read.reference_end // resolution + 1):
                if i < n_grid:
                    bins[i] += 1
    return np.asarray(bins, dtype=np.int64)


def ceiling_stats(counts: dict, chroms: list, depth: float, resolution: int) -> dict:
    """The keep-dup-1 ceiling block, field-for-field as `tower_census.py` writes it."""
    total = float(sum(int(v.sum()) for v in counts.values()))
    n_bins_total = int(sum(v.shape[0] for v in counts.values()))
    bpr = total / depth if depth else 0.0
    l_eff = resolution * bpr - (resolution - 1)
    ceiling = max(1, int(round(2 * (l_eff + resolution - 1))))
    n_over = sum(int((v > ceiling).sum()) for v in counts.values())
    excess = sum(float(np.maximum(v.astype(np.int64) - ceiling, 0).sum()) for v in counts.values())
    mass_in_over = sum(float(v[v > ceiling].sum()) for v in counts.values())
    gmax = max(int(v.max()) for v in counts.values())
    arg = max(chroms, key=lambda c: int(counts[c].max()))
    pos = int(np.argmax(counts[arg]))
    pooled = np.concatenate([np.sort(v)[-10:] for v in counts.values()])
    pooled = np.sort(pooled)[::-1]
    rec = {
        "depth": depth,
        "total_counts": total,
        "n_bins_total": n_bins_total,
        "bins_per_read": bpr,
        "L_eff_bp": l_eff,
        "keepdup1_ceiling": ceiling,
        "global_max_count": gmax,
        "global_max_locus": f"{arg}:{pos * resolution}",
        "max_over_ceiling_ratio": gmax / ceiling,
        "n_bins_over_ceiling": n_over,
        "frac_bins_over_ceiling": n_over / n_bins_total if n_bins_total else 0.0,
        "excess_mass_over_ceiling": excess,
        "frac_mass_over_ceiling": excess / total if total else 0.0,
        "mass_in_bins_over_ceiling": mass_in_over,
        "frac_mass_in_bins_over_ceiling": mass_in_over / total if total else 0.0,
    }
    for x in XMULT:
        rec[f"n_bins_over_{x}x_ceiling"] = sum(int((v > ceiling * x).sum()) for v in counts.values())
    for k in (1, 2, 10):
        rec[f"frac_mass_top{k}"] = float(pooled[:k].sum()) / total if total else 0.0
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bam", required=True, type=Path)
    ap.add_argument("--chrom-sizes", required=True, type=Path)
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--assay", default="DNase-seq")
    ap.add_argument("--resolution", type=int, default=25)
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="skip reads Picard already flagged 0x400 — the ATAC-template rule")
    ap.add_argument("--out", required=True, type=Path, help="h5, one 1-D dataset per chromosome")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--compare-store", type=Path, default=None,
                    help="a store biosample dir; its counts.h5 column for --assay is compared")
    ap.add_argument("--slow-reference", default=None,
                    help="also run EpiDenoise's literal Python loop on this chromosome")
    ap.add_argument("--chroms", default=None, help="comma list; default every main chromosome")
    args = ap.parse_args()

    res = args.resolution
    sizes = load_chrom_sizes(args.chrom_sizes)
    chroms = args.chroms.split(",") if args.chroms else [c for c in MAIN_CHROMS if c in sizes]

    t0 = time.time()
    counts_grid = {}
    n_reads = 0
    n_dropped = 0
    with pysam.AlignmentFile(str(args.bam), "rb") as bam:
        for c in chroms:
            n_grid = sizes[c] // res + 1
            v, nr, nd = count_chrom(bam, c, n_grid, res, args.drop_duplicates)
            counts_grid[c] = v
            n_reads += nr
            n_dropped += nd
            print(f"  {c:6s} grid={n_grid:9d} reads={nr:11d} dropped_dup={nd:11d} "
                  f"max={int(v.max()):9d}", flush=True)
    elapsed = time.time() - t0

    # EpiDenoise's coverage is over the `len//25 + 1` grid, before the store's floor truncation.
    bins_with_reads = int(sum(int((v > 0).sum()) for v in counts_grid.values()))
    grid_bins = int(sum(v.shape[0] for v in counts_grid.values()))
    coverage = bins_with_reads / grid_bins if grid_bins else 0.0

    # the store's grid: floor(len / resolution) — layout.py D13
    counts = {c: counts_grid[c][: sizes[c] // res] for c in chroms}

    result = {
        "biosample": args.biosample,
        "assay": args.assay,
        "bam": str(args.bam),
        "resolution": res,
        "drop_duplicates": bool(args.drop_duplicates),
        "chroms": chroms,
        "depth": n_reads,
        "n_reads_dropped_duplicate": n_dropped,
        "coverage": coverage,
        "seconds": elapsed,
        "rule": "EpiDenoise BAM_TO_SIGNAL.calculate_coverage_pysam, dsf=1, "
                "bins floor(start/res) .. floor(end/res) inclusive",
        "per_chrom": {c: {"n_bins": int(counts[c].shape[0]),
                          "counts_sum": int(counts[c].sum()),
                          "max": int(counts[c].max())} for c in chroms},
    }
    result["ceiling"] = ceiling_stats(counts, chroms, float(n_reads), res)

    if args.slow_reference:
        c = args.slow_reference
        ref = slow_reference(args.bam, c, sizes[c] // res + 1, res, args.drop_duplicates)
        same = bool(np.array_equal(ref[: counts[c].shape[0]], counts[c]))
        result["slow_reference"] = {"chrom": c, "identical": same,
                                    "n_mismatch": int((ref[: counts[c].shape[0]] != counts[c]).sum())}
        print(f"slow_reference {c}: identical={same}")

    if args.compare_store is not None:
        with h5py.File(args.compare_store / "counts.h5", "r") as f:
            tracks = list(json.loads(f.attrs["tracks"]) if isinstance(f.attrs["tracks"], str)
                          else f.attrs["tracks"])
            col = tracks.index(args.assay)
            cmp = {}
            for c in chroms:
                if c not in f:
                    continue
                old = f[c][:, col].astype(np.int64)
                new = counts[c][: old.shape[0]]
                cmp[c] = {"identical": bool(np.array_equal(old, new)),
                          "n_mismatch": int((old != new).sum()),
                          "old_sum": int(old.sum()), "new_sum": int(new.sum()),
                          "old_max": int(old.max()), "new_max": int(new.max())}
            result["compare_store"] = {
                "n_chroms": len(cmp),
                "n_identical": sum(1 for v in cmp.values() if v["identical"]),
                "old_total": sum(v["old_sum"] for v in cmp.values()),
                "new_total": sum(v["new_sum"] for v in cmp.values()),
                "per_chrom": cmp,
            }

    gmax = result["ceiling"]["global_max_count"]
    dtype = np.uint16 if gmax <= 65535 else np.uint32
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as fo:
        for c in chroms:
            fo.create_dataset(c, data=counts[c].astype(dtype),
                              chunks=(min(1024, counts[c].shape[0]),), compression="gzip",
                              compression_opts=4)
        fo.attrs["kind"] = "counts"
        fo.attrs["biosample"] = args.biosample
        fo.attrs["assay"] = args.assay
        fo.attrs["resolution"] = res
        fo.attrs["dsf"] = 1
        fo.attrs["depth"] = n_reads
        fo.attrs["coverage"] = coverage
        fo.attrs["dtype"] = "uint16" if dtype is np.uint16 else "uint32"
        fo.attrs["source_bam"] = str(args.bam)
        fo.attrs["duplicates_removed"] = bool(args.drop_duplicates)
    result["out"] = {"path": str(args.out), "dtype": str(np.dtype(dtype))}

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + "\n")
    cl = result["ceiling"]
    print(f"{args.biosample} depth={n_reads} dropped={n_dropped} total={cl['total_counts']:.0f} "
          f"ceiling={cl['keepdup1_ceiling']} max={cl['global_max_count']} "
          f"ratio={cl['max_over_ceiling_ratio']:.1f} "
          f"frac_mass_over={cl['frac_mass_over_ceiling']:.4f} -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
