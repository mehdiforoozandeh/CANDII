"""t112 — bin a pipeline tagAlign into store-rule 25 bp counts, and a pipeline pval bigwig into 25 bp p.

Two products per counterfactual arm are made here, both as one npz keyed by chromosome name:

**counts** — the store's counting rule, `tools/dnase_bam_rebuild.py` docstring, applied to tagAlign
coordinates. Each line adds 1 to bins `start // 25 … end // 25` **inclusive**, on a grid of
`len // 25 + 1` bins, then the grid is truncated to `len // 25` (`layout.py` D13). A tagAlign is
`bedtools bamtobed` output, so its BED `start`/`end` are pysam's `reference_start`/`reference_end`:
the rule is the same rule, not a re-derivation. The end bin is inclusive on purpose — a read whose
end falls on a bin boundary is credited to one more bin than it strictly overlaps, as the store does.
A bin index past the grid (a read overhanging the chromosome end) is dropped, as
`dnase_bam_rebuild.slow_reference` does. Lines on non-main chromosomes are ignored but counted.

**pval** — the pipeline's `*.pval.signal.bigwig` is `-log10 p` at base resolution. Its intervals are
written to one bedGraph and handed to `tools/dnase_macs2_pval.py::bin_bedgraph` unchanged: the exact
mean over each 25 bp bin. Floats are written with `repr`, so the text round trip is exact.

npz files differ run to run in their zip timestamps: compare arrays, never file md5s.

Deliberately standalone: numpy + pandas, plus pyBigWig (imported lazily) for `pval`. No `candi`
import. Note `dnase_macs2_pval` imports h5py at module level, so the `pval` path also needs h5py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: the store's chromosome set, `dnase_bam_rebuild.MAIN_CHROMS`.
MAIN_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)


def load_chrsz(path) -> dict:
    """A UCSC 2-column `.chrom.sizes` TSV; main chromosomes only, in MAIN_CHROMS order."""
    sizes = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, size = line.split("\t")[:2]
        if name in MAIN_CHROMS:
            sizes[name] = int(size)
    return {c: sizes[c] for c in MAIN_CHROMS if c in sizes}


def counts_from_tagalign(ta_path, sizes: dict, res: int = 25, chunksize: int = 5_000_000,
                         stats: dict | None = None) -> dict:
    """Store-rule counts of a (gzipped) tagAlign; `{chrom: uint32[len // res]}` in `sizes` order.

    Difference array per chromosome: +1 at `start // res`, -1 at `end // res + 1`, one `cumsum`.
    `stats`, if given, receives `n_lines` and `n_lines_main`.
    """
    diff = {c: np.zeros(sizes[c] // res + 2, dtype=np.int64) for c in sizes}
    n_lines = 0
    n_lines_main = 0
    reader = pd.read_csv(ta_path, sep="\t", header=None, usecols=[0, 1, 2],
                         names=["chrom", "start", "end"], chunksize=chunksize,
                         dtype={"chrom": str, "start": np.int64, "end": np.int64})
    for chunk in reader:
        n_lines += len(chunk)
        for chrom, g in chunk.groupby("chrom", sort=False):
            if chrom not in diff:
                continue
            n_lines_main += len(g)
            n_grid = sizes[chrom] // res + 1
            b0 = g["start"].to_numpy() // res
            b1 = g["end"].to_numpy() // res
            keep = b0 < n_grid
            b0 = b0[keep]
            b1 = np.minimum(b1[keep], n_grid - 1)
            d = diff[chrom]
            d += np.bincount(b0, minlength=d.shape[0])
            d -= np.bincount(b1 + 1, minlength=d.shape[0])
    out = {}
    for c in sizes:
        v = np.cumsum(diff[c][: sizes[c] // res + 1])[: sizes[c] // res]
        if v.size and int(v.max()) > np.iinfo(np.uint32).max:
            raise SystemExit(f"{c}: bin count {int(v.max())} overflows uint32")
        out[c] = v.astype(np.uint32)
    if stats is not None:
        stats["n_lines"] = n_lines
        stats["n_lines_main"] = n_lines_main
    return out


def pval_from_bigwig(bw_path, sizes: dict, tmpdir, res: int = 25,
                     stats: dict | None = None) -> dict:
    """Mean-binned `-log10 p` of a bigwig; `{chrom: float32[len // res]}` via `bin_bedgraph`.

    The bedGraph is written to `tmpdir` and removed after binning. `stats`, if given, receives
    `n_lines` (intervals written) and `n_lines_main` (those on a chromosome in `sizes`).
    """
    import pyBigWig

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dnase_macs2_pval import bin_bedgraph

    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    bdg = tmpdir / (Path(bw_path).name + ".bdg")
    n_lines = 0
    n_lines_main = 0
    bw = pyBigWig.open(str(bw_path))
    try:
        with open(bdg, "w") as fo:
            for chrom in bw.chroms():
                ivs = bw.intervals(chrom) or ()
                fo.write("".join(f"{chrom}\t{s}\t{e}\t{v!r}\n" for s, e, v in ivs))
                n_lines += len(ivs)
                if chrom in sizes:
                    n_lines_main += len(ivs)
    finally:
        bw.close()
    try:
        out = bin_bedgraph(bdg, sizes, res)[0]
    finally:
        bdg.unlink()
    if stats is not None:
        stats["n_lines"] = n_lines
        stats["n_lines_main"] = n_lines_main
    return {c: out[c] for c in sizes}


def write_npz(path, arrays: dict) -> None:
    """One npz, keys = chromosome names in MAIN_CHROMS order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{c: arrays[c] for c in MAIN_CHROMS if c in arrays})


def read_npz(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ac = sub.add_parser("counts", help="store-rule counts of a tagAlign")
    ac.add_argument("--ta", required=True, type=Path)
    ap_p = sub.add_parser("pval", help="mean-binned -log10 p of a pval bigwig")
    ap_p.add_argument("--bigwig", required=True, type=Path)
    ap_p.add_argument("--tmpdir", required=True, type=Path)
    for p in (ac, ap_p):
        p.add_argument("--chrsz", required=True, type=Path)
        p.add_argument("--res", type=int, default=25)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--json", required=True, type=Path)
    args = ap.parse_args(argv)

    sizes = load_chrsz(args.chrsz)
    stats = {}
    if args.cmd == "counts":
        arrays = counts_from_tagalign(args.ta, sizes, args.res, stats=stats)
        per = {c: {"n_bins": int(v.shape[0]), "sum": int(v.sum(dtype=np.int64)),
                   "max": int(v.max()) if v.size else 0} for c, v in arrays.items()}
    else:
        arrays = pval_from_bigwig(args.bigwig, sizes, args.tmpdir, args.res, stats=stats)
        per = {c: {"n_bins": int(v.shape[0]),
                   "mean": float(np.nanmean(v)) if v.size else 0.0,
                   "max": float(np.nanmax(v)) if v.size else 0.0,
                   "n_nan": int(np.isnan(v).sum())} for c, v in arrays.items()}
    write_npz(args.out, arrays)
    result = {"n_lines": stats["n_lines"], "n_lines_main": stats["n_lines_main"], "per_chrom": per}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.cmd}: n_lines={stats['n_lines']} n_lines_main={stats['n_lines_main']} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
