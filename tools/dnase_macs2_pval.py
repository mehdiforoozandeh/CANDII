"""t78 G1 fallback — turn MACS2's base-resolution `_ppois.bdg` into the store's 25 bp `pval` layer.

`plan/BENCHMARK_DESIGN.md` §10 named the fallback: "re-downloading the 40 DNase BAMs to scratch and
running MACS2 properly", where *properly* means at base resolution with the read shift and
extension that a 25 bp pre-binned pileup cannot express. This file is the second half of that — it
does not run MACS2, it consumes what MACS2 wrote, so the exact `macs2` command stays visible in the
job script rather than being buried in Python.

The upstream commands are ENCODE's own ATAC signal-track step, which §10 chose as the template
(`ENCODE-DCC/atac-seq-pipeline/src/encode_task_macs2_signal_track_atac.py`, `--smooth-win 150`,
`--pval-thresh 0.01`, so `shiftsize = -75`):

    macs2 callpeak -t <tagAlign.bed> -f BED -n P -g hs -p 0.01 \
        --shift -75 --extsize 150 --nomodel -B --SPMR --keep-dup all
    macs2 bdgcmp -t P_treat_pileup.bdg -c P_control_lambda.bdg --o-prefix P -m ppois -S <sval>

`--SPMR` writes both bedGraphs per million tags; `-S sval` with `sval = n_tags / 1e6` puts them
back into raw tag units, so the Poisson test is on raw pileup against raw local lambda. Without a
control, `P_control_lambda.bdg` is `max(lambda_bg, llocal)` and nothing else — `slocal` and the
d-size window are not calculated. That is the lambda rule `src/candi/store/pval_from_counts.py`
documents from the MACS2 source, and it is **not** re-derived here.

**Binning.** `_ppois.bdg` is `-log10 P(X > k)` at base resolution. The store's signal layers are
mean-binned at 25 bp (`DATA.md`: `signal_BW_res25` is "mean-binned and untransformed on disk"), so
this takes the mean of the base-resolution score over each 25 bp bin, exactly, from the interval
endpoints — no bigWig round trip. `n_bins = floor(chrom_len / 25)`, `layout.py` D13.

**Codec.** `uint16` of `round(arcsinh(x) * 2000)`, byte-identical to `layout.encode_pval` at
`transform="arcsinh"`. Reproduced rather than imported so this file keeps its no-`candi` promise.

Deliberately standalone: numpy + pandas + h5py, plus `pval_from_counts.py` only for the constants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PVAL_SCALE = 2000
PVAL_TRANSFORM = "arcsinh"
PVAL_UINT16_MAX = 65535
MAIN_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)


def encode_pval(values: np.ndarray, scale: int = PVAL_SCALE) -> tuple:
    """`-log10 p` -> `uint16`; `(encoded, n_clipped)`. `layout.encode_pval`, `nan_policy="error"`."""
    x = np.asarray(values, dtype=np.float64)
    n_nan = int(np.isnan(x).sum())
    if n_nan:
        raise SystemExit(f"{n_nan} NaN in a computed pval track; refusing to encode")
    scaled = np.rint(np.arcsinh(x) * scale)
    n_clipped = int(np.count_nonzero((scaled < 0) | (scaled > PVAL_UINT16_MAX)))
    return np.clip(scaled, 0, PVAL_UINT16_MAX).astype(np.uint16), n_clipped


def bin_bedgraph(path: Path, sizes: dict, resolution: int = 25,
                 chunksize: int = 20_000_000) -> dict:
    """Exact mean of a bedGraph over `floor(len / resolution)` bins per chromosome.

    Each interval `[s, e)` with value `v` contributes `v * overlap` to every bin it touches. The
    bins it covers WHOLLY are handled with a difference array so a 100 kb interval costs O(1);
    the at most two partial bins at either end are scattered with `np.add.at`. The divisor is the
    full `resolution` everywhere, so a bin the bedGraph does not cover reads 0 — which is what
    MACS2 means there, since `_ppois.bdg` covers every base of every chromosome it reports.
    """
    acc = {c: np.zeros(sizes[c] // resolution, dtype=np.float64) for c in sizes}
    dif = {c: np.zeros(sizes[c] // resolution + 2, dtype=np.float64) for c in sizes}
    seen = {c: 0 for c in sizes}
    reader = pd.read_csv(path, sep="\t", header=None, chunksize=chunksize,
                         names=["chrom", "start", "end", "value"],
                         dtype={"chrom": str, "start": np.int64, "end": np.int64,
                                "value": np.float64})
    for chunk in reader:
        for chrom, g in chunk.groupby("chrom", sort=False):
            if chrom not in acc:
                continue
            nb = acc[chrom].shape[0]
            s = g["start"].to_numpy()
            e = np.minimum(g["end"].to_numpy(), nb * resolution)
            v = g["value"].to_numpy()
            keep = e > s
            s, e, v = s[keep], e[keep], v[keep]
            if s.size == 0:
                continue
            seen[chrom] += int((e - s).sum())
            b0 = s // resolution
            b1 = (e - 1) // resolution
            same = b0 == b1
            # intervals inside one bin
            np.add.at(acc[chrom], b0[same], v[same] * (e[same] - s[same]))
            # intervals spanning >= 2 bins: head, tail, and the whole bins between
            m = ~same
            if m.any():
                b0m, b1m, sm, em, vm = b0[m], b1[m], s[m], e[m], v[m]
                np.add.at(acc[chrom], b0m, vm * ((b0m + 1) * resolution - sm))
                np.add.at(acc[chrom], b1m, vm * (em - b1m * resolution))
                full = b1m > b0m + 1
                if full.any():
                    np.add.at(dif[chrom], b0m[full] + 1, vm[full])
                    np.add.at(dif[chrom], b1m[full], -vm[full])
    out = {}
    for c in sizes:
        nb = acc[c].shape[0]
        acc[c] += np.cumsum(dif[c][:nb]) * resolution
        out[c] = (acc[c] / resolution).astype(np.float32)
    return out, seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bdg", required=True, type=Path, help="MACS2 `<prefix>_ppois.bdg`")
    ap.add_argument("--chrom-sizes", required=True, type=Path,
                    help="the store's genome/chrom_sizes.json, or a UCSC .chrom.sizes")
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--assay", default="DNase-seq")
    ap.add_argument("--resolution", type=int, default=25)
    ap.add_argument("--out", required=True, type=Path, help="the pval h5")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--macs2-version", default=None)
    ap.add_argument("--macs2-cmd", default=None, help="the exact command, recorded into attrs")
    ap.add_argument("--code-sha", default=None)
    args = ap.parse_args()

    p = Path(args.chrom_sizes)
    if p.suffix == ".json":
        raw = json.loads(p.read_text())["chrom_sizes"]
        sizes = {k: int(v) for k, v in raw.items() if k in MAIN_CHROMS}
    else:
        sizes = {}
        for line in p.read_text().splitlines():
            if line.strip():
                name, size = line.split("\t")[:2]
                if name in MAIN_CHROMS:
                    sizes[name] = int(size)
    sizes = {c: sizes[c] for c in MAIN_CHROMS if c in sizes}

    binned, seen = bin_bedgraph(args.bdg, sizes, args.resolution)

    n_clipped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as fo:
        for c in sizes:
            enc, nc = encode_pval(binned[c])
            n_clipped += nc
            fo.create_dataset(c, data=enc, chunks=(min(1024, enc.shape[0]),),
                              compression="gzip", compression_opts=4)
        fo.attrs["kind"] = "pval"
        fo.attrs["biosample"] = args.biosample
        fo.attrs["assay"] = args.assay
        fo.attrs["resolution"] = args.resolution
        fo.attrs["scale"] = PVAL_SCALE
        fo.attrs["transform"] = PVAL_TRANSFORM
        fo.attrs["method"] = "macs2_nocontrol_ppois_base_resolution"
        fo.attrs["source"] = str(args.bdg)
        if args.macs2_version:
            fo.attrs["macs2_version"] = args.macs2_version
        if args.macs2_cmd:
            fo.attrs["macs2_cmd"] = args.macs2_cmd
        if args.code_sha:
            fo.attrs["code_sha"] = args.code_sha

    result = {
        "biosample": args.biosample,
        "assay": args.assay,
        "bdg": str(args.bdg),
        "resolution": args.resolution,
        "macs2_version": args.macs2_version,
        "macs2_cmd": args.macs2_cmd,
        "codec": {"scale": PVAL_SCALE, "transform": PVAL_TRANSFORM, "n_clipped": n_clipped},
        "per_chrom": {c: {"n_bins": int(binned[c].shape[0]),
                          "bases_covered": seen[c],
                          "frac_bases_covered": seen[c] / sizes[c],
                          "max": float(binned[c].max()),
                          "mean": float(binned[c].mean())} for c in sizes},
        "global_max": float(max(float(v.max()) for v in binned.values())),
        "out": str(args.out),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.biosample} max_-log10p={result['global_max']:.4g} n_clipped={n_clipped} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
