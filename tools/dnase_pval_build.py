"""G1 Phase 2 — build one DNase experiment's `-log10 p` layer from the store's counts, and score
it against the two tracks we can compare it to.

Same `pval_from_counts` as the ATAC gate, same parameters, same statistics — the four the gate was
written against are imported from `atac_pval_gate` rather than rewritten, so a Phase 2 number and a
Phase 1 number are the same function of the same inputs by construction.

Three comparisons, all genome-wide over every chromosome the store has:

* `ours_vs_encode_rdns` — our computed p-value against what the store's `pval.h5` currently holds
  for DNase, which is ENCODE's `read-depth normalized signal`. This is a UNITS MISMATCH by
  construction and is reported to document the defect, not as an agreement statistic.
* `ours_vs_challenge` — our computed p-value against the ENCODE Imputation Challenge's own DNase
  `-log10 p` track. **The second independent check.** `T_` experiments only.
* `encode_rdns_vs_challenge` — the same challenge track against what the store holds today. The
  before-picture for the line above, on identical bins.

**Rule 1 is enforced here, in code.** `--challenge` is refused for any biosample that is not `T_`.

The challenge bigWig is binned exactly the way the challenge's own `bw_to_npy.py` does it — bins =
`ceil(chrom_len / 25)`, NaN -> 0, the final partial window recomputed with an exact `stats()` call
— and then truncated to the store's grid, which is `floor(chrom_len / 25)` and so is at most one
bin shorter per chromosome.

Deliberately standalone: h5py + numpy + scipy + pyBigWig, with `pval_from_counts.py` and
`atac_pval_gate.py` beside it on PYTHONPATH. It imports nothing from `candi`, so it runs against a
store on a cluster whose checkout of this repo is stale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from atac_pval_gate import _attrs, _decode_pval, _stats
from pval_from_counts import MACS2_GSIZE_HS, MACS2_LLOCAL, genome_lambda_bg, pval_from_counts

#: `src/candi/store/layout.py` — the store's pval codec, reproduced rather than imported so this
#: file keeps its no-`candi` promise. `round(arcsinh(x) * scale)` clipped into uint16.
PVAL_SCALE = 2000
PVAL_TRANSFORM = "arcsinh"
PVAL_UINT16_MAX = 65535


def encode_pval(values: np.ndarray, scale: int = PVAL_SCALE) -> tuple:
    """`-log10 p` -> `uint16`. Returns `(encoded, n_clipped)`; byte-identical to `layout.encode_pval`
    at `transform="arcsinh"`, `nan_policy="error"` — which raises rather than inventing a number."""
    x = np.asarray(values, dtype=np.float64)
    n_nan = int(np.isnan(x).sum())
    if n_nan:
        raise SystemExit(f"{n_nan} NaN in a computed pval track; refusing to encode")
    scaled = np.rint(np.arcsinh(x) * scale)
    n_clipped = int(np.count_nonzero((scaled < 0) | (scaled > PVAL_UINT16_MAX)))
    return np.clip(scaled, 0, PVAL_UINT16_MAX).astype(np.uint16), n_clipped


def bin_challenge_bigwig(path: Path, chroms: list, resolution: int) -> dict:
    """One challenge bigWig -> 25 bp bins, following the challenge's `bw_to_npy.py`.

    `n = ceil(chrom_len / resolution)`, the tail zero-padded, NaN -> 0, and the bin holding the last
    interval overwritten with an exact `stats()` call — the challenge's own fix for the partial
    final window. Returns float32 per chromosome, still on the CHALLENGE grid; the caller truncates.
    """
    import pyBigWig

    bw = pyBigWig.open(str(path))
    try:
        sizes = bw.chroms()
        out = {}
        for c in chroms:
            clen = int(sizes[c])
            n = (clen - 1) // resolution + 1
            raw = bw.values(c, 0, clen, numpy=True)
            buf = np.zeros(n * resolution, dtype=np.float64)
            buf[: raw.shape[0]] = np.nan_to_num(raw)
            del raw
            y = buf.reshape(-1, resolution).mean(axis=1)
            del buf
            # The end of the LAST interval, found from the tail rather than by listing every
            # interval on the chromosome — chr1 carries ~10 M of them and the list of tuples
            # costs over a GB. Doubling window, so a track that stops early is still found.
            last_end = None
            span = 1 << 20
            while last_end is None and span <= 2 * clen:
                ivs = bw.intervals(c, max(0, clen - span), clen)
                if ivs:
                    last_end = ivs[-1][1]
                span *= 4
            if last_end is not None:
                last = last_end // resolution
                if 0 <= last < n:
                    stat = bw.stats(c, last * resolution,
                                    min((last + 1) * resolution, clen), exact=True)
                    y[last] = 0.0 if stat[0] is None else stat[0]
            out[c] = y.astype(np.float32)
    finally:
        bw.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--assay", default="DNase-seq")
    ap.add_argument("--genome", type=Path, default=None, help="default: sibling genome/ dir")
    ap.add_argument("--out", required=True, type=Path, help="the JSON of statistics")
    ap.add_argument("--layer-out", type=Path, default=None,
                    help="write the built layer here as uint16 h5, one dataset per chromosome")
    ap.add_argument("--challenge", type=Path, default=None,
                    help="the challenge's DNase bigWig for this experiment; T_ biosamples only")
    ap.add_argument("--llocal", type=int, default=MACS2_LLOCAL)
    ap.add_argument("--gsize", type=float, default=MACS2_GSIZE_HS)
    ap.add_argument("--chroms", default=None, help="comma list; omit for every chromosome stored")
    ap.add_argument("--code-sha", default=None, help="recorded into the layer's attrs")
    args = ap.parse_args()

    # Rule 1, in code: a V_ or B_ track is never read for validation.
    if args.challenge is not None and not args.biosample.startswith("T_"):
        raise SystemExit(
            f"refusing --challenge for {args.biosample!r}: the second check is T_ only (Rule 1)"
        )

    bdir = args.corpus_root / "biosamples" / args.biosample
    gdir = args.genome or (args.corpus_root.parent / "genome")

    with h5py.File(bdir / "counts.h5", "r") as fc, h5py.File(bdir / "pval.h5", "r") as fp:
        ca, pa = _attrs(fc), _attrs(fp)
        res = int(ca["resolution"])
        ccol = list(ca["tracks"]).index(args.assay)
        pcol = list(pa["tracks"]).index(args.assay)
        chroms = args.chroms.split(",") if args.chroms else [
            c for c in fc.keys() if c in fp.keys()
        ]
        chroms = sorted(chroms, key=lambda c: (len(c), c))
        counts = {c: fc[c][:, ccol].astype(np.int32) for c in chroms}
        codes = {c: fp[c][:, pcol] for c in chroms}

    scale = float(pa.get("scale", 100))
    transform = str(pa.get("transform", "linear"))
    rdns = {c: _decode_pval(codes[c], scale, transform) for c in chroms}
    del codes

    total_counts = float(sum(int(v.sum()) for v in counts.values()))
    lam_bg = genome_lambda_bg(total_counts, gsize=args.gsize, resolution=res)
    ours = {c: pval_from_counts(counts[c], lambda_bg=lam_bg, llocal=args.llocal, resolution=res)
            for c in chroms}

    per_chrom = {c: {"n_bins": int(ours[c].shape[0]),
                     "counts_sum": int(counts[c].sum()),
                     "frac_nonzero_counts": float((counts[c] > 0).mean()),
                     "max_ours": float(ours[c].max())}
                 for c in chroms}
    nz = float(np.mean(np.concatenate([(counts[c] > 0) for c in chroms])))
    del counts

    result = {
        "biosample": args.biosample,
        "assay": args.assay,
        "corpus_root": str(args.corpus_root),
        "params": {
            "llocal": args.llocal,
            "slocal_used": False,
            "gsize": args.gsize,
            "resolution": res,
            "lambda_bg": lam_bg,
            "total_counts": total_counts,
            "shift": 0,
            "extsize": None,
            "tail": "P(X > k), MACS2 get_pscore",
        },
        "store_pval_codec": {"scale": scale, "transform": transform},
        "store_pval_is": "read-depth normalized signal (ENCODE), NOT a p-value",
        "chroms": chroms,
        "frac_nonzero_counts": nz,
        "per_chrom": per_chrom,
        "comparisons": {},
    }

    # ---- write the layer -------------------------------------------------------------------
    if args.layer_out is not None:
        args.layer_out.parent.mkdir(parents=True, exist_ok=True)
        n_clipped = 0
        with h5py.File(args.layer_out, "w") as fo:
            for c in chroms:
                enc, nc = encode_pval(ours[c])
                n_clipped += nc
                fo.create_dataset(c, data=enc)
            fo.attrs["scale"] = PVAL_SCALE
            fo.attrs["transform"] = PVAL_TRANSFORM
            fo.attrs["resolution"] = res
            fo.attrs["biosample"] = args.biosample
            fo.attrs["assay"] = args.assay
            fo.attrs["method"] = "macs2_nocontrol_poisson_from_counts"
            fo.attrs["params"] = json.dumps(result["params"])
            fo.attrs["source"] = "candi store counts.h5, DSF-1"
            if args.code_sha:
                fo.attrs["code_sha"] = args.code_sha
        result["layer"] = {"path": str(args.layer_out), "n_clipped": n_clipped,
                           "scale": PVAL_SCALE, "transform": PVAL_TRANSFORM}

    mpath = gdir / "mask.h5"
    mask = None
    if mpath.is_file():
        with h5py.File(mpath, "r") as fm:
            mask = {c: fm[c][:].astype(bool) for c in chroms}

    def compare(name, a: dict, b: dict, n_by_chrom: dict) -> None:
        """`_stats(a, b)` over every bin, and over mask-eligible bins as a diagnostic."""
        aa = np.concatenate([a[c][: n_by_chrom[c]] for c in chroms])
        bb = np.concatenate([b[c][: n_by_chrom[c]] for c in chroms])
        entry = {"all": _stats(aa, bb)}
        if mask is not None:
            mm = np.concatenate([mask[c][: n_by_chrom[c]] for c in chroms])
            if mm.shape[0] == aa.shape[0]:
                entry["masked"] = _stats(aa[mm], bb[mm])
        result["comparisons"][name] = entry

    store_n = {c: ours[c].shape[0] for c in chroms}
    compare("ours_vs_encode_rdns", ours, rdns, store_n)

    # ---- the second independent check ------------------------------------------------------
    if args.challenge is not None:
        chal = bin_challenge_bigwig(args.challenge, chroms, res)
        common = {c: min(store_n[c], chal[c].shape[0]) for c in chroms}
        result["challenge"] = {
            "path": str(args.challenge),
            "grid": {c: {"store": store_n[c], "challenge": int(chal[c].shape[0]),
                         "used": common[c]} for c in chroms},
            "max_grid_gap": max(abs(store_n[c] - int(chal[c].shape[0])) for c in chroms),
            "n_bins_used": int(sum(common.values())),
        }
        compare("ours_vs_challenge", ours, chal, common)
        compare("encode_rdns_vs_challenge", rdns, chal, common)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for name, entry in result["comparisons"].items():
        s = entry["all"]
        print(f"{args.biosample} {name}: pearson={s['pearson']:.4f} "
              f"spearman={s['spearman']:.4f} mse_ratio={s['mse_ratio']:.4f} "
              f"jaccard={s['jaccard_top1pct']:.4f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
