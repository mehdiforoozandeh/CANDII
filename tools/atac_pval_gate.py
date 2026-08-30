"""G1 Phase 1 — recompute one experiment's `-log10 p` from the store's counts and score it
against ENCODE's own p-value track, which the store already holds binned in `pval.h5`.

One experiment per invocation, genome-wide over every chromosome the store has. Writes one JSON.
The four statistics the gate is written against are `pearson`, `spearman`, `mse_ratio` and
`jaccard_top1pct`, computed over **every bin** (`scope = "all"`). The same four are also emitted
over mask-eligible bins only (`scope = "masked"`) as a diagnostic; the pre-registered bar is on
`all`, declared here before any number was seen.

Deliberately standalone: h5py + numpy + scipy and `pval_from_counts.py` beside it on PYTHONPATH,
so it runs against a store on a cluster whose checkout of this repo is stale.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from pval_from_counts import (
    MACS2_GSIZE_HS,
    MACS2_LLOCAL,
    genome_lambda_bg,
    pval_from_counts,
)


def _attrs(f) -> dict:
    out = {}
    for k, v in f.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        if isinstance(v, str) and v and v[0] in "[{":
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


def _decode_pval(codes: np.ndarray, scale: float, transform: str) -> np.ndarray:
    x = codes.astype(np.float64) / float(scale)
    if transform == "arcsinh":
        x = np.sinh(x)
    elif transform != "linear":
        raise SystemExit(f"unknown pval transform {transform!r}")
    return x.astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    na = math.sqrt(float(a @ a))
    nb = math.sqrt(float(b @ b))
    return float(a @ b) / (na * nb) if na > 0 and nb > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties sharing the run mean — `scipy.stats.rankdata(x, 'average')`.

    Written out rather than called because at 1.2e8 bins the temporaries decide whether the job
    fits in memory. Checked against `rankdata` on random ties-heavy input.
    """
    n = x.shape[0]
    order = np.argsort(x, kind="stable")
    sx = x[order]
    first = np.flatnonzero(np.r_[True, sx[1:] != sx[:-1]])
    run_len = np.diff(np.r_[first, n])
    avg = first.astype(np.float64) + (run_len + 1) / 2.0     # 1-based mean rank of the run
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, run_len)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rank(a), _rank(b))


def _top_frac_jaccard(a: np.ndarray, b: np.ndarray, frac: float) -> dict:
    n = a.shape[0]
    k = max(1, int(round(n * frac)))
    ta = np.partition(a, n - k)[n - k]
    tb = np.partition(b, n - k)[n - k]
    ma = a >= ta
    mb = b >= tb
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return {
        "jaccard": inter / union if union else float("nan"),
        "n_target": k,
        "n_ours": int(np.count_nonzero(ma)),
        "n_encode": int(np.count_nonzero(mb)),
        "threshold_ours": float(ta),
        "threshold_encode": float(tb),
    }


def _stats(ours: np.ndarray, enc: np.ndarray) -> dict:
    o = ours.astype(np.float64)
    e = enc.astype(np.float64)
    mo2 = float((o * o).mean())
    me2 = float((e * e).mean())
    top = _top_frac_jaccard(ours, enc, 0.01)
    return {
        "n_bins": int(ours.shape[0]),
        "pearson": _pearson(ours, enc),
        "spearman": _spearman(ours, enc),
        # the gate's statistic: each track's own mean square, ours over ENCODE's
        "mse_ratio": mo2 / me2 if me2 > 0 else float("nan"),
        "jaccard_top1pct": top["jaccard"],
        "top1pct": top,
        # context, so the ratio above can be read
        "mean_ours": float(o.mean()),
        "mean_encode": float(e.mean()),
        "mean_ratio": float(o.mean() / e.mean()) if e.mean() > 0 else float("nan"),
        "meansq_ours": mo2,
        "meansq_encode": me2,
        "var_ours": float(o.var()),
        "var_encode": float(e.var()),
        "max_ours": float(o.max()),
        "max_encode": float(e.max()),
        "mse_between": float(((o - e) ** 2).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--assay", default="ATAC-seq")
    ap.add_argument("--genome", type=Path, default=None, help="default: sibling genome/ dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--llocal", type=int, default=MACS2_LLOCAL)
    ap.add_argument("--gsize", type=float, default=MACS2_GSIZE_HS)
    ap.add_argument("--chroms", default=None, help="comma list; omit for every chromosome stored")
    args = ap.parse_args()

    bdir = args.corpus_root / "biosamples" / args.biosample
    gdir = args.genome or (args.corpus_root.parent / "genome")
    cpath, ppath = bdir / "counts.h5", bdir / "pval.h5"

    with h5py.File(cpath, "r") as fc, h5py.File(ppath, "r") as fp:
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
    encode = {c: _decode_pval(codes[c], scale, transform) for c in chroms}
    del codes

    total_counts = float(sum(int(v.sum()) for v in counts.values()))
    lam_bg = genome_lambda_bg(total_counts, gsize=args.gsize, resolution=res)

    ours = {c: pval_from_counts(counts[c], lambda_bg=lam_bg, llocal=args.llocal, resolution=res)
            for c in chroms}

    per_chrom = {c: {"n_bins": int(ours[c].shape[0]),
                     "pearson": _pearson(ours[c], encode[c]),
                     "counts_sum": int(counts[c].sum()),
                     "frac_nonzero_counts": float((counts[c] > 0).mean())}
                 for c in chroms}

    o_all = np.concatenate([ours[c] for c in chroms])
    e_all = np.concatenate([encode[c] for c in chroms])
    nz = float(np.mean(np.concatenate([(counts[c] > 0) for c in chroms])))
    del counts, ours, encode

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
        "encode_codec": {"scale": scale, "transform": transform},
        "chroms": chroms,
        "frac_nonzero_counts": nz,
        "per_chrom": per_chrom,
        "scopes": {"all": _stats(o_all, e_all)},
    }

    mpath = gdir / "mask.h5"
    if mpath.is_file():
        with h5py.File(mpath, "r") as fm:
            m_all = np.concatenate([fm[c][:].astype(bool) for c in chroms])
        if m_all.shape[0] == o_all.shape[0]:
            result["scopes"]["masked"] = _stats(o_all[m_all], e_all[m_all])
            result["mask_valid_frac"] = float(m_all.mean())
        else:
            result["mask_error"] = f"mask has {m_all.shape[0]} bins, tracks have {o_all.shape[0]}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{args.biosample}: pearson={result['scopes']['all']['pearson']:.4f} "
          f"spearman={result['scopes']['all']['spearman']:.4f} "
          f"mse_ratio={result['scopes']['all']['mse_ratio']:.4f} "
          f"jaccard={result['scopes']['all']['jaccard_top1pct']:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
