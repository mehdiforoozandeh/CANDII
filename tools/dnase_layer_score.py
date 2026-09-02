"""t78 G1 fallback — score a rebuilt DNase `pval` layer against the challenge's 2019 p-value track.

The same four statistics Phase 1 and Phase 2 reported, computed by the same functions, so a
rebuild number and a Phase 2 number are directly readable against each other: `_stats` and
`bin_challenge_bigwig` are IMPORTED from `atac_pval_gate.py` and `dnase_pval_build.py` rather than
rewritten. The only thing that changes is where the left-hand track comes from.

Three comparisons, genome-wide, on the store's grid:

* `ours_vs_challenge`      — the rebuilt layer against the challenge's DNase `-log10 p`.
* `encode_rdns_vs_challenge` — the store's current DNase `pval` column, which is ENCODE's
  `read-depth normalized signal`, against the same challenge track. The before-picture, on
  identical bins.
* `old_pval_vs_challenge`  — optional; the superseded 2026-08-29 Phase 2 layer, if it is passed.

**Rule 1 is enforced here, in code**, exactly as `dnase_pval_build.py` enforces it: `--challenge`
is refused for any biosample that is not `T_`. Rebuilding a `V_`/`B_` truth track is allowed —
truth is not a method — but no agreement metric may be computed on one.

Standalone: h5py + numpy + scipy + pyBigWig, with `pval_from_counts.py`, `atac_pval_gate.py` and
`dnase_pval_build.py` beside it on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from atac_pval_gate import _attrs, _decode_pval, _stats
from dnase_pval_build import bin_challenge_bigwig


def read_layer(path: Path) -> tuple:
    """One of our single-track layers -> `{chrom: -log10 p}` plus its attrs."""
    with h5py.File(path, "r") as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
        scale = float(attrs.get("scale", 2000))
        transform = str(attrs.get("transform", "arcsinh"))
        out = {c: _decode_pval(f[c][:], scale, transform) for c in f.keys()}
    return out, attrs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", required=True, type=Path, help="the rebuilt pval h5")
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--biosample", required=True)
    ap.add_argument("--assay", default="DNase-seq")
    ap.add_argument("--genome", type=Path, default=None)
    ap.add_argument("--challenge", type=Path, default=None, help="T_ biosamples only")
    ap.add_argument("--old-layer", type=Path, default=None,
                    help="the superseded 2026-08-29 Phase 2 layer, for a three-way read")
    ap.add_argument("--counts-json", type=Path, default=None,
                    help="the rebuild's counts JSON; its ceiling block is copied in")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.challenge is not None and not args.biosample.startswith("T_"):
        raise SystemExit(
            f"refusing --challenge for {args.biosample!r}: the check is T_ only (Rule 1)"
        )

    ours, layer_attrs = read_layer(args.layer)
    bdir = args.corpus_root / "biosamples" / args.biosample
    gdir = args.genome or (args.corpus_root.parent / "genome")

    with h5py.File(bdir / "pval.h5", "r") as fp:
        pa = _attrs(fp)
        res = int(pa["resolution"])
        pcol = list(pa["tracks"]).index(args.assay)
        chroms = sorted([c for c in fp.keys() if c in ours], key=lambda c: (len(c), c))
        codes = {c: fp[c][:, pcol] for c in chroms}
    scale = float(pa.get("scale", 100))
    transform = str(pa.get("transform", "linear"))
    rdns = {c: _decode_pval(codes[c], scale, transform) for c in chroms}
    del codes

    old = None
    if args.old_layer is not None and args.old_layer.is_file():
        old, _ = read_layer(args.old_layer)

    result = {
        "biosample": args.biosample,
        "assay": args.assay,
        "layer": str(args.layer),
        "layer_attrs": {k: (v.item() if hasattr(v, "item") else str(v))
                        for k, v in layer_attrs.items()},
        "chroms": chroms,
        "store_pval_codec": {"scale": scale, "transform": transform},
        "store_pval_is": "read-depth normalized signal (ENCODE), NOT a p-value",
        "max_ours": float(max(float(v.max()) for v in ours.values())),
        "comparisons": {},
    }
    if args.counts_json is not None and args.counts_json.is_file():
        cj = json.loads(args.counts_json.read_text())
        result["counts_rebuild"] = {"depth": cj["depth"],
                                    "n_reads_dropped_duplicate": cj["n_reads_dropped_duplicate"],
                                    "coverage": cj["coverage"],
                                    "ceiling": cj["ceiling"]}

    mpath = gdir / "mask.h5"
    mask = None
    if mpath.is_file():
        with h5py.File(mpath, "r") as fm:
            mask = {c: fm[c][:].astype(bool) for c in chroms}

    def compare(name, a: dict, b: dict, n_by_chrom: dict) -> None:
        aa = np.concatenate([a[c][: n_by_chrom[c]] for c in chroms])
        bb = np.concatenate([b[c][: n_by_chrom[c]] for c in chroms])
        entry = {"all": _stats(aa, bb)}
        if mask is not None:
            mm = np.concatenate([mask[c][: n_by_chrom[c]] for c in chroms])
            if mm.shape[0] == aa.shape[0]:
                entry["masked"] = _stats(aa[mm], bb[mm])
        result["comparisons"][name] = entry

    store_n = {c: min(ours[c].shape[0], rdns[c].shape[0]) for c in chroms}
    compare("ours_vs_encode_rdns", ours, rdns, store_n)

    if args.challenge is not None:
        chal = bin_challenge_bigwig(args.challenge, chroms, res)
        common = {c: min(store_n[c], int(chal[c].shape[0])) for c in chroms}
        result["challenge"] = {
            "path": str(args.challenge),
            "max_grid_gap": max(abs(store_n[c] - int(chal[c].shape[0])) for c in chroms),
            "n_bins_used": int(sum(common.values())),
        }
        compare("ours_vs_challenge", ours, chal, common)
        compare("encode_rdns_vs_challenge", rdns, chal, common)
        if old is not None:
            compare("old_pval_vs_challenge", old, chal, common)

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
