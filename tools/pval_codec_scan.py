#!/usr/bin/env python3
"""t25's gate — does every pval.h5 in a corpus carry the codec it was supposed to get?

    python tools/pval_codec_scan.py --corpus-root <…/CANDI_STORE/eic> [--json out.json]
    python tools/pval_codec_scan.py --corpus-root <…> --roundtrip --source-root <npz tree> \
        --spot T_X/H3K4me3 --spot T_Y/ATAC-seq

`PVAL_CODEC_PLAN.md` §6 defines "done" for the rebuild, and this is that definition as code. It
answers four questions, and a corpus passes only when all four are clean:

1. **schema / scale / transform.** Every `pval.h5` must read `schema == 2`, `transform ==
   "arcsinh"`, `scale == 2000`. A file that still says `linear` at 100 was not rebuilt; a file with
   a `scale` and no `transform` is a schema-1 file whose units are only recoverable by convention.
2. **`pval_clip_frac == 0.0`, everywhere** (D28). The new ceiling is `sinh(65535/2000)` = 8.5e13.
   Nothing real reaches it, so a nonzero entry does NOT mean "the codec is tight" — it means a
   source value above 8.5e13, which is a fault in the SOURCE and worth stopping for.
3. **The manifest agrees with the files.** `build-manifest` republishes `pval_scale` /
   `pval_transform` per track; a corpus whose files moved and whose manifest did not is exactly the
   "the units are recorded nowhere" complaint that started this.
4. **`--roundtrip`: the summits are actually back.** Decode a spot track out of the store and
   compare against its source npz. The bound is `eps * hypot(1, x)` with `eps = 1/(2*scale)` — the
   exact `cosh`-Jacobian form, absolute below `-log10 p` of 1 and relative above it. This also
   reports the track's maximum, which is the number the old codec capped at 655.35.

Exit code is 0 only when the corpus passes. Anything else prints what failed, first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

WANT_SCHEMA = 2
WANT_TRANSFORM = "arcsinh"
WANT_SCALE = 2000


def _attrs(f) -> dict:
    out = {}
    for k, v in f.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def scan_corpus(root: Path, *, want_scale: int, want_transform: str) -> dict:
    files = sorted((root / "biosamples").glob("*/pval.h5"))
    rep = {"corpus_root": str(root), "n_files": len(files),
           "want": {"schema": WANT_SCHEMA, "scale": want_scale, "transform": want_transform},
           "bad_codec": [], "clipping": [], "unreadable": [], "max_clip_frac": 0.0,
           "n_tracks": 0, "codecs_seen": {}}
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                a = _attrs(f)
                tracks = json.loads(a.get("tracks", "[]"))
                clip = json.loads(a.get("pval_clip_frac", "[]"))
        except Exception as e:                                  # a corrupt file is a failure, loudly
            rep["unreadable"].append({"path": str(p), "error": repr(e)})
            continue
        schema, scale = a.get("schema"), a.get("scale")
        transform = a.get("transform", "linear (absent = schema 1)")
        key = f"schema{schema}/{transform}/{scale}"
        rep["codecs_seen"][key] = rep["codecs_seen"].get(key, 0) + 1
        rep["n_tracks"] += len(tracks)
        if (schema != WANT_SCHEMA or scale != want_scale
                or a.get("transform") != want_transform):
            rep["bad_codec"].append({"path": str(p), "schema": schema, "scale": scale,
                                     "transform": a.get("transform")})
        for t, cf in zip(tracks, clip):
            cf = float(cf)
            rep["max_clip_frac"] = max(rep["max_clip_frac"], cf)
            if cf != 0.0:
                rep["clipping"].append({"path": str(p), "track": t, "clip_frac": cf})
    return rep


def check_manifest(root: Path, *, want_scale: int, want_transform: str) -> dict:
    mp = root / "manifest.json"
    out = {"path": str(mp), "present": mp.is_file(), "disagreements": [], "schema": None}
    if not out["present"]:
        return out
    m = json.loads(mp.read_text(encoding="utf-8"))
    out["schema"] = m.get("schema")
    for bios, entry in m.get("biosamples", {}).items():
        for rec in entry.get("tracks", []):
            if rec.get("pval_clip_frac") is None:               # this track has no pval layer
                continue
            if rec.get("pval_scale") != want_scale or rec.get("pval_transform") != want_transform:
                out["disagreements"].append(
                    {"biosample": bios, "track": rec.get("assay"),
                     "pval_scale": rec.get("pval_scale"),
                     "pval_transform": rec.get("pval_transform")})
    return out


def roundtrip(root: Path, source_root: Path, spots, *, want_scale: int, resolution: int = 25,
              chrom_pick: str | None = None):
    """Decode a spot track out of the store and compare against the source npz it was built from."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from candi.store import layout as L                          # noqa: E402
    from candi.store.reader import CorpusStore                   # noqa: E402

    eps = 1.0 / (2 * want_scale)
    corpus = CorpusStore(root)
    out = []
    for spot in spots:
        bios, _, assay = spot.partition("/")
        rec = {"biosample": bios, "assay": assay, "ok": False}
        try:
            bs = corpus[bios]
            sdir = source_root / bios / assay / f"signal_BW_res{resolution}"
            npzs = sorted(sdir.glob("*.npz"), key=lambda q: L.sort_chroms([q.stem])[0])
            if not npzs:
                raise FileNotFoundError(f"no pval npz under {sdir}")
            src = npzs[0] if chrom_pick is None else sdir / f"{chrom_pick}.npz"
            chrom = src.stem
            with np.load(src) as z:
                want = np.asarray(z[z.files[0]], dtype=np.float64)
            # `assays=[assay]` is not optional. Column 0 of the FILE is whichever track sorted
            # first, and reading it instead would compare one assay's store values against a
            # different assay's source — which looks exactly like a codec failure.
            got = bs.pval(chrom, 0, len(want), assays=[assay])[:, 0].astype(np.float64)
            n = min(len(got), len(want))                          # D13 truncates by at most one bin
            got, want = got[:n], want[:n]
            err = np.abs(got - want)
            bound = eps * np.hypot(1.0, want) * 1.01
            rec.update(
                chrom=chrom, n_bins=int(n),
                source_max=float(want.max()), store_max=float(got.max()),
                worst_err=float(err.max()),
                worst_err_over_bound=float((err / np.maximum(bound, 1e-300)).max()),
                n_over_bound=int((err > bound).sum()),
                n_above_old_ceiling=int((want > 655.35).sum()),
            )
            rec["ok"] = rec["n_over_bound"] == 0
        except Exception as e:
            rec["error"] = repr(e)
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-root", required=True, type=Path)
    ap.add_argument("--scale", type=int, default=WANT_SCALE)
    ap.add_argument("--transform", default=WANT_TRANSFORM)
    ap.add_argument("--roundtrip", action="store_true")
    ap.add_argument("--source-root", type=Path, default=None)
    ap.add_argument("--spot", action="append", default=[],
                    help="repeatable BIOSAMPLE/ASSAY, e.g. B_SJCRH30/H3K4me3")
    ap.add_argument("--chrom", default=None,
                    help="round-trip this chromosome; default is the first the source has")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    rep = scan_corpus(a.corpus_root, want_scale=a.scale, want_transform=a.transform)
    rep["manifest"] = check_manifest(a.corpus_root, want_scale=a.scale, want_transform=a.transform)
    if a.roundtrip:
        if a.source_root is None or not a.spot:
            print("--roundtrip needs --source-root and at least one --spot", file=sys.stderr)
            return 2
        rep["roundtrip"] = roundtrip(a.corpus_root, a.source_root, a.spot, want_scale=a.scale,
                                     chrom_pick=a.chrom)

    fails = []
    if rep["unreadable"]:
        fails.append(f"{len(rep['unreadable'])} unreadable pval.h5")
    if rep["bad_codec"]:
        fails.append(f"{len(rep['bad_codec'])} file(s) not at "
                     f"schema {WANT_SCHEMA}/{a.transform}/{a.scale}")
    if rep["clipping"]:
        fails.append(f"{len(rep['clipping'])} track(s) with pval_clip_frac > 0 (D28)")
    if not rep["manifest"]["present"]:
        fails.append("no manifest.json — run build-manifest")
    elif rep["manifest"]["disagreements"]:
        fails.append(f"{len(rep['manifest']['disagreements'])} manifest track(s) name a "
                     f"different codec than the file")
    if a.roundtrip:
        bad = [r for r in rep["roundtrip"] if not r["ok"]]
        if bad:
            fails.append(f"{len(bad)} spot track(s) failed the round trip")
    rep["ok"] = not fails
    rep["failures"] = fails

    text = json.dumps(rep, indent=2)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(text + "\n", encoding="utf-8")
    print(text)

    print(f"\n{a.corpus_root}: {rep['n_files']} pval.h5, {rep['n_tracks']} tracks, "
          f"codecs seen {rep['codecs_seen']}, max clip_frac {rep['max_clip_frac']:.3g}",
          file=sys.stderr)
    for r in rep.get("roundtrip", []):
        if "error" in r:
            print(f"  spot {r['biosample']}/{r['assay']}: ERROR {r['error']}", file=sys.stderr)
            continue
        print(f"  spot {r['biosample']}/{r['assay']} {r['chrom']}: source max "
              f"{r['source_max']:.1f} -> store max {r['store_max']:.1f}, "
              f"{r['n_above_old_ceiling']} bins were above the old 655.35 ceiling, "
              f"worst err {r['worst_err']:.4g} "
              f"({r['worst_err_over_bound']:.2f}x the bound)", file=sys.stderr)
    print(("PASS" if rep["ok"] else "FAIL: " + "; ".join(fails)), file=sys.stderr)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
