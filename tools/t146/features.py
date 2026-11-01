"""t146 (h31) — the per-25 bp-bin sequence inputs and the fixed mappability strata, built once.

    python tools/t146/features.py build <cache_dir> <MANIFEST.tsv> <annot_dir> <blacklist.bed> \
        <out_dir> [--t138-features features.npz]

Inputs: the t118 count cache (bin counts per chromosome), the t138 annotation dir (the SAME files
t138/t141 used: `k36.umap.bedgraph.gz`, Umap k36 multi-read mappability; `hg38.fa.gz`).
Writes, per main chromosome c (chr1-22, X), aligned to the cache's 25 bp bins (n_bins[c] bins):

  map25__<c>.npy  float32[n_bins]     mean Umap k36 value over the bin's 25 bp (uncovered bp = 0;
                                      t138 `features.mappability` at 25 bp instead of 1 kb)
  dna__<c>.npy    uint8[n_bins * 25]  A 0, C 1, G 2, T 3, anything else (N, past the end) 4
  dec__<c>.npy    int8[n_bins]        mappability decile 0..9 of the bin's 1 kb block, -1 where the
                                      block is blacklisted, or the bin is in the tail past the last
                                      whole 1 kb block

and strata.json. **Strata** (fixed here, before any run): the 1 kb mappability = mean of the 40
map25 bins of the block (== t138's `map_<c>`: mean Umap k36 over the 1000 bp, uncovered 0; with
`--t138-features` the max |difference| per chromosome is recorded and must be < 1e-4); 1 kb blocks
with any 25 bp sub-bin overlapping the blacklist by >= 1 bp are dropped (t138 `blacklist_1k`);
deciles = t138 `map.deciles` (rank deciles, ties averaged) over every kept block of the main
chromosomes, genome-wide. dec0 = the lowest mappability.

**Shifted channel** (null control, not stored): `shifted_window` reads map25 at
(i - SHIFT_BINS) mod n_bins[c], a 5 Mb circular shift within the chromosome.

Needs numpy, pandas, scipy (the venv).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
for _p in (_TOOLS / "t118", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data, pairs as lp  # noqa: E402
from t146.tasks import SHIFT_BINS  # noqa: E402

RES = 25
COARSE = 40                     # 25 bp bins per 1 kb block
DNA_CODE = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    DNA_CODE[_c] = _i
    DNA_CODE[_c | 0x20] = _i    # lower case (soft-masked) bases count as bases
N_CODE = 4


def deciles(x: np.ndarray) -> np.ndarray:
    """t138 `map.deciles`, restated: rank decile 0..9 over the finite values (ties averaged)."""
    from scipy.stats import rankdata
    out = np.full(x.shape, -1, dtype=np.int64)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n:
        rk = rankdata(x[ok], method="average")
        out[ok] = np.minimum(((rk - 1) * 10 // n).astype(np.int64), 9)
    return out


def chrom_bins(cache_dir, manifest) -> dict:
    """{chrom: n_bins} of the cache, identical across every usable product (checked)."""
    corpus = data.Corpus(cache_dir)
    rows = lp.usable_products(lp.read_manifest(manifest))
    ref = None
    for r in rows:
        nb = {c: corpus.n_bins(r["pid"], "counts", c) for c in corpus.chroms(r["pid"], "counts")}
        if ref is None:
            ref = nb
        elif nb != ref:
            raise ValueError(f"{r['pid']}: chromosome bins differ from {rows[0]['pid']}")
    return {c: ref[c] for c in lp.MAIN_CHROMS if c in ref}


def _t138_features():
    """tools/t138/features.py (`_read_table`, `bin_sums`), loaded by path: the bare name
    `features` is this module when it runs as a script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("t138_features", _TOOLS / "t138" / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def map25(bedgraph, n_bins: dict) -> dict:
    """{chrom: float32[n_bins]} mean Umap value per 25 bp bin (uncovered bp = 0)."""
    t138f = _t138_features()
    acc = {c: np.zeros(n) for c, n in n_bins.items()}
    for chunk in t138f._read_table(Path(bedgraph), [0, 1, 2, 3], chunksize=t138f.CHUNK):
        chunk.columns = range(4)
        for c, g in chunk.groupby(0, sort=False):
            if c not in acc:
                continue
            s, e = g[1].to_numpy(np.int64), g[2].to_numpy(np.int64)
            v = g[3].to_numpy(np.float64)
            o = np.argsort(s, kind="stable")
            acc[c] += t138f.bin_sums(s[o], e[o], v[o], n_bins[c], RES)
    return {c: (a / RES).astype(np.float32) for c, a in acc.items()}


def dna_codes(fasta_gz, n_bins: dict):
    """Yield (chrom, uint8[n_bins * 25]) per wanted chromosome, streamed from the fasta."""
    def finish(name, parts):
        seq = np.frombuffer(b"".join(parts), dtype=np.uint8)
        n = n_bins[name] * RES
        out = np.full(n, N_CODE, dtype=np.uint8)
        m = min(n, seq.size)
        out[:m] = DNA_CODE[seq[:m]]
        return name, out

    name, parts = None, []
    with gzip.open(fasta_gz, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name in n_bins:
                    yield finish(name, parts)
                name, parts = line[1:].split()[0].decode(), []
            elif name in n_bins:
                parts.append(line.rstrip(b"\r\n"))
    if name in n_bins:
        yield finish(name, parts)


def blacklist_blocks(blacklist, chrom: str, n_bins: int) -> np.ndarray:
    """bool[n_bins // 40]: the 1 kb block has a blacklisted 25 bp sub-bin (>= 1 bp overlap)."""
    nb = n_bins // COARSE
    flags = data.read_blacklist_flags(blacklist, chrom, nb * COARSE)
    return flags.reshape(nb, COARSE).any(1)


def strata(maps: dict, n_bins: dict, blacklist) -> tuple[dict, dict]:
    """({chrom: int8[n_bins]} per-25 bp decile, info)."""
    chroms = list(maps)
    blocks, bad = [], []
    for c in chroms:
        nb = n_bins[c] // COARSE
        blocks.append(maps[c][:nb * COARSE].astype(np.float64).reshape(nb, COARSE).mean(1))
        bad.append(blacklist_blocks(blacklist, c, n_bins[c]))
    v = np.concatenate(blocks)
    b = np.concatenate(bad)
    v_kept = np.where(b, np.nan, v)
    dec = deciles(v_kept)
    out, o = {}, 0
    edges = {}
    for c, blk in zip(chroms, blocks):
        d = dec[o:o + blk.size]
        o += blk.size
        a = np.full(n_bins[c], -1, dtype=np.int8)
        a[:blk.size * COARSE] = np.repeat(d, COARSE).astype(np.int8)
        out[c] = a
    for k in range(10):
        sel = v_kept[dec == k]
        edges[k] = [float(sel.min()), float(sel.max()), int(sel.size)] if sel.size else None
    info = {"n_blocks": int(v.size), "n_blacklisted_blocks": int(b.sum()),
            "decile_min_max_n": edges}
    return out, info


def shifted_window(m: np.ndarray, start: int, stop: int, shift: int = SHIFT_BINS) -> np.ndarray:
    """Bins [start, stop) of the circularly shifted track s[i] = m[(i - shift) mod n]; positions
    outside [0, n) read 0 (as `Corpus.window` pads)."""
    n = m.shape[0]
    out = np.zeros(stop - start, dtype=np.float32)
    s, e = max(start, 0), min(stop, n)
    if e > s:
        idx = (np.arange(s, e) - shift) % n
        out[s - start:e - start] = m[idx]
    return out


class Features:
    """Read access to a built feature dir: windows of map25 / shifted map25 / DNA codes / deciles."""

    def __init__(self, root):
        self.root = Path(root)
        self._mm: dict = {}

    def _arr(self, kind: str, chrom: str) -> np.ndarray:
        key = (kind, chrom)
        if key not in self._mm:
            self._mm[key] = np.load(self.root / f"{kind}__{chrom}.npy", mmap_mode="r")
        return self._mm[key]

    def map_full(self, chrom: str) -> np.ndarray:
        return self._arr("map25", chrom)

    def dec_full(self, chrom: str) -> np.ndarray:
        return self._arr("dec", chrom)

    def window(self, kind: str, chrom: str, start: int, stop: int) -> np.ndarray:
        """kind map / mapshift: float32[stop - start]; dna: uint8[(stop - start) * 25] (N past
        the chromosome)."""
        if kind == "map":
            m = self._arr("map25", chrom)
            out = np.zeros(stop - start, dtype=np.float32)
            s, e = max(start, 0), min(stop, m.shape[0])
            if e > s:
                out[s - start:e - start] = m[s:e]
            return out
        if kind == "mapshift":
            return shifted_window(self._arr("map25", chrom), start, stop)
        if kind == "dna":
            d = self._arr("dna", chrom)
            out = np.full((stop - start) * RES, N_CODE, dtype=np.uint8)
            s, e = max(start, 0) * RES, min(stop * RES, d.shape[0])
            if e > s:
                out[s - start * RES:e - start * RES] = d[s:e]
            return out
        raise ValueError(kind)


def _save(path: Path, a: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, path)


def build(cache_dir, manifest, annot_dir, blacklist, out_dir, t138_features=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annot = Path(annot_dir)
    n_bins = chrom_bins(cache_dir, manifest)
    maps = map25(annot / "k36.umap.bedgraph.gz", n_bins)
    for c, a in maps.items():
        _save(out_dir / f"map25__{c}.npy", a)
    for c, codes in dna_codes(annot / "hg38.fa.gz", n_bins):
        _save(out_dir / f"dna__{c}.npy", codes)
    missing = [c for c in n_bins if not (out_dir / f"dna__{c}.npy").is_file()]
    if missing:
        raise SystemExit(f"fasta lacks {missing}")
    dec, info = strata(maps, n_bins, blacklist)
    for c, a in dec.items():
        _save(out_dir / f"dec__{c}.npy", a)
    check = {}
    if t138_features:
        with np.load(t138_features, allow_pickle=False) as z:
            for c in n_bins:
                k = f"map_{c}"
                if k not in z.files:
                    continue
                ref = z[k].astype(np.float64)
                nb = min(ref.size, n_bins[c] // COARSE)
                mine = maps[c][:nb * COARSE].astype(np.float64).reshape(nb, COARSE).mean(1)
                check[c] = float(np.max(np.abs(mine - ref[:nb])))
        if any(v >= 1e-4 for v in check.values()):
            raise SystemExit(f"1 kb mappability differs from t138's map_<c>: {check}")
    meta = {"n_bins": n_bins, "res_bp": RES, "shift_bins": SHIFT_BINS,
            "annot_dir": str(annot),
            "annot_tsv": (annot / "ANNOT.tsv").read_text() if (annot / "ANNOT.tsv").is_file() else None,
            "blacklist": str(blacklist),
            "blacklist_sha256": hashlib.sha256(Path(blacklist).read_bytes()).hexdigest(),
            "strata": info, "t138_map_1kb_max_abs_diff": check,
            "rules": __doc__.split("**Strata**")[1].split("**Shifted")[0].strip()}
    (out_dir / "strata.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out_dir / "FEATURES_DONE").write_text("ok\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for a in ("cache_dir", "manifest", "annot_dir", "blacklist", "out_dir"):
        b.add_argument(a)
    b.add_argument("--t138-features", default=None)
    args = ap.parse_args(argv)
    if (Path(args.out_dir) / "FEATURES_DONE").is_file():
        print("FEATURES_DONE exists; nothing to do")
        return 0
    meta = build(args.cache_dir, args.manifest, args.annot_dir, args.blacklist, args.out_dir,
                 args.t138_features)
    print(json.dumps({k: meta[k] for k in ("strata", "t138_map_1kb_max_abs_diff")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
