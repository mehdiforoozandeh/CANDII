"""t138: the four per-1 kb-bin features of the regional map (exploratory; readings only).

    python features.py build <annot_dir> <narrowpeak_dir> <bins1k_dir> <out.npz>

Per chromosome c of the bins1k layout (G1: `<bins1k_dir>/<pid>.npz` with uint32 `a15_<c>`,
`b15_<c>`, `a375_<c>`, `b375_<c>`; every chromosome but chrM/chrY; 1 kb = 40 x 25 bp, tail
dropped) this writes:

  gc_<c>          float32  (G+C)/(A+C+G+T) of the bin in <annot_dir>/hg38.fa.gz; NaN where N > 50 %
  map_<c>         float32  mean Umap k36 value over the 1000 bp of the bin (uncovered bp count 0);
                           absent when <annot_dir>/MAPPABILITY_MISSING exists or no bedGraph is there
  rep_<c>         float32  fraction of the bin covered by the union of rmsk.txt.gz intervals
  ctx_<track>_<c> uint8    2 = the bin overlaps the track's base narrowPeak; 1 = not a peak and the
                           base product's 1 kb count (a15 + b15 of `<track>__base__base`), smoothed
                           by an 11-bin running mean, is above its 90th percentile over all bins with
                           a nonzero count; 0 = else

and `<out>.json` (features.json when out is features.npz) with the input files, their sha256, the
narrowPeak basename chosen per track, and the bin counts per context class per track.

Needs numpy and pandas only (Nibi scipy-stack). The fasta is streamed chromosome by chromosome.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BIN = 1000
DROP = ("chrM", "chrY")
TRACKS = ("C12M02", "C19M16", "C40M17", "C40M18", "C07M20", "C19M22", "C07M29")
SMOOTH_BINS = 11
DOMAIN_PCT = 90.0
N_MAX_FRACTION = 0.5
MAP_FILES = ("k36.umap.bedgraph.gz", "k36.umap.bed.gz")
CHUNK = 5_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def chrom_sizes(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            out[p[0]] = int(p[1])
    return out


def bins1k_chroms(npz_path: Path) -> dict:
    """{chrom: n_bins} from the a15_<c> keys of one bins1k npz, DROP removed."""
    with np.load(npz_path, allow_pickle=False) as z:
        return {k[4:]: int(z[k].size) for k in z.files
                if k.startswith("a15_") and k[4:] not in DROP}


def merge(starts: np.ndarray, ends: np.ndarray):
    """Union of half-open intervals -> sorted, non-overlapping (starts, ends)."""
    if starts.size == 0:
        return starts.astype(np.int64), ends.astype(np.int64)
    o = np.argsort(starts, kind="stable")
    s, e = starts[o].astype(np.int64), ends[o].astype(np.int64)
    run = np.maximum.accumulate(e)
    new = np.ones(s.size, dtype=bool)
    new[1:] = s[1:] > run[:-1]
    gid = np.cumsum(new) - 1
    ms = s[new]
    me = np.zeros(ms.size, dtype=np.int64)
    np.maximum.at(me, gid, e)
    return ms, me


def bin_sums(starts, ends, values, n_bins: int, binsize: int = BIN) -> np.ndarray:
    """Sum over bins [i*binsize, (i+1)*binsize) of value x overlap (bp), for SORTED,
    NON-OVERLAPPING half-open intervals. Evaluates the running integral G at the bin edges."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if starts.size == 0:
        return np.zeros(n_bins)
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("bin_sums needs sorted, non-overlapping intervals")
    full = values * (ends - starts)
    cum = np.concatenate([[0.0], np.cumsum(full)])
    x = np.arange(n_bins + 1, dtype=np.int64) * binsize
    k = np.searchsorted(starts, x, side="right") - 1
    kc = np.clip(k, 0, None)
    part = values[kc] * np.clip(x - starts[kc], 0, ends[kc] - starts[kc])
    g = np.where(k >= 0, cum[kc] + part, 0.0)
    return np.diff(g)


def gc_by_chrom(fasta_gz: Path, chroms: dict, sizes: dict) -> dict:
    """Stream the fasta; per wanted chromosome return float32 GC fraction per 1 kb bin."""
    out: dict = {}

    def finish(name, parts):
        if name not in chroms:
            return
        seq = np.frombuffer(b"".join(parts), dtype=np.uint8)
        if seq.size != sizes[name]:
            raise SystemExit(f"{name}: fasta length {seq.size} != chrom.sizes {sizes[name]}")
        n = chroms[name]
        a = (seq[: n * BIN] & 0xDF).reshape(n, BIN)  # upper-case ASCII letters
        gc = ((a == ord("G")) | (a == ord("C"))).sum(1)
        acgt = gc + ((a == ord("A")) | (a == ord("T"))).sum(1)
        nn = (a == ord("N")).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = gc / acgt
        f = np.where((nn > N_MAX_FRACTION * BIN) | (acgt == 0), np.nan, f)
        out[name] = f.astype(np.float32)

    name, parts = None, []
    with gzip.open(fasta_gz, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    finish(name, parts)
                name = line[1:].split()[0].decode()
                parts = []
            elif name in chroms:
                parts.append(line.rstrip(b"\r\n"))
    if name is not None:
        finish(name, parts)
    missing = sorted(set(chroms) - set(out))
    if missing:
        raise SystemExit(f"fasta lacks chromosomes {missing}")
    return out


def _read_table(path: Path, usecols, chunksize=None):
    """gzip TSV; leading header lines (track/browser/#) skipped."""
    skip = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(("track", "browser", "#")):
                skip += 1
            else:
                break
    return pd.read_csv(path, sep="\t", header=None, skiprows=skip, usecols=usecols,
                       chunksize=chunksize, dtype={usecols[0]: str}, compression="gzip")


def mappability(path: Path, chroms: dict) -> dict:
    """Mean value over each 1 kb bin (bp not covered count 0). A .bed.gz has no value column:
    every covered bp counts 1."""
    is_bed = path.name.endswith(".bed.gz")
    cols = [0, 1, 2] if is_bed else [0, 1, 2, 3]
    acc = {c: np.zeros(n) for c, n in chroms.items()}
    for chunk in _read_table(path, cols, chunksize=CHUNK):
        chunk.columns = range(len(cols))
        for c, g in chunk.groupby(0, sort=False):
            if c not in acc:
                continue
            s = g[1].to_numpy(np.int64)
            e = g[2].to_numpy(np.int64)
            v = np.ones(s.size) if is_bed else g[3].to_numpy(np.float64)
            o = np.argsort(s, kind="stable")
            s, e, v = s[o], e[o], v[o]
            if is_bed:
                s, e = merge(s, e)
                v = np.ones(s.size)
            acc[c] += bin_sums(s, e, v, chroms[c])
    return {c: (a / BIN).astype(np.float32) for c, a in acc.items()}


def coverage_fraction(df: pd.DataFrame, chroms: dict) -> dict:
    """df columns (chrom, start, end) -> fraction of each 1 kb bin covered by their union."""
    out = {c: np.zeros(n, dtype=np.float32) for c, n in chroms.items()}
    for c, g in df.groupby("chrom", sort=False):
        if c not in out:
            continue
        s, e = merge(g["start"].to_numpy(np.int64), g["end"].to_numpy(np.int64))
        out[c] = (bin_sums(s, e, np.ones(s.size), chroms[c]) / BIN).astype(np.float32)
    return out


def read_bed3(path: Path, cols) -> pd.DataFrame:
    df = _read_table(path, cols)
    df.columns = ["chrom", "start", "end"]
    return df


def pick_peak_file(narrowpeak_dir: Path, track: str):
    d = narrowpeak_dir / track
    cands = sorted(p.name for p in d.glob("call_peak_shard0_*.bfilt.narrowPeak.gz"))
    rule = "call_peak_shard0_*.bfilt.narrowPeak.gz"
    if not cands and track == "C12M02":
        cands = sorted(p.name for p in d.glob("call_peak_pr1_*.bfilt.narrowPeak.gz"))
        rule = "call_peak_pr1_*.bfilt.narrowPeak.gz"
    if not cands:
        raise SystemExit(f"{track}: no narrowPeak under {d} ({rule})")
    return d / cands[0], rule, cands


def running_mean(x: np.ndarray, w: int = SMOOTH_BINS) -> np.ndarray:
    """Centered w-bin mean; at the ends the mean over the bins that exist."""
    k = np.ones(w)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x), k, mode="same")
    return num / den


def build(annot_dir: Path, narrowpeak_dir: Path, bins1k_dir: Path, out: Path) -> dict:
    annot_dir, narrowpeak_dir, bins1k_dir, out = map(Path, (annot_dir, narrowpeak_dir,
                                                             bins1k_dir, out))
    tracks = [t for t in TRACKS if (bins1k_dir / f"{t}__base__base.npz").exists()]
    missing_tracks = [t for t in TRACKS if t not in tracks]
    if not tracks:
        raise SystemExit(f"no <track>__base__base.npz under {bins1k_dir}")
    for t in missing_tracks:
        print(f"WARNING: no base bins1k for {t}; ctx_{t}_* not written", file=sys.stderr)
    chroms = bins1k_chroms(bins1k_dir / f"{tracks[0]}__base__base.npz")
    sizes = chrom_sizes(annot_dir / "hg38.chrom.sizes")
    for c, n in chroms.items():
        if c not in sizes:
            raise SystemExit(f"{c} not in hg38.chrom.sizes")
        if sizes[c] // BIN != n:
            raise SystemExit(f"{c}: bins1k has {n} bins, chrom.sizes gives {sizes[c] // BIN}")

    arrays: dict = {}
    files: dict = {}
    fa = annot_dir / "hg38.fa.gz"
    files["hg38.fa.gz"] = sha256(fa)
    files["hg38.chrom.sizes"] = sha256(annot_dir / "hg38.chrom.sizes")
    for c, g in gc_by_chrom(fa, chroms, sizes).items():
        arrays[f"gc_{c}"] = g

    map_path = next((annot_dir / f for f in MAP_FILES if (annot_dir / f).exists()), None)
    map_missing = (annot_dir / "MAPPABILITY_MISSING").exists() or map_path is None
    if not map_missing:
        files[map_path.name] = sha256(map_path)
        for c, m in mappability(map_path, chroms).items():
            arrays[f"map_{c}"] = m

    rm = annot_dir / "rmsk.txt.gz"
    files["rmsk.txt.gz"] = sha256(rm)
    for c, r in coverage_fraction(read_bed3(rm, [5, 6, 7]), chroms).items():
        arrays[f"rep_{c}"] = r

    peaks: dict = {}
    ctx_counts: dict = {}
    for t in tracks:
        pf, rule, cands = pick_peak_file(narrowpeak_dir, t)
        files[f"{t}/{pf.name}"] = sha256(pf)
        peaks[t] = {"basename": pf.name, "rule": rule, "candidates": cands}
        is_peak = {c: v > 0 for c, v in
                   coverage_fraction(read_bed3(pf, [0, 1, 2]), chroms).items()}
        with np.load(bins1k_dir / f"{t}__base__base.npz", allow_pickle=False) as z:
            cnt = {c: z[f"a15_{c}"].astype(np.float64) + z[f"b15_{c}"].astype(np.float64)
                   for c in chroms}
        sm = {c: running_mean(cnt[c]) for c in chroms}
        kept = np.concatenate([sm[c][cnt[c] > 0] for c in chroms])
        thr = float(np.percentile(kept, DOMAIN_PCT)) if kept.size else np.inf
        counts = np.zeros(3, dtype=np.int64)
        for c in chroms:
            ctx = np.zeros(chroms[c], dtype=np.uint8)
            ctx[(sm[c] > thr) & ~is_peak[c]] = 1
            ctx[is_peak[c]] = 2
            arrays[f"ctx_{t}_{c}"] = ctx
            counts += np.bincount(ctx, minlength=3)
        ctx_counts[t] = {"0": int(counts[0]), "1": int(counts[1]), "2": int(counts[2]),
                         "domain_threshold_smoothed_count": thr}

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = {
        "chroms": chroms,
        "tracks": tracks,
        "tracks_missing": missing_tracks,
        "features": ["gc", "rep", "ctx"] if map_missing else ["gc", "map", "rep", "ctx"],
        "mappability_missing": bool(map_missing),
        "mappability_file": None if map_missing else map_path.name,
        "files_sha256": files,
        "peak_files": peaks,
        "ctx_counts": ctx_counts,
        "rules": {
            "bin": "1 kb = 40 x 25 bp, tail dropped (n = len // 1000)",
            "gc": "(G+C)/(A+C+G+T), case-insensitive; NaN where N > 50 % of the bin",
            "map": "sum(value x overlap bp) / 1000; uncovered bp count 0",
            "rep": "fraction of the bin covered by the union of rmsk genoName/genoStart/genoEnd",
            "ctx": "2 peak overlap (>= 1 bp); 1 not peak and 11-bin running mean of base "
                   "a15+b15 > its 90th percentile over bins with a nonzero count; 0 else",
        },
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 5 or a[0] != "build":
        raise SystemExit("usage: features.py build <annot_dir> <narrowpeak_dir> <bins1k_dir> "
                         "<out.npz>")
    meta = build(*map(Path, a[1:]))
    print(json.dumps({"features": meta["features"], "tracks": meta["tracks"],
                      "ctx_counts": meta["ctx_counts"]}))


if __name__ == "__main__":
    main()
