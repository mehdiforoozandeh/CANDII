"""Region strata masks at 25 bp on hg38 (t152 plan chunk E3, crux task t161).

    python tools/onion_eval/strata.py build --genome <genome dir> --src <download dir> \
        --t146 <t146 features dir> --umap <k36.umap.bedgraph.gz> --out <masks.h5>
    python tools/onion_eval/strata.py list --masks <masks.h5> [--genome <genome dir>]

`masks.h5` holds `/<stratum>/<chrom>` uint8 0/1 `(n_bins,)` on the grid of the genome layer
(`<genome>/mask.h5`: one dataset per chromosome, n_bins = floor(len / 25)), h5 chunks of at most
4096 bins. Root attr `sources` is a JSON list with, per annotation, the URL, release, file md5,
bytes and download date. Each stratum group carries attrs `source` and `rule`.

Rasterization rule (every stratum): an interval marks every 25 bp bin it overlaps by >= 1 bp.
BED intervals are 0-based half-open [start, end); GTF intervals are 1-based closed and are
converted to [start - 1, end). Base pairs past the last whole bin are dropped (floor grid).

Strata:
  promoters        ENCODE cCRE Registry V4, GRCh38, class PLS.
  enhancers        same file, classes pELS and dELS.
  gene_bodies      GENCODE release 40 basic (the release cCRE V4 used), transcripts of
                   protein-coding genes, TSS to TES.
  heterochromatin  full-stack ChromHMM hg38 (Vu and Ernst 2022), states HET1 to HET9.
  mappability_d0 .. mappability_d9
                   Umap k36 (Hoffman lab) deciles, the testbed's definition (tools/t146/features.py):
                   the mean Umap k36 value of each 1 kb block (40 bins, read from t146's map25
                   arrays), blocks with a blacklisted bin dropped, rank deciles with ties averaged,
                   genome-wide over chr1-22 and chrX. d0 is the least mappable. chrY and the tail
                   past the last whole block are in no decile.

Per-target foreground and background are cell-type-specific; they are a rule here, not a mask:
`foreground_mask(avg, frac)` and `background_mask(avg, frac)`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

RES = 25
CHUNK_BINS = 4096
BLOCK = 40                      # 25 bp bins per 1 kb block (t146 COARSE)
DECILE_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)

CCRE_URL = "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"
GENCODE_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_40/"
               "gencode.v40.basic.annotation.gtf.gz")
CHROMHMM_URL = ("https://public.hoffman2.idre.ucla.edu/ernst/2K9RS/full_stack/"
                "full_stack_annotation_public_release/hg38/hg38_genome_100_segments.bed.gz")
UMAP_URL = "https://bismap.hoffmanlab.org/raw/hg38/k36.umap.bedgraph.gz"

SOURCES = {
    "ccre": {"url": CCRE_URL, "release": "ENCODE cCRE Registry V4, GRCh38 (SCREEN)",
             "strata": ["promoters", "enhancers"]},
    "gencode": {"url": GENCODE_URL, "release": "GENCODE release 40, basic annotation",
                "strata": ["gene_bodies"]},
    "chromhmm": {"url": CHROMHMM_URL,
                 "release": "full-stack ChromHMM hg38, 100 states, 1-based state names "
                            "(README version 3, 2024-05-06 onward); Vu and Ernst 2022",
                 "strata": ["heterochromatin"]},
}
PROMOTER_CLASSES = ("PLS",)
ENHANCER_CLASSES = ("pELS", "dELS")
HET_STATE = re.compile(r"^\d+_HET[1-9]$")
DECILE_STRATA = tuple(f"mappability_d{k}" for k in range(10))


# -- rasterization ---------------------------------------------------------------------------

def interval_bins(starts, ends, n_bins: int, res: int = RES) -> np.ndarray:
    """uint8[n_bins]: 1 where a half-open interval [start, end) overlaps the bin by >= 1 bp."""
    s = np.asarray(starts, dtype=np.int64).ravel()
    e = np.asarray(ends, dtype=np.int64).ravel()
    keep = e > s
    s, e = s[keep], e[keep]
    b0 = np.clip(s // res, 0, n_bins)
    b1 = np.clip((e - 1) // res + 1, 0, n_bins)
    keep = b1 > b0
    diff = np.zeros(n_bins + 1, dtype=np.int64)
    np.add.at(diff, b0[keep], 1)
    np.add.at(diff, b1[keep], -1)
    return (np.cumsum(diff[:-1]) > 0).astype(np.uint8)


def gtf_to_half_open(start1, end1):
    """GTF 1-based closed [start, end] -> 0-based half-open [start - 1, end)."""
    return np.asarray(start1, np.int64) - 1, np.asarray(end1, np.int64)


def rasterize(intervals: dict, n_bins: dict) -> dict:
    """{chrom: ([starts], [ends])} -> {chrom: uint8[n_bins]} for every chrom of the frame."""
    out = {}
    for c, n in n_bins.items():
        s, e = intervals.get(c, ([], []))
        out[c] = interval_bins(s, e, n)
    return out


# -- deciles and the foreground rule ---------------------------------------------------------

def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties get the average rank (scipy rankdata method='average')."""
    x = np.asarray(x, dtype=np.float64)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    avg = upper - (counts - 1) / 2.0
    return avg[inv]


def deciles(x: np.ndarray) -> np.ndarray:
    """Rank decile 0..9 over the finite values (ties averaged), -1 elsewhere (t146 rule)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, -1, dtype=np.int64)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n:
        rk = average_ranks(x[ok])
        out[ok] = np.minimum(((rk - 1) * 10 // n).astype(np.int64), 9)
    return out


def block_deciles(map25: dict, bad: dict, n_bins: dict, block: int = BLOCK) -> tuple:
    """Per-bin decile (int8, -1 = none) of each bin's 1 kb block mean, genome-wide over the
    chromosomes of `map25`. A block with any bad bin is dropped. Returns (deciles, info)."""
    chroms = list(map25)
    vals, flags = [], []
    for c in chroms:
        nb = n_bins[c] // block
        vals.append(np.asarray(map25[c][:nb * block], np.float64).reshape(nb, block).mean(1))
        flags.append(np.asarray(bad[c][:nb * block], bool).reshape(nb, block).any(1))
    v = np.concatenate(vals) if vals else np.zeros(0)
    b = np.concatenate(flags) if flags else np.zeros(0, bool)
    dec = deciles(np.where(b, np.nan, v))
    out, o = {}, 0
    for c, blk in zip(chroms, vals):
        d = dec[o:o + blk.size]
        o += blk.size
        a = np.full(n_bins[c], -1, dtype=np.int8)
        a[:blk.size * block] = np.repeat(d, block).astype(np.int8)
        out[c] = a
    info = {"n_blocks": int(v.size), "n_dropped_blocks": int(b.sum()),
            "blocks_per_decile": [int((dec == k).sum()) for k in range(10)]}
    return out, info


def foreground_mask(avg, frac: float) -> np.ndarray:
    """bool: the top `frac` of bins by the leave-one-cell-out average rate.

    A bin is foreground when its value is strictly above the value at rank floor(frac * n) of
    the finite values (descending) and strictly above 0. Ties at the cut and zeros are never
    selected, so the selected fraction is at most `frac`, and an all-zero (or zero-heavy) track
    never selects every position (the old 98th-percentile trap). NaN bins are never selected.
    """
    a = np.asarray(avg, dtype=np.float64)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    ok = np.isfinite(a)
    n = int(ok.sum())
    out = np.zeros(a.shape, dtype=bool)
    k = int(np.floor(frac * n))
    if k == 0:
        return out
    v = np.sort(a[ok])[::-1]
    cut = max(v[k] if k < n else -np.inf, 0.0)
    out[ok] = a[ok] > cut
    return out


def background_mask(avg, frac: float) -> np.ndarray:
    """bool: finite bins that are not foreground under the same `frac`."""
    a = np.asarray(avg, dtype=np.float64)
    return np.isfinite(a) & ~foreground_mask(a, frac)


# -- annotation readers ----------------------------------------------------------------------

def _open_text(path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def read_ccre(path, classes) -> dict:
    """{chrom: (starts, ends)} of cCREs whose class (last column, before any comma) is in `classes`."""
    want = set(classes)
    acc: dict = {}
    with _open_text(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4 or f[-1].split(",")[0] not in want:
                continue
            s, e = acc.setdefault(f[0], ([], []))
            s.append(int(f[1]))
            e.append(int(f[2]))
    return acc


def ccre_class_counts(path) -> dict:
    counts: dict = {}
    with _open_text(path) as fh:
        for line in fh:
            k = line.rstrip("\n").split("\t")[-1]
            counts[k] = counts.get(k, 0) + 1
    return counts


def read_gencode_bodies(path, gene_type: str = "protein_coding") -> tuple:
    """{chrom: (starts, ends)} of transcript spans whose gene_type is `gene_type`, half-open."""
    acc: dict = {}
    n = 0
    tag = f'gene_type "{gene_type}"'
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if len(f) < 9 or f[2] != "transcript" or tag not in f[8]:
                continue
            s, e = gtf_to_half_open(int(f[3]), int(f[4]))
            ss, ee = acc.setdefault(f[0], ([], []))
            ss.append(int(s))
            ee.append(int(e))
            n += 1
    return acc, n


def read_chromhmm(path, state_re=HET_STATE) -> tuple:
    """{chrom: (starts, ends)} of segments whose state name matches, and the matched state names."""
    acc: dict = {}
    states = set()
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(("track", "browser", "#")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 4 or not state_re.match(f[3]):
                continue
            states.add(f[3])
            s, e = acc.setdefault(f[0], ([], []))
            s.append(int(f[1]))
            e.append(int(f[2]))
    return acc, sorted(states)


# -- provenance ------------------------------------------------------------------------------

def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> str:
    """Download `url` to `dest` unless present; return the download date (UTC, ISO), read from
    a `<dest>.date` sidecar when the file was already there."""
    side = dest.with_name(dest.name + ".date")
    if dest.is_file() and side.is_file():
        return side.read_text().strip()
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as out:
        shutil.copyfileobj(r, out, 1 << 22)
    part.rename(dest)
    day = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    side.write_text(day + "\n")
    return day


def genome_frame(genome_dir) -> tuple:
    """({chrom: n_bins}, {chrom: uint8 valid mask}, attrs) from `<genome_dir>/mask.h5`."""
    import h5py
    with h5py.File(Path(genome_dir) / "mask.h5", "r") as f:
        chroms = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        n_bins = {c: int(f[c].shape[0]) for c in chroms}
        valid = {c: f[c][:] for c in chroms}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    order = sorted(n_bins, key=_chrom_key)
    return {c: n_bins[c] for c in order}, valid, attrs


def _chrom_key(c: str):
    t = c[3:] if c.startswith("chr") else c
    return (0, int(t), "") if t.isdigit() else (1, 0, t)


# -- build -----------------------------------------------------------------------------------

def build(genome, src, t146, umap, out, log=print) -> dict:
    import h5py
    src = Path(src)
    src.mkdir(parents=True, exist_ok=True)
    n_bins, valid, gattrs = genome_frame(genome)
    log(f"frame: {len(n_bins)} chromosomes, {sum(n_bins.values())} bins, from {genome}/mask.h5")

    prov = []
    files = {}
    for key, meta in SOURCES.items():
        dest = src / meta["url"].rsplit("/", 1)[1]
        day = fetch(meta["url"], dest)
        files[key] = dest
        prov.append({"annotation": key, "url": meta["url"], "release": meta["release"],
                     "file": str(dest), "bytes": dest.stat().st_size, "md5": file_md5(dest),
                     "download_date": day, "strata": meta["strata"]})
        log(f"source {key}: {dest.name} md5 {prov[-1]['md5']} downloaded {day}")
    umap = Path(umap)
    prov.append({"annotation": "umap_k36", "url": UMAP_URL,
                 "release": "Umap k36 multi-read mappability, hg38 (bismap.hoffmanlab.org); the "
                            "testbed's copy (t138 annot), via t146 map25 arrays",
                 "file": str(umap), "bytes": umap.stat().st_size, "md5": file_md5(umap),
                 "download_date": _t138_download_date(umap), "strata": list(DECILE_STRATA),
                 "derived_from": str(Path(t146))})
    log(f"source umap_k36: md5 {prov[-1]['md5']}")

    masks, rules, checks = {}, {}, {}
    counts = ccre_class_counts(files["ccre"])
    checks["ccre_class_counts"] = counts
    log(f"cCRE classes: {counts}")
    for name, classes in (("promoters", PROMOTER_CLASSES), ("enhancers", ENHANCER_CLASSES)):
        missing = [k for k in classes if k not in counts]
        if missing:
            raise SystemExit(f"cCRE file has no class {missing}; classes are {sorted(counts)}")
        masks[name] = rasterize(read_ccre(files["ccre"], classes), n_bins)
        rules[name] = (f"ccre: cCRE V4 class in {list(classes)}", "any 1 bp overlap")
    iv, n_tx = read_gencode_bodies(files["gencode"])
    checks["gencode_protein_coding_basic_transcripts"] = n_tx
    masks["gene_bodies"] = rasterize(iv, n_bins)
    rules["gene_bodies"] = ("gencode: transcript lines with gene_type protein_coding, TSS to TES",
                            "any 1 bp overlap")
    iv, states = read_chromhmm(files["chromhmm"])
    if len(states) != 9:
        raise SystemExit(f"expected 9 HET states in the ChromHMM file, found {states}")
    checks["chromhmm_het_states"] = states
    masks["heterochromatin"] = rasterize(iv, n_bins)
    rules["heterochromatin"] = (f"chromhmm: states {states}", "any 1 bp overlap")

    bl_path = Path(genome) / "hg38-blacklist.v2.bed"
    bl: dict = {}
    with _open_text(bl_path) as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) >= 3:
                s, e = bl.setdefault(f[0], ([], []))
                s.append(int(f[1]))
                e.append(int(f[2]))
    bad = rasterize(bl, n_bins)
    t146 = Path(t146)
    map25 = {c: np.load(t146 / f"map25__{c}.npy") for c in DECILE_CHROMS}
    for c in DECILE_CHROMS:
        if map25[c].shape[0] != n_bins[c]:
            raise SystemExit(f"t146 map25 {c} has {map25[c].shape[0]} bins, frame has {n_bins[c]}")
    dec, info = block_deciles(map25, bad, n_bins)
    agree = {}
    for c in DECILE_CHROMS:
        p = t146 / f"dec__{c}.npy"
        if p.is_file():
            agree[c] = int((np.load(p) != dec[c]).sum())
    info["t146_dec_mismatched_bins"] = agree
    info["blacklist"] = str(bl_path)
    checks["mappability"] = info
    log(f"mappability: {info['blocks_per_decile']} blocks per decile; t146 mismatched bins "
        f"total {sum(agree.values())} over {len(agree)} chromosomes")
    for k, name in enumerate(DECILE_STRATA):
        masks[name] = {c: (dec[c] == k).astype(np.uint8) if c in dec
                       else np.zeros(n, np.uint8) for c, n in n_bins.items()}
        rules[name] = (f"umap_k36: decile {k} of 1 kb block mean (t146 rule), chr1-22 and chrX",
                       "whole 1 kb blocks; blocks with a blacklisted bin dropped")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema"] = 1
        f.attrs["resolution"] = RES
        f.attrs["sources"] = json.dumps(prov)
        f.attrs["checks"] = json.dumps(checks)
        f.attrs["genome_dir"] = str(genome)
        f.attrs["genome_fasta_sha256"] = str(gattrs.get("fasta_sha256", ""))
        f.attrs["n_bins"] = json.dumps(n_bins)
        f.attrs["built"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        for name, m in masks.items():
            g = f.create_group(name)
            g.attrs["source"], g.attrs["rule"] = rules[name]
            for c, n in n_bins.items():
                g.create_dataset(c, data=m[c].astype(np.uint8), chunks=(min(CHUNK_BINS, n),),
                                 compression="gzip", compression_opts=4)
    tmp.rename(out)
    log(f"wrote {out}")
    return {"strata": list(masks), "sources": prov, "checks": checks}


def _t138_download_date(umap: Path) -> str:
    """t138's annot dir records no date; report the file's mtime date (UTC)."""
    ts = umap.stat().st_mtime
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat() + " (file mtime)"


# -- read side -------------------------------------------------------------------------------

def load_strata(path, names) -> dict:
    """{name: {chrom: bool ndarray}} for the requested strata."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for n in names:
            if n not in f:
                raise KeyError(f"stratum {n!r} not in {path}; have {sorted(f.keys())}")
            out[n] = {c: f[n][c][:].astype(bool) for c in f[n].keys()}
    return out


def fractions(path, genome=None) -> list:
    """[(stratum, covered bins, fraction of all bins, fraction of valid bins or None)]."""
    import h5py
    rows = []
    with h5py.File(path, "r") as f:
        gdir = genome or f.attrs.get("genome_dir")
        valid = None
        if gdir and (Path(gdir) / "mask.h5").is_file():
            _, valid, _ = genome_frame(gdir)
        for name in f.keys():
            tot = cov = vtot = vcov = 0
            for c in f[name].keys():
                m = f[name][c][:].astype(bool)
                tot += m.size
                cov += int(m.sum())
                if valid is not None:
                    v = valid[c].astype(bool)
                    vtot += int(v.sum())
                    vcov += int((m & v).sum())
            rows.append((name, cov, cov / tot if tot else 0.0,
                         (vcov / vtot) if valid is not None and vtot else None))
    order = {n: i for i, n in enumerate(["promoters", "enhancers", "gene_bodies",
                                         "heterochromatin", *DECILE_STRATA])}
    rows.sort(key=lambda r: order.get(r[0], 99))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--genome", required=True, help="genome layer dir (mask.h5, blacklist bed)")
    b.add_argument("--src", required=True, help="download dir for the annotation files")
    b.add_argument("--t146", required=True, help="t146 features dir (map25__/dec__ arrays)")
    b.add_argument("--umap", required=True, help="the testbed's k36.umap.bedgraph.gz (for md5)")
    b.add_argument("--out", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--masks", required=True)
    ls.add_argument("--genome", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.genome, a.src, a.t146, a.umap, a.out)
        a.masks, a.genome = a.out, a.genome
    import h5py
    with h5py.File(a.masks, "r") as f:
        prov = json.loads(f.attrs["sources"])
    for p in prov:
        print(f"source {p['annotation']}: {p['release']} | {p['url']} | md5 {p['md5']} | "
              f"{p['download_date']}")
    print("stratum\tcovered_bins\tfraction_all_bins\tfraction_valid_bins")
    for name, cov, fa, fv in fractions(a.masks, a.genome):
        print(f"{name}\t{cov}\t{fa:.4f}\t{'' if fv is None else f'{fv:.4f}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
