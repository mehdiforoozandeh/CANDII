"""The shared genome layer: `genome/dna.h5` (D10), `genome/mask.h5` (D11) and window
eligibility (D12).

One genome layer serves every corpus — `eic/` and `merged/` are siblings of `genome/`, not
owners of it. Nothing here knows about biosamples, tracks or assays.

```
hg38.fa               --[build_dna]---->  genome/dna.h5    (chr_len,)  uint8 base codes
dna.h5 + blacklist    --[build_mask]--->  genome/mask.h5   (n_bins,)   uint8 0/1
mask.h5               --[eligible_starts]-> the window starts a regime may sample
```

Decisions implemented here (`STORE_PLAN.md` §3):

* **D10** `dna.h5`: one `uint8` array per chromosome at **base-pair** resolution, length
  `chr_len`, codes `A=0 C=1 G=2 T=3 N=4`, uppercase-folded. `fasta_sha256` in the root attrs so
  a build mismatch is loud rather than silent.
* **D11** `mask.h5`: one `uint8` 0/1 array per chromosome at **bin** resolution. A bin is
  invalid (0) if it contains any `N` **or** overlaps a blacklist interval.
* **D12** a window of `L` bins starting at bin `s` is eligible iff
  `mask[s : s + L].mean() >= min_valid_frac`, default `0.9`.
* **D13** `n_bins = floor(chr_len / resolution)`, via `layout.py::n_bins_for`. Never re-derived.

Everything about paths, chunking, compression, `n_bins` and chromosome order comes from
`layout.py`. If a rule appears both here and there, this copy is the bug.

Choices this module made where `STORE_PLAN.md` is silent — each is recorded in the root attrs
so a reader never has to guess:

* **Ambiguous IUPAC letters** (`R Y S W K M B D H V`, either case) are coded `N = 4`, because
  D11 makes `N` mean "this base is not a determined A/C/G/T" and an ambiguity code is exactly
  that. They are **counted** per letter into the `iupac_counts` root attr, so folding a large
  number of them is visible instead of silent. Soft-masked lowercase `acgt` are real bases and
  fold to their own codes (D10 says uppercase-folded, not "repeat-masked out").
* **The chromosome set is `chrom_sizes.json`'s, verbatim** — chr1–chr22, chrX, chrY for the t4
  file. Alts, randoms, `chrUn_*` and `chrM` are in the FASTA and are skipped: no corpus track
  covers them, so a mask bin there could never be sampled.
* **`mask.h5` also carries `fasta_sha256`**, copied from the `dna.h5` it was built from, so a
  mask built against a different genome than the DNA beside it is caught at open time.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

from candi.store import layout as L
from candi.store.layout import StoreError

__all__ = [
    "GENOME_BUILD",
    "DNA_CODES",
    "N_CODE",
    "DNA_CHUNK_BP",
    "IUPAC_LETTERS",
    "DEFAULT_MIN_VALID_FRAC",
    "ATTR_BUILD",
    "ATTR_FASTA_SHA256",
    "ATTR_FASTA_PATH",
    "ATTR_CODES",
    "ATTR_CHROM_LENGTHS",
    "ATTR_IUPAC_COUNTS",
    "ATTR_BLACKLIST_SHA256",
    "ATTR_BLACKLIST_SOURCE",
    "ATTR_RULE",
    "MASK_RULE",
    "sha256_file",
    "load_genome_chrom_sizes",
    "encode_bases",
    "read_blacklist",
    "blacklist_bin_flags",
    "build_dna",
    "build_mask",
    "build_genome",
    "window_valid_counts",
    "eligible_window_mask",
    "eligible_starts",
    "count_eligible",
    "GenomeLayer",
    "verify_genome",
    "genome_report",
]

# ---------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------

GENOME_BUILD = "GRCh38"

#: D10 — the base code table. Written into the `codes` root attr of `dna.h5` verbatim, so a
#: reader never has to trust this constant.
DNA_CODES = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
N_CODE = DNA_CODES["N"]

#: D10 — `dna.h5` chunk, in base pairs. 25600 bp is exactly `CHUNK_BINS * DEFAULT_RESOLUTION`,
#: so one DNA chunk covers the same span as one chunk of a counts/peaks/pval dataset.
DNA_CHUNK_BP = L.CHUNK_BINS * L.DEFAULT_RESOLUTION  # 25600

#: Ambiguity codes. Folded to `N` (see the module docstring) and counted.
IUPAC_LETTERS = ("R", "Y", "S", "W", "K", "M", "B", "D", "H", "V")

#: D12 — the default eligibility threshold. A regime file may override it; nothing is baked in.
DEFAULT_MIN_VALID_FRAC = 0.9

# root attrs of the genome layer. The per-biosample attr set is `layout.py::root_attrs`; the
# genome layer has no biosample, kind, tracks or control column, so it gets its own small set.
ATTR_BUILD = "build"
ATTR_FASTA_SHA256 = "fasta_sha256"
ATTR_FASTA_PATH = "fasta_path"
ATTR_CODES = "codes"                       # JSON dict, DNA_CODES
ATTR_CHROM_LENGTHS = "chrom_lengths"       # JSON dict {chrom: chr_len}
ATTR_IUPAC_COUNTS = "iupac_counts"         # JSON dict {letter: count}, dna.h5 only
ATTR_BLACKLIST_SHA256 = "blacklist_sha256"
ATTR_BLACKLIST_SOURCE = "blacklist_source"
ATTR_RULE = "rule"

MASK_RULE = (
    "mask[b] = 1 iff bin b contains no N base AND does not overlap any blacklist interval; "
    "bin b covers [b*resolution, (b+1)*resolution) bp; n_bins = floor(chr_len / resolution)"
)

_GENOME_JSON_ATTRS = frozenset(
    {ATTR_CODES, ATTR_CHROM_LENGTHS, ATTR_IUPAC_COUNTS, L.ATTR_N_BINS}
)


# ---------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------


def sha256_file(path: Path | str, block: int = 1 << 22) -> str:
    """Streaming sha256 of a file. The provenance hash of both the FASTA and the blacklist."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_genome_chrom_sizes(path: Path | str) -> dict:
    """`chrom_sizes.json` -> `{chrom: length}`, tolerating the t4 file's wrapper object.

    `layout.py::load_chrom_sizes` reads a **flat** `{chrom: length}` JSON object or a two-column
    `.sizes` TSV. The file t4 wrote at `genome/chrom_sizes.json` is richer — it wraps the map
    under a `chrom_sizes` key alongside `build`, `resolution`, `source` and a precomputed
    `n_bins`. This unwraps that form and delegates everything else to `layout.py`, so there is
    still exactly one parser for the flat forms.
    """
    p = Path(path)
    if p.suffix == ".json" and p.is_file():
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("chrom_sizes"), dict):
            return {str(k): int(v) for k, v in obj["chrom_sizes"].items()}
    return L.load_chrom_sizes(p)


def _base_lut() -> np.ndarray:
    """A 256-entry byte -> code table. Everything not `ACGTacgt` becomes `N`."""
    lut = np.full(256, N_CODE, dtype=np.uint8)
    for letter, code in DNA_CODES.items():
        if letter == "N":
            continue
        lut[ord(letter)] = code
        lut[ord(letter.lower())] = code
    return lut


_LUT = _base_lut()


def encode_bases(raw: "np.ndarray | bytes") -> tuple:
    """Raw FASTA sequence bytes -> `(codes uint8, byte_histogram)`.

    Whitespace (anything `<= 0x20`: newline, carriage return, space, tab) is dropped, so a
    caller may hand over a whole record including its line breaks. Case is folded; ambiguity
    codes fold to `N` and stay visible in the histogram.
    """
    arr = raw if isinstance(raw, np.ndarray) else np.frombuffer(raw, dtype=np.uint8)
    bases = arr[arr > 0x20]
    hist = np.bincount(bases, minlength=256)
    return _LUT[bases], hist


def _iupac_counts(hist: np.ndarray) -> dict:
    """The per-letter tally of ambiguity codes folded to `N`, plus an `other` bucket."""
    out = {}
    accounted = set()
    for letter in ("A", "C", "G", "T", "N"):
        accounted.update({ord(letter), ord(letter.lower())})
    for letter in IUPAC_LETTERS:
        n = int(hist[ord(letter)]) + int(hist[ord(letter.lower())])
        accounted.update({ord(letter), ord(letter.lower())})
        if n:
            out[letter] = n
    other = int(sum(int(hist[b]) for b in range(256) if b > 0x20 and b not in accounted))
    if other:
        out["other"] = other
    return out


def _write_genome_attrs(h5obj, attrs: Mapping) -> None:
    for k, v in attrs.items():
        h5obj.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v


def _read_genome_attrs(h5obj) -> dict:
    out = {}
    for k, v in h5obj.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, np.generic):
            v = v.item()
        if k in _GENOME_JSON_ATTRS and isinstance(v, str):
            v = json.loads(v)
        out[k] = v
    return out


def _refuse_existing(out: Path, overwrite: bool) -> None:
    if out.exists() and not overwrite:
        raise StoreError(f"{out} already exists; pass --overwrite to replace it")


# ---------------------------------------------------------------------------------------------
# FASTA -> dna.h5   (D10)
# ---------------------------------------------------------------------------------------------


def _index_fasta(mm: mmap.mmap) -> dict:
    """`{record_name: (seq_start_byte, seq_end_byte)}` for every record in the FASTA.

    The record name is the header up to the first whitespace, which is what every hg38 build
    uses (`>chr1  AC:CM000663.2 …` and a bare `>chr1` both give `chr1`).
    """
    size = mm.size()
    if size == 0:
        raise StoreError("empty FASTA")
    heads = []
    if mm[0:1] == b">":
        heads.append(0)
    p = mm.find(b"\n>")
    while p >= 0:
        heads.append(p + 1)
        p = mm.find(b"\n>", p + 1)
    if not heads:
        raise StoreError("no '>' header line in the FASTA; is this really a FASTA?")
    records = {}
    for i, h in enumerate(heads):
        nl = mm.find(b"\n", h)
        if nl < 0:
            nl = size
        name = mm[h + 1 : nl].split(None, 1)[0].decode("ascii") if nl > h + 1 else ""
        if not name:
            raise StoreError(f"FASTA record at byte {h} has an empty header")
        if name in records:
            raise StoreError(f"FASTA has two records named {name!r}; refusing to guess")
        end = heads[i + 1] if i + 1 < len(heads) else size
        records[name] = (min(nl + 1, size), end)
    return records


def build_dna(
    fasta: Path | str,
    out: Path | str,
    chrom_sizes: Mapping[str, int],
    *,
    chroms: Optional[Sequence[str]] = None,
    build: str = GENOME_BUILD,
    fasta_sha256: Optional[str] = None,
    overwrite: bool = False,
    progress=None,
) -> dict:
    """D10 — FASTA -> `dna.h5`, one `uint8` array per chromosome of length `chr_len`.

    `chrom_sizes` selects the chromosomes and is also the check: a record whose parsed length
    disagrees with `chrom_sizes` raises, because that is the cheapest way a wrong-build FASTA
    announces itself. `fasta_sha256` is the second: pass one to have it verified here, or let
    it be computed and written into the root attrs.

    Returns a per-chromosome summary dict; `bytes` is the file size on disk.
    """
    fasta = Path(fasta)
    out = Path(out)
    if fasta.suffix == ".gz":
        raise StoreError(
            f"{fasta}: build_dna needs an uncompressed FASTA (mmap random access). "
            f"Use the .fa beside it, or gunzip first."
        )
    _refuse_existing(out, overwrite)

    observed = sha256_file(fasta)
    if fasta_sha256 is not None and observed != fasta_sha256:
        raise StoreError(
            f"{fasta}: fasta_sha256 mismatch — expected {fasta_sha256}, got {observed}. "
            f"This is a different genome build or a truncated file; refusing to build dna.h5."
        )

    wanted = L.sort_chroms(chroms if chroms is not None else chrom_sizes.keys())
    missing_sizes = [c for c in wanted if c not in chrom_sizes]
    if missing_sizes:
        raise StoreError(f"no chrom_sizes entry for {missing_sizes}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    per_chrom = {}
    iupac_total = {}

    with open(fasta, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            records = _index_fasta(mm)
            absent = [c for c in wanted if c not in records]
            if absent:
                raise StoreError(
                    f"{fasta}: no FASTA record for {absent}. Present records include "
                    f"{sorted(records)[:8]}…"
                )
            with h5py.File(tmp, "w") as f:
                for chrom in wanted:
                    start, end = records[chrom]
                    # `mm[start:end]` copies the record's bytes out; a np.frombuffer VIEW on the
                    # mmap itself would keep an exported pointer and make `mm.close()` raise.
                    codes, hist = encode_bases(mm[start:end])
                    n = int(codes.shape[0])
                    expected = int(chrom_sizes[chrom])
                    if n != expected:
                        raise StoreError(
                            f"{fasta}:{chrom} is {n} bp but chrom_sizes says {expected}. "
                            f"The FASTA and the chrom sizes are different builds."
                        )
                    ds = f.create_dataset(
                        chrom,
                        shape=(n,),
                        dtype=np.uint8,
                        chunks=(min(DNA_CHUNK_BP, n),),
                        compression="gzip",
                        compression_opts=L.GZIP_LEVEL,
                        shuffle=False,
                    )
                    for a in range(0, n, 1 << 26):  # 64 Mbp slabs — bounded peak memory
                        b = min(a + (1 << 26), n)
                        ds[a:b] = codes[a:b]
                    amb = _iupac_counts(hist)
                    for k, v in amb.items():
                        iupac_total[k] = iupac_total.get(k, 0) + v
                    per_chrom[chrom] = {
                        "chr_len": n,
                        "n_A": int(hist[ord("A")] + hist[ord("a")]),
                        "n_C": int(hist[ord("C")] + hist[ord("c")]),
                        "n_G": int(hist[ord("G")] + hist[ord("g")]),
                        "n_T": int(hist[ord("T")] + hist[ord("t")]),
                        "n_N": int(hist[ord("N")] + hist[ord("n")]),
                        "n_softmasked": int(
                            sum(hist[ord(c)] for c in "acgt")
                        ),
                        "n_ambiguous_folded": int(sum(amb.values())),
                        "iupac_counts": amb,
                    }
                    del codes
                    if progress:
                        progress(f"dna {chrom}: {n} bp, N={per_chrom[chrom]['n_N']}")
                _write_genome_attrs(
                    f,
                    {
                        L.ATTR_SCHEMA: int(L.SCHEMA_VERSION),
                        ATTR_BUILD: str(build),
                        ATTR_FASTA_SHA256: observed,
                        ATTR_FASTA_PATH: str(fasta),
                        ATTR_CODES: dict(DNA_CODES),
                        ATTR_CHROM_LENGTHS: {c: per_chrom[c]["chr_len"] for c in wanted},
                        ATTR_IUPAC_COUNTS: iupac_total,
                        L.ATTR_KIT_VERSION: L.kit_version(),
                        L.ATTR_BUILT_UTC: L.utc_now(),
                    },
                )
        finally:
            mm.close()
    os.replace(tmp, out)
    return {
        "path": str(out),
        "bytes": out.stat().st_size,
        "fasta": str(fasta),
        "fasta_sha256": observed,
        "build": build,
        "chroms": wanted,
        "iupac_counts": iupac_total,
        "per_chrom": per_chrom,
    }


# ---------------------------------------------------------------------------------------------
# blacklist -> bin flags   (D11)
# ---------------------------------------------------------------------------------------------


def read_blacklist(path: Path | str) -> dict:
    """A BED -> `{chrom: (n, 2) int64}` of **merged, sorted, half-open** `[start, end)` intervals.

    The ENCODE hg38 blacklist v2 is delivered in **lexicographic** chromosome order
    (`chr1, chr10, chr11, …, chrY`), so this groups by chromosome and never assumes the file is
    sorted or grouped. It is also already merged and non-overlapping (636 intervals,
    227,162,400 bp) — merged again here anyway, because relying on that is how a re-download
    with a different provenance becomes a silent under-count.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"blacklist bed not found: {path}")
    raw: dict = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith(("#", "track", "browser")):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            raise StoreError(f"{path}:{lineno}: expected '<chrom>\\t<start>\\t<end>', got {line!r}")
        try:
            start, end = int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise StoreError(f"{path}:{lineno}: non-integer interval bounds in {line!r}") from exc
        if end < start:
            raise StoreError(f"{path}:{lineno}: end {end} < start {start}")
        if end == start:
            continue
        raw.setdefault(parts[0], []).append((start, end))
    out = {}
    for chrom, ivals in raw.items():
        arr = np.asarray(sorted(ivals), dtype=np.int64)
        merged = [arr[0].tolist()]
        for s, e in arr[1:].tolist():
            if s <= merged[-1][1]:          # touching or overlapping -> one interval
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[chrom] = np.asarray(merged, dtype=np.int64)
    return out


def blacklist_bin_flags(
    intervals: np.ndarray,
    n_bins: int,
    resolution: int = L.DEFAULT_RESOLUTION,
    chrom_len: Optional[int] = None,
) -> np.ndarray:
    """`(n_bins,)` bool — True where a bin **overlaps** any interval, by even a single bp.

    Bin `b` covers `[b*resolution, (b+1)*resolution)`. An interval `[s, e)` therefore touches
    bins `floor(s/resolution) … floor((e-1)/resolution)` inclusive. A difference array turns
    the whole set into one cumulative sum, so 636 intervals cost one pass, not 636.
    """
    flags = np.zeros(int(n_bins) + 1, dtype=np.int64)
    if intervals is None or len(intervals) == 0 or n_bins <= 0:
        return np.zeros(int(n_bins), dtype=bool)
    iv = np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
    s = iv[:, 0]
    e = iv[:, 1]
    if chrom_len is not None:
        s = np.clip(s, 0, int(chrom_len))
        e = np.clip(e, 0, int(chrom_len))
    keep = e > s
    s, e = s[keep], e[keep]
    if s.size == 0:
        return np.zeros(int(n_bins), dtype=bool)
    b0 = np.clip(s // resolution, 0, n_bins)
    b1 = np.clip((e - 1) // resolution, -1, n_bins - 1)
    ok = b1 >= b0
    b0, b1 = b0[ok], b1[ok]
    np.add.at(flags, b0, 1)
    np.add.at(flags, b1 + 1, -1)
    return np.cumsum(flags[:-1]) > 0


# ---------------------------------------------------------------------------------------------
# dna.h5 + blacklist -> mask.h5   (D11)
# ---------------------------------------------------------------------------------------------


def build_mask(
    dna: Path | str,
    blacklist: Path | str,
    out: Path | str,
    chrom_sizes: Mapping[str, int],
    *,
    chroms: Optional[Sequence[str]] = None,
    resolution: int = L.DEFAULT_RESOLUTION,
    blacklist_source: Optional[str] = None,
    overwrite: bool = False,
    progress=None,
) -> dict:
    """D11 — `dna.h5` + blacklist BED -> `mask.h5`, one `uint8` 0/1 array per chromosome.

    The N flags come from `dna.h5` rather than a second FASTA parse, so the mask cannot
    disagree with the DNA beside it; the `fasta_sha256` it copies out of `dna.h5` makes that
    provenance checkable at open time (`GenomeLayer`).

    The summary counts, per chromosome, bins invalid **because of N**, **because of the
    blacklist**, and **because of both** — the three are reported separately because the
    overlap is large (the blacklist covers centromeres, which are also N) and a single
    "invalid" number hides which rule is doing the work.
    """
    dna = Path(dna)
    out = Path(out)
    _refuse_existing(out, overwrite)
    bl_sha = sha256_file(blacklist)
    bl = read_blacklist(blacklist)

    with h5py.File(dna, "r") as fd:
        dna_attrs = _read_genome_attrs(fd)
        wanted = L.sort_chroms(chroms if chroms is not None else chrom_sizes.keys())
        absent = [c for c in wanted if c not in fd]
        if absent:
            raise StoreError(f"{dna}: no DNA dataset for {absent}")

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".h5.tmp")
        per_chrom = {}
        n_bins_all = {}
        with h5py.File(tmp, "w") as f:
            for chrom in wanted:
                chrom_len = int(chrom_sizes[chrom])
                if fd[chrom].shape[0] != chrom_len:
                    raise StoreError(
                        f"{dna}:{chrom} is {fd[chrom].shape[0]} bp but chrom_sizes says "
                        f"{chrom_len}; dna.h5 and chrom_sizes are different builds."
                    )
                n_bins = L.n_bins_for(chrom_len, resolution)        # D13
                n_bins_all[chrom] = n_bins

                has_n = np.zeros(n_bins, dtype=bool)
                step = 4_000_000                                    # 100 Mbp per slab
                for a in range(0, n_bins, step):
                    b = min(a + step, n_bins)
                    seg = fd[chrom][a * resolution : b * resolution]
                    has_n[a:b] = (seg.reshape(b - a, resolution) == N_CODE).any(axis=1)

                in_bl = blacklist_bin_flags(
                    bl.get(chrom), n_bins, resolution, chrom_len=chrom_len
                )
                valid = ~(has_n | in_bl)
                ds = f.create_dataset(
                    chrom,
                    shape=(n_bins,),
                    dtype=np.uint8,
                    chunks=(min(L.CHUNK_BINS, n_bins),),
                    compression="gzip",
                    compression_opts=L.GZIP_LEVEL,
                    shuffle=False,
                )
                ds[:] = valid.astype(np.uint8)
                per_chrom[chrom] = {
                    "n_bins": int(n_bins),
                    "invalid_n": int(has_n.sum()),
                    "invalid_blacklist": int(in_bl.sum()),
                    "invalid_both": int((has_n & in_bl).sum()),
                    "invalid_total": int((has_n | in_bl).sum()),
                    "n_valid": int(valid.sum()),
                    "valid_frac": float(valid.mean()) if n_bins else 0.0,
                    "blacklist_intervals": int(len(bl.get(chrom, ()))),
                }
                if progress:
                    progress(
                        f"mask {chrom}: {per_chrom[chrom]['valid_frac']:.4f} valid "
                        f"({per_chrom[chrom]['n_valid']}/{n_bins})"
                    )
            _write_genome_attrs(
                f,
                {
                    L.ATTR_SCHEMA: int(L.SCHEMA_VERSION),
                    ATTR_BUILD: str(dna_attrs.get(ATTR_BUILD, GENOME_BUILD)),
                    L.ATTR_RESOLUTION: int(resolution),
                    ATTR_BLACKLIST_SHA256: bl_sha,
                    ATTR_BLACKLIST_SOURCE: str(
                        blacklist_source if blacklist_source is not None else blacklist
                    ),
                    ATTR_RULE: MASK_RULE,
                    ATTR_FASTA_SHA256: str(dna_attrs.get(ATTR_FASTA_SHA256, "")),
                    L.ATTR_N_BINS: {c: int(v) for c, v in n_bins_all.items()},
                    L.ATTR_KIT_VERSION: L.kit_version(),
                    L.ATTR_BUILT_UTC: L.utc_now(),
                },
            )
    os.replace(tmp, out)
    total_bins = sum(v["n_bins"] for v in per_chrom.values())
    total_valid = sum(v["n_valid"] for v in per_chrom.values())
    return {
        "path": str(out),
        "bytes": out.stat().st_size,
        "blacklist": str(blacklist),
        "blacklist_sha256": bl_sha,
        "blacklist_intervals": int(sum(len(v) for v in bl.values())),
        "blacklist_bp": int(sum(int((v[:, 1] - v[:, 0]).sum()) for v in bl.values())),
        "resolution": int(resolution),
        "n_bins_total": int(total_bins),
        "n_valid_total": int(total_valid),
        "valid_frac_genome": float(total_valid / total_bins) if total_bins else 0.0,
        "per_chrom": per_chrom,
    }


def build_genome(
    store_root: Path | str,
    fasta: Path | str,
    blacklist: Path | str,
    *,
    chrom_sizes: Optional[Mapping[str, int]] = None,
    chroms: Optional[Sequence[str]] = None,
    resolution: int = L.DEFAULT_RESOLUTION,
    build: str = GENOME_BUILD,
    fasta_sha256: Optional[str] = None,
    blacklist_source: Optional[str] = None,
    what: str = "both",
    overwrite: bool = False,
    progress=None,
) -> dict:
    """Build `genome/dna.h5` then `genome/mask.h5` under `store_root`. Paths from `layout.py`."""
    if what not in ("both", "dna", "mask"):
        raise StoreError(f"unknown what={what!r}; expected 'both', 'dna' or 'mask'")
    store_root = Path(store_root)
    if chrom_sizes is None:
        chrom_sizes = load_genome_chrom_sizes(L.chrom_sizes_path(store_root))
    result = {"store_root": str(store_root)}
    if what in ("both", "dna"):
        result["dna"] = build_dna(
            fasta,
            L.dna_path(store_root),
            chrom_sizes,
            chroms=chroms,
            build=build,
            fasta_sha256=fasta_sha256,
            overwrite=overwrite,
            progress=progress,
        )
    if what in ("both", "mask"):
        result["mask"] = build_mask(
            L.dna_path(store_root),
            blacklist,
            L.mask_path(store_root),
            chrom_sizes,
            chroms=chroms,
            resolution=resolution,
            blacklist_source=blacklist_source,
            overwrite=overwrite,
            progress=progress,
        )
    return result


# ---------------------------------------------------------------------------------------------
# window eligibility   (D12)  —  the primitive `regime.py` imports
# ---------------------------------------------------------------------------------------------


def window_valid_counts(mask: np.ndarray, context_bins: int) -> np.ndarray:
    """`(n_bins - L + 1,)` int64 — the number of valid bins in the window starting at each bin.

    One cumulative sum over the mask, not a loop over windows: `O(n_bins)` regardless of `L`.
    Empty when the chromosome is shorter than one window.
    """
    L_ = int(context_bins)
    if L_ <= 0:
        raise StoreError(f"context_bins must be positive, got {context_bins}")
    m = np.asarray(mask)
    if m.ndim != 1:
        raise StoreError(f"mask must be 1-D, got shape {m.shape}")
    if m.shape[0] < L_:
        return np.zeros(0, dtype=np.int64)
    cs = np.empty(m.shape[0] + 1, dtype=np.int64)
    cs[0] = 0
    np.cumsum(m, dtype=np.int64, out=cs[1:])
    return cs[L_:] - cs[:-L_]


def eligible_window_mask(
    mask: np.ndarray,
    context_bins: int,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
) -> np.ndarray:
    """D12 — bool over every start `s`: `mask[s : s+L].mean() >= min_valid_frac`.

    Compared as `valid_count >= min_valid_frac * L` with a `1e-9` slack, so a threshold that
    lands exactly on an integer (`L=100, frac=0.9` -> 90) is inclusive rather than at the mercy
    of the last bit of the product.
    """
    frac = float(min_valid_frac)
    if not (0.0 <= frac <= 1.0):
        raise StoreError(f"min_valid_frac must be in [0, 1], got {min_valid_frac}")
    counts = window_valid_counts(mask, context_bins)
    return counts >= (frac * int(context_bins) - 1e-9)


def eligible_starts(
    mask: np.ndarray,
    context_bins: int,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
    *,
    stride: Optional[int] = None,
    offset: int = 0,
) -> np.ndarray:
    """The eligible window start bins, as an int64 array.

    `stride=None` means every start (`stride=1`). `stride=context_bins` is the tiled,
    non-overlapping plan — the `window_plan.type = "tile"` of the regime file. `offset` shifts
    the tiling, which is how a regime gets a second, disjoint tiling of the same chromosome.
    """
    ok = eligible_window_mask(mask, context_bins, min_valid_frac)
    if ok.size == 0:
        return np.zeros(0, dtype=np.int64)
    step = int(context_bins if stride is None else stride)
    if step <= 0:
        raise StoreError(f"stride must be positive, got {stride}")
    if step == 1 and offset == 0:
        return np.flatnonzero(ok).astype(np.int64)
    cand = np.arange(int(offset), ok.size, step, dtype=np.int64)
    return cand[ok[cand]]


def count_eligible(
    mask: np.ndarray,
    context_bins: int,
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
    *,
    stride: Optional[int] = None,
    offset: int = 0,
) -> int:
    """`len(eligible_starts(...))` without materializing the index array for `stride=1`."""
    ok = eligible_window_mask(mask, context_bins, min_valid_frac)
    if ok.size == 0:
        return 0
    step = int(context_bins if stride is None else stride)
    if step <= 0:
        raise StoreError(f"stride must be positive, got {stride}")
    if step == 1 and offset == 0:
        return int(ok.sum())
    return int(ok[int(offset) :: step].sum())


# ---------------------------------------------------------------------------------------------
# reading the genome layer
# ---------------------------------------------------------------------------------------------


class GenomeLayer:
    """Read-side handle on `genome/{dna.h5, mask.h5, chrom_sizes.json}`.

    ```python
    g = GenomeLayer("/…/CANDI_STORE/genome", fasta_sha256=manifest["genome"]["fasta_sha256"])
    seq  = g.dna("chr1", 1_000_000, 1_019_200)      # (19200,) uint8 codes
    m    = g.mask("chr1")                            # (n_bins,) uint8 0/1, cached
    idx  = g.eligible_starts("chr1", 768)            # int64 start bins
    ```

    **This is where the D10 build-mismatch check lives.** Two hashes are checked at open:
    `mask.h5`'s `fasta_sha256` against `dna.h5`'s — always, no argument needed — and, when the
    caller passes one (the corpus manifest's `genome.fasta_sha256`), the expected FASTA hash
    against what `dna.h5` records. Either mismatch raises `StoreError` before a single base is
    read, which is the entire point of writing the hash down. `verify_genome` is the same check
    as a problem list for the CLI.
    """

    def __init__(
        self,
        genome_dir: Path | str,
        *,
        fasta_sha256: Optional[str] = None,
        blacklist_sha256: Optional[str] = None,
        min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
    ) -> None:
        self.dir = Path(genome_dir)
        self.min_valid_frac = float(min_valid_frac)
        self._dna_path = self.dir / "dna.h5"
        self._mask_path = self.dir / "mask.h5"
        for p in (self._dna_path, self._mask_path):
            if not p.is_file():
                raise StoreError(f"{p} does not exist; build it with `build-genome` (t7)")
        self._dna = h5py.File(self._dna_path, "r")
        self._mask = h5py.File(self._mask_path, "r")
        self.dna_attrs = _read_genome_attrs(self._dna)
        self.mask_attrs = _read_genome_attrs(self._mask)
        self._mask_cache: dict = {}

        built = str(self.dna_attrs.get(ATTR_FASTA_SHA256, ""))
        mask_built = str(self.mask_attrs.get(ATTR_FASTA_SHA256, ""))
        if mask_built and built and mask_built != built:
            self.close()
            raise StoreError(
                f"{self._mask_path} was built from a genome with fasta_sha256 {mask_built} but "
                f"{self._dna_path} has {built}. The mask and the DNA are different builds; "
                f"rebuild the mask from this dna.h5."
            )
        if fasta_sha256 is not None and built != fasta_sha256:
            self.close()
            raise StoreError(
                f"{self._dna_path}: fasta_sha256 is {built or '<absent>'} but the caller expects "
                f"{fasta_sha256}. The store and the genome layer are different builds — every "
                f"coordinate in the store means something else. Refusing to read."
            )
        if blacklist_sha256 is not None:
            have = str(self.mask_attrs.get(ATTR_BLACKLIST_SHA256, ""))
            if have != blacklist_sha256:
                self.close()
                raise StoreError(
                    f"{self._mask_path}: blacklist_sha256 is {have or '<absent>'} but the caller "
                    f"expects {blacklist_sha256}; the mask was built from a different blacklist."
                )

        self.build = str(self.dna_attrs.get(ATTR_BUILD, GENOME_BUILD))
        self.codes = dict(self.dna_attrs.get(ATTR_CODES, DNA_CODES))
        self.resolution = int(self.mask_attrs.get(L.ATTR_RESOLUTION, L.DEFAULT_RESOLUTION))
        self.chrom_lengths = {
            str(k): int(v) for k, v in dict(self.dna_attrs.get(ATTR_CHROM_LENGTHS, {})).items()
        }
        self.n_bins = {
            str(k): int(v) for k, v in dict(self.mask_attrs.get(L.ATTR_N_BINS, {})).items()
        }

    # -- lifecycle ----------------------------------------------------------------------------

    def close(self) -> None:
        for h in ("_dna", "_mask"):
            f = getattr(self, h, None)
            if f is not None:
                try:
                    f.close()
                except Exception:  # pragma: no cover
                    pass
                setattr(self, h, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def chroms(self) -> list:
        return L.sort_chroms(self.n_bins.keys() or self.chrom_lengths.keys())

    # -- data ---------------------------------------------------------------------------------

    def dna(self, chrom: str, start_bp: int = 0, end_bp: Optional[int] = None) -> np.ndarray:
        """`(end_bp - start_bp,)` uint8 base codes. Coordinates are **base pairs**, not bins."""
        if chrom not in self._dna:
            raise StoreError(f"{self._dna_path}: no DNA for {chrom!r}; have {self.chroms}")
        ds = self._dna[chrom]
        stop = int(ds.shape[0] if end_bp is None else end_bp)
        start = int(start_bp)
        if start < 0 or stop > ds.shape[0] or stop < start:
            raise StoreError(
                f"{chrom}[{start}:{stop}] is outside [0, {ds.shape[0]}) bp"
            )
        return ds[start:stop]

    def dna_for_bins(self, chrom: str, start_bin: int, end_bin: int) -> np.ndarray:
        """The DNA under a bin window — `((end_bin - start_bin) * resolution,)` uint8."""
        return self.dna(chrom, int(start_bin) * self.resolution, int(end_bin) * self.resolution)

    def mask(self, chrom: str) -> np.ndarray:
        """The whole `(n_bins,)` uint8 0/1 mask, cached — the genome is ~124 MB of it."""
        if chrom not in self._mask_cache:
            if chrom not in self._mask:
                raise StoreError(f"{self._mask_path}: no mask for {chrom!r}; have {self.chroms}")
            self._mask_cache[chrom] = self._mask[chrom][:]
        return self._mask_cache[chrom]

    def mask_slice(self, chrom: str, start_bin: int, end_bin: int) -> np.ndarray:
        return self.mask(chrom)[int(start_bin) : int(end_bin)]

    # -- eligibility (D12) --------------------------------------------------------------------

    def eligible_starts(
        self,
        chrom: str,
        context_bins: int,
        min_valid_frac: Optional[float] = None,
        *,
        stride: Optional[int] = None,
        offset: int = 0,
    ) -> np.ndarray:
        return eligible_starts(
            self.mask(chrom),
            context_bins,
            self.min_valid_frac if min_valid_frac is None else min_valid_frac,
            stride=stride,
            offset=offset,
        )

    def is_eligible(
        self,
        chrom: str,
        start_bin: int,
        context_bins: int,
        min_valid_frac: Optional[float] = None,
    ) -> bool:
        frac = self.min_valid_frac if min_valid_frac is None else min_valid_frac
        w = self.mask_slice(chrom, start_bin, int(start_bin) + int(context_bins))
        if w.shape[0] < int(context_bins):
            return False
        return bool(w.mean() >= frac - 1e-9)

    def count_eligible(
        self,
        chrom: str,
        context_bins: int,
        min_valid_frac: Optional[float] = None,
        *,
        stride: Optional[int] = None,
    ) -> int:
        return count_eligible(
            self.mask(chrom),
            context_bins,
            self.min_valid_frac if min_valid_frac is None else min_valid_frac,
            stride=stride,
        )


def verify_genome(
    genome_dir: Path | str,
    *,
    fasta_sha256: Optional[str] = None,
    blacklist_sha256: Optional[str] = None,
    chrom_sizes: Optional[Mapping[str, int]] = None,
) -> list:
    """Structural check of a built genome layer. Returns a list of problem strings (empty = OK).

    Same hashes `GenomeLayer` refuses on, plus: every mask chromosome has a DNA chromosome,
    and every `n_bins` is `floor(chr_len / resolution)` (D13).
    """
    problems: list = []
    try:
        g = GenomeLayer(genome_dir, fasta_sha256=fasta_sha256, blacklist_sha256=blacklist_sha256)
    except StoreError as exc:
        return [str(exc)]
    try:
        for chrom, n_bins in sorted(g.n_bins.items()):
            chrom_len = g.chrom_lengths.get(chrom)
            if chrom_len is None:
                problems.append(f"{chrom}: in mask.h5 but not in dna.h5")
                continue
            want = L.n_bins_for(chrom_len, g.resolution)
            if want != n_bins:
                problems.append(
                    f"{chrom}: n_bins {n_bins} but floor({chrom_len}/{g.resolution}) = {want}"
                )
            if chrom_sizes is not None and chrom in chrom_sizes:
                if int(chrom_sizes[chrom]) != chrom_len:
                    problems.append(
                        f"{chrom}: dna.h5 length {chrom_len} but chrom_sizes says "
                        f"{int(chrom_sizes[chrom])}"
                    )
        if not g.mask_attrs.get(ATTR_RULE):
            problems.append("mask.h5 has no `rule` attr")
        if not g.mask_attrs.get(ATTR_BLACKLIST_SHA256):
            problems.append("mask.h5 has no `blacklist_sha256` attr")
        if not g.dna_attrs.get(ATTR_FASTA_SHA256):
            problems.append("dna.h5 has no `fasta_sha256` attr")
    finally:
        g.close()
    return problems


def genome_report(
    genome_dir: Path | str,
    context_bins: Sequence[int] = (768, 6144),
    min_valid_frac: float = DEFAULT_MIN_VALID_FRAC,
) -> dict:
    """Mask coverage and eligible-window counts per chromosome — the t7 deliverable numbers.

    For each `L` in `context_bins`, both **tiled** (`stride = L`, non-overlapping, the count a
    training epoch actually draws from under `window_plan.type = "tile"`) and **every start**
    (`stride = 1`, the size of the sampling space under a random-start plan).
    """
    out: dict = {"min_valid_frac": float(min_valid_frac), "per_chrom": {}, "genome": {}}
    with GenomeLayer(genome_dir, min_valid_frac=min_valid_frac) as g:
        out["build"] = g.build
        out["resolution"] = g.resolution
        out["fasta_sha256"] = str(g.dna_attrs.get(ATTR_FASTA_SHA256, ""))
        out["blacklist_sha256"] = str(g.mask_attrs.get(ATTR_BLACKLIST_SHA256, ""))
        out["blacklist_source"] = str(g.mask_attrs.get(ATTR_BLACKLIST_SOURCE, ""))
        out["rule"] = str(g.mask_attrs.get(ATTR_RULE, ""))
        totals = {"n_bins": 0, "n_valid": 0}
        tot_elig: dict = {}
        for chrom in g.chroms:
            m = g.mask(chrom)
            rec = {
                "n_bins": int(m.shape[0]),
                "n_valid": int(m.sum()),
                "valid_frac": float(m.mean()) if m.size else 0.0,
                "eligible": {},
            }
            totals["n_bins"] += rec["n_bins"]
            totals["n_valid"] += rec["n_valid"]
            for cb in context_bins:
                cb = int(cb)
                ok = eligible_window_mask(m, cb, min_valid_frac)
                tiled = int(ok[::cb].sum()) if ok.size else 0
                every = int(ok.sum()) if ok.size else 0
                rec["eligible"][str(cb)] = {"tiled": tiled, "every_start": every,
                                            "n_starts": int(ok.size)}
                agg = tot_elig.setdefault(str(cb), {"tiled": 0, "every_start": 0, "n_starts": 0})
                agg["tiled"] += tiled
                agg["every_start"] += every
                agg["n_starts"] += int(ok.size)
            out["per_chrom"][chrom] = rec
        totals["valid_frac"] = (
            float(totals["n_valid"] / totals["n_bins"]) if totals["n_bins"] else 0.0
        )
        totals["eligible"] = tot_elig
        out["genome"] = totals
    return out
