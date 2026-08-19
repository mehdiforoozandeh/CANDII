"""The single place that knows the CANDI_STORE on-disk contract.

Paths, attribute names, dtype rules, the chunk/compression policy, the `n_bins` rule and the
fixed-point pval codec all live here. `writer.py`, `manifest.py`, `cli.py` and — later —
`genome.py`, `reader.py`, `regime.py` and `dataset.py` import from this module rather than
re-deriving any of it. If a rule appears twice in this package, one of the copies is a bug.

Decisions implemented here, by number (`STORE_PLAN.md` §3):

* **D1/D3** one dataset per chromosome, shape `(n_bins, n_tracks)`, whole and contiguous.
* **D2** chunk `(1024 bins, all tracks)`, gzip level 4, every kind. Blosc/zstd are not
  registered on Fir — do not add them.
* **D4** one file per biosample per kind: `biosamples/<NAME>/{counts,peaks,pval}.h5`.
* **D7** counts dtype per file: `uint16` while the biosample's global max fits, else `uint32`.
* **D8** peaks `uint8`.
* **D9** pval `uint16` fixed-point, `scale=100`, stored in the original `-log10 p` space.
* **D13** every kind uses `n_bins = floor(chr_len / resolution)`; a source array one bin longer
  is truncated, anything else is an error.
* **D16** biosample names are opaque ids — nothing here parses them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_RESOLUTION",
    "KINDS",
    "CONTROL_TRACK",
    "CHUNK_BINS",
    "GZIP_LEVEL",
    "PVAL_SCALE",
    "PVAL_UINT16_MAX",
    "PVAL_MAX_ENCODABLE",
    "PEAKS_DTYPE",
    "PVAL_DTYPE",
    "COUNTS_DTYPES",
    "ATTR_SCHEMA",
    "ATTR_BIOSAMPLE",
    "ATTR_KIND",
    "ATTR_RESOLUTION",
    "ATTR_TRACKS",
    "ATTR_CONTROL_COL",
    "ATTR_DTYPE",
    "ATTR_N_BINS",
    "ATTR_SOURCE_ROOT",
    "ATTR_KIT_VERSION",
    "ATTR_BUILT_UTC",
    "ATTR_DSF",
    "ATTR_SCALE",
    "ATTR_PVAL_CLIP_FRAC",
    "ATTR_NPZ_DEPTH",
    "StoreError",
    "LengthMismatch",
    "genome_dir",
    "corpus_genome_dir",
    "chrom_sizes_path",
    "dna_path",
    "mask_path",
    "corpus_root",
    "biosamples_dir",
    "biosample_dir",
    "kind_path",
    "manifest_path",
    "n_bins_for",
    "n_bins_map",
    "fit_to_n_bins",
    "counts_dtype_for_max",
    "dtype_name",
    "chunk_shape",
    "dataset_kwargs",
    "order_tracks",
    "control_col_of",
    "sort_chroms",
    "load_chrom_sizes",
    "encode_pval",
    "decode_pval",
    "kit_version",
    "utc_now",
    "root_attrs",
    "write_root_attrs",
    "read_root_attrs",
]

# ---------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------

SCHEMA_VERSION = 1
DEFAULT_RESOLUTION = 25

#: The three kinds, in build order (D24). `counts` is the only one training strictly needs.
KINDS = ("counts", "peaks", "pval")

#: D18 — the control is a normal column of `counts.h5` under this exact name, flagged by
#: the `control_col` root attr. It is never an "assay".
CONTROL_TRACK = "chipseq-control"

#: D2 — chunk is `(CHUNK_BINS, n_tracks)` and the compressor is gzip at this level. Measured:
#: gzip4 is 22% smaller than gzip1 at identical read latency (`STORE_PLAN.md` §2).
CHUNK_BINS = 1024
GZIP_LEVEL = 4

#: D9 — fixed-point pval. `round(-log10 p * 100)` clipped into uint16.
PVAL_SCALE = 100
PVAL_UINT16_MAX = 65535
PVAL_MAX_ENCODABLE = PVAL_UINT16_MAX / PVAL_SCALE  # 655.35

PEAKS_DTYPE = np.uint8          # D8
PVAL_DTYPE = np.uint16          # D9
COUNTS_DTYPES = (np.uint16, np.uint32)   # D7, in widening order

# Root attribute names. Every kind carries the first eleven; counts adds `dsf`, pval adds `scale`.
ATTR_SCHEMA = "schema"
ATTR_BIOSAMPLE = "biosample"
ATTR_KIND = "kind"
ATTR_RESOLUTION = "resolution"
ATTR_TRACKS = "tracks"                 # JSON list — THE column order of this file
ATTR_CONTROL_COL = "control_col"       # int column index, or -1 when absent
ATTR_DTYPE = "dtype"                   # e.g. "uint16"
ATTR_N_BINS = "n_bins"                 # JSON dict {chrom: n_bins}
ATTR_SOURCE_ROOT = "source_root"
ATTR_KIT_VERSION = "kit_version"
ATTR_BUILT_UTC = "built_utc"
ATTR_DSF = "dsf"                       # counts only, always 1 (D6)
ATTR_SCALE = "scale"                   # pval only, always PVAL_SCALE

#: pval only — JSON list aligned with `tracks`, the fraction of bins clipped by the codec (D9).
ATTR_PVAL_CLIP_FRAC = "pval_clip_frac"
#: counts only — JSON list aligned with `tracks`, `depth` as the source npz tree reported it,
#: `null` where the source had none. Not authority (D20 gives that to the CSVs); it is here so a
#: CSV/npz depth disagreement is visible in the manifest instead of invisible.
ATTR_NPZ_DEPTH = "npz_depth"

_JSON_ATTRS = frozenset({ATTR_TRACKS, ATTR_N_BINS, ATTR_PVAL_CLIP_FRAC, ATTR_NPZ_DEPTH})


class StoreError(RuntimeError):
    """Any violation of the store contract. Always names the offending path."""


class LengthMismatch(StoreError):
    """A source array is neither `n_bins` nor `n_bins + 1` long (D13)."""


# ---------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------
# `store_root` is CANDI_STORE/ ; `corpus_root` is CANDI_STORE/<corpus>/ — the handle the reader
# takes (`CorpusStore("/…/CANDI_STORE/eic")`). The genome layer is shared and lives one level up.


def genome_dir(store_root: Path | str) -> Path:
    """`CANDI_STORE/genome` — the shared DNA + mask layer (t7)."""
    return Path(store_root) / "genome"


def corpus_genome_dir(corpus_root_: Path | str) -> Path:
    """The genome dir a corpus store belongs to. Corpora are siblings of `genome/`."""
    return Path(corpus_root_).resolve().parent / "genome"


def chrom_sizes_path(store_root: Path | str) -> Path:
    return genome_dir(store_root) / "chrom_sizes.json"


def dna_path(store_root: Path | str) -> Path:
    return genome_dir(store_root) / "dna.h5"


def mask_path(store_root: Path | str) -> Path:
    return genome_dir(store_root) / "mask.h5"


def corpus_root(store_root: Path | str, corpus: str) -> Path:
    return Path(store_root) / corpus


def biosamples_dir(corpus_root_: Path | str) -> Path:
    return Path(corpus_root_) / "biosamples"


def biosample_dir(corpus_root_: Path | str, biosample: str) -> Path:
    """D16 — `biosample` is used verbatim. No prefix parsing, no normalization."""
    return biosamples_dir(corpus_root_) / biosample


def kind_path(corpus_root_: Path | str, biosample: str, kind: str) -> Path:
    if kind not in KINDS:
        raise StoreError(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    return biosample_dir(corpus_root_, biosample) / f"{kind}.h5"


def manifest_path(corpus_root_: Path | str) -> Path:
    return Path(corpus_root_) / "manifest.json"


# ---------------------------------------------------------------------------------------------
# the n_bins rule (D13)
# ---------------------------------------------------------------------------------------------


def n_bins_for(chrom_len: int, resolution: int = DEFAULT_RESOLUTION) -> int:
    """D13 — **every** kind uses `floor(chr_len / resolution)`.

    The source counts npz is `ceil` and therefore one bin longer than the pval and peaks npz.
    Storing the floor for all three removes the off-by-one for good; see `fit_to_n_bins`.
    """
    if chrom_len <= 0:
        raise StoreError(f"chrom_len must be positive, got {chrom_len}")
    if resolution <= 0:
        raise StoreError(f"resolution must be positive, got {resolution}")
    return int(chrom_len // resolution)


def n_bins_map(chrom_sizes: Mapping[str, int], resolution: int = DEFAULT_RESOLUTION) -> dict:
    return {c: n_bins_for(int(n), resolution) for c, n in chrom_sizes.items()}


def fit_to_n_bins(arr: np.ndarray, n_bins: int, where: str) -> np.ndarray:
    """Truncate a source array to `n_bins` (D13), refusing anything but a one-bin overhang.

    `where` is quoted in the error, so pass something locating enough to act on — normally
    `f"{biosample}/{track}/{kind}/{chrom}"`.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise LengthMismatch(f"{where}: expected a 1-D array, got shape {arr.shape}")
    n = arr.shape[0]
    if n == n_bins:
        return arr
    if n == n_bins + 1:
        return arr[:n_bins]
    raise LengthMismatch(
        f"{where}: source array is {n} bins but n_bins is {n_bins}; only {n_bins} (floor) or "
        f"{n_bins + 1} (ceil) is allowed. Wrong chrom_sizes, wrong resolution, or a truncated npz."
    )


# ---------------------------------------------------------------------------------------------
# dtypes (D7, D8, D9)
# ---------------------------------------------------------------------------------------------


def counts_dtype_for_max(global_max: int) -> np.dtype:
    """D7 — one dtype per counts FILE, chosen from the biosample's global max.

    `uint16` while the max fits, else `uint32`. The reader upcasts to int32/int64, so a store
    built at either width is interchangeable downstream.
    """
    m = int(global_max)
    if m < 0:
        raise StoreError(f"counts max is negative ({m}); the source is not raw read counts")
    if m <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if m <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise StoreError(
        f"counts max {m} exceeds uint32; almost certainly the npz is not raw per-bin counts"
    )


def dtype_name(dt) -> str:
    return np.dtype(dt).name


def chunk_shape(n_bins: int, n_tracks: int) -> tuple:
    """D2 — `(1024 bins, all tracks)`, shrunk on a chromosome shorter than one chunk."""
    if n_bins <= 0 or n_tracks <= 0:
        raise StoreError(f"cannot chunk a ({n_bins}, {n_tracks}) dataset")
    return (min(CHUNK_BINS, n_bins), n_tracks)


def dataset_kwargs(n_bins: int, n_tracks: int, dtype) -> dict:
    """The one `create_dataset` policy. Never pass compression settings anywhere else."""
    return {
        "shape": (n_bins, n_tracks),
        "dtype": np.dtype(dtype),
        "chunks": chunk_shape(n_bins, n_tracks),
        "compression": "gzip",
        "compression_opts": GZIP_LEVEL,
        "shuffle": False,
    }


# ---------------------------------------------------------------------------------------------
# track and chromosome ordering
# ---------------------------------------------------------------------------------------------


def order_tracks(names: Iterable[str]) -> list:
    """The store's own column order: assays sorted, `chipseq-control` last (D18).

    This is storage order, not model order. **D14** puts the model's assay order in the regime
    file; `reader.py` maps the declared list onto these columns by name.
    """
    names = list(names)
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise StoreError(f"duplicate track names: {dupes}")
    assays = sorted(n for n in names if n != CONTROL_TRACK)
    return assays + ([CONTROL_TRACK] if CONTROL_TRACK in names else [])


def control_col_of(tracks: Sequence[str]) -> int:
    """D18 — the control's column index, or `-1` when this file has no control column."""
    return tracks.index(CONTROL_TRACK) if CONTROL_TRACK in tracks else -1


_CHROM_NUM = re.compile(r"^chr(\d+)$")
_CHROM_RANK = {"chrX": 100, "chrY": 101, "chrM": 102, "chrMT": 102}


def sort_chroms(chroms: Iterable[str]) -> list:
    """chr1…chr22, chrX, chrY, chrM, then anything else alphabetically."""

    def key(c: str):
        m = _CHROM_NUM.match(c)
        if m:
            return (0, int(m.group(1)), c)
        if c in _CHROM_RANK:
            return (1, _CHROM_RANK[c], c)
        return (2, 0, c)

    return sorted(set(chroms), key=key)


def load_chrom_sizes(path: Path | str) -> dict:
    """Read `chrom_sizes.json` or a two-column `.sizes` TSV, returning `{chrom: length}`.

    The JSON comes in two shapes and both are accepted. The bare one is `{chrom: length}`. The
    store's own `genome/chrom_sizes.json` is the **wrapped** one — a provenance record carrying
    `build`, `resolution`, `source` and a precomputed `n_bins` alongside the sizes — because a
    bare map cannot say which FASTA it came from.

    Symptom of getting this wrong, before the wrapper was understood here:
    `ValueError: invalid literal for int() with base 10: 'GRCh38'` out of `build-biosample`,
    which reads as a corrupt file rather than as the top level being a header.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"chrom sizes file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json" or text.lstrip().startswith("{"):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise StoreError(f"{p}: expected a JSON object of chrom -> length")
        if isinstance(obj.get("chrom_sizes"), dict):
            obj = obj["chrom_sizes"]
        return {str(k): int(v) for k, v in obj.items()}
    sizes = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise StoreError(f"{p}:{lineno}: expected '<chrom>\\t<length>', got {line!r}")
        sizes[parts[0]] = int(parts[1])
    if not sizes:
        raise StoreError(f"{p}: no chromosome entries")
    return sizes


# ---------------------------------------------------------------------------------------------
# the pval codec (D9)
# ---------------------------------------------------------------------------------------------


def encode_pval(values, scale: int = PVAL_SCALE, nan_policy: str = "error") -> tuple:
    """`-log10 p` (float) -> `uint16` fixed point. Returns `(encoded, n_clipped)`.

    `round(x * scale)` clipped to `[0, 65535]`, i.e. a resolution of `1/scale` and a ceiling of
    `65535 / scale` = 655.35 at the default. Measured quantization error on the real corpus:
    **max 0.005** on `-log10 p`, and chr1's observed max is 161.2, far under the ceiling.

    Stored in the **original space** — no arcsinh at bake time; every transform is the model's.

    `n_clipped` counts every bin whose value fell outside `[0, 65535/scale]`, returned as a count
    rather than a fraction so a caller can aggregate it across chromosomes.

    `+inf` is a real `-log10 p` (`p == 0`) and clips to the ceiling like any other overflow. **NaN
    is not a number at all**, so `nan_policy="error"` refuses it rather than inventing one;
    `"zero"` encodes it as 0 and counts it among the clipped.
    """
    if nan_policy not in ("error", "zero"):
        raise StoreError(f"unknown nan_policy {nan_policy!r}; expected 'error' or 'zero'")
    if scale <= 0:
        raise StoreError(f"pval scale must be positive, got {scale}")
    x = np.asarray(values, dtype=np.float64)
    nan = np.isnan(x)
    n_nan = int(nan.sum())
    if n_nan:
        if nan_policy == "error":
            raise StoreError(
                f"{n_nan} NaN (non-finite) values in a pval track; refusing to invent a number for "
                f"them. Pass nan_policy='zero' (CLI: --pval-nan zero) if that is what you want."
            )
        x = np.where(nan, 0.0, x)
    scaled = np.rint(x * scale)
    out_of_range = (scaled < 0) | (scaled > PVAL_UINT16_MAX)
    n_clipped = int(np.count_nonzero(out_of_range | nan))
    encoded = np.clip(scaled, 0, PVAL_UINT16_MAX).astype(PVAL_DTYPE)
    return encoded, n_clipped


def decode_pval(encoded, scale: int = PVAL_SCALE) -> np.ndarray:
    """`uint16` fixed point -> `-log10 p` as float32. The exact inverse of `encode_pval`."""
    return (np.asarray(encoded).astype(np.float32) / np.float32(scale)).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# root attrs
# ---------------------------------------------------------------------------------------------


def kit_version() -> str:
    try:
        from candi import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - only when candi is not importable at all
        return "unknown"


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def root_attrs(
    *,
    biosample: str,
    kind: str,
    tracks: Sequence[str],
    n_bins: Mapping[str, int],
    dtype,
    source_root: str,
    resolution: int = DEFAULT_RESOLUTION,
    built_utc: Optional[str] = None,
    version: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the root attr dict for one file. The only place the attr set is defined."""
    if kind not in KINDS:
        raise StoreError(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    attrs = {
        ATTR_SCHEMA: int(SCHEMA_VERSION),
        ATTR_BIOSAMPLE: str(biosample),
        ATTR_KIND: str(kind),
        ATTR_RESOLUTION: int(resolution),
        ATTR_TRACKS: json.dumps(list(tracks)),
        ATTR_CONTROL_COL: int(control_col_of(list(tracks))),
        ATTR_DTYPE: dtype_name(dtype),
        ATTR_N_BINS: json.dumps({str(k): int(v) for k, v in n_bins.items()}),
        ATTR_SOURCE_ROOT: str(source_root),
        ATTR_KIT_VERSION: str(version if version is not None else kit_version()),
        ATTR_BUILT_UTC: str(built_utc if built_utc is not None else utc_now()),
    }
    if kind == "counts":
        attrs[ATTR_DSF] = 1                     # D6 — DSF1 only; the loader thins.
    if kind == "pval":
        attrs[ATTR_SCALE] = int(PVAL_SCALE)     # D9
    if extra:
        attrs.update(extra)
    return attrs


def write_root_attrs(h5obj, attrs: Mapping[str, Any]) -> None:
    for k, v in attrs.items():
        h5obj.attrs[k] = v


def read_root_attrs(h5obj) -> dict:
    """Read root attrs back, decoding bytes and the JSON-encoded ones. Use this, not `.attrs`."""
    out = {}
    for k, v in h5obj.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, np.generic):
            v = v.item()
        if k in _JSON_ATTRS and isinstance(v, str):
            v = json.loads(v)
        out[k] = v
    return out
