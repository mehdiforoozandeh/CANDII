"""t118 ladder — the corpus reader over t112 npz products or the memory-mapped cache, and the cache builder.

**Two layouts, one reader.** `Corpus(root)` is a *cache* when `<root>` holds any
`<pid>__<space>.json` file, else a *products dir* of `<root>/<pid>/{counts25,pval25}.npz` (the t112
layout, `tools/t112/bin25.py::write_npz`: one array per main chromosome keyed by name, counts
uint32, -log10 p float32).

**Cache format** (one product, one space):
  `<pid>__<space>.npy`   1-D concatenation of the chromosomes in MAIN_CHROMS order, native dtype
  `<pid>__<space>.json`  `{"pid", "space", "dtype", "chroms": [...], "offsets": {chrom: [start,
                         end]}, "n_bins", "source_npz_md5"}` — written LAST, so its presence means
                         the npy is complete.
Cache reads are read-only memmap slices (no copy); npz reads decompress one chromosome and keep
the 64 most recent arrays in an LRU.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

from candi.store.genome import blacklist_bin_flags, read_blacklist

from ladder import pairs

RES = 25
#: space -> (npz file name, dtype)
SPACE_FILES = {"counts": ("counts25.npz", np.dtype(np.uint32)),
               "pval": ("pval25.npz", np.dtype(np.float32))}
LRU_SIZE = 64


def _check_space(space: str) -> None:
    if space not in SPACE_FILES:
        raise ValueError(f"space {space!r} not in {tuple(SPACE_FILES)}")


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Corpus:
    """Per-(pid, space, chrom) access to a products dir or a cache dir (see the module docstring)."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"corpus root not found: {self.root}")
        self.is_cache = any(self.root.glob("*__counts.json")) or any(self.root.glob("*__pval.json"))
        self._meta: dict = {}       # (pid, space) -> cache json dict
        self._mm: dict = {}         # (pid, space) -> read-only memmap of the whole npy
        self._chroms: dict = {}     # (pid, space) -> [chrom, ...]  (npz mode)
        self._lens: dict = {}       # (pid, space, chrom) -> n_bins (npz mode)
        self._lru: OrderedDict = OrderedDict()

    # -- cache mode -------------------------------------------------------------------------------

    def _cache_meta(self, pid: str, space: str) -> dict:
        key = (pid, space)
        if key not in self._meta:
            path = self.root / f"{pid}__{space}.json"
            if not path.is_file():
                raise FileNotFoundError(f"cache json not found: {path}")
            self._meta[key] = json.loads(path.read_text())
        return self._meta[key]

    def _cache_array(self, pid: str, space: str) -> np.ndarray:
        key = (pid, space)
        if key not in self._mm:
            meta = self._cache_meta(pid, space)
            mm = np.load(self.root / f"{pid}__{space}.npy", mmap_mode="r", allow_pickle=False)
            if mm.dtype != SPACE_FILES[space][1] or mm.ndim != 1 or mm.shape[0] != meta["n_bins"]:
                raise ValueError(f"{pid}__{space}.npy: dtype {mm.dtype} shape {mm.shape} does not "
                                 f"match its json ({meta['dtype']}, {meta['n_bins']})")
            self._mm[key] = mm
        return self._mm[key]

    # -- npz mode ---------------------------------------------------------------------------------

    def _npz_path(self, pid: str, space: str) -> Path:
        return self.root / pid / SPACE_FILES[space][0]

    def _npz_get(self, pid: str, space: str, chrom: str) -> np.ndarray:
        key = (pid, space, chrom)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        if chrom not in self.chroms(pid, space):
            raise KeyError(f"{pid}/{space}: no chromosome {chrom!r}")
        with np.load(self._npz_path(pid, space), allow_pickle=False) as z:
            a = z[chrom]
        want = SPACE_FILES[space][1]
        if a.ndim != 1 or a.dtype != want:
            raise ValueError(f"{self._npz_path(pid, space)}:{chrom}: dtype {a.dtype} ndim {a.ndim}, "
                             f"need 1-D {want}")
        a.setflags(write=False)
        self._lens[key] = a.shape[0]
        self._lru[key] = a
        while len(self._lru) > LRU_SIZE:
            self._lru.popitem(last=False)
        return a

    # -- public -----------------------------------------------------------------------------------

    def chroms(self, pid: str, space: str) -> list[str]:
        """The product's chromosomes in npz (cache: MAIN_CHROMS) order, minus DROP_CHROMS."""
        _check_space(space)
        if self.is_cache:
            return [c for c in self._cache_meta(pid, space)["chroms"] if c not in pairs.DROP_CHROMS]
        key = (pid, space)
        if key not in self._chroms:
            with np.load(self._npz_path(pid, space), allow_pickle=False) as z:
                self._chroms[key] = [c for c in z.files if c not in pairs.DROP_CHROMS]
        return list(self._chroms[key])

    def get(self, pid: str, space: str, chrom: str) -> np.ndarray:
        """The whole chromosome: uint32 counts / float32 p; read-only (memmap slice or LRU copy)."""
        _check_space(space)
        if self.is_cache:
            meta = self._cache_meta(pid, space)
            if chrom not in meta["offsets"] or chrom in pairs.DROP_CHROMS:
                raise KeyError(f"{pid}/{space}: no chromosome {chrom!r}")
            s, e = meta["offsets"][chrom]
            return self._cache_array(pid, space)[s:e]
        return self._npz_get(pid, space, chrom)

    def n_bins(self, pid: str, space: str, chrom: str) -> int:
        _check_space(space)
        if self.is_cache:
            s, e = self._cache_meta(pid, space)["offsets"][chrom]
            return int(e - s)
        key = (pid, space, chrom)
        if key not in self._lens:
            self._npz_get(pid, space, chrom)
        return int(self._lens[key])

    def window(self, pid: str, space: str, chrom: str, start: int, stop: int) -> np.ndarray:
        """Bins [start, stop) of `chrom`, zero-padded wherever the window leaves the chromosome."""
        start, stop = int(start), int(stop)
        if stop < start:
            raise ValueError(f"window stop {stop} < start {start}")
        a = self.get(pid, space, chrom)
        n = a.shape[0]
        out = np.zeros(stop - start, dtype=a.dtype)
        s, e = max(start, 0), min(stop, n)
        if e > s:
            out[s - start:e - start] = a[s:e]
        return out


def _npz_member_len(z, chrom: str, npz) -> int:
    """Length of one npz member read from its npy header, without decompressing the data."""
    with z.zip.open(f"{chrom}.npy") as f:
        version = np.lib.format.read_magic(f)
        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                else np.lib.format.read_array_header_2_0)
        shape, _, _ = read(f)
    if len(shape) != 1:
        raise ValueError(f"{npz}:{chrom}: shape {shape} is not 1-D")
    return int(shape[0])


def build_cache(products_dir, cache_dir, pid: str, manifest_row: dict) -> dict:
    """Cache both spaces of one product; returns `{space: "built" | "skipped"}`.

    Checks each npz's file md5 against `manifest_row["counts25_md5" | "pval25_md5"]` first, writes
    `<pid>__<space>.npy.tmp` then `os.replace`, then the json (tmp + replace). A space whose json
    already exists is skipped.
    """
    if manifest_row["pid"] != pid:
        raise ValueError(f"manifest row is {manifest_row['pid']!r}, not {pid!r}")
    products_dir, cache_dir = Path(products_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for space, (fname, dtype) in SPACE_FILES.items():
        js = cache_dir / f"{pid}__{space}.json"
        if js.is_file():
            status[space] = "skipped"
            continue
        npz = products_dir / pid / fname
        md5 = md5_file(npz)
        want = manifest_row[pairs.MD5_COLUMN[space]]
        if md5 != want:
            raise ValueError(f"{npz}: md5 {md5} != manifest {want}")
        with np.load(npz, allow_pickle=False) as z:
            present = set(z.files)
            chroms = [c for c in pairs.MAIN_CHROMS if c in present]
            lens = {c: _npz_member_len(z, c, npz) for c in chroms}
            offsets, n = {}, 0
            for c in chroms:
                offsets[c] = [n, n + lens[c]]
                n += lens[c]
            tmp = cache_dir / f"{pid}__{space}.npy.tmp"
            mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=(n,))
            for c in chroms:
                a = z[c]
                if a.ndim != 1 or a.dtype != dtype:
                    raise ValueError(f"{npz}:{c}: dtype {a.dtype} ndim {a.ndim}, need 1-D {dtype}")
                if space == "pval" and a.size and not (np.isfinite(a).all() and a.min() >= 0):
                    raise ValueError(f"{npz}:{c}: non-finite or negative -log10 p")
                s, e = offsets[c]
                mm[s:e] = a
            mm.flush()
            del mm
        os.replace(tmp, cache_dir / f"{pid}__{space}.npy")
        meta = {"pid": pid, "space": space, "dtype": dtype.name, "chroms": chroms,
                "offsets": offsets, "n_bins": n, "source_npz_md5": md5}
        jtmp = cache_dir / f"{pid}__{space}.json.tmp"
        jtmp.write_text(json.dumps(meta, indent=1) + "\n")
        os.replace(jtmp, js)
        status[space] = "built"
    return status


@lru_cache(maxsize=4)
def _blacklist(path: str) -> dict:
    return read_blacklist(path)


def read_blacklist_flags(blacklist_bed, chrom: str, n_bins: int) -> np.ndarray:
    """bool[n_bins]: True where a 25 bp bin overlaps a blacklist interval by >= 1 bp.

    Wraps `candi.store.genome.read_blacklist` (parsed once per path) and `blacklist_bin_flags`.
    """
    iv = _blacklist(str(Path(blacklist_bed).resolve())).get(chrom)
    return blacklist_bin_flags(iv, int(n_bins), RES)
