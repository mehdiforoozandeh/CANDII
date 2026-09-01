"""`CorpusStore` / `BiosampleStore` / `TrackView` — the OO tree over CANDI_STORE (D3, t8).

The tree people think in — `corpus -> biosample -> assay -> kind -> chromosome` — lives **here**,
in Python, not in HDF5 group paths. On disk there is one flat `(n_bins, n_tracks)` dataset per
chromosome per kind (D1/D3), so one read serves a whole window across every assay at once.

```python
corpus = CorpusStore("/…/CANDI_STORE/eic")
bs     = corpus["T_DND-41"]
counts = bs["H3K4me3"].counts("chr1", 0, 768)          # (L,)   int32
block  = bs.counts("chr1", 0, 768, assays=[...])       # (L, F) int32, DECLARED order (D14)
dna    = corpus.genome.dna("chr1", 0, 19_200)          # (Lbp,) uint8 codes
```

Three rules this module keeps:

* **The h5 root attrs are the authority, the manifest is a convenience.** Every file
  self-describes (`layout.py::read_root_attrs`); `manifest.json` is read for the corpus-level
  view (`assay_vocabulary`, per-track metadata) and **cross-checked**. A disagreement on track
  order, dtype, `control_col` or `n_bins` raises `StoreError` naming the file — it never picks
  a winner quietly.
* **The reader upcasts counts** to `int32` (D7), so a `uint16` store and a `uint32` store are
  interchangeable downstream, and `pval` is decoded back to `-log10 p` float32 through
  `layout.py::decode_pval` (D9). `peaks` stays `uint8` — it is a 0/1 indicator and already
  width-independent.
* **h5py handles are not fork-safe.** Handles are opened lazily and cached per *process*
  (`_HandlePool`); the cache is keyed by pid, so the first access inside a DataLoader worker
  opens its own file objects and the inherited ones are dropped, never reused and never closed
  from the child (closing a parent's HDF5 handle from a child corrupts the parent's cache).
  These objects also pickle cleanly — the pool is dropped by `__getstate__` — so a `spawn`
  start method behaves the same as `fork`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import h5py
import numpy as np

from candi.store import layout as L
from candi.store.layout import StoreError

__all__ = [
    "COUNTS_OUT_DTYPE",
    "PEAKS_OUT_DTYPE",
    "PVAL_OUT_DTYPE",
    "DNA_CODES",
    "TrackView",
    "BiosampleStore",
    "GenomeView",
    "CorpusStore",
    "one_hot_dna",
]

#: D7 — the reader's output width for counts. Wide enough for `uint32` stores and signed, so a
#: downstream `-1` MISSING sentinel fits without another cast.
COUNTS_OUT_DTYPE = np.int32
#: D8 — peaks are a 0/1 indicator; the stored width is already the narrowest correct one.
PEAKS_OUT_DTYPE = np.uint8
#: D9 — pval comes back in the original `-log10 p` space, decoded from the fixed point.
PVAL_OUT_DTYPE = np.float32

#: D10 — the base codes `genome/dna.h5` stores. `N` (4) one-hots to an all-zero row.
DNA_CODES = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


# ---------------------------------------------------------------------------------------------
# fork-safe handle pool
# ---------------------------------------------------------------------------------------------


class _HandlePool:
    """Lazily opened, read-only h5py handles, cached per process.

    A DataLoader worker forked from the parent inherits the parent's file descriptors. Reading
    through an inherited HDF5 handle races the parent's metadata cache and returns garbage or
    hangs; *closing* it from the child damages the parent. So on the first access from a new pid
    the map is **dropped without closing** and the worker opens its own handles.
    """

    __slots__ = ("_pid", "_files")

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._files: Dict[str, h5py.File] = {}

    def get(self, path: Path) -> h5py.File:
        pid = os.getpid()
        if pid != self._pid:
            # Inherited handles belong to the parent: forget them, do not close them.
            self._files = {}
            self._pid = pid
        key = str(path)
        f = self._files.get(key)
        if f is None:
            if not Path(path).is_file():
                raise StoreError(f"{path}: not found — the store does not have this file")
            f = h5py.File(path, "r")
            self._files[key] = f
        return f

    def close(self) -> None:
        """Close every handle this process opened. Never touches another process's."""
        if os.getpid() != self._pid:
            self._files = {}
            return
        for f in self._files.values():
            try:
                f.close()
            except Exception:  # pragma: no cover - a already-closed handle is not an error here
                pass
        self._files = {}


def _slice_bounds(start: int, end: Optional[int], n: int, where: str) -> tuple:
    start = int(start)
    end = int(n if end is None else end)
    if start < 0 or end > n or end <= start:
        raise StoreError(
            f"{where}: requested [{start}, {end}) is not inside [0, {n}). A window that runs off "
            f"the end of a chromosome is a window-plan bug, not something to clamp."
        )
    return start, end


def one_hot_dna(codes: np.ndarray) -> np.ndarray:
    """`(Lbp,)` uint8 base codes -> `(Lbp, 4)` float32 one-hot, ACGT. `N` (4) becomes all zeros.

    All-zero is the honest encoding of `N`: no base is asserted, and it is what the old bake's
    unknown-base column carried.
    """
    codes = np.asarray(codes)
    out = np.zeros((codes.shape[0], 4), dtype=np.float32)
    known = codes < 4
    out[np.arange(codes.shape[0])[known], codes[known].astype(np.intp)] = 1.0
    return out


# ---------------------------------------------------------------------------------------------
# one assay of one biosample
# ---------------------------------------------------------------------------------------------


class TrackView:
    """`bs["H3K4me3"]` — one column, across kinds and chromosomes.

    A thin addressing helper: every read goes through the owning `BiosampleStore`, so a
    single-assay read and a block read of the same interval hit the same handle and the same
    chunk cache.
    """

    __slots__ = ("_bs", "assay")

    def __init__(self, bs: "BiosampleStore", assay: str) -> None:
        self._bs = bs
        self.assay = assay

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TrackView {self._bs.name}/{self.assay}>"

    @property
    def kinds(self) -> List[str]:
        """The kinds that actually carry this assay (the control has counts only)."""
        return [k for k in self._bs.kinds if self._bs.has(self.assay, k)]

    def counts(self, chrom: str, start: int, end: Optional[int] = None) -> np.ndarray:
        return self._bs.counts(chrom, start, end, assays=[self.assay])[:, 0]

    def peaks(self, chrom: str, start: int, end: Optional[int] = None) -> np.ndarray:
        return self._bs.peaks(chrom, start, end, assays=[self.assay])[:, 0]

    def pval(self, chrom: str, start: int, end: Optional[int] = None) -> np.ndarray:
        return self._bs.pval(chrom, start, end, assays=[self.assay])[:, 0]

    def signal_rdns(self, chrom: str, start: int, end: Optional[int] = None) -> np.ndarray:
        """The archived predecessor of `pval`, read the same way `pval` is.

        An archive kind is reachable by every path a built kind is reachable by — that is the whole
        of what "stays queryable" means in `BENCHMARK_DESIGN.md` §10 Phase 3, and the per-track view
        is a documented path in `STORE.md`.
        """
        return self._bs.signal_rdns(chrom, start, end, assays=[self.assay])[:, 0]


# ---------------------------------------------------------------------------------------------
# one biosample
# ---------------------------------------------------------------------------------------------


class BiosampleStore:
    """`corpus["T_DND-41"]` — the `{counts,peaks,pval}.h5` triple of one biosample (D4).

    Every accessor takes **bin** coordinates on the store's own resolution grid, half-open
    `[start, end)`, and returns `(L,)` or `(L, F)` with the columns in the order you asked for
    (D14). Asking for an assay this biosample does not have raises and names it — the caller is
    expected to test with `has()` and emit `MISSING` itself, because "absent" is a modelling fact
    the loader must express, not something the reader may paper over with zeros.
    """

    def __init__(self, corpus_root: Path, name: str, pool: _HandlePool,
                 kinds: Sequence[str] = L.KINDS) -> None:
        self.corpus_root = Path(corpus_root)
        self.name = str(name)
        self._pool = pool
        self._attrs: Dict[str, dict] = {}
        self.kinds: List[str] = [
            k for k in kinds if L.kind_path(self.corpus_root, self.name, k).is_file()
        ]
        if not self.kinds:
            raise StoreError(
                f"{L.biosample_dir(self.corpus_root, self.name)}: no "
                f"{'/'.join(L.KINDS)}.h5 — the biosample directory exists but holds no kind"
            )

    # -- structure ----------------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<BiosampleStore {self.name} kinds={self.kinds} tracks={len(self.tracks())}>"

    def __getitem__(self, assay: str) -> TrackView:
        if not any(self.has(assay, k) for k in self.kinds):
            raise StoreError(
                f"{self.name}: no track named {assay!r}. This biosample has "
                f"{self.tracks()}"
            )
        return TrackView(self, assay)

    def __contains__(self, assay: str) -> bool:
        return any(self.has(assay, k) for k in self.kinds)

    def _path(self, kind: str) -> Path:
        if kind not in self.kinds:
            raise StoreError(
                f"{self.name}: kind {kind!r} is not in this store (has {self.kinds}). "
                f"pval is built last (D24) — a store without it is normal, not broken."
            )
        return L.kind_path(self.corpus_root, self.name, kind)

    def attrs(self, kind: str) -> dict:
        """Root attrs of one kind, decoded. The file is the authority; this is how you ask it."""
        if kind not in self._attrs:
            self._attrs[kind] = L.read_root_attrs(self._pool.get(self._path(kind)))
        return self._attrs[kind]

    def tracks(self, kind: Optional[str] = None) -> List[str]:
        """Storage column order of one kind, or of the canonical kind when `kind is None`."""
        return list(self.attrs(kind or self.kinds[0])[L.ATTR_TRACKS])

    def assays(self, kind: Optional[str] = None) -> List[str]:
        """`tracks()` without `chipseq-control` — the control is a column, not an assay (D18)."""
        return [t for t in self.tracks(kind) if t != L.CONTROL_TRACK]

    def has(self, assay: str, kind: str = "counts") -> bool:
        return kind in self.kinds and assay in self.attrs(kind)[L.ATTR_TRACKS]

    @property
    def control_col(self) -> int:
        return int(self.attrs(self.kinds[0])[L.ATTR_CONTROL_COL])

    @property
    def has_control(self) -> bool:
        return "counts" in self.kinds and L.CONTROL_TRACK in self.tracks("counts")

    @property
    def resolution(self) -> int:
        return int(self.attrs(self.kinds[0])[L.ATTR_RESOLUTION])

    @property
    def dtype(self) -> str:
        return str(self.attrs(self.kinds[0])[L.ATTR_DTYPE])

    def n_bins(self, chrom: Optional[str] = None, kind: Optional[str] = None):
        nb = self.attrs(kind or self.kinds[0])[L.ATTR_N_BINS]
        if chrom is None:
            return {str(k): int(v) for k, v in nb.items()}
        if chrom not in nb:
            raise StoreError(
                f"{self.name}: no chromosome {chrom!r}; this file has "
                f"{L.sort_chroms(nb)}"
            )
        return int(nb[chrom])

    def chroms(self, kind: Optional[str] = None) -> List[str]:
        return L.sort_chroms(self.n_bins(kind=kind))

    # -- reads --------------------------------------------------------------------------------

    def _read(self, kind: str, chrom: str, start: int, end: Optional[int],
              assays: Optional[Sequence[str]]) -> np.ndarray:
        path = self._path(kind)
        f = self._pool.get(path)
        tracks = self.tracks(kind)
        if chrom not in f:
            raise StoreError(
                f"{path}: no dataset for {chrom!r}; this file has {L.sort_chroms(f.keys())}"
            )
        ds = f[chrom]
        s, e = _slice_bounds(start, end, int(ds.shape[0]), f"{self.name}/{kind}/{chrom}")
        names = list(tracks) if assays is None else [str(a) for a in assays]
        missing = [a for a in names if a not in tracks]
        if missing:
            raise StoreError(
                f"{self.name}/{kind}: no such track(s) {missing}. Available: {tracks}. "
                f"An absent assay is a MISSING column for the loader to fill, not a read."
            )
        cols = np.array([tracks.index(a) for a in names], dtype=np.intp)
        # h5py fancy indexing needs strictly increasing indices, and the declared order is
        # arbitrary (D14), so read sorted and permute back in memory.
        order = np.argsort(cols, kind="stable")
        uniq, inverse = np.unique(cols[order], return_inverse=True)
        block = ds[s:e, uniq.tolist()]
        back = np.empty(len(cols), dtype=np.intp)
        back[order] = inverse
        return block[:, back]

    def counts(self, chrom: str, start: int, end: Optional[int] = None,
               assays: Optional[Sequence[str]] = None) -> np.ndarray:
        """`(L, F)` int32 raw counts at DSF1 (D6). The loader thins; the store never does."""
        return self._read("counts", chrom, start, end, assays).astype(COUNTS_OUT_DTYPE)

    def peaks(self, chrom: str, start: int, end: Optional[int] = None,
              assays: Optional[Sequence[str]] = None) -> np.ndarray:
        """`(L, F)` uint8 0/1 peak calls (D8)."""
        return self._read("peaks", chrom, start, end, assays).astype(PEAKS_OUT_DTYPE)

    def pval(self, chrom: str, start: int, end: Optional[int] = None,
             assays: Optional[Sequence[str]] = None) -> np.ndarray:
        """`(L, F)` float32 `-log10 p`, decoded from the fixed point in THE FILE'S OWN codec.

        D26 — the codec is invisible above this line. The store holds `arcsinh(-log10 p)` scaled to
        uint16 (D25) and this returns ordinary `-log10 p`; anything further is the model's, and is
        `--signal-target-transform` (D30).

        THIS IS THE WHOLE OF D27's BACK-COMPAT, and the reason it can be four lines: `decode_pval`
        has exactly two call sites in the repo, this one and one test, so nothing bypasses it. The
        codec is read off the file rather than off this package's defaults, and an ABSENT
        `transform` attr means `"linear"` — that is what a schema-1 file is, and defaulting to the
        current constant instead would silently reinterpret every unrebuilt file as arcsinh and
        return `sinh` of a number that was never compressed.
        """
        return self._decoded_signal("pval", chrom, start, end, assays)

    def signal_rdns(self, chrom: str, start: int, end: Optional[int] = None,
                    assays: Optional[Sequence[str]] = None) -> np.ndarray:
        """`(L, F)` float32 of the ARCHIVED signal layer this corpus's `pval` layer replaced.

        `BENCHMARK_DESIGN.md` §10 Phase 3 — the read-depth normalized signal ENCODE ships for
        DNase-seq. It is stored in the same fixed-point codec as `pval` and decoded the same way,
        which is why this is one line: the archive is the predecessor FILE, unmodified, so its
        codec attrs are its own and are read off it exactly as `pval`'s are. **The units are not
        `-log10 p`** — that is the whole reason it was archived rather than kept.
        """
        return self._decoded_signal("signal_rdns", chrom, start, end, assays)

    def _decoded_signal(self, kind: str, chrom: str, start: int, end: Optional[int],
                        assays: Optional[Sequence[str]]) -> np.ndarray:
        raw = self._read(kind, chrom, start, end, assays)
        a = self.attrs(kind)
        scale = int(a.get(L.ATTR_SCALE, L.PVAL_SCALE_LINEAR_V1))
        transform = str(a.get(L.ATTR_TRANSFORM, L.PVAL_TRANSFORM_LINEAR))
        return L.decode_pval(raw, scale, transform)

    def control(self, chrom: str, start: int, end: Optional[int] = None) -> np.ndarray:
        """`(L, 1)` int32 of the `chipseq-control` column (D18), or a loud error when absent."""
        if not self.has_control:
            raise StoreError(f"{self.name}: no {L.CONTROL_TRACK} column in counts.h5")
        return self.counts(chrom, start, end, assays=[L.CONTROL_TRACK])

    def block(self, kind: str, chrom: str, start: int, end: Optional[int] = None,
              assays: Optional[Sequence[str]] = None) -> np.ndarray:
        """Kind-dispatched read, for callers that carry the kind as data."""
        return {"counts": self.counts, "peaks": self.peaks, "pval": self.pval,
                "signal_rdns": self.signal_rdns}[kind](chrom, start, end, assays)

    # -- pickling (DataLoader workers under `spawn`) -------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_attrs"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._pool is None:
            self._pool = _HandlePool()


# ---------------------------------------------------------------------------------------------
# the shared genome layer
# ---------------------------------------------------------------------------------------------


class GenomeView:
    """`corpus.genome` — `dna.h5`, `mask.h5` and `chrom_sizes.json` (D10, D11), shared by corpora.

    t7 writes these files; this reads them. Both are optional here on purpose: `counts` is the
    only kind training strictly needs, and a store being built (D24 build order) legitimately has
    no genome layer yet. `has_dna` / `has_mask` say which of them are there.
    """

    def __init__(self, genome_dir: Path, pool: _HandlePool) -> None:
        self.dir = Path(genome_dir)
        self._pool = pool
        self._sizes: Optional[dict] = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<GenomeView {self.dir} dna={self.has_dna} mask={self.has_mask}>"

    @property
    def dna_path(self) -> Path:
        return self.dir / "dna.h5"

    @property
    def mask_path(self) -> Path:
        return self.dir / "mask.h5"

    @property
    def has_dna(self) -> bool:
        return self.dna_path.is_file()

    @property
    def has_mask(self) -> bool:
        return self.mask_path.is_file()

    @property
    def chrom_sizes(self) -> dict:
        if self._sizes is None:
            p = self.dir / "chrom_sizes.json"
            if p.is_file():
                self._sizes = L.load_chrom_sizes(p)
            elif self.has_dna:
                f = self._pool.get(self.dna_path)
                self._sizes = {c: int(f[c].shape[0]) for c in f.keys()}
            else:
                raise StoreError(
                    f"{p}: missing, and there is no dna.h5 to derive chromosome lengths from. "
                    f"The genome layer is t7's; build it or pass chrom sizes yourself."
                )
        return dict(self._sizes)

    def attrs(self, which: str = "dna") -> dict:
        return L.read_root_attrs(
            self._pool.get(self.dna_path if which == "dna" else self.mask_path)
        )

    def dna(self, chrom: str, start_bp: int, end_bp: Optional[int] = None) -> np.ndarray:
        """`(Lbp,)` uint8 base codes over a **base-pair** interval (D10)."""
        if not self.has_dna:
            raise StoreError(f"{self.dna_path}: missing — build the genome layer (t7) first")
        f = self._pool.get(self.dna_path)
        if chrom not in f:
            raise StoreError(f"{self.dna_path}: no sequence for {chrom!r}")
        s, e = _slice_bounds(start_bp, end_bp, int(f[chrom].shape[0]), f"dna/{chrom}")
        return np.asarray(f[chrom][s:e], dtype=np.uint8)

    def dna_onehot(self, chrom: str, start_bp: int, end_bp: Optional[int] = None) -> np.ndarray:
        """`(Lbp, 4)` float32 ACGT one-hot; `N` is an all-zero row."""
        return one_hot_dna(self.dna(chrom, start_bp, end_bp))

    def mask(self, chrom: str, start_bin: int = 0, end_bin: Optional[int] = None) -> np.ndarray:
        """`(L,)` uint8 0/1 validity at **bin** resolution (D11). 1 = usable bin."""
        if not self.has_mask:
            raise StoreError(f"{self.mask_path}: missing — build the genome layer (t7) first")
        f = self._pool.get(self.mask_path)
        if chrom not in f:
            raise StoreError(f"{self.mask_path}: no mask for {chrom!r}")
        s, e = _slice_bounds(start_bin, end_bin, int(f[chrom].shape[0]), f"mask/{chrom}")
        return np.asarray(f[chrom][s:e], dtype=np.uint8)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._pool is None:
            self._pool = _HandlePool()


# ---------------------------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------------------------


class CorpusStore:
    """`CorpusStore("/…/CANDI_STORE/eic")` — the root of the OO tree (D3).

    `manifest.json` is read when present and **cross-checked against the h5 root attrs** on every
    biosample it is asked for. The attrs win by construction — they are what a reader would use
    anyway — so the manifest's only jobs here are the corpus-level view (`assay_vocabulary`) and
    the per-track experimental metadata the h5 deliberately does not carry (D20: the CSVs are the
    authority for that, and the manifest is where they landed).
    """

    def __init__(self, corpus_root: Path | str, *, kinds: Sequence[str] = L.KINDS,
                 genome_dir: Optional[Path | str] = None, check_manifest: bool = True) -> None:
        self.root = Path(corpus_root)
        if not L.biosamples_dir(self.root).is_dir():
            raise StoreError(
                f"{self.root}: no biosamples/ directory. `corpus_root` is CANDI_STORE/<corpus> "
                f"(e.g. …/CANDI_STORE/eic), not CANDI_STORE itself."
            )
        self.kinds = tuple(k for k in kinds if k in L.KINDS)
        self._pool = _HandlePool()
        self._bios: Dict[str, BiosampleStore] = {}
        self._checked: set = set()
        self._check_manifest = bool(check_manifest)
        self._genome_dir = (
            Path(genome_dir) if genome_dir is not None else L.corpus_genome_dir(self.root)
        )
        self._genome: Optional[GenomeView] = None
        self._manifest: Optional[dict] = None
        mp = L.manifest_path(self.root)
        if mp.is_file():
            self._manifest = json.loads(mp.read_text(encoding="utf-8"))
            if self._manifest.get("schema") not in L.SUPPORTED_SCHEMA_VERSIONS:
                raise StoreError(
                    f"{mp}: schema {self._manifest.get('schema')} is not one of "
                    f"{list(L.SUPPORTED_SCHEMA_VERSIONS)}"
                )

    # -- structure ----------------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CorpusStore {self.root} n_biosamples={len(self.biosamples)}>"

    @property
    def manifest(self) -> Optional[dict]:
        """The corpus manifest, or None. Never authority for structure — the h5 attrs are."""
        return self._manifest

    @property
    def biosamples(self) -> List[str]:
        """Every biosample directory holding at least one kind, verbatim and sorted (D16)."""
        d = L.biosamples_dir(self.root)
        return sorted(
            p.name
            for p in d.iterdir()
            if p.is_dir() and any((p / f"{k}.h5").is_file() for k in self.kinds)
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.biosamples)

    def __len__(self) -> int:
        return len(self.biosamples)

    def __contains__(self, name: str) -> bool:
        return any((L.biosample_dir(self.root, name) / f"{k}.h5").is_file() for k in self.kinds)

    def __getitem__(self, name: str) -> BiosampleStore:
        if name not in self._bios:
            if name not in self:
                raise StoreError(
                    f"{name!r} is not in {self.root}. Names are opaque ids used verbatim (D16) — "
                    f"nothing here strips a T_/V_/B_ prefix for you."
                )
            bs = BiosampleStore(self.root, name, self._pool, self.kinds)
            if self._check_manifest:
                self._cross_check(bs)
            self._bios[name] = bs
        return self._bios[name]

    @property
    def genome(self) -> GenomeView:
        if self._genome is None:
            self._genome = GenomeView(self._genome_dir, self._pool)
        return self._genome

    @property
    def resolution(self) -> int:
        if self._manifest is not None:
            return int(self._manifest["resolution"])
        return self[self.biosamples[0]].resolution

    @property
    def assay_vocabulary(self) -> List[str]:
        """Every assay in the corpus, control excluded. From the manifest when there is one."""
        if self._manifest is not None:
            return list(self._manifest.get("assay_vocabulary", []))
        vocab: set = set()
        for name in self.biosamples:
            vocab.update(self[name].assays())
        return sorted(vocab)

    def n_bins(self, chrom: Optional[str] = None):
        """`{chrom: n_bins}` (D13). From the manifest when present, else the first biosample."""
        if self._manifest is not None:
            nb = {c: int(v) for c, v in self._manifest["genome"]["n_bins"].items()}
        else:
            nb = self[self.biosamples[0]].n_bins()
        return nb if chrom is None else int(nb[chrom])

    def chroms(self) -> List[str]:
        return L.sort_chroms(self.n_bins())

    def biosamples_with(self, assay: str, kind: str = "counts") -> List[str]:
        return [b for b in self.biosamples if self[b].has(assay, kind)]

    def track_meta(self, biosample: str, assay: str) -> Optional[dict]:
        """The manifest's per-track experimental metadata, or None when there is no manifest.

        `{}` means "the manifest has no row for this track" — different from None ("no manifest
        at all"), and both are cases the caller must decide about rather than fill in (D19).
        """
        if self._manifest is None:
            return None
        entry = self._manifest.get("biosamples", {}).get(biosample)
        if entry is None:
            return {}
        for rec in entry.get("tracks", []):
            if rec.get("assay") == assay:
                return dict(rec)
        return {}

    # -- the loud cross-check ------------------------------------------------------------------

    def _cross_check(self, bs: BiosampleStore) -> None:
        """Manifest vs root attrs. A disagreement is a bug in one of them; say which fields."""
        if self._manifest is None or bs.name in self._checked:
            return
        self._checked.add(bs.name)
        entry = self._manifest.get("biosamples", {}).get(bs.name)
        if entry is None:
            raise StoreError(
                f"{L.manifest_path(self.root)}: biosample {bs.name!r} exists on disk but the "
                f"manifest does not name it. Regenerate the manifest (`build-manifest`)."
            )
        bad: List[str] = []
        want_tracks = [t["assay"] for t in entry.get("tracks", [])]
        got_tracks = bs.tracks()
        if [t for t in got_tracks if t not in want_tracks]:
            bad.append(f"tracks {got_tracks} vs manifest {want_tracks}")
        if entry.get("control_col") is not None and int(entry["control_col"]) != bs.control_col:
            bad.append(f"control_col {bs.control_col} vs manifest {entry['control_col']}")
        if entry.get("dtype") is not None and "counts" in bs.kinds:
            if str(entry["dtype"]) != bs.attrs("counts")[L.ATTR_DTYPE]:
                bad.append(f"dtype {bs.attrs('counts')[L.ATTR_DTYPE]} vs manifest {entry['dtype']}")
        mnb = {c: int(v) for c, v in self._manifest.get("genome", {}).get("n_bins", {}).items()}
        for chrom, nb in bs.n_bins().items():
            if chrom in mnb and mnb[chrom] != nb:
                bad.append(f"n_bins[{chrom}] {nb} vs manifest {mnb[chrom]}")
        if bad:
            raise StoreError(
                f"{bs.name}: the h5 root attrs and {L.manifest_path(self.root)} disagree — "
                + "; ".join(bad)
                + ". The h5 is the record; regenerate the manifest rather than editing it."
            )

    # -- lifecycle ----------------------------------------------------------------------------

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_bios"] = {}
        state["_genome"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._pool is None:
            self._pool = _HandlePool()
