"""NPZ tree -> `counts.h5` / `peaks.h5` / `pval.h5` for ONE biosample.

One call builds one biosample, so the whole corpus is a SLURM array with no merge step (D4).
Nothing here reads a metadata CSV — that is `manifest.py`'s job — and nothing here filters a
track: **D15**, everything on disk is stored, RNA-seq included, and **D16**, the biosample name
is an opaque id used verbatim.

Source layout expected under `--source-root` (`DATA.md` "Expected on-disk layout"):

```
<ROOT>/<BIOSAMPLE>/<TRACK>/signal_DSF1_res25/{chr*.npz, metadata.json}   -> counts
                          /peaks_res25/chr*.npz                          -> peaks
                          /signal_BW_res25/chr*.npz                      -> pval
                          /file_metadata.json
```

`<TRACK>` is any subdirectory, including `chipseq-control` (D18). Only `data.files[0]` of each
npz is read — the array key is irrelevant, exactly as `handler.py::_load_npz` does it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

from candi.store import layout as L
from candi.store.layout import StoreError

__all__ = ["SourceLayout", "build_biosample", "discover_biosamples"]


# ---------------------------------------------------------------------------------------------
# source tree
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceLayout:
    """Where the npz for a (biosample, track, kind, chrom) lives. The only place that knows."""

    root: Path
    resolution: int = L.DEFAULT_RESOLUTION
    dsf: int = 1  # D6 — counts are DSF1 only

    def biosample_dir(self, biosample: str) -> Path:
        return Path(self.root) / biosample

    def track_dir(self, biosample: str, track: str) -> Path:
        return self.biosample_dir(biosample) / track

    def kind_dir(self, biosample: str, track: str, kind: str) -> Path:
        base = self.track_dir(biosample, track)
        if kind == "counts":
            return base / f"signal_DSF{self.dsf}_res{self.resolution}"
        if kind == "peaks":
            return base / f"peaks_res{self.resolution}"
        if kind == "pval":
            return base / f"signal_BW_res{self.resolution}"
        if kind in L.ARCHIVE_KINDS:
            raise StoreError(
                f"{kind!r} is an archive kind and has no npz source — it is the byte-identical "
                f"predecessor of pval.h5, placed by a promotion, never built here."
            )
        raise StoreError(f"unknown kind {kind!r}; expected one of {list(L.KINDS)}")

    def npz_path(self, biosample: str, track: str, kind: str, chrom: str) -> Path:
        return self.kind_dir(biosample, track, kind) / f"{chrom}.npz"

    def depth_json(self, biosample: str, track: str) -> Path:
        """`signal_DSF1_res25/metadata.json` — carries `depth` for this track at this DSF."""
        return self.kind_dir(biosample, track, "counts") / "metadata.json"

    def tracks(self, biosample: str) -> list:
        """Every subdirectory of the biosample dir, ordered per `layout.order_tracks`."""
        d = self.biosample_dir(biosample)
        if not d.is_dir():
            raise FileNotFoundError(f"biosample dir not found: {d}")
        names = [p.name for p in sorted(d.iterdir()) if p.is_dir() and not p.name.startswith(".")]
        # D15: no RNA-seq filter, no control filter. Everything on disk is stored.
        return L.order_tracks(names)

    def tracks_for_kind(self, biosample: str, kind: str) -> list:
        return [
            t
            for t in self.tracks(biosample)
            if self.kind_dir(biosample, t, kind).is_dir()
            and any(self.kind_dir(biosample, t, kind).glob("*.npz"))
        ]

    def chroms_for(self, biosample: str, track: str, kind: str) -> set:
        d = self.kind_dir(biosample, track, kind)
        if not d.is_dir():
            return set()
        return {p.stem for p in d.glob("*.npz")}


def discover_biosamples(source_root: Path | str) -> list:
    """Every biosample directory under the source root, verbatim and sorted (D16)."""
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"source root not found: {root}")
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "genome"
    )


def _load_npz(path: Path) -> np.ndarray:
    """Read the one array in an npz. Only `files[0]` is read; the key is irrelevant."""
    with np.load(path, allow_pickle=False) as data:
        if not data.files:
            raise StoreError(f"{path}: npz has no arrays")
        return data[data.files[0]]


def _read_depth(path: Path) -> Optional[float]:
    """`depth` out of the per-DSF `metadata.json`, or None. D19 — absent means null, not 50."""
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    v = obj.get("depth")
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if float(f).is_integer() else f


# ---------------------------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------------------------


def _resolve_chroms(
    src: SourceLayout,
    biosample: str,
    kind: str,
    tracks: Sequence[str],
    chrom_sizes: Mapping[str, int],
    chroms: Optional[Iterable[str]],
) -> list:
    """Chromosomes to build, and a loud failure when a track is missing one of them."""
    available = {t: src.chroms_for(biosample, t, kind) for t in tracks}
    if chroms is None:
        union = set().union(*available.values()) if available else set()
        wanted = L.sort_chroms(union & set(chrom_sizes))
        skipped = sorted(union - set(chrom_sizes))
        if skipped:
            print(
                f"NOTE {biosample}/{kind}: {len(skipped)} chrom(s) present in the source but absent "
                f"from chrom_sizes, not stored: {skipped[:8]}{' …' if len(skipped) > 8 else ''}",
                flush=True,
            )
    else:
        wanted = L.sort_chroms(chroms)
        missing_sizes = [c for c in wanted if c not in chrom_sizes]
        if missing_sizes:
            raise StoreError(f"{biosample}: no chrom_sizes entry for {missing_sizes}")
    if not wanted:
        raise StoreError(f"{biosample}/{kind}: no chromosomes to build")
    holes = [
        (t, c) for t in tracks for c in wanted if c not in available[t]
    ]
    if holes:
        raise StoreError(
            f"{biosample}/{kind}: {len(holes)} missing source npz — a ragged track set would give "
            f"columns that silently disagree about what a row means. First few: {holes[:5]}"
        )
    return wanted


def _scan_counts_max(
    src: SourceLayout, biosample: str, tracks: Sequence[str], chroms: Sequence[str]
) -> int:
    """D7 — the biosample's global counts max, which fixes the dtype of the whole FILE.

    This is a full extra read of the counts npz tree. It is the price of one dtype per file
    instead of one per chromosome; a per-chromosome dtype would make every read a special case.
    """
    gmax = 0
    for chrom in chroms:
        for track in tracks:
            arr = _load_npz(src.npz_path(biosample, track, "counts", chrom))
            if arr.size:
                m = int(np.max(arr))
                if m > gmax:
                    gmax = m
    return gmax


def _write_kind(
    *,
    src: SourceLayout,
    corpus_root: Path,
    biosample: str,
    kind: str,
    tracks: Sequence[str],
    chroms: Sequence[str],
    n_bins: Mapping[str, int],
    dtype: np.dtype,
    resolution: int,
    nan_policy: str,
    overwrite: bool,
    pval_scale: int = L.PVAL_SCALE,
    pval_transform: str = L.PVAL_TRANSFORM,
) -> dict:
    out = L.kind_path(corpus_root, biosample, kind)
    if out.exists() and not overwrite:
        raise StoreError(f"{out} already exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")

    clipped = np.zeros(len(tracks), dtype=np.int64)
    total_bins = 0
    depths: list = [None] * len(tracks)
    if kind == "counts":
        depths = [_read_depth(src.depth_json(biosample, t)) for t in tracks]

    with h5py.File(tmp, "w") as f:
        for chrom in chroms:
            nb = n_bins[chrom]
            block = np.zeros((nb, len(tracks)), dtype=dtype)
            for j, track in enumerate(tracks):
                where = f"{biosample}/{track}/{kind}/{chrom}"
                raw = _load_npz(src.npz_path(biosample, track, kind, chrom))
                raw = L.fit_to_n_bins(raw, nb, where)          # D13
                block[:, j] = _cast(raw, kind, dtype, where, nan_policy, clipped, j,
                                    pval_scale=pval_scale, pval_transform=pval_transform)
            ds = f.create_dataset(chrom, **L.dataset_kwargs(nb, len(tracks), dtype))
            ds[...] = block
            total_bins += nb
            del block
        extra = {}
        if kind == "pval":
            frac = (clipped / float(total_bins)).tolist() if total_bins else [0.0] * len(tracks)
            extra[L.ATTR_PVAL_CLIP_FRAC] = json.dumps(frac)
        if kind == "counts":
            extra[L.ATTR_NPZ_DEPTH] = json.dumps(depths)
        L.write_root_attrs(
            f,
            L.root_attrs(
                biosample=biosample,
                kind=kind,
                tracks=tracks,
                n_bins={c: n_bins[c] for c in chroms},
                dtype=dtype,
                source_root=str(src.root),
                resolution=resolution,
                pval_scale=pval_scale,
                pval_transform=pval_transform,
                extra=extra,
            ),
        )
    os.replace(tmp, out)
    return {
        "path": str(out),
        "kind": kind,
        "dtype": L.dtype_name(dtype),
        "tracks": list(tracks),
        "control_col": L.control_col_of(list(tracks)),
        "chroms": list(chroms),
        "bins": total_bins,
        "bytes": out.stat().st_size,
        "pval_clip_frac": (clipped / float(total_bins)).tolist() if kind == "pval" else None,
        # D25/D28 — the summary reports the codec it used, so a build log answers "what units" on
        # its own. Under D28 every entry of `pval_clip_frac` must be 0.0; a nonzero one means a
        # source value exceeded ~8.5e13 and is a fault in the source, not in the codec.
        "pval_scale": int(pval_scale) if kind == "pval" else None,
        "pval_transform": str(pval_transform) if kind == "pval" else None,
        "npz_depth": depths if kind == "counts" else None,
    }


def _cast(
    raw: np.ndarray,
    kind: str,
    dtype: np.dtype,
    where: str,
    nan_policy: str,
    clipped: np.ndarray,
    j: int,
    pval_scale: int = L.PVAL_SCALE,
    pval_transform: str = L.PVAL_TRANSFORM,
) -> np.ndarray:
    if kind == "pval":
        encoded, n_clipped = L.encode_pval(raw, pval_scale, nan_policy=nan_policy,   # D25
                                           transform=pval_transform)
        clipped[j] += n_clipped
        return encoded
    if raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)):
            raise StoreError(f"{where}: non-finite values in a {kind} track")
        if not np.all(raw == np.rint(raw)):
            raise StoreError(f"{where}: {kind} must be integer-valued, found fractional values")
    lo = int(raw.min()) if raw.size else 0
    hi = int(raw.max()) if raw.size else 0
    info = np.iinfo(dtype)
    if lo < info.min or hi > info.max:
        raise StoreError(
            f"{where}: values span [{lo}, {hi}] which does not fit {L.dtype_name(dtype)}. "
            f"For peaks that means the npz is not a 0/1 indicator; for counts the dtype scan "
            f"disagrees with the data — rebuild without --counts-dtype."
        )
    return raw.astype(dtype)


def build_biosample(
    source_root: Path | str,
    corpus_root: Path | str,
    biosample: str,
    *,
    chrom_sizes: Mapping[str, int],
    kinds: Sequence[str] = ("counts", "peaks"),
    chroms: Optional[Iterable[str]] = None,
    resolution: int = L.DEFAULT_RESOLUTION,
    dsf: int = 1,
    counts_dtype: Optional[str] = None,
    nan_policy: str = "error",
    overwrite: bool = False,
    pval_scale: int = L.PVAL_SCALE,
    pval_transform: str = L.PVAL_TRANSFORM,
) -> dict:
    """Build one biosample's `{counts,peaks,pval}.h5` from the npz tree.

    `chrom_sizes` fixes `n_bins = floor(len / resolution)` for every kind (D13). `chroms=None`
    builds every chromosome the source has that `chrom_sizes` also names; naming them explicitly
    makes a missing one an error rather than a silent omission.

    `counts_dtype` overrides the D7 scan (and skips it) — use it only when a prior run already
    told you the answer, e.g. a SLURM array rebuilding one biosample.

    Returns a summary dict, one entry per kind, suitable for a build log.
    """
    for kind in kinds:
        if kind not in L.KINDS:
            raise StoreError(f"unknown kind {kind!r}; expected a subset of {list(L.KINDS)}")
    src = SourceLayout(root=Path(source_root), resolution=resolution, dsf=dsf)
    corpus_root = Path(corpus_root)
    nbins_all = L.n_bins_map(chrom_sizes, resolution)

    summary = {"biosample": biosample, "resolution": resolution, "kinds": {}}
    for kind in kinds:
        tracks = src.tracks_for_kind(biosample, kind)
        if not tracks:
            raise StoreError(
                f"{biosample}: no track has a {src.kind_dir(biosample, '<TRACK>', kind).name} "
                f"directory with npz in it — nothing to write for kind {kind!r}"
            )
        chrom_list = _resolve_chroms(src, biosample, kind, tracks, chrom_sizes, chroms)
        if kind == "counts":
            if counts_dtype is not None:
                dtype = np.dtype(counts_dtype)
                if dtype not in [np.dtype(d) for d in L.COUNTS_DTYPES]:
                    raise StoreError(
                        f"counts dtype must be one of "
                        f"{[L.dtype_name(d) for d in L.COUNTS_DTYPES]}, got {counts_dtype}"
                    )
            else:
                gmax = _scan_counts_max(src, biosample, tracks, chrom_list)
                dtype = L.counts_dtype_for_max(gmax)
                summary["counts_max"] = gmax
        elif kind == "peaks":
            dtype = np.dtype(L.PEAKS_DTYPE)
        else:
            dtype = np.dtype(L.PVAL_DTYPE)
        summary["kinds"][kind] = _write_kind(
            src=src,
            corpus_root=corpus_root,
            biosample=biosample,
            kind=kind,
            tracks=tracks,
            chroms=chrom_list,
            n_bins=nbins_all,
            dtype=dtype,
            resolution=resolution,
            nan_policy=nan_policy,
            overwrite=overwrite,
            pval_scale=pval_scale,
            pval_transform=pval_transform,
        )
    return summary
