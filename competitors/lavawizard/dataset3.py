"""Dataset 3 — the signal tracks the 2019 challenge itself distributed, as bigwig.

This is the side the §7.4 **anchor** lives on: the port is retrained on the challenge's own
training data and must approach their published rows before it is retrained on ours. Max's 001
proved Dataset 2 and Dataset 3 are different quantities and that scores do not translate between
them (`RIVALS_PLAN.md` §1), so nothing computed here may be quoted in an internal table.

Landed on Fir at `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/{training_data,validation_data,
blind_truth}`. Filenames are `C##M##.bigwig` — cell id and mark id, the same keys
`Encode_meta.tsv` uses.

**The grid here is not our grid.** Upstream rounds the bin count *up*:

```python
nbins = (-(-end // 25))      # 00_data_generation.py:49 — CEIL
```

Our store uses `floor(chr_len / 25)` for every kind, always (`STORE.md`). On chr21 that is
1 868 400 against 1 868 399 — one bin, and the difference is real: their released checkpoints have
a `genome_25bp_embedding` with exactly 1 868 400 rows, which is how we know the ceil is what they
trained. `upstream_n_bins` is therefore the grid for anything on this side, and
`candi.store`'s is the grid for anything on ours. A track written for one and scored on the other
is silently misaligned after the first partial bin, which is why `emit.write_track` checks the
length it is given rather than trusting the caller.

**The binning reproduces theirs exactly**: NaN to zero, `arcsinh` **per base pair**, zero-pad up to
`nbins * 25`, then the mean of each 25-base group. The `arcsinh` goes on before the mean, so
`cross_cell_moments` is called with `transform="none"` on this side.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["CHROMS", "UPSTREAM_HYPERPARAMS", "factor_sizes", "schedule", "upstream_n_bins",
           "bin_values", "bin_arcsinh", "read_binned", "track_path", "parse_name", "read_meta",
           "Dataset3Reader", "contributor_pool"]

#: The 23 upstream trains on, in `Lavawizard_pipeline.sh` order. No chrY, no scaffolds — the
#: challenge bigwigs carry 123 contigs and their pipeline touches these.
CHROMS: Tuple[str, ...] = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)

#: Per-chromosome hyperparameters, transcribed from `Lavawizard_pipeline.sh` (d638b204):
#: `(batch_size, pretrain_epochs, train_epochs, n_25bp_factors, n_250bp_factors, n_5kbp_factors)`.
#:
#: There is no shared setting — the factor widths and the schedule both vary by chromosome, so a
#: model built with the wrong row will not load their checkpoint (the shapes disagree) and will not
#: reproduce their schedule if retrained. `factor_sizes()` is the accessor; use it rather than
#: indexing this table, so a missing chromosome fails by name.
UPSTREAM_HYPERPARAMS: Dict[str, Tuple[int, int, int, int, int, int]] = {
    "chr1": (21000, 150, 400, 10, 10, 45), "chr2": (20000, 150, 400, 10, 10, 45),
    "chr3": (17000, 150, 400, 25, 30, 45), "chr4": (16000, 150, 400, 25, 30, 45),
    "chr5": (15000, 150, 400, 25, 30, 45), "chr6": (14000, 150, 400, 25, 30, 45),
    "chr7": (10000, 200, 400, 25, 30, 45), "chr8": (10000, 200, 450, 25, 30, 45),
    "chr9": (10000, 200, 450, 25, 30, 45), "chr10": (10000, 150, 500, 25, 30, 45),
    "chr11": (10000, 200, 450, 25, 30, 45), "chr12": (10000, 200, 450, 25, 30, 45),
    "chr13": (10000, 200, 450, 25, 30, 45), "chr14": (10000, 200, 450, 25, 30, 45),
    "chr15": (10000, 200, 450, 25, 30, 45), "chr16": (10000, 200, 450, 25, 30, 45),
    "chr17": (10000, 200, 450, 25, 30, 45), "chr18": (10000, 200, 450, 25, 30, 45),
    "chr19": (10000, 200, 800, 25, 30, 60), "chr20": (10000, 200, 800, 25, 30, 60),
    "chr21": (10000, 200, 800, 25, 30, 60), "chr22": (10000, 200, 800, 25, 30, 60),
    "chrX": (10000, 150, 400, 25, 30, 45),
}


def factor_sizes(chrom: str) -> Dict[str, int]:
    """The three genome-embedding widths for `chrom`, as `model.Guacamole(**factor_sizes(c))`."""
    if chrom not in UPSTREAM_HYPERPARAMS:
        raise KeyError(f"{chrom!r} is not one of the 23 upstream trains on: {list(CHROMS)}")
    _, _, _, f25, f250, f5k = UPSTREAM_HYPERPARAMS[chrom]
    return {"n_25bp_factors": f25, "n_250bp_factors": f250, "n_5kbp_factors": f5k}


def schedule(chrom: str) -> Dict[str, int]:
    """`{batch_size, pretrain_epochs, train_epochs}` for `chrom`, for the anchor retrain."""
    if chrom not in UPSTREAM_HYPERPARAMS:
        raise KeyError(f"{chrom!r} is not one of the 23 upstream trains on: {list(CHROMS)}")
    bs, pre, tr, _, _, _ = UPSTREAM_HYPERPARAMS[chrom]
    return {"batch_size": bs, "pretrain_epochs": pre, "train_epochs": tr}


def upstream_n_bins(chrom_length: int, resolution: int = 25) -> int:
    """Their bin count: `ceil(len / 25)`, not our store's `floor`. See the module docstring."""
    return -(-int(chrom_length) // int(resolution))


def bin_values(values: np.ndarray, chrom_length: int, resolution: int = 25,
               *, transform: str = "arcsinh") -> np.ndarray:
    """NaN→0, optional `arcsinh` **per base**, zero-pad to a whole final bin, mean by `resolution`.

    `transform="arcsinh"` is `00_data_generation.py::get_binvals` exactly — the training signal
    space. `transform="none"` is the same grid without the transform, which is what scoring against
    `blind_truth` needs: the challenge's measures live in raw `-log10 p`, not in arcsinh.
    """
    if transform not in ("arcsinh", "none"):
        raise ValueError(f"transform must be 'arcsinh' or 'none', got {transform!r}")
    n_bins = upstream_n_bins(chrom_length, resolution)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.shape[0] != int(chrom_length):
        raise ValueError(f"expected {chrom_length} base pairs, got {v.shape[0]}")
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if transform == "arcsinh":
        v = np.arcsinh(v)                                # per BASE, before the mean — as upstream
    pad = n_bins * resolution - int(chrom_length)
    if pad:
        v = np.append(v, np.zeros(pad, dtype=np.float64))
    return v.reshape(n_bins, resolution).mean(axis=1).astype(np.float32)


def bin_arcsinh(values: np.ndarray, chrom_length: int, resolution: int = 25) -> np.ndarray:
    """`00_data_generation.py::get_binvals`, reproduced. See `bin_values`."""
    return bin_values(values, chrom_length, resolution, transform="arcsinh")


def read_binned(path: Path | str, chrom: str, *, transform: str = "arcsinh") -> np.ndarray:
    """One whole chromosome out of a bigwig, binned. The seam every consumer here shares."""
    import pyBigWig

    bw = pyBigWig.open(str(path))
    try:
        length = bw.chroms().get(str(chrom))
        if length is None:
            raise ValueError(f"{path} has no {chrom}")
        values = np.array(bw.values(str(chrom), 0, length))
    finally:
        bw.close()
    return bin_values(values, length, transform=transform)


def parse_name(filename: str) -> Tuple[str, str]:
    """`'C05M17.bigwig' -> ('C05', 'M17')`. Their own slicing (`00_data_generation.py:76-77`),
    which is positional and therefore assumes the fixed `C##M##` width."""
    stem = Path(filename).name
    if len(stem) < 6 or stem[0] != "C" or stem[3] != "M":
        raise ValueError(f"{filename!r} is not a C##M##.bigwig name")
    return stem[0:3], stem[3:6]


def track_path(root: Path | str, cell: str, mark: str) -> Path:
    return Path(root) / f"{cell}{mark}.bigwig"


def read_meta(meta_tsv: Path | str) -> List[Dict[str, str]]:
    """`Encode_meta.tsv` as a list of row dicts, with the long split column renamed `DataType`.

    Values are `T` (training), `V` (validation), `B` (blind test), matching their own rename in
    `00_data_generation.py:23-24`.
    """
    import csv

    rows: List[Dict[str, str]] = []
    with open(meta_tsv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row["DataType"] = row.pop("Training(T),Validation(V),Blind-test(B)")
            rows.append(row)
    if not rows:
        raise ValueError(f"{meta_tsv} is empty")
    return rows


class Dataset3Reader:
    """Binned `arcsinh` tracks off the challenge bigwigs, cached per `(cell, mark, chrom)`.

    One whole chromosome per read — their pipeline works that way and a bigwig is far happier with
    one big range than with thousands of small ones. chr1 is 10 M bins, so 40 MB per track cached;
    `max_cached` bounds it and the caller sizes that against the node.

    Built to plug straight into `features.cross_cell_moments`::

        reader = Dataset3Reader(root, chrom="chr21")
        avg, var, k = features.cross_cell_moments(
            reader.read, contributors, "chr21", 0, reader.n_bins,
            assay="M17", transform="none")     # <- already arcsinh'd, per base
    """

    def __init__(self, root: Path | str, chrom: str, *, max_cached: int = 8):
        import pyBigWig

        self.root = Path(root)
        self.chrom = str(chrom)
        self.max_cached = int(max_cached)
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._order: List[Tuple[str, str]] = []
        probe = sorted(self.root.glob("C*M*.bigwig"))
        if not probe:
            raise FileNotFoundError(f"no C##M##.bigwig under {self.root}")
        bw = pyBigWig.open(str(probe[0]))
        try:
            length = bw.chroms().get(self.chrom)
        finally:
            bw.close()
        if length is None:
            raise ValueError(f"{probe[0].name} has no {self.chrom}")
        #: Base pairs on this chromosome, read from the data rather than a hardcoded table.
        self.chrom_length = int(length)
        #: THEIR bin count (ceil), which is the grid everything on this side uses.
        self.n_bins = upstream_n_bins(self.chrom_length)

    def available(self) -> List[Tuple[str, str]]:
        """Every `(cell, mark)` this directory holds, sorted."""
        return sorted(parse_name(p.name) for p in self.root.glob("C*M*.bigwig"))

    def carries(self, cell: str, mark: str) -> bool:
        """The predicate `features.contributors` wants."""
        return track_path(self.root, cell, mark).exists()

    def _binned(self, cell: str, mark: str) -> np.ndarray:
        import pyBigWig

        key = (cell, mark)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        path = track_path(self.root, cell, mark)
        if not path.exists():
            raise FileNotFoundError(f"{path} — {cell}{mark} is not in {self.root}")
        bw = pyBigWig.open(str(path))
        try:
            values = np.array(bw.values(self.chrom, 0, self.chrom_length))
        finally:
            bw.close()
        binned = bin_arcsinh(values, self.chrom_length)
        while len(self._order) >= self.max_cached:
            self._cache.pop(self._order.pop(0), None)
        self._cache[key] = binned
        self._order.append(key)
        return binned

    def read(self, cell: str, mark: str, start: int, end: int) -> np.ndarray:
        """`features.ReadPvalFn`, in BIN coordinates on the upstream (ceil) grid."""
        return self._binned(cell, mark)[int(start):int(end)]


def contributor_pool(meta: Sequence[Dict[str, str]], mark: str, *,
                     splits: Sequence[str] = ("T",)) -> List[str]:
    """Cells carrying `mark` in the named splits, in metadata order.

    `splits=("T",)` is §5's rule for anything we report. `splits=("T", "V")` reproduces what
    upstream's own average spans (`00_data_generation.py` keeps both), and is the variant to use
    when the point is comparability with the challenge's published `Average` rows.
    """
    return [r["Cell_ID"] for r in meta if r["Mark_ID"] == mark and r["DataType"] in splits]
