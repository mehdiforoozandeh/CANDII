"""Batching for eDICE: the bag of (cell, assay, value) triples at one bin.

Two sources feed the same batch format.

* `RoadmapH5` -- the reference's own HDF5 layout (`targets` (n_bins, n_tracks) + `track_names`),
  used for the validation gate. Ships with their repo as a chr21 sample; the full file is the
  Edmond deposit doi:10.17617/3.VKEFB6.
* `panel_from_arrays` -- anything else, our EIC store included, once it has been read into a
  (n_bins, n_tracks) matrix with a cell id and an assay id per column.

Sampling, from `data_loaders/data_generators.py::TrainInMemGenerator`: shuffle the BINS once per
epoch, then within a batch draw one number `n_targets` for the whole batch and give each bin its own
independent permutation of the track axis. The first `n_targets` columns of that permutation are the
targets, the rest are the supports. So every track is a target in some bins and a support in others,
and the model never learns a fixed input/output split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Batch", "TrainSampler", "FixedTargetSampler", "RoadmapH5", "read_splits", "read_idmap"]

#: A batch, in the order `CellAssayCrossFactoriser.forward` wants it, plus the truth.
Batch = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def read_splits(path: Path | str) -> Dict[str, List[str]]:
    with open(path) as fh:
        return json.load(fh)


def read_idmap(path: Path | str) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(path) as fh:
        idmap = json.load(fh)
    return idmap["cell2id"], idmap["assay2id"]


class RoadmapH5:
    """The reference's Roadmap HDF5: `targets` (n_bins, n_tracks) float32, `track_names` bytes.

    Track names are `E<cell>-<assay>`, split on the FIRST hyphen (`data_loaders/metadata.py:209`) --
    which matters, because assay names such as `H2A.Z` and `H2BK120ac` contain no hyphen but a naive
    `split('-')[-1]` would still be wrong for anything that did.
    """

    def __init__(self, path: Path | str) -> None:
        import h5py

        self.path = Path(path)
        with h5py.File(self.path, "r") as fh:
            self.track_names = [t.decode() for t in fh["track_names"][:]]
            self.n_bins = int(fh["targets"].shape[0])
        self.track2col = {t: i for i, t in enumerate(self.track_names)}

    @staticmethod
    def track_cell(track: str) -> str:
        return track.split("-")[0]

    @staticmethod
    def track_assay(track: str) -> str:
        return track.split("-")[1]

    def load(self, tracks: Sequence[str], n_bins: Optional[int] = None) -> np.ndarray:
        """(n_bins, len(tracks)) in the ORDER GIVEN. h5py wants sorted fancy indices; we unsort."""
        import h5py

        cols = np.asarray([self.track2col[t] for t in tracks])
        order = cols.argsort()
        with h5py.File(self.path, "r") as fh:
            stop = self.n_bins if n_bins is None else min(n_bins, self.n_bins)
            block = fh["targets"][:stop, cols[order]]
        return np.ascontiguousarray(block[:, order.argsort()], dtype=np.float32)

    def ids_for(self, tracks: Sequence[str], cell2id: Dict[str, int],
                assay2id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        cells = np.asarray([cell2id[self.track_cell(t)] for t in tracks], dtype=np.int64)
        assays = np.asarray([assay2id[self.track_assay(t)] for t in tracks], dtype=np.int64)
        return cells, assays


class TrainSampler:
    """Per-bin random target/support partition over a fixed panel of training tracks."""

    def __init__(self, values: np.ndarray, cell_ids: np.ndarray, assay_ids: np.ndarray,
                 n_targets: int = 120, batch_size: int = 256, shuffle: bool = True,
                 rng: Optional[np.random.Generator] = None) -> None:
        if values.shape[1] != len(cell_ids) or values.shape[1] != len(assay_ids):
            raise ValueError("values must have one column per (cell id, assay id) pair")
        if not 0 < n_targets < values.shape[1]:
            raise ValueError(
                f"n_targets {n_targets} must leave at least one support out of "
                f"{values.shape[1]} tracks")
        self.values = values
        self.cell_ids = np.asarray(cell_ids, dtype=np.int64)
        self.assay_ids = np.asarray(assay_ids, dtype=np.int64)
        self.n_targets = int(n_targets)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n_tracks = values.shape[1]

    def __len__(self) -> int:
        return math.ceil(self.values.shape[0] / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        order = (self.rng.permutation(self.values.shape[0]) if self.shuffle
                 else np.arange(self.values.shape[0]))
        for b in range(len(self)):
            rows = order[b * self.batch_size:(b + 1) * self.batch_size]
            block = self.values[rows]
            n = block.shape[0]
            # one independent permutation per bin -- `TrainInMemGenerator.__getitem__`
            perm = np.argsort(self.rng.random((n, self.n_tracks)), axis=1)
            vals = np.take_along_axis(block, perm, axis=1)
            cells = self.cell_ids[perm]
            assays = self.assay_ids[perm]
            t = self.n_targets
            yield (vals[:, t:], cells[:, t:], assays[:, t:],
                   cells[:, :t], assays[:, :t], vals[:, :t])


class FixedTargetSampler:
    """Evaluation and prediction: supports and targets are fixed panels, in a fixed order.

    `truth` may be None when the targets are genuinely unobserved (the EIC prediction pass).
    """

    def __init__(self, supports: np.ndarray, support_cell_ids: np.ndarray,
                 support_assay_ids: np.ndarray, target_cell_ids: np.ndarray,
                 target_assay_ids: np.ndarray, truth: Optional[np.ndarray] = None,
                 batch_size: int = 256) -> None:
        self.supports = supports
        self.support_cell_ids = np.asarray(support_cell_ids, dtype=np.int64)
        self.support_assay_ids = np.asarray(support_assay_ids, dtype=np.int64)
        self.target_cell_ids = np.asarray(target_cell_ids, dtype=np.int64)
        self.target_assay_ids = np.asarray(target_assay_ids, dtype=np.int64)
        self.truth = truth
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return math.ceil(self.supports.shape[0] / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        for b in range(len(self)):
            lo, hi = b * self.batch_size, (b + 1) * self.batch_size
            block = self.supports[lo:hi]
            n = block.shape[0]
            tile = lambda v: np.broadcast_to(v, (n, len(v)))  # noqa: E731
            y = None if self.truth is None else self.truth[lo:hi]
            yield (block, tile(self.support_cell_ids), tile(self.support_assay_ids),
                   tile(self.target_cell_ids), tile(self.target_assay_ids), y)
