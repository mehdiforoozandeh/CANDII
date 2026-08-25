"""The EIC side of eDICE: which tracks it may see, and which it must predict.

**§6.2, and where it is enforced.** `training_panel` is built from `regime["biosamples"]["train"]`
and nothing else — the 51 `T_` cells. No `V_` or `B_` biosample is ever opened here, so no
validation or blind track can enter training, the support panel at prediction time, or the σ fit.
That is a one-line property of `training_panel` and it is asserted, not assumed
(`assert_no_eval_leakage`).

**The transductive substitution (§7.3, pre-registered).** eDICE has one learned embedding per cell
id, and a `V_X`/`B_X` cell has none — it was never in training, by construction. The regime declares
`(T_X, V_X)` as a pair of the SAME cell type, so a target on `V_X` is queried with `T_X`'s cell
embedding. This is the transductive analogue of the impute dial: the model is told which cell type
to answer for, not which experiment. Every eDICE row carries the caveat.

**Which tracks to emit.** Taken from `candi.bench.harness`, not re-derived: `source.pairs("impute")`
and `source.targets(pair, "impute")` are the same calls the scorer makes, so the set produced here
is the set `candi.bench.external` will demand and `provenance.missing_tracks` stays empty. The rule
they implement — an assay the TARGET cell has and the INPUT cell does NOT — is also what makes the
support panel leak-free: the queried assay is absent from `T_X`, so it cannot be copied off the
prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Panel", "TrackRequest", "load_regime", "training_panel", "requested_tracks",
           "assert_no_eval_leakage", "read_slab"]


@dataclass(frozen=True)
class Panel:
    """The training panel: one column per (training cell, assay) pval track that exists."""

    biosamples: List[str]           # the 51 T_ cells, in regime order -> cell id
    assays: List[str]               # the regime's assays, in regime order -> assay id
    tracks: List[Tuple[str, str]]   # (biosample, assay) per column
    cell_ids: np.ndarray
    assay_ids: np.ndarray

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)


@dataclass(frozen=True)
class TrackRequest:
    """One §4.1 output directory: the pair, the assay, and the (cell id, assay id) to query."""

    input_biosample: str
    target_biosample: str
    assay: str
    cell_id: int      # the INPUT cell's id -- the transductive substitution
    assay_id: int

    @property
    def dirname(self) -> str:
        return f"{self.input_biosample}__{self.target_biosample}__{self.assay}"


def load_regime(path: Path | str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def training_panel(store, regime: dict) -> Panel:
    """Every pval track carried by a `biosamples.train` cell. The 51-cell rule lives on this line."""
    biosamples = list(regime["biosamples"]["train"])
    assays = list(regime["assays"])
    assay_id = {a: i for i, a in enumerate(assays)}

    tracks: List[Tuple[str, str]] = []
    cell_ids: List[int] = []
    assay_ids: List[int] = []
    for c, name in enumerate(biosamples):
        if name not in store:
            raise KeyError(f"{name} is declared in biosamples.train but is not in the store")
        bs = store[name]
        for assay in assays:
            if bs.has(assay, "pval"):
                tracks.append((name, assay))
                cell_ids.append(c)
                assay_ids.append(assay_id[assay])
    if not tracks:
        raise ValueError("the training panel is empty; the store carries no pval for any T_ cell")
    return Panel(biosamples, assays, tracks,
                 np.asarray(cell_ids, dtype=np.int64), np.asarray(assay_ids, dtype=np.int64))


def assert_no_eval_leakage(panel: Panel) -> None:
    """§6.2 as a check, not a promise. A `V_`/`B_` column here would be a fairness violation."""
    bad = [b for b, _ in panel.tracks if b.startswith(("V_", "B_"))]
    if bad:
        raise AssertionError(
            f"§6.2: the training panel contains {sorted(set(bad))}. Rivals train on the regime's "
            f"training biosamples only; no V_ or B_ track may enter training, the support panel, "
            f"or the σ fit.")


def requested_tracks(source, panel: Panel) -> List[TrackRequest]:
    """The declared (pair, assay) set, with each target queried under its INPUT cell's embedding."""
    cell_id = {b: i for i, b in enumerate(panel.biosamples)}
    assay_id = {a: i for i, a in enumerate(panel.assays)}
    out: List[TrackRequest] = []
    for pair in source.pairs("impute"):
        if pair.input_biosample not in cell_id:
            raise KeyError(
                f"pair {pair} prompts with {pair.input_biosample}, which is not one of the "
                f"{len(panel.biosamples)} training cells eDICE learned an embedding for. The "
                f"transductive substitution has nothing to substitute.")
        for col in source.targets(pair, "impute"):
            assay = source.assays[col]
            out.append(TrackRequest(pair.input_biosample, pair.target_biosample, assay,
                                    cell_id[pair.input_biosample], assay_id[assay]))
    if not out:
        raise ValueError("the regime declares no (pair, assay) target")
    return out


def read_slab(store, panel: Panel, chrom: str, lo: int, hi: int) -> np.ndarray:
    """(hi-lo, panel.n_tracks) of raw `-log10 p`. Slabbed because a whole chromosome x the whole
    panel does not fit in memory: chr1 x ~800 tracks is ~32 GB at float32."""
    out = np.empty((hi - lo, panel.n_tracks), dtype=np.float32)
    for j, (name, assay) in enumerate(panel.tracks):
        out[:, j] = store[name][assay].pval(chrom, lo, hi)
    return out
