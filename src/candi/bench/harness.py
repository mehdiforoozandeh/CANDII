"""harness — whole eval chromosomes, one track at a time, both arms.

This is the part of the suite that touches a checkpoint. Everything under it (`eic`, `partitions`,
`distributional`, `binary`, `covariate`) is pure array -> scalar and knows nothing about h5 files,
stores or torch; this module is the only place the two meet.

Three things decide its shape.

**D2 — every 25 bp bin of the eval chromosomes.** The training loaders do not do that. Both of them
walk a *window plan*: `CandiKitH5Dataset.__iter__` advances one `(T_, V_/B_)` pair per window batch,
so a track gets one window in every `n_pairs`, and `Regime.windows` drops any tile the genome mask
rejects. Either is a fine way to sample; neither covers a chromosome. So the harness plans its own
windows — a full tile of every eval chromosome — and assembles its own batches. `SOURCES` below is
the whole of that: the datasets stay the authority on *metadata* (assay order, covariates,
availability, the depth centre), and the harness decides *which windows, in which order, for which
pair*.

**"without materialising the whole panel in memory" (`EVAL_PLAN.md` §8).** The loop is pair-outer,
window-inner: one `(input, target)` pair streams its whole chromosome into per-assay buffers, is
scored, and is released before the next pair opens. Peak memory is one pair's targets, not the
panel's. The one thing that outlives a pair is the binarised peak call for `pr_by_specificity`,
which is a panel-level measure by definition (§4.2) — it is kept bit-packed, at 1/8 the size.

**`bench` must never import `candi.train`.** An evaluation package that imports the training module
inverts the dependency and makes every eval run carry the training loop's imports. `open_source`
below is the factory that replaces `train.py::_make_dataset`, and it is deliberately smaller: it
opens a dataset for evaluation and nothing else.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from candi._vendored import CLOZE, MISSING
from candi.batch import make_masker, prepare_masked_batch
from candi.bench import annotations as ann
from candi.bench import binary as B
from candi.bench import covariate as C
from candi.bench import distributional as D
from candi.bench import eic as E
from candi.bench import partitions as P
from candi.precision import no_autocast

__all__ = [
    "Pair", "TrackRecord", "EvalSource", "H5Source", "StoreSource", "open_source", "cross_cell",
    "full_tiling", "decode_groups", "stream_tracks", "score_track", "panel_specificity", "c_block",
    "run_bench", "macro_mean",
    "SCOPE_HELD_OUT", "SCOPE_GENOME_WIDE", "SCOPES", "PANELS", "panel_of", "panel_macros",
]

#: Metadata row order, as `encoder.py` reads it. Named so nothing here indexes a bare integer.
DEPTH_ROW, ASSAY_ROW, READLEN_ROW, RUNTYPE_ROW = 0, 1, 2, 3

#: The two arms of D1. `count` scores the NB mean against raw counts and is available on every
#: checkpoint; `pval` scores the Gaussian signal head against the -log10 p-value track and needs
#: `--heads count,signal`.
ARMS: Tuple[str, ...] = ("count", "pval")

#: The kinds of track. `impute` is the headline and the only one `EVAL_PLAN.md` §4 names; `denoise`
#: is opt-in, and its key carries a fourth field so the two can never collide on a backend where the
#: input and target biosample are the same one.
KINDS: Tuple[str, ...] = ("impute", "denoise")


#: The two aggregations of one scoring pass (`plan/BENCHMARK_DESIGN.md` §4). They are aggregations,
#: not two passes: the model runs once, and every measure is then computed twice, over two different
#: sets of chromosomes. A metric is not linear in position, so a genome-wide MSE cannot be narrowed
#: to a held-out MSE afterwards — the split has to happen at the track, which is why `score_track`
#: takes a `chroms` argument rather than the caller slicing a finished number.
SCOPE_HELD_OUT = "held-out"
SCOPE_GENOME_WIDE = "genome-wide"
SCOPES: Tuple[str, ...] = (SCOPE_HELD_OUT, SCOPE_GENOME_WIDE)

#: The three panels of §5.2. `V_breadth` and `B` are ranked; `V_matched` exists ONLY so the
#: `V_`→`B_` delta is readable and is never ranked — see `panel_macros`.
PANELS: Tuple[str, ...] = ("V_breadth", "V_matched", "B")


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    """One `(input biosample, target biosample)`.

    They are equal for denoising, and for a store regime that declares no `eval_pairs` (D31) and so
    can only pose leave-one-assay-out within one cell. Under a declared pairing they differ on both
    backends: the input is the `T_` cell whose tracks are the prompt, the target is the `V_`/`B_`
    cell that holds the truth.
    """
    input_biosample: str
    target_biosample: str

    def __str__(self) -> str:                                   # pragma: no cover - display only
        return f"{self.input_biosample}->{self.target_biosample}"


def cross_cell(pair: Pair, kind: str) -> bool:
    """Does this pair's ground truth come from a DIFFERENT biosample than the prompt?

    When it does, the truth arrays are the `y_*_imp` keys — the target cell's counts, p-values and
    peaks — and the decoder prompt has to be spliced with the target's own covariates
    (`vb_natural_meta`). When it does not, the batch's own `y_*` keys are the truth. Both backends
    answer this the same way, off the pair alone, because both now emit the same five imputation
    keys: `H5Source.batch` builds them from the paired `V_`/`B_` biosample and
    `StoreDataset._imp_keys` builds them from the regime's declared `eval_pairs` (D31).
    """
    return kind == "impute" and pair.target_biosample != pair.input_biosample


def track_key(pair: Pair, assay: str, kind: str) -> str:
    """`T_cell|imp_cell|assay` for imputation — `EVAL_PLAN.md` §4's key, and `eval.py`'s too.

    Denoising appends a fourth field. Wherever the imputation target is a `V_`/`B_` cell the
    three-field keys are already distinct; on a self-paired store an imputed assay is held out of
    the same biosample that denoises the others, and without the suffix the two would overwrite
    each other.
    """
    base = f"{pair.input_biosample}|{pair.target_biosample}|{assay}"
    return base if kind == "impute" else f"{base}|{kind}"


@dataclass
class TrackRecord:
    """One track's assembled prediction and truth, per chromosome, over every scored bin.

    Arrays are `float32` and indexed by ABSOLUTE bin on the chromosome, so an annotation coordinate
    and a prediction refer to the same place without a translation step anywhere.
    """
    pair: Pair
    assay: str
    kind: str
    chroms: Tuple[str, ...]
    mu: Dict[str, np.ndarray] = field(default_factory=dict)
    n: Dict[str, np.ndarray] = field(default_factory=dict)
    counts: Dict[str, np.ndarray] = field(default_factory=dict)
    signal_mu: Dict[str, np.ndarray] = field(default_factory=dict)
    signal_sigma: Dict[str, np.ndarray] = field(default_factory=dict)
    pval: Dict[str, np.ndarray] = field(default_factory=dict)
    peak_score: Dict[str, np.ndarray] = field(default_factory=dict)
    peaks: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Whether `peak_score` is a real `sigmoid(peak_logit)` or the NB-mean FALLBACK. Recorded at
    #: stream time, where the model's output dict is the authority, because the two cannot be told
    #: apart afterwards: a value range is not evidence — a well-behaved NB mean is inside [0, 1] on
    #: most bins of most assays. `binary_suite` is rank-based and survives either, but a Bernoulli
    #: NLL of an unbounded count is meaningless, so `loss_block` gates on this.
    has_peak_head: bool = False
    #: t89 — the BIN SCOPE these arrays were COMPACTED to, or `None` when index `i` is still the bin
    #: starting at `i * resolution`. It travels on the record rather than beside it because the
    #: compaction is invisible in the data: a gathered array is a perfectly ordinary float32 vector,
    #: and the only thing that can tell `score_track` its indices are no longer genomic positions is
    #: the record itself. Every POSITIONAL measure is refused while it is set.
    bin_scope: Optional[str] = None

    @property
    def key(self) -> str:
        return track_key(self.pair, self.assay, self.kind)

    @property
    def has_pval(self) -> bool:
        return bool(self.signal_mu)

    @property
    def has_count(self) -> bool:
        """Is there a COUNT PREDICTION — not count truth, which every record carries?

        True for every record `stream_tracks` builds: the NB head is on every checkpoint, so `mu`
        and `n` are always filled. `bench.external` is the caller that can say no — a rival that
        predicts `-log10 p` has no count prediction, and B1b (`RIVALS_PLAN.md` §1) forbids inventing
        a read depth to manufacture one. An absent count arm is then ABSENT, by the same rule
        `loss_block` states for an absent head: a fabricated zero scored against real counts is a
        number a reader cannot tell from a real one.
        """
        return bool(self.mu)

    @property
    def has_sigma(self) -> bool:
        """Is the pval prediction DISTRIBUTIONAL, or a bare point?

        `stream_tracks` fills `signal_sigma` whenever it fills `signal_mu`, so on the model path
        this is exactly `has_pval`. An external point-only track (`bench.external`, §4.2) has the
        mean without the spread — and until a σ-table supplies one, `gauss_suite` and `gaussian_nll`
        are ABSENT rather than computed against a spread nobody predicted.
        """
        return bool(self.signal_sigma)

    def n_bins(self, chroms: Optional[Sequence[str]] = None) -> int:
        return int(sum(len(self.counts[c]) for c in (self.chroms if chroms is None else chroms)))


# ---------------------------------------------------------------------------
# window planning — D2
# ---------------------------------------------------------------------------

def full_tiling(n_bins: int, context_bins: int) -> List[int]:
    """Start bins of a tiling that covers `[0, n_bins)` with windows of `context_bins`.

    The last tile is pulled BACK to end exactly at `n_bins` rather than run past it, so it overlaps
    its predecessor instead of needing a short-window code path. A bin in that overlap is predicted
    twice, under two different contexts, and the harness keeps the second — the one from the window
    the bin sits further inside. Two legitimate predictions of the same bin is a property of a
    convolutional model with finite context, not an error; what would be an error is scoring a bin
    twice, and writing by absolute index makes that impossible.
    """
    if context_bins <= 0:
        raise ValueError(f"context_bins must be positive, got {context_bins}")
    if n_bins < context_bins:
        raise ValueError(
            f"chromosome is {n_bins} bins and the model's context is {context_bins}; there is no "
            f"window to score. Evaluate on a chromosome at least one context long."
        )
    starts = list(range(0, n_bins - context_bins + 1, context_bins))
    if starts[-1] + context_bins < n_bins:
        starts.append(n_bins - context_bins)
    return starts


# ---------------------------------------------------------------------------
# the metadata prompt for an imputation slot
# ---------------------------------------------------------------------------

def vb_natural_meta(t_meta: torch.Tensor, vb_meta: torch.Tensor,
                    y_avail: torch.Tensor) -> torch.Tensor:
    """Splice the target biosample's own covariates into the slots the input cell does not have.

    Lifted from `eval.py::_build_vb_natural_missing_meta` (which took it verbatim from
    `sandbox/train.py`), minus the canonical-median fallback: this suite does not have a canonical
    median table, and D7's discipline — say what the pool is rather than assume it away — applies to
    a prompt just as much as to a variance vector. A slot whose target covariates are the `-1`
    sentinel keeps the sentinel, and `stream_tracks` refuses to score it.

    Without this the depth-offset head is told the INPUT cell's sequencing depth for a track that
    cell does not have, so every imputed level is off by the depth ratio between two biosamples.
    """
    mixed = t_meta.clone()
    missing = (y_avail.to(mixed.device) == 0).unsqueeze(1).expand_as(mixed)
    vb = vb_meta.to(mixed.device)
    valid = (vb[:, DEPTH_ROW:DEPTH_ROW + 1, :] != float(MISSING)).expand_as(mixed)
    use = missing & valid
    mixed[use] = vb[use]
    return mixed


# ---------------------------------------------------------------------------
# sources — the factory that replaces train.py::_make_dataset
# ---------------------------------------------------------------------------

class EvalSource:
    """What the harness needs from a corpus. Two implementations: a baked h5, and a store.

    Everything below is metadata plus three verbs — `pairs`, `windows`, `batch` — and the two
    optional DSF verbs the C-block needs. Nothing here knows what a metric is.
    """

    assays: List[str]
    context_bins: int
    resolution: int
    eval_chroms: Tuple[str, ...]
    dsf_levels: Tuple[int, ...]
    meta_rows: int
    kind: str
    #: t89 — the EVAL SCOPE: a region set the scored window plan is cut down to, or `None` for every
    #: window of every eval chromosome, which is what every source did before this existed. It is
    #: the same `contain` rule D32 applies to the TRAIN split (`store.regime.RegionSet`) and the same
    #: chromosome-0-anchored tiling, so a regime that trains and evaluates on one BED shares a
    #: window grid with every other regime rather than re-anchoring per region.
    #:
    #: WHY IT LIVES ON THE SOURCE. `stream_tracks`, `external.stream_truth` and `c_block` all walk
    #: `windows()`, so one attribute here restricts a checkpoint's eval and a rival's eval by the
    #: same rule, in the same positions. A scope threaded through the scoring calls instead would
    #: have to be threaded through each of them separately, and the first one anybody forgot would
    #: score two methods on two different exams while both banners said `crps`.
    eval_regions: Optional[Any] = None

    def depth_center(self) -> float:
        raise NotImplementedError

    def n_bins(self, chrom: str) -> int:
        raise NotImplementedError

    def pairs(self, kind: str) -> List[Pair]:
        raise NotImplementedError

    def targets(self, pair: Pair, kind: str) -> List[int]:
        """Assay column indices this pair supplies ground truth for, under `kind`."""
        raise NotImplementedError

    def windows(self, chrom: str) -> List[int]:
        starts = full_tiling(self.n_bins(chrom), self.context_bins)
        if self.eval_regions is None:
            return starts
        return [int(s) for s in self.eval_regions.contained_starts(
            chrom, starts, self.context_bins, self.resolution)]

    def scored_bins(self, chrom: str) -> Optional[np.ndarray]:
        """The bins this source will actually predict on `chrom`, or `None` for all of them.

        `None` rather than `arange(n_bins)` is the load-bearing part: it is what lets every caller
        below keep the untouched path bit for bit, instead of gathering a full-length array through
        an identity index and hoping numpy gives back the same floats.

        Derived from `windows()` rather than from the region set, so it cannot disagree with what
        was predicted. `unique` because `full_tiling` pulls its last window back to end on the
        chromosome, and two windows may therefore claim the same bin — scoring it twice would
        weight it twice.
        """
        if self.eval_regions is None:
            return None
        s = np.asarray(self.windows(chrom), dtype=np.int64)
        if s.size == 0:
            return s
        cb = int(self.context_bins)
        return np.unique(np.repeat(s, cb) + np.tile(np.arange(cb, dtype=np.int64), s.size))

    def scope(self) -> Dict[str, Any]:
        """The eval scope, for `provenance`. Two runs are comparable only if these match."""
        if self.eval_regions is None:
            return {"name": "full", "note": "every window of every eval chromosome"}
        kept = {c: self.scored_bins(c) for c in self.eval_chroms}
        n_scored = int(sum(len(v) for v in kept.values()))
        n_full = int(sum(self.n_bins(c) for c in self.eval_chroms))
        return {
            "name": "regions",
            **self.eval_regions.to_dict(),
            "n_regions": len(self.eval_regions.intervals),
            "windows": {c: len(self.windows(c)) for c in self.eval_chroms},
            "scored_bins": n_scored,
            "full_bins": n_full,
            "fraction": (n_scored / n_full) if n_full else 0.0,
            "note": "SELECTION SCOPE, not the leaderboard's. Positional measures (mseprom, "
                    "msegene, mseenh, the P-block) are ABSENT here — a compacted array's index is "
                    "no longer a genomic position, so an annotation interval would read the wrong "
                    "loci.",
        }

    def batch(self, pair: Pair, chrom: str, starts: Sequence[int], kind: str, *,
              x_dsf: int = 1) -> Dict[str, Any]:
        raise NotImplementedError

    def counts_at_dsf(self, pair: Pair, chrom: str, starts: Sequence[int],
                      dsf: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def provenance(self) -> Dict[str, Any]:
        raise NotImplementedError


class H5Source(EvalSource):
    """A baked `candi.prep.bake` h5. Imputation targets are the paired `V_`/`B_` biosample's assays.

    The h5's windows are FROZEN — `counts_dsf1` is indexed by window row, not by genomic position —
    so this backend cannot re-tile. It uses the bake's own eval windows, places each one at
    `windows/start // resolution`, and **refuses to score a chromosome the bake left holes in**
    (D2). The gap is reported in bins rather than as a warning, because a partially covered
    chromosome scored as if it were whole is exactly the silent subsample D2 exists to forbid.
    """

    kind = "h5"

    def __init__(self, h5_path: Path | str, *, regime: str = "type1",
                 chroms: Optional[Sequence[str]] = None, imp_prefixes: Sequence[str] = ("V_", "B_"),
                 cell_cond: str = "off"):
        import h5py

        from candi.dataset import CandiKitH5Dataset, cell_id_map

        self.path = Path(h5_path)
        self.ds = CandiKitH5Dataset(
            self.path, regime, train=False, batch_size=1, biosample_prefix="T_",
            dsf_sampling="off", seed=0, shuffle=False, eval_include_vb_ground_truth=True,
            imp_prefixes=imp_prefixes, h5_cache_ram=False, cell_cond=cell_cond,
        )
        self.assays = list(self.ds.assays)
        self.context_bins = int(self.ds.context_bins)
        self.resolution = int(self.ds.resolution)
        self.dsf_levels = tuple(int(d) for d in self.ds.dsf_list)
        self.meta_rows = int(self.ds.meta_rows)
        self.cell_cond = str(cell_cond)
        self._cell_ids = cell_id_map(self.ds._bios_order) if cell_cond != "off" else {}
        self._h5 = h5py.File(self.path, "r")

        want = set(chroms) if chroms else set(self.ds.eval_chroms)
        # `windows/start` is in BASE PAIRS. Every placement below divides by the resolution once,
        # here, so no other line in this file has to remember which unit it is in.
        self._rows: Dict[str, Dict[int, int]] = {}
        for i in self.ds._eval_indices:
            chrom, start_bp, _end, _rt = self.ds._windows[i]
            if chrom in want:
                self._rows.setdefault(chrom, {})[int(start_bp) // self.resolution] = int(i)
        self.eval_chroms = tuple(c for c in self.ds.eval_chroms if c in self._rows)
        if not self.eval_chroms:
            raise ValueError(
                f"{self.path} has no eval windows on {sorted(want)}; the bake's eval chromosomes "
                f"are {list(self.ds.eval_chroms)}"
            )
        self._n_bins = {c: max(self._rows[c]) + self.context_bins for c in self.eval_chroms}
        self._check_coverage()

    def _check_coverage(self) -> None:
        for chrom in self.eval_chroms:
            covered = np.zeros(self._n_bins[chrom], dtype=bool)
            for s in self._rows[chrom]:
                covered[s:s + self.context_bins] = True
            missing = int((~covered).sum())
            if missing:
                raise ValueError(
                    f"{self.path}: {chrom} is {self._n_bins[chrom]} bins over the bake's eval "
                    f"windows but {missing} of them ({100.0 * missing / covered.size:.2f}%) are in "
                    f"no window. D2 scores every 25 bp bin, and a track assembled over a gapped "
                    f"tiling would score whatever the buffer was initialised to at those bins. "
                    f"Re-bake the eval chromosome as a contiguous tile, or score from a store."
                )

    def depth_center(self) -> float:
        from candi.dataset import h5_depth_center
        return float(h5_depth_center(self.path))

    def n_bins(self, chrom: str) -> int:
        return int(self._n_bins[chrom])

    def windows(self, chrom: str) -> List[int]:
        # NO EVAL SCOPE HERE, and it is a data property rather than an omission: a bake's windows
        # are frozen rows of `counts_dsf1`, so this backend cannot re-tile and a region cut would
        # have to drop whole baked rows. `open_source` refuses the combination by name.
        return sorted(self._rows[chrom])

    def pairs(self, kind: str) -> List[Pair]:
        t_bios = sorted(b for b in self.ds._bios_order if b.startswith("T_"))
        if kind == "denoise":
            return [Pair(t, t) for t in t_bios]
        return [Pair(t, imp) for t in t_bios for imp in self.ds._all_imp_biosamples(t)]

    def _meta1(self, bios: str) -> np.ndarray:
        return np.asarray(self._h5["biosamples"][bios.replace("/", "_")]["meta_dsf1"],
                          dtype=np.float64)

    def targets(self, pair: Pair, kind: str) -> List[int]:
        t_depth = self._meta1(pair.input_biosample)[DEPTH_ROW]
        if kind == "denoise":
            return [a for a in range(len(self.assays)) if t_depth[a] != MISSING]
        imp_depth = self._meta1(pair.target_biosample)[DEPTH_ROW]
        return [a for a in range(len(self.assays))
                if t_depth[a] == MISSING and imp_depth[a] != MISSING]

    def _rowsel(self, chrom: str, starts: Sequence[int]) -> np.ndarray:
        rows = self._rows[chrom]
        try:
            return np.asarray([rows[int(s)] for s in starts], dtype=np.int64)
        except KeyError as exc:                                 # pragma: no cover - guarded upstream
            raise KeyError(f"{chrom}: no baked window at start bin {exc}") from exc

    def _read(self, bios: str, name: str, wi: np.ndarray) -> np.ndarray:
        """`h5py` fancy indexing wants increasing indices; put the rows back in caller order."""
        g = self._h5["biosamples"][bios.replace("/", "_")]
        order = np.argsort(wi, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return np.asarray(g[name][wi[order]])[inv]

    def batch(self, pair: Pair, chrom: str, starts: Sequence[int], kind: str, *,
              x_dsf: int = 1) -> Dict[str, Any]:
        wi = self._rowsel(chrom, starts)
        t = pair.input_biosample
        B, L, F, R = len(starts), self.context_bins, len(self.assays), self.meta_rows
        if x_dsf not in self.dsf_levels:
            raise ValueError(f"{self.path} carries DSF levels {self.dsf_levels}, not {x_dsf}")

        x_counts = self._read(t, f"counts_dsf{x_dsf}", wi).astype(np.float32)
        y_counts = self._read(t, "counts_dsf1", wi).astype(np.float32)
        x_meta1 = np.asarray(self._h5["biosamples"][t.replace("/", "_")][f"meta_dsf{x_dsf}"],
                             dtype=np.float32)
        y_meta1 = np.asarray(self._h5["biosamples"][t.replace("/", "_")]["meta_dsf1"],
                             dtype=np.float32)

        out: Dict[str, Any] = {
            "x_data": torch.full((B, L, F), float(MISSING)),
            "x_meta": torch.full((B, R, F), float(MISSING)),
            "x_avail": torch.zeros(B, F),
            "y_data": torch.full((B, L, F), float(MISSING)),
            "y_meta": torch.full((B, R, F), float(MISSING)),
            "y_avail": torch.zeros(B, F),
            "x_dna": torch.from_numpy(self._read(t, "dna", wi).astype(np.float32)),
            "y_pval": torch.from_numpy(self._read(t, "pval", wi).astype(np.float32)),
            "y_peaks": torch.from_numpy(self._read(t, "peaks", wi).astype(np.float32)),
            "control_data": torch.from_numpy(self._read(t, "control", wi).astype(np.float32)),
            "control_meta": torch.zeros(B, R, 1),
            "control_avail": torch.zeros(B, 1),
            "y_dsf": torch.ones(B, F, dtype=torch.int64),
            "x_dsf": torch.full((B, F), int(x_dsf), dtype=torch.int64),
            "control_x_dsf": torch.ones(B, dtype=torch.int64),
            "biosample_name": t,
        }
        out["control_meta"][:, :4] = torch.from_numpy(
            self._read(t, "control_meta", wi).astype(np.float32))
        out["control_avail"][:, 0] = (out["control_data"] != 0).any(dim=1).any(dim=1).float()

        for a in range(F):
            if float(x_meta1[DEPTH_ROW, a]) != MISSING:
                out["x_data"][:, :, a] = torch.from_numpy(x_counts[:, :, a])
                out["x_meta"][:, :4, a] = torch.from_numpy(x_meta1[:, a])
                out["x_avail"][:, a] = 1.0
            if float(y_meta1[DEPTH_ROW, a]) != MISSING:
                out["y_data"][:, :, a] = torch.from_numpy(y_counts[:, :, a])
                out["y_meta"][:, :4, a] = torch.from_numpy(y_meta1[:, a])
                out["y_avail"][:, a] = 1.0

        if self.cell_cond != "off":
            from candi.dataset import base_cell_type
            cid = float(self._cell_ids[base_cell_type(t)])
            out["x_meta"][:, 4, :] = cid
            out["y_meta"][:, 4, :] = cid
            out["control_meta"][:, 4, :] = cid

        if kind == "impute":
            imp = pair.target_biosample
            imp_meta1 = np.asarray(
                self._h5["biosamples"][imp.replace("/", "_")]["meta_dsf1"], dtype=np.float32)
            imp_counts = self._read(imp, "counts_dsf1", wi).astype(np.float32)
            y_data_imp = torch.full((B, L, F), float(MISSING))
            for a in range(F):
                if float(imp_meta1[DEPTH_ROW, a]) != MISSING:
                    y_data_imp[:, :, a] = torch.from_numpy(imp_counts[:, :, a])
            out["y_data_imp"] = y_data_imp
            out["y_pval_imp"] = torch.from_numpy(self._read(imp, "pval", wi).astype(np.float32))
            out["y_peaks_imp"] = torch.from_numpy(self._read(imp, "peaks", wi).astype(np.float32))
            vb = torch.full((B, R, F), float(MISSING))
            vb[:, :4, :] = torch.from_numpy(imp_meta1)
            if self.cell_cond != "off":
                vb[:, 4, :] = out["y_meta"][:, 4, :]
            out["y_meta_imp"] = vb
            out["imp_biosample_name"] = imp
        return out

    def counts_at_dsf(self, pair: Pair, chrom: str, starts: Sequence[int],
                      dsf: int) -> np.ndarray:
        """The bake's OWN downsampling ladder, not a re-thinning of it (`depthcounterfact`'s truth)."""
        if dsf not in self.dsf_levels:
            raise ValueError(f"{self.path} carries DSF levels {self.dsf_levels}, not {dsf}")
        return self._read(pair.target_biosample, f"counts_dsf{dsf}",
                          self._rowsel(chrom, starts)).astype(np.float32)

    def close(self) -> None:
        self._h5.close()

    def provenance(self) -> Dict[str, Any]:
        return {
            "data_source": "h5",
            "h5": str(self.path),
            "assays": list(self.assays),
            "eval_chroms": list(self.eval_chroms),
            "n_bins": {c: self.n_bins(c) for c in self.eval_chroms},
            "context_bins": self.context_bins,
            "resolution": self.resolution,
            "dsf_levels": list(self.dsf_levels),
        }


class StoreSource(EvalSource):
    """A `CANDI_STORE` corpus behind a regime file. Imputation here has TWO cases.

    **The pairing is DECLARED (D31), never inferred (t80).** An imputation prompts with one
    biosample and scores against a different one that holds the held-out assays — `T_X` prompts,
    `V_X` and `B_X` are the truth. The bake finds the second by string surgery on the first's name;
    D16 forbids that here, because store biosample names are opaque ids. So the pairs come off the
    regime's `eval_pairs` list, which `tools/declare_eval_pairs.py` writes and can re-check against
    the corpus. `H5Source.pairs` and this now mean the same thing, and a store-scored panel and an
    h5-scored panel are the same exam.

    **With no `eval_pairs` the source self-pairs, exactly as it did before t80**, and says so out
    loud. A regime that declares no pairing has not asked for an imputation evaluation (D31's own
    words), and inventing one from the names is the surgery D16 rules out; but a self-paired run is
    a *different and harder exam* — the prompt is the eval cell's other blind assays rather than
    everything else known about that cell type — so it must never be mistaken for the benchmark's.
    Hence a printed warning rather than a silent fallback or a hard error.

    Imputation is still held out **by mask** on this path: `stream_tracks` calls `_apply_loo_mask`
    on the target column of every store forward pass and `decode_groups` forces one assay per pass.
    Under a declared pair the column is already `MISSING` in the prompt (the splits are disjoint on
    `(cell, assay)`, and `targets` below computes the panel so that it is), so the mask is a
    tripwire rather than the mechanism — and it is what keeps the path correct on a corpus whose
    splits are *not* disjoint.

    Batch assembly is `StoreDataset._make_batch`, called with a window list this class substitutes
    for the regime's plan. Re-implementing it here would duplicate the thinning, the depth-adjusted
    metadata row and the D19 availability rule, and those three are exactly the places a second copy
    would drift. Same package, deliberate: `_make_batch(name, widx, rng)` is a pure function of its
    arguments and `self._windows`.
    """

    kind = "store"

    def __init__(self, regime_path: Path | str, *, chroms: Optional[Sequence[str]] = None,
                 biosamples: Optional[Sequence[str]] = None,
                 eval_regions: Optional[Path | str] = None):
        from candi.store.dataset import StoreDataset
        from candi.store.regime import Regime, RegionSet

        self.regime_path = Path(regime_path)
        regime = Regime.from_file(self.regime_path)
        # A PATH, not the regime's own `regions` block, and not a boolean pointing at it. D32's
        # block says what the TRAIN split is cut to; `regime.eic_19` declares none and still has to
        # be able to select cheaply, and under `regime.eic_pilot` the two scopes being the same BED
        # is a fact worth reading off the run record rather than one the code assumes.
        self.eval_regions = None if eval_regions is None else RegionSet.from_bed(eval_regions)
        declared = [(str(a), str(b)) for a, b in regime.eval_pairs]
        pool, declared = self._select(declared, biosamples)
        self.ds = StoreDataset(regime, train=False, batch_size=1, dsf_sampling="off",
                               shuffle=False, deterministic=True, biosamples=pool)
        #: D31 — `((input, target), …)` as `Pair`s, restricted to what `--biosamples` asked for.
        self.eval_pairs: List[Pair] = [Pair(a, b) for a, b in declared]
        self.assays = list(self.ds.assays)
        self.context_bins = int(self.ds.context_bins)
        self.resolution = int(self.ds.resolution)
        self.dsf_levels = tuple(int(d) for d in self.ds.dsf_list) or (1,)
        self.meta_rows = int(self.ds.meta_rows)
        want = list(chroms) if chroms else list(self.ds.eval_chroms)
        self._n_bins = self.ds.corpus.n_bins()
        self.eval_chroms = tuple(c for c in want if c in self._n_bins)
        if not self.eval_chroms:
            raise ValueError(f"none of {want} is a chromosome of {self.ds.corpus.root}")
        self._run_seed = int(self.ds.run_seed)
        self.pair_overlaps = self._overlaps()
        self._announce_pairing()

    @staticmethod
    def _select(declared: List[Tuple[str, str]], biosamples: Optional[Sequence[str]]
                ) -> Tuple[Optional[List[str]], List[Tuple[str, str]]]:
        """Resolve `--biosamples` against the declared pairing. Returns `(loader pool, pairs)`.

        A caller who names a panel names the cells they care about, and under a declared pairing
        that is almost always the TRUTH cells — `--biosamples V_DND-41` means "score V_DND-41", not
        "prompt with it". So a named cell selects every declared pair it appears on, either side,
        and the loader pool becomes those pairs' inputs. Passing the pool through unexamined would
        hand `StoreDataset` a list of `V_` cells, leave `eval_pairs` matching none of them, and
        silently drop the panel back to the self-paired exam t80 exists to remove.

        Naming nothing the regime pairs is refused rather than self-paired, for the same reason.
        """
        if biosamples is None:
            return None, declared
        want = list(biosamples)
        if not declared:
            return want, declared
        keep = [p for p in declared if p[0] in want or p[1] in want]
        if not keep:
            raise ValueError(
                f"--biosamples names {want}, and this regime's `eval_pairs` name none of them on "
                f"either side. Name an input or a target of a declared pair, or drop the flag to "
                f"score the whole declared panel."
            )
        return list(dict.fromkeys(a for a, _ in keep)), keep

    def _overlaps(self) -> Dict[str, List[str]]:
        """`{pair: [assay, …]}` — assays a declared pair's INPUT and TARGET cells both hold.

        `BENCHMARK_DESIGN.md` §5.1 records the EIC splits as disjoint on `(cell, assay)`, with zero
        overlaps over all 89 cells, which is what makes "assays the truth cell has and the input
        cell does not" (`targets`) and "every assay the truth cell holds" the same set. This
        measures that claim on the corpus in front of us instead of assuming it. A non-empty result
        is not fatal — `targets` drops the overlapping assay, so nothing leaks — but it means the
        panel is SMALLER than the truth cells' assay count, and a shrunken panel that nobody
        mentioned is the failure this records rather than hides.
        """
        out: Dict[str, List[str]] = {}
        for pair in self.eval_pairs:
            x = self.ds._availability[pair.input_biosample]
            t = self.ds._availability[pair.target_biosample]
            both = [self.assays[a] for a in range(len(self.assays)) if bool(x[a]) and bool(t[a])]
            if both:
                out[str(pair)] = both
        return out

    def _announce_pairing(self) -> None:
        if not self.eval_pairs:
            print(
                f"[bench] {self.regime_path} declares no `eval_pairs` (D31), so every store pair "
                f"is SELF-PAIRED: the prompt is the eval cell's own other assays, held out by "
                f"mask, not its paired training cell's tracks. That is a different and harder exam "
                f"than the benchmark's, and a cell holding one assay has no leave-one-out at all. "
                f"Run tools/declare_eval_pairs.py to declare the pairing.", flush=True)
            return
        n_exp = sum(len(self.targets(p, "impute")) for p in self.eval_pairs)
        print(f"[bench] {len(self.eval_pairs)} declared eval pair(s), {n_exp} scoreable "
              f"experiment(s)", flush=True)
        if self.pair_overlaps:
            print(f"[bench] {len(self.pair_overlaps)} declared pair(s) are NOT disjoint on "
                  f"(cell, assay); the shared assays are not scored: {self.pair_overlaps}",
                  flush=True)

    def depth_center(self) -> float:
        return float(self.ds.depth_center())

    def n_bins(self, chrom: str) -> int:
        return int(self._n_bins[chrom])

    def pairs(self, kind: str) -> List[Pair]:
        """The declared `T_ -> V_/B_` pairs for imputation; self-pairs for denoising and for a
        regime that declares none.

        Denoising scores a cell against itself by definition, so its pairs are the prompt pool
        self-paired — the same `[Pair(t, t) for t in t_bios]` `H5Source.pairs` returns. Note that
        under a declared pairing `StoreDataset` makes the pool the pair INPUTS, so denoising runs
        on the `T_` cells, which is again what the h5 path does.
        """
        if kind == "denoise" or not self.eval_pairs:
            return [Pair(b, b) for b in self.ds.biosample_pool]
        return list(self.eval_pairs)

    def targets(self, pair: Pair, kind: str) -> List[int]:
        """The assay columns of the **target** biosample this pair supplies ground truth for.

        The panel belongs to the truth cell, never to the prompt: `BENCHMARK_DESIGN.md` §5.1 states
        it as *assays the truth cell has and the input cell does not*, and §8 says the same thing
        from the other end — every eval pair is a mark the `V_`/`B_` cell has and its paired `T_`
        cell lacks. That is `H5Source.targets`' rule, and it is now this one.

        Two degenerate cases keep their old answer. **Denoising** scores the prompt cell against
        itself, so its panel is the input's own assays. **A self-paired imputation** — the only
        thing a regime without `eval_pairs` can pose — has one cell playing both roles, where "has
        and does not have" is empty and "every assay the truth cell holds" is the only usable rule;
        that is what this returned for every store pair before t80.
        """
        if kind == "denoise":
            avail = self.ds._availability[pair.input_biosample]
            return [a for a in range(len(self.assays)) if bool(avail[a])]
        truth = self.ds._availability[pair.target_biosample]
        if not cross_cell(pair, kind):
            return [a for a in range(len(self.assays)) if bool(truth[a])]
        prompt = self.ds._availability[pair.input_biosample]
        return [a for a in range(len(self.assays)) if bool(truth[a]) and not bool(prompt[a])]

    def _raw_batch(self, pair: Pair, chrom: str, starts: Sequence[int]) -> Dict[str, Any]:
        self.ds._windows = [(chrom, int(s)) for s in starts]
        free = np.random.default_rng(self._run_seed)
        imp = pair.target_biosample if pair.target_biosample != pair.input_biosample else None
        return self.ds._make_batch(pair.input_biosample, list(range(len(starts))), free,
                                   imp_target=imp)

    def batch(self, pair: Pair, chrom: str, starts: Sequence[int], kind: str, *,
              x_dsf: int = 1) -> Dict[str, Any]:
        out = self._raw_batch(pair, chrom, starts)
        if x_dsf != 1:
            self._thin_input(out, pair, chrom, starts, x_dsf)
        return out

    def _thin_input(self, out: Dict[str, Any], pair: Pair, chrom: str,
                    starts: Sequence[int], dsf: int) -> None:
        """Thin the ENCODER INPUT to `dsf` and move the depth row with it.

        `StoreDataset` generates its ladder rather than storing one (D6), so a store's DSF-`d` input
        is a binomial thinning of its DSF-1 counts — the same operation the loader performs — and
        the covariate must fall by `log2(d)` alongside, or the model is told a depth its input
        contradicts. That identity is `prep/bake.py`'s F4 gate and `StoreDataset._depth_adjusted`.

        **`draw_seed` IS CALLED WITHOUT `side`, AND THAT IS THE POINT (t37).** `counts_at_dsf`
        below omits it too, so on a self-pair the two build the SAME generator over the same array
        and return the same bytes. That is ONE LADDER READ TWICE, not the identity-copy leak t27
        closed in `StoreDataset._thin`. The bake materialises `counts_dsf{d}` once and hands the
        same rows to whoever asks — `H5Source.batch` and `H5Source.counts_at_dsf` literally read
        one h5 dataset — and a generated ladder is only a substitute for a stored one if it agrees
        with itself. `_thin` is the opposite case: it draws TWO experiments, so its two calls must
        differ, which is why `side` is keyword-only and has no default there.

        Nothing compares the two anyway: this method's arrays reach only `_depthblind_latents`
        (latent invariance, no truth), and `counts_at_dsf`'s reach only `depthcounterfact` (whose
        latent is encoded at DSF 1). Adding `side` here would move both numbers by pure noise and
        split the store from the h5. `tests/test_bench_harness.py` pins that it stays absent.
        """
        from candi.store.dataset import draw_seed, dsf_milli, thin_counts

        x = out["x_data"].numpy()
        for a, assay in enumerate(self.assays):
            if float(out["x_avail"][0, a]) <= 0:
                continue
            for j, s in enumerate(starts):
                rng = np.random.default_rng(
                    draw_seed(self._run_seed, pair.input_biosample, assay, chrom, int(s),
                              dsf_milli(dsf)))
                x[j, :, a] = thin_counts(x[j, :, a].astype(np.int64), dsf, rng).astype(np.float32)
            out["x_meta"][:, DEPTH_ROW, a] -= math.log2(dsf)
        out["x_data"] = torch.from_numpy(x)
        out["x_dsf"] = torch.full_like(out["x_dsf"], int(dsf))

    def counts_at_dsf(self, pair: Pair, chrom: str, starts: Sequence[int],
                      dsf: int) -> np.ndarray:
        """The C-block's depth ladder, thinned off whichever cell holds the ground truth.

        Under a declared pair that is the TARGET cell, and its counts arrive as `y_data_imp`; the
        prompt cell's `y_data` column for the same assay is `MISSING`, so reading it would hand C3
        a ladder of `-1`s. The thinning is keyed on the truth cell's own name for the same reason
        `StoreDataset._imp_keys` keys it there: two pairs sharing a target must see the same ground
        truth at the same `(window, dsf)`.
        """
        from candi.store.dataset import draw_seed, dsf_milli, thin_counts

        out = self._raw_batch(pair, chrom, starts)
        cross = pair.target_biosample != pair.input_biosample
        truth_bios = pair.target_biosample if cross else pair.input_biosample
        y = (out["y_data_imp"] if cross else out["y_data"]).numpy().copy()
        if dsf == 1:
            return y
        avail = self.ds._availability[truth_bios]
        for a, assay in enumerate(self.assays):
            if not bool(avail[a]):
                continue
            for j, s in enumerate(starts):
                rng = np.random.default_rng(
                    draw_seed(self._run_seed, truth_bios, assay, chrom, int(s),
                              dsf_milli(dsf)))
                y[j, :, a] = thin_counts(y[j, :, a].astype(np.int64), dsf, rng).astype(np.float32)
        return y

    def close(self) -> None:
        self.ds.corpus.close()

    def provenance(self) -> Dict[str, Any]:
        from candi.store import layout as L

        manifest = L.manifest_path(self.ds.corpus.root)
        return {
            "data_source": "store",
            "regime": str(self.regime_path),
            "store": str(self.ds.corpus.root),
            "store_manifest": str(manifest) if manifest.exists() else None,
            "assays": list(self.assays),
            "eval_chroms": list(self.eval_chroms),
            "n_bins": {c: self.n_bins(c) for c in self.eval_chroms},
            "context_bins": self.context_bins,
            "resolution": self.resolution,
            "dsf_levels": list(self.dsf_levels),
            "biosamples": list(self.ds.biosample_pool),
            # t80 — which exam was sat. `eval_pairs: []` with `self_paired: true` is the pre-t80
            # leave-one-out-within-the-eval-cell run, and a report carrying it is not comparable
            # with one carrying a declared pairing.
            "eval_pairs": [[p.input_biosample, p.target_biosample] for p in self.eval_pairs],
            "self_paired": not self.eval_pairs,
            "eval_pair_assay_overlaps": self.pair_overlaps,
            # t89 — WHICH POSITIONS WERE SCORED. Two runs are comparable only if this matches; a
            # `full` run and a `regions` run are not two measurements of one quantity.
            "eval_scope": self.scope(),
        }


def open_source(*, h5: Optional[Path | str] = None, store: Optional[Path | str] = None,
                **kw) -> EvalSource:
    """Exactly one of a baked h5 or a store regime file. `bench`'s whole data entry point.

    This is deliberately NOT `train.py::DataSource.resolve`. The training resolver carries a
    reference table, a masking regime, cCRE loci and a run record; none of that is an evaluation
    concern, and importing it would put the training loop's dependencies behind every `bench`
    import. The rule the two share is the one that matters: neither has a default, and asking for
    both is an error rather than a precedence question.
    """
    if (h5 is None) == (store is None):
        raise ValueError(
            "exactly one of h5= / store= is required. h5=<baked.h5> reads a frozen bake; "
            f"store=<regime.json> reads a CANDI_STORE through a regime file. Got h5={h5!r} "
            f"store={store!r}."
        )
    if h5 is not None:
        if kw.get("eval_regions") is not None:
            raise ValueError(
                "eval_regions= is a store-only scope. A bake's eval windows are frozen rows of "
                "`counts_dsf1` and cannot be re-tiled, so restricting them would drop whole baked "
                "rows rather than cut the genome. Score an h5 at full coverage, or use a store.")
        return H5Source(h5, **{k: v for k, v in kw.items() if k != "eval_regions"})
    return StoreSource(store, **{k: v for k, v in kw.items()
                                 if k in ("chroms", "biosamples", "eval_regions")})


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

def _apply_loo_mask(batch: Dict[str, Any], assay: int) -> None:
    """Hold one assay out of the encoder input, exactly as `DataMasker._mask_full_assay` does.

    Data, metadata rows 0-3 and availability all take `CLOZE`. Row 4 (cell identity, when the model
    carries one) is deliberately left alone — the masker leaves it too, and which cell this is stays
    known however many assays are hidden.
    """
    batch["x_data"][:, :, assay] = float(CLOZE)
    batch["x_meta"][:, :4, assay] = float(CLOZE)
    batch["x_avail"][:, assay] = float(CLOZE)


def _decode_full(model, prep: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    return model(prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
                 prep.get("log_ref"))


def _is_loo_imputation(source: EvalSource, kind: str) -> bool:
    """Is this a store imputation — the one case that holds the target out BY MASK?

    Every store imputation, declared pairing or not (t80). Under a declared pair the target column
    is already `MISSING` in the `T_` prompt, so the mask removes nothing and is a tripwire; the
    condition stays unconditional so the task definition does not depend on the corpus happening to
    have disjoint splits. `decode_groups` reads the same answer for the same reason.
    """
    return kind == "impute" and source.kind == "store"


def decode_groups(source: EvalSource, kind: str, cols: Sequence[int]) -> List[List[int]]:
    """Which target assays can share one forward pass.

    On the h5, and for denoising anywhere, all of them: the imputation target is a `V_`/`B_` assay
    the input cell genuinely does not have, so nothing has to be hidden and one pass answers for
    every target at once.

    On the store, one assay per pass, and that is not an optimisation left on the table. Store
    imputation holds the assay out **by mask**, so masking every target at once would not be
    leave-one-out — it would be "impute all of them from whatever is left", a strictly harder task
    that is not the one the model trained on. On `V_aa`, which carries two assays, it would also
    empty the encoder input entirely.

    That stays true under a declared pairing (t80) even though the target column is then already
    absent from the `T_` prompt and masking it removes nothing. Grouping there would be free ONLY
    while the splits stay disjoint on `(cell, assay)`; the moment they are not, one grouped pass
    would hide several observed tracks at once and quietly change the exam. One assay per pass
    costs decode time and buys a task definition that does not depend on a corpus property.
    """
    return [[a] for a in cols] if _is_loo_imputation(source, kind) else [list(cols)]


@torch.no_grad()
def stream_tracks(model, source: EvalSource, device, *, kind: str = "impute",
                  batch_windows: int = 4, pairs: Optional[Sequence[Pair]] = None,
                  progress: bool = False) -> Iterator[TrackRecord]:
    """One `TrackRecord` per `(pair, assay)`, assembled over every bin of every eval chromosome.

    Pair-outer, window-inner. A pair's buffers are allocated when it opens and dropped once its
    records have been yielded, so peak memory is one pair's targets rather than the panel's.
    """
    model.eval()
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    plan = [(c, source.windows(c)) for c in source.eval_chroms]
    todo = list(pairs) if pairs is not None else source.pairs(kind)
    fields = ("mu", "n", "counts", "signal_mu", "signal_sigma", "pval", "peak_score", "peaks")

    for pi, pair in enumerate(todo):
        groups = [g for g in decode_groups(source, kind, source.targets(pair, kind)) if g]
        if not groups:
            continue
        cols = [a for g in groups for a in g]
        recs = {a: TrackRecord(pair=pair, assay=source.assays[a], kind=kind,
                               chroms=tuple(c for c, _ in plan)) for a in cols}
        have_signal = None
        have_peak = None
        dropped: List[int] = []
        for chrom, starts in plan:
            n = source.n_bins(chrom)
            buf = {a: {k: np.zeros(n, dtype=np.float32) for k in fields} for a in cols}
            for lo in range(0, len(starts), batch_windows):
                chunk = starts[lo:lo + batch_windows]
                for group in groups:
                    batch = source.batch(pair, chrom, chunk, kind)
                    if _is_loo_imputation(source, kind):
                        for a in group:
                            _apply_loo_mask(batch, a)
                    prep = prepare_masked_batch(batch, noop, device, apply_mask=False)
                    if prep is None:
                        # Only reachable when holding the group out empties the encoder input —
                        # a biosample with a single assay has no leave-one-out. Recorded and
                        # dropped, never scored against an all-MISSING buffer.
                        dropped.extend(a for a in group if a not in dropped)
                        continue
                    if cross_cell(pair, kind):
                        prep["y_meta"] = vb_natural_meta(prep["y_meta"],
                                                         batch["y_meta_imp"].to(device),
                                                         batch["y_avail"].to(device))
                        truth_counts = batch["y_data_imp"]
                        truth_pval = batch["y_pval_imp"]
                        truth_peaks = batch["y_peaks_imp"]
                    else:
                        truth_counts = batch["y_data"]
                        truth_pval = batch["y_pval"]
                        truth_peaks = batch["y_peaks"]

                    out = _decode_full(model, prep)
                    if have_signal is None:
                        have_signal = "signal_mu" in out
                    if have_peak is None:
                        have_peak = "peak_logit" in out
                    for j, s in enumerate(chunk):
                        sl = slice(int(s), int(s) + source.context_bins)
                        for a in group:
                            b = buf[a]
                            b["mu"][sl] = out["mu"][j, :, a].float().cpu().numpy()
                            b["n"][sl] = out["n"][j, :, a].float().cpu().numpy()
                            b["counts"][sl] = truth_counts[j, :, a].float().cpu().numpy()
                            b["pval"][sl] = truth_pval[j, :, a].float().cpu().numpy()
                            b["peaks"][sl] = truth_peaks[j, :, a].float().cpu().numpy()
                            if have_signal:
                                b["signal_mu"][sl] = out["signal_mu"][j, :, a].float().cpu().numpy()
                                b["signal_sigma"][sl] = (
                                    out["signal_var"][j, :, a].clamp_min(1e-12).sqrt()
                                    .float().cpu().numpy())
                            if "peak_logit" in out:
                                b["peak_score"][sl] = (
                                    torch.sigmoid(out["peak_logit"][j, :, a]).float().cpu().numpy())
                            else:
                                b["peak_score"][sl] = b["mu"][sl]
            # t89 — COMPACT to the scope, here rather than at scoring time. The buffers are
            # chromosome-length and only the planned windows were written into them, so under a
            # restricted plan every unwritten bin is still the zero it was allocated as, and a
            # scorer handed the whole vector would score a genome of mostly zeros. `None` keeps the
            # untouched path bit for bit: the arrays are the same objects they always were.
            keep = source.scored_bins(chrom)
            for a in cols:
                r, b = recs[a], buf[a]
                r.has_peak_head = bool(have_peak)
                if keep is not None:
                    b = {k: v[keep] for k, v in b.items()}
                    r.bin_scope = "regions"
                r.mu[chrom], r.n[chrom], r.counts[chrom] = b["mu"], b["n"], b["counts"]
                r.peaks[chrom], r.peak_score[chrom] = b["peaks"], b["peak_score"]
                if have_signal:
                    r.signal_mu[chrom] = b["signal_mu"]
                    r.signal_sigma[chrom] = b["signal_sigma"]
                    r.pval[chrom] = b["pval"]
            del buf
        if dropped:
            print(f"[bench] {pair}: {[source.assays[a] for a in dropped]} have no leave-one-out "
                  f"(holding them out empties the encoder input) — NOT scored", flush=True)
        if progress:
            print(f"[bench] {pi + 1}/{len(todo)} {pair} — "
                  f"{len(cols) - len(dropped)} {kind} track(s)", flush=True)
        for a in cols:
            if a not in dropped:
                yield recs[a]
        del recs


# ---------------------------------------------------------------------------
# per-track scoring — D4: the per-track score is the primitive
# ---------------------------------------------------------------------------

def _chrom_offsets(rec: TrackRecord, chroms: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Offsets into the concatenation of `chroms` — which is the SCOPE's chromosomes, not the
    record's. A held-out concatenation and a genome-wide one place the same chromosome at different
    positions, so an offset table built for one is wrong for the other."""
    off, acc = {}, 0
    for c in (rec.chroms if chroms is None else chroms):
        off[c] = acc
        acc += len(rec.counts[c])
    return off


def _p_block(rec: TrackRecord, gene_annotations: Sequence[str],
             chroms: Optional[Sequence[str]] = None,
             signal_mu: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, object]:
    """The P-block for one track, over the concatenation of every eval chromosome.

    Runs on the **pval arm only**, and that is the paper's own scoping rather than a shortcut: the
    binarisation is `Y >= 2` on a -log10 p-value and the strength bins run from 1e-1 to 10^2.5 in
    the same units. A count arm has no p-value, and inventing a count threshold to fill the slot
    would produce a number with no counterpart in the published work.

    THAT SCOPING IS ALSO WHY `signal_mu` IS AN ARGUMENT. The thresholds above are absolute numbers
    in `-log10 p`, so the prediction has to arrive in `-log10 p` too — `score_track` passes the
    INVERTED means (`D.invert_signal_prediction`) rather than `rec.signal_mu`, which on a store is
    in the head's training space. Defaults to `rec.signal_mu`, which is the same array whenever the
    run's transform is `"none"`.

    `accuracy_by_strength` is position-free, so it takes the whole concatenation at once and its
    strength bins are the panel's, not one chromosome's. The two region correlations ARE positional,
    so their windows are built per chromosome and shifted into the concatenation — the same
    construction `eic.dict_to_arr` relies on, for the same reason.
    """
    chroms = tuple(rec.chroms if chroms is None else chroms)
    sig = E.dict_to_arr(rec.pval, chroms)
    pred = E.dict_to_arr(rec.signal_mu if signal_mu is None else signal_mu, chroms)
    pk = E.dict_to_arr(rec.peaks, chroms).astype(bool)
    out: Dict[str, object] = {
        "acc_by_obs_strength": P.accuracy_by_strength(sig, pk, pred, bin_by="obs"),
        "acc_by_imp_strength": P.accuracy_by_strength(sig, pk, pred, bin_by="imp"),
    }
    off = _chrom_offsets(rec, chroms)
    if rec.assay == "H3K4me3":
        wins = [(lo + off[c], hi + off[c]) for c in chroms
                for lo, hi in P.promoter_windows(gene_annotations, c, len(rec.pval[c]))]
        out["prom_corr_h3k4me3"] = P.region_correlation(sig, pred, wins)
    if rec.assay in ("DNase-seq", "DNase"):
        wins = [(lo + off[c], hi + off[c]) for c in chroms
                for lo, hi in P.peak_regions(rec.peaks[c].astype(bool))]
        out["peak_shape_corr_dnase"] = P.region_correlation(sig, pred, wins)
    return out


def loss_block(rec: TrackRecord, *, signal_target_transform: str = "none",
               chroms: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The LOSS tier for one track: the three NLLs the training objective is made of.

    **This is the val/test loss, and it is not a benchmark comparison.** Every other block in this
    file asks how good a prediction is against a truth in the benchmark's own units. These three ask
    the one question the training loop asks — what likelihood did the model assign — so they are the
    only numbers here that are directly comparable to `train/nll` in a run's step log. `EVAL.md`
    lists them under the loss tier for that reason and not among the measures.

    **ARM-INDEPENDENT ON PURPOSE, so the same dict is spread into every arm.** The `count` / `pval`
    split is a split between two *comparisons* — NB predictions against raw counts, Gaussian
    predictions against p-values. A loss is not a comparison between arms; it is one scalar the
    objective summed. Putting `gaussian_nll` only under `pval` would then hide it from every reader
    of the count arm, including `monitor`, which scores that arm alone.

    **Head gating, from the model rather than from the numbers.** `nb_nll` needs a count prediction
    (`rec.has_count`), `gaussian_nll` needs a signal head predicting a SPREAD (`rec.has_pval` and
    `rec.has_sigma`) and `bernoulli_nll` needs a peak head (`rec.has_peak_head`, recorded in
    `stream_tracks` where the output dict is the authority). An absent head means an ABSENT KEY, not
    a nan: a nan travels into `macro_mean`, gets skipped by a finiteness filter, and leaves a reader
    unable to tell "no head" from "the head produced garbage". Every record `stream_tracks` builds
    satisfies the first two, so this gating is inert on the model path and only `bench.external`
    ever trips it.

    `signal_target_transform` is D30 and it is RECORDED beside the numbers, never inferred. A
    Gaussian NLL against `arcsinh(-log10 p)` and one against raw `-log10 p` are both finite and
    plausible and are not the same number, and the checkpoint carries no trace of which one it was
    trained for — the run json's `config.signal_target_transform` does.

    **THIS TIER IS THE ONE DELIBERATE EXCEPTION TO THE `-log10 p` RULE.** Every benchmark measure on
    the pval arm is quoted in `-log10 p`: `score_track` inverts the PREDICTION back into that space
    first (`D.invert_signal_prediction`). `gaussian_nll` does the opposite on purpose — it mirrors
    the training loss, so it takes the head's own `(µ, σ)` untouched and bends the TRUTH forward
    with `D.transform_signal_target`. It therefore lives in TRANSFORMED space whenever the run does,
    which is why the space travels with it as `signal_target_transform` and why this number must
    never be compared to one scored under a different transform. `EVAL.md` §spaces is the contract.
    """
    chroms = tuple(rec.chroms if chroms is None else chroms)
    out: Dict[str, object] = {}
    if rec.has_count:
        out["nb_nll"] = D.nb_nll(E.dict_to_arr(rec.n, chroms), E.dict_to_arr(rec.mu, chroms),
                                 E.dict_to_arr(rec.counts, chroms))
    out["signal_target_transform"] = str(signal_target_transform)
    if rec.has_pval and rec.has_sigma:
        # `stream_tracks` stores sigma; `gaussian_nll` takes the variance the loss was written in.
        sigma = E.dict_to_arr(rec.signal_sigma, chroms).astype(np.float64)
        out["gaussian_nll"] = D.gaussian_nll(
            E.dict_to_arr(rec.signal_mu, chroms), sigma * sigma,
            D.transform_signal_target(E.dict_to_arr(rec.pval, chroms), signal_target_transform))
    if rec.has_peak_head:
        out["bernoulli_nll"] = D.bernoulli_nll(E.dict_to_arr(rec.peak_score, chroms),
                                               E.dict_to_arr(rec.peaks, chroms))
    return out


def score_track(rec: TrackRecord, *, gene_annotations: Sequence[str],
                enh_annotations: Sequence[str],
                var: Optional[Mapping[str, np.ndarray]] = None,
                seed: int = 0, c_index_pairs: int = 200_000,
                signal_target_transform: str = "none",
                crps_approx: Optional[int] = None, crps_seed: int = 0,
                with_curve: bool = False,
                chroms: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, object]]:
    """Every block this track can carry, per arm — and only the arms whose PREDICTION exists.

    On the model path that reads "`count` always, `pval` when the signal head is on": `mu`/`n` are
    on every checkpoint. An external track (`bench.external`) may carry either arm alone, so each is
    gated on the record — `has_count`, `has_pval`, `has_sigma` — and an arm with no prediction is
    absent rather than scored against a buffer of zeros.

    `var` is the D7 variance pool as `{chrom: vector}`, aligned bin-for-bin with the track. It is
    only ever applied to the arm it was built in — the pool that matches the 267 EIC training
    experiments is a variance of -log10 p-values, and weighting squared count error by it would be
    a number with no interpretation. Omitting it omits `msevar`, which is the honest outcome; the
    organizers' code returns a bare `0.0` there and `annotations.load_variance_pool` says why we do
    not.

    `chroms` restricts the scoring to a SCOPE (§4). It defaults to every chromosome the record
    carries, which is the whole of the old behaviour. Passing a subset is how one prediction pass
    yields two aggregations: the measures are recomputed from the same arrays over fewer positions,
    never rescaled from a finished number, because none of them is linear in position. The scope
    reaches `loss_block` too, so a scoped arm's `n_bins` and its NLLs count the same positions.

    **A BIN SCOPE is the finer cut of the same idea, and it arrives on the RECORD.** t89's eval
    scope (`EvalSource.eval_regions`) compacts each chromosome's arrays down to the bins that were
    actually predicted, and `rec.bin_scope` says so. Every measure that is a function of the two
    vectors alone survives that untouched; the ones that look a genomic coordinate up in the array —
    `mseprom`, `msegene`, `mseenh` and the whole P-block — are ABSENT, because after a gather index
    `i` is no longer the bin at `i * 25` bp and an annotation interval would land on the wrong
    sequence. Absent rather than NaN so nothing can average them into a macro by mistake.

    **THE SPACE CONTRACT, AND WHERE `signal_target_transform` LANDS.** Storage is transformed (the
    store codec holds `arcsinh(-log10 p)`) and the reader inverts it, so the truth reaching this
    function is raw `-log10 p` on BOTH data paths. Training supervises in transformed space, so on a
    store the head's `(signal_mu, signal_sigma)` is a Gaussian over `arcsinh(-log10 p)`. Every
    BENCHMARK number on the pval arm is quoted in `-log10 p` (`D.SIGNAL_EVAL_SPACE`), so the
    prediction is bent back with `D.invert_signal_prediction` before the E-block, `gauss_suite` and
    `_p_block` see it, and the arm records `pred_space` to say so. `binary_suite` is the one block
    that needs nothing: it ranks by `peak_score`, and no signal-space value reaches it.

    **THE LOSS TIER IS THE DELIBERATE EXCEPTION.** `loss_block` gets the UNINVERTED prediction: its
    `gaussian_nll` mirrors the training objective, so it stays in the training space and bends the
    TRUTH forward instead, with `signal_target_transform` recorded beside the number. That is why
    the same `signal_target_transform` argument is used twice here in opposite directions — a
    likelihood is not a comparison, and `EVAL.md` §spaces states the rule both obey.

    `signal_target_transform` defaults to `"none"` — the identity, and the h5 path's real value — so
    every caller written before this existed keeps the arithmetic it had, bit for bit. A caller that
    knows better passes it: `monitor` from its own run config, `run_bench` from the CLI's
    `--arch-from`.

    `crps_approx` reaches the COUNT arm only, and only its CRPS family (`D.nb_suite`). It defaults
    to `None` = the closed form, so the same "bit for bit" promise covers it; a caller that asks for
    it must record `K` and `crps_seed` beside the numbers, because a sampled score is not
    reproducible from the prediction alone.
    """
    arms: Dict[str, Dict[str, object]] = {}
    chroms = tuple(rec.chroms if chroms is None else chroms)
    unknown = [c for c in chroms if c not in rec.chroms]
    if unknown:
        raise ValueError(f"{rec.key}: scope names {unknown}, which the record does not carry "
                         f"({list(rec.chroms)}). A scope is a subset of what was predicted, and a "
                         f"missing chromosome is a window-plan bug rather than something to skip.")
    loss = loss_block(rec, signal_target_transform=signal_target_transform, chroms=chroms)
    # t89 — a COMPACTED record's index is no longer a genomic position, so every measure that looks
    # one up is absent instead of wrong. It is read off the record rather than taken as an argument
    # because a gathered array is an ordinary float32 vector and nothing else can tell them apart.
    positional = rec.bin_scope is None

    if rec.has_count:
        y_c = E.dict_to_arr(rec.counts, chroms)
        arms["count"] = {
            **loss,
            **E.score_track(rec.counts, rec.mu, chroms, gene_annotations=gene_annotations,
                            enh_annotations=enh_annotations, positional=positional),
            **D.nb_suite(E.dict_to_arr(rec.n, chroms), E.dict_to_arr(rec.mu, chroms), y_c,
                         seed=seed, n_pairs=c_index_pairs,
                         crps_approx=crps_approx, crps_seed=crps_seed),
            **B.binary_suite(E.dict_to_arr(rec.peaks, chroms).astype(bool),
                             E.dict_to_arr(rec.peak_score, chroms), y_c, with_curve=with_curve),
        }
    if rec.has_pval:
        y_p = E.dict_to_arr(rec.pval, chroms)
        pool = None if var is None else E.dict_to_arr(var, chroms)
        # The prediction moves, the truth does not. `invert_signal_prediction` returns the caller's
        # own arrays under `"none"`, so the h5 path builds dicts of the same objects and every key
        # below is bit-identical to what it was before the inversion existed. A point-only external
        # track has no spread to invert, so it hands over zeros and keeps only the mean.
        inv = {c: D.invert_signal_prediction(
            rec.signal_mu[c],
            rec.signal_sigma[c] if rec.has_sigma else np.zeros_like(rec.signal_mu[c]),
            signal_target_transform) for c in chroms}
        mu_p = {c: v[0] for c, v in inv.items()}
        sigma_p = {c: v[1] for c, v in inv.items()}
        arms["pval"] = {
            **loss,
            **E.score_track(rec.pval, mu_p, chroms, gene_annotations=gene_annotations,
                            enh_annotations=enh_annotations, var=pool, positional=positional),
            # ABSENT, NOT NAN, when the track predicts a point and nothing else. Every key here is a
            # property of a FORECAST DISTRIBUTION — CRPS, PIT, 95% coverage, the C-index — and a
            # point track has none, so scoring it against a spread of zero would answer a question
            # it never asked. `RIVALS_PLAN.md` §4.2 is the rule; a σ-table is how a point-only rival
            # earns these keys back.
            **(D.gauss_suite(E.dict_to_arr(mu_p, chroms), E.dict_to_arr(sigma_p, chroms), y_p,
                             seed=seed, n_pairs=c_index_pairs) if rec.has_sigma else {}),
            # NOTHING TO INVERT HERE, and it is worth saying why rather than leaving a reader to
            # check: `binary_suite`'s ranking score is `peak_score` — the peak head's probability,
            # or the NB mean without that head — never `signal_mu`, so no signal-space value
            # reaches it. Its keys are rank-based besides, and every transform in the vocabulary is
            # strictly increasing, so they would be unchanged even if one did.
            **B.binary_suite(E.dict_to_arr(rec.peaks, chroms).astype(bool),
                             E.dict_to_arr(rec.peak_score, chroms), y_p, with_curve=with_curve),
            # POSITIONAL, and therefore absent under a scope: the two region correlations build
            # their windows from gene coordinates, and `accuracy_by_strength` rides along in the
            # same block. See `positional` above.
            **(_p_block(rec, gene_annotations, chroms, signal_mu=mu_p) if positional else {}),
            # WHICH SPACE THESE NUMBERS ARE IN, on the row itself. `pred_space` is the answer and
            # `pred_inversion` is how it was reached — `"none"` means the head already predicted
            # `-log10 p`, not that the question was skipped. A row from before the contract carries
            # neither key, which is how a reader tells the two apart.
            "pred_space": D.SIGNAL_EVAL_SPACE,
            "pred_inversion": str(signal_target_transform),
        }
    for arm in arms:
        arms[arm]["assay"] = rec.assay
        arms[arm]["kind"] = rec.kind
        arms[arm]["n_bins"] = rec.n_bins(chroms)
        arms[arm]["chroms"] = list(chroms)
        # On the row, not only in provenance: `n_bins` alone cannot tell a scoped track from a
        # short chromosome, and a reader diffing two per-track tables needs to know which.
        arms[arm]["bin_scope"] = rec.bin_scope
    return arms


def macro_mean(per_track: Mapping[str, Mapping[str, Mapping[str, object]]], arm: str,
               kind: str = "impute") -> Dict[str, float]:
    """The headline (D4): an unweighted mean over TRACKS of every scalar key the arm carries.

    Unweighted is the point. A weighted mean would let the deepest track and the longest chromosome
    decide the panel's number, which is the same background-domination failure the P-block exists to
    defeat, one level up.
    """
    rows = [t[arm] for t in per_track.values() if arm in t and t[arm].get("kind") == kind]
    if not rows:
        return {}
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, bool))})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))]
        if vals:
            out[k] = float(np.mean(vals))
            out[f"{k}_n_tracks"] = len(vals)
    out["n_tracks"] = len(rows)
    return out


# ---------------------------------------------------------------------------
# the three panel numbers — plan/BENCHMARK_DESIGN.md 5.2
# ---------------------------------------------------------------------------

def panel_of(target_biosample: str) -> Optional[str]:
    """Which panel a scored experiment belongs to, from its TARGET cell's prefix.

    The target, never the input: the prompt is always a `T_` cell, so pairing on the input would
    put every experiment in one panel. Anything that is neither `V_` nor `B_` returns None and is
    counted in no panel rather than silently folded into one — a self-paired denoise record is the
    ordinary case for that.
    """
    if target_biosample.startswith("V_"):
        return "V"
    if target_biosample.startswith("B_"):
        return "B"
    return None


def panel_macros(per_track: Mapping[str, Mapping[str, Mapping[str, object]]], arm: str,
                 kind: str = "impute") -> Dict[str, Dict[str, object]]:
    """`V_` breadth, `V_` matched and `B_` — the three numbers of 5.2, from one scored pass.

    `V_` and `B_` are different exams: `V_` poses 22 assays and `B_` poses 8. Ranking WITHIN a panel
    is unaffected by that, but the `V_`->`B_` delta a reader computes by eye is not, and it reads as
    a generalization gap when most of it is the exam changing. The middle number fixes that and
    costs nothing: it is the same scored tracks, aggregated over the subset of assays `B_` contains.

    The matched panel's assay set is **measured from `B_`**, never listed here. A hard-coded set
    would go stale the first time the panel moves and would be wrong silently.

    `V_matched` is marked `ranked: False`. It is a reading aid, and ranking it would invent a fourth
    placement nobody asked for and that has no counterpart on the board.
    """
    def rows(pred) -> Dict[str, Mapping[str, object]]:
        out = {}
        for key, arms in per_track.items():
            a = arms.get(arm)
            if a is None or a.get("kind") != kind:
                continue
            fields = key.split("|")
            if len(fields) < 3:
                continue
            if pred(fields[1], str(a.get("assay", fields[2]))):
                out[key] = arms
        return out

    b_rows = rows(lambda tgt, assay: panel_of(tgt) == "B")
    matched_assays = sorted({str(per_track[k][arm]["assay"]) for k in b_rows})

    out: Dict[str, Dict[str, object]] = {}
    for name, pred, ranked in (
        ("V_breadth", lambda tgt, assay: panel_of(tgt) == "V", True),
        ("V_matched", lambda tgt, assay: panel_of(tgt) == "V" and assay in matched_assays, False),
        ("B", lambda tgt, assay: panel_of(tgt) == "B", True),
    ):
        sel = rows(pred)
        macro = macro_mean(sel, arm, kind=kind)
        out[name] = {
            **macro,
            "n_experiments": len(sel),
            "assays": sorted({str(sel[k][arm]["assay"]) for k in sel}),
            "ranked": ranked,
        }
    out["V_matched"]["matched_to"] = matched_assays
    out["V_matched"]["note"] = (
        "NOT RANKED. It exists only so the V_->B_ delta is readable: V_ breadth -> V_ matched is "
        "the exam changing, V_ matched -> B_ is the generalization gap. Never subtract V_ breadth "
        "from B_ (5.3's reading rule).")
    return out


# ---------------------------------------------------------------------------
# the one panel-level measure (§4.2)
# ---------------------------------------------------------------------------

def panel_specificity(binarised: Mapping[str, Sequence[Tuple[str, np.ndarray, np.ndarray, int]]]
                      ) -> Dict[str, Dict[str, object]]:
    """`pr_by_specificity`, per assay, over every cell type in the panel.

    It cannot be a per-track measure: the specificity score of a locus is the column sum of the
    binarised cell-type x locus matrix FOR THAT ASSAY, so it is undefined until every cell has been
    scored. `binarised` is `{assay: [(track_key, truth_bits, call_bits, n_bins), ...]}` with both
    sides `np.packbits`-ed — one bit per bin instead of eight, which is what makes keeping the panel
    alive across the whole stream affordable while every other buffer is released per pair.
    """
    out: Dict[str, Dict[str, object]] = {}
    for assay, rows in sorted(binarised.items()):
        if len(rows) < 2:
            # A one-cell panel gives every locus specificity 0 or 1, and the measure — which exists
            # to separate a model from the average-activity baseline — has nothing to separate.
            out[assay] = {
                "n_cell_types": len(rows),
                "tracks": [r[0] for r in rows],
                "note": "fewer than two cell types carry this assay; cell-type specificity is not "
                        "defined over a panel of one",
            }
            continue
        n = int(rows[0][3])
        truth = np.stack([np.unpackbits(t)[:n].astype(bool) for _k, t, _c, _n in rows])
        spec = P.specificity_scores(truth)
        per: Dict[str, object] = {"n_cell_types": len(rows), "tracks": [r[0] for r in rows]}
        macro_p: List[float] = []
        macro_r: List[float] = []
        for i, (key, _t, call_bits, _n) in enumerate(rows):
            rec = P.precision_recall_by_specificity(
                spec, truth[i], np.unpackbits(call_bits)[:n].astype(bool))
            per[key] = rec
            for acc, k in ((macro_p, "macro_precision"), (macro_r, "macro_recall")):
                if np.isfinite(rec[k]):
                    acc.append(float(rec[k]))
        per["macro_precision"] = float(np.mean(macro_p)) if macro_p else float("nan")
        per["macro_recall"] = float(np.mean(macro_r)) if macro_r else float("nan")
        out[assay] = per
    return out


# ---------------------------------------------------------------------------
# C-block — covariate sensitivity, on cached latents
# ---------------------------------------------------------------------------

def decode_latent(model, z: torch.Tensor, y_meta: torch.Tensor,
                  log_ref: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Decode a cached encoder latent under a (possibly counterfactual) prompt.

    Lifted from `eval.py::decode_latent`, and for its reason: on a split-conditioning arm the
    y_meta attention blocks sit on the DECODER side of the boundary, so calling `model.decoder`
    directly would silently skip them — the run would score, the numbers would look plausible, and
    the arm would be measured as a different arm with extra untrained weights.
    """
    fn = getattr(model, "decode_latent", None)
    return fn(z, y_meta, log_ref) if fn is not None else model.decoder(z, y_meta, log_ref)


@dataclass
class _Context:
    """One encoded window batch, plus the `(row, assay)` slots the C-block treats as units."""
    z: torch.Tensor
    y_meta: torch.Tensor
    log_ref: Optional[torch.Tensor]
    slots: List[Tuple[int, int]]
    truth: np.ndarray                      # (len(slots), L)
    pair: Pair
    chrom: str
    starts: List[int]


def _spread(starts: Sequence[int], k: int) -> List[int]:
    """`k` window starts spaced across the whole chromosome — never a prefix.

    The deleted `eval.py::build_eval_units` recorded why in blood: the eval windows walk in genomic
    order, so the first k of them are a contiguous block at the START of the eval chromosome. On
    chr21 that block is the acrocentric p-arm, where every position of every held-out target is
    exactly zero.
    """
    if k >= len(starts):
        return list(starts)
    return [starts[int((i + 0.5) * len(starts) / k)] for i in range(k)]


@torch.no_grad()
def _c_contexts(model, source: EvalSource, device, *, kind: str, pairs: Sequence[Pair],
                n_windows: int) -> List[_Context]:
    """Encode once per counterfactual context; the prompt is what varies afterwards.

    The units are IMPUTATION slots wherever the backend can pose one, and that is a methodological
    requirement rather than a convenience. If the target assay is present in the encoder input, the
    model can read its depth off the signal and ignore the prompt entirely, and every instrument in
    the C-block would then measure a channel that is not the one under test. On the h5, and on a
    store with declared pairs, the target is an assay the input cell genuinely lacks, so one encode
    per pair answers for all of them; on a pair-less store it is held out by mask, one assay per
    context, hence one encode per held-out assay. `decode_groups` is where that choice is made, and
    asking it rather than re-deciding here is what keeps the two loops from drifting apart.
    """
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    model.eval()
    chrom = source.eval_chroms[0]
    picks = _spread(source.windows(chrom), n_windows)
    out: List[_Context] = []
    for pair in pairs:
        cols = source.targets(pair, kind)
        if not cols:
            continue
        groups = decode_groups(source, kind, cols)
        for group in groups:
            batch = source.batch(pair, chrom, picks, kind)
            if _is_loo_imputation(source, kind):
                for a in group:
                    _apply_loo_mask(batch, a)
            prep = prepare_masked_batch(batch, noop, device, apply_mask=False)
            if prep is None:                          # pragma: no cover - guarded by targets()
                continue
            y_meta = prep["y_meta"]
            truth_t = batch["y_data"]
            if cross_cell(pair, kind):
                y_meta = vb_natural_meta(y_meta, batch["y_meta_imp"].to(device),
                                         batch["y_avail"].to(device))
                truth_t = batch["y_data_imp"]
            z = model.encode(prep["x_data"], prep["x_dna"], prep["x_meta"])
            slots = [(j, a) for a in group for j in range(z.shape[0])]
            keep = [s for s in slots
                    if float(y_meta[s[0], DEPTH_ROW, s[1]]) not in (MISSING, CLOZE)]
            if not keep:
                continue
            truth = np.stack([truth_t[j, :, a].float().cpu().numpy() for j, a in keep])
            out.append(_Context(z=z, y_meta=y_meta.clone(), log_ref=prep.get("log_ref"),
                                slots=keep, truth=truth, pair=pair, chrom=chrom,
                                starts=list(picks)))
    return out


def _predictor(model, contexts: Sequence[_Context]) -> Tuple[C.Predictor, np.ndarray, np.ndarray]:
    """`(predict, cov, target)` — the three arguments every C-block instrument takes.

    `cov` is `(n_units, 4)` in `COVARIATES` order, read straight off the prompt rows. `predict`
    writes a candidate covariate matrix back into those rows, re-decodes the cached latents and
    gathers `mu` and `n` at the same slots. The encoder never re-runs, which is what makes `covuse`'s
    ~1,600 decodes affordable — and it is exact, because the latent does not depend on `y_meta`.

    **A candidate may have any number of rows.** `covuse`, `depthdir` and `covspec` pass
    one row per unit, but `covshare` passes an inner block of `n_inner` rows regardless of how many
    units exist. So a row is a free-standing covariate vector, not a unit: row `r` is evaluated in
    unit `r % n_units`, and a candidate longer than the unit list is processed in that many passes.
    Anything that assumed `len(candidate) == n_units` would index out of the batch on the first
    Shapley subset.
    """
    spans: List[Tuple[_Context, int]] = []
    cov_rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    pos = 0
    for ctx in contexts:
        spans.append((ctx, pos))
        pos += len(ctx.slots)
        for j, a in ctx.slots:
            cov_rows.append(ctx.y_meta[j, :4, a].float().cpu().numpy())
        targets.append(ctx.truth)
    n_units = pos
    cov = np.asarray(cov_rows, dtype=np.float64)
    target = np.concatenate(targets, axis=0)
    n_pos = target.shape[1]

    @torch.no_grad()
    def predict(candidate: np.ndarray) -> Dict[str, np.ndarray]:
        cand = np.asarray(candidate, dtype=np.float64)
        m = len(cand)
        mu = np.zeros((m, n_pos), dtype=np.float32)
        nn = np.zeros((m, n_pos), dtype=np.float32)
        for base in range(0, m, n_units):
            for ctx, lo in spans:
                ym = ctx.y_meta.clone()
                active = []
                for u, (j, a) in enumerate(ctx.slots):
                    r = base + lo + u
                    if r >= m:
                        continue
                    ym[j, :4, a] = torch.as_tensor(cand[r], dtype=ym.dtype, device=ym.device)
                    active.append((r, j, a))
                if not active:
                    continue
                out = decode_latent(model, ctx.z, ym, ctx.log_ref)
                for r, j, a in active:
                    mu[r] = out["mu"][j, :, a].float().cpu().numpy()
                    nn[r] = out["n"][j, :, a].float().cpu().numpy()
        return {"mu": mu, "n": nn}

    return predict, cov, target


@torch.no_grad()
def _depthblind_latents(model, source: EvalSource, device, *, pairs: Sequence[Pair],
                n_windows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled latents for the same regions read at every DSF level.

    `batch` is the DSF level — the nuisance the encoder should be blind to; `bio_label` is the
    region — the signal it must keep. `biokeep` reads the second one, and D13 is why both leave
    here together or not at all.
    """
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    model.eval()
    chrom = source.eval_chroms[0]
    picks = _spread(source.windows(chrom), n_windows)
    Z, dsf_lab, region = [], [], []
    for dsf in source.dsf_levels:
        for pair in pairs:
            batch = source.batch(pair, chrom, picks, "denoise", x_dsf=int(dsf))
            prep = prepare_masked_batch(batch, noop, device, apply_mask=False)
            if prep is None:                          # pragma: no cover
                continue
            z = model.encode(prep["x_data"], prep["x_dna"], prep["x_meta"])
            Z.append(z.mean(dim=1).float().cpu().numpy())
            dsf_lab.extend([float(dsf)] * z.shape[0])
            region.extend(f"{pair.input_biosample}@{s}" for s in picks[:z.shape[0]])
    if not Z:                                          # pragma: no cover
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    return np.concatenate(Z, axis=0), np.asarray(dsf_lab), np.asarray(region)


@torch.no_grad()
def c_block(model, source: EvalSource, device, *, kind: str = "impute",
            pairs: Optional[Sequence[Pair]] = None, n_windows: int = 8,
            n_resamples: int = 50, n_outer: int = 32, n_inner: int = 8,
            seed: int = 0) -> Dict[str, object]:
    """All seven covariate instruments (D8), on one shared set of counterfactual contexts.

    **This block is sampled and says so.** D2's no-subsampling rule is about scoring a TRACK, and
    this block does not score a track — it perturbs a prompt and watches the output move. Every
    count that decides its resolution is in the result (`n_units`, `n_windows`, `n_resamples`,
    `n_outer`, `n_inner`), so a reader can see what it rests on instead of inferring it.
    """
    pair_list = list(pairs) if pairs is not None else source.pairs(kind)
    contexts = _c_contexts(model, source, device, kind=kind, pairs=pair_list,
                           n_windows=n_windows)
    if not contexts:
        return {"error": "no counterfactual context: no pair supplies an imputation slot with a "
                         "valid target prompt"}
    predict, cov, target = _predictor(model, contexts)
    out: Dict[str, object] = {
        "n_units": int(cov.shape[0]),
        "n_windows": int(n_windows),
        "n_contexts": len(contexts),
        "covariates": list(C.COVARIATES),
        "kind": kind,
        "covuse": C.covuse(predict, cov, target, n_resamples=n_resamples, seed=seed),
        "covshare": C.covshare(predict, cov, n_outer=n_outer, n_inner=n_inner, seed=seed),
        "depthdir": C.depthdir(predict, cov, depth_col=DEPTH_ROW),
        "covspec": C.covspec(predict, cov, seed=seed),
    }
    out["covuse_n_resamples"] = int(n_resamples)
    out["covshare_n_outer"], out["covshare_n_inner"] = int(n_outer), int(n_inner)

    levels = [float(d) for d in source.dsf_levels] or [1.0]
    if len(levels) > 1:
        pred_by_told = {}
        for d in levels:
            c = np.array(cov, copy=True)
            c[:, DEPTH_ROW] = cov[:, DEPTH_ROW] - math.log2(d)
            pred_by_told[d] = predict(c)
        true_by_level = {}
        for d in levels:
            parts = []
            for ctx in contexts:
                arr = source.counts_at_dsf(ctx.pair, ctx.chrom, ctx.starts, int(d))
                parts.append(np.stack([arr[j, :, a] for j, a in ctx.slots]))
            true_by_level[d] = np.concatenate(parts, axis=0)
        out["depthcounterfact"] = C.depthcounterfact(pred_by_told, true_by_level)
        out["depthcounterfact_calibration"] = {
            "constant_answer_frac_min_at_true": 1.0 / len(levels),
            "note": "0.25 on a four-level ladder is the value of ALWAYS answering told-depth 1, "
                    "not a chance baseline. There is NO ceiling constant here any more: the ~0.73 "
                    "this key used to carry was a consequence of re-selecting the foreground from "
                    "the level-k realization being scored, and `depthcounterfact` now draws ONE "
                    "foreground on the deepest truth and reuses it at every level — see its "
                    "`n_fg`, `n_positions` and `fg_level`. What a perfect model caps at under the "
                    "fixed mask is a measurement nobody has made yet, and carrying the old number "
                    "across the change would be quoting it against a run it does not describe.",
        }

        Z, dsf_lab, region = _depthblind_latents(model, source, device, pairs=pair_list,
                                         n_windows=n_windows)
        if Z.shape[0] >= 4 and len(np.unique(dsf_lab)) > 1:
            k = max(2, min(20, Z.shape[0] - 1))
            # ONE key holding both halves: D13 says `depthblind` is never reported without
            # `biokeep`, and a single dict is what makes splitting them a code change.
            out["depthblind_biokeep"] = C.depthblind(Z, dsf_lab, region, k=k, seed=seed)
            out["depthblind_n_latents"] = int(Z.shape[0])
    else:
        out["depthcounterfact"] = {
            "error": "the corpus has one DSF level, so there is no depth ladder to sweep"}
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _binarise(rec: TrackRecord, signal_target_transform: str = "none"
              ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """MACS2 truth and the `>= 2` p-value call, bit-packed, for `panel_specificity`.

    `BINARISE_THRESHOLD` is an absolute number IN `-log10 p`, so the prediction has to be in
    `-log10 p` before it is compared — the same spaces contract `score_track` obeys, in the one
    pval-arm measure that lives at the panel level instead of on a track. Comparing a training-space
    `µ` to 2 would call a different set of positions and `panel_specificity` would be a number about
    the store's codec.
    """
    if not rec.has_pval:
        return None
    truth = E.dict_to_arr(rec.peaks, rec.chroms).astype(bool)
    mu, _ = D.invert_signal_prediction(E.dict_to_arr(rec.signal_mu, rec.chroms),
                                       np.zeros(1), signal_target_transform)
    return np.packbits(truth), np.packbits(mu >= P.BINARISE_THRESHOLD), int(truth.size)


def _varpool(varpool_root: Optional[Path | str], corpus: str, assay: str,
             chroms: Sequence[str], n_bins: Mapping[str, int],
             cache: Dict[Tuple[str, str], object]) -> Optional[Dict[str, np.ndarray]]:
    if varpool_root is None:
        return None
    vecs, members = {}, None
    for c in chroms:
        hit = cache.get((assay, c), "miss")
        if hit == "miss":
            try:
                hit = ann.load_variance_pool(varpool_root, corpus, assay, c)
            except ann.AssetError:
                hit = None
            cache[(assay, c)] = hit
        if hit is None:
            return None
        if len(hit.values) != n_bins[c]:
            raise ValueError(
                f"variance pool {corpus}/{assay}/{c} is {len(hit.values)} bins and the scored "
                f"track is {n_bins[c]}; msevar would weight the wrong positions")
        vecs[c] = np.asarray(hit.values, dtype=np.float64)
        members = hit
    if members is not None:
        cache.setdefault(("__members__", assay), {
            "n_biosamples": members.n_biosamples, "space": members.space,
            "biosamples": list(members.biosamples)})
    return vecs


def run_bench(model, source: EvalSource, device, *, kinds: Sequence[str] = ("impute",),
              batch_windows: int = 4, seed: int = 0, c_index_pairs: int = 200_000,
              varpool_root: Optional[Path | str] = None, varpool_corpus: str = "eic",
              blocks: Sequence[str] = ("E", "P", "D", "B", "C"),
              c_windows: int = 8, c_resamples: int = 50,
              signal_target_transform: str = "none",
              crps_approx: Optional[int] = None, crps_seed: int = 0,
              with_curve: bool = False, progress: bool = False,
              held_out_chroms: Optional[Sequence[str]] = None,
              extra_provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Score one checkpoint end to end and return the result JSON of `EVAL_PLAN.md` §4.

    **`held_out_chroms` turns one pass into two aggregations** (`plan/BENCHMARK_DESIGN.md` §4).
    Given a proper subset of the scored chromosomes, `per_track`, `macro` and `panels` carry the
    HELD-OUT scope — the ranked number, where no method's transferable parameters were fit — and a
    parallel `genome_wide` block carries the same three over every chromosome, for comparability
    with a literature that scores that way.

    Left `None`, or naming every scored chromosome, there is exactly one scope and no `genome_wide`
    block is produced. That is not a silent omission: it is §4's blanking rule, and the reason is
    written into `provenance.scope` rather than left for a reader to infer from an absent key. For
    a method whose parameters were fit at every position the genome-wide number is a memorisation
    score, so it is **not computed at all** — the run is given three chromosomes and the block
    never exists.

    `signal_target_transform` (D30) is recorded in `provenance` because no checkpoint carries which
    space it was trained in. It is used twice, in opposite directions: `loss_block` bends the TRUTH
    forward into that space (`gaussian_nll` is the training loss), and the pval arm bends the
    PREDICTION back out of it, so every benchmark key on that arm is in `-log10 p`. See
    `score_track` and `EVAL.md` §spaces.
    """
    gene = ann.gene_annotations()
    enh = ann.enhancer_annotations()
    n_bins = {c: source.n_bins(c) for c in source.eval_chroms}
    var_cache: Dict[Tuple[str, str], object] = {}

    scored = tuple(source.eval_chroms)
    if held_out_chroms is None:
        held = scored
    else:
        held = tuple(c for c in scored if c in set(held_out_chroms))
        missing = [c for c in held_out_chroms if c not in scored]
        if missing:
            raise ValueError(
                f"held_out_chroms names {missing}, which this run does not score "
                f"({list(scored)}). The held-out scope is a subset of what was predicted; naming a "
                f"chromosome outside it would rank on positions that were never scored.")
        if not held:
            raise ValueError("held_out_chroms selected nothing from the scored chromosomes")
    split = len(held) < len(scored)

    per_track: Dict[str, Dict[str, Dict[str, object]]] = {}
    per_track_gw: Dict[str, Dict[str, Dict[str, object]]] = {}
    binarised: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, int]]] = {}
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        for rec in stream_tracks(model, source, device, kind=kind,
                                 batch_windows=batch_windows, progress=progress):
            var = _varpool(varpool_root, varpool_corpus, rec.assay, rec.chroms, n_bins, var_cache)
            common = dict(gene_annotations=gene, enh_annotations=enh, var=var, seed=seed,
                          c_index_pairs=c_index_pairs, with_curve=with_curve,
                          signal_target_transform=signal_target_transform,
                          crps_approx=crps_approx, crps_seed=crps_seed)
            held_here = tuple(c for c in rec.chroms if c in set(held))
            per_track[rec.key] = score_track(rec, chroms=held_here or None, **common)
            if split:
                per_track_gw[rec.key] = score_track(rec, chroms=rec.chroms, **common)
            if "P" in blocks and kind == "impute":
                bits = _binarise(rec, signal_target_transform)
                if bits is not None:
                    binarised.setdefault(rec.assay, []).append((rec.key, *bits))

    result: Dict[str, Any] = {
        "provenance": {
            **source.provenance(),
            "suite": "candi.bench",
            "seed": int(seed),
            "kinds": list(kinds),
            "blocks": list(blocks),
            "depth_center": source.depth_center(),
            "annotation_assets": ann.verify_assets(),
            # D30 — WHICH SPACE the test-loss `gaussian_nll` was scored in. The key is written even
            # on a count-only run: "no signal head, so `none` was never applied to anything" and
            # "the field was forgotten" must not look the same in a result file.
            "signal_target_transform": str(signal_target_transform),
            # The space the pval arm's BENCHMARK keys are in, run-wide. It is a constant and that is
            # the point: a macro roll-up drops the per-track string keys (`macro_mean` averages
            # scalars), so without this a reader of the macro alone could not tell an inverted run
            # from a pre-contract one.
            "pval_pred_space": D.SIGNAL_EVAL_SPACE,
            # ABSENT WHEN THE CLOSED FORM WAS USED, and that is deliberate: the default run must
            # produce the same provenance dict it always did, so the presence of these three keys is
            # itself the flag that a count-arm CRPS in this file is an ESTIMATE. A sampled score is
            # not reproducible from the prediction alone — it needs k and the seed too.
            **({} if crps_approx is None else
               {"crps_estimator": "fair_sampled", "crps_k": int(crps_approx),
                "crps_seed": int(crps_seed)}),
            "c_index_note": "the C-index is the ONE documented exception to whole-track scoring "
                            "(EVAL_PLAN.md D3): it is pairwise, so it is estimated over a seeded "
                            "pair sample and never quoted without c_index_se.",
            "msevar": ({"pool": varpool_corpus,
                        **{k[1]: v for k, v in var_cache.items() if k[0] == "__members__"}}
                       if varpool_root else
                       {"pool": None,
                        "note": "no --varpool given, so msevar is ABSENT rather than 0.0"}),
            **(dict(extra_provenance) if extra_provenance else {}),
        },
        "tracks": sorted(per_track),
        "per_track": per_track,
        "macro": {arm: macro_mean(per_track, arm) for arm in ARMS},
        "panels": {arm: panel_macros(per_track, arm) for arm in ARMS},
        "panel": panel_specificity(binarised) if binarised else {},
        "ranking": None,
    }
    result["provenance"]["scope"] = {
        "ranked": SCOPE_HELD_OUT,
        "held_out_chroms": list(held),
        "scored_chroms": list(scored),
        "genome_wide_computed": bool(split),
        "note": (
            "`per_track`, `macro` and `panels` are the HELD-OUT scope, which is the ranked number "
            "(plan/BENCHMARK_DESIGN.md 4). `genome_wide` carries the same three over every scored "
            "chromosome."
            if split else
            "One scope only: the run scored exactly the held-out chromosomes, so there is no "
            "genome-wide aggregation to make. Under 4's blanking rule a method fit at every "
            "position is run this way on purpose -- its genome-wide number would be a memorisation "
            "score, so it is NOT COMPUTED rather than computed and withheld."),
    }
    if split:
        result["genome_wide"] = {
            "chroms": list(scored),
            "per_track": per_track_gw,
            "macro": {arm: macro_mean(per_track_gw, arm) for arm in ARMS},
            "panels": {arm: panel_macros(per_track_gw, arm) for arm in ARMS},
            "note": "Comparability with a literature that scores at the positions it fits. Not "
                    "ranked, and carries the per-cell in-sample fraction on the board (4).",
        }
    for kind in kinds:
        if kind != "impute":
            result[f"macro_{kind}"] = {arm: macro_mean(per_track, arm, kind=kind) for arm in ARMS}
            if split:
                result["genome_wide"][f"macro_{kind}"] = {
                    arm: macro_mean(per_track_gw, arm, kind=kind) for arm in ARMS}
    if "C" in blocks:
        result["C"] = c_block(model, source, device, kind=kinds[0], n_windows=c_windows,
                              n_resamples=c_resamples, seed=seed)
    return result
