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
    "Pair", "TrackRecord", "EvalSource", "H5Source", "StoreSource", "open_source",
    "full_tiling", "decode_groups", "stream_tracks", "score_track", "panel_specificity", "c_block",
    "run_bench", "macro_mean",
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


# ---------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    """One `(input biosample, target biosample)`. They are equal for denoising and for store LOO."""
    input_biosample: str
    target_biosample: str

    def __str__(self) -> str:                                   # pragma: no cover - display only
        return f"{self.input_biosample}->{self.target_biosample}"


def track_key(pair: Pair, assay: str, kind: str) -> str:
    """`T_cell|imp_cell|assay` for imputation — `EVAL_PLAN.md` §4's key, and `eval.py`'s too.

    Denoising appends a fourth field. On the h5 the imputation target is always a `V_`/`B_` cell so
    the three-field keys are already distinct; on the store an imputed assay is held out of the same
    biosample that denoises the others, and without the suffix the two would overwrite each other.
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

    @property
    def key(self) -> str:
        return track_key(self.pair, self.assay, self.kind)

    @property
    def has_pval(self) -> bool:
        return bool(self.signal_mu)

    def n_bins(self) -> int:
        return int(sum(len(self.counts[c]) for c in self.chroms))


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

    def depth_center(self) -> float:
        raise NotImplementedError

    def n_bins(self, chrom: str) -> int:
        raise NotImplementedError

    def pairs(self, kind: str) -> List[Pair]:
        raise NotImplementedError

    def targets(self, pair: Pair, kind: str) -> List[int]:
        """Assay column indices this pair supplies ground truth for, under `kind`."""
        raise NotImplementedError

    def cross_cell(self, kind: str) -> bool:
        """Does `kind` read its ground truth from a DIFFERENT biosample than the prompt?

        True means the batch carries the `y_*_imp` keys — the target cell's counts, p-values, peaks
        and covariates — and the harness scores against those rather than against the input cell's
        own tracks. It also means nothing has to be hidden from the encoder: the target assay is one
        the prompt cell genuinely does not have.

        Imputation on a bake is always cross-cell. On a store it depends on the regime (see
        `StoreSource.cross_cell`). Denoising never is, anywhere.
        """
        return kind == "impute"

    def windows(self, chrom: str) -> List[int]:
        return full_tiling(self.n_bins(chrom), self.context_bins)

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

    **Declared pairs (D31).** A regime may carry `eval_pairs`, a list of `[input, target]`
    biosample names — `configs/regime.eic_val.json` carries 26 of them. Then imputation is the task
    the training loop runs and the bake has always run: prompt with the input cell, score against
    the target cell's tracks, and hide nothing, because the target cell is a different biosample
    whose assays the prompt does not carry. The pairing is READ OFF THE REGIME and nowhere else —
    D16 makes biosample names opaque ids, so no `T_`/`V_`/`B_` prefix is parsed here.

    **No declared pairs.** Nothing to impute FROM, so imputation is leave-one-assay-out within an
    eval biosample: the assay is masked out of the encoder input exactly as
    `DataMasker._mask_full_assay` masks it during training (data, metadata rows 0-3 and availability
    all set to `CLOZE`), and the decoder is asked for it from the prompt alone. That is the same
    task the model was trained on, and it is the only imputation a pair-less regime can pose.

    Batch assembly is `StoreDataset._make_batch`, called with a window list this class substitutes
    for the regime's plan. Re-implementing it here would duplicate the thinning, the depth-adjusted
    metadata row and the D19 availability rule, and those three are exactly the places a second copy
    would drift. Same package, deliberate: `_make_batch(name, widx, rng)` is a pure function of its
    arguments and `self._windows`.
    """

    kind = "store"

    def __init__(self, regime_path: Path | str, *, chroms: Optional[Sequence[str]] = None,
                 biosamples: Optional[Sequence[str]] = None):
        from candi.store.dataset import StoreDataset
        from candi.store.regime import Regime

        self.regime_path = Path(regime_path)
        regime = Regime.from_file(self.regime_path)
        #: The parsed regime, kept because `eval_pairs` decides which imputation this source poses.
        self.regime = regime
        self.ds = StoreDataset(regime, train=False, batch_size=1, dsf_sampling="off",
                               shuffle=False, deterministic=True, biosamples=biosamples)
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

    def depth_center(self) -> float:
        return float(self.ds.depth_center())

    def n_bins(self, chrom: str) -> int:
        return int(self._n_bins[chrom])

    def cross_cell(self, kind: str) -> bool:
        """Only with declared pairs is there a second cell to read the ground truth from."""
        return kind == "impute" and self.regime.has_eval_pairs

    def pairs(self, kind: str) -> List[Pair]:
        """The declared `(input, target)` pairs when the regime has them; self-pairs otherwise.

        Declaration order, filtered to the inputs this source actually opened — `biosamples=` can
        narrow the pool, and a pair whose prompt cell is not in it has nothing to prompt with.
        Denoising is a self-pair over the pool whatever the regime declares: it reads and scores
        one cell.
        """
        if not self.cross_cell(kind):
            return [Pair(b, b) for b in self.ds.biosample_pool]
        pool = set(self.ds.biosample_pool)
        return [Pair(a, b) for a, b in self.ds.eval_pairs if a in pool]

    def targets(self, pair: Pair, kind: str) -> List[int]:
        """Under declared pairs: assays the TARGET cell has AND THE INPUT CELL DOES NOT.

        Both halves of that rule are load-bearing, and it is the same rule the other two
        implementations of this task enforce — `H5Source.targets` above, and
        `StoreDataset._imp_keys`, whose `imp_map = (y_avail <= 0) & (y_data_imp != -1)` reads
        `y_avail` off the INPUT cell. An assay neither cell has is not a target because there is no
        ground truth; an assay BOTH cells have is not a target because the encoder can read it
        straight off the prompt, so predicting it measures copying rather than imputation.

        Without declared pairs the pair is a self-pair and every available assay is a target: for
        `denoise` it is denoised, and for leave-one-out imputation it is held out one at a time.
        """
        if not self.cross_cell(kind):
            avail = self.ds._availability[pair.input_biosample]
            return [a for a in range(len(self.assays)) if bool(avail[a])]
        t_avail = self.ds._availability[pair.target_biosample]
        x_avail = self.ds._availability[pair.input_biosample]
        return [a for a in range(len(self.assays)) if bool(t_avail[a]) and not bool(x_avail[a])]

    def _raw_batch(self, pair: Pair, chrom: str, starts: Sequence[int], *,
                   imp_target: Optional[str] = None) -> Dict[str, Any]:
        self.ds._windows = [(chrom, int(s)) for s in starts]
        free = np.random.default_rng(self._run_seed)
        return self.ds._make_batch(pair.input_biosample, list(range(len(starts))), free,
                                   imp_target=imp_target)

    def batch(self, pair: Pair, chrom: str, starts: Sequence[int], kind: str, *,
              x_dsf: int = 1) -> Dict[str, Any]:
        out = self._raw_batch(
            pair, chrom, starts,
            imp_target=pair.target_biosample if self.cross_cell(kind) else None)
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
        """The ground truth at `dsf`, read off whichever cell holds it.

        Like `H5Source.counts_at_dsf` this takes no `kind`, so the cell is decided by the pair
        itself: two different biosamples can only be a declared eval pair, and then the truth is the
        TARGET's — thinned under the target's own name, as `StoreDataset._imp_keys` thins it, so two
        pairs sharing a target see the same numbers.

        **NO `side` IN THE SEED, DELIBERATELY (t37).** See `_thin_input` above for the full
        argument. In short: on a self-pair this and `_thin_input` share an entropy tuple and return
        byte-identical arrays, which is what the bake does — `counts_dsf{d}` is one stored array
        that both the input read and the truth read land on. On a declared pair `name` is the target
        cell, so the tuples differ and the two sites separate on their own. Do not "fix" this by
        passing `side="y"`: it moves `depthcounterfact` and `depthblind` by realization noise and
        makes the store disagree with the h5 on every self-pair.
        """
        from candi.store.dataset import draw_seed, dsf_milli, thin_counts

        imp = pair.target_biosample if pair.target_biosample != pair.input_biosample else None
        out = self._raw_batch(pair, chrom, starts, imp_target=imp)
        y = (out["y_data_imp"] if imp else out["y_data"]).numpy().copy()
        if dsf == 1:
            return y
        name = imp or pair.input_biosample
        avail = (self.ds._availability[imp] if imp
                 else (out["y_avail"][0].numpy() > 0))
        for a, assay in enumerate(self.assays):
            if not bool(avail[a]):
                continue
            for j, s in enumerate(starts):
                rng = np.random.default_rng(
                    draw_seed(self._run_seed, name, assay, chrom, int(s), dsf_milli(dsf)))
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
        return H5Source(h5, **kw)
    return StoreSource(store, **{k: v for k, v in kw.items() if k in ("chroms", "biosamples")})


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
    """Is this the store's leave-one-assay-out imputation — the one case that masks the input?"""
    return kind == "impute" and source.kind == "store" and not source.cross_cell(kind)


def decode_groups(source: EvalSource, kind: str, cols: Sequence[int]) -> List[List[int]]:
    """Which target assays can share one forward pass.

    On the h5, and for denoising anywhere, all of them: the imputation target is a `V_`/`B_` assay
    the input cell genuinely does not have, so nothing has to be hidden and one pass answers for
    every target at once.

    The store has both cases. Under DECLARED PAIRS it behaves like the h5 — the target cell is a
    different biosample, `targets()` keeps only the assays the prompt cell lacks, and one pass
    answers for all of them.

    Without declared pairs, one assay per pass, and that is not an optimisation left on the table.
    Leave-one-out holds the assay out **by mask**, so masking every target at once would not be
    leave-one-out — it would be "impute all of them from whatever is left", a strictly harder task
    that is not the one the model trained on. On `V_aa`, which carries two assays, it would also
    empty the encoder input entirely.
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
                    if source.cross_cell(kind):
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
            for a in cols:
                r, b = recs[a], buf[a]
                r.has_peak_head = bool(have_peak)
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

def _chrom_offsets(rec: TrackRecord) -> Dict[str, int]:
    off, acc = {}, 0
    for c in rec.chroms:
        off[c] = acc
        acc += len(rec.counts[c])
    return off


def _p_block(rec: TrackRecord, gene_annotations: Sequence[str],
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
    sig = E.dict_to_arr(rec.pval, rec.chroms)
    pred = E.dict_to_arr(rec.signal_mu if signal_mu is None else signal_mu, rec.chroms)
    pk = E.dict_to_arr(rec.peaks, rec.chroms).astype(bool)
    out: Dict[str, object] = {
        "acc_by_obs_strength": P.accuracy_by_strength(sig, pk, pred, bin_by="obs"),
        "acc_by_imp_strength": P.accuracy_by_strength(sig, pk, pred, bin_by="imp"),
    }
    off = _chrom_offsets(rec)
    if rec.assay == "H3K4me3":
        wins = [(lo + off[c], hi + off[c]) for c in rec.chroms
                for lo, hi in P.promoter_windows(gene_annotations, c, len(rec.pval[c]))]
        out["prom_corr_h3k4me3"] = P.region_correlation(sig, pred, wins)
    if rec.assay in ("DNase-seq", "DNase"):
        wins = [(lo + off[c], hi + off[c]) for c in rec.chroms
                for lo, hi in P.peak_regions(rec.peaks[c].astype(bool))]
        out["peak_shape_corr_dnase"] = P.region_correlation(sig, pred, wins)
    return out


def loss_block(rec: TrackRecord, *, signal_target_transform: str = "none") -> Dict[str, object]:
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

    **Head gating, from the model rather than from the numbers.** `gaussian_nll` needs a signal head
    (`rec.has_pval`) and `bernoulli_nll` needs a peak head (`rec.has_peak_head`, recorded in
    `stream_tracks` where the output dict is the authority). An absent head means an ABSENT KEY, not
    a nan: a nan travels into `macro_mean`, gets skipped by a finiteness filter, and leaves a reader
    unable to tell "no head" from "the head produced garbage".

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
    chroms = rec.chroms
    out: Dict[str, object] = {
        "nb_nll": D.nb_nll(E.dict_to_arr(rec.n, chroms), E.dict_to_arr(rec.mu, chroms),
                           E.dict_to_arr(rec.counts, chroms)),
        "signal_target_transform": str(signal_target_transform),
    }
    if rec.has_pval:
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
                with_curve: bool = False) -> Dict[str, Dict[str, object]]:
    """Every block this track can carry, per arm. `count` always; `pval` when the signal head is on.

    `var` is the D7 variance pool as `{chrom: vector}`, aligned bin-for-bin with the track. It is
    only ever applied to the arm it was built in — the pool that matches the 267 EIC training
    experiments is a variance of -log10 p-values, and weighting squared count error by it would be
    a number with no interpretation. Omitting it omits `msevar`, which is the honest outcome; the
    organizers' code returns a bare `0.0` there and `annotations.load_variance_pool` says why we do
    not.

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
    """
    arms: Dict[str, Dict[str, object]] = {}
    chroms = rec.chroms
    loss = loss_block(rec, signal_target_transform=signal_target_transform)

    y_c = E.dict_to_arr(rec.counts, chroms)
    arms["count"] = {
        **loss,
        **E.score_track(rec.counts, rec.mu, chroms, gene_annotations=gene_annotations,
                        enh_annotations=enh_annotations),
        **D.nb_suite(E.dict_to_arr(rec.n, chroms), E.dict_to_arr(rec.mu, chroms), y_c,
                     seed=seed, n_pairs=c_index_pairs),
        **B.binary_suite(E.dict_to_arr(rec.peaks, chroms).astype(bool),
                         E.dict_to_arr(rec.peak_score, chroms), y_c, with_curve=with_curve),
    }
    if rec.has_pval:
        y_p = E.dict_to_arr(rec.pval, chroms)
        pool = None if var is None else E.dict_to_arr(var, chroms)
        # The prediction moves, the truth does not. `invert_signal_prediction` returns the caller's
        # own arrays under `"none"`, so the h5 path builds dicts of the same objects and every key
        # below is bit-identical to what it was before the inversion existed.
        inv = {c: D.invert_signal_prediction(rec.signal_mu[c], rec.signal_sigma[c],
                                             signal_target_transform) for c in chroms}
        mu_p = {c: v[0] for c, v in inv.items()}
        sigma_p = {c: v[1] for c, v in inv.items()}
        arms["pval"] = {
            **loss,
            **E.score_track(rec.pval, mu_p, chroms, gene_annotations=gene_annotations,
                            enh_annotations=enh_annotations, var=pool),
            **D.gauss_suite(E.dict_to_arr(mu_p, chroms), E.dict_to_arr(sigma_p, chroms), y_p,
                            seed=seed, n_pairs=c_index_pairs),
            # NOTHING TO INVERT HERE, and it is worth saying why rather than leaving a reader to
            # check: `binary_suite`'s ranking score is `peak_score` — the peak head's probability,
            # or the NB mean without that head — never `signal_mu`, so no signal-space value
            # reaches it. Its keys are rank-based besides, and every transform in the vocabulary is
            # strictly increasing, so they would be unchanged even if one did.
            **B.binary_suite(E.dict_to_arr(rec.peaks, chroms).astype(bool),
                             E.dict_to_arr(rec.peak_score, chroms), y_p, with_curve=with_curve),
            **_p_block(rec, gene_annotations, signal_mu=mu_p),
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
        arms[arm]["n_bins"] = rec.n_bins()
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
            if source.cross_cell(kind):
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
              with_curve: bool = False, progress: bool = False,
              extra_provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Score one checkpoint end to end and return the result JSON of `EVAL_PLAN.md` §4.

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

    per_track: Dict[str, Dict[str, Dict[str, object]]] = {}
    binarised: Dict[str, List[Tuple[str, np.ndarray, np.ndarray, int]]] = {}
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        for rec in stream_tracks(model, source, device, kind=kind,
                                 batch_windows=batch_windows, progress=progress):
            var = _varpool(varpool_root, varpool_corpus, rec.assay, rec.chroms, n_bins, var_cache)
            per_track[rec.key] = score_track(
                rec, gene_annotations=gene, enh_annotations=enh, var=var, seed=seed,
                c_index_pairs=c_index_pairs, with_curve=with_curve,
                signal_target_transform=signal_target_transform)
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
        "panel": panel_specificity(binarised) if binarised else {},
        "ranking": None,
    }
    for kind in kinds:
        if kind != "impute":
            result[f"macro_{kind}"] = {arm: macro_mean(per_track, arm, kind=kind) for arm in ARMS}
    if "C" in blocks:
        result["C"] = c_block(model, source, device, kind=kinds[0], n_windows=c_windows,
                              n_resamples=c_resamples, seed=seed)
    return result
