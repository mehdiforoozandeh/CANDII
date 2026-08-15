"""h64 — the ARM SWITCH that brackets the covariate-gradient meter with two controls.

NEW FILE. `train.py::_grad_norms` already measures how much gradient reaches each covariate embedder
relative to the trunk. A meter with no bracket is a number, not a measurement: it cannot say whether a
small ratio means "starved" or "this is simply what a working pathway reads on this model". This
module supplies the two brackets, as a transform on the ASSEMBLED BATCH — never on the architecture.

Three arms, differing in the contents of metadata ROW 3 (`run_type`) and nothing else. Same
architecture, same parameter count, same `n_rows=4`, same seed, same steps, same data.

  off       row 3 = the true `run_type`. The reading under test. A STRICT no-op: `make_meta_probe`
            returns None, no object exists, no RNG is created and no call site fires.
  shuffled  row 3 permuted ACROSS THE BATCH, per assay column, among the non-sentinel entries only.
            Same marginal, zero association => the NEGATIVE control.
  planted   row 3 = a quantile-binned random scalar drawn PER SAMPLE, which is ALSO added to the
            TARGET in log space => the POSITIVE control. Row 3 is the sole route to the shift.

NO FIFTH ROW. Both arms OVERWRITE row 3. Appending a row would reuse the `num_cells > 0` five-row
machinery built for the cell-identity ablation, which a standing PI ruling forbids from spreading into
this node; it would also change `n_rows`, the parameter count and the constructor RNG stream, so the
arms would no longer be the same model.

THE SENTINEL LAYOUT IS PRESERVED EXACTLY. `MISSING` (-1) and `CLOZE` (-2) entries of row 3 are never
written and never moved, in either arm. `DataMasker._mask_full_assay` writes CLOZE into
`metadata[:, :4, masked_assays]`, so the sentinel layout carries the masking pattern; an arm that
scrambled it would differ from `off` in WHICH ASSAY IS BEING IMPUTED, not only in `run_type`, and the
contrast would be confounded at its root.

THE PERMUTATION AXIS IS DEGENERATE ON THIS PANEL, AND THAT IS MEASURED, NOT WORKED AROUND.
`CandiKitH5Dataset.__iter__` draws ONE biosample per batch (`dataset.py`, `bios = rng.choice(
bios_pool)`) and `run_type` is a property of `(biosample, assay)`. Row 3 is therefore CONSTANT down
the batch axis on `eic_full`, and permuting along B is the identity — measured on six consecutive real
batches, zero varying columns. The pre-registered axis is kept (a PI ruling on it is pending) and the
degeneracy is made LOUD instead: a one-time warning at the first no-op step, a per-step
`meta_probe/shuffle_is_noop`, a running `meta_probe/shuffle_noop_frac`, and an end-of-run summary. A
`shuffle_noop_frac` of 1.0 says, in the run's own record, that the negative control never differed
from the real arm.

THE PLANTED COVARIATE CARRIES ONE BIT, NOT A CONTINUUM. `MetadataEmbedding` builds
`runtype_embedding = nn.Embedding(num_runtypes + 2, embed_dim)` with `num_runtypes = 2`, and its
`forward` RAISES on any value >= `num_runtypes`. So the "quantile-binned scalar" is binned to
`{0, 1}` — a median split. The table is NOT widened: that would change the parameter count and break
the same-model premise the whole node rests on.

ONLY THE TARGET IS SHIFTED. `y_data` (counts) is multiplied by `2**(delta * (2*s - 1))`; `x_data` is
left bit-identical, and so are `y_pval` and `y_peaks` (the loss is counts-only). This is the entire
validity argument of the positive control: if the input carried the shift too, the model could read it
off the signal and would never need the covariate. The product is ROUNDED back to an integer because
the NB loss validates its support and refuses a half-count — see `_shift`.

THE RNG IS DEDICATED. Every draw comes from `np.random.default_rng(seed ^ 0xC0FF_EE64)`, never from
the loop's shared data stream — the same rule `unmask_rng` follows in `train.py`. Drawing from the
shared stream would advance it once per step and the arms would visit DIFFERENT DATA, not merely a
different covariate. That exact bug has been caught in this repo once already.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from candi._vendored import CLOZE, MISSING

__all__ = ["META_PROBE_MODES", "MetaProbe", "make_meta_probe", "RUN_TYPE_ROW",
           "DEFAULT_META_PROBE_DELTA"]

# Row 3 of the [B, n_rows, A] metadata tensor. h62 measured `run_type` as the least informative of the
# four covariates (0.1276 bits EIC / 0.0931 MERGED, conditional on assay + read_length), so
# overwriting it destroys the least real signal of any available row.
RUN_TYPE_ROW = 3

# log2 units. 1.0 => the two classes differ by 4x in expectation (2x up vs 2x down) — large and
# unmistakable, which is exactly what a positive control wants.
DEFAULT_META_PROBE_DELTA = 1.0

META_PROBE_MODES = ("off", "shuffled", "planted")

# XOR'd into the run seed to fork a stream that is provably disjoint from the data stream and from
# `unmask_rng`'s `0xC10A_5EED`. Never reuse either constant here.
_META_PROBE_SEED_XOR = 0xC0FF_EE64

_N_PLANTED_BINS = 2          # median split — see the module docstring; do NOT widen the table


def _is_sentinel(row: torch.Tensor) -> torch.Tensor:
    """`[B, C]` bool: True where row 3 holds MISSING (-1) or CLOZE (-2) rather than a run_type id."""
    return (row == float(MISSING)) | (row == float(CLOZE))


class MetaProbe:
    """The `shuffled` / `planted` transform. Constructed only for those two modes — `off` has no object.

    One `apply` per batch. Both arms are pure functions of the batch and of this object's own RNG;
    nothing here reads or writes the model, the optimizer, the loss, or the shared data stream.
    """

    def __init__(self, mode: str, *, seed: int, delta: float = DEFAULT_META_PROBE_DELTA,
                 num_runtypes: int = 2, row: int = RUN_TYPE_ROW) -> None:
        if mode not in ("shuffled", "planted"):
            raise ValueError(f"MetaProbe takes 'shuffled' or 'planted'; got {mode!r} "
                             "(use make_meta_probe, which maps 'off' to None)")
        if int(num_runtypes) < _N_PLANTED_BINS:
            raise ValueError(f"planted needs at least {_N_PLANTED_BINS} valid run_type levels; the "
                             f"embedding table declares num_runtypes={num_runtypes}")
        self.mode = str(mode)
        self.delta = float(delta)
        self.num_runtypes = int(num_runtypes)
        self.row = int(row)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed ^ _META_PROBE_SEED_XOR)
        # counters — `record=True` calls only, so the mid-training eval cannot dilute the training
        # arm's own no-op fraction
        self.n_steps = 0
        self.n_noop = 0
        self._last_noop: Optional[bool] = None
        self._last_frac_hi = float("nan")
        self._warned = False

    # -- reporting ---------------------------------------------------------------------------------

    @property
    def noop_frac(self) -> float:
        return (self.n_noop / self.n_steps) if self.n_steps else float("nan")

    def banner(self) -> str:
        if self.mode == "shuffled":
            return ("[meta-probe] mode=shuffled — NEGATIVE control. Metadata row 3 (run_type) is "
                    f"permuted along the BATCH axis, per assay column, among non-sentinel entries "
                    f"only. rng=seed({self.seed})^0x{_META_PROBE_SEED_XOR:X}, dedicated. Watch "
                    "meta_probe/shuffle_noop_frac: on a one-biosample-per-batch panel this "
                    "permutation is the IDENTITY and the arm is not a control at all.")
        return ("[meta-probe] mode=planted — POSITIVE control. Metadata row 3 (run_type) is "
                f"OVERWRITTEN with a per-sample median-split bin in {{0,1}} (ONE BIT, not a "
                f"continuum: the run_type table has only {self.num_runtypes} valid levels), and "
                f"y_data is multiplied by 2**(+/-{self.delta}) and rounded back to an integer (the "
                f"NB loss validates its support). x_data, y_pval and y_peaks are "
                f"untouched, so row 3 is the ONLY route to the shift. "
                f"rng=seed({self.seed})^0x{_META_PROBE_SEED_XOR:X}, dedicated.")

    def step_metrics(self) -> Dict[str, float]:
        """Per-step scalars for the run's metric stream. Empty until the first recorded step."""
        if not self.n_steps:
            return {}
        if self.mode == "shuffled":
            return {"meta_probe/shuffle_is_noop": float(bool(self._last_noop)),
                    "meta_probe/shuffle_noop_frac": self.noop_frac}
        return {"meta_probe/planted_frac_hi": self._last_frac_hi,
                "meta_probe/planted_delta": self.delta}

    def stats(self) -> Dict[str, object]:
        """The block that rides into the run JSON. `shuffle_noop_frac` is the whole point for the
        negative arm: it is the measured answer to "did this control ever do anything"."""
        return {
            "meta_probe": self.mode,
            "meta_probe_delta": self.delta,
            "meta_probe_steps": int(self.n_steps),
            "meta_probe_noop_steps": int(self.n_noop),
            "meta_probe_shuffle_noop_frac": (self.noop_frac if self.mode == "shuffled" else None),
            "meta_probe_row": self.row,
            "meta_probe_num_runtypes": self.num_runtypes,
            "meta_probe_planted_bins": (_N_PLANTED_BINS if self.mode == "planted" else None),
        }

    def summary(self) -> str:
        if self.mode != "shuffled":
            return (f"[meta-probe] SUMMARY mode=planted steps={self.n_steps} delta={self.delta} "
                    f"bins={_N_PLANTED_BINS} (1 bit) — row 3 was overwritten and y_data shifted on "
                    "every step; x_data was never touched.")
        f = self.noop_frac
        line = (f"[meta-probe] SUMMARY mode=shuffled steps={self.n_steps} "
                f"no-op steps={self.n_noop} shuffle_noop_frac={f:.4f}")
        if self.n_steps and f >= 1.0:
            line += ("\n[meta-probe] shuffle_noop_frac=1.000 — the permutation NEVER changed a "
                     "tensor. This arm trained on the REAL run_type and is NOT a negative control. "
                     "Any contrast against `off` is noise between two identical objectives.")
        elif self.n_steps and f > 0.0:
            line += (f"\n[meta-probe] the permutation was the identity on {f:.1%} of steps; the "
                     "negative control is only partial.")
        return line

    def _warn_noop(self, ) -> None:
        print(
            "\n[meta-probe] ============ WARNING: THE NEGATIVE CONTROL IS A NO-OP ============\n"
            "[meta-probe] The run_type shuffle changed NOTHING on this batch: the permuted tensor is\n"
            "[meta-probe] bit-identical to the original, so this `shuffled` arm is training on the\n"
            "[meta-probe] REAL run_type. On this batch it is not a negative control.\n"
            "[meta-probe] REASON: CandiKitH5Dataset.__iter__ draws ONE biosample per batch\n"
            "[meta-probe]   (`bios = rng.choice(bios_pool)`), and run_type is a property of\n"
            "[meta-probe]   (biosample, assay). Row 3 is therefore CONSTANT down the batch axis, and\n"
            "[meta-probe]   permuting along B is the identity.\n"
            "[meta-probe] The pre-registered permutation axis is NOT being worked around here.\n"
            "[meta-probe] Watch meta_probe/shuffle_noop_frac — it measures how often this happens,\n"
            "[meta-probe] and it is printed again at the end of training.\n"
            "[meta-probe] ==================================================================\n",
            flush=True)

    # -- the transform -----------------------------------------------------------------------------

    def apply(self, prep: Dict[str, object], *, record: bool = True) -> Dict[str, object]:
        """TRAINING seam: transform a `prepare_masked_batch` prep dict, in place on the DICT.

        The tensors themselves are never mutated — each write produces a new tensor and the dict entry
        is rebound. `prepare_masked_batch` hands back `batch["y_data"].to(device)`, which on CPU is the
        SAME object the dataset yielded; an in-place shift there would silently corrupt the batch.
        """
        metas, targets = self.apply_tensors(
            (prep["x_meta"], prep["y_meta"]), (prep["y_data"],), record=record)
        prep["x_meta"], prep["y_meta"] = metas
        prep["y_data"] = targets[0]
        return prep

    def apply_tensors(self, metas: Sequence[Optional[torch.Tensor]],
                      targets: Sequence[Optional[torch.Tensor]] = (),
                      *, record: bool = True,
                      ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """The core. `metas` are `[B, n_rows, C]` prompts; `targets` are `[B, L, A]` count tensors.

        ONE call = ONE batch = ONE draw, so every tensor handed in shares the same per-sample bin.
        The eval seam passes four tensors (both prompts and both count targets) in a single call for
        exactly that reason — two calls would hand the encoder and the decoder different bins.
        """
        metas = list(metas)
        targets = list(targets)
        if self.mode == "shuffled":
            out_metas, changed = [], False
            for m in metas:
                if m is None:
                    out_metas.append(None)
                    continue
                mm, ch = self._shuffle(m)
                out_metas.append(mm)
                changed = changed or ch
            if record:
                self.n_steps += 1
                self._last_noop = not changed
                self.n_noop += int(not changed)
                if (not changed) and not self._warned:
                    self._warned = True
                    self._warn_noop()
            return out_metas, targets                      # targets are returned untouched, as-is

        B = next((int(t.shape[0]) for t in list(metas) + targets if t is not None), 0)
        bins = self._draw_bins(B)
        out_metas = [None if m is None else self._plant(m, bins) for m in metas]
        out_targets = [None if t is None else self._shift(t, bins) for t in targets]
        if record:
            self.n_steps += 1
            self._last_frac_hi = float(bins.mean()) if bins.size else float("nan")
        return out_metas, out_targets

    # -- arms --------------------------------------------------------------------------------------

    def _shuffle(self, meta: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Permute row 3 along B, INDEPENDENTLY PER COLUMN, among non-sentinel entries only.

        Per column rather than per tensor because the pre-registration says "`run_type` permuted
        across the batch": each assay column has its own marginal, and a single shared permutation
        would mix them. Sentinels are not part of any column's value multiset and never move.
        """
        new = meta.clone()
        row = new[:, self.row, :]                          # basic-index VIEW into `new`
        orig = row.detach().to("cpu").numpy().copy()       # [B, C]
        out = orig.copy()
        for c in range(orig.shape[1]):
            col = orig[:, c]
            idx = np.flatnonzero((col != float(MISSING)) & (col != float(CLOZE)))
            if idx.size < 2:
                continue                                   # nothing to permute; sentinels stay put
            out[idx, c] = col[idx[self.rng.permutation(idx.size)]]
        row.copy_(torch.as_tensor(out, dtype=meta.dtype, device=meta.device))
        # THE DETECTOR. Compares the WHOLE tensor, so it also catches a bug that moved a sentinel.
        return new, not bool(torch.equal(new, meta))

    def _draw_bins(self, B: int) -> np.ndarray:
        """Per-sample median split of a U(0,1) scalar into `{0, 1}`.

        The split is at the POPULATION median (0.5), not the batch's empirical median: an empirical
        quantile would couple the samples in a batch to each other, and the pre-registration says the
        scalar is drawn per sample. `meta_probe/planted_frac_hi` reports the realized balance.
        """
        u = self.rng.random(int(B))
        return (u >= 0.5).astype(np.int64)

    def _plant(self, meta: torch.Tensor, bins: np.ndarray) -> torch.Tensor:
        """Overwrite row 3 with the per-sample bin, broadcast across every column, sentinels intact."""
        new = meta.clone()
        row = new[:, self.row, :]
        vals = torch.as_tensor(bins, dtype=meta.dtype, device=meta.device).unsqueeze(1).expand_as(row)
        row.copy_(torch.where(_is_sentinel(row), row, vals))
        return new

    def _shift(self, target: torch.Tensor, bins: np.ndarray) -> torch.Tensor:
        """`round(y * 2**(delta * (2*s - 1)))` at REAL COUNTS only.

        THE ROUNDING IS FORCED, NOT A CHOICE. The loss is `torch.distributions.NegativeBinomial`,
        whose support is `IntegerGreaterThan(0)`; `log_prob` VALIDATES it and raises. A count of 1
        scaled by 2**-1 is 0.5 and the arm crashes on its first step (measured, on gatec). So the
        shifted target is rounded back onto the counting numbers. The up-shift is exact whenever
        `2**delta` is an integer; only the down-shift rounds, by at most half a count, which is far
        below the 4x separation the control is built to create.

        Unavailable entries of `y_data` are the MISSING sentinel (-1.0), not a count; scaling one
        would turn it into -2.0, which is the CLOZE value. They are left exactly as they are.
        """
        expo = self.delta * (2.0 * bins.astype(np.float64) - 1.0)
        scale = torch.as_tensor(np.exp2(expo), dtype=target.dtype, device=target.device)
        scale = scale.view(-1, *([1] * (target.dim() - 1)))
        return torch.where(target >= 0, torch.round(target * scale), target)


def make_meta_probe(mode: str, *, seed: int, delta: float = DEFAULT_META_PROBE_DELTA,
                    num_runtypes: int = 2) -> Optional[MetaProbe]:
    """`off` -> None, and None is what makes `off` a STRICT no-op.

    No object exists, so no RNG is seeded, no counter moves and every call site's
    `if meta_probe is not None` is skipped. The `off` arm is the pre-h64 training path byte for byte.
    """
    if mode not in META_PROBE_MODES:
        raise ValueError(f"meta_probe must be one of {META_PROBE_MODES}; got {mode!r}")
    if mode == "off":
        return None
    return MetaProbe(mode, seed=seed, delta=delta, num_runtypes=num_runtypes)
