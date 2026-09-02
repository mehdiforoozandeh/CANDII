"""Training driver for the CANDI dual-conditioning recipe on real data (counts-only).

One invocation = one arm. Trains the golden-reference model (real 4-row metadata, per-assay `per_conv`
encoder + depth-offset NB head, COUNTS ONLY) on the train chromosomes with per-assay independent
DSF ON + cloze masking, watches held-out imputation every `--eval-every` epochs through
`candi.monitor`, and writes `{out_dir}/{tag}.json` + `{out_dir}/{tag}.ckpt`.

IT NO LONGER SCORES A CHECKPOINT. `candi.eval` owned the final `evaluate()` and it is deleted
(D15); scoring is `python -m candi.bench`, a separate command run after the training job. What this
module still owns is the mid-training dial and the best-checkpoint selection it drives.

Provenance: COPY+EDIT of
EpiDenoise/sandbox/diagnostics/dual_conditioning_real/run_real.py:1-286.

FROZEN: the optimizer/schedule/clip/loss/step order is load-bearing. Adam (coupled L2, NOT AdamW,
positional lr), grad-clip 1.0, cosine warmup_frac 0.1 / min_ratio 0.1, loss = mean NB NLL over unmasked
+ mean over cloze (unweighted sum), and the cudnn.deterministic / cudnn.benchmark pair with NO
`torch.use_deterministic_algorithms` (it crashes the candi_v2 encoder). Scale (num_assays,
context_length, assay order, DSF ladder, chromosomes) comes from the h5 attrs and is NEVER a flag.

h51 adds `--optimizer {adam,adamw}` (default `adam`) and `--trunk-wd` (default 0.0) WITHOUT relaxing
that freeze: on the default `adam` path the optimizer line below is the historical one, character for
character, so a run passing neither flag is bit-identical to the pre-h51 kit. `adamw` builds two
parameter groups (see `candi.param_groups`) and is the only new arithmetic here.

h64 adds `--meta-probe {off,shuffled,planted}` (default `off`) and `--meta-probe-delta` (default 1.0)
under the same rule. The switch lives entirely in `candi.meta_probe`, which transforms the
ASSEMBLED BATCH — metadata row 3 and, for the positive control, the count TARGET. At `off`
`make_meta_probe` returns None, so no object is built, no RNG is seeded and every call site below is
skipped: the training path is the pre-h64 one byte for byte. Nothing in the optimizer / schedule /
clip / loss / step order is touched by any of the three arms.

t23 adds `--store` (alias `--regime-file`) alongside `--h5`, mutually exclusive and exactly one
required, under the same rule. It changes WHERE batches come from and nothing else: both paths go
through the one factory `make_dataset`, which returns the pre-t23 `CandiKitH5Dataset` object for
`--h5` with the arguments its three call sites already passed. See `STORE.md` *Train off the store*.

The mid-training eval is `candi.monitor`, which scores every 25 bp bin of the regime's eval
chromosomes through `candi.bench` — two dials, imputation against the declared `eval_pairs` and
denoising against the prompt cells themselves, with the gap between them as the overfitting alarm.
Only imputation selects the checkpoint, and only imputation runs on the `--eval-every` cadence: the
denoising dial and the gap are computed once, at the end, on the selected checkpoint, because at the
wall-clocks measured in `cruxvault/results/t30/TIMING.md` two dials per check do not fit the budget.
It is STORE-ONLY, and that is now the whole story rather
than half of one: `--h5` used to keep `eval.quick_eval` for its mid-training curve, and that
function retired with `candi.eval`. An h5 run therefore TRAINS ONLY — no mid-training number, no
best-checkpoint selection, one checkpoint. The degradation is printed on the submit line, and
`python -m candi.bench --h5 …` scores the result afterwards.

Stage 2 adds `--heads` (default `count`) under the same rule. `nb_count_loss` is UNCHANGED, including
its `imp_weight` behaviour; the Gaussian signal and Bernoulli peak terms are auxiliary and are
constructed only when the model built the head that produces them, so at the default `aux_head_losses`
returns None and the loss expression `_train_step` evaluates is the pre-Stage-2 one. Evaluation stays
NB-only by ruling: the extra heads train, and nothing scores them yet.

t26 adds `--signal-target-transform` (default `auto`), which decides what space the SIGNAL head's
target sits in and applies it inside the loss — `PVAL_CODEC_PLAN.md` D30. It exists because the two
data paths do not hand over the same quantity: the bake stores `arcsinh(-log10 p)` and the store
stores the raw value, so before t26 the same head silently trained against two different targets
depending on which of `--h5` / `--store` opened the data. `auto` resolves to `none` on the h5 and
`arcsinh` on the store, which makes the h5 path bit-identical to its pre-t26 self, and the resolved
value is written into the run config so a run's target space is never inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from candi.batch import make_masker, prepare_masked_batch
from candi.dataset import CandiKitH5Dataset, h5_depth_center
from candi.meta_probe import DEFAULT_META_PROBE_DELTA, META_PROBE_MODES, make_meta_probe
from candi.encoder import FILM_INITS
from candi.model import (DEFAULT_FILM_TAPS, DEFAULT_HEADS, build_model, forward_full,
                         parse_film_taps, parse_heads)
from candi.monitor import Monitor, format_check, resolve_eval_every, tiered_keys
from candi.precision import (DEFAULT_PRECISION, PRECISION_HELP, PRECISIONS, assert_no_grad_scaler,
                             autocast_region, fp32_fence)


# ---------------------------------------------------------------------------
# t23 — where batches come from: a baked h5, or a CANDI_STORE through a regime file
# ---------------------------------------------------------------------------
# ONE new flag, `--store` (spelled `--regime-file` if you prefer), and it takes the REGIME FILE, not
# the corpus root. That is the whole reason one flag is enough: a regime already carries `store` as a
# required key (`store/regime.py::Regime.from_dict` refuses a regime without it), so a second
# `--store-root` flag would either duplicate the regime's own field or contradict it — and a
# contradiction between a flag and the file that is supposed to be the authority is exactly the class
# of bug the store was built to remove. `STORE_PLAN.md` §4: the regime file is the authority that
# replaced `h5.attrs`.
#
# THE NAME COLLISION IS REAL AND IS NOT PAPERED OVER. `--regime` already exists and means the MASKING
# regime (`type1` / `type2_loci`) — which windows the h5 loader draws from. It has nothing to do with
# a store regime file. Both help strings say so out loud.

#: `--store` help. Module-level so a test can assert it disambiguates `--regime` without re-parsing.
STORE_HELP = (
    "path to a CANDI_STORE REGIME FILE (json) — the store path. Mutually exclusive with --h5, and "
    "exactly one of the two is required. The regime's own `store` key names the corpus root, so no "
    "second flag is needed: see configs/regime.eic_smoke.json. NOT to be confused with --regime, "
    "which is the MASKING regime (type1 / type2_loci) and is unrelated. Also spelled --regime-file. "
    "On this path scale comes from the regime instead of h5.attrs, num_cells is 0 (cell_cond is "
    "refused, D16), and there is NO final evaluation in the training process — score the checkpoint "
    "afterwards with `python -m candi.bench`. The MID-TRAINING monitor does run here (candi.monitor: "
    "whole eval chromosomes, the imputation dial per check and the denoising dial once at the end) "
    "whenever the regime declares `eval_pairs`, and it is what selects the best checkpoint."
)

#: `--regime` help. It predates the store by a long way; the store gave its name a second meaning.
MASK_REGIME_HELP = (
    "the MASKING regime: type1 draws training windows from the h5's train chromosomes, type2_loci "
    "draws them from the baked cCRE/non-cCRE loci. NOTHING to do with --store, which takes a store "
    "REGIME FILE. type2_loci is h5-only — the store plans its own windows (regime.window_plan)."
)


def _manifest_sha256(corpus_root) -> Optional[str]:
    """sha256 of the corpus `manifest.json` bytes, or None when the store has no manifest.

    STORE_PLAN.md §4 requires the run record to carry this next to the regime's own hash: the regime
    says which columns, which biosamples and which chromosomes, and the manifest says which bytes
    were on disk under those names. Neither alone identifies the data a run saw.
    """
    try:
        from candi.store import layout as L
        p = L.manifest_path(corpus_root)
        return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
    except Exception:                                    # noqa: BLE001 — provenance is never fatal
        return None


@dataclass(frozen=True)
class DataSource:
    """Exactly one of a baked h5 or a store regime — the single object every call site is handed.

    It exists so that "which data path is this" is answered ONCE, at the top of `train_and_eval`,
    rather than three times inside three dataset constructions that would then be free to drift.
    """

    kind: str                                            # "h5" | "store"
    h5_path: Optional[str] = None
    regime_path: Optional[str] = None
    regime: Any = None                                   # candi.store.regime.Regime, store path only

    @classmethod
    def resolve(cls, *, h5=None, store=None) -> "DataSource":
        """Enforce exactly-one, then parse the regime. Argparse enforces it too; so does this.

        Both, deliberately: the mutually-exclusive group catches the CLI, and this catches every
        programmatic caller of `train_and_eval` — including the ones that used to be able to omit
        `h5_path` only by crashing three frames deeper.
        """
        if (h5 is None) == (store is None):
            raise ValueError(
                "exactly one of h5 / store is required. --h5 <baked.h5> reads the frozen bake; "
                "--store <regime.json> reads a CANDI_STORE through a regime file. "
                f"Got h5={h5!r} store={store!r} — neither has a default, and they are not "
                "combinable: the regime would then contradict the h5's own attrs.")
        if h5 is not None:
            return cls(kind="h5", h5_path=str(h5))
        from candi.store.regime import Regime
        return cls(kind="store", regime_path=str(store), regime=Regime.from_file(store))

    @classmethod
    def coerce(cls, obj) -> "DataSource":
        """A `DataSource` passes through; anything else is read as an h5 path.

        This is what keeps every pre-t23 caller of `train(model, h5_path, device, …)` working
        unchanged, including the ones in `tests/`.
        """
        return obj if isinstance(obj, cls) else cls(kind="h5", h5_path=str(obj))

    @property
    def is_store(self) -> bool:
        return self.kind == "store"

    def label(self) -> str:
        return self.regime_path if self.is_store else str(self.h5_path)

    def eval_pairs_declared(self) -> bool:
        """Can this source supply imputation targets to score against?

        On the store the pairing is DECLARED, in the regime's `eval_pairs` (D31), because D16 makes
        biosample names opaque ids that nothing may parse. With none declared the eval split is a
        plain held-out split with no imputation targets in it, and a mid-training eval would have
        nothing to select on. The h5 derives its pairing by string surgery on the `T_`/`V_`/`B_`
        prefixes, so it always has one.
        """
        return True if not self.is_store else bool(self.regime.eval_pairs)

    def provenance(self) -> dict:
        """What the run record keeps about the data. On the store path, STORE_PLAN.md §4 verbatim.

        The h5 path adds exactly ONE key to what it recorded before t23 (`data_source`), and keeps
        `h5` spelled the way every existing run json spells it. A reader of an old json sees the same
        field; a reader of a new one can tell the two paths apart without inferring it from which
        key is absent.
        """
        if not self.is_store:
            return {"data_source": "h5", "h5": str(self.h5_path)}
        return {
            "data_source": "store",
            "h5": None,
            "store": str(self.regime.store),
            "regime_file": self.regime_path,
            # The regime's sha256 is over the BYTES ON DISK, and `regime_json` is those bytes. A
            # hash with no text beside it cannot be checked by a reader who no longer has the file.
            "regime_sha256": self.regime.sha256,
            "regime_json": self.regime.raw,
            "store_manifest_sha256": _manifest_sha256(self.regime.store),
        }


def make_dataset(source, mask_regime, *, train=True, batch_size=8, dsf_sampling="uniform",
                 seed=42, shuffle=True, h5_cache_ram=True, cell_cond="off", reference=None,
                 only=None):
    """THE one place a training dataset is opened, on either path. All three call sites go here.

    On the h5 path this is `CandiKitH5Dataset(...)` with the arguments the three sites already
    passed, at the values they already passed them — the defaults here are that class's own
    defaults, so a site that used to omit an argument still gets the same object.

    `only` is the single knob the two paths spell differently. The h5 loader selects biosamples by
    NAME PREFIX (`_bios_candidates` uses `startswith`), so the full-coverage loop passes a whole
    biosample name as a "prefix" and gets exactly that one; the store selects by exact name (D16
    makes the names opaque ids that nothing may parse). `only=None` means "the split's whole pool"
    on both.
    """
    if not source.is_store:
        return CandiKitH5Dataset(source.h5_path, mask_regime, train=train, batch_size=batch_size,
                                 biosample_prefix=(only or "T_"), dsf_sampling=dsf_sampling,
                                 seed=seed, shuffle=shuffle, h5_cache_ram=h5_cache_ram,
                                 cell_cond=cell_cond, reference=reference)
    from candi.store.dataset import StoreDataset
    if mask_regime != "type1":
        raise ValueError(
            f"--regime {mask_regime!r} is h5-only. The store plans its own windows from the regime "
            "file's `window_plan`, and it has no cCRE annotation to tile by (every store window is "
            "a tile — `store/dataset.py::REGION_TILE`). Use --regime type1 with --store, or change "
            "the window plan in the regime file.")
    if reference is not None:
        raise ValueError(
            "--reference on is h5-only: `candi.reference.ReferenceTable` is built from a baked h5 "
            "and pins itself to that h5's fingerprint. A store-backed reference is not a flag.")
    return StoreDataset(source.regime, train=train, batch_size=batch_size,
                        dsf_sampling=dsf_sampling, seed=seed, shuffle=shuffle,
                        cell_cond=cell_cond,
                        biosamples=([str(only)] if only else None))


# ---------------------------------------------------------------------------
# loss + schedule
# ---------------------------------------------------------------------------

def _elem_nb_nll(p, n, target, eps: float = 1e-6):
    # FENCED IN fp32: this is the objective, and it is a log-likelihood, so its arithmetic is all
    # differences of large lgammas and logs of numbers near 0 and near 1 — the two places a short
    # mantissa cancels hardest. `1.0 - p` is the specific hazard: p is routinely within 1e-3 of 1.0,
    # bf16's ulp at 1.0 is 3.9e-3, so the subtraction would return exactly 0.0 for a large fraction
    # of positions and the clamp would replace the whole distribution's tail with a constant `eps`.
    # The gradient of a constant is 0, so those positions would silently stop supervising the model.
    #
    # The fence is inside the loss rather than at the training-loop call site so that every caller
    # gets it — `healthcheck` calls `nb_count_loss` directly, and so will the next reader.
    with fp32_fence(p, n, target) as (p, n, target):
        probs = (1.0 - p).clamp(eps, 1.0 - eps)
        total = n.clamp_min(eps)
        dist = torch.distributions.NegativeBinomial(total_count=total, probs=probs)
        return -dist.log_prob(target.clamp_min(0.0))       # [B, L, A]


def nb_count_loss(out, prep, imp_weight: float = 1.0):
    """Masked NB NLL over available assays, split obs (unmasked) + imp (cloze) — counts only.

    Two h74 changes on top of the h73 form, both applying to EVERY arm so neither can explain a
    contrast:

    `imp_weight` — evaluation scores imputation only, while the h73 objective weighted denoising and
    imputation equally and the denoising term dominated the observed loss curve. 3:1 by PI ruling. The
    sum is left UNNORMALIZED (so `1*obs + 3*imp`, not a convex combination): the point is to move the
    gradient direction, and a normalization would partly undo the reweighting by shrinking the step.
    `grad/total_preclip` is logged so the interaction with the 1.0 clip stays visible.

    `imp` is OMITTED from the returned terms when no position is masked, rather than logged as 0.0.
    In h73 30.9% of logged `imp` values were structural zeros from batches with nothing to impute,
    averaged into the curve as if they were measured likelihoods. The loss still gets a real
    zero-gradient term there; only the LOGGING changes. `imp_n` rides alongside so a reader can always
    tell how much supervision produced the number.
    """
    elem = _elem_nb_nll(out["p"], out["n"], prep["y_data"])
    obs, msk = prep["observed_map"], prep["masked_map"]
    n_obs, n_imp = int(obs.sum()), int(msk.sum())
    lo = elem[obs].mean() if n_obs else out["p"].sum() * 0.0
    li = elem[msk].mean() if n_imp else out["p"].sum() * 0.0
    terms = dict(obs=float(lo), obs_n=n_obs, imp_n=n_imp)
    if n_imp:
        terms["imp"] = float(li)
    return lo + imp_weight * li, terms


# ---------------------------------------------------------------------------
# auxiliary heads (Stage 2) — Gaussian signal + Bernoulli peak
# ---------------------------------------------------------------------------
# These ride ALONGSIDE the NB count loss and never modify it. A run at the default `--heads count`
# builds no optional head, so `aux_head_losses` returns `None` and `_train_step` never touches the
# loss expression at all — not "adds a zero", which would still put a new node in the autograd graph.
#
# WHY THE SIGNAL MAPS AND NOT `observed_map` / `masked_map`. `y_pval` and `y_peaks` are FULL-DEPTH
# quantities: the bake writes one `pval` and one `peaks` array per biosample, with no per-DSF variant,
# while the counts carry a `counts_dsf{1,2,4,8}` ladder. So on a step whose target DSF is 4, the model
# is prompted with a quarter-depth target and asked for that track's counts — and supervising its
# p-value head against the FULL-depth p-value on the same step would train it to contradict its own
# prompt. `batch.prepare_masked_batch` already computes `signal_observed_map` / `signal_masked_map` as
# exactly `obs/masked AND available AND y_dsf == 1` for this reason; until now nothing consumed them.
#
# THOSE MAPS ARE ALSO THE SENTINEL GUARD, and this is the part that would be silent if it were wrong.
# `y_pval` and `y_peaks` are NOT zero-filled for unavailable assays — the bake writes MISSING = -1
# into every column the biosample does not have (`prep/handler.py::make_bios_tensor_BW`,
# `make_bios_tensor_Peaks`, both `missing_value=-1`), and the loader copies the whole `[L, F]` slab
# through. A Gaussian mean is a softplus and can never reach -1, so an unmasked loss would be
# dominated by columns that carry no data; BCE against -1 is not even defined. The maps require
# `y_avail > 0`, which drops those columns entirely.
#
# `obs` / `imp` here mean UNMASKED / MASKED, as everywhere else in this package.



def _elem_gaussian_nll(mu, var, target):
    """Element-wise Gaussian NLL, `full=True` so the 0.5*log(2*pi) constant is included.

    `full=True` matches production (`torch.nn.GaussianNLLLoss(reduction="none", full=True)`), which
    matters because the constant is what makes the number a comparable log-likelihood rather than an
    unnormalised score — the same reason the NB term uses `torch.distributions`' own `log_prob`.
    """
    return torch.nn.functional.gaussian_nll_loss(mu, target, var, full=True, reduction="none")


def _elem_peak_bce(logit, target):
    """Element-wise BCE from a LOGIT — `PeakHead` no longer applies the sigmoid.

    NO CLAMP, AND THE ABSENCE IS THE POINT. `binary_cross_entropy_with_logits` fuses the sigmoid into
    the loss and evaluates it through the log-sum-exp identity, so it never forms `1 - p` and has
    nothing to saturate. The previous probability-input version needed `clamp(eps, 1-eps)` to keep
    `log(0)` out of the batch — and under bf16 that clamp was a NO-OP, because `1 - 1e-6` rounds to
    exactly 1.0 and the bound landed on the value it was excluding.

    What that cost was NOT a crash. `binary_cross_entropy` clamps its own log at -100, so the loss
    merely froze at 13.8023 for every logit >= 8 and the gradient went to EXACTLY ZERO — measured.
    The head quietly stopped learning from its most confident errors while the loss curve stayed
    plausible. Re-adding a clamp here would be reintroducing that, wearing the look of a safety
    measure.

    THE TARGET CHECK IS EXPLICIT BECAUSE THE KERNEL NO LONGER DOES IT. `binary_cross_entropy` rejects
    a target outside [0, 1]; `binary_cross_entropy_with_logits` does NOT — it accepts -1 and returns
    a finite, plausible number (1.4741 at logit 0.5). Peaks are 0/1 where available and MISSING = -1
    where the assay is absent, so a -1 arriving here means the availability mask failed, and without
    this check the head would train against assays the biosample does not have while every loss curve
    looked healthy. Switching to logits removed a safety property that torch was providing for free;
    this restores it deliberately rather than losing it silently.

    Callers must still select with the availability mask BEFORE calling this — `aux_head_losses` does
    — so in normal operation the check never fires and costs one reduction per optimizer step.
    """
    if torch.is_tensor(target) and target.numel() and bool(((target < 0) | (target > 1)).any()):
        raise ValueError(
            "peak BCE received a target outside [0, 1] — almost certainly the MISSING = -1 sentinel "
            "for an assay this biosample does not have. The availability mask must be applied before "
            "the elementwise loss, not after it; see `aux_head_losses`.")
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")


#: D30 — how the Gaussian signal head's TARGET is transformed, applied inside the loss.
#: `signal_transform` in `EncoderConfig` is a different thing entirely: that one transforms the
#: encoder's INPUT counts. `decoder.py::GaussianSignalHead` records why the two are easy to
#: conflate. Nothing here touches the loader — `y_pval` arrives in raw `-log10 p` space from both
#: data paths and this is the only place it is bent.
SIGNAL_TARGET_TRANSFORMS = ("none", "arcsinh", "log1p")

#: The default is DERIVED FROM THE DATA SOURCE, not a single constant, because the two paths do not
#: hand over the same quantity. The bake pre-transforms (`prep/handler.py` loads the bigwig with
#: `arcsinh=True`), so applying one again would square the compression; the store (STORE_PLAN.md D9,
#: as amended by PVAL_CODEC_PLAN.md D26) returns raw `-log10 p`, so the head needs one. This is what
#: makes the h5 path bit-identical to its pre-t26 self: `none` is the identity.
SIGNAL_TARGET_TRANSFORM_BY_SOURCE = {"h5": "none", "store": "arcsinh"}

#: What `--signal-target-transform` defaults to. `auto` resolves through the table above; any other
#: value overrides it and is recorded verbatim in the run config.
SIGNAL_TARGET_TRANSFORM_AUTO = "auto"


def resolve_signal_target_transform(source, requested: str = SIGNAL_TARGET_TRANSFORM_AUTO) -> str:
    """`auto` -> the source's default; anything else -> itself, validated.

    The RESOLVED value is what goes in the run config. A run whose target space has to be inferred
    from which data flag was passed is a run that can be compared against one it does not belong
    beside — the same rule `EVAL.md` already states for `--heads`.
    """
    if requested != SIGNAL_TARGET_TRANSFORM_AUTO and requested not in SIGNAL_TARGET_TRANSFORMS:
        raise ValueError(
            f"signal_target_transform must be {SIGNAL_TARGET_TRANSFORM_AUTO!r} or one of "
            f"{SIGNAL_TARGET_TRANSFORMS}; got {requested!r}")
    if requested != SIGNAL_TARGET_TRANSFORM_AUTO:
        return requested
    kind = DataSource.coerce(source).kind
    try:
        return SIGNAL_TARGET_TRANSFORM_BY_SOURCE[kind]
    except KeyError:  # pragma: no cover - DataSource.resolve already fences the kind
        raise ValueError(f"no default signal target transform for data source kind {kind!r}")


def _apply_signal_target_transform(target, mode: str):
    """Bend `y_pval` into the space the signal head predicts in. `none` returns the tensor itself.

    `none` returns the ARGUMENT, not a clone: the pre-t26 loss expression evaluated `y` directly, and
    an identity op inserted here would be a different autograd graph on a path that is supposed to be
    unchanged. `y_pval` is a target and carries no grad, so there is nothing to alias badly.

    Both transforms are monotone and defined at 0, which `-log10 p` reaches constantly, and both keep
    a non-negative input non-negative — the property `GaussianSignalHead`'s softplus mean relies on.
    """
    if mode == "none":
        return target
    if mode == "arcsinh":
        return torch.arcsinh(target)
    if mode == "log1p":
        return torch.log1p(target)
    raise ValueError(
        f"signal_target_transform must be one of {SIGNAL_TARGET_TRANSFORMS}; got {mode!r}")


def aux_head_losses(out, prep, imp_weight: float = 1.0,
                    signal_target_transform: str = "none"):
    """`(extra_loss, terms)` for whichever optional heads the model built, or `(None, {})` for none.

    THE SWITCH IS THE PRESENCE OF THE OUTPUT KEY, not a flag threaded down from `main`. A model built
    at `--heads count` emits no `signal_mu`, so there is nothing here to weigh and nothing is added.
    This is the same contract `forward_full` uses for `log_ref` — a dict without the key IS the
    earlier model — and it keeps the head set in exactly one place, the constructor that built it.

    Each term carries the SAME `imp_weight` the count loss uses. At the shipped 1.0 that is an
    identity, and above it the reweighting stays a property of the OBJECTIVE rather than of one head:
    an arm that emphasised imputation for counts and not for signal would differ from its control in
    two ways. This is a reasoned choice, not a measured one — no run has yet varied it.

    The two heads are summed into the count loss with COEFFICIENT 1.0. Nobody has measured what the
    right relative weight is, and inventing one here would bury an unmeasured constant inside a
    "no-op" stage; 1.0 is the honest placeholder and the obvious next knob.

    `signal_target_transform` (D30) bends `y_pval` — and ONLY the signal head's target — before the
    Gaussian NLL. It defaults to `"none"`, the identity, so every caller written before t26 gets the
    pre-t26 arithmetic; `train_and_eval` resolves the real value from the data source and passes it.
    The peak head is untouched: `y_peaks` is 0/1 and there is no space to move it into.
    """
    has_signal = "signal_mu" in out
    has_peak = "peak_logit" in out
    if not (has_signal or has_peak):
        return None, {}

    obs, msk = prep["signal_observed_map"], prep["signal_masked_map"]
    n_obs, n_imp = int(obs.sum()), int(msk.sum())
    total = None
    # Reported once, not per head: both heads read the same two maps, so a second copy of these
    # counts under a `peak_` prefix would invite a reader to look for a difference that cannot exist.
    terms = dict(sig_obs_n=n_obs, sig_imp_n=n_imp)

    def _split(elem_of, prefix, anchor):
        """obs + imp_weight*imp over the signal maps, logged the way `nb_count_loss` logs its own.

        `elem_of(mask)` SELECTS AND THEN COMPUTES, which is not the order `nb_count_loss` uses and is
        deliberate. The count loss evaluates its NLL over the whole tensor and indexes afterwards,
        which is safe because `y_data` is raw counts everywhere. `y_peaks` is not: unavailable assays
        hold -1, and `binary_cross_entropy` VALIDATES that its target lies in [0, 1] — it would raise
        on the sentinel before the mask ever got a chance to drop it. Selecting first means the
        sentinel never reaches the kernel, and it is also strictly less arithmetic.

        `anchor` is any tensor from the head, used to make a real zero when a map is empty: it keeps
        the head's parameters attached to the graph so `backward()` still reaches them, which a python
        `0.0` would not.
        """
        nonlocal total
        lo = elem_of(obs).mean() if n_obs else anchor.sum() * 0.0
        li = elem_of(msk).mean() if n_imp else anchor.sum() * 0.0
        terms[f"{prefix}_obs"] = float(lo)
        # Absent, not zero, when nothing was masked — the same rule as the count loss. A structural
        # zero averaged into the curve reads as a confidently well-fit batch.
        if n_imp:
            terms[f"{prefix}_imp"] = float(li)
        part = lo + imp_weight * li
        total = part if total is None else total + part

    if has_signal:
        mu, var = out["signal_mu"], out["signal_var"]
        # D30 — the ONE place `y_pval` changes space. The loader hands over raw `-log10 p` from both
        # data paths and never transforms it; see `SIGNAL_TARGET_TRANSFORMS` above.
        y = _apply_signal_target_transform(prep["y_pval"], signal_target_transform)
        _split(lambda m: _elem_gaussian_nll(mu[m], var[m], y[m]), "signal", mu)
    if has_peak:
        logit, y = out["peak_logit"], prep["y_peaks"]
        _split(lambda m: _elem_peak_bce(logit[m], y[m]), "peak", logit)
    return total, terms


LR_SCHEDULES = ("cosine", "linear", "constant")


def make_lr_schedule(opt, total_steps, schedule="cosine", warmup_frac=0.1, min_ratio=0.1):
    """Linear warmup, then decay to `min_ratio` of the peak LR.

    `cosine` at the default 0.1 / 0.1 is the frozen historical schedule, character for character —
    every recorded run used it, and it is what `tests/test_flags.py` pins. `linear` decays on a
    straight line to the same floor; `constant` holds the peak after warmup, which is the right
    control when the question is whether the decay itself is doing the work.
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}; got {schedule!r}")
    warm = max(1, int(warmup_frac * total_steps))

    def fn(s):
        if s < warm:
            return s / warm
        t = (s - warm) / max(1, total_steps - warm)
        if schedule == "cosine":
            return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))
        if schedule == "linear":
            return min_ratio + (1 - min_ratio) * (1 - t)
        return 1.0                                   # constant: hold the peak after warmup

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def _t_biosamples(h5_path) -> list:
    import json
    import h5py
    with h5py.File(h5_path, "r") as h:
        order = json.loads(h["biosamples"].attrs["order"])
    return [b for b in order if b.startswith("T_")]


def _coverage_biosamples(source) -> list:
    """The biosamples `--full-coverage` iterates, on either path.

    The h5 answers "every `T_` in the bake"; the store answers "the regime's declared train split",
    because D16 forbids reading meaning out of a store biosample name — `T_` is not a thing the
    store knows. A regime that declares no train split has not said what to iterate, and guessing
    "all of them" would put eval biosamples into training.
    """
    if not source.is_store:
        return _t_biosamples(source.h5_path)
    names = list(source.regime.biosamples("train"))
    if not names:
        raise ValueError(
            f"{source.regime_path}: --full-coverage needs `biosamples.train` to name the biosamples "
            "to cover. The store has no `T_` convention to fall back on (D16), and covering every "
            "biosample in the corpus would train on the eval split.")
    return names


# Covariate-pathway parameter groups: field -> substring patterns matched against
# `model.named_parameters()`. Every pattern deliberately matches BOTH towers
# (`encoder.metadata_embedding.*` and `decoder.meta_embedding.*`), because the covariate pathway is
# the pair of them, not either one alone.
#   * `_embedding.fusion` matches `metadata_embedding.fusion` AND `meta_embedding.fusion`, and does
#     NOT match `encoder.fusion` (the signal+DNA tower fusion — a different module entirely).
#   * the four sentinel vectors are standalone `nn.Parameter`s, so they get their own rows; the two
#     categorical tables carry their MISSING/CLOZE entries as ordinary rows, so a whole-table norm
#     already includes them.
_META_FIELDS = {
    "depth_proj": ("depth_proj",),
    "assay_embedding": ("assay_embedding",),
    "read_length_proj": ("read_length_proj",),
    "runtype_embedding": ("runtype_embedding",),
    "depth_missing_emb": ("depth_missing_emb",),
    "depth_cloze_emb": ("depth_cloze_emb",),
    "readlen_missing_emb": ("readlen_missing_emb",),
    "readlen_cloze_emb": ("readlen_cloze_emb",),
    "meta_fusion": ("_embedding.fusion",),
    # candi_kit's single `decoder.film_proj` becomes many taps in candi — `decoder.film_layers.N`
    # and `encoder.*_blocks.N.film`. All three spellings are collected under one key so the aggregate
    # "how much gradient reaches the FiLM projections" number stays comparable across every arm; the
    # PER-TAP breakdown is in the `film/` block below, which is where a dead tap shows up.
    "film_proj": ("film_proj", "film_layers.", ".film.proj"),
    # reads 0.0 on the default 4-row model (no such parameter exists); kept so h73-era runs stay
    # comparable, no longer what this function is about.
    "cell_embedding": ("cell_embedding",),
}

# Block-level gradient flow for the candi arms. Each entry answers one diagnostic question.
# VALUE FORMAT: a list of ALTERNATIVES; each alternative is a tuple of substrings that must ALL
# appear in the parameter name. `enc_blocks` and `lane_attn` are not contiguous in
# `encoder.enc_blocks.0.lane_attn.…`, so the plain OR-matching `_META_FIELDS` uses would fold the
# position axis into the lane axis and hide exactly the failure this block exists to catch.
_BLOCK_FIELDS = {
    "enc_conv":       [("encoder.signal_tower",)],    # does the grouped signal tower still learn
    "enc_dna":        [("encoder.dna_tower",)],       # does the shared sequence trunk learn
    "fusion_w_sig":   [("fusion.w_sig",)],            # per-lane signal readout
    "fusion_w_dna":   [("fusion.w_dna",)],            # per-assay DNA readout — the broadcast itself
    "dna_prior_flag": [("fusion.miss_flag",)],        # "this lane is unobserved" flag (a4/a5 only)
    "mask_token":     [("mask_embedding",)],          # the constant it replaces (every other arm)
    # THE cross-assay mixer. If this reads ~0 while enc_attn_pos does not, the lane design has
    # collapsed to 35 independent per-assay models and no arm above a1b means what it claims.
    "enc_attn_lane":  [("enc_blocks", "lane_attn")],
    "enc_attn_pos":   [("enc_blocks", "pos_attn")],
    "dec_attn_lane":  [("dec_blocks", "lane_attn")],
    "dec_attn_pos":   [("dec_blocks", "pos_attn")],
    "dec_deconv":     [("decoder.blocks",)],          # the grouped deconv trunk
    "dec_input_proj": [("decoder.input_proj",), ("decoder.w_in",)],
    "dec_head":       [("head_eta",), ("head_n",)],
}


def _grad_norms(model) -> dict:
    """Per-FIELD gradient norms for the COVARIATE PATHWAY, read BETWEEN backward() and clip_grad_norm_.

    The four covariate rows (depth, assay_id, read_length, run_type) reach the trunk only through the
    metadata embedder and then through `film_proj`, so a covariate that cannot steer the model shows
    up here — as a per-field norm near zero, or as a `_over_trunk` ratio orders of magnitude below the
    decoder trunk — long before it shows up in the loss curve, which it never does. Each field is
    reported BOTH absolutely (`grad/<field>`) and relative to the trunk (`grad/<field>_over_trunk`),
    because the absolute scale moves with the loss and only the ratio is comparable across steps.

    `meta_embed/<field>_absmax` is the absmax of the field's PARAMETERS (not their gradients),
    sentinel rows and vectors included. A field whose parameters never move away from init is a
    different failure from one that receives no gradient, and only the pair distinguishes them.

    Pure logging: nothing here reads or writes anything the model computes.
    """
    def n(*pats):
        return math.sqrt(sum(float(p.grad.pow(2).sum()) for k, p in model.named_parameters()
                             if any(q in k for q in pats) and p.grad is not None))

    def amax(*pats):
        vals = [float(p.detach().abs().max()) for k, p in model.named_parameters()
                if any(q in k for q in pats) and p.numel()]
        return max(vals) if vals else 0.0

    trunk = max(n("decoder.trunk", "decoder.blocks"), n("encoder.signal_tower"), 1e-30)
    out = {"grad/trunk": trunk,
           "grad/encoder_transformer": n("encoder.transformer", "encoder.enc_blocks"),
           "grad/decoder_film_proj": n("decoder.film_proj", "decoder.film_layers")}

    # ---- candi block-level flow. Reads 0.0 on any arm that does not build the block, so one
    # dashboard serves the whole ladder and a flat curve means "not in this arm", not "broken".
    def n_alts(alts):
        """sqrt of the summed squared grads over params matching ANY alternative (ALL substrings)."""
        tot = 0.0
        for k, p in model.named_parameters():
            if p.grad is None:
                continue
            if any(all(q in k for q in alt) for alt in alts):
                tot += float(p.grad.pow(2).sum())
        return math.sqrt(tot)

    for key, alts in _BLOCK_FIELDS.items():
        g = n_alts(alts)
        out[f"grad/{key}"] = g
        out[f"grad/{key}_over_trunk"] = g / trunk

    # ---- per-tap FiLM, absolute and as a share of the trunk.
    # A tap whose gradient is ~0 is not conditioning anything, and because every FiLM projection here
    # is adaLN-ZERO the failure is silent: a tap that never leaves its zero init produces gamma=beta=0
    # forever, i.e. an exact identity, and the loss curve looks entirely normal. `film/*_absmax` is
    # the parameter absmax, so `grad ~ 0 AND absmax ~ 0` = never switched on, while `grad ~ 0 AND
    # absmax > 0` = converged. Only the pair separates them.
    for name, p in model.named_parameters():
        if ".film_layers." in name or ".film.proj." in name:
            if not name.endswith(".weight"):
                continue
            tap = name.rsplit(".proj.", 1)[0].replace("encoder.", "enc_").replace("decoder.", "dec_")
            out[f"film/{tap}_grad"] = float(p.grad.pow(2).sum().sqrt()) if p.grad is not None else 0.0
            out[f"film/{tap}_absmax"] = float(p.detach().abs().max())

    # ---- "is anything being killed": share of parameter TENSORS with no gradient signal at all.
    # Distinct from the norms above, which average a dead block away inside a live one.
    tot = dead = 0
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        tot += 1
        if p.grad is None or float(p.grad.abs().max()) < 1e-12:
            dead += 1
    out["grad/dead_tensor_frac"] = dead / max(tot, 1)

    for field, pats in _META_FIELDS.items():
        g = n(*pats)
        out[f"grad/{field}"] = g
        out[f"grad/{field}_over_trunk"] = g / trunk
        out[f"meta_embed/{field}_absmax"] = amax(*pats)
    # historical alias, kept verbatim so h73/h74 dashboards keep resolving
    out["grad/cell_over_trunk"] = out["grad/cell_embedding_over_trunk"]
    return out


CLIP_NORM = 1.0            # the frozen default; `--clip-norm` overrides it per run


def _train_step(model, prep, opt, sched, losses, *, want_grads: bool = False,
                imp_weight: float = 1.0, probe=None, clip_state=None,
                clip_norm: float = CLIP_NORM, precision=DEFAULT_PRECISION, scaler=None,
                signal_target_transform: str = "none"):
    # NO GradScaler, ON PURPOSE, and this call is what keeps it that way. See
    # `precision.assert_no_grad_scaler`: bf16 has fp32's exponent range so there is nothing to
    # rescue, and a scaler added quietly would multiply every number `_grad_norms` reads below by a
    # drifting scale factor. The parameter exists only so that adding one is a loud failure here
    # rather than a silent corruption twelve lines down.
    assert_no_grad_scaler(scaler)
    # The probe (when present) captures exactly this forward pass and nothing else. Its hooks return
    # on their first statement while unarmed, so the 24 of every 25 steps that do not log pay nothing,
    # and a hook that returns None cannot alter the module output — the arithmetic is unchanged.
    if probe is not None and want_grads:
        probe.arm()
    # The autocast region covers the FORWARD ONLY. `backward` must not be inside it — autograd
    # replays each op in the dtype its forward ran in, so wrapping the backward adds nothing and the
    # PyTorch AMP recipe says so explicitly. The loss is outside it as well, and additionally fences
    # itself; at `--precision fp32` this context is `enabled=False` and changes no arithmetic, which
    # is what `tools/golden.py` and `test_defaults_are_a_no_op` assert.
    #
    # THE DEVICE IS READ OFF THE MODEL, not threaded in from the caller. A device string passed down
    # can disagree with where the model actually lives, and `autocast('cpu')` wrapped around a CUDA
    # model is a SILENT no-op: every op stays fp32, the run finishes normally, and the run JSON
    # records `precision=bf16` for a run that was nothing of the kind. There is exactly one correct
    # answer to "which device is this forward on" and the model already holds it.
    with autocast_region(next(model.parameters()).device, precision):
        out = forward_full(model, prep)
    loss, terms = nb_count_loss(out, prep, imp_weight=imp_weight)
    # Stage 2: auxiliary heads, added only when the model built one. `aux is None` on the default
    # `--heads count` path, so `loss` is the object `nb_count_loss` returned and the autograd graph is
    # the pre-Stage-2 one — not the same graph plus a zero, which would still be a different graph.
    aux, aux_terms = aux_head_losses(out, prep, imp_weight=imp_weight,
                                     signal_target_transform=signal_target_transform)
    if aux is not None:
        loss = loss + aux
        terms = {**terms, **aux_terms}
    opt.zero_grad()
    loss.backward()
    # grad norms are read HERE — after backward, before clipping — or clip_grad_norm_ rescales them
    # and the reported ratio becomes a property of the clip threshold rather than of the model. The
    # same argument is why no scaler may be in play at this point without an `unscale_` before it.
    if want_grads:
        terms = {**terms, **_grad_norms(model)}
        if probe is not None:
            terms.update(probe.read())
    # clip_grad_norm_ RETURNS the pre-clip total norm, which is the only way to see whether the 1.0
    # threshold binds — and with imp_weight > 1 raising the gradient scale, whether it now binds more
    # often than it did in h73.
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if clip_state is not None:
        # Accumulated ON DEVICE as a tensor. `float(total)` would be a host synchronisation on every
        # step; comparing and summing in torch keeps the hot path free of one, and the single `.item()`
        # happens on logging steps, which already synchronise.
        hit = (total > clip_norm).float()
        clip_state["binds"] = hit if clip_state["binds"] is None else clip_state["binds"] + hit
        clip_state["steps"] += 1
    if want_grads:
        terms["grad/total_preclip"] = float(total)
        if clip_state is not None and clip_state["steps"]:
            terms["train/clip_bind_frac"] = float(clip_state["binds"]) / clip_state["steps"]
    opt.step()
    sched.step()
    losses.append(float(loss))
    return terms


def train(model, source, device, *, regime="type1", epochs=25, steps_per_epoch=200,
          batch_size=8, lr=5e-4, weight_decay=0.0, seed=0, dsf_sampling="uniform",
          log_every=0, full_coverage=False, p_full_assay=1.0, mask_fraction=0.2,
          cell_cond="off", log_fn=None, reference=None, imp_weight=1.0, unmask_frac=0.0,
          eval_hook=None, eval_every=3, terms_out=None,
          optimizer="adam", trunk_wd=0.0, meta_probe=None,
          lr_schedule="cosine", warmup_frac=0.1, lr_min_ratio=0.1,
          clip_norm=CLIP_NORM, precision=DEFAULT_PRECISION,
          signal_target_transform="none") -> list:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")
    # t23: `source` is a `DataSource`. A bare path still works and still means the h5 — every caller
    # written before t23, this package's own and the tests', passes one positionally.
    source = DataSource.coerce(source)
    # D30. `"none"` rather than `"auto"` is the default HERE, on purpose: `train` is the low-level
    # entry point every test and probe calls directly, and a default that reads the data source
    # would change what those callers descend. `train_and_eval` is where `auto` is resolved.
    if signal_target_transform not in SIGNAL_TARGET_TRANSFORMS:
        raise ValueError(f"signal_target_transform must be one of {SIGNAL_TARGET_TRANSFORMS}; "
                         f"got {signal_target_transform!r}")
    if p_full_assay == 1.0 and mask_fraction != 0.2:
        print("[train] WARNING: --mask-fraction is INERT under --p-full-assay 1.0 "
              "(DataMasker._mask_full_assay never reads it)", flush=True)
    masker = make_masker(p_full_assay=p_full_assay, mask_fraction=mask_fraction,
                         p_full_loci=0.0, p_chunks=0.0)   # whole-assay cloze -> imputation supervision
    # --- h51: ADDITIVE, DEFAULT-OFF -----------------------------------------------------------------
    # The `adam` branch is the historical line VERBATIM — same class, same positional lr, same
    # coupled-L2 `weight_decay`, one implicit parameter group. Nothing about it is rewritten "for
    # symmetry" with the adamw branch, because a rewrite is exactly how a frozen path stops being
    # frozen. The `trunk_wd` guard costs nothing at the default 0.0 and refuses a run that would
    # RECORD a decay it never applied.
    if optimizer == "adam":
        if float(trunk_wd) != 0.0:
            raise ValueError(
                f"trunk_wd={trunk_wd} is inert under optimizer='adam' (the single-group path never "
                "reads it); use optimizer='adamw' for the h51 split.")
        opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
        print(f"[train] optimizer=adam weight_decay={weight_decay} trunk_wd=0.0 | ONE parameter "
              f"group: {sum(1 for _ in model.parameters())} tensors / "
              f"{sum(p.numel() for p in model.parameters()):,} params (h51 partition NOT used)",
              flush=True)
    elif optimizer == "adamw":
        # imported HERE, not at module scope, so the default path does not even load the partition
        from candi.param_groups import build_adamw
        opt = build_adamw(model, lr=lr, weight_decay=weight_decay, trunk_wd=trunk_wd)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'adamw'; got {optimizer!r}")
    losses, step = [], 0

    # h74 fix 3: a fraction of steps run FULLY UNMASKED. The encoder was trained exclusively on inputs
    # containing CLOZE tokens and evaluated exclusively on inputs with none, so eval sat off the
    # training input distribution. This is a PARTIAL fix by explicit ruling — it corrects the encoder's
    # input distribution, not the decoder prompt structure (eval encodes one biosample and prompts with
    # another's metadata, a regime training still never presents).
    #
    # The coin comes from a DEDICATED generator, never the shared data stream. Drawing it from the
    # loop's `rng` would advance that stream once per step, so two arms would visit different
    # biosamples and DSF levels and would differ in the DATA they saw, not only in the offset. This is
    # the same failure `_batch_cell_id` guards against, and it is exactly the bug caught in h73's build.
    unmask_rng = np.random.default_rng(seed ^ 0xC10A_5EED)
    n_unmasked = 0

    def _mask_this_step() -> bool:
        return float(unmask_rng.random()) >= unmask_frac

    # --- per-step diagnostics (logging only; none of this is read by the model) --------------------
    # The metadata-pathway probe is built ONLY when logging is on, so a run without W&B is untouched:
    # no hooks are attached at all, not merely inert ones.
    probe = None
    if log_fn is not None:
        from candi.probes import make_meta_path_probe
        probe = make_meta_path_probe(model)
    clip_state = {"binds": None, "steps": 0}
    pace = {"t": time.time(), "n": 0}

    def _pace() -> dict:
        """Seconds per step SINCE THE LAST LOG, not since the run started: a cumulative average hides
        the slowdown that matters (a run that degrades halfway through reads fine to the end)."""
        now = time.time()
        out = {"train/sec_per_step": (now - pace["t"]) / max(1, pace["n"])}
        pace["t"], pace["n"] = now, 0
        return out

    def _record(step, ep, terms):
        """Keep the obs/imp split IN THE RUN JSON, at the same 25-step cadence as the grad norms.

        h73's training-side forensics had to be reconstructed through the W&B API because the json
        carried only the summed loss, and the obs/imp split is the whole question V4 asks. `imp` is
        absent on unmasked steps by design, so the reader records what was measured rather than
        substituting a zero.
        """
        if terms_out is None or step % 25:
            return
        terms_out.append(dict(step=step, epoch=ep, obs=terms["obs"], imp=terms.get("imp"),
                              imp_n=terms["imp_n"], obs_n=terms["obs_n"], nll=losses[-1]))

    if full_coverage:
        # DETERMINISTIC full coverage: every epoch iterates ALL train windows for ALL T_ biosamples
        # (no random-biosample sampling). One per-biosample dataset each, sharing ONE RAM buffer, pulled
        # round-robin so biosamples interleave within the epoch.
        bios = _coverage_biosamples(source)
        # h5 only: one RAM copy of the file, shared by every per-biosample dataset. The store reads
        # whole chromosomes out of many files through its own handle pool and has nothing to share.
        shared = None if source.is_store else Path(source.h5_path).read_bytes()
        datasets = []
        for b in bios:
            ds = make_dataset(source, regime, train=True, batch_size=batch_size, only=b,
                              dsf_sampling=dsf_sampling, seed=seed, shuffle=True,
                              h5_cache_ram=False, cell_cond=cell_cond, reference=reference)
            if shared is not None:
                ds._ram_buf = shared                 # share the single 1.6 GB buffer (no duplication)
            datasets.append(ds)
        per_epoch = sum(d.estimate_steps_per_epoch() for d in datasets)
        sched = make_lr_schedule(opt, max(1, epochs * per_epoch), lr_schedule,
                                 warmup_frac, lr_min_ratio)
        # `T_` is the h5's own naming convention and is kept verbatim there; the store has no such
        # convention (D16), so it says what it actually iterated.
        what = "regime train-split" if source.is_store else "T_"
        print(f"[train] full-coverage: {len(bios)} {what} biosamples x all train windows "
              f"= ~{per_epoch} batches/epoch x {epochs} epochs", flush=True)
        for ep in range(epochs):
            model.train()
            iters = []
            for ds in datasets:
                ds.seed = seed + ep                  # per-epoch reshuffle (per biosample)
                iters.append(iter(ds))
            active = list(range(len(iters)))
            while active:
                for i in list(active):
                    try:
                        batch = next(iters[i])
                    except StopIteration:
                        active.remove(i)
                        continue
                    do_mask = _mask_this_step()
                    n_unmasked += (not do_mask)
                    prep = prepare_masked_batch(batch, masker, device, apply_mask=do_mask)
                    if prep is None:
                        continue
                    # h64: the arm switch, on the ASSEMBLED batch. None at `off` — see the banner.
                    if meta_probe is not None:
                        meta_probe.apply(prep)
                    want = bool(log_fn) and (step % 25 == 0)
                    terms = _train_step(model, prep, opt, sched, losses, want_grads=want,
                                        imp_weight=imp_weight, probe=probe,
                                        clip_state=clip_state, clip_norm=clip_norm,
                                        precision=precision,
                                        signal_target_transform=signal_target_transform)
                    step += 1
                    pace["n"] += 1
                    if want:
                        terms.update(_pace())
                    _record(step, ep, terms)
                    if log_fn:
                        log_fn(step, dict(terms, **{"train/nll": losses[-1],
                                                    "train/lr": sched.get_last_lr()[0],
                                                    "train/epoch": ep,
                                                    "train/frac_unmasked": n_unmasked / step},
                                          **(meta_probe.step_metrics() if meta_probe is not None else {})))
                    if log_every and step % log_every == 0:
                        print(f"  ep{ep} step{step} nll={losses[-1]:.3f} (obs={terms['obs']:.3f} "
                              f"imp={terms.get('imp', float('nan')):.3f}) "
                              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if eval_hook and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
                # A hook that returns truthy is asking to stop — see `train_and_eval`'s patience.
                # A hook that returns None never stops anything, so every caller predating this
                # (the probes, the tests, `train()` driven directly) is unchanged.
                if eval_hook(step, ep):
                    break
        if probe is not None:
            probe.close()
        if meta_probe is not None:
            print(meta_probe.summary(), flush=True)
        return losses

    # sampled path: random train biosample per batch, steps_per_epoch batches/epoch
    ds = make_dataset(source, regime, train=True, batch_size=batch_size, only=None,
                      dsf_sampling=dsf_sampling, seed=seed, shuffle=True, cell_cond=cell_cond,
                      reference=reference)
    sched = make_lr_schedule(opt, epochs * steps_per_epoch, lr_schedule,
                             warmup_frac, lr_min_ratio)
    for ep in range(epochs):
        model.train()
        ds.seed = seed + ep                          # per-epoch variety (RAM buffer persists)
        it = iter(ds)
        for _ in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(ds)
                batch = next(it)
            do_mask = _mask_this_step()
            n_unmasked += (not do_mask)
            prep = prepare_masked_batch(batch, masker, device, apply_mask=do_mask)
            if prep is None:
                continue
            # h64: the arm switch, on the ASSEMBLED batch. None at `off` — see the banner.
            if meta_probe is not None:
                meta_probe.apply(prep)
            want = bool(log_fn) and (step % 25 == 0)
            terms = _train_step(model, prep, opt, sched, losses, want_grads=want,
                                imp_weight=imp_weight, probe=probe, clip_state=clip_state,
                                clip_norm=clip_norm, precision=precision,
                                signal_target_transform=signal_target_transform)
            step += 1
            pace["n"] += 1
            if want:
                terms.update(_pace())
            _record(step, ep, terms)
            if log_fn:
                log_fn(step, dict(terms, **{"train/nll": losses[-1],
                                            "train/lr": sched.get_last_lr()[0],
                                            "train/epoch": ep,
                                            "train/frac_unmasked": n_unmasked / step},
                                  **(meta_probe.step_metrics() if meta_probe is not None else {})))
            if log_every and step % log_every == 0:
                print(f"  ep{ep} step{step} nll={losses[-1]:.3f} (obs={terms['obs']:.3f} "
                      f"imp={terms.get('imp', float('nan')):.3f}) "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
        # end of epoch: mid-training eval + best-checkpoint selection (h74 fix 4)
        if eval_hook and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
            stop = eval_hook(step, ep)
            model.train()
            if stop:
                break
    if probe is not None:
        probe.close()
    if meta_probe is not None:
        print(meta_probe.summary(), flush=True)
    return losses


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
# NOTHING IN THIS MODULE SCORES A CHECKPOINT ANY MORE. `evaluate` came from `candi.eval`, which is
# deleted (D15); `python -m candi.bench` is the scorer and it takes either `--h5` or `--store`. What
# remains here is the mid-training dial (`candi.monitor`), which is a training-loop concern: it
# exists to select a checkpoint, not to produce a recorded number.


# ---------------------------------------------------------------------------
# json + main
# ---------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {(f"{k[0]}|{k[1]}|{k[2]}" if isinstance(k, tuple) else str(k)): _jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _wandb_init(project, run_name, config):
    """Return a live wandb run, or None. NEVER raises: a dead dashboard must not kill a 3-hour arm.

    Credentials come from `~/.netrc` (already present on Fir). Nothing here reads, prints or writes a
    key. `WANDB_MODE=offline` in the environment still works and syncs later with `wandb sync`.
    """
    if not project:
        return None
    try:
        import wandb
        run = wandb.init(project=project, name=run_name, config=config, reinit=True)
        print(f"[wandb] {run.url}", flush=True)
        return run
    except Exception as e:                                    # noqa: BLE001 — logging is never fatal
        print(f"[wandb] DISABLED: {type(e).__name__}: {e}", flush=True)
        return None


def _flat_scalars(d, prefix=""):
    """Flatten the eval result dict to `a/b/c -> float`, keeping only finite scalars.

    Everything else in the eval payload — the calibration grids, the per-target maps, the bootstrap
    cluster values — belongs in the run json, not in a metrics dashboard.
    """
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat_scalars(v, f"{key}/"))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            if math.isfinite(f):
                out[key] = f
    return out


def _wb_safe(what: str, fn):
    """Run one logging call and swallow whatever it raises, exactly like `_wandb_init`.

    A run that trained for seven hours must not die on its last line because a table failed to
    serialise. Every new W&B call below goes through here.
    """
    try:
        fn()
    except Exception as e:                                    # noqa: BLE001 — logging is never fatal
        print(f"[wandb] {what} FAILED (ignored): {type(e).__name__}: {e}", flush=True)


def _cell(v):
    """Coerce one eval-record field into something a `wandb.Table` column accepts."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)          # `target` tuples -> the "a|b|c" form the h5-era M1 used
    if v is None or isinstance(v, str):
        return v
    return str(v)


def _flat_record(r: dict) -> dict:
    """Expand one level of nested dicts into `parent.child` columns.

    The h5-era S14 per-target records carried `crps_matrix`, `min_at_true` and `beats_told1` as
    sub-dicts; left nested they would have landed in the table as one unreadable stringified blob
    per row.
    """
    out = {}
    for k, v in (r or {}).items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
        else:
            out[k] = v
    return out


def _wb_table(rows):
    """`list[dict]` -> a `wandb.Table` over the UNION of keys, first-seen column order."""
    import wandb
    rows = [_flat_record(r) for r in rows]
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    t = wandb.Table(columns=cols)
    for r in rows:
        t.add_data(*[_cell(r.get(c)) for c in cols])
    return t


def _rows_from_map(d, key_col: str = "target") -> list:
    """`{"T_x|V_x|assay": {...}}` (the h5-era M1 per-target form) -> `[{"target": "T_x|V_x|assay", ...}]`."""
    return [dict({key_col: k}, **v) for k, v in sorted((d or {}).items()) if isinstance(v, dict)]


def _h5_fingerprint_safe(h5_path):
    """The same cheap content fingerprint `reference.py` pins its tables with; None if unavailable.

    Reusing that mechanism is deliberate: it hashes the file size, every top-level attr and all of
    `meta_dsf1`, so two runs claiming the same h5 can be proven to have seen the same panel, the same
    biosample order and the same per-assay depths — without hashing 24 GB.
    """
    try:
        from candi.reference import h5_fingerprint
        return h5_fingerprint(h5_path)
    except Exception:                                         # noqa: BLE001 — provenance is never fatal
        return None


def _git_provenance() -> dict:
    """`{sha, dirty}` for the checkout this module was imported from; `unknown` if git is unavailable."""
    import subprocess
    here = str(Path(__file__).resolve().parent)
    out = {"git_sha": "unknown", "git_dirty": None}
    try:
        out["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                        text=True, timeout=30, check=True).stdout.strip()
        out["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here,
                                               capture_output=True, text=True, timeout=60,
                                               check=True).stdout.strip())
    except Exception:                                         # noqa: BLE001 — provenance is never fatal
        pass
    return out


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _g(d, *keys, default=float("nan")):
    for k in keys:
        if k in d:
            return d[k]
    return default


def train_and_eval(*, h5_path=None, out_dir, store=None, regime="type1", epochs=25,
                   steps_per_epoch=200, batch_size=8,
                   lr=5e-4, weight_decay=0.0, use_offset=True, dsf_sampling="uniform", device="cpu",
                   seed=0, embed_dim=32, dropout=0.1, n_transformer_layers=2,
                   depth_center=None, d_model=0, nhead=4, p_full_assay=1.0, mask_fraction=0.2,
                   eval_batch_size=4, eval_max_batches=None, fg_frac=0.02, n_boot=1000,
                   eval_budget=200_000, m3_regions=8, include_deprecated=False, log_every=0,
                   full_coverage=False, decoder_lane=8, deconv_norm="lane",
                   resolution=None, n_cnn_layers=3, conv_kernel_size=3, pool_size=2,
                   expansion_factor=2, n_deconv_layers=3, deconv_upsample=2, deconv_kernel_size=3,
                   conv_norm="layer", transformer_norm="layer", transformer_norm_placement="pre",
                   attn_qk_norm=False, film_taps=DEFAULT_FILM_TAPS,
                   film_init_encoder="xavier", film_init_decoder="zero",
                   head_sharing="shared", head_hidden=0, heads=DEFAULT_HEADS,
                   lr_schedule="cosine", warmup_frac=0.1, lr_min_ratio=0.1, clip_norm=CLIP_NORM,
                   ckpt_path=None, cell_cond="off",
                   wandb_project=None, wandb_run_name=None, reference="off", reference_path=None,
                   reference_pseudocount=None, imp_weight=1.0, unmask_frac=0.0,
                   eval_every=None, eval_batches_per_pair=4, early_stop_epochs=3,
                   eval_regions=None,
                   meta_embed_layernorm=True,
                   optimizer="adam", trunk_wd=0.0, meta_gain=1.0,
                   meta_probe="off", meta_probe_delta=DEFAULT_META_PROBE_DELTA,
                   precision=DEFAULT_PRECISION,
                   signal_target_transform=SIGNAL_TARGET_TRANSFORM_AUTO) -> dict:
    # Checked on the submit line, for the same reason as `meta_probe` below: a mistyped precision
    # must fail before the h5 is opened, not after an hour of training.
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")
    # h51 ambiguity guard, run BEFORE anything expensive: `adamw` + a non-zero `weight_decay` would
    # double-specify decay (a global coupled term on top of the per-group decoupled one) and the two
    # arms would stop differing in decay alone. One authoritative home for the rule; `train()` and
    # `build_adamw` re-check it for direct callers.
    from candi.param_groups import validate_optimizer_config
    optimizer = validate_optimizer_config(optimizer, weight_decay, trunk_wd)
    # h64, checked here for the same reason: a mistyped arm must fail on the submit line, not after
    # the h5 has been opened and the model built.
    if meta_probe not in META_PROBE_MODES:
        raise ValueError(f"meta_probe must be one of {META_PROBE_MODES}; got {meta_probe!r}")
    # t23: which data path, decided ONCE and here. Everything below asks `source`, never the flags.
    source = DataSource.resolve(h5=h5_path, store=store)
    # D30 — resolved ONCE, here, and from now on carried as a concrete value. `auto` never reaches
    # the loss: `auto` is a request, and what a run must record is the answer.
    signal_target_transform = resolve_signal_target_transform(source, signal_target_transform)
    # Resolved the same way and for the same reason: `None` is a request ("whatever suits this
    # source"), and what a run records must be the answer. An explicit value — including an
    # explicit 0 — always wins; unset means the settled cadence of 3 when the source has
    # imputation targets to score, and 0 when it does not. The guard below is UNCHANGED and still
    # fires on an explicit non-zero against a pair-less regime, which is the case worth a message.
    eval_every = resolve_eval_every(eval_every,
                                    eval_pairs_declared=source.eval_pairs_declared())
    if source.is_store:
        # Said before anything is opened, because each of these is a way a store-backed run could
        # look like a normal one in the run json while being something else.
        if reference != "off":
            raise ValueError(
                "--reference on is h5-only: the table is built from a baked h5 and pins itself to "
                "that h5's fingerprint (`candi.reference.ReferenceTable`). Not yet a store thing.")
        if eval_every and not source.eval_pairs_declared():
            print("[train] store path: --eval-every is OFF. This regime declares no `eval_pairs`, "
                  "so the eval split has no imputation targets and a mid-training eval would "
                  "select the 'best' checkpoint on nothing. Declare `eval_pairs` (D31) to turn it "
                  "back on.", flush=True)
            eval_every = 0
    elif eval_every:
        # THE h5 PATH NO LONGER EVALUATES, IN PROCESS, AT ALL — degraded deliberately rather than
        # refused, because the training half of this path is untouched and still runs. `candi.eval`
        # is deleted (D15): it owned BOTH the mid-training `quick_eval` and the final `evaluate`,
        # and neither has an h5-capable replacement here — `candi.monitor` opens its data with
        # `bench.harness.open_source(store=…)` and is store-only by construction. Scoring an h5
        # checkpoint is now a separate command AFTER the run, which `candi.bench` does support:
        #
        #     python -m candi.bench --h5 <panel.h5> --ckpt <run>.ckpt --arch-from <run>.json \
        #         --out <scores>.json
        #
        # The cost is stated rather than discovered: with no mid-training number there is no
        # best-checkpoint selection on this path, so `<tag>.ckpt` (the LAST epoch) is the only
        # checkpoint an h5 run produces.
        print("[train] h5 path: --eval-every is OFF and there is NO final evaluation. `candi.eval` "
              "is deleted and its replacement (`candi.monitor` / `candi.bench`) reads a store, so "
              "an h5 run now trains only. This run json carries the training curve, the config and "
              "the LAST checkpoint — no best-checkpoint selection, and none of the h5-era "
              "M1/M2/M3/S14 keys. Score the "
              "checkpoint afterwards with `python -m candi.bench --h5 <panel.h5> --ckpt "
              "<tag>.ckpt --arch-from <tag>.json --out <scores>.json`.", flush=True)
        eval_every = 0
    # Scale is read from the h5 (or from the regime file), never from a flag. `h5_cache_ram=False`
    # so this probe does not duplicate the shared RAM buffer that the full-coverage loop allocates.
    ds = make_dataset(source, regime, train=True, batch_size=batch_size,
                      dsf_sampling=dsf_sampling, seed=seed, h5_cache_ram=False,
                      cell_cond=cell_cond)
    if depth_center is None:
        if source.is_store:
            depth_center = ds.depth_center()
            print(f"[train] depth_center derived from the store (median log2 depth over the "
                  f"regime's train split) = {depth_center:.4f} — override with --depth-center",
                  flush=True)
        else:
            depth_center = h5_depth_center(h5_path)
            print(f"[train] depth_center derived from h5 (median T_ meta_dsf1[0]) = {depth_center:.4f} "
                  "— override with --depth-center", flush=True)

    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    # num_cells comes from the h5's own biosample list, never from a flag — same rule as num_assays.
    # THE ARCHITECTURE, as one dict. It is what `build_model` is called with AND what lands in the
    # run JSON as `config.arch`, so `candi.bench --arch-from <run>.json` rebuilds exactly this model
    # without a single flag being retyped. Scale (num_assays, context_bins, resolution, num_cells)
    # still comes from the h5 and is never a flag.
    arch = dict(embed_dim=embed_dim, dropout=dropout,
                depth_center=depth_center, use_offset=use_offset,
                num_assays=ds.num_assays, context_length=ds.context_bins,
                resolution=int(ds.resolution if resolution is None else resolution),
                d_model=d_model, nhead=nhead, num_cells=ds.num_cells,
                meta_embed_layernorm=bool(meta_embed_layernorm), meta_gain=float(meta_gain),
                decoder_lane=int(decoder_lane), deconv_norm=str(deconv_norm),
                n_transformer_layers=int(n_transformer_layers),
                n_cnn_layers=int(n_cnn_layers), conv_kernel_size=int(conv_kernel_size),
                pool_size=int(pool_size), expansion_factor=int(expansion_factor),
                n_deconv_layers=int(n_deconv_layers), deconv_upsample=int(deconv_upsample),
                deconv_kernel_size=int(deconv_kernel_size),
                conv_norm=str(conv_norm), transformer_norm=str(transformer_norm),
                transformer_norm_placement=str(transformer_norm_placement),
                attn_qk_norm=bool(attn_qk_norm),
                film_taps=list(parse_film_taps(film_taps)),
                film_init_encoder=str(film_init_encoder),
                film_init_decoder=str(film_init_decoder),
                head_sharing=str(head_sharing), head_hidden=int(head_hidden),
                # Stored as a LIST, like film_taps, because this dict is written to the run JSON and
                # read back by `--arch-from`; a tuple round-trips through JSON as a list anyway, and
                # storing what comes back is what keeps the two comparable.
                heads=list(parse_heads(heads)))
    model = build_model(**arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params:,} (decoder_lane={decoder_lane} deconv_norm={deconv_norm} "
          f"attn_depth={n_transformer_layers} film_taps={','.join(arch['film_taps'])} "
          f"heads={','.join(arch['heads'])})", flush=True)
    if tuple(arch["heads"]) != tuple(DEFAULT_HEADS):
        # Said out loud because the extra heads change the OBJECTIVE while the mid-training monitor
        # scores the count arm alone. The numbers in this run's JSON are therefore comparable to a
        # default arm only if the reader knows the arm was trained against a different total loss.
        print(f"[train] heads={','.join(arch['heads'])}: the Gaussian signal and/or Bernoulli peak "
              "terms are ADDED to the NB loss (coefficient 1.0 each). The monitor scores the count "
              "arm, so this run's curve measures a model trained on a different objective. This is "
              "NOT the control arm; the control is --heads count.", flush=True)
    if float(meta_gain) != 1.0:
        print(f"[train] meta_gain={float(meta_gain)} (h76): `memb` is scaled by this FIXED, "
              "non-learnable factor immediately before decoder.film_proj. This is NOT the control "
              "arm; the control is meta_gain=1.0, where the multiply is skipped entirely.", flush=True)
    if not meta_embed_layernorm:
        print("[train] meta_embed_layernorm=OFF (h75): the metadata-fusion LayerNorm is REMOVED on "
              "both towers. This is NOT the control arm.", flush=True)
    if ds.num_cells:
        print(f"[train] cell_cond={cell_cond}: 5th metadata row ON, {ds.num_cells} cell types",
              flush=True)

    # --- h64 arm switch ----------------------------------------------------------------------------
    # Built AFTER the model so `num_runtypes` is read off the embedding table that will actually
    # consume the planted index, rather than assumed. `off` returns None and nothing below fires.
    mprobe = make_meta_probe(
        meta_probe, seed=seed, delta=float(meta_probe_delta),
        num_runtypes=int(getattr(model.encoder.metadata_embedding, "num_runtypes", 2)))
    if mprobe is not None:
        print(mprobe.banner(), flush=True)

    # --- h74 reference offset ---------------------------------------------------------------------
    # `off` never constructs the table, so nothing about the arm's arithmetic, memory or IO changes:
    # `forward_full` sees no `log_ref` key and the decoder takes the pre-h74 branch exactly.
    ref = None
    if reference == "on":
        from candi.reference import REFERENCE_PSEUDOCOUNT, ReferenceTable, reference_path_for
        rp = reference_path_for(h5_path) if reference_path is None else Path(reference_path)
        pc = REFERENCE_PSEUDOCOUNT if reference_pseudocount is None else float(reference_pseudocount)
        # src_h5 is passed so a re-baked h5 aborts the run instead of training against a stale
        # reference — silently wrong offsets would look like a modelling result.
        ref = ReferenceTable(rp, src_h5=h5_path, pseudocount=pc)
        # The two offsets must live on ONE exposure scale. `R` is normalized to the reference's
        # depth_center and the decoder subtracts the model's, so a disagreement silently biases every
        # prediction by 2^(difference) — invisible in the loss curve, fatal to the comparison. Both
        # default to `h5_depth_center` on the same file and agree; `--depth-center` can break that.
        if abs(float(ref.depth_center) - float(depth_center)) > 1e-6:
            raise ValueError(
                f"depth_center mismatch: the reference was built at {ref.depth_center:.6f} but this "
                f"run uses {depth_center:.6f}. The reference offset is expressed at ITS centre, so "
                "the two would compose on different scales. Rebuild the reference with "
                f"`--depth-center {depth_center}`, or drop --depth-center so both derive it from the h5.")
        print(f"[train] reference=on: {rp} — {len(ref.contributors)} T_ contributors, "
              f"pseudocount={pc}, depth_center={ref.depth_center:.4f}; leave-one-out ACTIVE",
              flush=True)
        zero = [ref.assays[a] for a in range(len(ref.assays)) if ref.count[a] == 0]
        if zero:
            print(f"[train] NOTE {len(zero)} assays have zero T_ contributors, so their reference is "
                  f"the pseudocount floor and carries no information: {zero}", flush=True)

    # t89 — WHICH POSITIONS SELECT THE CHECKPOINT. Hashed from the BED here, before the monitor
    # exists, so the dashboard config carries the pin from the first logged step; the monitor
    # upgrades it below with the window and bin counts it actually planned. The sha256 is the
    # load-bearing half — `configs/regions/…` is a mutable file, and the path alone would let two
    # runs claim the same scope while scoring different loci.
    eval_scope_cfg: dict = {"name": "full", "note": "every window of every eval chromosome"}
    if eval_regions is not None and not eval_every:
        # A scope with nothing to score is a claim in the run json about a curve that does not
        # exist. Blanked here rather than recorded, which is the same rule D32 follows for an
        # unset `regions` and the same one the h5 degradation notice above follows.
        print(f"[train] --eval-regions {eval_regions} is INERT: --eval-every resolved to 0, so "
              f"there is no mid-training check to scope. config.eval_scope stays `full`.",
              flush=True)
        eval_regions = None
    if eval_regions is not None:
        from candi.store.regime import RegionSet
        eval_scope_cfg = {"name": "regions", **RegionSet.from_bed(eval_regions).to_dict()}

    run_cfg = dict(regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                   batch_size=batch_size, lr=lr, weight_decay=weight_decay, seed=seed,
                   cell_cond=cell_cond, num_cells=int(ds.num_cells), num_assays=ds.num_assays,
                   context_bins=ds.context_bins, full_coverage=full_coverage,
                   depth_center=float(depth_center),
                   # t23: which data path, and — on the store — the regime verbatim plus its
                   # sha256 and the manifest's (STORE_PLAN.md §4). On the h5 path this is the same
                   # `h5=<path>` key it has always been, plus `data_source`.
                   **source.provenance(),
                   reference=reference, imp_weight=imp_weight, unmask_frac=unmask_frac,
                   eval_every=eval_every, eval_batches_per_pair=eval_batches_per_pair,
                   # t89 — WHICH POSITIONS SELECTED THIS CHECKPOINT. `source.provenance()` above
                   # describes the TRAINING source and says nothing about the monitor's scope, so
                   # without this key a `regions` run and a `full` run produce config blocks that
                   # are identical while their curves are not comparable. The path alone would not
                   # do: `configs/regions/…` is a mutable file, and it is the sha256 that pins what
                   # was actually scored. Resolved below, once the monitor has hashed the BED.
                   eval_scope=eval_scope_cfg,
                   meta_embed_layernorm=bool(meta_embed_layernorm),
                   # h51: the two arms differ ONLY in trunk_wd, so both values ride into the
                   # dashboard config — a run whose arm has to be inferred is a run that can be
                   # mislabelled.
                   optimizer=optimizer, trunk_wd=float(trunk_wd),
                   # h76: the arm label for the gain sweep. 1.0 is the control.
                   meta_gain=float(meta_gain),
                   # h64: the SAME rule. A `planted` run trains against a target nobody else's run
                   # has; its numbers are not comparable to a real arm's, and the config is where a
                   # reader finds that out.
                   meta_probe=meta_probe, meta_probe_delta=float(meta_probe_delta),
                   # Stage 3: which precision trained this. It leaves NO trace in the state_dict —
                   # autocast keeps master weights in fp32, so a bf16 checkpoint and an fp32 one are
                   # the same file format — and it is deliberately not part of `arch`, so the config
                   # is the only place a reader can recover it.
                   precision=str(precision),
                   # D30. The RESOLVED value, never `auto`. Two runs are comparable on the signal
                   # head only if this field matches, exactly as `EVAL.md` says of `--heads`.
                   signal_target_transform=str(signal_target_transform))
    wb = _wandb_init(wandb_project, wandb_run_name, run_cfg)
    log_fn = (lambda step, d: wb.log(d, step=step)) if wb else None

    # --- mid-training eval + best-checkpoint selection ---------------------------------------------
    # h73 evaluated ONCE, after training, so over- and under-fitting were indistinguishable. This
    # scores held-out imputation every `eval_every` epochs and keeps the best checkpoint by it.
    # Selection is on the imputation dial by PI ruling: the bias it introduces is common-mode across
    # arms, so the paired delta survives it. Which biosamples that dial reads is the regime file's
    # business — on `configs/regime.eic_val.json` the targets are all `V_`, so `B_` is never opened.
    best = {"crps": float("inf"), "epoch": -1, "step": -1}
    curve: list = []
    best_path = (str(Path(ckpt_path).with_suffix(".best.ckpt")) if ckpt_path else None)

    # OPENED ONCE, OUTSIDE THE HOOK, and re-iterated at every check (t28). Two reasons, and the
    # second is the load-bearing one. Re-opening per check would re-read the store's manifest and
    # re-plan the windows eight times a run for nothing; but more importantly, ONE source is what
    # pins ONE window set across every epoch. Selection compares epoch 6 against epoch 12, and that
    # comparison is only paired if both saw the same positions. The monitor's plan is the harness's
    # full tiling of the regime's eval chromosomes — every 25 bp bin, so "the same positions" is
    # "all of them" and the pairing is exact rather than merely deterministic.
    #
    # ONE PATH NOW. `--store` gets `candi.monitor`: whole eval chromosomes, an imputation dial that
    # selects on every check and a denoising dial that only watches, once, at the end. `--h5` gets
    # NOTHING — see the degradation
    # notice printed above, where `eval_every` was forced to 0. The mid-training scorer the h5 path
    # used was `eval.quick_eval`, and `candi.eval` is deleted; the monitor is store-only because
    # `Monitor.__init__` opens its source with `open_source(store=…)`, and teaching it the h5 would
    # be new code written against a data path that is being retired.
    # TWO MONITORS, NOT ONE, and only when `--eval-regions` narrows the selection scope (t89).
    # `monitor` is the one the hook calls every `eval_every` epochs and is the one that may be
    # cheap; `final_monitor` is the end-of-run check and is ALWAYS full coverage. The split is the
    # PI's ruling and not an optimisation: a cheap number is good enough to say which of two
    # checkpoints is better, and not good enough to be the number a run reports. Both are opened
    # here rather than inside the hook for the reason below — one source is one window plan, and
    # selection is only a paired comparison if every epoch saw the same positions.
    monitor = None
    final_monitor = None
    if eval_every and source.is_store:
        # `batch_windows` IS the eval batch size — how many context windows go through the model in
        # one forward pass — so the existing flag keeps its meaning rather than gaining a twin.
        monitor = Monitor(store=source.regime_path, device=device, seed=seed,
                          batch_windows=eval_batch_size, eval_regions=eval_regions,
                          signal_target_transform=signal_target_transform, log_fn=log_fn)
        final_monitor = monitor if eval_regions is None else Monitor(
            store=source.regime_path, device=device, seed=seed, batch_windows=eval_batch_size,
            signal_target_transform=signal_target_transform, log_fn=log_fn)
        sc = monitor.source.scope()
        where = (f"{','.join(monitor.source.eval_chroms)} end to end" if sc["name"] == "full" else
                 f"{sc['scored_bins']:,} of {sc['full_bins']:,} bins "
                 f"({sc['fraction']:.2%}) of {','.join(monitor.source.eval_chroms)}, inside "
                 f"{sc['n_regions']} regions of {sc['bed']}")
        print(f"[train] monitor: every {eval_every} epoch(s), {where} — imputation ONLY (it is what "
              f"selects), count arm, val loss in "
              f"signal_target_transform={signal_target_transform}. Reported: "
              f"{', '.join(tiered_keys(model))}", flush=True)
        print("[train] monitor: the denoising dial (watch-only) and the overfitting gap run ONCE, "
              "at the end, on the SELECTED checkpoint — the PI's cost ruling, measured in "
              "cruxvault/results/t30/TIMING.md.", flush=True)
        run_cfg["eval_scope"] = eval_scope_cfg = sc
        if sc["name"] != "full":
            print(f"[train] monitor: the SELECTION scope above is NARROWER than the eval scope. "
                  f"Two runs are comparable on the mid-training curve only if their "
                  f"`eval_scope` blocks match — the BED, its sha256 {sc['sha256'][:12]} and the "
                  f"chromosome list. The END-OF-RUN check on the selected checkpoint is FULL "
                  f"COVERAGE and is the number to quote; the positional measures (mseprom, "
                  f"msegene, mseenh, the P-block) are absent mid-training because a compacted "
                  f"array's index is no longer a genomic position.", flush=True)

    def _keep_best(value, step, ep):
        """Lower is better, and only a finite number may select."""
        improved = bool(np.isfinite(value)) and float(value) < best["crps"]
        if improved:
            best.update(crps=float(value), epoch=ep, step=step)
            if best_path:
                torch.save(model.state_dict(), best_path)
        return improved

    def eval_hook(step, ep):
        # `model.eval()` and the `no_grad` fence are `stream_tracks`'s, and the monitor puts the
        # training mode back itself — the two loops above disagreed about whose job that was.
        #
        # IMPUTE ONLY, by the PI's cost ruling on the measured wall-clocks in
        # `cruxvault/results/t30/TIMING.md`: a whole-chromosome impute pass already spends most of
        # the 20 % budget the cadence is held to, and the denoise dial scores ~2.3x its tracks. So
        # the mid-training row carries the selection metric and no `gap`; the alarm is computed once
        # below, on the checkpoint this loop ends up selecting.
        row = monitor.check(model, epoch=ep, step=step, kinds=("impute",))
        curve.append(row)
        _keep_best(row["selection"]["value"], step, ep)
        print(format_check(row, best=best), flush=True)
        # Stop when the selection metric has not improved for more than `early_stop_epochs` epochs.
        # Counted in EPOCHS, not in evals, because `eval_every` changes how many evals an epoch buys
        # and the patience the operator asked for is a number of epochs either way. The resolution is
        # still `eval_every`: at the default 3 the earliest detectable stall is 6 epochs, and a
        # patience below `eval_every` can therefore never fire. Nothing is lost by stopping — the
        # best checkpoint is already on disk (`_keep_best` writes it the moment it improves), and the
        # block below loads it rather than the last one.
        if early_stop_epochs and best["epoch"] >= 0 and (ep - best["epoch"]) > early_stop_epochs:
            print(f"[train] EARLY STOP at epoch {ep}: no V_ improvement since epoch "
                  f"{best['epoch']} ({ep - best['epoch']} epochs > patience "
                  f"{early_stop_epochs}). Best crps={best['crps']:.4f}, and it is the checkpoint "
                  f"that will be scored.", flush=True)
            return True
        return False

    terms_log: list = []
    losses = train(model, source, device, regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                   batch_size=batch_size, lr=lr, weight_decay=weight_decay, seed=seed,
                   dsf_sampling=dsf_sampling, log_every=log_every, full_coverage=full_coverage,
                   p_full_assay=p_full_assay, mask_fraction=mask_fraction, cell_cond=cell_cond,
                   log_fn=log_fn, reference=ref, imp_weight=imp_weight, unmask_frac=unmask_frac,
                   eval_hook=(eval_hook if eval_every else None), eval_every=eval_every,
                   terms_out=terms_log, optimizer=optimizer, trunk_wd=trunk_wd,
                   meta_probe=mprobe, lr_schedule=lr_schedule, warmup_frac=warmup_frac,
                   lr_min_ratio=lr_min_ratio, clip_norm=clip_norm, precision=precision,
                   signal_target_transform=signal_target_transform)
    # The monitor is NOT closed here: the end-of-run check below still needs its source, and it has
    # to run after the selected weights are loaded. `monitor.close()` is further down, after
    # everything that reads the eval chromosomes.
    model.eval()
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)          # the LAST checkpoint, always written
    # Score the BEST checkpoint, not the last one. Written before the monitor's last check, so a
    # walltime kill still leaves both checkpoints on disk and the run is re-scorable with
    # `python -m candi.bench`.
    selected = "last"
    if best_path and Path(best_path).exists() and best["epoch"] != epochs - 1:
        model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
        model.eval()
        selected = "best"
        print(f"[train] keeping the BEST checkpoint (epoch {best['epoch']}, impute macro "
              f"crps={best['crps']:.4f}), not the last", flush=True)

    # --- the end-of-run check: the denoising dial and the overfitting alarm, ONCE -----------------
    # THE MODEL HERE IS THE ONE THE RUN SHIPS. That is the whole reason this sits below the load
    # above rather than at the end of the loop: the alarm is a statement about a specific set of
    # weights, and the last epoch's weights are not the selected ones unless the last epoch won.
    #
    # Mid-training rows carry the impute dial alone (see `eval_hook`), so this is the only row in
    # the run json with a `denoise` block and a `gap`. Cost: one extra pass of each dial per RUN,
    # not per check — which is what makes the ruling in `cruxvault/results/t30/TIMING.md` fit the
    # budget at all.
    final_row = None
    if monitor is not None and curve:
        if best["epoch"] < 0:
            # Nothing finite ever selected, so `<tag>.best.ckpt` was never written and `model` is
            # the last epoch's. Say so beside the numbers rather than letting the row imply that a
            # selection happened.
            print("[train] NO finite selection was ever made (every impute macro crps was "
                  "non-finite), so there is no best checkpoint. The end-of-run check below scores "
                  "the LAST model.", flush=True)
        final_epoch = best["epoch"] if selected == "best" else epochs - 1
        # W&B: `Monitor.log_fn` is the same `log_fn` the loop logs through, so the row reaches the
        # dashboard from inside `final_check` — under its own `monitor_final/...` keys, at the last
        # training step, which is the step below.
        # `final_monitor`, not `monitor` — under `--eval-regions` the mid-training one is cut down
        # to the selection scope, and the row a run REPORTS is full coverage whatever the curve that
        # picked the weights was measured on. They are the same object when no scope was asked for.
        final_row = final_monitor.final_check(model, epoch=int(final_epoch),
                                              step=int(curve[-1]["step"]), selected=selected)
        print(format_check(final_row), flush=True)
    for m in {id(x): x for x in (monitor, final_monitor) if x is not None}.values():
        # The store handle pool is the monitor's, not the training loader's, so it has to be given
        # back explicitly — nothing below reads the eval chromosomes again. Keyed by identity so the
        # unscoped case, where both names are one object, closes it once.
        m.close()
    # h64: THE PROBE IS NOW APPLIED IN TRAINING ONLY, ON EVERY PATH. `quick_eval` took
    # `meta_probe=mprobe` and transformed the assembled batch, and it retired with `candi.eval`
    # (D15); the monitor scores through `candi.bench`, which knows nothing about h64 and must not be
    # taught — an eval package that imports a training-side arm switch is the dependency inversion
    # the bench boundary exists to prevent. So a planted arm's mid-training curve measures it
    # against the REAL objective it was never trained on, and the two are not comparable.
    if mprobe is not None:
        print(f"\n[meta-probe] WARNING: meta_probe={meta_probe} was applied in TRAINING ONLY.\n"
              "[meta-probe] The mid-training monitor scores through candi.bench, which does not "
              "apply the probe: its curve\n"
              "[meta-probe] measures this arm against the REAL objective. Do NOT compare it to a "
              "--meta-probe off arm.\n"
              "[meta-probe] See config.final_eval_meta_probe_applied = false.\n", flush=True)
    # NO FINAL EVALUATION ON EITHER PATH. `evaluate()` was `candi.eval`'s h5-only top-level entry
    # point and it is deleted (D15); scoring a checkpoint is now `python -m candi.bench`, a separate
    # command that takes either `--h5` or `--store`. Writing empty M1/M2 keys instead would read in
    # a json exactly like a finished evaluation, so the keys are ABSENT, not empty.
    #
    # `final_check` is NOT that final evaluation coming back. It is the same monitor, on the same
    # tiered subset, run once more so the overfitting alarm exists at all — a selection instrument,
    # not the recorded number (`EVAL.md`, *Two instruments*).
    print("\n[train] NO final evaluation in this process — scoring is `python -m candi.bench` now. "
          "This run json carries the training curve, the config, the checkpoints, the "
          "mid-training monitor curve and its one end-of-run check (store path, `eval_pairs` "
          "declared) — and none of the h5-era M1/M2/M3/S14 keys.\n", flush=True)
    ev: dict = {}
    ev["eval_curve"] = curve
    # The end-of-run row, under its own key rather than appended to `eval_curve`: it is the only row
    # with a `denoise` block and a `gap`, and it describes the SELECTED weights rather than an epoch
    # of the curve. `None` when the monitor was off or never fired, which is absence, not zero.
    ev["final_check"] = final_row
    ev["best_checkpoint"] = dict(best, path=best_path, scored=selected)
    # h64: the arm, its magnitude, and — unambiguously — WHERE it was and was not applied.
    # `meta_probe_shuffle_noop_frac` is the negative control's own verdict on itself: 1.0 means the
    # permutation never changed a tensor, so the arm trained on the real run_type.
    # `meta_probe_quick_eval_applied` keeps its name so a reader of an OLD run json still finds the
    # field, and it is now flatly False: `quick_eval` was the only scorer that ever applied the
    # probe and it retired with `candi.eval` (D15). The mid-training monitor scores through
    # `candi.bench`, which never applies it, so on every path the probe is training-only.
    h64_cfg = dict(meta_probe=meta_probe, meta_probe_delta=float(meta_probe_delta),
                   meta_probe_train_applied=(mprobe is not None),
                   meta_probe_quick_eval_applied=False,
                   final_eval_meta_probe_applied=False)
    if mprobe is not None:
        h64_cfg.update(mprobe.stats())
    cfg_block = dict(regime=regime, epochs=epochs, steps_per_epoch=steps_per_epoch,
                     batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                     # h51: recorded in the run JSON as well as the W&B config. `optimizer` +
                     # `trunk_wd` are the ONLY fields that differ between this node's two arms, so a
                     # reader of the json must be able to tell them apart without guessing.
                     optimizer=optimizer, trunk_wd=float(trunk_wd),
                     use_offset=use_offset, dsf_sampling=dsf_sampling, seed=seed,
                     fg_frac=fg_frac, n_boot=n_boot, full_coverage=full_coverage,
                     eval_max_batches=eval_max_batches, eval_budget=eval_budget,
                     m3_regions=m3_regions, p_full_assay=p_full_assay,
                     mask_fraction=mask_fraction, embed_dim=embed_dim, dropout=dropout,
                     n_transformer_layers=n_transformer_layers,
                     include_deprecated=include_deprecated,
                     # The model identity. `n_params` rides along because it is the first thing any
                     # reader comparing two runs wants, and it cannot be reconstructed from the rest.
                     decoder_lane=int(decoder_lane), deconv_norm=str(deconv_norm),
                     # `lane_norm` is the pre-rename spelling of the same value, mirrored so a diff
                     # against a candi_kit2 ladder JSON still lines up on this field.
                     lane_norm=str(deconv_norm),
                     arch=dict(arch), n_params=int(n_params),
                     lr_schedule=str(lr_schedule), warmup_frac=float(warmup_frac),
                     lr_min_ratio=float(lr_min_ratio), clip_norm=float(clip_norm),
                     # Stage 3. NOT inside `arch`, deliberately: precision changes no module, no
                     # shape and no state_dict key, so putting it there would make `--arch-from`
                     # refuse to re-score a bf16 checkpoint in fp32 — which is the only way eval
                     # ever scores it, since evaluation is never autocast.
                     precision=str(precision),
                     cell_cond=cell_cond, num_cells=int(ds.num_cells),
                     # h75: recorded here as well as in the W&B config, so a reader of the run JSON
                     # cannot mistake an LN-off run for a control.
                     meta_embed_layernorm=bool(meta_embed_layernorm),
                     # h76: recorded here for the same reason — the gain leaves no trace in the
                     # state_dict (it is a plain float, deliberately), so the config is the ONLY
                     # place a reader can recover which arm produced this JSON.
                     meta_gain=float(meta_gain),
                     reference=reference,
                     reference_path=(str(ref.path) if ref else None),
                     reference_pseudocount=(ref.pseudocount if ref else None),
                     reference_fingerprint=(ref.src_fingerprint if ref else None),
                     reference_n_contributors=(len(ref.contributors) if ref else 0),
                     imp_weight=imp_weight, unmask_frac=unmask_frac,
                     # D30. Resolved, not requested. The signal head leaves no trace of its target
                     # space in the state_dict — a Gaussian head trained on `arcsinh(-log10 p)` and
                     # one trained on the raw value are the same file — so this field is the only
                     # place a reader can recover which one produced these numbers.
                     signal_target_transform=str(signal_target_transform),
                     **h64_cfg,
                     eval_every=eval_every, eval_batches_per_pair=eval_batches_per_pair,
                     # t89 — the SELECTION scope. Not the same statement as `eval_chroms` below:
                     # that says which chromosomes were predicted, this says how much of them the
                     # curve that picked these weights was measured over.
                     eval_scope=eval_scope_cfg,
                     out_dir=str(out_dir),
                     # t23: the same provenance block `run_cfg` carries. On the store path
                     # that is the regime file VERBATIM plus its sha256 and the store
                     # manifest's sha256 — STORE_PLAN.md §4's last line, which asks for all
                     # three so a run can be re-pointed at the data it actually saw.
                     **source.provenance(),
                     assays=list(ds.assays), num_assays=ds.num_assays,
                     context_bins=ds.context_bins, resolution=ds.resolution,
                     dsf_list=list(ds.dsf_list), train_chroms=list(ds.train_chroms),
                     eval_chroms=list(ds.eval_chroms), d_model=int(model.encoder.d_model),
                     nhead=nhead, depth_center=float(depth_center),
                     kit_version=_pkg_version("candi"), torch_version=torch.__version__,
                     x_transformers_version=_pkg_version("x-transformers"))
    if wb:
        # Eval scalars land as summary values, not as a time series: they are measured once, at the
        # end, and a step-indexed point would imply a curve that does not exist. S14 and the
        # reference-only bar ride along — both were dropped entirely before, and S14 is the only
        # metric here with a real counterfactual ground truth.
        #
        # HISTORICAL, AND INERT SINCE D15. `M1`/`M2`/`M3`/`S14` are the h5-era `candi.eval` keys.
        # Nothing writes them any more — `ev` above is built from the training curve alone — so
        # every lookup below misses and logs nothing. Kept as the record of what the dashboard
        # carried when the run json still had them; the covariate instruments they became are named
        # `covuse` / `covshare` / `depthdir` / `depthcounterfact` / `covspec` / `depthblind` /
        # `biokeep` and live in a bench json, not here.
        _wb_safe("summary", lambda: wb.summary.update(_flat_scalars(
            {k: ev[k] for k in ("M1", "M2", "M3", "S14", "reference_only_baseline") if k in ev})))
        # PER-TARGET, as tables. `_flat_scalars` silently drops every one of these because they are
        # lists and maps, so the dashboard carried only pooled numbers — and pooling has already
        # overstated an arm gap threefold once. Each table is logged separately so one that fails to
        # serialise cannot take the others with it.
        m1, m2 = (ev.get("M1") or {}), (ev.get("M2") or {})
        tables = {
            "eval/M1_imp_per_target": _rows_from_map(m1.get("imp_per_target")),
            "eval/M1_den_per_target": _rows_from_map(m1.get("den_per_target")),
            "eval/M2_depth_per_target": (m2.get("depth") or {}).get("per_target"),
            "eval/M2_run_type_per_target": (m2.get("run_type") or {}).get("per_target"),
            "eval/S14_per_target": (ev.get("S14") or {}).get("per_target"),
            "eval/reference_only_per_target": (ev.get("reference_only_baseline") or {}).get("per_target"),
            "eval/curve": curve,
        }
        for _name, _rows in tables.items():
            if _rows:
                _wb_safe(f"table {_name}", lambda n=_name, r=_rows: wb.log({n: _wb_table(list(r))}))
        # Provenance: WHICH code, WHICH data, WHICH panel, WHICH seed. Without it a dashboard row is
        # a number with no way back to the thing that produced it.
        prov = dict(cfg_block)
        prov.update(_git_provenance())
        prov["h5_fingerprint"] = None if source.is_store else _h5_fingerprint_safe(h5_path)
        _wb_safe("config", lambda: wb.config.update(prov, allow_val_change=True))
        _wb_safe("finish", wb.finish)
    return dict(config=cfg_block, train_losses=losses, train_terms=terms_log, **ev)


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a value. `main()` parses it; a test can inspect it without running anything."""
    ap = argparse.ArgumentParser()
    # t23 — EXACTLY ONE data path, enforced by argparse itself so the error arrives on the submit
    # line with argparse's own wording ("one of the arguments --h5 --store is required" /
    # "not allowed with argument"). `--h5` is no longer `required=True`, and it did not become
    # optional: the GROUP is required, so omitting both still fails.
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", default=None,
                     help="path to a baked h5 (candi.prep.bake). The frozen path. Mutually "
                          "exclusive with --store, and exactly one of the two is required.")
    src.add_argument("--store", "--regime-file", dest="store", default=None, help=STORE_HELP)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regime", default="type1", choices=["type1", "type2_loci"],
                    help=MASK_REGIME_HELP)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                    help="h51: adam = the frozen default — ONE parameter group, coupled L2 through "
                         "--weight-decay, bit-identical to the pre-h51 kit. adamw = two groups: the "
                         "conditioning pathway (embedding tables, film_proj, every norm affine, "
                         "every bias) pinned at wd=0 and trunk+heads at --trunk-wd. At --trunk-wd 0 "
                         "the adamw path is numerically the adam default (decoupled shrink by "
                         "1-lr*0 is a multiply by 1.0), which is what makes it a clean control. "
                         "CANNOT be combined with a non-zero --weight-decay.")
    ap.add_argument("--trunk-wd", type=float, default=0.0,
                    help="h51: weight decay on the DECAY group (trunk + heads) in the adamw path; "
                         "the no-decay group is always 0.0. 0.0 = the control arm, 0.1 = the split "
                         "arm (the GPT-3 / LLaMA convention). Inert — and therefore rejected — under "
                         "--optimizer adam.")
    ap.add_argument("--offset", "--arm", dest="offset", default="on",
                    choices=["on", "off", "offset_on", "offset_off"])
    ap.add_argument("--dsf-sampling", default="uniform", choices=["uniform", "off", "x_eq_y", "upsample_only"])
    ap.add_argument("--p-full-assay", type=float, default=1.0)
    ap.add_argument("--mask-fraction", type=float, default=0.2, help="INERT under --p-full-assay 1.0")
    ap.add_argument("--depth-center", type=float, default=None,
                    help="default: median of finite meta_dsf1[0] over T_ biosamples (printed)")
    ap.add_argument("--d-model", type=int, default=0,
                    help="0 = auto = (num_assays+1)*expansion**n_cnn_layers; SET IT when num_assays != 8")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--n-transformer-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    # THE SIX FLAGS MARKED INERT BELOW ARE INERT (`--full-coverage`, between them, is not — it is a
    # TRAINING knob and still does what it says). Each set a knob on `eval.evaluate` — the scorer this
    # module used to call — and `candi.eval` is deleted (D15). They are still parsed and still
    # recorded in the run config so an old submit script does not fail on an unrecognised argument,
    # and they change no number. `candi.bench` refuses its own spellings of --eval-budget and
    # --eval-max-batches BY NAME (bench.cli.RETIRED, D2) because there the mistake is live.
    ap.add_argument("--eval-max-batches", type=int, default=0, help="INERT — see above.")
    ap.add_argument("--fg-frac", type=float, default=0.02, help="INERT — see above.")
    ap.add_argument("--n-boot", type=int, default=1000, help="INERT — see above.")
    ap.add_argument("--full-coverage", action="store_true",
                    help="deterministic full coverage: every epoch = all train windows x all T_ biosamples")
    ap.add_argument("--eval-budget", type=int, default=200_000, help="INERT — see above.")
    ap.add_argument("--m3-regions", type=int, default=8, help="INERT — see above.")
    ap.add_argument("--include-deprecated", action="store_true", help="INERT — see above.")
    # -- geometry ---------------------------------------------------------------------------------
    # The encoder's total downsampling and the decoder's total upsampling MUST agree; `build_model`
    # refuses the pair before anything is constructed. `--dna-pool-size` is deliberately NOT a flag:
    # it is isqrt(resolution), derived from the panel the h5 declares.
    ap.add_argument("--n-cnn-layers", type=int, default=3,
                    help="conv blocks in the signal tower. Each halves the sequence by --pool-size "
                         "and multiplies the per-assay channel count by --expansion-factor.")
    ap.add_argument("--conv-kernel-size", type=int, default=3)
    ap.add_argument("--pool-size", type=int, default=2)
    ap.add_argument("--expansion-factor", type=int, default=2)
    ap.add_argument("--n-deconv-layers", type=int, default=3)
    ap.add_argument("--deconv-upsample", type=int, default=2)
    ap.add_argument("--deconv-kernel-size", type=int, default=3)
    # -- normalisation ------------------------------------------------------------------------------
    ap.add_argument("--conv-norm", default="layer",
                    choices=["layer", "lane", "group", "batch", "instance"],
                    help="normaliser after every conv in BOTH towers. 'layer' is the shipped default "
                         "and, despite the grouped conv, pools its statistic across the WHOLE panel "
                         "at each position — so the conv tower is not per-assay under it, and never "
                         "has been (see encoder.ConvBlock). 'lane' is the same statistic computed "
                         "within one assay; 'group' additionally pools positions; 'instance' pools "
                         "positions per channel and mixes nothing.")
    ap.add_argument("--deconv-norm", default="lane", choices=["lane", "group"],
                    help="how the deconv trunk's per-assay normaliser pools its statistic. Both are "
                         "per-assay and neither leaks. 'lane' = the C channels of one lane AT EACH "
                         "POSITION, which fixes that lane's energy per bin so amplitude survives only "
                         "as a pattern across C. 'group' = the nn.GroupNorm statistic at assay "
                         "granularity — the C channels ACROSS ALL POSITIONS — which removes the "
                         "lane's overall scale and leaves the profile along the genome intact. "
                         "(Renamed from --lane-norm: the old name collided with its own first value.)")
    ap.add_argument("--transformer-norm", default="layer",
                    choices=["layer", "rmsnorm", "simple_rmsnorm", "scalenorm"],
                    help="norm inside each x-transformers block.")
    ap.add_argument("--transformer-norm-placement", default="pre",
                    choices=["pre", "post", "sandwich"],
                    help="where that norm sits relative to the residual. 'resi_dual' was in the "
                         "agreed set and is NOT offered: the pinned x-transformers does not accept "
                         "the keyword, so the choice would have raised the moment it was selected.")
    ap.add_argument("--attn-qk-norm", action="store_true",
                    help="normalise Q and K before attention (x-transformers attn_qk_norm).")
    # -- conditioning -------------------------------------------------------------------------------
    ap.add_argument("--film-taps", default=",".join(DEFAULT_FILM_TAPS),
                    help="comma-separated set naming EVERY place metadata conditioning enters, both "
                         "towers. Encoder: pre_conv | per_conv | post_conv (alternative placements "
                         "of one FiLM — pick at most one) and per_transformer (requires per_conv). "
                         "Decoder: pre_deconv, per_deconv, post_head. Default "
                         f"'{','.join(DEFAULT_FILM_TAPS)}'. An empty string removes conditioning "
                         "entirely, which is the null control for every metadata question.")
    ap.add_argument("--film-init-encoder", default="xavier", choices=list(FILM_INITS),
                    help="init of the conv tower's FiLM projections. xavier = live from step 0 (the "
                         "historical pair: Xavier-uniform weight, N(0,0.1) bias).")
    ap.add_argument("--film-init-decoder", default="zero", choices=list(FILM_INITS),
                    help="init of the decoder's FiLM projections. zero = adaLN-zero, so the decoder "
                         "is born un-conditioned and steering is learned. CHANGING EITHER MOVES THE "
                         "GLOBAL RNG STREAM, so a non-default init is not comparable weight-for-"
                         "weight with a recorded run even at the same seed.")
    # -- heads --------------------------------------------------------------------------------------
    ap.add_argument("--head-sharing", default="shared", choices=["shared", "per_assay"],
                    help="shared = ONE small MLP reads every assay's lane (the shipped default, 1/A "
                         "the parameters). per_assay = A independent copies. The shared head assumes "
                         "the four FiLM taps have already put assay identity in the lane; that is "
                         "reasonable and unmeasured, which is why the alternative is here.")
    ap.add_argument("--head-hidden", type=int, default=0,
                    help="hidden width of each head MLP. 0 = match the decoder lane width.")
    ap.add_argument("--signal-target-transform", dest="signal_target_transform",
                    default=SIGNAL_TARGET_TRANSFORM_AUTO,
                    choices=[SIGNAL_TARGET_TRANSFORM_AUTO, *SIGNAL_TARGET_TRANSFORMS],
                    help="D30: how the Gaussian signal head's TARGET (y_pval) is transformed, INSIDE "
                         "the loss. Not the loader's job and not the same knob as the encoder's own "
                         "input transform. 'auto' (the default) derives it from the data source — "
                         "'none' under --h5, because the bake already stores arcsinh(-log10 p), and "
                         "'arcsinh' under --store, which returns the raw value. The RESOLVED value is "
                         "written to the run config; two runs are comparable on the signal head only "
                         "if it matches. INERT unless 'signal' is in --heads.")
    ap.add_argument("--heads", default=",".join(DEFAULT_HEADS),
                    help="comma-separated set of output heads. 'count' (NB over raw counts) is the "
                         "shipped model, is REQUIRED, and is the only head the mid-training monitor "
                         "scores. "
                         "'signal' adds a Gaussian (mu, var) over the arcsinh log-p-value track and "
                         "'peak' a Bernoulli over the peak calls, each supervised on the FULL-DEPTH "
                         "targets only (y_dsf == 1) and added to the NB loss with coefficient 1.0. "
                         f"Default '{','.join(DEFAULT_HEADS)}'. Either extra head changes the "
                         "objective while evaluation stays NB-only, so such a run is not a control.")
    # -- optimisation -------------------------------------------------------------------------------
    ap.add_argument("--lr-schedule", default="cosine", choices=list(LR_SCHEDULES))
    ap.add_argument("--warmup-frac", type=float, default=0.1,
                    help="fraction of total steps spent warming the LR up linearly from 0.")
    ap.add_argument("--lr-min-ratio", type=float, default=0.1,
                    help="floor of the decay, as a fraction of the peak LR. Inert under "
                         "--lr-schedule constant.")
    ap.add_argument("--clip-norm", type=float, default=CLIP_NORM,
                    help="global grad-norm clip. The pre-clip norm is logged as grad/total_preclip "
                         "and the bind rate as train/clip_bind_frac, so a threshold that has stopped "
                         "binding is visible rather than assumed.")
    ap.add_argument("--decoder-lane", type=int, default=8,
                    help="channels per assay inside the grouped deconv trunk. Held CONSTANT across "
                         "all three upsample stages rather than tapering 8->4->2->1, so the final "
                         "FiLM tap and the head both have more than one channel to act on.")
    ap.add_argument("--cell-cond", default="off", choices=["off", "id", "random"],
                    help="5th metadata row = cell identity. off = the historical 4-row model "
                         "(bit-identical, RNG included). id = the real cell type. random = a valid "
                         "id redrawn per batch: the NULL CONTROL, same channel and parameter count, "
                         "zero information (a fixed label permutation would preserve it all)")
    ap.add_argument("--meta-embed-layernorm", default="on", choices=["on", "off"],
                    help="h75: trailing LayerNorm of the metadata fusion MLP, on BOTH towers. "
                         "on = the historical architecture and a strict no-op (LayerNorm draws "
                         "nothing from the global RNG, so the stream and every golden are unchanged). "
                         "off = REMOVED, which is what h75 tests: h48/F2 measured this LayerNorm "
                         "attenuating the conditioning perturbation ~70x.")
    ap.add_argument("--meta-gain", type=float, default=1.0,
                    help="h76: FIXED, NON-LEARNABLE scalar applied to the metadata embedding `memb` "
                         "immediately before decoder.film_proj — a pure upstream gain that adds no "
                         "parameter, alters no architecture and touches no RNG. 1.0 = the control, "
                         "where the multiply is SKIPPED rather than performed with one, so the path "
                         "is bit-identical and every golden holds at 0 ULP. h76 sweeps 0.1 / 1.0 / "
                         "10.0 to ask whether the conditioning pathway regulates its output scale "
                         "back to a loss-determined set point.")
    ap.add_argument("--meta-probe", default="off", choices=list(META_PROBE_MODES),
                    help="h64: the ARM SWITCH for the covariate-gradient meter. All three arms share "
                         "one architecture, one parameter count, n_rows=4 and one seed, and differ "
                         "ONLY in the contents of metadata row 3 (run_type). "
                         "off = the true run_type and a STRICT no-op (no object is built, no RNG is "
                         "seeded, the training path is bit-identical to the pre-h64 kit). "
                         "shuffled = row 3 permuted along the BATCH axis per assay column among "
                         "non-sentinel entries: same marginal, zero association — the NEGATIVE "
                         "control. On a one-biosample-per-batch panel this permutation is the "
                         "IDENTITY; that is not hidden, it is measured as "
                         "meta_probe/shuffle_noop_frac and printed. "
                         "planted = row 3 OVERWRITTEN with a per-sample median-split bin that is "
                         "ALSO added to y_data in log space — the POSITIVE control. x_data is never "
                         "shifted, which is what makes row 3 the sole route.")
    ap.add_argument("--meta-probe-delta", type=float, default=DEFAULT_META_PROBE_DELTA,
                    help="h64: planted shift magnitude in log2 units. y_data *= 2**(+/-delta), so "
                         "the default 1.0 separates the two classes by 4x in expectation. INERT "
                         "under --meta-probe off/shuffled; recorded in the run config either way.")
    ap.add_argument("--reference", default="off", choices=["off", "on"],
                    help="h74: predict the DEVIATION from a per-assay, per-position average over T_ "
                         "biosamples, as a log-mean offset. off = the raw-signal arm and a strict "
                         "no-op (no table is built, no log_ref key exists, the decoder arithmetic is "
                         "bit-identical). THE ONLY FLAG THAT MAY DIFFER BETWEEN THE TWO ARMS.")
    ap.add_argument("--reference-path", default=None,
                    help="default: <h5 stem>.reference.h5 beside the h5")
    ap.add_argument("--reference-pseudocount", type=float, default=None,
                    help="c in log2(R + c); default 0.25 (see candi.reference)")
    ap.add_argument("--imp-weight", type=float, default=1.0,
                    help="loss = obs + w*imp. Eval scores imputation only; h74 runs w=3.")
    ap.add_argument("--unmask-frac", type=float, default=0.0,
                    help="fraction of steps run fully unmasked, drawn from a DEDICATED RNG so two "
                         "arms see identical batches in identical order. h74 runs 0.15.")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="STORE ONLY — run `candi.monitor`'s IMPUTATION dial every N epochs (0 = "
                         "off, the h73 behaviour) and keep the best imputation checkpoint. The "
                         "denoising dial and the overfitting gap are NOT on this cadence: they run "
                         "once, at the end, on the selected checkpoint (cost ruling, measured in "
                         "cruxvault/results/t30/TIMING.md). UNSET is not 0: it "
                         "resolves to 3 when the source has imputation targets to score and to 0 "
                         "when it does not. An explicit value always wins, including an explicit 0. "
                         "Under --h5 it is forced to 0 whatever you pass, loudly: the h5 path's "
                         "mid-training scorer was `eval.quick_eval` and `candi.eval` is deleted "
                         "(D15), so an h5 run trains only and is scored afterwards with "
                         "`python -m candi.bench --h5 …`.")
    ap.add_argument("--eval-regions", default=None, metavar="BED",
                    help="STORE ONLY — cut the MID-TRAINING selection scope down to a BED: every "
                         "25 bp bin of every window that lies WHOLLY inside one region, and "
                         "nothing else. UNSET is the default and is today's behaviour, every "
                         "window of every eval chromosome. It does not thin the panel — all 45 "
                         "tracks are still scored end to end — it scores them over fewer "
                         "POSITIONS. Measured: a 2.67%% scope is 5.85 min against a 94 min full "
                         "check (16x, 21%% of one epoch; not the 37x the bin count implies, "
                         "because ~206 s of a check is scope-invariant). "
                         "NOT EVERY BED IS FIT TO SELECT ON — "
                         "configs/regions/encode_pilot_hg38.bed is the eic.pilot TRAINING scope "
                         "and was MEASURED to flip checkpoint selections on gaps of 0.069 (its "
                         "bias drifts with the model, so it does not cancel in a difference); a "
                         "seeded random window set of the same size did not. See EVAL.md before "
                         "choosing one. The BED is hashed and the hash goes in "
                         "`config.eval_scope`: two runs are comparable on the mid-training curve "
                         "only if that block matches. The END-OF-RUN check on the selected "
                         "checkpoint is FULL COVERAGE whatever this says, and the positional "
                         "measures (mseprom, msegene, mseenh, the P-block) are absent "
                         "mid-training because a compacted array's index is no longer a genomic "
                         "position.")
    ap.add_argument("--early-stop-epochs", type=int, default=3,
                    help="Stop when the mid-training selection metric has not improved for MORE "
                         "than N epochs (0 = off, train the full --epochs). Counted in epochs, so "
                         "the effective resolution is --eval-every: at the default 3 the earliest "
                         "stop is 6 epochs after the best, and N < --eval-every can never fire. "
                         "Nothing is lost by stopping — the best checkpoint is written the moment "
                         "it improves and is the one scored. Added after job 57620803_0 took its "
                         "best V_ checkpoint at epoch 2 and then ran nine more GPU-hours while "
                         "V_imp_crps rose 0.5604 -> 0.5820 -> 0.5864.")
    ap.add_argument("--eval-batches-per-pair", type=int, default=4,
                    help="INERT. It set the window thinning of `eval.quick_eval`, the h5 path's "
                         "mid-training scorer, and that function was deleted with the rest of "
                         "`candi.eval` (D15). Nothing reads it now: the monitor scores every 25 bp "
                         "bin of the regime's eval chromosomes and has nothing to thin, and the h5 "
                         "path has no mid-training eval at all. Accepted and recorded in the run "
                         "config so an old submit script still parses; it changes no number.")
    ap.add_argument("--precision", default=DEFAULT_PRECISION, choices=list(PRECISIONS),
                    help=PRECISION_HELP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wandb-project", default=None,
                    help="W&B project; omit to disable. Auth comes from ~/.netrc. Set "
                         "WANDB_MODE=offline in the environment to buffer and `wandb sync` later.")
    ap.add_argument("--wandb-run-name", default=None, help="defaults to --tag")
    return ap


def main():
    a = build_parser().parse_args()
    use_offset = a.offset in ("on", "offset_on")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.tag or f"candi_{a.deconv_norm}_ep{a.epochs}_seed{a.seed}"
    meta_ln = (a.meta_embed_layernorm == "on")
    if a.store:
        print(f"[run] data=STORE regime_file={a.store} (masking regime={a.regime}) — num_cells=0 "
              "(cell_cond refused, D16), --reference refused, and NO final evaluate(): this run "
              "json gets none of the h5-era M1/M2/M3/S14 keys. The MID-TRAINING MONITOR does run "
              "when the regime "
              "declares `eval_pairs`: every 25 bp bin of the regime's eval chromosomes, the "
              "imputation dial ONLY, which is what selects the best checkpoint. The denoising dial "
              "and the gap between them — the overfitting alarm — run ONCE at the end, on the "
              "selected checkpoint, because two dials on every check cost more than the run can "
              "afford (cruxvault/results/t30/TIMING.md). With no pairs declared "
              "--eval-every resolves to off. See STORE.md.", flush=True)
    print(f"[run] deconv_norm={a.deconv_norm} decoder_lane={a.decoder_lane} "
          f"attn_depth={a.n_transformer_layers} conv_norm={a.conv_norm} "
          f"film_taps={a.film_taps or '(none)'} heads={a.heads}\n"
          f"[run] tag={tag} device={device} full_coverage={a.full_coverage} "
          f"eval_max_batches={a.eval_max_batches or 'ALL'} reference={a.reference} "
          f"imp_weight={a.imp_weight} unmask_frac={a.unmask_frac} "
          f"eval_every={'auto' if a.eval_every is None else (a.eval_every or 'OFF')} "
          f"meta_embed_layernorm={a.meta_embed_layernorm} "
          f"optimizer={a.optimizer} weight_decay={a.weight_decay} trunk_wd={a.trunk_wd} "
          f"meta_probe={a.meta_probe} meta_probe_delta={a.meta_probe_delta} "
          f"meta_gain={a.meta_gain} precision={a.precision} "
          f"signal_target_transform={a.signal_target_transform}", flush=True)
    if a.precision != DEFAULT_PRECISION:
        print(f"[run] precision={a.precision}: the TRAINING forward is autocast; the metadata "
              "embedders, every FiLM tap, LaneNorm, the NB head and the whole of eval stay fp32. "
              "Expect MEMORY, not speed — a null wall-clock delta is the expected result here.",
              flush=True)
    t0 = time.time()
    res = train_and_eval(
        h5_path=a.h5, store=a.store, out_dir=out_dir, regime=a.regime, epochs=a.epochs,
        signal_target_transform=a.signal_target_transform,
        steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, use_offset=use_offset, dsf_sampling=a.dsf_sampling, device=device,
        seed=a.seed, embed_dim=a.embed_dim, dropout=a.dropout,
        n_transformer_layers=a.n_transformer_layers,
        depth_center=a.depth_center, d_model=a.d_model, nhead=a.nhead, p_full_assay=a.p_full_assay,
        mask_fraction=a.mask_fraction, eval_batch_size=a.eval_batch_size,
        eval_max_batches=(a.eval_max_batches or None), fg_frac=a.fg_frac, n_boot=a.n_boot,
        eval_budget=a.eval_budget, m3_regions=a.m3_regions, include_deprecated=a.include_deprecated,
        log_every=200, full_coverage=a.full_coverage,
        decoder_lane=a.decoder_lane, deconv_norm=a.deconv_norm,
        n_cnn_layers=a.n_cnn_layers, conv_kernel_size=a.conv_kernel_size, pool_size=a.pool_size,
        expansion_factor=a.expansion_factor, n_deconv_layers=a.n_deconv_layers,
        deconv_upsample=a.deconv_upsample, deconv_kernel_size=a.deconv_kernel_size,
        conv_norm=a.conv_norm, transformer_norm=a.transformer_norm,
        transformer_norm_placement=a.transformer_norm_placement, attn_qk_norm=a.attn_qk_norm,
        film_taps=a.film_taps, film_init_encoder=a.film_init_encoder,
        film_init_decoder=a.film_init_decoder,
        head_sharing=a.head_sharing, head_hidden=a.head_hidden, heads=a.heads,
        lr_schedule=a.lr_schedule, warmup_frac=a.warmup_frac, lr_min_ratio=a.lr_min_ratio,
        clip_norm=a.clip_norm,
        cell_cond=a.cell_cond, ckpt_path=str(out_dir / f"{tag}.ckpt"),
        wandb_project=a.wandb_project, wandb_run_name=(a.wandb_run_name or tag),
        reference=a.reference, reference_path=a.reference_path,
        reference_pseudocount=a.reference_pseudocount, imp_weight=a.imp_weight,
        unmask_frac=a.unmask_frac, eval_every=a.eval_every, eval_regions=a.eval_regions,
        eval_batches_per_pair=a.eval_batches_per_pair, early_stop_epochs=a.early_stop_epochs,
        meta_embed_layernorm=meta_ln,
        optimizer=a.optimizer, trunk_wd=a.trunk_wd, meta_gain=a.meta_gain,
        meta_probe=a.meta_probe, meta_probe_delta=a.meta_probe_delta,
        precision=a.precision)
    res["config"]["tag"] = tag
    res["wall_s"] = round(time.time() - t0, 1)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    # HISTORICAL BANNER, AND THE ONLY LIVE BRANCH SINCE D15. `M1`/`M2`/`M3` are the h5-era
    # `candi.eval` keys; no run json carries them any more, so this test always takes the first
    # branch and the second is kept only as the record of what the banner used to print.
    if "M1" not in res:
        # t23 store path: there is no evaluation, so there is nothing to summarise but the loss.
        losses = res.get("train_losses") or [float("nan")]
        print(f"[{tag}] STORE run, no eval: {len(losses)} steps, "
              f"first nll={losses[0]:.3f} last nll={losses[-1]:.3f} wall={res['wall_s']}s",
              flush=True)
        return
    m1, m2 = res["M1"], res["M2"]
    print(f"[{tag}] imp_spear={_g(m1['imp'], 'spearman_raw', 'spearman'):.3f} "
          f"den_spear={_g(m1['den'], 'spearman_raw', 'spearman'):.3f} "
          f"eff_rank={_g(m1, 'encoder_eff_rank_perpos', 'encoder_eff_rank'):.2f} | "
          f"M2 total_slope={_g(m2['depth'], 'median_total_slope'):.3f} | "
          f"M3 ratio={res['M3']['ratio']:.3f} wall={res['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
