"""monitor — the mid-training dial: whole-chromosome imputation every N epochs, and denoising once.

`bench <- monitor <- train`. This module imports `candi.bench` and is imported BY `candi.train`.
It must never import `candi.train`: that would close the cycle the bench boundary exists to keep
open, and `tests/test_monitor.py` pins it in a subprocess exactly as
`tests/test_bench_harness.py` pins the bench half.

WHAT IT REPLACED, AND WHY. `eval.quick_eval` — deleted with the rest of `candi.eval` (D15) —
scored a THINNED sample of windows: whole pair
cycles spread across the eval chromosome, `--eval-batches-per-pair` of them. A sampled number
selects a checkpoint on a draw. This scores every 25 bp bin of every eval chromosome the regime
declares, through `bench.harness.stream_tracks`, so the per-track rows below are real scores and
the macro roll-up is the panel's, not a sample's.

**TWO DIALS, AND ONLY ONE OF THEM SELECTS.**

- `impute` — the regime's declared `eval_pairs`, prompt with the input cell, score against the
  target cell's tracks. This is the one that drives best-checkpoint selection.
- `denoise` — the same input cells, self-paired: read a cell and score it against itself. WATCH
  ONLY. It never selects anything.

The gap between them is the overfitting alarm. Denoising is the task the model is trained on and
imputation is the task it is asked to generalise to, so a run whose denoise score keeps improving
while its impute score stalls is memorising the cells it has seen. That difference is emitted per
metric rather than leaving a reader to subtract two banners.

**THE TWO DIALS DO NOT RUN AT THE SAME CADENCE, AND THAT IS A COST RULING RATHER THAN AN
OVERSIGHT.** Mid-training checks run the IMPUTE dial ALONE. The denoise dial and the gap are
computed ONCE, at the end of the run, on the checkpoint that was actually selected. The measurement
behind the ruling is `cruxvault/results/t30/TIMING.md`: one whole-chr21 impute pass over the 26
declared pairs is 1254 s, which at `--eval-every 3` already sits just inside the PI's 20 % budget on
its own. The denoise dial scores ~2.3x as many tracks as the impute dial on the same panel — it
self-pairs every prompt cell and every assay that cell carries — so running both on every check
spends most of the budget's multiple, not a fraction of it. Read TIMING.md for the arithmetic and
its caveats; nothing here re-derives it.

`check(..., kinds=("impute",))` is what the training hook calls, and `final_check()` is the
end-of-run one. AN IMPUTE-ONLY ROW CARRIES NO `denoise` BLOCK AND NO `gap` — the alarm is a
difference, and a difference with one side missing is not a smaller alarm, it is no alarm. A reader
of `eval_curve` therefore sees the selection statistic per check and the alarm exactly once, on the
`final_check` row, where the model it describes is the one the run shipped.

Neither dial picks a biosample. `StoreSource` reads the pool off the regime — with `eval_pairs`
declared, `StoreDataset.__init__` sets the pool to `regime.eval_inputs`, so `pairs("impute")` is
the declared pairs and `pairs("denoise")` is those same input cells self-paired. Which cells those
are is the regime file's business alone. `configs/regime.eic_val.json` names only `V_` targets, so
a run pointed at it never touches `B_`; a run pointed at something else touches whatever that file
declares, and this module neither knows nor guesses the difference.

**THE COUNT ARM ONLY, AND THAT IS A RULING RATHER THAN AN OMISSION.** Two reasons, and neither is a
defect in `candi.bench`. First the TIERS: every key this module reports — `POINT_KEYS`, `DIST_KEYS`,
`PEAK_KEYS` and the selection metric `SELECTION_KEY` — is a count-arm key, and `crps` on the count
arm is what selects the checkpoint. A pval arm would add a second set of numbers that decides
nothing mid-training. Second the COST: the per-track seconds measured at the bottom of this
docstring would be paid again for a second arm, inside the budget the cadence ruling above already
spends on the impute dial alone. So a caller who asks for the other arm is refused by name.

**THE PVAL ARM IS SCOREABLE — JUST NOT HERE.** It used to be unscoreable, and that history is worth
one paragraph so the refusal is not read as still meaning it. On a store the Gaussian head is
trained against `arcsinh(-log10 p)` while the truth arrives as raw `-log10 p`, and `bench` once
compared the two untransformed. It no longer does: `bench.harness.score_track` inverts the
prediction back into `-log10 p` before the pval arm's benchmark keys and records `pred_space` on the
row. The end-of-run `python -m candi.bench` therefore scores that arm correctly, and this module
stays count-arm-only for the tier ruling and the cost above.

**THE `nll` TIER IS THE ONE FAMILY HERE THAT IS NOT A COUNT-ARM COMPARISON.** `loss_block` is not a
comparison at all — it is the objective, and it bends `y_pval` into the head's own space
(`signal_target_transform`, D30) before taking a likelihood. So `gaussian_nll` off a store is the
number the training loop would print, in TRANSFORMED space, with the transform recorded beside it;
and it is emitted into every arm because a loss has no arm.

**THE LOSS TIER IS WHY THIS MODULE IS THE `val_loss`.** NB, Gaussian and Bernoulli NLL are the three
terms the objective is built from; the training loop logs them every step as `train/nll` and its
per-head obs/imp terms, `candi.bench`'s CLI emits them as the test loss, and `check()` emits them
here on both dials. The same formula on three populations — train batches, the eval panel
mid-training, the eval panel at the end — is what makes the three numbers comparable at all.
Selection is UNCHANGED: `SELECTION_KEY` is still `crps`.

COST, MEASURED RATHER THAN ASSERTED. Per track, on one numpy core over a chr21-length array
(1,868,399 bins at 25 bp): `eic.score_track` 0.55 s, `distributional.nb_suite` 7.6 s (of which
`oracle_scale` alone is 4.5 s and has no dial — `c_index_pairs` is not where the time goes),
`binary.binary_suite` 0.68 s. Call it ~8.8 s of CPU per track, times however many tracks the two
dials produce. Memory is `stream_tracks`'s: it buffers eight float32 vectors per TARGET ASSAY of
the open pair, which is ~60 MB per assay on chr21 — small for an impute pair with three targets,
about 1.2 GB for a denoise pair carrying twenty assays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The point tier. Every method can produce these, probabilistic or not, which is the whole reason
#: they are a separate tier from the one below: a leaderboard row for a non-probabilistic baseline
#: has these and nothing else. Spelled as `bench.eic` spells them.
POINT_KEYS: Tuple[str, ...] = ("mse", "gwcorr", "gwspear", "mse1obs")

#: The distribution tier — CANDI-likes only, and never `crps` on its own. `AGENTS.md` §7.2: a raw
#: CRPS confounds the shape of the predicted distribution with its scale, so the oracle-scaled
#: value and the difference between the two travel with it or none of them do.
DIST_KEYS: Tuple[str, ...] = ("crps", "crps_oracle_scaled", "scale_error")

#: The peak tier. `auprc` is how `bench.binary` spells it — NOT `aupr`. `peak_base_rate` rides
#: along because an AUPRC without the prevalence it is measured against is uninterpretable.
PEAK_KEYS: Tuple[str, ...] = ("auprc", "peak_base_rate")

#: The LOSS tier — the val-loss half of `train_loss / val_loss / test_loss`, from
#: `bench.harness.loss_block`. Keyed by the HEAD that produces each term, because this is the one
#: tier whose keys do not switch on and off together: a `count,peak` model owes an `nb_nll` and a
#: `bernoulli_nll` and no `gaussian_nll`, which no whole-tier gate can express. That is also the
#: answer to why it is a tier of its own rather than three keys folded into the tiers above —
#: `dist` is gated on the count head alone and `peak` on the peak head alone, so folding
#: `gaussian_nll` in would need a fourth tier for the signal head anyway, and `nb_nll` sitting in
#: `dist` would put a likelihood among the CRPS split that `AGENTS.md` §7.2 governs.
NLL_KEYS_BY_HEAD: Dict[str, Tuple[str, ...]] = {
    "count": ("nb_nll",), "signal": ("gaussian_nll",), "peak": ("bernoulli_nll",)}

#: Tier order for `tiered_keys`. The loss tier is last because it is the newest and because the
#: banner reads point -> distribution -> peak -> loss.
TIER_ORDER: Tuple[str, ...] = ("point", "dist", "peak", "nll")

#: The arm this module scores. See the module docstring and `assert_prediction_space`.
COUNT_ARM = "count"

#: The selection metric, read off the impute dial's macro roll-up. Lower is better.
SELECTION_KEY = "crps"


def model_heads(model) -> Tuple[str, ...]:
    """Which output heads this model actually built, read off the decoder that built them.

    `decoder.heads` is the constructor's own parsed tuple (`decoder.py::SymmetricDecoder.__init__`
    stores `parse_heads(heads)`), so this is the model answering for itself rather than a flag
    threaded down from a command line and possibly out of date. A model that predates the attribute
    is a count-only model, which is what `DEFAULT_HEADS` has always been.
    """
    dec = getattr(model, "decoder", None)
    return tuple(getattr(dec, "heads", ("count",)))


def tiers(model) -> Dict[str, Tuple[str, ...]]:
    """The metric keys this model may report, by tier. Detected, never assumed.

    The peak tier is the one that needs saying out loud. `bench.harness.stream_tracks` fills
    `peak_score` from `sigmoid(peak_logit)` when the peak head is built AND FROM THE NB MEAN WHEN
    IT IS NOT, so `binary.binary_suite` returns a perfectly finite `auprc` for a count-only model —
    a number that ranks bins by predicted coverage and has nothing to do with a peak head. Reading
    it as "the peak metric" would be reading a fallback as a result, so the key is suppressed here
    rather than trusted downstream.

    The `nll` tier is detected the same way and for a sharper version of the same reason: `bench`
    omits `bernoulli_nll` outright without a peak head — there is no fallback to suppress, because
    a BCE of the NB mean is not merely misleading, it is arithmetic on the wrong kind of number.
    """
    heads = model_heads(model)
    out: Dict[str, Tuple[str, ...]] = {"point": POINT_KEYS}
    if "count" in heads:
        out["dist"] = DIST_KEYS
    if "peak" in heads:
        out["peak"] = PEAK_KEYS
    nll = tuple(k for h, ks in NLL_KEYS_BY_HEAD.items() if h in heads for k in ks)
    if nll:
        out["nll"] = nll
    return out


def tiered_keys(model) -> Tuple[str, ...]:
    """Every key `check()` will report, flattened, in tier order."""
    t = tiers(model)
    return tuple(k for tier in TIER_ORDER for k in t.get(tier, ()))


def assert_prediction_space(arm: str, *, signal_target_transform: str) -> None:
    """Hold the mid-training monitor to the count arm. A SCOPE RULING, not a space bug.

    THE NAME IS OLDER THAN THE REASON. This function was written when the pval arm really was
    unscoreable — the Gaussian head predicting `arcsinh(-log10 p)` against a raw `-log10 p` truth —
    and it refused the arm on those grounds. `bench.harness.score_track` now inverts the prediction
    back into `-log10 p` (`bench.distributional.invert_signal_prediction`) and stamps `pred_space`
    on the row, so that mismatch is gone and the end-of-run `python -m candi.bench` scores the pval
    arm correctly.

    WHAT STILL HOLDS, AND WHY THE REFUSAL STAYS. Every key this module reports is a count-arm key
    (`POINT_KEYS`, `DIST_KEYS`, `PEAK_KEYS`) and `SELECTION_KEY` selects on the count arm, so a pval
    arm here would add numbers that decide nothing; and the second arm costs the per-track seconds
    in the module docstring again, inside a budget the impute dial already fills. Mid-training is
    for selection; the full picture is the end-of-run bench's job.

    `signal_target_transform` is still taken and still reported, because it is what the caller
    needs to know to read `gaussian_nll` — that tier is in TRANSFORMED space by design.

    Raised at construction, not at the first check, so a run configured for an arm this module does
    not report dies on the submit line rather than after an hour of training.
    """
    if arm == COUNT_ARM:
        return
    raise ValueError(
        f"the mid-training monitor scores the {COUNT_ARM!r} arm only; got arm={arm!r}. This is a "
        "SCOPE ruling, not a space bug: every tier here is a count-arm tier and SELECTION_KEY "
        "selects on the count arm, and a second arm costs the whole per-track budget again. The "
        "pval arm IS correctly scored — in -log10 p, with the prediction inverted out of "
        f"signal_target_transform={signal_target_transform!r} by harness.score_track — by the "
        "end-of-run `python -m candi.bench`. Run it there. The one number here that is not a "
        "count-arm comparison is gaussian_nll, which is the training loss and stays in the "
        "transformed space it was trained in."
    )


@dataclass(frozen=True)
class DialResult:
    """One dial's whole answer: the per-track rows AND the roll-up over them.

    Both, deliberately. At whole-chromosome coverage a per-track number is a real score rather than
    a draw, so it is worth keeping — a macro that moved because one assay collapsed and a macro
    that moved because everything drifted are the same number until you can see the rows.
    """

    kind: str
    macro: Dict[str, float] = field(default_factory=dict)
    per_track: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_tracks: int = 0
    wall_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"macro": dict(self.macro), "per_track": {k: dict(v) for k, v in
                                                         self.per_track.items()},
                "n_tracks": int(self.n_tracks), "wall_s": float(self.wall_s)}


def _select(row: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, float]:
    """The tiered subset of one score row, finite values only."""
    out: Dict[str, float] = {}
    for k in keys:
        if k in row:
            v = float(row[k])
            if np.isfinite(v):
                out[k] = v
    return out


class Monitor:
    """The mid-training dial. Built once per run, called once per `--eval-every` epochs.

    `kinds` is what this monitor MAY score, not what every call does score. Under the cadence ruling
    the training hook asks for `("impute",)` per call and `final_check()` asks for all of them once,
    so the constructor default stays both dials and the caller narrows it per check.

    OPENED ONCE, OUTSIDE THE HOOK, for the reason t28 gives about the dataset it replaces: one
    source is what pins one window plan across every epoch, and selection compares epoch 6 against
    epoch 12, which is only a paired comparison if both saw the same positions. Re-opening per check
    would also re-read the store manifest and re-plan the windows once per check for nothing.

    The chromosomes are the regime's own `eval_chroms` — `configs/regime.eic_val.json` says
    `["chr21"]`, which is the settled coverage. `chroms=` overrides it for tests; production passes
    nothing and lets the regime be the authority.
    """

    def __init__(self, *, store, device, chroms: Optional[Sequence[str]] = None,
                 batch_windows: int = 4, seed: int = 0, c_index_pairs: int = 200_000,
                 signal_target_transform: str = "none",
                 kinds: Sequence[str] = ("impute", "denoise"),
                 arm: str = COUNT_ARM,
                 eval_regions: Optional[Any] = None,
                 log_fn: Optional[Callable[[int, Mapping[str, Any]], None]] = None) -> None:
        from candi.bench import annotations as ann
        from candi.bench.harness import KINDS, open_source

        # Before anything is opened: an arm this monitor cannot honestly score must fail on the
        # submit line, not after the store is read and the first epoch is burnt.
        assert_prediction_space(arm, signal_target_transform=signal_target_transform)
        bad = [k for k in kinds if k not in KINDS]
        if bad:
            raise ValueError(f"kinds must be drawn from {KINDS}; got {bad}")
        if "impute" not in kinds:
            raise ValueError(
                "the impute dial is what selects the checkpoint, so it is not optional. `denoise` "
                "is the one that can be dropped — it is watch-only.")
        self.arm = arm
        self.kinds = tuple(kinds)
        self.device = device
        self.batch_windows = int(batch_windows)
        self.seed = int(seed)
        self.c_index_pairs = int(c_index_pairs)
        self.signal_target_transform = str(signal_target_transform)
        self.log_fn = log_fn
        self.source = open_source(store=store, eval_regions=eval_regions,
                                  **({"chroms": list(chroms)} if chroms else {}))
        # Loaded once. These are pinned repo assets (`bench.annotations.verify_assets`), so this is
        # a read of files already on disk rather than a new dependency of the training loop.
        self._gene = ann.gene_annotations()
        self._enh = ann.enhancer_annotations()

    # -- one dial -----------------------------------------------------------------------------

    def score(self, model, *, kind: str) -> DialResult:
        """Stream every track of one dial and score each one. No sampling anywhere in here.

        `signal_target_transform` is the run's OWN resolved value, held since construction, and it
        reaches `loss_block` alone. That is what makes `gaussian_nll` here the training loop's
        number rather than a cross-space comparison — see the exception in the module docstring.
        """
        from candi.bench.harness import macro_mean, score_track, stream_tracks

        t0 = time.time()
        keys = tiered_keys(model)
        per_track: Dict[str, Dict[str, float]] = {}
        full: Dict[str, Dict[str, Dict[str, object]]] = {}
        for rec in stream_tracks(model, self.source, self.device, kind=kind,
                                 batch_windows=self.batch_windows):
            arms = score_track(rec, gene_annotations=self._gene, enh_annotations=self._enh,
                               seed=self.seed, c_index_pairs=self.c_index_pairs,
                               signal_target_transform=self.signal_target_transform)
            full[rec.key] = arms
            row = _select(arms[self.arm], keys)
            row["assay"] = rec.assay
            per_track[rec.key] = row
        macro = _select(macro_mean(full, self.arm, kind=kind), keys)
        return DialResult(kind=kind, macro=macro, per_track=per_track,
                          n_tracks=len(per_track), wall_s=round(time.time() - t0, 1))

    # -- both dials, plus the alarm -----------------------------------------------------------

    def check(self, model, *, epoch: int, step: int,
              kinds: Optional[Sequence[str]] = None,
              final: bool = False, selected: Optional[str] = None) -> Dict[str, Any]:
        """Score the requested dials and return one row of the run json's `eval_curve`.

        `kinds=None` means every dial this monitor was opened with — which is what `final_check`
        wants. The TRAINING HOOK passes `kinds=("impute",)` instead, by the cadence ruling in the
        module docstring: at the measured cost of a whole-chromosome pass the denoise dial cannot
        ride along on every check. A subset must still contain `impute`, for the same reason the
        constructor demands it — a check that does not produce the selection metric selects nothing
        — and it may only name dials this monitor actually opened.

        `model.train()` is restored on the way out, whatever it was on the way in. `stream_tracks`
        calls `model.eval()` itself and never puts it back, and the two training loops disagree
        about whose job that is — the sampled path calls `model.train()` after the hook, the
        full-coverage path does not — so the restore lives here and both call sites behave the same.
        """
        want = self.kinds if kinds is None else tuple(kinds)
        unopened = [k for k in want if k not in self.kinds]
        if unopened:
            raise ValueError(
                f"this monitor was opened with kinds={self.kinds}; cannot check {unopened}")
        if "impute" not in want:
            raise ValueError(
                "the impute dial is what selects the checkpoint, so it is not optional — not at "
                "construction and not per call. `denoise` is the one that can be dropped.")
        was_training = bool(model.training)
        t0 = time.time()
        try:
            dials = {k: self.score(model, kind=k) for k in want}
        finally:
            model.train(was_training)

        imp = dials["impute"]
        row: Dict[str, Any] = {
            "epoch": int(epoch), "step": int(step),
            "wall_s": round(time.time() - t0, 1),
            "arm": self.arm,
            # D30 — recorded beside the numbers, because `gaussian_nll` in one space and the same
            # key in another are both finite and are not the same quantity.
            "signal_target_transform": self.signal_target_transform,
            "heads": list(model_heads(model)),
            "tiers": {t: list(ks) for t, ks in tiers(model).items()},
            "chroms": list(self.source.eval_chroms),
            # t89 — WHICH POSITIONS this number was measured over, on the row itself. `chroms` alone
            # no longer answers it: a `full` check and a `regions` check name the same three
            # chromosomes and score 37x different amounts of them, and a curve that silently
            # changed scope mid-run would read as the model moving.
            "eval_scope": self.source.scope(),
        }
        for k, d in dials.items():
            row[k] = d.as_dict()
        if "denoise" in dials:
            den = dials["denoise"].macro
            row["gap"] = {k: float(imp.macro[k] - den[k]) for k in imp.macro if k in den}
        sel = imp.macro.get(SELECTION_KEY, float("nan"))
        row["selection"] = {"metric": SELECTION_KEY, "value": float(sel), "kind": "impute",
                            "n_tracks": imp.n_tracks}
        if final:
            # Set BEFORE the dashboard call: `_wandb_payload` reads these to give the end-of-run row
            # its own key namespace, and a row that arrived under `monitor/impute/...` would land in
            # the mid-training series while describing a different set of weights.
            row["final"] = True
            row["selected"] = str(selected) if selected is not None else "best"
        if self.log_fn:
            self.log_fn(step, _wandb_payload(row))
        return row

    def final_check(self, model, *, epoch: int, step: int,
                    selected: str = "best") -> Dict[str, Any]:
        """The end-of-run check: every dial this monitor holds, and therefore the gap.

        Run ONCE, on the checkpoint the run actually selected — not on whatever the last epoch left
        in memory, unless the last epoch is what was selected. `selected` records which of the two
        it was, because the overfitting alarm is a statement about a specific set of weights and
        "the best checkpoint" and "the last checkpoint" are not the same weights.

        `epoch` is the epoch the SELECTED weights came from, so this row is placeable on the curve
        rather than floating after it.
        """
        return self.check(model, epoch=epoch, step=step, kinds=self.kinds,
                          final=True, selected=selected)

    def close(self) -> None:
        self.source.close()


def _wandb_payload(row: Mapping[str, Any]) -> Dict[str, float]:
    """The flat scalars. Per-track rows stay out of the dashboard, in the json.

    The end-of-run row gets its OWN namespace, `monitor_final/...`. It is logged at the last
    training step, but it describes the SELECTED checkpoint rather than the model that step left
    behind, so folding it into `monitor/impute/crps` would put a point from different weights on the
    end of the mid-training series and make the curve read as if it had moved one last time.
    """
    pre = "monitor_final" if row.get("final") else "monitor"
    out: Dict[str, float] = {f"{pre}/epoch": float(row["epoch"])}
    for kind in ("impute", "denoise"):
        if kind not in row:
            continue
        for k, v in row[kind]["macro"].items():
            out[f"{pre}/{kind}/{k}"] = float(v)
        out[f"{pre}/{kind}/n_tracks"] = float(row[kind]["n_tracks"])
    for k, v in row.get("gap", {}).items():
        out[f"{pre}/gap/{k}"] = float(v)
    out[f"{pre}/selection"] = float(row["selection"]["value"])
    out[f"{pre}/wall_s"] = float(row["wall_s"])
    return out


def format_check(row: Mapping[str, Any], *, best: Optional[Mapping[str, Any]] = None) -> str:
    """The one-line banner. Names the dial each number came from — neither is 'the' score.

    The end-of-run row says so in the tag, and names the checkpoint it scored. A reader scrolling a
    log has to be able to tell the one row carrying the overfitting alarm from the checks that only
    selected, without counting epochs.
    """
    imp, sel = row["impute"], row["selection"]
    tag = (f"[monitor@final:{row.get('selected', 'best')} ckpt, ep{row['epoch']}]"
           if row.get("final") else f"[monitor@ep{row['epoch']}]")
    # The SCOPE is in the banner because the banner is what a reader compares across epochs, and a
    # `regions` number and a `full` number are not two measurements of one quantity. Printed only
    # when it is not the full scope, so an untouched run's log is byte-for-byte what it was.
    scope = row.get("eval_scope") or {}
    where = ",".join(row["chroms"])
    if scope.get("name") not in (None, "full"):
        where += f" @{scope['name']} {scope.get('fraction', float('nan')):.2%}"
    head = (f"{tag} impute {sel['metric']}={sel['value']:.4f} "
            f"(n={imp['n_tracks']} tracks, {where})")
    if "denoise" in row:
        den = row["denoise"]
        gap = row.get("gap", {}).get(sel["metric"])
        head += (f" | denoise {sel['metric']}="
                 f"{den['macro'].get(sel['metric'], float('nan')):.4f} "
                 f"(n={den['n_tracks']}, watch-only)")
        if gap is not None:
            head += f" | end-of-run overfitting alarm: gap={gap:+.4f}"
    if best is not None:
        head += (" [*BEST*]" if best.get("epoch") == row["epoch"]
                 else " [best {:.4f} @ep{}]".format(best.get("crps", float("nan")),
                                                    best.get("epoch", -1)))
    return head + f" {row['wall_s']}s"


def resolve_eval_every(requested: Optional[int], *, eval_pairs_declared: bool) -> int:
    """`--eval-every` when it was not given: the settled cadence when there is anything to score.

    An explicit value always wins, including an explicit 0. Unset means 3 — the PI's settled
    cadence — when the source declares `eval_pairs`, and 0 when it does not, because a pair-less
    regime has no imputation targets and the guard in `train_and_eval` would only turn it off again
    one line later. That guard is unchanged: an EXPLICIT non-zero value against a pair-less regime
    still hits it and still gets told why.
    """
    if requested is not None:
        return int(requested)
    return 3 if eval_pairs_declared else 0
