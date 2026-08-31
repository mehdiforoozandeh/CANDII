"""The mid-training monitor, against a REAL store built on disk by `candi.store.writer`.

Nothing here is mocked, for the reason `tests/test_bench_harness.py` gives: the monitor's whole job
is to be right about a corpus, and a test against a stand-in checks the code against a picture of
the thing rather than the thing.

Five properties, each of them a way the monitor could be silently wrong:

1. **Two dials, both real, on two cadences.** Imputation and denoising each produce per-track rows,
   and the gap between their roll-ups is the difference the overfitting alarm reports — but only on
   the END-OF-RUN row. A mid-training check runs the impute dial alone (the PI's cost ruling on the
   wall-clocks in `cruxvault/results/t30/TIMING.md`), so an absent `gap` there is the design and a
   present one is the bug.
2. **The tiers switch on the model.** A count-only model must not report `auprc` — `stream_tracks`
   fills `peak_score` from the NB mean when there is no peak head, so `binary_suite` returns a
   finite number regardless and reading it would be reading a fallback as a result.
3. **The space assertion fires.** Asking for the pval arm is refused by name, at construction.
4. **W&B is off unless asked for.**
5. **The dependency direction holds.** `bench <- monitor <- train`, checked in a subprocess.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from candi import monitor as MON
from candi.model import build_model

from tests.test_store_reader import ASSAYS, make_store
from tests.test_store_regime import CTX, regime_dict

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

DEVICE = torch.device("cpu")

#: `T_aa` prompts and `V_aa` is the truth, and `V_aa` carries an assay `T_aa` does not. Both halves
#: are load-bearing: an assay BOTH cells have is not an imputation target (the encoder can read it
#: off the prompt), and an assay NEITHER has cannot be one. Copied in spirit from
#: `tests/test_store_eval_units.py`, which needed the same shape for the same reason.
EVAL_TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq"),
    "V_aa": ("ATAC-seq", "DNase-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "DNase-seq"),
    "V_bb": ("ATAC-seq", "DNase-seq", "H3K4me3"),
}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("monstore"), tracks=EVAL_TRACKS)


@pytest.fixture(scope="module")
def regime_file(tmp_path_factory, store) -> Path:
    """A regime that DECLARES its pairs — without them there is no imputation dial to monitor."""
    p = tmp_path_factory.mktemp("monregime") / "regime.json"
    p.write_text(json.dumps(regime_dict(
        store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
        kinds=["counts", "peaks", "pval"],
        eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]]), indent=2), encoding="utf-8")
    return p


def _model(heads=("count",)):
    torch.manual_seed(0)
    m = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                    depth_center=24.25, d_model=32, embed_dim=8, n_transformer_layers=1,
                    heads=heads)
    m.eval()
    return m


@pytest.fixture(scope="module")
def count_model():
    return _model(("count",))


@pytest.fixture(scope="module")
def peak_model():
    return _model(("count", "peak"))


def _monitor(regime_file, **kw):
    return MON.Monitor(store=regime_file, device=DEVICE, batch_windows=2, **kw)


# ---------------------------------------------------------------------------
# 1 — two dials, both producing real per-track rows, on two cadences
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows(regime_file, count_model):
    """One mid-training check and one end-of-run check, from ONE monitor.

    Shared because they are the expensive fixtures in this file, and produced from the same monitor
    on purpose: the cadence ruling is about which dials a CALL runs, not about opening two different
    monitors, so a test that built a fresh monitor per cadence would be checking something else.
    """
    m = _monitor(regime_file)
    try:
        mid = m.check(count_model, epoch=2, step=120, kinds=("impute",))
        final = m.final_check(count_model, epoch=2, step=120, selected="best")
    finally:
        m.close()
    return {"mid": mid, "final": final}


@pytest.fixture(scope="module")
def row(rows):
    """The END-OF-RUN row — the only one carrying both dials and the gap."""
    return rows["final"]


@pytest.fixture(scope="module")
def mid_row(rows):
    """A MID-TRAINING row: the impute dial alone."""
    return rows["mid"]


def test_the_mid_training_row_carries_the_impute_dial_alone_and_the_final_one_carries_both(
        mid_row, row) -> None:
    """The cadence ruling, as a shape.

    Mid-training is impute-only because a whole-chromosome denoise pass does not fit the budget
    (`cruxvault/results/t30/TIMING.md`), and the overfitting alarm is a DIFFERENCE — with one side
    missing it is not a weaker alarm, it is no alarm, so the key is absent rather than empty.
    """
    assert "impute" in mid_row and mid_row["impute"]["n_tracks"] > 0
    assert "denoise" not in mid_row, "the denoise dial rode along on a mid-training check"
    assert "gap" not in mid_row, "an impute-only row cannot carry the overfitting alarm"
    assert not mid_row.get("final")
    # Selection still works off it — that is the whole reason impute is the one that stays.
    assert mid_row["selection"]["kind"] == "impute"
    assert np.isfinite(mid_row["selection"]["value"])

    # ... and the end-of-run row is the one that has everything.
    assert row["final"] is True and row["selected"] == "best"
    assert "denoise" in row and row["denoise"]["n_tracks"] > 0
    assert row["gap"], "the end-of-run check produced no overfitting alarm"


def test_the_impute_dial_cannot_be_dropped_per_call_either(regime_file, count_model) -> None:
    """The constructor guard has a twin: a call that drops impute would select nothing."""
    m = _monitor(regime_file)
    try:
        with pytest.raises(ValueError, match="not optional"):
            m.check(count_model, epoch=0, step=0, kinds=("denoise",))
        # And a dial this monitor never opened cannot be asked for on a whim.
        m2 = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000)
        try:
            with pytest.raises(ValueError, match="opened with"):
                m2.check(count_model, epoch=0, step=0, kinds=("impute", "denoise"))
        finally:
            m2.close()
    finally:
        m.close()


def test_both_dials_produce_per_track_rows(row) -> None:
    for kind in ("impute", "denoise"):
        dial = row[kind]
        assert dial["n_tracks"] > 0, f"the {kind} dial scored nothing"
        assert len(dial["per_track"]) == dial["n_tracks"]
        for key, track in dial["per_track"].items():
            assert track["assay"] in ASSAYS
            # Whole-coverage scoring, so these are scores rather than draws — every point-tier key
            # must be there, not merely most of them.
            for k in MON.POINT_KEYS:
                assert k in track, f"{kind}/{key} is missing {k}"
    # The two dials must be telling us about different work: the denoise keys carry the fourth
    # field `track_key` appends, so they can never collide with an imputation key.
    assert not set(row["impute"]["per_track"]) & set(row["denoise"]["per_track"])
    assert all(k.endswith("|denoise") for k in row["denoise"]["per_track"])


def test_the_gap_is_impute_minus_denoise_on_every_tiered_key(row) -> None:
    imp, den = row["impute"]["macro"], row["denoise"]["macro"]
    assert row["gap"], "the overfitting alarm reported nothing"
    for k, v in row["gap"].items():
        assert k in imp and k in den
        assert v == pytest.approx(imp[k] - den[k])


def test_selection_reads_the_impute_dial_and_never_the_denoise_one(row) -> None:
    sel = row["selection"]
    assert sel["kind"] == "impute" and sel["metric"] == MON.SELECTION_KEY
    assert sel["value"] == pytest.approx(row["impute"]["macro"][MON.SELECTION_KEY])
    assert sel["n_tracks"] == row["impute"]["n_tracks"]
    # And it is NOT the denoise number, which is the whole point of the dial being watch-only.
    assert sel["value"] != pytest.approx(row["denoise"]["macro"][MON.SELECTION_KEY])


# ---------------------------------------------------------------------------
# 2 — the tiers switch on the heads the model actually built
# ---------------------------------------------------------------------------

def test_the_point_tier_is_always_present(count_model, peak_model) -> None:
    for m in (count_model, peak_model):
        assert MON.tiers(m)["point"] == MON.POINT_KEYS


def test_the_distribution_tier_carries_the_crps_split_not_bare_crps(count_model) -> None:
    """AGENTS.md §7.2: a raw CRPS confounds distribution shape with scale."""
    dist = MON.tiers(count_model)["dist"]
    assert "crps" in dist and "crps_oracle_scaled" in dist and "scale_error" in dist
    # A bare `nll` is still not a key anywhere: the three loss terms are named for the distribution
    # each one is, in their own tier, because a model may owe two of them and not the third.
    assert "nll" not in MON.tiered_keys(count_model)
    assert "nll" not in dist


def test_the_peak_tier_appears_only_when_the_peak_head_is_built(count_model, peak_model) -> None:
    assert "peak" not in MON.tiers(count_model)
    assert MON.tiers(peak_model)["peak"] == MON.PEAK_KEYS
    assert MON.model_heads(count_model) == ("count",)
    assert "peak" in MON.model_heads(peak_model)


def test_a_count_only_model_does_not_report_auprc_even_though_bench_computes_it(
        regime_file, count_model, row) -> None:
    """`stream_tracks` falls back to the NB mean for `peak_score`, so the number exists anyway.

    Suppressing it here is the difference between a peak metric and a coverage ranking wearing a
    peak metric's name.
    """
    from candi.bench.harness import score_track, stream_tracks
    from candi.bench import annotations as ann

    for track in row["impute"]["per_track"].values():
        assert "auprc" not in track and "peak_base_rate" not in track
    assert "auprc" not in row["impute"]["macro"]

    # ... and bench really does emit it for this same count-only model, so the absence above is a
    # decision this module made rather than a number that was never computed.
    src = MON.Monitor(store=regime_file, device=DEVICE, batch_windows=2).source
    rec = next(iter(stream_tracks(count_model, src, DEVICE, kind="impute", batch_windows=2)))
    arms = score_track(rec, gene_annotations=ann.gene_annotations(),
                       enh_annotations=ann.enhancer_annotations(), c_index_pairs=1000)
    assert "auprc" in arms["count"], "bench stopped emitting auprc; this test's premise is gone"
    src.close()


def test_the_peak_tier_is_reported_when_the_head_is_on(regime_file, peak_model) -> None:
    m = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000)
    try:
        got = m.check(peak_model, epoch=0, step=1)
    finally:
        m.close()
    assert "auprc" in got["impute"]["macro"]
    for track in got["impute"]["per_track"].values():
        assert "auprc" in track


# ---------------------------------------------------------------------------
# 2b — the loss tier: this module is the val_loss
# ---------------------------------------------------------------------------
#
# NB, Gaussian and Bernoulli NLL are the three terms the objective is built from. The training loop
# logs them every step and `candi.bench`'s CLI emits them as the test loss; this is the mid-training
# third of `train_loss / val_loss / test_loss`. What is checked here is that both dials carry them
# and that the head gate is the model's, not a guess.


def test_the_loss_tier_is_head_gated_key_by_key_and_not_tier_by_tier(
        count_model, peak_model) -> None:
    """A `count,peak` model owes `nb_nll` and `bernoulli_nll` and no `gaussian_nll` — which is why
    this is a tier of its own rather than three keys folded into the tiers above."""
    assert MON.tiers(count_model)["nll"] == ("nb_nll",)
    assert MON.tiers(peak_model)["nll"] == ("nb_nll", "bernoulli_nll")
    assert "gaussian_nll" not in MON.tiered_keys(peak_model)
    # Tier order is stable, and the loss comes last.
    assert MON.tiered_keys(count_model)[-1] == "nb_nll"


def test_both_dials_carry_the_val_loss_per_track_and_in_the_macro(row) -> None:
    """Per-track AND macro, on impute and on denoise: the gap between them is read on this key too."""
    for kind in ("impute", "denoise"):
        assert "nb_nll" in row[kind]["macro"]
        assert np.isfinite(row[kind]["macro"]["nb_nll"])
        for key, track in row[kind]["per_track"].items():
            assert "nb_nll" in track, f"{kind}/{key} has no val loss"
    assert "nb_nll" in row["gap"], "the overfitting alarm skipped the loss"
    # The count-only model owes exactly one of the three, and the other two are ABSENT keys.
    for track in row["impute"]["per_track"].values():
        assert "gaussian_nll" not in track and "bernoulli_nll" not in track


def test_the_target_space_the_loss_was_scored_in_is_on_the_row(regime_file, count_model) -> None:
    """D30 — `gaussian_nll` in one space and the same key in another are two different numbers."""
    m = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000,
                 signal_target_transform="arcsinh")
    try:
        got = m.check(count_model, epoch=0, step=1)
    finally:
        m.close()
    assert got["signal_target_transform"] == "arcsinh"
    assert m.arm == "count", "the count-arm ruling is untouched by the loss tier"


def test_the_peak_loss_appears_only_with_a_peak_head(regime_file, peak_model) -> None:
    """`stream_tracks` fills `peak_score` from the NB MEAN without one, and a BCE of an unbounded
    count is not merely misleading — it is arithmetic on the wrong kind of number."""
    m = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000)
    try:
        got = m.check(peak_model, epoch=0, step=1)
    finally:
        m.close()
    assert "bernoulli_nll" in got["impute"]["macro"]
    for track in got["impute"]["per_track"].values():
        assert np.isfinite(track["bernoulli_nll"]) and np.isfinite(track["nb_nll"])
        assert "gaussian_nll" not in track


# ---------------------------------------------------------------------------
# 3 — the space assertion
# ---------------------------------------------------------------------------

def test_the_space_assertion_fires_on_a_deliberate_mismatch(regime_file) -> None:
    """Any arm but `count` is refused, by name and at construction — a SCOPE ruling."""
    with pytest.raises(ValueError, match="arcsinh"):
        MON.assert_prediction_space("pval", signal_target_transform="arcsinh")
    # And it fires at CONSTRUCTION, before an hour of training is spent on a number nobody can use.
    with pytest.raises(ValueError, match="count.*arm only"):
        MON.Monitor(store=regime_file, device=DEVICE, arm="pval",
                    signal_target_transform="arcsinh")
    # The count arm is always fine: raw counts on both sides, no transform anywhere.
    MON.assert_prediction_space("count", signal_target_transform="arcsinh")


def test_the_refusal_gives_the_reason_that_is_still_true(regime_file) -> None:
    """The pval arm WAS unscoreable and now is not: `bench.harness.score_track` inverts the
    prediction into -log10 p. A refusal that still blamed the space would send a reader looking for
    a bug that has been fixed, so the message has to name the reasons that remain — scope and cost
    — and point at the command that does score that arm."""
    with pytest.raises(ValueError) as e:
        MON.assert_prediction_space("pval", signal_target_transform="arcsinh")
    msg = str(e.value)
    assert "candi.bench" in msg, "the refusal must say where the pval arm IS scored"
    assert "SCOPE" in msg and "SELECTION_KEY" in msg
    assert "gaussian_nll" in msg, "the one number here that is not a count-arm comparison"
    # The dead reason is gone: nothing in the message may claim bench compares two spaces.
    low = msg.lower()
    assert "cannot be scored" not in low and "untransformed" not in low


def test_the_impute_dial_is_not_optional_because_it_is_the_one_that_selects(regime_file) -> None:
    with pytest.raises(ValueError, match="not optional"):
        MON.Monitor(store=regime_file, device=DEVICE, kinds=("denoise",))


# ---------------------------------------------------------------------------
# 4 — W&B
# ---------------------------------------------------------------------------

def test_wandb_is_off_by_default(regime_file, count_model) -> None:
    import inspect

    assert inspect.signature(MON.Monitor.__init__).parameters["log_fn"].default is None
    seen: list = []
    m = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000,
                 log_fn=lambda step, d: seen.append((step, d)))
    try:
        got = m.check(count_model, epoch=1, step=42)
    finally:
        m.close()
    assert len(seen) == 1 and seen[0][0] == 42
    payload = seen[0][1]
    assert payload["monitor/selection"] == pytest.approx(got["selection"]["value"])
    assert all(isinstance(v, float) for v in payload.values()), \
        "the dashboard takes scalars; per-track rows belong in the run json"
    # `_wandb_payload` iterates the macros, so a new macro key reaches the dashboard with no
    # per-key list to keep in step. The val loss is the first key that relies on that.
    assert payload["monitor/impute/nb_nll"] == pytest.approx(got["impute"]["macro"]["nb_nll"])
    assert "monitor/impute/gaussian_nll" not in payload, "no signal head on this model"


def test_the_end_of_run_row_logs_under_its_own_dashboard_keys(regime_file, count_model) -> None:
    """It is logged at the LAST training step but describes the SELECTED checkpoint.

    Under `monitor/impute/crps` it would land on the end of the mid-training series as if the curve
    had moved one more time, so it gets `monitor_final/...` instead.
    """
    seen: list = []
    m = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000,
                 log_fn=lambda step, d: seen.append((step, d)))
    try:
        got = m.final_check(count_model, epoch=4, step=99, selected="last")
    finally:
        m.close()
    assert len(seen) == 1 and seen[0][0] == 99
    payload = seen[0][1]
    assert payload["monitor_final/selection"] == pytest.approx(got["selection"]["value"])
    assert payload["monitor_final/epoch"] == 4.0
    assert not any(k.startswith("monitor/") for k in payload), \
        "the end-of-run row leaked into the mid-training series"
    assert got["selected"] == "last"


# ---------------------------------------------------------------------------
# 5 — the boundaries
# ---------------------------------------------------------------------------

def test_the_monitor_never_imports_the_training_module() -> None:
    """`bench <- monitor <- train`. A monitor that imports the training loop closes the cycle.

    A subprocess, for the reason `tests/test_bench_harness.py` gives about the bench half: tearing
    `candi` out of `sys.modules` in-process leaves every later test holding classes from a module
    object that no longer exists.
    """
    import os
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parent.parent / "src")
    code = (
        "import sys, importlib\n"
        "importlib.import_module('candi.monitor')\n"
        # `candi.eval` was on this list until it was deleted (D15); a name that can never be
        # imported asserts nothing, so the list is `candi.train` alone — which is the one that
        # would close the cycle.
        "print(','.join(m for m in ('candi.train',) if m in sys.modules))\n"
    )
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    assert got.stdout.strip() == "", f"candi.monitor dragged in {got.stdout.strip()}"


def test_the_monitor_puts_the_training_mode_back(regime_file, count_model) -> None:
    """`stream_tracks` calls `model.eval()` and never puts it back; the two loops disagreed."""
    m = _monitor(regime_file, kinds=("impute",), c_index_pairs=1000)
    try:
        count_model.train()
        m.check(count_model, epoch=0, step=0)
        assert count_model.training is True
        count_model.eval()
        m.check(count_model, epoch=0, step=0)
        assert count_model.training is False
    finally:
        m.close()


# ---------------------------------------------------------------------------
# the --eval-every resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested,declared,want", [
    (None, True, 3),        # unset + something to score -> the settled cadence
    (None, False, 0),       # unset + nothing to score -> off, without tripping the guard
    (0, True, 0),           # an explicit 0 wins over the cadence
    (7, False, 7),          # an explicit value wins and still meets the guard below
    (1, True, 1),
])
def test_eval_every_resolves_from_the_source_when_it_was_not_given(requested, declared, want):
    assert MON.resolve_eval_every(requested, eval_pairs_declared=declared) == want


def test_the_pair_less_guard_is_untouched_by_the_new_default(tmp_path, store) -> None:
    """An EXPLICIT non-zero against a pair-less regime still gets told why it is off."""
    from candi.train import DataSource

    p = tmp_path / "nopairs.json"
    p.write_text(json.dumps(regime_dict(store)), encoding="utf-8")
    src = DataSource.resolve(store=str(p))
    assert src.eval_pairs_declared() is False
    assert MON.resolve_eval_every(5, eval_pairs_declared=False) == 5


def test_the_banner_names_which_dial_each_number_came_from(row) -> None:
    line = MON.format_check(row, best={"crps": 0.5, "epoch": 2})
    assert "impute" in line and "denoise" in line and "watch-only" in line and "gap=" in line
    assert np.isfinite(row["selection"]["value"])


def test_the_banner_tells_the_two_cadences_apart(mid_row, row) -> None:
    """A reader scrolling a log must see which row carries the alarm without counting epochs."""
    mid = MON.format_check(mid_row, best={"crps": 0.5, "epoch": 2})
    assert "monitor@ep2" in mid
    assert "denoise" not in mid and "gap=" not in mid, "an impute-only banner promised an alarm"

    fin = MON.format_check(row)
    assert "monitor@final" in fin and "best" in fin
    assert "end-of-run overfitting alarm" in fin
