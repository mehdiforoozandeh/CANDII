"""h64 — the `--meta-probe` arm switch: known-answer tests for both controls and for the `off` no-op.

The four things that can silently invalidate this node, each pinned below:

  1. `off` is not really off — some tensor moves, or an RNG is advanced, and the "control" arm is not
     the pre-h64 path. Tested by identity of every tensor AND by `make_meta_probe` returning None.
  2. The negative control corrupts the sentinel layout, so the arms differ in WHICH ASSAY IS BEING
     IMPUTED rather than only in `run_type`.
  3. The no-op detector has no power — it would report "identity" on a batch that genuinely changed,
     and its 1.0 reading on real data would mean nothing. Tested on a synthetic batch where row 3
     genuinely varies down B.
  4. The positive control leaks the shift into the input, so the model never needs the covariate.
     Tested by bit-identity of `x_data` alongside an exact `2**(+/-delta)` check on `y_data`.

Plus the RNG-independence property: two arms must visit the SAME batches. That is what
`_META_PROBE_SEED_XOR`'s dedicated stream buys, and the test asserts it on the shared data stream
rather than trusting the comment.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from candi._vendored import CLOZE, MISSING
from candi.meta_probe import (
    DEFAULT_META_PROBE_DELTA, META_PROBE_MODES, MetaProbe, RUN_TYPE_ROW, make_meta_probe,
)

NUM_RUNTYPES = 2                     # what MetadataEmbedding declares; the planted bins must fit it


# ---------------------------------------------------------------------------
# synthetic batches
# ---------------------------------------------------------------------------

def _meta(rows_row3, B, C, dtype=torch.float32) -> torch.Tensor:
    """A `[B, 4, C]` prompt whose row 3 is exactly `rows_row3` and whose other rows are distinctive."""
    m = torch.zeros(B, 4, C, dtype=dtype)
    m[:, 0, :] = 25.0 + torch.arange(B, dtype=dtype).unsqueeze(1)      # log2 depth
    m[:, 1, :] = torch.arange(C, dtype=dtype).unsqueeze(0)             # assay id
    m[:, 2, :] = 76.0                                                  # read length
    m[:, 3, :] = torch.as_tensor(rows_row3, dtype=dtype)
    return m


def _constant_down_b(B=4, C=3) -> torch.Tensor:
    """THE REAL CASE. One biosample per batch => run_type is constant down B in every column."""
    return _meta([[0.0, 1.0, 0.0]] * B, B, C)


def _varying_down_b(B=4, C=3) -> torch.Tensor:
    """The synthetic case that gives the no-op detector its power."""
    return _meta([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]], B, C)


def _with_sentinels(B=4, C=4) -> torch.Tensor:
    """Row 3 with MISSING and CLOZE scattered through it, exactly as masking leaves them."""
    m = _meta([[0.0, float(MISSING), 1.0, float(CLOZE)],
               [1.0, float(MISSING), 0.0, 1.0],
               [1.0, 0.0, float(CLOZE), 0.0],
               [0.0, 1.0, 1.0, float(CLOZE)]], B, C)
    return m


def _prep(x_meta, y_meta, B=4, L=6, A=3, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(B, L, A, generator=g) * 40).round()
    y[:, :, A - 1] = float(MISSING)                     # one wholly unavailable assay column
    return dict(
        x_data=(torch.rand(B, L, A + 1, generator=g) * 40).round(),
        x_dna=torch.zeros(B, L * 25, 4),
        x_meta=x_meta, y_meta=y_meta, y_data=y,
        y_pval=torch.rand(B, L, A, generator=g),
        y_peaks=torch.zeros(B, L, A),
    )


def _clone(d: dict) -> dict:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def _row3(m: torch.Tensor) -> torch.Tensor:
    return m[:, RUN_TYPE_ROW, :]


def _sentinel_mask(m: torch.Tensor) -> torch.Tensor:
    r = _row3(m)
    return (r == float(MISSING)) | (r == float(CLOZE))


# ---------------------------------------------------------------------------
# 1. `off` is a STRICT no-op
# ---------------------------------------------------------------------------

def test_off_builds_no_object_at_all():
    assert make_meta_probe("off", seed=0) is None
    assert make_meta_probe("off", seed=7, delta=3.0) is None


def test_off_leaves_every_tensor_bit_identical():
    """`off` has no object, so the call site is skipped and the prep dict is untouched by identity."""
    prep = _prep(_with_sentinels(), _with_sentinels())
    before = _clone(prep)
    probe = make_meta_probe("off", seed=0)
    if probe is not None:                                  # the branch train.py actually executes
        probe.apply(prep)
    for k, v in before.items():
        assert torch.equal(prep[k], v), f"{k} moved under --meta-probe off"


def test_off_advances_no_rng():
    """No global torch RNG draw, no numpy default_rng draw: `off` cannot desynchronise the arms."""
    torch.manual_seed(1234)
    ref = torch.rand(8)
    torch.manual_seed(1234)
    probe = make_meta_probe("off", seed=1234)
    assert probe is None
    assert torch.equal(torch.rand(8), ref)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        make_meta_probe("shufled", seed=0)
    assert META_PROBE_MODES == ("off", "shuffled", "planted")


# ---------------------------------------------------------------------------
# 2. `shuffled` — multiset preserved, sentinels frozen
# ---------------------------------------------------------------------------

def test_shuffled_preserves_the_per_column_multiset_and_every_sentinel():
    x, y = _with_sentinels(), _with_sentinels()
    prep = _prep(x.clone(), y.clone())
    before = _clone(prep)
    make_meta_probe("shuffled", seed=0).apply(prep)

    for key in ("x_meta", "y_meta"):
        old, new = before[key], prep[key]
        # sentinels: same positions, same VALUES (a MISSING must not become a CLOZE)
        sm_old, sm_new = _sentinel_mask(old), _sentinel_mask(new)
        assert torch.equal(sm_old, sm_new), f"{key}: sentinel layout moved"
        assert torch.equal(_row3(old)[sm_old], _row3(new)[sm_new]), f"{key}: sentinel values changed"
        # per column, the multiset of non-sentinel values is exactly preserved
        for c in range(old.shape[2]):
            o = _row3(old)[:, c][~sm_old[:, c]]
            n = _row3(new)[:, c][~sm_new[:, c]]
            assert torch.equal(torch.sort(o).values, torch.sort(n).values), f"{key} col {c}"
        # nothing outside row 3 moved
        for r in range(old.shape[1]):
            if r != RUN_TYPE_ROW:
                assert torch.equal(old[:, r, :], new[:, r, :]), f"{key} row {r} moved"


def test_shuffled_never_touches_the_targets_or_the_input():
    prep = _prep(_varying_down_b(), _varying_down_b())
    before = _clone(prep)
    make_meta_probe("shuffled", seed=0).apply(prep)
    for k in ("x_data", "x_dna", "y_data", "y_pval", "y_peaks"):
        assert torch.equal(prep[k], before[k]), f"shuffled moved {k}"


# ---------------------------------------------------------------------------
# 3. the no-op detector has POWER, and reads 1.0 on the real case
# ---------------------------------------------------------------------------

def test_detector_has_power_when_run_type_genuinely_varies_down_b():
    """If this fails, the 1.0 reading on real data means nothing — it would be an untested instrument."""
    changed_any = False
    for seed in range(8):
        prep = _prep(_varying_down_b(), _varying_down_b())
        before = _clone(prep)
        p = make_meta_probe("shuffled", seed=seed)
        p.apply(prep)
        if not torch.equal(prep["x_meta"], before["x_meta"]):
            changed_any = True
            assert p.noop_frac == 0.0
            assert p.step_metrics()["meta_probe/shuffle_is_noop"] == 0.0
    assert changed_any, "the permutation never changed a varying batch — the detector has no power"


def test_detector_reports_noop_on_the_real_constant_down_b_case(capsys):
    """One biosample per batch => row 3 constant down B => the permutation IS the identity."""
    p = make_meta_probe("shuffled", seed=0)
    for _ in range(5):
        prep = _prep(_constant_down_b(), _constant_down_b())
        before = _clone(prep)
        p.apply(prep)
        assert torch.equal(prep["x_meta"], before["x_meta"])
        assert torch.equal(prep["y_meta"], before["y_meta"])
    assert p.n_steps == 5 and p.n_noop == 5
    assert p.noop_frac == 1.0
    assert p.step_metrics()["meta_probe/shuffle_is_noop"] == 1.0
    assert p.step_metrics()["meta_probe/shuffle_noop_frac"] == 1.0
    assert p.stats()["meta_probe_shuffle_noop_frac"] == 1.0

    out = capsys.readouterr().out
    assert "NEGATIVE CONTROL IS A NO-OP" in out
    assert "ONE biosample per batch" in out
    assert out.count("NEGATIVE CONTROL IS A NO-OP") == 1, "the warning must be printed ONCE"
    assert "shuffle_noop_frac=1.000" in p.summary()
    assert "NOT a negative control" in p.summary()


def test_single_row_batch_cannot_permute_and_is_reported_as_a_noop():
    p = make_meta_probe("shuffled", seed=0)
    prep = _prep(_meta([[0.0, 1.0]], 1, 2), _meta([[0.0, 1.0]], 1, 2), B=1, A=2)
    p.apply(prep)
    assert p.noop_frac == 1.0


def test_eval_application_does_not_pollute_the_training_noop_fraction():
    p = make_meta_probe("shuffled", seed=0)
    prep = _prep(_varying_down_b(), _varying_down_b())
    p.apply_tensors((prep["x_meta"], prep["y_meta"]), (prep["y_data"],), record=False)
    assert p.n_steps == 0 and p.step_metrics() == {}


# ---------------------------------------------------------------------------
# 4. `planted` — valid indices, sentinels intact, target-only shift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [1.0, 0.5, 2.0])
def test_planted_writes_only_valid_indices_and_shifts_only_the_target(delta):
    prep = _prep(_with_sentinels(), _with_sentinels())
    before = _clone(prep)
    p = make_meta_probe("planted", seed=3, delta=delta, num_runtypes=NUM_RUNTYPES)
    p.apply(prep)

    for key in ("x_meta", "y_meta"):
        old, new = before[key], prep[key]
        sm = _sentinel_mask(old)
        assert torch.equal(sm, _sentinel_mask(new)), f"{key}: sentinel layout moved"
        assert torch.equal(_row3(old)[sm], _row3(new)[sm]), f"{key}: sentinel values changed"
        live = _row3(new)[~sm]
        assert live.numel()
        assert bool(((live >= 0) & (live < NUM_RUNTYPES)).all()), "planted index outside the table"
        assert bool((live == live.round()).all()), "planted index is not integral"
        for r in range(old.shape[1]):
            if r != RUN_TYPE_ROW:
                assert torch.equal(old[:, r, :], new[:, r, :]), f"{key} row {r} moved"

    # THE VALIDITY ARGUMENT: the input never carries the shift.
    for k in ("x_data", "x_dna", "y_pval", "y_peaks"):
        assert torch.equal(prep[k], before[k]), f"planted moved {k} — the shift leaked out of y_data"

    # the target is scaled by exactly 2**(+/-delta) and rounded back onto the counting numbers,
    # per sample, at real counts only. The rounding is forced by the NB support — see _shift.
    bins = []
    for b in range(before["x_meta"].shape[0]):
        row = _row3(prep["x_meta"])[b]
        live = row[~_sentinel_mask(prep["x_meta"])[b]]
        assert bool((live == live[0]).all()), "the bin must be constant across the assay axis"
        bins.append(float(live[0]))
    for b, s in enumerate(bins):
        scale = float(np.exp2(delta * (2.0 * s - 1.0)))
        old, new = before["y_data"][b], prep["y_data"][b]
        real = old >= 0
        assert torch.equal(new[~real], old[~real]), "a MISSING sentinel in y_data was scaled"
        want = torch.round(old * torch.as_tensor(scale, dtype=old.dtype))
        assert torch.equal(new[real], want[real])


def test_planted_target_stays_on_the_negative_binomial_support():
    """REGRESSION. `torch.distributions.NegativeBinomial.log_prob` VALIDATES its support, so a count
    of 1 scaled by 2**-1 = 0.5 kills the arm on its first step. Caught by the gatec smoke, pinned here.
    """
    prep = _prep(_with_sentinels(), _with_sentinels())
    assert torch.equal(prep["y_data"], prep["y_data"].round()), "fixture must start integral"
    make_meta_probe("planted", seed=5, delta=1.0, num_runtypes=NUM_RUNTYPES).apply(prep)
    y = prep["y_data"]
    real = y >= 0
    assert torch.equal(y[real], y[real].round()), "planted put a half-count on the NB support"
    # and the real loss must actually accept it
    n = torch.full_like(y, 5.0)
    p = torch.full_like(y, 0.5)
    torch.distributions.NegativeBinomial(
        total_count=n.clamp_min(1e-6), probs=p, validate_args=True
    ).log_prob(y.clamp_min(0.0))


def test_planted_uses_both_bins_and_reports_the_balance():
    p = make_meta_probe("planted", seed=0, delta=1.0, num_runtypes=NUM_RUNTYPES)
    seen = set()
    for _ in range(20):
        prep = _prep(_with_sentinels(), _with_sentinels())
        p.apply(prep)
        row = _row3(prep["x_meta"])
        seen.update(int(v) for v in row[~_sentinel_mask(prep["x_meta"])].unique().tolist())
        frac = p.step_metrics()["meta_probe/planted_frac_hi"]
        assert 0.0 <= frac <= 1.0
    assert seen == {0, 1}, f"the median split must produce both bins; saw {seen}"


def test_planted_refuses_a_table_that_cannot_hold_two_bins():
    with pytest.raises(ValueError):
        MetaProbe("planted", seed=0, num_runtypes=1)


def test_planted_default_delta_is_one_log2_unit():
    assert DEFAULT_META_PROBE_DELTA == 1.0
    p = make_meta_probe("planted", seed=0)
    assert p.delta == 1.0
    assert p.stats()["meta_probe_planted_bins"] == 2      # ONE BIT, not a continuum


def test_planted_shares_one_bin_across_every_tensor_in_a_call():
    """The eval seam hands four tensors in ONE call precisely so this holds."""
    p = make_meta_probe("planted", seed=1, delta=1.0, num_runtypes=NUM_RUNTYPES)
    xm, ym = _with_sentinels(), _with_sentinels()
    yd = torch.full((4, 5, 3), 10.0)                     # integral: the NB support demands it
    ydi = torch.full((4, 5, 3), 10.0)
    (xm2, ym2), (yd2, ydi2) = p.apply_tensors((xm, ym), (yd, ydi))
    assert torch.equal(_row3(xm2)[~_sentinel_mask(xm2)], _row3(ym2)[~_sentinel_mask(ym2)])
    assert torch.equal(yd2, ydi2), "the two targets got different bins"


# ---------------------------------------------------------------------------
# 5. the probe RNG is DEDICATED — the arms must visit the same data
# ---------------------------------------------------------------------------

def test_probe_rng_never_touches_the_shared_data_stream():
    """Two arms differing ONLY in --meta-probe must draw the same batches from the data stream."""
    def run(mode):
        shared = np.random.default_rng(0)               # stands in for the loop's data stream
        probe = make_meta_probe(mode, seed=0, delta=1.0, num_runtypes=NUM_RUNTYPES)
        drawn, batches = [], []
        for _ in range(6):
            drawn.append(float(shared.random()))        # "which biosample / which DSF" — the data
            prep = _prep(_with_sentinels(), _with_sentinels(), seed=len(drawn))
            batches.append(prep["x_data"].clone())      # the pre-transform data tensor
            if probe is not None:
                probe.apply(prep)
                assert torch.equal(prep["x_data"], batches[-1]), "x_data was mutated"
        return drawn, batches

    d_off, b_off = run("off")
    d_shuf, b_shuf = run("shuffled")
    d_plant, b_plant = run("planted")
    assert d_off == d_shuf == d_plant, "the probe advanced the shared data stream"
    for a, b, c in zip(b_off, b_shuf, b_plant):
        assert torch.equal(a, b) and torch.equal(a, c), "the arms saw different data"


def test_probe_streams_are_reproducible_from_the_seed():
    def bins(seed):
        p = make_meta_probe("planted", seed=seed, num_runtypes=NUM_RUNTYPES)
        prep = _prep(_with_sentinels(), _with_sentinels())
        p.apply(prep)
        return _row3(prep["x_meta"]).clone()
    assert torch.equal(bins(11), bins(11)), "the same seed must give the same arm"


# ---------------------------------------------------------------------------
# 6. wiring: the flag reaches train_and_eval, the CLI and the run config
# ---------------------------------------------------------------------------

def test_train_and_eval_defaults_to_off():
    import inspect

    from candi.train import train, train_and_eval
    sig = inspect.signature(train_and_eval).parameters
    assert sig["meta_probe"].default == "off"
    assert sig["meta_probe_delta"].default == DEFAULT_META_PROBE_DELTA
    assert inspect.signature(train).parameters["meta_probe"].default is None


def test_a_mistyped_arm_fails_before_the_h5_is_opened():
    """The guard must fire on the submit line, not three minutes into a queued job."""
    from candi.train import train_and_eval
    with pytest.raises(ValueError, match="meta_probe must be one of"):
        train_and_eval(h5_path="/nonexistent/never.h5", out_dir="/nonexistent",
                       meta_probe="shufled")

def _cli_actions():
    """Build train.main's parser and hand back its actions, without parsing or running anything."""
    import argparse

    import candi.train as T

    captured = {}

    class _Stop(Exception):
        pass

    real = argparse.ArgumentParser.parse_args

    def fake(self, *a, **kw):
        captured["ap"] = self
        raise _Stop

    argparse.ArgumentParser.parse_args = fake
    try:
        with pytest.raises(_Stop):
            T.main()
    finally:
        argparse.ArgumentParser.parse_args = real
    return {act.dest: act for act in captured["ap"]._actions}


def test_cli_exposes_both_flags_with_the_right_defaults():
    acts = _cli_actions()
    assert acts["meta_probe"].default == "off"
    assert list(acts["meta_probe"].choices) == list(META_PROBE_MODES)
    assert acts["meta_probe_delta"].default == DEFAULT_META_PROBE_DELTA


def test_quick_eval_and_build_eval_units_take_the_probe():
    import inspect

    from candi.eval import build_eval_units, quick_eval
    assert inspect.signature(quick_eval).parameters["meta_probe"].default is None
    assert inspect.signature(build_eval_units).parameters["meta_probe"].default is None
