"""t28 — the scorer takes a ready dataset, so a store-backed run can actually be scored.

`build_eval_units` used to CONSTRUCT its own `CandiKitH5Dataset` from a path, and that single line
was the whole reason a store-backed run could not be evaluated: `train.py` printed a warning and
silently set `--eval-every` to 0, so a run on the corpus we are migrating to had no mid-training
number at all and no best-checkpoint selection.

The three names the builder reads off a dataset for its slot arithmetic — `_eval_indices`,
`_bios_candidates`, `_all_imp_biosamples` — were already spelled identically by `StoreDataset`
(t14, `store/dataset.py`), deliberately, so that "the eval builder must not be able to tell the two
datasets apart". With the construction removed, that promise is finally testable, and this file is
where it is tested.

Everything runs against a real synthetic store built by `candi.store.writer`. Nothing here
hand-rolls a dataset: a test that built its own stand-in would be checking the builder against a
mock of the thing it is supposed to work with.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from candi.eval import build_eval_units, eval_cell_cond, quick_eval
from candi.store import layout as L
from candi.store.regime import Regime
from candi.train import DataSource, make_dataset

from tests.test_store_reader import ASSAYS, make_store
from tests.test_store_regime import CTX, regime_dict

DEVICE = torch.device("cpu")


#: An imputation eval needs a prompt that LACKS something the truth carries: `imp_map` is
#: `(y_avail <= 0) & (y_data_imp != -1)`, so an assay present in both is not a target and an assay
#: present in neither is not either. The shared `TRACKS` layout has `V_aa` as a strict subset of
#: `T_aa`, which is the right shape for testing MISSING and the wrong one for testing imputation --
#: it yields zero targets and a NaN selection metric, which is exactly what "selecting on nothing"
#: looked like before t28. Here `T_aa` is missing H3K4me3 and `V_aa` has it.
EVAL_TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_aa": ("ATAC-seq", "DNase-seq", "H3K4me3"),
    # TWO pairs, not one. With a single pair the cycle arithmetic below cannot be wrong: every
    # batch belongs to the only target there is. The bug this file now pins needed two.
    "T_bb": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_bb": ("ATAC-seq", "DNase-seq", "H3K4me3"),
}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("t28store"), tracks=EVAL_TRACKS)


@pytest.fixture(scope="module")
def paired_source(tmp_path_factory, store) -> DataSource:
    """A store whose regime DECLARES an imputation pair — `T_aa` prompts, `V_aa` is the truth."""
    p = tmp_path_factory.mktemp("t28regime") / "regime.json"
    p.write_text(json.dumps(regime_dict(
        store, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
        eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]]), indent=2), encoding="utf-8")
    return DataSource(kind="store", regime_path=str(p), regime=Regime.from_file(p))


@pytest.fixture(scope="module")
def bare_source(tmp_path_factory, store) -> DataSource:
    """The same store with NO `eval_pairs` — a held-out split with nothing to impute."""
    p = tmp_path_factory.mktemp("t28bare") / "regime.json"
    p.write_text(json.dumps(regime_dict(store), indent=2), encoding="utf-8")
    return DataSource(kind="store", regime_path=str(p), regime=Regime.from_file(p))


def _eval_ds(source, model, batch_size=2):
    return make_dataset(source, "type1", train=False, batch_size=batch_size, dsf_sampling="off",
                        seed=0, shuffle=False, h5_cache_ram=False,
                        cell_cond=eval_cell_cond(model))


def _model(ds):
    from candi.model import build_model
    torch.manual_seed(0)
    return build_model(num_assays=ds.num_assays, context_length=ds.context_bins,
                       resolution=ds.resolution, num_cells=ds.num_cells,
                       depth_center=ds.depth_center(), d_model=32, embed_dim=8,
                       n_transformer_layers=1).eval()


# ---------------------------------------------------------------------------------------------
# the signature: it receives, it does not construct
# ---------------------------------------------------------------------------------------------


def test_the_builder_no_longer_takes_a_path_or_knows_how_to_open_one():
    """The regression this file exists to prevent: a path parameter invites a hard-coded loader."""
    params = inspect.signature(build_eval_units).parameters
    assert "h5_path" not in params and "regime" not in params
    assert "imp_prefixes" not in params and "reference" not in params
    # batch_size is read off the dataset -- a caller free to disagree with the loader would
    # mis-space `batches_per_pair` across the chromosome without saying anything.
    assert "batch_size" not in params


def test_quick_eval_no_longer_takes_a_path_either():
    params = inspect.signature(quick_eval).parameters
    assert "h5_path" not in params and "reference" not in params
    assert inspect.signature(quick_eval).parameters["meta_probe"].default is None


# ---------------------------------------------------------------------------------------------
# the capability: a store-backed run is scorable
# ---------------------------------------------------------------------------------------------


def test_the_builder_scores_a_store_backed_dataset(paired_source):
    """The point of t28. Before this, these units could not be built at all."""
    probe = make_dataset(paired_source, "type1", train=True, batch_size=2, seed=0)
    model = _model(probe)
    ds = _eval_ds(paired_source, model)
    units, assays = build_eval_units(model, ds, DEVICE)
    assert units, "a store-backed eval produced no units"
    assert list(assays) == list(ds.assays)
    assert any(u.imp_map is not None for u in units), "no imputation target was scored"
    assert any(u.imp_biosample == "V_aa" for u in units)


def test_quick_eval_returns_a_selection_metric_off_a_store(paired_source):
    """`train.py` selects on `V_imp_crps`; a NaN here is what "selecting on nothing" looked like."""
    probe = make_dataset(paired_source, "type1", train=True, batch_size=2, seed=0)
    model = _model(probe)
    q = quick_eval(model, _eval_ds(paired_source, model), DEVICE, batches_per_pair=1)
    assert q["V_n_targets"] > 0
    assert q["V_imp_crps"] == q["V_imp_crps"], "V_imp_crps is NaN"


def test_one_dataset_re_iterated_gives_the_same_windows_every_time(paired_source):
    """Why `train.py` opens the eval dataset ONCE and re-iterates it.

    Selection compares epoch 6 against epoch 12, and that comparison is only paired if both saw the
    same positions. The loop relies on re-iteration being stable; this is that assumption, asserted.
    """
    probe = make_dataset(paired_source, "type1", train=True, batch_size=2, seed=0)
    model = _model(probe)
    ds = _eval_ds(paired_source, model)
    first = [int(b["window_idx"][0]) for b in ds]
    second = [int(b["window_idx"][0]) for b in ds]
    assert first and first == second


# ---------------------------------------------------------------------------------------------
# the guard: narrowed to the condition that actually matters
# ---------------------------------------------------------------------------------------------


def test_a_regime_with_declared_pairs_reports_that_it_can_be_scored(paired_source):
    assert paired_source.eval_pairs_declared() is True


def test_a_regime_with_no_declared_pairs_reports_that_it_cannot(bare_source):
    """D31: the store DECLARES its pairing, because D16 makes the names opaque ids.

    With none declared there is genuinely nothing to impute, and turning the mid-training eval on
    would select the 'best' checkpoint on an empty target set. That is the one case the guard in
    `train.py` should still catch -- and the only one.
    """
    assert bare_source.eval_pairs_declared() is False


def test_the_h5_path_always_reports_that_it_can_be_scored(tmp_path_factory):
    """It derives its pairing from the `T_`/`V_`/`B_` prefixes, so it can never lack one."""
    src = DataSource.coerce(str(tmp_path_factory.mktemp("t28h5") / "nothing.h5"))
    assert src.eval_pairs_declared() is True


# ---------------------------------------------------------------------------------------------
# thinning WINDOWS must never thin TARGETS
# ---------------------------------------------------------------------------------------------


def test_batches_per_pair_keeps_every_target_and_shrinks_only_the_windows(paired_source):
    """The knob's entire reason for existing, and it did not hold on the store.

    `CandiKitH5Dataset` interleaves eval batches across pairs -- batch `bi` belongs to pair
    `bi % n_pairs` -- while `StoreDataset` groups them pair-major: all of pair 0's batches, then all
    of pair 1's. `build_eval_units` selected cycles as `bi // n_slots`, which is right for the first
    ordering and, on the second, picks a contiguous block lying entirely inside the FIRST pair.
    Measured before the fix: two pairs in, one target scored, one dropped without a word.

    `max_batches` warns when it drops targets. `batches_per_pair` is documented as the knob that
    CANNOT -- "keeps EVERY entry and thins the WINDOWS" -- which is exactly why its failing silently
    was worse than the one that shouts.
    """
    probe = make_dataset(paired_source, "type1", train=True, batch_size=2, seed=0)
    model = _model(probe)
    ds = _eval_ds(paired_source, model)
    n_pairs = len(paired_source.regime.eval_pairs)
    assert n_pairs >= 2, "a one-pair fixture cannot see this bug"
    full, _ = build_eval_units(model, ds, DEVICE)
    every_target = {u.imp_biosample for u in full if u.imp_biosample}
    for k in (1, 2):
        units, _ = build_eval_units(model, ds, DEVICE, batches_per_pair=k)
        assert {u.imp_biosample for u in units if u.imp_biosample} == every_target, (
            f"batches_per_pair={k} dropped a target")


def test_the_cycle_count_is_asked_of_the_dataset_not_derived_from_the_batch_index(paired_source):
    """The two loaders divide the eval chromosome differently, so only they can answer this.

    The store hands EVERY window to EVERY pair, so a pair's share is the whole chromosome. The h5
    walks the pool once and cycles the pair index, so a pair's share is the pool divided by the
    pair count. That is the "position scope" difference the t22 equivalence report separated out --
    a real difference in what a target is scored on, not an iteration detail.
    """
    probe = make_dataset(paired_source, "type1", train=True, batch_size=2, seed=0)
    ds = _eval_ds(paired_source, _model(probe))
    import math
    assert ds.eval_batches_per_pair() == math.ceil(len(ds._eval_indices) / ds.batch_size)


def test_more_coverage_means_more_windows_per_target_not_more_targets(paired_source):
    probe = make_dataset(paired_source, "type1", train=True, batch_size=2, seed=0)
    model = _model(probe)
    ds = _eval_ds(paired_source, model)
    a, _ = build_eval_units(model, ds, DEVICE, batches_per_pair=1)
    b, _ = build_eval_units(model, ds, DEVICE, batches_per_pair=2)
    assert len(b) > len(a)
    assert ({u.imp_biosample for u in a if u.imp_biosample}
            == {u.imp_biosample for u in b if u.imp_biosample})
