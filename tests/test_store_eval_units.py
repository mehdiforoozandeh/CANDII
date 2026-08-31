"""t28 — what survives of it now that the scorer it fixed has been deleted.

t28's finding was that `build_eval_units` CONSTRUCTED its own `CandiKitH5Dataset` from a path, and
that single line was the whole reason a store-backed run could not be evaluated. The fix made the
builder take a ready dataset. `candi.eval` — builder, `quick_eval` and all — is now deleted (D15)
and `candi.monitor` scores every 25 bp bin of the regime's eval chromosomes instead, so the window
thinning those tests pinned (`batches_per_pair`, the slot arithmetic, the cycle count) describes
nothing that runs. `tests/test_monitor.py` is where "a store-backed run is scorable" is tested now.

Two things t28 established do outlive it, and they are what is left here:

- **`DataSource.eval_pairs_declared()`** — the guard `train.py` still uses to decide whether a
  mid-training eval has anything to select on. It is `train.py`'s own, not the scorer's.
- **A store eval dataset re-iterates to the same windows.** The monitor pins its window plan by
  opening ONE source outside the hook; a loader that handed back different windows on the second
  pass would break that pairing whatever the scorer is.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from candi.store import layout as L
from candi.store.regime import Regime
from candi.train import DataSource, make_dataset

from tests.test_store_reader import make_store
from tests.test_store_regime import regime_dict


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
    # batch belongs to the only target there is. The bug this file once pinned needed two.
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


# ---------------------------------------------------------------------------------------------
# the window plan: one dataset, re-iterated, is the same windows
# ---------------------------------------------------------------------------------------------


def test_one_dataset_re_iterated_gives_the_same_windows_every_time(paired_source):
    """Why the mid-training eval opens its source ONCE and re-iterates it.

    Selection compares epoch 6 against epoch 12, and that comparison is only paired if both saw the
    same positions. `Monitor.__init__` opens `bench.harness.open_source` outside the hook for
    exactly this reason. The loader assumption underneath is the same either way, and this is it.
    """
    ds = make_dataset(paired_source, "type1", train=False, batch_size=2, dsf_sampling="off",
                      seed=0, shuffle=False, h5_cache_ram=False, cell_cond="off")
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
    """It derives its pairing from the `T_`/`V_`/`B_` prefixes, so it can never lack one.

    Note this no longer means an h5 run gets a mid-training eval: `train.py` turns `--eval-every`
    off on the h5 path outright, because the scorer that served it was `eval.quick_eval` and
    `candi.eval` is deleted (D15). What this answers is "could this source supply imputation
    targets", which is still the right question for a source and still True here.
    """
    src = DataSource.coerce(str(tmp_path_factory.mktemp("t28h5") / "nothing.h5"))
    assert src.eval_pairs_declared() is True
