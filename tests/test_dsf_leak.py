"""t16 — the identity-copy leak the DSF ladder gives the model, pinned as a rate.

Every batch draws a per-assay `x_dsf` and `y_dsf`. When the two land on the same level, the input
column for that assay can be the *same numbers* as the target column — a free identity copy. It is
not a bug in either loader: it is what "sample x and y from one ladder" means. What differs between
the two data paths is how often it bites, and that is what these pin, because the rate is a property
of the SAMPLING RULE and not of the data — so a synthetic fixture measures it exactly.

`tools/dsf_leak.py` is the instrument; this file is the standing check that the three structural
claims below stay true.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dsf_leak import measure                                    # noqa: E402

from candi.dataset import CandiKitH5Dataset                           # noqa: E402
from candi.store.dataset import StoreDataset                          # noqa: E402
from candi.store.regime import Regime                                 # noqa: E402
from tests.test_bake_gates import write_v2_h5                         # noqa: E402
from tests.test_store_regime import regime_dict                       # noqa: E402
from tests.test_store_reader import make_store                        # noqa: E402

N_BATCHES = 24


@pytest.fixture(scope="module")
def bake(tmp_path_factory):
    return write_v2_h5(tmp_path_factory.mktemp("leak") / "leak.h5",
                       num_assays=8, n_train=64, n_eval=8, order=("T_a", "V_a"))


@pytest.fixture(scope="module")
def store_regime(tmp_path_factory):
    root = make_store(tmp_path_factory.mktemp("leakstore"))
    return Regime.from_dict(regime_dict(root, biosamples={"train": ["T_aa"], "eval": ["V_aa"]}))


def _bake_ds(path, mode):
    return CandiKitH5Dataset(path, "type1", train=True, batch_size=8, dsf_sampling=mode,
                             seed=0, shuffle=False, h5_cache_ram=False)


def _store_ds(reg, mode, deterministic):
    return StoreDataset(reg, train=True, batch_size=2, dsf_sampling=mode, seed=0,
                        shuffle=False, deterministic=deterministic)


def test_the_bake_leaks_on_every_equal_dsf_column(bake):
    """The ladder is MATERIALIZED, so equal `d` is one stored array read twice. Always identical.

    At the shipped `--dsf-sampling uniform` over `{1, 2, 4, 8}` that is `1/K` = 25% of available
    columns, and no RNG anywhere can change it.
    """
    r = measure(_bake_ds(bake, "uniform"), N_BATCHES)
    assert r["identical_given_equal_dsf"] == 1.0
    assert 0.15 < r["rate_equal_dsf"] < 0.35            # 1/4, with fixture-sized sampling noise
    assert r["rate_identical"] == r["rate_equal_dsf"]


def test_x_eq_y_makes_every_available_column_an_identity_copy(bake):
    """The mode exists as a control, and this is the cost of using it as anything else."""
    r = measure(_bake_ds(bake, "x_eq_y"), N_BATCHES)
    assert r["rate_equal_dsf"] == 1.0 and r["rate_identical"] == 1.0


def test_the_store_leaks_only_at_dsf_1_while_training(store_regime):
    """D6 — the store GENERATES the ladder, so equal `d` is two independent binomial draws.

    They collide only at `d == 1`, where `_thin` is the identity rather than a draw. Under uniform
    sampling over four levels that is `1/K^2` = 1/16, not `1/K`.
    """
    r = measure(_store_ds(store_regime, "uniform", deterministic=False), N_BATCHES)
    assert r["identical_given_equal_dsf"] is not None
    assert r["identical_given_equal_dsf"] < 1.0
    assert r["rate_identical"] < r["rate_equal_dsf"]
    # every identical column is a DSF-1 column, and nothing above it is
    lv = r["by_equal_dsf_level"]
    for d in (2, 4, 8):
        assert lv.get(f"dsf{d}_identical", 0) == 0, f"a DSF-{d} pair collided while training"
    assert lv.get("dsf1_identical", 0) == lv.get("dsf1", 0)


def test_the_store_leaks_at_every_level_under_the_deterministic_rng(store_regime):
    """The finding t16 turned up, and the reason the leak is NOT simply 16x smaller on the store.

    D22's counter-based eval RNG seeds each draw from
    `(run_seed, biosample, assay, chrom, window_start, dsf_milli(dsf))`. There is NO x/y term in
    that tuple, so when `x_dsf == y_dsf` the two `_thin` calls build the SAME generator and return
    the SAME draw — the store reproduces the bake's leak in full, at `1/K`, not `1/K^2`.

    It is latent today only because `eval.py::build_eval_units` passes `dsf_sampling="off"`, which
    pins every level at 1. Any deterministic dataset with the ladder on collides everywhere. This
    test pins the CURRENT behaviour, not the desired one: adding an x/y term to the seed would move
    every deterministic eval number, so it is a decision, not a cleanup.
    """
    r = measure(_store_ds(store_regime, "uniform", deterministic=True), N_BATCHES)
    assert r["identical_given_equal_dsf"] == 1.0
    lv = r["by_equal_dsf_level"]
    for d in (1, 2, 4, 8):
        if lv.get(f"dsf{d}", 0):
            assert lv[f"dsf{d}_identical"] == lv[f"dsf{d}"], f"DSF-{d} did not collide"


def test_the_leak_lands_on_the_denoising_half_only(bake):
    """A clozed column's input is the CLOZE sentinel, so there is nothing there to copy.

    So the leak is entirely on the OBSERVED half of the objective — the denoising term, which
    `EVAL.md` says nothing scores, and whose loss curve is the one that moves first. That is why a
    leak this size can make one data path look like it descends faster without any of it reaching
    the number the leaderboard would read.
    """
    r = measure(_bake_ds(bake, "off"), N_BATCHES)
    assert r["rate_identical"] == 1.0                        # every column, at DSF 1 throughout
    assert 0.3 < r["rate_leak_observed"] < 0.8               # only the unclozed share of them
    assert r["leak_observed"] < r["identical"]
