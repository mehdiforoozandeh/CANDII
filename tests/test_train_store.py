"""t23 — `train.py` trains off a CANDI_STORE, and the old `--h5` path is untouched.

Everything here runs against a **real** synthetic store built by `candi.store.writer` (the
`make_store` fixture `tests/test_store_reader.py` owns) and, for the h5 side, against a real
schema-v2 h5 (`tests/test_bake_gates.py::write_v2_h5`). Neither is hand-rolled here: a test that
built its own files would be checking this module against itself.

The last test drives the actual `candi.train.main()` entry point through a handful of optimizer
steps. That is the only thing that proves the wiring — a factory that returns a dataset and a
`train_and_eval` that accepts a regime can both be correct while the loop still cannot take a
step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from candi.dataset import CandiKitH5Dataset
from candi.store.layout import StoreError
from candi.store.regime import Regime
from candi.train import DataSource, build_parser, make_dataset

from tests.test_store_reader import ASSAYS, make_store
from tests.test_store_regime import CTX, regime_dict


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("trainstore"))


@pytest.fixture(scope="module")
def regime_file(tmp_path_factory, store) -> Path:
    """A regime over the synthetic store, written to disk so it has a sha256 to record."""
    p = tmp_path_factory.mktemp("regime") / "regime.json"
    p.write_text(json.dumps(regime_dict(store), indent=2), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def v2_h5(tmp_path_factory) -> Path:
    from tests.test_bake_gates import write_v2_h5
    return write_v2_h5(tmp_path_factory.mktemp("t23h5") / "v2.h5")


@pytest.fixture()
def store_source(regime_file) -> DataSource:
    return DataSource.resolve(store=str(regime_file))


# ---------------------------------------------------------------------------------------------
# 1. exactly one of --h5 / --store, both directions
# ---------------------------------------------------------------------------------------------


def test_the_cli_refuses_neither_and_refuses_both(regime_file, tmp_path):
    """Argparse itself, on the submit line — not three frames into `train_and_eval`."""
    ap = build_parser()
    base = ["--out-dir", str(tmp_path)]

    with pytest.raises(SystemExit):                       # neither
        ap.parse_args(base)
    with pytest.raises(SystemExit):                       # both
        ap.parse_args(base + ["--h5", "a.h5", "--store", str(regime_file)])

    assert ap.parse_args(base + ["--h5", "a.h5"]).store is None
    assert ap.parse_args(base + ["--store", str(regime_file)]).h5 is None
    # `--regime-file` is the same destination, not a second knob.
    assert ap.parse_args(base + ["--regime-file", str(regime_file)]).store == str(regime_file)


def test_h5_is_no_longer_required_but_did_not_become_optional(tmp_path):
    """The distinction the task turns on: the GROUP is required, so omitting both still fails."""
    ap = build_parser()
    h5_action = next(a for a in ap._actions if "--h5" in a.option_strings)
    assert h5_action.required is False
    with pytest.raises(SystemExit):
        ap.parse_args(["--out-dir", str(tmp_path)])


@pytest.mark.parametrize("kw", [{}, {"h5": "a.h5", "store": "r.json"}])
def test_the_python_api_refuses_neither_and_both_too(kw):
    """Argparse guards the CLI; this guards every programmatic caller of `train_and_eval`."""
    with pytest.raises(ValueError, match="exactly one of h5 / store"):
        DataSource.resolve(**kw)


def test_train_and_eval_refuses_neither_and_both(tmp_path):
    from candi.train import train_and_eval
    with pytest.raises(ValueError, match="exactly one of h5 / store"):
        train_and_eval(out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="exactly one of h5 / store"):
        train_and_eval(h5_path="a.h5", store="r.json", out_dir=str(tmp_path))


def test_the_two_regime_flags_say_they_are_different_things():
    """`--regime` (masking) and `--store`/`--regime-file` collide by name and must not by meaning."""
    import candi.train as T
    assert "--regime" in T.STORE_HELP and "MASKING" in T.STORE_HELP
    assert "--store" in T.MASK_REGIME_HELP and "MASKING" in T.MASK_REGIME_HELP


# ---------------------------------------------------------------------------------------------
# 2. the factory: the store branch
# ---------------------------------------------------------------------------------------------


def test_the_store_branch_yields_a_training_batch_with_num_cells_zero(store_source):
    from candi.store.dataset import StoreDataset

    ds = make_dataset(store_source, "type1", train=True, batch_size=2, seed=0)
    assert isinstance(ds, StoreDataset)
    # The whole reason `num_cells` had to exist: `build_model(num_cells=0)` builds NO
    # `cell_embedding`, so the model is the historical 4-row one rather than a 5-row one whose
    # extra row is constant.
    assert ds.num_cells == 0
    assert ds.num_assays == len(ASSAYS) and ds.context_bins == CTX

    batch = next(iter(ds))
    # The training-path key set, taken from the old loader's contract rather than typed out here.
    want = {"x_data", "x_meta", "x_avail", "x_dna", "y_data", "y_meta", "y_avail", "y_pval",
            "y_peaks", "control_data", "control_meta", "control_avail", "x_dsf", "y_dsf",
            "control_x_dsf", "biosample_name", "region_type", "window_idx"}
    assert want <= set(batch)
    assert batch["x_data"].shape == (2, CTX, len(ASSAYS))


def test_num_cells_zero_reaches_build_model_without_the_fifth_row(store_source):
    """`ds.num_cells` flows into `arch`, and at 0 the encoder stays 4-row and owns no cell table."""
    from candi.model import build_model

    ds = make_dataset(store_source, "type1", train=True, batch_size=2, seed=0)
    torch.manual_seed(0)
    model = build_model(num_assays=ds.num_assays, context_length=ds.context_bins,
                        resolution=ds.resolution, num_cells=ds.num_cells,
                        depth_center=ds.depth_center(), d_model=32, embed_dim=8,
                        n_transformer_layers=1)
    assert model.encoder.metadata_embedding.n_rows == 4
    assert not hasattr(model.encoder.metadata_embedding, "cell_embedding")


def test_the_store_derives_a_depth_centre_the_way_the_h5_does(store_source):
    """Median log2(depth) over the available columns of the split — the store's `h5_depth_center`."""
    ds = make_dataset(store_source, "type1", train=True, batch_size=2, seed=0)
    assert ds.depth_center() == pytest.approx(float(np.log2(20_000_000)))


# ---------------------------------------------------------------------------------------------
# 3. the factory: the h5 branch is the object it always was
# ---------------------------------------------------------------------------------------------


def test_the_factory_returns_the_unchanged_old_dataset_for_h5(v2_h5):
    """Same class, same values, for every attribute `train.py` reads off a dataset."""
    direct = CandiKitH5Dataset(str(v2_h5), "type1", train=True, batch_size=2,
                               biosample_prefix="T_", dsf_sampling="uniform", seed=3,
                               shuffle=True, h5_cache_ram=False, cell_cond="off", reference=None)
    viafac = make_dataset(DataSource.resolve(h5=str(v2_h5)), "type1", train=True, batch_size=2,
                          dsf_sampling="uniform", seed=3, h5_cache_ram=False, cell_cond="off")
    assert type(viafac) is CandiKitH5Dataset is type(direct)
    for attr in ("num_assays", "context_bins", "resolution", "num_cells", "assays", "dsf_list",
                 "train_chroms", "eval_chroms", "signal_dim", "meta_rows", "biosample_prefix",
                 "seed", "shuffle", "batch_size", "dsf_sampling", "cell_cond", "train"):
        assert getattr(viafac, attr) == getattr(direct, attr), attr
    assert viafac.estimate_steps_per_epoch() == direct.estimate_steps_per_epoch()


def test_the_full_coverage_site_still_selects_one_biosample_on_each_path(v2_h5, store_source):
    """`only=` is the one knob the two paths spell differently, and it must mean the same thing."""
    h5ds = make_dataset(DataSource.resolve(h5=str(v2_h5)), "type1", batch_size=2, only="T_a")
    assert h5ds.biosample_prefix == "T_a"
    stds = make_dataset(store_source, "type1", batch_size=2, only="T_aa")
    assert stds.biosample_pool == ["T_aa"]


@pytest.mark.parametrize("label,old_kw,new_kw", [
    ("sampled",       dict(biosample_prefix="T_", shuffle=True), dict(only=None, shuffle=True)),
    ("full_coverage", dict(biosample_prefix="T_a", shuffle=True, h5_cache_ram=False),
                      dict(only="T_a", shuffle=True, h5_cache_ram=False)),
    ("probe",         dict(h5_cache_ram=False), dict(h5_cache_ram=False)),
])
def test_each_h5_call_site_yields_the_same_batches_through_the_factory(v2_h5, label, old_kw, new_kw):
    """The pre-t23 construction line for each of the three sites, against the factory. Bit for bit.

    `tools/golden.py` proves the MODEL did not move; this proves the DATA did not, which is the
    half of "the old path is unchanged" that a golden cannot see.
    """
    import itertools

    old = CandiKitH5Dataset(str(v2_h5), "type1", train=True, batch_size=2, dsf_sampling="uniform",
                            seed=7, cell_cond="off", reference=None, **old_kw)
    new = make_dataset(DataSource.resolve(h5=str(v2_h5)), "type1", train=True, batch_size=2,
                       dsf_sampling="uniform", seed=7, cell_cond="off", reference=None, **new_kw)
    n = 0
    for a, b in itertools.islice(zip(iter(old), iter(new)), 8):
        assert set(a) == set(b)
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k]), (label, k)
            else:
                assert a[k] == b[k], (label, k)
        n += 1
    assert n, "the fixture yielded no batches, so this test asserted nothing"


# ---------------------------------------------------------------------------------------------
# 4. what the store path refuses, loudly
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["id", "random"])
def test_cell_cond_on_the_store_path_fails_rather_than_being_ignored(store_source, mode):
    """Silently dropping it would train a 4-row model while the run json claimed cell_cond=id."""
    with pytest.raises(StoreError, match="cell_cond"):
        make_dataset(store_source, "type1", batch_size=2, cell_cond=mode)


def test_cell_cond_on_the_store_path_stops_train_and_eval_too(regime_file, tmp_path):
    from candi.train import train_and_eval
    with pytest.raises(StoreError, match="cell_cond"):
        train_and_eval(store=str(regime_file), out_dir=str(tmp_path), cell_cond="id")


def test_type2_loci_is_refused_on_the_store_path(store_source):
    with pytest.raises(ValueError, match="h5-only"):
        make_dataset(store_source, "type2_loci", batch_size=2)


def test_the_reference_arm_is_refused_on_the_store_path(regime_file, tmp_path):
    from candi.train import train_and_eval
    with pytest.raises(ValueError, match="h5-only"):
        train_and_eval(store=str(regime_file), out_dir=str(tmp_path), reference="on")


# ---------------------------------------------------------------------------------------------
# 5. the run record — and a real training run to produce it
# ---------------------------------------------------------------------------------------------


def test_the_h5_paths_provenance_block_is_what_it_always_was(v2_h5):
    """One new key (`data_source`) and the same `h5` spelling every existing run json uses."""
    prov = DataSource.resolve(h5=str(v2_h5)).provenance()
    assert prov == {"data_source": "h5", "h5": str(v2_h5)}


@pytest.fixture(scope="module")
def store_run(tmp_path_factory, regime_file) -> dict:
    """Drive the REAL `candi.train.main()` for a handful of optimizer steps off the store.

    Module-scoped: it is the expensive fixture, and the run json it produces is what the
    provenance assertions below read.
    """
    import candi.train as T

    out = tmp_path_factory.mktemp("storerun")
    argv = ["candi.train", "--store", str(regime_file), "--out-dir", str(out),
            "--tag", "t23_smoke", "--epochs", "1", "--steps-per-epoch", "6",
            "--batch-size", "2", "--d-model", "32", "--embed-dim", "8",
            "--n-transformer-layers", "1", "--seed", "0"]
    old = sys.argv
    sys.argv = argv
    try:
        T.main()
    finally:
        sys.argv = old
    return json.loads((out / "t23_smoke.json").read_text(encoding="utf-8"))


def test_a_real_run_off_the_store_actually_takes_optimizer_steps(store_run):
    losses = store_run["train_losses"]
    assert len(losses) == 6, losses
    assert all(np.isfinite(x) for x in losses), losses
    # It is a negative log-likelihood of raw counts: strictly positive, and not a constant.
    assert min(losses) > 0.0
    assert len(set(losses)) > 1, "the loss never moved — the optimizer did not step"


def test_the_run_record_says_which_path_and_carries_both_hashes(store_run, regime_file, store):
    import hashlib

    from candi.store import layout as L

    cfg = store_run["config"]
    assert cfg["data_source"] == "store"
    assert cfg["h5"] is None
    assert cfg["store"] == str(store)
    assert cfg["regime_file"] == str(regime_file)

    raw = regime_file.read_text(encoding="utf-8")
    assert cfg["regime_json"] == raw, "the regime must be recorded VERBATIM, not re-serialised"
    assert cfg["regime_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert cfg["store_manifest_sha256"] == hashlib.sha256(
        Path(L.manifest_path(store)).read_bytes()).hexdigest()
    # And it round-trips: the recorded text alone rebuilds the regime that ran.
    assert Regime.from_dict(json.loads(cfg["regime_json"])).assays == tuple(ASSAYS)


def test_full_coverage_trains_off_the_store_too(regime_file, tmp_path):
    """The third construction site. It iterates the regime's declared train split, not `T_*`."""
    from candi.train import train_and_eval

    res = train_and_eval(store=str(regime_file), out_dir=str(tmp_path), epochs=1, batch_size=2,
                         full_coverage=True, d_model=32, embed_dim=8, n_transformer_layers=1,
                         seed=0)
    assert len(res["train_losses"]) >= 5
    assert all(np.isfinite(x) for x in res["train_losses"])


def test_full_coverage_needs_the_regime_to_declare_its_train_split(tmp_path, store):
    """Guessing "all of them" would put the eval biosamples into training."""
    from candi.train import _coverage_biosamples

    p = tmp_path / "nosplit.json"
    p.write_text(json.dumps(regime_dict(store, biosamples={"eval": ["V_aa"]})), encoding="utf-8")
    with pytest.raises(ValueError, match="biosamples.train"):
        _coverage_biosamples(DataSource.resolve(store=str(p)))


def test_the_store_run_records_num_cells_zero_and_scores_nothing(store_run):
    cfg = store_run["config"]
    assert cfg["num_cells"] == 0 and cfg["cell_cond"] == "off"
    assert cfg["num_assays"] == len(ASSAYS) and cfg["context_bins"] == CTX
    # t14: no eval keys at all, rather than an M1 pooled over zero targets that reads as a result.
    for key in ("M1", "M2", "M3", "S14"):
        assert key not in store_run, f"{key} must be absent on the store path, not empty"
