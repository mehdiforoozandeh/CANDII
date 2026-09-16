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
from candi.decoder import HEADS
from candi.store.layout import StoreError
from candi.store.regime import Regime
from candi.train import DataSource, build_parser, make_dataset

from tests.test_store_reader import ASSAYS, make_store
from tests.test_store_regime import CTX, REPO, regime_dict


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


@pytest.fixture(scope="module")
def monitored_run(tmp_path_factory) -> dict:
    """A REAL run with the monitor live: a regime that declares `eval_pairs`, `--eval-every 1`.

    The store above declares no pairs, so `--eval-every` resolves to 0 there and the whole
    monitor branch of `train_and_eval` goes unexercised. This is the fixture that drives it, and it
    is the only place the CADENCE wiring is checked end to end — `tests/test_monitor.py` checks what
    `check()` and `final_check()` each produce, not which one the training loop calls.
    """
    from candi.train import train_and_eval
    from tests.test_monitor import EVAL_TRACKS

    root = tmp_path_factory.mktemp("monrun")
    st = make_store(root / "store", tracks=EVAL_TRACKS)
    p = root / "regime.json"
    p.write_text(json.dumps(regime_dict(
        st, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
        kinds=["counts", "peaks", "pval"],
        eval_pairs=[["T_aa", "V_aa"]]), indent=2), encoding="utf-8")
    out = root / "out"
    # Created here rather than left to `train_and_eval`: `_keep_best` writes `<tag>.best.ckpt` from
    # inside the eval hook, which runs BEFORE the `mkdir` that guards the last-checkpoint write.
    # That ordering is a pre-existing bug in `train.py` and is not this test's subject.
    out.mkdir(parents=True, exist_ok=True)
    return train_and_eval(store=str(p), out_dir=str(out), epochs=1,
                          steps_per_epoch=2, batch_size=2, d_model=32, embed_dim=8,
                          n_transformer_layers=1, seed=0, eval_every=1, eval_batch_size=2,
                          ckpt_path=str(out / "cadence.ckpt"))


def test_mid_training_checks_run_the_impute_dial_alone(monitored_run) -> None:
    """The PI's cost ruling, as the shape of the run json (`cruxvault/results/t30/TIMING.md`)."""
    curve = monitored_run["eval_curve"]
    assert curve, "the monitor never fired, so this test asserted nothing"
    for row in curve:
        assert row["impute"]["n_tracks"] > 0
        assert "denoise" not in row, "the denoise dial rode along on a mid-training check"
        assert "gap" not in row, "an impute-only row cannot carry the overfitting alarm"


def test_the_end_of_run_check_is_the_one_that_carries_both_dials_and_the_gap(monitored_run) -> None:
    fin = monitored_run["final_check"]
    assert fin is not None and fin["final"] is True
    assert fin["impute"]["n_tracks"] > 0 and fin["denoise"]["n_tracks"] > 0
    assert fin["gap"], "the end-of-run overfitting alarm reported nothing"
    for k, v in fin["gap"].items():
        assert v == pytest.approx(fin["impute"]["macro"][k] - fin["denoise"]["macro"][k])
    # It names the weights it scored, because "best" and "last" are not the same model.
    assert fin["selected"] == monitored_run["best_checkpoint"]["scored"]


def test_the_store_run_records_num_cells_zero_and_scores_nothing(store_run):
    cfg = store_run["config"]
    assert cfg["num_cells"] == 0 and cfg["cell_cond"] == "off"
    assert cfg["num_assays"] == len(ASSAYS) and cfg["context_bins"] == CTX
    # t14: no eval keys at all, rather than an M1 pooled over zero targets that reads as a result.
    for key in ("M1", "M2", "M3", "S14"):
        assert key not in store_run, f"{key} must be absent on the store path, not empty"


# ---------------------------------------------------------------------------------------------
# a head is supervised by a store layer, and the regime is what loads the layer
# ---------------------------------------------------------------------------------------------

#: Which store layer carries each head's target. Driven off `decoder.HEADS` by the test below, so a
#: fourth head cannot be added without declaring the layer that supervises it.
HEAD_TARGET_KIND = {"count": "counts", "signal": "pval", "peak": "peaks"}

#: The two live regimes of `plan/BENCHMARK_DESIGN.md` §3. `regime.eic_smoke.json` and
#: `regime.equiv.json` are not here on purpose: neither trains a board row.
LIVE_REGIMES = ("regime.eic_19.json", "regime.eic_pilot.json")


def test_every_head_declares_the_layer_that_supervises_it():
    """Fails when a head is added to `decoder.HEADS` and nothing says what its target layer is."""
    assert set(HEAD_TARGET_KIND) == set(HEADS), (
        f"HEADS={HEADS} but HEAD_TARGET_KIND covers {sorted(HEAD_TARGET_KIND)}; a head with no "
        "declared target layer cannot be checked against a regime's `kinds`"
    )


@pytest.mark.parametrize("name", LIVE_REGIMES)
def test_a_live_regime_loads_the_layer_behind_every_head(name):
    """A live regime must declare `kinds` for every head, because omitting one is SILENT.

    `StoreDataset._batch` allocates `y_pval` and `y_peaks` as `torch.zeros` and fills a column only
    when `"<kind>" in self.regime.kinds` (`store/dataset.py:605-606`). When the regime omits the
    kind the tensor stays **all zero** and is emitted anyway.

    Nothing downstream catches that. `batch.prepare_masked_batch` gates the auxiliary heads on
    `signal_observed_map` / `signal_masked_map`, which require `y_avail > 0` — and `y_avail` is set
    from the **counts** availability (`store/dataset.py:601`), not from the layer the head actually
    reads. So a `--heads count,signal` run off a regime whose `kinds` lacks `pval` trains its
    Gaussian head against a target of 0.0 on every present column. A softplus mean reaches 0
    happily, so the loss descends and the head learns to predict nothing, with no error raised.
    That is `AGENTS.md` invariant 8's failure family, and this test is what stands in for it.

    The narrower case the same mechanism produces — a regime that DOES declare `pval` but a column
    whose biosample lacks that layer — is not covered here and is on record as needing a ruling.
    """
    kinds = set(Regime.from_file(REPO / "configs" / name).kinds)
    missing = {h: k for h, k in HEAD_TARGET_KIND.items() if k not in kinds}
    assert not missing, (
        f"configs/{name} declares kinds={sorted(kinds)}, so head(s) "
        f"{sorted(missing)} would be supervised on a plane of zeros: "
        + ", ".join(f"{h} needs '{k}'" for h, k in sorted(missing.items()))
    )


# ---------------------------------------------------------------------------------------------
# t85 — training stops when the validation metric stops improving
# ---------------------------------------------------------------------------------------------
#
# Job 57620803_0 took its best `V_` checkpoint at epoch 2 and then ran nine more GPU-hours while
# `V_imp_crps` rose 0.5604 -> 0.5820 -> 0.5864. Nothing in the loop could stop it. These pin the
# two halves of the fix: `train()` honours a hook that asks to stop, and the patience that decides
# is counted in EPOCHS.


@pytest.mark.parametrize("full_coverage", [False, True],
                         ids=["sampled-path", "full-coverage-path"])
def test_a_hook_that_asks_to_stop_stops_the_epoch_loop(regime_file, full_coverage):
    """Both construction sites, because the break had to be written into each loop separately.

    A stub hook rather than a real metric: whether a 32-unit model's CRPS happens to rise on
    epoch 3 of a synthetic store is not a property this test may depend on. What is being checked
    is the wiring — a truthy return ends training — and that is deterministic.
    """
    from candi.model import build_model
    from candi.train import train

    src = DataSource.resolve(store=str(regime_file))
    ds = make_dataset(src, "type1", train=True, batch_size=2, seed=0)
    model = build_model(num_cells=0, num_assays=ds.num_assays, context_length=ds.context_bins,
                        d_model=32, embed_dim=8, n_transformer_layers=1)
    calls = []

    def hook(step, ep):
        calls.append(ep)
        return len(calls) == 2          # stop at the second eval, whatever epoch that is

    train(model, src, "cpu", epochs=8, steps_per_epoch=2, batch_size=2, seed=0,
          full_coverage=full_coverage, eval_hook=hook, eval_every=1)
    assert calls == [0, 1], f"the loop ran past the stop request: hook saw epochs {calls}"


def test_a_hook_that_returns_none_never_stops_anything(regime_file):
    """Every caller predating t85 returns None, and none of them may change behaviour."""
    from candi.model import build_model
    from candi.train import train

    src = DataSource.resolve(store=str(regime_file))
    ds = make_dataset(src, "type1", train=True, batch_size=2, seed=0)
    model = build_model(num_cells=0, num_assays=ds.num_assays, context_length=ds.context_bins,
                        d_model=32, embed_dim=8, n_transformer_layers=1)
    calls = []
    train(model, src, "cpu", epochs=4, steps_per_epoch=2, batch_size=2, seed=0,
          eval_hook=lambda step, ep: calls.append(ep), eval_every=1)
    assert calls == [0, 1, 2, 3], f"a None-returning hook changed the epoch count: {calls}"


def test_the_patience_is_counted_in_epochs_and_defaults_to_three():
    """`--early-stop-epochs 3` must mean three EPOCHS, not three evals.

    The distinction is load-bearing: `--eval-every` is 3 by default, so a patience read as three
    *evals* would be nine epochs, three times what the operator asked for.
    """
    import inspect
    from candi.train import train_and_eval

    assert inspect.signature(train_and_eval).parameters["early_stop_epochs"].default == 3
    src = inspect.getsource(train_and_eval)
    assert "(ep - best[\"epoch\"]) > early_stop_epochs" in src, \
        "the stop condition must compare EPOCH numbers, not a count of evals"
    assert "if early_stop_epochs and" in src, "0 must switch it off"


def test_the_cli_exposes_the_patience_and_says_zero_is_off():
    import candi.train as T

    ap = T.build_arg_parser() if hasattr(T, "build_arg_parser") else None
    if ap is None:                       # the parser is built inside main(); read --help instead
        import subprocess
        out = subprocess.run([sys.executable, "-m", "candi.train", "--help"],
                             capture_output=True, text=True).stdout
        assert "--early-stop-epochs" in out
        assert "0 = off" in out


# ---------------------------------------------------------------------------------------------
# t89 — the selection scope, wired end to end through the training loop
# ---------------------------------------------------------------------------------------------
# `tests/test_bench_harness.py` checks that a scoped source scores the scoped bins and
# `tests/test_monitor.py` checks that a scoped row still selects. Neither can see the thing that
# only exists here: the loop runs TWO monitors, a cheap one for the curve and a full-coverage one
# for the row the run reports, and the run json has to say which was which.

@pytest.fixture(scope="module")
def scoped_run(tmp_path_factory):
    """The same one-epoch monitored run as `monitored_run`, with `--eval-regions` on.

    The BED covers part of the eval chromosome and its edges are off the 25 bp bin grid, as the
    hg38 Pilot Regions' are.
    """
    from candi.train import train_and_eval
    from tests.test_monitor import EVAL_TRACKS

    root = tmp_path_factory.mktemp("scopedrun")
    st = make_store(root / "store", tracks=EVAL_TRACKS)
    bed = root / "scope.bed"
    bed.write_text("chr2\t3210\t11190\tR0\n", encoding="utf-8")
    p = root / "regime.json"
    p.write_text(json.dumps(regime_dict(
        st, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
        kinds=["counts", "peaks", "pval"],
        eval_pairs=[["T_aa", "V_aa"]]), indent=2), encoding="utf-8")
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    res = train_and_eval(store=str(p), out_dir=str(out), epochs=1,
                         steps_per_epoch=2, batch_size=2, d_model=32, embed_dim=8,
                         n_transformer_layers=1, seed=0, eval_every=1, eval_batch_size=2,
                         eval_regions=str(bed), ckpt_path=str(out / "scoped.ckpt"))
    return res, bed


def test_the_run_json_says_which_positions_selected_the_checkpoint(scoped_run,
                                                                   monitored_run) -> None:
    """Two runs are comparable only if this matches. An unscoped run says `full` rather than
    omitting the key — "scored everything" and "the field was forgotten" must not look alike."""
    import hashlib

    res, bed = scoped_run
    assert monitored_run["config"]["eval_scope"]["name"] == "full"
    sc = res["config"]["eval_scope"]
    assert sc["name"] == "regions"
    assert sc["sha256"] == hashlib.sha256(bed.read_bytes()).hexdigest()
    assert 0.0 < sc["fraction"] < 1.0


def test_the_curve_is_cheap_and_the_end_of_run_row_is_not(scoped_run) -> None:
    """THE RULING THIS TEST EXISTS FOR. Cheap enough to select on every check; the number the run
    REPORTS is the thorough one. A run whose final row inherited the selection scope would report a
    region cut as if it were the panel."""
    res, _ = scoped_run
    curve = res["eval_curve"]
    assert curve, "the monitor never fired, so this test asserted nothing"
    for row in curve:
        assert row["eval_scope"]["name"] == "regions"
    fin = res["final_check"]
    assert fin is not None and fin["final"] is True
    assert fin["eval_scope"]["name"] == "full"
    assert fin["impute"]["n_tracks"] > 0 and fin["denoise"]["n_tracks"] > 0


def test_a_scoped_curve_scores_fewer_bins_than_the_row_the_run_reports(scoped_run) -> None:
    """The saving is in the BINS, not in the tracks — the same panel over fewer positions. That is
    the whole difference between this and the sampled scorer that was deleted, which bought its
    speed by dropping windows per track and therefore selected on a draw."""
    res, _ = scoped_run
    mid = res["eval_curve"][-1]
    fin = res["final_check"]
    assert set(mid["impute"]["per_track"]) == set(fin["impute"]["per_track"])
    assert mid["impute"]["n_tracks"] == fin["impute"]["n_tracks"] > 0
    assert mid["eval_scope"]["scored_bins"] < fin["eval_scope"]["scored_bins"]
    assert fin["eval_scope"]["fraction"] == 1.0
    assert mid["eval_scope"]["full_bins"] == fin["eval_scope"]["full_bins"]


def test_a_scope_with_no_check_to_scope_is_blanked_rather_than_recorded(tmp_path) -> None:
    """A `regions` block in a run json is a claim about a curve. With `--eval-every 0` there is no
    curve, so the claim would be about nothing."""
    from candi.train import train_and_eval
    from tests.test_monitor import EVAL_TRACKS

    st = make_store(tmp_path / "store", tracks=EVAL_TRACKS)
    bed = tmp_path / "scope.bed"
    bed.write_text("chr2\t3210\t11190\tR0\n", encoding="utf-8")
    p = tmp_path / "regime.json"
    p.write_text(json.dumps(regime_dict(
        st, biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
        kinds=["counts", "peaks", "pval"]), indent=2), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    res = train_and_eval(store=str(p), out_dir=str(out), epochs=1, steps_per_epoch=2,
                         batch_size=2, d_model=32, embed_dim=8, n_transformer_layers=1,
                         seed=0, eval_every=0, eval_batch_size=2, eval_regions=str(bed))
    assert res["config"]["eval_scope"]["name"] == "full"
    assert not res["eval_curve"]
