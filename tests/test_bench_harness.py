"""t20 — the harness and the CLI, against a REAL store built on disk by `candi.store.writer`.

Nothing here is mocked. `make_store` writes an actual `CANDI_STORE` (the same code that will build
EIC on Fir), a real `CandiModel` is constructed and saved as a real checkpoint, and the CLI is
driven through `main(argv)` exactly as a shell would drive it. A test that passes here is a test
against the on-disk contract.

Four things are pinned, and each of them is a way the harness could be silently wrong:

1. **Coverage.** Every 25 bp bin of every eval chromosome is scored, exactly once each, and the
   window plan is the harness's own full tiling rather than the regime's (D2).
2. **No drift from the loader.** The harness assembles its own batches; the DSF-1 batch it builds
   must equal the one `StoreDataset` builds for the same biosample and windows.
3. **The keys.** `EVAL_PLAN.md` §4 names them; a block that quietly stops emitting one is a block
   nobody notices is gone.
4. **The forbidden imports and flags.** `bench` must not reach `candi.train`, and the two flags
   that used to shrink a run unevenly must be refused by name.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # t80 — tools/ is importable

from candi._vendored import MISSING                                   # noqa: E402
from candi.bench import annotations as ann                            # noqa: E402
from candi.bench import harness as H
from candi.bench.harness import Pair, decode_groups, full_tiling, open_source
from candi.model import build_model                                   # noqa: E402
from candi.store import layout as L                                   # noqa: E402
from candi.store import layout as SL                                  # noqa: E402
from candi.store.dataset import StoreDataset                          # noqa: E402
from candi.store.regime import Regime                                 # noqa: E402

from tests.test_store_reader import ASSAYS, N_BINS, make_store
from tests.test_store_regime import CTX, regime_dict

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---------------------------------------------------------------------------
# a real store, a real regime file, a real checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("benchstore"))


@pytest.fixture(scope="module")
def regime_file(store, tmp_path_factory) -> Path:
    """Both biosamples are eval, so `pr_by_specificity` has more than one cell type to compare."""
    d = tmp_path_factory.mktemp("benchregime")
    obj = regime_dict(store, biosamples={"train": ["T_aa"], "eval": ["T_aa", "V_aa"]},
                      kinds=["counts", "peaks", "pval"])
    p = d / "regime.json"
    p.write_text(json.dumps(obj))
    return p


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    """A real `CandiModel` with all three heads, saved as a real state_dict."""
    torch.manual_seed(0)
    model = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                        depth_center=24.25, heads=("count", "signal", "peak"))
    p = tmp_path_factory.mktemp("benchckpt") / "model.pt"
    torch.save(model.state_dict(), p)
    return p


@pytest.fixture()
def source(regime_file):
    s = open_source(store=regime_file)
    yield s
    s.close()


@pytest.fixture(scope="module")
def model(ckpt):
    m = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                    depth_center=24.25, heads=("count", "signal", "peak"))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1 — D2: every bin, exactly once, from the harness's own plan
# ---------------------------------------------------------------------------

def test_the_tiling_covers_every_bin_and_ends_flush_with_the_chromosome() -> None:
    """The tail tile is pulled BACK rather than run past the end, so coverage is exact."""
    for n in (100, 128, 129, 255, 256, 2_000):
        starts = full_tiling(n, 64)
        covered = np.zeros(n, dtype=int)
        for s in starts:
            assert s + 64 <= n
            covered[s:s + 64] += 1
        assert covered.min() >= 1, f"n={n} leaves a hole"
        assert starts[-1] + 64 == n


def test_a_chromosome_shorter_than_the_context_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="no window to score"):
        full_tiling(30, 64)


def test_the_harness_plans_its_own_windows_not_the_regimes(source) -> None:
    """The regime's plan drops mask-rejected tiles; D2 does not allow that.

    `make_store` blacklists bins 100-400 of chr2 precisely so the regime plan has something to
    reject. The harness must score them anyway — a blacklist applies to peaks, never to signal (D6).
    """
    regime = Regime.from_file(source.regime_path)
    planned = {s for c, s in regime.windows(source.ds.corpus, "eval") if c == "chr2"}
    ours = set(source.windows("chr2"))
    assert ours - planned, "the harness plan is a subset of the regime's — the mask leaked into D2"
    covered = np.zeros(source.n_bins("chr2"), dtype=bool)
    for s in ours:
        covered[s:s + source.context_bins] = True
    assert covered.all()


def test_every_scored_bin_of_every_eval_chromosome_lands_in_the_record(source, model) -> None:
    rec = next(H.stream_tracks(model, source, "cpu", kind="impute", batch_windows=2,
                               pairs=[Pair("V_aa", "V_aa")]))
    assert set(rec.chroms) == set(source.eval_chroms)
    for c in rec.chroms:
        assert len(rec.counts[c]) == N_BINS[c]
        assert len(rec.mu[c]) == N_BINS[c]
    assert rec.n_bins() == sum(N_BINS[c] for c in source.eval_chroms)


# ---------------------------------------------------------------------------
# 2 — the harness's batch is the loader's batch
# ---------------------------------------------------------------------------

def test_the_harness_batch_equals_the_loaders_batch_for_the_same_windows(source, regime_file):
    """The harness drives its own windows; it must not also drift on tensor assembly.

    Same biosample, same window starts, DSF 1 on both sides. Every tensor the encoder or the loss
    reads has to agree bit for bit, or the harness is scoring a different input than training saw.
    """
    starts = source.windows("chr2")[:2]
    ours = source.batch(Pair("T_aa", "T_aa"), "chr2", starts, "denoise")

    ds = StoreDataset(Regime.from_file(regime_file), train=False, batch_size=2,
                      dsf_sampling="off", shuffle=False, deterministic=True)
    ds._windows = [("chr2", int(s)) for s in starts]
    theirs = ds._make_batch("T_aa", [0, 1], np.random.default_rng(int(ds.run_seed)))
    try:
        for k, v in theirs.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(ours[k], v), f"{k} differs between harness and loader"
    finally:
        ds.corpus.close()


def test_thinning_the_input_moves_the_depth_covariate_with_it(source) -> None:
    """`x_meta[0] -= log2(d)` is `prep/bake.py`'s F4 identity; without it the prompt lies."""
    starts = source.windows("chr2")[:2]
    one = source.batch(Pair("T_aa", "T_aa"), "chr2", starts, "denoise", x_dsf=1)
    four = source.batch(Pair("T_aa", "T_aa"), "chr2", starts, "denoise", x_dsf=4)
    avail = one["x_avail"][0] > 0
    assert avail.any()
    assert torch.allclose(four["x_meta"][:, 0, avail], one["x_meta"][:, 0, avail] - 2.0)
    assert four["x_data"][:, :, avail].sum() < one["x_data"][:, :, avail].sum()


def test_bench_thins_one_ladder_and_reads_it_twice(source) -> None:
    """t37 — the two bench thinning sites SHARE a seed on a self-pair, and must keep sharing it.

    `_thin_input` and `counts_at_dsf` both call `draw_seed` WITHOUT `side`. On a self-pair that
    makes them one generator over one array, so the input at depth `d` and the truth at depth `d`
    are the same bytes. That is the bake's behaviour, not t27's identity-copy leak: `H5Source`
    serves both reads out of a single `counts_dsf{d}` dataset, and a generated ladder is only a
    substitute for a stored one if it agrees with itself. Nothing compares the two arrays anyway —
    `_thin_input` feeds `depthblind`'s latents (no truth) and `counts_at_dsf` feeds `depthcounterfact` (latent at DSF 1).

    Passing `side=` here would move `depthcounterfact` and `depthblind` by pure realization noise and split the store from
    the h5. This test is the tripwire for that edit.
    """
    pair = Pair("T_aa", "T_aa")
    starts = source.windows("chr2")[:2]
    avail = (source.batch(pair, "chr2", starts, "denoise")["x_avail"][0] > 0).numpy()
    assert avail.any()
    for d in source.dsf_levels:
        if int(d) == 1:
            continue
        x = source.batch(pair, "chr2", starts, "denoise", x_dsf=int(d))["x_data"].numpy()
        y = source.counts_at_dsf(pair, "chr2", starts, int(d))
        assert np.array_equal(x[:, :, avail], y[:, :, avail]), (
            f"the two bench thinning sites diverged at dsf {int(d)}; if a `side=` was added to "
            f"draw_seed in _thin_input or counts_at_dsf, read cruxvault/results/t37/FINDING.md "
            f"before keeping it")
        assert x[:, :, avail].sum() < source.counts_at_dsf(
            pair, "chr2", starts, 1)[:, :, avail].sum()


def test_a_declared_pair_separates_the_two_thinning_sites_on_its_own(source) -> None:
    """The self-pair collision above is confined to the self-pair.

    `counts_at_dsf` seeds under the TARGET cell's name and `_thin_input` under the INPUT's, so a
    declared eval pair puts a different biosample in the entropy tuple and the two sites separate
    with no `side` term needed. This pins WHY §5 of the t37 finding is safe under declared pairs.
    """
    from candi.store.dataset import draw_seed, dsf_milli

    a = draw_seed(source._run_seed, "T_aa", "H3K4me3", "chr2", 0, dsf_milli(2))
    b = draw_seed(source._run_seed, "V_aa", "H3K4me3", "chr2", 0, dsf_milli(2))
    assert list(a.entropy) != list(b.entropy)
    assert len(list(a.entropy)) == 6, "bench must keep the pre-t27 six-element tuple"


# ---------------------------------------------------------------------------
# 2b — the DECLARED eval pairs (D31)
# ---------------------------------------------------------------------------
#
# The shared `TRACKS` layout has `V_aa` as a strict subset of `T_aa`, which is the right shape for
# testing MISSING and the wrong one for testing imputation: an imputation eval needs a prompt that
# LACKS something the truth carries. Here `T_aa` has no H3K4me3 and `V_aa` has it, so H3K4me3 is a
# real target — while ATAC-seq, which BOTH carry, is not one, and pins the filter. Two pairs, so
# declaration order is something the assertions can see.

PAIRED_TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "T_bb": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_bb": ("ATAC-seq", "H3K4me3"),
}
IMPUTED = ASSAYS.index("H3K4me3")
SHARED = ASSAYS.index("ATAC-seq")


@pytest.fixture(scope="module")
def paired_store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("pairedstore"), tracks=PAIRED_TRACKS)


@pytest.fixture(scope="module")
def paired_regime_file(paired_store, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("pairedregime")
    obj = regime_dict(paired_store,
                      biosamples={"train": ["T_aa", "T_bb"], "eval": ["V_aa", "V_bb"]},
                      kinds=["counts", "peaks", "pval"],
                      eval_pairs=[["T_aa", "V_aa"], ["T_bb", "V_bb"]])
    p = d / "regime.json"
    p.write_text(json.dumps(obj))
    return p


@pytest.fixture()
def paired_source(paired_regime_file):
    s = open_source(store=paired_regime_file)
    yield s
    s.close()


def test_a_declared_regime_pairs_the_prompt_cell_with_the_cell_holding_the_truth(paired_source):
    """D31 — the pairing is READ OFF the regime, in declaration order. No name is parsed (D16)."""
    assert H.cross_cell(Pair("T_aa", "V_aa"), "impute") is True
    assert paired_source.pairs("impute") == [Pair("T_aa", "V_aa"), Pair("T_bb", "V_bb")]
    # Denoising reads and scores ONE cell whatever the regime declares.
    assert H.cross_cell(Pair("T_aa", "T_aa"), "denoise") is False
    assert paired_source.pairs("denoise") == [Pair("T_aa", "T_aa"), Pair("T_bb", "T_bb")]


def test_a_target_is_an_assay_the_truth_cell_has_and_the_prompt_cell_does_not(paired_source):
    """Both halves of the rule. `H5Source.targets` and `StoreDataset._imp_keys` enforce the same."""
    cols = paired_source.targets(Pair("T_aa", "V_aa"), "impute")
    assert cols == [IMPUTED]
    assert SHARED not in cols, (
        "ATAC-seq is in the encoder input, so predicting it measures copying, not imputation")
    # The self-pair is unchanged: everything the cell has is a denoising target.
    assert paired_source.targets(Pair("T_aa", "T_aa"), "denoise") == [SHARED,
                                                                     ASSAYS.index("DNase-seq")]


def test_a_regime_with_no_declared_pairs_still_imputes_by_leaving_one_assay_out(source):
    """The pair-less path is untouched: self-pairs, and one forward pass per held-out assay."""
    assert H.cross_cell(Pair("T_aa", "T_aa"), "impute") is False
    assert source.pairs("impute") == [Pair("T_aa", "T_aa"), Pair("V_aa", "V_aa")]
    cols = source.targets(Pair("T_aa", "T_aa"), "impute")
    assert cols == [0, 1, 2]
    assert decode_groups(source, "impute", cols) == [[0], [1], [2]]


def test_the_batch_carries_the_target_cells_own_truth_and_covariates(paired_source):
    """The five `y_*_imp` keys come from `StoreDataset`, not from anything re-derived here."""
    starts = paired_source.windows("chr2")[:2]
    batch = paired_source.batch(Pair("T_aa", "V_aa"), "chr2", starts, "impute")
    assert batch["biosample_name"] == "T_aa" and batch["imp_biosample_name"] == "V_aa"
    assert float(batch["x_avail"][0, IMPUTED]) == 0.0, "T_aa has no H3K4me3 to prompt with"
    assert float(batch["y_data_imp"][:, :, IMPUTED].max()) > 0, "V_aa's H3K4me3 is the truth"
    # The prompt for the imputed slot must carry V_aa's depth, not T_aa's — without it every
    # imputed level is off by the depth ratio between the two cells.
    mixed = H.vb_natural_meta(batch["y_meta"], batch["y_meta_imp"], batch["y_avail"])
    assert float(batch["y_meta"][0, 0, IMPUTED]) == float(MISSING)
    assert float(mixed[0, 0, IMPUTED]) == pytest.approx(
        float(batch["y_meta_imp"][0, 0, IMPUTED]))
    assert float(mixed[0, 0, IMPUTED]) != float(MISSING)


def test_the_declared_pair_is_scored_against_the_target_cell_end_to_end(paired_source, model):
    recs = list(H.stream_tracks(model, paired_source, "cpu", kind="impute", batch_windows=4))
    assert {r.key for r in recs} == {"T_aa|V_aa|H3K4me3", "T_bb|V_bb|H3K4me3"}
    for rec in recs:
        assert rec.n_bins() == N_BINS["chr2"]
        # T_aa carries no H3K4me3 at all, so a non-zero truth can only have come from V_aa.
        assert rec.counts["chr2"].max() > 0


def test_the_dsf_ladder_of_a_declared_pair_is_the_target_cells_ladder(paired_source):
    """`depthcounterfact`'s ground truth is the cell being scored, and that is the TARGET under a declared pair."""
    starts = paired_source.windows("chr2")[:2]
    pair = Pair("T_aa", "V_aa")
    one = paired_source.counts_at_dsf(pair, "chr2", starts, 1)
    batch = paired_source.batch(pair, "chr2", starts, "impute")
    assert np.array_equal(one[:, :, IMPUTED], batch["y_data_imp"][:, :, IMPUTED].numpy())
    two = paired_source.counts_at_dsf(pair, "chr2", starts, 2)
    assert two[:, :, IMPUTED].sum() < one[:, :, IMPUTED].sum()


# ---------------------------------------------------------------------------
# 3 — the keys EVAL_PLAN.md §4 names
# ---------------------------------------------------------------------------

E_KEYS = {"mse", "gwcorr", "gwspear", "mseprom", "msegene", "mseenh", "mse1obs", "mse1imp"}
D_COUNT_KEYS = {"crps", "crps_oracle_scaled", "scale_error", "ece", "coverage_95",
                "c_index", "c_index_se", "c_index_n_pairs", "marg_crps", "beats_marginal"}
D_PVAL_KEYS = {"crps", "pit_ks", "coverage_95", "c_index", "c_index_se"}
B_KEYS = {"auprc", "peak_overlap_0.01", "peak_base_rate"}
P_KEYS = {"acc_by_obs_strength", "acc_by_imp_strength"}


@pytest.fixture(scope="module")
def scored(regime_file, model):
    src = open_source(store=regime_file)
    try:
        return H.run_bench(model, src, "cpu", kinds=("impute",), batch_windows=4,
                           c_index_pairs=2_000, c_windows=3, c_resamples=6, seed=0)
    finally:
        src.close()


def test_the_result_has_the_shape_the_plan_specifies(scored) -> None:
    assert set(scored) >= {"provenance", "tracks", "per_track", "macro", "panel", "ranking", "C"}
    assert scored["ranking"] is None, "the rank aggregation is inert without competitors (D4)"
    assert scored["tracks"] == sorted(scored["per_track"])


def test_the_track_key_is_input_target_assay(scored) -> None:
    for key in scored["tracks"]:
        cell_in, cell_out, assay = key.split("|")
        assert assay in ASSAYS
        assert cell_in in ("T_aa", "V_aa") and cell_out in ("T_aa", "V_aa")


def test_both_arms_carry_their_blocks(scored) -> None:
    arms = next(iter(scored["per_track"].values()))
    assert set(arms) == {"count", "pval"}, "the signal head is on, so the pval arm must be there"
    assert E_KEYS <= set(arms["count"]) and E_KEYS <= set(arms["pval"])
    assert D_COUNT_KEYS <= set(arms["count"])
    assert D_PVAL_KEYS <= set(arms["pval"])
    assert B_KEYS <= set(arms["count"]) and B_KEYS <= set(arms["pval"])
    assert P_KEYS <= set(arms["pval"]), "the P-block is a p-value measure (§4.2)"
    assert not (P_KEYS & set(arms["count"])), "a count arm has no p-value to threshold at 2"


def test_msevar_is_absent_rather_than_zero_when_no_pool_was_given(scored) -> None:
    """The organizers' code returns a bare 0.0 here; a sentinel that looks like a score is worse."""
    for arms in scored["per_track"].values():
        assert "msevar" not in arms["count"]
        assert "msevar" not in arms["pval"]
    assert scored["provenance"]["msevar"]["pool"] is None


def test_the_c_index_never_travels_without_its_standard_error(scored) -> None:
    """D3 — it is the one sampled measure, so the SE is not optional."""
    for arms in scored["per_track"].values():
        for arm in ("count", "pval"):
            assert np.isfinite(arms[arm]["c_index_se"])
            assert arms[arm]["c_index_n_pairs"] > 0


def test_crps_is_never_emitted_without_its_scale_split(scored) -> None:
    for arms in scored["per_track"].values():
        c = arms["count"]
        assert c["scale_error"] == pytest.approx(c["crps"] - c["crps_oracle_scaled"], abs=1e-9)


def test_the_macro_is_an_unweighted_mean_over_tracks(scored) -> None:
    rows = [a["count"] for a in scored["per_track"].values()]
    assert scored["macro"]["count"]["n_tracks"] == len(rows)
    assert scored["macro"]["count"]["mse"] == pytest.approx(
        float(np.mean([r["mse"] for r in rows])))


def test_the_panel_measure_sees_every_cell_type_that_carries_the_assay(scored) -> None:
    """`pr_by_specificity` is panel-level by definition — it needs the whole cell x locus matrix."""
    shared = scored["panel"]["H3K4me3"]
    assert shared["n_cell_types"] == 2
    assert np.isfinite(shared["macro_precision"]) or np.isfinite(shared["macro_recall"])
    # ATAC-seq is in T_aa only, so its panel is one cell and the measure declines to pretend.
    assert scored["panel"]["ATAC-seq"]["n_cell_types"] == 1
    assert "note" in scored["panel"]["ATAC-seq"]


def test_the_c_block_reports_all_seven_instruments_and_what_they_rest_on(scored) -> None:
    # keys renamed from C1..C6 codes to names; `C5_C6_invariance` is now `depthblind_biokeep`.
    c = scored["C"]
    assert {"covuse", "covshare", "depthdir", "covspec"} <= set(c)
    assert "depthcounterfact" in c and "depthblind_biokeep" in c
    assert c["n_units"] > 0 and c["n_windows"] == 3 and c["covuse_n_resamples"] == 6
    # D13 — `depthblind` never leaves without `biokeep`, and the shared key is what enforces it.
    assert "bio_silhouette" in c["depthblind_biokeep"]


def test_the_within_batch_tripwire_reads_exactly_zero(scored) -> None:
    """`covuse`'s cheapest falsifier: substitute the value already there and the loss must not move."""
    for cov, rec in scored["C"]["covuse"].items():
        assert rec["within_batch_d_crps"] == 0.0, cov


# ---------------------------------------------------------------------------
# 4 — the boundaries
# ---------------------------------------------------------------------------

def test_bench_never_imports_the_training_module() -> None:
    """`EVAL_PLAN.md` §11.2. An eval package that imports the training loop inverts the dependency.

    Checked in a SUBPROCESS with a clean interpreter. Doing it in-process would mean tearing
    `candi` out of `sys.modules` and re-importing it, which leaves every later test in this session
    holding classes from a module object that no longer exists — the fixtures compare fine and the
    `isinstance` checks stop matching. A fresh interpreter answers the same question and answers it
    about a real cold import, which is the situation that matters.
    """
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parent.parent / "src")
    code = (
        "import sys, importlib\n"
        "for m in ('candi.bench.harness', 'candi.bench.cli', 'candi.bench.eic'):\n"
        "    importlib.import_module(m)\n"
        # `candi.eval` used to be on this list and is deleted (D15), so naming it would assert
        # nothing. `candi.monitor` takes its place: the direction is bench <- monitor <- train, and
        # a bench that imported the monitor would close the same loop from the other end.
        "leaked = [m for m in ('candi.train', 'candi.monitor') if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    assert got.stdout.strip() == "", f"candi.bench dragged in {got.stdout.strip()}"


@pytest.mark.parametrize("flag", ["--eval-budget", "--max-batches", "--batches-per-pair"])
def test_the_flags_that_shrank_a_run_unevenly_are_refused_by_name(flag) -> None:
    from candi.bench.cli import main

    with pytest.raises(SystemExit) as e:
        main([flag, "8", "--ckpt", "x", "--out", "y"])
    assert "D2" in str(e.value) or "see --max-batches" in str(e.value)


def test_exactly_one_of_h5_and_store(regime_file) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        open_source()
    with pytest.raises(ValueError, match="exactly one"):
        open_source(h5="a.h5", store=regime_file)


# ---------------------------------------------------------------------------
# the CLI, end to end
# ---------------------------------------------------------------------------

def test_the_cli_scores_a_real_checkpoint_end_to_end(regime_file, ckpt, tmp_path) -> None:
    from candi.bench.cli import main

    out = tmp_path / "bench.json"
    rc = main(["--store", str(regime_file), "--ckpt", str(ckpt), "--out", str(out),
               "--heads", "count,signal,peak", "--depth-center", "24.25",
               "--blocks", "E,P,D,B", "--c-index-pairs", "500", "--quiet"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["provenance"]["data_source"] == "store"
    assert res["provenance"]["ckpt"] == str(ckpt)
    assert res["per_track"], "the CLI wrote no track"
    assert res["macro"]["count"]["n_tracks"] == len(res["per_track"])
    # JSON has no NaN; `jsonable` spells a non-finite float `null` rather than emitting invalid JSON.
    assert "NaN" not in out.read_text()


# ---------------------------------------------------------------------------
# the h5 backend
# ---------------------------------------------------------------------------
#
# A real v2 bake, small enough to read in a glance and complete enough that `CandiKitH5Dataset`
# opens it without complaint. `T_a` holds assays 0 and 1, `V_a` holds 1 and 2 — so assay 2 is a
# genuine imputation target (absent from the input cell, present in its paired cell) and assay 1 is
# a denoising target, which is exactly the pairing the EIC panel has and the store cannot express.

H5_A, H5_L, H5_RES = 3, 64, 25
H5_TRAIN_W, H5_EVAL_W = 4, 6
H5_HAS = {"T_a": [0, 1], "V_a": [1, 2]}
H5_DEPTH = {"T_a": 24.0, "V_a": 25.0}


def _bake(path: Path, *, eval_gap: bool = False) -> Path:
    """A minimal but complete v2 h5. `eval_gap` leaves a hole in the eval tiling, for D2's guard."""
    import h5py

    rng = np.random.default_rng(0)
    W = H5_TRAIN_W + H5_EVAL_W
    span = H5_L * H5_RES
    chroms = [b"chr19"] * H5_TRAIN_W + [b"chr21"] * H5_EVAL_W
    starts = [i * span for i in range(H5_TRAIN_W)]
    # The eval chromosome is tiled from 0; `eval_gap` skips the second tile and leaves a hole.
    ev = [i for i in range(H5_EVAL_W)] if not eval_gap else [0] + list(range(2, H5_EVAL_W + 1))
    starts += [i * span for i in ev]

    with h5py.File(str(path), "w") as h:
        h.attrs["version"] = 2
        h.attrs["assays"] = json.dumps([f"assay{i}" for i in range(H5_A)])
        h.attrs["context_bins"] = H5_L
        h.attrs["resolution"] = H5_RES
        h.attrs["dsf_list"] = json.dumps([1, 2])
        h.attrs["train_chroms"] = json.dumps(["chr19"])
        h.attrs["eval_chroms"] = json.dumps(["chr21"])
        h.create_dataset("windows/chrom", data=np.array(chroms))
        h.create_dataset("windows/start", data=np.array(starts, dtype=np.int64))
        h.create_dataset("windows/end", data=np.array(starts, dtype=np.int64) + span)
        h.create_dataset("windows/region_type", data=np.zeros(W, dtype=np.int32))
        g0 = h.create_group("biosamples")
        g0.attrs["order"] = json.dumps(sorted(H5_HAS))
        for b, cols in H5_HAS.items():
            g = g0.create_group(b)
            counts = np.zeros((W, H5_L, H5_A), dtype=np.int32)
            meta = np.full((4, H5_A), -1.0, dtype=np.float32)
            pval = np.zeros((W, H5_L, H5_A), dtype=np.float32)
            peaks = np.zeros((W, H5_L, H5_A), dtype=np.float32)
            for a in cols:
                counts[:, :, a] = rng.integers(1, 60, size=(W, H5_L))
                pval[:, :, a] = rng.random((W, H5_L)) * 8.0
                peaks[:, :, a] = (rng.random((W, H5_L)) < 0.1).astype(np.float32)
                meta[:, a] = (H5_DEPTH[b], float(a), 76.0, 1.0)
            g.create_dataset("counts_dsf1", data=counts)
            g.create_dataset("counts_dsf2", data=(counts // 2).astype(np.int32))
            g.create_dataset("meta_dsf1", data=meta)
            m2 = meta.copy()
            m2[0, cols] -= 1.0                      # depth falls by log2(2) with the thinning
            g.create_dataset("meta_dsf2", data=m2)
            g.create_dataset("pval", data=pval)
            g.create_dataset("peaks", data=peaks)
            g.create_dataset("dna", data=rng.random((W, H5_L * H5_RES, 4)).astype(np.float32))
            g.create_dataset("control", data=rng.integers(0, 5, (W, H5_L, 1)).astype(np.float32))
            g.create_dataset("control_meta",
                             data=np.tile(np.array([[20.0], [0.0], [76.0], [1.0]], np.float32),
                                          (W, 1, 1)))
    return path


@pytest.fixture(scope="module")
def baked(tmp_path_factory) -> Path:
    return _bake(tmp_path_factory.mktemp("bake") / "toy.h5")


@pytest.fixture()
def h5_source(baked):
    s = open_source(h5=baked)
    yield s
    s.close()


def test_the_h5_backend_pairs_each_input_cell_with_its_imputation_counterpart(h5_source) -> None:
    assert h5_source.pairs("impute") == [Pair("T_a", "V_a")]
    assert h5_source.pairs("denoise") == [Pair("T_a", "T_a")]
    # assay2 is in V_a and not in T_a: the prompt is the only channel carrying it.
    assert h5_source.targets(Pair("T_a", "V_a"), "impute") == [2]
    assert h5_source.targets(Pair("T_a", "T_a"), "denoise") == [0, 1]


def test_a_gapped_eval_tiling_is_refused_with_the_gap_measured(tmp_path) -> None:
    """D2 — a hole would be scored against whatever the buffer was initialised to."""
    gapped = _bake(tmp_path / "gapped.h5", eval_gap=True)
    with pytest.raises(ValueError, match="in no window"):
        open_source(h5=gapped)


def test_the_h5_bakes_own_dsf_ladder_is_used_rather_than_a_re_thinning(h5_source) -> None:
    """`depthcounterfact`'s ground truth is `counts_dsf{k}` as baked. Re-thinning it here would be a second draw."""
    pair = Pair("T_a", "V_a")
    starts = h5_source.windows("chr21")[:2]
    one = h5_source.counts_at_dsf(pair, "chr21", starts, 1)
    two = h5_source.counts_at_dsf(pair, "chr21", starts, 2)
    assert np.array_equal(two[:, :, 1], one[:, :, 1] // 2)
    with pytest.raises(ValueError, match="DSF levels"):
        h5_source.counts_at_dsf(pair, "chr21", starts, 8)


def test_the_imputation_prompt_carries_the_target_cells_own_covariates(h5_source) -> None:
    """Without this the depth-offset head is told the INPUT cell's depth for a track it lacks."""
    pair = Pair("T_a", "V_a")
    batch = h5_source.batch(pair, "chr21", h5_source.windows("chr21")[:2], "impute")
    mixed = H.vb_natural_meta(batch["y_meta"], batch["y_meta_imp"], batch["y_avail"])
    assert float(batch["y_meta"][0, 0, 2]) == -1.0, "assay2 is absent from T_a"
    assert float(mixed[0, 0, 2]) == pytest.approx(H5_DEPTH["V_a"]), "V_a's depth must be spliced in"
    # A slot the input cell DOES have is untouched — the splice is for missing slots only.
    assert float(mixed[0, 0, 1]) == pytest.approx(H5_DEPTH["T_a"])


@pytest.fixture(scope="module")
def h5_model():
    torch.manual_seed(0)
    m = build_model(num_assays=H5_A, context_length=H5_L, resolution=H5_RES,
                    depth_center=24.0, heads=("count", "signal"))
    m.eval()
    return m


def test_the_h5_path_scores_the_imputation_target_over_the_whole_eval_chromosome(
        baked, h5_model) -> None:
    src = open_source(h5=baked)
    try:
        recs = list(H.stream_tracks(h5_model, src, "cpu", kind="impute", batch_windows=2))
        assert len(recs) == 1
        rec = recs[0]
        assert rec.assay == "assay2" and rec.key == "T_a|V_a|assay2"
        assert rec.n_bins() == H5_EVAL_W * H5_L
        # The ground truth is V_a's track, not T_a's — T_a has no assay2 at all.
        assert rec.counts["chr21"].max() > 0
        assert rec.has_pval
    finally:
        src.close()


def test_denoise_tracks_carry_a_fourth_key_field_so_they_cannot_collide(baked, h5_model) -> None:
    src = open_source(h5=baked)
    try:
        recs = list(H.stream_tracks(h5_model, src, "cpu", kind="denoise", batch_windows=3))
        assert {r.key for r in recs} == {"T_a|T_a|assay0|denoise", "T_a|T_a|assay1|denoise"}
    finally:
        src.close()


def test_the_h5_harness_batch_equals_the_loaders_batch(baked) -> None:
    """The h5 backend re-implements batch assembly; this is what stops it drifting.

    The store backend calls `StoreDataset._make_batch` and cannot drift by construction. The h5
    backend cannot do the same — `CandiKitH5Dataset.__iter__` advances one `(T_, V_/B_)` pair per
    window batch and owns the window order, which is precisely what D2 needs taken away from it —
    so assembly is written twice and pinned here instead.

    The toy bake has exactly one pair, so the loader's own first batch is windows 0..2 of the eval
    chromosome for `(T_a, V_a)`: the same rows the harness is asked for.
    """
    from candi.dataset import CandiKitH5Dataset

    ds = CandiKitH5Dataset(baked, "type1", train=False, batch_size=3, biosample_prefix="T_",
                           dsf_sampling="off", seed=0, shuffle=False,
                           eval_include_vb_ground_truth=True, h5_cache_ram=False)
    theirs = next(iter(ds))
    assert theirs["biosample_name"] == "T_a" and theirs["imp_biosample_name"] == "V_a"

    src = open_source(h5=baked)
    try:
        starts = src.windows("chr21")[:3]
        ours = src.batch(Pair("T_a", "V_a"), "chr21", starts, "impute")
        for k, v in theirs.items():
            if isinstance(v, torch.Tensor) and k not in ("x_dsf", "y_dsf", "region_type",
                                                         "window_idx", "control_x_dsf"):
                assert torch.equal(ours[k], v), f"{k} differs between harness and loader"
    finally:
        src.close()


def test_the_cli_runs_the_h5_path_and_both_kinds_at_once(baked, tmp_path) -> None:
    """Imputation and denoising in one run, with keys that cannot collide, and separate macros."""
    from candi.bench.cli import main

    torch.manual_seed(0)
    m = build_model(num_assays=H5_A, context_length=H5_L, resolution=H5_RES,
                    depth_center=24.0, heads=("count", "signal"))
    ck = tmp_path / "m.pt"
    torch.save(m.state_dict(), ck)
    out = tmp_path / "h5bench.json"
    rc = main(["--h5", str(baked), "--ckpt", str(ck), "--out", str(out),
               "--heads", "count,signal", "--depth-center", "24.0",
               "--kinds", "impute,denoise", "--blocks", "E,D,B",
               "--c-index-pairs", "500", "--quiet"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["provenance"]["data_source"] == "h5"
    assert set(res["tracks"]) == {"T_a|V_a|assay2", "T_a|T_a|assay0|denoise",
                                  "T_a|T_a|assay1|denoise"}
    assert res["macro"]["count"]["n_tracks"] == 1, "the headline macro is over IMPUTED tracks"
    assert res["macro_denoise"]["count"]["n_tracks"] == 2
    assert "C" not in res, "--blocks did not name C"


def test_gwcorr_of_a_track_against_itself_is_one_through_the_whole_stack(baked, h5_model) -> None:
    """An end-to-end sanity anchor: score a record whose prediction IS its target.

    Every layer between the h5 and the number — assembly, placement, concatenation, the E-block —
    has to be right for this to come out at 1.0, and a placement bug that shuffled bins would not.
    """
    from candi.bench.harness import score_track

    src = open_source(h5=baked)
    try:
        rec = next(H.stream_tracks(h5_model, src, "cpu", kind="impute", batch_windows=3))
        rec.mu = {c: rec.counts[c].astype(np.float32) for c in rec.chroms}
        rec.signal_mu = {c: rec.pval[c].astype(np.float32) for c in rec.chroms}
        # The REAL pinned beds, not an empty list: an annotation set matching no line at all is
        # `score_metrics.py` quirk 8, which raises rather than returning nan, and the point here is
        # the identity of the genome-wide measures — not to re-test the quirk.
        scored = score_track(rec, gene_annotations=ann.gene_annotations(),
                             enh_annotations=ann.enhancer_annotations(), c_index_pairs=1_000)
    finally:
        src.close()
    for arm in ("count", "pval"):
        assert scored[arm]["gwcorr"] == pytest.approx(1.0)
        assert scored[arm]["gwspear"] == pytest.approx(1.0)
        assert scored[arm]["mse"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# t80 — the declared pairing: the T_ cell prompts, the V_ cell holds the truth
# ---------------------------------------------------------------------------
#
# `StoreSource.pairs` used to be a hardcoded `[Pair(b, b) for b in biosample_pool]`, so CANDI's
# prompt on the store path was the eval cell's OTHER blind assays — never the paired training
# cell's tracks. Two things follow, and this layout reproduces both in miniature.
#
# `V_pp` holds exactly ONE assay, the shape 16 of the 26 EIC `V_` cells have. Self-paired, holding
# it out empties the encoder input, `prepare_masked_batch` returns None and the track is dropped
# unscored — the collapse that left the `V_` panel able to pose 29 of its 45 experiments. Paired
# with `T_pp` it is an ordinary experiment. `V_qq` holds two, so the panel is three, not two, and
# the self-paired panel is two, not three.
#
# Both pairs are DISJOINT on (cell, assay), which is what `BENCHMARK_DESIGN.md` §5.1 measured over
# all 89 EIC cells and what makes the panel rule and "every assay the truth cell holds" agree.

PAIR_TRACKS = {
    "T_pp": ("ATAC-seq", "DNase-seq", SL.CONTROL_TRACK),
    "V_pp": ("H3K4me3",),
    "T_qq": ("ATAC-seq", SL.CONTROL_TRACK),
    "V_qq": ("DNase-seq", "H3K4me3"),
}
PAIRS = [["T_pp", "V_pp"], ["T_qq", "V_qq"]]
#: `{f"{input}->{target}": [assay, …]}` — the panel each declared pair poses, by hand off the
#: layout above, so the assertions do not re-derive the rule they are checking.
PAIR_PANEL = {"T_pp->V_pp": ["H3K4me3"], "T_qq->V_qq": ["DNase-seq", "H3K4me3"]}


@pytest.fixture(scope="module")
def pair_store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("t80store"), tracks=PAIR_TRACKS)


def _pair_regime(dirpath: Path, store: Path, **over) -> Path:
    obj = regime_dict(store, biosamples={"train": ["T_pp", "T_qq"], "eval": ["V_pp", "V_qq"]},
                      kinds=["counts", "peaks", "pval"], **over)
    p = dirpath / "regime.json"
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def paired_regime(pair_store, tmp_path_factory) -> Path:
    return _pair_regime(tmp_path_factory.mktemp("t80paired"), pair_store, eval_pairs=PAIRS)


@pytest.fixture(scope="module")
def bare_regime(pair_store, tmp_path_factory) -> Path:
    """The same store with no `eval_pairs` — the pre-t80 self-paired exam."""
    return _pair_regime(tmp_path_factory.mktemp("t80bare"), pair_store)


@pytest.fixture()
def paired(paired_regime):
    s = open_source(store=paired_regime)
    yield s
    s.close()


@pytest.fixture()
def bare(bare_regime):
    s = open_source(store=bare_regime)
    yield s
    s.close()


def _panel(src) -> dict:
    return {str(p): [src.assays[a] for a in src.targets(p, "impute")]
            for p in src.pairs("impute")}


def test_the_store_prompts_with_the_paired_training_cell_not_the_eval_cell(paired) -> None:
    """The headline. `pairs` is the regime's declared list, and both halves differ."""
    assert [(p.input_biosample, p.target_biosample) for p in paired.pairs("impute")] == [
        ("T_pp", "V_pp"), ("T_qq", "V_qq")]
    assert all(H.cross_cell(p, "impute") for p in paired.pairs("impute"))
    # …and the loader is prompted with the T_ cells, so `depth_center` is the training population's.
    assert paired.ds.biosample_pool == ["T_pp", "T_qq"]


def test_the_scored_panel_belongs_to_the_truth_cell_not_the_prompt(paired) -> None:
    """`assays the truth cell has and the input cell does not` — §5.1, §8, `H5Source.targets`."""
    assert _panel(paired) == PAIR_PANEL


def test_denoising_stays_self_paired_and_keeps_the_prompt_cells_own_panel(paired) -> None:
    assert [(p.input_biosample, p.target_biosample) for p in paired.pairs("denoise")] == [
        ("T_pp", "T_pp"), ("T_qq", "T_qq")]
    got = {p.input_biosample: [paired.assays[a] for a in paired.targets(p, "denoise")]
           for p in paired.pairs("denoise")}
    assert got == {"T_pp": ["ATAC-seq", "DNase-seq"], "T_qq": ["ATAC-seq"]}


def test_a_regime_that_declares_no_pairs_keeps_the_old_self_paired_behaviour(bare) -> None:
    """D31 — absent means "no imputation declared", never "infer it from the names" (D16)."""
    assert [(p.input_biosample, p.target_biosample) for p in bare.pairs("impute")] == [
        ("V_pp", "V_pp"), ("V_qq", "V_qq")]
    assert _panel(bare) == {"V_pp->V_pp": ["H3K4me3"],
                            "V_qq->V_qq": ["DNase-seq", "H3K4me3"]}
    assert bare.provenance()["self_paired"] is True and bare.provenance()["eval_pairs"] == []


def test_pairing_restores_the_experiments_leave_one_out_could_not_pose(paired, bare, model) -> None:
    """The 45-vs-29 collapse, in miniature: `V_pp` holds one assay and self-pairing drops it.

    The paired source scores every experiment in the panel; the self-paired one scores the panel
    minus every single-assay truth cell, because holding out its only assay empties the encoder
    input and `prepare_masked_batch` returns `None`.
    """
    def scored(src):
        return sorted(f"{r.pair}|{r.assay}"
                      for r in H.stream_tracks(model, src, "cpu", batch_windows=4))

    assert scored(paired) == ["T_pp->V_pp|H3K4me3", "T_qq->V_qq|DNase-seq",
                              "T_qq->V_qq|H3K4me3"]
    assert scored(bare) == ["V_qq->V_qq|DNase-seq", "V_qq->V_qq|H3K4me3"]
    assert sum(len(v) for v in PAIR_PANEL.values()) == 3


def test_the_prompt_never_carries_the_target_assay(paired) -> None:
    """No leak, checked two ways, for every declared (pair, assay).

    First the corpus: the `T_` cell must not hold the assay at all — that is the disjointness
    `BENCHMARK_DESIGN.md` §5.1 claims and `_overlaps` measures. Then the tensor the model actually
    sees: after `_apply_loo_mask` the target column is `CLOZE` in data, metadata and availability,
    so `prepare_masked_batch` classes it masked rather than observed however the corpus is shaped.
    """
    from candi._vendored import CLOZE

    assert paired.pair_overlaps == {}
    chrom = paired.eval_chroms[0]
    starts = paired.windows(chrom)[:2]
    for pair in paired.pairs("impute"):
        prompt_has = paired.ds._availability[pair.input_biosample]
        for a in paired.targets(pair, "impute"):
            assert not bool(prompt_has[a]), f"{pair} prompt holds its own target {paired.assays[a]}"
            batch = paired.batch(pair, chrom, starts, "impute")
            assert H.decode_groups(paired, "impute",
                                   paired.targets(pair, "impute")) == [
                [c] for c in paired.targets(pair, "impute")]
            H._apply_loo_mask(batch, a)
            assert float(batch["x_avail"][0, a]) == float(CLOZE)
            assert bool((batch["x_data"][:, :, a] == float(CLOZE)).all())
            assert bool((batch["x_meta"][:, :4, a] == float(CLOZE)).all())


def test_the_truth_is_the_target_cells_track_read_off_the_store(paired) -> None:
    """`y_data_imp`, not `y_data`. The prompt cell has no such column to give."""
    from candi._vendored import MISSING

    chrom = paired.eval_chroms[0]
    starts = paired.windows(chrom)[:2]
    pair = paired.pairs("impute")[0]
    a = paired.targets(pair, "impute")[0]
    batch = paired.batch(pair, chrom, starts, "impute")
    assert batch["imp_biosample_name"] == pair.target_biosample
    assert bool((batch["y_data"][:, :, a] == float(MISSING)).all())
    want = paired.ds.corpus[pair.target_biosample].counts(
        chrom, starts[0], starts[0] + paired.context_bins, assays=[paired.assays[a]])[:, 0]
    assert np.array_equal(batch["y_data_imp"][0, :, a].numpy(), want.astype(np.float32))
    # …and the DSF-1 rung of the C-block ladder reads the same cell, not the prompt's MISSING.
    ladder = paired.counts_at_dsf(pair, chrom, starts, 1)
    assert np.array_equal(ladder[:, :, a], batch["y_data_imp"][:, :, a].numpy())


def test_the_decoder_is_prompted_with_the_target_cells_own_covariates(paired, model) -> None:
    """Without this the depth head is told the `T_` cell's depth for a `V_` cell's track."""
    from candi._vendored import MISSING
    from candi.batch import make_masker, prepare_masked_batch

    chrom = paired.eval_chroms[0]
    starts = paired.windows(chrom)[:1]
    pair = paired.pairs("impute")[0]
    a = paired.targets(pair, "impute")[0]
    batch = paired.batch(pair, chrom, starts, "impute")
    H._apply_loo_mask(batch, a)
    noop = make_masker(p_full_loci=0.0, p_full_assay=0.0, p_chunks=0.0, mask_fraction=0.0)
    prep = prepare_masked_batch(batch, noop, "cpu", apply_mask=False)
    assert prep is not None, "the T_ prompt still carries its own tracks"
    assert float(prep["y_meta"][0, H.DEPTH_ROW, a]) == float(MISSING)
    mixed = H.vb_natural_meta(prep["y_meta"], batch["y_meta_imp"], batch["y_avail"])
    want = float(paired.ds._meta[pair.target_biosample][H.DEPTH_ROW, a])
    assert float(mixed[0, H.DEPTH_ROW, a]) == pytest.approx(want)


def test_biosamples_selects_a_declared_pair_from_either_side(paired_regime) -> None:
    """`--biosamples V_pp` means "score V_pp", and must not drop the panel back to self-pairing."""
    for named in ("V_pp", "T_pp"):
        src = open_source(store=paired_regime, biosamples=[named])
        try:
            assert [(p.input_biosample, p.target_biosample) for p in src.pairs("impute")] == [
                ("T_pp", "V_pp")]
        finally:
            src.close()
    with pytest.raises(ValueError, match="none of them"):
        open_source(store=paired_regime, biosamples=["T_qq_nope"])


def test_the_provenance_records_which_exam_was_sat(paired) -> None:
    prov = paired.provenance()
    assert prov["eval_pairs"] == PAIRS
    assert prov["self_paired"] is False
    assert prov["eval_pair_assay_overlaps"] == {}


def test_a_pair_whose_cells_share_an_assay_is_recorded_and_the_assay_is_not_scored(
        tmp_path_factory) -> None:
    """Disjointness is measured, not assumed. A shared assay drops out of the panel and is named."""
    store = make_store(tmp_path_factory.mktemp("t80overlap"), tracks={
        "T_rr": ("ATAC-seq", "DNase-seq", SL.CONTROL_TRACK),
        "V_rr": ("DNase-seq", "H3K4me3"),
    })
    d = tmp_path_factory.mktemp("t80overlapregime")
    p = d / "regime.json"
    p.write_text(json.dumps(regime_dict(
        store, biosamples={"train": ["T_rr"], "eval": ["V_rr"]},
        eval_pairs=[["T_rr", "V_rr"]])), encoding="utf-8")
    src = open_source(store=p)
    try:
        assert src.pair_overlaps == {"T_rr->V_rr": ["DNase-seq"]}
        assert _panel(src) == {"T_rr->V_rr": ["H3K4me3"]}
    finally:
        src.close()


# the loss tier — val/test loss, and the head gate that keeps it honest
# ---------------------------------------------------------------------------
#
# These are the three NLLs the training objective is built from, computed on the eval panel instead
# of on a training batch. What they must be right about here is not the arithmetic — that is pinned
# against `candi.train` in `tests/test_bench_analytic.py` — but the PLUMBING: which heads produce
# which keys, that an absent head means an absent key rather than a nan, and that the target space
# the Gaussian term was scored in is recorded next to the number.

NLL_KEYS = {"nb_nll", "gaussian_nll", "bernoulli_nll"}


def test_the_loss_tier_lands_in_every_arm_because_a_loss_has_no_arm(scored) -> None:
    """The `count`/`pval` split is between two COMPARISONS; the objective is one scalar."""
    for arms in scored["per_track"].values():
        for arm in ("count", "pval"):
            assert NLL_KEYS <= set(arms[arm]), f"{arm} is missing a loss term"
            assert all(np.isfinite(arms[arm][k]) for k in NLL_KEYS)
    for arm in ("count", "pval"):
        assert NLL_KEYS <= set(scored["macro"][arm]), "the loss did not reach the macro roll-up"


def test_the_gaussian_target_space_is_recorded_beside_the_number(scored) -> None:
    """D30 — the same key in two spaces is two different quantities, both finite and plausible."""
    assert scored["provenance"]["signal_target_transform"] == "none"
    for arms in scored["per_track"].values():
        for arm in ("count", "pval"):
            assert arms[arm]["signal_target_transform"] == "none"
    # A string cannot become a macro: `macro_mean` averages scalars only, and averaging a label
    # would be nonsense even if it were possible.
    assert "signal_target_transform" not in scored["macro"]["count"]


@pytest.fixture(scope="module")
def count_only_model():
    """No signal head, no peak head — the checkpoint shape most of this repo's runs have."""
    torch.manual_seed(0)
    m = build_model(num_assays=len(ASSAYS), context_length=CTX, resolution=25,
                    depth_center=24.25, heads=("count",))
    m.eval()
    return m


def _one_track(model, regime_file, **kw):
    src = open_source(store=regime_file)
    try:
        rec = next(H.stream_tracks(model, src, "cpu", kind="impute", batch_windows=4))
        return rec, H.score_track(rec, gene_annotations=ann.gene_annotations(),
                                  enh_annotations=ann.enhancer_annotations(),
                                  c_index_pairs=500, **kw)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# t80 — tools/declare_eval_pairs.py, the thing §14 says owns the pairing
# ---------------------------------------------------------------------------

def test_the_declaring_tool_derives_the_pairing_and_records_its_provenance(
        pair_store, paired_regime, tmp_path) -> None:
    from tools.declare_eval_pairs import main as declare_main

    out = tmp_path / "pairs.json"
    rc = declare_main(["declare", "--store", str(pair_store), "--input-prefix", "T_",
                       "--target-prefix", "V_", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rc == 0
    assert rec["eval_pairs"] == PAIRS
    assert rec["disjoint"] is True and rec["overlaps"] == {}
    assert rec["summary"] == {"pairs": 2, "experiments": 3,
                              "by_target_prefix": {"V_": {"cells": 2, "experiments": 3}}}
    # the pre-t80 exam, recorded beside the fix so the gap stays legible
    assert rec["self_paired_would_score"] == 2
    assert rec["rule"] == {"kind": "prefix", "input_prefix": "T_", "target_prefixes": ["V_"]}
    assert rec["store"] == str(pair_store) and len(rec["manifest_sha256"]) == 64


def test_the_tool_and_the_harness_read_the_same_panel(pair_store, paired, tmp_path) -> None:
    """Two independent readings of one claim. The tool must never be the harness's echo."""
    from tools.declare_eval_pairs import main as declare_main

    out = tmp_path / "pairs.json"
    declare_main(["declare", "--store", str(pair_store), "--input-prefix", "T_",
                  "--target-prefix", "V_", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert {k: v["panel"] for k, v in rec["panels"].items()} == _panel(paired) == PAIR_PANEL


def test_the_tool_refuses_to_guess_a_pairing(pair_store) -> None:
    """D31 — a default naming rule would make the pairing inferred rather than declared."""
    from tools.declare_eval_pairs import main as declare_main

    with pytest.raises(SystemExit, match="no pairing rule"):
        declare_main(["declare", "--store", str(pair_store)])


def test_the_tool_takes_a_csv_so_it_need_not_parse_a_name_at_all(pair_store, tmp_path) -> None:
    """D16-clean input: the operator states the pairing and no string is dissected."""
    from tools.declare_eval_pairs import main as declare_main

    csv = tmp_path / "pairs.csv"
    csv.write_text("input,target\nT_pp,V_pp\nT_qq,V_qq\n", encoding="utf-8")
    out = tmp_path / "pairs.json"
    assert declare_main(["declare", "--store", str(pair_store), "--pairs-from", str(csv),
                         "--out", str(out)]) == 0
    assert json.loads(out.read_text())["eval_pairs"] == PAIRS


def test_the_tool_writes_a_regime_that_loads_and_the_harness_then_pairs(
        bare_regime, pair_store, tmp_path) -> None:
    from tools.declare_eval_pairs import main as declare_main

    paired_out = tmp_path / "regime.paired.json"
    assert declare_main(["declare", "--regime", str(bare_regime), "--input-prefix", "T_",
                         "--target-prefix", "V_", "--regime-out", str(paired_out)]) == 0
    assert Regime.from_file(paired_out).eval_pairs == (("T_pp", "V_pp"), ("T_qq", "V_qq"))
    src = open_source(store=paired_out)
    try:
        assert _panel(src) == PAIR_PANEL
    finally:
        src.close()


def test_check_fails_a_regime_that_declares_nothing_and_passes_one_that_does(
        bare_regime, paired_regime) -> None:
    from tools.declare_eval_pairs import main as declare_main

    assert declare_main(["check", "--regime", str(bare_regime)]) == 1
    assert declare_main(["check", "--regime", str(paired_regime)]) == 0


def test_a_paired_run_scores_end_to_end_and_the_c_block_ladder_reads_the_truth_cell(
        paired_regime, model) -> None:
    """`run_bench` over a declared pairing, C-block included.

    The C-block is the other half of the fix: `_c_contexts` has to take its truth from `y_data_imp`
    like `stream_tracks` does, and `counts_at_dsf` has to thin the TARGET cell's counts — the
    prompt cell's column for a held-out assay is `MISSING`, so a depth ladder read off it would be
    a ladder of `-1`s and C3 would score noise against noise.
    """
    src = open_source(store=paired_regime)
    try:
        res = H.run_bench(model, src, "cpu", batch_windows=4, c_windows=2, c_resamples=2,
                          c_index_pairs=500)
        for d in src.dsf_levels:
            ladder = src.counts_at_dsf(src.pairs("impute")[1], src.eval_chroms[0],
                                       src.windows(src.eval_chroms[0])[:2], int(d))
            for a in src.targets(src.pairs("impute")[1], "impute"):
                assert (ladder[:, :, a] >= 0).all(), f"dsf {d} read the prompt cell's MISSING"
    finally:
        src.close()
    assert res["tracks"] == ["T_pp|V_pp|H3K4me3", "T_qq|V_qq|DNase-seq", "T_qq|V_qq|H3K4me3"]
    assert res["provenance"]["eval_pairs"] == PAIRS
    assert res["macro"]["count"]["n_tracks"] == 3
    assert res["C"]["n_units"] > 0 and "error" not in res["C"]
    assert "error" not in res["C"]["C3_depth_counterfactual"]


# ---------------------------------------------------------------------------
# t80 — the two scopes and the three panel numbers (plan/BENCHMARK_DESIGN.md §4, §5.2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_chrom_regime(tmp_path_factory) -> Path:
    """TWO eval chromosomes, so a held-out scope can be a proper SUBSET of what is scored.

    The fixtures above cannot express the split at all. They have two chromosomes total, and a
    regime refuses to train and evaluate on the same one, so exactly one is left to evaluate on —
    which makes the held-out scope and the genome-wide scope the same set. That is §4's blanking
    case, tested separately; a third chromosome is what makes the split itself testable.
    """
    sizes = {"chr1": 2_000 * 25 + 7, "chr2": 800 * 25 + 3, "chr3": 600 * 25 + 11}
    store3 = make_store(tmp_path_factory.mktemp("benchstore3"), chrom_sizes=sizes)
    d = tmp_path_factory.mktemp("benchregime2")
    obj = regime_dict(store3, biosamples={"train": ["T_aa"], "eval": ["T_aa", "V_aa"]},
                      kinds=["counts", "peaks", "pval"],
                      train_chroms=["chr1"], eval_chroms=["chr2", "chr3"])
    p = d / "regime.json"
    p.write_text(json.dumps(obj))
    return p


def test_panel_of_reads_the_target_cell_and_nothing_else() -> None:
    assert H.panel_of("V_K562") == "V"
    assert H.panel_of("B_HAP-1") == "B"
    # A `T_` cell is the PROMPT. Pairing on the input would put every experiment in one panel.
    assert H.panel_of("T_K562") is None


def _fake_track(assay: str, mse: float) -> dict:
    return {"pval": {"assay": assay, "kind": "impute", "mse": mse, "n_bins": 100}}


def test_the_matched_panel_is_measured_from_b_not_listed() -> None:
    """§5.2's middle number: the same `V_` tracks, over the assays `B_` actually contains.

    The set is read off `B_` at aggregation time on purpose. A literal list here would go stale the
    first time the blind panel moves, and would go stale silently.
    """
    per_track = {
        "T_a|V_a|H3K4me3": _fake_track("H3K4me3", 1.0),
        "T_b|V_b|H3K27ac": _fake_track("H3K27ac", 3.0),
        "T_c|V_c|H3K9me3": _fake_track("H3K9me3", 5.0),      # V_ only — not in B_
        "T_d|B_d|H3K4me3": _fake_track("H3K4me3", 2.0),
        "T_e|B_e|H3K27ac": _fake_track("H3K27ac", 4.0),
    }
    got = H.panel_macros(per_track, "pval")

    assert got["V_breadth"]["n_experiments"] == 3
    assert got["V_breadth"]["assays"] == ["H3K27ac", "H3K4me3", "H3K9me3"]
    assert got["V_breadth"]["mse"] == pytest.approx(3.0)     # (1+3+5)/3

    assert got["V_matched"]["matched_to"] == ["H3K27ac", "H3K4me3"]
    assert got["V_matched"]["n_experiments"] == 2
    assert got["V_matched"]["mse"] == pytest.approx(2.0)     # (1+3)/2, H3K9me3 dropped

    assert got["B"]["n_experiments"] == 2
    assert got["B"]["mse"] == pytest.approx(3.0)             # (2+4)/2


def test_only_the_matched_panel_is_unranked() -> None:
    """It is a reading aid. Ranking it would invent a placement with no counterpart on the board."""
    per_track = {"T_a|V_a|H3K4me3": _fake_track("H3K4me3", 1.0),
                 "T_d|B_d|H3K4me3": _fake_track("H3K4me3", 2.0)}
    got = H.panel_macros(per_track, "pval")
    assert got["V_breadth"]["ranked"] is True
    assert got["B"]["ranked"] is True
    assert got["V_matched"]["ranked"] is False


def test_a_scope_is_a_subset_of_what_was_predicted(source, model) -> None:
    """`score_track` refuses a chromosome the record does not carry, rather than skipping it.

    Skipping would be the dangerous behaviour: a scope quietly narrowed to whatever happened to be
    present is a number whose address nobody can state, which is the whole failure this design is
    against.
    """
    rec = next(iter(H.stream_tracks(model, source, "cpu", kind="impute", batch_windows=4)))
    with pytest.raises(ValueError, match="which the record does not carry"):
        H.score_track(rec, gene_annotations=ann.gene_annotations(),
                      enh_annotations=ann.enhancer_annotations(), chroms=("chrNope",))


def test_the_two_scopes_are_two_aggregations_of_one_pass(two_chrom_regime, model) -> None:
    """§4. One prediction pass, measures recomputed over two chromosome sets — never rescaled.

    The check that matters is that the held-out numbers and the genome-wide numbers actually
    DIFFER. If they did not, the split would be decorative and a reader could not tell which one
    the board is ranking.
    """
    src = open_source(store=two_chrom_regime)
    try:
        res = H.run_bench(model, src, "cpu", kinds=("impute",), batch_windows=4,
                          c_index_pairs=2_000, c_windows=3, c_resamples=6, seed=0,
                          blocks=("E", "D", "B"), held_out_chroms=["chr3"])
    finally:
        src.close()

    scope = res["provenance"]["scope"]
    assert scope["held_out_chroms"] == ["chr3"]
    assert scope["scored_chroms"] == ["chr2", "chr3"]
    assert scope["genome_wide_computed"] is True
    assert "genome_wide" in res

    # the ranked scope is chr2 only; the comparability scope is both
    for key in res["tracks"]:
        assert res["per_track"][key]["count"]["chroms"] == ["chr3"]
        assert res["genome_wide"]["per_track"][key]["count"]["chroms"] == ["chr2", "chr3"]
        assert (res["genome_wide"]["per_track"][key]["count"]["n_bins"]
                > res["per_track"][key]["count"]["n_bins"])

    held = res["macro"]["count"]["mse"]
    gw = res["genome_wide"]["macro"]["count"]["mse"]
    assert held != gw, "a scope split that does not move a number is not a split"
    assert set(res["panels"]) == {"count", "pval"}
    assert set(res["panels"]["count"]) == set(H.PANELS)


def test_one_scope_means_the_genome_wide_block_does_not_exist(scored) -> None:
    """§4's blanking rule: not computed, rather than computed and withheld.

    A method whose transferable parameters were fit at every position has no honest genome-wide
    number, so the run is given the held-out chromosomes only and the block never exists. The
    reason is written into the result rather than left for a reader to infer from an absent key.
    """
    assert "genome_wide" not in scored
    scope = scored["provenance"]["scope"]
    assert scope["genome_wide_computed"] is False
    assert "NOT COMPUTED" in scope["note"]
    assert scope["held_out_chroms"] == scope["scored_chroms"]


def test_the_held_out_scope_cannot_name_an_unscored_chromosome(two_chrom_regime, model) -> None:
    src = open_source(store=two_chrom_regime)
    try:
        with pytest.raises(ValueError, match="which this run does not score"):
            H.run_bench(model, src, "cpu", kinds=("impute",), batch_windows=4,
                        blocks=("E",), held_out_chroms=["chr3", "chr22"])
    finally:
        src.close()

def test_a_count_only_model_gets_nb_nll_and_the_other_two_keys_are_ABSENT(
        regime_file, count_only_model) -> None:
    """Absent, not nan. A nan is skipped by every finiteness filter downstream and then reads as
    "the head produced garbage" rather than "there is no head"."""
    rec, arms = _one_track(count_only_model, regime_file)
    assert set(arms) == {"count"}, "no signal head, so there is no pval arm to score"
    assert np.isfinite(arms["count"]["nb_nll"])
    assert "gaussian_nll" not in arms["count"]
    assert "bernoulli_nll" not in arms["count"]
    # ... and bench still computes `auprc` off the NB-mean fallback, which is exactly the number
    # `bernoulli_nll` refuses to take a logarithm of.
    assert "auprc" in arms["count"]


def test_the_peak_head_is_detected_at_stream_time_rather_than_from_the_value_range(
        regime_file, count_only_model, model) -> None:
    """`peak_score` is `sigmoid(peak_logit)` OR the NB mean, and the two are not distinguishable
    afterwards: an NB mean sits inside [0, 1] on most bins of most assays."""
    rec_count, _ = _one_track(count_only_model, regime_file)
    rec_all, _ = _one_track(model, regime_file)
    assert rec_count.has_peak_head is False
    assert rec_all.has_peak_head is True
    assert rec_count.has_pval is False and rec_all.has_pval is True
    # The fallback really is in range, so a range check would have said "peak head" here.
    vals = np.concatenate([rec_count.peak_score[c] for c in rec_count.chroms])
    assert ((vals >= 0.0) & (vals <= 1.0)).mean() > 0.5


def test_the_transform_moves_the_gaussian_test_loss_and_leaves_the_other_two_alone(
        regime_file, model) -> None:
    """The whole reason the loss path carries a transform: on a store the head predicts
    `arcsinh(-log10 p)`, so scoring it against raw `-log10 p` is a finite, plausible, wrong loss."""
    _, plain = _one_track(model, regime_file, signal_target_transform="none")
    _, bent = _one_track(model, regime_file, signal_target_transform="arcsinh")
    assert plain["count"]["gaussian_nll"] != pytest.approx(bent["count"]["gaussian_nll"], rel=1e-6)
    assert plain["count"]["nb_nll"] == pytest.approx(bent["count"]["nb_nll"])
    assert plain["count"]["bernoulli_nll"] == pytest.approx(bent["count"]["bernoulli_nll"])
    assert bent["pval"]["signal_target_transform"] == "arcsinh"


# ---------------------------------------------------------------------------
# 6 — the spaces contract: every pval-arm BENCHMARK number is in -log10 p
# ---------------------------------------------------------------------------
#
# Storage is transformed, the reader inverts it (pinned in `tests/test_store_reader.py` and, at the
# batch, in `tests/test_store_dataset.py`), training supervises in transformed space — so the
# benchmark path has to bend the PREDICTION back. These four tests are that last link: the number
# emitted is the one computed on the inverted prediction, the loss tier is deliberately NOT
# inverted, and the `none` path is untouched to the bit.

def _cat(rec, d):
    from candi.bench import eic as E
    return E.dict_to_arr(d, rec.chroms)


def test_the_pval_arm_scores_the_INVERTED_prediction_against_the_raw_truth(regime_file, model):
    """The head trained on `arcsinh(-log10 p)`; the truth is `-log10 p`; the metric is `-log10 p`.

    `mse` is the check because it is the least forgiving: it is not rank-based, so the only way to
    land on the hand-computed value is to have applied `sinh` to the mean before squaring. The
    uninverted number is computed too and asserted DIFFERENT — a test that only matched one formula
    would pass on a harness that had silently stopped inverting if the two happened to be close.
    """
    from candi.bench import distributional as D

    rec, bent = _one_track(model, regime_file, signal_target_transform="arcsinh")
    y = _cat(rec, rec.pval)
    # float64, because that is what the inversion returns and a float32 `sinh` here would differ
    # from the harness's number at float32's epsilon for a reason that has nothing to do with space.
    mu_t = _cat(rec, rec.signal_mu).astype(np.float64)     # the head's own output, arcsinh space
    sig_t = _cat(rec, rec.signal_sigma).astype(np.float64)
    want = float(np.mean((y - np.sinh(mu_t)) ** 2))
    naive = float(np.mean((y - mu_t) ** 2))               # what the pre-contract harness computed

    assert bent["pval"]["mse"] == pytest.approx(want, rel=1e-12)
    assert bent["pval"]["mse"] != pytest.approx(naive, rel=1e-3), "the inversion did nothing"
    # `crps` reads the whole distribution, so it needs the inverted sigma as well.
    assert bent["pval"]["crps"] == pytest.approx(
        D.gauss_crps_mean(np.sinh(mu_t), np.cosh(mu_t) * sig_t, y), rel=1e-12)


def test_the_arm_records_the_space_it_was_scored_in(regime_file, model):
    """A reader must be able to tell an inverted row from a pre-contract one, and `pred_inversion`
    must say `"none"` out loud rather than being absent — absent is what a pre-contract row is."""
    from candi.bench import distributional as D

    _, bent = _one_track(model, regime_file, signal_target_transform="arcsinh")
    _, plain = _one_track(model, regime_file, signal_target_transform="none")
    assert bent["pval"]["pred_space"] == plain["pval"]["pred_space"] == D.SIGNAL_EVAL_SPACE
    assert bent["pval"]["pred_inversion"] == "arcsinh"
    assert plain["pval"]["pred_inversion"] == "none"
    # A label is not a measure: `macro_mean` averages scalars, so neither key can reach a macro.
    assert "pred_space" not in H.macro_mean({"k": bent}, "pval", kind=bent["pval"]["kind"])


def test_the_inversion_moves_the_measures_and_leaves_the_rank_keys_alone(regime_file, model):
    """`sinh` is strictly increasing, so anything that reads only the ORDER of the predictions is
    invariant under it. `gwspear` moving would mean the inversion did something other than a
    monotone map; `mse` not moving would mean it did nothing at all."""
    _, plain = _one_track(model, regime_file, signal_target_transform="none")
    _, bent = _one_track(model, regime_file, signal_target_transform="arcsinh")
    assert plain["pval"]["mse"] != pytest.approx(bent["pval"]["mse"], rel=1e-6)
    assert plain["pval"]["gwspear"] == pytest.approx(bent["pval"]["gwspear"], rel=1e-9)
    # The B-block ranks by `peak_score`, which is the peak head's — no signal-space value reaches it.
    assert plain["pval"]["auprc"] == bent["pval"]["auprc"]
    assert plain["pval"]["peak_overlap_0.01"] == bent["pval"]["peak_overlap_0.01"]


def test_the_panel_binarisation_thresholds_in_minus_log10_p_as_well(regime_file, model):
    """`panel_specificity` is the one pval-arm measure that lives at the PANEL level, and its call
    is `prediction >= 2` — an absolute number in `-log10 p`. So `_binarise` inverts too, or the
    panel would be scored on a threshold that means something different on a store than on an h5.
    """
    from candi.bench import partitions as P

    rec, _ = _one_track(model, regime_file, signal_target_transform="arcsinh")
    mu = _cat(rec, rec.signal_mu).astype(np.float64)
    _, call_arc, _ = H._binarise(rec, "arcsinh")
    _, call_none, _ = H._binarise(rec, "none")
    assert np.array_equal(call_arc, np.packbits(np.sinh(mu) >= P.BINARISE_THRESHOLD))
    assert np.array_equal(call_none, np.packbits(mu >= P.BINARISE_THRESHOLD))
    # Inverting the prediction is the same as moving the threshold to `arcsinh(2)`, so the two calls
    # differ on exactly the bins that fall between the two cut points — and nowhere else.
    straddles = bool(((mu >= np.arcsinh(P.BINARISE_THRESHOLD))
                      & (mu < P.BINARISE_THRESHOLD)).any())
    assert (not np.array_equal(call_arc, call_none)) is straddles


def test_the_loss_tier_is_the_exception_and_keeps_the_UNINVERTED_prediction(regime_file, model):
    """`gaussian_nll` mirrors the training loss: prediction untouched, TRUTH bent forward. Scoring
    it on the inverted prediction would give a finite, plausible number that is not the one the
    training loop prints."""
    from candi.bench import distributional as D

    rec, bent = _one_track(model, regime_file, signal_target_transform="arcsinh")
    y_arr = _cat(rec, rec.pval)
    mu_t = _cat(rec, rec.signal_mu).astype(np.float64)
    sig_t = _cat(rec, rec.signal_sigma).astype(np.float64)
    want = D.gaussian_nll(mu_t, sig_t * sig_t, D.transform_signal_target(y_arr, "arcsinh"))
    assert bent["pval"]["gaussian_nll"] == pytest.approx(want, rel=1e-12)
    assert bent["count"]["gaussian_nll"] == pytest.approx(want, rel=1e-12)
    # The inverted alternative is a different number, which is what makes the choice load-bearing.
    inverted = D.gaussian_nll(np.sinh(mu_t), (np.cosh(mu_t) * sig_t) ** 2, y_arr)
    assert bent["pval"]["gaussian_nll"] != pytest.approx(inverted, rel=1e-3)


def test_the_none_path_is_bit_identical_to_scoring_without_any_inversion(regime_file, model):
    """The h5 path resolves to `none`, and `none` must be the identity to the BIT — not to a
    tolerance. `invert_signal_prediction` returns the caller's own arrays there, so the blocks below
    are handed exactly what `score_track` hands them and `==` is the right comparison."""
    from candi.bench import distributional as D
    from candi.bench import eic as E

    rec, plain = _one_track(model, regime_file, signal_target_transform="none")
    y = _cat(rec, rec.pval)
    direct = {
        **E.score_track(rec.pval, rec.signal_mu, rec.chroms,
                        gene_annotations=ann.gene_annotations(),
                        enh_annotations=ann.enhancer_annotations()),
        **D.gauss_suite(_cat(rec, rec.signal_mu), _cat(rec, rec.signal_sigma), y,
                        seed=0, n_pairs=500),
    }
    for key, want in direct.items():
        if isinstance(want, float):
            # `assert_array_equal` rather than `==` for one reason only: a block whose window set is
            # empty on this synthetic panel returns nan, and nan is not equal to itself.
            np.testing.assert_array_equal(plain["pval"][key], want,
                                          err_msg=f"{key} moved on the identity path")


def test_the_cli_reads_the_target_space_off_the_run_json_and_a_flag_overrides_it(tmp_path) -> None:
    """The run json holds the RESOLVED value; a checkpoint holds nothing. So the json wins by
    default — and an explicit flag still wins over the json, because a flag argparse accepts and the
    program discards is worse than either precedence order."""
    from candi.bench.cli import build_parser, resolve_signal_target_transform

    run = tmp_path / "run.json"
    run.write_text(json.dumps({"config": {"signal_target_transform": "arcsinh",
                                          "arch": {"num_assays": 3}}}))
    p = build_parser()
    base = ["--store", "x", "--ckpt", "c", "--out", "o"]

    a = p.parse_args(base)
    assert resolve_signal_target_transform(a) == ("none", "default")

    a = p.parse_args(base + ["--arch-from", str(run)])
    got, src = resolve_signal_target_transform(a)
    assert got == "arcsinh" and str(run) in src

    a = p.parse_args(base + ["--arch-from", str(run), "--signal-target-transform", "log1p"])
    assert resolve_signal_target_transform(a) == ("log1p", "--signal-target-transform")

    # A run json from before D30 has no such field, and falling back is the whole point.
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"config": {"arch": {"num_assays": 3}}}))
    a = p.parse_args(base + ["--arch-from", str(old)])
    assert resolve_signal_target_transform(a) == ("none", "default")


def test_the_cli_emits_the_test_loss_per_track_and_in_the_macro(regime_file, ckpt,
                                                                tmp_path) -> None:
    """`python -m candi.bench` IS the test-loss reporter: the same three keys the training loop
    prints as `train/nll` and the monitor reports as the val loss."""
    from candi.bench.cli import main

    out = tmp_path / "loss.json"
    rc = main(["--store", str(regime_file), "--ckpt", str(ckpt), "--out", str(out),
               "--heads", "count,signal,peak", "--depth-center", "24.25",
               "--blocks", "E,D,B", "--c-index-pairs", "500",
               "--signal-target-transform", "arcsinh", "--quiet"])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["provenance"]["signal_target_transform"] == "arcsinh"
    assert res["provenance"]["signal_target_transform_source"] == "--signal-target-transform"
    assert NLL_KEYS <= set(res["macro"]["count"])
    for arms in res["per_track"].values():
        assert NLL_KEYS <= set(arms["count"])
        assert arms["count"]["signal_target_transform"] == "arcsinh"
