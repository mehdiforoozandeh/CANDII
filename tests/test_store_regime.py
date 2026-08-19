"""The regime file: the declared column order (D14), window eligibility (D12), DSF policy (D23).

The regime is the file that replaced `h5.attrs`. Everything the old bake froze into the h5 lives
here now, which means a silent misparse of this file is a silently different experiment — hence
the coverage on the *failure* paths, not only the happy one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from candi.store import layout as L
from candi.store.reader import CorpusStore
from candi.store.regime import (
    DEFAULT_DSF_LEVELS, DEFAULT_MIN_VALID_FRAC, DsfPolicy, Regime, RegimeError, WindowPlan,
    eligible_starts,
)

from tests.test_store_reader import (
    ASSAYS, BIOSAMPLES, BLACKLIST_BINS, CHROM_SIZES, N_BINS, make_store,
)

CTX = 64
REPO = Path(__file__).resolve().parent.parent


def regime_dict(store: Path, **over) -> dict:
    obj = {
        "store": str(store),
        "assays": list(ASSAYS),
        "biosamples": {"train": ["T_aa"], "eval": ["V_aa"]},
        "context_bins": CTX,
        "train_chroms": ["chr1"],
        "eval_chroms": ["chr2"],
        "window_plan": {"type": "tile", "stride_bins": CTX, "min_valid_frac": 0.9},
        "dsf": {"policy": "discrete", "levels": [1, 2, 4, 8]},
        "kinds": ["counts", "peaks"],
        "seed": 42,
    }
    obj.update(over)
    return obj


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    return make_store(tmp_path_factory.mktemp("regime"))


@pytest.fixture()
def corpus(store):
    c = CorpusStore(store)
    yield c
    c.close()


def _regime(store: Path, tmp_path: Path, **over) -> Regime:
    p = tmp_path / "regime.json"
    p.write_text(json.dumps(regime_dict(store, **over)), encoding="utf-8")
    return Regime.from_file(p)


# ---------------------------------------------------------------------------------------------
# parsing — the schema in STORE_PLAN.md §4, and the shipped example
# ---------------------------------------------------------------------------------------------


def test_the_documented_schema_round_trips(store, tmp_path):
    r = _regime(store, tmp_path)
    assert r.assays == ASSAYS and r.context_bins == CTX and r.seed == 42
    assert r.train_biosamples == ("T_aa",) and r.eval_biosamples == ("V_aa",)
    assert r.train_chroms == ("chr1",) and r.eval_chroms == ("chr2",)
    assert r.kinds == ("counts", "peaks")
    assert r.window_plan.min_valid_frac == 0.9 and r.window_plan.stride(CTX) == CTX
    assert r.dsf.policy == "discrete" and r.dsf.levels == (1.0, 2.0, 4.0, 8.0)
    assert Regime.from_dict(r.to_dict()).to_dict() == r.to_dict()
    assert len(r.sha256) == 64 and r.raw is not None


def test_the_shipped_example_regime_parses():
    """`configs/regime.eic_smoke.json` is the template people copy; a broken one is a trap."""
    r = Regime.from_file(REPO / "configs" / "regime.eic_smoke.json")
    assert r.context_bins == 768 and r.num_assays == 8
    assert r.train_chroms == ("chr19",) and r.eval_chroms == ("chr21",)
    assert r.dsf.levels == (1.0, 2.0, 4.0, 8.0)
    assert r.window_plan.min_valid_frac == DEFAULT_MIN_VALID_FRAC


@pytest.mark.parametrize(
    "over, match",
    [
        ({"assays": []}, "empty"),
        ({"assays": ["H3K4me3", "H3K4me3"]}, "duplicate"),
        ({"assays": ["H3K4me3", L.CONTROL_TRACK]}, "control"),
        ({"context_bins": 0}, "positive"),
        ({"kinds": ["peaks"]}, "must include 'counts'"),
        ({"kinds": ["counts", "signal"]}, "unknown kind"),
        ({"window_plan": {"type": "random"}}, "not supported"),
        ({"window_plan": {"min_valid_frac": 1.5}}, "min_valid_frac"),
        ({"biosamples": {"holdout": []}}, "unknown split"),
        ({"dsf": {"policy": "sqrt"}}, "policy"),
        ({"dsf": {"policy": "loguniform", "min": 8, "max": 2}}, "min <= max"),
    ],
)
def test_a_malformed_regime_names_the_field(store, tmp_path, over, match):
    with pytest.raises(RegimeError, match=match):
        _regime(store, tmp_path, **over)


def test_a_missing_required_key_is_named(store, tmp_path):
    obj = regime_dict(store)
    obj.pop("assays")
    p = tmp_path / "r.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(RegimeError, match="assays"):
        Regime.from_file(p)


# ---------------------------------------------------------------------------------------------
# D14 — the declared order IS the column order
# ---------------------------------------------------------------------------------------------


def test_the_declared_list_maps_onto_storage_columns(store, tmp_path, corpus):
    r = _regime(store, tmp_path)
    tracks = corpus["T_aa"].tracks("counts")
    assert r.assay_columns(tracks) == [tracks.index(a) for a in ASSAYS]


def test_a_permuted_declaration_permutes_the_columns(store, tmp_path, corpus):
    """Not a silent pass: the column INDICES must actually move with the declaration."""
    tracks = corpus["T_aa"].tracks("counts")
    forward = _regime(store, tmp_path).assay_columns(tracks)
    permuted = _regime(store, tmp_path, assays=list(reversed(ASSAYS))).assay_columns(tracks)
    assert permuted == forward[::-1] != forward


def test_an_assay_absent_from_the_store_raises_naming_it(store, tmp_path, corpus):
    r = _regime(store, tmp_path, assays=["H3K4me3", "H3K27ac"])
    with pytest.raises(RegimeError, match="H3K27ac"):
        r.validate_against(corpus)
    with pytest.raises(RegimeError, match="H3K27ac"):
        r.assay_columns(corpus["T_aa"].tracks("counts"))


def test_an_assay_only_some_biosamples_have_is_fine(store, tmp_path, corpus):
    """`V_aa` has no ATAC-seq. That is a MISSING column at load time, not a regime error."""
    _regime(store, tmp_path).validate_against(corpus)
    assert not corpus["V_aa"].has("ATAC-seq", "counts")


def test_validation_names_a_bad_biosample_chrom_or_kind(store, tmp_path, corpus):
    with pytest.raises(RegimeError, match="T_nope"):
        _regime(store, tmp_path, biosamples={"train": ["T_nope"], "eval": []}).validate_against(corpus)
    with pytest.raises(RegimeError, match="chr9"):
        _regime(store, tmp_path, train_chroms=["chr9"]).validate_against(corpus)
    with pytest.raises(RegimeError, match="not an evaluation"):
        _regime(store, tmp_path, eval_chroms=["chr1"]).validate_against(corpus)


# ---------------------------------------------------------------------------------------------
# D12 — window eligibility
# ---------------------------------------------------------------------------------------------


def test_eligible_starts_is_the_mean_rule(store):
    mask = np.ones(1000, dtype=np.uint8)
    mask[200:260] = 0                              # 60 invalid bins
    got = eligible_starts(mask, 100, 0.9, 10)
    for s in got:
        assert mask[s:s + 100].mean() >= 0.9
    for s in range(0, 901, 10):
        if s not in set(got.tolist()):
            assert mask[s:s + 100].mean() < 0.9
    assert eligible_starts(mask, 2000, 0.9, 10).size == 0     # context longer than the chromosome


def test_no_planned_window_is_below_min_valid_frac(store, tmp_path, corpus):
    r = _regime(store, tmp_path)
    wins = r.windows(corpus, "eval")
    full = corpus.genome.mask("chr2", 0, N_BINS["chr2"])
    assert wins and all(c == "chr2" for c, _ in wins)
    for _, s in wins:
        assert full[s:s + CTX].mean() >= 0.9
    n_tiles = len(range(0, N_BINS["chr2"] - CTX + 1, CTX))
    assert len(wins) < n_tiles                     # the blacklist really rejected something
    assert all(not (s < BLACKLIST_BINS[1] and s + CTX > BLACKLIST_BINS[0]) for _, s in wins)


def test_min_valid_frac_is_overridable_per_regime(store, tmp_path, corpus):
    loose = _regime(store, tmp_path,
                    window_plan={"type": "tile", "stride_bins": CTX, "min_valid_frac": 0.0})
    strict = _regime(store, tmp_path)
    assert len(loose.windows(corpus, "eval")) > len(strict.windows(corpus, "eval"))
    assert len(loose.windows(corpus, "eval")) == len(range(0, N_BINS["chr2"] - CTX + 1, CTX))


def test_the_stride_controls_overlap(store, tmp_path, corpus):
    tiled = _regime(store, tmp_path).windows(corpus, "train")
    dense = _regime(store, tmp_path,
                    window_plan={"type": "tile", "stride_bins": CTX // 2}).windows(corpus, "train")
    assert len(dense) > len(tiled)
    assert tiled[:3] == [("chr1", 0), ("chr1", CTX), ("chr1", 2 * CTX)]


def test_windows_without_a_mask_layer_are_all_eligible(tmp_path, store):
    """t7 builds `mask.h5`; a store mid-build has none, and that must not stop a plumbing run."""
    corpus_root = make_store(tmp_path / "nomask")
    L.mask_path(corpus_root.parent).unlink()
    c = CorpusStore(corpus_root)
    r = _regime(corpus_root, tmp_path)
    assert not r.mask_available(c)
    assert len(r.windows(c, "eval")) == len(range(0, N_BINS["chr2"] - CTX + 1, CTX))


def test_a_context_longer_than_every_chromosome_is_an_error(store, tmp_path, corpus):
    r = _regime(store, tmp_path, context_bins=10 * N_BINS["chr2"])
    with pytest.raises(RegimeError, match="no eligible"):
        r.windows(corpus, "eval")


# ---------------------------------------------------------------------------------------------
# D23 — the DSF policy
# ---------------------------------------------------------------------------------------------


def test_the_default_dsf_policy_is_the_discrete_ladder():
    p = DsfPolicy.from_obj(None)
    assert p.policy == "discrete" and p.levels == DEFAULT_DSF_LEVELS
    rng = np.random.default_rng(0)
    draws = {p.sample(rng) for _ in range(200)}
    assert draws == {1.0, 2.0, 4.0, 8.0}


def test_loguniform_is_continuous_and_stays_inside_its_bounds():
    p = DsfPolicy.from_obj({"policy": "loguniform", "min": 1, "max": 8})
    rng = np.random.default_rng(0)
    draws = np.array([p.sample(rng) for _ in range(500)])
    assert draws.min() >= 1.0 and draws.max() <= 8.0
    assert len(set(draws.tolist())) > 400          # continuous, not the ladder
    assert not p.is_trivial


def test_dsf_off_and_a_single_level_are_both_trivial():
    assert DsfPolicy.from_obj({"policy": "off"}).is_trivial
    assert DsfPolicy.from_obj({"policy": "discrete", "levels": [1]}).is_trivial
    assert DsfPolicy.from_obj({"policy": "off"}).sample(np.random.default_rng(0)) == 1.0


def test_window_plan_defaults_to_non_overlapping_tiles():
    wp = WindowPlan.from_obj(None)
    assert wp.type == "tile" and wp.stride(768) == 768
    assert wp.min_valid_frac == DEFAULT_MIN_VALID_FRAC
