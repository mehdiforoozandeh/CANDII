"""The regime file: the declared column order (D14), window eligibility (D12), DSF policy (D23).

The regime is the file that replaced `h5.attrs`. Everything the old bake froze into the h5 lives
here now, which means a silent misparse of this file is a silently different experiment — hence
the coverage on the *failure* paths, not only the happy one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from candi.store import layout as L
from candi.store.reader import CorpusStore
from candi.store.regime import (
    DEFAULT_DSF_LEVELS, DEFAULT_MIN_VALID_FRAC, DsfPolicy, Regime, RegimeError, RegionSet,
    WindowPlan, eligible_starts,
)

from tests.test_store_reader import (
    ASSAYS, BIOSAMPLES, BLACKLIST_BINS, CHROM_SIZES, N_BINS, RES, make_store,
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


# ---------------------------------------------------------------------------------------------
# t14 / D31 — eval_pairs
# ---------------------------------------------------------------------------------------------
# The bake finds the imputation target by string surgery on the prompt's name (T_X -> V_X / B_X).
# D16 makes store biosample names opaque ids, so here the pairing is DECLARED. These pin the two
# things a declaration has to buy over a derivation: it can be wrong in ways a validator catches,
# and it is absent by default.


def test_eval_pairs_default_to_empty_which_means_no_imputation_eval(store):
    r = Regime.from_dict(regime_dict(store))
    assert r.eval_pairs == () and not r.has_eval_pairs
    assert r.eval_inputs == () and r.eval_targets == ()


def test_eval_pairs_accept_both_spellings_and_round_trip(store):
    a = Regime.from_dict(regime_dict(store, eval_pairs=[["T_aa", "V_aa"]]))
    b = Regime.from_dict(regime_dict(
        store, eval_pairs=[{"input": "T_aa", "target": "V_aa"}]))
    assert a.eval_pairs == b.eval_pairs == (("T_aa", "V_aa"),)
    assert a.has_eval_pairs and a.eval_inputs == ("T_aa",) and a.eval_targets == ("V_aa",)
    assert Regime.from_dict(a.to_dict()).eval_pairs == a.eval_pairs


def test_eval_inputs_and_targets_dedupe_in_declaration_order(store):
    r = Regime.from_dict(regime_dict(
        store, biosamples={"train": [], "eval": ["T_aa", "V_aa"]},
        eval_pairs=[["T_aa", "V_aa"], ["V_aa", "T_aa"]]))
    assert r.eval_inputs == ("T_aa", "V_aa")
    assert r.eval_targets == ("V_aa", "T_aa")


def test_a_biosample_paired_with_itself_is_refused(store):
    """An identity copy dressed as an imputation — the one mistake the list exists to prevent."""
    with pytest.raises(RegimeError, match="with itself"):
        Regime.from_dict(regime_dict(store, eval_pairs=[["T_aa", "T_aa"]]))


def test_a_repeated_pair_is_refused(store):
    with pytest.raises(RegimeError, match="repeats the pair"):
        Regime.from_dict(regime_dict(store, eval_pairs=[["T_aa", "V_aa"], ["T_aa", "V_aa"]]))


@pytest.mark.parametrize("bad", [["T_aa"], "T_aa,V_aa", [{"input": "T_aa"}]])
def test_a_malformed_eval_pair_names_itself(store, bad):
    with pytest.raises(RegimeError, match="eval_pairs"):
        Regime.from_dict(regime_dict(store, eval_pairs=[bad]))


def test_a_target_that_also_trains_is_refused(store, corpus):
    """The target holds the ground truth being scored; one the model trained on is not imputation.

    The INPUT is deliberately allowed to be a training biosample — the imputation prompt is exactly
    a biosample the model knows — so only one half of the pair is checked.
    """
    r = Regime.from_dict(regime_dict(
        store, biosamples={"train": ["V_aa"], "eval": []}, eval_pairs=[["T_aa", "V_aa"]]))
    with pytest.raises(RegimeError, match="also in"):
        r.validate_against(corpus)
    ok = Regime.from_dict(regime_dict(
        store, biosamples={"train": ["T_aa"], "eval": ["V_aa"]}, eval_pairs=[["T_aa", "V_aa"]]))
    ok.validate_against(corpus)


def test_a_pair_naming_a_biosample_the_store_lacks_is_refused(store, corpus):
    r = Regime.from_dict(regime_dict(store, eval_pairs=[["T_aa", "Z_nope"]]))
    with pytest.raises(RegimeError, match="not in"):
        r.validate_against(corpus)


# ---------------------------------------------------------------------------------------------
# D32 — restricting the train split to a BED (t79)
# ---------------------------------------------------------------------------------------------

PILOT_BED = REPO / "configs" / "regions" / "encode_pilot_hg38.bed"
PILOT_REGIME = REPO / "configs" / "regime.eic_pilot.json"


def _bed(path: Path, spans_in_bins) -> Path:
    """Write a BED4 whose intervals are given in BIN coordinates, so the tests read as bins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{c}\t{a * RES}\t{b * RES}\tR{i}\n" for i, (c, a, b) in enumerate(spans_in_bins)),
        encoding="utf-8",
    )
    return path


def _regions(path: Path, **over) -> dict:
    obj = {"bed": path.name,
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "policy": "contain"}
    obj.update(over)
    return obj


def test_a_regime_without_regions_plans_exactly_what_it_planned_before(store, tmp_path, corpus):
    """D32 is additive. Pinned lists, not a re-derivation — a re-derivation cannot catch a drift."""
    r = _regime(store, tmp_path)
    assert r.regions is None
    assert r.windows(corpus, "train") == [("chr1", 64 * i) for i in range(31)]
    assert r.windows(corpus, "eval") == [("chr2", s) for s in (0, 448, 512, 576, 640, 704)]


def test_every_shipped_regime_carries_regions_exactly_when_its_json_declares_them():
    """Sweeps `configs/`, so a regime added tomorrow is covered without editing this test.

    Both sides of D32 are live in the shipped set — `regime.eic_pilot.json` declares a `regions`
    BED and the rest omit the key — so this checks the mapping, not just the None half. It reads
    only tracked files on purpose: it used to pin two regimes under `cruxvault/results/t9/`, which
    is gitignored, and so it failed in every fresh clone and worktree.
    """
    shipped = sorted((REPO / "configs").glob("regime.*.json"))
    assert len(shipped) >= 2, "configs/ should ship regimes; the glob found none to check"
    declared = [p.name for p in shipped if "regions" in json.loads(p.read_text(encoding="utf-8"))]
    assert declared, "no shipped regime declares regions — the set-case is no longer covered"
    assert len(declared) < len(shipped), "every shipped regime declares regions — None is uncovered"
    for path in shipped:
        has_block = "regions" in json.loads(path.read_text(encoding="utf-8"))
        regions = Regime.from_file(path).regions
        assert (regions is not None) == has_block, path.name


def test_to_dict_omits_regions_when_unset(store, tmp_path):
    """A `"regions": null` in run.json would be a claim about the training scope no run made."""
    d = _regime(store, tmp_path).to_dict()
    assert "regions" not in d
    assert Regime.from_dict(d).to_dict() == d


def test_to_dict_writes_regions_back_verbatim_when_set(store, tmp_path):
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400)])
    r = _regime(store, tmp_path, regions=_regions(bed))
    assert r.to_dict()["regions"] == {"bed": "r.bed", "sha256": r.regions.sha256,
                                      "policy": "contain"}


def test_every_planned_train_window_lies_inside_one_region(store, tmp_path, corpus):
    """Hand-computed: bins [100,400) admits 128/192/256/320; bins [500,756) admits 512/576/640."""
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400), ("chr1", 500, 756)])
    wins = _regime(store, tmp_path, regions=_regions(bed)).windows(corpus, "train")
    assert wins == [("chr1", s) for s in (128, 192, 256, 320, 512, 576, 640)]
    spans = ((100, 400), (500, 756))
    assert all(any(a <= s and s + CTX <= b for a, b in spans) for _, s in wins)


def test_regions_do_not_touch_the_eval_split(store, tmp_path, corpus):
    """Rule 2 is about training loci; eval is whole chromosomes."""
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400)])
    r = _regime(store, tmp_path, regions=_regions(bed))
    assert r.windows(corpus, "eval") == _regime(store, tmp_path).windows(corpus, "eval")


def test_a_window_straddling_a_region_edge_is_dropped_even_at_90_percent(store, tmp_path, corpus):
    """The test that separates containment from mask-ANDing.

    Region bins [0,122). The tile at 64 has 58 of its 64 bins inside — 90.6 %, above
    `min_valid_frac` — so ANDing the region into the mask would keep it and admit 6 bins of
    non-region sequence. Containment drops it.
    """
    bed = _bed(tmp_path / "r.bed", [("chr1", 0, 122)])
    wins = _regime(store, tmp_path, regions=_regions(bed)).windows(corpus, "train")
    assert wins == [("chr1", 0)]
    assert (122 - 64) / CTX >= DEFAULT_MIN_VALID_FRAC        # it would have survived semantics (i)


def test_d12_still_rejects_a_masked_window_inside_a_region(store, tmp_path, corpus):
    """chr2 is blacklisted over bins [100,400). A region around it does not resurrect them."""
    bed = _bed(tmp_path / "r.bed", [("chr2", 64, 512)])
    r = _regime(store, tmp_path, train_chroms=["chr2"], eval_chroms=["chr1"],
                regions=_regions(bed))
    assert r.windows(corpus, "train") == [("chr2", 448)]


def test_a_region_on_a_chromosome_outside_train_chroms_contributes_nothing(store, tmp_path, corpus):
    """The chromosome list is consulted first — that is how the chr20/21/22 cut is declared."""
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400), ("chr2", 500, 700)])
    wins = _regime(store, tmp_path, regions=_regions(bed)).windows(corpus, "train")
    assert wins == [("chr1", s) for s in (128, 192, 256, 320)]


def test_regions_shorter_than_the_context_name_the_bed_as_the_cause(store, tmp_path, corpus):
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 150), ("chr1", 300, 340)])
    r = _regime(store, tmp_path, regions=_regions(bed))
    with pytest.raises(RegimeError, match=r"no eligible train window.*wholly inside a region"):
        r.windows(corpus, "train")


def test_a_bed_whose_hash_does_not_match_is_loud(store, tmp_path):
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400)])
    wrong = "0" * 64
    with pytest.raises(RegimeError, match=r"r\.bed.*" + wrong):
        _regime(store, tmp_path, regions=_regions(bed, sha256=wrong))
    bed.write_text(bed.read_text() + "chr1\t20000\t30000\tRx\n", encoding="utf-8")
    with pytest.raises(RegimeError, match=r"r\.bed"):
        Regime.from_file(tmp_path / "regime.json")


def test_the_sha256_is_required_not_optional(store, tmp_path):
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400)])
    obj = _regions(bed)
    obj.pop("sha256")
    with pytest.raises(RegimeError, match="sha256"):
        _regime(store, tmp_path, regions=obj)


@pytest.mark.parametrize(
    "over, match",
    [
        ({"policy": "intersect"}, "policy"),
        ({"bed": "nope.bed"}, "not a file"),
    ],
)
def test_a_malformed_regions_block_names_the_field(store, tmp_path, over, match):
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400)])
    with pytest.raises(RegimeError, match=match):
        _regime(store, tmp_path, regions=_regions(bed, **over))


def test_the_bed_resolves_against_the_regime_files_own_directory(store, tmp_path, corpus):
    """The regime is copied into run.json and read back from elsewhere; cwd must not decide."""
    bed = _bed(tmp_path / "sub" / "r.bed", [("chr1", 100, 400)])
    r = _regime(store, tmp_path, regions=_regions(bed, bed="sub/r.bed"))
    assert Path(r.regions.resolved) == bed
    absolute = _regime(store, tmp_path, regions=_regions(bed, bed=str(bed)))
    assert Path(absolute.regions.resolved) == bed


@pytest.mark.parametrize("bad", ["chr1\t100\n", "chr1\tx\t200\n", "chr1\t200\t100\n", "# only\n"])
def test_a_malformed_bed_line_is_named(store, tmp_path, bad):
    p = tmp_path / "r.bed"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(RegimeError, match="BED"):
        _regime(store, tmp_path, regions=_regions(p))


# -- the shipped ENCODE Pilot Regions ---------------------------------------------------------


def _pilot_spans(r: Regime, chroms):
    return [(c, a, b) for c in chroms for a, b in r.regions.bin_spans(c, RES)]


def test_the_shipped_pilot_bed_is_pinned():
    """Catches a regeneration of the BED under different liftOver options (t79 §3)."""
    r = Regime.from_file(PILOT_REGIME)
    iv = r.regions.intervals
    assert len(iv) == 44 and sum(e - s for _, s, e, _ in iv) == 29_984_074
    assert r.regions.sha256 == hashlib.sha256(PILOT_BED.read_bytes()).hexdigest()
    assert len(r.regions.chroms) == 21
    cut = [x for x in iv if x[0] in set(r.eval_chroms)]
    assert len(cut) == 4 and sum(e - s for _, s, e, _ in cut) == 4_395_877
    train = [x for x in iv if x[0] not in set(r.eval_chroms)]
    assert len(train) == 40 and sum(e - s for _, s, e, _ in train) == 25_588_197
    assert sorted({x[0] for x in train}) == sorted(r.train_chroms) and "chrX" in r.train_chroms
    assert len(r.train_chroms) == 18 and r.eval_chroms == ("chr20", "chr21", "chr22")


def test_the_pilot_training_scope_is_a_containment_count_not_a_division():
    r = Regime.from_file(PILOT_REGIME)
    spans = _pilot_spans(r, r.train_chroms)
    assert sum(b - a for _, a, b in spans) == 1_023_489
    assert 25_588_197 % RES != 0                 # so `bp // 25` is the wrong rule, and is not used


def test_the_pilot_regime_plans_its_training_windows(tmp_path):
    """The contained-window count at context_bins=768, stride=768, over an all-valid mask.

    TWO numbers, and they are not the same number.

    * **1,328** is the region-anchored packing CAPACITY — how many disjoint 768-bin windows fit
      inside the regions if each region's tiling starts at that region's own first bin. This is
      the figure in `plan/BENCHMARK_DESIGN.md` §3.1 and in `cruxvault/results/t79/G2_PILOT_HG38.md`
      §5.3 ("1,328 fully-contained windows = 1,019,904 bins = 99.6 %").
    * **1,294** is what the APPROVED sampler produces. D32 filters the tiling `eligible_starts`
      already lays down, and that tiling is anchored at bin 0 of the CHROMOSOME, not at each
      region. 34 of the 40 training regions therefore lose their leading partial tile.

    Both are pinned so the gap cannot be lost. Which one the regime should produce is a PI
    decision: reaching 1,328 means anchoring the tiling per region, which changes the candidate
    starts and is not what §5.3 approved.
    """
    r = Regime.from_file(PILOT_REGIME)
    ctx = r.context_bins
    stride = r.window_plan.stride(ctx)
    capacity = sum((b - a) // ctx for _, a, b in _pilot_spans(r, r.train_chroms))
    assert capacity == 1_328 and capacity * ctx == 1_019_904

    planned = 0
    for chrom in r.train_chroms:
        end = max(b for _, a, b in _pilot_spans(r, [chrom]))
        cand = np.arange(0, end + ctx, stride, dtype=np.int64)
        kept = r.regions.contained_starts(chrom, cand, ctx, RES)
        spans = r.regions.bin_spans(chrom, RES)
        assert all(any(a <= s and s + ctx <= b for a, b in spans) for s in kept.tolist())
        planned += int(kept.size)
    assert planned == 1_294


def test_a_bed_named_on_a_command_line_is_hashed_rather_than_hash_checked(tmp_path):
    """`from_obj` checks a DECLARED hash because the regime cannot pin the BED any other way. A
    scope named on a command line (t89's `--eval-regions`) has no declaration to check against, so
    the hash is computed and travels in the run's provenance instead — which is what makes two runs
    comparable or not. Same object, same intervals, same `contain` rule."""
    bed = _bed(tmp_path / "r.bed", [("chr1", 100, 400)])
    got = RegionSet.from_bed(bed)
    assert got.sha256 == hashlib.sha256(bed.read_bytes()).hexdigest()
    assert got.policy == "contain"
    assert got.intervals == RegionSet.from_obj(_regions(bed), base=tmp_path).intervals
    assert got.to_dict()["sha256"] == got.sha256


def test_the_shipped_pilot_bed_reads_as_the_44_regions_the_design_names():
    """Read off the file, not off the doc. §3.1 pins 44 regions and 29,984,074 bp in hg38, and the
    same BED is the eval scope t89 offers — so a lift that silently changed would move the training
    scope and the selection scope at once."""
    rs = RegionSet.from_bed(PILOT_BED)
    assert len(rs.intervals) == 44
    assert sum(e - s for _, s, e, _ in rs.intervals) == 29_984_074
    assert rs.sha256 == "13e11a198fdee08edb7797d1e402b5d985846b5a7d973ade91e8511462acb7a3"


def test_a_bed_that_is_not_there_is_refused_by_name(tmp_path):
    with pytest.raises(RegimeError, match="is not a file"):
        RegionSet.from_bed(tmp_path / "absent.bed")
