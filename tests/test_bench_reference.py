"""Layer 1 — the E-block equals the organizers' own code.

`EVAL_PLAN.md` §6. `candi.bench.eic` is written independently of `score_metrics.py`, so this is a
comparison between two implementations rather than a file against itself. The bar in the plan is
1e-12 relative; in practice the interval accumulation is arranged to give **0 ULP**, and these tests
assert the strict version — a drift to "merely 1e-12" means someone reordered a float sum, which is
worth knowing about even though it would not change any conclusion.

Adversarial inputs are the point. Random arrays agree trivially. What separates two implementations
is all-zero targets, constant predictions, heavy ties, single non-zero positions, arrays shorter than
100 (where `mse1obs` degenerates), and annotations that run off the front of the array.
"""
from __future__ import annotations

import numpy as np
import pytest

from candi.bench import eic
from tests.fixtures.reference import UPSTREAM_BLOB_SHA1, VENDORED, load_reference

CHROMS = ["chrT"]


@pytest.fixture(scope="module")
def ref():
    return load_reference()


def _bed_gene(rows):
    return [f"chrT\t{s}\t{e}\tG{i}\t.\t{strand}\n" for i, (s, e, strand) in enumerate(rows)]


def _bed_enh(rows):
    return [f"chrT\t{s}\t{e}\tE{i}\t0\t.\t{s}\t{e}\t0,0,0\t1\t{e-s}\t0\n"
            for i, (s, e) in enumerate(rows)]


def _pairs():
    """(label, y_true, y_pred) — the adversarial catalogue plus a few random draws."""
    rng = np.random.default_rng(0)
    n = 4000
    cases = [
        ("random_uniform", rng.random(n) * 10, rng.random(n) * 10),
        ("random_exponential", rng.exponential(2.0, n), rng.exponential(2.0, n)),
        ("pred_equals_true", (t := rng.random(n) * 5), t.copy()),
        ("pred_offset", (t := rng.random(n) * 5), t + 1.7),
        ("pred_scaled", (t := rng.random(n) * 5), t * 3.0),
        ("true_all_zero", np.zeros(n), rng.random(n)),
        ("pred_all_zero", rng.random(n), np.zeros(n)),
        ("pred_constant", rng.random(n), np.full(n, 0.5)),
        ("heavy_ties", rng.integers(0, 3, n).astype(float), rng.integers(0, 3, n).astype(float)),
        ("single_nonzero", np.eye(1, n, 17)[0] * 9.0, rng.random(n)),
        ("sparse_true", (rng.random(n) < 0.01) * rng.random(n) * 100, rng.random(n)),
        ("short_array_99", rng.random(99) * 4, rng.random(99) * 4),
        ("short_array_100", rng.random(100) * 4, rng.random(100) * 4),
        ("tiny_array_7", rng.random(7), rng.random(7)),
    ]
    return cases


@pytest.mark.parametrize("label,y_true,y_pred", _pairs(), ids=[c[0] for c in _pairs()])
def test_genomewide_measures_are_bit_identical(ref, label, y_true, y_pred) -> None:
    for name, ours, theirs in [
        ("mse", eic.mse(y_true, y_pred), ref.mse(y_true, y_pred)),
        ("gwcorr", eic.gwcorr(y_true, y_pred), ref.gwcorr(y_true, y_pred)),
        ("gwspear", eic.gwspear(y_true, y_pred), ref.gwspear(y_true, y_pred)),
        ("mse1obs", eic.mse1obs(y_true, y_pred), ref.mse1obs(y_true, y_pred)),
        ("mse1imp", eic.mse1imp(y_true, y_pred), ref.mse1imp(y_true, y_pred)),
    ]:
        if np.isnan(theirs):
            assert np.isnan(ours), f"{label}/{name}: reference nan, ours {ours}"
        else:
            assert ours == theirs, f"{label}/{name}: {ours!r} != {theirs!r}"


@pytest.mark.parametrize("label,y_true,y_pred", _pairs(), ids=[c[0] for c in _pairs()])
def test_msevar_is_bit_identical(ref, label, y_true, y_pred) -> None:
    rng = np.random.default_rng(1)
    var = rng.random(len(y_true)) + 1e-3
    assert eic.msevar(y_true, y_pred, var) == ref.msevar(y_true, y_pred, var=var)


@pytest.mark.parametrize("label,y_true,y_pred", _pairs(), ids=[c[0] for c in _pairs()])
def test_region_measures_are_bit_identical(ref, label, y_true, y_pred) -> None:
    """Genes chosen to hit every edge: off the front, off the back, overlapping, zero-length."""
    n = len(y_true)
    bp = 25 * n
    rows = [
        (0, 1000, "+"),              # start bin 0 -> promoter runs off the front (quirk 2)
        (500, 3000, "+"),
        (1200, 1800, "-"),           # minus strand -> promoter is downstream of `end`
        (1000, 5000, "+"),           # overlaps the previous ones (quirk 1)
        (max(bp - 500, 0), bp, "-"), # minus strand at the very end -> promoter runs off the back
        (2000, 2000, "+"),           # zero-length gene: end//25+1 makes it one bin wide (quirk 3)
    ]
    genes = _bed_gene([r for r in rows if r[0] <= r[1]])
    enh = _bed_enh([(0, 300), (900, 1500), (1000, 1400), (max(bp - 100, 0), bp)])
    td, pd_ = {"chrT": y_true}, {"chrT": y_pred}

    assert eic.mseprom(td, pd_, CHROMS, genes) == ref.mseprom(td, pd_, CHROMS, genes)
    assert eic.msegene(td, pd_, CHROMS, genes) == ref.msegene(td, pd_, CHROMS, genes)
    assert eic.mseenh(td, pd_, CHROMS, enh) == ref.mseenh(td, pd_, CHROMS, enh)


def test_multichromosome_concatenation_matches(ref) -> None:
    """`dict_to_arr` order is part of the number: the top 1% spans every scored chromosome."""
    rng = np.random.default_rng(7)
    chroms = ["chrA", "chrB", "chrC"]
    td = {c: rng.random(1500) * (i + 1) * 4 for i, c in enumerate(chroms)}
    pd_ = {c: rng.random(1500) * 4 for c in chroms}
    ours_t = eic.dict_to_arr(td, chroms)
    theirs_t = np.array([v for c in chroms for v in td[c]])
    assert np.array_equal(ours_t, theirs_t)

    ours_p = eic.dict_to_arr(pd_, chroms)
    assert eic.mse1obs(ours_t, ours_p) == ref.mse1obs(theirs_t, ours_p)


def test_region_measures_match_on_the_real_pinned_beds(ref) -> None:
    """Not synthetic: the actual 58,721-line GENCODE bed and 63,285-line FANTOM5 bed.

    chr21 at 25 bp is 1,868,399 bins, which is the real scoring geometry. Slow by the standards of
    this file and worth every second — synthetic beds cannot exercise 58k real coordinate rows.
    """
    from candi.bench.annotations import enhancer_annotations, gene_annotations
    rng = np.random.default_rng(11)
    n = 1_868_399                                       # chr21 / 25 bp
    y_true = rng.exponential(1.0, n)
    y_pred = y_true * 0.8 + rng.random(n) * 0.3
    td, pd_ = {"chr21": y_true}, {"chr21": y_pred}
    genes, enh = gene_annotations(), enhancer_annotations()
    ch = ["chr21"]

    assert eic.mseprom(td, pd_, ch, genes) == ref.mseprom(td, pd_, ch, genes)
    assert eic.msegene(td, pd_, ch, genes) == ref.msegene(td, pd_, ch, genes)
    assert eic.mseenh(td, pd_, ch, enh) == ref.mseenh(td, pd_, ch, enh)


def test_the_vendored_fixture_is_still_upstreams_exact_bytes() -> None:
    """The fixture's value is that it is unmodified. Assert it, or it silently stops being so."""
    import hashlib
    raw = VENDORED.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert blob == UPSTREAM_BLOB_SHA1, (
        f"tests/fixtures/encode_score_metrics_vendored.py is no longer upstream's bytes "
        f"({blob} != {UPSTREAM_BLOB_SHA1}). It is a frozen fixture; stub imports in "
        f"tests/fixtures/reference.py instead of editing it."
    )


def test_an_empty_selection_fails_two_different_ways_and_both_match(ref) -> None:
    """Quirk 8. An empty result is not one behaviour upstream, it is two, and they must both match.

    `sse` starts as a Python float and is promoted to numpy the first time any slice — empty or not —
    is summed into it. So an annotation set that selects zero bins yields `np.float64(0)/0` = **nan**,
    while an annotation set that matches no line at all leaves `sse` a Python float and yields a
    **ZeroDivisionError**. Same empty answer, different failure, decided by whether the loop body ran.
    """
    y = np.linspace(0.0, 1.0, 200)
    td, pd_ = {"chrT": y}, {"chrT": y * 2}

    # (a) lines match the chromosome, but the promoter [200:280] is off the end of a 200-bin array
    off = _bed_gene([(0, 4975, "-")])
    with np.errstate(invalid="ignore"):
        ours, theirs = eic.mseprom(td, pd_, CHROMS, off), ref.mseprom(td, pd_, CHROMS, off)
    assert np.isnan(ours) and np.isnan(theirs)

    # (b) no line matches the chromosome at all -- the loop body never runs
    other = _bed_gene([(1000, 2000, "+")])           # these are chrT lines; we score chrOTHER
    td2, pd2 = {"chrOTHER": y}, {"chrOTHER": y * 2}
    with pytest.raises(ZeroDivisionError):
        eic.msegene(td2, pd2, ["chrOTHER"], other)
    with pytest.raises(ZeroDivisionError):
        ref.msegene(td2, pd2, ["chrOTHER"], other)


def test_rank_direction_matches_the_reference_table(ref) -> None:
    """`ranking.py` inverts the sign for DESCENDING measures; a mismatch would silently flip a rank."""
    assert eic.RANK_DIRECTION == ref.RANK_METHOD_FOR_EACH_METRIC
    assert set(eic.MEASURES) == set(ref.Score._fields)
