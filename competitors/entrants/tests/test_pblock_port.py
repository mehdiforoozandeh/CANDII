"""The port in `pblock_bigwig.py` must agree with `candi.bench.partitions` bit-for-bit.

This test is deliberately NOT under `tests/`. `tests/` is the candi suite and must stay free of any
dependency on `competitors/`; this file imports both sides, so it lives on the competitors side of
the boundary and runs on its own:

    cd competitors/entrants && PYTHONPATH=../../src pytest tests/ -q

The port exists because the Fir scorer environment has numpy/scipy/pyBigWig and no torch, so it
cannot `import candi`. Two copies of a definition is a maintenance hazard, and this test is the
whole mitigation: identical input, identical output, to the bit. Tolerance is exact equality rather
than `approx` on purpose -- the port is a transcription, so any float difference at all means the
arithmetic diverged and is a defect, not rounding.
"""
from __future__ import annotations

import numpy as np
import pytest

from candi.bench import partitions as REF

import pblock_bigwig as PORT


def _assert_same(a, b, path="root"):
    """Recursive exact comparison, with nan == nan (a nan is a meaningful value in these dicts)."""
    assert type(a) is type(b), f"{path}: type {type(a)} != {type(b)}"
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))} differ"
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"{path}: {a} != {b}"
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.fixture
def rng():
    return np.random.default_rng(20260825)


def _synthetic(rng, n=4000):
    """A track with the shape the P-block cares about: heavy background, a sparse strong tail.

    Lognormal background puts most bins below the first strength edge (1e-1) so the underflow bin is
    exercised, and the injected spikes reach past the last edge (10^2.5 = 316) so the overflow bin is
    too. A uniform draw would leave both empty and the test would pass without ever checking them.
    The spike parameters put roughly a quarter of the spikes above 316, which
    `test_accuracy_by_strength_covers_underflow_and_overflow` asserts rather than assumes.
    """
    sig = rng.lognormal(mean=-2.0, sigma=1.8, size=n)
    spikes = rng.choice(n, size=n // 50, replace=False)
    sig[spikes] += rng.lognormal(mean=5.0, sigma=1.2, size=spikes.size)
    pred = np.abs(sig * rng.normal(1.0, 0.4, size=n) + rng.normal(0, 0.3, size=n))
    peaks = rng.random(n) < 0.03
    return sig, pred, peaks


# ---------------------------------------------------------------------------
# the individual functions
# ---------------------------------------------------------------------------

def test_strength_bin_edges_identical():
    np.testing.assert_array_equal(PORT.strength_bin_edges(), REF.strength_bin_edges())
    assert len(PORT.strength_bin_edges()) == 36        # 35 bins, 36 edges
    assert PORT.BINARISE_THRESHOLD == REF.BINARISE_THRESHOLD


@pytest.mark.parametrize("bin_by", ["obs", "imp"])
def test_accuracy_by_strength_identical(rng, bin_by):
    sig, pred, peaks = _synthetic(rng)
    _assert_same(PORT.accuracy_by_strength(sig, peaks, pred, bin_by=bin_by),
                 REF.accuracy_by_strength(sig, peaks, pred, bin_by=bin_by))


def test_accuracy_by_strength_covers_underflow_and_overflow(rng):
    """The fixture must actually occupy the tail bins, or the parity test above proves little."""
    sig, pred, peaks = _synthetic(rng)
    got = PORT.accuracy_by_strength(sig, peaks, pred, bin_by="obs")
    assert got["n"][0] > 0, "underflow bin empty -- fixture no longer exercises it"
    assert got["n"][-1] > 0, "overflow bin empty -- fixture no longer exercises it"


def test_peak_regions_identical(rng):
    b = rng.random(3000) < 0.05
    assert PORT.peak_regions(b) == REF.peak_regions(b)
    # a run at each array edge, and a one-bin run that both sides must drop
    hand = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert PORT.peak_regions(hand) == REF.peak_regions(hand) == [(0, 2), (5, 8)]
    assert PORT.peak_regions(np.zeros(10, dtype=bool)) == REF.peak_regions(np.zeros(10, dtype=bool))


def test_region_correlation_identical(rng):
    sig, pred, _ = _synthetic(rng)
    regions = [(0, 100), (250, 400), (1000, 1002), (3900, 4000)]
    _assert_same(PORT.region_correlation(sig, pred, regions),
                 REF.region_correlation(sig, pred, regions))


def test_region_correlation_constant_window_is_undefined_on_both_sides():
    """A flat window must land in `n_undefined`, not be scored 0 -- on both implementations."""
    t = np.concatenate([np.full(50, 3.0), np.arange(50, dtype=float)])
    p = np.concatenate([np.full(50, 7.0), np.arange(50, dtype=float) * 2])
    regions = [(0, 50), (50, 100)]
    got, ref = (PORT.region_correlation(t, p, regions), REF.region_correlation(t, p, regions))
    _assert_same(got, ref)
    assert got["n_undefined"] == 1 and got["n_scored"] == 1


def test_promoter_windows_identical():
    genes = [
        "chr21 5000000 5010000 ENSG1 0 +",
        "chr21 5100000 5140000 ENSG2 0 -",
        "chr21 1000 2000 ENSG3 0 +",          # within 80 bins of the start -> clipped to 0
        "chr22 5000000 5010000 ENSG4 0 +",    # wrong chromosome -> dropped
    ]
    assert (PORT.promoter_windows(genes, "chr21", 300_000)
            == REF.promoter_windows(genes, "chr21", 300_000))
    # and the array-length clip must bite identically on a short array
    assert PORT.promoter_windows(genes, "chr21", 200_100) == REF.promoter_windows(
        genes, "chr21", 200_100)


def test_specificity_and_pr_identical(rng):
    mat = rng.random((12, 2000)) < 0.08
    spec_port, spec_ref = PORT.specificity_scores(mat), REF.specificity_scores(mat)
    np.testing.assert_array_equal(spec_port, spec_ref)
    truth = mat[0]
    call = rng.random(2000) < 0.09
    _assert_same(PORT.precision_recall_by_specificity(spec_port, truth, call),
                 REF.precision_recall_by_specificity(spec_ref, truth, call))


# ---------------------------------------------------------------------------
# the whole block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("assay", ["H3K4me3", "H3K27me3", "DNase-seq"])
def test_partition_suite_identical(rng, assay):
    sig, pred, peaks = _synthetic(rng)
    genes = ["chr21 %d %d G%d 0 %s" % (s, s + 8000, i, "+" if i % 2 else "-")
             for i, s in enumerate(range(10_000, 90_000, 3_000))]
    _assert_same(
        PORT.partition_suite(sig, pred, peaks, assay=assay, chrom="chr21",
                             gene_annotations=genes),
        REF.partition_suite(sig, pred, peaks, assay=assay, chrom="chr21",
                            gene_annotations=genes))


def test_multichrom_matches_harness_offset_construction(rng):
    """`partition_suite_multichrom` must equal the per-chromosome-shift construction in
    `harness._p_block`: windows built per chromosome, then offset into the concatenation.

    The harness itself is not importable without a store, so the construction is rebuilt here from
    the reference module's own primitives -- which is what the harness calls.
    """
    genes = ["chr21 %d %d G%d 0 +" % (s, s + 4000, i)
             for i, s in enumerate(range(10_000, 60_000, 2_500))]
    genes += ["chr22 %d %d H%d 0 -" % (s, s + 4000, i)
              for i, s in enumerate(range(10_000, 60_000, 2_500))]
    chroms = ["chr21", "chr22"]
    truth, pred = {}, {}
    for c in chroms:
        s, p, _ = _synthetic(rng, n=3000)
        truth[c], pred[c] = s, p

    got = PORT.partition_suite_multichrom(truth, pred, chroms, assay="H3K4me3",
                                          gene_annotations=genes)

    sig = np.concatenate([truth[c] for c in chroms])
    prd = np.concatenate([pred[c] for c in chroms])
    pk = sig >= REF.BINARISE_THRESHOLD
    offset, acc = {}, 0
    for c in chroms:
        offset[c] = acc
        acc += len(truth[c])
    wins = [(lo + offset[c], hi + offset[c]) for c in chroms
            for lo, hi in REF.promoter_windows(genes, c, len(truth[c]))]

    _assert_same(got["acc_by_obs_strength"],
                 REF.accuracy_by_strength(sig, pk, prd, bin_by="obs"))
    _assert_same(got["acc_by_imp_strength"],
                 REF.accuracy_by_strength(sig, pk, prd, bin_by="imp"))
    _assert_same(got["prom_corr_h3k4me3"], REF.region_correlation(sig, prd, wins))
    assert got["prom_corr_h3k4me3"]["n_regions"] > 0, "no promoter windows -- fixture is inert"


def test_multichrom_records_its_dataset3_caveats(rng):
    """The two Dataset-3 deviations must be stamped in the output, not left to a reader's memory."""
    truth = {"chr21": _synthetic(rng, 2000)[0]}
    pred = {"chr21": _synthetic(rng, 2000)[1]}
    got = PORT.partition_suite_multichrom(truth, pred, ["chr21"], assay="H3K27me3")
    assert got["truth_binarisation"] == "signal>=2"
    assert got["blacklist_deleted"] is False
    assert "peak_shape_corr_dnase" not in got


def test_multichrom_chrom_set_changes_the_number(rng):
    """Strength bins are the panel's, not one chromosome's -- so adding a chromosome must move the
    macro accuracy. If it does not, the concatenation is not happening."""
    truth, pred = {}, {}
    for c in ("chr21", "chr22"):
        s, p, _ = _synthetic(rng, n=3000)
        truth[c], pred[c] = s, p
    one = PORT.partition_suite_multichrom(truth, pred, ["chr21"], assay="H3K27me3")
    two = PORT.partition_suite_multichrom(truth, pred, ["chr21", "chr22"], assay="H3K27me3")
    assert (one["acc_by_obs_strength"]["macro_accuracy"]
            != two["acc_by_obs_strength"]["macro_accuracy"])
