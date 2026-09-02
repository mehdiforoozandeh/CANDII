"""t49 — the naive baseline suite (`RIVALS_PLAN.md` §5), against hand arithmetic and a real store.

Two gates, both named in §5.5.

**The fixture gate.** Three contributors are built by hand, at three different sequencing depths, and
every number the suite emits for them — `mu`, `n`, `signal_mu`, `signal_sigma`, `peak_score` — is
compared against a value worked out on paper in the test itself. Not against a second
implementation, and not against a recorded blob: a baseline whose arithmetic is only checked against
its own previous output can drift a factor of two and stay green forever.

**The depth gate.** `tests/test_reference.py::test_L3_reference_is_depth_free` doubles one
contributor's sequencing depth AND its counts and requires the average not to move. The same
property is asserted here twice: once on the arithmetic directly, and once end to end on the store
path, by building two real stores that differ only in that doubling and diffing the npz files the
generator writes.

The rest of the file is the fairness rules — the exclusion rule, the target rule, the depth centre —
each checked against the thing it has to agree with (`bench.harness`), because the generator
re-derives them rather than importing them and a copy that silently disagrees is the one defect that
would make every baseline number wrong in the same direction.
"""
from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from candi.bench.external import _expected, score_external, track_dirname
from candi.bench.harness import Pair, open_source
from candi.store import layout as L
from candi.store.reader import CorpusStore

from competitors.baselines import generate as Gen
from competitors.baselines import heads as Hd
from competitors.baselines import leaderboard as LB
from competitors.baselines.generate import (
    METHODS, REGIME_DEPENDENT, REGIME_INDEPENDENT, Panel, RegimeIdentityError, available,
    cell_type, depth_center, generate, log2_depth, similarity_table, top_k,
)
from tests.test_store_reader import RES, make_store
from tests.test_store_regime import regime_dict

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

#: Three cells carry the imputed assay and two do not — the layout §5 needs. `T_aa`/`T_ee` are the
#: prompt cells and lack H3K4me3, so `(T_aa, V_aa)` and `(T_ee, V_ee)` pose imputation; `T_bb`,
#: `T_cc`, `T_dd` are the contributors, which makes `k = 3` and gives `ddof=1` something to divide by.
TRACKS = {
    "T_aa": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "T_bb": ("ATAC-seq", "DNase-seq", "H3K4me3", L.CONTROL_TRACK),
    "T_cc": ("ATAC-seq", "DNase-seq", "H3K4me3", L.CONTROL_TRACK),
    "T_dd": ("ATAC-seq", "DNase-seq", "H3K4me3", L.CONTROL_TRACK),
    "T_ee": ("ATAC-seq", "DNase-seq", L.CONTROL_TRACK),
    "V_aa": ("ATAC-seq", "H3K4me3"),
    "V_ee": ("ATAC-seq", "H3K4me3"),
}
TRAIN = ["T_aa", "T_bb", "T_cc", "T_dd", "T_ee"]
EVAL = ["V_aa", "V_ee"]
PAIRS = [["T_aa", "V_aa"], ["T_ee", "V_ee"]]
TARGET = "H3K4me3"


# ---------------------------------------------------------------------------
# §5.5 gate one — three contributors, arithmetic done on paper
# ---------------------------------------------------------------------------
#
# depth_center = 20. Three contributors at log2 depths 20, 21, 19, so `reference.py`'s
# `2 ** (depth_center - d)` rescales their raw counts by 1, 1/2 and 2 respectively:
#
#     raw       scale     at depth_center
#     [4, 0, 10]   1      [4, 0, 10]
#     [8, 0,  4]   1/2    [4, 0,  2]
#     [2, 0,  3]   2      [4, 0,  6]
#
# The target track sits at log2 depth 21, so `s = 2 ** (21 - 20) = 2`.
#
#   bin 0: m = 4, v = 0             -> mu = 8,     V = 0   -> V !> mu -> n = POISSON_N
#   bin 1: m = 0, v = 0             -> mu = MU_FLOOR (a hard zero has no valid NB), n = POISSON_N
#   bin 2: m = 6, v = ((10-6)^2 + (2-6)^2 + (6-6)^2) / 2 = 16
#                                   -> mu = 12,    V = 4*16 = 64 > 12 -> n = 144 / 52
DC = 20.0
RAW = np.array([[4.0, 0.0, 10.0], [8.0, 0.0, 4.0], [2.0, 0.0, 3.0]])
DEPTHS = np.array([20.0, 21.0, 19.0])
D_T = 21.0

PVAL = np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 3.0], [3.0, 2.0, 6.0]])
PEAKS = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_fixture_counts_are_the_hand_computed_moment_matched_nb():
    x = Hd.normalize_to_center(RAW, DEPTHS, DC)
    assert np.array_equal(x, np.array([[4.0, 0.0, 10.0], [4.0, 0.0, 2.0], [4.0, 0.0, 6.0]])), \
        "the depth rescale is not reference.py's 2**(depth_center - d)"

    mu, n = Hd.moment_matched_nb(x, D_T, DC)
    assert mu == pytest.approx([8.0, Hd.MU_FLOOR, 12.0], rel=0, abs=0)
    assert n == pytest.approx([Hd.POISSON_N, Hd.POISSON_N, 144.0 / 52.0], rel=1e-12)


def test_fixture_pval_head_is_the_plain_mean_and_the_cross_cell_std():
    assert Hd.plain_mean(PVAL) == pytest.approx([2.0, 2.0, 3.0], rel=0, abs=0)
    # bin 0: values 1,2,3 -> var = 1        bin 1: all equal -> 0, floored
    # bin 2: values 0,3,6 -> var = 9
    assert Hd.cross_cell_sigma(PVAL) == pytest.approx([1.0, Hd.SIGMA_FLOOR, 3.0], rel=1e-12)


def test_fixture_arcsinh_variant_is_sinh_of_the_mean_arcsinh():
    got = Hd.arcsinh_mean(PVAL)
    want = [math.sinh((math.asinh(1) + math.asinh(2) + math.asinh(3)) / 3.0),
            2.0,
            math.sinh((math.asinh(0) + math.asinh(3) + math.asinh(6)) / 3.0)]
    assert got == pytest.approx(want, rel=1e-12)
    assert got[0] != pytest.approx(2.0), \
        "the arcsinh variant must NOT reproduce the plain mean, or it is not a variant"


def test_fixture_peak_score_is_the_contributor_fraction():
    assert Hd.peak_fraction(PEAKS) == pytest.approx([2.0 / 3.0, 0.0, 2.0 / 3.0], rel=1e-12)


def test_one_contributor_takes_the_poisson_floor_and_offers_no_sigma():
    """§5.1 — `v` is undefined at k = 1, so the NB cannot be over-dispersed and says so."""
    mu, n = Hd.moment_matched_nb(RAW[:1], DC, DC)
    assert np.array_equal(mu, np.maximum(RAW[0], Hd.MU_FLOOR))
    assert np.all(n == Hd.POISSON_N)
    assert Hd.cross_cell_sigma(PVAL[:1]) is None, \
        "a single contributor measures no cross-cell spread; a zero sigma would claim certainty"


def test_the_preregistered_poisson_floor_is_scoreable_by_candi_bench():
    """t49's finding, closed by t56 — pinned here so a regression reopens it loudly.

    §5.1 pre-registers `n = 1e6` for the Poisson floor. `candi.metrics.nb_crps` used to evaluate
    the Gini mean difference as `hyp2f1(0.5, 1 - n, 2, w)`, which returns NaN above roughly
    `n = 1e4` at every µ — one floored bin NaN'd the whole track's `crps`, `macro_mean` dropped the
    key, and the count arm silently lost its distributional tier. Since t56 (PR #26), `nb_crps`
    scores `n > N_GINI_HYP2F1_MAX` in the sd-standardized Poisson limit, so the floor is a number
    rather than an absence.

    Finiteness alone would be satisfiable by garbage, so the value is refereed: at `n = 1e6` the NB
    is Poisson(µ) to a part in ~2e4, and the Poisson CRPS is a finite pmf sum that touches no
    `candi.metrics` code.
    """
    from scipy.stats import poisson

    from candi.bench.distributional import p_from_mu
    from candi.metrics import nb_crps

    mu = np.array([50.0])
    y = np.array([50.0])
    ok = nb_crps(np.array([1e4]), p_from_mu(np.array([1e4]), mu), y)[0]
    floor = nb_crps(np.array([Hd.POISSON_N]), p_from_mu(np.array([Hd.POISSON_N]), mu), y)[0]
    assert np.isfinite(ok), "the exact hyp2f1 branch broke below the switch — a deeper defect"
    assert np.isfinite(floor), (
        "the pre-registered Poisson floor is unscoreable again — the t56 large-dispersion branch "
        "of nb_crps regressed. Fix it in candi.metrics; do not re-cap `--poisson-n` on the runs.")
    x = np.arange(int(50.0 + 12.0 * math.sqrt(50.0) + 60.0) + 1)
    poisson_ref = float(((poisson.cdf(x, 50.0) - (x >= 50.0)) ** 2).sum())
    assert floor == pytest.approx(poisson_ref, rel=1e-3), \
        "finite but wrong at the floor — the Poisson-limit branch no longer scores the NB's limit"


def test_the_nb_is_never_under_dispersed():
    """The pre-registered floor (§5.1): cross-cell agreement bottoms out at Poisson, never below."""
    tight = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.001]])
    _, n = Hd.moment_matched_nb(tight, DC, DC)
    assert np.all(n == Hd.POISSON_N)


def test_n_never_exceeds_the_poisson_floor_from_either_branch():
    """The clip in `nb_from_moments`, and the defect it closes.

    As `V` falls towards `mu` from above, `mu**2 / (V - mu)` diverges — so a bin whose contributors
    very nearly agree returns an enormous `n` through the MOMENT-MATCHED branch while a bin whose
    contributors agree exactly returns `poisson_n` through the floor. Both are the statement "this is
    Poisson"; without the clip they are different numbers, and the big one is unscoreable.

    Not cosmetic: uncapped, 26 of the 45 P1 tracks came back with a NaN CRPS at `poisson_n = 1e4`,
    and `beats_marginal` counted every one of them as a loss.
    """
    nearly = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.000_01]])
    for floor in (1e4, Hd.POISSON_N):
        _, n = Hd.moment_matched_nb(nearly, DC, DC, poisson_n=floor)
        assert np.all(n <= floor), f"n above the declared Poisson floor {floor}: {n.max()}"
        # The two branches must MEET: the exact-agreement bin and the near-agreement bin report the
        # same limit, which is what makes the cap continuous rather than an arbitrary clamp.
        assert n[0] == floor and n[1] == floor


def test_the_nb_crps_ceiling_is_gone_at_every_mu():
    """The ceiling that used to size the Poisson floor, re-measured after t56: there is none.

    Pre-t56, `nb_crps` was finite up to `n = 1e4` and NaN from `n = 3e4` at every `mu` — which is
    why the generator capped `n` and why §5.1's 1e6 was unreachable. The t56 branch scores
    `n > N_GINI_HYP2F1_MAX` in the sd-standardized Poisson limit, so every `n` the generator can
    emit — both sides of the switch and the pre-registered floor itself — must now be finite at
    every `mu`. A NaN anywhere on this grid silently re-drops the count arm's CRPS tier.
    """
    from candi.bench.distributional import p_from_mu
    from candi.metrics import nb_crps

    for mu in (0.01, 1.0, 100.0, 1e5):
        m, y = np.array([mu]), np.array([float(round(mu))])
        for n in (1e4, 3e4, Hd.POISSON_N):
            v = nb_crps(np.array([n]), p_from_mu(np.array([n]), m), y)[0]
            assert np.isfinite(v), \
                f"nb_crps is NaN again at n={n}, mu={mu} — the t56 large-dispersion branch regressed"


def test_L3_arithmetic_is_depth_free():
    """`reference.py`'s L3, on the generator's arithmetic: double a depth AND its counts, nothing moves."""
    raw2 = RAW.copy()
    raw2[1] *= 2.0
    d2 = DEPTHS.copy()
    d2[1] += 1.0                                        # log2 depth + 1 == sequenced 2x deeper
    a = Hd.moment_matched_nb(Hd.normalize_to_center(RAW, DEPTHS, DC), D_T, DC)
    b = Hd.moment_matched_nb(Hd.normalize_to_center(raw2, d2, DC), D_T, DC)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]), \
        "the count baseline moved when a contributor's depth changed — it is NOT depth-free"


def test_depth_center_cancels_out_of_every_count_prediction():
    """Documented in `heads.py`: `mu = 2**d_t * mean(c_j * 2**-d_j)` whatever the centre is.

    Worth a test rather than a comment, because it is what makes a baseline generated on one pool
    comparable to one generated on another.
    """
    a = Hd.moment_matched_nb(Hd.normalize_to_center(RAW, DEPTHS, DC), D_T, DC)
    b = Hd.moment_matched_nb(Hd.normalize_to_center(RAW, DEPTHS, 3.5), D_T, 3.5)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1], rtol=1e-9)


# ---------------------------------------------------------------------------
# a real store
# ---------------------------------------------------------------------------

def _set_depth(corpus_root: Path, biosample: str, assay: str, depth: float) -> None:
    """Rewrite one track's `depth` in the store manifest. The manifest is what the loader reads."""
    p = corpus_root / "manifest.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    for rec in obj["biosamples"][biosample]["tracks"]:
        if rec["assay"] == assay:
            rec["depth"] = depth
    p.write_text(json.dumps(obj), encoding="utf-8")


def _scale_counts(corpus_root: Path, biosample: str, assay: str, factor: int) -> None:
    """Multiply one assay's counts column by `factor`, in place, on every chromosome."""
    path = L.kind_path(corpus_root, biosample, "counts")
    with h5py.File(str(path), "r+") as f:
        col = list(L.read_root_attrs(f)[L.ATTR_TRACKS]).index(assay)
        for chrom in [k for k in f.keys()]:
            ds = f[chrom]
            ds[:, col] = ds[:, col] * factor


def _structure_pval(corpus_root: Path, assay: str, cells, seed: int = 11) -> None:
    """Give the imputed assay a SHARED per-bin profile plus small per-cell deviation.

    `make_store` draws every track from its own seed, so the cells of its synthetic corpus are
    independent noise — and on independent noise a cross-cell average is genuinely WORSE than one
    constant per assay (the average adds `var/k` to the irreducible variance and buys nothing back).
    §5.5's first sanity anchor is a claim about epigenomes, which share a profile across cell types;
    a fixture with no shared profile cannot express it either way.

    So the pval column is rewritten as `shared(bin) + N(0, 0.4)`, encoded in the file's own codec.
    That is the only assumption the anchor rests on, and it is stated here rather than smuggled in.
    """
    for b in cells:
        path = L.kind_path(corpus_root, b, "pval")
        with h5py.File(str(path), "r+") as f:
            attrs = L.read_root_attrs(f)
            col = list(attrs[L.ATTR_TRACKS]).index(assay)
            scale = int(attrs.get(L.ATTR_SCALE, L.PVAL_SCALE))
            transform = str(attrs.get(L.ATTR_TRANSFORM, L.PVAL_TRANSFORM))
            for chrom in list(f.keys()):
                n = f[chrom].shape[0]
                shared = np.abs(np.random.default_rng(seed + len(chrom) + ord(chrom[-1]))
                                .normal(8.0, 6.0, n))
                # `zlib.crc32`, not `hash` — str hashing is salted per interpreter, so a fixture
                # seeded from it would differ between runs and the anchor test would be flaky.
                v = np.clip(shared + np.random.default_rng(
                    seed + zlib.crc32(b.encode())).normal(0.0, 0.4, n), 0.0, None)
                enc = v if transform == "linear" else np.arcsinh(v)
                f[chrom][:, col] = np.clip(np.round(enc * scale), 0,
                                           L.PVAL_UINT16_MAX).astype(np.uint16)


def _build(tmp: Path, *, deeper: bool = False, chrom_sizes=None) -> tuple:
    """A store + a regime declaring two imputation pairs. `deeper` doubles T_bb's H3K4me3 exposure.

    `chrom_sizes` overrides the two-chromosome default. The D1 identity assertion needs THREE — two
    disjoint training slices and one eval chromosome — because a regime whose train and eval
    chromosomes overlap is refused by `Regime.check_against_store` and would be a fixture that no
    real regime resembles.
    """
    root = make_store(tmp, tracks=TRACKS, chrom_sizes=chrom_sizes)
    # `make_store` gives every track the same depth; a baseline that never sees two depths never
    # exercises the rescale it is built around.
    for i, b in enumerate(TRAIN):
        _set_depth(root, b, "ATAC-seq", 10_000_000 * (i + 1))
        if "H3K4me3" in TRACKS[b]:
            _set_depth(root, b, "H3K4me3", 8_000_000 * (i + 1))
    _set_depth(root, "V_aa", "H3K4me3", 25_000_000)
    _set_depth(root, "V_ee", "H3K4me3", 12_000_000)
    _structure_pval(root, TARGET, [b for b in TRACKS if TARGET in TRACKS[b]])
    if deeper:
        _set_depth(root, "T_bb", "H3K4me3", 2 * 8_000_000 * (TRAIN.index("T_bb") + 1))
        _scale_counts(root, "T_bb", "H3K4me3", 2)
    obj = regime_dict(root, biosamples={"train": TRAIN, "eval": EVAL},
                      kinds=["counts", "peaks", "pval"], eval_pairs=PAIRS)
    rp = tmp / "regime.json"
    rp.write_text(json.dumps(obj), encoding="utf-8")
    return root, rp


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("basestore"))


#: The Poisson floor these fixtures generate at. NOT the pre-registered 1e6 — no longer because
#: `nb_crps` cannot evaluate it (t56 fixed that; `test_the_preregistered_poisson_floor_is_
#: scoreable_by_candi_bench`), but because 1e4 keeps every fixture bin on the exact hyp2f1 branch,
#: bit-identical to the vendored closed form, instead of the t56 Poisson-limit approximation.
#: `test_the_preregistered_floor_keeps_the_count_arm_its_crps_tier` covers the spec value itself.
SCOREABLE_POISSON_N = 1e4


@pytest.fixture(scope="module")
def roots(built, tmp_path_factory):
    _, regime = built
    return generate(regime, tmp_path_factory.mktemp("preds"),
                    poisson_n=SCOREABLE_POISSON_N, progress=False)


# ---------------------------------------------------------------------------
# the rules the generator re-derives must agree with the ones bench enforces
# ---------------------------------------------------------------------------

def test_panel_agrees_with_the_harness_on_targets_and_depth_center(built):
    _, regime = built
    panel = Panel(regime)
    source = open_source(store=regime)
    try:
        for pair in PAIRS:
            want = [source.assays[a] for a in source.targets(Pair(*pair), "impute")]
            assert sorted(panel.targets(tuple(pair))) == sorted(want), \
                f"{pair}: the generator and bench disagree about what is being imputed"
        # float32, not float64: `StoreDataset` keeps `log2(depth)` in a float32 meta array and takes
        # the median of that, so the two agree to float32 and no further. It does not matter — the
        # centre cancels out of every count prediction (`test_depth_center_cancels_out_of_every_
        # count_prediction`) — but a disagreement in the fourth digit would mean a different POOL,
        # which would.
        assert panel.depth_center == pytest.approx(source.depth_center(), rel=1e-6), \
            "the generator's depth centre is not StoreDataset's"
    finally:
        source.close()
        panel.close()


def test_the_exclusion_rule_drops_the_paired_training_cell(built):
    """§5 — `T_aa` never contributes to `V_aa`, for any assay. The whole leakage rule is this."""
    _, regime = built
    panel = Panel(regime)
    try:
        for pair in PAIRS:
            contribs = panel.contributors(tuple(pair), TARGET)
            assert pair[0] not in contribs
            assert not any(cell_type(b) == cell_type(pair[1]) for b in contribs)
            assert all(b.startswith("T_") for b in contribs), "an eval biosample reached a baseline"
            assert sorted(contribs) == ["T_bb", "T_cc", "T_dd"]
    finally:
        panel.close()


def test_cell_type_strips_only_the_split_prefix():
    assert cell_type("T_K562") == "K562" and cell_type("V_K562") == "K562"
    assert cell_type("B_adrenal_gland") == "adrenal_gland"
    assert cell_type("K562") == "K562"


def test_the_roots_cover_exactly_the_declared_tracks(built, roots):
    _, regime = built
    source = open_source(store=regime)
    try:
        want = set(_expected(source))
        assert want == {track_dirname(Pair(*p), TARGET) for p in PAIRS}
        for method, root in roots.items():
            got = {d.name for d in root.iterdir() if d.is_dir()}
            assert got == want, f"{method}: root covers {got}, regime declares {want}"
    finally:
        source.close()


def test_every_method_writes_a_traceable_manifest(roots):
    assert sorted(roots) == sorted(METHODS)
    for method, root in roots.items():
        m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert m["method"] == method
        assert m["generated_by"] == "competitors/baselines/generate.py"
        assert m["notes"] and m["arms"]
        assert set(m["tracks"]) == {d.name for d in root.iterdir() if d.is_dir()}
        for row in m["tracks"].values():
            assert row["n_contributors"] >= 1 and row["n_eligible"] == 3
        # Three ELIGIBLE contributors, so nothing is sparse — including `knn1`, which uses one on
        # purpose. §5's flag is about a thin training split, not about a method's own choice.
        assert m["sparse_assays"] == []
        assert m["poisson_n"] == SCOREABLE_POISSON_N
        assert m["poisson_n_is_preregistered"] is False


def test_the_arrays_each_method_promises_are_the_arrays_on_disk(roots):
    want = {
        "avg": {"mu", "n", "signal_mu", "signal_sigma", "peak_score"},
        "avg-arcsinh": {"signal_mu"},
        "knn1": {"mu", "n", "signal_mu", "peak_score"},          # k = 1: no cross-cell sigma
        "knn5": {"mu", "n", "signal_mu", "signal_sigma", "peak_score"},
        "marginal": {"mu", "n", "signal_mu", "signal_sigma"},    # no peak head
    }
    for method, root in roots.items():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            with np.load(d / "chr2.npz") as z:
                assert set(z.files) == want[method], f"{method}/{d.name}"
                for k in z.files:
                    assert z[k].dtype == np.float32 and z[k].shape == (800,)


def test_knn1_is_bestsingle_and_knn5_falls_back_to_what_exists(roots):
    """k = 1 picks exactly one cell; k = 5 with three eligible contributors takes all three."""
    one = json.loads((roots["knn1"] / "manifest.json").read_text(encoding="utf-8"))
    five = json.loads((roots["knn5"] / "manifest.json").read_text(encoding="utf-8"))
    for name, row in one["tracks"].items():
        assert row["n_contributors"] == 1 and len(row["contributors"]) == 1
        assert row["n_eligible"] == 3
        assert five["tracks"][name]["n_contributors"] == 3
        assert row["contributors"][0] in five["tracks"][name]["contributors"]


def test_the_similarity_table_never_reads_an_eval_chromosome(built, monkeypatch):
    """§5.4 — the kNN ranking is fitted on train chromosomes only. Asserted by watching the reads."""
    _, regime = built
    panel = Panel(regime)
    seen = []
    orig = CorpusStore.__getitem__

    def spy(self, name):
        bs = orig(self, name)
        real = bs.pval

        def pval(chrom, start, end=None, assays=None):
            seen.append((name, chrom))
            return real(chrom, start, end, assays)
        bs.pval = pval
        return bs

    monkeypatch.setattr(CorpusStore, "__getitem__", spy)
    try:
        similarity_table(panel, panel.train)
    finally:
        panel.close()
    assert seen, "the spy caught nothing — the test is not testing anything"
    assert {c for _, c in seen} == {"chr1"}, f"eval chromosome read while ranking: {set(seen)}"
    assert all(b.startswith("T_") for b, _ in seen), "an eval biosample reached the kNN table"


def test_top_k_is_deterministic_under_ties():
    sim = {("T_aa", "T_bb"): 0.5, ("T_aa", "T_cc"): 0.5}
    assert top_k(sim, "T_aa", ["T_dd", "T_cc", "T_bb"], 2) == ["T_bb", "T_cc"]
    assert top_k(sim, "T_aa", ["T_dd", "T_cc", "T_bb"], 3)[-1] == "T_dd"   # -inf sorts last


# ---------------------------------------------------------------------------
# D1 — which of the five collapse to one run, asserted rather than argued
# ---------------------------------------------------------------------------
#
# `BENCHMARK_DESIGN.md` §12.2 ruled all five naive baselines run once rather than once per regime,
# "because their fit is regime-independent", and said the first implementation of `avg` asserts it.
# It did not: there was no test anywhere that predicted under two regimes and compared. These are
# that assertion, and they also pin the half §12.2 got wrong — `knn1`, `knn5` and `marginal` fit on
# the regime's training chromosomes and do NOT collapse.
#
# The fixture is one store with three chromosomes and two regimes over it that differ in NOTHING
# but the training slice: chr1 against chr3, both predicting chr2, the same declared pairs.

_ASSERT_SIZES = {"chr1": 400 * RES + 7, "chr2": 300 * RES + 3, "chr3": 500 * RES + 11}


@pytest.fixture(scope="module")
def two_regimes(tmp_path_factory):
    """`(store, regime_A, regime_B)` — same store, same pairs, disjoint training chromosomes."""
    tmp = tmp_path_factory.mktemp("tworegime")
    root, a = _build(tmp, chrom_sizes=_ASSERT_SIZES)
    obj = json.loads(a.read_text(encoding="utf-8"))
    assert obj["train_chroms"] == ["chr1"] and obj["eval_chroms"] == ["chr2"]
    obj["train_chroms"] = ["chr3"]
    b = tmp / "regime_b.json"
    b.write_text(json.dumps(obj), encoding="utf-8")
    return root, a, b


def test_the_collapsed_methods_are_identical_under_two_training_slices(two_regimes, tmp_path):
    """§12.2's claim, for the two methods it is true of — and the manifest records that it was run."""
    _, a, b = two_regimes
    out = tmp_path / "preds"
    roots = generate(a, out, methods=list(REGIME_INDEPENDENT), poisson_n=SCOREABLE_POISSON_N,
                     progress=False, assert_against=b)
    assert sorted(roots) == sorted(REGIME_INDEPENDENT)
    for m, root in roots.items():
        got = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert got["regime_independent"] == {"asserted_against": "regime_b.json",
                                             "chrom": "chr2", "identical": True}, m
        assert got["tracks"], f"{m} wrote a manifest with no track — the comparison saw nothing"


def test_the_assertion_fails_when_the_training_slice_is_made_to_matter(two_regimes, tmp_path):
    """The other half of D1: `marginal` pools over `train_chroms`, so the two regimes disagree.

    This is the control. An identity check that cannot fail licenses nothing, and the failure is
    produced by a real regime-dependent method rather than by a doctored one.
    """
    _, a, b = two_regimes
    out = tmp_path / "preds"
    with pytest.raises(RegimeIdentityError) as exc:
        generate(a, out, methods=["marginal"], poisson_n=SCOREABLE_POISSON_N,
                 progress=False, assert_against=b)
    assert "marginal/" in str(exc.value) and "differ" in str(exc.value)
    assert not (out / "marginal" / "manifest.json").exists(), \
        "a failed assertion left a manifest behind that a reader could quote"


def test_the_assertion_refuses_a_panel_the_other_regime_does_not_declare(two_regimes, tmp_path):
    """Identical output for two DIFFERENT panels is an identity between two different objects."""
    _, a, b = two_regimes
    obj = json.loads(b.read_text(encoding="utf-8"))
    obj["eval_pairs"] = [PAIRS[0]]
    narrow = tmp_path / "regime_narrow.json"
    narrow.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="SAME panel"):
        generate(a, tmp_path / "preds", methods=["avg"], poisson_n=SCOREABLE_POISSON_N,
                 progress=False, assert_against=narrow)


def test_the_cli_never_asserts_regime_independence_for_a_fitted_method(two_regimes, tmp_path):
    """Exit 2, and before any store is opened: the flag is not offered for the fitted three."""
    _, a, b = two_regimes
    for m in REGIME_DEPENDENT:
        rc = Gen.main(["--store", str(a), "--out", str(tmp_path / m), "--methods", m,
                       "--assert-regime-independent", str(b), "--quiet"])
        assert rc == 2, m
        assert not (tmp_path / m).exists(), f"{m} was generated before the flag was refused"


def test_a_difference_leaves_the_cli_with_exit_5(two_regimes, tmp_path, monkeypatch):
    """The exit code the launchers branch on. `marginal` is let past the CLI guard to produce one."""
    _, a, b = two_regimes
    monkeypatch.setattr(Gen, "REGIME_INDEPENDENT", ("avg", "avg-arcsinh", "marginal"))
    rc = Gen.main(["--store", str(a), "--out", str(tmp_path / "preds"), "--methods", "marginal",
                   "--assert-regime-independent", str(b), "--poisson-n",
                   str(SCOREABLE_POISSON_N), "--quiet"])
    assert rc == 5
    assert Gen.main(["--store", str(a), "--out", str(tmp_path / "ok"), "--methods", "avg",
                     "--assert-regime-independent", str(b), "--poisson-n",
                     str(SCOREABLE_POISSON_N), "--quiet"]) == 0


# ---------------------------------------------------------------------------
# §5.5 gate two — the L3 depth-free property, end to end on the store path
# ---------------------------------------------------------------------------

def test_L3_generator_is_depth_free(tmp_path):
    """Adapted from `tests/test_reference.py::test_L3_reference_is_depth_free`.

    Two stores identical but for one contributor sequenced twice as deep, with its counts doubled to
    match. Every count array the generator writes must be unchanged — otherwise the baseline carries
    a depth mixture and its `mu` means "how deep were the contributors" rather than "how much signal
    is here".
    """
    _, ra = _build(tmp_path / "a")
    _, rb = _build(tmp_path / "b", deeper=True)
    a = generate(ra, tmp_path / "pa", methods=["avg"], progress=False)["avg"]
    b = generate(rb, tmp_path / "pb", methods=["avg"], progress=False)["avg"]
    names = sorted(d.name for d in a.iterdir() if d.is_dir())
    assert names
    for name in names:
        with np.load(a / name / "chr2.npz") as za, np.load(b / name / "chr2.npz") as zb:
            for k in ("mu", "n"):
                assert np.allclose(za[k], zb[k], rtol=1e-5), \
                    f"{name}/{k} moved when a contributor's depth doubled — NOT depth-free"


def test_the_baseline_tracks_the_targets_depth(tmp_path, built, roots):
    """The other half of §5.1: `mu` is quoted at the TARGET track's exposure, not at the centre.

    `V_aa` is declared about 2x deeper than `V_ee` for the same assay and the same contributors, so
    its predicted mean must be about 2x larger. Without this the count arm would score every target
    at a common exposure the truth was never measured at.
    """
    root, regime = built
    panel = Panel(regime)
    try:
        ratio = 2.0 ** (log2_depth(panel.corpus, "V_aa", TARGET)
                        - log2_depth(panel.corpus, "V_ee", TARGET))
    finally:
        panel.close()
    with np.load(roots["avg"] / f"T_aa__V_aa__{TARGET}" / "chr2.npz") as za, \
            np.load(roots["avg"] / f"T_ee__V_ee__{TARGET}" / "chr2.npz") as ze:
        live = za["mu"] > Hd.MU_FLOOR
        assert live.any()
        assert np.allclose(za["mu"][live] / ze["mu"][live], ratio, rtol=1e-4)


# ---------------------------------------------------------------------------
# §5.5 gate three — it goes through the external entry, end to end
# ---------------------------------------------------------------------------

def test_every_root_scores_through_the_external_entry(built, roots):
    _, regime = built
    source = open_source(store=regime)
    try:
        for method, root in roots.items():
            res = score_external(source, root, c_index_pairs=2_000)
            assert res["provenance"]["method"] == method
            assert res["provenance"]["missing_tracks"] == []
            assert len(res["tracks"]) == len(PAIRS)
            arms = {a for t in res["per_track"].values() for a in t}
            if method == "avg-arcsinh":
                assert arms == {"pval"}, "a pval-only variant must not grow a count arm"
                assert "crps" not in res["macro"]["pval"], \
                    "a point-only track earned gauss_suite keys without a sigma (§4.2)"
            else:
                assert arms == {"count", "pval"}
                assert "crps" in res["macro"]["count"]
            if method in ("avg", "knn1", "knn5"):
                assert all(t["count"].get("bernoulli_nll") is not None
                           for t in res["per_track"].values()), \
                    "the fraction-of-contributors peak score is a real peak head (§5.3)"
    finally:
        source.close()


def test_the_preregistered_floor_keeps_the_count_arm_its_crps_tier(built, tmp_path):
    """The unit-level fix above, seen end to end: generate at §5.1's `n = 1e6` and the count arm's
    whole distributional tier stays in the macro roll-up. Pre-t56 this exact run came back with the
    tier ABSENT — the reason the Fir P1 runs passed `--poisson-n`. Once those runs are re-scored at
    the spec value, that workaround can be dropped."""
    _, regime = built
    root = generate(regime, tmp_path / "spec", methods=["avg"], progress=False)["avg"]
    source = open_source(store=regime)
    try:
        res = score_external(source, root, c_index_pairs=2_000)
    finally:
        source.close()
    macro = res["macro"]["count"]
    assert "mse" in macro and "nb_nll" in macro, "the point and loss tiers must be unaffected"
    for k in ("crps", "crps_oracle_scaled", "scale_error"):
        assert k in macro and np.isfinite(macro[k]), \
            f"{k} went absent or NaN at the pre-registered floor — the t56 fix regressed end to end"
    # t56's other half: `beats_marginal` is None only when a side of the comparison is non-finite,
    # so at a now-scoreable floor every track must report the boolean, not the absence.
    for name, t in res["per_track"].items():
        assert t["count"].get("beats_marginal") is not None, \
            f"{name}: beats_marginal went absent — a non-finite CRPS reached nb_suite at the floor"


def test_the_average_beats_the_marginal_on_pval_mse(built, roots):
    """§5.5's first sanity anchor, on the synthetic panel.

    A per-bin cross-cell average must be closer to the truth than one constant per assay. This is
    the same assertion the Fir run makes on the real panel; failing it means the average is not
    carrying per-bin information and the plan says stop rather than tune.
    """
    _, regime = built
    source = open_source(store=regime)
    try:
        avg = score_external(source, roots["avg"], c_index_pairs=2_000)
        marg = score_external(source, roots["marginal"], c_index_pairs=2_000)
    finally:
        source.close()
    assert avg["macro"]["pval"]["mse"] < marg["macro"]["pval"]["mse"]


# ---------------------------------------------------------------------------
# the leaderboard's §6.4 invariants
# ---------------------------------------------------------------------------

def _score_file(tmp_path, source, root, name):
    from candi.bench.cli import jsonable
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(jsonable(score_external(source, root, c_index_pairs=2_000))))
    return p


def test_the_leaderboard_reports_per_assay_first_and_medians_beside_the_pool(built, roots, tmp_path):
    _, regime = built
    source = open_source(store=regime)
    try:
        files = {m: _score_file(tmp_path, source, roots[m], m) for m in ("avg", "marginal")}
    finally:
        source.close()
    board = LB.assemble(files, protocol="P1")

    for name, block in board["methods"].items():
        assert list(block)[0] == "per_assay" and list(block)[1] == "macro", \
            f"{name}: §6.4 wants per-assay rows first, macro second"
        row = block["per_assay"]["count"][TARGET]
        assert row["group"] == "punctate" and row["peak_ranking"] in ("peak_head", "coverage_ranking")
        assert "auprc" in row and "peak_base_rate" in row
        macro = block["macro"]["count"]
        assert "crps" in macro and "crps_oracle_scaled" in macro and "scale_error" in macro
        assert "mse_punctate_median" in macro and "mse_punctate_n_assays" in macro
    assert board["noise_floors"]["macro_crps_target_clustered"] == 0.09
    assert board["sanity"]["avg_beats_marginal_on_macro_pval_mse"]["pass"] is True


def test_the_leaderboard_refuses_a_crps_without_its_companions():
    """§6.4, as clarified by the PI on 2026-08-26: the CRPS companions are PER ARM."""
    with pytest.raises(ValueError, match="§6.4|crps_oracle_scaled"):
        LB._check_companions("t/count", "count", ["crps", "mse"])
    with pytest.raises(ValueError, match="§6.4|peak_base_rate"):
        LB._check_companions("t/pval", "pval", ["auprc"])
    # The Gaussian CRPS has no oracle-scale split to demand — asking for one would want a key
    # `candi.bench` does not compute — so the pval arm takes CALIBRATION companions instead.
    with pytest.raises(ValueError, match="pit_ks|coverage_95"):
        LB._check_companions("t/pval", "pval", ["crps", "peak_base_rate", "auprc"])
    LB._check_companions("t/pval", "pval", ["crps", "pit_ks", "coverage_95"])
    # ...and the count arm is never asked for the pval arm's pair.
    LB._check_companions("t/count", "count", ["crps", "crps_oracle_scaled", "scale_error"])


def test_the_pval_crps_gap_is_marked_unquotable_until_its_floor_exists():
    """PI 2026-08-26 — a pval CRPS is quotable as a value; a between-method GAP on it is not.

    The caveat is stamped onto the macro row rather than left in a caption, and the header says the
    floor is missing rather than staying silent about it. `AGENTS.md` §7.2 supplies count-arm CRPS
    floors and nothing for the pval arm, so silence here would read as "no floor needed".
    """
    scores = {"pval": {"crps": 0.5, "pit_ks": 0.1, "coverage_95": 0.94, "mse": 1.0,
                       "auprc": 0.3, "peak_base_rate": 0.05}}
    obj = {"provenance": {"method": "m", "manifest": {}}, "tracks": ["t"],
           "per_track": {"t": {"pval": {**scores["pval"], "assay": "H3K4me3", "kind": "impute"}}},
           "macro": {"pval": {**scores["pval"], "n_tracks": 1}}, "panel": {}, "ranking": None}
    import json as _json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text(_json.dumps(obj), encoding="utf-8")
        board = LB.assemble({"m": p}, protocol="P1")
    macro = board["methods"]["m"]["macro"]["pval"]
    assert "crps_gap_not_quotable" in macro and "noise floor" in macro["crps_gap_not_quotable"]
    assert "pval_macro_crps" in board["noise_floors_absent"]
    # The count arm has floors, so it carries no such stamp.
    assert board["noise_floors"]["macro_crps_target_clustered"] == 0.09


def _fake_score(tmp: Path, name: str, *, estimator=None, k=None, seed=None) -> Path:
    """One minimal count-arm score json, with or without t56's estimator stamp."""
    count = {"assay": "H3K4me3", "kind": "impute", "crps": 0.44, "crps_oracle_scaled": 0.43,
             "scale_error": 0.01, "auprc": 0.3, "peak_base_rate": 0.05,
             "n_star_log2": -0.4, "crps_oracle_scaled_and_n": 0.42}
    prov = {"method": name, "manifest": {}}
    if estimator:
        prov.update(crps_estimator=estimator, crps_k=k, crps_seed=seed)
    obj = {"provenance": prov, "tracks": ["t"], "per_track": {"t": {"count": count}},
           "macro": {"count": {**{x: v for x, v in count.items() if x not in ("assay", "kind")},
                               "n_tracks": 1}},
           "panel": {}, "ranking": None}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_a_sampled_crps_can_never_be_read_as_an_exact_one(tmp_path):
    """PI 2026-08-26 — t56's estimator is GO for P2 at k=100, so a table may now MIX estimators.

    P2's `avg-arcsinh` has no count arm and was scored before the switch, so a mixed table is a
    legitimate state — but which method used which must be visible on the row, not inferable from
    whether an optional key happens to be present.
    """
    files = {"sampled": _fake_score(tmp_path, "sampled", estimator="fair_sampled", k=100, seed=0),
             "exact": _fake_score(tmp_path, "exact")}
    b = LB.assemble(files, protocol="P2")
    s = b["methods"]["sampled"]
    assert s["macro"]["count"]["crps_estimator"] == "fair_sampled"
    assert s["macro"]["count"]["crps_k"] == 100 and s["macro"]["count"]["crps_seed"] == 0
    assert s["provenance"]["crps_estimator"] == "fair_sampled"
    e = b["methods"]["exact"]
    assert e["provenance"]["crps_estimator"] == "closed_form"
    assert "crps_estimator" not in e["macro"]["count"], "an exact row must not claim an estimator"
    assert b["reporting"]["crps_estimator"] == {"sampled": "fair_sampled", "exact": "closed_form"}


def test_the_two_n_grid_keys_are_flagged_unreliable_on_a_closed_form_row(tmp_path):
    """t56 finding: `oracle_scale` searches `n * 2**k` for k in -4..4, so a track whose fitted n is
    near the Poisson floor is evaluated up to 16x higher — where the closed-form `nb_crps` is NaN.
    `n_star_log2` and `crps_oracle_scaled_and_n` are read off that grid and must not be quoted from
    a closed-form score file. The sampled estimator stays finite there, so its rows are not flagged.
    """
    files = {"exact": _fake_score(tmp_path, "exact"),
             "sampled": _fake_score(tmp_path, "sampled", estimator="fair_sampled", k=100, seed=0)}
    b = LB.assemble(files, protocol="P2")
    ex = b["methods"]["exact"]["macro"]["count"]
    for key in ("n_star_log2", "crps_oracle_scaled_and_n"):
        assert f"{key}_unreliable" in ex, f"{key} not flagged on a closed-form row"
        assert "NaN" in ex[f"{key}_unreliable"]
    sm = b["methods"]["sampled"]["macro"]["count"]
    assert not any(k.endswith("_unreliable") for k in sm), \
        "the sampled estimator stays finite on the n-grid; its rows must not be flagged"


def test_the_reporting_ruling_travels_in_the_header_not_on_an_anchor():
    """A reporting-scoped ruling governs table semantics, so it rides on `reporting` — and must not
    be attached to a §5.5 anchor, where it would read as a verdict on a result."""
    for key in ("crps_companions_are_per_arm", "sampled_crps_for_p2"):
        assert LB.PI_RULINGS[key]["scope"] == "reporting"
        assert LB.PI_RULINGS[key]["date"] == "2026-08-26"
    assert "k=100" in LB.PI_RULINGS["sampled_crps_for_p2"]["ruling"]
    assert "0.000087" in LB.PI_RULINGS["sampled_crps_for_p2"]["ruling"]
    rulings = LB.PI_RULINGS["crps_companions_are_per_arm"]
    assert rulings["scope"] == "reporting" and rulings["date"] == "2026-08-26"
    assert "count-arm (NB) only" in rulings["ruling"]
    assert "pit_ks and coverage_95" in rulings["ruling"]
    anchors = LB.check_anchors({
        "avg": {"macro": {"pval": {"mse": 1.0}, "count": {"beats_marginal": 0.99}}},
        "marginal": {"macro": {"pval": {"mse": 2.0}}},
    })
    for name, block in anchors.items():
        if isinstance(block, dict):
            assert block.get("pi_ruling", {}).get("scope") != "reporting", \
                f"{name} carries a reporting ruling as if it were a verdict on a result"


def test_broad_marks_are_never_folded_into_the_punctate_median():
    assert LB._group("H3K27me3") == "broad" and LB._group("H3K9me3") == "broad"
    assert LB._group("H3K4me3") == "punctate" and LB._group("H3K27ac") == "punctate"
    assert LB._group("DNase-seq") == "accessibility" and LB._group("ATAC-seq") == "accessibility"


def test_a_pi_ruling_travels_with_the_anchor_and_never_flips_its_verdict():
    """The PI's 2026-08-25 ruling on anchor 2, and the one thing a ruling may not do.

    A ruling records what the PI CONCLUDED from a failing anchor. It must not turn `pass: False`
    into `pass: True`, or the anchor stops being a check and becomes a formality — and it must
    survive regeneration, which is why it lives in `PI_RULINGS` rather than in the json.
    """
    methods = {
        "avg": {"macro": {"pval": {"mse": 7.13}, "count": {"beats_marginal": 0.8444}}},
        "marginal": {"macro": {"pval": {"mse": 9.31}}},
    }
    got = LB.check_anchors(methods)
    a2 = got["avg_beats_marginal_near_universal"]
    assert a2["pass"] is False and a2["fraction_of_tracks"] == 0.8444
    assert got["all_pass"] is False, "a ruled-on anchor must not silently pass the whole set"
    assert "k=3" in a2["pi_ruling"]["ruling"] and "median 16" in a2["pi_ruling"]["ruling"]
    assert "declined a post-hoc sparse-threshold change" in a2["pi_ruling"]["ruling"]
    assert got["avg_beats_marginal_on_macro_pval_mse"]["pass"] is True
    assert "pi_ruling" not in got["avg_beats_marginal_on_macro_pval_mse"], \
        "anchor 1 passed on its own; it carries no ruling"


# ---------------------------------------------------------------------------
# the one-way rule (§3)
# ---------------------------------------------------------------------------

def test_candi_never_imports_competitors():
    """§3 — the dependency runs one way. `competitors` reads the store; `candi` knows nothing of it.

    Matched on the import statement, not on the word: `bench.cli` and `bench.external` talk ABOUT
    competitors in prose and in a `--competitors` flag, which is the point of them.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "candi"
    hits = [str(p) for p in src.rglob("*.py")
            if re.search(r"^\s*(import\s+competitors|from\s+competitors)", p.read_text("utf-8"),
                         re.M)]
    assert not hits, f"core imports the competitors tree: {hits}"
