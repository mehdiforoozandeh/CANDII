"""TVf: regression tests for the defects of the total-variance audit (cruxvault/results/t233/AUDIT.md).

Each test plants a known value with the audit's simulator (`tools/onion_eval/total_variance.py`, the
chain of plan Amendment 3 section A) and checks the repaired tool against it, as the audit does:
planted log1p values are direct pair values 0.5 mean(d^2) of a second Poisson draw of the same latent
rates. Where a second method exists (the halves as two measurements, the audit's own thinned branch)
the test compares the two."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import layers as LY  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import mixed_model as MM  # noqa: E402
from onion_eval import resolution as RS  # noqa: E402
from onion_eval import total_variance as TV  # noqa: E402

pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def sims():
    return dict(sim=TV.simulate(TV.BASE_LAYERS, n_cells=6, seed=0),
                big=TV.simulate(TV.BASE_LAYERS, n_cells=6, seed=0, count_seed=977),
                big_half=TV.simulate(TV.BASE_LAYERS, {"scale": 0.5}, n_cells=6, seed=0, count_seed=978))


def _in(row, planted, lo="ci_low", hi="ci_high"):
    return float(row[lo]) <= planted <= float(row[hi])


# ---------------------------------------------------------------------------------------------------
# 1 and 4. The pseudoreplicate row (a == b) reads its halves; layers.tsv carries the common depth

def test_pseudoreplicate_self_pair_reads_halves(sims):
    sim, big, big_half = sims["sim"], sims["big"], sims["big_half"]
    pairs = [p for p in sim["pairs"] if p["rung"] in ("pseudoreplicate", "sequencing")]
    got = TV._layers_run(sim, pairs, "self", n_boot=200)          # pairs.tsv as build_tables writes it
    pr = got["pseudoreplicate"]
    plant = TV.pair_values(big, "log1p", ["pseudoreplicate"])["pseudoreplicate"]   # halves, half depth
    assert float(pr["variance"]) > 0 and _in(pr, plant), (pr["variance"], plant)
    assert float(pr["depth"]) == 5e6                              # the halves' depth (1e7 / 2) is common
    # second method: the halves written as two measurements give the same rung bit for bit
    got2 = TV._layers_run(sim, pairs, "halves", n_boot=200)
    assert float(got2["pseudoreplicate"]["variance"]) == float(pr["variance"])
    # the sampling term stays in the pseudoreplicate rung: the sequencing increment is the half-depth layer
    pv_full = TV.pair_values(big, "log1p", ["pseudoreplicate"])
    pv_half = TV.pair_values(big_half, "log1p", ["sequencing"])
    assert _in(got["sequencing"], pv_half["sequencing"] - pv_full["pseudoreplicate"])


def test_half_depths_read_the_files_attributes(tmp_path):
    import h5py
    for h, d in (("counts", 1000), ("pr1", 300), ("pr2", 700)):
        with h5py.File(tmp_path / f"{h}.h5", "w") as f:
            f["chr1"] = np.zeros(4, np.uint32)
            f.attrs["depth_reads"] = d
    row = {"counts_path": f"h5:{tmp_path / 'counts.h5'}", "depth_reads": "1000"}
    (p1, d1), (p2, d2) = LY.half_depths(row)
    assert (d1, d2) == (300.0, 700.0) and p1.endswith("pr1.h5") and p2.endswith("pr2.h5")
    from onion_eval import score as SC
    assert SC._half_shares(row) == pytest.approx((0.3, 0.7))
    for h in ("pr1", "pr2"):                                      # no attributes: half the row's depth each
        with h5py.File(tmp_path / f"{h}.h5", "w") as f:
            f["chr1"] = np.zeros(4, np.uint32)
    assert [d for _, d in LY.half_depths(row)] == [500.0, 500.0]
    assert SC._half_shares(row) == (0.5, 0.5)
    assert LY.half_depths({"counts_path": "store:/x:bs:A", "depth_reads": "1"}) is None


# ---------------------------------------------------------------------------------------------------
# 2. variance and layer_increment are the pair value 0.5 E[(Y - Y')^2]

def test_layers_variance_is_the_pair_value(sims):
    sim, big = sims["sim"], sims["big"]
    recs = TV._records(sim, [p for p in sim["pairs"] if p["rung"] == "technical"], depth=1.0)
    t = TV._lt(recs)["technical"]
    plant = TV.pair_values(big, "log1p", ["technical"])["technical"]
    assert _in(t, plant), (t["variance"], plant)
    # second method: 0.5 mean(d^2) of the same pairs (no centring, no thinning at equal depth)
    direct = TV.pair_values(sim, "log1p", ["technical"])["technical"]
    assert t["variance"] / direct == pytest.approx(1.0, abs=0.01)
    # pair_variance itself stays Var(d): twice the table's value on the same records
    assert np.mean([r["v"][TV.ALL] for r in recs]) == pytest.approx(2 * t["variance"], rel=1e-12)


def test_resolution_takes_the_pair_value_without_halving_again():
    rng = np.random.default_rng(0)
    pr = [dict(cell=f"c{c}", v=rng.uniform(0.1, 0.2, len(RS.LEVELS))) for c in range(4)]
    rungs = [dict(rung="technical", cell=f"c{c}", v=r["v"] + rng.uniform(0.0, 0.05, len(RS.LEVELS)))
             for c, r in enumerate(pr)]
    rows = RS.class_rows("narrow_histone", {"pr": pr, "rungs": rungs, "knobs": [], "grams": []}, n_boot=50)
    got = {(r["source"], r["level"]): r["variance"] for r in rows}
    for i, lv in enumerate(RS.LEVELS):
        want_pr = np.mean([0.5 * r["v"][i] for r in pr])
        want_t = np.mean([0.5 * (t["v"][i] - p["v"][i]) for t, p in zip(rungs, pr)])
        assert got[("pseudoreplicate", lv)] == pytest.approx(want_pr, rel=1e-12)
        assert got[("technical", lv)] == pytest.approx(want_t, rel=1e-9, abs=1e-15)


# ---------------------------------------------------------------------------------------------------
# 3. an increment is taken per cell over the cells of both rungs, and a cell mismatch is flagged

def test_increment_is_paired_by_cell():
    sim = TV.simulate(TV.HET_LAYERS, n_cells=6, seed=0)
    late = ("c3", "c4", "c5")
    tech = [p for p in sim["pairs"] if p["rung"] == "technical"]                   # all 6 cells
    iso = [p for p in sim["pairs"] if p["rung"] == "isogenic" and p["cell"] in late]
    t = TV._lt(TV._records(sim, tech + iso, depth=1.0))["isogenic"]
    assert t["flag"] == LY.FLAG_CELLS and t["n_cells"] == 3
    # planted: the layer on the common cells, mean of 3 reference draws of the same latent rates (their
    # spread is about 0.0005, wider than the 3-cell interval, so the check is a 10% band, not the interval)
    plant = []
    for cs in (977, 979, 1001):
        big = TV.simulate(TV.HET_LAYERS, n_cells=6, seed=0, count_seed=cs)
        pv = TV.pair_values(dict(big, pairs=[p for p in big["pairs"] if p["cell"] in late]), "log1p",
                            ["technical", "isogenic"])
        plant.append(pv["isogenic"] - pv["technical"])
    plant = float(np.mean(plant))
    assert t["layer_increment"] == pytest.approx(plant, rel=0.1), (t["layer_increment"], plant)
    # second method: both rungs restricted to the common cells give the same increment
    same = TV._lt(TV._records(sim, [p for p in tech if p["cell"] in late] + iso, depth=1.0))["isogenic"]
    assert same["flag"] == LY.FLAG_OK and t["layer_increment"] == pytest.approx(same["layer_increment"], rel=1e-12)
    # disjoint cells: no increment at all, never one taken across cell sets
    early = [p for p in tech if p["cell"] not in late]
    t2 = TV._lt(TV._records(sim, early + iso, depth=1.0))["isogenic"]
    assert np.isnan(t2["layer_increment"]) and np.isnan(t2["ci_width"]) and t2["flag"] == LY.FLAG_CELLS


def test_pooled_row_of_assays_at_different_depths_is_flagged():
    rng = np.random.default_rng(1)
    recs = []
    for assay, depth, n_cells in (("A", 1e7, 6), ("B", 5e6, 1)):
        for c, rung in product(range(n_cells), ("pseudoreplicate", "technical")):
            recs.append(dict(assay=assay, assay_class="narrow_histone", rung=rung, cell=f"c{c}", depth=depth,
                             v={TV.ALL: float(rng.uniform(0.1, 0.2))}))
    rows = LY.assemble(recs, n_boot=50)
    by = {(r["assay_or_class"], r["rung"]): r for r in rows}
    assert by[("A", "technical")]["depth"] == 1e7 and by[("B", "technical")]["depth"] == 5e6
    pooled = by[("narrow_histone", "technical")]
    assert pooled["depth"] == "mixed" and LY.FLAG_DEPTHS in pooled["flag"]


# ---------------------------------------------------------------------------------------------------
# 5. mixed model: a named donor is labelled by cell and donor (crossed with experiment)

def test_mixed_model_donor_is_crossed_with_experiment():
    seed, K = 0, 600
    rng = np.random.default_rng(seed)
    var = dict(cell=1.0, donor=0.5, experiment=0.5, library=0.3, residual=0.2)    # the audit's planted values
    mu = rng.normal(0, 1, K)
    meas, Y = [], []
    for c in range(6):
        uc = rng.normal(0, np.sqrt(var["cell"]), K)
        ud = [rng.normal(0, np.sqrt(var["donor"]), K) for _ in range(2)]
        ux = [rng.normal(0, np.sqrt(var["experiment"]), K) for _ in range(2)]
        for d, x, l in product(range(2), range(2), range(2)):
            ul = rng.normal(0, np.sqrt(var["library"]), K)
            for _ in range(2):                                     # the two halves of the library
                Y.append(mu + uc + ud[d] + ux[x] + ul + rng.normal(0, np.sqrt(var["residual"]), K))
            meas.append(dict(measurement_id=f"c{c}x{x}d{d}l{l}", assay="A", in_eic="train", arm_knob="",
                             unit=f"br{d + 1}tr{l + 1}", layer="uniform", experiment=f"c{c}x{x}", cell=f"c{c}",
                             donor=f"D{c}{d}", bio_rep=str(d + 1), depth_reads="1000", has_pseudoreps="1",
                             counts_path=f"h5:/none/c{c}x{x}d{d}l{l}/counts.h5"))
    obs, _ = MM.select_observations(meas, "A")
    assert len(obs) == len(Y) and obs[0]["replicate"] == "c0|D00" and obs[2]["replicate"] == "c0|D00"
    Y = np.array(Y)
    labels = {f: [o[f] for o in obs] for f in MM.FACTORS}
    rows, _ = MM.decompose(Y @ Y.T, Y.sum(1), K, labels, None, n_boot=30, seed=seed)
    by = {r["factor"]: r for r in rows}
    tot = sum(var.values())
    for f, key in (("experiment", "experiment"), ("replicate", "donor")):
        assert _in(by[f], var[key] / tot), (f, by[f]["variance_fraction"], var[key] / tot)
    # no donor in the table: the biological replicate within its experiment, as before
    obs2, _ = MM.select_observations([dict(m, donor="") for m in meas], "A")
    assert obs2[0]["replicate"] == "c0x0|1"


# ---------------------------------------------------------------------------------------------------
# 6. mixed model: every observation is thinned to one depth before log1p

def _mm_experiment(seed, exp_depth, depth_arg):
    sim = TV.simulate(dict(cell_type=0.5, bio_rep=0.08, library=0.05), {"experiment": exp_depth}, n_cells=6,
                      seed=seed)
    obs = []
    for c, x, r, l in product(range(6), range(2), range(2), range(2)):
        mid = TV._leaf(c, 0, x, r, l, 0)
        obs.append(dict(counts_path=mid, cell=f"c{c}", experiment=f"c{c}x{x}", replicate=f"c{c}x{x}|r{r}",
                        library=f"c{c}x{x}|r{r}l{l}", depth=sim["depth"][mid]))

    class Reader:
        def read(self, path, chrom, start, end):
            return sim["tracks"][path][start:end]

    S, ysum, K = MM.sufficient_stats(obs, [("chr1", 0)], "fine", Reader(), depth=depth_arg)
    rows, _ = MM.decompose(S, ysum, K, {f: [o[f] for o in obs] for f in MM.FACTORS}, None, n_boot=30, seed=seed)
    return {r["factor"]: r for r in rows}["experiment"]


def test_mixed_model_thins_to_one_depth():
    planted = 0.0                                  # no experiment variance planted
    thinned = _mm_experiment(0, [1.0, 4.0], "common")
    assert thinned["variance_fraction"] - planted < 0.005 and thinned["ci_high"] < 0.005
    # power: the same data read unthinned put depth into the experiment row (the audit's +0.076)
    raw = _mm_experiment(0, [1.0, 4.0], None)
    assert raw["variance_fraction"] > 0.03 and raw["ci_low"] > 0.01
    assert MM.common_depth([{"depth": 2.0}, {"depth": 0.5}]) == 0.5
    assert MM.common_depth([{"depth": 2.0}, {}]) is None
    with pytest.raises(ValueError):
        MM.sufficient_stats([dict(counts_path="x", depth=1.0)], [("chr1", 0)], "fine", None, depth=2.0)


# ---------------------------------------------------------------------------------------------------
# 7. noise floor: real depth shares and the unequal-length Spearman-Brown form; the log1p bias diagnostic

def test_noise_floor_unequal_shares_on_additive_noise():
    rng = np.random.default_rng(3)
    M, f = 2 ** 14, (0.3, 0.7)
    sig = rng.normal(0, 1, M)                                     # signal variance 1
    target = sig + rng.normal(0, 1, M)                            # target noise variance 1: share 0.5
    h1, h2 = (sig + rng.normal(0, np.sqrt(1 / fi), M) for fi in f)   # half i: noise 1 / f_i at target scale
    new = Mx.noise_floor(h1, h2, target, J=10, shares=f)
    old = Mx.noise_floor(h1, h2, target, 2.0, J=10)
    assert new == pytest.approx(0.5, abs=0.02) and abs(old - 0.5) > 0.03
    # equal shares give the Spearman-Brown form with k = 1 / f, bit for bit
    assert Mx.noise_floor(h1, h2, target, J=10, shares=(0.5, 0.5)) == Mx.noise_floor(h1, h2, target, 2.0, J=10)
    assert Mx.noise_floor(sig, sig, sig, J=10, shares=f) == pytest.approx(0.0, abs=1e-12)


def _noise_case(frac, seed=0):
    """The audit's Spearman-Brown case, with the real shares passed as score.py now does."""
    sim = TV.simulate(TV.BASE_LAYERS, {"pr_frac": frac}, n_cells=6, seed=seed)
    rng = np.random.default_rng(seed + 7)
    est, corr, plant, cells = [], [], [], []
    for c, d, x in product(range(6), range(2), range(2)):
        mid = TV._leaf(c, d, x, 0, 0, 0)
        y, h = sim["tracks"][mid], [sim["tracks"][f"{mid}.{k}"] for k in ("pr1", "pr2")]
        ts = Mx.to_space(y, "counts_log1p")
        f = tuple(float(hh.sum() / y.sum()) for hh in h)
        n = Mx.noise_floor(*(Mx.to_space(hh, "counts_log1p", rescale=1 / fi) for hh, fi in zip(h, f)), ts,
                           shares=f)
        est.append(n)
        corr.append(n - Mx.noise_floor_log1p_bias(y, f, seed=c * 4 + d * 2 + x))
        lam = sim["rate"][mid].astype(float)
        y1, y2 = (np.log1p(rng.poisson(lam)) for _ in range(2))
        plant.append(Mx.noise_floor(y1, y2, ts, 1.0))
        cells.append(f"c{c}")
    e = RS.mean_ci(np.array(est)[:, None], cells)
    k = RS.mean_ci(np.array(corr)[:, None], cells)
    return float(np.mean(plant)), (e[0][0], e[1][0], e[2][0]), (k[0][0], k[1][0], k[2][0])


def test_noise_floor_unequal_halves_on_the_audit_simulation():
    plant, even, even_c = _noise_case(0.5)
    _, uneq, uneq_c = _noise_case(0.3)
    # the log1p step's own bias remains (not corrected; larger for the 0.3 half, which has fewer reads):
    # N understates the planted share for both splits, where the old form overstated it for 0.3 / 0.7
    assert even[2] < plant and uneq[2] < plant
    # second method: N minus the simulated bias of the diagnostic holds the planted share for both splits
    for est, lo, hi in (even_c, uneq_c):
        assert lo <= plant <= hi, (plant, est, lo, hi)


# ===================================================================================================
# TVg: PI rulings 30 (sibling layers) and 31 (headline 7 per rung), the simulator on the pair-value scale,
# and the error budget with real read shares

def test_lower_terms_sibling_structure_and_fallbacks():
    full = LY.lower_terms(LY.RUNGS)
    fmt = {r: (LY.format_terms(t), lay) for r, (t, lay) in full.items()}
    assert fmt["isogenic"] == ("technical", "biological_replicate")
    assert fmt["anisogenic"] == ("isogenic", "donor")
    assert fmt["other_experiment"] == ("isogenic", "experiment")
    assert fmt["other_cell_type"] == ("anisogenic+other_experiment-isogenic", "cell_type")
    assert LY.parse_terms(fmt["other_cell_type"][0]) == full["other_cell_type"][0]
    # other_experiment empty (build_tables writes none): cell type over anisogenic, holding the experiment
    part = LY.lower_terms(["pseudoreplicate", "isogenic", "anisogenic", "other_cell_type"])
    assert LY.format_terms(part["other_cell_type"][0]) == "anisogenic"
    assert part["other_cell_type"][1] == "cell_type+experiment"
    assert part["isogenic"][1] == "sequencing+library+biological_replicate"
    # isogenic empty: each sibling over the nearest nested rung, holding the biological-replicate layer
    noiso = LY.lower_terms(["technical", "anisogenic", "other_experiment", "other_cell_type"])
    assert noiso["anisogenic"] == ([("technical", 1)], "biological_replicate+donor")
    assert noiso["other_cell_type"] == ([("other_experiment", 1)], "cell_type+donor")


def test_crossed_layers_recover_planted_donor_experiment_and_cell_type(sims):
    """PI ruling 30 on the simulator with donor and experiment crossed (cv2 donor 0.15, experiment 0.10, cell
    type 0.5). The cell-type layer is planted over a true pair that shares only the cell type."""
    sim, big = sims["sim"], sims["big"]
    rows = TV.case_crossed_layers(0, big)
    for r in rows:
        assert r["defect"] == "no", (r["place"], r["planted"], r["estimated"])
    # all rungs on the same cells: the increments add up to the other_cell_type pair value
    ps = [p for p in sim["pairs"]]
    rows_all = LY.assemble([dict(r, assay="A", assay_class="narrow_histone")
                            for r in TV._records(sim, ps, depth=0.5)], n_boot=50)   # halves set the depth
    chk = TV.telescoping_check(rows_all)
    assert chk and all(c["ok"] for c in chk), [c for c in chk if not c["ok"]][:1]


def test_error_budget_noise_with_real_shares():
    from onion_eval import error_budget as EB
    rng = np.random.default_rng(5)
    M, f = 2 ** 12, (0.3, 0.7)
    sig = rng.normal(0, 1, (3, M))
    h1, h2 = (sig + rng.normal(0, np.sqrt(1 / fi), (3, M)) for fi in f)
    even = EB.noise_share_from_halves(h1, h2, 2.0)
    assert np.array_equal(EB.noise_share_from_halves(h1, h2, shares=(0.5, 0.5)), even)    # bit-identical
    got = EB.noise_share_from_halves(h1, h2, shares=np.tile(f, (3, 1)))
    assert got[:, :8].mean() == pytest.approx(0.5, abs=0.03)       # planted noise share 1 / (1 + 1) on fine scales
    assert abs(even[:, :8].mean() - 0.5) > 0.03
    # the same unequal form as metrics.noise_floor on one track and its own energy weights
    one = Mx.noise_floor(h1[0], h2[0], sig[0] + rng.normal(0, 1, M), J=12, shares=f, return_levels=True)
    assert one["share"][:8].mean() == pytest.approx(got[0, :8].mean(), abs=0.02)


class _FakeScorer:
    """The attributes `Scorer._h7_per_rung` and `_h7_metric` read, over one simulated cell."""

    def __init__(self, sim, cell, pairs, depth):
        self.sim, self.cell, self.pairs, self.depth = sim, cell, pairs, depth
        self.regions, self.rb, self.external, self.n_boot, self.seed = [("chr1", 0)], TV.L, False, 20, 0

    def _rung_pairs(self, assay, cell):
        out = {}
        for p in self.pairs:
            if p["cell"] == cell:
                out.setdefault(p["rung"], []).append((p["a"], p["b"]))
        return out

    def _row(self, mid):
        return {"depth_reads": str(self.depth[mid])}

    def _ds_read(self, mid, chrom, start, L):
        return self.sim["tracks"][mid][start:start + L]


def _h7(sim, c, n_scale=1.0):
    from onion_eval import score as SC
    n, mu = TV._z1_model(sim, c)
    pairs = [p for p in sim["pairs"] if p["rung"] != "pseudoreplicate"]
    fake = _FakeScorer(sim, f"c{c}", pairs, {k: v * 1e7 for k, v in sim["depth"].items()})
    leaf = TV._leaf(c, 0, 0, 0, 0, 0)
    vr = Mx.variance_ratio(n * n_scale, 0.5 * mu, sim["tracks"][leaf + ".pr1"], sim["tracks"][leaf + ".pr2"])["pooled"]
    job = {"assay": "A", "cell": f"c{c}", "target": leaf}
    return SC.Scorer._h7_per_rung(fake, job, {"depth_reads": "1e7"}, n * n_scale, mu, True, vr, TV.L), fake


def test_headline7_per_rung_and_pass_flag():
    """A calibrated Z1 model on a design with one donor per cell type (a cell line): >= 1 at the halves, about
    1 at other_experiment, so it passes. A model with a much smaller NB size (over-dispersed) fails the upper
    bound."""
    from onion_eval import score as SC
    sim = TV.simulate(dict(TV.BASE_LAYERS, donor=0.0), n_cells=6, seed=0)
    passes, ratios = [], []
    for c in range(6):
        out, _ = _h7(sim, c)
        per = out["variance_ratio_per_rung"]
        assert set(per) == set(LY.RUNGS)
        ratios.append(per["other_experiment"])
        passes.append(out["variance_ratio_pass"])
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05)
    assert sum(passes) >= 3                     # about 1 at the upper bound: a pass in about half the cells
    over, fake = _h7(sim, 0, n_scale=0.2)
    assert over["variance_ratio_per_rung"]["other_experiment"] > 1.5 and over["variance_ratio_pass"] == 0.0
    # the metric block: old scalar kept, described as the lower-bound check, per_rung and pass_rate added
    recs = []
    for c in range(6):
        out, _ = _h7(sim, c)
        recs.append(dict(target=f"t{c}", assay="A", cell=f"c{c}", variance_ratio=out["variance_ratio_per_rung"][
            "pseudoreplicate"], **out))
    blk = SC.Scorer._h7_metric(fake, recs, ["A"])
    assert blk["description"].startswith("lower-bound check")
    assert blk["macro"] == pytest.approx(np.mean([r["variance_ratio"] for r in recs]))
    assert set(blk["per_rung"]) == set(LY.RUNGS) and "pass_rate" in blk and len(blk["per_target"]) == 6
    assert blk["upper_bound_rung"]["imputation"] == "other_experiment"
