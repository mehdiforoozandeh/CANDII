"""RS resolution study: per-Haar-level variance sources.

Two kinds of check. (1) The level split itself, by a second method: per-level variances from the tool
equal block-sum energies computed here without `errdecomp.haar`, and the 15 levels add up to the plain
variance, on signals with known scales. (2) A synthetic ladder on disk with a processing effect (the
mapq knob) planted at levels 1-3, a cell effect at levels 8-11, a shared field at levels 4-7 and an
isogenic replicate component at levels 12-14: `run` puts each in its levels within the interval."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from eic_decomp import errdecomp  # noqa: E402
from onion_eval import resolution as RS  # noqa: E402

J = errdecomp.J
L = 2 ** J
K = J + 1
FRAC = np.array([2.0 ** -(i + 1) for i in range(J)] + [2.0 ** -J])   # Haar coefficients per bin, by level


def block_level_energy(v):
    """Second method: energy per Haar level of rows of v (R, L), from block sums only.
    Level j detail = (sum of left half - sum of right half) / sqrt(2^j) over blocks of 2^j bins; the
    remainder = block sum / sqrt(2^J). Returns (K,) energies summed over rows."""
    v = np.atleast_2d(np.asarray(v, np.float64))
    out = []
    for j in range(1, J + 1):
        blk = v.reshape(v.shape[0], -1, 2 ** j)
        h = 2 ** (j - 1)
        d = (blk[..., :h].sum(-1) - blk[..., h:].sum(-1)) / math.sqrt(2 ** j)
        out.append(float((d ** 2).sum()))
    out.append(float((v.reshape(v.shape[0], -1, L).sum(-1) ** 2).sum() / L))
    return np.array(out)


def field(rng, levels, coef_var, R):
    """A real signal (R, L) with Haar coefficients N(0, coef_var) at the given levels (1..14) only."""
    coefs = [np.zeros((R, c.shape[-1])) for c in errdecomp.haar(np.zeros((R, L)), J)]
    for j in levels:
        coefs[j - 1] = rng.normal(0, math.sqrt(coef_var), coefs[j - 1].shape)
    return errdecomp.haar_inverse(coefs)


# ---------------------------------------------------------------------------------------------------
# The level split, by a second method

def test_known_scales_land_at_their_levels_and_sum_to_total_variance():
    rng = np.random.default_rng(0)
    R = 3
    planted = {2: 0.5, 9: 4.0}
    x = sum(field(rng, [j], c, R) for j, c in planted.items()) + 7.0          # an offset, removed by centring
    tracks = np.stack([x.reshape(-1), (2 * x).reshape(-1)])
    sh, sp, tot = RS.track_parts(tracks, ["c0", "c1"])
    for t, scale in ((0, 1.0), (1, 4.0)):
        xc = tracks[t] - tracks[t].mean()
        want = block_level_energy(xc.reshape(R, L)) / xc.size
        np.testing.assert_allclose(tot[t], want, rtol=1e-10, atol=1e-14)
        assert tot[t].sum() == pytest.approx(np.var(tracks[t]), rel=1e-12)
        off = [i for i in range(K) if i + 1 not in planted]
        assert np.abs(tot[t][off]).max() < 1e-20 * scale + 1e-24
        for j, c in planted.items():                                       # near the planted c x 2^-j
            assert tot[t][j - 1] == pytest.approx(scale * c * FRAC[j - 1], rel=0.2)
    np.testing.assert_allclose(sh + sp, tot, rtol=1e-12)
    # two tracks of one signal (scaled 1 and 2): their shared part is the cross energy, 2 x energy of x
    np.testing.assert_allclose(sh[0], 2 * tot[0], rtol=1e-10, atol=1e-20)


def test_level_variance_equals_block_sums_of_the_log1p_difference():
    rng = np.random.default_rng(1)
    R = 2
    lam = np.exp(6.0 + field(rng, [1, 5, 13], 0.05, R))
    a, b = rng.poisson(lam), rng.poisson(lam)
    v = RS.level_variance(a, b, 1e7, 1e7, 1e7)                         # equal depths: no thinning
    d = np.log1p(a) - np.log1p(b)
    d = d - d.mean()
    np.testing.assert_allclose(v, block_level_energy(d) / d.size, rtol=1e-10)
    assert v.sum() == pytest.approx(np.var(np.log1p(a) - np.log1p(b)), rel=1e-12)


def test_track_parts_one_cell_has_no_shared_part():
    x = np.random.default_rng(2).normal(size=(2, L))
    sh, sp, tot = RS.track_parts(x, ["c0", "c0"])
    assert np.isnan(sh).all() and np.isnan(sp).all() and np.isfinite(tot).all()


def test_bin_bp():
    assert RS.bin_bp("1") == 50 and RS.bin_bp("14") == 25 * 2 ** 14 and RS.bin_bp("remainder") == 25 * 2 ** 14
    assert RS.LEVELS[0] == "1" and RS.LEVELS[-1] == "remainder" and len(RS.LEVELS) == K


# ---------------------------------------------------------------------------------------------------
# A synthetic ladder on disk

N_CELLS, R = 8, 4
LH = 7.0                                    # log rate of one half: ~1100 reads per bin
C_KNOB, C_CELL, C_SHARED, C_REP = 0.04, 0.4, 0.2, 0.4      # Haar coefficient variances
KNOB_LEVELS, CELL_LEVELS, SHARED_LEVELS, REP_LEVELS = [1, 2, 3], [8, 9, 10, 11], [4, 5, 6, 7], [12, 13, 14]
DEPTH = 2e7
LADDER_SEED = 103
CHROM = "chr1"


def planted(levels, c):
    out = np.zeros(K)
    for j in levels:
        out[j - 1] = c * FRAC[j - 1]
    return out


def _write(path, x):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f[CHROM] = np.asarray(x, np.uint32).reshape(-1)


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    pytest.importorskip("h5py")
    root = tmp_path_factory.mktemp("rs")
    rng = np.random.default_rng(LADDER_SEED)
    tables = root / "ONION" / "tables" / "toy"
    tables.mkdir(parents=True)
    shared = field(rng, SHARED_LEVELS, C_SHARED, R)
    mrows, prows = [], []

    def meas(mid, cell, halves, depth, layer="uniform", knob="", value="", role="val"):
        mdir = root / "m" / mid
        h1, h2 = halves
        _write(mdir / "pr1.h5", h1)
        _write(mdir / "pr2.h5", h2)
        _write(mdir / "counts.h5", h1 + h2)
        mrows.append(dict(measurement_id=mid, assay="H3K4me3", assay_class="narrow_histone", cell=cell,
                          in_eic=role, unit="pooled", layer=layer, arm_knob=knob, arm_value=value,
                          depth_reads=f"{depth:.0f}", depth_fragments="", has_pseudoreps="1",
                          counts_path=f"h5:{mdir / 'counts.h5'}"))
        return mid

    def pair(a, b, rung, cell, knob="", subset="0", split="val"):
        prows.append(dict(pair_id=f"p{len(prows):04d}", a=a, b=b, rung=rung, assay="H3K4me3", cell=cell,
                          knobs_differing=knob, subset_pair=subset, split=split, never_trained="0"))

    for c in range(N_CELLS):
        cell = f"C{c:02d}"
        role = "train" if c % 2 else "val"
        lr = LH + shared + field(rng, CELL_LEVELS, C_CELL, R)
        base_lr = lr + field(rng, REP_LEVELS, C_REP, R)
        bh = (rng.poisson(np.exp(base_lr)), rng.poisson(np.exp(base_lr)))
        base = meas(f"E{c}.base", cell, bh, DEPTH, role=role)
        arm_lr = base_lr + field(rng, KNOB_LEVELS, C_KNOB, R)
        arm = meas(f"E{c}.mapq0", cell, (rng.poisson(np.exp(arm_lr)), rng.poisson(np.exp(arm_lr))), DEPTH,
                   knob="mapq", value="0", role=role)
        sub = meas(f"E{c}.depth0.50", cell, (rng.binomial(bh[0], 0.5), rng.binomial(bh[1], 0.5)), DEPTH / 2,
                   knob="depth", value="0.50", role=role)
        rep_lr = lr + field(rng, REP_LEVELS, C_REP, R)
        rep = meas(f"E{c}.rep2", cell, (rng.poisson(np.exp(rep_lr)), rng.poisson(np.exp(rep_lr))), DEPTH,
                   layer="native", role=role)
        for x, y in ((base, arm), (arm, base)):                      # designed pairs come in both directions
            pair(x, y, "designed", cell, "mapq", split=role)
        pair(base, sub, "designed", cell, "depth", subset="1", split=role)
        pair(base, rep, "isogenic", cell, split=role)
    # a blind track whose files do not exist: reading it would raise
    mrows.append(dict(mrows[0], measurement_id="BLIND.base", cell="C99", in_eic="blind",
                      counts_path=f"h5:{root / 'nowhere' / 'counts.h5'}"))
    for name, rows in (("measurements.tsv", mrows), ("pairs.tsv", prows)):
        with open(tables / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    ev = root / "evalset.json"
    ev.write_text(json.dumps({"regions": [[CHROM, r * L] for r in range(R)]}))
    out = root / "resolution.tsv"
    assert RS.main(["run", "--tables", str(tables), "--evalset", str(ev), "--out", str(out),
                    "--n-boot", "400"]) == 0
    with open(out, newline="") as fh:
        got = list(csv.DictReader(fh, delimiter="\t"))
    return dict(rows=got, root=root, shared=shared)


def _series(rows, source, col="variance"):
    rs = {r["level"]: r for r in rows if r["source"] == source and r["assay_class"] == "narrow_histone"}
    assert set(rs) == set(RS.LEVELS), source
    return np.array([float(rs[lv][col]) for lv in RS.LEVELS])


# Over fixture seeds 3, 103, ..., 503 the planted value fell outside its 95% interval in 14 of 360
# (source, level) cells (3.9%), plus one systematic case: log1p of a Poisson count shrinks a log-rate
# signal by about 1 / lambda (here 0.09%), which the cell bootstrap does not cover where the shared field
# is largest. Hence a relative slack of 0.5% of the planted value and at most one miss in 15 levels.
SLACK, MAX_MISS = 0.005, 1


def _covers(rows, source, want):
    lo, hi = _series(rows, source, "ci_low"), _series(rows, source, "ci_high")
    pad = SLACK * np.abs(want)
    miss = [(lv, w, a, b) for lv, w, a, b, d in zip(RS.LEVELS, want, lo, hi, pad) if not a - d <= w <= b + d]
    assert len(miss) <= MAX_MISS, f"{source}: planted value outside the interval at {miss}"


def test_columns_and_sources(ladder):
    rows = ladder["rows"]
    assert list(rows[0]) == RS.COLUMNS
    assert {r["source"] for r in rows} == {"pseudoreplicate", "isogenic", "knob:mapq", "depth_subset", "shared",
                                           "cell_specific", "total"}
    n = {r["source"]: (r["n_pairs"], r["n_cells"]) for r in rows if r["level"] == "1"}
    assert n["knob:mapq"] == ("8", "8")                 # both directions of a designed pair count once
    assert n["shared"] == n["pseudoreplicate"] == ("8", "8")         # the blind cell is not read


def test_processing_effect_sits_at_levels_1_to_3(ladder):
    rows = ladder["rows"]
    want = planted(KNOB_LEVELS, C_KNOB)
    _covers(rows, "knob:mapq", want)
    v = _series(rows, "knob:mapq")
    assert v[:3].min() > 10 * np.abs(v[3:]).max()


def test_cell_effect_sits_at_levels_8_to_11(ladder):
    rows = ladder["rows"]
    want = planted(CELL_LEVELS, C_CELL) + planted(REP_LEVELS, C_REP)   # the replicate part is specific too
    _covers(rows, "cell_specific", want)
    v = _series(rows, "cell_specific")
    assert v[7:11].min() > 10 * np.abs(v[:7]).max()


def test_replicate_rung_and_depth_subset(ladder):
    rows = ladder["rows"]
    _covers(rows, "isogenic", planted(REP_LEVELS, C_REP))
    _covers(rows, "depth_subset", np.zeros(K))


def test_shared_field_and_total_decomposition(ladder):
    rows = ladder["rows"]
    s = ladder["shared"] - ladder["shared"].mean()
    realized = block_level_energy(s) / s.size                       # second method on the planted field
    _covers(rows, "shared", realized)
    tot = _series(rows, "total")
    np.testing.assert_allclose(_series(rows, "shared") + _series(rows, "cell_specific")
                               + _series(rows, "pseudoreplicate"), tot, rtol=1e-9)
    np.testing.assert_allclose(_series(rows, "total", "share_of_level_total"), 1.0)


def test_pseudoreplicate_levels_sum_to_the_plain_noise_variance(ladder):
    """Second method: with halves at the matched depth nothing is thinned, so the 15 levels of the
    pseudoreplicate row must add up to the mean of var(log1p pr1 - log1p pr2) / 2 over base tracks."""
    h5py = pytest.importorskip("h5py")
    root = ladder["root"]
    direct = []
    for c in range(N_CELLS):
        with h5py.File(root / "m" / f"E{c}.base" / "pr1.h5") as f1, \
                h5py.File(root / "m" / f"E{c}.base" / "pr2.h5") as f2:
            direct.append(np.var(np.log1p(f1[CHROM][:].astype(float)) - np.log1p(f2[CHROM][:].astype(float))) / 2)
    assert _series(ladder["rows"], "pseudoreplicate").sum() == pytest.approx(np.mean(direct), rel=1e-9)


def test_figure(ladder, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "res.png"
    tsv = tmp_path / "res.tsv"
    with open(tsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RS.COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(ladder["rows"])
    assert RS.main(["figure", "--in", str(tsv), "--out", str(out)]) == 0
    assert out.stat().st_size > 0
