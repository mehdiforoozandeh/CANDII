"""t130 nested-read oracle (tools/t130/nested_oracle.py) on synthetic data: the exact discrete CRPS
against brute force and the Poisson closed form, the scale search against baseline_rungs, a pair
made by true binomial thinning (the oracle beats the one-warp; the interval is not inverted), the
dispersion check (~1 on true thinning), the run + aggregate path, and the sbatch dry run."""
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, poisson

ROOT = Path(__file__).resolve().parents[1]
T130 = ROOT / "tools" / "t130"
sys.path.insert(0, str(ROOT / "tools" / "t118"))
import baseline_rungs as br  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(f"t130_{name}", T130 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NO = _load("nested_oracle")


def brute_crps(pmf: np.ndarray, y: int) -> float:
    """E|X - y| - 1/2 E|X - X'| summed over the support of the pmf."""
    k = np.arange(pmf.size)
    exy = float(np.sum(pmf * np.abs(k - y)))
    exx = float(np.sum(pmf[:, None] * pmf[None, :] * np.abs(k[:, None] - k[None, :])))
    return exy - 0.5 * exx


@pytest.mark.parametrize("x,q,mu", [(0, 0.5, 0.0), (1, 0.5, 0.0), (7, 0.25, 0.0), (40, 0.125, 0.0),
                                    (5, 1.0, 1.0), (12, 0.5, 0.3), (30, 0.3, 0.1)])
def test_exact_crps_against_brute_force(x, q, mu):
    K = 200
    pmf = binom.pmf(np.arange(x + 1), x, q)
    pmf = np.convolve(pmf, poisson.pmf(np.arange(K + 1), x * mu) if mu > 0 else [1.0])[:K + 1]
    ys = np.array([0, 1, 2, 3, x, x + 4, 60])
    got = NO.grouped_crps(np.full(ys.size, x), ys, lambda v, ymax: NO.family_cdf(v, q, mu, ymax))
    want = [brute_crps(pmf, int(y)) for y in ys]
    assert got == pytest.approx(want, abs=1e-10)


def test_poisson_remainder_equals_the_closed_form():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 60, 500)
    y = x + rng.integers(-3, 40, 500).clip(-x)
    rho = 0.25
    got = NO.family_crps(x, y, rho, 1.0)                 # arm -> base: x + Poisson(x (1-r)/r)
    lam = x * (1 - rho) / rho
    want = np.where(lam > 0, br.poisson_crps(np.maximum(lam, 1e-300), y - x), np.abs(y - x))
    assert got == pytest.approx(want, abs=1e-9)


def test_split_reproduces_scale_split():
    rng = np.random.default_rng(2)
    m = rng.gamma(2.0, 2.0, 25_000)
    y = rng.poisson(m * 1.3)
    want = br.scale_split(m, y, br.poisson_crps)
    got = NO.split(m, y, lambda mm, yy, c: br.poisson_crps(mm * 2.0 ** c, yy))
    for k in want:
        assert got[k] == pytest.approx(want[k], abs=1e-12)


def _thinned(n, rho, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.poisson(rng.gamma(0.6, 6.0, n))           # overdispersed, many zeros
    return base, rng.binomial(base, rho)


def _onewarp(xs_tr, ys_tr, x, y):
    """QuantileMatching fitted on training bins, Poisson spread (the rungs_v2 rule)."""
    def hist(a):
        h = np.bincount(a)
        nz = np.flatnonzero(h)
        return nz, h[nz]
    knots = br.qm_fit(*hist(xs_tr), *hist(ys_tr))
    return float(np.mean(br.poisson_crps(np.maximum(br.qm_apply(knots, x), br.PRED_FLOOR), y)))


@pytest.mark.parametrize("rho", [0.5, 0.125])
def test_true_thinning_oracle_beats_onewarp(rho):
    btr, atr = _thinned(60_000, rho, seed=3)
    b, a = _thinned(30_000, rho, seed=4)
    d_onewarp = _onewarp(btr, atr, b, a)
    d_oracle = float(np.mean(NO.family_crps(b, a, 1.0, rho)))
    d_model = (d_onewarp + d_oracle) / 2
    assert d_oracle < d_onewarp                           # the interval is not inverted
    assert (d_onewarp - d_model) / (d_onewarp - d_oracle) == pytest.approx(0.5)
    sp = NO.split(b, a, lambda xx, yy, c: NO.family_crps(xx, yy, 1.0, rho, c))
    assert abs(sp["c_star"]) <= 0.05 and sp["scale_error"] < 1e-3   # the true model needs no scale


def test_arm_to_base_eb_oracle_beats_onewarp_where_poisson_remainder_inverts():
    """At rho 0.125 the brief's poisson_remainder (a point mass at 0 for an empty arm bin) sits
    ABOVE the one-warp; the exact posterior with the base's training histogram as prior does not.
    This is why analyze.py reads eb_binomial for arm -> base."""
    rng = np.random.default_rng(8)
    btr = rng.poisson(rng.gamma(0.3, 40.0, 200_000))
    atr = rng.binomial(btr, 0.125)
    b = rng.poisson(rng.gamma(0.3, 40.0, 100_000))
    a = rng.binomial(b, 0.125)
    d_onewarp = _onewarp(atr, btr, a, b)
    prior = np.bincount(btr) / btr.size
    d_eb = float(np.mean(NO.eb_crps(a, b, 0.125, prior)))
    d_pr = float(np.mean(NO.family_crps(a, b, 0.125, 1.0)))
    assert d_eb < d_onewarp < d_pr


def test_dispersion_reads_one_on_true_thinning():
    b, a = _thinned(400_000, 0.25, seed=5)
    rows = NO.dispersion(b, a, 0.25, a.sum() / b.sum())
    big = [r for r in rows if r["n_bins"] > 5000 and r["base_lo"] >= 1]
    assert len(big) >= 4
    for r in big:
        assert r["dispersion_manifest"] == pytest.approx(1.0, abs=0.06)
        assert r["mean_ratio_manifest"] == pytest.approx(1.0, abs=0.03)
        assert r["n_arm_gt_base"] == 0
    # an overdispersed arm (not a thinning) reads clearly above 1
    rng = np.random.default_rng(6)
    noisy = rng.poisson(rng.gamma(4.0, 0.25 * b / 4.0 + 1e-9))
    hi = [r for r in NO.dispersion(b, noisy, 0.25, 0.25) if r["base_lo"] >= 9 and r["n_bins"] > 5000]
    assert all(r["dispersion_manifest"] > 1.5 for r in hi)


# ---------------------------------------------------------------------------------------------
# run + aggregate on a synthetic product tree
# ---------------------------------------------------------------------------------------------

CHROMS = {"chr1": 40_000, "chr19": 12_000, "chr21": 9_000, "chr22": 5_000}


def _products(root: Path):
    rng = np.random.default_rng(7)
    base = {c: rng.poisson(rng.gamma(0.6, 6.0, n)).astype(np.uint32) for c, n in CHROMS.items()}
    arms = {"15M": {c: rng.binomial(v, 0.5).astype(np.uint32) for c, v in base.items()},
            "7.5M": {c: rng.binomial(v, 0.25).astype(np.uint32) for c, v in base.items()}}
    rows = [("T1__base__base", "base", "base", 30_000_000)]
    rows += [(f"T1__depth__{lv}", "depth", lv, int(float(lv[:-1]) * 1e6)) for lv in arms]
    rows += [("T1__pe__pe", "pe", "pe", 30_000_000)]
    data = {"T1__base__base": base, "T1__pe__pe": base,
            **{f"T1__depth__{lv}": v for lv, v in arms.items()}}
    for pid, d in data.items():
        (root / pid).mkdir(parents=True)
        np.savez_compressed(root / pid / "counts25.npz", **d)
    with (root / "MANIFEST.tsv").open("w") as fh:
        fh.write("pid\ttrack\tassay\tarm\tlevel\tdepth\n")
        for pid, arm, lv, dep in rows:
            fh.write(f"{pid}\tT1\tH3K27ac\t{arm}\t{lv}\t{dep}\n")
    bl = root / "blacklist.bed"
    bl.write_text("chr19\t0\t2500\nchr21\t100\t200\n")
    return root / "MANIFEST.tsv", bl


def _read(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_run_and_aggregate(tmp_path):
    man, bl = _products(tmp_path / "products")
    out = tmp_path / "x" / "t130" / "nested"
    assert NO.depth_tracks(NO.read_manifest(man)) == ["T1"]
    assert NO.aggregate(man, out) == 1                    # no part yet: refuses
    r = subprocess.run([sys.executable, str(T130 / "nested_oracle.py"), "run", str(man),
                        str(tmp_path / "products"), str(bl), str(out), "0"],
                       capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0, r.stderr
    assert NO.aggregate(man, out) == 0
    rows = _read(out / "nested_oracle.tsv")
    # 2 arms: b2a (1 predictive) + a2b (2) per arm, a2a 2 ordered pairs (1); x 2 rho sources
    assert len(rows) == (2 * (1 + 2) + 2) * 2
    assert not any(v == "NA" for r in rows for v in r.values())
    b2a = {(r["pid"], r["r_source"]): r for r in rows if r["direction"] == "base_to_arm"}
    assert float(b2a[("T1__depth__15M", "manifest")]["rho_tgt"]) == 0.5
    assert float(b2a[("T1__depth__7.5M", "empirical")]["rho_tgt"]) == pytest.approx(0.25, abs=0.01)
    assert {r["exact"] for r in rows if r["predictive"] in ("binomial", "eb_binomial")} == {"1"}
    assert {r["exact"] for r in rows if r["predictive"] in ("poisson_remainder",
                                                            "base_mediated")} == {"0"}
    a2a = [r for r in rows if r["direction"] == "arm_to_arm"]
    assert {r["pid"] for r in a2a} == {"T1__depth__15M->T1__depth__7.5M",
                                       "T1__depth__7.5M->T1__depth__15M"}
    n_bl = 100 + 4                                        # chr19 bins 0..99, chr21 bins 4..8
    assert int(rows[0]["n_all"]) == CHROMS["chr19"] + CHROMS["chr21"] - n_bl
    # the oracle beats the one-warp on the scored bins (one-warp fitted on chr1)
    z = {pid: np.load(tmp_path / "products" / pid / "counts25.npz")
         for pid in ("T1__base__base", "T1__depth__7.5M")}
    keep = {c: ~br.blacklist_bin_flags(br.read_blacklist(bl).get(c), CHROMS[c], br.RES)
            for c in ("chr19", "chr21")}
    x = np.concatenate([z["T1__base__base"][c][keep[c]] for c in ("chr19", "chr21")]).astype(int)
    y = np.concatenate([z["T1__depth__7.5M"][c][keep[c]] for c in ("chr19", "chr21")]).astype(int)
    ow = _onewarp(z["T1__base__base"]["chr1"].astype(int), z["T1__depth__7.5M"]["chr1"].astype(int),
                  x, y)
    assert float(b2a[("T1__depth__7.5M", "manifest")]["crps_all"]) < ow
    disp = _read(out / "nested_dispersion.tsv")
    assert {d["pid"] for d in disp} == {"T1__depth__15M", "T1__depth__7.5M"}
    assert all(d["n_arm_gt_base"] == "0" for d in disp)
    assert (out / "NESTED_DONE").exists()


def test_nested_oracle_sh_dry_run_and_guard():
    sh = str(ROOT / "slurm" / "t130" / "nested_oracle.sh")
    env = {**os.environ, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "3"}
    r = subprocess.run(["bash", sh, "run", "/k", "/p", "/b.bed", "/x/t130/nested"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ("python3 /k/tools/t130/nested_oracle.py run /p/MANIFEST.tsv /p "
                                "/b.bed /x/t130/nested 3")
    r = subprocess.run(["bash", sh, "aggregate", "/k", "/p", "/b.bed", "/x/t130/nested"], env=env,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "python3 /k/tools/t130/nested_oracle.py aggregate /p/MANIFEST.tsv /x/t130/nested"
    bad = subprocess.run(["bash", sh, "run", "/k", "/p", "/b.bed", "/x/t118"], env=env,
                         capture_output=True, text=True)
    assert bad.returncode != 0 and bad.stdout == ""
