"""t125-P2h: the merged-mixture CRPS for own-noise h in the chain (PI decision 2026-10-16).

The rule: per bin, the components k of the mixture sum_k NB(k) N(mu_k, sigma_k^2) are sorted by
mu_k, adjacent components are merged into at most G (256 in the chain) moment-matched Gaussians
(each group keeps its weight, mean and variance, the variance including the spread of the group's
means), and the CRPS pair sum runs over the merged components. Exact stays the default (alone
scoring, `score.py chain` without --mix-merge); a merged record says so (`crps_mix_merged`,
`crps_mix_groups`, `n_mix_merged_bins`). Push-only h are never merged.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hmap import pushforward as pf  # noqa: E402
from hmap import score  # noqa: E402
from test_t125_h_score import S, _h_dir  # noqa: E402,F401  (the shared synthetic set-up)

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")


def _mixture(n, K, seed, smooth=True, wiggle=0.05):
    rng = np.random.default_rng(seed)
    k = np.arange(K, dtype=np.float64)[None, :]
    if smooth:                                     # mu(k), sigma(k) smooth in k, as an h's are
        mu = 0.5 + 0.002 * k * rng.uniform(0.5, 1.5, (n, 1)) + wiggle * np.sin(k / 37.0)
        sg = 0.2 + 0.05 * np.cos(k / 53.0) * rng.uniform(0.5, 1.0, (n, 1))
        centre = rng.uniform(0.3, 0.7, (n, 1)) * K
        w = np.exp(-0.5 * ((k - centre) / (0.15 * K)) ** 2)
    else:
        mu = rng.normal(0, 1, (n, K))
        sg = np.exp(rng.normal(-1, 0.3, (n, K)))
        w = rng.random((n, K))
    w[:, K - K // 7:] *= rng.random((n, 1)) > 0.5  # some rows shorter (padding)
    y = mu[:, K // 2] + rng.normal(0, 0.3, n)
    return [torch.from_numpy(np.ascontiguousarray(a)) for a in (mu, sg, w, y)]


def test_merge_keeps_each_rows_weight_mean_and_variance():
    mu, sg, w, _ = _mixture(40, 900, 0, smooth=False)
    wn = w / w.sum(1, keepdim=True)
    M, V, W = pf.merge_mixture_t(mu, sg * sg, wn, 64)
    assert M.shape == (40, 64)
    np.testing.assert_allclose(W.sum(1).numpy(), 1.0, rtol=1e-12)
    mean = (wn * mu).sum(1)
    np.testing.assert_allclose((W * M).sum(1).numpy(), mean.numpy(), rtol=1e-11, atol=1e-13)
    var = (wn * (sg * sg + mu * mu)).sum(1) - mean ** 2
    np.testing.assert_allclose(((W * (V + M * M)).sum(1) - (W * M).sum(1) ** 2).numpy(),
                               var.numpy(), rtol=1e-9)
    assert bool((V[W > 0] > 0).all())
    assert ((W > 0).sum(1) <= 64).all() and ((W > 0).sum(1) > 32).all()


def test_merge_groups_are_adjacent_in_mu():
    """Equal weights, counts and gaps: the cut is the equal-count one, in sorted mu order."""
    mu = torch.tensor([[5.0, 1.0, 3.0, 2.0, 4.0, 0.0]], dtype=torch.float64)
    w = torch.full((1, 6), 1 / 6, dtype=torch.float64)
    M, V, W = pf.merge_mixture_t(mu, torch.zeros_like(mu) + 1e-6, w, 3)
    np.testing.assert_allclose(M.numpy(), [[0.5, 2.5, 4.5]])
    np.testing.assert_allclose(W.numpy(), [[1 / 3, 1 / 3, 1 / 3]])
    np.testing.assert_allclose(V.numpy(), [[0.25 + 1e-6] * 3])


def test_merge_groups_are_finer_where_the_weight_is():
    rng = np.random.default_rng(5)
    mu = torch.from_numpy(np.sort(rng.normal(0, 1, (1, 1000)), axis=1))
    w = torch.from_numpy(np.exp(-0.5 * (np.arange(1000) - 500.0) ** 2 / 50.0 ** 2))[None, :]
    w = w / w.sum()
    M, V, W = pf.merge_mixture_t(mu, torch.full_like(mu, 0.01), w, 50)
    used = W[0] > 0
    assert int(used.sum()) <= 50
    assert bool((torch.diff(M[0][used]) > 0).all())          # adjacent, ordered groups
    assert float(W[0][used].max()) < 0.15                    # no group swallows the bulk


@pytest.mark.parametrize("K", [5, 200])
def test_rows_within_the_cap_are_exact(K):
    mu, sg, w, y = _mixture(30, K, 1)
    exact = pf.crps_gauss_mixture_t(mu, sg, w, y)
    merged = pf.crps_gauss_mixture_t(mu, sg, w, y, max_groups=256)
    np.testing.assert_array_equal(merged.numpy(), exact.numpy())    # nothing to merge: untouched
    np.testing.assert_allclose(exact.numpy(), pf.crps_gauss_mixture(*[t.numpy() for t in
                                                                      (mu, sg, w, y)]),
                               rtol=1e-12)


@pytest.mark.parametrize("wiggle,tol", [(0.0, 1e-6), (0.05, 1e-4)])
def test_merged_crps_is_close_on_a_wide_smooth_mixture(wiggle, tol):
    """A wide support (2,500 components, as a deep bin's) merged to 256: within 1e-6 per bin when
    mu rises with k; within 1e-4 when mu wiggles (components of different sigma then sit side by
    side in mu order, and merging them is not exact at any group count)."""
    mu, sg, w, y = _mixture(12, 2500, 2, wiggle=wiggle)
    exact = pf.crps_gauss_mixture_t(mu, sg, w, y)
    merged = pf.crps_gauss_mixture_t(mu, sg, w, y, max_groups=256)
    rel = (merged - exact).abs() / exact.abs()
    assert float(rel.max()) < tol, float(rel.max())
    assert float(rel.max()) > 0                   # it did merge


def test_merged_with_row_key_and_mixed_widths():
    mu, sg, w, y = _mixture(20, 600, 3, smooth=False)
    w[::2, 50:] = 0.0                             # half the rows within the cap
    key = torch.arange(20) % 5
    mu, sg, w = mu[key], sg[key], w[key]
    a = pf.crps_gauss_mixture_t(mu, sg, w, y, max_groups=100, row_key=key)
    b = pf.crps_gauss_mixture_t(mu, sg, w, y, max_groups=100)
    np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-12)
    exact = pf.crps_gauss_mixture_t(mu, sg, w, y)
    small = (w > 0).sum(1) <= 100                 # not merged: their own components (only the
    np.testing.assert_allclose(b[small].numpy(), exact[small].numpy(), rtol=1e-13)  # sum order)
    assert not np.allclose(b[~small].numpy(), exact[~small].numpy(), rtol=1e-13)


# -- the chain -------------------------------------------------------------------------------------


def _chain(S, specs, out, mix_groups=None):
    fd = score.score_chain(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], S["f_dir"],
                           out, specs, workers=1, control_rl_tsv=S["rl"], device="cpu",
                           mix_groups=mix_groups)
    return {p.name: json.loads((p / "scores.json").read_text())
            for p in Path(fd).iterdir() if p.is_dir()}


@pytest.fixture(scope="module")
def chains(S, tmp_path_factory):
    root = tmp_path_factory.mktemp("mixmerge")
    specs = ["formula:frag", str(S["h_mlp"]), str(S["h_noise"])]
    return {"exact": _chain(S, specs, root / "exact"),
            "merged": _chain(S, specs, root / "merged", mix_groups=4),
            "wide": _chain(S, specs, root / "wide", mix_groups=10 ** 6)}


def test_exact_chain_has_no_merge_fields(chains):
    for res in chains["exact"].values():
        assert res["config"]["mix_groups"] is None
        for r in res["records"]:
            assert not {"crps_mix_merged", "crps_mix_groups", "n_mix_merged_bins"} & set(r)


def test_merged_chain_flags_only_the_own_noise_h(chains):
    noise = "h_formula_affine_r5_noise_true_s2"
    for name, res in chains["merged"].items():
        assert res["config"]["mix_groups"] == 4
        for r, e in zip(res["records"], chains["exact"][name]["records"]):
            if name == noise:
                assert r["crps_mix_merged"] is True and r["crps_mix_groups"] == 4
                assert 0 < r["n_mix_merged_bins"] <= r["n_all"]
                assert r["crps_log_all"] == pytest.approx(e["crps_log_all"], rel=0.05)
                assert r["crps_mean_point_all"] == pytest.approx(e["crps_mean_point_all"],
                                                                 rel=1e-9)
            else:                                 # push only: never merged, the same numbers
                assert "crps_mix_merged" not in r
                for k, v in e.items():
                    if isinstance(v, float):
                        assert r[k] == pytest.approx(v, rel=1e-9, abs=1e-12), (name, k)
    assert sum(r["n_mix_merged_bins"] for r in chains["merged"][noise]["records"]) > 0


def test_a_cap_above_every_support_gives_the_exact_numbers(chains):
    noise = "h_formula_affine_r5_noise_true_s2"
    for r, e in zip(chains["wide"][noise]["records"], chains["exact"][noise]["records"]):
        assert r["n_mix_merged_bins"] == 0 and r["crps_mix_merged"] is True
        for k, v in e.items():
            if isinstance(v, float):
                assert r[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_cli_passes_mix_merge(monkeypatch, tmp_path):
    got = {}

    def fake(*a, **k):
        got.update(k)
        return tmp_path
    monkeypatch.setattr(score, "score_chain", fake)
    base = ["chain", "m", "c", "d", "ctl", "bl", "f_run", str(tmp_path), "--h", "formula:bin"]
    score.main(base)
    assert got["mix_groups"] is None
    score.main(base + ["--mix-merge", "256"])
    assert got["mix_groups"] == 256


def test_merge_needs_the_fast_path(S):
    ctx = {"y": np.zeros(0), "mode": "nb"}
    with pytest.raises(ValueError):
        score.score_context([], ctx, device="cpu", fast=False, mix_groups=8)
