"""Rung 1, the pointwise set model (chunk T1): budget, guardrail part 1, pointwise, planted effect."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.components import MaskHandler, NotApplicable  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from minicandi.rungs import pointwise as P  # noqa: E402
from onion_eval import guardrail as G  # noqa: E402

L = 64
BASE = _ROOT / "tools" / "minicandi" / "configs" / "rung1" / "base.json"
PINNED_KEYS = ("rung", "context_bins", "covariate_injection", "mask_mode", "dna",
               "scramble_covariates", "count_transform", "qc_on_input", "two_step", "base_depth",
               "average_input", "control_input", "predict_control", "param_budget",
               "windows_seen", "lr", "batch_size", "simpler_than_base")
ARMS = [(inj, mm) for inj in ("film", "concat", "tokens") for mm in ("drop", "token")]


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_r1"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


def _small(**kw):
    cfg = json.loads(BASE.read_text())
    cfg.update(context_bins=L, param_budget=1.4e5)
    cfg.update(kw)
    return cfg


def _jiggle(model, seed=0):
    """Move every parameter off its initial value (the NB head starts at zero, so an untrained
    model ignores its inputs and would pass a wiring check trivially)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g))
    return model.eval()


def test_base_config_holds_every_pinned_key_and_the_rung1_base_choices():
    cfg = json.loads(BASE.read_text())
    assert set(PINNED_KEYS) <= set(cfg)
    assert cfg["rung"] == 1 and cfg["mask_mode"] == "drop"
    assert cfg["covariate_injection"] == "film" and cfg["simpler_than_base"] is None
    assert R.module_name(cfg["rung"]) == "pointwise"


@pytest.mark.parametrize("inj,mm", ARMS)
def test_param_count_is_within_10pct_of_the_budget(inj, mm):
    cfg = json.loads(BASE.read_text())
    cfg.update(covariate_injection=inj, mask_mode=mm)
    m = R.build_model(cfg)
    assert abs(m.param_count() - cfg["param_budget"]) <= 0.1 * cfg["param_budget"]


def test_dna_off_and_qc_and_average_arms_build_in_budget():
    for kw in ({"dna": "off"}, {"qc_on_input": True}, {"average_input": True},
               {"count_transform": "quantile"}):
        cfg = dict(json.loads(BASE.read_text()), **kw)
        m = R.build_model(cfg)
        assert abs(m.param_count() - 1e6) <= 1e5, kw


def test_build_is_deterministic_and_width_search_draws_nothing():
    torch.manual_seed(7)
    a = R.build_model(_small())
    torch.manual_seed(7)
    state = torch.get_rng_state()
    P.choose_width(_small())
    assert torch.equal(state, torch.get_rng_state())
    b = R.build_model(_small())
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_two_step_is_refused():
    with pytest.raises(NotApplicable, match="later ablation on the winning design"):
        P.Model(_small(two_step=True))
    assert type(R.build_model(_small(two_step=True))).__name__ == "TwoStep"   # chunk A1


@pytest.mark.parametrize("inj,mm", ARMS)
@pytest.mark.parametrize("transform", ["offset", "local_ratio", "quantile"])
def test_wiring_check_passes(tables, inj, mm, transform):
    cfg = _small(covariate_injection=inj, mask_mode=mm, count_transform=transform)
    torch.manual_seed(0)
    m = _jiggle(R.build_model(cfg))
    b = OnionDataset(tables, "train", cfg, 0).batch(4, np.random.default_rng(1))
    assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
    res = C.wiring_check(m, b)
    assert res["pass"], res
    # the check has teeth: a visible input change does move the output
    pert = dict(b)
    vis = b["x_avail"][:, None, :] & ~b["x_bin_mask"]
    noise = torch.randint(1, 20, b["x_counts"].shape, generator=torch.Generator().manual_seed(4),
                          dtype=b["x_counts"].dtype)            # rank-changing (quantile)
    pert["x_counts"] = torch.where(vis, b["x_counts"] + noise, b["x_counts"])
    with torch.no_grad():
        assert not torch.allclose(m(b)["mu"], m(pert)["mu"])


@pytest.mark.parametrize("mm", ["drop", "token"])
def test_wiring_check_fails_with_the_mask_bypassed(tables, mm, monkeypatch):
    """Planted negative control: with every bin and track called visible, hidden and missing
    counts reach the perturbed model and the check must fail (so the passes above are real)."""
    cfg = _small(mask_mode=mm)
    torch.manual_seed(0)
    m = _jiggle(R.build_model(cfg))
    b = OnionDataset(tables, "train", cfg, 0).batch(4, np.random.default_rng(1))
    assert C.wiring_check(m, b)["pass"]
    monkeypatch.setattr(MaskHandler, "visible", staticmethod(lambda a, mk: torch.ones_like(mk)))
    assert not C.wiring_check(m, b)["pass"]
    assert not G.copy_leak(m, [b])["part1_wiring_pass"]


@pytest.mark.parametrize("inj,mm", [("film", "drop"), ("tokens", "token")])
def test_merged_guardrail_part1_passes(tables, inj, mm):
    cfg = _small(covariate_injection=inj, mask_mode=mm)
    torch.manual_seed(0)
    m = _jiggle(R.build_model(cfg))
    vds = OnionDataset(tables, "val", cfg, 0)
    batches = [b for _, b in zip(range(3), vds)]
    res = G.copy_leak(m, batches)
    assert res["part1_wiring_pass"], res
    assert res["n_hidden_bins"] > 0 and res["n_unavailable_bins"] > 0


def test_each_bin_is_predicted_from_that_bin_alone(tables):
    cfg = _small(dna="off")                # the DNA tower's 15 bp kernel crosses bin edges
    torch.manual_seed(0)
    m = _jiggle(R.build_model(cfg))
    b = OnionDataset(tables, "train", cfg, 0).batch(2, np.random.default_rng(2))
    j = 17
    pert = dict(b)
    x = b["x_counts"].clone()
    vis = b["x_avail"][:, None, :] & ~b["x_bin_mask"]
    x[:, j] = torch.where(vis[:, j], x[:, j] + 11, x[:, j])
    pert["x_counts"] = x
    with torch.no_grad():
        a, p = m(b)["mu"], m(pert)["mu"]
    moved = (a - p).abs().amax(dim=(0, 2))
    assert moved[j] > 0
    assert torch.all(moved[torch.arange(L) != j] == 0)


def test_track_order_of_the_input_set_does_not_matter_given_the_track_embedding(tables):
    """Permutation invariance of the pool: the pooled vector is a mean, so swapping two members'
    features and their track embeddings leaves the output unchanged."""
    cfg = _small(dna="off")
    torch.manual_seed(0)
    m = _jiggle(R.build_model(cfg))
    b = OnionDataset(tables, "train", cfg, 0).batch(2, np.random.default_rng(3))
    with torch.no_grad():
        ref = m(b)["mu"]
        perm = torch.randperm(C.A)
        m2 = m
        x_counts, avail, bm, c_in = (b["x_counts"][..., perm], b["x_avail"][:, perm],
                                     b["x_bin_mask"][..., perm], b["c_in"][:, perm])
        m2.in_track.data = m.in_track.data[perm]
        out = m2(dict(b, x_counts=x_counts, x_avail=avail, x_bin_mask=bm, c_in=c_in))["mu"]
    assert torch.allclose(ref, out, rtol=1e-5, atol=1e-6)


def test_planted_cross_assay_effect_is_recovered(tables, tmp_path):
    """Train 400 steps with DNA off; on imputation targets (kind 1) the within-window correlation
    of log mu with log1p(truth) is clearly positive, and it vanishes when the inputs are shuffled
    along the window (measured across seeds 0-2 at 400 steps: 0.18 to 0.27, shuffled -0.01 to 0)."""
    cfg = _small(dna="off", batch_size=8, windows_seen=8 * 400, lr=3e-3)
    T.train(cfg, tables, 0, tmp_path / "run", log_every=1000)
    model, c = R.load_run(tmp_path / "run")
    ds = OnionDataset(tables, "train", c, 123)

    def corr(shuffle: bool) -> float:
        rng, rs = np.random.default_rng(5), []
        for _ in range(4):
            b = ds.batch(16, rng)
            if shuffle:
                p = torch.randperm(L, generator=torch.Generator().manual_seed(0))
                b = dict(b, x_counts=b["x_counts"][:, p], x_bin_mask=b["x_bin_mask"][:, p])
            with torch.no_grad():
                mu = model(b)["mu"]
            for i, a in zip(*torch.nonzero(b["y_kind"] == C.KIND_IMPUTE, as_tuple=True)):
                x, y = torch.log(mu[i, :, a]).numpy(), np.log1p(b["y_counts"][i, :, a].numpy())
                rs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0)
        return float(np.mean(rs))

    real, shuffled = corr(False), corr(True)
    assert real > 0.1, real
    assert real - shuffled > 0.1, (real, shuffled)
