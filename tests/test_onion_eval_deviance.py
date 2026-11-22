"""t371: the deviance-based metrics (tools/onion_eval/deviance.py) and the scorer switch.

Requirements checked:
  - the unit deviance, the expected deviances and the gap are correct on known cases;
  - the simulation's pre-registered checks C1-C5 pass (tools/onion_eval/deviance_sim.py);
  - the switch adds the key `deviance` and changes no other key of scores.json, on the in-process
    path and on the prediction-store path (predict + metrics phases, as the kit scores a run).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import deviance as DV  # noqa: E402
from onion_eval import deviance_sim as SIM  # noqa: E402
from onion_eval import score as S  # noqa: E402


# ------------------------------------------------------------------ unit cases
def test_unit_deviance_known_values():
    y = np.array([0.0, 1.0, 3.0])
    mu = np.array([0.5, 1.0, 1.0])
    want = 2 * np.array([0.5, 0.0, 3 * np.log(3) - 2])
    assert np.allclose(DV.unit_deviance(y, mu), want)
    assert DV.unit_deviance(np.array([2.0]), np.array([2.0]))[0] == 0.0


def _exact_nb(n, m):
    p = n / (n + m)
    k = np.arange(int(stats.nbinom.ppf(1 - 1e-12, n, p)) + 1)
    pk = stats.nbinom.pmf(k, n, p)
    yl = np.where(k > 0, k * np.log(np.maximum(k, 1)), 0.0)
    return 2 * ((pk * yl).sum() / pk.sum() - m * np.log(m))


@pytest.mark.parametrize("n,m", [(0.3, 0.05), (1.0, 1.0), (5.0, 20.0), (0.2, 200.0), (300.0, 3.0)])
def test_nb_expected_deviance_matches_the_exact_sum(n, m):
    got = float(DV.nb_expected_deviance(np.array([n]), np.array([m]))[0])
    assert got == pytest.approx(_exact_nb(n, m), rel=2e-3)


def test_poisson_expected_deviance_is_the_large_n_limit_of_nb():
    m = np.array([0.01, 0.5, 3.0, 40.0])
    assert np.allclose(DV.poisson_expected_deviance(m), DV.nb_expected_deviance(np.full(4, 1e4), m),
                       rtol=5e-3)


def test_isotonic_fit_is_monotone_in_mu():
    rng = np.random.default_rng(0)
    mu = rng.gamma(1.0, 2.0, 5000)
    y = rng.poisson(mu).astype(float)
    lam = DV.isotonic_fit(mu, y)
    o = np.argsort(mu)
    assert (np.diff(lam[o]) >= -1e-12).all()
    assert lam.mean() == pytest.approx(y.mean(), rel=0.05)


def test_gap_is_zero_at_the_baseline_and_one_at_the_floor():
    rec = {"D0": 10.0, "D_ref": 8.0, "D_model": 8.0, "floor": 4.0}
    assert DV.gap([rec], "D_ref") == 0.0
    assert DV.gap([dict(rec, D_model=4.0)], "D_ref") == 1.0
    assert np.isnan(DV.gap([dict(rec, D_ref=4.0)], "D_ref"))


# ------------------------------------------------------------------ simulation checks
def test_simulation_checks_pass():
    dr = SIM.sim_depth(50000, 1)
    cr = SIM.sim_calibration(50000, 1)
    kr = SIM.sim_correction(50000, 1)
    res = SIM.checks(dr, cr, kr)
    assert {k: v for k, v in res.items() if k.endswith("_pass") and not v} == {}, res


# ------------------------------------------------------------------ scorer switch
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (4, "chipseq-control"))
L, RB, B = 64, 256, 4
KW = dict(n_boot=5, guardrail_windows=4, log=lambda *a: None)
MKW = {k: v for k, v in KW.items() if k != "guardrail_windows"}
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads", "pred_store")
CONFIG = REPO / "tools/minicandi/configs/ablations/predict_control/rung2/predict_control_off.json"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import tables_sha256
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("dev")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = [p for p in ev["correction_pairs"]
                              if p["never_trained"] and not p["subset_pair"]][:4]
    ev["replicate_pairs"] = []
    out.write_text(json.dumps(ev))
    refs = root / "refs"
    for name in S.REFS_FROM_DIRS:
        S.score(out, refs / S.REFS_FROM_DIRS[name], baseline=name, **KW)
    from minicandi import registry as R
    cfg = {**json.loads(CONFIG.read_text()), "context_bins": L, "batch_size": B}
    torch.manual_seed(2)
    m = R.build_model(cfg)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return {"root": root, "evalset": out, "refs": refs, "sha": tables_sha256(t),
            "model": m.eval(), "config": cfg}


def _scores(d):
    s = json.loads((d / "scores.json").read_text())
    for k in VOLATILE:
        s.pop(k, None)
    return s


def _check(off, on):
    assert "deviance" not in off
    blk = on.pop("deviance")
    assert off == on                     # every other key is unchanged
    for k in ("dev_imputation_skill", "dev_correction_gap_closed", "dev_calibration_ratio",
              "dev_imputation_floor", "dev_correction_depth_only"):
        assert blk[k]["macro"] is not None and np.isfinite(blk[k]["macro"]), (k, blk[k])
    assert blk["n_pairs"] > 0 and blk["n_targets"] > 0
    assert 0 < blk["dev_imputation_floor"]["macro"] < 1


def test_switch_adds_only_the_deviance_key_in_process(env):
    for name, dev in (("off", False), ("on", True)):
        S.score(env["evalset"], env["root"] / f"ip_{name}", model=env["model"], config=env["config"],
                run_tables_sha256=env["sha"], deviance=dev, **KW)
    _check(_scores(env["root"] / "ip_off"), _scores(env["root"] / "ip_on"))


def test_switch_adds_only_the_deviance_key_on_the_store_path(env):
    from onion_eval import predict_store as P
    sc = S.Scorer(env["evalset"], main="model", model=env["model"], config=env["config"],
                  run_tables_sha256=env["sha"], **KW)
    store = env["root"] / "store"
    P.predict(sc, P.requests(sc), store, B)
    for name, dev in (("off", False), ("on", True)):
        S.score(env["evalset"], env["root"] / f"m_{name}", phase="metrics", pred_store=store,
                config=env["config"], run_tables_sha256=env["sha"], refs_from=env["refs"],
                deviance=dev, **MKW)
    _check(_scores(env["root"] / "m_off"), _scores(env["root"] / "m_on"))
