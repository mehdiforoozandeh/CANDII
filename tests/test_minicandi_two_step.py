"""miniCANDI two-step correction (chunk A1, decision D15) and the registry's wrapper order and
`model_file` hook (plan pinned interface E, "Wrappers" and "Two-step correction")."""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import minicandi  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi import two_step as TS  # noqa: E402
from minicandi.components import NotApplicable  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
RUNGS = (1, 2, 5)


def _cfg(r: int, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, param_budget=1.5e5, batch_size=4)
    cfg.update(over)
    return cfg


def _perturb(model, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_two_step"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    b = OnionDataset(tables, "train", _cfg(1, two_step=True), 0).batch(
        4, np.random.default_rng(0))
    C.validate_batch(b)
    assert "c_base" in b and "y_base" in b
    assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
    assert (b["c_base"][..., 0] != 0).any()
    return b


# ------------------------------------------------------------------- the wrapper on real rungs


@pytest.mark.parametrize("r", RUNGS)
def test_registry_wraps_and_the_wiring_check_passes_with_a_perturbed_head(r, batch):
    torch.manual_seed(0)
    m = R.build_model(_cfg(r, two_step=True))
    assert isinstance(m, TS.TwoStep) and type(m.inner) is R.rung_module(r).Model
    assert m.param_count() == m.inner.param_count()          # shared weights, no new parameter
    _perturb(m)
    out = m(batch)
    assert set(out) == {"n", "mu", "n_base", "mu_base"}
    assert not torch.allclose(out["mu"], out["mu_base"])     # leg 2 is not leg 1
    w = C.wiring_check(m, batch)
    assert w["pass"] and w["n_perturbed"] > 0, w


@pytest.mark.parametrize("r", RUNGS)
def test_no_output_reads_y_base_or_y_counts(r, batch):
    torch.manual_seed(0)
    m = _perturb(R.build_model(_cfg(r, two_step=True)))
    g = torch.Generator().manual_seed(1)
    pert = dict(batch)
    pert["y_base"] = torch.randint(0, 500, batch["y_base"].shape, generator=g, dtype=torch.int32)
    pert["y_counts"] = torch.randint(0, 500, batch["y_counts"].shape, generator=g,
                                     dtype=torch.int32)
    with torch.no_grad():
        a, b = m(batch), m(pert)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_two_step_with_qc_on_input_is_not_applicable():
    with pytest.raises(NotApplicable, match="QC"):
        R.build_model(_cfg(1, two_step=True, qc_on_input=True))


def test_missing_c_base_is_an_error(batch):
    torch.manual_seed(0)
    m = R.build_model(_cfg(1, two_step=True)).eval()
    b = {k: v for k, v in batch.items() if k != "c_base"}
    with pytest.raises(KeyError, match="c_base"):
        m(b)


# ------------------------------------------------------------------- hand-computed legs


class _Stub(C.MiniCandi):
    """mu = exp(a) + 0.5 * x_counts (unmasked, so legs are visible), n = 1; records each call."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.3))
        self.calls = []

    def forward(self, batch):
        self.calls.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                           for k, v in batch.items()})
        mu = torch.exp(self.a) + 0.5 * batch["x_counts"].float()
        return {"n": torch.ones_like(mu), "mu": mu}


def test_legs_by_hand(batch):
    b = dict(batch)
    c_base = b["c_base"].clone()
    t0 = int(b["y_avail"][0].nonzero()[0])
    c_base[0, t0] = 0.0                        # one target track without a base view
    b["c_base"] = c_base
    stub = _Stub()
    out = TS.wrap(stub, {})(b)
    leg1, leg2 = stub.calls
    has = c_base[..., 0] != 0
    want_c1 = torch.where(has[..., None], c_base, b["c_out"])
    assert torch.equal(leg1["c_out"], want_c1)
    assert torch.equal(leg1["c_out"][0, t0], b["c_out"][0, t0])       # the fallback
    for k in ("x_counts", "x_avail", "x_bin_mask", "c_in"):
        assert torch.equal(leg1[k], b[k]), k
    tgt = b["y_avail"]
    mu1 = torch.exp(torch.tensor(0.3)) + 0.5 * b["x_counts"].float()
    assert torch.allclose(out["mu_base"], mu1)
    assert torch.allclose(leg2["x_counts"], mu1 * tgt[:, None, :])
    assert leg2["x_counts"].dtype == torch.float32
    assert torch.equal(leg2["x_avail"], tgt)
    assert not leg2["x_bin_mask"].any()
    assert torch.equal(leg2["c_in"], want_c1 * tgt[..., None])
    assert torch.equal(leg2["c_out"], b["c_out"])
    # mu2 = e^a + 0.5 * tgt * (e^a + 0.5 x): d sum(mu2)/da = e^a * (N + 0.5 * #target bins)
    # when mu_base is NOT detached (it would be e^a * N if it were)
    out["mu"].sum().backward()
    N = out["mu"].numel()
    n_tgt = int(tgt.sum()) * L
    want = float(np.exp(0.3)) * (N + 0.5 * n_tgt)
    assert stub.a.grad.item() == pytest.approx(want, rel=1e-5)


# ------------------------------------------------------------------- train, load_run, FLOPs


@pytest.mark.parametrize("r", RUNGS)
def test_load_run_round_trips_a_two_step_run_and_flops_double(r, tables, tmp_path):
    runs = {}
    for name, two in (("plain", False), ("two", True)):
        info = T.train(_cfg(r, two_step=two, windows_seen=8), tables, 0, tmp_path / name,
                       log_every=1)
        runs[name] = info
    ratio = runs["two"]["flops_per_window"] / runs["plain"]["flops_per_window"]
    assert 1.6 < ratio < 2.4, ratio
    fwd = runs["two"]["flops_per_window_forward"] / runs["plain"]["flops_per_window_forward"]
    assert 1.9 < fwd < 2.1, fwd
    rec = [json.loads(x) for x in (tmp_path / "two/log.jsonl").read_text().splitlines()]
    assert "base" in rec[-1]["parts"] and "main" in rec[-1]["parts"]
    model, cfg = R.load_run(tmp_path / "two")
    assert cfg["two_step"] is True and isinstance(model, TS.TwoStep)
    b = OnionDataset(tables, "train", cfg, 3).batch(2, np.random.default_rng(3))
    ck = torch.load(tmp_path / "two/ckpt.pt", weights_only=False)
    torch.manual_seed(5)
    again = R.build_model(cfg)
    again.load_state_dict(ck["state_dict"])
    again.eval()
    with torch.no_grad():
        o1, o2 = model(b), again(b)
    for k in o1:
        assert torch.equal(o1[k], o2[k]), k


# ------------------------------------------------------------------- wrapper order, model_file


def test_control_head_then_two_step_order_with_a_stub_module(monkeypatch):
    seen = []

    class _CH(C.MiniCandi):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, batch):
            return self.inner(batch)

    def wrap(inner, config):
        seen.append((type(inner), dict(config)))
        return _CH(inner)

    stub = types.ModuleType("minicandi.control_head")
    stub.wrap = wrap
    monkeypatch.setitem(sys.modules, "minicandi.control_head", stub)
    monkeypatch.setattr(minicandi, "control_head", stub, raising=False)

    built = []
    real = R.rung_module(1).Model

    class Spy(real):
        def __init__(self, config):
            built.append(dict(config))
            super().__init__(config)

    monkeypatch.setattr(R, "model_class", lambda config: Spy)
    torch.manual_seed(0)
    m = R.build_model(_cfg(1, control_head="lambda_smooth", control_input=False,
                           two_step=True))
    assert isinstance(m, TS.TwoStep) and isinstance(m.inner, _CH)
    assert isinstance(m.inner.inner, Spy)
    assert len(seen) == 1 and seen[0][1]["control_head"] == "lambda_smooth"
    assert "two_step" not in built[-1] and "control_head" not in built[-1]
    seen.clear()
    m = R.build_model(_cfg(1, control_head="channel"))
    assert isinstance(m, Spy) and not seen                   # channel: no wrapper


def test_model_file_loads_class_model_from_a_copied_rung_module(tables, tmp_path):
    src = R.RUNGS_DIR / "conv_ae.py"
    dst = tmp_path / "hybrid_copy.py"
    shutil.copy(src, dst)
    cfg = _cfg(2, model_file=str(dst))
    torch.manual_seed(0)
    m = R.build_model(cfg)
    assert type(m).__module__.startswith("minicandi_model_file.hybrid_copy_")
    assert type(m) is not R.rung_module(2).Model
    torch.manual_seed(0)
    ref = R.build_model(_cfg(2))
    sa, sb = m.state_dict(), ref.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    # a path relative to the repository root works too
    rel = R.build_model(_cfg(2, model_file="tools/minicandi/rungs/conv_ae.py"))
    assert type(rel).__module__.startswith("minicandi_model_file.conv_ae_")
    # load_run follows model_file, also under two_step
    info = T.train(_cfg(2, model_file=str(dst), two_step=True, windows_seen=4), tables, 0,
                   tmp_path / "run")
    assert info["done"]
    model, saved = R.load_run(tmp_path / "run")
    assert saved["model_file"] == str(dst)
    assert isinstance(model, TS.TwoStep)
    assert type(model.inner).__module__.startswith("minicandi_model_file.hybrid_copy_")
    with pytest.raises(ModuleNotFoundError):
        R.build_model(_cfg(2, model_file=str(tmp_path / "absent.py")))
