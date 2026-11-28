"""t370 chunk C1: trunk FiLM placement (rungs 2, 3, 4) and the U-Net skip switches.

(1) Defaults are today's model: the base configs build 1,042,514 / 1,038,018 / 993,599
    parameters, and with the new keys absent or at their defaults each rung has the state_dict,
    parameter count and outputs of the rung module before this chunk (`git show BASE_COMMIT`).
(2) Every non-default value builds within 10% of `param_budget`, starts as the identity where the
    plan says so (trunk FiLM, skip_film = c_cprime, head_skip = film), changes the output after
    one optimizer step on its own parameters, passes `contract.wiring_check`, and is accepted by
    `train.check_config_keys`.
(3) The NotApplicable cases.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools" / "t118", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.components import NotApplicable  # noqa: E402
from minicandi.rungs import conv_ae, conv_attn_ae, unet  # noqa: E402
from minicandi.trunk_film import PLACEMENTS, TrunkFiLM, pooled_condition  # noqa: E402

#: the umbrella commit this chunk was cut from: its rung modules are "the code before the change"
BASE_COMMIT = "57f3c9e8"
L = 64
MODS = {2: conv_ae, 3: conv_attn_ae, 4: unet}
REF_PARAMS = {2: 1_042_514, 3: 1_038_018, 4: 993_599}
EXPLICIT = {2: {"film_placement": "ends"}, 3: {"film_placement": "ends"},
            4: {"film_placement": "ends", "skip_film": "cprime", "head_skip": "plain"}}
NEW_MODULES = ("trunk_film", "head_skip_film")     # built only by a non-default value


def _base(r: int) -> dict:
    return json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base_heldout.json").read_text())


def _cfg(r: int, **over) -> dict:
    return {**_base(r), "context_bins": L, **over}


def _arms():
    for r in (2, 3, 4):
        for p in PLACEMENTS[1:]:
            yield r, {"film_placement": p}
    yield 4, {"skip_film": "c_cprime"}
    yield 4, {"head_skip": "off"}
    yield 4, {"head_skip": "film"}
    yield 4, {"film_placement": "all", "skip_film": "c_cprime", "head_skip": "film"}


ARMS = list(_arms())


def _id(v):
    return "_".join(f"{k}={x}" for k, x in v.items()) if isinstance(v, dict) else f"rung{v}"


def _batch(K: int, B: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [0, 3, 7, 15, 21, 35]] = True
    x_avail[1, 9] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 3] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, [1, 16, 30]] = C.KIND_IMPUTE
    y_kind[1, 2] = C.KIND_IMPUTE
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[:, ::3, 3] = True
    bm[:, 5:9, 0] = True
    bm[:, 20:40, 7] = True
    x = torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0

    def cov(avail, k):
        v = torch.rand(B, C.A, k, generator=g)
        c = torch.zeros(B, C.A, k)
        c[..., 0] = 20.0 + 8.0 * v[..., 0]
        c[..., 1] = (v[..., 1] > 0.5).float()
        c[..., 2] = 36.0 + 100.0 * v[..., 2]
        c[..., 3] = (v[..., 3] > 0.5).float()
        c[..., 4] = 30.0 * v[..., 4]
        c[..., 5:C.K_META] = torch.randint(1, 6, (B, C.A, C.K_META - 5), generator=g).float()
        if k > C.K_META:
            c[..., C.K_META:] = v[..., C.K_META:]
        return torch.where(avail[..., None], c, torch.zeros_like(c))

    y_avail = y_kind > 0
    return {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
            "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
            "c_in": cov(x_avail, K), "c_out": cov(y_avail, C.K_META),
            "y_counts": torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32),
            "y_avail": y_avail, "y_kind": y_kind,
            "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}


def _perturb(model, seed: int = 0):
    """Noise on every parameter (the NB head starts at zero, so an untrained model ignores its
    inputs and a wiring check or a gradient check would pass vacuously)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def _out(model, batch) -> tuple:
    model.eval()
    with torch.no_grad():
        o = model(batch)
    return o["n"], o["mu"]


def _build(r: int, cfg: dict, width: int | None = None):
    torch.manual_seed(0)
    if width is None:
        return MODS[r].Model(cfg)
    if r == 4:
        return unet.Model({**cfg, "width": width})
    return MODS[r].Model(cfg, width=width)


def _new_params(model) -> list:
    steer_new = getattr(model, "skip_film", None) == "c_cprime"
    return [p for n, p in model.named_parameters()
            if n.split(".")[0] in NEW_MODULES or (steer_new and n.split(".")[0] == "steer")]


# ------------------------------------------------------------------ (1) defaults are today's


@pytest.mark.parametrize("r", [2, 3, 4], ids=_id)
@pytest.mark.parametrize("explicit", [False, True])
def test_base_configs_keep_their_parameter_counts(r, explicit):
    cfg = {**_base(r), **(EXPLICIT[r] if explicit else {})}
    torch.manual_seed(0)
    assert R.build_model(cfg).param_count() == REF_PARAMS[r]


@pytest.fixture(scope="module")
def old_modules(tmp_path_factory) -> dict:
    """{rung: the rung module as of BASE_COMMIT}, loaded from `git show` under a private name."""
    out, d = {}, tmp_path_factory.mktemp("pre_c1")
    for r, mod in MODS.items():
        rel = f"tools/minicandi/rungs/{mod.NAME}.py"
        res = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_COMMIT}:{rel}"],
                             capture_output=True, text=True)
        if res.returncode:
            pytest.skip(f"git show {BASE_COMMIT} unavailable here: {res.stderr.strip()[:80]}")
        path = d / f"pre_c1_{mod.NAME}.py"
        path.write_text(res.stdout)
        spec = importlib.util.spec_from_file_location(f"pre_c1_{mod.NAME}", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out[r] = m
    return out


@pytest.mark.parametrize("r", [2, 3, 4], ids=_id)
@pytest.mark.parametrize("explicit", [False, True])
def test_defaults_give_the_pre_change_model_to_zero_difference(r, explicit, old_modules):
    torch.manual_seed(0)
    old = old_modules[r].Model(_cfg(r))
    new = _build(r, _cfg(r, **(EXPLICIT[r] if explicit else {})))
    so, sn = old.state_dict(), new.state_dict()
    assert [(k, tuple(v.shape)) for k, v in so.items()] == \
        [(k, tuple(v.shape)) for k, v in sn.items()]
    assert all(torch.equal(so[k], sn[k]) for k in so)
    assert old.param_count() == new.param_count()
    _perturb(old), _perturb(new)
    b = _batch(C.k_input(_cfg(r)))
    assert all(torch.equal(x, y) for x, y in zip(_out(old, b), _out(new, b)))


# ------------------------------------------------------------------ (2) every non-default value


def test_trunk_film_starts_as_identity_and_pools_available_tracks():
    tf = TrunkFiLM(8, [3, 5])
    for gamma, beta in tf(torch.randn(4, 8)):
        assert torch.equal(gamma, torch.ones_like(gamma))
        assert torch.equal(beta, torch.zeros_like(beta))
    assert [tuple(g.shape) for g, _ in tf(torch.randn(4, 8))] == [(4, 3, 1), (4, 5, 1)]
    e_in, e_out = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    xa = torch.tensor([[True, False, True, False], [False] * 4])
    ya = torch.tensor([[False, True, False, False], [True] * 4])
    c = pooled_condition(e_in, xa, e_out, ya)
    assert torch.allclose(c[0, :3], e_in[0, [0, 2]].mean(0))
    assert torch.equal(c[1, :3], torch.zeros(3))                   # no available input track
    assert torch.allclose(c[0, 3:], e_out[0, 1])
    assert torch.allclose(c[1, 3:], e_out[1].mean(0))


@pytest.mark.parametrize("r,arm", ARMS, ids=_id)
def test_arm_builds_within_budget(r, arm):
    cfg = _cfg(r, **arm)
    m = _build(r, cfg)
    assert abs(m.param_count() - cfg["param_budget"]) <= 0.10 * cfg["param_budget"]
    assert any(n.split(".")[0] in NEW_MODULES for n, _ in m.named_parameters()) \
        or arm.get("film_placement", "ends") == "ends"
    if arm.get("head_skip") == "off":
        assert not hasattr(m, "head_skip")
    T.check_config_keys(cfg)


def _pair(r: int, arm: dict):
    """(today's model perturbed, the arm at the same width with every shared parameter copied
    from it and its new modules at their initial values)."""
    base = _build(r, _cfg(r))
    _perturb(base)
    if arm.get("skip_film") == "c_cprime":        # steer is the identity at the start in the arm
        with torch.no_grad():
            for g in base.steer:
                g.net[-1].weight.zero_()
    m = _build(r, _cfg(r, **arm), width=base.width)
    sb, sm = base.state_dict(), m.state_dict()
    keep = {k: v for k, v in sb.items() if k in sm and sm[k].shape == v.shape}
    m.load_state_dict({**sm, **keep})
    return base, m


@pytest.mark.parametrize("r,arm", [a for a in ARMS if a[1].get("head_skip") != "off"], ids=_id)
def test_arm_starts_as_identity(r, arm):
    base, m = _pair(r, arm)
    b = _batch(C.k_input(_cfg(r)))
    for x, y in zip(_out(base, b), _out(m, b)):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-6), (x - y).abs().max()


@pytest.mark.parametrize("r,arm", ARMS, ids=_id)
def test_one_step_changes_the_output_and_wiring_holds(r, arm):
    _, m = _pair(r, arm)
    b = _batch(C.k_input(_cfg(r)))
    before = _out(m, b)
    params = _new_params(m) or list(m.parameters())              # head_skip = off adds nothing
    opt = torch.optim.Adam(params, lr=1e-2)
    m.train()
    o = m(b)
    loss = o["mu"].log().mean() + o["n"].log().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    opt.step()
    after = _out(m, b)
    assert not all(torch.equal(x, y) for x, y in zip(before, after))
    _perturb(m, seed=1)
    res = C.wiring_check(m.eval(), b)
    assert res["pass"], res


# ------------------------------------------------------------------ (3) NotApplicable


@pytest.mark.parametrize("r,arm", [
    (2, {"skip_film": "c_cprime"}), (2, {"head_skip": "off"}), (3, {"skip_film": "c_cprime"}),
    (3, {"head_skip": "film"}),
    (3, {"covariate_injection": "tokens", "film_placement": "encoder"}),
    (4, {"covariate_injection": "tokens", "film_placement": "bottleneck"}),
    (4, {"skip_source": "dna", "skip_film": "c_cprime"}),
    (4, {"skip_source": "dna", "head_skip": "off"}),
    (4, {"skip_source": "dna", "head_skip": "film"}),
], ids=_id)
def test_not_applicable(r, arm):
    with pytest.raises(NotApplicable):
        _build(r, _cfg(r, **arm))


@pytest.mark.parametrize("r,arm", [(2, {"film_placement": "middle"}),
                                   (4, {"skip_film": "c"}), (4, {"head_skip": "on"})], ids=_id)
def test_bad_values_refused(r, arm):
    with pytest.raises(ValueError):
        _build(r, _cfg(r, **arm))
