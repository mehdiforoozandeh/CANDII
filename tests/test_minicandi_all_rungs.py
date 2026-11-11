"""miniCANDI, all six rungs together (chunk RC): one registration form, one two_step refusal, one
scramble rule, real wiring checks, the support matrix, and an end-to-end pass of every rung through
the train CLI, `load_run` and the merged copy-leak guardrail."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.components import (CountTransform, MaskHandler, NBHead,  # noqa: E402
                                  NotApplicable, TRANSFORMS)
from minicandi.covariates import CovariateEmbedding, scramble_covariates  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402
from onion_eval import guardrail as G  # noqa: E402

L = 64
RUNGS = sorted(R.RUNGS)
SHARED = ("check_rung", "scramble_mode", "model_covariates")


def _base(r: int) -> dict:
    return json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())


def _cfg(r: int, **over) -> dict:
    return {**_base(r), "context_bins": L, **over}


def _perturb(model, seed: int = 0):
    """Noise on every parameter: the NB head starts at zero, so an untrained model ignores its
    inputs and a wiring check would pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


def _bypass_mask(monkeypatch):
    monkeypatch.setattr(MaskHandler, "visible", staticmethod(lambda a, mk: torch.ones_like(mk)))


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_all"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    b = OnionDataset(tables, "train", _cfg(1), 0).batch(4, np.random.default_rng(0))
    C.validate_batch(b)
    assert b["x_bin_mask"].any() and (~b["x_avail"]).any()
    return b


# ------------------------------------------------------------------------- (c) registration


@pytest.mark.parametrize("r", RUNGS)
def test_every_rung_exposes_model_rung_and_name_and_registers_nothing(r):
    mod = R.rung_module(r)
    assert mod.RUNG == r and mod.NAME == R.RUNGS[r]
    assert isinstance(mod.Model, type) and issubclass(mod.Model, C.MiniCandi)
    assert not hasattr(mod, "build")
    src = inspect.getsource(mod)
    assert "register(" not in src and "import registry" not in src
    for rung in (r, R.RUNGS[r]):                    # the registry accepts the number or the name
        assert type(R.build_model(_cfg(r, rung=rung))) is mod.Model


@pytest.mark.parametrize("r", RUNGS)
def test_another_rungs_config_is_refused(r):
    other = 1 + r % 6
    with pytest.raises(ValueError):
        R.rung_module(r).Model(_cfg(r, rung=other))


def test_the_shared_rule_is_identical_in_all_six_modules():
    mods = [R.rung_module(r) for r in RUNGS]
    for fn in SHARED:
        srcs = {inspect.getsource(getattr(m, fn)) for m in mods}
        assert len(srcs) == 1, fn
    assert len({m.TWO_STEP_MESSAGE for m in mods}) == 1
    assert len({m.SCRAMBLE_SEED for m in mods}) == 1


# ------------------------------------------------------------------------- (d) two_step


@pytest.mark.parametrize("r", RUNGS)
def test_two_step_is_not_applicable_with_one_message(r):
    mod = R.rung_module(r)
    with pytest.raises(NotApplicable) as e:
        R.build_model(_cfg(r, two_step=True))
    assert str(e.value) == mod.TWO_STEP_MESSAGE
    assert "later ablation on the winning design" in mod.TWO_STEP_MESSAGE


# ------------------------------------------------------------------------- (e) base configs


def test_base_configs_share_the_placeholder_budget_and_scramble_off():
    for r in RUNGS:
        b = _base(r)
        assert b["rung"] == r and b["windows_seen"] == 200000
        assert b["scramble_covariates"] is False and b["scramble_all"] is False
        assert b["two_step"] is False
        assert b["lr"] > 0 and b["batch_size"] > 0
    lr = {r: _base(r)["lr"] for r in RUNGS}
    bs = {r: _base(r)["batch_size"] for r in RUNGS}
    assert lr == {1: 1e-3, 2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 3e-4, 6: 3e-4}, lr
    assert set(bs.values()) == {8}, bs


# ------------------------------------------------------------------------- (b) scrambling


def _spy(model):
    """Record what the covariate encoders, the count transform and the NB head receive."""
    seen = {"embed": [], "depth_in": [], "c_out_head": []}
    hooks = []
    for mod in model.modules():
        if isinstance(mod, CovariateEmbedding):
            hooks.append(mod.register_forward_pre_hook(
                lambda m, a: seen["embed"].append(a[0].detach().clone())))
        elif isinstance(mod, CountTransform):
            hooks.append(mod.register_forward_pre_hook(
                lambda m, a: seen["depth_in"].append(a[1].detach().clone())))
        elif isinstance(mod, NBHead):
            hooks.append(mod.register_forward_pre_hook(
                lambda m, a: seen["c_out_head"].append(a[1].detach().clone())))
    return seen, hooks


def _reference(batch, seed):
    g = torch.Generator().manual_seed(seed)
    return (scramble_covariates(batch["c_in"], batch["x_avail"], g),
            scramble_covariates(batch["c_out"], batch["y_avail"], g))


@pytest.mark.parametrize("mode", ["scramble_covariates", "scramble_all"])
@pytest.mark.parametrize("r", RUNGS)
def test_scrambling_is_applied_once_and_keeps_the_true_depth_unless_scramble_all(r, mode, batch):
    """Eval mode: every covariate encoder input equals ONE scramble of C or C' (seeded as the
    model seeds it; not the original, not a second scramble). The count transform and the NB head
    read the true depth under scramble_covariates and the same scrambled depth under
    scramble_all. The outputs move against the unscrambled model with the same weights."""
    mod = R.rung_module(r)
    torch.manual_seed(0)
    plain = _perturb(R.build_model(_cfg(r)))
    m = R.build_model(_cfg(r, **{mode: True})).eval()
    m.load_state_dict(plain.state_dict())
    ref_in, ref_out = _reference(batch, mod.SCRAMBLE_SEED)
    assert not torch.equal(ref_in, batch["c_in"]) and not torch.equal(ref_out, batch["c_out"])
    twice_in = scramble_covariates(ref_in, batch["x_avail"],
                                   torch.Generator().manual_seed(mod.SCRAMBLE_SEED))
    assert not torch.equal(twice_in, ref_in)

    seen, hooks = _spy(m)
    with torch.no_grad():
        out = m(batch)
    for h in hooks:
        h.remove()
    B, A = batch["c_in"].shape[:2]
    got = [e.reshape(B, A, e.shape[-1]) for e in seen["embed"]]
    assert got, "no covariate encoder was called"
    assert any(torch.equal(e, ref_in) for e in got)
    assert any(torch.equal(e, ref_out) for e in got)
    for e in got:
        assert torch.equal(e, ref_in) or torch.equal(e, ref_out), "encoder input not scrambled once"
    want_in = (ref_in if mode == "scramble_all" else batch["c_in"])[..., C.LOG2_DEPTH_COL]
    want_out = (ref_out if mode == "scramble_all" else batch["c_out"])[..., C.LOG2_DEPTH_COL]
    assert seen["depth_in"] and all(torch.equal(d, want_in) for d in seen["depth_in"])
    assert seen["c_out_head"]
    assert all(torch.equal(c[..., C.LOG2_DEPTH_COL], want_out) for c in seen["c_out_head"])

    with torch.no_grad():
        ref = plain(batch)
    assert not torch.allclose(out["mu"], ref["mu"])
    if mode == "scramble_all":
        cov = R.build_model(_cfg(r, scramble_covariates=True)).eval()
        cov.load_state_dict(plain.state_dict())
        with torch.no_grad():
            assert not torch.allclose(out["mu"], cov(batch)["mu"])


@pytest.mark.xfail(strict=True, reason="train.py (not owned by chunk RC) scrambles c_in and c_out "
                   "itself, so a scrambling config is scrambled twice in training and the count "
                   "transform loses the true depth; delete the scramble block in train.train")
def test_the_train_loop_leaves_scrambling_to_the_model(tables, tmp_path, monkeypatch):
    calls = []
    real = T.scramble_covariates
    monkeypatch.setattr(T, "scramble_covariates", lambda *a, **k: calls.append(1) or real(*a, **k))
    T.train(_cfg(1, scramble_covariates=True, batch_size=2, windows_seen=2, param_budget=1.4e5),
            tables, 0, tmp_path / "run")
    assert not calls


# ------------------------------------------------------------------------- (a) wiring, (f) matrix


@pytest.mark.parametrize("r", RUNGS)
def test_wiring_check_is_real_on_every_rung(r, batch, monkeypatch):
    torch.manual_seed(0)
    m = _perturb(R.build_model(_cfg(r)))
    assert C.wiring_check(m, batch)["pass"]
    _bypass_mask(monkeypatch)
    assert not C.wiring_check(m, batch)["pass"]


ARMS = {**{f"covariate_injection={m}": {"covariate_injection": m}
           for m in ("concat", "film", "tokens")},
        **{f"mask_mode={m}": {"mask_mode": m} for m in ("token", "drop")},
        **{f"dna={m}": {"dna": m} for m in ("own", "off", "shuffled")},
        **{f"count_transform={t}": {"count_transform": t} for t in TRANSFORMS},
        **{f"{k}=true": {k: True} for k in ("qc_on_input", "average_input", "control_input",
                                            "predict_control")}}
#: per rung, the arms whose construction raises NotApplicable; every other arm builds
NOT_APPLICABLE = {1: set(), 2: {"covariate_injection=tokens"}, 3: set(), 4: set(),
                  5: {"average_input=true"}, 6: set()}


def support_matrix() -> dict:
    """{rung: {arm: "ok" | "NotApplicable"}}, read by constructing each config."""
    out = {}
    for r in RUNGS:
        out[r] = {}
        for arm, over in ARMS.items():
            try:
                R.build_model(_cfg(r, **over))
                out[r][arm] = "ok"
            except NotApplicable:
                out[r][arm] = "NotApplicable"
    return out


def test_support_matrix():
    got = support_matrix()
    for r in RUNGS:
        na = {a for a, v in got[r].items() if v == "NotApplicable"}
        assert na == NOT_APPLICABLE[r], (r, na)


# ------------------------------------------------------------------------- end to end


@pytest.mark.parametrize("r", RUNGS)
def test_end_to_end_train_cli_load_run_and_guardrail(r, tables, tmp_path, monkeypatch):
    """The train CLI for 4 windows, then `load_run` reproduces the trained model's outputs
    exactly, FLOPs per window are logged, and the merged guardrail's part 1 (run through
    `load_run`) passes on the perturbed model and fails with the mask bypassed."""
    cfg = _cfg(r, batch_size=2)
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    built = []
    monkeypatch.setattr(T, "build_model", lambda c: built.append(R.build_model(c)) or built[-1])
    run = tmp_path / "run"
    assert T.main(["--config", str(tmp_path / "cfg.json"), "--tables", str(tables), "--seed", "0",
                   "--out", str(run), "--max-windows", "4", "--log-every", "1"]) == 0
    info = json.loads((run / "run.json").read_text())
    assert info["windows_seen"] == 4 and info["done"] is False and not (run / "DONE").exists()
    assert info["flops_per_window"] > 0 and info["flops_per_window_forward"] > 0
    print(f"rung {r} {R.RUNGS[r]}: params {info['param_count']}, training FLOPs per window "
          f"{info['flops_per_window']:.3e} at L = {L}")

    model, config = R.load_run(run)
    assert config == T.full_config(cfg)
    vb = [b for _, b in zip(range(2), OnionDataset(tables, "val", config, 0))]
    trained = built[0].eval()
    with torch.no_grad():
        for b in vb:
            a, o = trained(b), model(b)
            assert torch.equal(a["n"], o["n"]) and torch.equal(a["mu"], o["mu"])

    ck = torch.load(run / R.CKPT_NAME, weights_only=False)
    _perturb(model)
    pert = tmp_path / "perturbed"
    pert.mkdir()
    torch.save(C.make_ckpt(model, config, ck["windows_seen"], ck["tables_sha256"],
                           ck["git_sha"]), pert / R.CKPT_NAME)
    args = ["--run", str(pert), "--tables", str(tables), "--max-batches", "2"]
    G.main(args + ["--out", str(tmp_path / "leak.json")])
    res = json.loads((tmp_path / "leak.json").read_text())
    assert res["part1_wiring_pass"], res
    assert res["n_hidden_bins"] > 0 and res["n_unavailable_bins"] > 0
    _bypass_mask(monkeypatch)
    G.main(args + ["--out", str(tmp_path / "leak_bypassed.json")])
    assert not json.loads((tmp_path / "leak_bypassed.json").read_text())["part1_wiring_pass"]
