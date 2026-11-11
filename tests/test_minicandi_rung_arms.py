"""miniCANDI rung support for the stage-2 arm keys (chunk RX; pinned interface E, "Rung support").

(a) Defaults are today's model: on every rung, a config without the new keys (and one that names
    them at their defaults) builds the same state_dict, layout and values, and gives the same
    outputs, to 0 difference, as the rung module of the tree before this chunk (`git show BASE`).
(b) Every new arm key builds on every rung (the NotApplicable table below is empty), lands within
    10% of `param_budget`, passes `contract.wiring_check` with a perturbed head (and fails it with
    the mask bypassed, so the check is not vacuous), and is read: each arm has its own assertion.
(c) Keys that act in the dataset, the loss or the wrappers leave the rung's network unchanged.
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
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi.components import (CountTransform, CovariateEmbedding,  # noqa: E402
                                  CovariateInjector, DnaTower, MaskHandler, NBHead)

#: the umbrella commit this chunk was cut from: its rung modules are "the code before the change"
BASE_COMMIT = "3c689a2"
L = 64
D_EMB = 128                        # dna_embedding_dim after PCA (chunk A5)
RUNGS = sorted(R.RUNGS)
CAT_COLS = list(range(5, C.K_META))  # the categorical id columns of C and C'

#: the arm keys of pinned interface E that act inside the rung
NEW_ARMS = {
    "inject_adaln_zero": {"covariate_injection": "adaln_zero"},
    "cov_dropout": {"covariate_dropout": 0.3},
    "cov_numeric_only": {"covariate_embedding": "numeric_only"},
    "dna_precomputed": {"dna": "precomputed", "dna_embedding_dim": D_EMB},
    "offset_eps0.1": {"offset_eps": 0.1},
    "tf_soft_clip": {"count_transform": "soft_clip"},
    "head_shared_size": {"head": "nb_shared_size"},
    "average_input": {"average_input": True},
}
#: (rung, arm) -> the reason the rung refuses it. Every rung reads every key, so it is empty.
NOT_APPLICABLE: dict = {}

#: keys that act in the dataset, the loss, or the registry's wrappers; the rung ignores them
OUTSIDE_KEYS = {"input_depth": "fixed", "mask_scheme": "dirichlet", "bin_mask_shape": "span",
                "train_on_validation": True, "final_model": True, "dna_shift_bp": 5_000_000,
                "dna_embedding_h5": "/nonexistent/emb", "qc_table": "/nonexistent/qc.tsv",
                "loss": "nb_crps", "loss_weights": {"impute": 4, "correct": 1, "same": 1,
                                                    "base": 1},
                "base_depth": "fixed", "control_input": False, "predict_control": False}

#: arms of the tree before this chunk, checked against it besides the base (a)
OLD_ARMS = {
    1: [{"covariate_injection": "tokens"}, {"mask_mode": "token"}, {"dna": "off"},
        {"average_input": True}],
    2: [{"covariate_injection": "concat"}, {"mask_mode": "drop"}, {"dna": "off"},
        {"average_input": True}],
    3: [{"covariate_injection": "tokens"}, {"mask_mode": "drop"}, {"dna": "shuffled"},
        {"average_input": True}],
    4: [{"covariate_injection": "tokens"}, {"skip_source": "dna"}, {"dna": "off"},
        {"average_input": True}],
    5: [{"covariate_injection": "film"}, {"mask_mode": "token"}, {"dna": "off"},
        {"axis": "tied_position"}],
    6: [{"covariate_injection": "concat"}, {"mask_mode": "token"}, {"dna": "off"},
        {"average_input": True}],
}
#: the new keys named at their defaults
EXPLICIT_DEFAULTS = {"covariate_dropout": 0.0, "covariate_embedding": "full", "offset_eps": 1.0,
                     "head": "nb"}


def _base(r: int) -> dict:
    return json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())


def _cfg(r: int, **over) -> dict:
    return {**_base(r), "context_bins": L, **over}


def _batch(B=2, K=C.K_META, seed=0) -> dict:
    """A contract batch from a fixed generator, with `avg` and `x_dna_emb` (width D_EMB)."""
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [0, 3, 7, 15, 21, 35]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 3] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, [1, 16, 30]] = C.KIND_IMPUTE
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
            "x_dna_emb": torch.randn(B, L, D_EMB, generator=g),
            "avg": 5.0 * torch.rand(B, L, C.A, generator=g),
            "c_in": cov(x_avail, K), "c_out": cov(y_avail, C.K_META),
            "y_counts": torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32),
            "y_avail": y_avail, "y_kind": y_kind,
            "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}


def _perturb(model, seed: int = 0):
    """Noise on every parameter: the NB head starts at zero, so an untrained model would ignore
    its inputs and a wiring check would pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


def _out(model, batch) -> tuple:
    with torch.no_grad():
        o = model(batch)
    return o["n"], o["mu"]


def _same(a: tuple, b: tuple) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _build(r: int, **over):
    torch.manual_seed(0)
    return R.rung_module(r).Model(_cfg(r, **over))


# ------------------------------------------------------------------ (a) defaults are today's


@pytest.fixture(scope="module")
def old_modules(tmp_path_factory) -> dict:
    """{rung: the rung module as of BASE_COMMIT}, loaded from `git show` under a private name."""
    out, d = {}, tmp_path_factory.mktemp("pre_rx")
    for r in RUNGS:
        rel = f"tools/minicandi/rungs/{R.RUNGS[r]}.py"
        res = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_COMMIT}:{rel}"],
                             capture_output=True, text=True)
        if res.returncode:
            pytest.skip(f"git show {BASE_COMMIT} unavailable here: {res.stderr.strip()[:80]}")
        path = d / f"pre_rx_{R.RUNGS[r]}.py"
        path.write_text(res.stdout)
        spec = importlib.util.spec_from_file_location(f"pre_rx_{R.RUNGS[r]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[r] = mod
    return out


def _cases():
    for r in RUNGS:
        yield r, "base", {}
        yield r, "explicit_defaults", dict(EXPLICIT_DEFAULTS)
        for over in OLD_ARMS[r]:
            yield r, "old_" + "_".join(f"{k}={v}" for k, v in over.items()), over


@pytest.mark.parametrize("r,name,over", list(_cases()), ids=lambda v: str(v))
def test_defaults_give_the_pre_change_model_to_zero_difference(r, name, over, old_modules):
    cfg_new = _cfg(r, **over)
    cfg_old = _cfg(r, **{k: v for k, v in over.items() if k not in EXPLICIT_DEFAULTS})
    torch.manual_seed(0)
    old = old_modules[r].Model(cfg_old)
    torch.manual_seed(0)
    new = R.rung_module(r).Model(cfg_new)
    so, sn = old.state_dict(), new.state_dict()
    assert [(k, tuple(v.shape)) for k, v in so.items()] == \
        [(k, tuple(v.shape)) for k, v in sn.items()]
    assert all(torch.equal(so[k], sn[k]) for k in so)
    assert old.param_count() == new.param_count()
    _perturb(old), _perturb(new)
    b = _batch(K=C.k_input(cfg_new))
    assert _same(_out(old, b), _out(new, b))


def test_rung5_average_input_was_refused_before_and_builds_now(old_modules):
    with pytest.raises(R.rung_module(5).NotApplicable):
        old_modules[5].Model(_cfg(5, average_input=True))
    m = _build(5, average_input=True)
    assert hasattr(m, "avg_patch")


# ------------------------------------------------------------------ (b) every new arm, every rung


def _modules(model, kind):
    return [m for m in model.modules() if isinstance(m, kind)]


def _bypass_mask(monkeypatch):
    monkeypatch.setattr(MaskHandler, "visible", staticmethod(lambda a, mk: torch.ones_like(mk)))


@pytest.mark.parametrize("arm", sorted(NEW_ARMS))
@pytest.mark.parametrize("r", RUNGS)
def test_every_new_arm_builds_in_budget_passes_the_wiring_check_and_is_read(r, arm, monkeypatch):
    if (r, arm) in NOT_APPLICABLE:
        with pytest.raises(R.rung_module(r).NotApplicable, match=NOT_APPLICABLE[(r, arm)]):
            _build(r, **NEW_ARMS[arm])
        return
    over = NEW_ARMS[arm]
    m = _build(r, **over)
    budget = float(_cfg(r)["param_budget"])
    assert abs(m.param_count() - budget) <= 0.10 * budget, (r, arm, m.param_count())
    _perturb(m)
    b = _batch()
    out = _out(m, b)
    assert all(torch.isfinite(t).all() and (t > 0).all() for t in out)

    # the key is read
    if arm == "inject_adaln_zero":
        inj = _modules(m, CovariateInjector)
        assert inj and all(i.mode == "adaln_zero" for i in inj)
    elif arm == "cov_dropout":
        emb = _modules(m, CovariateEmbedding)
        assert emb and all(e.null is not None for e in emb)
        seen = []
        hooks = [e.register_forward_pre_hook(lambda mod, a: seen.append(bool(a[0].isnan().any())))
                 for e in emb]
        _out(m, b)
        assert seen and not any(seen)                       # eval mode: nothing dropped
        seen.clear()
        m.train()
        torch.manual_seed(1)
        with torch.no_grad():
            tr = m(b)
        m.eval()
        for h in hooks:
            h.remove()
        assert any(seen)                                    # training: rows dropped
        assert torch.isfinite(tr["mu"]).all() and torch.isfinite(tr["n"]).all()
    elif arm == "cov_numeric_only":
        emb = _modules(m, CovariateEmbedding)
        assert emb and all(e.numeric_only for e in emb)
        b2 = dict(b)
        for key in ("c_in", "c_out"):
            c = b[key].clone()
            c[..., CAT_COLS] = torch.where(c[..., CAT_COLS] > 0, c[..., CAT_COLS] + 3,
                                           c[..., CAT_COLS])
            b2[key] = c
        assert _same(out, _out(m, b2))
    elif arm == "dna_precomputed":
        towers = [t for t in _modules(m, DnaTower) if t.mode != "off"]
        assert towers and all(t.mode == "precomputed" and t.d_emb == D_EMB for t in towers)
        other_seq = dict(b, x_dna=torch.flip(b["x_dna"], dims=[1]))
        assert _same(out, _out(m, other_seq))               # bases are not read
        other_emb = dict(b, x_dna_emb=b["x_dna_emb"] + 1.0)
        assert not _same(out, _out(m, other_emb))           # x_dna_emb is
    elif arm in ("offset_eps0.1", "tf_soft_clip"):
        tf = _modules(m, CountTransform)
        assert tf
        if arm == "offset_eps0.1":
            assert all(t.offset_eps == 0.1 for t in tf)
        else:
            assert all(t.name == "soft_clip" for t in tf)
        base = _build(r)                                    # same layout: same weights, new key
        base.load_state_dict(m.state_dict())
        assert not _same(out, _out(base.eval(), b))
    elif arm == "head_shared_size":
        heads = _modules(m, NBHead)
        assert heads and all(h.head == "nb_shared_size" for h in heads)
        n = out[0]
        assert torch.equal(n, n[:, :1].expand_as(n))        # one NB size per (window, track)
    elif arm == "average_input":
        other = dict(b, avg=b["avg"] * 2.0 + 1.0)
        assert not _same(out, _out(m, other))
    else:
        raise AssertionError(f"no read check for {arm}")

    # the wiring check passes, and is real
    assert C.wiring_check(m, b)["pass"], (r, arm)
    _bypass_mask(monkeypatch)
    assert not C.wiring_check(m, b)["pass"], (r, arm)


def test_not_applicable_table_names_only_known_rungs_and_arms():
    for (r, arm), reason in NOT_APPLICABLE.items():
        assert r in RUNGS and arm in NEW_ARMS and reason


@pytest.mark.parametrize("r", RUNGS)
def test_precomputed_dna_needs_its_width(r):
    with pytest.raises(ValueError, match="dna_embedding_dim"):
        R.rung_module(r).Model(_cfg(r, dna="precomputed"))


def test_rung4_dna_skips_read_the_precomputed_embedding():
    m = _perturb(_build(4, skip_source="dna", dna="precomputed", dna_embedding_dim=D_EMB))
    b = _batch()
    out = _out(m, b)
    assert not _same(out, _out(m, dict(b, x_dna_emb=b["x_dna_emb"] * -1.0)))
    assert C.wiring_check(m, b)["pass"]


@pytest.mark.parametrize("r", RUNGS)
def test_average_input_without_avg(r):
    """Every rung but rung 1 refuses a batch without `avg` under average_input (rung 1 skips the
    lift, as it did before this chunk)."""
    m = _perturb(_build(r, average_input=True))
    b = {k: v for k, v in _batch().items() if k != "avg"}
    if r == 1:
        _out(m, b)
        return
    with pytest.raises((ValueError, KeyError)):
        _out(m, b)


# ------------------------------------------------------------------ (c) keys outside the rung


@pytest.mark.parametrize("r", RUNGS)
def test_dataset_loss_and_wrapper_keys_leave_the_network_unchanged(r):
    a, b = _build(r), _build(r, **OUTSIDE_KEYS)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    _perturb(a), _perturb(b)
    batch = _batch()
    assert _same(_out(a, batch), _out(b, batch))
