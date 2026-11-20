"""DNA fusion arms (close-out item A11; `tools/minicandi/dna_fusion.py`).

(a) Without `dna_fusion` (absent or null) every rung of 2, 3, 4 is the model of the umbrella commit
    before the change (`BASE_COMMIT`), state_dict and outputs to 0 difference, under the base DNA
    modes own, off, shuffled and precomputed.
(b) Every arm keeps the base trunk width, passes the copy-leak wiring check, reads the DNA, joins
    it at the stage the arm says (forward-hook probes before and after the join; the U-Net
    bottleneck join is also checked with the DNA skips off), and sends gradient to every parameter
    of its join layer.
(c) The arms a rung cannot express, `dna = off`, unknown values and rungs 1, 5, 6 are refused.
(d) The three layers: FiLM starts as the identity; patch pooling and cross-attention shapes.
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
from minicandi.components import NotApplicable  # noqa: E402
from minicandi.dna_fusion import (APPLIES, NOT_APPLICABLE, DnaCrossAttn, DnaFiLM,  # noqa: E402
                                  DnaPatchDown)

#: the umbrella commit this item was cut from: its rung modules are "the code before the change"
BASE_COMMIT = "309d0f6"
L = 64
D_EMB = 128
NAMES = {2: "conv_ae", 3: "conv_attn_ae", 4: "unet"}
ARMS = [(r, a) for r in (2, 3, 4) for a in APPLIES[NAMES[r]]]
BASE_DNA = [{}, {"dna": "off"}, {"dna": "shuffled"},
            {"dna": "precomputed", "dna_embedding_dim": D_EMB}, {"dna_fusion": None}]


def _ref(r: int) -> dict:
    p = _ROOT / f"tools/minicandi/configs/final/rung{r}/reference_equal.json"
    return {**json.loads(p.read_text()), "context_bins": L}


def _batch(B=2, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [0, 3, 7, 15, 21, 35]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 3] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, [1, 16, 30]] = C.KIND_IMPUTE
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[:, ::3, 3] = True
    bm[:, 20:40, 7] = True
    x = torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0

    def cov(avail):
        v = torch.rand(B, C.A, C.K_META, generator=g)
        c = torch.zeros(B, C.A, C.K_META)
        c[..., 0] = 20.0 + 8.0 * v[..., 0]
        c[..., 1] = (v[..., 1] > 0.5).float()
        c[..., 2] = 36.0 + 100.0 * v[..., 2]
        c[..., 3] = (v[..., 3] > 0.5).float()
        c[..., 4] = 30.0 * v[..., 4]
        c[..., 5:] = torch.randint(1, 6, (B, C.A, C.K_META - 5), generator=g).float()
        return torch.where(avail[..., None], c, torch.zeros_like(c))

    y_avail = y_kind > 0
    return {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
            "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
            "x_dna_emb": torch.randn(B, L, D_EMB, generator=g),
            "c_in": cov(x_avail), "c_out": cov(y_avail),
            "y_counts": torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32),
            "y_avail": y_avail, "y_kind": y_kind,
            "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}


def _perturb(model, seed=0):
    """Noise on every parameter (the NB head and the FiLM start at zero, so an untrained model
    would ignore its inputs and the checks below would pass vacuously)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


def _out(model, batch):
    with torch.no_grad():
        o = model(batch)
    return o["n"], o["mu"]


def _build(cfg):
    torch.manual_seed(0)
    return R.build_model(cfg)


# ------------------------------------------------------------------ (a) off = before the change


@pytest.fixture(scope="module")
def old_modules(tmp_path_factory) -> dict:
    out, d = {}, tmp_path_factory.mktemp("pre_a11")
    for r, name in NAMES.items():
        res = subprocess.run(["git", "-C", str(_ROOT), "show",
                              f"{BASE_COMMIT}:tools/minicandi/rungs/{name}.py"],
                             capture_output=True, text=True)
        if res.returncode:
            pytest.skip(f"git show {BASE_COMMIT} unavailable here: {res.stderr.strip()[:80]}")
        path = d / f"pre_a11_{name}.py"
        path.write_text(res.stdout)
        spec = importlib.util.spec_from_file_location(f"pre_a11_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[r] = mod
    return out


@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("over", BASE_DNA, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items())
                         or "own")
def test_without_the_key_the_model_is_the_pre_change_model(r, over, old_modules):
    cfg = {**_ref(r), **over}
    torch.manual_seed(0)
    old = old_modules[r].Model({k: v for k, v in cfg.items() if k != "dna_fusion"})
    torch.manual_seed(0)
    new = R.rung_module(r).Model(cfg)
    so, sn = old.state_dict(), new.state_dict()
    assert [(k, tuple(v.shape)) for k, v in so.items()] == \
        [(k, tuple(v.shape)) for k, v in sn.items()]
    assert all(torch.equal(so[k], sn[k]) for k in so)
    _perturb(old), _perturb(new)
    b = _batch()
    assert all(torch.equal(x, y) for x, y in zip(_out(old, b), _out(new, b)))


# ------------------------------------------------------------------ (b) every arm


@pytest.mark.parametrize("r,arm", ARMS, ids=lambda v: str(v))
@pytest.mark.parametrize("dna", ["own", "precomputed"])
def test_arm_keeps_the_base_width_reads_dna_and_keeps_the_mask(r, arm, dna):
    cfg = {**_ref(r), "dna_fusion": arm, "dna": dna}
    if dna == "precomputed":
        cfg["dna_embedding_dim"] = D_EMB
    m = _build(cfg)
    base = _build({k: v for k, v in cfg.items() if k != "dna_fusion"})
    assert m.width == base.width, (m.width, base.width)
    _perturb(m)
    b = _batch()
    assert C.wiring_check(m, b)["pass"]
    assert not torch.allclose(_out(m, b)[1], _out(m, _other_dna(b, dna))[1]), \
        "the arm does not read the DNA"
    n_, mu = _out(m, b)
    assert n_.shape == mu.shape == (2, L, C.A) and bool((mu > 0).all() and (n_ > 0).all())


def _other_dna(b: dict, dna: str = "own") -> dict:
    key = "x_dna_emb" if dna == "precomputed" else "x_dna"
    b2 = dict(b)
    g = torch.Generator().manual_seed(9)
    b2[key] = torch.randn(b[key].shape, generator=g) if dna == "precomputed" else \
        torch.randint(0, 5, b[key].shape, generator=g).to(b[key].dtype)
    return b2


#: (rung, arm) -> (does the probe BEFORE the join stage see the DNA, does the probe after it).
#: Probes (module whose first input is captured): before = the first encoder block (conv models)
#: or the count tower (U-Net); after = the bottleneck (`mid[0]` / the self-attention `attn`).
STAGE = {(2, "film"): (True, True), (2, "late"): (False, True),
         (3, "film"): (True, True), (3, "late"): (False, True), (3, "xattn"): (False, True),
         (4, "film"): (False, True), (4, "xattn"): (False, True), (4, "early"): (True, True)}


def _probes(m, r):
    if r == 2:
        return m.enc[0][0], m.mid[0]
    if r == 3:
        return m.enc[0], m.attn
    return m.count_tower, m.attn


def _captured(m, r, b):
    seen = []
    hooks = [mod.register_forward_pre_hook(lambda _m, a: seen.append(a[0].detach().clone()))
             for mod in _probes(m, r)]
    _out(m, b)
    for h in hooks:
        h.remove()
    return seen


@pytest.mark.parametrize("r,arm", ARMS, ids=lambda v: str(v))
def test_arm_joins_dna_at_its_stage(r, arm):
    m = _perturb(_build({**_ref(r), "dna_fusion": arm}))
    b = _batch()
    (pre1, post1), (pre2, post2) = _captured(m, r, b), _captured(m, r, _other_dna(b))
    before, after = STAGE[(r, arm)]
    assert (not torch.equal(pre1, pre2)) == before, "DNA reaches the stage before the join"
    assert (not torch.equal(post1, post2)) == after, "DNA does not reach the bottleneck"


@pytest.mark.parametrize("arm", ["film", "xattn"])
def test_unet_bottleneck_join_is_live_without_dna_skips(arm):
    """With `skip_source: counts`, the bottleneck join is the only DNA path: dropping it must
    leave the output blind to the DNA."""
    m = _perturb(_build({**_ref(4), "dna_fusion": arm, "skip_source": "counts"}))
    b = _batch()
    assert not torch.allclose(_out(m, b)[1], _out(m, _other_dna(b))[1])


@pytest.mark.parametrize("r,arm", ARMS, ids=lambda v: str(v))
def test_join_layer_gets_gradient(r, arm):
    m = _perturb(_build({**_ref(r), "dna_fusion": arm}))
    out = m(_batch())
    (out["mu"].log().mean() + out["n"].log().mean()).backward()
    names = {"film": ["dna_film"], "late": ["dna_down"], "xattn": ["dna_xattn"],
             "early": ["dna", "stem"]}[arm] + (["dna_down"] if (r, arm) == (3, "xattn") else []) \
        + (["count_proj"] if r == 4 and arm in ("film", "xattn") else [])
    for name in names:
        for pn, p in getattr(m, name).named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}.{pn} has no gradient"


@pytest.mark.parametrize("r", [2, 3])
def test_conv_arms_take_the_dna_out_of_the_input_projection(r):
    base = _build(_ref(r))
    proj = (lambda m: m.stem.in_channels) if r == 2 else (lambda m: m.fuse.in_features)
    for arm in APPLIES[NAMES[r]]:
        m = _build({**_ref(r), "dna_fusion": arm})
        assert proj(m) == proj(base) - 32, arm
        assert hasattr(m, {"film": "dna_film", "late": "dna_down", "xattn": "dna_xattn"}[arm])


def test_unet_arms_replace_the_bottleneck_join():
    base = _build(_ref(4))
    assert hasattr(base, "fuse") and base.skip_dna
    for arm, mod in (("film", "dna_film"), ("xattn", "dna_xattn")):
        m = _build({**_ref(4), "dna_fusion": arm})
        assert not hasattr(m, "fuse") and hasattr(m, mod) and hasattr(m, "dna_tower")
        assert hasattr(m, "count_proj")                     # the count half of `fuse` stays
        assert m.skip_dna                                   # the DNA skips stay
    m = _build({**_ref(4), "dna_fusion": "early"})
    assert not hasattr(m, "fuse") and not hasattr(m, "dna_tower") and not m.skip_dna
    assert m.stem.in_channels == base.stem.in_channels + m.chans[0]


# ------------------------------------------------------------------ (c) refusals


@pytest.mark.parametrize("name,arm", sorted(NOT_APPLICABLE))
def test_not_applicable_pairs_are_refused(name, arm):
    r = {v: k for k, v in NAMES.items()}[name]
    with pytest.raises(NotApplicable):
        _build({**_ref(r), "dna_fusion": arm})


@pytest.mark.parametrize("r", [2, 3, 4])
def test_dna_off_and_bad_values_are_refused(r):
    with pytest.raises(NotApplicable):
        _build({**_ref(r), "dna": "off", "dna_fusion": "film"})
    with pytest.raises(ValueError):
        _build({**_ref(r), "dna_fusion": "add"})


@pytest.mark.parametrize("r", [1, 5, 6])
def test_other_rungs_refuse_the_key(r):
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    with pytest.raises(NotApplicable):
        R.build_model({**cfg, "dna_fusion": "film"})


def test_unet_early_with_dna_only_skips_is_refused():
    with pytest.raises(NotApplicable):
        _build({**_ref(4), "dna_fusion": "early", "skip_source": "dna"})


def test_training_accepts_the_key():
    from minicandi.train import check_config_keys
    check_config_keys({**_ref(2), "dna_fusion": "film"})


# ------------------------------------------------------------------ (d) the layers


def test_film_starts_as_the_identity_and_learns():
    torch.manual_seed(0)
    f = DnaFiLM(8, 6)
    z, dna = torch.randn(2, 6, 10), torch.randn(2, 8, 10)
    assert torch.equal(f(z, dna), z)
    with torch.no_grad():
        f.proj.weight.normal_()
    assert not torch.allclose(f(z, dna), z)


def test_patch_down_and_cross_attention_shapes():
    torch.manual_seed(0)
    down = DnaPatchDown(8, 12, 4)
    assert down(torch.randn(2, 8, 16)).shape == (2, 12, 4)
    assert sum(p.numel() for p in down.parameters()) == 8 * 12 + 12
    with pytest.raises(ValueError):
        down(torch.randn(2, 8, 18))
    xa = DnaCrossAttn(16, 4)
    q = torch.randn(2, 5, 16)
    assert xa(q, torch.randn(2, 5, 16)).shape == q.shape
    with pytest.raises(ValueError):
        xa(q, torch.randn(2, 6, 16))
