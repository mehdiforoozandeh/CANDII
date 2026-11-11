"""miniCANDI component arms (chunk A3; pinned interface E, "Component arms"): adaLN-Zero injection,
covariate dropout, numeric-only covariate embedding, the precomputed DNA tower, `offset_eps`, the
soft-clip count transform and the shared-size NB head. Defaults must give today's components."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import components as M  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi import covariates as COV  # noqa: E402
from minicandi import registry as R  # noqa: E402

L = 64
LAB = C.META_COLUMNS.index("lab_id")
DONOR = C.META_COLUMNS.index("donor_id")


def _batch(B=2, L=L, K=C.K_META, seed=0, d_emb=None) -> dict:
    """A contract batch built from a fixed generator only (no tables, no dataset code)."""
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [0, 3, 7, 15, 21, 35]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 3] = C.KIND_SAME
    y_kind[:, 15] = C.KIND_CORRECT
    y_kind[:, [1, 16, 30]] = C.KIND_IMPUTE
    y_avail = y_kind > 0
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[:, ::3, 3] = True
    bm[:, 5:9, 0] = True
    x = torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32)
    x[bm | ~x_avail[:, None, :]] = 0

    def cov(avail, k):
        c = torch.zeros(B, C.A, k)
        v = torch.rand(B, C.A, k, generator=g)
        c[..., 0] = 20.0 + 8.0 * v[..., 0]                              # log2 depth
        c[..., 1] = (v[..., 1] > 0.5).float()
        c[..., 2] = 36.0 + 100.0 * v[..., 2]
        c[..., 3] = (v[..., 3] > 0.5).float()
        c[..., 4] = 30.0 * v[..., 4]
        c[..., 5:C.K_META] = torch.randint(1, 6, (B, C.A, C.K_META - 5), generator=g).float()
        if k > C.K_META:
            c[..., C.K_META:] = v[..., C.K_META:]
        return torch.where(avail[..., None], c, torch.zeros_like(c))

    b = {"x_counts": x, "x_avail": x_avail, "x_bin_mask": bm,
         "x_dna": torch.randint(0, 5, (B, L * C.RES), generator=g).to(torch.uint8),
         "c_in": cov(x_avail, K), "c_out": cov(y_avail, C.K_META),
         "y_counts": torch.randint(0, 60, (B, L, C.A), generator=g, dtype=torch.int32),
         "y_avail": y_avail, "y_kind": y_kind,
         "meta": [{"cell": "x", "chrom": "chr1", "start_bin": 0}] * B}
    if d_emb is not None:
        b["x_dna_emb"] = torch.randn(B, L, d_emb, generator=g)
    return b


def _perturb(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


# ------------------------------------------------------------- defaults are today's components

#: recorded on the tree before this chunk (umbrella 6d5dac3), darwin arm64, torch 2.6.0:
#: rung -> (param_count, sha256 of the state_dict keys and shapes, sha256 of eval-mode n and mu
#: on `_batch()` after `_perturb`, all from torch.manual_seed(0) and the rung's base.json)
REFERENCE = {
    1: (1001418, "c40363e945ae46f72c2409f3d41ee84d03d42e04a8e31202c3a6e627ebe9f626",
        "c3ff89e6018de11a41b2072e08735c0e2ef38bdf601920c502d236e4a3eea83d"),
    2: (1042514, "39eb4ce575e5bd11c5304f3e104f9f68b3355ff6bd8d898d707465f996e52d39",
        "3e030b02c7ecae97df6cfd367396f0527eed2380f6cd1432806a1c23decad7ff"),
    3: (1038018, "25eed73122cc646f40dabfcec3bb9f49e289eead0af82383e58bcc85c65c1957",
        "082908b0805943b2ae633235781619f4e67e4a51f1b9f7ad4c1e1b99a7f61794"),
    4: (993599, "7736ae9b1639cfdf51b957d58d743b2eaf76b4146adfad31176f79ebfe9be15a",
        "c938a4816a0ba4efb9d13a6d3aabbc57c2371732c5099ebe8e3c38407b55ebae"),
    5: (1000158, "1040a82d70c853b6cb311d36c6c68519faef5a72af736d33411b8446e513ad6c",
        "1d84e420411d325e56a8600b842c1c67106ce57beaaf1dfabf24eebcb4e6dbc0"),
    6: (1025120, "a819d378eabc38748e0756200aea008fff687ba5e3d1b2dd3783ffc5f33127e1",
        "47c81cfe812435deec263c3afe2d002ebfb10365699e751cbf30151a53cd5a41"),
}
REFERENCE_PLATFORM = ("2.6.0", "Darwin", "arm64")


def _base(r: int) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    return {**cfg, "context_bins": L}


def rung_digests(r: int) -> tuple:
    torch.manual_seed(0)
    m = R.build_model(_base(r))
    keys = hashlib.sha256(";".join(f"{k}:{tuple(v.shape)}" for k, v in m.state_dict().items())
                          .encode()).hexdigest()
    _perturb(m).eval()
    with torch.no_grad():
        o = m(_batch())
    out = hashlib.sha256(o["n"].numpy().tobytes() + o["mu"].numpy().tobytes()).hexdigest()
    return m.param_count(), keys, out


@pytest.mark.parametrize("r", sorted(REFERENCE))
def test_six_rung_bases_give_todays_parameters_and_outputs(r):
    count, keys, out = rung_digests(r)
    assert (count, keys) == REFERENCE[r][:2]
    here = (torch.__version__.split("+")[0], platform.system(), platform.machine())
    if here == REFERENCE_PLATFORM:
        assert out == REFERENCE[r][2]                                  # 0 difference, bit for bit


def test_explicit_default_keys_build_the_same_components():
    kw = M.component_kwargs({})
    assert kw == M.component_kwargs({"covariate_embedding": "full", "covariate_dropout": 0.0,
                                     "offset_eps": 1.0, "head": "nb", "dna": "own"})
    assert kw["injector"] == {"embedding": "full", "null_vector": False}
    assert kw["dna"] == {} and kw["covariate_dropout"] == 0.0
    b = _batch()
    for mode in ("concat", "film", "tokens"):
        torch.manual_seed(0)
        a = M.CovariateInjector(mode, 8, C.K_META)
        torch.manual_seed(0)
        e = M.CovariateInjector(mode, 8, C.K_META, **kw["injector"])
        assert a.state_dict().keys() == e.state_dict().keys()
        for k, v in a.state_dict().items():
            assert torch.equal(v, e.state_dict()[k])
    for name in M.TRANSFORMS:
        assert torch.equal(M.CountTransform(name)(b["x_counts"], b["c_in"][..., 0]),
                           M.CountTransform(name, **kw["transform"])(b["x_counts"],
                                                                      b["c_in"][..., 0]))
    h = torch.randn(2, L, C.A, 8)
    _perturb(hd := M.NBHead(8))
    he = M.NBHead(8, **kw["head"])
    he.load_state_dict(hd.state_dict())
    oa, ob = hd(h, b["c_out"]), he(h, b["c_out"])
    assert torch.equal(oa["n"], ob["n"]) and torch.equal(oa["mu"], ob["mu"])


def test_model_covariates_with_zero_dropout_draws_nothing_and_returns_todays_tensors():
    b = _batch()
    for mode in ("none", "covariates", "all"):
        torch.manual_seed(3)
        a = COV.model_covariates(b, mode, True)
        state = torch.get_rng_state()
        torch.manual_seed(3)
        z = COV.model_covariates(b, mode, True, dropout=0.0)
        assert torch.equal(state, torch.get_rng_state())
        assert all(torch.equal(x, y) for x, y in zip(a, z))


@pytest.mark.parametrize("bad", [{"covariate_embedding": "ids"}, {"covariate_dropout": 1.0},
                                 {"covariate_dropout": -0.1}, {"offset_eps": 0.0},
                                 {"head": "poisson"}, {"dna": "precomputed"},
                                 {"dna": "precomputed", "dna_embedding_dim": 0}])
def test_bad_component_keys_raise(bad):
    with pytest.raises(ValueError):
        M.component_kwargs(bad)


# ---------------------------------------------------------------------------------- adaln_zero


def test_adaln_zero_starts_as_the_identity_and_learns():
    torch.manual_seed(0)
    inj = M.CovariateInjector("adaln_zero", 8, C.K_META)
    h = torch.randn(2, L, C.A, 8, requires_grad=True)
    c = _batch()["c_in"][:, None]
    out = inj(h, c)
    assert torch.equal(out, h)                                         # exact identity at init
    out.pow(2).sum().backward()
    gate_w = inj.g.net[-1].weight.grad[16:]                             # the gate rows
    assert gate_w.abs().sum() > 0                                       # the gate can open
    _perturb(inj)
    assert not torch.allclose(inj(h, c), h)
    assert not torch.allclose(inj(h, c), inj(h, c * 0.5))               # it reads the covariates
    with pytest.raises(M.NotApplicable):
        inj.tokens(_batch()["c_in"])


# --------------------------------------------------------------------------- covariate dropout


def test_covariate_dropout_only_in_training_and_never_on_the_depth_the_transform_reads():
    b = _batch()
    for mode in ("none", "covariates"):
        ev = COV.model_covariates(b, mode, False, dropout=0.5)
        assert all(torch.equal(x, y) for x, y in zip(ev, COV.model_covariates(b, mode, False)))
        torch.manual_seed(0)
        e_in, e_out, t_in, t_out = COV.model_covariates(b, mode, True, dropout=0.5)
        torch.manual_seed(0)
        ref = COV.model_covariates(b, mode, True)
        assert torch.equal(t_in, ref[2]) and torch.equal(t_out, ref[3])  # transform copies kept
        assert not torch.isnan(t_in).any() and not torch.isnan(t_out).any()
        dropped = torch.isnan(e_in).all(-1)
        assert torch.equal(dropped, torch.isnan(e_out).all(-1))         # C and C' of a track
        assert 0.3 < dropped.float().mean() < 0.7
        assert torch.equal(e_in[~dropped], ref[0][~dropped])


def test_dropped_rows_embed_as_one_learned_null_vector():
    torch.manual_seed(0)
    emb = COV.CovariateEmbedding(C.K_META, 6, null_vector=True)
    with torch.no_grad():
        emb.null.copy_(torch.arange(6.0))
    c = _batch()["c_in"]
    c[0, 3] = float("nan")
    e = emb(c)
    assert torch.equal(e[0, 3], torch.arange(6.0))
    assert torch.isfinite(e).all()
    e.sum().backward()
    assert emb.null.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="null_vector"):
        COV.CovariateEmbedding(C.K_META, 6)(c)                    # a NaN row needs the null vector
    assert sum(p.numel() for p in COV.CovariateEmbedding(C.K_META, 6, null_vector=True)
               .parameters()) == sum(p.numel() for p in COV.CovariateEmbedding(C.K_META, 6)
                                     .parameters()) + 6


def test_dropout_through_the_injector_keeps_film_finite_in_training():
    torch.manual_seed(0)
    kw = M.component_kwargs({"covariate_dropout": 0.5})
    inj = _perturb(M.CovariateInjector("film", 8, C.K_META, **kw["injector"]))
    b = _batch()
    torch.manual_seed(1)
    e_in, *_ = COV.model_covariates(b, "none", True, dropout=kw["covariate_dropout"])
    out = inj(torch.randn(2, L, C.A, 8), e_in[:, None])
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- numeric_only


def test_numeric_only_ignores_categorical_ids_and_full_reads_them():
    c = _batch()["c_in"]
    c2 = c.clone()
    c2[..., LAB] += 1.0
    c2[..., DONOR] += 2.0
    c3 = c.clone()
    c3[..., 0] += 1.0
    torch.manual_seed(0)
    num = _perturb(COV.CovariateEmbedding(C.K_META, 6, numeric_only=True))
    torch.manual_seed(0)
    full = _perturb(COV.CovariateEmbedding(C.K_META, 6))
    assert torch.equal(num(c), num(c2))
    assert not torch.allclose(num(c), num(c3))
    assert not torch.allclose(full(c), full(c2))
    inj = M.CovariateInjector("concat", 8, C.K_META, embedding="numeric_only")
    assert inj.embed.numeric_only


# ------------------------------------------------------------------------- DNA: precomputed


def test_precomputed_dna_tower_has_no_convolution_and_reads_each_bin_alone():
    tower = M.DnaTower("precomputed", 6, d_emb=12)
    assert not any(isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
                   for m in tower.modules())
    _perturb(tower)
    x = torch.randn(2, L, 12)
    out = tower(x)
    assert out.shape == (2, L, 6)
    x2 = x.clone()
    x2[:, 10] += 5.0
    diff = (tower(x2) - out).abs().sum(dim=(0, 2))
    assert diff[10] > 0 and diff[torch.arange(L) != 10].max() == 0
    with pytest.raises(ValueError):
        tower(torch.randint(0, 5, (2, L * C.RES)).to(torch.uint8))     # base codes refused
    with pytest.raises(ValueError):
        tower(torch.randn(2, L, 11))
    with pytest.raises(ValueError):
        M.DnaTower("precomputed", 6)
    assert M.component_kwargs({"dna": "precomputed", "dna_embedding_dim": 12})["dna"] == \
        {"d_emb": 12}


# ------------------------------------------------------------------- count transforms


@pytest.mark.parametrize("name,kw", [("soft_clip", {}), ("offset", {"offset_eps": 0.1}),
                                     ("offset", {})])
def test_per_bin_transforms_are_j_invariant(name, kw):
    g = torch.Generator().manual_seed(1)
    y = torch.randint(0, 30, (2, 600, C.A), generator=g, dtype=torch.int32)
    vis = torch.rand(2, 600, C.A, generator=g) > 0.3
    d = torch.full((2, C.A), 24.0)
    tr = M.CountTransform(name, **kw)
    y2 = torch.where(vis, y, y + 1000)                              # hidden bins perturbed
    assert torch.equal(tr(y, d, vis)[vis], tr(y2, d, vis)[vis])
    assert torch.equal(tr(y, d)[vis], tr(y2, d)[vis])               # per bin: even without vis
    assert torch.isfinite(tr(y2, d, vis)).all()


def test_offset_eps_is_the_offset_pseudocount():
    y = torch.zeros(1, 4, C.A, dtype=torch.int32)
    d = torch.full((1, C.A), 24.0)
    assert torch.equal(M.CountTransform("offset", offset_eps=0.1)(y, d),
                       torch.full((1, 4, C.A), torch.log2(torch.tensor(0.1)).item()))
    assert torch.equal(M.CountTransform("offset")(y, d), torch.zeros(1, 4, C.A))
    with pytest.raises(ValueError):
        M.CountTransform("offset", offset_eps=0.0)


def test_soft_clip_is_the_borzoi_squash_and_decodes_by_the_offset_form():
    tr = M.CountTransform("soft_clip")
    y = torch.tensor([0, 1, 16, 2 ** 11, 2 ** 16]).reshape(1, 5, 1).expand(1, 5, C.A).contiguous()
    d = torch.full((1, C.A), 24.0)                                   # s = 1
    x = tr(y, d)[0, :, 0]
    z = y[0, :, 0].double() ** 0.75
    want = torch.where(z > 384, 384 + torch.sqrt((z - 384).clamp_min(0)), z).float()
    assert torch.allclose(x, want, rtol=1e-6)
    assert x[3] < 384 < x[4] < 450              # 2048**0.75 = 304; 65536**0.75 = 4096 -> 444.9
    assert torch.allclose(tr(y, torch.full((1, C.A), 25.0))[0, :, 0],
                          torch.where((z / 2 ** 0.75) > 384,
                                      384 + torch.sqrt((z / 2 ** 0.75 - 384).clamp_min(0)),
                                      z / 2 ** 0.75).float(), rtol=1e-6)
    eta = torch.randn(1, 5, C.A)
    assert torch.equal(tr.add_back(eta, d), M.CountTransform("offset").add_back(eta, d))
    assert "soft_clip" in M.TRANSFORMS and "soft_clip" not in M.INPUT_ONLY


# ----------------------------------------------------------------------------------- head


def test_shared_size_head_gives_one_size_per_track():
    head = _perturb(M.NBHead(8, head="nb_shared_size"))
    h = torch.randn(2, L, C.A, 8)
    c_out = _batch()["c_out"]
    o = head(h, c_out)
    assert o["n"].shape == o["mu"].shape == (2, L, C.A)
    assert torch.equal(o["n"], o["n"][:1, :1].expand_as(o["n"]))     # same at every bin, window
    assert not torch.equal(o["n"][0, 0, 0], o["n"][0, 0, 1])          # but per track
    assert torch.equal(head(h * 3.0, c_out)["n"], o["n"])              # independent of features
    assert not torch.allclose(head(h * 3.0, c_out)["mu"], o["mu"])
    assert (o["n"] > 0).all() and (o["mu"] > 0).all()
    assert head.lin.out_features == 1 and M.NBHead(8).lin.out_features == 2


# -------------------------------------------------------- a model built from every new arm


class _ArmToy(C.MiniCandi):
    """Per-bin model from the shared components with every component arm switched on."""

    def __init__(self, cfg: dict, d=8, d_emb=12):
        super().__init__()
        kw = M.component_kwargs({**cfg, "dna_embedding_dim": d_emb})
        self.p = kw["covariate_dropout"]
        self.tr = M.CountTransform(cfg.get("count_transform", "offset"), **kw["transform"])
        self.inp = torch.nn.Linear(1, d)
        self.mask = M.MaskHandler("token", d)
        self.cin = M.CovariateInjector(cfg["covariate_injection"], d, C.K_META, **kw["injector"])
        self.cout = M.CovariateInjector(cfg["covariate_injection"], d, C.K_META,
                                        **kw["injector"])
        self.dna = M.DnaTower(cfg["dna"], d, **kw["dna"])
        self.mix = torch.nn.Linear(C.A * d, C.A * d)
        self.head = M.NBHead(d, self.tr, **kw["head"])

    def forward(self, batch):
        B, Lb, Ac = batch["x_counts"].shape
        c_in, c_out, c_in_t, c_out_t = COV.model_covariates(batch, "none", self.training, self.p)
        vis = M.MaskHandler.visible(batch["x_avail"], batch["x_bin_mask"])
        x = self.tr(batch["x_counts"], c_in_t[..., C.LOG2_DEPTH_COL], vis)
        h, _ = self.mask(self.inp(x[..., None]), batch["x_avail"], batch["x_bin_mask"])
        h = self.cin(h, c_in[:, None])
        dna_in = batch["x_dna_emb"] if self.dna.mode == "precomputed" else batch["x_dna"]
        h = h + self.dna(dna_in)[:, :, None, :]
        h = self.mix(h.reshape(B, Lb, -1)).reshape(B, Lb, Ac, -1)
        h = self.cout(h, c_out[:, None])
        return self.head(h, c_out_t)


ARM_CFGS = {
    "adaln_zero": {"covariate_injection": "adaln_zero", "dna": "own"},
    "all_arms": {"covariate_injection": "adaln_zero", "dna": "precomputed",
                 "covariate_dropout": 0.3, "covariate_embedding": "numeric_only",
                 "count_transform": "soft_clip", "head": "nb_shared_size"},
    "offset_eps": {"covariate_injection": "film", "dna": "precomputed", "offset_eps": 0.1},
}


@pytest.mark.parametrize("arm", sorted(ARM_CFGS))
def test_component_arms_pass_the_wiring_check_and_train(arm):
    torch.manual_seed(0)
    model = _perturb(_ArmToy(ARM_CFGS[arm]))
    b = _batch(d_emb=12)
    r = C.wiring_check(model.eval(), b)
    assert r["pass"] and r["n_perturbed"] > 0, r
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        out = model(b)
        loss = C.nb_nll(out["n"], out["mu"], b["y_counts"], C.loss_mask(b))
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
