"""t370 chunk C15: the control training switches (decision D9, second step).

(1) `control_holdout_cells` (dataset.py): absent = the training and validation batches of
    dataset.py at BASE_COMMIT, bit for bit; set, a held-out cell's control is never in `x_avail`
    or `y_avail` of a training window over 200 batches (the other cells still show theirs); val
    mode is unchanged; bad values raise.
(2) `control_loss_weight` (losses.py): absent or 1.0 = losses.py at BASE_COMMIT, bit for bit;
    weight 4 multiplies the control part of the main term by 4 and leaves `main_no_control`.
(3) `control_branch` (rungs 2, 3, 4): absent or `shared` = the same state_dict and outputs;
    `separate` builds within 10% of `param_budget`, adds only `ctl_*` modules, and a change to
    them moves track 35 only; tracks 0 to 34 equal the shared model with the same weights.
(4) recipe_arms: the control family and CONTROL_HOLDOUT (8 cells); check_config_keys accepts
    the three keys.
"""
from __future__ import annotations

import hashlib
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
from minicandi import dataset as D  # noqa: E402
from minicandi import losses as LS  # noqa: E402
from minicandi import recipe_arms as RA  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.rungs import conv_ae, conv_attn_ae, unet  # noqa: E402

BASE_COMMIT = "1da07953"          # the umbrella commit this chunk was cut from
L = 64
MODS = {2: conv_ae, 3: conv_attn_ae, 4: unet}
HELD = ["s0", "s2"]


def _old_module(rel: str, name: str, tmp_path_factory):
    src = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_COMMIT}:{rel}"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip(f"git show {BASE_COMMIT} failed: {src.stderr.strip()[:80]}")
    p = tmp_path_factory.mktemp(name) / f"{name}.py"
    p.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------ dataset

@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_ctl"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


def _digest(batches) -> str:
    h = hashlib.sha256()
    for b in batches:
        for k in sorted(b):
            h.update(k.encode())
            if k == "meta":
                h.update(json.dumps(b[k], sort_keys=True).encode())
            else:
                t = b[k].contiguous()
                h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
                h.update(t.numpy().tobytes())
    return h.hexdigest()


def _train(mod, tables, cfg, n=30):
    it = iter(mod.OnionDataset(tables, "train", cfg, seed=3))
    return [next(it) for _ in range(n)]


def test_holdout_absent_is_the_old_dataset(tables, tmp_path_factory):
    old = _old_module("tools/minicandi/dataset.py", "dataset_before_c15", tmp_path_factory)
    cfg = {"context_bins": L, "batch_size": 4}
    want = _digest(_train(old, tables, cfg))
    assert _digest(_train(D, tables, cfg)) == want
    assert _digest(_train(D, tables, {**cfg, "control_holdout_cells": None})) == want
    vcfg = {**cfg, "region_bins": 4 * L}
    vo = old.OnionDataset(tables, "val", vcfg, seed=3)
    vn = D.OnionDataset(tables, "val", {**vcfg, "control_holdout_cells": HELD}, seed=3)
    assert _digest(vn) == _digest(vo)


def test_held_out_control_never_in_a_training_window(tables):
    cfg = {"context_bins": L, "batch_size": 4, "control_holdout_cells": HELD}
    ds = D.OnionDataset(tables, "train", cfg, seed=5)
    assert ds.control_holdout == frozenset(HELD)
    seen_held, seen_other_ctl = 0, 0
    it = iter(ds)
    for _ in range(200):
        b = next(it)
        for i, m in enumerate(b["meta"]):
            if m["cell"] in HELD:
                seen_held += 1
                assert not bool(b["x_avail"][i, C.CONTROL_INDEX]), m
                assert not bool(b["y_avail"][i, C.CONTROL_INDEX]), m
            elif bool(b["x_avail"][i, C.CONTROL_INDEX] | b["y_avail"][i, C.CONTROL_INDEX]):
                seen_other_ctl += 1
    assert seen_held > 0 and seen_other_ctl > 0          # not vacuous
    # the held-out cells keep their other tracks; train_cells is unchanged
    assert ds.train_cells == D.OnionDataset(tables, "train", {"context_bins": L}).train_cells


def test_holdout_refusals(tables):
    with pytest.raises(ValueError):
        D.OnionDataset(tables, "train", {"context_bins": L, "control_holdout_cells": "s0"})
    with pytest.raises(ValueError):
        D.OnionDataset(tables, "train", {"context_bins": L, "control_holdout_cells": ["nope"]})


# ------------------------------------------------------------------------------ losses

def _loss_inputs(seed=0, B=3, A=C.A):
    g = torch.Generator().manual_seed(seed)
    mu = torch.rand(B, L, A, generator=g) * 20 + 0.1
    n = torch.rand(B, L, A, generator=g) * 5 + 0.5
    y = torch.randint(0, 40, (B, L, A), generator=g, dtype=torch.int32)
    y_kind = torch.zeros(B, A, dtype=torch.uint8)
    y_kind[:, [1, 4, C.CONTROL_INDEX]] = C.KIND_IMPUTE
    y_kind[:, 7] = C.KIND_CORRECT
    y_kind[:, 9] = C.KIND_SAME
    bm = torch.zeros(B, L, A, dtype=torch.bool)
    bm[:, ::4, 9] = True
    batch = {"y_counts": y, "y_avail": y_kind > 0, "y_kind": y_kind, "x_bin_mask": bm,
             "x_avail": y_kind == C.KIND_SAME}
    return {"n": n, "mu": mu}, batch


@pytest.fixture(scope="module")
def old_losses(tmp_path_factory):
    return _old_module("tools/minicandi/losses.py", "losses_before_c15", tmp_path_factory)


@pytest.mark.parametrize("cfg", [{}, {"loss_weights": {"impute": 2.0}}, {"loss": "nb_crps"},
                                 {"loss_assay_weight": "equal"}, {"loss_shape_weight": 0.5}])
def test_weight_one_is_the_old_loss(cfg, old_losses):
    out, batch = _loss_inputs()
    lo, po = old_losses.total_loss(out, batch, cfg)
    for c in (cfg, {**cfg, "control_loss_weight": 1.0}):
        ln, pn = LS.total_loss(out, batch, c)
        assert torch.equal(ln, lo) and pn == po


def test_weight_four_multiplies_the_control_part():
    out, batch = _loss_inputs(seed=1)
    mask = C.loss_mask(batch)
    per = LS.nb_nll_per(out["n"], out["mu"], batch["y_counts"])
    ctl = torch.zeros_like(mask)
    ctl[..., C.CONTROL_INDEX] = True
    k = int(mask.sum())
    a, c = float(per[mask & ~ctl].sum()) / k, float(per[mask & ctl].sum()) / k
    l1, p1 = LS.total_loss(out, batch, {})
    l4, p4 = LS.total_loss(out, batch, {"control_loss_weight": 4.0})
    assert float(l1) == pytest.approx(a + c, rel=1e-5)
    assert float(l4) == pytest.approx(a + 4.0 * c, rel=1e-5)
    assert p4["main_no_control"] == pytest.approx(p1["main_no_control"], rel=1e-6)
    l0, _ = LS.total_loss(out, batch, {"control_loss_weight": 0.0})
    assert float(l0) == pytest.approx(a, rel=1e-5)
    with pytest.raises(ValueError):
        LS.total_loss(out, batch, {"control_loss_weight": -1.0})


# ------------------------------------------------------------------------------ models

def _base(r: int) -> dict:
    return json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base_heldout.json").read_text())


def _cfg(r: int, **over) -> dict:
    return {**_base(r), "context_bins": L, **over}


def _batch(K: int, B: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_avail = torch.zeros(B, C.A, dtype=torch.bool)
    x_avail[:, [0, 3, 7, 15, 21]] = True
    y_kind = torch.zeros(B, C.A, dtype=torch.uint8)
    y_kind[:, 3] = C.KIND_SAME
    y_kind[:, [1, 16, 30, C.CONTROL_INDEX]] = C.KIND_IMPUTE
    bm = torch.zeros(B, L, C.A, dtype=torch.bool)
    bm[:, ::3, 3] = True
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


def _build(r: int, cfg: dict, width: int | None = None):
    torch.manual_seed(0)
    if width is None:
        return MODS[r].Model(cfg)
    if r == 4:
        return unet.Model({**cfg, "width": width})
    return MODS[r].Model(cfg, width=width)


def _out(model, batch):
    model.eval()
    with torch.no_grad():
        o = model(batch)
    return o["n"], o["mu"]


def _perturb(params, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in params:
            p.add_(0.05 * torch.randn(p.shape, generator=g))


@pytest.mark.parametrize("r", [2, 3, 4])
def test_shared_is_the_default_model(r):
    a = _build(r, _cfg(r))
    b = _build(r, _cfg(r, control_branch="shared"))
    assert a.width == b.width and a.param_count() == b.param_count()
    assert list(a.state_dict()) == list(b.state_dict())
    _perturb(a.parameters())
    b.load_state_dict(a.state_dict())
    batch = _batch(C.k_input(_cfg(r)))
    assert all(torch.equal(x, y) for x, y in zip(_out(a, batch), _out(b, batch)))


@pytest.mark.parametrize("r", [2, 3, 4])
def test_separate_changes_only_track_35(r):
    cfg = _cfg(r)
    shared = _build(r, cfg)
    sep_cfg = _cfg(r, control_branch="separate")
    sep = _build(r, sep_cfg)
    budget = float(sep_cfg["param_budget"])
    assert abs(sep.param_count() - budget) <= 0.10 * budget
    # same width: the shared weights load into the separate model, which adds ctl_* only
    sep_w = _build(r, sep_cfg, width=shared.width)
    new = set(sep_w.state_dict()) - set(shared.state_dict())
    assert new and all(k.split(".")[0].startswith("ctl_") for k in new)
    assert set(shared.state_dict()) <= set(sep_w.state_dict())
    _perturb(shared.parameters())
    missing, unexpected = sep_w.load_state_dict(shared.state_dict(), strict=False)
    assert not unexpected and set(missing) == new
    ctl = [p for n, p in sep_w.named_parameters() if n.split(".")[0].startswith("ctl_")]
    _perturb(ctl, seed=1)
    batch = _batch(C.k_input(cfg))
    (n0, m0), (n1, m1) = _out(shared, batch), _out(sep_w, batch)
    i = C.CONTROL_INDEX
    assert torch.equal(n0[..., :i], n1[..., :i]) and torch.equal(m0[..., :i], m1[..., :i])
    assert not torch.equal(m0[..., i], m1[..., i])
    res = C.wiring_check(sep_w.eval(), batch)
    assert res["max_rel_change"] <= 1e-6


@pytest.mark.parametrize("r", [2, 3, 4])
def test_bad_branch_raises(r):
    with pytest.raises(ValueError):
        _build(r, _cfg(r, control_branch="own"))


# ------------------------------------------------------------------------------ arms, keys

def test_control_family_and_holdout_list():
    assert len(RA.CONTROL_HOLDOUT) == 8 and RA.CONTROL_HOLDOUT == sorted(RA.CONTROL_HOLDOUT)
    fam = {a: x for a, x in RA.ARMS.items() if x["family"] == "control"}
    assert set(fam) == {"ctl_holdout", "ctl_holdout_w4", "ctl_holdout_separate"}
    for x in fam.values():
        assert x["models"] == (2, 3, 4)
        assert x["keys"]["control_holdout_cells"] == RA.CONTROL_HOLDOUT
    assert fam["ctl_holdout_w4"]["keys"]["control_loss_weight"] == 4.0
    assert fam["ctl_holdout_separate"]["keys"]["control_branch"] == "separate"
    assert all(RA.run_name(r, a).startswith("rcp_ctl_") for r, a in
               RA.select(arms=list(fam)))


def test_check_config_keys_accepts_the_three_keys():
    T.check_config_keys({"control_holdout_cells": ["s0"], "control_loss_weight": 4.0,
                         "control_branch": "separate"})
