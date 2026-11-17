"""Close-out item A12: the per-field covariate drop switch (`covariate_drop_fields`).

Off (key absent or empty) the model is today's: same parameters, same forward, bit for bit, on the
three ablation models (conv autoencoder, conv + attention, U-Net). On, the listed columns of the
two ENCODER copies (C, C') read their neutral value on available rows, in training and eval mode,
and the transform copies (count transform, NB depth offset) keep the true values."""
from __future__ import annotations

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
from minicandi import covariates as CV  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi.rungs import conv_ae, conv_attn_ae, unet  # noqa: E402

L, B = 64, 4
CFG = _ROOT / "tools/minicandi/configs"
RUNGS = {2: conv_ae, 3: conv_attn_ae, 4: unet}


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    tables = synth.make_synthetic_tables(tmp_path_factory.mktemp("a12_drop"))
    cfg = dict(json.loads((CFG / "rung2/base_heldout.json").read_text()), context_bins=L)
    b = D.OnionDataset(tables, "train", cfg, seed=0).batch(B, np.random.default_rng(0))
    C.validate_batch(b)
    return b


def _cfg(r, **kw):
    return {**json.loads((CFG / f"rung{r}/base_heldout.json").read_text()), "context_bins": L, **kw}


def _model(r, cfg):
    torch.manual_seed(0)
    m = RUNGS[r].Model(cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():               # move the zero-initialised parts so covariates matter
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval()


def _neutral_batch(b, col):
    out = dict(b)
    i = C.META_COLUMNS.index(col)
    for key, av in (("c_in", "x_avail"), ("c_out", "y_avail")):
        c = b[key].clone()
        c[..., i] = torch.where(b[av].bool(), torch.full_like(c[..., i], CV.neutral_value(col)),
                                c[..., i])
        out[key] = c
    return out


def test_indices_and_refusals():
    assert CV.drop_field_indices({}) is None
    assert CV.drop_field_indices({"covariate_drop_fields": []}) is None
    assert CV.drop_field_indices({"covariate_drop_fields": ["lab_id", "log2_depth"]}) == \
        ((7, 0), (7, 0))
    qc = CV.drop_field_indices({"covariate_drop_fields": ["nsc", "donor_id"], "qc_on_input": True})
    assert qc == ((11, 10), (10,))
    for bad in ({"covariate_drop_fields": ["nsc"]},                    # QC needs qc_on_input
                {"covariate_drop_fields": ["lab"]},                    # not a column name
                {"covariate_drop_fields": ["lab_id", "lab_id"]},
                {"covariate_drop_fields": "lab_id"},
                {"covariate_drop_fields": ["lab_id"], "two_step": True}):
        with pytest.raises(ValueError):
            CV.drop_field_indices(bad)
    assert CV.neutral_value("lab_id") == 0.0 and CV.neutral_value("read_length") == 75.0
    assert CV.neutral_value("log2_depth") == 24.0


def test_model_covariates_off_returns_the_batch_tensors(batch):
    for training in (False, True):
        e_in, e_out, t_in, t_out = CV.model_covariates(batch, "none", training, drop=None)
        assert e_in is batch["c_in"] and e_out is batch["c_out"]
        assert t_in is batch["c_in"] and t_out is batch["c_out"]


@pytest.mark.parametrize("training", [False, True])
def test_model_covariates_on(batch, training):
    before = {k: batch[k].clone() for k in ("c_in", "c_out")}
    drop = CV.drop_field_indices({"covariate_drop_fields": ["log2_depth", "lab_id"]})
    e_in, e_out, t_in, t_out = CV.model_covariates(batch, "none", training, drop=drop)
    assert torch.equal(batch["c_in"], before["c_in"]) and torch.equal(batch["c_out"], before["c_out"])
    assert torch.equal(t_in, before["c_in"]) and torch.equal(t_out, before["c_out"])   # transforms
    for e, key, av in ((e_in, "c_in", "x_avail"), (e_out, "c_out", "y_avail")):
        a = batch[av].bool()
        assert torch.equal(e[..., 0][a], torch.full_like(e[..., 0][a], 24.0))
        assert torch.equal(e[..., 7][a], torch.zeros_like(e[..., 7][a]))
        assert torch.equal(e[~a], before[key][~a])                       # unavailable rows kept
        keep = [i for i in range(e.shape[-1]) if i not in (0, 7)]
        assert torch.equal(e[..., keep], before[key][..., keep])
    assert bool((before["c_out"][..., 7][batch["y_avail"].bool()] != 0).any())   # the test has power


@pytest.mark.parametrize("r", [2, 3, 4])
def test_off_is_bit_identical(batch, r):
    m0 = _model(r, _cfg(r))
    m1 = _model(r, _cfg(r, covariate_drop_fields=[]))
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0)
    with torch.no_grad():
        o0, o1 = m0(batch), m1(batch)
    assert all(torch.equal(o0[k], o1[k]) for k in ("n", "mu"))


@pytest.mark.parametrize("r", [2, 3, 4])
def test_on_equals_a_neutral_batch_and_changes_the_output(batch, r):
    plain = _model(r, _cfg(r))
    drop = _model(r, _cfg(r, covariate_drop_fields=["lab_id"]))
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(),
                                                 drop.state_dict().values()))
    with torch.no_grad():
        od, on, op = drop(batch), plain(_neutral_batch(batch, "lab_id")), plain(batch)
    for k in ("n", "mu"):
        torch.testing.assert_close(od[k], on[k], rtol=0, atol=0)
    assert not torch.equal(od["mu"], op["mu"])


@pytest.mark.parametrize("r", [2, 3, 4])
def test_depth_drop_keeps_the_offset(batch, r):
    """Dropping log2_depth from the encoders differs from neutralising it in the batch (which
    also moves the count transform and the NB offset)."""
    drop = _model(r, _cfg(r, covariate_drop_fields=["log2_depth"]))
    plain = _model(r, _cfg(r))
    with torch.no_grad():
        od, on = drop(batch), plain(_neutral_batch(batch, "log2_depth"))
    assert not torch.equal(od["mu"], on["mu"])
