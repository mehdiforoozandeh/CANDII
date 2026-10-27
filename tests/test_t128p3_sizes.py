"""t128 phase 3: g's hidden width and form D's channel count are per-run settings.

  * form D at the default size is unchanged; at 64 channels n_theta is 512 and the map is still the
    identity at init; the class attribute (read by form_ds) stays 256;
  * the Ladder at default sizes is bit-equal to today's; g_hidden / f_channels resize g and f;
    a non-plain g refuses g_hidden;
  * a checkpoint without the new keys loads as 64 / 32;
  * run_training records the sizes and the arm, load_run and the via-base readout rebuild them;
  * a default run records 64 / 32, no arm, and keeps its name.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools", REPO / "tools" / "t128"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import model, pairs, synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from ladder.fforms import form_d, form_ds  # noqa: E402
import viabase  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
STATS = {"knots_x": np.linspace(0.0, 3.0, 12), "n0": 2.5, "sigma0": 1.0}
SMALL = dict(window=256, batch=4, knot_sample=20_000, knot_block=256)
HOLDOUT = (("depth", "7.5M"),)


def _n(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _near(got: int, want: int) -> bool:
    return abs(got - want) <= 0.1 * want


# (1) form D sizes
def test_form_d_default_and_wide():
    a = form_d.FormD("counts", STATS)
    b = form_d.FormD("counts", STATS, channels=32)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)
    assert a.n_theta == b.n_theta == 256 and a.channels == 32
    assert torch.equal(a.init_theta(), torch.cat([torch.ones(128), torch.zeros(128)]))

    w = form_d.FormD("counts", STATS, channels=64)
    assert w.n_theta == 512 and w.channels == 64
    assert tuple(w.head.weight.shape) == (2, 64, 1)
    assert list(w.state_dict()) == list(sa)
    x = torch.randn(3, 100 + 2 * form_d.CONTEXT)
    loc, disp = w(x, w.init_theta().unsqueeze(0).expand(3, -1))
    torch.testing.assert_close(loc, x[:, form_d.CONTEXT:-form_d.CONTEXT], atol=1e-6, rtol=0)
    torch.testing.assert_close(disp, torch.full_like(disp, math.log(2.5)), atol=1e-6, rtol=0)
    assert form_d.FormD.n_theta == 256
    assert form_ds.FILM == 256


# (2) Ladder sizes
def test_ladder_sizes():
    torch.manual_seed(0)
    a = model.Ladder("D", "counts", STATS)
    torch.manual_seed(0)
    b = model.Ladder("D", "counts", STATS, g_hidden=64, f_channels=32)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert (a.g_hidden, a.f_channels, a.n_theta) == (64, 32, 256)

    big_g = model.Ladder("D", "counts", STATS, g_hidden=256)
    assert big_g.g.net[0].out_features == 256
    assert big_g.g.net[2].in_features == 256
    assert big_g.g.net[4].out_features == 256
    assert (big_g.g_hidden, big_g.f_channels) == (256, 32)

    big_f = model.Ladder("D", "counts", STATS, f_channels=64)
    assert big_f.n_theta == 512
    assert big_f.g.net[4].out_features == 512
    assert big_f.g.net[0].out_features == 64

    counts = {"g_default": _n(a.g), "g_256": _n(big_g.g), "f_default": _n(a.form),
              "f_64": _n(big_f.form), "g_ntheta512": _n(big_f.g)}
    print("parameter counts:", counts)
    assert _near(counts["g_default"], 23_000)
    assert _near(counts["g_256"], 142_000)
    assert _near(counts["f_default"], 16_000)
    assert _near(counts["f_64"], 62_000)
    assert _near(counts["g_ntheta512"], 40_000)

    with pytest.raises(ValueError, match="plain g only"):
        model.Ladder("DS", "counts", STATS, g_hidden=256)
    assert model.Ladder("A", "counts", STATS).f_channels is None


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128p3sizes")
    manifest, cov = synth.make_products(root / "products", 0)
    return {"root": root, "manifest": manifest, "cov": cov, "data": root / "products",
            "rows": pairs.read_manifest(manifest)}


def _train(prod, runs, **kw):
    return ltrain.run_training(prod["manifest"], prod["cov"], prod["data"], prod["root"] / runs,
                               "D", "T1", "counts", "real", 0, max_steps=4, eval_every=2,
                               device="cpu", holdout=HOLDOUT, **SMALL, **kw)


def _load(prod, run_dir):
    return model.load_run(run_dir, prod["data"], prod["cov"], prod["manifest"], device="cpu")


# (5) default run
@pytest.fixture(scope="module")
def default_run(prod):
    return _train(prod, "default")


def test_default_run_records_default_sizes(default_run):
    assert default_run.name == "D_T1_counts_real_s0"
    cfg = json.loads((default_run / "config.json").read_text())
    assert cfg["g_hidden"] == 64 and cfg["f_channels"] == 32
    assert "arm" not in cfg
    assert cfg["run_name"] == "D_T1_counts_real_s0" and cfg["n_theta"] == 256


# (3) a checkpoint without the keys
def test_old_checkpoint_loads_as_default(prod, default_run, tmp_path):
    ckpt = torch.load(default_run / "ckpt.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(5)
    lad = model.Ladder("D", "counts", STATS)
    with torch.no_grad():
        for p in lad.parameters():
            p.add_(0.01 * torch.randn_like(p))
    cfg = {k: v for k, v in ckpt["config"].items() if k not in ("g_hidden", "f_channels", "arm")}
    old = {"g": {k: v.clone() for k, v in lad.g.state_dict().items()},
           "form": {k: v.clone() for k, v in lad.form.state_dict().items()},
           "stats": ckpt["stats"], "encoder": ckpt["encoder"], "config": cfg,
           "step": ckpt["step"], "val_nll": ckpt["val_nll"]}
    assert set(old) == {"g", "form", "stats", "encoder", "config", "step", "val_nll"}
    run = tmp_path / "D_T1_counts_real_s0"
    run.mkdir()
    torch.save(old, run / "ckpt.pt")
    pred = _load(prod, run)
    assert pred.ladder.g_hidden == 64 and pred.ladder.f_channels == 32
    for part in ("g", "form"):
        got = getattr(pred.ladder, part).state_dict()
        assert list(got) == list(old[part])
        assert all(torch.equal(got[k], old[part][k]) for k in got)


# (4) sized runs, read back
@pytest.mark.parametrize("arm,gh,fc,nt", [("Dbigg", 256, 32, 256), ("Dbigf", 64, 64, 512)])
def test_sized_run_recorded_and_rebuilt(prod, arm, gh, fc, nt):
    run = _train(prod, "sized", g_hidden=gh, f_channels=fc, arm=arm)
    assert run.name == f"{arm}_T1_counts_real_s0"
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["rung"] == "D" and cfg["arm"] == arm
    assert (cfg["g_hidden"], cfg["f_channels"], cfg["n_theta"]) == (gh, fc, nt)
    assert cfg["run_name"] == f"{arm}_T1_counts_real_s0"
    pred = _load(prod, run)
    assert pred.ladder.g_hidden == gh and pred.ladder.form.channels == fc
    assert pred.ladder.g.net[0].out_features == gh
    assert pred.ladder.n_theta == nt
    if arm == "Dbigf":
        vb = viabase.load_viabase(run, prod["data"], prod["cov"], prod["manifest"], prod["rows"])
        assert isinstance(vb, viabase.ViaBasePredictor)
        assert vb.ladder.form.channels == 64
        p = pairs.law_pairs(prod["rows"], "T1")[0]
        a, b = p["source_pid"], p["target_pid"]
        n = vb.corpus.get(a, "counts", "chr19").shape[0]
        loc, disp = vb.predict(a, a, b, "chr19")
        assert loc.shape == (n,) and disp.shape == (n,)
