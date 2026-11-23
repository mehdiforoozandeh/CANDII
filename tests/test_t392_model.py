"""t392 chunk C8: strand channels in the rung-4 U-Net and the training entry.

Against plan chunk C8 (cruxvault/INSTRUCTIONS/t392_read_starts/run_state/plan.md) and
INSTRUCTIONS section 3: the read-start model is the reference config
(`configs/recipe/combo/rung4/halves_mix_avg_pearson.json`) plus `read_start`; only the counts it
reads and predicts become read starts. Bit-identity without `read_start` is
tests/test_minicandi_base_golden.py. Batches come from the synthetic read-start store of
tests/test_t392_dataset.py (its `env` fixture).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from test_t392_dataset import _make, env  # noqa: E402,F401  (fixture reused)

REF = REPO / "tools/minicandi/configs/recipe/combo/rung4/halves_mix_avg_pearson.json"
RS = REPO / "tools/minicandi/configs/t392/rs_halves_mix_avg_pearson.json"
L = 64


def _model_cfg(env, **extra):
    cfg = json.loads(RS.read_text())
    cfg.update({"context_bins": L, "read_start_store": env["cfg"]["read_start_store"],
                "read_start_average_h5": env["cfg"]["read_start_average_h5"],
                "average_h5": env["cfg"]["average_h5"], "width": 8})
    cfg.update(extra)
    return cfg


def _batch(env, seed=1):
    ds = _make(env, {"train_halves": "pairs"}, seed=seed)
    return ds, ds.batch(2, np.random.default_rng([seed, 9]))


def _build(cfg, seed=0):
    from minicandi import registry as R
    torch.manual_seed(seed)
    m = R.build_model(cfg)
    with torch.no_grad():                      # the NB head starts at zero: move every weight
        g = torch.Generator().manual_seed(7)
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval()


# ------------------------------------------------------------------------------ config

def test_config_is_the_reference_plus_the_three_read_start_keys():
    ref, rs = json.loads(REF.read_text()), json.loads(RS.read_text())
    extra = {k: rs[k] for k in set(rs) - set(ref)}
    assert extra == {"read_start": True,
                     "read_start_store": "/scratch/mforooz/rs392/store",
                     "read_start_average_h5":
                         "/scratch/mforooz/rs392/aux/eic_v1_2_read_start_average.h5"}
    assert {k: rs[k] for k in ref} == ref
    assert rs["param_budget"] == 1e6 and rs["windows_seen"] == 80000


def test_param_count_under_the_budget_rule_is_within_2_percent_of_1e6():
    from minicandi import registry as R
    ref = R.build_model(json.loads(REF.read_text()))
    rs = R.build_model(json.loads(RS.read_text()))
    assert abs(rs.param_count() - 1e6) <= 0.02 * 1e6
    assert ref.param_count() == 993615
    assert rs.width == ref.width


# ------------------------------------------------------------------------------ model

def test_read_start_model_emits_two_strand_channels_and_their_strand_output(env):
    from minicandi import contract as C
    from minicandi.read_start import strand_output
    m = _build(_model_cfg(env))
    assert m.track_in[0].in_features == 2 and m.avg_in.in_features == 2
    assert tuple(m.out_sub_emb.shape)[0] == 2
    _, b = _batch(env)
    with torch.no_grad():
        out = m(b)
    B = b["x_counts"].shape[0]
    assert tuple(out["mu_sub"].shape) == (B, L, C.A, 2) == tuple(out["n_sub"].shape)
    assert tuple(out["mu"].shape) == (B, L, C.A) == tuple(out["n"].shape)
    so = strand_output(out)
    assert torch.equal(out["mu"], so["mu"]) and torch.equal(out["n"], so["n"])
    assert torch.isfinite(out["mu"]).all() and torch.isfinite(out["n"]).all()


def test_the_average_enters_through_avg_sub(env):
    m = _build(_model_cfg(env))
    _, b = _batch(env)
    b2 = dict(b, avg_sub=b["avg_sub"].clone())
    b2["avg_sub"][..., 1] += 5.0
    b3 = dict(b, avg=b["avg"] + 5.0)                # the channel sum is not read
    with torch.no_grad():
        o1, o2, o3 = m(b)["mu_sub"], m(b2)["mu_sub"], m(b3)["mu_sub"]
    assert not torch.equal(o1, o2)
    assert torch.equal(o1, o3)


def test_hidden_bins_of_x_counts_sub_move_no_output(env):
    m = _build(_model_cfg(env))
    _, b = _batch(env)
    hidden = b["x_bin_mask"].bool() | ~b["x_avail"].bool()[:, None, :]
    assert hidden.any() and (~hidden).any()
    noise = torch.randint(1, 50, b["x_counts_sub"].shape, generator=torch.Generator().manual_seed(3),
                          dtype=b["x_counts_sub"].dtype)
    b2 = dict(b, x_counts_sub=torch.where(hidden[..., None], b["x_counts_sub"] + noise,
                                          b["x_counts_sub"]))
    with torch.no_grad():
        o1, o2 = m(b), m(b2)
    for k in ("mu_sub", "n_sub", "mu", "n"):
        assert torch.equal(o1[k], o2[k]), k
    # a visible bin does move the output (the test is not vacuous)
    b3 = dict(b, x_counts_sub=torch.where(~hidden[..., None], b["x_counts_sub"] + noise,
                                          b["x_counts_sub"]))
    with torch.no_grad():
        assert not torch.equal(o1["mu_sub"], m(b3)["mu_sub"])


@pytest.mark.parametrize("key", ["res_in_sub", "res_out_sub"])
def test_read_start_with_a_sub_bin_key_above_1_raises(env, key):
    from minicandi import registry as R
    with pytest.raises(ValueError):
        R.build_model(_model_cfg(env, **{key: 5}))


def test_read_start_loss_is_finite_and_has_a_gradient(env):
    from minicandi.read_start import read_start_loss
    m = _build(_model_cfg(env)).train()
    _, b = _batch(env)
    loss, parts = read_start_loss(m(b), b, _model_cfg(env))
    assert torch.isfinite(loss)
    loss.backward()
    assert m.track_in[0].weight.grad is not None


# ------------------------------------------------------------------------------ pearson stats

def test_pearson_stats_under_read_start_skip_channel_1_of_unstranded_tracks(env):
    from minicandi import contract as C
    from minicandi.components import CountTransform
    from minicandi.train import set_pearson_stats
    m = _build(_model_cfg(env))
    _, b = _batch(env)
    tr = next(x for x in m.modules() if isinstance(x, CountTransform))

    def stats(batch):
        set_pearson_stats(m, [batch], read_start=True)
        return tr.lam.clone(), tr.r.clone()

    lam1, r1 = stats(b)
    sub = b["x_counts_sub"].clone()
    for t in ("ATAC-seq", "DNase-seq"):
        sub[..., C.TRACK_INDEX[t], 1] += 1000
    lam2, r2 = stats(dict(b, x_counts_sub=sub))
    assert torch.equal(lam1, lam2) and torch.equal(r1, r2)
    # the per-channel rate of a stranded track is half its strand-sum rate (both channels count)
    set_pearson_stats(m, [b])
    lam_sum = tr.lam.clone()
    h = C.TRACK_INDEX["H3K27ac"]
    vis = (b["x_avail"][:, None, :] & ~b["x_bin_mask"])[..., h]
    if vis.any():
        assert torch.isclose(lam1[h], lam_sum[h] / 2, rtol=1e-5)


# ------------------------------------------------------------------------------ training entry

def test_train_main_routes_a_read_start_config_to_res_dataset(env, tmp_path, monkeypatch):
    from minicandi import res_dataset, train as T
    seen = {}

    def fake(config, tables, seed, out, *args):
        seen["args"] = (config.get("read_start"), args[-1])
        return {k: 0 for k in ("windows_seen", "steps", "param_count", "flops_per_window",
                               "training_flops", "done")}

    monkeypatch.setattr(res_dataset, "train", fake)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(_model_cfg(env)))
    T.main(["--config", str(cfg), "--tables", str(env["tables"]), "--seed", "0",
            "--out", str(tmp_path / "o"), "--stage-dir", str(tmp_path / "st")])
    assert seen["args"] == (True, str(tmp_path / "st"))


def test_stage_read_starts_copies_and_keeps_the_batches(env, tmp_path):
    from minicandi.train import stage_read_starts
    ds0, b0 = _batch(env, seed=4)
    ds = _make(env, {"train_halves": "pairs"}, seed=4)
    st = stage_read_starts(ds, tmp_path / "stage")
    assert st["skipped"] is None and st["files"] > 1
    assert Path(ds.rs_store) == tmp_path / "stage" / "read_starts"
    assert str(ds.rs_average_h5).startswith(str(tmp_path / "stage" / "aux"))
    one = next(p for p in (tmp_path / "stage" / "read_starts").iterdir())
    assert (one / "starts.h5").is_file() and not (one / "starts.h5").is_symlink()
    assert (one / "starts_pr1.h5").is_symlink()
    b1 = ds.batch(2, np.random.default_rng([4, 9]))
    for k in ("x_counts_sub", "y_counts_sub", "avg_sub", "x_counts", "y_counts"):
        assert torch.equal(b0[k], b1[k]), k


def test_stage_read_starts_skips_when_the_disk_is_too_small(env, tmp_path, monkeypatch):
    import shutil
    from minicandi.train import stage_read_starts
    ds = _make(env, {"train_halves": "pairs"}, seed=4)
    before = Path(ds.rs_store)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10, 10, 0))
    st = stage_read_starts(ds, tmp_path / "stage")
    assert st["skipped"] and st["files"] == 0
    assert Path(ds.rs_store) == before


def test_rs_train_sh_header_and_command():
    sh = REPO / "slurm/t392/rs_train.sh"
    text = sh.read_text()
    for line in ("#SBATCH --cpus-per-task=12", "#SBATCH --mem=48000M", "#SBATCH --gres=none"):
        assert line in text
    env_ = dict(os.environ, DRY_RUN="1", SLURM_CPUS_PER_TASK="12", SLURM_TMPDIR="/tmpx")
    cmd = subprocess.run(["bash", str(sh), "/kit", "/c.json", "/tables", "1", "/out"],
                         env=env_, capture_output=True, text=True, check=True).stdout.split()
    assert cmd[:2] == ["python3", "/kit/tools/minicandi/train.py"]
    for a, v in (("--device", "cpu"), ("--workers", "11"), ("--seed", "1"),
                 ("--stage-dir", "/tmpx/stage")):
        assert cmd[cmd.index(a) + 1] == v
