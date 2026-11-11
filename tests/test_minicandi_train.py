"""miniCANDI train loop and registry (chunk T1): CLI, ckpt.pt, load_run, resume, FLOPs, scramble."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
SMALL = {"rung": 1, "context_bins": L, "batch_size": 4, "param_budget": 1.4e5, "lr": 3e-3,
         "dna": "own"}


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_t1"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


def _batch(tables, cfg, seed=0, n=4):
    return OnionDataset(tables, "train", cfg, seed).batch(n, np.random.default_rng(seed))


# ------------------------------------------------------------------------------- registry


def test_registry_maps_the_six_rungs_lazily():
    assert R.RUNGS == {1: "pointwise", 2: "conv_ae", 3: "conv_attn_ae", 4: "unet", 5: "axial",
                       6: "perceiver"}
    assert R.module_name(1) == R.module_name("1") == "pointwise"
    with pytest.raises(ValueError):
        R.module_name(7)
    with pytest.raises(ValueError):
        R.module_name(True)
    assert "pointwise" in R.available()
    m = R.build_model(dict(SMALL))
    assert isinstance(m, C.MiniCandi)


FAKE = '''
import torch
from minicandi.contract import MiniCandi


class Model(MiniCandi):
    def __init__(self, config):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((1,), float(config.get("fake_value", 3.0))))

    def forward(self, batch):
        z = torch.zeros(batch["y_counts"].shape) + self.w
        return {"n": z.exp(), "mu": z.exp()}
'''


def test_registry_finds_a_rung_module_dropped_into_rungs(tables, tmp_path):
    name = f"_t1_fake_{uuid.uuid4().hex[:8]}"
    path = R.RUNGS_DIR / f"{name}.py"
    try:
        path.write_text(FAKE, encoding="utf-8")
        assert name in R.available()
        m = R.build_model({"rung": name, "fake_value": 1.5})
        assert isinstance(m, C.MiniCandi) and float(m.w) == 1.5
        # and through the full ckpt.pt -> load_run path
        run = tmp_path / "fake_run"
        run.mkdir()
        torch.save(C.make_ckpt(m, {"rung": name, "fake_value": 1.5}, 0, "x", "y"),
                   run / "ckpt.pt")
        m2, cfg = R.load_run(run)
        assert cfg["rung"] == name and float(m2.w) == 1.5 and not m2.training
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"{R.PACKAGE}.{name}", None)
    assert name not in R.available()


def test_registry_rejects_a_module_without_model(tmp_path):
    name = f"_t1_nomodel_{uuid.uuid4().hex[:8]}"
    path = R.RUNGS_DIR / f"{name}.py"
    try:
        path.write_text("X = 1\n", encoding="utf-8")
        with pytest.raises(TypeError):
            R.build_model({"rung": name})
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"{R.PACKAGE}.{name}", None)
    with pytest.raises(ModuleNotFoundError):
        R.build_model({"rung": "no_such_rung_module"})


def test_save_load_run_gives_identical_outputs(tables, tmp_path):
    torch.manual_seed(0)
    m = R.build_model(dict(SMALL))
    with torch.no_grad():                                 # move the zero-initialised head
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    m.eval()
    b = _batch(tables, SMALL)
    with torch.no_grad():
        a = m(b)
    torch.save(C.make_ckpt(m, SMALL, 123, "sha", "git"), tmp_path / "ckpt.pt")
    state = torch.get_rng_state()
    m2, cfg = R.load_run(tmp_path)
    assert torch.equal(state, torch.get_rng_state())     # loading draws nothing
    assert cfg == SMALL and not m2.training
    with torch.no_grad():
        b2 = m2(b)
    for k in ("n", "mu"):
        assert torch.equal(a[k], b2[k])


# ------------------------------------------------------------------------------- train loop


def test_cli_trains_lowers_the_loss_and_writes_the_pinned_files(tables, tmp_path):
    cfg = dict(SMALL, batch_size=8, windows_seen=8 * 60)
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg))
    out = tmp_path / "run"
    rc = T.main(["--config", str(cpath), "--tables", str(tables), "--seed", "0", "--out", str(out),
                 "--device", "cpu", "--log-every", "5", "--val-every", "160",
                 "--val-batches", "2"])
    assert rc == 0
    for f in ("ckpt.pt", "run.json", "DONE", "log.jsonl", "train_state.pt"):
        assert (out / f).is_file(), f
    ck = torch.load(out / "ckpt.pt", weights_only=False)
    C.check_ckpt(ck)
    assert ck["config"] == T.full_config(cfg) and ck["windows_seen"] == 480
    run = json.loads((out / "run.json").read_text())
    for k in ("config", "seed", "git_sha", "tables_sha256", "param_count", "flops_per_window",
              "training_flops", "windows_seen", "wall_seconds", "device"):
        assert k in run, k
    assert run["windows_seen"] == 480 and run["steps"] == 60 and run["device"] == "cpu"
    assert run["training_flops"] == pytest.approx(480 * run["flops_per_window"])
    assert run["tables_sha256"] == ck["tables_sha256"] and run["param_count"] == ck["param_count"]
    recs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    train_loss = [r["loss"] for r in recs if "loss" in r]
    val_loss = [r["val_loss"] for r in recs if "val_loss" in r]
    assert np.mean(train_loss[-3:]) < 0.9 * np.mean(train_loss[:2])
    assert len(val_loss) == 3 and val_loss[-1] < val_loss[0]
    # rerun with DONE present: a no-op
    before = (out / "ckpt.pt").stat().st_mtime_ns
    T.main(["--config", str(cpath), "--tables", str(tables), "--seed", "0", "--out", str(out)])
    assert (out / "ckpt.pt").stat().st_mtime_ns == before


def test_resume_continues_the_step_count_and_is_exact(tables, tmp_path):
    cfg = dict(SMALL, windows_seen=32, dna="shuffled", scramble_covariates=True)
    straight = T.train(cfg, tables, 3, tmp_path / "a", log_every=1)
    part = T.train(cfg, tables, 3, tmp_path / "b", max_windows=16, log_every=1)
    assert part["windows_seen"] == 16 and part["steps"] == 4 and not part["done"]
    assert not (tmp_path / "b" / "DONE").exists()
    ck = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert ck["windows_seen"] == 16
    full = T.train(cfg, tables, 3, tmp_path / "b", log_every=1)
    assert full["steps"] == 8 and full["windows_seen"] == 32 and full["done"]
    assert full["n_resumes"] == 1 and (tmp_path / "b" / "DONE").is_file()
    steps = [json.loads(x)["step"] for x in (tmp_path / "b" / "log.jsonl").read_text().splitlines()]
    assert steps == list(range(1, 9))
    sa = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=False)["state_dict"]
    sb = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)["state_dict"]
    assert straight["steps"] == 8
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_resume_refuses_a_changed_config(tables, tmp_path):
    cfg = dict(SMALL, windows_seen=16)
    T.train(cfg, tables, 0, tmp_path / "r", max_windows=8)
    with pytest.raises(ValueError, match="config"):
        T.train(dict(cfg, lr=1e-4), tables, 0, tmp_path / "r")


def test_same_seed_same_run(tables, tmp_path):
    cfg = dict(SMALL, windows_seen=12)
    T.train(cfg, tables, 5, tmp_path / "x")
    T.train(cfg, tables, 5, tmp_path / "y")
    T.train(cfg, tables, 6, tmp_path / "z")
    sx, sy, sz = (torch.load(tmp_path / d / "ckpt.pt", weights_only=False)["state_dict"]
                  for d in "xyz")
    assert all(torch.equal(sx[k], sy[k]) for k in sx)
    assert not all(torch.equal(sx[k], sz[k]) for k in sx)


def test_scramble_covariates_is_applied_by_the_loop(tables, tmp_path, monkeypatch):
    calls = []
    real = T.scramble_covariates

    def spy(c, avail, generator=None):
        calls.append(tuple(c.shape))
        return real(c, avail, generator)

    monkeypatch.setattr(T, "scramble_covariates", spy)
    T.train(dict(SMALL, windows_seen=8), tables, 0, tmp_path / "off")
    assert calls == []
    T.train(dict(SMALL, windows_seen=8, scramble_covariates=True), tables, 0, tmp_path / "on")
    assert len(calls) == 4                     # 2 steps x (c_in, c_out)
    assert calls[1][-1] == C.K_META


def test_flops_per_window_counts_forward_and_backward_and_scales_with_context(tables):
    out = {}
    for ctx in (32, 64, 128):
        cfg = dict(SMALL, context_bins=ctx)
        torch.manual_seed(0)
        m = R.build_model(cfg)
        b = _batch(tables, cfg)
        state = torch.get_rng_state()
        out[ctx] = T.flops_per_window(m, b, "cpu")
        assert torch.equal(state, torch.get_rng_state())
        assert all(p.grad is None for p in m.parameters())
    (t32, f32), (t64, _), (t128, _) = out[32], out[64], out[128]
    assert f32 > 0 and t32 > 2 * f32                   # backward adds about 2x forward
    # affine in the context: per-bin work plus a per-track covariate part that does not scale
    assert t128 - t64 == pytest.approx(2 * (t64 - t32), rel=1e-9)
    assert t64 > 1.5 * t32


def test_pearson_statistics_are_set_from_training_data(tables, tmp_path):
    info = T.train(dict(SMALL, windows_seen=4, count_transform="pearson"), tables, 0,
                   tmp_path / "p")
    m, _ = R.load_run(tmp_path / "p")
    lam = m.transform.lam
    for a in ("DNase-seq", "H3K27ac", "H3K4me3"):
        assert lam[C.TRACK_INDEX[a]] != 1.0
    assert lam[C.TRACK_INDEX["H3K9me3"]] == 1.0       # not in the synthetic panel: default
    assert info["done"]


def test_lr_schedule_warms_up_and_decays_to_the_floor():
    lrs = [T.lr_at(s, 100, 1.0) for s in range(100)]
    assert lrs[0] < lrs[4] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(T.LR_FLOOR, abs=0.01)
    assert all(a >= b for a, b in zip(lrs[5:], lrs[6:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu_shape(tables, tmp_path):
    info = T.train(dict(SMALL, windows_seen=8), tables, 0, tmp_path / "g", device="cuda")
    assert info["device"].startswith("cuda") and info["done"]
