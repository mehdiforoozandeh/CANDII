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
    assert (tmp_path / "b" / "CAPPED").is_file() and (tmp_path / "b" / "DONE").is_file()
    ck = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert ck["windows_seen"] == 16
    full = T.train(cfg, tables, 3, tmp_path / "b", log_every=1)
    assert full["steps"] == 8 and full["windows_seen"] == 32 and full["done"]
    assert full["n_resumes"] == 1 and (tmp_path / "b" / "DONE").is_file()
    assert not (tmp_path / "b" / "CAPPED").exists() and "capped" not in full
    steps = [json.loads(x)["step"] for x in (tmp_path / "b" / "log.jsonl").read_text().splitlines()]
    assert steps == list(range(1, 9))
    sa = torch.load(tmp_path / "a" / "ckpt.pt", weights_only=False)["state_dict"]
    sb = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)["state_dict"]
    assert straight["steps"] == 8
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_max_windows_cap_ends_in_capped_done_and_never_reads_as_full(tables, tmp_path):
    """Arm smoke defect 3: a run stopped by --max-windows wrote no DONE, so submit_arms.sh
    resubmitted it. It now ends in CAPPED + DONE; run.json says done false, capped true."""
    cfg = dict(SMALL, windows_seen=32)
    out = tmp_path / "c"
    part = T.train(cfg, tables, 0, out, max_windows=8, log_every=1)
    assert not part["done"] and part["capped"] and part["max_windows"] == 8
    cap = json.loads((out / "CAPPED").read_text())
    assert cap == {"windows_seen": 8, "max_windows": 8, "budget": 32}
    assert "capped" in (out / "DONE").read_text()
    run = json.loads((out / "run.json").read_text())
    assert run["done"] is False and run["capped"] is True and run["windows_seen"] == 8
    # the same command again (a resubmission) is a no-op, as for a full run
    before = (out / "ckpt.pt").stat().st_mtime_ns
    again = T.train(cfg, tables, 0, out, max_windows=8, log_every=1)
    assert (out / "ckpt.pt").stat().st_mtime_ns == before and again["capped"]
    # a larger cap extends it; CAPPED moves with it
    more = T.train(cfg, tables, 0, out, max_windows=16, log_every=1)
    assert more["windows_seen"] == 16 and more["capped"] and more["n_resumes"] == 1
    assert json.loads((out / "CAPPED").read_text())["windows_seen"] == 16
    # a full run never has the capped key
    full = T.train(dict(SMALL, windows_seen=8), tables, 0, tmp_path / "f", log_every=1)
    assert full["done"] and "capped" not in full and not (tmp_path / "f" / "CAPPED").exists()


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


# ------------------------------------------------------------------ F2: DataLoader workers

def _worker_batches(tables, seed, workers, n=3, first=0):
    ds = OnionDataset(tables, "train", dict(SMALL, windows_seen=64), seed)
    it = T.worker_batches(ds, 4, seed, first, first + n, workers)
    return [next(it) for _ in range(n)]


def _same(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def test_workers_same_seed_same_batches(tables):
    """Same seed and same N give identical batches across two runs; the stream does not depend on
    N or on the first step (so a resume draws the same batches)."""
    a, b = _worker_batches(tables, 7, 2), _worker_batches(tables, 7, 2)
    assert all(_same(x, y) for x, y in zip(a, b))
    assert not _same(a[0], a[1])
    assert _same(_worker_batches(tables, 7, 1, n=1, first=2)[0], a[2])      # N = 1, from step 2


def test_workers_run_resumes_exactly(tables, tmp_path):
    cfg = dict(SMALL, windows_seen=24)
    T.train(cfg, tables, 2, tmp_path / "w1", workers=1)
    T.train(cfg, tables, 2, tmp_path / "w3", max_windows=12, workers=1)
    info = T.train(cfg, tables, 2, tmp_path / "w3", workers=1)
    assert info["workers"] == 1 and info["done"] and info["n_resumes"] == 1
    s1, s3 = (torch.load(tmp_path / d / "ckpt.pt", weights_only=False)["state_dict"]
              for d in ("w1", "w3"))
    assert all(torch.equal(s1[k], s3[k]) for k in s1)
    with pytest.raises(ValueError, match="workers"):                       # 0 <-> N > 0 refused
        T.train(cfg, tables, 2, tmp_path / "w3x", max_windows=12)
        T.train(cfg, tables, 2, tmp_path / "w3x", workers=1)


def test_workers_zero_is_the_default_and_unrecorded(tables, tmp_path):
    info = T.train(dict(SMALL, windows_seen=8), tables, 0, tmp_path / "z")
    assert "workers" not in info
    st = torch.load(tmp_path / "z" / T.STATE_NAME, weights_only=False)
    assert "workers" not in st
    with pytest.raises(ValueError, match="workers"):
        T.train(dict(SMALL, windows_seen=8), tables, 0, tmp_path / "n", workers=-1)


@pytest.mark.parametrize("cpus,want", [("", "0"), ("1", "0"), ("8", "7")])
def test_train_sh_passes_workers(cpus, want):
    import os
    import subprocess
    env = dict(os.environ, DRY_RUN="1", SLURM_CPUS_PER_TASK=cpus)
    r = subprocess.run(["bash", str(_ROOT / "slurm/minicandi/train.sh"), "K", "c.json", "T", "0",
                        "O", "cpu"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-2:] == ["--workers", want]


def _fake_slurm(tmp_path, venv_rc: int, update_rc: int = 0) -> dict:
    """A PATH whose module/virtualenv/pip/sbatch/scontrol/squeue are logging fakes."""
    import os
    b = tmp_path / "bin"
    b.mkdir()
    log = tmp_path / "calls.log"
    rec = ("JobId=1234 JobName=mc_rung2_x_s0 TimeLimit=00:20:00 CPUs/Task=8 "
           "MinMemoryNode=32000M ExcNodeList=fc1 StdOut=/runs/x/seed0/train_1234.out")
    fakes = {
        "module": "exit 0",
        "virtualenv": f'mkdir -p "$3/bin"; : > "$3/bin/activate"; exit {venv_rc}',
        "pip": "exit 0",
        "sbatch": f'echo "sbatch $*" >> {log}; echo 4242',
        "scontrol": f'if [ "$1" = show ]; then echo "{rec}"; else echo "scontrol $*" >> {log}; exit {update_rc}; fi',
        "squeue": 'printf "777 afterok:1234(unfulfilled)\\n778 afterok:12345(unfulfilled)\\n"',
        "hostname": "echo badnode",
    }
    for name, body in fakes.items():
        (b / name).write_text(f"#!/bin/bash\n{body}\n")
        (b / name).chmod(0o755)
    (tmp_path / "tmpdir").mkdir()
    return dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", SLURM_JOB_ID="1234",
                SLURM_TMPDIR=str(tmp_path / "tmpdir"), USER="me")


@pytest.mark.parametrize("retried,update_rc", [(False, 0), (True, 0), (False, 1)])
def test_train_sh_setup_failure_retries_once_elsewhere(tmp_path, retried, update_rc):
    """Arm smoke defect 2: a CVMFS I/O error (exit 126) in setup killed 11 jobs for good. Now the
    job resubmits itself once, off this node, moves its afterok dependent and exits 3; a retried
    job never retries again."""
    import subprocess
    env = _fake_slurm(tmp_path, 126, update_rc)
    if retried:
        env["ONION_RETRY_OF"] = "1000"
    r = subprocess.run(["bash", str(_ROOT / "slurm/minicandi/train.sh"), "/kit", "c.json", "T",
                        "0", str(tmp_path / "out"), "cpu"], env=env, capture_output=True,
                       text=True)
    assert r.returncode == 3, r.stdout + r.stderr
    calls = (tmp_path / "calls.log").read_text().splitlines() \
        if (tmp_path / "calls.log").exists() else []
    if retried:
        assert calls == [] and "retried-as" not in r.stdout
        return
    assert len(calls) == 2, calls
    sb = calls[0].split()
    assert "--exclude=fc1,badnode" in sb and "--time=00:20:00" in sb and "--gres=none" in sb
    assert "--output=/runs/x/seed0/train_%j.out" in sb and "--job-name=mc_rung2_x_s0" in sb
    assert sb[-7:] == ["/kit/slurm/minicandi/train.sh", "/kit", "c.json", "T", "0",
                       str(tmp_path / "out"), "cpu"]
    assert calls[1] == "scontrol update JobId=777 Dependency=afterok:4242"   # not 778
    assert "[minicandi-train] retried-as=4242 host=badnode" in r.stdout
    moved = "[minicandi-train] dependent 777 now waits on 4242" in r.stdout
    failed = ("[minicandi-train] dependency-move-failed scoring_job=777 retry_job=4242"
              in r.stdout)
    assert (moved, failed) == ((True, False) if update_rc == 0 else (False, True))
