"""Predict command and its job script (plan t152, Amendment 10, pinned interface M; chunk SP2a).

On synthetic tables and a saved rung-1 run: `predict_store.py run` in 2 shards gives the store that
`predict()` gives in one; a rerun stores nothing new; both TF32 flags are false inside; another
evaluation set is refused with the store's message; `predict.sh` under DRY_RUN=1 prints the
command with each variable."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import registry as R  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import predict_store as P  # noqa: E402
from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (6, "H3K27me3"), (7, "chipseq-control"))
L, RB, LEGACY_B, GW = 64, 256, 4, 8
SCRIPT = REPO / "slurm/onion_eval/predict.sh"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("prun")
    t = make_synthetic_tables(root, n_cells=8, val_tracks=VAL, blind_tracks=())
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = ev["correction_pairs"][:4]
    out.write_text(json.dumps(ev))
    other = out.with_name("evalset_other.json")              # same targets, another file hash
    other.write_text(json.dumps(dict(ev, note="another evaluation set")))
    cfg = {**json.loads((REPO / "tools/minicandi/configs/rung1/base.json").read_text()),
           "context_bins": L, "batch_size": LEGACY_B}
    torch.manual_seed(1)
    m = R.build_model(cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():       # the NB head starts at zero: perturb so outputs read the inputs
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    run = root / "runs" / "rung1_base" / "seed0"
    run.mkdir(parents=True)
    torch.save({"state_dict": m.state_dict(), "config": cfg, "param_count": m.param_count(),
                "windows_seen": 0, "tables_sha256": tables_sha256(t), "git_sha": "test"},
               run / R.CKPT_NAME)
    return {"tables": t, "evalset": out, "other": other, "run": run}


def _cli(env, store, *extra, evalset=None):
    return P.run(["--run", str(env["run"]), "--evalset", str(evalset or env["evalset"]),
                  "--pred-store", str(store), "--device", "cpu",
                  "--guardrail-windows", str(GW), *extra])


def _reference(env, store):
    """The store that predict() gives in one shard, from the scorer score.py builds for the run."""
    model, cfg = R.load_run(env["run"])
    sc = S.Scorer(env["evalset"], main="model", model=model, config=cfg, device="cpu",
                  run_tables_sha256=S.run_tables_sha256(env["run"]), guardrail_windows=GW,
                  log=lambda *a: None)
    reqs = P.requests(sc)
    P.predict(sc, reqs, store, LEGACY_B, run=env["run"])
    return sc, reqs


def test_two_shards_equal_predict_and_rerun_stores_nothing(env, tmp_path, capsys):
    sc, reqs = _reference(env, tmp_path / "ref")
    d = tmp_path / "cli"
    assert _cli(env, d, "--shard", "0/2") == 0
    st = P.PredictionStore.for_scorer(d, sc, run=env["run"])
    assert not st.complete(reqs)                                   # shard 1 missing
    assert _cli(env, d, "--shard", "1/2", "--stage-dir", str(tmp_path / "stage")) == 0
    assert st.complete(reqs) and any((tmp_path / "stage").rglob("counts.h5"))
    head = st.info
    assert head["n_shards"] == 2 and head["infer_batch"] == LEGACY_B and head["device"] == "cpu"
    ref = P.PredictionStore(tmp_path / "ref")
    for q in reqs:
        x, y = st.read(q["key"]), ref.read(q["key"])
        assert set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x), q["key"]
    assert (json.loads((d / P.GUARDRAIL_FILE).read_text())
            == json.loads((tmp_path / "ref" / P.GUARDRAIL_FILE).read_text()))
    out = capsys.readouterr().out
    assert "timing by phase:" in out and "forward_seconds" in out
    # rerun: nothing new is stored and no array file is rewritten
    files = sorted(d.glob("*.npz"))
    stamp = {f: f.stat().st_mtime_ns for f in files}
    assert _cli(env, d, "--shard", "0/2") == 0 and _cli(env, d, "--shard", "1/2") == 0
    out = capsys.readouterr().out
    assert out.count("stored: 0 new") == 2
    assert {f: f.stat().st_mtime_ns for f in sorted(d.glob("*.npz"))} == stamp


def test_tf32_flags_are_false_inside_run(env, tmp_path, monkeypatch):
    seen = []
    inner = P.request_arrays

    def spy(*a, **k):
        seen.append(P.tf32_flags())
        return inner(*a, **k)
    monkeypatch.setattr(P, "request_arrays", spy)
    old = P.tf32_flags()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        assert _cli(env, tmp_path / "s", "--infer-batch", "8") == 0
        assert P.tf32_flags() == (True, True)                     # restored after the phase
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    assert seen and all(f == (False, False) for f in seen)
    head = json.loads((tmp_path / "s" / P.STORE_FILE).read_text())
    assert head["tf32"] is False and head["amp"] is False and head["infer_batch"] == 8


def test_another_evalset_is_refused_with_the_store_message(env, tmp_path, capsys):
    d = tmp_path / "s"
    assert _cli(env, d, "--shard", "0/4") == 0
    n = len(list(d.glob("*.npz")))
    capsys.readouterr()
    assert _cli(env, d, "--shard", "1/4", evalset=env["other"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("refused: ") and "a prediction store of another scoring" in err
    assert "evalset_sha256" in err
    assert len(list(d.glob("*.npz"))) == n                         # nothing written


def _dry(tmp_path, **envv):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("ONION_", "SLURM_")) and k != "CUDA_VISIBLE_DEVICES"}
    e.update(DRY_RUN="1", **envv)
    r = subprocess.run(["bash", str(SCRIPT), "/kit", "/runs/r/seed0", "/ev/evalset.json",
                        "/store/r/seed0/p16"], env=e, capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def test_predict_sh_dry_run_prints_the_command_with_each_variable(tmp_path):
    head, cmd = _dry(tmp_path)
    assert head == ("ONION_SHARD=0/1 ONION_DEVICE=cpu ONION_INFER_BATCH=run batch_size "
                    "ONION_STAGE=1 ONION_PREDICT_THREADS=1")
    assert cmd == ("python3 /kit/tools/onion_eval/predict_store.py run --run /runs/r/seed0 "
                   "--evalset /ev/evalset.json --pred-store /store/r/seed0/p16 --shard 0/1 "
                   "--device cpu")
    head, cmd = _dry(tmp_path, ONION_SHARD="3/8", ONION_INFER_BATCH="32", SLURM_JOB_GPUS="0",
                     SLURM_CPUS_PER_TASK="4", SLURM_TMPDIR="/tmp/job")
    assert head == ("ONION_SHARD=3/8 ONION_DEVICE=cuda ONION_INFER_BATCH=32 ONION_STAGE=1 "
                    "ONION_PREDICT_THREADS=4")
    assert cmd.endswith("--shard 3/8 --device cuda --infer-batch 32 --stage-dir /tmp/job/stage")
    head, cmd = _dry(tmp_path, ONION_DEVICE="cpu", ONION_STAGE="0", SLURM_JOB_GPUS="0",
                     SLURM_TMPDIR="/tmp/job", ONION_PREDICT_THREADS="8")
    assert "ONION_DEVICE=cpu" in head and "ONION_STAGE=0" in head and "THREADS=8" in head
    assert cmd.endswith("--device cpu") and "--stage-dir" not in cmd
    bad = subprocess.run(["bash", str(SCRIPT), "/kit", "/r", "/e", "/s"], capture_output=True,
                         text=True, env={**os.environ, "DRY_RUN": "1", "ONION_SHARD": "x"})
    assert bad.returncode == 2 and "want i/N" in bad.stderr
