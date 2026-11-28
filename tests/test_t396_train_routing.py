"""t396 iteration 2, chunk A1: `train.py` routes every t396 arm to its trainer and stages every
file that arm reads.

On the synthetic tables and read-start store of `tests/test_t392_end_to_end.py`:
- `kernel_head` without `read_start` exits non-zero and writes nothing;
- a kernel-head run with `--stage-dir` opens no measurement h5 file outside the stage dir after
  staging (the genome files are not staged by any run), and its first 3 batches equal those of
  an unstaged run;
- too little free space stages nothing and logs `staging skipped`;
- a count config (neither key) still runs `train.train`; `read_start` or `kernel_head` runs
  `res_dataset.train`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import torch

from tests.test_t392_end_to_end import env  # noqa: F401  (the module-scoped synthetic fixture)

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "tools" / "minicandi" / "train.py"


def _kh_cfg(env, windows=24):
    cfg = dict(env["cfg"])
    cfg.update({"kernel_head": True, "windows_seen": windows})
    return cfg


def _record_batches(monkeypatch):
    """Record the tensors of every batch that reaches the kernel-head loss."""
    from minicandi import kernel_head as KH
    seen = []
    orig = KH.kernel_head_loss

    def rec(out, batch, config):
        seen.append({k: v.detach().clone() for k, v in batch.items() if torch.is_tensor(v)})
        return orig(out, batch, config)
    monkeypatch.setattr(KH, "kernel_head_loss", rec)
    return seen


def _train_kh(env, out, stage_dir=None):
    from minicandi.res_dataset import train
    torch.manual_seed(0)
    return train(_kh_cfg(env), env["tables"], 0, out, log_every=1000, stage_dir=stage_dir)


def test_kernel_head_without_read_start_exits_nonzero(env, tmp_path):
    cfg = dict(env["cfg"])
    cfg.pop("read_start", None)
    cfg["kernel_head"] = True
    cp = tmp_path / "cfg.json"
    cp.write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(TRAIN), "--config", str(cp), "--tables",
                        str(env["tables"]), "--seed", "0", "--out", str(out)],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{REPO / 'tools'}:{REPO / 'src'}"})
    assert r.returncode != 0
    assert "kernel_head needs read_start" in r.stderr
    assert not out.exists()


def test_kernel_head_staged_reads_only_the_stage_dir_and_same_batches(env, tmp_path, monkeypatch):
    from minicandi import train as T
    stage = tmp_path / "stage"
    opened, after = [], {"on": False}
    orig_file, orig_stage_counts = h5py.File, T.stage_counts

    def file_rec(path, *a, **kw):
        if after["on"]:
            opened.append(str(path))
        return orig_file(path, *a, **kw)

    def stage_counts_rec(*a, **kw):
        r = orig_stage_counts(*a, **kw)
        after["on"] = True                     # the last staging step of a kernel-head run
        return r
    monkeypatch.setattr(h5py, "File", file_rec)
    monkeypatch.setattr(T, "stage_counts", stage_counts_rec)

    seen_staged = _record_batches(monkeypatch)
    _train_kh(env, tmp_path / "staged", stage_dir=stage)
    assert after["on"], "stage_counts was not called for a kernel-head run"
    assert opened, "no h5 file was opened after staging"
    # the genome (dna.h5, mask.h5) is not a measurement store and no run stages it
    outside = sorted({p for p in opened
                      if not p.startswith(str(stage)) and Path(p).parent.name != "genome"})
    assert not outside, outside[:5]
    assert any("/read_starts/" in p for p in opened)          # starts read from the stage dir
    assert any(p.endswith("counts.h5") and "/read_starts/" not in p for p in opened)  # overlaps

    monkeypatch.setattr(T, "stage_counts", orig_stage_counts)
    monkeypatch.undo()                         # unwrap the recorder before the unstaged run
    seen_plain = _record_batches(monkeypatch)
    _train_kh(env, tmp_path / "plain")
    assert len(seen_staged) >= 3 and len(seen_plain) == len(seen_staged)
    for a, b in zip(seen_staged[:3], seen_plain[:3]):
        assert a.keys() == b.keys() and "y_counts_overlap" in a
        for k in a:
            assert torch.equal(a[k], b[k]), k


def test_too_little_space_stages_nothing(env, tmp_path, monkeypatch, capsys):
    import shutil
    from collections import namedtuple
    du = namedtuple("du", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: du(10, 10, 0))
    stage = tmp_path / "stage"
    _train_kh(env, tmp_path / "run", stage_dir=stage)
    assert "staging skipped" in capsys.readouterr().out
    assert not [p for p in stage.rglob("*") if p.is_file()]


@pytest.mark.parametrize("cfg,want", [({}, "train"), ({"read_start": True}, "res"),
                                      ({"kernel_head": True}, "res"),
                                      ({"read_start": True, "kernel_head": True}, "res")])
def test_main_routes_by_config(tmp_path, monkeypatch, cfg, want):
    from minicandi import res_dataset as R
    from minicandi import train as T
    called = []
    info = {"windows_seen": 0, "steps": 0, "param_count": 0, "flops_per_window": 0,
            "training_flops": 0, "done": True}
    monkeypatch.setattr(T, "train", lambda *a, **k: called.append("train") or info)
    monkeypatch.setattr(R, "train", lambda *a, **k: called.append("res") or info)
    cp = tmp_path / "cfg.json"
    cp.write_text(json.dumps(cfg), encoding="utf-8")
    assert T.main(["--config", str(cp), "--tables", str(tmp_path), "--seed", "0",
                   "--out", str(tmp_path / "run")]) == 0
    assert called == [want]
