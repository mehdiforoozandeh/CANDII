"""The D10 and D15 arms on rung 1's base (chunk A1k; P7 of the parallel-iterations plan).

`tools/minicandi/configs/arms/stage2_rung1/{average_input,two_step_native}.json` are written by
`stage2_arms.py write --incumbent tools/minicandi/configs/arms/rung1/base.json --arms
average_input,two_step_native --average-h5 /scratch/mforooz/onion/aux/eic_v1_average.h5`.
Checks: (1) the committed files equal the writer's output (one change from rung 1's base);
(2) `--arms` writes only the named arms; (3) the per-position average is leave-one-cell-out:
read through an `average_h5`, a training cell's average equals the mean over the OTHER training
cells and differs from the mean that includes the cell; (4) each arm trains a few steps on
synthetic tables with a finite loss, and its batch carries the arm's extra input (`avg`, or
`c_base` and `y_base`) where the base batch does not.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import stage2_arms as S2  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset, build_average_h5  # noqa: E402

ARMS = ("average_input", "two_step_native")
FROZEN_AVERAGE = "/scratch/mforooz/onion/aux/eic_v1_average.h5"
BASE = _ROOT / "tools/minicandi/configs/arms/rung1/base.json"
OUT = _ROOT / "tools/minicandi/configs/arms/stage2_rung1"
L, BS, STEPS = 64, 4, 3


def _base() -> dict:
    return json.loads(BASE.read_text())


def _small(cfg: dict, **over) -> dict:
    return {**cfg, "context_bins": L, "batch_size": BS, "windows_seen": STEPS * BS,
            "param_budget": 1e5, **over}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("iter1")
    td = synth.make_synthetic_tables(root)
    avg = build_average_h5(td, root / "aux" / "average.h5")
    return {"root": root, "tables": td, "avg": avg}


def test_committed_configs_are_the_writers_output():
    base = _base()
    for arm in ARMS:
        disk = json.loads((OUT / f"{arm}.json").read_text())
        assert disk == S2.arm_config(base, arm, {"average_h5": FROZEN_AVERAGE}), arm
        assert S2.changed_keys(base, disk) == sorted(
            {"average_input": ["average_input", "average_h5"],
             "two_step_native": ["two_step"]}[arm])
        assert disk["windows_seen"] == 80000 and disk["rung"] == 1
    assert json.loads((OUT / "two_step_native.json").read_text())["base_depth"] == "native"
    assert sorted(p.stem for p in OUT.glob("*.json")) == sorted(ARMS)


def test_arms_filter_writes_only_the_named_arms(tmp_path):
    rows = S2.write(_base(), tmp_path, {"average_h5": "/a.h5"}, build_check=lambda c: None,
                    arms=list(ARMS))
    assert [r["arm"] for r in rows] == ["two_step_native", "average_input"]
    assert sorted(p.stem for p in tmp_path.glob("*.json")) == sorted(ARMS)
    lines = (tmp_path / "arms.tsv").read_text().splitlines()
    assert len(lines) == 3 and lines[0].split("\t") == list(S2.ARMS_TSV_COLUMNS)
    with pytest.raises(ValueError, match="unknown arms"):
        S2.plan(_base(), arms=["no_such_arm"], build_check=lambda c: None)
    # default: every arm, as before
    assert len(S2.plan(_base(), build_check=lambda c: None)) == len(S2.ARMS)


def test_average_is_leave_one_cell_out(env):
    cfg = _small(_base(), average_input=True)
    direct = OnionDataset(env["tables"], "train", cfg, 0)
    via_h5 = OnionDataset(env["tables"], "train", {**cfg, "average_h5": str(env["avg"])}, 0)
    chrom, start = direct.chroms[0], 128
    checked = 0
    for cell in direct.train_cells:
        a_h5 = via_h5.average(cell, chrom, start)
        a_dir = direct.average(cell, chrom, start)
        np.testing.assert_allclose(a_h5, a_dir, rtol=1e-4, atol=1e-3)
        # the mean over ALL training cells (own cell included) differs on the cell's own tracks
        for (c, a), m in direct.train_base.items():
            if c != cell:
                continue
            t = C.TRACK_INDEX[a]
            others = [mm for (cc, aa), mm in direct.train_base.items() if aa == a and cc != cell]
            rates = [direct._rate(mm, chrom, start, L) for mm in others + [m]]
            incl = np.mean(rates, axis=0)
            excl = np.mean(rates[:-1], axis=0)
            np.testing.assert_allclose(a_dir[:, t], excl, rtol=1e-5, atol=1e-6)
            assert np.abs(incl - excl).max() > 1e-3
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("arm", ARMS)
def test_arm_trains_and_its_input_differs_from_base(env, arm):
    over = {"average_h5": str(env["avg"])} if arm == "average_input" else {}
    cfg = _small(json.loads((OUT / f"{arm}.json").read_text()), **over)
    run = env["root"] / "runs" / arm
    T.train(cfg, env["tables"], 0, run, log_every=1)
    losses = [json.loads(x)["loss"] for x in (run / "log.jsonl").read_text().splitlines()]
    assert len(losses) == STEPS and all(math.isfinite(x) for x in losses), losses

    rng = lambda: np.random.default_rng(1)  # noqa: E731
    b_arm = OnionDataset(env["tables"], "train", T.full_config(cfg), 1).batch(BS, rng())
    b_base = OnionDataset(env["tables"], "train", T.full_config(_small(_base())), 1).batch(
        BS, rng())
    extra = {"average_input": ("avg",), "two_step_native": ("c_base", "y_base")}[arm]
    for k in extra:
        assert k in b_arm and k not in b_base, k
        assert b_arm[k].float().abs().sum() > 0, k
    # the shared keys are the same stream: the arm only adds an input
    for k in ("x_counts", "y_counts", "c_in", "c_out"):
        assert (b_arm[k] == b_base[k]).all(), k


# ------------------------------------------------- train.py fixes (foreman, 2026-11-14)
import torch  # noqa: E402

from minicandi import registry as R  # noqa: E402


def _ckpt(run):
    return torch.load(run / "ckpt.pt", weights_only=False)["state_dict"]


def test_staging_copies_the_average_h5_and_keeps_the_run_identical(env, tmp_path, capsys):
    cfg = _small(json.loads((OUT / "average_input.json").read_text()),
                 average_h5=str(env["avg"]))
    T.train(cfg, env["tables"], 0, tmp_path / "plain", log_every=1)
    T.train(cfg, env["tables"], 0, tmp_path / "staged", log_every=1, stage_dir=tmp_path / "st")
    out = capsys.readouterr().out
    staged = tmp_path / "st" / "aux" / env["avg"].name
    assert f"to {staged}" in out and staged.is_file()
    assert staged.stat().st_size == env["avg"].stat().st_size
    a, b = _ckpt(tmp_path / "plain"), _ckpt(tmp_path / "staged")
    assert all(torch.equal(a[k], b[k]) for k in a)
    losses = [json.loads(x)["loss"] for x in (tmp_path / "plain" / "log.jsonl").read_text()
              .splitlines()]
    assert losses == [json.loads(x)["loss"] for x in (tmp_path / "staged" / "log.jsonl")
                      .read_text().splitlines()]
    # the run's config keeps the original path; only the dataset reads the copy
    run = json.loads((tmp_path / "staged" / "run.json").read_text())
    assert run["config"]["average_h5"] == str(env["avg"])
    ds = OnionDataset(env["tables"], "train", T.full_config(cfg), 0)
    info = T.stage_average(ds, tmp_path / "st2")
    assert ds.config["average_h5"] == info["dst"] and Path(info["dst"]).is_file()
    assert T.stage_average(OnionDataset(env["tables"], "train", T.full_config(_small(_base())),
                                        0), tmp_path / "st3") is None


@pytest.mark.parametrize("arm", ("base",) + ARMS)
def test_one_window_probe_gives_the_old_flops_per_window(env, arm):
    cfg = _small(_base() if arm == "base" else json.loads((OUT / f"{arm}.json").read_text()))
    if arm == "average_input":
        cfg["average_h5"] = str(env["avg"])
    cfg = T.full_config(cfg)
    ds = OnionDataset(env["tables"], "train", cfg, 0)
    torch.manual_seed(0)
    model = R.build_model(cfg).train()
    full = ds.batch(BS, np.random.default_rng([0, 2]))
    one = ds.batch(1, np.random.default_rng([0, 2]))
    assert one["meta"][0] == full["meta"][0]          # the first window of the old probe
    old = T.flops_per_window(model, full, "cpu", cfg)
    new = T.flops_per_window(model, one, "cpu", cfg)
    assert new == old, (arm, old, new)


def test_probe_oom_never_stops_a_training(env, tmp_path, monkeypatch, capsys):
    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
    monkeypatch.setattr(T, "flops_per_window", oom)
    T.train(_small(_base()), env["tables"], 0, tmp_path / "run", log_every=1)
    assert "flops_per_window: unavailable (OOM)" in capsys.readouterr().out
    recs = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert len(recs) == STEPS and all(r["training_flops"] is None for r in recs)
    assert all(math.isfinite(r["loss"]) for r in recs)
    assert (tmp_path / "run" / "DONE").is_file()
