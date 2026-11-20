"""Chunk FC (fair chance): the optional `heldout_loss` key of the train loop, and the configs that
give each ladder rung a fair chance (learning rate, latent count, matched compute, 160k windows)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
SMALL = {"rung": 1, "context_bins": L, "batch_size": 4, "param_budget": 1.4e5, "lr": 3e-3,
         "dna": "own", "windows_seen": 16}
HELD = {"batches": 2, "every": 2}
CFG = _ROOT / "tools/minicandi/configs"
BASE_SHA = "cc99d65"              # the tree before chunk FC
H = {"batches": 16, "every": 20}
# name -> (rung, source under configs/arms/rung<r>/, the one named change)
NEW = {"rung6/lr1e-3": (6, "base", {"lr": 0.001}),
       "rung6/latents_large": (6, "latents_large", {}),
       "rung5/matched_compute": (5, "matched_compute", {}),
       "rung5/lr1e-3": (5, "base", {"lr": 0.001}),
       "rung2/windows160k": (2, "base", {"windows_seen": 160000}),
       "rung3/windows160k": (3, "base", {"windows_seen": 160000}),
       "rung4/windows160k": (4, "base", {"windows_seen": 160000}),
       **{f"rung{r}/base_heldout": (r, "base", {}) for r in (2, 3, 4, 5, 6)}}


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_fc"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("wall_seconds", "git_sha", "host")}


def _log(out: Path) -> list:
    return [_strip(json.loads(x)) for x in (out / "log.jsonl").read_text().splitlines()]


def _overlaps(chrom, start, n, regions) -> bool:
    return any(c == chrom and s < start + n and start < s + m for c, s, m in regions)


def test_default_off_is_bit_identical_to_the_tree_before(tables, tmp_path):
    src = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_SHA}:tools/minicandi/train.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip(f"git show {BASE_SHA} failed: {src.stderr.strip()}")
    (tmp_path / "train_before.py").write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("train_before", tmp_path / "train_before.py")
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    a = T.train(dict(SMALL), tables, 0, tmp_path / "now", log_every=1)
    b = before.train(dict(SMALL), tables, 0, tmp_path / "before", log_every=1)
    assert _strip(a) == _strip(b) and "heldout" not in a
    assert _log(tmp_path / "now") == _log(tmp_path / "before")
    assert not any("heldout_loss" in r or "exclude_rejections" in r for r in _log(tmp_path / "now"))
    sa = torch.load(tmp_path / "now" / "ckpt.pt", weights_only=False)["state_dict"]
    sb = torch.load(tmp_path / "before" / "ckpt.pt", weights_only=False)["state_dict"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_heldout_batches_are_the_same_for_every_run_seed(tables, tmp_path):
    cfg = dict(SMALL, heldout_loss=HELD)
    x, y = T.heldout_batches(tables, cfg), T.heldout_batches(tables, cfg)
    for bx, by in zip(x, y):
        assert torch.equal(bx["x_counts"], by["x_counts"]) and bx["meta"] == by["meta"]
    runs = [T.train(cfg, tables, s, tmp_path / f"s{s}", log_every=1) for s in (0, 1)]
    assert runs[0]["heldout"] == runs[1]["heldout"]
    assert runs[0]["heldout"]["windows"] == HELD["batches"] * SMALL["batch_size"]
    logs = [[r for r in _log(tmp_path / f"s{s}") if "heldout_loss" in r] for s in (0, 1)]
    assert [r["step"] for r in logs[0]] == [2, 4] == [r["step"] for r in logs[1]]
    assert all(np.isfinite(r["heldout_loss"]) for r in logs[0] + logs[1])
    assert "last_heldout_loss" in runs[0]


def test_heldout_windows_are_off_the_training_sampler_and_the_eval_regions(tables):
    cfg = T.full_config(dict(SMALL, heldout_loss=HELD))
    held = T.heldout_batches(tables, cfg)
    loci = [(m["chrom"], m["start_bin"], L) for b in held for m in b["meta"]]
    ev = json.loads((tables.parent.parent / "eval" / tables.name / "evalset.json").read_text())
    evr = [(c, s, 4 * L) for c, s in ev["regions"]]
    assert not any(_overlaps(c, s, n, evr) for c, s, n in loci)
    for seed in (0, 1):
        ds = OnionDataset(tables, "train", cfg, seed)
        ds.exclude_windows(loci)
        rng = np.random.default_rng([seed, 1])
        drawn = [ds.plan(rng) for _ in range(400)]
        assert not any(_overlaps(p["chrom"], p["start_bin"], L, loci) for p in drawn)
        assert sum(p["exclude_rejections"] for p in drawn) > 0     # the exclusion was exercised


def _source(name: str) -> dict:
    r, src, change = NEW[name]
    return dict(json.loads((CFG / f"arms/rung{r}/{src}.json").read_text()), **change)


@pytest.mark.parametrize("name", sorted(NEW))
def test_new_config_is_its_source_plus_the_named_change_and_builds(name, capsys):
    cfg = json.loads((CFG / f"{name}.json").read_text())
    assert cfg == dict(_source(name), heldout_loss=H)
    model = R.build_model(T.full_config(cfg))
    n = model.param_count()
    with capsys.disabled():
        print(f"\n{name}: {n} parameters")
    assert abs(n - cfg["param_budget"]) <= 0.1 * cfg["param_budget"]
    r, src, _ = NEW[name]
    if src == "base":                    # the held-out key and lr / windows move no parameter
        base = json.loads((CFG / f"arms/rung{r}/base.json").read_text())
        torch.manual_seed(0)
        a = R.build_model(T.full_config(base)).state_dict()
        torch.manual_seed(0)
        b = R.build_model(T.full_config(cfg)).state_dict()
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_rung5_matched_compute_windows_follow_budgets():
    bud = json.loads((CFG / "budgets.json").read_text())
    w = round(bud["flops_matched_compute"] / bud["flops_per_window_measured"]["5"])
    assert w == 106667 == bud["windows_seen_at_matched_compute"]["5"]
    assert json.loads((CFG / "rung5/matched_compute.json").read_text())["windows_seen"] == w
