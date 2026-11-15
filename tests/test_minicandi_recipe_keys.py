"""Chunk S1: the optional recipe keys of the train loop (`weight_decay`, `grad_clip`,
`warmup_frac`, `warmup_max`) and the unknown-key guard (`train.check_config_keys`)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402

L = 64
SMALL = {"rung": 1, "context_bins": L, "batch_size": 4, "param_budget": 1.4e5, "lr": 3e-3,
         "dna": "own", "windows_seen": 16}               # 4 steps: default warm-up is 1 step
BASE_SHA = "0e9abfc"              # the tree before chunk S1
CFG = _ROOT / "tools/minicandi/configs"


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_s1"))


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("wall_seconds", "git_sha", "host")}


def _log(out: Path) -> list:
    return [_strip(json.loads(x)) for x in (out / "log.jsonl").read_text().splitlines()]


def test_defaults_are_bit_identical_to_the_tree_before(tables, tmp_path):
    src = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_SHA}:tools/minicandi/train.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip(f"git show {BASE_SHA} failed: {src.stderr.strip()}")
    (tmp_path / "train_before.py").write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("train_before_s1", tmp_path / "train_before.py")
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    a = T.train(dict(SMALL), tables, 0, tmp_path / "now", log_every=1)
    b = before.train(dict(SMALL), tables, 0, tmp_path / "before", log_every=1)
    assert _strip(a) == _strip(b) and "recipe" not in a
    assert _log(tmp_path / "now") == _log(tmp_path / "before")
    sa = torch.load(tmp_path / "now" / "ckpt.pt", weights_only=False)["state_dict"]
    sb = torch.load(tmp_path / "before" / "ckpt.pt", weights_only=False)["state_dict"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_default_values_written_out_equal_the_absent_keys(tables, tmp_path):
    """Writing today's constants explicitly trains the same weights; run.json gains `recipe`."""
    a = T.train(dict(SMALL), tables, 0, tmp_path / "absent")
    b = T.train({**SMALL, **T.RECIPE_DEFAULTS}, tables, 0, tmp_path / "explicit")
    assert b["recipe"] == {"lr": 3e-3, "weight_decay": 0.01, "grad_clip": 1.0, "warmup_steps": 1,
                           "total_steps": 4}
    sa = torch.load(tmp_path / "absent" / "ckpt.pt", weights_only=False)["state_dict"]
    sb = torch.load(tmp_path / "explicit" / "ckpt.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert a["last_train_loss"] == b["last_train_loss"]


def test_weight_decay_reaches_the_optimizer(tables, tmp_path):
    info = T.train({**SMALL, "weight_decay": 0.1}, tables, 0, tmp_path / "r")
    st = torch.load(tmp_path / "r" / T.STATE_NAME, weights_only=False)
    assert {g["weight_decay"] for g in st["optimizer"]["param_groups"]} == {0.1}
    assert info["recipe"]["weight_decay"] == 0.1


def test_grad_clip_is_the_threshold_used(tables, tmp_path, monkeypatch):
    seen = []
    orig = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        seen.append(max_norm)
        return orig(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    info = T.train({**SMALL, "grad_clip": 3.0}, tables, 0, tmp_path / "r")
    assert seen == [3.0] * 4 and info["recipe"]["grad_clip"] == 3.0
    seen.clear()
    T.train(dict(SMALL), tables, 0, tmp_path / "base")
    assert seen == [1.0] * 4


@pytest.mark.parametrize("keys,warm", [({"warmup_frac": 0.5}, 2),
                                       ({"warmup_frac": 1.0, "warmup_max": 3}, 3)])
def test_warmup_keys_set_the_warmup_step_count(tables, tmp_path, keys, warm):
    info = T.train({**SMALL, **keys}, tables, 0, tmp_path / "r", log_every=1)
    assert info["recipe"]["warmup_steps"] == warm
    lrs = [r["lr"] for r in _log(tmp_path / "r")]
    assert lrs[:warm] == pytest.approx([3e-3 * (s + 1) / warm for s in range(warm)])
    assert T.warmup_steps(4) == 1                       # the default on the same budget


def test_unknown_key_is_refused_and_comment_keys_pass(tables, tmp_path):
    with pytest.raises(KeyError, match="weight_decai"):
        T.train({**SMALL, "weight_decai": 0.1}, tables, 0, tmp_path / "typo")
    assert not (tmp_path / "typo" / "ckpt.pt").exists()
    T.check_config_keys({**SMALL, "_note": "a comment", "_stage_b_rule": "x"})


def test_code_keys_learns_keys_from_a_model_file(tmp_path):
    f = tmp_path / "model.py"
    f.write_text("class Model:\n    def __init__(self, cfg):\n"
                 "        self.h = cfg.get('my_new_width', 4)\n")
    cfg = {**SMALL, "model_file": str(f), "my_new_width": 8}
    T.check_config_keys(cfg)
    with pytest.raises(KeyError):
        T.check_config_keys({**SMALL, "my_new_width": 8})


def test_every_config_on_this_branch_passes_the_guard():
    known = T.code_keys()
    paths = sorted(CFG.rglob("*.json"))
    n = 0
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not (isinstance(d, dict) and "rung" in d):
            continue                                       # budgets.json is not a train config
        bad = [k for k in d if not k.startswith("_") and k not in known]
        assert not bad, f"{p.relative_to(_ROOT)}: {bad}"
        n += 1
    assert n >= 100
    assert set(T.RECIPE_DEFAULTS) <= known
