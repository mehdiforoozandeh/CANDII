"""miniCANDI ladder kit (chunk LK): the per-rung arm configs under tools/minicandi/configs/arms/
equal what `ladder_arms.py` generates, build through the registry within 10% of the parameter
budget, and differ from their base in exactly the declared keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import ladder_arms as LA  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi.components import NotApplicable  # noqa: E402

BUDGETS = json.loads((_ROOT / "tools/minicandi/configs/budgets.json").read_text())
ARMS_DIR = _ROOT / "tools/minicandi/configs/arms"
N_FILES = {1: 8, 2: 6, 3: 7, 4: 9, 5: 10, 6: 9}
ALL = [(r, a) for r in LA.RUNGS for a in LA.arm_names(r)]


def _disk(r: int, arm: str) -> dict:
    return json.loads((ARMS_DIR / f"rung{r}" / f"{arm}.json").read_text())


def test_files_on_disk_equal_the_generated_ones():
    assert LA.stale() == []
    for r, n in N_FILES.items():
        assert len(list((ARMS_DIR / f"rung{r}").glob("*.json"))) == n, r
    assert sum(N_FILES.values()) == 49


def test_budgets_gain_the_two_keys():
    assert BUDGETS["cuda_cpus"] == {"fir": 1, "nibi": 2}
    assert BUDGETS["score_time"] == {"1": "08:00:00", "2": "08:00:00", "3": "08:00:00",
                                     "4": "08:00:00", "5": "1-00:00:00", "6": "08:00:00"}


@pytest.mark.parametrize("r", LA.RUNGS)
def test_arms_base_is_the_rung_base_at_matched_params_windows(r):
    rung_base = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    assert rung_base["windows_seen"] == 200000          # the rung base file is left untouched
    base = _disk(r, "base")
    assert {k for k in base if base[k] != rung_base.get(k)} == {"windows_seen"}
    assert base["windows_seen"] == BUDGETS["windows_seen_matched_params"] == 80000
    mc = _disk(r, "matched_compute")
    assert mc["windows_seen"] == BUDGETS["windows_seen_at_matched_compute"][str(r)]


@pytest.mark.parametrize("r,arm", ALL)
def test_each_arm_differs_from_its_base_in_exactly_the_declared_keys(r, arm):
    base, cfg = _disk(r, "base"), _disk(r, arm)
    assert set(cfg) == set(base)
    changed = {k for k in cfg if cfg[k] != base[k]}
    assert changed == set(LA.arm_change(r, arm))
    if arm not in ("base", "matched_compute"):
        assert len(changed - {"simpler_than_base"}) == 1


@pytest.mark.parametrize("r,arm", ALL)
def test_each_arm_builds_within_10_percent_of_the_budget(r, arm):
    cfg = _disk(r, arm)
    torch.manual_seed(0)
    n = R.build_model(cfg).param_count()
    assert abs(n / float(cfg["param_budget"]) - 1) <= 0.10, (r, arm, n)


def test_simpler_than_base_per_arm():
    want_true = {"dna_off", "inject_concat", "skip_dna", "skip_counts", "axis_position_only",
                 "axis_track_only", "latents_small"}
    for r, arm in ALL:
        s = _disk(r, arm)["simpler_than_base"]
        if arm in want_true:
            assert s is True, (r, arm)
        elif arm in ("latents_large", "scrambled_covariates", "scramble_all") or \
                (arm == "inject_tokens" and r in (1, 3, 4)):
            assert s is False, (r, arm)
        else:
            assert s is None, (r, arm)


def test_the_table_of_arms():
    names = {r: set(LA.arm_names(r)) for r in LA.RUNGS}
    common = {"base", "dna_off", "scrambled_covariates", "inject_concat", "matched_compute"}
    assert names[1] == common | {"scramble_all", "inject_tokens", "mask_token"}
    assert names[2] == common | {"mask_drop"}
    assert names[3] == common | {"inject_tokens", "mask_drop"}
    assert names[4] == names[3] | {"skip_dna", "skip_counts"}
    assert names[5] == common | {"inject_film", "mask_token", "axis_position_only",
                                 "axis_track_only", "axis_tied"}
    assert names[6] == common | {"inject_film", "mask_token", "latents_small", "latents_large"}
    assert _disk(6, "latents_small")["n_latents"] == 32
    assert _disk(6, "latents_large")["n_latents"] == 512
    assert "scramble_all" not in set().union(*(names[r] for r in range(2, 7)))


def test_inject_tokens_on_rung_2_is_not_applicable():
    cfg = {**_disk(2, "base"), **LA.NOT_APPLICABLE[2]["inject_tokens"]}
    with pytest.raises(NotApplicable):
        R.build_model(cfg)
    assert not (ARMS_DIR / "rung2" / "inject_tokens.json").exists()
