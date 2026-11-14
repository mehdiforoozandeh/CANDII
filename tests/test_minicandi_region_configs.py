"""Configs of the region experiments (Amendment 7, chunk RBk) under
tools/minicandi/configs/region/: each file differs from its ladder source in the stated keys only,
every list path is under $O/region/lists/, and the file counts are 30 budget files, 8 + 12 region
files and 9 confirmation files (section B.4 as amended on 2026-11-13: seven selection arms run;
five older selection files stay on disk, marked not run in the README). One test loads a new
config through the trainer's config parser (no training)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import ladder_arms as LA  # noqa: E402

REGION = _ROOT / "tools/minicandi/configs/region"
ARMS_DIR = _ROOT / "tools/minicandi/configs/arms"
O = "/project/def-maxwl/mforooz/ONION"
LISTS = f"{O}/region/lists/"
EXCLUDE = f"{O}/eval/regions_v1/regions.json"
# arms with a list of their own and a confirmation file; the first four are not run (amended B.4)
NOT_RUN_SELECTION = ("peakfreq_w", "peakfreq_u", "ccre_like_w", "variance_w")
TILT = ("ccredens_e50_w", "ccredens_e25_w", "ccredens_e50_u", "variance_e50_w")
SELECTION = (*NOT_RUN_SELECTION, *TILT, "kcenter")
RUN_20K = ("tile_uniform", "terr_r1024_w20k", *TILT, "kcenter")
NOT_RUN_20K = (*NOT_RUN_SELECTION, "blocks_minus_eval_w20k")
NOT_RUN_MARK = "not run (near-uniform lists; see STRATA_REAL.md)"
MODEL_LOSS_KEYS = ("rung", "context_bins", "param_budget", "lr", "batch_size", "skip_source",
                   "covariate_injection", "mask_mode", "dna", "count_transform", "two_step")

# the stated change of every region file from rung 4 `base` at 80,000 windows
REGION80K = {
    **{f"terr_r{r}": {"train_regions": f"{LISTS}terr_r{r}_s{{seed}}.tsv"}
       for r in (64, 256, 1024, 4096)},
    "blocks_minus_eval": {"train_regions": f"{LISTS}blocks_minus_eval.tsv"},
    "eval_heavy": {"train_regions": f"{LISTS}eval_heavy.tsv"},
    "no_eval": {"exclude_regions": EXCLUDE},
    "today": {},
}
SELECTION20K = {
    "blocks_minus_eval_w20k": f"{LISTS}blocks_minus_eval.tsv",
    "terr_r1024_w20k": f"{LISTS}terr_r1024_s{{seed}}.tsv",
    "tile_uniform": f"{LISTS}tile_uniform.tsv",
    **{a: f"{LISTS}{a}.tsv" for a in SELECTION},
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _diff(a: dict, b: dict) -> set:
    return {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}


def _equal_budget_arms(rung: int) -> list[str]:
    return [a for a in LA.arm_names(rung) if a != "matched_compute"]


def _ladder(rung: int, arm: str) -> dict:
    """The arm's config as the ladder resolves it, two ways: generated and on disk."""
    gen = LA.arm_configs(rung)[arm]
    disk = _load(ARMS_DIR / f"rung{rung}" / f"{arm}.json")
    assert gen == disk, f"arms/rung{rung}/{arm}.json is stale against ladder_arms.py"
    return disk


def test_counts():
    def n(sub):
        return len(list((REGION / sub).glob("*.json")))
    assert len(_equal_budget_arms(1)) == 7 and len(_equal_budget_arms(4)) == 8
    assert (n("budget"), n("region80k"), n("selection20k"), n("confirm80k")) == (30, 8, 12, 9)
    assert sorted(p.name for p in REGION.iterdir() if p.is_file()) == ["README.md"]
    assert sorted(p.name for p in REGION.iterdir() if p.is_dir()) == \
        ["budget", "confirm80k", "region80k", "selection20k"]


@pytest.mark.parametrize("rung,arm,b", [(r, a, b) for r in (1, 4) for a in _equal_budget_arms(r)
                                        for b in (20000, 40000)])
def test_budget_file(rung, arm, b):
    cfg = _load(REGION / "budget" / f"rung{rung}_{arm}_w{b}.json")
    src = _ladder(rung, arm)
    assert _diff(cfg, src) == {"windows_seen"}
    assert cfg["windows_seen"] == b and src["windows_seen"] == 80000


def test_budget_files_are_exactly_the_expected_set():
    want = {f"rung{r}_{a}_w{b}.json" for r in (1, 4) for a in _equal_budget_arms(r)
            for b in (20000, 40000)}
    assert {p.name for p in (REGION / "budget").glob("*.json")} == want


@pytest.mark.parametrize("arm", sorted(REGION80K))
def test_region80k(arm):
    cfg = _load(REGION / "region80k" / f"{arm}.json")
    src = _ladder(4, "base")
    assert _diff(cfg, src) == set(REGION80K[arm])
    for k, v in REGION80K[arm].items():
        assert cfg[k] == v
    assert cfg["windows_seen"] == 80000
    assert not ("train_regions" in cfg and "exclude_regions" in cfg)


@pytest.mark.parametrize("arm", sorted(SELECTION20K))
def test_selection20k(arm):
    cfg = _load(REGION / "selection20k" / f"{arm}.json")
    src = _load(REGION / "budget" / "rung4_base_w20000.json")
    assert _diff(cfg, src) == {"train_regions"}
    assert cfg["train_regions"] == SELECTION20K[arm]
    assert _diff(cfg, _ladder(4, "base")) == {"windows_seen", "train_regions"}
    assert cfg["windows_seen"] == 20000


@pytest.mark.parametrize("arm", SELECTION)
def test_confirmation(arm):
    cfg = _load(REGION / "confirm80k" / f"{arm}_w80k.json")
    assert _diff(cfg, _ladder(4, "base")) == {"train_regions"}
    assert cfg["train_regions"] == _load(REGION / "selection20k" / f"{arm}.json")["train_regions"]
    assert cfg["windows_seen"] == 80000


def test_file_sets_are_exact():
    assert {p.stem for p in (REGION / "region80k").glob("*.json")} == set(REGION80K)
    assert {p.stem for p in (REGION / "selection20k").glob("*.json")} == set(SELECTION20K)
    assert {p.stem for p in (REGION / "confirm80k").glob("*.json")} == \
        {f"{a}_w80k" for a in SELECTION}


def test_paths_and_invariants_over_every_file():
    files = sorted(REGION.glob("*/*.json"))
    assert len(files) == 59
    base4 = _ladder(4, "base")
    for p in files:
        cfg = _load(p)
        if "train_regions" in cfg:
            tr = cfg["train_regions"]
            assert tr.startswith(LISTS) and tr.endswith(".tsv"), p
            assert "/" not in tr[len(LISTS):], p
        if "exclude_regions" in cfg:
            assert cfg["exclude_regions"] == EXCLUDE, p
        src = base4 if p.parent.name != "budget" else \
            _ladder(int(p.name[4]), p.stem[6:].rsplit("_w", 1)[0])
        for k in MODEL_LOSS_KEYS:
            assert cfg.get(k, "<absent>") == src.get(k, "<absent>"), (p, k)
        assert p.read_text(encoding="utf-8") == LA.render(cfg), f"{p} is not in render() form"


def test_amended_selection_arms():
    assert set(RUN_20K) | set(NOT_RUN_20K) == set(SELECTION20K)
    assert not set(RUN_20K) & set(NOT_RUN_20K) and len(RUN_20K) == 7
    for a in TILT:
        assert SELECTION20K[a] == f"{LISTS}{a}.tsv"
    readme = (REGION / "README.md").read_text(encoding="utf-8")
    rows = {line.split("|")[1].strip().strip("`"): line for line in readme.splitlines()
            if line.startswith("| `")}
    for a in NOT_RUN_20K:
        assert NOT_RUN_MARK in rows[a], a
    for a in RUN_20K:
        assert "not run" not in rows[a], a


def test_new_config_loads_through_the_trainer_parser():
    from minicandi import train as T
    cfg = _load(REGION / "selection20k" / "ccredens_e50_w.json")
    full = T.full_config(cfg)
    assert full["train_regions"] == f"{LISTS}ccredens_e50_w.tsv"
    assert full["windows_seen"] == 20000 and full["rung"] == 4
    assert {k: full[k] for k in cfg} == cfg
