"""Chunk CF: the train-mode-only `train_views` key, strict `same_view_prob = 1.0`, and the
counterfactual-data ablation configs under configs/views/rung4/ (synthetic tables)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402

L = 64
BASE_SHA = "cc83ee4"              # the tree before chunk CF
CFG = _ROOT / "tools/minicandi/configs"
NEW = {"views_base_only": {"train_views": ["base"]},
       "views_all_same_view_only": {"same_view_prob": 1.0},
       "views_base_depth": {"train_views": ["base", "depth"]},
       "views_base_fastq": {"train_views": ["base", "se", "crop"]},
       "views_base_alignment": {"train_views": ["base", "nodedup", "mapq"]}}
REF_CONFIGS = {"plain": {"context_bins": L, "batch_size": 4},
               "existing_keys": {"context_bins": L, "batch_size": 4, "qc_on_input": True,
                                 "two_step": True, "base_depth": "fixed",
                                 "average_input": True},
               "panel": {"context_bins": L, "batch_size": 4, "mask_scheme": "panel"},
               "same_0.9": {"context_bins": L, "batch_size": 4, "same_view_prob": 0.9}}


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_cf"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def before(tmp_path_factory):
    """dataset.py as it was at BASE_SHA, loaded as its own module."""
    src = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_SHA}:tools/minicandi/dataset.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip(f"git show {BASE_SHA} failed: {src.stderr.strip()}")
    p = tmp_path_factory.mktemp("before") / "dataset_before.py"
    p.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("dataset_before", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _digest(batches) -> str:
    h = hashlib.sha256()
    for b in batches:
        for k in sorted(b):
            h.update(k.encode())
            if k == "meta":
                h.update(json.dumps(b[k], sort_keys=True).encode())
            else:
                t = b[k].contiguous()
                h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
                h.update(t.numpy().tobytes())
    return h.hexdigest()


def _train_digest(mod, tables, cfg, n=50) -> str:
    it = iter(mod.OnionDataset(tables, "train", cfg, seed=3))
    return _digest(next(it) for _ in range(n))


def _eval_digest(mod, tables, cfg, mode="val") -> str:
    return _digest(mod.OnionDataset(tables, mode, dict(cfg, region_bins=4 * L), seed=3))


def _plans(tables, cfg, n=200, seed=0) -> list:
    ds = D.OnionDataset(tables, "train", cfg, seed)
    rng = np.random.default_rng([seed, 1])
    return ds, [ds.plan(rng) for _ in range(n)]


def _kind(ds, mid) -> str:
    return D.view_of(ds.rows[mid])[0]


# ------------------------------------------------------------------------------- defaults


@pytest.mark.parametrize("name", sorted(REF_CONFIGS))
def test_default_is_bit_identical_to_the_tree_before(tables, before, name):
    cfg = REF_CONFIGS[name]
    old = _train_digest(before, tables, cfg)
    assert _train_digest(D, tables, cfg) == old
    assert _train_digest(D, tables, dict(cfg, train_views=None)) == old
    assert _eval_digest(D, tables, cfg) == _eval_digest(before, tables, cfg)


# ------------------------------------------------------------------------------- train_views


def test_base_only_reads_no_other_view_in_200_training_windows(tables):
    ds, plans = _plans(tables, {"context_bins": L}, seed=0)
    assert any(_kind(ds, m) != "base" for p in plans for e in p["entries"]
               for m in e[1:3] if m)                    # the default does read other views
    cfg = {"context_bins": L, "batch_size": 8, "train_views": ["base"]}
    it = iter(D.OnionDataset(tables, "train", cfg, seed=0))
    n, kinds = 0, set()
    for _ in range(25):
        for m in next(it)["meta"]:
            n += 1
            for mid in m["inputs"] + m["targets"]:
                kinds.add(_kind(ds, mid))
    assert n == 200 and kinds == {"base"}


@pytest.mark.parametrize("views", [["base", "depth"], ["base", "se"], ["base", "mapq"],
                                   ["se"]])
def test_only_listed_kinds_and_a_track_without_one_is_missing(tables, views):
    ds, plans = _plans(tables, {"context_bins": L, "train_views": views}, seed=1)
    seen = {_kind(ds, m) for p in plans for e in p["entries"] for m in e[1:3] if m}
    assert seen <= set(views) and seen
    allowed = {k for k, v in ds.train_tracks.items()
               if any(_kind(ds, m) in views for m in v)}
    for p in plans:                                     # no track without a listed view appears
        assert {(p["cell"], e[0]) for e in p["entries"]} <= allowed
        assert not ds.window_violations([e[1] for e in p["entries"] if e[1]],
                                        [e[2] for e in p["entries"]])


def test_bad_train_views_raise(tables):
    for bad in ([], "base", ["fastq"], ["base", "depth0.50"]):
        with pytest.raises(ValueError):
            D.OnionDataset(tables, "train", {"context_bins": L, "train_views": bad})


def test_train_views_leaves_average_and_train_cells_alone(tables):
    a = D.OnionDataset(tables, "train", {"context_bins": L})
    b = D.OnionDataset(tables, "train", {"context_bins": L, "train_views": ["se"]})
    assert a.train_tracks == b.train_tracks and a.train_base == b.train_base
    assert a.train_cells == b.train_cells
    assert np.array_equal(a.average("s0", "chr1", 0, L), b.average("s0", "chr1", 0, L))


@pytest.mark.parametrize("mode", ["val", "blind"])
def test_validation_and_scoring_batches_ignore_the_key(tables, mode):
    cfg = {"context_bins": L, "batch_size": 4, "allow_blind": True}
    ref = _eval_digest(D, tables, cfg, mode)
    for change in ({"train_views": ["base"]}, {"train_views": ["base", "depth"]}):
        assert _eval_digest(D, tables, dict(cfg, **change), mode) == ref


# ------------------------------------------------------------------------------- same view only


def _cross(plans) -> int:
    return sum(e[3] == C.KIND_CORRECT for p in plans for e in p["entries"])


def test_same_view_prob_one_draws_no_cross_view_pair(tables, before):
    cfg = {"context_bins": L, "same_view_prob": 1.0}
    old = before.OnionDataset(tables, "train", cfg, 0)
    rng = np.random.default_rng([0, 1])
    old_cross = _cross([old.plan(rng) for _ in range(2000)])
    ds, plans = _plans(tables, cfg, n=2000, seed=0)
    assert _cross(plans) == 0
    assert sum(e[3] == C.KIND_SAME for p in plans for e in p["entries"]) > 0
    for p in plans:
        assert all(e[1] == e[2] for e in p["entries"] if e[1] is not None)
    it = iter(D.OnionDataset(tables, "train", dict(cfg, batch_size=8), seed=0))
    for _ in range(25):                                  # 200 windows: no kind 2 in y_kind
        assert not (next(it)["y_kind"] == C.KIND_CORRECT).any()
    print(f"\ncross-view entries at same_view_prob 1.0 in 2000 windows: before {old_cross}, "
          f"now 0")


# ------------------------------------------------------------------------------- configs


@pytest.mark.parametrize("name", sorted(NEW))
def test_config_is_base_plus_one_change_and_builds(name):
    base = json.loads((CFG / "arms/rung4/base.json").read_text())
    cfg = json.loads((CFG / f"views/rung4/{name}.json").read_text())
    assert cfg == dict(base, **NEW[name])
    T.check_config_keys(cfg)
    torch.manual_seed(0)
    a = R.build_model(T.full_config(base)).state_dict()
    torch.manual_seed(0)
    m = R.build_model(T.full_config(cfg))
    b = m.state_dict()
    assert m.param_count() == R.build_model(T.full_config(base)).param_count()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
