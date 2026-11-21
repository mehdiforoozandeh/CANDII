"""Task t370, chunk C29: read-disjoint training pairs (`train_halves`, `train_halves_prob`) in
tools/minicandi/dataset.py, on synthetic tables (synth writes pr1.h5 / pr2.h5 beside counts.h5).

Requirements checked: (1) with the key absent or null the training and validation batches are
bit-identical to dataset.py at BASE_SHA; (2) under `pairs` the source of every correction and
same-view pair is the source view's pr1 file at depth scale 0.5 and its target the target view's
pr2 file at depth scale 0.5, the kind is the sampler's, and same-view pairs keep their bin mask;
(3) kind-1 targets still read `counts` at full depth; (4) `train_halves_prob` halves per pair;
(5) bad values raise; (6) train.check_config_keys accepts both keys."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402

L = 64
BASE_SHA = "345c0c7c"             # the tree before chunk C29
REF_CONFIGS = {"plain": {"context_bins": L, "batch_size": 4},
               "existing_keys": {"context_bins": L, "batch_size": 4, "qc_on_input": True,
                                 "two_step": True, "base_depth": "fixed",
                                 "average_input": True},
               "fixed_input": {"context_bins": L, "batch_size": 4, "input_depth": "fixed"}}
PAIRS = {"context_bins": L, "batch_size": 4, "train_halves": "pairs"}


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_halves"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def before(tmp_path_factory):
    src = subprocess.run(["git", "-C", str(_ROOT), "show", f"{BASE_SHA}:tools/minicandi/dataset.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip(f"git show {BASE_SHA} failed: {src.stderr.strip()}")
    p = tmp_path_factory.mktemp("before") / "dataset_before_c29.py"
    p.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location("dataset_before_c29", p)
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


def _train_digest(mod, tables, cfg, n=40) -> str:
    it = iter(mod.OnionDataset(tables, "train", cfg, seed=3))
    return _digest(next(it) for _ in range(n))


def _eval_digest(mod, tables, cfg) -> str:
    return _digest(mod.OnionDataset(tables, "val", dict(cfg, region_bins=4 * L), seed=3))


def _windows(tables, cfg, n=120, seed=0):
    ds = D.OnionDataset(tables, "train", cfg, seed)
    rng = np.random.default_rng([seed, 1])
    out = []
    for _ in range(n):
        plan = ds.plan(rng)
        out.append((plan, ds.materialize(plan, rng)))
    return ds, out


# --------------------------------------------------------------------------- default path


@pytest.mark.parametrize("name", sorted(REF_CONFIGS))
def test_default_is_bit_identical_to_the_tree_before(tables, before, name):
    cfg = REF_CONFIGS[name]
    old = _train_digest(before, tables, cfg)
    assert _train_digest(D, tables, cfg) == old
    assert _train_digest(D, tables, dict(cfg, train_halves=None)) == old
    assert _eval_digest(D, tables, cfg) == _eval_digest(before, tables, cfg)


def test_validation_batches_ignore_the_key(tables, before):
    cfg = REF_CONFIGS["plain"]
    assert _eval_digest(D, tables, dict(cfg, train_halves="pairs")) == \
        _eval_digest(before, tables, cfg)


# --------------------------------------------------------------------------- pairs


def test_pairs_read_pr1_to_pr2_at_half_depth_and_kind1_reads_counts(tables):
    ds, wins = _windows(tables, PAIRS)
    enc, seen = ds.encoder, {C.KIND_IMPUTE: 0, C.KIND_CORRECT: 0, C.KIND_SAME: 0}
    for plan, w in wins:
        meta = w["meta"]
        chrom, s0 = meta["chrom"], meta["start_bin"]
        x, y = w["x_counts"].numpy(), w["y_counts"].numpy()
        bm, kind = w["x_bin_mask"].numpy(), w["y_kind"].numpy()
        want_halved = sorted(a for a, _i, _o, k in plan["entries"]
                             if k in (C.KIND_SAME, C.KIND_CORRECT))
        assert meta["halves_tracks"] == want_halved
        for a, v_in, v_out, k in plan["entries"]:
            t = C.TRACK_INDEX[a]
            r_out = ds.rows[v_out]
            assert kind[t] == k, (a, v_in, v_out)
            seen[k] += 1
            if k == C.KIND_IMPUTE:
                assert np.array_equal(y[:, t], ds.read(v_out, chrom, s0, L, "counts"))
                assert np.allclose(w["c_out"][t].numpy(), enc.c_out(r_out))
                continue
            src = ds.read(v_in, chrom, s0, L, "pr1")
            assert np.array_equal(x[:, t], np.where(bm[:, t], 0, src))
            assert np.array_equal(y[:, t], ds.read(v_out, chrom, s0, L, "pr2"))
            assert np.allclose(w["c_in"][t].numpy(), enc.c_in(ds.rows[v_in], False, 0.5))
            assert np.allclose(w["c_out"][t].numpy(), enc.c_out(r_out, 0.5))
            if k == C.KIND_SAME:
                assert bm[:, t].any() and v_in == v_out      # today's bin-mask rule
            else:
                assert not bm[:, t].any() and v_in != v_out
    assert all(v > 0 for v in seen.values()), seen


def test_halves_share_no_reads_with_counts_by_construction(tables):
    """pr1 + pr2 = counts on synthetic rows, so the pair halves are read-disjoint."""
    ds = D.OnionDataset(tables, "train", PAIRS, 0)
    m = sorted(ds.allowed)[0]
    chrom = ds.chroms[0]
    a, b = ds.read(m, chrom, 0, L, "pr1"), ds.read(m, chrom, 0, L, "pr2")
    assert np.array_equal(a + b, ds.read(m, chrom, 0, L, "counts"))


def test_prob_halves_per_pair(tables):
    ds, wins = _windows(tables, dict(PAIRS, train_halves_prob=0.5), n=150)
    halved = unhalved = 0
    for plan, w in wins:
        pairs = {a for a, _i, _o, k in plan["entries"] if k in (C.KIND_SAME, C.KIND_CORRECT)}
        h = set(w["meta"]["halves_tracks"])
        assert h <= pairs
        halved += len(h)
        unhalved += len(pairs - h)
        y, kind = w["y_counts"].numpy(), w["y_kind"].numpy()
        chrom, s0 = w["meta"]["chrom"], w["meta"]["start_bin"]
        for a, v_in, v_out, k in plan["entries"]:
            t = C.TRACK_INDEX[a]
            assert kind[t] == k
            if a in pairs - h and k == C.KIND_CORRECT:       # an unhalved pair reads counts
                assert np.array_equal(y[:, t], ds.read(v_out, chrom, s0, L, "counts"))
    assert halved > 0 and unhalved > 0


def test_prob_zero_halves_nothing(tables):
    _, wins = _windows(tables, dict(PAIRS, train_halves_prob=0.0), n=40)
    assert all(w["meta"]["halves_tracks"] == [] for _, w in wins)


@pytest.mark.parametrize("bad", [{"train_halves": "all"}, {"train_halves": True},
                                 {"train_halves": "pairs", "train_halves_prob": 1.5},
                                 {"train_halves": "pairs", "train_halves_prob": -0.1}])
def test_bad_values_raise(tables, bad):
    with pytest.raises(ValueError):
        D.OnionDataset(tables, "train", dict(PAIRS, **bad), 0)


def test_check_config_keys_accepts_both_keys():
    T.check_config_keys({"train_halves": "pairs", "train_halves_prob": 0.5})
