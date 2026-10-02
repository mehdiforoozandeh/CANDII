"""t118 N1 — the row-2 no-covariates averaged map made cheap, with exact fixes only.

F3  `model.theta_levels(average=True)` applies g's last (linear) layer after the mean over pairs:
    within 1e-5 of the mean of the full g.
F4  `train.validate` reuses each source's chr22 distinct values and inverse index across calls
    (`level_cache`): bit-identical.
F1  the nocov (loc, disp) is computed once per (source pid, chromosome), in validation and in
    `score.py` trained / law: bit-identical, records in their own order.
Row 1, and the row-2 real / ids / xshuf paths, never reach the new code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import data, model, pairs, score, synth, train  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
FAST = {"window": 256, "batch": 16, "knot_sample": 20_000, "knot_block": 256}
CPU = torch.device("cpu")
STATS = {"knots_x": np.linspace(-6.9, 5.5, 12), "n0": 5.0, "sigma0": 0.5}
REL = 1e-5


def _old_theta_levels(ladder, x_values, cov, average=False):
    """The pre-N1 `theta_levels`, verbatim: the mean over pairs of the full g."""
    if not ladder.per_bin:
        raise ValueError(f"rung {ladder.rung}: theta_levels needs a per-bin form")
    cov = cov.reshape(-1, cov.shape[-1])
    k = int(cov.shape[0])
    if not average and k != 1:
        raise ValueError(f"theta_levels: {k} covariate rows without average=True")
    x_values = x_values.reshape(-1)
    step = max(1, model.LEVEL_ROWS // k)
    out = []
    for u0 in range(0, max(1, x_values.shape[0]), step):
        t = ladder.g.levels(cov, x_values[u0:u0 + step])
        out.append(t.mean(0) if average else t[0])
    return torch.cat(out, 0)


def _old_run_jobs(jobs, workers):
    """The pre-N1 `score._run_jobs`: one prediction per job."""
    if workers <= 1 or len(jobs) <= 1:
        return [score.score_job(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, len(jobs)),
                                     initializer=score._worker_init) as pool:
        return pool.map(score.score_job, jobs, chunksize=1)


def _trained_like(lad, seed=0):
    """Random non-zero weights everywhere, so theta varies with x and with the covariates."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in lad.g.parameters():
            p.add_(0.1 * torch.randn_like(p))
        for p in lad.form.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return lad.eval()


def _levels(n, seed=1):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.exponential(0.5, n), rng.exponential(20, n // 10)])
    return torch.from_numpy(np.unique(np.log(np.maximum(p, 1e-3)).astype(np.float32))[:n])


#: fields F3's float-order change may move by more than REL relative, with the absolute bound
#: allowed: Spearman is a rank statistic (a 1e-7 change in theta reorders near-tied predictions;
#: the synthetic top-1 % subsets are a few dozen bins), and the swap ratio and the scale error are
#: near-zero differences whose relative error is amplified.
LOOSE_ABS = {"spearman_all": 0.02, "spearman_nonzero": 0.02, "spearman_top1": 0.02,
             "swap_median_abs_log_ratio": 1e-6, "scale_error_all": 1e-6}


def _rel_close(a, b, path="", key=None) -> list:
    """Paths where a and b differ: floats beyond REL relative (LOOSE_ABS absolute for the fields
    named there), anything else not equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in _rel_close(a[k], b[k], f"{path}.{k}", k)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in _rel_close(x, y, f"{path}[{i}]", key)]
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        if a == b or abs(a - b) <= REL * max(abs(a), abs(b)):
            return []
        if key in LOOSE_ABS and abs(a - b) <= LOOSE_ABS[key]:
            return []
        return [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    root = tmp_path_factory.mktemp("nocovfast")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\nchr21\t5000\t7500\nchr22\t0\t250\n")
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov,
            "bed": bed, "rows": pairs.read_manifest(manifest)}


def _train(kit, rung, g, space, model_name, sub="runs"):
    return train.run_training(kit["manifest"], kit["cov"], kit["products"], kit["root"] / sub,
                              rung, g, space, model_name, 0, max_steps=5, eval_every=5,
                              device="cpu", lr=1e-2, **FAST)


# -- F3 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rung", ["A2", "D2"])
@pytest.mark.parametrize("k", [2, 17, 242])
def test_f3_theta_bar_matches_the_mean_of_the_full_g(rung, k):
    lad = _trained_like(model.Ladder(rung, "pval", STATS), seed=k)
    torch.manual_seed(100 + k)
    cov = torch.randn(k, 40)
    x = _levels(3000)
    with torch.no_grad():
        want = _old_theta_levels(lad, x, cov, average=True)
        got = model.theta_levels(lad, x, cov, average=True)
    assert got.shape == want.shape == (x.shape[0], lad.n_theta)
    assert float(want.abs().max()) > 0.1 and float(want.std(0).max()) > 1e-3   # not constant
    assert float((got - want).abs().max()) < 1e-5
    old = model.LEVEL_ROWS
    try:                                        # the level blocks change nothing
        model.LEVEL_ROWS = 999
        with torch.no_grad():
            blocked = model.theta_levels(lad, x, cov, average=True)
    finally:
        model.LEVEL_ROWS = old
    assert float((blocked - want).abs().max()) < 1e-5


@pytest.mark.parametrize("rung", ["A2", "D2"])
def test_f3_leaves_the_one_pair_path_bit_identical(rung):
    lad = _trained_like(model.Ladder(rung, "counts", STATS), seed=3)
    cov = torch.randn(1, 40, generator=torch.Generator().manual_seed(4))
    x = _levels(2000, seed=5)
    with torch.no_grad():
        assert torch.equal(model.theta_levels(lad, x, cov), _old_theta_levels(lad, x, cov))


def test_f3_is_exact_only_because_nothing_follows_the_last_linear_layer():
    lad = model.Ladder("A2", "pval", STATS)
    layers = list(lad.g.net)
    assert isinstance(layers[-1], torch.nn.Linear) and isinstance(layers[-2], torch.nn.ReLU)
    lad.g.net.append(torch.nn.ReLU())           # a nonlinearity after it: F3 must refuse
    with pytest.raises(TypeError):
        model.theta_levels(lad, _levels(10), torch.randn(3, 40), average=True)


# -- F4 and F1 in validation ------------------------------------------------------------------------

def _per_pair_validate(lad, corpus, tp, cov, space):
    """The pre-N1 per-pair loop of `train.validate` for the nocov per-bin twin."""
    vals = []
    with torch.no_grad():
        for p in tp:
            X = corpus.get(p["source_pid"], space, "chr22")
            Y = corpus.get(p["target_pid"], space, "chr22")
            loc, disp = model.predict_chrom_bins(lad, X, cov, CPU, average=True)
            y = torch.from_numpy(np.array(Y, dtype=np.float32))
            vals.append(float(lad.nll(torch.from_numpy(loc), torch.from_numpy(disp), y)))
    return float(np.mean(vals))


@pytest.mark.parametrize("rung,space", [("A2", "pval"), ("D2", "counts")])
def test_validate_equals_the_per_pair_loop_and_reuses_the_levels(kit, monkeypatch, rung, space):
    corpus = data.Corpus(kit["products"])
    tp = pairs.train_pairs(kit["rows"], "all")
    srcs = {p["source_pid"] for p in tp}
    assert len(srcs) < len(tp)                  # several pairs share a source
    lad = _trained_like(model.Ladder(rung, space, STATS), seed=7)
    cov = torch.randn(len(tp), 40, generator=torch.Generator().manual_seed(8))
    want = _per_pair_validate(lad, corpus, tp, cov, space)
    assert train.validate(lad, corpus, tp, cov, space, CPU, True) == want
    cache: dict = {}
    assert train.validate(lad, corpus, tp, cov, space, CPU, True, level_cache=cache) == want
    assert set(cache) == {(s, "chr22") for s in srcs}
    calls = {"unique": 0, "predict": 0}
    real_unique, real_pcb = np.unique, model.predict_chrom_bins

    def counting_unique(*a, **k):
        calls["unique"] += 1
        return real_unique(*a, **k)

    def counting_pcb(*a, **k):
        calls["predict"] += 1
        return real_pcb(*a, **k)

    monkeypatch.setattr(model.np, "unique", counting_unique)
    monkeypatch.setattr(model, "predict_chrom_bins", counting_pcb)
    assert train.validate(lad, corpus, tp, cov, space, CPU, True, level_cache=cache) == want
    assert calls == {"unique": 0, "predict": len(srcs)}     # F4: no unique; F1: once per source


def test_level_cache_is_bit_identical_and_checks_its_x(kit):
    corpus = data.Corpus(kit["products"])
    lad = _trained_like(model.Ladder("D2", "pval", STATS), seed=9)
    cov = torch.randn(5, 40, generator=torch.Generator().manual_seed(10))
    X = corpus.get("T1__base__base", "pval", "chr22")
    want = model.predict_chrom_bins(lad, X, cov, CPU, average=True, chunk=128)
    lc: dict = {}
    for _ in range(2):                          # fill, then reuse
        got = model.predict_chrom_bins(lad, X, cov, CPU, average=True, chunk=128, level_cache=lc)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    assert lc["inv"].dtype == np.int32
    with pytest.raises(ValueError):             # a different padded length is refused
        model.predict_chrom_bins(lad, X[:-300], cov, CPU, average=True, chunk=128, level_cache=lc)


# -- F1 in scoring ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nocov_run(kit):
    return _train(kit, "A2", "all", "pval", "nocov")


def _score_docs(kit, run_dir, workers, monkeypatch, old_levels=False, old_jobs=False):
    with monkeypatch.context() as m:
        if old_levels:
            m.setattr(model, "theta_levels", _old_theta_levels)
        if old_jobs:
            m.setattr(score, "_run_jobs", _old_run_jobs)
        pred = model.load_run(run_dir, kit["products"], kit["cov"], kit["manifest"], device="cpu")
        s = json.loads(score.score_trained(pred, kit["rows"], kit["bed"], run_dir,
                                           workers=workers).read_text())
        law = json.loads(score.score_law(pred, kit["rows"], kit["bed"], run_dir,
                                         workers=workers).read_text())
    return s, law


def _order(recs):
    return [(r["kind"], r["eval"], r["source_pid"], r["target_pid"], r["cov_src_pid"],
             r["cov_tgt_pid"]) for r in recs]


@pytest.mark.parametrize("workers", [1, 2])
def test_nocov_scoring_by_source_is_bit_equal_without_f3(kit, nocov_run, monkeypatch, workers):
    old_s, old_law = _score_docs(kit, nocov_run, workers, monkeypatch, old_levels=True,
                                 old_jobs=True)
    new_s, new_law = _score_docs(kit, nocov_run, workers, monkeypatch, old_levels=True)
    assert len(old_s["records"]) > 20 and len(old_law["records"]) > 10
    assert _order(new_s["records"]) == _order(old_s["records"])
    assert new_s["records"] == old_s["records"]
    assert new_s["skipped"] == old_s["skipped"] and new_s["snippets"] == old_s["snippets"]
    assert new_law["records"] == old_law["records"]


def test_nocov_scoring_with_f3_within_1e5_relative(kit, nocov_run, monkeypatch):
    old_s, old_law = _score_docs(kit, nocov_run, 1, monkeypatch, old_levels=True, old_jobs=True)
    new_s, new_law = _score_docs(kit, nocov_run, 1, monkeypatch)
    assert _order(new_s["records"]) == _order(old_s["records"])
    assert _order(new_law["records"]) == _order(old_law["records"])
    assert _rel_close(new_s["records"], old_s["records"]) == []
    assert _rel_close(new_law["records"], old_law["records"]) == []
    for r_new, r_old in zip(new_s["records"] + new_law["records"],
                            old_s["records"] + old_law["records"]):
        for k in r_old:                         # every CRPS field within REL, no exemption
            if k.startswith("crps") and r_old[k] is not None:
                assert abs(r_new[k] - r_old[k]) <= REL * abs(r_old[k]), k


def test_nocov_scoring_predicts_once_per_source_and_chromosome(kit, nocov_run, monkeypatch):
    seen = []
    real = model.Predictor.predict

    def spy(self, x_src_pid, cov_src_pid, cov_tgt_pid, chrom):
        seen.append((x_src_pid, chrom))
        return real(self, x_src_pid, cov_src_pid, cov_tgt_pid, chrom)

    monkeypatch.setattr(model.Predictor, "predict", spy)
    pred = model.load_run(nocov_run, kit["products"], kit["cov"], kit["manifest"], device="cpu")
    doc = json.loads(score.score_trained(pred, kit["rows"], kit["bed"], nocov_run,
                                         workers=1).read_text())
    assert len(seen) == len(set(seen))
    n_trained = sum(r["kind"] == "trained" for r in doc["records"])
    assert len(seen) < n_trained                # one per job would be at least this many
    seen.clear()
    law = json.loads(score.score_law(pred, kit["rows"], kit["bed"], nocov_run,
                                     workers=1).read_text())
    assert len(seen) == len(set(seen)) < 2 * len(law["records"])


# -- every other path is untouched ------------------------------------------------------------------

@pytest.mark.parametrize("rung,model_name", [("A2", "real"), ("A2", "ids"), ("A2", "xshuf"),
                                             ("A", "nocov"), ("A", "real"), ("A2", "nocov")])
def test_only_the_row2_nocov_path_uses_the_new_code(kit, monkeypatch, rung, model_name):
    calls = {"level_cache": [], "average": []}
    real_pcb, real_tl = model.predict_chrom_bins, model.theta_levels

    def spy_pcb(*a, **k):
        calls["level_cache"].append(k.get("level_cache") is not None)
        return real_pcb(*a, **k)

    def spy_tl(ladder, x_values, cov, average=False):
        calls["average"].append(bool(average))
        return real_tl(ladder, x_values, cov, average)

    monkeypatch.setattr(model, "predict_chrom_bins", spy_pcb)
    monkeypatch.setattr(model, "theta_levels", spy_tl)
    rd = _train(kit, rung, "T1", "counts", model_name, sub=f"paths_{rung}_{model_name}")
    pred = model.load_run(rd, kit["products"], kit["cov"], kit["manifest"], device="cpu")
    score._setup(pred, kit["rows"], kit["bed"])
    nocov2 = rung == "A2" and model_name == "nocov"
    assert score._CTX["by_source"] is nocov2
    if nocov2:
        assert calls["level_cache"] and all(calls["level_cache"]) and all(calls["average"])
        return
    assert not any(calls["level_cache"]) and not any(calls["average"])
    if rung == "A2" and model_name == "real":   # scoring never groups by source
        def boom(*a, **k):
            raise AssertionError("_run_by_source called for a non-nocov run")
        monkeypatch.setattr(score, "_run_by_source", boom)
        score.score_trained(pred, kit["rows"], kit["bed"], rd, workers=1)
        score.score_law(pred, kit["rows"], kit["bed"], rd, workers=1)
        assert not any(calls["average"])
