"""t118-C6a — the ladder end to end: the smoke driver, and the seams between the chunks.

  * the smoke (`tools/t118/ladder/smoke.py`) for rung A, 10 steps, 1 seed, through every CLI of the
    chain on synthetic products, in under 90 s; its results.json passes `figures.check_schema`;
  * rung C with a one-hot (identity) kernel gives exactly rung B's (loc, disp) for the same curve
    theta (C carries its own copy of B's curve);
  * the SLURM DRY_RUN command lines of task 54 (train, score trained, law) and of the cache and agg
    scripts are argument lists the real CLIs parse and dispatch as intended;
  * the cache path: `train.py cache` over every synthetic product, then a short train that reads
    the cache dir, which matches the same train read from the npz products.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "tools" / "t118" / "ladder"
SLURM = REPO / "slurm" / "t118"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
BLACKLIST = "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"
sys.path.insert(0, str(REPO / "tools" / "t118"))
from ladder import aggregate, base, data, figures, pairs, score, synth, train  # noqa: E402
from ladder.fforms import form_b, form_c  # noqa: E402


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO / "src")
    return e


# ---- the smoke -------------------------------------------------------------------------------

def test_smoke_rung_a_end_to_end(tmp_path):
    work = tmp_path / "smoke_A"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(LADDER / "smoke.py"), str(work), "A", "--steps", "10",
                        "--seeds", "1"], env=_env(), capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.strip().splitlines()[-1] == "SMOKE OK A"
    assert wall < 90, f"smoke took {wall:.1f} s"
    res = figures.load_results(work / "agg")
    assert figures.check_schema(res) == []
    runs = {f"A_{g}_{s}_{m}_s0" for g in ("T1", "all") for s in pairs.SPACES
            for m in pairs.MODELS}
    assert runs <= set(res["runs_present"])
    for f in ("ckpt.pt", "scores.json", "figdata.npz", "law.json"):
        assert (work / "runs" / "A_T1_counts_real_s0" / f).is_file()
    for f in ("report.md", "checks_A.json"):
        assert (work / "agg" / "A" / f).is_file()
    kinds = {r["kind"] for r in res["per_pair"]}
    assert {"trained", "swap"} <= kinds
    # every run's records reached the aggregate: trained records of each run in per_pair
    assert {r["run_name"] for r in res["per_pair"] if r["kind"] == "trained"} == runs


# ---- rung C with an identity kernel is rung B ---------------------------------------------------

@pytest.mark.parametrize("space", pairs.SPACES)
def test_rung_c_with_one_hot_kernel_is_exactly_rung_b(space):
    rng = np.random.default_rng(3)
    xs = np.log1p(rng.poisson(rng.gamma(0.6, 3.0, 200_000))).astype(np.float64)
    if space == "pval":
        xs = np.log(np.maximum(rng.gamma(0.5, 1.0, 200_000), 1e-3))
    stats = {"knots_x": base.knot_stats(xs), "n0": 5.0, "sigma0": 0.5}
    fb, fc = form_b.build(space, stats), form_c.build(space, stats)
    assert fc.n_theta == fb.n_theta + 33 and fc.context == 16 and fb.context == 0
    g = torch.Generator().manual_seed(0)
    n, L = 6, 3000
    theta_b = fb.init_theta() + 0.7 * torch.randn(n, fb.n_theta, generator=g)
    kernel = torch.zeros(n, 33)
    kernel[:, 16] = 1.0
    theta_c = torch.cat([kernel, theta_b], dim=1)
    lo, hi = float(stats["knots_x"][0]), float(stats["knots_x"][-1])
    # x spans below, inside and above the knot range (both extrapolated end segments)
    x = lo - 1.0 + (hi - lo + 2.0) * torch.rand(n, L + 32, generator=g)
    loc_b, disp_b = fb(x[:, 16:-16], theta_b)
    loc_c, disp_c = fc(x, theta_c)
    assert torch.equal(loc_b, loc_c)
    assert torch.equal(disp_b, disp_c)
    # the identity theta of each form is the same map too
    lb0, _ = fb(x[:, 16:-16], fb.init_theta().expand(n, -1))
    lc0, _ = fc(x, fc.init_theta().expand(n, -1))
    assert torch.allclose(lb0, lc0, atol=1e-5)
    d_b, d_c = fb.describe(theta_b[0]), fc.describe(theta_c[0])
    assert np.allclose(d_b["curve_y"], d_c["curve_y"], atol=1e-6)
    assert np.allclose(d_b["disp"], d_c["disp"], atol=1e-6)


# ---- the SLURM command lines are what the CLIs parse --------------------------------------------

def _dry(script: str, args: list[str], index=54) -> list[list[str]]:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"))
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    r = subprocess.run(["bash", str(SLURM / f"ladder_{script}.sh"), *args], env=e,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return [shlex.split(line) for line in r.stdout.splitlines()]


def _script_argv(cmd: list[str], kit: str, tool: str) -> list[str]:
    """Check `python3 <kit>/tools/t118/ladder/<tool>` and that the tool exists in this checkout;
    return the argument list after the script."""
    assert cmd[0] == "python3"
    assert cmd[1] == f"{kit}/tools/t118/ladder/{tool}"
    assert (LADDER / tool).is_file()
    return cmd[2:]


@pytest.fixture
def fixture_rows(monkeypatch):
    """`pairs.read_manifest` returns the md5-pinned t112 fixture for any path, and records it."""
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    seen = []

    def fake(path):
        seen.append(str(path))
        return rows
    monkeypatch.setattr(pairs, "read_manifest", fake)
    return seen


def test_dry_run_train_lines_parse_as_train_index_then_score_trained(monkeypatch, fixture_rows):
    cmds = _dry("train", ["/kit", "/products", "/cache", "/out"])
    assert len(cmds) == 2
    got = {}

    def fake_train(manifest, covariates, data_dir, runs_dir, rung, g, space, model_name, seed,
                   max_steps=None, eval_every=None, device="auto", **kw):
        got["train"] = dict(manifest=str(manifest), covariates=str(covariates),
                            data_dir=str(data_dir), runs_dir=str(runs_dir), rung=rung, g=g,
                            space=space, model=model_name, seed=seed, max_steps=max_steps,
                            device=device)
        return Path(runs_dir) / train.run_name(rung, g, space, model_name, seed)
    monkeypatch.setattr(train, "run_training", fake_train)
    assert train.main(_script_argv(cmds[0], "/kit", "train.py")) == 0
    assert got["train"] == dict(manifest="/products/MANIFEST.tsv",
                                covariates="/kit/tools/t118/covariates.tsv", data_dir="/cache",
                                runs_dir="/out/runs", rung="A", g="C19M16", space="counts",
                                model="real", seed=0, max_steps=None, device="auto")
    run_dir = Path(got["train"]["runs_dir"]) / "A_C19M16_counts_real_s0"
    _check_score_line(monkeypatch, cmds[1], "trained", run_dir, BLACKLIST, 4)


def test_dry_run_law_line_parses_as_score_law(monkeypatch, fixture_rows):
    cmds = _dry("law", ["/kit", "/products", "/cache", "/blacklist.bed", "/out"])
    assert len(cmds) == 1
    _check_score_line(monkeypatch, cmds[0], "law", Path("/out/runs/A_C19M16_counts_real_s0"),
                      "/blacklist.bed", 8)


def _check_score_line(monkeypatch, cmd, verb, run_dir, blacklist, workers):
    from ladder import model
    argv = _script_argv(cmd, "/kit", "score.py")
    assert argv[0] == verb
    got = {}

    class FakePred:
        pass

    def fake_load(rd, data_dir, cov, man, device="auto"):
        got["load"] = (str(rd), str(data_dir), str(cov), str(man), device)
        return FakePred()

    def fake_score(pred, rows, bl, rd, workers=1):
        got["score"] = (str(bl), str(rd), workers, len(rows))
        return Path(rd) / "x.json"
    monkeypatch.setattr(model, "load_run", fake_load)
    monkeypatch.setattr(score, "score_trained" if verb == "trained" else "score_law", fake_score)
    assert score.main(argv[:]) == 0
    assert got["load"] == (str(run_dir), "/cache", "/kit/tools/t118/covariates.tsv",
                           "/products/MANIFEST.tsv", "cpu")
    assert got["score"] == (blacklist, str(run_dir), workers, 130)


def test_dry_run_cache_line_parses_and_indexes_the_sorted_products(monkeypatch, fixture_rows):
    cmds = _dry("cache", ["/kit", "/products", "/cache"], 7)
    got = {}

    def fake_build(products_dir, cache_dir, pid, row):
        got["build"] = (str(products_dir), str(cache_dir), pid, row["pid"])
        return {"counts": "built", "pval": "built"}
    monkeypatch.setattr(data, "build_cache", fake_build)
    assert train.main(_script_argv(cmds[0], "/kit", "train.py")) == 0
    pid7 = sorted(r["pid"] for r in pairs.read_manifest(FIXTURE_MANIFEST))[7]
    assert got["build"] == ("/products", "/cache", pid7, pid7)


def test_dry_run_agg_lines_parse(monkeypatch, tmp_path):
    cmds = _dry("agg", ["/kit", "/products", "/out", "/refs.tsv", "B"], None)
    assert [c[1].rsplit("/", 1)[1] for c in cmds] == ["aggregate.py", "aggregate.py",
                                                      "figures.py", "report.py"]
    got = {}
    monkeypatch.setattr(aggregate, "aggregate", lambda *a: got.setdefault("agg", a) and
                        {"runs_present": [], "runs_missing": [], "checks": []})
    monkeypatch.setattr(aggregate, "qm_curves", lambda *a: got.setdefault("qm", a))
    assert aggregate.main(_script_argv(cmds[0], "/kit", "aggregate.py")) == 0
    assert [str(v) for v in got["agg"]] == ["/products/MANIFEST.tsv",
                                            "/kit/tools/t118/covariates.tsv", "/out/runs",
                                            "/refs.tsv", "/out/agg"]
    assert aggregate.main(_script_argv(cmds[1], "/kit", "aggregate.py")) == 0
    assert [str(v) for v in got["qm"]] == ["/products/MANIFEST.tsv", "/products",
                                           "/out/agg/qm_curves.json"]
    monkeypatch.setattr(figures, "draw_all", lambda *a: got.setdefault("fig", a) and [])
    assert figures.main(_script_argv(cmds[2], "/kit", "figures.py")) == 0
    assert got["fig"] == ("/out/agg", "B", "/out/agg/qm_curves.json", None)
    from ladder import report
    argv = _script_argv(cmds[3], "/kit", "report.py")
    assert argv == ["/out/agg", "B"]
    monkeypatch.setattr(report, "build_report", lambda agg, rung: got.setdefault("rep", (agg, rung))
                        and "stub\n")
    local = [a.replace("/out", str(tmp_path)) for a in argv]   # report.main writes its file
    assert report.main(local) == 0
    assert got["rep"] == (str(tmp_path / "agg"), "B")
    assert (tmp_path / "agg" / "B" / "report.md").read_text() == "stub\n"


# ---- the cache path ---------------------------------------------------------------------------

def test_cache_then_train_from_the_cache(tmp_path):
    products = tmp_path / "products"
    manifest, covariates = synth.make_products(products, seed=0)
    cache = tmp_path / "cache"
    rows = pairs.read_manifest(manifest)
    for i in range(len(rows)):
        assert train.main(["cache", str(manifest), str(products), str(cache), str(i)]) == 0
    assert len(list(cache.glob("*__*.json"))) == 2 * len(rows)
    npz, mm = data.Corpus(products), data.Corpus(cache)
    assert mm.is_cache and not npz.is_cache
    for pid in ("T1__depth__15M", "T2__pe__pe"):
        for space in pairs.SPACES:
            assert mm.chroms(pid, space) == npz.chroms(pid, space)
            for c in mm.chroms(pid, space):
                assert np.array_equal(mm.get(pid, space, c), npz.get(pid, space, c))

    logs = {}
    for name, data_dir in (("cache", cache), ("npz", products)):
        runs = tmp_path / f"runs_{name}"
        r = subprocess.run([sys.executable, str(LADDER / "train.py"), "train", str(manifest),
                            str(covariates), str(data_dir), str(runs), "A", "T1", "counts", "real",
                            "0", "--max-steps", "6", "--eval-every", "3", "--device", "cpu"],
                           env=_env(), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-3000:]
        rd = runs / "A_T1_counts_real_s0"
        assert (rd / "TRAIN_DONE").is_file() and (rd / "ckpt.pt").is_file()
        assert json.loads((rd / "config.json").read_text())["data_dir"] == str(data_dir)
        assert ("not the memory-mapped cache" in r.stderr) == (name == "npz")
        logs[name] = np.genfromtxt(rd / "train_log.tsv", delimiter="\t", names=True)
    # the same seed on the same data: the cache and the npz read give the same training run
    for col in ("step", "val_nll"):
        assert np.array_equal(logs["cache"][col], logs["npz"][col])
    assert np.allclose(logs["cache"]["loss"], logs["npz"]["loss"], equal_nan=True)
