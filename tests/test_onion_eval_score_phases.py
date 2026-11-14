"""SP2b (plan t152, Amendment 10, pinned interface M): the metric phase of the two-phase scorer.

Pins, on synthetic tables and a small perturbed model of each of the six rungs:
  - round trip: a store written by `predict_store.predict` (all six rungs) and by the merged SP2a
    command `predict_store.py run` (a saved run, rungs 1 and 5, through both command lines) is read
    back by the metric phase, and every numeric leaf of scores.json (apart from the version,
    settings, refs and timing keys), every suffstats.npz array, pairs_per_region.npz and every
    dumped method track file equal the legacy single-job output at the legacy batch;
  - the pair reference terms read from the baselines (`refs_from`) equal the terms the legacy path
    computes in the run;
  - refusals: a refs directory of another eval set, other tables, another lambda source or other pair
    regions; an incomplete store; a store with another `git_sha` and equal identity is accepted;
  - TF32 is off for matmul and cuDNN inside the legacy model path and the metric phase;
  - review fixes of SP2b: the store records the checkpoint's sha256 and the metric phase refuses a
    store of another checkpoint (a store of an older kit without the hash is accepted with a warning
    and recorded as null; shards of mixed kits do not refuse each other); the legacy model path
    writes `scorer_settings`; workers are refused after a model forward in the process; the metric
    phase refuses the predict-phase options and runs the previous-eval-set guard; a CRPS of 0.0
    stays 0.0 in suffstats;
  - review items of the scorer base (SF0): a killed worker raises; `--pairs-only` refuses a directory
    that holds SCORED; a staged copy whose size matches but whose mtime differs is copied fresh; a
    strict-subset rebuild from pairs_per_region.npz equals a direct scoring on that subset; the
    exact set of keys the SC before/after test ignores as new."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import registry as R  # noqa: E402
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import predict_store as P  # noqa: E402
from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (6, "H3K27me3"), (7, "chipseq-control"))
L, RB, LEGACY_B = 64, 256, 4
KW = dict(n_boot=10, guardrail_windows=8, log=lambda *a: None)
#: keys that differ between the legacy and the two-phase output by design (or are timings)
#: the metric phase's keyword arguments (no guardrail_windows: the predict phase owns it)
MKW = {k: v for k, v in KW.items() if k != "guardrail_windows"}
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads", "scorer_version", "scorer_settings",
            "refs", "pred_store", "pair_regions_n")


def _tsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """The fixture of test_onion_eval_predict_store.py: items, a predicted-control route, correction
    and replicate pairs, control targets; plus the two reference baselines scored on it."""
    root = tmp_path_factory.mktemp("phases")
    t = make_synthetic_tables(root, n_cells=8, val_tracks=VAL, blind_tracks=())
    pairs = _tsv(t / "pairs.tsv")
    meas = {r["measurement_id"]: r for r in _tsv(t / "measurements.tsv")}
    extra = []
    for r in meas.values():
        if r["in_eic"] == "val" and r["arm_knob"] == "" and r["assay"] == "H3K4me3":
            extra.append(dict(pairs[0], pair_id=f"rep{len(extra)}", a=r["measurement_id"],
                              b=r["measurement_id"].replace(".base", ".mapq0"), rung="technical",
                              assay=r["assay"], cell=r["cell"], subset_pair="0", split="val",
                              never_trained="0", knobs_differing="mapq"))
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
    with open(t / "pairs.tsv", "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]), delimiter="\t", lineterminator="\n")
        w.writerows(extra)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = ev["correction_pairs"][:4]
    ev["replicate_pairs"] = [p["pair_id"] for p in extra]
    out.write_text(json.dumps(ev))
    refs = root / "refs"
    for name in S.REFS_FROM_DIRS:              # baselines with their own default config
        S.score(out, refs / S.REFS_FROM_DIRS[name], baseline=name, **KW)
    return {"root": root, "tables": t, "evalset": out, "refs": refs, "sha": tables_sha256(t)}


def _model(r: int):
    cfg = {**json.loads((REPO / f"tools/minicandi/configs/rung{r}/base.json").read_text()),
           "context_bins": L, "batch_size": LEGACY_B}
    torch.manual_seed(r)
    m = R.build_model(cfg)
    g = torch.Generator().manual_seed(r)
    with torch.no_grad():       # the NB head starts at zero: perturb so outputs read the inputs
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval(), cfg


def _store(env, r, d, B=LEGACY_B, shards=1):
    m, cfg = _model(r)
    sc = S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256=env["sha"],
                  **KW)
    reqs = P.requests(sc)
    for i in range(shards):
        P.predict(sc, reqs, d, B, shard=(i, shards))
    return cfg


def _metrics(env, cfg, store, out, **kw):
    kw = {"refs_from": env["refs"], "workers": 1, **kw}
    return S.score(env["evalset"], out, phase="metrics", pred_store=store, config=cfg,
                   run_tables_sha256=env["sha"], **MKW, **kw)


def _subprocess(args, code=None):
    """Run score.py (or `code`) in a fresh interpreter. Forked workers must start from a process
    that has run no multi-threaded torch work (`Scorer._parallel`): on Linux a GNU OpenMP pool
    started by a model forward in this pytest process hangs the children (seen on Nibi), so every
    test with more than one worker runs the scorer the way score.sh does, in its own process."""
    import subprocess
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "tools")]),
               OMP_NUM_THREADS="2", OMP_WAIT_POLICY="PASSIVE")
    cmd = [sys.executable] + (["-c", code] if code else [str(REPO / "tools/onion_eval/score.py")])
    return subprocess.run(cmd + list(args), env=env, capture_output=True, text=True, timeout=1800)


def _save_run(root: Path, r: int, sha: str) -> Path:
    m, cfg = _model(r)
    run = root / "runs" / f"rung{r}_base" / "seed0"
    run.mkdir(parents=True)
    torch.save({"state_dict": m.state_dict(), "config": cfg, "param_count": m.param_count(),
                "windows_seen": 0, "tables_sha256": sha, "git_sha": "test"}, run / R.CKPT_NAME)
    return run


def _flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, json.dumps(x)


def leaf_differences(a: Path, b: Path, extra_volatile=()) -> list:
    """Differing leaves of two score directories: scores.json (VOLATILE dropped; a key on one side
    only is a difference), every array of suffstats.npz and pairs_per_region.npz (dtype, shape and
    bytes), and every file of a `dump` subdirectory (bytes)."""
    diff = []
    sa, sb = (json.loads((d / "scores.json").read_text()) for d in (a, b))
    for s in (sa, sb):
        for k in VOLATILE + tuple(extra_volatile):
            s.pop(k, None)
    fa, fb = dict(_flat(sa)), dict(_flat(sb))
    diff += [f"scores.json{k}" for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    for f in ("suffstats.npz", "pairs_per_region.npz"):
        if (a / f).is_file() != (b / f).is_file():
            diff.append(f)
            continue
        if not (a / f).is_file():
            continue
        with np.load(a / f) as za, np.load(b / f) as zb:
            for k in sorted(set(za.files) | set(zb.files)):
                if k not in za.files or k not in zb.files:
                    diff.append(f"{f}:{k}")
                    continue
                x, y = za[k], zb[k]
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    diff.append(f"{f}:{k}")
    da, db = a / "dump", b / "dump"
    if da.is_dir() or db.is_dir():
        ra = sorted(p.relative_to(da) for p in da.rglob("*") if p.is_file())
        rb = sorted(p.relative_to(db) for p in db.rglob("*") if p.is_file())
        diff += [f"dump/{r}" for r in sorted(set(ra) ^ set(rb))]
        diff += [f"dump/{r}" for r in sorted(set(ra) & set(rb))
                 if (da / r).read_bytes() != (db / r).read_bytes()]
    return diff


# ------------------------------------------------------------------------------- the round trip

@pytest.mark.parametrize("r", sorted(R.RUNGS))
def test_metric_phase_equals_legacy_for_each_rung(env, tmp_path, r):
    m, cfg = _model(r)
    leg = tmp_path / "legacy"
    S.score(env["evalset"], leg, model=m, config=cfg, run_tables_sha256=env["sha"],
            dump_signal=str(leg / "dump"), **KW)
    store = tmp_path / "store"
    _store(env, r, store, shards=2)
    tp = tmp_path / "two_phase"
    s = _metrics(env, cfg, store, tp, dump_signal=str(tp / "dump"))
    assert leaf_differences(leg, tp) == []
    assert s["scorer_version"] == 2 and s["pair_regions_n"] == 2
    assert s["scorer_settings"] == {"device": "cpu", "infer_batch": LEGACY_B, "tf32": False,
                                    "amp": False, "phase": "metrics"}
    assert set(s["refs"]) == {"dir", "sha256_depth_only", "sha256_floor"}
    assert s["n_pairs"] > 0 and s["n_replicate_pairs"] > 0
    assert s["predicted_control"]["n_targets"] >= (1 if cfg.get("predict_control") else 0)
    legacy = json.loads((leg / "scores.json").read_text())
    assert legacy["scorer_version"] == 1 and legacy["scorer_settings"]["phase"] == "legacy"


@pytest.mark.parametrize("r", (1, 5))
def test_cli_round_trip_with_the_predict_command(env, tmp_path, r):
    """A run directory: `predict_store.py run` (SP2a, 2 shards) then `score.py --phase metrics`
    against `score.py --run` (legacy), through the command lines."""
    run = _save_run(tmp_path, r, env["sha"])
    common = ["--run", str(run), "--evalset", str(env["evalset"])]
    gw = ["--guardrail-windows", str(KW["guardrail_windows"])]
    assert S.main(common + ["--out", str(tmp_path / "legacy"), "--device", "cpu", "--n-boot",
                            str(KW["n_boot"])] + gw) == 0
    store = tmp_path / "store"
    for i in range(2):
        assert P.run(common + ["--pred-store", str(store), "--shard", f"{i}/2", "--device",
                               "cpu"] + gw) == 0
    p = _subprocess(common + ["--out", str(tmp_path / "tp"), "--phase", "metrics", "--pred-store",
                              str(store), "--refs-from", str(env["refs"]), "--workers", "2",
                              "--stage-dir", str(tmp_path / "stage"), "--n-boot",
                              str(KW["n_boot"])])
    assert p.returncode == 0, p.stderr[-2000:]
    assert leaf_differences(tmp_path / "legacy", tmp_path / "tp") == []
    s = json.loads((tmp_path / "tp" / "scores.json").read_text())
    assert s["run"] == str(run.resolve()) and s["scorer_version"] == 2
    # the command line refuses a metric phase without its inputs, and the inputs without the phase
    with pytest.raises(SystemExit):
        S.main(common + ["--out", str(tmp_path / "x"), "--phase", "metrics", "--pred-store",
                         str(store)])
    with pytest.raises(SystemExit):
        S.main(common + ["--out", str(tmp_path / "x"), "--refs-from", str(env["refs"])])


def test_reference_terms_read_equal_terms_computed_in_the_run(env, tmp_path):
    cfg = _store(env, 1, tmp_path / "store")
    a = _metrics(env, cfg, tmp_path / "store", tmp_path / "read")
    b = _metrics(env, cfg, tmp_path / "store", tmp_path / "computed", refs_from=None)
    assert b["refs"] == {"dir": None, "computed_in_run": True}
    assert leaf_differences(tmp_path / "read", tmp_path / "computed") == []
    with np.load(tmp_path / "read" / "suffstats.npz") as z:
        pk = [k for k in z.files if k.startswith("pair/")]
        assert pk
        with np.load(env["refs"] / "depth_only_scaling/seed0/suffstats.npz") as d, \
                np.load(env["refs"] / "pseudoreplicate_floor/seed0/suffstats.npz") as f:
            for k in pk:
                assert z[k][1].tobytes() == d[k][1].tobytes()
                assert z[k][2].tobytes() == f[k][2].tobytes()
    assert a["refs"]["sha256_floor"] == S.sha256_file(
        env["refs"] / "pseudoreplicate_floor/seed0/suffstats.npz")


def test_workers_and_store_batch_do_not_move_a_value(env, tmp_path):
    """1 against 3 workers give the same outputs; a store at another inference batch is read, and
    its batch is recorded (its floats may differ in the last bits: a property of the store)."""
    run = _save_run(tmp_path, 4, env["sha"])
    common = ["--run", str(run), "--evalset", str(env["evalset"])]
    gw = ["--guardrail-windows", str(KW["guardrail_windows"])]
    for d, b in (("s4", LEGACY_B), ("s16", 16)):
        assert P.run(common + ["--pred-store", str(tmp_path / d), "--device", "cpu",
                               "--infer-batch", str(b)] + gw) == 0
    S.score(env["evalset"], tmp_path / "w1", run=run, phase="metrics", pred_store=tmp_path / "s4",
            refs_from=env["refs"], workers=1, **MKW)
    p = _subprocess(common + ["--out", str(tmp_path / "w3"), "--phase", "metrics", "--pred-store",
                              str(tmp_path / "s4"), "--refs-from", str(env["refs"]), "--workers",
                              "3", "--n-boot", str(KW["n_boot"])])
    assert p.returncode == 0, p.stderr[-2000:]
    s = S.score(env["evalset"], tmp_path / "b16", run=run, phase="metrics",
                pred_store=tmp_path / "s16", refs_from=env["refs"], **MKW)
    assert leaf_differences(tmp_path / "w1", tmp_path / "w3") == []
    assert s["infer_batch"] == 16 and s["scorer_settings"]["infer_batch"] == 16


def test_pairs_only_in_the_metric_phase(env, tmp_path):
    cfg = _store(env, 1, tmp_path / "store")
    _metrics(env, cfg, tmp_path / "store", tmp_path / "full")
    res = _metrics(env, cfg, tmp_path / "store", tmp_path / "po", pairs_only=True)
    assert res["scorer_version"] == 2 and not (tmp_path / "po" / "scores.json").exists()
    with np.load(tmp_path / "full" / "pairs_per_region.npz") as a, \
            np.load(tmp_path / "po" / "pairs_per_region.npz") as b:
        assert sorted(a.files) == sorted(b.files)
        assert all(a[k].tobytes() == b[k].tobytes() for k in a.files)
    with pytest.raises(S.ScoreRefused, match="SCORED"):
        _metrics(env, cfg, tmp_path / "store", tmp_path / "full", pairs_only=True)


# ------------------------------------------------------------------------------- refusals

@pytest.fixture(scope="module")
def store1(env, tmp_path_factory):
    d = tmp_path_factory.mktemp("store1") / "store"
    return _store(env, 1, d), d


def _planted_refs(env, tmp_path, edit):
    d = tmp_path / "refs"
    shutil.copytree(env["refs"], d)
    for name, sub in S.REFS_FROM_DIRS.items():
        edit(d / sub, name)
    return d


def _edit_scores(key, value):
    def edit(d, name):
        p = d / "scores.json"
        s = json.loads(p.read_text())
        s[key] = value
        p.write_text(json.dumps(s))
    return edit


def _edit_regions(d, name):
    p = d / "pairs_per_region.npz"
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    a["region_indices"] = a["region_indices"][:1]
    np.savez_compressed(p, **a)


@pytest.mark.parametrize("what,edit,match", [
    ("evaluation set", _edit_scores("evalset_sha256", "0" * 64), "evalset_sha256"),
    ("tables", _edit_scores("tables_sha256", "0" * 64), "tables_sha256"),
    ("lambda source", _edit_scores("lambda_source_option", "observed_control"), "lambda source"),
    ("pair regions (count)", _edit_scores("pair_regions_n", 1), "pair_regions_n"),
    ("pair regions (file)", _edit_regions, "pair regions"),
])
def test_planted_refs_mismatch_is_refused(env, store1, tmp_path, what, edit, match):
    cfg, store = store1
    refs = _planted_refs(env, tmp_path, edit)
    with pytest.raises(S.ScoreRefused, match=match):
        _metrics(env, cfg, store, tmp_path / "out", refs_from=refs)
    assert not (tmp_path / "out" / "scores.json").exists()


def test_refs_missing_a_pair_is_refused(env, store1, tmp_path):
    cfg, store = store1

    def drop(d, name):
        p = d / "suffstats.npz"
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        del a[sorted(k for k in a if k.startswith("pair/"))[0]]
        np.savez_compressed(p, **a)
    with pytest.raises(S.ScoreRefused, match="not in the reference"):
        _metrics(env, cfg, store, tmp_path / "out", refs_from=_planted_refs(env, tmp_path, drop))


def test_incomplete_store_is_refused(env, tmp_path):
    cfg = _store(env, 1, tmp_path / "a")
    st = P.PredictionStore(tmp_path / "a")
    one = next(p for p in (tmp_path / "a").glob("*.npz"))
    one.unlink()
    with pytest.raises(S.ScoreRefused, match="not complete"):
        _metrics(env, cfg, tmp_path / "a", tmp_path / "out")
    _store(env, 1, tmp_path / "b", shards=2)
    (tmp_path / "b" / "PREDICTED_1of2.json").unlink()
    with pytest.raises(S.ScoreRefused, match="not complete"):
        _metrics(env, cfg, tmp_path / "b", tmp_path / "out")
    assert st.info["n_shards"] == 1


def test_store_of_another_scoring_is_refused(env, store1, tmp_path):
    cfg, store = store1
    other = env["evalset"].with_name("evalset_other.json")
    other.write_text(json.dumps(dict(json.loads(env["evalset"].read_text()), note="other")))
    with pytest.raises(S.ScoreRefused, match="another scoring"):
        S.score(other, tmp_path / "out", phase="metrics", pred_store=store, config=cfg,
                run_tables_sha256=env["sha"], **MKW)


def test_store_with_another_git_sha_is_accepted(env, tmp_path):
    cfg = _store(env, 1, tmp_path / "s")
    p = tmp_path / "s" / "PREDICTED_0of1.json"
    d = json.loads(p.read_text())
    d["git_sha"] = "f" * 40
    p.write_text(json.dumps(d))
    s = _metrics(env, cfg, tmp_path / "s", tmp_path / "out")
    assert s["scorer_version"] == 2 and (tmp_path / "out" / S.SCORED).is_file()


def test_metric_phase_builds_no_model(env, store1, tmp_path, monkeypatch):
    cfg, store = store1

    def boom(*a, **k):
        raise AssertionError("the metric phase built or called a model")
    monkeypatch.setattr(S.ModelPredictor, "__call__", boom)
    monkeypatch.setattr(R, "build_model", boom)
    _metrics(env, cfg, store, tmp_path / "out", workers=1)


# ------------------------------------------------------------------------------- TF32

def test_tf32_off_in_the_legacy_model_path_and_the_metric_phase(env, store1, tmp_path, monkeypatch):
    seen = []
    orig = S.ModelPredictor.__call__

    def spy(self, batch):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return orig(self, batch)
    monkeypatch.setattr(S.ModelPredictor, "__call__", spy)
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        m, cfg = _model(1)
        S.score(env["evalset"], tmp_path / "leg", model=m, config=cfg, run_tables_sha256=env["sha"],
                **KW)
        assert seen and set(seen) == {(False, False)}
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        cfg, store = store1
        _metrics(env, cfg, store, tmp_path / "tp", workers=1)
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (False,
                                                                                              False)
        assert json.loads((store / "STORE.json").read_text())["tf32"] is False
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


# ------------------------------------------------------------------------------- review items (SF0)

def test_a_killed_worker_raises(env):
    """A worker that dies raises BrokenProcessPool in the parent at once (in a fresh process)."""
    code = (
        "import os, sys\n"
        "from concurrent.futures.process import BrokenProcessPool\n"
        "from onion_eval import score as S\n"
        "orig = S._worker_task\n"
        "def die(i):\n"
        "    if i == 1:\n"
        "        os._exit(3)\n"
        "    return orig(i)\n"
        "S._worker_task = die\n"
        "try:\n"
        "    S.score(sys.argv[1], sys.argv[2], baseline='marginal_nb', workers=2, n_boot=0,\n"
        "            log=lambda *a: None)\n"
        "except BrokenProcessPool:\n"
        "    print('BROKEN')\n")
    out = env["root"] / "killed"
    p = _subprocess([str(env["evalset"]), str(out)], code=code)
    assert p.returncode == 0 and "BROKEN" in p.stdout, p.stderr[-2000:]
    assert not (out / "scores.json").exists()


def test_pairs_only_refuses_a_scored_directory(env, tmp_path):
    out = tmp_path / "out"
    S.score(env["evalset"], out, baseline="marginal_nb", **KW)
    before = (out / "pairs_per_region.npz").read_bytes()
    with pytest.raises(S.ScoreRefused, match="SCORED"):
        S.score(env["evalset"], out, baseline="marginal_nb", pairs_only=True, **KW)
    assert (out / "pairs_per_region.npz").read_bytes() == before
    assert S.main(["--baseline", "marginal_nb", "--evalset", str(env["evalset"]), "--out",
                   str(out), "--pairs-only", "--force", "--n-boot", "0"]) == 2


def test_staged_copy_with_equal_size_and_other_mtime_is_copied_fresh(env, tmp_path):
    orig = {Path(r["counts_path"][3:]).parent.name: Path(r["counts_path"][3:]).parent
            for r in S.Scorer(env["evalset"], main="marginal_nb", **KW).ds.rows.values()
            if r["counts_path"].startswith("h5:")}

    def stage():
        sc = S.Scorer(env["evalset"], main="marginal_nb", stage_dir=tmp_path / "stage", **KW)
        sc._stage(*sc.jobs())
    stage()
    dst = next(p for p in sorted((tmp_path / "stage").rglob("counts.h5")) if not p.is_symlink())
    src = orig[dst.parent.name] / dst.name
    good = src.read_bytes()
    assert dst.read_bytes() == good and dst.stat().st_mtime_ns == src.stat().st_mtime_ns
    dst.write_bytes(bytes(len(good)))            # same size, other bytes
    os.utime(dst, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10 ** 9))
    stage()
    assert dst.read_bytes() == good and dst.stat().st_mtime_ns == src.stat().st_mtime_ns


def test_strict_subset_rebuild_equals_direct_scoring(tmp_path):
    """A strict subset of the pair regions rebuilt from pairs_per_region.npz (the check tool's
    aggregation) equals a scoring whose eval set has `pair_regions` set to that subset."""
    import test_onion_eval_score_pairs_rules as PR
    from onion_eval import pair_regions_check as PC
    env = PR._build(tmp_path / "b", True)
    PR._score(env, tmp_path / "all")
    ev = json.loads(env["evalset"].read_text())
    sub = [0, 2]
    assert len(sub) < PR.N_REGIONS
    p = env["evalset"].with_name("evalset_sub.json")
    p.write_text(json.dumps(dict(ev, pair_regions=sub, tables_dir=str(env["tables"]))))
    PR._score(env, tmp_path / "sub", evalset=p)
    w = np.isin(np.arange(PR.N_REGIONS), sub).astype(np.float64)
    got = PC.values(PC._agg(PC.Run(tmp_path / "all", ev), w, S.pair_metrics_from_records,
                            ev["scorable"]))
    direct = PC.values(PR._load(tmp_path / "sub"))

    def close(a, b):
        return (a is None and b is None) or (a is not None and b is not None
                                             and a == pytest.approx(b, rel=1e-9, abs=1e-12))
    for ref in (direct,):
        assert close(got["h4"], ref["h4"]) and close(got["h5"], ref["h5"])
        assert set(got["h4_assay"]) == set(ref["h4_assay"])
        assert all(close(got["h4_assay"][a], ref["h4_assay"][a]) for a in got["h4_assay"])
        # a rung with no pair job (Undefined in the scoring) is not in the rebuilt records
        assert got["rep"] and set(got["rep"]) <= set(ref["rep"])
        assert all(ref["rep"][g] is None for g in set(ref["rep"]) - set(got["rep"]))
        assert all(close(v, ref["rep"][g]) for g, v in got["rep"].items())
    with np.load(tmp_path / "all" / "pairs_per_region.npz") as za, \
            np.load(tmp_path / "sub" / "pairs_per_region.npz") as zs:
        assert list(zs["region_indices"]) == sub and list(zs["pair_ids"]) == list(za["pair_ids"])
        np.testing.assert_allclose(zs["crps_sum_model"], za["crps_sum_model"][:, sub], rtol=1e-12)
    a_s, s_s = PR._suff(tmp_path / "all"), PR._suff(tmp_path / "sub")
    assert any(a_s[k][0] != s_s[k][0] for k in a_s if k.startswith("pair/"))   # not a no-op


#: the keys the SC before/after test (test_onion_eval_score_arms.py) ignores as added after its base
#: (6d5dac3): headline 7 and the explained fraction (later chunks), the noise-floor bias and depth of
#: a per-target block, the F4 label, and SF0's scorer_version; measured on the laptop 2026-11-14
ARMS_NEW_KEYS = {
    "/accessibility_only/explained_fraction_per_layer",
    "/accessibility_only/predicted_over_replicate_variance/description",
    "/accessibility_only/predicted_over_replicate_variance/pass_rate",
    "/accessibility_only/predicted_over_replicate_variance/per_rung",
    "/accessibility_only/predicted_over_replicate_variance/per_target",
    "/accessibility_only/predicted_over_replicate_variance/upper_bound_rung",
    "/explained_fraction_per_layer",
    "/predicted_control/explained_fraction_per_layer",
    "/predicted_control/predicted_over_replicate_variance/description",
    "/predicted_control/predicted_over_replicate_variance/pass_rate",
    "/predicted_control/predicted_over_replicate_variance/per_rung",
    "/predicted_control/predicted_over_replicate_variance/per_target",
    "/predicted_control/predicted_over_replicate_variance/upper_bound_rung",
    "/predicted_control/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000040.pooled.uniform.base/N_log1p_bias_sim",
    "/predicted_control/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000040.pooled.uniform.base/depth_reads",
    "/predicted_over_replicate_variance/description",
    "/predicted_over_replicate_variance/pass_rate",
    "/predicted_over_replicate_variance/per_rung",
    "/predicted_over_replicate_variance/per_target",
    "/predicted_over_replicate_variance/upper_bound_rung",
    "/scorer_settings",
    "/scorer_version",
    "/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000003.pooled.uniform.base/N_log1p_bias_sim",
    "/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000003.pooled.uniform.base/depth_reads",
    "/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000008.pooled.uniform.base/N_log1p_bias_sim",
    "/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000008.pooled.uniform.base/depth_reads",
    "/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000013.pooled.uniform.base/N_log1p_bias_sim",
    "/specific_variance_recovered_above_noise/counts_log1p/per_target/ENCSR000013.pooled.uniform.base/depth_reads",
    "/specific_variance_recovered_above_noise/signal_arcsinh/per_target/ENCSR000008.pooled.uniform.base/lambda_source_training_average",
    "/specific_variance_recovered_above_noise/signal_arcsinh/per_target/ENCSR000013.pooled.uniform.base/lambda_source_training_average",
    "/specific_variance_recovered_above_noise/signal_arcsinh_plug_in/per_target/ENCSR000008.pooled.uniform.base/lambda_source_training_average",
    "/specific_variance_recovered_above_noise/signal_arcsinh_plug_in/per_target/ENCSR000013.pooled.uniform.base/lambda_source_training_average",
}


def test_arms_before_after_new_keys_are_the_expected_set(tmp_path_factory):
    import test_onion_eval_score_arms as A
    env = A.env.__wrapped__(tmp_path_factory)
    A.IGNORED_NEW_KEYS.clear()
    A.test_before_after_bit_identical(env, A.default_model.__wrapped__(env), tmp_path_factory)
    assert A.IGNORED_NEW_KEYS == ARMS_NEW_KEYS


# ------------------------------------------------------------------------------- review fixes (SP2b)

def _predict_run(env, run, store, shards=1):
    gw = ["--guardrail-windows", str(KW["guardrail_windows"])]
    for i in range(shards):
        assert P.run(["--run", str(run), "--evalset", str(env["evalset"]), "--pred-store",
                      str(store), "--shard", f"{i}/{shards}", "--device", "cpu"] + gw) == 0


def _metrics_run(env, run, store, out, log=None):
    kw = dict(MKW, log=log) if log else MKW
    return S.score(env["evalset"], out, run=run, phase="metrics", pred_store=store,
                   refs_from=env["refs"], **kw)


def test_store_records_the_checkpoint_and_the_metric_phase_checks_it(env, tmp_path):
    run = _save_run(tmp_path, 1, env["sha"])
    sha = P.ckpt_sha256(run)
    _predict_run(env, run, tmp_path / "store")
    assert json.loads((tmp_path / "store" / "STORE.json").read_text())[P.CKPT_KEY] == sha
    s = _metrics_run(env, run, tmp_path / "store", tmp_path / "ok")
    assert s["pred_store"]["ckpt_sha256"] == sha
    # the run's checkpoint changed after the store was made: refused, nothing scored
    ck = torch.load(run / R.CKPT_NAME, weights_only=False)
    with torch.no_grad():
        ck["state_dict"] = {k: v + 1e-3 if v.is_floating_point() else v
                            for k, v in ck["state_dict"].items()}
    torch.save(ck, run / R.CKPT_NAME)
    with pytest.raises(S.ScoreRefused, match="another checkpoint"):
        _metrics_run(env, run, tmp_path / "store", tmp_path / "stale")
    assert not (tmp_path / "stale" / S.SCORED).exists()


def test_store_of_an_older_kit_is_accepted_with_a_warning(env, tmp_path):
    run = _save_run(tmp_path, 1, env["sha"])
    _predict_run(env, run, tmp_path / "store")
    for f in [tmp_path / "store" / "STORE.json", *(tmp_path / "store").glob("PREDICTED_*.json")]:
        d = json.loads(f.read_text())
        d.pop(P.CKPT_KEY)
        f.write_text(json.dumps(d))
    lines = []
    s = _metrics_run(env, run, tmp_path / "store", tmp_path / "out", log=lines.append)
    assert s["pred_store"]["ckpt_sha256"] is None
    assert sum("WARNING" in str(x) and "checkpoint hash" in str(x) for x in lines) == 1


def test_shards_of_mixed_kits_do_not_refuse_each_other(env, tmp_path):
    head = {"format": 1, "run": "/r", "n_shards": 2}
    d = tmp_path / "a"
    d.mkdir()
    P._claim_store(d, head)                                   # an older kit: no hash
    P._claim_store(d, dict(head, **{P.CKPT_KEY: "a" * 64}))   # a newer kit: accepted
    e = tmp_path / "b"
    e.mkdir()
    P._claim_store(e, dict(head, **{P.CKPT_KEY: "a" * 64}))
    P._claim_store(e, head)                                   # the other order: accepted
    with pytest.raises(S.ScoreRefused, match=P.CKPT_KEY):     # two different hashes: refused
        P._claim_store(e, dict(head, **{P.CKPT_KEY: "b" * 64}))


def test_legacy_model_path_writes_scorer_settings_and_a_baseline_does_not(env, tmp_path):
    m, cfg = _model(1)
    s = S.score(env["evalset"], tmp_path / "m", model=m, config=cfg, run_tables_sha256=env["sha"],
                **KW)
    assert s["scorer_settings"] == {"device": "cpu", "infer_batch": LEGACY_B, "tf32": False,
                                    "amp": False, "phase": "legacy"}
    assert s["scorer_version"] == 1
    b = S.score(env["evalset"], tmp_path / "b", baseline="marginal_nb", **KW)
    assert "scorer_settings" not in b


def test_workers_refused_after_a_model_forward(env, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_MODEL_RAN", False)
    m, cfg = _model(1)
    sc = S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256=env["sha"],
                  **KW)
    req = P.requests(sc)[0]
    P.request_arrays(sc, dict(req, region_indices=req["region_indices"][:1]), LEGACY_B)
    assert S._MODEL_RAN is True
    with pytest.raises(RuntimeError, match="model forward"):
        S.score(env["evalset"], tmp_path / "out", baseline="marginal_nb", workers=2, **KW)


def test_metric_phase_refuses_predict_phase_options_and_runs_the_evalset_guard(env, store1,
                                                                               tmp_path,
                                                                               monkeypatch):
    cfg, store = store1
    for k, v in (("infer_batch", 8), ("guardrail_windows", 8)):
        with pytest.raises(ValueError, match="predict phase"):
            S.score(env["evalset"], tmp_path / "x", phase="metrics", pred_store=store, config=cfg,
                    run_tables_sha256=env["sha"], refs_from=env["refs"], **{k: v}, **MKW)
    run = _save_run(tmp_path, 1, env["sha"])
    base = ["--run", str(run), "--evalset", str(env["evalset"]), "--out", str(tmp_path / "y"),
            "--phase", "metrics", "--pred-store", str(store), "--refs-from", str(env["refs"])]
    for extra in (["--infer-batch", "8"], ["--guardrail-windows", "8"]):
        with pytest.raises(SystemExit):
            S.main(base + extra)
    seen = []
    monkeypatch.setattr(S, "_guard_previous_evalset", lambda out, ev, kw: seen.append(out))
    _metrics(env, cfg, store, tmp_path / "z")
    assert seen == [tmp_path / "z"]


def test_a_crps_of_exactly_zero_is_kept_in_suffstats():
    assert S._nan_if_none(0.0) == 0.0
    assert np.isnan(S._nan_if_none(S.Mx.Undefined("undefined")))
    assert np.isnan(S._nan_if_none(None))
    assert S._nan_if_none(1.5) == 1.5
