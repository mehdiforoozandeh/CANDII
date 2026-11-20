"""CM (PI ruling 2026-11-16, review item 9): the opt-in same-view pass and the correction-only term.

Pure: `correction_only_block` is 1 for a model whose cross-view and same-view scores are equal, 0
for the depth-only model (m = d, s = f), pooled per pair type then the macro over pair types;
`same_view_jobs` gives one job per distinct headline target. End to end on synthetic tables with
a small perturbed model: the switch off leaves scores.json, suffstats.npz and pairs_per_region.npz
as they are (only the new key and file appear with it on); the depth-only baseline scores 0 through
the scorer; the prediction-store path (predict + metric phase) gives the legacy same-view values;
a store made without the switch is refused by a metric phase with it."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import score as S  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (7, "chipseq-control"))
L, RB, B = 64, 256, 4
KW = dict(n_boot=5, guardrail_windows=4, log=lambda *a: None)
MKW = {k: v for k, v in KW.items() if k != "guardrail_windows"}
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads")


# ------------------------------------------------------------------------------------------ pure

def _rec(knob, m, d, f, target="t1"):
    return {"pair_id": f"{knob}-{target}", "source": "s", "target": target, "knob": knob, "rung": "",
            "never_trained": 1, "subset_pair": 0, "crps_model": m, "crps_depth_only": d,
            "crps_floor": f}


def test_term_is_one_when_cross_view_equals_same_view_and_zero_for_depth_only():
    cor = [_rec("crop", 0.7, 1.0, 0.8, "t1"), _rec("crop", 0.9, 1.2, 0.9, "t2"),
           _rec("mapq+nodedup", 0.6, 0.9, 0.5, "t3")]
    same = [{"target": r["target"], "crps_model_same_view": r["crps_model"]} for r in cor]
    b = S.correction_only_block(cor, same)
    assert b["value"] == 1.0 and b["n_pairs"] == 3 and b["n_pairs_by_type"] == {"crop": 2,
                                                                                "mapq+nodedup": 1}
    dep = [dict(r, crps_model=r["crps_depth_only"]) for r in cor]
    same_dep = [{"target": r["target"], "crps_model_same_view": r["crps_floor"]} for r in cor]
    assert S.correction_only_block(dep, same_dep)["value"] == 0.0


def test_pooled_per_pair_type_then_macro():
    cor = [_rec("crop", 0.7, 1.0, 0.8, "t1"), _rec("crop", 0.9, 1.2, 0.9, "t2"),
           _rec("mapq+nodedup", 0.6, 0.9, 0.5, "t3"),
           dict(_rec("crop", 9.0, 9.0, 0.0, "t4"), never_trained=0)]       # not a headline pair
    same = [{"target": "t1", "crps_model_same_view": 0.6}, {"target": "t2", "crps_model_same_view": 0.8},
            {"target": "t3", "crps_model_same_view": 0.55}]
    b = S.correction_only_block(cor, same)
    crop = 1 - ((0.7 + 0.9) / 2 - (0.6 + 0.8) / 2) / ((1.0 + 1.2) / 2 - (0.8 + 0.9) / 2)
    mq = 1 - (0.6 - 0.55) / (0.9 - 0.5)
    assert b["per_pair_type"]["crop"]["value"] == pytest.approx(crop)
    assert b["per_pair_type"]["mapq+nodedup"]["value"] == pytest.approx(mq)
    assert b["value"] == pytest.approx((crop + mq) / 2)
    assert b["n_pairs"] == 3 and b["s_mean"] == pytest.approx((0.6 + 0.8 + 0.55) / 3)
    # a target without a same-view score: undefined with the reason
    u = S.correction_only_block(cor, same[:2])
    assert u["value"] is None and "no same-view score" in u["reason"]


def test_same_view_jobs_one_per_distinct_headline_target():
    cor = [{"kind": "correction", "source": "a", "target": "t", "never_trained": 1, "subset_pair": 0,
            "assay": "X", "cell": "c"},
           {"kind": "correction", "source": "b", "target": "t", "never_trained": 1, "subset_pair": 0,
            "assay": "X", "cell": "c"},
           {"kind": "correction", "source": "a", "target": "u", "never_trained": 0, "subset_pair": 0,
            "assay": "X", "cell": "c"}]
    js = S.same_view_jobs(cor)
    assert [(j["source"], j["target"], j["pair_id"]) for j in js] == [("t", "t", "same_view/t")]


# ------------------------------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import tables_sha256
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("cor_only")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = [p for p in ev["correction_pairs"]
                              if p["never_trained"] and not p["subset_pair"]][:4]
    ev["replicate_pairs"] = []
    assert ev["correction_pairs"]
    out.write_text(json.dumps(ev))
    refs = root / "refs"
    for name in S.REFS_FROM_DIRS:
        S.score(out, refs / S.REFS_FROM_DIRS[name], baseline=name, **KW)
    return {"root": root, "evalset": out, "refs": refs, "sha": tables_sha256(t), "ev": ev}


def _model():
    from minicandi import registry as R
    cfg = {**json.loads((REPO / "tools/minicandi/configs/rung2/base.json").read_text()),
           "context_bins": L, "batch_size": B}
    torch.manual_seed(2)
    m = R.build_model(cfg)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return m.eval(), cfg


def _legacy(env, out, **kw):
    m, cfg = _model()
    return S.score(env["evalset"], out, model=m, config=cfg, run_tables_sha256=env["sha"], **KW, **kw)


def _npz(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _same_npz(a, b):
    za, zb = _npz(a), _npz(b)
    return za.keys() == zb.keys() and all(
        za[k].dtype == zb[k].dtype and za[k].shape == zb[k].shape and za[k].tobytes() == zb[k].tobytes()
        for k in za)


def test_default_off_leaves_outputs_identical(env, monkeypatch):
    monkeypatch.delenv(S.SAME_VIEW_ENV, raising=False)
    off, on = env["root"] / "off", env["root"] / "on"
    _legacy(env, off)
    _legacy(env, on, same_view=True)
    so, sn = (json.loads((d / "scores.json").read_text()) for d in (off, on))
    assert S.CORRECTION_ONLY_KEY not in so and not (off / "same_view_per_region.npz").exists()
    assert S.CORRECTION_ONLY_KEY in sn and (on / "same_view_per_region.npz").is_file()
    sn.pop(S.CORRECTION_ONLY_KEY)
    for s in (so, sn):
        for k in VOLATILE:
            s.pop(k, None)
    assert so == sn                     # every other key and value as without the switch
    assert _same_npz(off / "suffstats.npz", on / "suffstats.npz")
    assert _same_npz(off / "pairs_per_region.npz", on / "pairs_per_region.npz")
    blk = json.loads((on / "scores.json").read_text())[S.CORRECTION_ONLY_KEY]
    n_t = len({p["target"] for p in env["ev"]["correction_pairs"]})
    assert blk["n_same_view_targets"] == n_t and blk["n_pairs"] == len(env["ev"]["correction_pairs"])
    assert blk["value"] is not None and np.isfinite(blk["value"])
    z = _npz(on / "same_view_per_region.npz")
    assert z["crps_sum_model"].shape == (n_t, len(z["region_indices"]))
    # the per-region sums add up to the pair CRPS times its bins
    tot = z["crps_sum_model"].sum(1) / z["bins"].sum(1)
    assert np.allclose(tot, z["crps_model_same_view"], rtol=1e-12)


def test_depth_only_baseline_scores_zero(env, monkeypatch):
    monkeypatch.setenv(S.SAME_VIEW_ENV, "1")            # the env route of submit_arms.sh
    out = env["root"] / "dos_sv"
    assert S.main(["--baseline", "depth_only_scaling", "--evalset", str(env["evalset"]), "--out",
                   str(out), "--n-boot", "5", "--guardrail-windows", "4"]) == 0
    blk = json.loads((out / "scores.json").read_text())[S.CORRECTION_ONLY_KEY]
    # m = d exactly; s = f up to the float32 depth covariates and the two NB-size fits (2.5e-10 seen)
    assert blk["value"] == pytest.approx(0.0, abs=1e-8)
    for v in blk["per_pair_type"].values():
        assert v["m_mean"] == pytest.approx(v["d_mean"], rel=1e-12)
        assert v["s_mean"] == pytest.approx(v["f_mean"], rel=1e-8)


def test_store_path_matches_legacy_and_refuses_a_store_without_the_switch(env, monkeypatch):
    from onion_eval import predict_store as P
    monkeypatch.delenv(S.SAME_VIEW_ENV, raising=False)
    m, cfg = _model()
    sc = S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256=env["sha"],
                  same_view=True, **KW)
    reqs = P.requests(sc)
    n_sv = sum(r["key"].startswith("pair/" + S.SAME_VIEW_PREFIX) for r in reqs)
    base = S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256=env["sha"],
                    **KW)
    assert len(reqs) == len(P.requests(base)) + n_sv and n_sv > 0
    store = env["root"] / "store_sv"
    P.predict(sc, reqs, store, B)
    out = env["root"] / "metrics_sv"
    S.score(env["evalset"], out, phase="metrics", pred_store=store, config=cfg,
            run_tables_sha256=env["sha"], refs_from=env["refs"], same_view=True, **MKW)
    leg = env["root"] / "on"
    if not (leg / "scores.json").is_file():
        _legacy(env, leg, same_view=True)
    a, b = (json.loads((d / "scores.json").read_text())[S.CORRECTION_ONLY_KEY] for d in (out, leg))
    assert a == b
    assert _same_npz(out / "same_view_per_region.npz", leg / "same_view_per_region.npz")
    store0 = env["root"] / "store_plain"
    P.predict(base, P.requests(base), store0, B)
    with pytest.raises(S.ScoreRefused, match="requests_sha256"):
        S.score(env["evalset"], env["root"] / "metrics_bad", phase="metrics", pred_store=store0,
                config=cfg, run_tables_sha256=env["sha"], refs_from=env["refs"], same_view=True,
                **MKW)


# ------------------------------------------------------------------------------------------ review fixes

def test_pair_type_below_the_denominator_tolerance_is_left_out_and_named():
    from onion_eval.select_rule import CORRECTION_DENOM_REL_TOL as tol
    cor = [_rec("crop", 0.7, 1.0, 0.8, "t1"),
           _rec("tiny", 0.5, 1.0, 1.0 - tol / 2, "t2")]           # (d - f) / d = tol / 2
    same = [{"target": "t1", "crps_model_same_view": 0.6}, {"target": "t2", "crps_model_same_view": 0.1}]
    b = S.correction_only_block(cor, same)
    assert set(b["pair_types_excluded"]) == {"tiny"} and b["pair_types_in_macro"] == ["crop"]
    assert b["value"] == pytest.approx(1 - (0.7 - 0.6) / (1.0 - 0.8))
    assert b["per_pair_type"]["tiny"]["below_tolerance"] is True
    only = S.correction_only_block(cor[1:], same[1:])
    assert only["value"] is None and "below the denominator tolerance" in only["reason"]


def _cli(args, extra_env=None):
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "tools")]),
             OMP_NUM_THREADS="2", OMP_WAIT_POLICY="PASSIVE")
    e.pop(S.SAME_VIEW_ENV, None)
    e.update(extra_env or {})
    return subprocess.run([sys.executable, str(REPO / "tools/onion_eval/score.py"), *args], env=e,
                          capture_output=True, text=True, timeout=1200)


def test_parallel_same_view_equals_serial(env):
    """Forked workers (fresh processes, as score.sh runs them) give the serial same-view result."""
    outs = {}
    for w in (1, 2):
        d = env["root"] / f"par{w}"
        p = _cli(["--baseline", "position_average", "--evalset", str(env["evalset"]), "--out", str(d),
                  "--n-boot", "5", "--guardrail-windows", "4", "--workers", str(w),
                  "--score-same-view"])
        assert p.returncode == 0, p.stderr[-2000:]
        outs[w] = d
    a, b = (json.loads((outs[w] / "scores.json").read_text()) for w in (1, 2))
    assert a[S.CORRECTION_ONLY_KEY] == b[S.CORRECTION_ONLY_KEY]
    assert a[S.CORRECTION_ONLY_KEY]["n_same_view_targets"] > 0
    for k in VOLATILE:
        a.pop(k, None), b.pop(k, None)
    assert a == b
    assert _same_npz(outs[1] / "same_view_per_region.npz", outs[2] / "same_view_per_region.npz")


# ------------------------------------------------------------------------------------------ submit_arms

FAKE_SBATCH = """#!/bin/bash
n=$(( $(wc -l < "$FAKE_LOG" 2>/dev/null || echo 0) + 1000 ))
echo "$n SV=${ONION_SCORE_SAME_VIEW:-unset} $*" >> "$FAKE_LOG"
echo "$n"
"""
FAKE_SQUEUE = "#!/bin/bash\nexit 0\n"
SUBMIT = REPO / "slurm/minicandi/submit_arms.sh"


@pytest.fixture()
def sub_env(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    for d in ("tables", "refs"):
        (tmp_path / d).mkdir()
    (tmp_path / "evalset.json").write_text("{}")
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SCORE_", "ONION_", "PRED_"))}
    e.update(PATH=f"{b}:{e['PATH']}", PYTHON=sys.executable, CLUSTER="nibi", USER="tester",
             FAKE_LOG=str(tmp_path / "sbatch.log"), PRED_STORE_ROOT=str(tmp_path / "ps"),
             SEEDS="0", SCORE_PHASES="1", REFS_FROM=str(tmp_path / "refs"), SCORE_ONLY="1")
    run = tmp_path / "runs" / "rung1_base" / "seed0"
    run.mkdir(parents=True)
    (run / "DONE").write_text("")
    return tmp_path, e


def _submit(tmp, e, **over):
    p = subprocess.run(["bash", str(SUBMIT), str(REPO), str(tmp / "tables"), str(tmp / "evalset.json"),
                        str(tmp / "runs"), "1", "base"], env={**e, **over}, capture_output=True,
                       text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_submit_dry_run_names_the_switch_on_predict_and_metrics_only_when_set(sub_env):
    tmp, e = sub_env
    out = _submit(tmp, e, DRY_RUN="1", SCORE_SAME_VIEW="1")
    pred = [ln for ln in out.splitlines() if "predict.sh" in ln]
    met = [ln for ln in out.splitlines() if "ONION_SCORE_PHASE=metrics" in ln]
    assert pred and met and all("ONION_SCORE_SAME_VIEW=1" in ln for ln in pred + met)
    assert all("/rung1_base/same_view/seed0" in ln for ln in pred + met)
    out = _submit(tmp, e, DRY_RUN="1", ONION_SCORE_SAME_VIEW="1")        # unset; shell has a value
    lines = [ln for ln in out.splitlines() if "predict.sh" in ln or "metrics" in ln]
    assert lines and not any("SAME_VIEW" in ln or "same_view" in ln for ln in lines)


def test_submit_never_leaks_the_shell_value_into_jobs(sub_env):
    tmp, e = sub_env
    _submit(tmp, e, ONION_SCORE_SAME_VIEW="1")                          # SCORE_SAME_VIEW unset
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log and all(" SV=unset " in ln for ln in log)
    (tmp / "sbatch.log").unlink()
    _submit(tmp, e, SCORE_SAME_VIEW="1", PRED_STORE_ROOT=str(tmp / "ps2"))
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log and all(" SV=1 " in ln for ln in log)


def _complete_store(d: Path):
    d.mkdir(parents=True)
    (d / "STORE.json").write_text(json.dumps({"format": 1, "n_shards": 1}))
    (d / "PREDICTED_0of1.json").write_text("{}")


def test_same_view_store_is_apart_from_the_plain_store(sub_env):
    """The path alone decides: a complete plain store is never used for a same-view scoring, and a
    complete store under the same_view path is reused."""
    tmp, e = sub_env
    _complete_store(tmp / "ps" / "runs" / "rung1_base" / "seed0")
    out = _submit(tmp, e, DRY_RUN="1")
    assert "is complete; no predict job" in out and "predict.sh" not in out
    out = _submit(tmp, e, DRY_RUN="1", SCORE_SAME_VIEW="1")
    assert "predict.sh" in out and "/rung1_base/same_view/seed0" in out
    _complete_store(tmp / "ps" / "runs" / "rung1_base" / "same_view" / "seed0")
    out = _submit(tmp, e, DRY_RUN="1", SCORE_SAME_VIEW="1")
    assert "is complete; no predict job" in out and "predict.sh" not in out
    assert "predict_store.py" not in SUBMIT.read_text().replace("score.py and predict_store.py", "")


def test_checkpoint_of_another_tables_version_is_refused(env):
    """Close-out D1 finding 6: a run trained on one tables version (its tables_sha256) is never
    scored on validation against an eval set of another version."""
    m, cfg = _model()
    with pytest.raises(S.ScoreRefused, match="trained on tables sha256"):
        S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256="0" * 64,
                 **KW)
