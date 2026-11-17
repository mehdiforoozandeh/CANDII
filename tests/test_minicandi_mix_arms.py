"""Close-out item A13, mix-and-match combinations (tools/minicandi/mix_arms.py,
tools/minicandi/configs/ablations/mix, and the native-depth scoring of score.py).

(a) The configs on disk equal the generator; each switch equals the key set of its single-ablation
    config; each arm is the base recipe plus the union of its switches' keys; the design is the
    full factorial (conv autoencoder, U-Net) or the resolution-V half fraction I = +ABCDE
    (conv + attention).
(b) Every arm builds through the registry at full size within 10% of 1.0e6 parameters and, at a
    small width on synthetic tables, trains 3 steps with a finite loss, reloads and passes the
    wiring check with a perturbed head: every combined switch set can be on together.
(c) ONION_SCORE_INPUT_DEPTH=native: a run trained with input_depth = fixed is scored from
    native-depth inputs; the switch is refused where it means nothing; off leaves the scores equal;
    the store and output-directory guards; submit_arms.sh SCORE_INPUT_DEPTH.
"""
from __future__ import annotations

import itertools
import json
import math
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

from minicandi import contract as C  # noqa: E402
from minicandi import mix_arms as M  # noqa: E402
from minicandi import registry as R  # noqa: E402

ABL = REPO / "tools/minicandi/configs/ablations"
FINAL = REPO / "tools/minicandi/configs/final"
ARMS = [(r, sw) for r in M.POOL for sw in M.design(r)]
IDS = [f"rung{r}-{M.arm_name(sw)}" for r, sw in ARMS]
L, BS, STEPS = 64, 4, 3
SMALL = {2: 1e5, 3: 2e5, 4: 2e5}


def _ref(r):
    return json.loads((FINAL / f"rung{r}" / "reference_equal.json").read_text())


# ------------------------------------------------------------------------------- (a) configs

def test_configs_on_disk_equal_the_generator():
    assert M.main(["check"]) == 0


def test_each_switch_equals_its_single_ablation():
    for name, (_, src, keys) in M.SWITCHES.items():
        ab, arm = src.split("/")
        rungs = [r for r in M.POOL if name in M.POOL[r]]
        assert rungs, name
        for r in rungs:
            single = json.loads((ABL / ab / f"rung{r}" / f"{arm}.json").read_text())
            ref = _ref(r)
            diff = {k: single[k] for k in set(single) | set(ref)
                    if single.get(k) != ref.get(k) and k != "simpler_than_base"}
            if "param_budget" in diff and float(diff["param_budget"]) == float(ref["param_budget"]):
                diff.pop("param_budget")                         # 1000000 vs 1000000.0
            assert diff == keys, (name, r, diff)


@pytest.mark.parametrize("r,sw", ARMS, ids=IDS)
def test_arm_is_reference_plus_its_switches(r, sw):
    cfg = json.loads((ABL / "mix" / f"rung{r}" / f"{M.arm_name(sw)}.json").read_text())
    ref = _ref(r)
    want = dict(ref)
    for s in sw:
        want.update(M.SWITCHES[s][2])
    want["simpler_than_base"] = False
    assert cfg == want


def test_design():
    assert [len(M.design(r)) for r in (2, 3, 4)] == [1, 16, 7]
    assert M.design(2) == [("same_view_only",)]
    assert set(M.design(4)) == {s for k in (1, 2, 3)
                                for s in itertools.combinations(M.POOL[4], k)}
    d3 = M.design(3)
    assert all(len(s) % 2 == 1 for s in d3)                      # I = +ABCDE: 1, 3 or 5 on
    assert {(s,) for s in M.POOL[3]} <= set(d3) and tuple(M.POOL[3]) in d3
    # resolution V: intercept, 5 main effects and 10 two-switch interactions are orthogonal
    X = np.array([[1 if s in arm else -1 for s in M.POOL[3]] for arm in d3], float)
    cols = [np.ones(len(d3))] + [X[:, i] for i in range(5)] + \
           [X[:, i] * X[:, j] for i, j in itertools.combinations(range(5), 2)]
    D = np.stack(cols, 1)
    assert np.allclose(D.T @ D, 16 * np.eye(16))


HEAD = ["arm", "n_seeds"] + [f"{k}.{x}" for _, k in M.TERMS for x in ("mean", "range")]


def _summary(tmp_path, rows):
    p = tmp_path / "summary.tsv"
    lines = ["\t".join(HEAD)] + ["\t".join(str(v) for v in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")
    return p


def _row(arm, imp, cor, cal, rng=0.01, n=3):
    return [arm, n, imp, rng, cor, rng, cal, rng]


def test_qualification_gate(tmp_path, capsys):
    """The gate re-applies the rule per switch: up beyond on one term and down beyond on none;
    a switch without a 3-seed row waits; U-Net fixed depth is in by PI ruling."""
    rows = [_row(f"rung{r}_base_heldout", 0.40, 0.60, 0.60) for r in (2, 3, 4)]
    rows += [_row(M.source_run(2, "same_view_only"), 0.45, 0.60, 0.60),          # qualifies
             _row(M.source_run(3, "concat"), 0.40, 0.605, 0.60),                 # within: fails
             _row(M.source_run(3, "fixed_depth"), 0.45, 0.50, 0.60),             # up and down
             _row(M.source_run(3, "impute_x4"), 0.45, 0.60, 0.60, n=2),          # 2 seeds
             _row(M.source_run(4, "fixed_depth"), 0.30, 0.80, 0.60)]             # PI ruling
    st = {(r, x["switch"]): x["status"] for r in M.POOL
          for x in M.qualify(M.read_summary(_summary(tmp_path, rows)), r)}
    assert st[(2, "same_view_only")] == "qualifies"
    assert st[(3, "concat")] == "fails" and st[(3, "fixed_depth")] == "fails"
    assert st[(3, "impute_x4")] == "no_score" and st[(3, "nb_crps")] == "no_score"
    assert st[(4, "fixed_depth")] == "in_by_pi_ruling" and st[(4, "pearson")] == "no_score"
    assert M.main(["qualify", "--summary", str(_summary(tmp_path, rows))]) == 3   # WAIT
    full = rows[:4] + [_row(M.source_run(r, s), 0.45, 0.60, 0.60)
                       for r in M.POOL for s in M.POOL[r] if (r, s) != (2, "same_view_only")]
    assert M.main(["qualify", "--summary", str(_summary(tmp_path, full))]) == 0    # PASS
    bad = full + [_row(M.source_run(3, "concat"), 0.40, 0.60, 0.60)]              # last row wins
    assert M.main(["qualify", "--summary", str(_summary(tmp_path, bad))]) == 1     # FAIL
    capsys.readouterr()


def test_gate_reproduces_the_16_region_pool():
    """On the 16-region table of the ablation read, every switch of the pool qualifies (the
    provisional design); the table is in the foreman's worktree, so skip when absent."""
    p = Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/"
             "cruxvault/results/ablations_closeout/arm_summary_p16_sv_2026-11-17.tsv")
    if not p.is_file():
        pytest.skip("16-region summary not on this machine")
    summ = M.read_summary(p)
    st = {(r, x["switch"]): x["status"] for r in M.POOL for x in M.qualify(summ, r)}
    assert set(st.values()) == {"qualifies", "in_by_pi_ruling"}
    assert st[(4, "fixed_depth")] == "in_by_pi_ruling"


# ------------------------------------------------------------------------------- (b) build, train

@pytest.mark.parametrize("r,sw", ARMS, ids=IDS)
def test_full_size_parameter_count(r, sw):
    cfg = json.loads((ABL / "mix" / f"rung{r}" / f"{M.arm_name(sw)}.json").read_text())
    torch.manual_seed(0)
    n = sum(p.numel() for p in R.build_model(cfg).parameters() if p.requires_grad)
    assert abs(n - 1.0e6) <= 1.0e5, n


@pytest.fixture(scope="module")
def synth_env(tmp_path_factory):
    from minicandi import synth
    root = tmp_path_factory.mktemp("mix")
    return {"root": root, "tables": synth.make_synthetic_tables(root)}


def _perturb(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model.eval()


@pytest.mark.parametrize("r,sw", ARMS, ids=IDS)
def test_arm_trains_reloads_and_passes_the_wiring_check(synth_env, r, sw):
    from minicandi import train as T
    from minicandi.dataset import OnionDataset
    cfg = json.loads((ABL / "mix" / f"rung{r}" / f"{M.arm_name(sw)}.json").read_text())
    cfg.pop("average_h5", None)            # small tables: the average reads the cells directly
    cfg.pop("heldout_loss", None)          # a monitor that needs an eval set; no switch reads it
    cfg.update(context_bins=L, batch_size=BS, windows_seen=STEPS * BS, param_budget=SMALL[r])
    run = synth_env["root"] / "runs" / f"rung{r}_{M.arm_name(sw)}"
    T.train(cfg, synth_env["tables"], 0, run, log_every=1)
    losses = [json.loads(x)["loss"] for x in (run / "log.jsonl").read_text().splitlines()]
    assert len(losses) == STEPS and all(math.isfinite(x) for x in losses), losses
    model, saved = R.load_run(run)
    for s in sw:
        for k, v in M.SWITCHES[s][2].items():
            if k != "average_h5":
                assert saved[k] == v, (s, k)
    b = OnionDataset(synth_env["tables"], "train", saved, 1).batch(BS, np.random.default_rng(1))
    if "average_input" in sw:
        assert "avg" in b
    w = C.wiring_check(_perturb(model), b)
    assert w["pass"], (r, sw, w)


# ------------------------------------------------------------------------------- (c) native depth

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (7, "chipseq-control"))
RB, B = 256, 4
KW = dict(n_boot=5, guardrail_windows=4, log=lambda *a: None)
MKW = {k: v for k, v in KW.items() if k != "guardrail_windows"}


@pytest.fixture(scope="module")
def ev_env(tmp_path_factory):
    from minicandi.dataset import tables_sha256
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    from onion_eval import score as S
    root = tmp_path_factory.mktemp("mixnd")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = [p for p in ev["correction_pairs"]
                              if p["never_trained"] and not p["subset_pair"]][:4]
    ev["replicate_pairs"] = []
    out.write_text(json.dumps(ev))
    refs = root / "refs"
    for name in S.REFS_FROM_DIRS:
        S.score(out, refs / S.REFS_FROM_DIRS[name], baseline=name, **KW)
    return {"root": root, "tables": t, "evalset": out, "refs": refs, "sha": tables_sha256(t)}


@pytest.fixture()
def low_cap(monkeypatch):
    """A depth cap below every synthetic depth, so `input_depth = fixed` thins every input."""
    from minicandi import dataset as D
    for k in list(D.FIXED_BASE_LOG2_DEPTH):
        monkeypatch.setitem(D.FIXED_BASE_LOG2_DEPTH, k, 12.0)


def _fixed_model(fixed=True):
    cfg = {**_ref(2), "context_bins": L, "batch_size": B, "param_budget": 1e5}
    if fixed:
        cfg["input_depth"] = "fixed"
    torch.manual_seed(2)
    m = R.build_model(cfg)
    return _perturb(m, 2), cfg


def _score(ev_env, out, model, cfg, **kw):
    from onion_eval import score as S
    return S.score(ev_env["evalset"], out, model=model, config=cfg,
                   run_tables_sha256=ev_env["sha"], **KW, **kw)


def test_native_depth_reads_unthinned_inputs(ev_env, low_cap):
    from onion_eval import score as S
    m, cfg = _fixed_model()
    fx = S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                  run_tables_sha256=ev_env["sha"], **KW)
    nd = S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                  run_tables_sha256=ev_env["sha"], input_depth_at_scoring="native", **KW)
    assert fx.ds.config["input_depth"] == "fixed" and nd.ds.config["input_depth"] == "native"
    assert cfg["input_depth"] == "fixed"                         # the caller's config is not changed
    imp = fx.jobs()[0][0]
    c, s0 = fx.regions[0]
    wf = fx.ds.window(c, s0, imp["inputs"], [imp["target"]])
    wn = nd.ds.window(c, s0, imp["inputs"], [imp["target"]])
    t = [C.TRACK_INDEX[fx.ds.rows[x]["assay"]] for x in imp["inputs"]]
    assert torch.all(wf["c_in"][t, C.LOG2_DEPTH_COL] == 12.0)
    assert torch.all(wn["c_in"][t, C.LOG2_DEPTH_COL] > 12.0)
    assert wn["x_counts"][:, t].sum() > wf["x_counts"][:, t].sum()
    assert torch.equal(wf["y_counts"], wn["y_counts"]) and torch.equal(wf["c_out"], wn["c_out"])


def test_native_depth_is_refused_where_it_means_nothing(ev_env):
    from onion_eval import score as S
    m, cfg = _fixed_model(fixed=False)
    with pytest.raises(ValueError, match="input_depth = fixed"):
        S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                 run_tables_sha256=ev_env["sha"], input_depth_at_scoring="native", **KW)
    with pytest.raises(ValueError, match="model"):
        S.Scorer(ev_env["evalset"], main="position_average", input_depth_at_scoring="native", **KW)
    m, cfg = _fixed_model()
    with pytest.raises(ValueError, match="not in"):
        S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                 run_tables_sha256=ev_env["sha"], input_depth_at_scoring="deep", **KW)


def test_foreign_swap_depth_guard_reads_the_runs_own_config(ev_env):
    """The guard against swap_depth on side C of a fixed-depth run tests the run's config, which a
    native-depth scoring does not change (it changes only the scorer's dataset config)."""
    from onion_eval import score as S
    m, cfg = _fixed_model()
    for kw in ({}, {"input_depth_at_scoring": "native"}):
        with pytest.raises(ValueError, match="swap_depth"):
            S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                     run_tables_sha256=ev_env["sha"], foreign_covariates="in_swap_depth", **kw, **KW)


def test_env_parse(monkeypatch):
    from onion_eval import score as S
    monkeypatch.delenv(S.INPUT_DEPTH_ENV, raising=False)
    assert S.input_depth_kwargs() == {}
    monkeypatch.setenv(S.INPUT_DEPTH_ENV, "")
    assert S.input_depth_kwargs() == {}
    monkeypatch.setenv(S.INPUT_DEPTH_ENV, "native")
    assert S.input_depth_kwargs() == {"input_depth_at_scoring": "native"}
    monkeypatch.setenv(S.INPUT_DEPTH_ENV, "fixed")
    with pytest.raises(ValueError):
        S.input_depth_kwargs()


def test_end_to_end_legacy_and_store(ev_env, low_cap):
    from onion_eval import predict_store as P
    from onion_eval import score as S
    from onion_eval import select_rule as SR
    root = ev_env["root"]
    m, cfg = _fixed_model()
    fx = _score(ev_env, root / "e2e" / "p256_sv", m, cfg)
    nd = _score(ev_env, root / "e2e" / "p256_sv_native_depth", m, cfg,
                input_depth_at_scoring="native")
    assert S.INPUT_DEPTH_KEY not in fx and S.INPUT_DEPTH_KEY not in fx["scorer_settings"]
    assert nd[S.INPUT_DEPTH_KEY] == "native"
    assert nd["scorer_settings"][S.INPUT_DEPTH_KEY] == "native"
    assert nd["count_crps"] != fx["count_crps"]                # the inputs differ
    # a comparability key: never merged silently; the deployment comparison needs --allow-mix
    a, k = SR.scorer_keys(fx), SR.scorer_keys(nd)
    assert "input_depth_at_scoring" not in a and k["input_depth_at_scoring"] == "native"
    with pytest.raises(SR.Refused, match="input_depth_at_scoring"):
        SR.check_scorer_keys([("fx", a), ("nd", k)])
    SR.check_scorer_keys([("fx", a), ("nd", k)], allow=("input_depth_at_scoring",))
    # the output-directory guard, both ways
    with pytest.raises(S.ScoreRefused, match="native_depth"):
        _score(ev_env, root / "e2e" / "p256_other", m, cfg, input_depth_at_scoring="native")
    with pytest.raises(S.ScoreRefused, match="normal score"):
        _score(ev_env, root / "e2e" / "p256_sv_native_depth", m, cfg)
    with pytest.raises(S.ScoreRefused, match="takes only native-depth"):
        _score(ev_env, root / "e2e" / "p256_sv_native_depth_fresh", m, cfg)
    # the store path equals the legacy path and guards the switch both ways
    sc = S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                  run_tables_sha256=ev_env["sha"], input_depth_at_scoring="native", **KW)
    store = root / "store_nd"
    P.predict(sc, P.requests(sc), store, B)
    assert json.loads((store / P.STORE_FILE).read_text())[S.INPUT_DEPTH_KEY] == "native"
    s = S.score(ev_env["evalset"], root / "st" / "p256_native_depth", phase="metrics",
                pred_store=store, config=cfg, run_tables_sha256=ev_env["sha"],
                refs_from=ev_env["refs"], input_depth_at_scoring="native", **MKW)
    for k in ("count_crps", "pit_deviation_pooled", "correction_crps_model"):
        assert s[k] == nd[k], k
    with pytest.raises(S.ScoreRefused, match="input depth at scoring"):
        S.score(ev_env["evalset"], root / "st" / "p256_bad", phase="metrics", pred_store=store,
                config=cfg, run_tables_sha256=ev_env["sha"], refs_from=ev_env["refs"], **MKW)
    base = S.Scorer(ev_env["evalset"], main="model", model=m, config=cfg,
                    run_tables_sha256=ev_env["sha"], **KW)
    store0 = root / "store_plain"
    P.predict(base, P.requests(base), store0, B)
    assert S.INPUT_DEPTH_KEY not in json.loads((store0 / P.STORE_FILE).read_text())
    with pytest.raises(S.ScoreRefused, match="input depth at scoring"):
        S.score(ev_env["evalset"], root / "st" / "p256_native_depth_bad", phase="metrics",
                pred_store=store0, config=cfg, run_tables_sha256=ev_env["sha"],
                refs_from=ev_env["refs"], input_depth_at_scoring="native", **MKW)


# ------------------------------------------------------------------------------- (c) submit

FAKE_SBATCH = """#!/bin/bash
n=$(( $(wc -l < "$FAKE_LOG" 2>/dev/null || echo 0) + 1000 ))
echo "$n ND=${ONION_SCORE_INPUT_DEPTH:-unset} $*" >> "$FAKE_LOG"
echo "$n"
"""


@pytest.fixture()
def sub_env(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    for name, text in (("sbatch", FAKE_SBATCH), ("squeue", "#!/bin/bash\nexit 0\n")):
        (b / name).write_text(text)
        (b / name).chmod(0o755)
    for d in ("tables", "refs"):
        (tmp_path / d).mkdir()
    (tmp_path / "evalset.json").write_text("{}")
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SCORE_", "ONION_", "PRED_", "FOREIGN_"))}
    e.update(PATH=f"{b}:{e['PATH']}", PYTHON=sys.executable, CLUSTER="nibi", USER="tester",
             FAKE_LOG=str(tmp_path / "sbatch.log"), PRED_STORE_ROOT=str(tmp_path / "ps"),
             SEEDS="0", SCORE_PHASES="1", REFS_FROM=str(tmp_path / "refs"))
    run = tmp_path / "runs" / "rung1_base" / "seed0"
    run.mkdir(parents=True)
    (run / "DONE").write_text("")
    return tmp_path, e


def _submit(tmp, e, **over):
    return subprocess.run(["bash", str(REPO / "slurm/minicandi/submit_arms.sh"), str(REPO),
                           str(tmp / "tables"), str(tmp / "evalset.json"), str(tmp / "runs"), "1",
                           "base"], env={**e, **over}, capture_output=True, text=True)


def test_submit_native_depth(sub_env):
    tmp, e = sub_env
    p = _submit(tmp, e, DRY_RUN="1", SCORE_INPUT_DEPTH="native", SCORE_SUBDIR="p256_sv")
    assert p.returncode == 2 and "native_depth" in p.stderr
    p = _submit(tmp, e, DRY_RUN="1", SCORE_INPUT_DEPTH="deep", SCORE_SUBDIR="p256_native_depth")
    assert p.returncode == 2
    p = _submit(tmp, e, DRY_RUN="1", SCORE_INPUT_DEPTH="native", SCORE_PHASES="0",
                SCORE_SUBDIR="p256_native_depth")
    assert p.returncode == 2
    p = _submit(tmp, e, DRY_RUN="1", SCORE_INPUT_DEPTH="native", SCORE_SUBDIR="p256_sv_native_depth",
                SCORE_SAME_VIEW="1")
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("ONION_") and "sbatch" in ln]
    assert lines and all("ONION_SCORE_INPUT_DEPTH=native" in ln for ln in lines)
    assert all("--job-name=mc_rung1_base_s0_nd" in ln for ln in lines)
    st = [ln for ln in lines if "ONION_PRED_STORE" in ln or "predict.sh" in ln]
    assert st and all("/rung1_base/same_view/native_depth/seed0/p256_sv_native_depth" in ln
                      for ln in st)
    assert not any("train.sh" in ln for ln in p.stdout.splitlines())
    # a real submission exports the switch; a shell value never leaks without SCORE_INPUT_DEPTH
    p = _submit(tmp, e, SCORE_INPUT_DEPTH="native", SCORE_SUBDIR="p256_native_depth")
    assert p.returncode == 0, p.stderr
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log and all(" ND=native " in ln for ln in log)
    (tmp / "sbatch.log").unlink()
    p = _submit(tmp, e, ONION_SCORE_INPUT_DEPTH="native", SCORE_SUBDIR="p256_x",
                PRED_STORE_ROOT=str(tmp / "ps2"))
    assert p.returncode == 0, p.stderr
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log and all(" ND=unset " in ln for ln in log)


def test_submit_native_depth_waits_on_a_live_training(sub_env):
    """The native-depth jobs carry their own name; their predict jobs wait on the run's training
    job, found by the run's own job name in squeue."""
    tmp, e = sub_env
    (tmp / "runs" / "rung1_base" / "seed0" / "DONE").unlink()
    (tmp / "bin" / "squeue").write_text(
        "#!/bin/bash\necho '4242 mc_rung1_base_s0 /k/slurm/minicandi/train.sh a b'\n")
    p = _submit(tmp, e, DRY_RUN="1", SCORE_INPUT_DEPTH="native", SCORE_SUBDIR="p256_native_depth")
    assert p.returncode == 0, p.stderr
    pred = [ln for ln in p.stdout.splitlines() if "predict.sh" in ln]
    assert pred and all("--dependency=afterok:4242" in ln for ln in pred)
