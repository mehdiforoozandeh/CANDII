"""Close-out item E5: the scorer extensions, on synthetic tables.

1. Foreign covariates at scoring (`score.py --foreign-covariates`): the donor draw (same assay,
   another cell, base view, seeded, logged), the model batch (side C / C', depth kept or swapped,
   the batch itself when off), end to end through the legacy path and the prediction store, the
   comparability key, the store and output-directory guards, and submit_arms.sh FOREIGN_COVARIATES.
2. Artifact variance: mixed_model.py `--extra-factors` (filled, partly empty, absent columns) and
   report_extras.py by-covariate and by-scale.
3. Every stratum's PIT deviation (`pit_deviation_per_stratum`), its worst-stratum maximum, and the
   report-time value from suffstats.npz.
4. Headline 7's robust aggregates (median and ratio of means) beside the unchanged mean.
And `selection.composite_status`, the additive label of the superseded composite."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.contract import MiniCandi  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import report_extras as RX  # noqa: E402
from onion_eval import score as S  # noqa: E402
from onion_eval import select_rule as SR  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (3, "H3K27ac"), (7, "chipseq-control"))
L, RB, B = 64, 256, 4
KW = dict(n_boot=5, guardrail_windows=4, log=lambda *a: None)
MKW = {k: v for k, v in KW.items() if k != "guardrail_windows"}
VOLATILE = ("wall_seconds", "timing_seconds", "torch_threads")
NEW_KEYS = (S.PIT_STRATA_KEY, S.H7_ROBUST_KEY)


class Trivial(MiniCandi):
    """mu = a constant rate times the target depth from C' column 0; reads nothing else."""

    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.full((36,), 3.0), requires_grad=False)

    def forward(self, b):
        Lw = b["y_counts"].shape[1]
        mu = (2.0 ** b["c_out"][..., 0] / 2 ** 24 * self.rate).clamp_min(1e-3)
        mu = mu[:, None, :].expand(-1, Lw, -1).contiguous()
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import tables_sha256
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    root = tmp_path_factory.mktemp("e5")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    sp = root / "strata.h5"
    sizes = json.loads((root / "genome" / "chrom_sizes.json").read_text())
    with h5py.File(sp, "w") as f:
        rng = np.random.default_rng(1)
        hi, lo = f.create_group("high"), f.create_group("low")
        for c, bp in sizes.items():
            m = rng.random(bp // 25) < 0.3
            hi.create_dataset(c, data=m)
            lo.create_dataset(c, data=~m)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, strata=sp, n_regions=2, n_fit_regions=1, region_bins=RB, seed=0)
    ev["correction_pairs"] = [p for p in ev["correction_pairs"]
                              if p["never_trained"] and not p["subset_pair"]][:4]
    ev["replicate_pairs"] = []
    assert ev["correction_pairs"] and ev["strata"] == ["high", "low"]
    out.write_text(json.dumps(ev))
    refs = root / "refs"
    for name in S.REFS_FROM_DIRS:
        S.score(out, refs / S.REFS_FROM_DIRS[name], baseline=name, **KW)
    return {"root": root, "tables": t, "evalset": out, "refs": refs, "sha": tables_sha256(t),
            "ev": ev}


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


def _score(env, out, model=None, cfg=None, **kw):
    if model is None:
        model, cfg = _model()
    return S.score(env["evalset"], out, model=model, config=cfg, run_tables_sha256=env["sha"],
                   **KW, **kw)


def _clean(s):
    s = dict(s)
    for k in VOLATILE:
        s.pop(k, None)
    return s


def _all_assays(m):
    return {**(m.get("non_scorable_per_assay") or {}), **(m.get("per_assay") or {})}


def _rows(tables):
    with open(Path(tables) / "measurements.tsv", newline="") as fh:
        return {r["measurement_id"]: r for r in csv.DictReader(fh, delimiter="\t")}


# ------------------------------------------------------------------------------------------ 1. donors

def test_foreign_modes_and_spec():
    assert set(S.FOREIGN_MODES) == {f"{s}_{d}" for s in ("in", "out", "both")
                                    for d in ("keep_depth", "swap_depth")}
    assert S.foreign_spec(None) is None
    assert S.foreign_spec("both_keep_depth", 3) == {"mode": "both_keep_depth", "sides": ["in", "out"],
                                                    "depth": "keep_depth", "seed": 3}
    with pytest.raises(ValueError):
        S.foreign_spec("out")


def test_donor_is_another_cell_same_assay_base_view_and_seeded(env):
    from minicandi.dataset import view_of
    rows = _rows(env["tables"])
    mids = sorted(rows)
    d0 = S.foreign_donors(rows, mids, 0)
    assert d0 == S.foreign_donors(rows, list(reversed(mids)), 0)          # order-free
    assert d0 == {m: S.foreign_donors(rows, [m], 0)[m] for m in mids[:20]} | {
        m: d0[m] for m in mids[20:]}                                     # per measurement
    for m, d in d0.items():
        assert d is not None
        assert rows[d]["assay"] == rows[m]["assay"] and rows[d]["cell"] != rows[m]["cell"]
        assert view_of(rows[d])[0] == "base" and rows[d]["in_eic"] in ("train", "val")
    d1 = S.foreign_donors(rows, mids, 1)
    assert d1 != d0                                                      # the seed matters
    one = {m: r for m, r in rows.items() if r["cell"] == rows[mids[0]]["cell"]}
    assert set(S.foreign_donors(one, list(one), 0).values()) == {None}  # no other cell


def _scorer(env, mode, model=None, cfg=None, seed=0):
    if model is None:
        model, cfg = _model()
    return S.Scorer(env["evalset"], main="model", model=model, config=cfg,
                    run_tables_sha256=env["sha"], foreign_covariates=mode, foreign_seed=seed, **KW)


def test_model_batch_off_is_the_batch_itself(env):
    sc = _scorer(env, None)
    imp = sc.jobs()[0][0]
    c, s0 = sc.regions[0]
    batch = sc.C.collate([sc.ds.window(c, s0, imp["inputs"], [imp["target"]])])
    assert sc._model_batch(batch, imp) is batch


@pytest.mark.parametrize("mode", S.FOREIGN_MODES)
def test_model_batch_rows(env, mode):
    sc = _scorer(env, mode)
    imp, cor = sc.jobs()[0][0], sc.jobs()[1][0]
    c, s0 = sc.regions[0]
    for job, pair in ((imp, False), (cor, True)):
        win = (sc._pair_window(job, c, s0) if pair
               else sc.ds.window(c, s0, job["inputs"], [job["target"]]))
        batch = sc.C.collate([win, win])
        before = {k: batch[k].clone() for k in ("c_in", "c_out")}
        mb = sc._model_batch(batch, job)
        assert all(torch.equal(batch[k], before[k]) for k in before)    # the input is not changed
        for k in batch:
            if k not in ("c_in", "c_out"):
                assert mb[k] is batch[k]
        scale = 0.5 if pair else 1.0
        side, depth = mode.split("_", 1)
        sides = S.FOREIGN_SIDES[side]
        ins = [job["source"]] if pair else job["inputs"]
        for key, sd, mids in (("c_in", "in", ins), ("c_out", "out", [job["target"]])):
            for m in mids:
                t = sc.C.TRACK_INDEX[sc.ds.rows[m]["assay"]]
                avail = batch["x_avail" if sd == "in" else "y_avail"][:, t]
                got, old = mb[key][:, t], before[key][:, t]
                if sd not in sides or not bool(avail.all()):
                    assert torch.equal(got, old)
                    continue
                d = sc.ds.rows[sc._donor(m)]
                want = torch.as_tensor(sc.ds.encoder.c_in(d, sc.ds.qc_on_input, scale) if sd == "in"
                                       else sc.ds.encoder.c_out(d, scale))
                assert torch.equal(got[:, 1:], want[1:].expand_as(got[:, 1:]))
                if depth == "keep_depth":
                    assert torch.equal(got[:, 0], old[:, 0])
                else:
                    assert torch.equal(got[:, 0], want[0].expand_as(got[:, 0]))
        # every other track's rows are untouched
        touched = {sc.C.TRACK_INDEX[sc.ds.rows[m]["assay"]] for m in [*ins, job["target"]]}
        for t in set(range(sc.C.A)) - touched:
            assert torch.equal(mb["c_in"][:, t], before["c_in"][:, t])
            assert torch.equal(mb["c_out"][:, t], before["c_out"][:, t])


def test_foreign_needs_a_model(env):
    with pytest.raises(ValueError, match="foreign"):
        S.Scorer(env["evalset"], main="position_average", foreign_covariates="out_keep_depth", **KW)


# ------------------------------------------------------------------------------------------ 1. end to end

def test_trivial_model_keep_depth_scores_as_normal_swap_depth_does_not(env):
    """Trivial reads only C' column 0: out_keep_depth leaves every prediction, so every score is the
    normal one; out_swap_depth moves the depth offset."""
    cfg = {"context_bins": L, "batch_size": B}
    root = env["root"]
    n = _score(env, root / "triv" / "p16", model=Trivial(), cfg=cfg)
    k = _score(env, root / "triv" / "p16_foreign_ok", model=Trivial(), cfg=cfg,
               foreign_covariates="out_keep_depth")
    w = _score(env, root / "triv" / "p16_foreign_os", model=Trivial(), cfg=cfg,
               foreign_covariates="out_swap_depth")
    fk = k.pop(S.FOREIGN_KEY)
    assert k["scorer_settings"].pop(S.FOREIGN_KEY) == "out_keep_depth"
    assert k["scorer_settings"].pop("foreign_seed") == 0
    assert _clean(k) == _clean(n)
    assert fk["mode"] == "out_keep_depth" and fk["n_without_donor"] == 0 and fk["n_measurements"] > 0
    assert all("log2_depth" not in e["columns_changed"] for e in fk["assignment"].values())
    assert w[S.FOREIGN_KEY]["n_changed_by_column"].get("log2_depth", 0) > 0
    assert _all_assays(w["count_crps"]) != _all_assays(n["count_crps"])
    # the guardrail keeps the true covariates
    assert w["copy_leak"] == n["copy_leak"]


def test_real_model_both_modes_change_predictions_and_keys_separate(env):
    root = env["root"]
    n = _score(env, root / "real" / "seed0" / "p16")
    f = _score(env, root / "real" / "seed0" / "p16_foreign_bk", foreign_covariates="both_keep_depth")
    assert f[S.FOREIGN_KEY]["n_changed_by_column"].get("lab_id", 0) > 0
    assert _all_assays(f["count_crps"]) != _all_assays(n["count_crps"])
    a, b = SR.scorer_keys(n), SR.scorer_keys(f)
    assert "foreign_covariates" not in a and b["foreign_covariates"] == "both_keep_depth:seed0"
    with pytest.raises(SR.Refused, match="foreign_covariates"):
        SR.check_scorer_keys([("n", a), ("f", b)])
    with pytest.raises(SR.Refused):
        SR.check_scorer_keys([("n", a), ("f", b)], allow=("foreign_covariates",))


def test_output_directory_guard(env):
    with pytest.raises(S.ScoreRefused, match="foreign"):
        _score(env, env["root"] / "guard" / "p16_sv", foreign_covariates="in_keep_depth")
    d = env["root"] / "guard" / "p16_foreign_x"
    d.mkdir(parents=True)
    (d / "scores.json").write_text(json.dumps({S.FOREIGN_KEY: {"mode": "out_keep_depth"}}))
    with pytest.raises(S.ScoreRefused, match="out_keep_depth"):
        _score(env, d, foreign_covariates="in_keep_depth")


def test_store_path_equals_legacy_and_guards_the_mode(env):
    from onion_eval import predict_store as P
    m, cfg = _model()
    root = env["root"]
    sc = _scorer(env, "both_swap_depth", m, cfg)
    reqs = P.requests(sc)
    store = root / "store_fc"
    P.predict(sc, reqs, store, B)
    head = json.loads((store / P.STORE_FILE).read_text())
    assert head[S.FOREIGN_KEY] == {"mode": "both_swap_depth", "seed": 0}
    out = root / "st" / "p16_foreign_bs"
    s = S.score(env["evalset"], out, phase="metrics", pred_store=store, config=cfg,
                run_tables_sha256=env["sha"], refs_from=env["refs"],
                foreign_covariates="both_swap_depth", **MKW)
    leg = _score(env, root / "st" / "legacy_foreign_bs", model=m, cfg=cfg,
                 foreign_covariates="both_swap_depth")
    for k in ("count_crps", "pit_deviation_pooled", "correction_crps_model", S.H7_ROBUST_KEY):
        assert s[k] == leg[k], k
    assert s[S.FOREIGN_KEY] == leg[S.FOREIGN_KEY]
    # a normal metric phase refuses the foreign store; a foreign one refuses a normal store
    with pytest.raises(S.ScoreRefused, match="foreign covariates"):
        S.score(env["evalset"], root / "st" / "p16_bad", phase="metrics", pred_store=store,
                config=cfg, run_tables_sha256=env["sha"], refs_from=env["refs"], **MKW)
    base = _scorer(env, None, m, cfg)
    store0 = root / "store_plain"
    P.predict(base, P.requests(base), store0, B)
    with pytest.raises(S.ScoreRefused, match="foreign covariates"):
        S.score(env["evalset"], root / "st" / "p16_foreign_bad", phase="metrics", pred_store=store0,
                config=cfg, run_tables_sha256=env["sha"], refs_from=env["refs"],
                foreign_covariates="both_swap_depth", **MKW)
    # a foreign predict refuses to write into the normal store (STORE.json differs)
    with pytest.raises(S.ScoreRefused, match="another scoring"):
        P.predict(sc, reqs, store0, B)


def test_cli_and_env(env, monkeypatch, capsys):
    ap = __import__("argparse").ArgumentParser()
    S.add_foreign_arguments(ap)
    monkeypatch.delenv(S.FOREIGN_ENV, raising=False)
    monkeypatch.delenv(S.FOREIGN_SEED_ENV, raising=False)
    assert S.foreign_kwargs(ap.parse_args([])) == {}
    assert S.foreign_kwargs(ap.parse_args(["--foreign-covariates", "in_swap_depth", "--foreign-seed",
                                           "4"])) == {"foreign_covariates": "in_swap_depth",
                                                      "foreign_seed": 4}
    monkeypatch.setenv(S.FOREIGN_ENV, "out_keep_depth")
    monkeypatch.setenv(S.FOREIGN_SEED_ENV, "2")
    assert S.foreign_kwargs(ap.parse_args([])) == {"foreign_covariates": "out_keep_depth",
                                                   "foreign_seed": 2}
    monkeypatch.delenv(S.FOREIGN_ENV)
    with pytest.raises(SystemExit):
        S.foreign_kwargs(ap.parse_args(["--foreign-seed", "1"]))


# ------------------------------------------------------------------------------------------ 1. submit

FAKE_SBATCH = """#!/bin/bash
n=$(( $(wc -l < "$FAKE_LOG" 2>/dev/null || echo 0) + 1000 ))
echo "$n FC=${ONION_FOREIGN_COVARIATES:-unset}:${ONION_FOREIGN_SEED:-unset} $*" >> "$FAKE_LOG"
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


def test_submit_foreign(sub_env):
    tmp, e = sub_env
    p = _submit(tmp, e, DRY_RUN="1", FOREIGN_COVARIATES="out_keep_depth", SCORE_SUBDIR="p16_sv")
    assert p.returncode == 2 and "foreign" in p.stderr
    p = _submit(tmp, e, DRY_RUN="1", FOREIGN_COVARIATES="sideways", SCORE_SUBDIR="p16_foreign")
    assert p.returncode == 2
    p = _submit(tmp, e, DRY_RUN="1", FOREIGN_COVARIATES="out_keep_depth", FOREIGN_SEED="3",
                SCORE_SUBDIR="p16_foreign_ok")
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("ONION_") and "sbatch" in ln]
    assert lines and all("ONION_FOREIGN_COVARIATES=out_keep_depth" in ln
                         and "ONION_FOREIGN_SEED=3" in ln for ln in lines)
    assert all("/rung1_base/foreign_out_keep_depth_s3/seed0/p16_foreign_ok" in ln for ln in lines
               if "ONION_PRED_STORE" in ln or "predict.sh" in ln)
    assert not any("train.sh" in ln for ln in p.stdout.splitlines())
    # a real submission exports the switch; a shell value never leaks without FOREIGN_COVARIATES
    p = _submit(tmp, e, FOREIGN_COVARIATES="in_swap_depth", SCORE_SUBDIR="p16_foreign_is")
    assert p.returncode == 0, p.stderr
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log and all(" FC=in_swap_depth:0 " in ln for ln in log)
    (tmp / "sbatch.log").unlink()
    p = _submit(tmp, e, ONION_FOREIGN_COVARIATES="in_swap_depth", SCORE_SUBDIR="p16_x",
                PRED_STORE_ROOT=str(tmp / "ps2"))
    assert p.returncode == 0, p.stderr
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log and all(" FC=unset:unset " in ln for ln in log)


# ------------------------------------------------------------------------------------------ 3, 4, label

def test_per_stratum_pit_worst_and_report_time(env):
    out = env["root"] / "real" / "seed0" / "p16"
    s = json.loads((out / "scores.json").read_text()) if (out / "scores.json").is_file() else \
        _score(env, out)
    blk = s[S.PIT_STRATA_KEY]
    assert blk["strata"] == ["high", "low"] and set(blk["per_stratum"]) == {"high", "low"}
    worst = s["pit_deviation_worst_stratum"]["per_assay"]
    for a, v in worst.items():
        assert v == pytest.approx(max(blk["per_stratum"][k]["per_assay"][a] for k in ("high", "low")),
                                  rel=0, abs=0)
    rows = RX.table_strata({"dir": out, "scores": s})
    got = [r for r in rows if r["reason"] == "" or r["assay"] == "__macro__"]
    assert got and all(r["value"] == r["stored"] for r in got)
    # accessibility table carries the key too
    assert S.PIT_STRATA_KEY in s["accessibility_only"]


def test_variance_ratio_robust_aggregates():
    rng = np.random.default_rng(0)
    n, mu = np.full(2000, 4.0), rng.gamma(2, 3, 2000)
    a, b = rng.poisson(mu), rng.poisson(mu)
    v = Mx.variance_ratio(n, mu, a, b)
    assert v["pooled"] == pytest.approx(v["pred_var_mean"] / v["rep_var_mean"], rel=1e-12)

    def rec(assay, ratio, pv, rv):
        return {"assay": assay, "cell": f"c{ratio}", "variance_ratio": ratio,
                "variance_ratio_means": (pv, rv)}
    rs = [rec("A", 1.0, 1.0, 1.0), rec("A", 2.0, 4.0, 2.0), rec("A", 47.0, 47.0, 1.0),
          rec("B", 3.0, 3.0, 1.0)]
    med = S.metric(rs, S._median_of("variance_ratio"), {"A", "B"}, 0, 0)
    rom = S.metric(rs, S._ratio_of_means, {"A", "B"}, 0, 0)
    mean = S.metric(rs, S._mean_of("variance_ratio"), {"A", "B"}, 0, 0)
    assert med["per_assay"] == {"A": 2.0, "B": 3.0}
    assert rom["per_assay"]["A"] == pytest.approx(52.0 / 4.0)
    assert mean["per_assay"]["A"] == pytest.approx(50.0 / 3.0)
    bad = rs + [{"assay": "B", "cell": "x", "variance_ratio": Mx.Undefined("no halves"),
                 "variance_ratio_means": None}]
    assert S.metric(bad, S._ratio_of_means, {"A", "B"}, 0, 0)["macro"] is None
    assert S.metric(bad, S._median_of("variance_ratio"), {"A", "B"}, 0, 0)["macro"] is None


def test_robust_block_in_scores_and_report_time(env):
    out = env["root"] / "real" / "seed0" / "p16"
    s = json.loads((out / "scores.json").read_text())
    blk = s[S.H7_ROBUST_KEY]
    assert blk["rung"] == "pseudoreplicate"
    rows = {r["assay"]: r for r in RX.table_variance_ratio({"dir": out, "scores": s})}
    for a, v in blk["median_of_targets"]["per_assay"].items():
        assert rows[a]["median_of_targets"] == pytest.approx(v, rel=1e-12)
        assert rows[a]["ratio_of_means"] == pytest.approx(blk["ratio_of_means"]["per_assay"][a])
    # an older score (no robust key): the median still follows; the ratio of means is not stored
    old = {k: v for k, v in s.items() if k != S.H7_ROBUST_KEY}
    rows = {r["assay"]: r for r in RX.table_variance_ratio({"dir": out, "scores": old})}
    assert all(r["ratio_of_means"] is None and r["reason"] == RX.NOT_STORED for r in rows.values())
    assert rows["__macro__"]["median_of_targets"] == pytest.approx(blk["median_of_targets"]["macro"])


def test_composite_status_label_and_value_unchanged(env):
    s = json.loads((env["root"] / "real" / "seed0" / "p16" / "scores.json").read_text())
    sel = s["selection"]
    assert sel["composite_status"] == S.COMPOSITE_STATUS and "superseded" in sel["composite_status"]
    assert sel["composite"] == S._j(S.composite(sel["imputation_gap_closed"],
                                                sel["correction_gap_closed"],
                                                sel["uncertainty_gap_closed"]))


def test_external_entry_blanks_the_new_keys():
    out = {}
    sc = S.Scorer.__new__(S.Scorer)
    out = {"specific_variance_recovered_above_noise": {}, "r_spec": {}}
    S.Scorer._blank_count_space(sc, out, [], ["A"])
    for k in NEW_KEYS:
        assert out[k]["macro"] is None and out[k]["reason"] == S.EXTERNAL_REASON


# ------------------------------------------------------------------------------------------ 2. artifacts

def test_report_by_scale_and_by_covariate(env, tmp_path):
    out = env["root"] / "real" / "seed0" / "p16"
    s = json.loads((out / "scores.json").read_text())
    it = {"dir": out, "scores": s, "arm": "real", "seed": "seed0"}
    rows = RX.table_by_scale(it)
    assert rows and {r["band"] for r in rows if r["band"]} <= {"fine", "peak", "domain", "broad"}
    for r in rows:
        if r["band"]:
            fc = r["floor_correction"]
            assert r["V"] == pytest.approx(r["explained"] + r["unexplained_with_noise"] + fc, abs=1e-12)
            assert r["E"] == pytest.approx(r["unexplained_with_noise"] + fc + r["added"], abs=1e-12)
            assert (fc != 0) == bool(r["floor_flag"]) or fc == 0
    rows = RX.table_by_covariate(it, tables=env["tables"])
    lab = [r for r in rows if r["column"] == "lab" and r["assay"] == RX.ALL]
    assert lab and sum(r["n_targets"] for r in lab) == len(
        s["specific_variance_recovered_above_noise"]["counts_log1p"]["per_target"])
    # a table version without the column, or with it empty: "not available in this table version"
    t2 = tmp_path / "t2"
    t2.mkdir()
    with open(Path(env["tables"]) / "measurements.tsv", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols = [c for c in rd.fieldnames if c != "antibody"]
        mrows = [{**{c: r[c] for c in cols}, "lab": ""} for r in rd]
    with open(t2 / "measurements.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(mrows)
    rows = {r["column"]: r for r in RX.table_by_covariate(it, tables=t2) if r["assay"] == RX.ALL
            and r["value"] == RX.NOT_AVAILABLE}
    assert "absent" in rows["antibody"]["reason"] and "empty" in rows["lab"]["reason"]
    # the command line, with a seed summary
    o, sm = tmp_path / "rows.tsv", tmp_path / "sum.tsv"
    assert RX.main(["by-scale", "--dirs", str(out), "--out", str(o), "--summary", str(sm)]) == 0
    assert o.read_text().count("\n") > 1 and "V.mean" in sm.read_text()


def test_report_refuses_mixed_foreign_and_normal(env):
    a = env["root"] / "real" / "seed0" / "p16"
    b = env["root"] / "real" / "seed0" / "p16_foreign_bk"
    assert RX.main(["strata", "--dirs", str(a), str(b), "--out",
                    str(env["root"] / "mix.tsv")]) == 2


def _mm_tables(tmp_path, platform_of, antibody=True):
    from onion_eval import mixed_model as mm
    import test_onion_eval_mixed_model as T
    rng = np.random.default_rng(0)
    n_bins = 2 * 2 ** mm.J
    base = rng.gamma(2.0, 2.0, n_bins)
    rows = []
    for c in range(4):
        cell_rate = base * rng.gamma(4.0, 0.25, n_bins)
        for br in (1, 2):
            exp = f"ENCSR{c:03d}"
            mid = f"{exp}.br{br}.uniform.base"
            d = tmp_path / "m" / mid
            d.mkdir(parents=True)
            rate = cell_rate * rng.gamma(10.0, 0.1, n_bins)
            for h in ("counts", "pr1", "pr2"):
                T._write_h5(d / f"{h}.h5", rng, rate if h == "counts" else rate / 2, n_bins)
            r = {k: "" for k in T.MEAS_COLS}
            r.update(measurement_id=mid, experiment=exp, assay="H3K4me3", assay_class="narrow_histone",
                     cell=f"C{c}", in_eic="train", unit=f"br{br}", bio_rep=str(br), layer="uniform",
                     arm_knob="", depth_reads=str(int(rate.sum())), lab=f"lab{c}",
                     platform=platform_of(c, br), antibody=f"ab{c}" if antibody else "",
                     has_pseudoreps="1", counts_path="h5:" + str(d / "counts.h5"))
            rows.append(r)
    tables = tmp_path / "tables"
    tables.mkdir()
    with open(tables / "measurements.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, T.MEAS_COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ev = tmp_path / "evalset.json"
    ev.write_text(json.dumps({"tables": str(tables), "regions": [["chr1", 0], ["chr1", 2 ** mm.J]]}))
    return tables, ev


def test_mixed_model_extra_factors(tmp_path):
    sys.path.insert(0, str(REPO / "tests"))
    from onion_eval import mixed_model as mm
    tables, ev = _mm_tables(tmp_path, lambda c, br: "HiSeq" if c % 2 else "NovaSeq", antibody=False)
    off, on = tmp_path / "off.tsv", tmp_path / "on.tsv"
    assert mm.main(["run", "--tables", str(tables), "--evalset", str(ev), "--band", "peak",
                    "--out", str(off), "--n-boot", "5"]) == 0
    assert mm.main(["run", "--tables", str(tables), "--evalset", str(ev), "--band", "peak",
                    "--out", str(on), "--n-boot", "5", "--extra-factors", "lab,platform,antibody"]) == 0
    read = lambda p: list(csv.DictReader(open(p, newline=""), delimiter="\t"))  # noqa: E731
    a, b = read(off), read(on)
    by = {r["factor"]: r for r in b}
    assert by["platform"]["estimable"] == "1"            # shared across cells: crossed, estimable
    assert by["lab"]["estimable"] == "0" and "confounded" in by["lab"]["note"]
    assert by["antibody"]["estimable"] == "0" and mm.NOT_AVAILABLE in by["antibody"]["note"]
    assert b[-1]["factor"] == "log2_depth" and b[-2]["factor"] == "residual"
    fr = sum(float(r["variance_fraction"]) for r in b if r["estimable"] == "1"
             and r["factor"] != "log2_depth")
    assert fr == pytest.approx(1.0, abs=1e-4)
    # off: the pre-E5 rows (the fixed lab row, no extra factor)
    assert [r["factor"] for r in a] == ["cell", "experiment", "replicate", "library", "residual",
                                        "lab", "log2_depth"]
    with pytest.raises(ValueError):
        mm.run(tables, ev, "peak", tmp_path / "x.tsv", n_boot=0, extra=("donor",))
