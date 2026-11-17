"""Close-out item A12: per-column screening at scoring (`score.py --foreign-covariates
<side>_keep_depth.<col>...` and `<side>_neutral.<col>...`). Reuses the synthetic eval set of the E5
tests. The whole-row E5 modes are pinned by tests/test_onion_eval_scorer_extensions.py."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "e5_tests", REPO / "tests/test_onion_eval_scorer_extensions.py")
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)
env = X.env                      # the module-scoped synthetic eval set fixture
sub_env = X.sub_env
S, SR = X.S, X.SR

from minicandi.contract import META_COLUMNS  # noqa: E402
from minicandi.covariates import neutral_value  # noqa: E402


def test_spec_of_column_modes():
    s = S.foreign_spec("out_keep_depth.lab_id.platform_id", 2)
    assert s == {"mode": "out_keep_depth.lab_id.platform_id", "sides": ["out"], "depth": "keep_depth",
                 "seed": 2, "columns": ["lab_id", "platform_id"], "kind": "keep_depth"}
    assert S.foreign_spec("both_neutral.donor_id")["kind"] == "neutral"
    assert S.foreign_spec("in_neutral.nsc")["columns"] == ["nsc"]
    for bad in ("out_neutral", "out_neutral.", "out_neutral.log2_depth", "out_swap_depth.lab_id",
                "out_neutral.lab", "out_neutral.lab_id.lab_id", "out_neutral.nsc", "both_neutral.frip",
                "up_neutral.lab_id"):
        with pytest.raises(ValueError):
            S.foreign_spec(bad)
    assert "columns" not in S.foreign_spec("out_keep_depth")          # whole-row modes unchanged


def _batch(sc, job, pair):
    c, s0 = sc.regions[0]
    win = sc._pair_window(job, c, s0) if pair else sc.ds.window(c, s0, job["inputs"], [job["target"]])
    return sc.C.collate([win, win])


@pytest.mark.parametrize("mode", ["out_keep_depth.lab_id", "both_neutral.lab_id.read_length",
                                  "in_keep_depth.platform_id.donor_id"])
def test_model_batch_changes_only_the_named_columns(env, mode):
    sc = X._scorer(env, mode)
    spec = S.foreign_spec(mode)
    idx = [META_COLUMNS.index(c) for c in spec["columns"]]
    jobs = sc.jobs()
    n_changed = 0
    for job, pair in [(j, False) for j in jobs[0][:8]] + [(j, True) for j in jobs[1][:8]]:
        batch = _batch(sc, job, pair)
        before = {k: batch[k].clone() for k in ("c_in", "c_out")}
        mb = sc._model_batch(batch, job)
        assert all(torch.equal(batch[k], before[k]) for k in before)
        scale = 0.5 if pair else 1.0
        ins = [job["source"]] if pair else job["inputs"]
        for key, sd, mids in (("c_in", "in", ins), ("c_out", "out", [job["target"]])):
            for m in mids:
                t = sc.C.TRACK_INDEX[sc.ds.rows[m]["assay"]]
                av = batch["x_avail" if sd == "in" else "y_avail"][:, t].bool()
                got, old = mb[key][:, t], before[key][:, t]
                other = [i for i in range(got.shape[-1]) if i not in idx]
                assert torch.equal(got[:, other], old[:, other])          # depth and the rest kept
                if sd not in spec["sides"] or not bool(av.all()):
                    assert torch.equal(got, old)
                    continue
                if spec["kind"] == "neutral":
                    want = torch.tensor([neutral_value(c) for c in spec["columns"]])
                else:
                    d = sc.ds.rows[sc._donor(m)]
                    v = torch.as_tensor(sc.ds.encoder.c_in(d, sc.ds.qc_on_input, scale) if sd == "in"
                                        else sc.ds.encoder.c_out(d, scale))
                    want = v[idx]
                assert torch.equal(got[:, idx], want.to(got.dtype).expand(got.shape[0], -1))
                n_changed += int(not torch.equal(got, old))
    assert n_changed > 0                                                  # the test has power


def test_qc_column_needs_qc_on_input(env):
    with pytest.raises(ValueError, match="qc_on_input"):
        X._scorer(env, "in_neutral.nsc")


def test_end_to_end_trivial_and_real(env):
    root = env["root"]
    cfg = {"context_bins": X.L, "batch_size": X.B}
    n = X._score(env, root / "a12triv" / "p16", model=X.Trivial(), cfg=cfg)
    k = X._score(env, root / "a12triv" / "p16_foreign_on", model=X.Trivial(), cfg=cfg,
                 foreign_covariates="out_neutral.lab_id")
    fk = k.pop(S.FOREIGN_KEY)
    assert k["scorer_settings"].pop(S.FOREIGN_KEY) == "out_neutral.lab_id"
    k["scorer_settings"].pop("foreign_seed")
    assert X._clean(k) == X._clean(n)                    # Trivial reads depth only: no change
    assert fk["columns"] == ["lab_id"] and "columns_definition" in fk
    assert set(fk["n_changed_by_column"]["out"]) <= {"lab_id"}
    assert fk["n_changed_by_column"]["out"].get("lab_id", 0) > 0
    assert all(e["neutral"] for e in fk["assignment"]["out"].values())
    real = X._score(env, root / "a12real" / "seed0" / "p16")
    f = X._score(env, root / "a12real" / "seed0" / "p16_foreign_okl",
                 foreign_covariates="out_keep_depth.lab_id")
    fb = f[S.FOREIGN_KEY]
    assert set(fb["n_changed_by_column"]["out"]) == {"lab_id"}
    assert X._all_assays(f["count_crps"]) != X._all_assays(real["count_crps"])
    assert SR.scorer_keys(f)["foreign_covariates"] == "out_keep_depth.lab_id:seed0"


def test_cli_accepts_column_modes(monkeypatch):
    ap = argparse.ArgumentParser()
    S.add_foreign_arguments(ap)
    monkeypatch.delenv(S.FOREIGN_ENV, raising=False)
    assert S.foreign_kwargs(ap.parse_args(["--foreign-covariates", "in_neutral.lab_id"])) == {
        "foreign_covariates": "in_neutral.lab_id", "foreign_seed": 0}
    with pytest.raises(SystemExit):
        ap.parse_args(["--foreign-covariates", "in_neutral.log2_depth"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--foreign-covariates", "sideways"])


def test_submit_accepts_column_modes(sub_env):
    tmp, e = sub_env
    p = X._submit(tmp, e, DRY_RUN="1", FOREIGN_COVARIATES="out_neutral.lab_id.platform_id",
                  SCORE_SUBDIR="p256_foreign_out_neutral.lab_id.platform_id")
    assert p.returncode == 0, p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("ONION_") and "sbatch" in ln]
    assert lines and all("ONION_FOREIGN_COVARIATES=out_neutral.lab_id.platform_id" in ln
                         for ln in lines)
    for bad in ("out_neutral", "out_neutral.Lab", "out_swap_depth.lab_id"):
        p = X._submit(tmp, e, DRY_RUN="1", FOREIGN_COVARIATES=bad, SCORE_SUBDIR="p256_foreign_x")
        assert p.returncode == 2, bad


def test_column_modes_combine_with_native_input_depth(env):
    """Per-column modes never touch log2 depth, so they combine with the native-depth switch
    (ONION_SCORE_INPUT_DEPTH, a run trained with input_depth = fixed): the inputs are unthinned and
    only the named column changes; the whole-row swap_depth on side C stays refused."""
    from minicandi import registry as R
    cfg = {**json.loads((REPO / "tools/minicandi/configs/rung2/base.json").read_text()),
           "context_bins": X.L, "batch_size": X.B, "input_depth": "fixed"}
    torch.manual_seed(2)
    m = R.build_model(cfg).eval()
    sc = S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256=env["sha"],
                  foreign_covariates="both_keep_depth.lab_id", input_depth_at_scoring="native", **X.KW)
    assert sc.ds.config["input_depth"] == "native" and sc.foreign["columns"] == ["lab_id"]
    i = META_COLUMNS.index("lab_id")
    changed = 0
    for job in sc.jobs()[0][:8]:
        batch = _batch(sc, job, False)
        mb = sc._model_batch(batch, job)
        for k in ("c_in", "c_out"):
            other = [j for j in range(batch[k].shape[-1]) if j != i]
            assert torch.equal(mb[k][..., other], batch[k][..., other])   # depth included
            changed += int(not torch.equal(mb[k], batch[k]))
    assert changed > 0
    with pytest.raises(ValueError, match="swap_depth"):
        S.Scorer(env["evalset"], main="model", model=m, config=cfg, run_tables_sha256=env["sha"],
                 foreign_covariates="in_swap_depth", input_depth_at_scoring="native", **X.KW)
