"""SC: scorer options (--lambda-source, the signal_override hook for external entries, the
read-overlap flag, tagged blind eval sets). Before/after: with default options the scorer gives
scores.json and suffstats identical, to 0 difference, to the scorer before SC (the module at the
base commit, read from git), for a model and for a baseline, on a synthetic run."""
from __future__ import annotations

import csv
import json
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
from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import score as S  # noqa: E402

#: the umbrella commit SC was cut from (the code before SC)
BASE = "6d5dac3"


def old_module(name: str, relpath: str, tmp: Path):
    """The module as it was at BASE (git show), imported under `name`; skip without git history."""
    import importlib.util
    import subprocess
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:{relpath}"], check=True,
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # pragma: no cover - a kit copy without .git
        pytest.skip(f"git history of {BASE} not available: {e}")
    p = tmp / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"), (7, "chipseq-control"))
BLIND = ((8, "H3K4me3"),)
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=10, guardrail_windows=4, log=lambda *a: None)
SKIP_KEYS = {"wall_seconds", "timing_seconds", "torch_threads"}
PCK = "signal_arcsinh_predicted_control"


class Reader(MiniCandi):
    """mu = (1 + mean visible input count) x the target depth from C'; reads the counts input."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, b):
        x = (b["x_counts"].float() * b["x_avail"][:, None, :].float()).mean(-1, keepdim=True)
        depth = (2.0 ** b["c_out"][..., 0] / 2 ** 24)[:, None, :]
        mu = (self.w * (1.0 + x) * depth).clamp_min(1e-3)
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


def tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


def write_tsv(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("scarms")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    rows = tsv(t / "measurements.tsv")
    base = {(r["cell"], r["assay"]): r for r in rows if r["arm_knob"] == ""}
    # s0's H3K4me3 target has an observed control; s1 and s2 have none
    base[("s0", "H3K4me3")]["control_measurement_id"] = base[("s0", "chipseq-control")]["measurement_id"]
    write_tsv(t / "measurements.tsv", rows)
    targets = {}
    for (cell, a), r in base.items():
        if a == "H3K4me3" and r["in_eic"] in ("val", "blind"):
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
            targets[cell] = r["measurement_id"]
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"] if x["assay"] == "H3K4me3"
                                and x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"] = [p for p in ev["correction_pairs"] if p["assay"] == "H3K4me3"
                              and p["never_trained"] and not p["subset_pair"]][:2]
    ev["replicate_pairs"] = []
    ev["scorable"]["imputation"] = ["H3K4me3"]      # the only assay with targets kept here
    out.write_text(json.dumps(ev))
    return {"root": root, "tables": t, "evalset": out, "ev": ev, "targets": targets}


def _scorer(mod, env, evalset=None, model=True, **kw):
    if model:
        return mod.Scorer(evalset or env["evalset"], main="model", model=Reader(), config=CFG,
                          run_tables_sha256=tables_sha256(env["tables"]), **dict(KW, **kw))
    return mod.Scorer(evalset or env["evalset"], config=CFG, **dict(KW, **kw))


def canon(scores: dict) -> str:
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in SKIP_KEYS}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(scores), sort_keys=True, allow_nan=False)


@pytest.fixture(scope="module")
def default_model(env):
    sc = _scorer(S, env)
    return sc.run(), sc.suff


def test_before_after_bit_identical(env, default_model, tmp_path_factory):
    """Default options: the scorer before SC and after SC write the same scores and suffstats."""
    old = old_module("score_before_sc", "tools/onion_eval/score.py", tmp_path_factory.mktemp("old"))
    new_s, new_suff = default_model
    o = _scorer(old, env)
    old_s = o.run()
    assert canon(old_s) == canon(new_s)
    assert set(o.suff) == set(new_suff)
    for k in o.suff:
        np.testing.assert_array_equal(o.suff[k], new_suff[k])
    # an explicit `auto` is the default
    assert canon(_scorer(S, env, lambda_source="auto").run()) == canon(new_s)
    # a baseline too, and through score() / write_outputs (bytes of scores.json, minus timings)
    d = tmp_path_factory.mktemp("bl")
    a = old.score(env["evalset"], d / "old", baseline="position_average", config=CFG, **KW)
    b = S.score(env["evalset"], d / "new", baseline="position_average", config=CFG, **KW)
    assert canon(a) == canon(b)
    assert "lambda_source_option" not in b and "signal_source" not in b
    assert "correction_pairs_not_read_disjoint" not in b


def _drop_pc(scores):
    """The scores without the keys the predicted-control option adds (in every table)."""
    n = [0]

    def strip(x):
        if isinstance(x, dict):
            n[0] += sum(k in (PCK, "lambda_source_option") for k in x)
            return {k: strip(v) for k, v in x.items() if k not in (PCK, "lambda_source_option")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    s = strip(json.loads(canon(scores)))
    assert n[0] >= 3
    return json.dumps(s, sort_keys=True)


def test_lambda_source_predicted_control(env, default_model):
    base, _ = default_model
    s = _scorer(S, env, lambda_source="predicted_control").run()
    assert s["lambda_source_option"] == "predicted_control"
    # every existing key is unchanged; the new route sits beside it
    assert _drop_pc(s) == json.dumps(json.loads(canon(base)), sort_keys=True)
    h1 = s["specific_variance_recovered_above_noise"]
    auto, pc = h1["signal_arcsinh"]["per_target"], h1[PCK]["per_target"]
    t0, t1 = env["targets"]["s0"], env["targets"]["s1"]
    assert auto[t0]["lambda_source"] == "observed_control"
    assert auto[t1]["lambda_source"] == "predicted_control"
    assert {v["lambda_source"] for v in pc.values()} == {"predicted_control"}
    assert pc[t0]["level"] != auto[t0]["level"]            # s0's control is now the predicted one
    assert h1[PCK]["macro"] is not None
    # r_spec's training average is the `auto` one; on a ChIP target without an observed control
    # it is undefined in both routes (the position average has no control; pre-existing)
    assert s["r_spec"][PCK]["reason"] == s["r_spec"]["signal_arcsinh"]["reason"] != ""
    assert h1[PCK]["lambda_source_option"] == "predicted_control"


def test_lambda_source_observed_control(env):
    s = _scorer(S, env, lambda_source="observed_control").run()
    per = s["specific_variance_recovered_above_noise"]["signal_arcsinh"]["per_target"]
    t0, t1 = env["targets"]["s0"], env["targets"]["s1"]
    assert per[t0]["lambda_source"] == "observed_control"
    assert per[t1]["lambda_source"] is None and "no observed control" in per[t1]["lambda_reason"]
    assert s["specific_variance_recovered_above_noise"]["signal_arcsinh"]["macro"] is None
    assert PCK not in s["r_spec"]
    with pytest.raises(ValueError, match="lambda_source"):
        _scorer(S, env, lambda_source="observed")


def test_predicted_control_on_a_baseline_is_undefined_with_reason(env, tmp_path):
    out = tmp_path / "bl" / "seed0"
    rc = S.main(["--baseline", "position_average", "--evalset", str(env["evalset"]), "--out",
                 str(out), "--n-boot", "10", "--lambda-source", "predicted_control"])
    assert rc == 0
    s = json.loads((out / "scores.json").read_text())
    assert s["lambda_source_option"] == "predicted_control"
    h = s["specific_variance_recovered_above_noise"][PCK]
    assert h["macro"] is None
    assert all("no control prediction" in v["lambda_reason"] for v in h["per_target"].values())


def _truth_override(calls):
    def f(row, regions):
        calls.append(row["measurement_id"])
        d = Path(row["counts_path"][3:]).parent
        with h5py.File(d / "signal.h5", "r") as fh:
            return [np.asarray(fh[c][s:s + 256], np.float32) for c, s in regions]
    return f


def test_signal_override_external_entry(env, tmp_path):
    calls = []
    s = S.score(env["evalset"], tmp_path / "ext" / "seed0", signal_override=_truth_override(calls),
                config=CFG, **KW)
    assert sorted(set(calls)) == sorted(t["target"] for t in env["ev"]["imputation_targets"])
    assert s["predictor"] == S.EXTERNAL and s["count_space_reason"] == S.EXTERNAL_REASON
    h1 = s["specific_variance_recovered_above_noise"]
    assert h1["counts_log1p"]["macro"] is None and h1["counts_log1p"]["reason"] == S.EXTERNAL_REASON
    for k in S.Scorer.COUNT_SPACE_KEYS + ("correction_gap_closed_vs_depth_only_never_trained",
                                          "correction_crps_model", "subset_pairs_gap_closed",
                                          "depth_law_abs_log2_error_never_trained"):
        assert s[k]["macro"] is None and s[k]["reason"] == S.EXTERNAL_REASON, k
    assert s["r_spec"]["counts_log1p"]["reason"] == S.EXTERNAL_REASON
    assert s["error_budget"]["reason"] == S.EXTERNAL_REASON
    assert s["replicate_pair_gap_closed_vs_depth_only"]["reason"] == S.EXTERNAL_REASON
    assert s["n_pairs"] == 0 and s["selection"]["composite"] is None
    assert s["selection"]["imputation_reason"] == S.EXTERNAL_REASON
    # the signal space is scored from the override (here the truth itself)
    assert h1["signal_arcsinh"]["macro"] is not None
    per = h1["signal_arcsinh"]["per_target"]
    assert {v["lambda_source"] for v in per.values()} == {"external_entry"}
    assert all(v["level"] is not None and v["N"] is not None for v in per.values())
    assert h1["signal_arcsinh_plug_in"]["macro"] is None
    assert "external entry" in h1["signal_arcsinh_plug_in"]["reason"]
    assert s["copy_leak"]["copy_leak_pass"] is None


def test_signal_override_shapes(env):
    calls = []
    sc = S.Scorer(env["evalset"], main=S.EXTERNAL, signal_override=_truth_override(calls),
                  config=CFG, **KW)
    row = sc.ds.rows[env["targets"]["s1"]]
    a = sc._override_signal(row)
    sc.signal_override = lambda r, regs: np.concatenate(_truth_override([])(r, regs))
    np.testing.assert_array_equal(a, sc._override_signal(row))
    sc.signal_override = lambda r, regs: np.zeros(10, np.float32)
    with pytest.raises(ValueError, match="bins"):
        sc._override_signal(row)
    with pytest.raises(ValueError, match="signal_override"):
        S.Scorer(env["evalset"], main="model", model=Reader(), signal_override=lambda r, g: 0,
                 config=CFG, run_tables_sha256=tables_sha256(env["tables"]), **KW)
    with pytest.raises(ValueError, match="exactly one"):
        S.score(env["evalset"], "x", baseline="marginal_nb", signal_override=lambda r, g: 0)


def test_pair_not_read_disjoint_is_not_scored(env, tmp_path):
    ev = json.loads(env["evalset"].read_text())
    assert len(ev["correction_pairs"]) == 2
    ev["correction_pairs"][0].update(read_disjoint=0, read_disjoint_reason="test")
    p = env["evalset"].parent / "evalset_flag.json"
    p.write_text(json.dumps(ev))
    sc = _scorer(S, env, evalset=p, model=False, main="depth_only_scaling")
    _, cor, _, _ = sc.jobs()
    assert [c["pair_id"] for c in cor] == [ev["correction_pairs"][1]["pair_id"]]
    assert sc.n_pairs_not_read_disjoint == 1
    s = S.score(p, tmp_path / "dos" / "seed0", baseline="depth_only_scaling", config=CFG, **KW)
    assert s["n_pairs"] == 1 and s["correction_pairs_not_read_disjoint"] == 1


def test_tagged_blind_evalset_is_scored(env, tmp_path):
    out = env["root"] / "eval" / "synth2" / "evalset_blind_tv.json"
    ev = E.build(env["tables"], out, include_blind=True, n_regions=1, n_fit_regions=1,
                 region_bins=256, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"] if x["split"] == "blind"
                                and x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"], ev["control_targets"] = [], [], []
    out.write_text(json.dumps(ev))
    s = S.score(out, tmp_path / "mnb" / "seed0" / "blind", baseline="marginal_nb", config=CFG,
                split="blind", allow_blind=True, **KW)
    assert s["split"] == "blind" and s["n_targets"] == 1
    assert (out.parent / S.BLIND_LEDGER).is_file()
