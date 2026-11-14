"""E6 scorer end to end on synthetic tables: a trivial model (through the --run CLI and a stand-in
`load_run`) and each training-free baseline. Depth-only must close exactly 0 of the correction gap
and the half-against-half floor exactly 1; marginal NB has an uncertainty term of exactly 0."""
from __future__ import annotations

import csv
import json
import sys
import types
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
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import score as S  # noqa: E402
from onion_eval import select_rule as R  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"), (7, "chipseq-control"))
BLIND = ((8, "H3K4me3"),)
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=30, guardrail_windows=8, log=lambda *a: None)

PINNED = ("specific_variance_recovered_above_noise", "r_spec", "count_crps_skill_vs_position_average",
          "count_crps", "count_crps_oracle_scaled", "count_crps_scale_error",
          "correction_gap_closed_vs_depth_only_never_trained", "correction_crps_model",
          "correction_crps_depth_only", "correction_crps_pseudoreplicate_floor",
          "subset_pairs_gap_closed", "depth_law_abs_log2_error_never_trained",
          "pit_deviation_pooled", "pit_deviation_worst_stratum", "predicted_over_replicate_variance",
          "replicate_pair_gap_closed_vs_depth_only", "copy_leak", "error_budget", "selection",
          "n_targets", "n_pairs", "regime", "split", "scorable")


class Trivial(MiniCandi):
    """Reads no count input: mu = a per-track constant rate times the target depth from C'."""

    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.full((36,), 3.0), requires_grad=False)

    def forward(self, b):
        L = b["y_counts"].shape[1]
        mu = (2.0 ** b["c_out"][..., 0] / 2 ** 24 * self.rate).clamp_min(1e-3)
        mu = mu[:, None, :].expand(-1, L, -1).contiguous()
        return {"n": torch.full_like(mu, 5.0), "mu": mu}


def _tsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("score")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    # a few replicate-ladder pairs so the per-rung path runs (base -> mapq0 views as "technical")
    pairs = _tsv(t / "pairs.tsv")
    meas = {r["measurement_id"]: r for r in _tsv(t / "measurements.tsv")}
    extra = []
    for r in meas.values():
        if r["in_eic"] == "val" and r["arm_knob"] == "" and r["assay"] == "H3K4me3":
            extra.append(dict(pairs[0], pair_id=f"rep{len(extra)}", a=r["measurement_id"],
                              b=r["measurement_id"].replace(".base", ".mapq0"), rung="technical",
                              assay=r["assay"], cell=r["cell"], subset_pair="0", split="val",
                              never_trained="0", knobs_differing="mapq"))
    with open(t / "pairs.tsv", "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]), delimiter="\t", lineterminator="\n")
        w.writerows(extra)
    # pipeline signal tracks for the H3K4me3 validation targets (the signal space, X2)
    for r in meas.values():
        if r["in_eic"] == "val" and r["arm_knob"] == "" and r["assay"] == "H3K4me3":
            d = Path(r["counts_path"][3:]).parent
            for nm, src in (("signal", "counts"), ("pr1_signal", "pr1"), ("pr2_signal", "pr2")):
                with h5py.File(d / f"{src}.h5", "r") as fi, h5py.File(d / f"{nm}.h5", "w") as fo:
                    for c in fi:
                        fo.create_dataset(c, data=np.asarray(fi[c][:], np.float32) * 0.7)
    sp = root / "eval" / "strata" / "masks.h5"
    sp.parent.mkdir(parents=True)
    sizes = json.loads((root / "genome" / "chrom_sizes.json").read_text())
    with h5py.File(sp, "w") as f:
        rng = np.random.default_rng(1)
        hi, lo = f.create_group("high"), f.create_group("low")
        for c, bp in sizes.items():
            m = rng.random(bp // 25) < 0.3
            hi.create_dataset(c, data=m)
            lo.create_dataset(c, data=~m)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, strata=sp, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    # keep the scored pairs small: every never-trained non-subset pair plus two subset pairs
    # (the library's oracle rescale costs about 1 s per target, so the scored set stays small)
    nt = [p for p in ev["correction_pairs"] if p["never_trained"] and not p["subset_pair"]
          and p["cell"] in ("s0", "s1", "s3", "s4")]
    acc = [x for x in ev["imputation_targets"] if x["regime"] == "accessibility_only"]
    ev["imputation_targets"] = [x for x in ev["imputation_targets"] if x not in acc] + [
        x for x in acc if x["cell"] in ("s0", "s3")]
    sub = [p for p in ev["correction_pairs"] if p["subset_pair"]
           and p["assay"] in ev["scorable"]["correction"]][:2]
    ev["correction_pairs"] = nt + sub
    out.write_text(json.dumps(ev))
    res = {}
    bdir = root / "eval" / "synth" / "baselines"
    for name in S.REFS:
        res[name] = S.score(out, bdir / name / "seed0", baseline=name, config=CFG, **KW)
    return {"root": root, "tables": t, "evalset": out, "ev": ev, "res": res, "bdir": bdir}


@pytest.fixture(scope="module")
def model_run(env, tmp_path_factory):
    """The trivial model scored through the CLI `--run`, with a stand-in for T1's load_run."""
    run = tmp_path_factory.mktemp("runs") / "rung1_trivial" / "seed0"
    run.mkdir(parents=True)
    # training wrote DONE and ckpt.pt (finding 3: the scorer used to skip on this DONE)
    (run / "DONE").write_text("1\n")
    torch.save({"tables_sha256": tables_sha256(env["tables"])}, run / "ckpt.pt")
    fake = types.ModuleType("minicandi.registry")
    fake.load_run = lambda d: (Trivial(), dict(CFG))
    old = sys.modules.get("minicandi.registry")
    sys.modules["minicandi.registry"] = fake
    try:
        rc = S.main(["--run", str(run), "--evalset", str(env["evalset"]), "--out", str(run),
                     "--n-boot", "30", "--guardrail-windows", "8"])
    finally:
        if old is None:
            sys.modules.pop("minicandi.registry", None)
        else:
            sys.modules["minicandi.registry"] = old
    assert rc == 0
    return run


def _strict(path):
    def bad(x):
        raise ValueError(f"non-JSON constant {x}")
    return json.loads(Path(path).read_text(), parse_constant=bad)


def test_every_pinned_key_and_level(model_run, env):
    s = _strict(model_run / "scores.json")
    assert set(PINNED) <= set(s)
    for k in ("specific_variance_recovered_above_noise", "r_spec"):
        assert set(s[k]) == {"counts_log1p", "signal_arcsinh", "signal_arcsinh_plug_in"}
    h1 = s["specific_variance_recovered_above_noise"]["counts_log1p"]
    assert set(h1) >= {"macro", "per_assay", "ci", "per_target"}
    loao = [x["target"] for x in env["ev"]["imputation_targets"]
            if x["regime"] == "leave_one_assay_out"]
    assert set(h1["per_target"]) == set(loao)
    assert all(np.isfinite(v["level"]) for v in h1["per_target"].values())
    sig = s["specific_variance_recovered_above_noise"]["signal_arcsinh"]
    # H3K4me3 has signal tracks (no observed control: the model's predicted control); H3K27ac has none
    assert sig["per_assay"]["H3K4me3"] is not None
    assert sig["macro"] is None and "signal.h5" in sig["reason"]
    for tg, v in sig["per_target"].items():
        if v["assay"] == "H3K4me3":
            assert v["lambda_source"] == "predicted_control" and np.isfinite(v["level"])
            assert v["level"] != h1["per_target"][tg]["level"]         # finding 15
        else:
            assert v["lambda_source"] is None and "signal.h5" in v["lambda_reason"]
    # ruling 20: the plug-in variant (score of the NB mean) beside the headline expectation
    pi = s["specific_variance_recovered_above_noise"]["signal_arcsinh_plug_in"]
    assert sig["signal_mode"] == "expectation" and pi["signal_mode"] == "plug_in"
    assert pi["per_assay"]["H3K4me3"] is not None and pi["macro"] is None
    for tg, v in pi["per_target"].items():
        if v["assay"] == "H3K4me3":                # same truth and floor, another prediction
            assert np.isfinite(v["level"]) and v["N"] == sig["per_target"][tg]["N"]
            assert v["level"] != sig["per_target"][tg]["level"]
    assert set(s["r_spec"]["signal_arcsinh_plug_in"]) >= {"macro", "per_assay", "reason"}
    # ruling 19: the DNase-seq route defaults to MACS2 without control
    assert sig["dnase_route"] == s["dnase_route"] == "macs2_no_control"
    assert set(s["copy_leak"]) >= set(("hidden_input_max_abs_change", "unavailable_input_max_abs_change",
                                       "same_view_copy_gain", "copy_leak_pass"))
    assert s["copy_leak"]["hidden_input_max_abs_change"] == 0.0
    assert len(s["copy_leak"]["same_view_copy_gain_relative_by_mask_seed"]) == 3
    assert isinstance(s["copy_leak"]["same_view_copy_gain_spread_ok"], bool)
    assert isinstance(s["copy_leak"]["same_view_copy_gain_flag"], bool)
    assert s["copy_leak"]["gate_on_copy_gain"] is False
    assert set(s["error_budget"]) >= set(S.BUDGET_KEYS)
    assert set(s["selection"]) >= {"space", "imputation_gap_closed", "correction_gap_closed",
                                   "uncertainty_gap_closed", "composite"}
    assert s["selection"]["space"] == "counts_log1p"
    assert s["regime"] == "leave_one_assay_out" and s["split"] == "val"
    assert s["n_targets"] == len(loao) and s["n_pairs"] == len(env["ev"]["correction_pairs"])
    assert (model_run / "SCORED").is_file()
    suff = np.load(model_run / "suffstats.npz")
    assert any(k.startswith("gram/leave_one_assay_out/H3K4me3/") for k in suff.files)
    assert any(k.startswith("pit/") for k in suff.files) and any(k.startswith("pair/") for k in suff.files)
    # a second call skips on SCORED
    assert S.main(["--run", str(model_run), "--evalset", str(env["evalset"]),
                   "--out", str(model_run)]) == 0
    assert s["tables_sha256"] == s["run_tables_sha256"] == tables_sha256(env["tables"])
    assert s["noise_floor"] == "pseudoreplicate_halves" and s["n_targets_without_halves"] == 0


def test_trivial_model_selection_terms(model_run):
    sel = _strict(model_run / "scores.json")["selection"]
    for k in ("imputation_gap_closed", "correction_gap_closed", "uncertainty_gap_closed", "composite"):
        assert sel[k] is not None, (k, sel)
    assert sel["composite"] == pytest.approx(S.composite(sel["imputation_gap_closed"],
                                                         sel["correction_gap_closed"],
                                                         sel["uncertainty_gap_closed"]))


def test_depth_only_closes_zero_and_floor_one(env):
    d = env["res"]["depth_only_scaling"]["correction_gap_closed_vs_depth_only_never_trained"]
    f = env["res"]["pseudoreplicate_floor"]["correction_gap_closed_vs_depth_only_never_trained"]
    assert d["macro"] == 0.0 and set(d["per_assay"].values()) == {0.0}
    assert f["macro"] == 1.0 and set(f["per_assay"].values()) == {1.0}
    assert all(v == 0.0 for v in d["per_knob"].values()) and d["per_knob"]
    # subset pairs are reported apart; on pure thinning depth-only is exact in expectation, so it
    # sits at the half-against-half floor and the gap is undefined (never in headline 4)
    sub = env["res"]["depth_only_scaling"]["subset_pairs_gap_closed"]
    assert sub["n"] == 2
    assert sub["macro"] == 0.0 or "not above the floor" in sub["reason"]
    rep = env["res"]["depth_only_scaling"]["replicate_pair_gap_closed_vs_depth_only"]["per_rung"]
    assert set(rep) == {"technical"}
    assert rep["technical"]["non_scorable_per_assay"]["H3K4me3"] == 0.0


def test_baselines_terms(env):
    m = env["res"]["marginal_nb"]["selection"]
    assert m["uncertainty_gap_closed"] == 0.0 and m["imputation_gap_closed"] is not None
    pa = env["res"]["position_average"]
    assert pa["selection"]["imputation_gap_closed"] is not None
    # the position average scores zero CRPS skill against itself
    assert pa["count_crps_skill_vs_position_average"]["macro"] == 0.0
    # depth-only is undefined on imputation targets: the composite is undefined, with a reason
    d = env["res"]["depth_only_scaling"]["selection"]
    assert d["imputation_gap_closed"] is None and d["composite"] is None and d["reason"]
    for name in S.REFS:
        assert env["res"][name]["copy_leak"]["copy_leak_pass"] is None


def test_scorable_set_drives_macro(env):
    s = env["res"]["position_average"]
    h1 = s["specific_variance_recovered_above_noise"]["counts_log1p"]
    assert set(h1["per_assay"]) == {"H3K4me3", "H3K27ac"}
    assert "H3K27me3" in s["count_crps"]["non_scorable_per_assay"]
    assert "H3K27me3" not in s["count_crps"]["per_assay"]
    assert s["count_crps"]["macro"] == pytest.approx(np.mean(list(s["count_crps"]["per_assay"].values())))
    lo, hi = s["count_crps"]["ci"]
    assert lo is not None and lo <= hi


def test_accessibility_only_table(env):
    acc = env["res"]["marginal_nb"]["accessibility_only"]
    assert acc["regime"] == "accessibility_only"
    assert set(acc["count_crps"]["per_assay"]) == {"H3K4me3", "H3K27ac"}
    assert acc["count_crps"]["n"] == 2
    assert acc["count_crps"]["macro"] is not None


def test_baselines_read_as_runs(env):
    rows = R.table(env["bdir"])
    assert {r["arm"] for r in rows} == set(S.REFS) and {r["state"] for r in rows} == {"done"}


def test_blind_refused(env):
    with pytest.raises(PermissionError):
        S.Scorer(env["evalset"], main="marginal_nb", config=CFG, split="blind")
    out = env["root"] / "blind_out"
    assert S.main(["--baseline", "marginal_nb", "--evalset", str(env["evalset"]), "--out", str(out),
                   "--split", "blind"]) == 2
    # a blind row smuggled into a validation item is refused too
    ev = dict(env["ev"])
    meas = {r["measurement_id"]: r for r in _tsv(env["tables"] / "measurements.tsv")}
    blind = next(m for m, r in meas.items() if r["in_eic"] == "blind" and r["arm_knob"] == "")
    ev["imputation_targets"] = [dict(ev["imputation_targets"][0], target=blind)]
    p = env["root"] / "eval" / "synth" / "evalset_smuggled.json"
    p.write_text(json.dumps(ev))
    with pytest.raises(PermissionError):
        S.Scorer(p, main="marginal_nb", config=CFG).jobs()
    # finding 14: --allow-blind on a validation pass still refuses a blind row
    with pytest.raises(PermissionError, match="blind"):
        S.Scorer(p, main="marginal_nb", config=CFG, split="val", allow_blind=True).jobs()


def test_pooled_pit_equals_concatenated():
    rng = np.random.default_rng(0)
    parts = []
    for k in range(3):
        mu = rng.uniform(0.5, 20, 300 + 50 * k)
        y = rng.negative_binomial(4, 4 / (4 + mu))
        parts.append((np.full(mu.size, 4.0), mu, y))
    pooled = S.pooled_pit_deviation([(S.pit_curve(*p), p[2].size) for p in parts])
    cat = [np.concatenate([p[i] for p in parts]) for i in range(3)]
    assert pooled == pytest.approx(Mx.pit_deviation(*cat)["pooled"], abs=1e-12)


def test_composite_weights():
    assert S.composite(0.4, 0.8, 0.2) == pytest.approx(0.5 * 0.4 + 0.25 * 0.8 + 0.25 * 0.2)
    assert np.isnan(S.composite(0.4, float("nan"), 0.2))


# ------------------------------------------------------------------ review E6 regression tests

def _model_scorer(env, **kw):
    return S.Scorer(env["evalset"], main="model", model=Trivial(), config=CFG,
                    run_tables_sha256=tables_sha256(env["tables"]), **dict(KW, **kw))


def _fake_guardrail(G, gains, part1):
    def fake(model, batches, seed=0):
        g = gains[seed]
        ok = part1[seed] and (g <= G.GAIN_THRESHOLD or not G.GATE_ON_COPY_GAIN)
        return {"hidden_input_max_abs_change": 0.0 if part1[seed] else 0.5,
                "unavailable_input_max_abs_change": 0.0,
                "same_view_copy_gain": g, "same_view_copy_gain_relative": g,
                "same_view_copy_gain_flag": g > G.GAIN_THRESHOLD,
                "copy_leak_pass": ok, "part1_wiring_pass": part1[seed],
                "gate_on_copy_gain": G.GATE_ON_COPY_GAIN}
    return fake


def test_f2_copy_leak_pass_needs_every_mask_seed(env, monkeypatch):
    """Finding 2: copy_leak_pass came from mask seed 0 only. The gate is part 1 on every mask seed;
    the same-view copy gain (0.05, 0.15, 0.05 here) is written and flagged but does not gate."""
    from onion_eval import guardrail as G
    gains, part1 = {0: 0.05, 1: 0.15, 2: 0.05}, {0: True, 1: True, 2: True}
    monkeypatch.setattr(G, "copy_leak", _fake_guardrail(G, gains, part1))
    cl = _model_scorer(env)._copy_leak()
    assert cl["copy_leak_pass"] is True and cl["gate_on_copy_gain"] is False
    assert cl["same_view_copy_gain_flag"] is True
    assert cl["same_view_copy_gain_flag_by_mask_seed"] == [False, True, False]
    assert cl["same_view_copy_gain_relative"] == pytest.approx(0.15)
    assert cl["same_view_copy_gain_relative_spread"] == pytest.approx(0.10)
    assert cl["same_view_copy_gain_spread_ok"] is False         # 0.10 > 0.5 x 0.10, written only
    assert cl["hidden_input_max_abs_change_by_mask_seed"] == [0.0, 0.0, 0.0]
    assert cl["reason"] == ""
    part1[1] = False                                            # one mask seed leaks: out
    cl = _model_scorer(env)._copy_leak()
    assert cl["copy_leak_pass"] is False and cl["copy_leak_pass_by_mask_seed"] == [True, False, True]
    assert cl["part1_wiring_pass"] is False and "part 1 failed" in cl["reason"]


def test_f2_gate_on_copy_gain_restores_the_old_gate(env, monkeypatch):
    from onion_eval import guardrail as G
    monkeypatch.setattr(G, "GATE_ON_COPY_GAIN", True)
    gains, part1 = {0: 0.05, 1: 0.15, 2: 0.05}, {0: True, 1: True, 2: True}
    monkeypatch.setattr(G, "copy_leak", _fake_guardrail(G, gains, part1))
    cl = _model_scorer(env)._copy_leak()
    assert cl["gate_on_copy_gain"] is True
    assert cl["copy_leak_pass"] is False and cl["copy_leak_pass_by_mask_seed"] == [True, False, True]
    assert cl["same_view_copy_gain_spread_ok"] is False and "across mask seeds" in cl["reason"]
    gains.update({1: 0.07})
    cl = _model_scorer(env)._copy_leak()
    assert cl["copy_leak_pass"] is True and cl["same_view_copy_gain_spread_ok"] is True
    assert cl["same_view_copy_gain_flag"] is False


def test_f3_done_from_training_does_not_skip_scoring(model_run):
    """Finding 3: a run dir always holds training's DONE; the scorer skipped it and never scored."""
    assert (model_run / "DONE").is_file() and (model_run / "scores.json").is_file()
    assert R.seed_state(model_run) == "done"
    for sh in ("slurm/onion_eval/score.sh", "slurm/minicandi/train.sh"):
        text = (REPO / sh).read_text()
        assert '-f "$OUT/SCORED"' in text and '-f "$OUT/DONE"' not in text, sh


def test_f6_undefined_noise_floor_makes_headline_1_undefined():
    """Finding 6 repro: (E=.2, V=.5, N=.1), (E=.45, V=.5, N=Undefined), (E=.2, V=.5, N=.1) gave
    per_assay A = 0.75 with no reason."""
    recs = [dict(assay="A", cell="c0", E_sp=.2, V_sp=.5, N=.1),
            dict(assay="A", cell="c1", E_sp=.45, V_sp=.5, N=Mx.Undefined("halves unreadable")),
            dict(assay="A", cell="c2", E_sp=.2, V_sp=.5, N=.1)]
    m = S.metric(recs, S.recovered_fn(), {"A"}, n_boot=0)
    assert m["per_assay"]["A"] is None and m["macro"] is None
    assert "halves unreadable" in m["per_assay_reason"]["A"] and m["reason"]
    recs[1]["V_sp"] = Mx.Undefined("no other group")
    recs[1]["N"] = 0.1
    assert "no other group" in S.metric(recs, S.recovered_fn(), {"A"}, n_boot=0)["per_assay_reason"]["A"]
    # a target WITHOUT halves is left out (finding 11), not counted as undefined
    recs[1].update(V_sp=.5, N=Mx.Undefined("no halves"), has_halves=False)
    m = S.metric(recs, S.recovered_fn(), {"A"}, n_boot=0)
    assert m["per_assay"]["A"] == pytest.approx(1 - (0.2 - 0.1) / (0.5 - 0.1))


def test_f10_missing_scorable_assay_is_listed_and_voids_the_headline():
    recs = [dict(assay="A", cell="c0", x=1.0), dict(assay="A", cell="c1", x=2.0)]
    m = S.metric(recs, S._mean_of("x"), {"A", "B"}, n_boot=0)
    assert m["macro"] == 1.5 and m["missing_scorable"] == ["B"]
    m = S.metric(recs, S._mean_of("x"), {"A", "B"}, n_boot=0, require_all=True)
    assert m["macro"] is None and "B" in m["reason"] and m["missing_scorable"] == ["B"]


def test_f4_selection_writes_denominators(model_run, env):
    sel = _strict(model_run / "scores.json")["selection"]
    den = sel["denominators"]
    assert den["uncertainty_marginal_pit_deviation"] > 0
    rel = den["correction_baseline_minus_floor_relative_by_assay"]
    assert set(rel) == set(env["ev"]["scorable"]["correction"])
    assert den["correction_baseline_minus_floor_relative_min"] == pytest.approx(min(rel.values()))
    # the review's input: metrics.uncertainty_gap_closed([0.01], [1e-9]) explodes; the rule escalates
    assert Mx.uncertainty_gap_closed([0.01], [1e-9]) < -1e6
    s = {"selection": dict(sel, uncertainty_gap_closed=-9999999.0,
                           denominators=dict(den, uncertainty_marginal_pit_deviation=1e-9))}
    seed = R.load_seed(_write(env["root"] / "f4.json", s))
    probs, _ = R.screen_terms([seed])
    assert {k for k, _ in probs} == {"uncertainty"}


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def test_f16_control_scored_apart(model_run, env):
    s = _strict(model_run / "scores.json")
    loao = s["specific_variance_recovered_above_noise"]["counts_log1p"]["per_target"]
    assert all(v["assay"] != "chipseq-control" for v in loao.values())
    pc = s["predicted_control"]
    assert pc["regime"] == "control" and pc["n_targets"] == 1
    assert pc["count_crps"]["per_assay"]["chipseq-control"] is not None
    assert env["res"]["marginal_nb"]["predicted_control"]["n_targets"] == 0


def test_f9_tables_checked_against_evalset_and_checkpoint(env, tmp_path):
    with pytest.raises(S.ScoreRefused, match="trained on tables"):
        S.Scorer(env["evalset"], main="model", model=Trivial(), config=CFG,
                 run_tables_sha256="0" * 64, **KW)
    ev = json.loads(env["evalset"].read_text())
    p = env["evalset"].parent / "evalset_badsha.json"
    p.write_text(json.dumps(dict(ev, tables_sha256="0" * 64)))
    with pytest.raises(S.ScoreRefused, match="changed since"):
        S.Scorer(p, main="marginal_nb", config=CFG, **KW)
    p.write_text(json.dumps({k: v for k, v in ev.items() if k != "tables_sha256"}))
    with pytest.raises(S.ScoreRefused, match="no tables_sha256"):
        S.Scorer(p, main="marginal_nb", config=CFG, **KW)
    run = tmp_path / "rung1_x" / "seed0"
    run.mkdir(parents=True)
    assert S.main(["--run", str(run), "--evalset", str(env["evalset"]), "--out", str(run)]) == 2


def test_f9_blind_ledger_and_repeat_reason(env, capsys):
    """Finding 9: a second blind pass with --force silently overwrote the first, with no record."""
    t = env["tables"]
    bdir = env["root"] / "eval" / "synth_blind"
    ev = E.build(t, bdir / "evalset_blind.json", n_regions=1, n_fit_regions=1, region_bins=256,
                 include_blind=True)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["split"] == "blind" and x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"] = [], []
    (bdir / "evalset_blind.json").write_text(json.dumps(ev))
    out = bdir / "baselines" / "marginal_nb" / "seed0" / "blind"
    args = ["--baseline", "marginal_nb", "--evalset", str(bdir / "evalset_blind.json"),
            "--split", "blind", "--allow-blind", "--n-boot", "0"]
    # the pinned layout: a blind pass writes into .../blind, and a val eval set is refused
    assert S.main(args + ["--out", str(out.parent)]) == 2
    assert S.main(["--baseline", "marginal_nb", "--evalset", str(env["evalset"]), "--split",
                   "blind", "--allow-blind", "--out", str(out)]) == 2
    assert S.main(args + ["--out", str(out)]) == 0
    s = _strict(out / "scores.json")
    assert s["split"] == "blind" and s["n_targets"] == 1
    led = _tsv(bdir / "BLIND_PASSES.tsv")
    assert [r["status"] for r in led] == ["started", "scored"]
    assert led[0]["run"] == "baseline:marginal_nb" and led[0]["evalset_sha256"] == s["evalset_sha256"]
    assert led[0]["tables_sha256"] == tables_sha256(t) and led[0]["git_sha"]
    capsys.readouterr()
    assert S.main(args + ["--out", str(out), "--force"]) == 2
    assert "--repeat-blind-reason" in capsys.readouterr().err
    assert len(_tsv(bdir / "BLIND_PASSES.tsv")) == 2
    assert S.main(args + ["--out", str(out), "--force", "--repeat-blind-reason",
                          "node died mid-write"]) == 0
    led = _tsv(bdir / "BLIND_PASSES.tsv")
    assert len(led) == 4 and led[-1]["repeat_reason"] == "node died mid-write"


@pytest.fixture(scope="module")
def halves_env(tmp_path_factory):
    """Tables where one H3K4me3 validation target has no halves on disk (finding 11 repro: the
    scorer read pr1/pr2 of every target and raised)."""
    root = tmp_path_factory.mktemp("halves")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL[:6], blind_tracks=BLIND)
    rows = _tsv(t / "measurements.tsv")
    v = [r for r in rows if r["in_eic"] == "val" and r["arm_knob"] == "" and r["assay"] == "H3K4me3"]
    v[0]["has_pseudoreps"] = "0"
    d = Path(v[0]["counts_path"][3:]).parent
    (d / "pr1.h5").unlink()
    (d / "pr2.h5").unlink()
    with open(t / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=1, n_fit_regions=1, region_bins=256, seed=0)
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"], ev["replicate_pairs"] = [], []
    out.write_text(json.dumps(ev))
    return {"tables": t, "evalset": out, "victim": v[0]["measurement_id"], "root": root}


def test_f11_target_without_halves_is_counted_not_raised(halves_env):
    s = S.score(halves_env["evalset"], halves_env["root"] / "pa", baseline="position_average",
                config=CFG, **KW)
    h1 = s["specific_variance_recovered_above_noise"]["counts_log1p"]
    assert s["n_targets_without_halves"] == 1 and h1["targets_without_halves"] == [halves_env["victim"]]
    pt = h1["per_target"][halves_env["victim"]]
    assert pt["N"] is None and pt["has_halves"] is False and pt["E_sp"] is not None
    # H3K4me3 is still scored from its two targets with halves... but it is no longer scorable
    assert "H3K4me3" in h1["non_scorable_per_assay"]
    assert s["predicted_over_replicate_variance"]["per_assay_reason"]


def test_f11_no_halves_pilot_mode(env, monkeypatch):
    """`--no-halves`: the pilot's store: tables. No half is read; headline 1 without a noise floor
    goes under a key the selection rule cannot mistake for the headline, and is refused there."""
    sc = S.Scorer(env["evalset"], main="position_average", config=CFG, no_halves=True, **KW)
    real = sc.ds.read

    def read(mid, chrom, start, L=None, which="counts"):
        assert which == "counts", f"read {which} in --no-halves mode"
        return real(mid, chrom, start, L, which)
    monkeypatch.setattr(sc.ds, "read", read)
    s = sc.run()
    assert s["noise_floor"] == "none_pilot" and s["n_pairs"] == 0
    assert s["specific_variance_recovered_above_noise"]["counts_log1p"]["macro"] is None
    assert "--no-halves" in s["specific_variance_recovered_above_noise"]["counts_log1p"]["reason"]
    assert s[S.PILOT_H1_KEY]["macro"] is not None
    assert s["selection"]["imputation_gap_closed"] is None and s["selection"]["imputation_reason"]
    assert s["pit_deviation_pooled"]["macro"] is not None
    # F3 (DEFECTS item 3): the uncertainty term over every assay, under a pilot-only key
    pu = s[S.PILOT_UNC_KEY]
    assert pu["value"] is not None and pu["marginal_pit_deviation"] > 0 and pu["assays"]
    st = s["selection"]["headline1_per_assay_status"]
    assert st and all(not v["floor_exists"] and not v["in_imputation_term"]
                      and v["fallback"] == S.FALLBACK_PILOT for v in st.values())


# ------------------------------------------------------------------ end to end: train, score, select

E2E = {"rung": 2, "context_bins": 64, "batch_size": 4, "param_budget": 2.0e4, "windows_seen": 8,
       "lr": 1e-3, "covariate_injection": "film", "mask_mode": "token", "dna": "own",
       "scramble_covariates": False, "scramble_all": False, "count_transform": "offset",
       "qc_on_input": False, "two_step": False, "base_depth": "native", "average_input": False,
       "control_input": True, "predict_control": True}


def test_end_to_end_train_score_compare(env, tmp_path):
    """Train rung 2 for a few windows via the train CLI (the run dir then holds training's DONE),
    score each run via the scorer CLI, then `select_rule.py compare` on three seeds per arm: the
    path that finding 3 broke ("0 scored seeds")."""
    from minicandi import train as T
    ev = json.loads(env["evalset"].read_text())
    ev["imputation_targets"] = [x for x in ev["imputation_targets"]
                                if x["regime"] == "leave_one_assay_out"]
    ev["correction_pairs"] = [p for p in ev["correction_pairs"] if not p["subset_pair"]][:2]
    ev["replicate_pairs"], ev["control_targets"] = [], []
    evp = env["evalset"].parent / "evalset_e2e.json"
    evp.write_text(json.dumps(ev))
    runs = tmp_path / "runs"
    arms = {"rung2_base": dict(E2E, simpler_than_base=None),
            "rung2_dna_off": dict(E2E, dna="off", simpler_than_base=True)}
    for arm, cfg in arms.items():
        cp = tmp_path / f"{arm}.json"
        cp.write_text(json.dumps(cfg))
        for k in range(3):
            run = runs / arm / f"seed{k}"
            assert T.main(["--config", str(cp), "--tables", str(env["tables"]), "--seed", str(k),
                           "--out", str(run), "--log-every", "1000"]) in (0, None)
            assert (run / "DONE").is_file() and not (run / "SCORED").exists()
            assert S.main(["--run", str(run), "--evalset", str(evp), "--out", str(run),
                           "--n-boot", "0", "--guardrail-windows", "4"]) == 0
            assert (run / "SCORED").is_file()
    rows = R.table(runs, "rung2_")
    assert len(rows) == 6 and {r["state"] for r in rows} == {"done"}
    res = R.compare(runs / "rung2_base", runs / "rung2_dna_off", None,
                    settlements=runs / "settlements.tsv")
    assert res["verdict"] in R.VERDICTS, res
    assert _tsv(runs / "settlements.tsv")[0]["verdict"] == res["verdict"]


# ------------------------------------------------------------------ F3: pilot defects

def test_f3_guardrail_windows_are_drawn_per_mask_seed(env, monkeypatch):
    """DEFECTS item 9: every mask seed used the first N reconstruction items (one target's windows),
    so the copy gain's spread missed window-sampling noise. Now each seed draws its own windows."""
    from onion_eval import guardrail as G
    sc = _model_scorer(env, guardrail_windows=6)
    seen = {}

    def fake(model, batches, seed=0):
        seen[seed] = [(m["measurements_out"], m["chrom"], m["start_bin"])
                      for b in batches for m in b["meta"]]
        return _fake_guardrail(G, {0: 0.01, 1: 0.02, 2: 0.03}, {0: True, 1: True, 2: True})(model, batches, seed)
    monkeypatch.setattr(G, "copy_leak", fake)
    cl = sc._copy_leak()
    assert set(seen) == {0, 1, 2} and all(len(v) == 6 for v in seen.values())
    assert seen[0] != seen[1] and seen[1] != seen[2]
    assert cl["window_draw"] == S.GUARD_DRAW and cl["n_reconstruction_items"] > 6
    assert cl["same_view_copy_gain_relative_spread"] == pytest.approx(0.02)
    items0, n0 = S.guardrail_items(sc.ds, 6, 0)
    assert S.guardrail_items(sc.ds, 6, 0)[0] == items0                  # deterministic per seed
    assert [it["item"] for it in items0] == sorted(it["item"] for it in items0)
    assert all(it["task"] == "reconstruction" for it in items0)
    assert len({it["targets"][0] for s_ in (0, 1, 2) for it in S.guardrail_items(sc.ds, 6, s_)[0]}) > 1


def test_f3_headline1_status_per_assay(model_run, env):
    """DEFECTS item 5: per assay, whether the floor exists, whether headline 1 is defined, whether
    the assay is in the composite's imputation term, and the fallback (none is ever computed)."""
    s = _strict(model_run / "scores.json")
    sel = s["selection"]
    st = sel["headline1_per_assay_status"]
    assert st == s["specific_variance_recovered_above_noise"]["counts_log1p"]["per_assay_status"]
    sc = set(env["ev"]["scorable"]["imputation"])
    assert sel["imputation_assays_in_term"] == sorted(sc) and sel["imputation_term_definition"]
    for a, v in st.items():
        assert v["in_imputation_term"] == (a in sc) and v["floor_exists"]
        if a in sc:
            assert v["headline1_defined"] and v["fallback"] == S.FALLBACK_NONE_NEEDED
    one = st["H3K27me3"]                     # one validation target: no shared/specific split
    assert one["n_targets"] == 1 and not one["headline1_defined"] and one["value"] is None
    assert "1 validation target" in one["reason"] and one["fallback"] == S.FALLBACK_NON_SCORABLE
    assert R.h1_status(s) == st


def test_f3_pair_only_baseline_skips_without_pairs(env, tmp_path, capsys):
    """DEFECTS item 4: depth-only scaling is defined on pairs only; with no pair the job writes
    SKIPPED and no scores.json, instead of a table of NaN rows."""
    ev = dict(env["ev"], correction_pairs=[], replicate_pairs=[])
    p = env["root"] / "eval" / "synth" / "evalset_nopairs.json"
    p.write_text(json.dumps(ev))
    out = tmp_path / "depth_only_scaling" / "seed0"
    assert S.main(["--baseline", "depth_only_scaling", "--evalset", str(p), "--out", str(out),
                   "--n-boot", "0"]) == 0
    assert "skipped:" in capsys.readouterr().out
    assert (out / S.SKIPPED).is_file() and not (out / "scores.json").exists()
    assert R.seed_state(out) == "skipped"
    # with pairs it still scores
    assert "skipped" not in env["res"]["depth_only_scaling"]


class ConvReader(MiniCandi):
    """Reads its count input and the DNA (a fixed random conv over log1p counts)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(36, 36, 5, generator=g) * 0.05, requires_grad=False)

    def forward(self, b):
        x = torch.log1p(b["x_counts"].double()).permute(0, 2, 1)
        h = torch.nn.functional.conv1d(x, self.w.double(), padding=2).permute(0, 2, 1)
        dna = (b["x_dna"].double().reshape(h.shape[0], h.shape[1], -1) == 1).double().mean(-1)
        mu = torch.exp(0.5 * torch.tanh(h) + dna[..., None]) * (2.0 ** b["c_out"][..., 0] / 2 ** 24)[:, None, :]
        return {"n": torch.full_like(mu, 4.0), "mu": mu.clamp_min(1e-3)}


SKIP_KEYS = {"wall_seconds", "timing_seconds", "infer_batch", "region_cache", "torch_threads"}


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP_KEYS:
                yield from _leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def max_score_difference(a: dict, b: dict) -> float:
    """Max |a - b| over the numeric leaves; every non-numeric leaf must be equal."""
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    assert set(la) == set(lb), set(la) ^ set(lb)
    worst = 0.0
    for k, v in la.items():
        w = lb[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and w is not None:
            worst = max(worst, abs(float(v) - float(w)))
        else:
            assert v == w, (k, v, w)
    return worst


def test_f3_region_cache_and_inference_batch_keep_every_score(env):
    """DEFECTS item 6: the region cache (one whole-region read per measurement, tiles are slices)
    and a larger inference batch change no score."""
    common = dict(main="model", model=ConvReader(), config=CFG,
                  run_tables_sha256=tables_sha256(env["tables"]), n_boot=30, guardrail_windows=8,
                  log=lambda *a: None)
    old = S.Scorer(env["evalset"], region_cache=False, **common)
    reads = []
    real = old._ds_read
    old.ds.read = lambda *a, **k: (reads.append(a), real(*a, **k))[1]
    a = old.run()
    new = S.Scorer(env["evalset"], region_cache=True, infer_batch=16, **common)
    reads_new = []
    real_new = new._ds_read
    new._ds_read = lambda *a, **k: (reads_new.append(a), real_new(*a, **k))[1]
    b = new.run()
    assert max_score_difference(a, b) <= 1e-9
    assert len(reads_new) < len(reads)          # 2 tiles per region here; 16 on the pilot
    assert b["infer_batch"] == 16 and b["region_cache"] is True and b["timing_seconds"]["model_forward"] >= 0
