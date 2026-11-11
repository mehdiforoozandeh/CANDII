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
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402
from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import score as S  # noqa: E402
from onion_eval import select_rule as R  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"))
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
        assert set(s[k]) == {"counts_log1p", "signal_arcsinh"}
    h1 = s["specific_variance_recovered_above_noise"]["counts_log1p"]
    assert set(h1) >= {"macro", "per_assay", "ci", "per_target"}
    loao = [x["target"] for x in env["ev"]["imputation_targets"]
            if x["regime"] == "leave_one_assay_out"]
    assert set(h1["per_target"]) == set(loao)
    assert all(np.isfinite(v["level"]) for v in h1["per_target"].values())
    sig = s["specific_variance_recovered_above_noise"]["signal_arcsinh"]
    assert sig["macro"] is None and "signal_from_counts" in sig["reason"]
    assert set(s["copy_leak"]) >= set(("hidden_input_max_abs_change", "unavailable_input_max_abs_change",
                                       "same_view_copy_gain", "copy_leak_pass"))
    assert s["copy_leak"]["hidden_input_max_abs_change"] == 0.0
    assert len(s["copy_leak"]["same_view_copy_gain_relative_by_mask_seed"]) == 3
    assert set(s["error_budget"]) >= set(S.BUDGET_KEYS)
    assert set(s["selection"]) >= {"space", "imputation_gap_closed", "correction_gap_closed",
                                   "uncertainty_gap_closed", "composite"}
    assert s["selection"]["space"] == "counts_log1p"
    assert s["regime"] == "leave_one_assay_out" and s["split"] == "val"
    assert s["n_targets"] == len(loao) and s["n_pairs"] == len(env["ev"]["correction_pairs"])
    assert (model_run / "DONE").is_file()
    suff = np.load(model_run / "suffstats.npz")
    assert any(k.startswith("gram/leave_one_assay_out/H3K4me3/") for k in suff.files)
    assert any(k.startswith("pit/") for k in suff.files) and any(k.startswith("pair/") for k in suff.files)
    # a second call skips on DONE
    assert S.main(["--run", str(model_run), "--evalset", str(env["evalset"]),
                   "--out", str(model_run)]) == 0


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
    p = env["root"] / "eval" / "synth" / "evalset_blind.json"
    p.write_text(json.dumps(ev))
    with pytest.raises(PermissionError):
        S.Scorer(p, main="marginal_nb", config=CFG).jobs()


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
