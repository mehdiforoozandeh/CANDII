"""SP2c (Amendment 10, pinned interface M): `--sub <SUB>` on select_rule compare/table and on
arm_report, and the comparability keys (evalset, tables, scorer version, scorer settings, refs).
Without `--sub` every output is byte-identical to the tools before SP2c (digest tests)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import arm_report as AR  # noqa: E402
from onion_eval import select_rule as R  # noqa: E402

EV, TB = "E" * 64, "T" * 64


def _m(v):
    return {"macro": v, "per_assay": {"H3K4me3": v}, "non_scorable_per_assay": {"ATAC-seq": v / 4},
            "ci": [None, None]}


def scores(v, goals=(0.3, 0.4, 0.5), **top):
    """A scores.json that both select_rule and arm_report read; `top` overrides top-level keys."""
    imp, cor, unc = goals
    s = {"evalset_sha256": EV, "tables_sha256": TB, "split": "val",
         "regime": "leave_one_assay_out", "noise_floor": "pseudoreplicate_halves",
         "evalset": "/x/eval/eic_v1/evalset.json",
         "selection": {"space": "counts_log1p", "imputation_gap_closed": imp,
                       "correction_gap_closed": cor, "uncertainty_gap_closed": unc,
                       "denominators": {"uncertainty_marginal_pit_deviation": 0.1,
                                        "correction_baseline_minus_floor_relative_min": 0.2}},
         "copy_leak": {"copy_leak_pass": True, "part1_wiring_pass": True,
                       "same_view_copy_gain_spread_ok": True, "gate_on_copy_gain": False,
                       "same_view_copy_gain_flag": False}}
    for path, _ in AR.HEADLINE:
        if path[0] in AR.DERIVED_SCALE_ERROR:
            continue
        if len(path) == 2:
            s.setdefault(path[0], {})[path[1]] = _m(v)
        else:
            s[path[0]] = _m(v)
    s["correction_crps_model_oracle_scaled"] = _m(v - 0.1)
    s["correction_gap_closed_vs_depth_only_never_trained"]["per_knob"] = {"crop36": v, "se": 2 * v}
    s["error_budget"] = {**{k: v for k in AR.BUDGET_KEYS},
                         "per_assay": {"H3K4me3": {k: v + 1 for k in AR.BUDGET_KEYS}}}
    acc = {}
    for path in AR.IMPUTATION:
        if len(path) == 2:
            acc.setdefault(path[0], {})[path[1]] = _m(v / 2)
        else:
            acc[path[0]] = _m(v / 2)
    s["accessibility_only"] = acc
    s.update(top)
    return s


def put(d: Path, s: dict, scored=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps(s), encoding="utf-8")
    if scored:
        (d / "SCORED").write_text("x")


GOALS = {"rung1_base": [(0.30, 0.40, 0.50), (0.31, 0.41, 0.50), (0.30, 0.40, 0.51)],
         "rung1_x": [(0.40, 0.40, 0.50), (0.41, 0.41, 0.50), (0.40, 0.40, 0.51)]}


def build(root: Path, sub: str = "", **top) -> tuple[Path, Path]:
    """runs/<arm>/seed<k>[/<sub>]/scores.json for two arms of 3 seeds, baselines/<name>/seed0."""
    runs, base = root / "runs", root / "baselines"
    for arm, goals in GOALS.items():
        for k, g in enumerate(goals):
            d = runs / arm / f"seed{k}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "DONE").write_text("x")
            (d / "run.json").write_text(json.dumps({"config": {"simpler_than_base": False}}))
            put(d / sub if sub else d, scores(0.2 + 0.1 * k + (0.5 if arm == "rung1_x" else 0), g,
                                              **top))
    for name, v in (("marginal_nb", 0.1), ("position_average", 0.25), ("depth_only_scaling", 1.2)):
        put(base / name / "seed0", scores(v, (0.0, 0.0, 0.0)), scored=False)
    return runs, base


def _run(fn, argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(argv)
    return rc, buf.getvalue()


def outputs(root: Path, sub_args: tuple = ()) -> dict:
    """Every command output (stdout or the written tsv) on the fixture tree, as text."""
    runs, base = build(root, sub_args[1] if sub_args else "")
    out = {}
    rc, txt = _run(R.main, ["compare", "--incumbent", str(runs / "rung1_base"), "--challenger",
                            str(runs / "rung1_x"), "--simpler", "none", "--settlements",
                            str(root / "s.tsv"), *sub_args])
    out["compare"] = f"{rc}\n{txt}"
    rc, txt = _run(R.main, ["table", "--runs", str(runs), *sub_args])
    out["table"] = f"{rc}\n{txt}"
    for cmd in ("headline", "dev-evals", "accessibility", "per-knob"):
        p = root / f"{cmd}.tsv"
        rc, _ = _run(AR.main, [cmd, "--runs", str(runs), "--baselines", str(base), "--out", str(p),
                               *sub_args])
        out[cmd] = f"{rc}\n{p.read_text(encoding='utf-8')}"
    p = root / "ss.tsv"
    rc, _ = _run(AR.main, ["shared-specific", "--runs", str(runs), "--arms",
                           "rung1_base,rung1_x,marginal_nb", "--baselines", str(base), "--out",
                           str(p), *sub_args])
    out["shared-specific"] = f"{rc}\n{p.read_text(encoding='utf-8')}"
    return out


def digests(root: Path) -> dict:
    return {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in outputs(root).items()}


# Computed with the tools at 57f387b (before SP2c) on the same fixture tree.
BEFORE = {
    "compare": "0b09fb464ce1e8dd",
    "table": "4a350bd3a739f1cc",
    "headline": "e11331948a27c470",
    "dev-evals": "f1a4f42508628913",
    "accessibility": "b8f0b34a676269d1",
    "per-knob": "32586a3a2aa0b7b1",
    "shared-specific": "3bfcb1ba2ba409b2",
}


def test_without_sub_outputs_are_byte_identical(tmp_path):
    assert digests(tmp_path) == BEFORE


def test_sub_p16_reads_the_subdirectory(tmp_path):
    a = outputs(tmp_path / "a")
    b = outputs(tmp_path / "b", ("--sub", "p16"))
    assert a == b                                   # the same scores, one level down
    assert "\trunning\t" not in b["table"] and "\tdone\t" in b["table"]
    # an arm with scores only at the top level is not scored under --sub p16
    runs, _ = build(tmp_path / "c")
    assert {r["state"] for r in R.table(runs, sub="p16")} == {"running"}
    with pytest.raises(R.Missing):
        R.load_arm(runs / "rung1_base", "p16")


def test_sub_baselines_fall_back_to_seed0(tmp_path):
    """A p16 baseline lives in its own baselines dir at seed0/ (baselines_p16/<name>/seed0)."""
    runs, base = build(tmp_path, "p16")
    rows, _ = AR.headline(runs, "", base, sub="p16")
    assert {r["arm"] for r in rows if r["kind"] == "baseline"} == {
        "marginal_nb", "position_average", "depth_only_scaling"}
    put(base / "marginal_nb" / "seed0" / "p16", scores(7.0, (0.0, 0.0, 0.0)))
    rows, _ = AR.headline(runs, "", base, sub="p16")
    assert [r["count_crps.mean"] for r in rows if r["arm"] == "marginal_nb"] == [pytest.approx(7.0)]


def test_bad_sub_refused():
    with pytest.raises(SystemExit):
        AR.main(["headline", "--runs", "x", "--baselines", "y", "--out", "z", "--sub", "../p16"])
    with pytest.raises(SystemExit):
        R.main(["table", "--runs", "x", "--sub", "a/b"])


V2 = {"scorer_version": 2, "scorer_settings": {"device": "cuda", "infer_batch": 32, "tf32": False,
                                               "amp": False},
      "pair_regions_n": 16, "refs": {"dir": "/b", "sha256_depth_only": "d" * 64,
                                     "sha256_floor": "f" * 64}}
MIXES = {
    "evalset_sha256": {"evalset_sha256": "X" * 64},
    "tables_sha256": {"tables_sha256": "X" * 64},
    "scorer_version": {"scorer_version": 1},
    "scorer_settings.device": {"scorer_settings": {**V2["scorer_settings"], "device": "cpu"}},
    "scorer_settings.infer_batch": {"scorer_settings": {**V2["scorer_settings"], "infer_batch": 8}},
    "refs.sha256_depth_only": {"refs": {**V2["refs"], "sha256_depth_only": "x" * 64}},
    "refs.sha256_floor": {"refs": {**V2["refs"], "sha256_floor": "x" * 64}},
}


def _mix(tmp_path, key):
    runs, base = build(tmp_path, "p16", **V2)
    p = runs / "rung1_x" / "seed1" / "p16" / "scores.json"
    s = json.loads(p.read_text())
    s.update(MIXES[key])
    p.write_text(json.dumps(s))
    return runs, base


@pytest.mark.parametrize("key", sorted(MIXES))
def test_a_mix_of_each_key_is_refused(tmp_path, key):
    runs, base = _mix(tmp_path, key)
    with pytest.raises(R.Refused, match=key.split(".")[0]):
        R.compare(runs / "rung1_base", runs / "rung1_x", "none", settlements=tmp_path / "s.tsv",
                  sub="p16")
    rc, txt = _run(R.main, ["compare", "--incumbent", str(runs / "rung1_base"), "--challenger",
                            str(runs / "rung1_x"), "--simpler", "none", "--settlements",
                            str(tmp_path / "s.tsv"), "--sub", "p16"])
    assert rc == 2 and txt.startswith("REFUSED")
    assert not (tmp_path / "s.tsv").exists()
    for cmd in ("headline", "dev-evals", "accessibility", "per-knob"):
        assert AR.main([cmd, "--runs", str(runs), "--baselines", str(base), "--out",
                        str(tmp_path / "o.tsv"), "--sub", "p16"]) == 2, cmd
    assert AR.main(["shared-specific", "--runs", str(runs), "--arms", "rung1_base,rung1_x",
                    "--out", str(tmp_path / "o.tsv"), "--sub", "p16"]) == 2
    assert not (tmp_path / "o.tsv").exists()


def test_uniform_v2_arms_with_v1_baselines_are_accepted(tmp_path):
    """A scorer_version 1 baseline is comparable with version 2 runs as a reference directory."""
    runs, base = build(tmp_path, "p16", **V2)
    res = R.compare(runs / "rung1_base", runs / "rung1_x", "none", settlements=tmp_path / "s.tsv",
                    sub="p16")
    assert res["verdict"] == "CHALLENGER"
    rows, _ = AR.headline(runs, "", base, sub="p16")
    assert len(rows) == 5


def test_a_v1_score_is_never_an_arm_beside_v2(tmp_path):
    runs, base = build(tmp_path, "p16", **V2)
    for k in range(3):                      # a whole v1 arm (no new keys) against a v2 arm
        p = runs / "rung1_base" / f"seed{k}" / "p16" / "scores.json"
        s = json.loads(p.read_text())
        for key in ("scorer_version", "scorer_settings", "refs", "pair_regions_n"):
            s.pop(key)
        p.write_text(json.dumps(s))
    with pytest.raises(R.Refused, match="scorer_version"):
        R.compare(runs / "rung1_base", runs / "rung1_x", "none", settlements=tmp_path / "s.tsv",
                  sub="p16")
    with pytest.raises(AR.Refused, match="scorer_version"):
        AR.headline(runs, "", base, sub="p16")


@pytest.mark.parametrize("key", ["scorer_settings.device", "scorer_settings.infer_batch"])
def test_allow_mix_of_a_setting(tmp_path, key):
    """--allow-mix accepts a mix of the device or the batch only, never of the other keys."""
    runs, base = _mix(tmp_path, key)
    assert _run(R.main, ["compare", "--incumbent", str(runs / "rung1_base"), "--challenger",
                         str(runs / "rung1_x"), "--simpler", "none", "--settlements",
                         str(tmp_path / "s.tsv"), "--sub", "p16", "--allow-mix", key])[0] == 0
    assert AR.main(["headline", "--runs", str(runs), "--baselines", str(base), "--out",
                    str(tmp_path / "o.tsv"), "--sub", "p16", "--allow-mix", key]) == 0
    with pytest.raises(SystemExit):
        R.main(["compare", "--incumbent", "a", "--challenger", "b", "--allow-mix",
                "scorer_version"])
