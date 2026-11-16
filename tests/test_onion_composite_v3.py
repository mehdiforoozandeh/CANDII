"""CM (PI ruling 2026-11-16; foreman review): composite version 3 -- the correction term is the
oracle-floor gap closed on the headline pairs with no duplicate-keeping side, computed at report
time only.

Synthetic tables (the oracle-floor fixture: the single-end views keep duplicates), the pair
baselines, the oracle reference. Checks: version 3 equals the stored-records key
`correction_gap_closed_oracle_no_nodedup_side` and is the default everywhere; version 2 is
unchanged and reachable explicitly; the v3 term is undefined with a reason when an assay with such
a pair does not enter it; a version 2 score reads under the version 3 table and the arm summary
with no rescoring, the numerator and denominator columns describe the printed term; `compare` and
`check-done` run under version 3; settlements.tsv rows carry the version and never mix; with an
oracle reference present the scorer's outputs are bit-identical to those of the pre-change tree
52bfde4."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import oracle_floor as OF  # noqa: E402
from onion_eval import select_rule as R  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (7, "chipseq-control"))
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=20, guardrail_windows=4, log=lambda *a: None)
BASE_COMMIT = "52bfde4"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    from onion_eval import score as S
    root = tmp_path_factory.mktemp("cv3")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    mp = t / "measurements.tsv"
    with open(mp, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        cols, rows = rd.fieldnames, list(rd)
    for r in rows:                         # the se views keep duplicates (a nodedup side)
        if r["arm_knob"] == "se":
            r["dedup"] = "0"
    with open(mp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    nt = [p for p in ev["correction_pairs"] if p["never_trained"] and not p["subset_pair"]]
    ev["correction_pairs"] = nt[:12]
    out.write_text(json.dumps(ev))
    bdir = root / "eval" / "synth" / "baselines"
    for n in ("marginal_nb", "position_average", "depth_only_scaling", "pseudoreplicate_floor"):
        S.score(out, bdir / n / "seed0", baseline=n, config=CFG, **KW)
    OF.compute(out, bdir / "oracle_floor" / "seed0", log=lambda *a: None)
    return {"root": root, "evalset": out, "bdir": bdir}


def _p(env, name="position_average"):
    return env["bdir"] / name / "seed0" / "scores.json"


#: the baseline fixture has no imputation term; stored seeds get these (per seed) so that a
#: composite is defined, as in a scored run of version 2
TERMS = ((0.44, 0.60), (0.42, 0.58), (0.47, 0.61))


def _runs(env, tmp_path, n=3, arm="rung1_base", shift=0.0, edit=None):
    """n seeds of one arm: copies of a version 2 score (no composite keys) with defined terms."""
    runs = tmp_path / "runs"
    for k in range(n):
        d = runs / arm / f"seed{k}"
        d.mkdir(parents=True)
        for f in ("scores.json", "pairs_per_region.npz", "SCORED"):
            shutil.copy(env["bdir"] / "position_average" / "seed0" / f, d / f)
        s = json.loads((d / "scores.json").read_text())
        s["selection"]["imputation_gap_closed"] = TERMS[k][0] + shift
        s["selection"]["uncertainty_gap_closed"] = TERMS[k][1]
        s["copy_leak"] = {"copy_leak_pass": True, "part1_wiring_pass": True}   # as a scored run
        s["selection"].setdefault("denominators", {})["uncertainty_marginal_pit_deviation"] = 0.05
        if k:                              # the seeds' correction terms differ a little
            z = dict(np.load(d / "pairs_per_region.npz"))
            z["crps_sum_model"] = z["crps_sum_model"] * (1.0 - 0.01 * k)
            np.savez_compressed(d / "pairs_per_region.npz", **z)
        if edit:
            edit(s)
        (d / "scores.json").write_text(json.dumps(s))
    return runs


def test_v3_equals_the_no_nodedup_side_key_and_is_the_default(env, tmp_path):
    p = _runs(env, tmp_path, 1) / "rung1_base" / "seed0" / "scores.json"
    s = json.loads(p.read_text())
    t = OF.seed_terms(p, s)
    b = t["block"]["no_nodedup_side"]
    want = b["gap_closed_oracle"]
    assert want is not None
    assert set(t["evalset_assays_no_nodedup_side"]) == set(b["per_assay"])
    v3, dflt = R.load_seed(p, composite_version=3), R.load_seed(p)
    assert v3["composite_version"] == dflt["composite_version"] == 3
    assert v3["goals"] == dflt["goals"] and v3["scorer_keys"]["composite_version"] == 3
    assert v3["goals"]["correction"] == want == v3["correction_gap_closed_oracle_no_nodedup_side"]
    assert v3["denominators"]["correction"] == b["denominator_relative_min"]
    assert (v3["correction_n_assays"], v3["correction_n_pairs"]) == (b["n_assays"], b["n_pairs"])
    assert (v3["correction_numerator"], v3["correction_denominator"]) == (b["numerator"],
                                                                          b["denominator"])
    g = R.screen_terms([v3])[1][0]
    c = g["clipped_goals"]
    assert c["imputation"] == 0.44 and c["uncertainty"] == 0.60
    assert g["composite"] == 0.5 * 0.44 + 0.25 * R._clip(want) + 0.25 * 0.60


def test_v2_unchanged_when_asked_for(env):
    p = _p(env)
    t = OF.seed_terms(p, json.loads(p.read_text()))
    v2 = R.load_seed(p, composite_version=2)
    assert v2["composite_version"] == 2
    assert v2["goals"]["correction"] == t["value"]
    assert v2["denominators"]["correction"] == t["denominator_relative_min"]
    assert v2["denominator_names"]["correction"] == "correction_depth_only_minus_oracle_relative_min"
    a = t["block"]["all"]
    assert (v2["correction_numerator"], v2["correction_n_pairs"]) == (a["numerator"], a["n_pairs"])
    with pytest.raises(ValueError):
        R.load_seed(p, composite_version=4)


def test_v3_expected_assays_are_the_scorable_ones_with_such_pairs(env, tmp_path):
    p0 = _p(env)
    have = OF.seed_terms(p0, json.loads(p0.read_text()))["evalset_assays_no_nodedup_side"]
    drop = have[0]

    def unscorable(s):                   # not correction-scorable: simply not expected
        s["scorable"]["correction"] = [a for a in s["scorable"]["correction"] if a != drop]
    p = _runs(env, tmp_path / "a", 1, edit=unscorable) / "rung1_base" / "seed0" / "scores.json"
    v3 = R.load_seed(p)
    assert v3["goals"]["correction"] is not None and v3["goal_reasons"]["correction"] == ""
    assert v3["correction_n_assays"] == v3["correction_n_assays_expected"] == len(have) - 1
    # scorable but with no per-pair record: undefined, with the reason
    p = _runs(env, tmp_path / "b", 1) / "rung1_base" / "seed0" / "scores.json"
    z = dict(np.load(p.parent / "pairs_per_region.npz"))
    o = OF.load_oracle(env["bdir"] / "oracle_floor" / "seed0")
    nnd = {str(k) for k, a, n, r in zip(o["pair_ids"], o["assay"], o["nodedup_side"], o["rung"])
           if str(a) == drop and n == 0 and str(r) == ""}
    keep = np.array([str(k) not in nnd for k in z["pair_ids"]])
    np.savez_compressed(p.parent / "pairs_per_region.npz",
                        **{k: (v[keep] if v.ndim and v.shape[0] == keep.size else v) for k, v in z.items()})
    v3 = R.load_seed(p)
    assert v3["goals"]["correction"] is None
    why = v3["goal_reasons"]["correction"]
    assert drop in why and "no per-pair record" in why
    rows = R.table(tmp_path / "b" / "runs")
    assert rows[0]["composite_v3"] is None and rows[0]["composite"] is None


def test_versions_never_mix(env):
    p = _p(env)
    v2, v3 = R.load_seed(p, composite_version=2), R.load_seed(p, composite_version=3)
    with pytest.raises(R.Refused, match="version 3"):
        R.check_composite_versions([v2, v3])
    with pytest.raises(R.Refused, match="composite_version"):
        R.check_scorer_keys([(v2["path"], v2["scorer_keys"]), (v3["path"], v3["scorer_keys"])])


def test_version2_score_reads_under_the_v3_table_and_summary(env, tmp_path, capsys):
    assert "composite_version" not in json.loads(_p(env).read_text())
    runs = _runs(env, tmp_path)
    rows = R.table(runs)
    assert rows and all(r["composite_version"] == 3 for r in rows)
    c3, c2 = [], []
    for k, r in enumerate(rows):
        p = runs / "rung1_base" / f"seed{k}" / "scores.json"
        v3 = R.screen_terms([R.load_seed(p, composite_version=3)])[1][0]
        v2 = R.screen_terms([R.load_seed(p, composite_version=2)])[1][0]
        assert r["composite"] == r["composite_v3"] == v3["composite"] is not None
        assert r["composite_v2"] == v2["composite"] is not None
        assert r["correction_raw"] == r["correction_gap_closed_oracle_no_nodedup_side"]
        b = v3["oracle_block"]["no_nodedup_side"]
        assert (r["correction_numerator"], r["correction_denominator"]) == (b["numerator"],
                                                                            b["denominator"])
        assert (r["correction_n_assays"], r["correction_n_pairs"]) == (b["n_assays"], b["n_pairs"])
        c3.append(v3["composite"])
        c2.append(v2["composite"])
    assert R.main(["table", "--runs", str(runs)]) == 0               # CLI default: version 3
    out = capsys.readouterr().out.splitlines()
    head = out[0].split("\t")
    assert {"composite_v2", "composite_v3", "correction_n_assays", "correction_n_pairs"} <= set(head)
    i = head.index("composite_version")
    assert all(ln.split("\t")[i] == "3" for ln in out[1:])
    assert R.main(["table", "--runs", str(runs), "--composite-version", "2"]) == 0
    out2 = capsys.readouterr().out.splitlines()
    assert all(ln.split("\t")[i] == "2" for ln in out2[1:])
    assert R.main(["arm-summary", "--runs", str(runs)]) == 0
    s = capsys.readouterr().out.splitlines()
    h, row = s[0].split("\t"), s[1].split("\t")
    assert row[h.index("n_seeds")] == "3" and row[h.index("composite_version")] == "3"
    assert float(row[h.index("composite_v3.mean")]) == pytest.approx(sum(c3) / 3, abs=1e-4)
    assert float(row[h.index("composite_v3.range")]) == pytest.approx(max(c3) - min(c3), abs=1e-4)
    assert float(row[h.index("composite_v2.mean")]) == pytest.approx(sum(c2) / 3, abs=1e-4)


def test_compare_and_check_done_under_v3_use_their_own_file(env, tmp_path, capsys):
    runs = _runs(env, tmp_path, arm="rung1_base")
    _runs(env, tmp_path, arm="rung1_better", shift=0.2)
    legacy = runs / "settlements.tsv"
    legacy.write_text("comparison\tverdict\ttime\nc1\tINCUMBENT\tt0\n")
    before = legacy.read_bytes()
    lad = tmp_path / "ladder.json"
    lad.write_text(json.dumps({"comparisons": [
        {"name": "c1", "incumbent": "rung1_base", "challenger": "rung1_better", "simpler": "none",
         "d14_condition_1": True, "verdict": "", "notepad_ref": ""}]}))
    # no version-3 row yet: MISSING, never FAIL and never "does not match (none)"
    assert R.check_rung(runs, 1, lad) == [f"c1 ({R.V3_MISSING})"]
    assert R.main(["check-rung", "--runs", str(runs), "--rung", "1", "--ladder", str(lad)]) == 1
    assert f"MISSING: c1 ({R.V3_MISSING})" in capsys.readouterr().out
    ok, probs = R.done_condition_1(runs, lad)
    assert not ok and all(R.V3_MISSING in x for x in probs if x.startswith("c1"))
    res = R.compare(runs / "rung1_base", runs / "rung1_better", "none", name="c1")
    assert res["composite_version"] == 3 and res["verdict"] == "CHALLENGER", res["reason"]
    v3f = runs / "settlements_v3.tsv"
    rows = list(csv.DictReader(open(v3f), delimiter="\t"))
    assert list(rows[0]) == list(R.SETTLEMENT_COLUMNS_V3) and rows[0]["composite_version"] == "3"
    assert legacy.read_bytes() == before                     # the legacy file is never touched
    assert R.latest_settlements(runs) == {"c1": "CHALLENGER"}
    assert R.latest_settlements(runs, 2) == {"c1": "INCUMBENT"}
    assert R.check_rung(runs, 1, lad) == []
    assert R.main(["check-rung", "--runs", str(runs), "--rung", "1", "--ladder", str(lad)]) == 0
    assert "composite_version: 3" in capsys.readouterr().out
    shutil.copytree(runs / "rung1_better", runs / "final")
    out = R.check_done(runs, lad, baselines=env["bdir"])
    assert out["composite_version"] == 3
    assert out["validation"][0] != "REFUSED", out["validation"]
    assert any("composite_version: 3" in ln for ln in out["validation"][1])
    assert "c1" not in " ".join(out["problems_1"])


def test_v3_compare_refuses_an_arm_without_a_reference_and_writes_nothing(tmp_path):
    from tests import test_onion_eval_select_rule as T
    inc = T.make_arm(tmp_path, "rung1_base", T.BASE)
    ch = T.make_arm(tmp_path, "rung1_x", T.shift(T.BASE, di=0.1))
    with pytest.raises(R.Refused, match="oracle-floor reference"):
        R.compare(inc, ch, "none")
    assert not list(tmp_path.glob("settlements*.tsv"))
    assert R.main(["compare", "--incumbent", str(inc), "--challenger", str(ch), "--simpler",
                   "none"]) == 2
    assert not list(tmp_path.glob("settlements*.tsv"))


LEGACY_RUN = """
import json, sys
from pathlib import Path
from onion_eval import select_rule as R
runs, lad, bl, extra = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
inc, ch = runs / "rung1_base", runs / "rung1_x"
res = R.compare(inc, ch, "none", name="r1_dna", settlements=runs / "settlements.tsv",
                **({"composite_version": 2} if extra else {}))
print("compare", res["verdict"], res.get("composite_version"))
R.main(["check-rung", "--runs", str(runs), "--rung", "1", "--ladder", lad] + extra)
R.main(["check-done", "--runs", str(runs), "--ladder", lad, "--baselines", bl] + extra)
"""


@pytest.mark.parametrize("old_cols", [11, 14])
def test_legacy_settlements_identical_to_the_base_tree(tmp_path, old_cols):
    """Foreman item 1: with --composite-version 2, compare + check-rung + check-done on a legacy
    settlements.tsv (the old 11-column header, or the 14 columns with the clip ones) write and
    print exactly what the 52bfde4 code does."""
    from tests import test_onion_eval_select_rule as T
    old = tmp_path / "old"
    if not _base_tree(old):
        pytest.skip(f"git archive {BASE_COMMIT} not available (a kit without .git)")
    outs = {}
    for tree, name, extra in ((REPO, "new", ["--composite-version", "2"]), (old, "old", [])):
        root = tmp_path / name
        runs = root / "runs"
        lad = T._full_ladder(root, runs)
        T.make_arm(runs, "rung1_x", T.shift(T.BASE, di=0.1))
        T.make_arm(runs, "final", T.BASE)
        bl = root / "baselines"
        T._baselines(bl, T.GOOD_BASELINES)
        cols = list(R.SETTLEMENT_COLUMNS[:old_cols])
        st = runs / "settlements.tsv"
        st.write_text("\t".join(cols) + "\n" + "".join(
            "\t".join([f"r{r}_dna", f"rung{r}_base", f"rung{r}_dna_off", "challenger", "", "", "", "",
                       "", "INCUMBENT", "2026-11-11T00:00:00+00:00"] + [""] * (old_cols - 11)) + "\n"
            for r in range(1, 7)))
        e = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tree / "src"), str(tree / "tools")]))
        p = subprocess.run([sys.executable, "-c", LEGACY_RUN, str(runs), str(lad), str(bl), *extra],
                           env=e, capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr[-2000:]
        text = st.read_text().splitlines()
        # the time of the appended row is the clock's: mask it
        last = text[-1].split("\t")
        last[10] = "<time>"
        outs[name] = (p.stdout.replace(str(root), "<root>"), text[:-1] + ["\t".join(last)],
                      sorted(x.name for x in runs.glob("**/settlements*.tsv")))
    assert outs["new"] == outs["old"]
    assert outs["new"][2] == ["settlements.tsv"]


def _base_tree(dst: Path) -> bool:
    """The tools and src trees of BASE_COMMIT under dst (git archive); False when not available."""
    try:
        a = subprocess.run(["git", "-C", str(REPO), "archive", BASE_COMMIT, "tools", "src"],
                           capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    dst.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(dst)], input=a, check=True)
    return True


SCORE_RUN = """
import json, sys
from pathlib import Path
from onion_eval import score as S
ev, out, run = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(n_boot=20, guardrail_windows=4, log=lambda *a: None)
if run == "-":
    S.score(ev, Path(out), baseline="position_average", config={"context_bins": 128,
            "batch_size": 4}, **kw)
else:
    S.score(ev, Path(out), run=run, **kw)
"""


def test_outputs_bit_identical_to_the_base_tree_with_an_oracle_reference(env, tmp_path):
    """Foreman item 1: with the switch off the scorer writes exactly what 52bfde4 writes, on an
    eval set that HAS an oracle reference (so a version key would have been found)."""
    import torch
    from minicandi import registry as MR
    from minicandi.dataset import tables_sha256
    from onion_eval.evalset import tables_dir_of
    old = tmp_path / "old"
    if not _base_tree(old):
        pytest.skip(f"git archive {BASE_COMMIT} not available (a kit without .git)")
    ev = json.loads(env["evalset"].read_text())
    sha = tables_sha256(tables_dir_of(ev, env["evalset"]))
    cfg = {**json.loads((REPO / "tools/minicandi/configs/rung2/base.json").read_text()),
           "context_bins": 128, "batch_size": 4}
    torch.manual_seed(2)
    m = MR.build_model(cfg)
    run = tmp_path / "run" / "seed0"
    run.mkdir(parents=True)
    torch.save({"state_dict": m.state_dict(), "config": cfg, "param_count": m.param_count(),
                "windows_seen": 0, "tables_sha256": sha, "git_sha": "test"}, run / MR.CKPT_NAME)
    outs = {}
    for tree, name in ((REPO, "new"), (old, "old")):
        e = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tree / "src"), str(tree / "tools")]))
        e.pop("ONION_SCORE_SAME_VIEW", None)
        for what, r in (("pa", "-"), ("model", str(run))):
            d = env["bdir"] / f"cmp_{name}_{what}" / "seed0"     # beside the oracle reference
            p = subprocess.run([sys.executable, "-c", SCORE_RUN, str(env["evalset"]), str(d), r],
                               env=e, capture_output=True, text=True, timeout=1200)
            assert p.returncode == 0, p.stderr[-2000:]
            outs[(name, what)] = d
    assert OF.find_oracle(outs[("new", "pa")] / "scores.json",
                          json.loads((outs[("new", "pa")] / "scores.json").read_text()))
    for what in ("pa", "model"):
        a, b = outs[("new", what)], outs[("old", what)]
        assert sorted(f.name for f in a.iterdir()) == sorted(f.name for f in b.iterdir())
        sa, sb = (json.loads((d / "scores.json").read_text()) for d in (a, b))
        for s in (sa, sb):
            for k in ("wall_seconds", "timing_seconds", "torch_threads"):
                s.pop(k, None)
        assert sa == sb
        for f in ("suffstats.npz", "pairs_per_region.npz"):
            with np.load(a / f) as za, np.load(b / f) as zb:
                assert za.files == zb.files
                for k in za.files:
                    assert za[k].dtype == zb[k].dtype and za[k].tobytes() == zb[k].tobytes(), (f, k)
    for d in outs.values():
        shutil.rmtree(d.parent)
