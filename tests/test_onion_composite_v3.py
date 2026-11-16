"""CM (PI ruling 2026-11-16): composite version 3 -- the correction term is the oracle-floor gap
closed on the headline pairs with no duplicate-keeping side.

Synthetic tables (the oracle-floor fixture: the single-end views keep duplicates), the three pair
baselines, the oracle reference. Checks: version 3 equals the stored-records key
`correction_gap_closed_oracle_no_nodedup_side`; version 2 is unchanged (the oracle_floor `all`
term, the default of the python API); a version 2 score (no composite keys in scores.json) reads
under the version 3 table and the arm summary with no rescoring; score.py writes the version keys
when it finds a reference; versions never mix."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

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
    for n in ("position_average", "depth_only_scaling", "pseudoreplicate_floor"):
        S.score(out, bdir / n / "seed0", baseline=n, config=CFG, **KW)
    OF.compute(out, bdir / "oracle_floor" / "seed0", log=lambda *a: None)
    return {"root": root, "evalset": out, "bdir": bdir}


def _p(env, name="position_average"):
    return env["bdir"] / name / "seed0" / "scores.json"


#: the baseline fixture has no imputation term; stored seeds get these (per seed) so that a
#: composite is defined, as in a scored run of version 2
TERMS = ((0.44, 0.60), (0.42, 0.58), (0.47, 0.61))


def _runs(env, tmp_path, n=3):
    """n seeds of one arm: copies of a version 2 score (no composite keys) with defined terms."""
    runs = tmp_path / "runs"
    for k in range(n):
        d = runs / "rung1_base" / f"seed{k}"
        d.mkdir(parents=True)
        for f in ("scores.json", "pairs_per_region.npz", "SCORED"):
            shutil.copy(env["bdir"] / "position_average" / "seed0" / f, d / f)
        s = json.loads((d / "scores.json").read_text())
        s["selection"]["imputation_gap_closed"], s["selection"]["uncertainty_gap_closed"] = TERMS[k]
        (d / "scores.json").write_text(json.dumps(s))
    return runs


def test_v3_equals_the_no_nodedup_side_key(env, tmp_path):
    p = _runs(env, tmp_path, 1) / "rung1_base" / "seed0" / "scores.json"
    s = json.loads(p.read_text())
    t = OF.seed_terms(p, s)
    want = t["block"]["no_nodedup_side"]["gap_closed_oracle"]
    assert want is not None
    v3 = R.load_seed(p, composite_version=3)
    assert v3["composite_version"] == 3 and v3["scorer_keys"]["composite_version"] == 3
    assert v3["goals"]["correction"] == want
    assert v3["correction_gap_closed_oracle_no_nodedup_side"] == want
    assert v3["denominators"]["correction"] == t["block"]["no_nodedup_side"]["denominator_relative_min"]
    g = R.screen_terms([v3])[1][0]
    c = g["clipped_goals"]
    assert c["imputation"] == 0.44 and c["uncertainty"] == 0.60
    assert g["composite"] == 0.5 * 0.44 + 0.25 * R._clip(want) + 0.25 * 0.60


def test_v2_unchanged_and_the_python_default(env):
    p = _p(env)
    s = json.loads(p.read_text())
    t = OF.seed_terms(p, s)
    v2, dflt = R.load_seed(p, composite_version=2), R.load_seed(p)
    assert v2["composite_version"] == dflt["composite_version"] == 2
    for k in ("goals", "composite", "denominators", "goal_reasons", "denominator_names"):
        assert v2[k] == dflt[k]
    assert v2["goals"]["correction"] == t["value"]
    assert v2["denominators"]["correction"] == t["denominator_relative_min"]
    assert v2["denominator_names"]["correction"] == "correction_depth_only_minus_oracle_relative_min"
    with pytest.raises(ValueError):
        R.load_seed(p, composite_version=4)


def test_versions_never_mix(env):
    p = _p(env)
    v2, v3 = R.load_seed(p, composite_version=2), R.load_seed(p, composite_version=3)
    with pytest.raises(R.Refused, match="version 3"):
        R.check_composite_versions([v2, v3])
    with pytest.raises(R.Refused, match="composite_version"):
        R.check_scorer_keys([(v2["path"], v2["scorer_keys"]), (v3["path"], v3["scorer_keys"])])


def test_version2_score_reads_under_the_v3_table_and_summary(env, tmp_path, capsys):
    assert "composite_version" not in json.loads(_p(env).read_text())   # scored before the reference
    runs = _runs(env, tmp_path)
    rows = R.table(runs, composite_version=3)
    assert rows and all(r["composite_version"] == 3 for r in rows)
    c3, c2 = [], []
    for k, r in enumerate(rows):
        p = runs / "rung1_base" / f"seed{k}" / "scores.json"
        v3 = R.screen_terms([R.load_seed(p, composite_version=3)])[1][0]
        v2 = R.screen_terms([R.load_seed(p, composite_version=2)])[1][0]
        assert r["composite"] == r["composite_v3"] == v3["composite"] is not None
        assert r["composite_v2"] == v2["composite"] is not None
        c3.append(v3["composite"])
        c2.append(v2["composite"])
        assert r["correction_raw"] == r["correction_gap_closed_oracle_no_nodedup_side"]
        assert r["correction_clipped"] == R._clip(r["correction_raw"])
    assert R.main(["table", "--runs", str(runs)]) == 0               # CLI default: version 3
    out = capsys.readouterr().out.splitlines()
    head = out[0].split("\t")
    assert {"composite_v2", "composite_v3", "correction_gap_closed_oracle_all"} <= set(head)
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
    assert float(row[h.index("imputation_raw.mean")]) == pytest.approx((0.44 + 0.42 + 0.47) / 3,
                                                                       abs=1e-4)


def test_score_writes_the_version_keys_with_a_reference(env):
    from onion_eval import score as S
    out = env["bdir"] / "position_average_v3" / "seed0"
    s = S.score(env["evalset"], out, baseline="position_average", config=CFG, **KW)
    assert s["composite_version"] == 3
    disk = json.loads((out / "scores.json").read_text())
    v3 = R.screen_terms([R.load_seed(out / "scores.json", composite_version=3)])[1][0]
    v2 = R.screen_terms([R.load_seed(out / "scores.json", composite_version=2)])[1][0]
    assert disk["composite_version"] == 3
    assert disk["composite_v3"]["composite"] == v3["composite"]
    assert disk["composite_v2"]["composite"] == v2["composite"]
    assert disk["composite_v3"]["correction_subset"] == "no_nodedup_side"
    assert disk["composite_v3"]["terms_raw"]["correction"] == v3["raw_goals"]["correction"]
    # the old half-read composite under `selection` is untouched
    old = json.loads(_p(env).read_text())
    assert disk["selection"] == old["selection"]
    shutil.rmtree(out.parent)
