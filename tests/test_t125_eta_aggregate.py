"""t125 part 1 — `tools/t125/eta/aggregate.py`: every scored eta run -> class tables with seed
spread, the readings of handoff Phase 7 part 1, the depth law, the explosion counts, grid.md.

Written from the plan (chunk t125-P1f), handoff Phase 7 and plan/ETA_OFFSET_AND_H_DESIGN.md
"Scoring". The runs are synthetic `scores.json` / `law.json` files written here from the record
schema pinned in chunk t125-P1e (t118's records plus mse_log1p_*); every expected number below is
worked out by hand from the formulas in `_crps`.

Hand formula (trained/score crps_all of one pair record):
    BASE[rung, model] + TRACK[track] + PAIR[i] + SEEDSTEP[rung, model] * seed
with the pairs of a track averaged first, then the tracks of a class (macro), per seed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eta import aggregate as ea  # noqa: E402
from eta import pairs as ep  # noqa: E402

MANIFEST = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
COV = ROOT / "tools" / "t118" / "covariates.tsv"

TRACK_CLASS = {"C07M20": "narrow", "C07M29": "narrow", "C19M16": "DNase"}
TRACK = {"C07M20": 0.0, "C07M29": 1.0, "C19M16": 2.0}
PAIR = (0.0, 0.2)
BASE = {("A", "real"): 1.0, ("A", "nocov"): 1.5, ("Aold", "real"): 1.2, ("Aold", "nocov"): 1.6}
SEEDSTEP = {("A", "real"): 0.1, ("A", "nocov"): 0.15, ("Aold", "real"): 0.1,
            ("Aold", "nocov"): 0.1}
GS = ("all", "C07M20")
APPROX = dict(abs=1e-9)


def _pair(track: str, i: int) -> dict:
    src, tgt = f"{track}__base__base", f"{track}__arm{i}__l{i}"
    return {"track": track, "cell": track[:3], "assay": "X", "mark_class": TRACK_CLASS[track],
            "arm_src": "base", "level_src": "base", "arm_tgt": f"arm{i}", "level_tgt": f"l{i}",
            "source_pid": src, "target_pid": tgt, "direction": "base_to_arm",
            "knob_combo": f"base:base->arm{i}:l{i}", "counts_identical": i == 1,
            "pval_identical": False}


def _rec(kind: str, ev: str, pair: dict, crps, split: bool = False) -> dict:
    """One record of the pinned schema; every metric derived from crps_all by fixed ratios."""
    r = {"kind": kind, "eval": ev, **pair, "cov_src_pid": pair["source_pid"],
         "cov_tgt_pid": pair["target_pid"], "n_blacklisted": 0,
         "n_crps_nonfinite": 0 if crps is not None else 3,
         "n_all": 100, "n_nonzero": 50, "n_top1": 1}
    c = crps
    r.update({"crps_all": c, "crps_nonzero": None if c is None else 2 * c,
              "crps_top1": None if c is None else 10 * c,
              "spearman_all": 0.5, "spearman_nonzero": 0.4, "spearman_top1": 0.1,
              "mse_log1p_all": None if c is None else c / 10,
              "mse_log1p_nonzero": None if c is None else c / 5,
              "mse_log1p_top1": None if c is None else c,
              "crps_all_gt_20": None if c is None else c > 20})
    if split:
        r.update({"crps_oracle_scaled_all": None if c is None else 0.8 * c,
                  "scale_error_all": None if c is None else 0.2 * c, "c_star_all": 1.0})
    return r


def _crps(rung, model, track, i, seed):
    return BASE[(rung, model)] + TRACK[track] + PAIR[i] + SEEDSTEP[(rung, model)] * seed


def _tracks(g):
    return list(TRACK) if g == "all" else [g]


def _write_run(runs_dir: Path, rung: str, g: str, model: str, seed: int,
               overrides: dict | None = None, law: bool = True) -> Path:
    """scores.json (+ law.json) of one run. `overrides` maps (track, i) -> crps_all of that
    trained/score record."""
    name = ep.run_name(rung, g, model, seed)
    d = runs_dir / name
    d.mkdir(parents=True)
    recs = []
    for t in _tracks(g):
        for i in (0, 1):
            p = _pair(t, i)
            c = _crps(rung, model, t, i, seed)
            if overrides and (t, i) in overrides:
                c = overrides[(t, i)]
            recs.append(_rec("trained", "score", p, c, split=True))
            recs.append(_rec("trained", "val", p, 0.5))
            recs.append(_rec("shuffle", "score", p, None if c is None else c + 1.0))
        recs.append({**_rec("swap", "score", _pair(t, 0), 0.3),
                     "swap_median_abs_log_ratio": 0.05})
        # the trained base <-> depth pair: a scored trained record AND its depthlaw copy
        dp = {**_pair(t, 0), "arm_tgt": "depth", "level_tgt": "15M",
              "target_pid": f"{t}__depth__15M"}
        crps_dp = 1.0
        if overrides and (t, "depth") in overrides:
            crps_dp = overrides[(t, "depth")]
        tr = _rec("trained", "score", dp, crps_dp, split=True)
        recs.append(tr)
        dl = {**tr, "kind": "depthlaw", "depth_source": "trained",
              "depth_log2_ratio_true": 0.5, "depth_log2_scale_pred": 0.5}
        for k in ("crps_oracle_scaled_all", "scale_error_all", "c_star_all"):
            dl.pop(k, None)
        recs.append(dl)
    run = {"run_name": name, "rung": rung, "g": g,
           "g_version": "across" if g == "all" else "per_track", "space": "counts",
           "model": model, "seed": seed, "steps": 10, "best_step": 10, "val_nll": 1.0}
    (d / "scores.json").write_text(json.dumps({"run": run, "records": recs,
                                               "t125": {"law_scope": None}}))
    if law:
        lrecs = []
        for t in _tracks(g):
            for j, (true, preds) in enumerate(((1.0, (1.1, 1.0, 0.9)),
                                               (-1.0, (-0.8, -0.8, -0.8)))):
                p = {**_pair(t, 0), "arm_src": "depth", "level_src": f"a{j}",
                     "arm_tgt": "depth", "level_tgt": f"b{j}",
                     "source_pid": f"{t}__depth__a{j}", "target_pid": f"{t}__depth__b{j}"}
                lr = _rec("law", "score", p, 2.0)
                lrecs.append(lr)
                pred = preds[seed] if model == "real" else true + 0.5
                lrecs.append({**lr, "kind": "depthlaw", "depth_source": "law",
                              "depth_log2_ratio_true": true, "depth_log2_scale_pred": pred})
        (d / "law.json").write_text(json.dumps({"run": run, "records": lrecs,
                                                "t125": {"law_scope": "depth"}}))
    (d / "SCORE_DONE").write_text("x\n")
    return d


OLD_COLS = ("rung g_version space model kind mark_class metric variant mean seed_wobble "
            "per_seed_0 per_seed_1 per_seed_2 n_tracks").split()


def _write_old_tsv(path: Path) -> Path:
    rows = [
        ("A", "across", "counts", "real", "trained", "narrow", "crps_all", "all", 2.0, 0.3),
        ("A", "across", "counts", "real", "trained", "narrow", "crps_top1", "all", 30.0, 1.0),
        ("A", "across", "counts", "nocov", "trained", "narrow", "crps_all", "all", 2.5, 0.1),
        ("A", "across", "pval", "real", "trained", "narrow", "crps_all", "all", 99.0, 0.1),
        ("A", "across", "counts", "real", "law", "narrow", "crps_all", "all", 99.0, 0.1),
        ("A", "across", "counts", "real", "trained", "narrow", "crps_all", "nonidentical",
         99.0, 0.1),
    ]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(OLD_COLS)
        for r in rows:
            w.writerow(list(r) + [r[8] - r[9] / 2, r[8], r[8] + r[9] / 2, 2])
    return path


def _write_old_json(path: Path) -> Path:
    """The old grid's results.json: per_pair trained/score records with the run fields flat."""
    recs = []
    for seed in (0, 1, 2):
        for t, base in (("C07M20", 1.0), ("C07M29", 2.0)):
            step = 0.1 if t == "C07M20" else 0.0
            r = _rec("trained", "score", _pair(t, 0), 1.0, split=True)
            r.update({"crps_oracle_scaled_all": base + step * seed, "scale_error_all": 0.5,
                      "run_name": f"A_all_counts_real_s{seed}", "rung": "A", "g": "all",
                      "g_version": "across", "space": "counts", "model": "real", "seed": seed})
            if t == "C07M29" and seed == 2:
                r["crps_all"] = 30.0
            recs.append(r)
    path.write_text(json.dumps({"per_pair": recs}))
    return path


@pytest.fixture(scope="module")
def agg(tmp_path_factory):
    root = tmp_path_factory.mktemp("agg")
    runs = root / "runs"
    for rung in ("A", "Aold"):
        for g in GS:
            for model in ("real", "nocov"):
                for seed in (0, 1, 2):
                    ov = None
                    if (rung, g, model) == ("A", "C07M20", "real"):
                        ov = {1: {("C07M20", 1): 30.0},
                              2: {("C07M20", 0): None, ("C07M20", "depth"): 25.0}}.get(seed)
                    law = not (rung, g, model, seed) == ("Aold", "C07M20", "nocov", 2)
                    _write_run(runs, rung, g, model, seed, ov, law=law)
    old_tsv = _write_old_tsv(root / "old.tsv")
    old_json = _write_old_json(root / "old.json")
    out = root / "agg"
    rc = ea.main([str(MANIFEST), str(COV), str(runs), str(out), "--old-grid-tsv", str(old_tsv),
                  "--old-grid-json", str(old_json)])
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    return {"root": root, "runs": runs, "out": out, "res": res, "old_tsv": old_tsv}


def _reading(res, reading, rung, gv, mc, metric, model="real"):
    hits = [r for r in res["readings"] if r["reading"] == reading and r["rung"] == rung
            and r["g_version"] == gv and r["mark_class"] == mc and r["metric"] == metric
            and r.get("model", "real") == model]
    assert len(hits) == 1, (reading, rung, gv, mc, metric, hits)
    return hits[0]


def _summary_rows(out: Path) -> list[dict]:
    with (out / "results_summary.tsv").open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ---------------------------------------------------------------------------------------------
# files, missing runs
# ---------------------------------------------------------------------------------------------


def test_outputs_written(agg):
    for f in ("results.json", "results_summary.tsv", "pairs_summary.tsv", "depth_law.tsv",
              "grid.md", "summary_data.json"):
        assert (agg["out"] / f).is_file(), f


def test_missing_runs_listed_not_fatal(agg):
    res = agg["res"]
    expected = [t["run_name"] for t in ep.tasks()]
    assert len(expected) == 432
    assert len(res["runs_present"]) == 24
    assert set(res["runs_missing"]) == set(expected) - set(res["runs_present"])
    assert "B_all_counts_real_s0" in res["runs_missing"]
    assert "A_all_counts_real_s0" not in res["runs_missing"]
    assert res["runs_without_law"] == ["Aold_C07M20_counts_nocov_s2"]


# ---------------------------------------------------------------------------------------------
# aggregation: pair -> track -> class, seeds
# ---------------------------------------------------------------------------------------------


def test_class_mean_and_wobble_by_hand(agg):
    rows = _summary_rows(agg["out"])
    assert list(rows[0]) == OLD_COLS
    assert {r["space"] for r in rows} == {"counts"}

    def row(rung, model, metric, mc="narrow", gv="across", kind="trained", variant="all"):
        hit = [r for r in rows if (r["rung"], r["g_version"], r["model"], r["kind"],
                                   r["mark_class"], r["metric"], r["variant"])
               == (rung, gv, model, kind, mc, metric, variant)]
        assert len(hit) == 1
        return hit[0]

    # narrow, g across, A real: tracks C07M20 (pairs 1.0+0.1s, 1.2+0.1s, depth pair 1.0)
    # and C07M29 (2.0+0.1s, 2.2+0.1s, depth pair 1.0).
    # per track: (3.2+0.2s)/3 and (5.2+0.2s)/3; class: (8.4+0.4s)/6
    r = row("A", "real", "crps_all")
    per_seed = [(8.4 + 0.4 * s) / 6 for s in (0, 1, 2)]
    for k, v in enumerate(per_seed):
        assert float(r[f"per_seed_{k}"]) == pytest.approx(v, **APPROX)
    assert float(r["mean"]) == pytest.approx(sum(per_seed) / 3, **APPROX)
    assert float(r["seed_wobble"]) == pytest.approx(0.8 / 6, **APPROX)
    assert r["n_tracks"] == "2"
    # the split parts and the MSE follow the same rule
    assert float(row("A", "real", "crps_oracle_scaled_all")["mean"]) == \
        pytest.approx(0.8 * sum(per_seed) / 3, **APPROX)
    assert float(row("A", "real", "scale_error_all")["mean"]) == \
        pytest.approx(0.2 * sum(per_seed) / 3, **APPROX)
    assert float(row("A", "real", "mse_log1p_all")["mean"]) == \
        pytest.approx(sum(per_seed) / 30, **APPROX)
    # nonidentical drops pair 1 (counts_identical): C07M20 (1.0+0.1s, 1.0), C07M29 (2.0+0.1s, 1.0)
    ni = row("A", "real", "crps_all", variant="nonidentical")
    assert float(ni["per_seed_0"]) == pytest.approx((2.0 / 2 + 3.0 / 2) / 2, **APPROX)
    # spearman is aggregated (reported only), the swap key on swap records
    assert float(row("A", "real", "spearman_all")["mean"]) == pytest.approx(0.5)
    assert float(row("A", "real", "swap_median_abs_log_ratio", kind="swap")["mean"]) == \
        pytest.approx(0.05)
    assert row("A", "real", "crps_all", kind="shuffle")
    assert row("A", "real", "crps_all", kind="law")


# ---------------------------------------------------------------------------------------------
# readings
# ---------------------------------------------------------------------------------------------


def _per_seed(rung, model):
    b, st = BASE[(rung, model)], SEEDSTEP[(rung, model)]
    out = []
    for s in (0, 1, 2):
        t1 = ((b + st * s) + (b + 0.2 + st * s) + 1.0) / 3
        t2 = ((b + 1.0 + st * s) + (b + 1.2 + st * s) + 1.0) / 3
        out.append((t1 + t2) / 2)
    return out


def _mw(vals):
    return sum(vals) / len(vals), max(vals) - min(vals)


def test_beatstwin_bar_uses_the_larger_wobble(agg):
    m_real, w_real = _mw(_per_seed("A", "real"))
    m_noc, w_noc = _mw(_per_seed("A", "nocov"))
    r = _reading(agg["res"], "beatstwin", "A", "across", "narrow", "crps_all")
    assert r["value"] == pytest.approx(m_noc - m_real, **APPROX)       # positive = real better
    assert r["bar"] == pytest.approx(2 * max(w_real, w_noc), **APPROX)
    assert w_noc > w_real                                               # the larger one is nocov
    assert r["seed_wobble"] == pytest.approx(max(w_real, w_noc), **APPROX)
    assert r["met"] is (r["value"] > r["bar"])
    # by hand: class = (4b + 4.4 + 4 st s)/6 -> value (2.0 + 0.2)/6, bar 2 x 0.2 (nocov's);
    # t118's bar (2 x wobble_real = 0.267) would have counted this gap, the plan's does not
    assert r["value"] == pytest.approx(2.2 / 6, **APPROX)
    assert r["bar"] == pytest.approx(0.4, **APPROX)
    assert r["met"] is False
    # the note that the twin keeps the offset travels with the reading
    assert "offset" in r["components"]["note"]
    # the old grid's own twin gap is beside it (2.5 - 2.0), never subtracted from it
    assert r["components"]["oldgrid_gap"] == pytest.approx(0.5)


def test_vs_oldgrid_and_aold_against_old_a(agg):
    res = agg["res"]
    m_a, w_a = _mw(_per_seed("A", "real"))
    r = _reading(res, "vs_oldgrid", "A", "across", "narrow", "crps_all")
    assert r["value"] == pytest.approx(2.0 - m_a, **APPROX)            # positive = t125 better
    assert r["bar"] == pytest.approx(2 * max(0.3, w_a), **APPROX)
    assert r["met"] is (r["value"] > r["bar"])
    top = _reading(res, "vs_oldgrid", "A", "across", "narrow", "crps_top1")
    assert top["value"] == pytest.approx(30.0 - 10 * m_a, **APPROX)
    assert top["bar"] == pytest.approx(2 * max(1.0, 10 * w_a), **APPROX)
    assert top["met"] is True                                           # 15.3 > 2.67
    # Aold compares with the old grid's A
    m_o, w_o = _mw(_per_seed("Aold", "real"))
    ro = _reading(res, "vs_oldgrid", "Aold", "across", "narrow", "crps_all")
    assert ro["value"] == pytest.approx(2.0 - m_o, **APPROX)
    assert ro["components"]["oldgrid_rung"] == "A"
    # no old-grid row (DNase, per_track, MSE) -> no reading
    assert not [x for x in res["readings"] if x["reading"] == "vs_oldgrid"
                and (x["mark_class"] == "DNase" or x["g_version"] == "per_track"
                     or x["metric"].startswith("mse"))]


def test_vs_oldspread_is_aold_minus_a(agg):
    m_a, w_a = _mw(_per_seed("A", "real"))
    m_o, w_o = _mw(_per_seed("Aold", "real"))
    r = _reading(agg["res"], "vs_oldspread", "A", "across", "narrow", "crps_all")
    assert r["value"] == pytest.approx(m_o - m_a, **APPROX)            # positive = new spread better
    assert r["bar"] == pytest.approx(2 * max(w_a, w_o), **APPROX)
    assert r["value"] == pytest.approx(0.8 / 6, **APPROX)
    assert r["met"] is False                                            # 0.133 < 2 x 0.133
    assert not [x for x in agg["res"]["readings"]
                if x["reading"] == "vs_oldspread" and x["rung"] != "A"]


def test_mean_or_spread_parts(agg):
    res = agg["res"]
    r = _reading(res, "mean_or_spread", "A", "across", "narrow", "beatstwin")
    assert r["value"] is None and r["bar"] is None and r["met"] is None
    parts = r["components"]["parts"]
    for k in ("crps_all", "crps_oracle_scaled_all", "scale_error_all", "mse_log1p_all",
              "mse_log1p_top1", "crps_top1"):
        assert k in parts
    m_real, w_real = _mw(_per_seed("A", "real"))
    m_noc, w_noc = _mw(_per_seed("A", "nocov"))
    p = parts["crps_oracle_scaled_all"]
    assert p["gain"] == pytest.approx(0.8 * (m_noc - m_real), **APPROX)
    assert p["bar"] == pytest.approx(2 * 0.8 * max(w_real, w_noc), **APPROX)
    assert p["gain_exceeds_bar"] is (p["gain"] > p["bar"])
    assert parts["mse_log1p_all"]["gain"] == pytest.approx((m_noc - m_real) / 10, **APPROX)
    # against the old grid: the split comes from the old json (narrow, across, A):
    # per seed (1.0+0.1s + 2.0)/2 -> mean 1.55, wobble 0.1; the old grid has no MSE
    og = _reading(res, "mean_or_spread", "A", "across", "narrow", "vs_oldgrid")
    q = og["components"]["parts"]["crps_oracle_scaled_all"]
    assert q["other"] == pytest.approx(1.55, **APPROX)
    assert q["other_wobble"] == pytest.approx(0.1, **APPROX)
    assert q["gain"] == pytest.approx(1.55 - 0.8 * m_real, **APPROX)
    assert og["components"]["parts"]["mse_log1p_all"]["gain"] is None
    sp = _reading(res, "mean_or_spread", "A", "across", "narrow", "vs_oldspread")
    assert sp["components"]["parts"]["scale_error_all"]["gain"] == \
        pytest.approx(0.2 * (_mw(_per_seed("Aold", "real"))[0] - m_real), **APPROX)


def test_no_correlation_in_readings(agg):
    assert not [r for r in agg["res"]["readings"] if "spearman" in str(r["metric"])]


def test_one_seed_reading_never_met(tmp_path):
    runs = tmp_path / "runs"
    for model in ("real", "nocov"):
        _write_run(runs, "A", "all", model, 0)
    ea.aggregate(MANIFEST, COV, runs, tmp_path / "agg")
    res = json.loads((tmp_path / "agg" / "results.json").read_text())
    r = _reading(res, "beatstwin", "A", "across", "narrow", "crps_all")
    assert r["bar"] is None and r["met"] is False
    assert "seed" in r["components"]["met_reason"]


# ---------------------------------------------------------------------------------------------
# explosions and non-finite CRPS
# ---------------------------------------------------------------------------------------------


def test_explosion_and_nonfinite_counts(agg):
    res = agg["res"]
    # per_track g (C07M20), A real: seed 1 pair 1 at 30.0; seed 2 the depth pair at 25.0
    # (its depthlaw copy is NOT counted again) and pair 0 non-finite
    r = _reading(res, "n_pairs_crps_gt_20", "A", "per_track", "narrow", "crps_all")
    assert r["value"] == 2
    nf = _reading(res, "n_pairs_crps_nonfinite", "A", "per_track", "narrow", "crps_all")
    assert nf["value"] == 1
    for reading in ("n_pairs_crps_gt_20", "n_pairs_crps_nonfinite"):
        assert _reading(res, reading, "A", "across", "narrow", "crps_all")["value"] == 0
        assert _reading(res, reading, "A", "per_track", "narrow", "crps_all",
                        model="nocov")["value"] == 0
    ex = {(e["rung"], e["g_version"], e["mark_class"], e["model"], e["kind"], e["eval"]): e
          for e in res["explosions"]}
    e = ex[("A", "per_track", "narrow", "real", "trained", "score")]
    # 3 seeds x (2 pairs + 1 depth pair) trained/score records
    assert (e["n_records"], e["n_crps_gt_20"], e["n_crps_nonfinite"]) == (9, 2, 1)
    # shuffle records are their own kind: 31.0 at seed 1; seed 2 pair 0 non-finite
    s = ex[("A", "per_track", "narrow", "real", "shuffle", "score")]
    assert (s["n_crps_gt_20"], s["n_crps_nonfinite"]) == (1, 1)
    assert ("A", "per_track", "narrow", "real", "depthlaw", "score") not in ex
    assert ex[("A", "across", "narrow", "real", "law", "score")]["n_records"] == 3 * 2 * 2


def test_old_grid_explosions_from_json(agg):
    og = agg["res"]["oldgrid_explosions"]
    hit = [e for e in og if (e["rung"], e["g_version"], e["mark_class"]) ==
           ("A", "across", "narrow")]
    assert len(hit) == 1 and hit[0]["n_crps_gt_20"] == 1 and hit[0]["n_records"] == 6


# ---------------------------------------------------------------------------------------------
# depth law
# ---------------------------------------------------------------------------------------------


def test_depthlaw_reading(agg):
    r = _reading(agg["res"], "depthlaw", "A", "across", "all", "depth_scale_rel_error")
    # law pairs: true 1.0, pred (1.1, 1.0, 0.9) -> mean err 0; true -1, pred -0.8 -> |2^0.2 - 1|
    assert r["value"] == pytest.approx(2 ** 0.2 - 1, **APPROX)
    assert r["bar"] == 0.10
    assert r["met"] is False                                   # met = value <= bar
    assert r["seed_wobble"] == pytest.approx(0.0, abs=1e-12)
    c = r["components"]
    assert c["n_pairs"] == 3 * 2                                # 3 tracks x 2 depth->depth pairs
    assert c["trained_value"] == pytest.approx(0.0, abs=1e-12)  # base<->depth, pred == true
    assert c["trained_n_pairs"] == 3
    assert c["nocov_value"] == pytest.approx(2 ** 0.5 - 1, **APPROX)


def test_depth_law_tsv(agg):
    with (agg["out"] / "depth_law.tsv").open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert list(rows[0])[:9] == ["rung", "g_version", "model", "seed", "depth_source",
                                 "source_pid", "target_pid", "depth_log2_ratio_true",
                                 "depth_log2_scale_pred"]
    # law depthlaw: per run (#tracks x 2); trained depthlaw: per run #tracks; 23 runs with law
    n_law = sum(len(_tracks(g)) * 2 for rung in ("A", "Aold") for g in GS
                for _m in ("real", "nocov") for _s in (0, 1, 2)) - 1 * 2
    n_tr = sum(len(_tracks(g)) for rung in ("A", "Aold") for g in GS
               for _m in ("real", "nocov") for _s in (0, 1, 2))
    assert sum(r["depth_source"] == "law" for r in rows) == n_law
    assert sum(r["depth_source"] == "trained" for r in rows) == n_tr


# ---------------------------------------------------------------------------------------------
# pairs summary, grid.md, summary data
# ---------------------------------------------------------------------------------------------


def test_pairs_summary(agg):
    with (agg["out"] / "pairs_summary.tsv").open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    cols = ["rung", "g_version", "model", "seed", "track", "mark_class", "source_pid",
            "target_pid", "knob_combo", "crps_all", "crps_top1", "crps_oracle_scaled_all",
            "scale_error_all", "mse_log1p_all", "mse_log1p_top1"]
    assert [c for c in rows[0] if c in cols] == cols
    # trained/score only: per run #tracks x 3 records
    assert len(rows) == sum(len(_tracks(g)) * 3 for _r in ("A", "Aold") for g in GS
                            for _m in ("real", "nocov") for _s in (0, 1, 2))
    hit = [r for r in rows if r["run_name"] == "A_all_counts_real_s1"
           and r["source_pid"] == "C07M29__base__base" and r["target_pid"] == "C07M29__arm1__l1"]
    assert len(hit) == 1 and float(hit[0]["crps_all"]) == pytest.approx(1.0 + 1.0 + 0.2 + 0.1)


def test_grid_md(agg):
    md = (agg["out"] / "grid.md").read_text()
    assert "one g across tracks" in md and "one g per track" in md
    for mc in ("DNase", "narrow"):
        assert f"### {mc}" in md
    for row in ("row 1", "row 2", "old grid row 1", "old grid row 2", "twin row 1",
                "twin row 2"):
        assert f"| {row}" in md, row
    assert "Aold" in md
    m_a, w_a = _mw(_per_seed("A", "real"))
    assert f"{m_a:.4g} ± {w_a:.4g}" in md
    assert "2 ± 0.3" in md                                     # the old grid's A cell
    assert "oracle-scaled" in md and "scale error" in md
    assert "not directly comparable" in md
    assert "CRPS > 20" in md and "non-finite" in md
    assert "Depth law" in md
    assert "Nothing in this file is a verdict" in md


def test_summary_data(agg):
    sd = json.loads((agg["out"] / "summary_data.json").read_text())
    for k in ("cells", "readings", "depth_law", "explosions", "runs_present", "runs_missing"):
        assert k in sd
    assert sd["runs_present"] == 24
    cell = [c for c in sd["cells"] if (c["rung"], c["g_version"], c["mark_class"], c["metric"],
                                       c["source"]) == ("A", "across", "narrow", "crps_all",
                                                        "t125 real")]
    assert len(cell) == 1
    assert cell[0]["mean"] == pytest.approx(_mw(_per_seed("A", "real"))[0], **APPROX)


def test_empty_old_grid_tsv_is_not_an_error(tmp_path, agg):
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    ea.aggregate(MANIFEST, COV, agg["runs"], tmp_path / "agg", old_grid_tsv=empty)
    res = json.loads((tmp_path / "agg" / "results.json").read_text())
    assert not [r for r in res["readings"] if r["reading"] == "vs_oldgrid"]
    assert [r for r in res["readings"] if r["reading"] == "vs_oldspread"]
    assert (tmp_path / "agg" / "grid.md").is_file()
