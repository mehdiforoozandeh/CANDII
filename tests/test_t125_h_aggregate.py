"""t125 part 2 — `tools/t125/hmap/aggregate.py` (h scores -> the part-2 tables of handoff Phase 7)
and `tools/t125/hmap/synth.py` (synthetic controls for the smokes).

Written from the plan (chunk t125-P2g), handoff Phase 7, plan/ETA_OFFSET_AND_H_DESIGN.md "Part 2"
and "Scoring", and the PI decisions in plan/T125_STATUS.md. The h scores are synthetic
`scores.json` files written here in the record schema of `hmap/score.py` (chunk t125-P2f):
push only carries `crps_*` (proper CRPS on P) with `crps_oracle_scaled_all` / `scale_error_all`
and the flag `crps_all_gt_20`; own noise carries `crps_mean_point_*` (the point CRPS of the mean)
with its own split and flag; both carry `crps_log_*` with `crps_log_oracle_scaled_all` /
`crps_log_scale_error_all`, `mse_log1p_*` and the non-finite counts.

Hand formulas. Alone (one record per product): the class value per seed is the mean over the
products of the class (plan), then the mean over seeds and the wobble = max - min over seeds.
Chain (one record per pair): the pairs of a track are averaged first, then the tracks of the class
(macro, as t118), per seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hmap import aggregate as ha  # noqa: E402
from hmap import controls, products, synth, tasks  # noqa: E402
from ladder import synth as lsynth  # noqa: E402

MANIFEST = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
COV = ROOT / "tools" / "t118" / "covariates.tsv"
AP = dict(abs=1e-9)

# ---------------------------------------------------------------------------------------------
# synthetic h scores
# ---------------------------------------------------------------------------------------------

#: alone: pid -> (track, mark_class, offset)
PRODUCTS = {"P1": ("Ta", "narrow", 0.0), "P2": ("Ta", "narrow", 0.2), "P3": ("Tb", "narrow", 0.4),
            "P4": ("Td", "DNase", 0.0)}
#: chain: (track, source, target) -> (mark_class, offset)
PAIRS = {("Ta", "a1", "a2"): ("narrow", 0.0), ("Ta", "a2", "a1"): ("narrow", 0.3),
         ("Ta", "a1", "a3"): ("narrow", 0.6), ("Tb", "b1", "b2"): ("narrow", 0.9),
         ("Td", "d1", "d2"): ("DNase", 0.0)}
F1, F2 = "A_all_counts_real_s0", "A2_all_counts_real_s0"
F_RUNG = {F1: "A", F2: "A2"}


def rec(noise: bool, nat: float, log: float, mse: float, **over) -> dict:
    """A score.py record body: native CRPS `nat` (+10 on top 1 %), split os = nat - 0.05,
    log CRPS `log` (+1 on top 1 %, split os = log - 0.01), MSE `mse` (+2 on top 1 %)."""
    k = "crps_mean_point" if noise else "crps"
    r = {"n_all": 100, "n_nonzero": 60, "n_top1": 1,
         f"{k}_all": nat, f"{k}_nonzero": nat, f"{k}_top1": nat + 10.0,
         "crps_log_all": log, "crps_log_nonzero": log, "crps_log_top1": log + 1.0,
         "mse_log1p_all": mse, "mse_log1p_nonzero": mse, "mse_log1p_top1": mse + 2.0,
         "mse_all": mse, "mse_nonzero": mse, "mse_top1": mse,
         f"n_{k}_nonfinite": 0, "n_crps_log_nonfinite": 0,
         "crps_log_oracle_scaled_all": log - 0.01, "crps_log_scale_error_all": 0.01,
         "crps_log_c_star_all": 0.0, f"{k}_all_gt_20": nat > 20.0}
    if noise:
        r.update({"crps_mean_point_oracle_scaled_all": nat - 0.05,
                  "crps_mean_point_scale_error_all": 0.05, "crps_mean_point_c_star_all": 0.0})
    else:
        r.update({"crps_oracle_scaled_all": nat - 0.05, "scale_error_all": 0.05,
                  "c_star_all": 0.0})
    r.update(over)
    return r


def h_info(variant, rung, noise, source, seed, f_run=None, window=None) -> dict:
    if variant == "formula":
        return {"h_name": f"h_formula_{window}", "spec": f"formula:{window}", "kind": "formula",
                "variant": "formula", "window": window, "rung": None, "noise": 0,
                "source": None, "seed": None}
    tag = "true" if source == "true" else f_run
    return {"h_name": tasks.h_name(variant, rung, noise, tag, seed), "spec": "/x",
            "kind": "learned", "variant": variant, "rung": rung, "noise": int(noise),
            "source": "true" if source == "true" else f"fnb:{f_run}", "seed": seed,
            "fnb_run": None if source == "true" else f"/runs/{f_run}", "step": 5,
            "val_loss": 0.1}


def write_alone(root: Path, info: dict, records: list[dict]) -> None:
    d = root / "alone" / info["h_name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "scores.json").write_text(json.dumps({"mode": "alone", "h": info, "records": records}))
    (d / "ALONE_DONE").write_text("x\n")


def write_chain(root: Path, f_run: str, info: dict, records: list[dict], done=True) -> None:
    """One h's chain scores.json; `done` also writes the f's CHAIN_DONE (score.py writes it once
    every h of the f is scored; a chain without it is not finished)."""
    d = root / "chain" / f_run / info["h_name"]
    d.mkdir(parents=True, exist_ok=True)
    g = f_run.split("_")[1]
    f_info = {"run_name": f_run, "rung": F_RUNG.get(f_run, f_run.split("_")[0]), "g": g,
              "g_version": "across" if g == "all" else "per_track", "space": "counts",
              "model": "real", "seed": 0}
    (d / "scores.json").write_text(json.dumps({"mode": "chain", "f_run": f_info, "h": info,
                                               "records": records}))
    if done:
        with (root / "chain" / f_run / "CHAIN_DONE").open("a") as fh:
            fh.write(info["h_name"] + "\n")


def alone_recs(info, noise, fn) -> list[dict]:
    """fn(pid, offset) -> rec kwargs, or None to leave the product out."""
    out = []
    for pid, (track, mc, off) in PRODUCTS.items():
        kw = fn(pid, off)
        if kw is None:
            continue
        out.append({"h_name": info["h_name"], "pid": pid, "track": track,
                    "assay": "DNase-seq" if mc == "DNase" else "H3K27ac", "mark_class": mc,
                    "arm": "base", "level": "base", "has_control": int(mc != "DNase"),
                    **rec(noise, **kw)})
    return out


def chain_recs(info, f_run, noise, fn) -> list[dict]:
    out = []
    for (track, src, tgt), (mc, off) in PAIRS.items():
        kw = fn((track, src, tgt), off)
        if kw is None:
            continue
        out.append({"track": track, "cell": "c", "assay": "x", "mark_class": mc,
                    "arm_src": "base", "level_src": "base", "arm_tgt": "depth",
                    "level_tgt": "15M", "source_pid": src, "target_pid": tgt,
                    "direction": "base_to_arm", "knob_combo": "k", "counts_identical": False,
                    "pval_identical": False, "f_run_name": f_run, "h_name": info["h_name"],
                    "n_k_mean": 3.0, "n_k_max": 9,
                    "n_capped": 2 if (f_run == F1 and (src, tgt) == ("a1", "a2")) else 0,
                    **rec(noise, **kw)})
    return out


def build_h_root(root: Path) -> Path:
    h_root = root / "h"
    # ---- alone
    fb = h_info("formula", None, 0, None, None, window="bin")
    write_alone(h_root, fb, alone_recs(fb, False, lambda p, o: dict(nat=1.0 + o, log=0.9 + o,
                                                                    mse=0.8 + o)))
    for s in (0, 1, 2):
        hp = h_info("affine", 1, 0, "true", s)
        write_alone(h_root, hp, alone_recs(hp, False, lambda p, o, s=s: dict(
            nat=2.0 + o + 0.1 * s, log=0.5 + o + 0.01 * s, mse=1.0 + o)))
        hn = h_info("affine", 1, 1, "true", s)

        def fn_noise(p, o, s=s):
            kw = dict(nat=3.0 + o + 0.1 * s, log=0.4 + o + 0.03 * s, mse=0.5 + o)
            if p == "P4" and s == 0:
                kw["nat"] = 30.0                              # an explosion (own-noise kind)
            if p == "P1" and s == 2:                          # non-finite native and log bins
                kw.update(crps_mean_point_all=None, crps_mean_point_top1=None,
                          n_crps_mean_point_nonfinite=2, crps_log_all=None, crps_log_top1=None,
                          n_crps_log_nonfinite=1, crps_mean_point_all_gt_20=False)
            return kw
        write_alone(h_root, hn, alone_recs(hn, True, fn_noise))
    # ---- chain, on two f runs
    for f in (F1, F2):
        write_chain(h_root, f, fb, chain_recs(fb, f, False, lambda k, o: dict(
            nat=2.0 + o, log=1.0 + o, mse=1.0 + o)))
        for s in (0, 1, 2):
            hp = h_info("affine", 1, 0, "true", s)

            def fn_push(k, o, s=s, f=f):
                kw = dict(nat=1.0 + o + 0.1 * s, log=0.5 + o + 0.01 * s, mse=1.0 + o)
                if f == F1 and s == 0 and k == ("Ta", "a2", "a1"):
                    kw.update(crps_all=None, crps_top1=None, n_crps_nonfinite=1,
                              crps_oracle_scaled_all=None, scale_error_all=None,
                              crps_all_gt_20=False)
                if f == F1 and s == 1 and k == ("Td", "d1", "d2"):   # a non-finite LOG bin only
                    kw.update(crps_log_all=None, crps_log_top1=None, n_crps_log_nonfinite=1,
                              crps_log_oracle_scaled_all=None, crps_log_scale_error_all=None)
                return kw
            write_chain(h_root, f, hp, chain_recs(hp, f, False, fn_push))
            hn = h_info("affine", 1, 1, "true", s)

            def fn_cn(k, o, s=s, f=f):
                kw = dict(nat=1.2 + o + 0.05 * s, log=0.3 + o + 0.01 * s, mse=0.9 + o)
                if f == F1 and s == 1 and k == ("Tb", "b1", "b2"):
                    kw["nat"] = 40.0
                return kw
            write_chain(h_root, f, hn, chain_recs(hn, f, True, fn_cn))
        fbase = 0.8 if f == F1 else 0.5
        hf = h_info("affine", 1, 0, "fnb", 0, f_run=f)
        write_chain(h_root, f, hf, chain_recs(hf, f, False, lambda k, o, b=fbase: dict(
            nat=b + o, log=0.4 + o, mse=0.9 + o)))
    return h_root


OLD_COLS = ("rung", "g_version", "space", "model", "kind", "mark_class", "metric", "variant",
            "mean", "seed_wobble", "per_seed_0", "per_seed_1", "per_seed_2", "n_tracks")


def write_old_tsv(path: Path) -> Path:
    rows = [("A", "across", "pval", "real", "trained", "narrow", "crps_all", "all", 2.0, 0.1,
             1.95, 2.05, 2.0, 2),
            ("A", "across", "pval", "real", "trained", "narrow", "crps_top1", "all", 12.0, 0.1,
             11.95, 12.05, 12.0, 2),
            ("A", "across", "pval", "real", "trained", "DNase", "crps_all", "all", 0.7, 0.02,
             0.7, 0.71, 0.69, 1),
            # ignored: counts space, another model, another variant, another kind
            ("A", "across", "counts", "real", "trained", "narrow", "crps_all", "all", 99, 1,
             99, 99, 99, 2),
            ("A", "across", "pval", "nocov", "trained", "narrow", "crps_all", "all", 98, 1,
             98, 98, 98, 2),
            ("A", "across", "pval", "real", "trained", "narrow", "crps_all", "nonidentical", 97,
             1, 97, 97, 97, 2),
            ("A", "across", "pval", "real", "law", "narrow", "crps_all", "all", 96, 1, 96, 96,
             96, 2)]
    with path.open("w") as fh:
        fh.write("\t".join(OLD_COLS) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    return path


def write_old_json(path: Path) -> Path:
    """Old grid per_pair pval records of rung A, across, real, seeds 0..2: crps_all = 2 + offset
    + 0.05 seed; the pair a1->a3 is absent; one DNase record explodes (seed 1)."""
    per_pair = []
    for s in (0, 1, 2):
        for (track, src, tgt), (mc, off) in PAIRS.items():
            if (src, tgt) == ("a1", "a3"):
                continue
            v = 2.0 + off + 0.05 * s
            if mc == "DNase" and s == 1:
                v = 25.0
            base = {"kind": "trained", "eval": "score", "rung": "A", "g_version": "across",
                    "model": "real", "seed": s, "track": track, "mark_class": mc,
                    "source_pid": src, "target_pid": tgt, "n_crps_nonfinite": 0}
            per_pair.append({**base, "space": "pval", "crps_all": v, "crps_top1": v + 10,
                             "crps_oracle_scaled_all": v - 0.05, "scale_error_all": 0.05})
            per_pair.append({**base, "space": "counts", "crps_all": 500.0, "crps_top1": 500.0})
            per_pair.append({**base, "space": "pval", "model": "nocov", "crps_all": 600.0,
                             "crps_top1": 600.0})
    path.write_text(json.dumps({"per_pair": per_pair}))
    return path


@pytest.fixture(scope="module")
def agg(tmp_path_factory):
    root = tmp_path_factory.mktemp("h_agg")
    h_root = build_h_root(root)
    old_tsv = write_old_tsv(root / "results_summary.tsv")
    res = ha.aggregate(MANIFEST, COV, h_root, root / "agg", old_grid_tsv=old_tsv,
                       chain_track=None)
    return {"root": root, "h_root": h_root, "res": res, "agg": root / "agg", "old_tsv": old_tsv}


@pytest.fixture(scope="module")
def agg_json(tmp_path_factory):
    root = tmp_path_factory.mktemp("h_agg_json")
    h_root = build_h_root(root)
    old_tsv = write_old_tsv(root / "results_summary.tsv")
    old_json = write_old_json(root / "results.json")
    res = ha.aggregate(MANIFEST, COV, h_root, root / "agg", old_grid_tsv=old_tsv,
                       old_grid_json=old_json, chain_track=None)
    return {"res": res, "agg": root / "agg"}


def read_tsv(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pick(rows, **kw) -> list[dict]:
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


def one(rows, **kw) -> dict:
    got = pick(rows, **kw)
    assert len(got) == 1, (kw, got)
    return got[0]


def num(v):
    return None if v in ("", None) else float(v)


# ---------------------------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------------------------


def test_outputs_written(agg):
    for name in ("results_h.json", "h_alone_summary.tsv", "h_chain_summary.tsv", "h_repair.tsv",
                 "h_vs_oldgrid.tsv", "grid_h.md", "summary_data_h.json"):
        assert (agg["agg"] / name).is_file(), name
    cols = read_tsv(agg["agg"] / "h_alone_summary.tsv")[0].keys()
    for c in ("h_name", "variant", "rung", "noise", "source", "mark_class", "metric", "mean",
              "seed_wobble", "per_seed_0", "per_seed_1", "per_seed_2", "n_products"):
        assert c in cols
    cols = read_tsv(agg["agg"] / "h_chain_summary.tsv")[0].keys()
    for c in ("f_rung", "f_g_version", "h_name", "variant", "rung", "noise", "source",
              "mark_class", "metric", "mean", "seed_wobble", "n_pairs", "n_pairs_crps_gt_20"):
        assert c in cols


def test_missing_and_present_runs_listed(agg):
    res = agg["res"]
    assert "h_formula_bin" in res["alone_present"]
    assert "h_formula_frag" in res["alone_missing"]
    assert "h_affine_r1_push_true_s0" not in res["alone_missing"]
    assert len(res["alone_present"]) + len(res["alone_missing"]) == len(tasks.tasks_alone())
    miss = res["chain_missing"]
    assert F1 in miss and "h_formula_frag" in miss[F1]
    assert "h_affine_r1_push_true_s0" not in miss[F1]
    assert "A_C07M20_counts_real_s0" in res["chain_f_runs_missing"]


# ---------------------------------------------------------------------------------------------
# alone: class means over products, seeds, explosions
# ---------------------------------------------------------------------------------------------


def test_alone_push_mean_over_products_then_seeds(agg):
    rows = read_tsv(agg["agg"] / "h_alone_summary.tsv")
    r = one(rows, h_name="h_affine_r1_push_true", mark_class="narrow", metric="crps_all")
    # products 0, 0.2, 0.4 -> 0.2 (a track macro would give 0.25); seeds 2.2, 2.3, 2.4
    assert num(r["mean"]) == pytest.approx(2.3, **AP)
    assert num(r["seed_wobble"]) == pytest.approx(0.2, **AP)
    assert [num(r[f"per_seed_{k}"]) for k in range(3)] == pytest.approx([2.2, 2.3, 2.4], **AP)
    assert r["variant"] == "affine" and r["rung"] == "1" and r["noise"] == "0"
    assert r["source"] == "true" and r["n_products"] == "3"
    sp = one(rows, h_name="h_affine_r1_push_true", mark_class="narrow",
             metric="crps_oracle_scaled_all")
    assert num(sp["mean"]) == pytest.approx(2.25, **AP)
    lg = one(rows, h_name="h_affine_r1_push_true", mark_class="narrow", metric="crps_log_all")
    assert num(lg["mean"]) == pytest.approx(0.71, **AP)


def test_alone_formula_has_no_seed(agg):
    rows = read_tsv(agg["agg"] / "h_alone_summary.tsv")
    r = one(rows, h_name="h_formula_bin", mark_class="narrow", metric="crps_all")
    assert num(r["mean"]) == pytest.approx(1.2, **AP)
    assert r["seed_wobble"] == "" and r["per_seed_0"] == "" and r["source"] == "formula"


def test_alone_noise_native_is_point_crps_never_crps_all(agg):
    rows = read_tsv(agg["agg"] / "h_alone_summary.tsv")
    assert not pick(rows, h_name="h_affine_r1_noise_true", metric="crps_all")
    r = one(rows, h_name="h_affine_r1_noise_true", mark_class="narrow",
            metric="crps_mean_point_all")
    # seed 2 leaves out P1 (non-finite): (3.4 + 3.6) / 2 = 3.5; seeds 3.2, 3.3, 3.5
    assert [num(r[f"per_seed_{k}"]) for k in range(3)] == pytest.approx([3.2, 3.3, 3.5], **AP)
    assert num(r["mean"]) == pytest.approx(10.0 / 3.0, **AP)
    assert num(r["seed_wobble"]) == pytest.approx(0.3, **AP)
    assert pick(rows, h_name="h_affine_r1_noise_true", metric="crps_mean_point_oracle_scaled_all")
    assert pick(rows, h_name="h_affine_r1_noise_true", metric="crps_log_scale_error_all")


def test_alone_explosions_by_kind_and_nonfinite_apart(agg):
    ex = agg["res"]["explosions"]
    n = one(ex, scope="alone", h_name="h_affine_r1_noise_true", mark_class="narrow")
    assert n["kind"] == "crps_mean_point_all_gt_20"
    assert (n["n_records"], n["n_gt_20"], n["n_native_nonfinite"], n["n_log_nonfinite"]) == \
        (9, 0, 1, 1)
    d = one(ex, scope="alone", h_name="h_affine_r1_noise_true", mark_class="DNase")
    assert (d["n_records"], d["n_gt_20"], d["n_native_nonfinite"]) == (3, 1, 0)
    p = one(ex, scope="alone", h_name="h_affine_r1_push_true", mark_class="DNase")
    assert p["kind"] == "crps_all_gt_20" and p["n_gt_20"] == 0


# ---------------------------------------------------------------------------------------------
# push only against push + own noise (log-scale CRPS and MSE; pairs finite in both)
# ---------------------------------------------------------------------------------------------


def test_push_vs_noise_alone(agg):
    rd = [r for r in agg["res"]["readings"] if r["reading"] == "push_vs_noise"
          and r["scope"] == "alone" and r["mark_class"] == "narrow" and r["variant"] == "affine"
          and r["rung"] == 1 and r["source"] == "true"]
    metrics = {r["metric"] for r in rd}
    assert metrics == {"crps_log_all", "crps_log_top1", "mse_log1p_all", "mse_log1p_top1"}
    r = one(rd, metric="crps_log_all")
    # P1 is non-finite in the noise arm at seed 2: dropped from both arms at every seed.
    # push: 0.8 + 0.01 s (0.81 +- 0.02); noise: 0.7 + 0.03 s (0.73 +- 0.06)
    assert r["push_mean"] == pytest.approx(0.81, **AP)
    assert r["noise_mean"] == pytest.approx(0.73, **AP)
    assert r["gain"] == pytest.approx(0.08, **AP)                       # push - noise
    assert r["bar"] == pytest.approx(0.12, **AP)
    assert r["gain_exceeds_bar"] is False
    pr = r["components"]["pairs"]
    assert pr["n_used"] == 2 and pr["n_dropped_not_finite"] == 1
    sg = r["components"]["split_gain"]
    assert set(sg) == {"crps_log_oracle_scaled_all", "crps_log_scale_error_all"}
    m = one(rd, metric="mse_log1p_all")
    assert m["gain"] == pytest.approx(0.5, **AP) and m["bar"] == pytest.approx(0.0, **AP)
    assert m["gain_exceeds_bar"] is True
    # no native CRPS reading: the two variants' native numbers are of different kinds
    assert not [r for r in agg["res"]["readings"] if r["reading"] == "push_vs_noise"
                and r["metric"] in ("crps_all", "crps_mean_point_all")]


# ---------------------------------------------------------------------------------------------
# chain: class macro over tracks, explosions, capped rows
# ---------------------------------------------------------------------------------------------


def test_chain_class_macro_over_tracks(agg):
    rows = read_tsv(agg["agg"] / "h_chain_summary.tsv")
    r = one(rows, f_rung="A2", f_g_version="across", h_name="h_affine_r1_push_true",
            mark_class="narrow", metric="crps_all")
    # Ta mean 0.3, Tb 0.9 -> 0.6 (a pair mean would give 0.45); seeds 1.6, 1.7, 1.8
    assert num(r["mean"]) == pytest.approx(1.7, **AP)
    assert num(r["seed_wobble"]) == pytest.approx(0.2, **AP)
    assert r["n_pairs"] == "4"
    f = one(rows, f_rung="A", h_name="h_affine_r1_push_fnb", mark_class="narrow",
            metric="crps_all")
    assert num(f["mean"]) == pytest.approx(1.4, **AP) and f["seed_wobble"] == ""
    assert f["source"] == "fnb"
    fo = one(rows, f_rung="A", h_name="h_formula_bin", mark_class="narrow", metric="crps_all")
    assert num(fo["mean"]) == pytest.approx(2.6, **AP)


def test_chain_explosions_and_capped(agg):
    rows = read_tsv(agg["agg"] / "h_chain_summary.tsv")
    r = one(rows, f_rung="A", h_name="h_affine_r1_noise_true", mark_class="narrow",
            metric="crps_mean_point_all")
    assert r["n_pairs_crps_gt_20"] == "1"
    p = one(rows, f_rung="A", h_name="h_affine_r1_push_true", mark_class="narrow",
            metric="crps_all")
    assert p["n_pairs_crps_gt_20"] == "0" and p["n_pairs_crps_nonfinite"] == "1"
    ex = agg["res"]["explosions"]
    e = one(ex, scope="chain", f_rung="A", h_name="h_affine_r1_noise_true", mark_class="narrow")
    assert e["kind"] == "crps_mean_point_all_gt_20" and e["n_gt_20"] == 1
    cap = one(agg["res"]["capped"], f_run=F1)
    assert cap["n_capped_rows"] == 2 and cap["n_pairs_capped"] == 1 and cap["n_pairs"] == 5
    assert one(agg["res"]["capped"], f_run=F2)["n_capped_rows"] == 0


# ---------------------------------------------------------------------------------------------
# repair: true-X'-trained h against the fnb-trained h, in the chain on the same f
# ---------------------------------------------------------------------------------------------


def test_repair_delta_and_bar_from_true_h(agg):
    rows = read_tsv(agg["agg"] / "h_repair.tsv")
    a = one(rows, f_rung="A", mark_class="narrow", variant="affine", rung="1", noise="0",
            metric="crps_all")
    # the pair a2->a1 is non-finite in the true h (seed 0): dropped; Ta (0, 0.6) and Tb 0.9
    # true: 1.6 + 0.1 s (1.7 +- 0.2); fnb: 0.8 + 0.6 = 1.4; delta = true - fnb
    assert num(a["true_mean"]) == pytest.approx(1.7, **AP)
    assert num(a["true_wobble"]) == pytest.approx(0.2, **AP)
    assert num(a["fnb_mean"]) == pytest.approx(1.4, **AP)
    assert num(a["delta"]) == pytest.approx(0.3, **AP)
    assert num(a["bar"]) == pytest.approx(0.4, **AP)
    assert a["abs_delta_exceeds_bar"] == "no"
    assert a["n_pairs_used"] == "3" and a["n_pairs_dropped_not_finite"] == "1"
    assert "true-X" in a["note"]
    b = one(rows, f_rung="A2", mark_class="narrow", variant="affine", rung="1", noise="0",
            metric="crps_all")
    assert num(b["delta"]) == pytest.approx(1.6 + 0.1 - 1.1, **AP)
    assert b["abs_delta_exceeds_bar"] == "yes"
    for m in ("crps_top1", "crps_oracle_scaled_all", "scale_error_all", "crps_log_all",
              "crps_log_top1", "mse_log1p_all", "mse_log1p_top1"):
        assert pick(rows, f_rung="A", mark_class="narrow", variant="affine", noise="0", metric=m)
    # no fnb own-noise h was trained here: no repair row, a skip with its reason
    assert not pick(rows, noise="1")
    sk = [s for s in agg["res"]["readings_skipped"] if s["reading"] == "repair"
          and s.get("noise") == 1]
    assert sk and all(s["reason"] for s in sk)


# ---------------------------------------------------------------------------------------------
# the chain against the old grid's p results
# ---------------------------------------------------------------------------------------------


def test_vs_oldgrid_tsv(agg):
    rows = read_tsv(agg["agg"] / "h_vs_oldgrid.tsv")
    r = one(rows, f_rung="A", f_g_version="across", mark_class="narrow",
            h_name="h_affine_r1_push_true", metric="crps_all")
    assert num(r["old_mean"]) == pytest.approx(2.0, **AP)
    assert num(r["old_wobble"]) == pytest.approx(0.1, **AP)
    assert num(r["h_mean"]) == pytest.approx(1.7, **AP)
    assert num(r["delta"]) == pytest.approx(0.3, **AP)                  # old - chain h
    assert num(r["bar"]) == pytest.approx(0.4, **AP)
    assert r["gain_exceeds_bar"] == "no" and r["matched"] == "no"
    # the chain's f is seed 0: the old grid's seed-0 value and delta beside the 3-seed mean
    assert num(r["old_seed0"]) == pytest.approx(1.95, **AP)
    assert num(r["delta_seed0"]) == pytest.approx(0.25, **AP)
    assert r["gain_seed0_exceeds_bar"] == "no"
    t = one(rows, f_rung="A", mark_class="narrow", h_name="h_affine_r1_push_true",
            metric="crps_top1")
    assert num(t["old_mean"]) == pytest.approx(12.0, **AP)
    fo = one(rows, f_rung="A", mark_class="narrow", h_name="h_formula_bin", metric="crps_all")
    assert num(fo["delta"]) == pytest.approx(2.0 - 2.6, **AP)
    assert num(fo["bar"]) == pytest.approx(0.2, **AP)                   # old wobble only
    fn = one(rows, f_rung="A", mark_class="narrow", h_name="h_affine_r1_push_fnb",
             metric="crps_all")
    # one fnb seed: the bar uses the true-X' h's wobble (0.2) beside the old grid's (0.1)
    assert num(fn["delta"]) == pytest.approx(0.6, **AP)
    assert num(fn["bar"]) == pytest.approx(0.4, **AP)
    # own-noise h: a point CRPS is never put beside the old grid's proper CRPS
    assert not pick(rows, h_name="h_affine_r1_noise_true")
    sk = [s for s in agg["res"]["readings_skipped"] if s["reading"] == "vs_oldgrid"
          and s.get("h_name") == "h_affine_r1_noise_true"]
    assert sk and all(s["reason"] == ha.NOISE_VS_OLD for s in sk)
    assert {r["h_metric"] for r in rows} <= {"crps_all", "crps_top1", "crps_oracle_scaled_all",
                                             "scale_error_all"}
    # no old-grid row for A2: no comparison (not an error)
    assert not pick(rows, f_rung="A2")
    assert r["old_n_crps_gt_20"] == ""                                  # TSV only: unknown
    assert r["h_n_crps_nonfinite"] == "1"


def test_vs_oldgrid_json_pair_matched(agg_json):
    rows = read_tsv(agg_json["agg"] / "h_vs_oldgrid.tsv")
    r = one(rows, f_rung="A", f_g_version="across", mark_class="narrow",
            h_name="h_affine_r1_push_true", metric="crps_all")
    # common pairs finite in both: a1->a2 (Ta) and b1->b2 (Tb); a1->a3 absent from the old
    # grid, a2->a1 non-finite in the chain h at seed 0.  Ta 0, Tb 0.9 -> 0.45
    assert r["matched"] == "yes"
    assert num(r["h_mean"]) == pytest.approx(1.55, **AP)
    assert num(r["old_mean"]) == pytest.approx(2.5, **AP)
    assert num(r["old_wobble"]) == pytest.approx(0.1, **AP)
    assert num(r["delta"]) == pytest.approx(0.95, **AP)
    assert r["gain_exceeds_bar"] == "yes"
    assert num(r["old_seed0"]) == pytest.approx(2.45, **AP)
    assert num(r["delta_seed0"]) == pytest.approx(0.9, **AP)
    assert r["gain_seed0_exceeds_bar"] == "yes"
    assert r["n_pairs_used"] == "2" and r["n_pairs_dropped_not_finite"] == "1"
    assert r["n_pairs_only_one_arm"] == "1"
    assert num(r["old_tsv_mean"]) == pytest.approx(2.0, **AP)
    d = one(rows, f_rung="A", mark_class="DNase", h_name="h_affine_r1_push_true",
            metric="crps_all")
    assert d["old_n_crps_gt_20"] == "1"
    # d1->d2 has a non-finite log-scale bin at seed 1 in the chain h: it stays in the native
    # comparison (only the compared metrics must be finite)
    assert d["matched"] == "yes" and d["n_pairs_used"] == "1"
    assert d["n_pairs_dropped_not_finite"] == "0"
    assert num(d["h_mean"]) == pytest.approx(1.1, **AP)
    assert num(d["old_mean"]) == pytest.approx((2.0 + 25.0 + 2.1) / 3, **AP)
    sp = one(rows, f_rung="A", mark_class="narrow", h_name="h_affine_r1_push_true",
             metric="crps_oracle_scaled_all")
    assert num(sp["old_mean"]) == pytest.approx(2.45, **AP)


# ---------------------------------------------------------------------------------------------
# the grid and the summary data
# ---------------------------------------------------------------------------------------------


def test_grid_md_rows_and_rules(agg):
    md = (agg["agg"] / "grid_h.md").read_text()
    for label in ("formula, bin only", "formula, fragment window", "affine, rung 1",
                  "affine, rung 5", "MLP, rung 1", "MLP, rung 5", "formula + affine, rung 5",
                  "formula + MLP, rung 5"):
        assert label in md, label
    assert "push only" in md and "push + own noise" in md
    assert "point CRPS of the mean" in md
    assert "CRPS > 20" in md
    assert "2.3 ± 0.2" in md
    for word in ("supported", "refuted", "correlation"):
        assert word not in md.lower()


def test_summary_data(agg):
    sd = json.loads((agg["agg"] / "summary_data_h.json").read_text())
    for k in ("alone_cells", "chain_cells", "readings", "readings_skipped", "explosions",
              "capped"):
        assert k in sd
    res = json.loads((agg["agg"] / "results_h.json").read_text())
    assert res["readings"] and res["alone_rows"] and res["chain_rows"]


def test_vs_oldgrid_json_with_empty_tsv(tmp_path):
    h_root = build_h_root(tmp_path)
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    old_json = write_old_json(tmp_path / "results.json")
    ha.aggregate(MANIFEST, COV, h_root, tmp_path / "agg", old_grid_tsv=empty,
                 old_grid_json=old_json)
    rows = read_tsv(tmp_path / "agg" / "h_vs_oldgrid.tsv")
    r = one(rows, f_rung="A", mark_class="narrow", h_name="h_affine_r1_push_true",
            metric="crps_all")
    assert r["matched"] == "yes" and num(r["delta"]) == pytest.approx(0.95, **AP)
    assert r["old_tsv_mean"] == ""


def test_rec_ok_checks_only_compared_metrics():
    r = rec(False, nat=1.0, log=0.5, mse=1.0, crps_log_all=None, n_crps_log_nonfinite=3)
    assert ha._rec_ok(r, ("crps_all", "crps_top1"))
    assert not ha._rec_ok(r, ("crps_all", "crps_log_all"))


def test_cli(tmp_path):
    h_root = build_h_root(tmp_path)
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    assert ha.main([str(MANIFEST), str(COV), str(h_root), str(tmp_path / "agg"),
                    "--old-grid-tsv", str(empty)]) == 0
    assert (tmp_path / "agg" / "grid_h.md").is_file()
    assert read_tsv(tmp_path / "agg" / "h_vs_oldgrid.tsv") == []


# ---------------------------------------------------------------------------------------------
# a partial chain set (PI 2026-10-17: aggregate on the finished chains, refresh later)
# ---------------------------------------------------------------------------------------------

C19 = [f"{r}_C19M16_counts_real_s0" for r in ("A", "B", "C", "D", "A2", "B2", "C2", "D2")]


def _chain_f(h_root: Path, f_run: str, done: bool) -> None:
    """A per-track f with a formula h and one true push h (3 seeds); `done` writes CHAIN_DONE."""
    fb = h_info("formula", None, 0, None, None, window="bin")
    write_chain(h_root, f_run, fb, chain_recs(fb, f_run, False, lambda k, o: dict(
        nat=2.0 + o, log=1.0 + o, mse=1.0 + o)), done=done)
    for s in (0, 1, 2):
        hp = h_info("affine", 1, 0, "true", s)
        write_chain(h_root, f_run, hp, chain_recs(hp, f_run, False, lambda k, o, s=s: dict(
            nat=1.0 + o + 0.1 * s, log=0.5 + o, mse=1.0 + o)), done=done)


def _merge_error(h_root: Path, f_run: str, rows: dict) -> None:
    h = {name: {"variant": "affine", "rung": 1, "noise": 1, "source": "true", "n_pairs": 5,
                "n_bins": nb, "support_median": None if not nb else 600.0,
                "support_max": None if not nb else 1028, "rel_err_median": None,
                "rel_err_q99": None, "rel_err_max": mx, "signed_rel_err_mean": None,
                "crps_log_exact_mean": None, "crps_log_merged_mean": None, "avg_rel_err": av}
         for name, (nb, mx, av) in rows.items()}
    (h_root / "chain" / f_run / "merge_error.json").write_text(json.dumps(
        {"f_run_name": f_run, "mix_groups": 512, "mix_check_bins": 64, "check": "x", "h": h}))


def _fit(h_root: Path, sub: str, name: str, source: str, steps: int, best: int, capped: int):
    d = h_root / sub / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"h_name": name, "source": source, "max_steps": 4000,
                                               "DEFAULTS": {"max_steps": 4000}}))
    (d / "timing.json").write_text(json.dumps({"steps_run": steps, "best_step": best,
                                               "n_capped": capped, "n_capped_train": capped,
                                               "n_capped_val": 0}))
    (d / "TRAIN_DONE").write_text("x\n")


@pytest.fixture(scope="module")
def partial(tmp_path_factory):
    """The C19M16 scope (the default): C and D2 finished (CHAIN_DONE), A scored but not finished
    (no CHAIN_DONE), B only shard files in B.partial; the two across f of build_h_root (out of
    this scope) are present too."""
    root = tmp_path_factory.mktemp("h_agg_partial")
    h_root = build_h_root(root)
    _chain_f(h_root, C19[2], True)
    _chain_f(h_root, C19[7], True)
    _chain_f(h_root, C19[0], False)
    (h_root / "chain" / (C19[1] + ".partial")).mkdir(parents=True)
    (h_root / "chain" / (C19[1] + ".partial") / "pair_0000.json").write_text("{}")
    _merge_error(h_root, C19[2], {"h_a": (0, None, None), "h_b": (0, None, None)})
    _merge_error(h_root, C19[7], {"h_a": (91, 6.6e-4, 1.7e-4), "h_b": (91, 2e-7, 1e-7),
                                  "h_c": (0, None, None)})
    _fit(h_root, "runs_fnb", f"h_mlp_r5_push_{C19[2]}_s0", f"fnb:{C19[2]}", 4000, 3250, 3)
    _fit(h_root, "runs_fnb", f"h_mlp_r1_push_{C19[2]}_s0", f"fnb:{C19[2]}", 2250, 1250, 0)
    _fit(h_root, "runs_fnb", f"h_mlp_r1_push_{C19[7]}_s0", f"fnb:{C19[7]}", 4000, 4000, 0)
    _fit(h_root, "runs_true", "h_mlp_r1_push_true_s0", "true", 4000, 4000, 0)
    res = ha.aggregate(MANIFEST, COV, h_root, root / "agg",
                       old_grid_tsv=write_old_tsv(root / "results_summary.tsv"))
    return {"res": res, "agg": root / "agg", "h_root": h_root}


def test_partial_chains_expected_scope_is_the_c19m16_designs(partial):
    res = partial["res"]
    assert res["chain_track"] == "C19M16"
    assert res["chain_f_expected_runs"] == C19
    assert res["chain_f_finished_runs"] == [C19[2], C19[7]]
    assert res["chain_f_runs_missing"] == [f for f in C19 if f not in (C19[2], C19[7])]
    assert res["chain_f_not_finished"] == [C19[0]]          # scores present, no CHAIN_DONE
    assert sorted(res["chain_f_out_of_scope"]) == sorted([F1, F2])
    sd = json.loads((partial["agg"] / "summary_data_h.json").read_text())
    assert sd["chain_f_present"] == 2 and sd["chain_f_expected"] == 8
    for k in ("chain_f_expected_runs", "chain_f_finished_runs", "chain_f_runs_missing",
              "chain_track", "merge_errors", "fits"):
        assert k in sd, k


def test_partial_chains_missing_f_skipped_with_reason(partial):
    sk = [k for k in partial["res"]["readings_skipped"] if k["reason"] == "chain not finished"]
    by_f = {}
    for k in sk:
        by_f.setdefault(k["f_run"], set()).add(k["reading"])
    assert set(by_f) == set(partial["res"]["chain_f_runs_missing"])
    for f, readings in by_f.items():
        assert readings == {"push_vs_noise", "repair", "vs_oldgrid"}, f
    a = [k for k in sk if k["f_run"] == C19[0]][0]
    assert a["f_rung"] == "A" and a["f_g_version"] == "per_track"
    assert "CHAIN_DONE" in a["detail"]
    md = (partial["agg"] / "grid_h.md").read_text()
    assert "chain not finished" in md


def test_partial_chains_unfinished_scores_left_out(partial):
    rows = read_tsv(partial["agg"] / "h_chain_summary.tsv")
    assert not pick(rows, f_rung="A", f_g_version="per_track")      # A: no CHAIN_DONE
    assert pick(rows, f_rung="C", f_g_version="per_track")
    assert pick(rows, f_rung="D2", f_g_version="per_track")
    assert pick(rows, f_rung="A", f_g_version="across")             # out of scope, still read
    assert C19[0] not in partial["res"]["chain_present"]
    assert not any(c["f_run"].endswith(".partial") for c in partial["res"]["capped"])


def test_merge_errors_per_f(partial):
    me = {m["f_run"]: m for m in partial["res"]["merge_errors"]}
    assert set(me) == {C19[2], C19[7]}
    d2 = me[C19[7]]
    assert d2["mix_groups"] == 512 and d2["n_h"] == 3 and d2["n_h_checked"] == 2
    assert d2["rel_err_max"] == pytest.approx(6.6e-4) and d2["rel_err_max_h"] == "h_a"
    assert d2["avg_rel_err_max"] == pytest.approx(1.7e-4) and d2["avg_rel_err_max_h"] == "h_a"
    assert d2["n_bins_max"] == 91 and d2["n_h_rel_err_max_ge_bar"] == 0
    c = me[C19[2]]
    assert c["n_h_checked"] == 0 and c["rel_err_max"] is None
    assert "merged mixture equals the exact one" in c["note"]
    md = (partial["agg"] / "grid_h.md").read_text()
    assert "Merged mixture" in md and "512" in md


def test_fits_step_cap_and_capped_rows(partial):
    fits = partial["res"]["fits"]
    fnb = {r["h_name"]: r for r in fits["runs"] if r["source"] == "fnb"}
    assert len(fnb) == 3
    assert fnb[f"h_mlp_r5_push_{C19[2]}_s0"]["at_step_cap"] is True
    assert fnb[f"h_mlp_r1_push_{C19[2]}_s0"]["at_step_cap"] is False
    assert fnb[f"h_mlp_r1_push_{C19[7]}_s0"]["best_at_step_cap"] is True
    tot = {t["source"]: t for t in fits["totals"]}
    assert tot["fnb"]["n_fits"] == 3 and tot["fnb"]["n_at_step_cap"] == 2
    assert tot["fnb"]["n_best_at_step_cap"] == 1
    assert tot["fnb"]["n_fits_with_capped_rows"] == 1 and tot["fnb"]["n_capped_rows"] == 3
    assert tot["true"]["n_fits"] == 1 and tot["true"]["n_at_step_cap"] == 1
    md = (partial["agg"] / "grid_h.md").read_text()
    assert "step cap" in md


# ---------------------------------------------------------------------------------------------
# synthetic controls
# ---------------------------------------------------------------------------------------------

N_BINS = {"chr19": 400, "chr21": 300, "chr22": 200}


def test_add_controls(tmp_path):
    prod = tmp_path / "products"
    manifest, cov = lsynth.make_products(prod, 0, n_bins=N_BINS)
    rl = tmp_path / "control_read_length.tsv"
    pids = synth.add_controls(prod, manifest, seed=0, rl_tsv=rl)
    rows = {r["pid"]: r for r in read_tsv(manifest)}
    t1 = sorted(p for p, r in rows.items() if r["track"] == "T1")
    assert sorted(pids) == t1 and t1
    for pid, r in rows.items():
        npz = prod / pid / "control_counts25.npz"
        if r["track"] == "T1":
            assert r["control_counts25_md5"] == hashlib.md5(npz.read_bytes()).hexdigest()
        else:
            assert r["control_counts25_md5"] == "" and not npz.exists()
    # lambda = 0.6 x the track base's mean count
    with np.load(prod / "T1__base__base" / "counts25.npz") as z:
        base_mean = float(np.concatenate([z[c] for c in z.files]).astype(float).mean())
    with np.load(prod / t1[0] / "control_counts25.npz") as z:
        c = np.concatenate([z[k] for k in z.files]).astype(float)
        assert set(z.files) == set(N_BINS)
        assert z["chr19"].dtype == np.uint32
    assert c.mean() == pytest.approx(0.6 * base_mean, rel=0.15)
    # the manifest keeps its columns in order; the control cache and products.load accept it
    with open(manifest) as fh:
        assert fh.readline().rstrip("\n").split("\t") == list(lsynth.MANIFEST_COLUMNS)
    for pid, r in rows.items():
        controls.build_control_cache(prod, tmp_path / "ctl", pid, r)
    assert len(list((tmp_path / "ctl").glob("*__counts.json"))) == len(t1)
    infos = products.load(manifest, cov, rl)
    assert infos[t1[0]].has_control == 1 and infos[t1[0]].control_read_length > 0
    # deterministic in the seed
    with np.load(prod / t1[0] / "control_counts25.npz") as z:
        first = {k: z[k] for k in z.files}
    synth.add_controls(prod, manifest, seed=0)
    with np.load(prod / t1[0] / "control_counts25.npz") as z:
        assert all(np.array_equal(first[k], z[k]) for k in first)
